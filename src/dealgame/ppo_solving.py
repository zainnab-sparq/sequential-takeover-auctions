"""Proper PPO self-play for deal games.

The seed paper's headline method is PPO. OpenSpiel's bundled PPO is built for
vectorized single-agent gym environments; rather than bend it to two-player
turn-based self-play, we implement clipped PPO directly in our self-play
framework, which keeps the evaluation (exact exploitability on a tabular
tail-average) identical to the other deep solvers.

These games have terminal rewards only (no intermediate payoffs) and are
undiscounted, so the return-to-go from every decision state a player visits is
the same final payoff and the advantage reduces to ``return - V(info_state)`` at
each such state. That is exactly GAE with ``gamma = lambda = 1``, so a player that
acts in several rounds needs no multi-step bootstrapping: ``_collect`` logs one
sample per decision it makes and assigns the terminal return to each. The
implementation uses the full PPO machinery (clipped surrogate, value baseline,
entropy bonus, multiple epochs over minibatches), so it is PPO proper on both the
single-round (sealed) and multi-round (sequential) auctions.
"""

from __future__ import annotations

import collections
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_spiel.python import rl_agent

from dealgame.deep_solving import (RLPolicies, RunningAverageTabularPolicy,
                                   isolated_exploitability)

_NEG_INF = -1e9


class _ActorCritic(nn.Module):
    def __init__(self, in_dim, num_actions, hidden):
        super().__init__()
        layers = []
        dim = in_dim
        for width in hidden:
            layers += [nn.Linear(dim, width), nn.Tanh()]
            dim = width
        self.torso = nn.Sequential(*layers)
        self.policy_head = nn.Linear(dim, num_actions)
        self.value_head = nn.Linear(dim, 1)

    def forward(self, x):
        z = self.torso(x)
        return self.policy_head(z), self.value_head(z).squeeze(-1)


class _ValueNet(nn.Module):
    """Standalone value network (PPG keeps value separate from the policy trunk)."""

    def __init__(self, in_dim, hidden):
        super().__init__()
        layers = []
        dim = in_dim
        for width in hidden:
            layers += [nn.Linear(dim, width), nn.Tanh()]
            dim = width
        layers += [nn.Linear(dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PPOAgent:
    """One PPO actor-critic for a single player.

    Exposes ``step(time_step, is_evaluation=True)`` returning a full action-prob
    vector so the shared evaluation wrappers (:class:`RLPolicies`,
    :class:`RunningAverageTabularPolicy`) work unchanged.
    """

    def __init__(self, player_id, info_state_size, num_actions, hidden=(64,),
                 lr=3e-3, clip=0.2, epochs=4, minibatches=4, ent_coef=0.01,
                 vf_coef=0.5, max_grad_norm=0.5):
        self.player_id = player_id
        self.num_actions = num_actions
        self.net = _ActorCritic(info_state_size, num_actions, hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.clip = clip
        self.epochs = epochs
        self.minibatches = minibatches
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

    def _mask(self, legal_actions):
        m = torch.zeros(self.num_actions, dtype=torch.bool)
        m[legal_actions] = True
        return m

    def step(self, time_step, is_evaluation=False):
        info = np.asarray(time_step.observations["info_state"][self.player_id],
                          dtype=np.float32)
        legal = time_step.observations["legal_actions"][self.player_id]
        with torch.no_grad():
            logits, _ = self.net(torch.from_numpy(info).unsqueeze(0))
            masked = logits[0].masked_fill(~self._mask(legal), _NEG_INF)
            probs = F.softmax(masked, dim=-1).numpy()
        action = int(np.random.choice(self.num_actions, p=probs)) if legal else None
        return rl_agent.StepOutput(action=action, probs=probs)

    def update(self, batch):
        states = torch.as_tensor(np.asarray(batch["states"]), dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long)
        old_logps = torch.as_tensor(batch["logps"], dtype=torch.float32)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32)
        old_values = torch.as_tensor(batch["values"], dtype=torch.float32)
        masks = torch.as_tensor(np.asarray(batch["masks"]), dtype=torch.bool)
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(actions)
        order = np.arange(n)
        mb_size = max(1, n // self.minibatches)
        for _ in range(self.epochs):
            np.random.shuffle(order)
            for start in range(0, n, mb_size):
                idx = order[start:start + mb_size]
                logits, values = self.net(states[idx])
                masked = logits.masked_fill(~masks[idx], _NEG_INF)
                dist = torch.distributions.Categorical(logits=masked)
                new_logp = dist.log_prob(actions[idx])
                ratio = torch.exp(new_logp - old_logps[idx])
                adv = advantages[idx]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv
                pi_loss = -torch.min(surr1, surr2).mean()
                v_loss = F.mse_loss(values, returns[idx])
                entropy = dist.entropy().mean()
                loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()


def _collect(game, agents, batch_episodes):
    """Self-play rollouts; returns a per-player batch dict of trajectory arrays."""
    buf = {p: collections.defaultdict(list) for p in range(2)}
    for _ in range(batch_episodes):
        state = game.new_initial_state()
        ep_steps = {0: [], 1: []}
        while not state.is_terminal():
            if state.is_chance_node():
                actions, probs = zip(*state.chance_outcomes())
                state.apply_action(int(np.random.choice(actions, p=np.asarray(probs))))
                continue
            player = state.current_player()
            agent = agents[player]
            info = np.asarray(state.information_state_tensor(player),
                              dtype=np.float32)
            legal = state.legal_actions(player)
            with torch.no_grad():
                logits, value = agent.net(torch.from_numpy(info).unsqueeze(0))
                masked = logits[0].masked_fill(~agent._mask(legal), _NEG_INF)
                dist = torch.distributions.Categorical(logits=masked)
                action = dist.sample()
                logp = dist.log_prob(action)
            ep_steps[player].append(
                (info, int(action.item()), float(logp.item()),
                 float(value.item()), agent._mask(legal).numpy()))
            state.apply_action(int(action.item()))
        returns = state.returns()
        for player in range(2):
            for info, action, logp, value, mask in ep_steps[player]:
                b = buf[player]
                b["states"].append(info)
                b["actions"].append(action)
                b["logps"].append(logp)
                b["values"].append(value)
                b["returns"].append(returns[player])
                b["masks"].append(mask)
    return buf


def train_ppo(game, episodes=300000, eval_every=15000, batch_episodes=256,
              hidden=(64,), lr=3e-3, clip=0.2, epochs=4, minibatches=4,
              ent_coef=0.01, seed=0, burn_in_frac=0.5, snapshot_every=None):
    """Clipped PPO in self-play, evaluated like the other deep solvers.

    Reports the live last iterate (``"last"``) and the tabular tail-average
    (``"avg"``, the convergent metric in zero-sum games). ``seconds`` counts
    training time only; evaluation/snapshot RNG is isolated so training does not
    depend on the evaluation cadence.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    info_state_size = game.information_state_tensor_size()
    num_actions = game.num_distinct_actions()
    agents = [PPOAgent(p, info_state_size, num_actions, hidden=hidden, lr=lr,
                       clip=clip, epochs=epochs, minibatches=minibatches,
                       ent_coef=ent_coef) for p in range(2)]
    average = RunningAverageTabularPolicy(game, agents)
    last_iterate = RLPolicies(game, agents)
    if snapshot_every is None:
        snapshot_every = max(1, eval_every // 50)
    burn_in = int(burn_in_frac * episodes)

    curve = []
    train_seconds = 0.0
    episodes_done = 0
    next_eval = 1            # record an initial point
    next_snapshot = snapshot_every
    while episodes_done < episodes:
        t0 = time.time()
        batch = _collect(game, agents, batch_episodes)
        for player in range(2):
            agents[player].update(batch[player])
        episodes_done += batch_episodes
        train_seconds += time.time() - t0

        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        if episodes_done >= burn_in and episodes_done >= next_snapshot:
            average.snapshot()
            next_snapshot += snapshot_every
        if episodes_done >= next_eval:
            row = {"episode": episodes_done, "seconds": train_seconds,
                   "last": isolated_exploitability(game, last_iterate)}
            if average.count > 0:
                row["avg"] = isolated_exploitability(game, average)
            curve.append(row)
            next_eval = (episodes_done // eval_every + 1) * eval_every
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
    return curve


class PPGAgent:
    """Phasic Policy Gradient actor (Cobbe et al. 2021), one per player.

    Decouples policy and value optimization. The policy network (``net``, with an
    auxiliary value head) is trained with the clipped PPO objective during the
    *policy phase*, using advantages from a separate value network (``vf``). Every
    ``n_policy`` policy phases, an *auxiliary phase* distills returns into the
    policy trunk's auxiliary value head while a behavioral-cloning KL term pins the
    policy output to its pre-phase snapshot, so improving the shared representation
    does not move the policy. Exposes the same ``step`` interface as
    :class:`PPOAgent` so the shared evaluation wrappers work unchanged.
    """

    def __init__(self, player_id, info_state_size, num_actions, hidden=(64,),
                 lr=3e-3, clip=0.2, epochs=4, minibatches=4, ent_coef=0.01,
                 max_grad_norm=0.5, n_policy=8, aux_epochs=6, bc_coef=1.0):
        self.player_id = player_id
        self.num_actions = num_actions
        self.net = _ActorCritic(info_state_size, num_actions, hidden)
        self.vf = _ValueNet(info_state_size, hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.vf_opt = torch.optim.Adam(self.vf.parameters(), lr=lr)
        self.clip = clip
        self.epochs = epochs
        self.minibatches = minibatches
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.n_policy = n_policy
        self.aux_epochs = aux_epochs
        self.bc_coef = bc_coef
        self._aux_buffer = {"states": [], "returns": [], "masks": []}

    def _mask(self, legal_actions):
        m = torch.zeros(self.num_actions, dtype=torch.bool)
        m[legal_actions] = True
        return m

    def step(self, time_step, is_evaluation=False):
        info = np.asarray(time_step.observations["info_state"][self.player_id],
                          dtype=np.float32)
        legal = time_step.observations["legal_actions"][self.player_id]
        with torch.no_grad():
            logits, _ = self.net(torch.from_numpy(info).unsqueeze(0))
            masked = logits[0].masked_fill(~self._mask(legal), _NEG_INF)
            probs = F.softmax(masked, dim=-1).numpy()
        action = int(np.random.choice(self.num_actions, p=probs)) if legal else None
        return rl_agent.StepOutput(action=action, probs=probs)

    def policy_phase(self, batch):
        """One PPO policy-phase update; advantages come from the separate ``vf``."""
        states = torch.as_tensor(np.asarray(batch["states"]), dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long)
        old_logps = torch.as_tensor(batch["logps"], dtype=torch.float32)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32)
        masks = torch.as_tensor(np.asarray(batch["masks"]), dtype=torch.bool)
        with torch.no_grad():
            values = self.vf(states)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(actions)
        order = np.arange(n)
        mb_size = max(1, n // self.minibatches)
        for _ in range(self.epochs):
            np.random.shuffle(order)
            for start in range(0, n, mb_size):
                idx = order[start:start + mb_size]
                logits, _ = self.net(states[idx])
                masked = logits.masked_fill(~masks[idx], _NEG_INF)
                dist = torch.distributions.Categorical(logits=masked)
                ratio = torch.exp(dist.log_prob(actions[idx]) - old_logps[idx])
                adv = advantages[idx]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv
                pi_loss = -torch.min(surr1, surr2).mean()
                entropy = dist.entropy().mean()
                loss = pi_loss - self.ent_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()
        # Fit the separate value network on returns.
        for _ in range(self.epochs):
            np.random.shuffle(order)
            for start in range(0, n, mb_size):
                idx = order[start:start + mb_size]
                v_loss = F.mse_loss(self.vf(states[idx]), returns[idx])
                self.vf_opt.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.vf.parameters(), self.max_grad_norm)
                self.vf_opt.step()
        # Stash this phase's data for the next auxiliary phase.
        self._aux_buffer["states"].append(np.asarray(batch["states"]))
        self._aux_buffer["returns"].append(np.asarray(batch["returns"]))
        self._aux_buffer["masks"].append(np.asarray(batch["masks"]))

    def auxiliary_phase(self):
        """Distill returns into the policy trunk while preserving the policy."""
        states = torch.as_tensor(np.concatenate(self._aux_buffer["states"]),
                                 dtype=torch.float32)
        returns = torch.as_tensor(np.concatenate(self._aux_buffer["returns"]),
                                  dtype=torch.float32)
        masks = torch.as_tensor(np.concatenate(self._aux_buffer["masks"]),
                                dtype=torch.bool)
        with torch.no_grad():
            old_logits, _ = self.net(states)
            old_logp = F.log_softmax(old_logits.masked_fill(~masks, _NEG_INF), dim=-1)
        n = len(returns)
        order = np.arange(n)
        mb_size = max(1, n // self.minibatches)
        for _ in range(self.aux_epochs):
            np.random.shuffle(order)
            for start in range(0, n, mb_size):
                idx = order[start:start + mb_size]
                logits, aux_value = self.net(states[idx])
                aux_loss = F.mse_loss(aux_value, returns[idx])
                new_logp = F.log_softmax(
                    logits.masked_fill(~masks[idx], _NEG_INF), dim=-1)
                # KL(old || new) as the policy-preservation (behavioral-cloning) term.
                kl = (old_logp[idx].exp() * (old_logp[idx] - new_logp)).sum(-1).mean()
                loss = aux_loss + self.bc_coef * kl
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()
                v_loss = F.mse_loss(self.vf(states[idx]), returns[idx])
                self.vf_opt.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.vf.parameters(), self.max_grad_norm)
                self.vf_opt.step()
        self._aux_buffer = {"states": [], "returns": [], "masks": []}


def train_ppg(game, episodes=300000, eval_every=15000, batch_episodes=256,
              hidden=(64,), lr=3e-3, clip=0.2, epochs=4, minibatches=4,
              ent_coef=0.01, n_policy=8, aux_epochs=6, bc_coef=1.0,
              seed=0, burn_in_frac=0.5, snapshot_every=None):
    """Phasic Policy Gradient (Cobbe et al. 2021) in self-play.

    The seed paper's other recommended generic method. Same evaluation as
    :func:`train_ppo` (tabular tail-average, key ``"avg"``, plus the live last
    iterate ``"last"``); ``seconds`` counts training time only.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    info_state_size = game.information_state_tensor_size()
    num_actions = game.num_distinct_actions()
    agents = [PPGAgent(p, info_state_size, num_actions, hidden=hidden, lr=lr,
                       clip=clip, epochs=epochs, minibatches=minibatches,
                       ent_coef=ent_coef, n_policy=n_policy, aux_epochs=aux_epochs,
                       bc_coef=bc_coef) for p in range(2)]
    average = RunningAverageTabularPolicy(game, agents)
    last_iterate = RLPolicies(game, agents)
    if snapshot_every is None:
        snapshot_every = max(1, eval_every // 50)
    burn_in = int(burn_in_frac * episodes)

    curve = []
    train_seconds = 0.0
    episodes_done = 0
    phase_count = 0
    next_eval = 1
    next_snapshot = snapshot_every
    while episodes_done < episodes:
        t0 = time.time()
        batch = _collect(game, agents, batch_episodes)
        for player in range(2):
            agents[player].policy_phase(batch[player])
        phase_count += 1
        if phase_count % n_policy == 0:
            for player in range(2):
                agents[player].auxiliary_phase()
        episodes_done += batch_episodes
        train_seconds += time.time() - t0

        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        if episodes_done >= burn_in and episodes_done >= next_snapshot:
            average.snapshot()
            next_snapshot += snapshot_every
        if episodes_done >= next_eval:
            row = {"episode": episodes_done, "seconds": train_seconds,
                   "last": isolated_exploitability(game, last_iterate)}
            if average.count > 0:
                row["avg"] = isolated_exploitability(game, average)
            curve.append(row)
            next_eval = (episodes_done // eval_every + 1) * eval_every
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
    return curve
