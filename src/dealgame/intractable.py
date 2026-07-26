"""Approximate exploitability and tabulation-free training for intractable games.

When a deal game is large enough that its tree cannot be enumerated (e.g. the
multi-signal common-value auction with several signals per bidder), exact
exploitability is unavailable and the tabular tail-average used elsewhere cannot be
built. This module provides the tools to operate in that regime:

- :func:`approx_exploitability` estimates exploitability with a *learned best
  response*: freeze the policy, train a PPO best-responder against it for each
  player, and Monte-Carlo the gain from deviating. Learned BR underestimates the
  true best response, so this is a lower bound on exploitability; we validate it
  against exact exploitability on a small game where both can be computed.
- :func:`train_intractable` trains PPO or PPG self-play without ever tabulating the
  game (weights are tail-averaged directly), evaluating with
  :func:`approx_exploitability`.

Everything here queries policies only at states reached by rollouts, so nothing
enumerates the game tree.
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch

from dealgame.deep_solving import RLPolicies
from dealgame.ppo_solving import _NEG_INF, PPGAgent, PPOAgent, _collect


class UniformPolicy:
    """Uniform-random policy; a tabulation-free upper anchor for exploitability."""

    def __init__(self, game):
        self.game = game

    def action_probabilities(self, state, player_id=None):
        legal = state.legal_actions(state.current_player())
        return {a: 1.0 / len(legal) for a in legal}


class NaiveBidder:
    """Bids its posterior-mean estimate of W with no winner's-curse shading.

    A non-trivial but economically *exploitable* reference for the multi-signal
    common-value auction: it computes the Bayesian posterior over W from its own
    signals and bids that estimate, ignoring the adverse selection of winning, so it
    overpays. A best-responder that shades should beat it. Used to show the learned BR
    has real power on the intractable game (it exploits this policy), so a learned
    policy that the same BR cannot exploit is genuinely near-unexploitable, not just
    beyond a weak BR.
    """

    def __init__(self, game):
        self._n = game.num_values
        self._num_bids = game.num_bids
        self._noise = game.signal_noise_0

    def action_probabilities(self, state, player_id=None):
        player = state.current_player()
        signals = state._signals[player]
        logpost = []
        for w in range(self._n):
            lp = 0.0
            for s in signals:
                p = self._noise / self._n + (1.0 - self._noise if s == w else 0.0)
                lp += math.log(max(p, 1e-12))
            logpost.append(lp)
        m = max(logpost)
        weights = [math.exp(x - m) for x in logpost]
        z = sum(weights)
        post_mean = sum((w + 1) * weights[w] / z for w in range(self._n))
        bid = min(range(self._num_bids), key=lambda b: abs(b - post_mean))
        legal = state.legal_actions(player)
        return {a: (1.0 if a == bid else 0.0) for a in legal}


def _policy_act(policy):
    """Action function sampling from a policy's action probabilities."""
    def act(state):
        probs = policy.action_probabilities(state)
        actions = list(probs)
        p = np.asarray([probs[a] for a in actions], dtype=float)
        return int(np.random.choice(actions, p=p / p.sum()))
    return act


def _net_greedy_act(agent, player):
    """Greedy (argmax) action function for a trained agent's network."""
    def act(state):
        info = np.asarray(state.information_state_tensor(player), dtype=np.float32)
        legal = state.legal_actions(player)
        with torch.no_grad():
            logits, _ = agent.net(torch.from_numpy(info).unsqueeze(0))
            masked = logits[0].masked_fill(~agent._mask(legal), _NEG_INF)
        return int(torch.argmax(masked).item())
    return act


def _rollout_value(game, act_fns, episodes, with_se=False):
    """Mean per-player return over ``episodes`` rollouts, each player using its
    action function in ``act_fns`` (chance sampled from the game).

    When ``with_se`` is set, also returns the Monte-Carlo standard error of each
    per-player mean, so callers can report the estimator's resolution.
    """
    totals = [0.0, 0.0]
    sq_totals = [0.0, 0.0]
    for _ in range(episodes):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                actions, probs = zip(*state.chance_outcomes())
                state.apply_action(int(np.random.choice(actions, p=np.asarray(probs))))
                continue
            cur = state.current_player()
            state.apply_action(act_fns[cur](state))
        r = state.returns()
        for i in (0, 1):
            totals[i] += r[i]
            sq_totals[i] += r[i] * r[i]
    means = [t / episodes for t in totals]
    if not with_se:
        return means
    ses = []
    for i in (0, 1):
        var = max(sq_totals[i] / episodes - means[i] ** 2, 0.0)
        ses.append(math.sqrt(var / episodes))
    return means, ses


def _collect_vs_fixed(game, br_agent, br_player, opp_policy, batch_episodes):
    """Rollouts where ``br_player`` learns and the opponent plays ``opp_policy``."""
    opp_act = _policy_act(opp_policy)
    batch = {k: [] for k in
             ("states", "actions", "logps", "values", "returns", "masks")}
    for _ in range(batch_episodes):
        state = game.new_initial_state()
        steps = []
        while not state.is_terminal():
            if state.is_chance_node():
                actions, probs = zip(*state.chance_outcomes())
                state.apply_action(int(np.random.choice(actions, p=np.asarray(probs))))
                continue
            cur = state.current_player()
            if cur == br_player:
                info = np.asarray(state.information_state_tensor(cur),
                                  dtype=np.float32)
                legal = state.legal_actions(cur)
                with torch.no_grad():
                    logits, value = br_agent.net(torch.from_numpy(info).unsqueeze(0))
                    masked = logits[0].masked_fill(~br_agent._mask(legal), _NEG_INF)
                    dist = torch.distributions.Categorical(logits=masked)
                    action = dist.sample()
                    logp = dist.log_prob(action)
                steps.append((info, int(action.item()), float(logp.item()),
                              float(value.item()), br_agent._mask(legal).numpy()))
                state.apply_action(int(action.item()))
            else:
                state.apply_action(opp_act(state))
        ret = state.returns()[br_player]
        for info, action, logp, value, mask in steps:
            batch["states"].append(info)
            batch["actions"].append(action)
            batch["logps"].append(logp)
            batch["values"].append(value)
            batch["returns"].append(ret)
            batch["masks"].append(mask)
    return batch


def _train_br(game, opp_policy, br_player, batches, batch_episodes, seed,
              hidden=(64,), lr=3e-3):
    """Train a PPO best-responder for ``br_player`` against a fixed opponent."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    info_size = game.information_state_tensor_size()
    num_actions = game.num_distinct_actions()
    br = PPOAgent(br_player, info_size, num_actions, hidden=hidden, lr=lr)
    for _ in range(batches):
        batch = _collect_vs_fixed(game, br, br_player, opp_policy, batch_episodes)
        br.update(batch)
    return br


def approx_exploitability(game, policy, br_batches=300, br_batch_episodes=256,
                          mc_episodes=20000, seed=0, br_hidden=(64,), br_lr=3e-3):
    """Lower-bound exploitability via a learned best response.

    For each player, train a PPO best-responder against ``policy`` and Monte-Carlo
    its gain over playing ``policy``. Returns ``(approx_exploitability, gains,
    base_values)`` where ``approx_exploitability`` is half the summed gains (the
    NashConv estimate halved). Because a learned BR is imperfect, this underestimates
    the true exploitability; treat it as a lower bound.
    """
    np.random.seed(seed)
    pol_act = _policy_act(policy)
    base = _rollout_value(game, [pol_act, pol_act], mc_episodes)
    gains = []
    for p in range(2):
        br = _train_br(game, policy, p, br_batches, br_batch_episodes, seed + 1 + p,
                       hidden=br_hidden, lr=br_lr)
        acts = [None, None]
        acts[p] = _net_greedy_act(br, p)
        acts[1 - p] = pol_act
        br_value = _rollout_value(game, acts, mc_episodes)[p]
        gains.append(br_value - base[p])
    approx = max(sum(gains) / 2.0, 0.0)
    return approx, gains, base


def approx_exploitability_detailed(game, policy, br_batches=300,
                                   br_batch_episodes=256, mc_episodes=20000, seed=0,
                                   br_hidden=(64,), br_lr=3e-3):
    """Like :func:`approx_exploitability`, but also report the *unclamped* estimate
    and its Monte-Carlo standard error.

    Returns a dict with ``approx`` (the clamped, non-negative estimate reported
    elsewhere), ``approx_unclamped`` (the raw half-sum of gains, which can dip
    negative when the policy is near-optimal and the learned BR is no better than
    the policy itself), ``gains``, ``base``, and ``se`` (the standard error of the
    half-sum estimate, propagated from the base and best-response rollout
    variances). ``se`` quantifies the estimator's resolution floor: an
    ``approx`` of zero means "no profitable deviation found above roughly ``se``."
    """
    np.random.seed(seed)
    pol_act = _policy_act(policy)
    base, base_se = _rollout_value(game, [pol_act, pol_act], mc_episodes,
                                   with_se=True)
    gains = []
    gain_vars = []
    for p in range(2):
        br = _train_br(game, policy, p, br_batches, br_batch_episodes, seed + 1 + p,
                       hidden=br_hidden, lr=br_lr)
        acts = [None, None]
        acts[p] = _net_greedy_act(br, p)
        acts[1 - p] = pol_act
        br_value, br_se = _rollout_value(game, acts, mc_episodes, with_se=True)
        gains.append(br_value[p] - base[p])
        gain_vars.append(br_se[p] ** 2 + base_se[p] ** 2)
    half = sum(gains) / 2.0
    se = 0.5 * math.sqrt(sum(gain_vars))
    return {"approx": max(half, 0.0), "approx_unclamped": half, "gains": gains,
            "base": base, "se": se}


def _averaged_policy(game, agents, avg_state_dicts):
    """Wrap tail-averaged network weights as a tabulation-free joint policy."""
    shells = []
    for p, agent in enumerate(agents):
        shell = PPOAgent(p, game.information_state_tensor_size(),
                         game.num_distinct_actions())
        shell.net = copy.deepcopy(agent.net)
        shell.net.load_state_dict(avg_state_dicts[p])
        shells.append(shell)
    return RLPolicies(game, shells)


def train_intractable(game, kind="ppo", episodes=300000, batch_episodes=256,
                      hidden=(64,), lr=3e-3, eval_points=4, burn_in_frac=0.5,
                      seed=0, approx_kw=None):
    """Train PPO or PPG self-play without tabulating the game.

    Weights are tail-averaged (after a burn-in) directly, so no TabularPolicy is
    ever built. At ``eval_points`` checkpoints (and at the end) we estimate
    exploitability of the weight-averaged policy with :func:`approx_exploitability`.
    Returns a curve of ``{episode, seconds, approx_expl, gains}``.
    """
    approx_kw = approx_kw or {}
    torch.manual_seed(seed)
    np.random.seed(seed)
    info_size = game.information_state_tensor_size()
    num_actions = game.num_distinct_actions()
    if kind == "ppg":
        agents = [PPGAgent(p, info_size, num_actions, hidden=hidden, lr=lr)
                  for p in range(2)]
    else:
        agents = [PPOAgent(p, info_size, num_actions, hidden=hidden, lr=lr)
                  for p in range(2)]

    burn_in = int(burn_in_frac * episodes)
    avg_sum = [None, None]
    avg_count = 0
    eval_episodes = sorted(
        {int(burn_in + (episodes - burn_in) * (i + 1) / eval_points)
         for i in range(eval_points)})

    curve = []
    train_seconds = 0.0
    episodes_done = 0
    phase_count = 0
    next_eval_idx = 0
    while episodes_done < episodes:
        t0 = time.time()
        batch = _collect(game, agents, batch_episodes)
        if kind == "ppg":
            for p in range(2):
                agents[p].policy_phase(batch[p])
            phase_count += 1
            if phase_count % agents[0].n_policy == 0:
                for p in range(2):
                    agents[p].auxiliary_phase()
        else:
            for p in range(2):
                agents[p].update(batch[p])
        episodes_done += batch_episodes
        train_seconds += time.time() - t0

        if episodes_done >= burn_in:
            for p in range(2):
                sd = {k: v.clone() for k, v in agents[p].net.state_dict().items()}
                if avg_sum[p] is None:
                    avg_sum[p] = sd
                else:
                    for k in avg_sum[p]:
                        avg_sum[p][k] += sd[k]
            avg_count += 1

        if (next_eval_idx < len(eval_episodes)
                and episodes_done >= eval_episodes[next_eval_idx] and avg_count):
            avg_sds = [{k: v / avg_count for k, v in avg_sum[p].items()}
                       for p in range(2)]
            policy = _averaged_policy(game, agents, avg_sds)
            np_state = np.random.get_state()
            torch_state = torch.get_rng_state()
            approx, gains, _ = approx_exploitability(game, policy, **approx_kw)
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
            curve.append({"episode": episodes_done, "seconds": train_seconds,
                          "approx_expl": approx, "gains": gains})
            next_eval_idx += 1
    return curve


def tree_and_strategy_size(num_values, num_bids, num_signals):
    """(terminal histories, information sets per player) for a takeover instance.

    Used to document why exact methods are infeasible: both grow exponentially in
    ``num_signals``."""
    terminals = num_values ** (1 + 2 * num_signals) * num_bids ** 2
    infosets = num_values ** num_signals
    return terminals, infosets
