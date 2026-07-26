"""Deep (neural) self-play solvers for deal games.

These are the methods that matter for the seed paper's actual claim, which is
about *deep* policy gradient, not tabular play:

- :func:`train_policy_gradient` -- a generic neural policy gradient (OpenSpiel's
  PolicyGradient agent, e.g. A2C/RPG), the deep analogue of our tabular REINFORCE
  and the stand-in for the seed paper's PPO/PPG.
- :func:`train_nfsp` -- Neural Fictitious Self-Play, the fictitious-play
  representative among specialized deep methods.

Both train two agents in self-play via ``rl_environment`` and are evaluated by
exact OpenSpiel exploitability. Every curve records wall-clock seconds alongside
the episode count so methods can be compared on a normalized compute budget, not
just iterations.

Iterate averaging (why it matters): in two-player zero-sum imperfect-information
games the *last iterate* of policy gradient cycles around the equilibrium and does
not converge in exploitability; only the *time-average* of the iterates converges
(the same reason CFR averages its strategies and NFSP trains a separate
average-policy network). We therefore evaluate deep policy gradient on a tabular
running average of the network's per-information-state action probabilities -- the
games are small enough to tabulate a neural policy exactly -- and report both the
average (the honest convergent metric) and the oscillating last iterate.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from open_spiel.python import policy as policy_lib
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import policy_aggregator
from open_spiel.python.algorithms.psro_v2 import best_response_oracle
from open_spiel.python.algorithms.psro_v2 import psro_v2
from open_spiel.python.pytorch import deep_cfr
from open_spiel.python.pytorch import nfsp
from open_spiel.python.pytorch import policy_gradient as pg


class RLPolicies(policy_lib.Policy):
    """Wrap trained RL agents as an OpenSpiel joint policy for exploitability.

    This evaluates each agent's *current* (last-iterate) policy. For NFSP, set
    ``nfsp_mode=nfsp.MODE.AVERAGE_POLICY`` so the convergent average-policy network
    is queried; for plain policy gradient the last iterate does not converge, so
    use :class:`RunningAverageTabularPolicy` instead for the headline metric.
    """

    def __init__(self, game, agents, nfsp_mode=None):
        super().__init__(game, list(range(len(agents))))
        self._agents = agents
        self._nfsp_mode = nfsp_mode
        n = len(agents)
        self._obs = {"info_state": [None] * n, "legal_actions": [None] * n}

    def action_probabilities(self, state, player_id=None):
        cur = state.current_player()
        legal = state.legal_actions(cur)
        self._obs["current_player"] = cur
        self._obs["info_state"][cur] = state.information_state_tensor(cur)
        self._obs["legal_actions"][cur] = legal
        time_step = rl_environment.TimeStep(
            observations=self._obs, rewards=None, discounts=None, step_type=None)
        agent = self._agents[cur]
        if self._nfsp_mode is not None:
            with agent.temp_mode_as(self._nfsp_mode):
                probs = agent.step(time_step, is_evaluation=True).probs
        else:
            probs = agent.step(time_step, is_evaluation=True).probs
        return {action: probs[action] for action in legal}


class RunningAverageTabularPolicy(policy_lib.Policy):
    """Tabular time-average of a live RL policy.

    Maintains, per information state, a running sum of the live network's action
    probabilities. :meth:`snapshot` adds one observation of the current policy;
    :meth:`action_probabilities` returns the normalized average so far. Because the
    average of the iterates (not the last iterate) is what converges to a Nash
    equilibrium in zero-sum games, exploitability is measured on this object.
    """

    def __init__(self, game, agents):
        self._tabular = policy_lib.TabularPolicy(game)
        super().__init__(game, list(range(len(agents))))
        self._agents = agents
        self._sum = np.zeros_like(self._tabular.action_probability_array)
        self.count = 0
        n = len(agents)
        self._obs = {"info_state": [None] * n, "legal_actions": [None] * n}

    def snapshot(self):
        """Accumulate one observation of the agents' current policy."""
        for state in self._tabular.states:
            cur = state.current_player()
            legal = state.legal_actions()
            idx = self._tabular.state_lookup[state.information_state_string()]
            self._obs["current_player"] = cur
            self._obs["info_state"][cur] = state.information_state_tensor(cur)
            self._obs["legal_actions"][cur] = legal
            time_step = rl_environment.TimeStep(
                observations=self._obs, rewards=None, discounts=None, step_type=None)
            probs = self._agents[cur].step(time_step, is_evaluation=True).probs
            for action in legal:
                self._sum[idx, action] += probs[action]
        self.count += 1

    def action_probabilities(self, state, player_id=None):
        legal = state.legal_actions()
        if self.count == 0:
            return {action: 1.0 / len(legal) for action in legal}
        row = self._sum[self._tabular.state_lookup[state.information_state_string()]]
        total = sum(row[action] for action in legal)
        if total <= 0:
            return {action: 1.0 / len(legal) for action in legal}
        return {action: row[action] / total for action in legal}


def _run_self_play(game, agents, episodes, eval_every, evaluate,
                   snapshot_every=None, snapshot=None, seed=0):
    """Self-play loop. ``evaluate()`` returns a dict of named exploitabilities.

    Returns a list of rows ``{"episode", "seconds", **evaluate()}``. If
    ``snapshot`` is given it is called with the current episode number every
    ``snapshot_every`` episodes so a running-average policy can accumulate between
    evaluations (the callback decides whether to record, e.g. after a burn-in).

    The global numpy/torch RNG state is saved and restored around the evaluation
    and snapshot calls, so the training trajectory is independent of the evaluation
    cadence (the agents sample actions from the global numpy RNG even in evaluation
    mode). ``seconds`` accumulates training time only; evaluation and snapshot
    overhead is excluded so wall-clock is comparable across methods.
    """
    env = rl_environment.Environment(game, seed=seed)
    curve = []
    train_seconds = 0.0
    for ep in range(1, episodes + 1):
        t0 = time.time()
        time_step = env.reset()
        while not time_step.last():
            pid = time_step.observations["current_player"]
            action = agents[pid].step(time_step).action
            time_step = env.step([action])
        for agent in agents:
            agent.step(time_step)
        train_seconds += time.time() - t0
        do_snapshot = snapshot is not None and ep % snapshot_every == 0
        do_eval = ep % eval_every == 0 or ep == 1
        if do_snapshot or do_eval:
            np_state = np.random.get_state()
            torch_state = torch.get_rng_state()
            if do_snapshot:
                snapshot(ep)
            if do_eval:
                row = {"episode": ep, "seconds": train_seconds}
                row.update(evaluate())
                curve.append(row)
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
    return curve


def train_policy_gradient(game, episodes=40000, eval_every=2000,
                          loss_str="rpg", hidden=(64,), batch_size=16,
                          pi_lr=0.005, critic_lr=0.05, seed=0,
                          snapshot_every=None, burn_in_frac=0.5):
    """Generic deep policy gradient in self-play (the seed paper's method class).

    Defaults are the stable configuration found by sweep: the regret-policy-gradient
    loss with a modest learning rate, which converges monotonically rather than
    diverging into a limit cycle (higher learning rates overshoot and lock into a
    high-exploitability cycle).

    Evaluated on the live last iterate (key ``"last"``, the convergent metric for
    this stable config and the seed paper's reporting convention) and on a tabular
    *tail* time-average of the network policy (key ``"avg"``), which discards the
    first ``burn_in_frac`` of training so the random-init transient does not
    pollute the average. The tail-average is a config-robust check: it also smooths
    out the limit cycle that less-stable settings exhibit.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)  # the agents sample actions from the global numpy RNG
    env = rl_environment.Environment(game)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]
    agents = [
        pg.PolicyGradient(
            player_id=i,
            info_state_size=info_state_size,
            num_actions=num_actions,
            loss_str=loss_str,
            hidden_layers_sizes=list(hidden),
            batch_size=batch_size,
            pi_learning_rate=pi_lr,
            critic_learning_rate=critic_lr,
            optimizer_str="adam",
        )
        for i in range(2)
    ]
    if snapshot_every is None:
        snapshot_every = max(1, eval_every // 50)
    burn_in = int(burn_in_frac * episodes)
    average = RunningAverageTabularPolicy(game, agents)
    last_iterate = RLPolicies(game, agents)

    def snapshot_after_burn_in(ep):
        if ep >= burn_in:
            average.snapshot()

    def evaluate():
        row = {"last": exploitability.exploitability(game, last_iterate)}
        if average.count > 0:
            row["avg"] = exploitability.exploitability(game, average)
        return row

    return _run_self_play(game, agents, episodes, eval_every, evaluate,
                          snapshot_every=snapshot_every,
                          snapshot=snapshot_after_burn_in, seed=seed)


def train_nfsp(game, episodes=40000, eval_every=2000, hidden=(64,),
               batch_size=128, rl_lr=0.01, sl_lr=0.01,
               anticipatory_param=0.1, seed=0):
    """Neural Fictitious Self-Play, the FP-based deep baseline.

    Evaluated on its average-policy network (key ``"nfsp"``), which is NFSP's own
    convergent time-average. The inner best-response DQN's exploration is scheduled
    to decay over the training horizon (its default ~1e6-step decay would leave the
    agent nearly random for a short run, so NFSP would appear not to learn).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)  # the agents sample actions from the global numpy RNG
    env = rl_environment.Environment(game)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]
    agents = [
        nfsp.NFSP(
            i,
            info_state_size,
            num_actions,
            hidden_layers_sizes=list(hidden),
            reservoir_buffer_capacity=int(2e6),
            min_buffer_size_to_learn=1000,
            anticipatory_param=anticipatory_param,
            batch_size=batch_size,
            rl_learning_rate=rl_lr,
            sl_learning_rate=sl_lr,
            optimizer_str="adam",
            # forwarded to the inner DQN best-responder
            replay_buffer_capacity=int(2e5),
            epsilon_start=0.6,
            epsilon_end=0.01,
            epsilon_decay_duration=int(episodes),
            update_target_network_every=1000,
            discount_factor=1.0,
        )
        for i in range(2)
    ]
    eval_policy = RLPolicies(game, agents, nfsp_mode=nfsp.MODE.AVERAGE_POLICY)

    def evaluate():
        return {"nfsp": exploitability.exploitability(game, eval_policy)}

    return _run_self_play(game, agents, episodes, eval_every, evaluate, seed=seed)


def isolated_exploitability(game, pol):
    """Exploitability that does not perturb the global RNG.

    Querying a neural policy at every information state samples actions from the
    global numpy/torch RNG; saving and restoring that state keeps training
    independent of how often we evaluate.
    """
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        return exploitability.exploitability(game, pol)
    finally:
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)


def train_deep_cfr(game, num_iterations=100, num_traversals=100,
                   policy_layers=(64, 64), advantage_layers=(64, 64),
                   learning_rate=1e-3, advantage_train_steps=600,
                   policy_train_steps=600, seed=0):
    """Deep CFR (Brown et al. 2019), the deep counterfactual-regret baseline.

    Deep CFR trains advantage networks by external-sampling traversals and fits a
    final average-strategy network; we score that average policy by exact
    exploitability. It stands in for the deep CFR-family methods (including ESCHER,
    a later variance-reduced variant). Returns a one-row curve (the solver runs all
    iterations internally) for a uniform interface with the other solvers.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    solver = deep_cfr.DeepCFRSolver(
        game,
        policy_network_layers=tuple(policy_layers),
        advantage_network_layers=tuple(advantage_layers),
        num_iterations=num_iterations,
        num_traversals=num_traversals,
        learning_rate=learning_rate,
        batch_size_advantage=2048,
        batch_size_strategy=2048,
        memory_capacity=int(1e6),
        advantage_network_train_steps=advantage_train_steps,
        policy_network_train_steps=policy_train_steps,
        reinitialize_advantage_networks=True,
        device="cpu",
        seed=seed,
    )
    start = time.time()
    solver.solve()
    expl = isolated_exploitability(game, solver.to_tabular())
    return [{"iteration": num_iterations, "seconds": time.time() - start,
             "deepcfr": expl}]


def train_psro(game, iterations=15, eval_every=1, seed=0, meta_method="prd",
               sims_per_entry=1000):
    """PSRO (Lanctot et al. 2017) with exact best-response oracles.

    Double-oracle policy-space response: each iteration adds an exact best response
    to the current meta-strategy, then re-solves the empirical meta-game (projected
    replicator dynamics by default). We score the aggregated meta-strategy by exact
    exploitability. Exact BR oracles are the strongest oracle and are appropriate
    for games small enough to tabulate; the OpenSpiel RL-oracle variants depend on
    JAX, which we do not install.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random_policy = policy_lib.TabularPolicy(game)
    oracle = best_response_oracle.BestResponseOracle(game=game,
                                                     policy=random_policy)
    agents = [random_policy.__copy__() for _ in range(2)]
    solver = psro_v2.PSROSolver(
        game,
        oracle,
        initial_policies=agents,
        training_strategy_selector="probabilistic",
        rectifier="",
        sims_per_entry=sims_per_entry,
        number_policies_selected=1,
        meta_strategy_method=meta_method,
        prd_iterations=50000,
        prd_gamma=1e-10,
        sample_from_marginals=True,
        symmetric_game=False,
    )
    aggregator = policy_aggregator.PolicyAggregator(game)
    curve = []
    start = time.time()
    for it in range(1, iterations + 1):
        solver.iteration()
        if it % eval_every == 0 or it == iterations:
            policies = solver.get_policies()
            probs = solver.get_meta_strategies()
            aggr = aggregator.aggregate(list(range(2)), policies, probs)
            expl = exploitability.exploitability(game, aggr)
            curve.append({"iteration": it, "seconds": time.time() - start,
                          "psro": expl})
    return curve
