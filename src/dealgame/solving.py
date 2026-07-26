"""Solvers and evaluation for deal games.

Three methods are compared, mirroring the seed paper's thesis in miniature:

- :func:`run_cfr` -- Counterfactual Regret Minimization (OpenSpiel), the
  specialized regret-minimization baseline.
- :func:`run_mmd` -- Magnetic Mirror Descent (OpenSpiel), the regularized
  method at the heart of the seed paper.
- :class:`ReinforcePolicyGradient` -- a from-scratch tabular REINFORCE policy
  gradient ("roll our own" generic PG), the method the seed paper argues should
  be competitive despite its simplicity.

All three are scored by the same exact metric: OpenSpiel exploitability
(NashConv / 2), the distance from Nash for a two-player zero-sum game.
"""

from __future__ import annotations

import numpy as np
import pyspiel
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import cfr, exploitability, mmd_dilated


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / e.sum()


def run_cfr(game, iterations: int, eval_every: int = 10):
    """Run CFR, returning (iteration, exploitability) checkpoints."""
    solver = cfr.CFRSolver(game)
    curve = [(0, exploitability.exploitability(game, solver.average_policy()))]
    for it in range(1, iterations + 1):
        solver.evaluate_and_update_policy()
        if it % eval_every == 0:
            curve.append((it, exploitability.exploitability(game, solver.average_policy())))
    return curve


def run_mmd(game, iterations: int, eval_every: int = 10, alpha: float = 0.0,
            stepsize: float = 0.1):
    """Run Magnetic Mirror Descent, returning (iteration, exploitability).

    ``alpha`` is the regularization temperature; ``alpha=0`` targets the Nash
    equilibrium (so exploitability drives toward zero, comparable to CFR).
    """
    solver = mmd_dilated.MMDDilatedEnt(game, alpha, stepsize)
    curve = [(0, exploitability.exploitability(game, solver.get_avg_policies()))]
    for it in range(1, iterations + 1):
        solver.update_sequences()
        if it % eval_every == 0:
            curve.append((it, exploitability.exploitability(game, solver.get_avg_policies())))
    return curve


class ReinforcePolicyGradient:
    """Tabular REINFORCE self-play with a per-infostate baseline and entropy bonus.

    A deliberately simple, dependency-free generic policy-gradient method. Each
    iteration samples ``batch`` self-play episodes, then nudges each visited
    infostate's softmax policy toward actions that beat the running baseline.
    Exploitability is measured on the iterate-averaged policy, which is what
    converges in self-play for zero-sum games.
    """

    def __init__(self, game, lr: float = 0.5, entropy_coef: float = 0.01,
                 batch: int = 256, seed: int = 0):
        self._game = game
        self._lr = lr
        self._entropy_coef = entropy_coef
        self._batch = batch
        self._rng = np.random.default_rng(seed)
        self._logits: dict[str, np.ndarray] = {}
        self._legal: dict[str, list[int]] = {}
        self._baseline: dict[str, float] = {}
        # Running sum of policy snapshots, for the averaged evaluation policy.
        self._avg_sum: dict[str, np.ndarray] = {}
        self._avg_count = 0

    def _policy_at(self, iset: str, legal: list[int]) -> np.ndarray:
        if iset not in self._logits:
            self._logits[iset] = np.zeros(len(legal))
            self._legal[iset] = list(legal)
            self._baseline[iset] = 0.0
            self._avg_sum[iset] = np.zeros(len(legal))
        return _softmax(self._logits[iset])

    def _sample_episode(self):
        state = self._game.new_initial_state()
        traj = []  # (player, iset, action_index)
        while not state.is_terminal():
            if state.is_chance_node():
                actions, probs = zip(*state.chance_outcomes())
                action = int(self._rng.choice(actions, p=np.asarray(probs)))
                state.apply_action(action)
                continue
            player = state.current_player()
            iset = state.information_state_string(player)
            legal = state.legal_actions(player)
            probs = self._policy_at(iset, legal)
            idx = int(self._rng.choice(len(legal), p=probs))
            traj.append((player, iset, idx))
            state.apply_action(legal[idx])
        return traj, state.returns()

    def _update_from(self, traj, returns):
        for player, iset, idx in traj:
            probs = _softmax(self._logits[iset])
            g = returns[player]
            advantage = g - self._baseline[iset]
            score_grad = -probs.copy()
            score_grad[idx] += 1.0  # d log pi(a) / d logits
            logp = np.log(np.clip(probs, 1e-12, 1.0))
            entropy = -float(np.sum(probs * logp))
            entropy_grad = -probs * (logp + entropy)
            self._logits[iset] += self._lr * (
                advantage * score_grad + self._entropy_coef * entropy_grad
            )
            self._baseline[iset] += 0.01 * (g - self._baseline[iset])

    def _snapshot_average(self):
        self._avg_count += 1
        for iset, logits in self._logits.items():
            self._avg_sum[iset] += _softmax(logits)

    def average_policy(self) -> "policy_lib.TabularPolicy":
        tab = policy_lib.TabularPolicy(self._game)
        for iset, total in self._avg_sum.items():
            if iset not in tab.state_lookup:
                continue
            row = tab.state_lookup[iset]
            avg = total / max(self._avg_count, 1)
            tab.action_probability_array[row] = 0.0
            for pos, action_id in enumerate(self._legal[iset]):
                tab.action_probability_array[row][action_id] = avg[pos]
        return tab

    def run(self, iterations: int, eval_every: int = 10):
        curve = []
        for it in range(1, iterations + 1):
            for _ in range(self._batch):
                traj, returns = self._sample_episode()
                self._update_from(traj, returns)
            self._snapshot_average()
            if it % eval_every == 0 or it == 1:
                expl = exploitability.exploitability(self._game, self.average_policy())
                curve.append((it, expl))
        return curve
