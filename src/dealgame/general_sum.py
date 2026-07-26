"""Exact general-sum (own-profit) Bayes-Nash equilibrium for the takeover auction.

The zero-sum solvers (CFR/MMD) elsewhere in this package optimize the *profit
difference*, which injects a rivalry term and so finds the equilibrium of the
relativized game, not the auction's own Bayes-Nash equilibrium (see
:mod:`dealgame.base`). For the economic comparative statics (the value of
information, the value of diligence) the object that matters is the genuine
general-sum equilibrium in which each bidder maximizes its *own* profit.

This module computes that equilibrium exactly. The auction factorizes: terminal
profit depends only on ``(w, b0, b1)`` and each bidder's signals are conditionally
independent given ``w``. So the whole game collapses into small dense tensors
(``num_values**k`` information sets, ``num_bids`` actions, ``num_values`` common
values) and the equilibrium is found by fictitious play on own profit, with no tree
traversal and no Monte-Carlo noise. ``EnumeratedAuction`` mirrors the payoff and
signal logic of :class:`dealgame.takeover.TakeoverAuctionState`; the equivalence is
pinned by ``tests/test_general_sum.py``.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass

import numpy as np

FP_TOLERANCE = 1e-4          # NashConv below this counts as converged
FP_MAX_ITERATIONS = 60000


def _signal_given_value(num_values: int, noise: float) -> np.ndarray:
    """Matrix ``P[signal, w]`` for one noisy signal of the common value.

    With probability ``1 - noise`` the signal equals the true value index; the
    remaining mass is spread uniformly over all values (matching the takeover
    game's ``_signal_distribution``).
    """
    base = noise / num_values
    mat = np.full((num_values, num_values), base)
    mat[np.arange(num_values), np.arange(num_values)] += 1.0 - noise
    return mat  # column w is the signal distribution given value w


def _tuple_given_value(num_values: int, num_signals: int,
                       noise: float) -> np.ndarray:
    """Matrix ``P[tuple, w]`` for a bidder's ordered ``k``-tuple of signals."""
    per_signal = _signal_given_value(num_values, noise)
    tuples = list(itertools.product(range(num_values), repeat=num_signals))
    out = np.empty((len(tuples), num_values))
    for row, tup in enumerate(tuples):
        prob = np.ones(num_values)
        for sig in tup:
            prob *= per_signal[sig]
        out[row] = prob
    return out


def _profit_tables(num_values: int, num_bids: int, toehold: float,
                   bid_values: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Own-profit tensors ``profit_p[w, b0, b1]`` for each bidder.

    Ties go to bidder 0; the winner pays its bid (the toehold holder pays only the
    share it does not already own); the loser with a toehold collects its share of
    the winning price. Mirrors ``TakeoverAuctionState.raw_profits``.

    ``bid_values`` maps a bid index to its monetary amount; the default
    ``[0, 1, ..., num_bids-1]`` makes the index the amount (the takeover game's
    convention). Passing a finer grid over the same range (e.g. a linspace) lets a
    caller check that a result is not an artifact of the bid-grid resolution. The
    grid must be increasing so the index tie-break (ties to bidder 0) still matches
    the amount ordering.
    """
    if bid_values is None:
        bid_values = np.arange(num_bids, dtype=float)
    profit0 = np.zeros((num_values, num_bids, num_bids))
    profit1 = np.zeros((num_values, num_bids, num_bids))
    for w in range(num_values):
        w_val = w + 1  # value grid is 1..num_values
        for b0 in range(num_bids):
            for b1 in range(num_bids):
                if b0 >= b1:
                    profit0[w, b0, b1] = w_val - (1.0 - toehold) * bid_values[b0]
                else:
                    profit1[w, b0, b1] = w_val - bid_values[b1]
                    profit0[w, b0, b1] = toehold * bid_values[b1]
    return profit0, profit1


@dataclass
class EnumeratedAuction:
    """Dense tensor view of the takeover auction for exact general-sum solving."""

    num_values: int
    num_bids: int
    num_signals_0: int = 1
    num_signals_1: int = 1
    noise_0: float = 0.5
    noise_1: float = 0.5
    toehold: float = 0.0
    bid_values: np.ndarray | None = None  # bid index -> amount; default is the index

    def __post_init__(self) -> None:
        v = self.num_values
        self._pw = np.full(v, 1.0 / v)  # P(w), uniform common value
        # P(tuple, w) weighted by P(w): row sums give the joint over (tuple, w).
        self._joint0 = _tuple_given_value(v, self.num_signals_0, self.noise_0) * self._pw
        self._joint1 = _tuple_given_value(v, self.num_signals_1, self.noise_1) * self._pw
        self._bid_values = (np.arange(self.num_bids, dtype=float)
                            if self.bid_values is None
                            else np.asarray(self.bid_values, dtype=float))
        self._profit0, self._profit1 = _profit_tables(
            v, self.num_bids, self.toehold, self._bid_values)

    @property
    def n_infosets_0(self) -> int:
        return self._joint0.shape[0]

    @property
    def n_infosets_1(self) -> int:
        return self._joint1.shape[0]

    def uniform_policy(self, player: int) -> np.ndarray:
        n = self.n_infosets_0 if player == 0 else self.n_infosets_1
        return np.full((n, self.num_bids), 1.0 / self.num_bids)

    def _bid_given_value(self, opp_policy: np.ndarray, opp_player: int) -> np.ndarray:
        """``P[w, b]`` that the opponent bids ``b`` given true value ``w``."""
        joint = self._joint0 if opp_player == 0 else self._joint1
        cond = joint / self._pw  # back out P(tuple | w)
        return cond.T @ opp_policy  # [w, b]

    def best_response_q(self, br_player: int, opp_policy: np.ndarray) -> np.ndarray:
        """Reach-weighted own-profit ``Q[infoset, bid]`` for ``br_player``.

        Summing the per-infoset maximum gives the best response's expected own
        profit; the per-infoset argmax gives the (pure) best-response policy.
        """
        opp = 1 - br_player
        m_opp = self._bid_given_value(opp_policy, opp)            # [w, b_opp]
        profit = self._profit0 if br_player == 0 else self._profit1
        if br_player == 0:
            # profit0[w, b0, b1] contracted over opponent bid b1.
            r = np.einsum("wij,wj->wi", profit, m_opp)           # [w, b0]
        else:
            r = np.einsum("wji,wj->wi", profit, m_opp)           # [w, b1]
        joint = self._joint0 if br_player == 0 else self._joint1  # [infoset, w]
        return joint @ r                                          # [infoset, b]

    def own_profit_value(self, policy0: np.ndarray, policy1: np.ndarray) -> tuple[float, float]:
        q0 = self.best_response_q(0, policy1)
        q1 = self.best_response_q(1, policy0)
        return float(np.sum(q0 * policy0)), float(np.sum(q1 * policy1))


def _onehot_best_response(q: np.ndarray, num_bids: int) -> np.ndarray:
    br = np.zeros_like(q)
    br[np.arange(q.shape[0]), np.argmax(q, axis=1)] = 1.0
    return br


def own_profit_fictitious_play(
    auction: EnumeratedAuction,
    max_iterations: int = FP_MAX_ITERATIONS,
    tolerance: float = FP_TOLERANCE,
) -> dict:
    """Solve the auction's own-profit Bayes-Nash equilibrium by fictitious play.

    Each round both bidders best-respond (on own profit) to the opponent's running
    average, then the averages are updated. NashConv (the summed own-profit gain
    from a unilateral deviation) measures distance from equilibrium; it is reported
    so convergence is honest rather than assumed.
    """
    avg0 = auction.uniform_policy(0)
    avg1 = auction.uniform_policy(1)
    nashconv_trace: list[tuple[int, float]] = []
    for it in range(1, max_iterations + 1):
        q0 = auction.best_response_q(0, avg1)
        q1 = auction.best_response_q(1, avg0)
        br_val0, br_val1 = float(np.sum(np.max(q0, axis=1))), float(np.sum(np.max(q1, axis=1)))
        cur_val0, cur_val1 = float(np.sum(q0 * avg0)), float(np.sum(q1 * avg1))
        nashconv = (br_val0 - cur_val0) + (br_val1 - cur_val1)
        nashconv_trace.append((it, nashconv))
        if nashconv < tolerance:
            break
        weight = 1.0 / (it + 1)
        avg0 = (1.0 - weight) * avg0 + weight * _onehot_best_response(q0, auction.num_bids)
        avg1 = (1.0 - weight) * avg1 + weight * _onehot_best_response(q1, auction.num_bids)
    final_nashconv = nashconv_trace[-1][1]
    converged = final_nashconv < tolerance
    if not converged:
        warnings.warn(
            f"fictitious play did not reach tolerance {tolerance:g} in "
            f"{nashconv_trace[-1][0]} iterations (NashConv={final_nashconv:.3e}); "
            "the returned values are a near-equilibrium, not converged. This is "
            "expected at the fully-informed corner (noise=0), where averaged "
            "best-response play cycles slowly.",
            RuntimeWarning,
            stacklevel=2,
        )
    value0, value1 = auction.own_profit_value(avg0, avg1)
    return {
        "policy0": avg0,
        "policy1": avg1,
        "value0": value0,
        "value1": value1,
        "nashconv": final_nashconv,
        "converged": converged,
        "iterations": nashconv_trace[-1][0],
        "nashconv_trace": nashconv_trace,
    }


def expected_bid(auction: EnumeratedAuction, policy: np.ndarray, player: int) -> float:
    """Signal-weighted expected bid level for ``player`` under ``policy``."""
    joint = auction._joint0 if player == 0 else auction._joint1
    mass = joint.sum(axis=1)              # P(infoset)
    bids = auction._bid_values
    return float(mass @ (policy @ bids))
