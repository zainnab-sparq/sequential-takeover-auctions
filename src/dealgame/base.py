"""Reusable building blocks shared by all deal games.

A "deal game" is an imperfect-information game in which players draw private
types from chance and then split a deal surplus. Two pieces are notoriously
error-prone and identical across such games, so they live here once:

1. **Information-set strings.** The single most common silent bug in
   imperfect-information games is leaking an opponent's private draw into a
   player's information state, which quietly makes the game easier than intended
   and invalidates every exploitability number. ``info_string`` centralizes the
   construction so a player only ever sees their own private tokens plus public
   tokens.

2. **The payoff contract.** Two renderings, with different strategic content:

   - ``zero_sum_returns`` handles a *constant*-sum pie split by subtracting the
     equal-split reference. This is a per-player affine shift, so it preserves
     every strategic incentive while making returns sum to zero.

   - ``zero_sum_from_profits`` handles a *general*-sum game (e.g. two bidders in
     an auction, where both can lose) by scoring the profit *difference*. This is
     NOT incentive-preserving: subtracting the opponent's (endogenous) profit adds
     a rivalry/spite term, so the Nash equilibrium of the difference game differs
     in general from the Bayes-Nash equilibrium of the underlying auction. We use
     it because it makes exact two-player exploitability and the seed paper's
     head-to-head benchmark applicable; ``general_sum_returns`` keeps the raw
     profits for the undistorted (general-sum) treatment.
"""

from __future__ import annotations

from typing import Sequence

_EQUAL_SPLIT = 0.5


def zero_sum_returns(acquirer_share: float) -> list[float]:
    """Two-player constant-sum split rendered strictly zero-sum.

    ``acquirer_share`` and the implied target share lie in [0, 1] and sum to 1.
    Subtracting the equal split makes the two returns sum to 0.
    """
    target_share = 1.0 - acquirer_share
    return [acquirer_share - _EQUAL_SPLIT, target_share - _EQUAL_SPLIT]


def general_sum_returns(player_shares: Sequence[float]) -> list[float]:
    """Raw shares, for the general-sum payoff contract (n-player extension)."""
    return list(player_shares)


def zero_sum_from_profits(profit0: float, profit1: float) -> list[float]:
    """Two-player zero-sum *relative* scoring from raw profits.

    Scores the game as one player's profit minus the other's, which benchmarks two
    strategies head-to-head exactly as the seed paper does and lets exact two-player
    exploitability apply. Caveat: this is a relativization of a general-sum auction,
    not an affine shift of it. Subtracting the opponent's endogenous profit injects
    a rivalry term, so the equilibrium of this difference game is the equilibrium of
    the relativized game, not the auction's Bayes-Nash equilibrium. Use
    :func:`general_sum_returns` / ``raw_profits`` for the undistorted object.
    """
    return [profit0 - profit1, profit1 - profit0]


def info_string(player: int, private_tokens: dict, public_tokens: dict) -> str:
    """Build a perfect-recall information-state string.

    ``private_tokens`` are visible only to ``player``; ``public_tokens`` are
    visible to everyone. Tokens whose value is ``None`` (not yet revealed) are
    omitted so that earlier and later decision nodes map to distinct info sets.
    """
    parts = [f"P{player}"]
    for key in sorted(private_tokens):
        value = private_tokens[key]
        if value is not None:
            parts.append(f"{key}={value}")
    for key in sorted(public_tokens):
        value = public_tokens[key]
        if value is not None:
            parts.append(f"{key}={value}")
    return "|".join(parts)
