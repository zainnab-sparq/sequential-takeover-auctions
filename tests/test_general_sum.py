"""Pin the vectorized general-sum model to the real OpenSpiel game.

``dealgame.general_sum.EnumeratedAuction`` reimplements the auction's payoff and
signal logic as dense tensors for speed. These tests guarantee it cannot silently
diverge from :class:`dealgame.takeover.TakeoverAuctionState`: for the same
signal-dependent policy, the expected own profit computed by enumerating the
OpenSpiel tree must equal the tensor computation, across several game shapes.
"""

import itertools

import numpy as np
import pyspiel
import pytest

import dealgame  # noqa: F401  (registers the game)
from dealgame.general_sum import EnumeratedAuction, own_profit_fictitious_play

GAME_NAME = "dealgame_takeover_auction"


def _openspiel_own_profit(game, policy_fn):
    """Expected own profit per player by enumerating the OpenSpiel tree."""
    acc = [0.0, 0.0]

    def rec(state, reach):
        if state.is_terminal():
            profit = state.raw_profits()
            acc[0] += reach * profit[0]
            acc[1] += reach * profit[1]
            return
        if state.is_chance_node():
            for action, prob in state.chance_outcomes():
                rec(state.child(action), reach * prob)
            return
        player = state.current_player()
        for action, prob in policy_fn(state, player).items():
            if prob > 0.0:
                rec(state.child(action), reach * prob)

    rec(game.new_initial_state(), 1.0)
    return acc


def _first_signal_policy_fn(num_bids):
    """Deterministic policy: bid = min(first signal index, num_bids - 1)."""
    def policy_fn(state, player):
        first_signal = state._signals[player][0]
        return {min(first_signal, num_bids - 1): 1.0}
    return policy_fn


def _first_signal_policy_matrix(num_values, num_bids, num_signals):
    tuples = list(itertools.product(range(num_values), repeat=num_signals))
    policy = np.zeros((len(tuples), num_bids))
    for row, tup in enumerate(tuples):
        policy[row, min(tup[0], num_bids - 1)] = 1.0
    return policy


@pytest.mark.parametrize("num_values,num_bids,k,noise0,noise1,toehold", [
    (3, 4, 1, 0.5, 0.5, 0.0),
    (4, 5, 1, 0.2, 0.7, 0.0),
    (3, 4, 2, 0.5, 0.5, 0.3),
    (5, 6, 1, 0.0, 0.5, 0.4),
    (3, 5, 2, 0.6, 0.3, 0.0),
])
def test_enumerated_matches_openspiel(num_values, num_bids, k, noise0, noise1, toehold):
    params = {"num_values": num_values, "num_bids": num_bids,
              "signal_noise_0": noise0, "signal_noise_1": noise1,
              "toehold": toehold, "num_signals": k}
    game = pyspiel.load_game(GAME_NAME, params)
    tree_v0, tree_v1 = _openspiel_own_profit(game, _first_signal_policy_fn(num_bids))

    auction = EnumeratedAuction(
        num_values=num_values, num_bids=num_bids, num_signals_0=k, num_signals_1=k,
        noise_0=noise0, noise_1=noise1, toehold=toehold)
    pol = _first_signal_policy_matrix(num_values, num_bids, k)
    tensor_v0, tensor_v1 = auction.own_profit_value(pol, pol)

    assert tensor_v0 == pytest.approx(tree_v0, abs=1e-9)
    assert tensor_v1 == pytest.approx(tree_v1, abs=1e-9)


def test_fictitious_play_converges_and_is_an_equilibrium():
    auction = EnumeratedAuction(num_values=5, num_bids=6)
    result = own_profit_fictitious_play(auction)
    # NashConv is the summed own-profit gain from a unilateral deviation.
    assert result["converged"], "fictitious play did not reach the tolerance"
    assert result["nashconv"] < 1e-2, "fictitious play did not approach equilibrium"
    # Equilibrium values must be non-negative (a bidder can guarantee 0 by bidding 0).
    assert result["value0"] >= -1e-9 and result["value1"] >= -1e-9


# The fully-informed corner (noise=0) cycles slowly and plateaus near NashConv 8e-3
# rather than reaching tolerance; assert that documented near-equilibrium bound
# explicitly so a comparative static is never read off a badly unconverged solve.
NEAR_EQ_NASHCONV = 1e-2


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_more_information_weakly_helps():
    """At the own-profit equilibrium, a sharper signal must not lower a bidder's value."""
    sharp = EnumeratedAuction(num_values=5, num_bids=6, noise_0=0.0, noise_1=0.5)
    dull = EnumeratedAuction(num_values=5, num_bids=6, noise_0=1.0, noise_1=0.5)
    res_sharp = own_profit_fictitious_play(sharp)
    res_dull = own_profit_fictitious_play(dull)
    assert res_sharp["nashconv"] < NEAR_EQ_NASHCONV  # fully-informed corner: near-eq
    assert res_dull["converged"]
    assert res_sharp["value0"] >= res_dull["value0"] - 1e-6


def test_useless_signals_do_not_change_value():
    """Validates the asymmetric-k path (which OpenSpiel cannot build): extra
    pure-noise signals carry no information, so a bidder's equilibrium value must
    not depend on how many of them it holds."""
    one = EnumeratedAuction(num_values=4, num_bids=5, num_signals_0=1, num_signals_1=1,
                            noise_0=1.0, noise_1=0.5)
    three = EnumeratedAuction(num_values=4, num_bids=5, num_signals_0=3, num_signals_1=1,
                              noise_0=1.0, noise_1=0.5)
    res_one = own_profit_fictitious_play(one)
    res_three = own_profit_fictitious_play(three)
    assert res_one["converged"] and res_three["converged"]
    assert res_three["value0"] == pytest.approx(res_one["value0"], abs=1e-6)


def test_more_diligence_weakly_helps():
    """More informative signals (asymmetric k, rival fixed) must not lower value."""
    base = EnumeratedAuction(num_values=5, num_bids=6, num_signals_0=1, num_signals_1=1)
    more = EnumeratedAuction(num_values=5, num_bids=6, num_signals_0=3, num_signals_1=1)
    res_base = own_profit_fictitious_play(base)
    res_more = own_profit_fictitious_play(more)
    assert res_base["converged"] and res_more["converged"]
    assert res_more["value0"] >= res_base["value0"] - 1e-6
