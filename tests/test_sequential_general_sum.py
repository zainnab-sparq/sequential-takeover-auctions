"""Pin the extensive-form own-profit solver to the real OpenSpiel game.

``dealgame.sequential_general_sum`` caches the sequential auction's tree in a
pyspiel-free form so fictitious play can iterate on it cheaply. These tests
guarantee the cache and the best-response computation cannot silently diverge
from :class:`dealgame.sequential_takeover.SequentialTakeoverState`: expected own
profit from the cache must equal a direct enumeration of the OpenSpiel tree, and
the best response must equal a brute-force search over every pure policy on a
game small enough to enumerate them all.
"""

import itertools

import numpy as np
import pyspiel
import pytest

import dealgame  # noqa: F401  (registers the game)
from dealgame.sequential_general_sum import (
    SequentialAuctionTree,
    equilibrium_statistics,
    opening_bid_profit_curve,
    opening_infosets,
    own_profit_best_response,
    own_profit_fictitious_play,
    policy_value,
)

GAME_NAME = "dealgame_sequential_takeover"
SMALL = {"num_values": 2, "num_bids": 3, "num_rounds": 2}
TINY = {"num_values": 2, "num_bids": 2, "num_rounds": 1}


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


def _uniform_policy_fn(state, player):
    legal = state.legal_actions()
    return {action: 1.0 / len(legal) for action in legal}


def _tree_policy_fn(tree, policies):
    """Turn the solver's per-infoset policies into an OpenSpiel policy function."""
    def policy_fn(state, player):
        iset = state.information_state_string(player)
        index = tree.infoset_index(player, iset)
        probs = policies[player][index]
        actions = tree.infoset_actions(player, index)
        return dict(zip(actions, probs))
    return policy_fn


@pytest.mark.parametrize("params", [
    SMALL,
    dict(SMALL, toehold=0.3),
    dict(SMALL, num_rounds=1),
    dict(SMALL, num_rounds=3, num_bids=4),
    {"num_values": 3, "num_bids": 4, "num_rounds": 2, "num_signals": 2,
     "signal_noise_0": 0.2, "signal_noise_1": 0.7, "toehold": 0.25},
])
def test_cached_tree_matches_openspiel(params):
    game = pyspiel.load_game(GAME_NAME, params)
    tree = SequentialAuctionTree(game)
    uniform = [tree.uniform_policy(0), tree.uniform_policy(1)]

    tree_v0, tree_v1 = policy_value(tree, uniform[0], uniform[1])
    spiel_v0, spiel_v1 = _openspiel_own_profit(game, _uniform_policy_fn)

    assert tree_v0 == pytest.approx(spiel_v0, abs=1e-9)
    assert tree_v1 == pytest.approx(spiel_v1, abs=1e-9)


def test_best_response_matches_brute_force_over_pure_policies():
    """On a game small enough to enumerate every pure policy, the best response's
    own-profit value must equal the best value any pure policy attains."""
    game = pyspiel.load_game(GAME_NAME, TINY)
    tree = SequentialAuctionTree(game)
    opponent = tree.uniform_policy(1)

    br_policy, br_value = own_profit_best_response(tree, 0, opponent)

    n_infosets = tree.num_infosets(0)
    action_choices = [tree.infoset_actions(0, i) for i in range(n_infosets)]
    best = -np.inf
    for combo in itertools.product(*action_choices):
        pure = [np.zeros(len(action_choices[i])) for i in range(n_infosets)]
        for i, action in enumerate(combo):
            pure[i][action_choices[i].index(action)] = 1.0
        value, _ = policy_value(tree, pure, opponent)
        best = max(best, value)

    assert br_value == pytest.approx(best, abs=1e-9)
    # And the returned policy must actually attain that value.
    attained, _ = policy_value(tree, br_policy, opponent)
    assert attained == pytest.approx(br_value, abs=1e-9)


def test_best_response_is_at_least_as_good_as_the_policy_it_replaces():
    game = pyspiel.load_game(GAME_NAME, SMALL)
    tree = SequentialAuctionTree(game)
    uniform0, uniform1 = tree.uniform_policy(0), tree.uniform_policy(1)
    _, br_value = own_profit_best_response(tree, 0, uniform1)
    uniform_value, _ = policy_value(tree, uniform0, uniform1)
    assert br_value >= uniform_value - 1e-9


def test_fictitious_play_converges_to_an_equilibrium():
    game = pyspiel.load_game(GAME_NAME, SMALL)
    tree = SequentialAuctionTree(game)
    result = own_profit_fictitious_play(tree)
    assert result["converged"], "fictitious play did not reach the tolerance"
    assert result["nashconv"] < 1e-2
    # A bidder can guarantee zero by passing, so equilibrium value cannot be negative.
    assert result["value0"] >= -1e-9 and result["value1"] >= -1e-9


def test_equilibrium_policies_are_consistent_with_the_openspiel_game():
    """The solved policies, replayed on the real game, reproduce the solver's values."""
    game = pyspiel.load_game(GAME_NAME, dict(SMALL, toehold=0.2))
    tree = SequentialAuctionTree(game)
    result = own_profit_fictitious_play(tree)
    policies = [result["policy0"], result["policy1"]]
    spiel_v0, spiel_v1 = _openspiel_own_profit(game, _tree_policy_fn(tree, policies))
    assert spiel_v0 == pytest.approx(result["value0"], abs=1e-9)
    assert spiel_v1 == pytest.approx(result["value1"], abs=1e-9)


def test_statistics_are_a_coherent_distribution_over_outcomes():
    game = pyspiel.load_game(GAME_NAME, dict(SMALL, toehold=0.25))
    tree = SequentialAuctionTree(game)
    result = own_profit_fictitious_play(tree)
    stats = equilibrium_statistics(tree, result["policy0"], result["policy1"])

    total = stats["p_win0"] + stats["p_win1"] + stats["p_no_sale"]
    assert total == pytest.approx(1.0, abs=1e-9)
    assert stats["value0"] == pytest.approx(result["value0"], abs=1e-9)
    assert stats["value1"] == pytest.approx(result["value1"], abs=1e-9)
    # Every reported probability is a probability.
    for key in ("p_win0", "p_win1", "p_no_sale", "p_open_wait0", "p_rival_deterred"):
        assert -1e-9 <= stats[key] <= 1.0 + 1e-9, f"{key}={stats[key]}"


def test_opening_bid_is_reported_in_money_not_bid_index():
    """With a refined grid the bid index is not the price; the statistic must be money."""
    game = pyspiel.load_game(GAME_NAME, dict(SMALL, num_bids=5, bid_step=0.5))
    tree = SequentialAuctionTree(game)

    open_at_two = []
    for index in range(tree.num_infosets(0)):
        actions = tree.infoset_actions(0, index)
        probs = np.zeros(len(actions))
        probs[actions.index(2) if 2 in actions else 0] = 1.0
        open_at_two.append(probs)

    stats = equilibrium_statistics(tree, open_at_two, tree.uniform_policy(1))
    # Bid index 2 at a step of 0.5 is a price of 1.0, not 2.
    assert stats["expected_opening_bid0"] == pytest.approx(1.0, abs=1e-9)


def test_fictitious_play_from_a_random_start_finds_the_same_value():
    """Equilibrium selection: a general-sum game can have many equilibria, so the
    economics must not depend on where fictitious play happens to start."""
    game = pyspiel.load_game(GAME_NAME, dict(SMALL, toehold=0.3))
    tree = SequentialAuctionTree(game)

    from_uniform = own_profit_fictitious_play(tree)
    rng = np.random.default_rng(20260713)
    from_random = own_profit_fictitious_play(
        tree, initial_policies=[tree.random_policy(0, rng), tree.random_policy(1, rng)])

    assert from_uniform["converged"] and from_random["converged"]
    assert from_random["value0"] == pytest.approx(from_uniform["value0"], abs=1e-2)
    assert from_random["value1"] == pytest.approx(from_uniform["value1"], abs=1e-2)


def test_random_policy_is_a_valid_distribution():
    game = pyspiel.load_game(GAME_NAME, SMALL)
    tree = SequentialAuctionTree(game)
    rng = np.random.default_rng(7)
    for player in (0, 1):
        policy = tree.random_policy(player, rng)
        for index, probs in enumerate(policy):
            assert len(probs) == len(tree.infoset_actions(player, index))
            assert probs.sum() == pytest.approx(1.0, abs=1e-12)
            assert (probs > 0).all(), "full support keeps every information set reachable"


def test_deterrence_statistic_counts_an_immediate_concession():
    """If bidder 0 always opens high and bidder 1 always passes, deterrence is 1."""
    game = pyspiel.load_game(GAME_NAME, SMALL)
    tree = SequentialAuctionTree(game)
    pass_action = tree.pass_action

    def pure(player, choose):
        policy = []
        for index in range(tree.num_infosets(player)):
            actions = tree.infoset_actions(player, index)
            probs = np.zeros(len(actions))
            probs[actions.index(choose(actions))] = 1.0
            policy.append(probs)
        return policy

    # Bidder 0 always takes the highest available bid; bidder 1 always passes.
    always_high = pure(0, lambda actions: max(a for a in actions if a != pass_action)
                       if any(a != pass_action for a in actions) else pass_action)
    always_pass = pure(1, lambda actions: pass_action)

    stats = equilibrium_statistics(tree, always_high, always_pass)
    assert stats["p_open_wait0"] == pytest.approx(0.0, abs=1e-9)
    assert stats["p_rival_deterred"] == pytest.approx(1.0, abs=1e-9)
    assert stats["p_win0"] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Constrained best response: the exact "what would preempting be worth?" object.
#
# The headline economics rest on bidder 0's profit from jump-bidding versus
# standing back. Reading that off a fictitious-play solve is circular when the
# solve is exactly what we suspect of failing, so these pin an *exact* answer:
# force the opening bid, let every later decision best-respond, and measure.
# ---------------------------------------------------------------------------


def test_opening_infosets_are_the_first_bidding_decisions():
    game = pyspiel.load_game(GAME_NAME, SMALL)
    tree = SequentialAuctionTree(game)

    opening = opening_infosets(tree, 0)

    # One per signal draw bidder 0 could hold, and nothing else.
    assert len(opening) == SMALL["num_values"] ** game.num_signals
    # No bid stands yet, so every bid plus PASS is legal at each of them.
    for index in opening:
        assert len(tree.infoset_actions(0, index)) == SMALL["num_bids"] + 1


def test_forced_best_response_reproduces_the_unconstrained_value_when_it_agrees():
    """Forcing the action the best response would have picked anyway must change
    nothing. This is what makes the forced sweep comparable to the free solve."""
    game = pyspiel.load_game(GAME_NAME, dict(SMALL, toehold=0.3))
    tree = SequentialAuctionTree(game)
    opponent = tree.uniform_policy(1)

    free_policy, free_value = own_profit_best_response(tree, 0, opponent)
    forced = {index: int(np.argmax(free_policy[index]))
              for index in opening_infosets(tree, 0)}
    forced = {index: tree.infoset_actions(0, index)[action]
              for index, action in forced.items()}

    _, forced_value = own_profit_best_response(tree, 0, opponent, forced=forced)

    assert forced_value == pytest.approx(free_value, abs=1e-9)


def test_forcing_an_opening_bid_never_beats_the_free_best_response():
    game = pyspiel.load_game(GAME_NAME, dict(SMALL, toehold=0.3))
    tree = SequentialAuctionTree(game)
    opponent = tree.uniform_policy(1)
    _, free_value = own_profit_best_response(tree, 0, opponent)

    for action in tree.infoset_actions(0, opening_infosets(tree, 0)[0]):
        forced = {index: action for index in opening_infosets(tree, 0)}
        _, value = own_profit_best_response(tree, 0, opponent, forced=forced)
        assert value <= free_value + 1e-9


def test_forced_best_response_matches_brute_force_over_pure_policies():
    """The gold standard, mirroring the unconstrained brute-force pin: with the
    opening forced, the constrained best response must equal the best value any
    pure policy that opens that way can attain."""
    game = pyspiel.load_game(GAME_NAME, TINY)
    tree = SequentialAuctionTree(game)
    opponent = tree.uniform_policy(1)
    opening = opening_infosets(tree, 0)
    n_infosets = tree.num_infosets(0)
    action_choices = [tree.infoset_actions(0, i) for i in range(n_infosets)]

    for forced_action in action_choices[opening[0]]:
        forced = {index: forced_action for index in opening}
        _, value = own_profit_best_response(tree, 0, opponent, forced=forced)

        best = -np.inf
        for combo in itertools.product(*action_choices):
            if any(combo[index] != forced_action for index in opening):
                continue
            pure = [np.zeros(len(action_choices[i])) for i in range(n_infosets)]
            for i, action in enumerate(combo):
                pure[i][action_choices[i].index(action)] = 1.0
            candidate, _ = policy_value(tree, pure, opponent)
            best = max(best, candidate)

        assert value == pytest.approx(best, abs=1e-9)


def test_forcing_an_illegal_action_is_an_error():
    game = pyspiel.load_game(GAME_NAME, SMALL)
    tree = SequentialAuctionTree(game)
    index = opening_infosets(tree, 0)[0]
    illegal = max(tree.infoset_actions(0, index)) + 1

    with pytest.raises(ValueError):
        own_profit_best_response(tree, 0, tree.uniform_policy(1),
                                 forced={index: illegal})


def test_opening_bid_profit_curve_peaks_at_the_free_best_response_value():
    """The curve sweeps every opening the bidder could commit to, so its maximum
    is exactly what an unconstrained best response earns."""
    game = pyspiel.load_game(GAME_NAME, dict(SMALL, toehold=0.2))
    tree = SequentialAuctionTree(game)
    opponent = tree.uniform_policy(1)
    _, free_value = own_profit_best_response(tree, 0, opponent)

    bids, profits = opening_bid_profit_curve(tree, opponent)

    # One entry per bid level plus the "wait" option.
    assert len(bids) == len(profits) == SMALL["num_bids"] + 1
    assert float(np.max(profits)) <= free_value + 1e-9
    # The free best response opens signal-by-signal; a single forced opening can
    # only match it when the free opening happens to be signal-independent. It is
    # never better, and here it is attained.
    assert float(np.max(profits)) == pytest.approx(free_value, abs=1e-9)


def test_opening_bid_profit_curve_reports_money_not_bid_indices():
    game = pyspiel.load_game(GAME_NAME, dict(SMALL, num_bids=5, bid_step=0.5))
    tree = SequentialAuctionTree(game)

    bids, _ = opening_bid_profit_curve(tree, tree.uniform_policy(1))

    assert bids[:5] == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])
    assert np.isnan(bids[5])  # the PASS option has no price
