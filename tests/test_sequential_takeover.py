"""Validity tests for the sequential (multi-round ascending) takeover auction."""

import pyspiel
import pytest
from open_spiel.python.algorithms import cfr, exploitability, get_all_states

import dealgame  # noqa: F401  (registers the game)

GAME_NAME = "dealgame_sequential_takeover"

# A small instance that is fast to enumerate: 2 values, 3 bid levels, 2 rounds.
SMALL = {"num_values": 2, "num_bids": 3, "num_rounds": 2}


@pytest.fixture
def game():
    return pyspiel.load_game(GAME_NAME, SMALL)


def _deal(game, w, sig0, sig1):
    """Advance past the chance nodes to the first bidding decision."""
    state = game.new_initial_state()
    state.apply_action(w)
    state.apply_action(sig0)
    state.apply_action(sig1)
    assert state.current_player() == 0
    return state


def test_game_loads(game):
    assert game.num_players() == 2
    gt = game.get_type()
    assert gt.information == pyspiel.GameType.Information.IMPERFECT_INFORMATION
    assert gt.utility == pyspiel.GameType.Utility.ZERO_SUM


def test_all_terminals_are_zero_sum(game):
    states = get_all_states.get_all_states(
        game, depth_limit=-1, include_terminals=True, include_chance_states=False)
    terminals = [s for s in states.values() if s.is_terminal()]
    assert terminals, "expected terminal states"
    for state in terminals:
        returns = state.returns()
        assert abs(returns[0] + returns[1]) < 1e-9, f"not zero-sum: {returns}"


def test_opening_legal_actions_are_every_bid_plus_pass(game):
    state = _deal(game, w=0, sig0=0, sig1=0)
    # 3 bid levels (actions 0,1,2) plus PASS (action 3).
    assert state.legal_actions() == [0, 1, 2, 3]


def test_a_raise_must_strictly_exceed_the_standing_bid(game):
    state = _deal(game, w=0, sig0=0, sig1=0)
    state.apply_action(1)  # bidder 0 opens at level 1
    assert state.current_player() == 1
    # Bidder 1 may only jump to level 2, or pass.
    assert state.legal_actions() == [2, 3]


def test_passing_while_behind_ends_the_auction(game):
    state = _deal(game, w=1, sig0=0, sig1=0)
    state.apply_action(1)  # bidder 0 opens at level 1
    state.apply_action(3)  # bidder 1 passes
    assert state.is_terminal()
    # Bidder 0 wins at its own standing bid of 1; W index 1 is worth 2.
    assert state.raw_profits() == pytest.approx([2.0 - 1.0, 0.0])


def test_both_passing_leaves_no_sale(game):
    state = _deal(game, w=1, sig0=0, sig1=0)
    state.apply_action(3)  # bidder 0 passes with no standing bid
    assert not state.is_terminal()
    assert state.current_player() == 1
    state.apply_action(3)  # bidder 1 passes too
    assert state.is_terminal()
    assert state.raw_profits() == pytest.approx([0.0, 0.0])


def test_round_limit_ends_the_auction(game):
    """With num_rounds=2 each bidder gets two turns; the last raise stands."""
    state = _deal(game, w=1, sig0=0, sig1=0)
    state.apply_action(0)  # turn 0, bidder 0 opens at 0
    state.apply_action(1)  # turn 1, bidder 1 raises to 1
    state.apply_action(2)  # turn 2, bidder 0 raises to 2
    assert state.current_player() == 1
    assert state.legal_actions() == [3]  # nothing left above level 2 but PASS
    state.apply_action(3)  # turn 3, bidder 1 passes; bidder 0 wins at 2
    assert state.is_terminal()
    assert state.raw_profits() == pytest.approx([2.0 - 2.0, 0.0])


def test_a_bidder_can_face_more_than_one_decision(game):
    """Multi-decision play is the whole point; PPO collection depends on it."""
    state = _deal(game, w=1, sig0=0, sig1=0)
    state.apply_action(0)
    state.apply_action(1)
    assert state.current_player() == 0  # bidder 0 acts a second time


def test_information_set_hides_the_opponent_signal(game):
    a = _deal(game, w=0, sig0=1, sig1=0)
    b = _deal(game, w=1, sig0=1, sig1=1)
    assert a.information_state_string(0) == b.information_state_string(0)


def test_information_set_reveals_the_bid_history(game):
    """Bids are public: a different history is a different information set."""
    a = _deal(game, w=0, sig0=1, sig1=0)
    a.apply_action(0)  # bidder 0 opens low
    b = _deal(game, w=0, sig0=1, sig1=0)
    b.apply_action(2)  # bidder 0 jumps high
    assert a.information_state_string(1) != b.information_state_string(1)


def test_information_state_has_perfect_recall(game):
    """A player's own earlier action is recoverable from its later info set."""
    a = _deal(game, w=0, sig0=0, sig1=0)
    a.apply_action(0)  # bidder 0 opens at 0
    a.apply_action(1)  # bidder 1 raises to 1
    b = _deal(game, w=0, sig0=0, sig1=0)
    b.apply_action(3)  # bidder 0 passes
    b.apply_action(1)  # bidder 1 opens at 1
    # Same standing bid and same turn, but bidder 0 played differently, so these
    # must not collapse into one information set.
    assert a.current_player() == 0 and b.current_player() == 0
    assert a.information_state_string(0) != b.information_state_string(0)


def test_toehold_loser_collects_its_share_of_the_price():
    game = pyspiel.load_game(GAME_NAME, dict(SMALL, toehold=0.25))
    state = _deal(game, w=1, sig0=0, sig1=0)
    state.apply_action(0)  # bidder 0 opens at 0
    state.apply_action(2)  # bidder 1 jumps to 2
    state.apply_action(3)  # bidder 0 passes; bidder 1 wins at 2
    assert state.is_terminal()
    # Bidder 1 pays 2 for a target worth 2; bidder 0 sells its 25% at that price.
    assert state.raw_profits() == pytest.approx([0.25 * 2.0, 0.0])


def test_toehold_winner_only_buys_what_it_does_not_own():
    game = pyspiel.load_game(GAME_NAME, dict(SMALL, toehold=0.25))
    state = _deal(game, w=1, sig0=0, sig1=0)
    state.apply_action(2)  # bidder 0 opens at 2
    state.apply_action(3)  # bidder 1 passes
    assert state.raw_profits() == pytest.approx([2.0 - 0.75 * 2.0, 0.0])


def test_bid_step_refines_the_price_grid():
    """bid_step decouples the bid amount from the bid index, so the grid can be
    refined over a fixed price range without inventing dominated high bids."""
    game = pyspiel.load_game(GAME_NAME, dict(SMALL, num_bids=5, bid_step=0.5))
    state = _deal(game, w=1, sig0=0, sig1=0)
    state.apply_action(3)  # bid index 3 is a price of 1.5
    state.apply_action(4)  # bidder 1 outbids at index 4, a price of 2.0
    state.apply_action(5)  # bidder 0 concedes (PASS is action num_bids)
    assert state.is_terminal()
    assert state.auction_outcome() == (1, 2.0)
    # Target worth 2 bought for 2.0.
    assert state.raw_profits() == pytest.approx([0.0, 0.0])


def test_bid_step_defaults_to_the_bid_index():
    game = pyspiel.load_game(GAME_NAME, SMALL)
    state = _deal(game, w=1, sig0=0, sig1=0)
    state.apply_action(2)
    state.apply_action(3)  # pass
    assert state.auction_outcome() == (0, 2.0)


def test_standing_bid_reports_the_leading_bid(game):
    state = _deal(game, w=0, sig0=0, sig1=0)
    assert state.standing_bid() is None
    state.apply_action(1)
    assert state.standing_bid() == 1
    state.apply_action(2)
    assert state.standing_bid() == 2


def test_auction_outcome_reports_the_winner_and_price(game):
    state = _deal(game, w=1, sig0=0, sig1=0)
    state.apply_action(0)  # bidder 0 opens at 0
    state.apply_action(2)  # bidder 1 jumps to 2
    state.apply_action(3)  # bidder 0 concedes
    assert state.auction_outcome() == (1, 2.0)


def test_auction_outcome_reports_no_sale(game):
    state = _deal(game, w=1, sig0=0, sig1=0)
    state.apply_action(3)
    state.apply_action(3)
    assert state.auction_outcome() == (None, 0.0)


def test_information_state_tensor_size(game):
    # own signal one-hot (1 x 2 values) + 4 turns x (3 bids + pass) one-hot.
    assert game.information_state_tensor_size() == 2 + 4 * 4


def test_cfr_converges(game):
    solver = cfr.CFRSolver(game)
    for _ in range(300):
        solver.evaluate_and_update_policy()
    expl = exploitability.exploitability(game, solver.average_policy())
    assert expl < 0.01, f"CFR did not converge: exploitability={expl}"


def test_ppo_collection_records_multiple_decisions_for_a_player(game):
    """The multi-step PPO path depends on _collect logging EVERY decision a player
    makes, not just its first. In a multi-round auction bidder 0 opens and then may
    act again; over many episodes that second decision must show up in the buffer,
    so the sample count for a player exceeds the episode count."""
    import numpy as np
    import torch

    from dealgame.ppo_solving import PPOAgent, _collect

    torch.manual_seed(0)
    np.random.seed(0)
    info_size = game.information_state_tensor_size()
    num_actions = game.num_distinct_actions()
    agents = [PPOAgent(p, info_size, num_actions) for p in range(2)]

    episodes = 200
    buf = _collect(game, agents, episodes)
    # A one-decision-per-episode game would give exactly `episodes` samples; more
    # means at least one episode drove a player to a second (later-round) decision.
    assert len(buf[0]["states"]) > episodes
    # Every logged sample carries the terminal return as its target (GAE gamma=lambda=1).
    assert len(buf[0]["returns"]) == len(buf[0]["states"])


def test_ppo_learns_on_the_sequential_game(game):
    """End-to-end: PPO self-play on the multi-round game must drive the tabular
    tail-average exploitability well below uniform-random play. This is the load-
    bearing assumption behind the deep benchmark, so it is checked directly rather
    than assumed from the single-round case."""
    from open_spiel.python import policy as policy_lib

    from dealgame.ppo_solving import train_ppo

    uniform_expl = exploitability.exploitability(game, policy_lib.TabularPolicy(game))
    curve = train_ppo(game, episodes=20000, eval_every=20000, batch_episodes=128,
                      seed=0)
    final = curve[-1]
    assert "avg" in final, "expected a tabular tail-average reading"
    # Learning, not just noise: at least a 40% cut below uniform on this tiny game.
    assert final["avg"] < 0.6 * uniform_expl, (
        f"PPO did not learn: avg={final['avg']:.4f} vs uniform={uniform_expl:.4f}")
