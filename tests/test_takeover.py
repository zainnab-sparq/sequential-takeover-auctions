"""Validity tests for the common-value takeover auction game."""

import pyspiel
import pytest
from open_spiel.python.algorithms import cfr, exploitability, get_all_states

import dealgame  # noqa: F401  (registers the game)

GAME_NAME = "dealgame_takeover_auction"


@pytest.fixture
def game():
    return pyspiel.load_game(GAME_NAME)


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
    for s in terminals:
        r = s.returns()
        assert abs(r[0] + r[1]) < 1e-9, f"returns not zero-sum: {r}"


def test_information_set_hides_opponent(game):
    # Two plays with the same bidder-0 signal but different W and bidder-1 signal
    # must map to the same bidder-0 information set (no leakage).
    def state_at_bid0(w, s0, s1):
        st = game.new_initial_state()
        st.apply_action(w)   # draw W
        st.apply_action(s0)  # bidder 0 signal
        st.apply_action(s1)  # bidder 1 signal
        assert st.current_player() == 0
        return st

    a = state_at_bid0(w=0, s0=1, s1=0)
    b = state_at_bid0(w=2, s0=1, s1=2)
    assert a.information_state_string(0) == b.information_state_string(0)
    assert a.information_state_string(0) == "P0|sig=1"


def test_cfr_converges(game):
    solver = cfr.CFRSolver(game)
    for _ in range(300):
        solver.evaluate_and_update_policy()
    expl = exploitability.exploitability(game, solver.average_policy())
    assert expl < 0.01, f"CFR did not converge: exploitability={expl}"


def test_multi_signal_game():
    """k>1 gives each bidder k signals; info set = own k-tuple, tensor = k one-hots."""
    k = 3
    g = pyspiel.load_game(GAME_NAME, {"num_values": 4, "num_bids": 4,
                                      "num_signals": k})
    assert g.information_state_tensor_size() == 4 * k
    st = g.new_initial_state()
    st.apply_action(0)                      # draw W
    for _ in range(2 * k):                   # draw k signals for each bidder
        assert st.is_chance_node()
        st.apply_action(st.chance_outcomes()[0][0])
    assert st.current_player() == 0
    iset = st.information_state_string(0)
    assert "sig0=" in iset and f"sig{k - 1}=" in iset
    # zero-sum still holds at a terminal
    st.apply_action(0)
    st.apply_action(0)
    r = st.returns()
    assert abs(r[0] + r[1]) < 1e-9


def test_signal_noise_changes_value():
    """A bidder with a perfect signal should do strictly better than with none."""
    from open_spiel.python.algorithms import expected_game_score

    def bidder0_value(noise0):
        g = pyspiel.load_game(
            GAME_NAME, {"signal_noise_0": noise0, "signal_noise_1": 0.5})
        solver = cfr.CFRSolver(g)
        for _ in range(300):
            solver.evaluate_and_update_policy()
        v0, _ = expected_game_score.policy_value(
            g.new_initial_state(), [solver.average_policy()] * 2)
        return v0

    assert bidder0_value(0.0) > bidder0_value(1.0) + 0.1
