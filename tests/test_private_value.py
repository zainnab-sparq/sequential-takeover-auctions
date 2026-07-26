"""Validity tests for the private-value first-price auction."""

import pyspiel
import pytest
from open_spiel.python.algorithms import cfr, exploitability, get_all_states

import dealgame  # noqa: F401

GAME_NAME = "dealgame_private_value_auction"


@pytest.fixture
def game():
    return pyspiel.load_game(GAME_NAME)


def test_game_loads(game):
    assert game.num_players() == 2
    assert game.get_type().utility == pyspiel.GameType.Utility.ZERO_SUM


def test_all_terminals_are_zero_sum(game):
    states = get_all_states.get_all_states(
        game, depth_limit=-1, include_terminals=True, include_chance_states=False)
    terminals = [s for s in states.values() if s.is_terminal()]
    assert terminals
    for s in terminals:
        r = s.returns()
        assert abs(r[0] + r[1]) < 1e-9


def test_information_set_hides_opponent_value(game):
    def at_bid0(v0, v1):
        st = game.new_initial_state()
        st.apply_action(v0)
        st.apply_action(v1)
        assert st.current_player() == 0
        return st

    a = at_bid0(v0=2, v1=0)
    b = at_bid0(v0=2, v1=3)
    assert a.information_state_string(0) == b.information_state_string(0)
    assert a.information_state_string(0) == "P0|val=2"


def test_cfr_converges(game):
    solver = cfr.CFRSolver(game)
    for _ in range(300):
        solver.evaluate_and_update_policy()
    assert exploitability.exploitability(game, solver.average_policy()) < 0.01
