"""Tests for the three solvers, including the from-scratch policy gradient."""

import pyspiel
import pytest

import dealgame  # noqa: F401  (registers the game)
from dealgame.solving import ReinforcePolicyGradient, run_cfr, run_mmd

GAME_NAME = "dealgame_takeover_auction"


@pytest.fixture
def game():
    return pyspiel.load_game(GAME_NAME)


def test_run_cfr_converges(game):
    curve = run_cfr(game, iterations=200, eval_every=50)
    assert curve[-1][1] < 0.01


def test_run_mmd_converges(game):
    curve = run_mmd(game, iterations=300, eval_every=100, alpha=0.0, stepsize=1.0)
    assert curve[-1][1] < 0.05


def test_reinforce_is_competitive(game):
    """Our generic policy gradient should reach low exploitability and improve."""
    pg = ReinforcePolicyGradient(game, lr=0.5, entropy_coef=0.01, batch=200, seed=0)
    curve = pg.run(iterations=150, eval_every=50)
    first, last = curve[0][1], curve[-1][1]
    assert last < first, f"PG did not improve: {first} -> {last}"
    assert last < 0.05, f"PG exploitability too high: {last}"
