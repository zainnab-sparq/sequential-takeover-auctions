"""Smoke tests for the deep self-play solvers.

Kept to small episode budgets so they run quickly; they check the training loop,
exploitability wiring, and that the deep policy gradient improves, not that the
agents reach equilibrium (which needs far more episodes).
"""

import math

import pyspiel
import pytest

import dealgame  # noqa: F401
from dealgame.deep_solving import train_nfsp, train_policy_gradient

GAME_NAME = "dealgame_takeover_auction"


@pytest.fixture
def game():
    return pyspiel.load_game(GAME_NAME)


def _valid_curve(curve, required_keys):
    assert curve, "empty curve"
    for row in curve:
        assert row["episode"] >= 1 and row["seconds"] >= 0.0
        for key in required_keys:
            assert math.isfinite(row[key]) and row[key] >= 0.0


def test_policy_gradient_runs_and_improves(game):
    curve = train_policy_gradient(
        game, episodes=8000, eval_every=4000, loss_str="rpg",
        pi_lr=0.005, critic_lr=0.05)
    _valid_curve(curve, ("last",))  # "avg" only appears after the burn-in
    assert any("avg" in row for row in curve), "tail-average never recorded"
    # Last iterate should improve from the (near-random) start.
    assert curve[-1]["last"] <= curve[0]["last"] + 1e-6, \
        "deep PG did not improve exploitability"


def test_nfsp_runs(game):
    curve = train_nfsp(game, episodes=6000, eval_every=3000)
    _valid_curve(curve, ("nfsp",))
