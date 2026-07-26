"""Smoke tests for the added baselines: Deep CFR, PSRO, and PPO.

Small budgets keep them fast; they check the wiring, exploitability evaluation,
and that each method learns something better than random on the small game.
"""

import math

import pyspiel
import pytest

import dealgame  # noqa: F401
from dealgame.deep_solving import train_deep_cfr, train_psro
from dealgame.intractable import (UniformPolicy, approx_exploitability,
                                  train_intractable)
from dealgame.ppo_solving import train_ppg, train_ppo

GAME_NAME = "dealgame_takeover_auction"
# Random-play exploitability on the small game is ~0.5; learners must beat this.
RANDOM_LEVEL = 0.45


@pytest.fixture
def game():
    return pyspiel.load_game(GAME_NAME, {"num_values": 3, "num_bids": 4})


def test_deep_cfr_runs(game):
    # Wiring smoke only: Deep CFR needs a large budget to converge (slow), which
    # the benchmark exercises; here we just check it runs and returns a sane value.
    row = train_deep_cfr(game, num_iterations=10, num_traversals=20,
                         advantage_train_steps=50, policy_train_steps=50)[0]
    assert math.isfinite(row["deepcfr"]) and 0.0 <= row["deepcfr"] <= 1.0
    assert row["seconds"] >= 0.0


def test_psro_runs_and_converges(game):
    curve = train_psro(game, iterations=8, eval_every=4)
    assert curve and all(math.isfinite(r["psro"]) for r in curve)
    # Exact best-response PSRO should drive exploitability low on a small game.
    assert curve[-1]["psro"] < 0.1


def test_ppo_runs_and_learns(game):
    curve = train_ppo(game, episodes=20000, eval_every=5000, batch_episodes=128)
    assert curve, "empty curve"
    for row in curve:
        assert row["episode"] >= 1 and row["seconds"] >= 0.0
        assert math.isfinite(row["last"]) and row["last"] >= 0.0
    assert any("avg" in r for r in curve), "tail-average never recorded"
    assert curve[-1]["last"] < RANDOM_LEVEL


def test_ppg_runs_and_learns(game):
    # n_policy small so an auxiliary phase actually fires within the budget.
    curve = train_ppg(game, episodes=20000, eval_every=5000, batch_episodes=128,
                      n_policy=4, aux_epochs=3)
    assert curve, "empty curve"
    assert any("avg" in r for r in curve), "tail-average never recorded"
    assert math.isfinite(curve[-1]["last"]) and curve[-1]["last"] < RANDOM_LEVEL


def test_approx_exploitability_detects_uniform(game):
    # Uniform play is highly exploitable; the learned BR must recover a large gain.
    approx, gains, _ = approx_exploitability(
        game, UniformPolicy(game), br_batches=60, br_batch_episodes=128,
        mc_episodes=3000, seed=0)
    assert approx > 0.15, f"learned BR failed to exploit uniform: {approx}"


def test_train_intractable_runs(game):
    # Tabulation-free training returns approximate-exploitability checkpoints.
    curve = train_intractable(
        game, kind="ppo", episodes=4000, batch_episodes=128, eval_points=2, seed=0,
        approx_kw=dict(br_batches=10, br_batch_episodes=64, mc_episodes=500))
    assert curve and all(math.isfinite(r["approx_expl"]) for r in curve)
