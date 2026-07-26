"""Angle C toy experiment: the imperfect-information M&A on-ramp.

Produces two artifacts in ``results/``:

1. ``convergence.png`` / ``convergence.csv`` -- exploitability vs iterations for
   CFR, MMD, and our REINFORCE policy gradient on the bilateral takeover auction.
   This is the miniature test of the seed paper's thesis: does a simple generic
   policy gradient stay competitive with the specialized solvers?

2. ``asymmetry.png`` / ``asymmetry.csv`` -- comparative statics: as the acquirer's
   signal gets noisier (information asymmetry rises), how does the equilibrium
   value it can capture move? The economically expected direction is down.

Run inside the container:
    docker run --rm -v "<repo>:/work" imperfect-info:latest \
        python experiments/angle_c_toy.py
"""

from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyspiel
from open_spiel.python.algorithms import cfr, exploitability, expected_game_score

import dealgame  # noqa: F401  (registers the game)
from dealgame.solving import ReinforcePolicyGradient, run_cfr, run_mmd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
ITERATIONS = 300
EVAL_EVERY = 10
PG_ITERATIONS = 300
PG_EVAL_EVERY = 20
NOISE_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def convergence_experiment():
    game = pyspiel.load_game("dealgame_takeover_auction")
    print("Running CFR ...")
    cfr_curve = run_cfr(game, ITERATIONS, EVAL_EVERY)
    print("Running MMD ...")
    mmd_curve = run_mmd(game, ITERATIONS, EVAL_EVERY, alpha=0.0, stepsize=1.0)
    print("Running REINFORCE policy gradient ...")
    pg = ReinforcePolicyGradient(game, lr=0.5, entropy_coef=0.01, batch=256, seed=0)
    pg_curve = pg.run(PG_ITERATIONS, PG_EVAL_EVERY)

    plt.figure(figsize=(7, 5))
    for label, curve in [("CFR", cfr_curve), ("MMD", mmd_curve),
                         ("REINFORCE PG (ours)", pg_curve)]:
        xs, ys = zip(*curve)
        plt.plot(xs, ys, marker="o", markersize=3, label=label)
    plt.yscale("log")
    plt.xlabel("iterations")
    plt.ylabel("exploitability (NashConv / 2, log scale)")
    plt.title("Takeover auction: simple PG vs specialized solvers")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    out_png = os.path.join(RESULTS_DIR, "convergence.png")
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close()

    out_csv = os.path.join(RESULTS_DIR, "convergence.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "iteration", "exploitability"])
        for label, curve in [("CFR", cfr_curve), ("MMD", mmd_curve),
                             ("REINFORCE_PG", pg_curve)]:
            for it, e in curve:
                w.writerow([label, it, f"{e:.8f}"])

    print(f"  final exploitability  CFR={cfr_curve[-1][1]:.5f}  "
          f"MMD={mmd_curve[-1][1]:.5f}  PG={pg_curve[-1][1]:.5f}")
    print(f"  wrote {out_png} and {out_csv}")


def _p0_ownprofit_br_value(game, opp_pol):
    """Bidder 0's expected OWN profit under the bid that maximizes its own profit,
    holding bidder 1 fixed at ``opp_pol``.

    The zero-sum value read off the equilibrium mixes bidder 0's own incentive with
    the rivalry term that the profit-difference scoring injects. This isolates the
    undistorted objective: for each bidder-0 signal it picks the bid maximizing
    bidder 0's own (general-sum) profit against the fixed opponent, then sums the
    reach-weighted best own profit over signals. Mirrors the toehold study's
    own-profit best-response check in the main benchmark.
    """
    q = {}      # q[s0][b0] = sum of reach * own-profit over terminals
    mass = {}   # mass[s0] = P(signal s0)

    def rec(state, reach, s0, b0):
        if state.is_terminal():
            q[s0][b0] += reach * state.raw_profits()[0]
            return
        if state.is_chance_node():
            for a, p in state.chance_outcomes():
                rec(state.child(a), reach * p, s0, b0)
            return
        cur = state.current_player()
        if cur == 0:
            s0 = state.information_state_string(0)
            if s0 not in q:
                q[s0] = {a: 0.0 for a in state.legal_actions(0)}
                mass[s0] = 0.0
            mass[s0] += reach
            for a in state.legal_actions(0):
                rec(state.child(a), reach, s0, a)
        else:
            for a, p in opp_pol.action_probabilities(state).items():
                rec(state.child(a), reach * p, s0, b0)

    rec(game.new_initial_state(), 1.0, None, None)
    return sum(max(actions.values()) for actions in q.values())


def asymmetry_experiment():
    """Vary bidder 0's signal noise (bidder 1 fixed at 0.5) and read off both the
    equilibrium relative (zero-sum) value and, as a robustness check on the
    undistorted objective, bidder 0's own-profit best-response value against a fixed
    opponent. More information should be weakly better on both. The opponent is
    fixed at the symmetric (0.5/0.5) equilibrium so only bidder 0's information
    varies, mirroring the toehold study's fixed-opponent own-profit check."""
    ref_game = pyspiel.load_game(
        "dealgame_takeover_auction",
        {"signal_noise_0": 0.5, "signal_noise_1": 0.5})
    ref_solver = cfr.CFRSolver(ref_game)
    for _ in range(ITERATIONS):
        ref_solver.evaluate_and_update_policy()
    ref_opp = ref_solver.average_policy()  # bidder 1 held fixed at symmetric eq

    rows = []
    for noise in NOISE_GRID:
        game = pyspiel.load_game(
            "dealgame_takeover_auction",
            {"signal_noise_0": noise, "signal_noise_1": 0.5})
        solver = cfr.CFRSolver(game)
        for _ in range(ITERATIONS):
            solver.evaluate_and_update_policy()
        avg = solver.average_policy()
        expl = exploitability.exploitability(game, avg)
        v0, v1 = expected_game_score.policy_value(
            game.new_initial_state(), [avg, avg])
        v0_own = _p0_ownprofit_br_value(game, ref_opp)
        rows.append((noise, v0, v1, v0_own, expl))
        print(f"  bidder0 noise={noise:.2f}  zero-sum value={v0:+.4f}  "
              f"own-profit BR value={v0_own:+.4f}  exploitability={expl:.5f}")

    xs = [r[0] for r in rows]
    fig, (ax_zs, ax_op) = plt.subplots(1, 2, figsize=(12, 5))
    ax_zs.plot(xs, [r[1] for r in rows], marker="o", label="bidder 0 value")
    ax_zs.plot(xs, [r[2] for r in rows], marker="s", label="bidder 1 value (noise=0.5)")
    ax_zs.axhline(0.0, color="gray", lw=0.8)
    ax_zs.set_xlabel("bidder 0 signal noise (its information disadvantage)")
    ax_zs.set_ylabel("equilibrium relative value (zero-sum)")
    ax_zs.set_title("Zero-sum equilibrium value")
    ax_zs.legend()
    ax_zs.grid(True, alpha=0.3)
    ax_op.plot(xs, [r[3] for r in rows], marker="o", color="C2",
               label="bidder 0 own-profit BR value")
    ax_op.set_xlabel("bidder 0 signal noise (its information disadvantage)")
    ax_op.set_ylabel("own (general-sum) profit, fixed opponent")
    ax_op.set_title("Undistorted own-profit best response")
    ax_op.legend()
    ax_op.grid(True, alpha=0.3)
    fig.suptitle("Common-value takeover auction: information advantage and value")
    out_png = os.path.join(RESULTS_DIR, "asymmetry.png")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)

    out_csv = os.path.join(RESULTS_DIR, "asymmetry.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["bidder0_noise", "bidder0_value", "bidder1_value",
                    "bidder0_ownprofit_br", "exploitability"])
        for noise, v0, v1, v0_own, e in rows:
            w.writerow([noise, f"{v0:.6f}", f"{v1:.6f}", f"{v0_own:.6f}", f"{e:.8f}"])
    print(f"  wrote {out_png} and {out_csv}")


def main():
    _ensure_results_dir()
    print("== Convergence experiment ==")
    convergence_experiment()
    print("== Information-asymmetry comparative statics ==")
    asymmetry_experiment()
    print("Done.")


if __name__ == "__main__":
    main()
