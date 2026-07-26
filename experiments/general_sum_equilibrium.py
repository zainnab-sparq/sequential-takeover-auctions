"""Economic comparative statics at the true general-sum (own-profit) equilibrium.

The zero-sum solvers used elsewhere optimize the profit difference, which injects a
rivalry term; the economic questions here (the value of information, the value of
diligence) are properties of the underlying general-sum auction. This script solves
that auction's own-profit Bayes-Nash equilibrium exactly (see
:mod:`dealgame.general_sum`) and reads the comparative statics off it, so the
results do not depend on the zero-sum relativization.

Produces two artifacts in ``results/``:

1. ``asymmetry_gs.{png,csv}`` -- bidder 0's equilibrium own profit as its signal
   noise rises (bidder 1 fixed). The value-of-information comparative static
   measured in a genuine equilibrium, not as a one-sided best response.

2. ``diligence.{png,csv}`` -- bidder 0's equilibrium own profit as a function of the
   number of independent diligence signals it acquires (rival fixed at one signal).
   With a per-signal diligence cost ``c``, the profit-maximizing amount of diligence
   is ``argmax_k [value(k) - c*k]``; the second panel reports that cutoff, which is
   the question in the title.

Run inside the container:
    docker run --rm -v "<repo>:/work" imperfect-info:latest \
        python experiments/general_sum_equilibrium.py
"""

from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dealgame.general_sum import (EnumeratedAuction, expected_bid,
                                  own_profit_fictitious_play)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DILIGENCE_BASE = {"num_values": 5, "num_bids": 6}
NOISE_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
TOEHOLD_GRID = [0.0, 0.2, 0.4, 0.6]
DILIGENCE_K = [1, 2, 3, 4, 5]
RIVAL_SIGNALS = 1                # the rival's fixed diligence level
DILIGENCE_NOISE = 0.5           # quality of each individual signal
COST_GRID = [round(0.02 * i, 3) for i in range(0, 16)]  # per-signal diligence cost

# Robustness sweep for the diligence cutoff: does the lumpy (odd-signal) marginal and
# the k=2-skipping cutoff hold away from the single (num_values=5, noise=0.5) base, or
# is it an artifact of that parameterization? We keep the bid grid covering the value
# range (num_bids = num_values + 1, as in the base) so the comparison is fair across
# num_values. Noise extremes (0, 1) are excluded: noise=0 is the slow-converging
# fully-informed corner and noise=1 carries no information.
ROBUST_NUM_VALUES = [4, 5, 6]
ROBUST_NOISE = [0.3, 0.5, 0.7]
ROBUST_MAX_ITER = 400000  # larger games (nv=6, high k) need more FP iterations to converge


def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def asymmetry_gs_experiment():
    """Value of information at the own-profit equilibrium, varying bidder 0's noise."""
    print("== Information asymmetry at the general-sum equilibrium ==")
    rows = []
    for noise in NOISE_GRID:
        auction = EnumeratedAuction(
            num_values=DILIGENCE_BASE["num_values"], num_bids=DILIGENCE_BASE["num_bids"],
            noise_0=noise, noise_1=DILIGENCE_NOISE)
        res = own_profit_fictitious_play(auction)
        bid0 = expected_bid(auction, res["policy0"], 0)
        rows.append((noise, res["value0"], res["value1"], bid0, res["nashconv"]))
        print(f"  bidder0 noise={noise:.2f}  own-profit value={res['value0']:+.4f}  "
              f"rival value={res['value1']:+.4f}  bid={bid0:.3f}  "
              f"NashConv={res['nashconv']:.2e}")

    plt.figure(figsize=(7, 5))
    plt.plot([r[0] for r in rows], [r[1] for r in rows], marker="o",
             label="bidder 0 (varying information)")
    plt.plot([r[0] for r in rows], [r[2] for r in rows], marker="s",
             label="bidder 1 (fixed, noise=0.5)")
    plt.xlabel("bidder 0 signal noise (its information disadvantage)")
    plt.ylabel("equilibrium own profit (general-sum BNE)")
    plt.title("Value of information at the own-profit equilibrium")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_png = os.path.join(RESULTS_DIR, "asymmetry_gs.png")
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close()

    out_csv = os.path.join(RESULTS_DIR, "asymmetry_gs.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["bidder0_noise", "bidder0_value", "bidder1_value",
                    "bidder0_bid", "nashconv"])
        for noise, v0, v1, bid0, nc in rows:
            w.writerow([noise, f"{v0:.6f}", f"{v1:.6f}", f"{bid0:.6f}", f"{nc:.8f}"])
    print(f"  wrote {out_png} and {out_csv}")


def toehold_gs_experiment():
    """Toehold aggressiveness at the own-profit equilibrium, varying bidder 0's
    toehold. Validates the comparative static in a genuine BNE rather than as a
    one-sided best response against a fixed (zero-sum) opponent."""
    print("== Toehold aggressiveness at the general-sum equilibrium ==")
    rows = []
    for theta in TOEHOLD_GRID:
        auction = EnumeratedAuction(
            num_values=DILIGENCE_BASE["num_values"], num_bids=DILIGENCE_BASE["num_bids"],
            toehold=theta)
        res = own_profit_fictitious_play(auction)
        bid0 = expected_bid(auction, res["policy0"], 0)
        rows.append((theta, bid0, res["value0"], res["nashconv"]))
        print(f"  toehold={theta:.1f}  bidder0 eq bid={bid0:.3f}  "
              f"value={res['value0']:+.4f}  NashConv={res['nashconv']:.2e}")

    fig, (ax_v, ax_b) = plt.subplots(1, 2, figsize=(12, 5))
    ax_v.plot([r[0] for r in rows], [r[2] for r in rows], marker="o", color="C2")
    ax_v.set_xlabel("bidder 0 toehold $\\theta$")
    ax_v.set_ylabel("equilibrium own profit (general-sum BNE)")
    ax_v.set_title("A toehold raises the holder's profit")
    ax_v.grid(True, alpha=0.3)
    ax_b.plot([r[0] for r in rows], [r[1] for r in rows], marker="s", color="C0")
    ax_b.set_xlabel("bidder 0 toehold $\\theta$")
    ax_b.set_ylabel("equilibrium expected bid")
    ax_b.set_title("but its equilibrium bid is competed flat")
    ax_b.set_ylim(0, DILIGENCE_BASE["num_bids"] - 1)
    ax_b.grid(True, alpha=0.3)
    fig.suptitle("Toehold at the own-profit equilibrium: value rises, aggressiveness is competed away")
    fig.savefig(os.path.join(RESULTS_DIR, "toehold_gs.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    out_csv = os.path.join(RESULTS_DIR, "toehold_gs.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["toehold", "bidder0_bid", "bidder0_value", "nashconv"])
        for theta, bid0, v0, nc in rows:
            w.writerow([theta, f"{bid0:.6f}", f"{v0:.6f}", f"{nc:.8f}"])
    print(f"  wrote results/toehold_gs.{{png,csv}}")


def _optimal_diligence(values_by_k, ks, cost):
    """Profit-maximizing number of signals: argmax_k [value(k) - cost*k]."""
    net = [values_by_k[k] - cost * k for k in ks]
    return ks[int(np.argmax(net))]


def diligence_experiment():
    """How much due diligence: equilibrium value vs number of signals, then the
    cost cutoff. Bidder 0 acquires k independent signals; bidder 1 is fixed at
    one. Both sides play the resulting general-sum equilibrium."""
    print("== How much due diligence: value of signals and the cost cutoff ==")
    values_by_k = {}
    rows = []
    for k in DILIGENCE_K:
        auction = EnumeratedAuction(
            num_values=DILIGENCE_BASE["num_values"], num_bids=DILIGENCE_BASE["num_bids"],
            num_signals_0=k, num_signals_1=RIVAL_SIGNALS,
            noise_0=DILIGENCE_NOISE, noise_1=DILIGENCE_NOISE)
        res = own_profit_fictitious_play(auction)
        values_by_k[k] = res["value0"]
        marginal = res["value0"] - values_by_k.get(k - 1, res["value0"]) if k > DILIGENCE_K[0] else float("nan")
        rows.append((k, res["value0"], res["value1"], marginal, res["nashconv"]))
        print(f"  k={k}  bidder0 value={res['value0']:+.4f}  marginal={marginal:+.4f}  "
              f"rival value={res['value1']:+.4f}  NashConv={res['nashconv']:.2e}")

    cutoffs = [(c, _optimal_diligence(values_by_k, DILIGENCE_K, c)) for c in COST_GRID]
    for c, kstar in cutoffs:
        print(f"  cost/signal={c:.3f}  optimal diligence k*={kstar}")

    fig, (ax_v, ax_k) = plt.subplots(1, 2, figsize=(12, 5))
    ax_v.plot(DILIGENCE_K, [values_by_k[k] for k in DILIGENCE_K], marker="o", color="C2")
    ax_v.set_xlabel("number of diligence signals $k$")
    ax_v.set_ylabel("equilibrium own profit (rival fixed)")
    ax_v.set_title("Gross value of diligence (lumpy marginal)")
    ax_v.set_xticks(DILIGENCE_K)
    ax_v.grid(True, alpha=0.3)
    ax_k.step([c for c, _ in cutoffs], [k for _, k in cutoffs], where="post",
              marker="o", color="C3")
    ax_k.set_xlabel("cost per diligence signal $c$")
    ax_k.set_ylabel("profit-maximizing diligence $k^*$")
    ax_k.set_title("How much diligence before you bid")
    ax_k.set_yticks(DILIGENCE_K)
    ax_k.grid(True, alpha=0.3)
    fig.suptitle("Common-value takeover auction: the optimal amount of due diligence")
    out_png = os.path.join(RESULTS_DIR, "diligence.png")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)

    out_csv = os.path.join(RESULTS_DIR, "diligence.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["num_signals", "bidder0_value", "bidder1_value",
                    "marginal_value", "nashconv"])
        for k, v0, v1, marginal, nc in rows:
            w.writerow([k, f"{v0:.6f}", f"{v1:.6f}", f"{marginal:.6f}", f"{nc:.8f}"])
    out_cut = os.path.join(RESULTS_DIR, "diligence_cutoff.csv")
    with open(out_cut, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cost_per_signal", "optimal_k"])
        for c, kstar in cutoffs:
            w.writerow([c, kstar])
    print(f"  wrote {out_png}, {out_csv}, and {out_cut}")


def _diligence_value_curve(num_values, num_bids, noise, ks):
    """value(k) for bidder 0 acquiring k signals, rival fixed at one, at the BNE.

    Returns (values, all_converged): a dict k -> equilibrium own profit, and whether
    every solve in the curve reached the FP tolerance (so non-converged curves can be
    excluded from robustness conclusions rather than silently trusted)."""
    values = {}
    all_converged = True
    for k in ks:
        auction = EnumeratedAuction(
            num_values=num_values, num_bids=num_bids,
            num_signals_0=k, num_signals_1=RIVAL_SIGNALS,
            noise_0=noise, noise_1=noise)
        res = own_profit_fictitious_play(auction, max_iterations=ROBUST_MAX_ITER)
        values[k] = res["value0"]
        all_converged = all_converged and res["converged"]
    return values, all_converged


def diligence_robustness_experiment():
    """Is the lumpy/odd-signal marginal and the k=2-skipping cutoff robust?

    Sweeps (num_values, noise) and, for each, recomputes the diligence value curve and
    the cost cutoff. Reports, per parameterization, whether the value is increasing,
    whether the marginal is non-monotone (the third signal worth more than the second),
    and whether the cost cutoff ever selects k=2. This tells a referee whether the
    headline 'how much diligence' shape generalizes beyond the base parameterization."""
    print("== Diligence-cutoff robustness across (num_values, noise) ==")
    rows = []
    summary = []
    for nv in ROBUST_NUM_VALUES:
        nb = nv + 1  # keep max bid index == max value, as in the base game
        for noise in ROBUST_NOISE:
            values, converged = _diligence_value_curve(nv, nb, noise, DILIGENCE_K)
            marginals = {k: (values[k] - values[k - 1] if k > DILIGENCE_K[0] else float("nan"))
                         for k in DILIGENCE_K}
            cutoffs = {c: _optimal_diligence(values, DILIGENCE_K, c) for c in COST_GRID}
            kstars = set(cutoffs.values())
            net_positive = values[DILIGENCE_K[-1]] > values[DILIGENCE_K[0]]
            monotone = all(values[k] >= values[k - 1] - 1e-9 for k in DILIGENCE_K[1:])
            # The base finding: the 3rd signal's marginal exceeds the 2nd's.
            third_beats_second = marginals[3] > marginals[2]
            skips_two = 2 not in kstars
            falls_with_cost = cutoffs[COST_GRID[0]] >= cutoffs[COST_GRID[-1]]
            summary.append((nv, noise, converged, net_positive, monotone,
                            third_beats_second, skips_two, falls_with_cost, sorted(kstars)))
            tag = "" if converged else "  [NOT CONVERGED -- excluded from robustness counts]"
            print(f"  num_values={nv} noise={noise:.1f}: "
                  f"value {values[DILIGENCE_K[0]]:.3f}->{values[DILIGENCE_K[-1]]:.3f}  "
                  f"net+={net_positive} monotone={monotone} 3rd>2nd={third_beats_second}  "
                  f"k* falls with cost={falls_with_cost} k* in {sorted(kstars)} skips2={skips_two}{tag}")
            for k in DILIGENCE_K:
                rows.append((nv, nb, noise, k, values[k], marginals[k], int(converged)))

    conv = [s for s in summary if s[2]]
    n = len(conv)
    print(f"  -- of {len(summary)} parameterizations, {n} converged; among those: "
          f"net value positive (k=5 > k=1) in {sum(1 for s in conv if s[3])}/{n}; "
          f"monotone in k in {sum(1 for s in conv if s[4])}/{n}; "
          f"3rd-signal marginal beats 2nd in {sum(1 for s in conv if s[5])}/{n}; "
          f"cutoff falls with cost in {sum(1 for s in conv if s[7])}/{n}; "
          f"cutoff skips k=2 in {sum(1 for s in conv if s[6])}/{n}")

    fig, (ax_v, ax_m) = plt.subplots(1, 2, figsize=(12, 5))
    for nv in ROBUST_NUM_VALUES:
        for noise in ROBUST_NOISE:
            series = [r for r in rows if r[0] == nv and r[2] == noise]
            vals = [r[4] for r in series]
            margs = [r[5] for r in series]
            ok = bool(series[0][6])
            label = f"nv={nv}, noise={noise:.1f}" + ("" if ok else " (unconv.)")
            style = dict(alpha=0.85) if ok else dict(alpha=0.35, ls=":")
            ax_v.plot(DILIGENCE_K, vals, marker="o", label=label, **style)
            ax_m.plot(DILIGENCE_K[1:], margs[1:], marker="s", label=label, **style)
    ax_v.set_xlabel("number of diligence signals $k$")
    ax_v.set_ylabel("equilibrium own profit")
    ax_v.set_title("Value of diligence (net positive but not always monotone)")
    ax_v.set_xticks(DILIGENCE_K)
    ax_v.grid(True, alpha=0.3)
    ax_m.set_xlabel("number of diligence signals $k$")
    ax_m.set_ylabel("marginal value of the $k$-th signal")
    ax_m.axhline(0.0, color="#888888", lw=0.8)
    ax_m.set_title("Marginal is lumpy; its shape is parameterization-specific")
    ax_m.set_xticks(DILIGENCE_K[1:])
    ax_m.grid(True, alpha=0.3)
    ax_m.legend(fontsize=7, ncol=1)
    fig.suptitle("Robustness of the diligence cutoff across game size and signal noise")
    fig.savefig(os.path.join(RESULTS_DIR, "diligence_robustness.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    out_csv = os.path.join(RESULTS_DIR, "diligence_robustness.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["num_values", "num_bids", "noise", "num_signals",
                    "bidder0_value", "marginal_value", "converged"])
        for nv, nb, noise, k, val, marg, conv in rows:
            w.writerow([nv, nb, noise, k, f"{val:.6f}", f"{marg:.6f}", conv])
    print(f"  wrote {out_csv} and results/diligence_robustness.png")


# Symmetric acquisition equilibrium. The diligence curve above fixes the rival at one
# signal, so it is a one-sided best response; the toehold section warns that one-sided
# analyses overstate effects because the opponent does not re-optimize. Here we let BOTH
# bidders choose how many signals to acquire. We solve the bidding BNE for every pair
# (k0, k1) to get the value matrix V[k0][k1] (bidder-0 own profit holding k0 signals
# against a rival holding k1), then at each per-signal cost find the SYMMETRIC pure
# acquisition equilibria k* (both acquire k*, neither gains by deviating). Comparing
# k*(c) to the one-sided cutoff shows whether competition in acquisition changes how
# much diligence is bought.
ACQ_K = [1, 2, 3, 4, 5]

# Grid-refinement check: the robustness sweep found cells where more signals LOWER
# equilibrium profit (non-monotone value in k). The base game uses a coarse bid grid
# (num_bids = num_values + 1). We re-solve one such cell at finer bid grids over the
# SAME bid range [0, num_values]; if the decline persists it is a genuine equilibrium
# effect, not a grid-quantization artifact (the same check we applied to the toehold).
GRID_REFINE_CELL = {"num_values": 4, "noise": 0.5}
GRID_REFINE_RESOLUTIONS = [5, 9, 17]  # bid-grid points over [0, num_values]; 5 == base


def _value_matrix(num_values, num_bids, noise, ks):
    """``V[(ka, kb)]`` = bidder-0 equilibrium own profit holding ``ka`` signals against
    a rival holding ``kb``, at the bidding BNE. Also returns whether every solve in the
    matrix reached the FP tolerance."""
    V = {}
    all_conv = True
    for ka in ks:
        for kb in ks:
            auction = EnumeratedAuction(
                num_values=num_values, num_bids=num_bids,
                num_signals_0=ka, num_signals_1=kb,
                noise_0=noise, noise_1=noise)
            res = own_profit_fictitious_play(auction, max_iterations=ROBUST_MAX_ITER)
            V[(ka, kb)] = res["value0"]
            all_conv = all_conv and res["converged"]
    return V, all_conv


def _symmetric_acquisition_equilibria(V, ks, cost):
    """Pure-strategy symmetric Nash of the acquisition stage at per-signal ``cost``.
    ``k`` is an equilibrium iff, when the rival also acquires ``k``, no deviation
    ``k'`` raises bidder 0's net payoff ``V[(k', k)] - cost*k'``."""
    eqs = []
    for k in ks:
        base = V[(k, k)] - cost * k
        if all(base >= V[(kp, k)] - cost * kp - 1e-9 for kp in ks):
            eqs.append(k)
    return eqs


def _one_sided_cutoff(V, ks, cost, rival=RIVAL_SIGNALS):
    """The one-sided cutoff: argmax_k [V[(k, rival)] - cost*k] (rival fixed)."""
    net = [V[(k, rival)] - cost * k for k in ks]
    return ks[int(np.argmax(net))]


def acquisition_symmetric_experiment():
    """How much diligence when BOTH bidders choose it: the symmetric acquisition
    equilibrium vs the one-sided cutoff, on the base game."""
    print("== Symmetric acquisition equilibrium (both bidders choose diligence) ==")
    nv = DILIGENCE_BASE["num_values"]
    nb = DILIGENCE_BASE["num_bids"]
    V, all_conv = _value_matrix(nv, nb, DILIGENCE_NOISE, ACQ_K)
    if not all_conv:
        print("  [warning] not every (k0,k1) solve converged; see nashconv in solves")
    for ka in ACQ_K:
        print("  V[k0=%d][k1=*] = " % ka
              + "  ".join(f"{V[(ka, kb)]:.3f}" for kb in ACQ_K))

    rows = []
    for c in COST_GRID:
        sym = _symmetric_acquisition_equilibria(V, ACQ_K, c)
        one = _one_sided_cutoff(V, ACQ_K, c)
        selected = max(sym, key=lambda k: V[(k, k)] - c * k) if sym else None
        rows.append((c, sym, selected, one))
        print(f"  cost/signal={c:.3f}  symmetric k*={sym or '[none pure]'}  "
              f"(selected {selected})  one-sided k*={one}")

    fig, (ax_v, ax_k) = plt.subplots(1, 2, figsize=(12, 5))
    for kb in ACQ_K:
        ax_v.plot(ACQ_K, [V[(ka, kb)] for ka in ACQ_K], marker="o",
                  label=f"rival holds {kb}")
    ax_v.set_xlabel("own diligence signals $k_0$")
    ax_v.set_ylabel("equilibrium own profit")
    ax_v.set_title("Own value falls as the rival acquires more diligence")
    ax_v.set_xticks(ACQ_K)
    ax_v.legend(fontsize=8)
    ax_v.grid(True, alpha=0.3)
    costs = [r[0] for r in rows]
    sym_sel = [r[2] if r[2] is not None else float("nan") for r in rows]
    one_sided = [r[3] for r in rows]
    ax_k.step(costs, sym_sel, where="post", marker="o", color="C3",
              label="symmetric acquisition $k^*$")
    ax_k.step(costs, one_sided, where="post", marker="s", color="C0",
              label="one-sided cutoff (rival fixed at 1)")
    ax_k.set_xlabel("cost per diligence signal $c$")
    ax_k.set_ylabel("profit-maximizing diligence $k^*$")
    ax_k.set_title("Both schedules fall with cost; competition lowers $k^*$")
    ax_k.set_yticks(ACQ_K)
    ax_k.legend(fontsize=8)
    ax_k.grid(True, alpha=0.3)
    fig.suptitle("Diligence when both bidders choose it: symmetric equilibrium vs one-sided")
    fig.savefig(os.path.join(RESULTS_DIR, "acquisition_symmetric.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    out_mat = os.path.join(RESULTS_DIR, "acquisition_matrix.csv")
    with open(out_mat, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["own_signals_k0", "rival_signals_k1", "bidder0_value"])
        for ka in ACQ_K:
            for kb in ACQ_K:
                w.writerow([ka, kb, f"{V[(ka, kb)]:.6f}"])
    out_csv = os.path.join(RESULTS_DIR, "acquisition_symmetric.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cost_per_signal", "symmetric_kstars", "selected_symmetric_kstar",
                    "one_sided_kstar"])
        for c, sym, selected, one in rows:
            w.writerow([c, ";".join(str(k) for k in sym) if sym else "",
                        "" if selected is None else selected, one])
    print(f"  wrote {out_mat}, {out_csv}, and results/acquisition_symmetric.png")


def grid_refinement_experiment():
    """Does the value non-monotonicity in k survive a finer bid grid?"""
    nv = GRID_REFINE_CELL["num_values"]
    noise = GRID_REFINE_CELL["noise"]
    print(f"== Grid-refinement check on non-monotone cell nv={nv}, noise={noise} ==")
    rows = []
    for R in GRID_REFINE_RESOLUTIONS:
        bid_values = np.linspace(0.0, float(nv), R)
        curve = {}
        for k in DILIGENCE_K:
            auction = EnumeratedAuction(
                num_values=nv, num_bids=R, num_signals_0=k, num_signals_1=RIVAL_SIGNALS,
                noise_0=noise, noise_1=noise, bid_values=bid_values)
            res = own_profit_fictitious_play(auction, max_iterations=ROBUST_MAX_ITER)
            curve[k] = res
            rows.append((R, nv, noise, k, res["value0"], int(res["converged"]),
                         res["nashconv"]))
        decline_12 = curve[1]["value0"] - curve[2]["value0"]  # >0 => value falls k1->k2
        print(f"  bid-grid points={R} (step={nv / (R - 1):.3f}): "
              f"value k=1->2 {curve[1]['value0']:.3f}->{curve[2]['value0']:.3f} "
              f"(decline {decline_12:+.3f}); "
              f"monotone={'no' if decline_12 > 1e-4 else 'yes'}  "
              f"converged={all(curve[k]['converged'] for k in DILIGENCE_K)}")

    plt.figure(figsize=(7, 5))
    for R in GRID_REFINE_RESOLUTIONS:
        series = [r for r in rows if r[0] == R]
        plt.plot(DILIGENCE_K, [s[4] for s in series], marker="o",
                 label=f"{R} bid levels (step {nv / (R - 1):.2f})")
    plt.xlabel("number of diligence signals $k$")
    plt.ylabel("equilibrium own profit")
    plt.title(f"Value non-monotonicity survives bid-grid refinement (nv={nv}, noise={noise})")
    plt.xticks(DILIGENCE_K)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "grid_refinement.png"),
                dpi=130, bbox_inches="tight")
    plt.close()

    out_csv = os.path.join(RESULTS_DIR, "grid_refinement.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["bid_levels", "num_values", "noise", "num_signals",
                    "bidder0_value", "converged", "nashconv"])
        for R, nvv, ns, k, val, conv, nc in rows:
            w.writerow([R, nvv, ns, k, f"{val:.6f}", conv, f"{nc:.8f}"])
    print(f"  wrote {out_csv} and results/grid_refinement.png")


def main():
    _ensure_results_dir()
    asymmetry_gs_experiment()
    toehold_gs_experiment()
    diligence_experiment()
    diligence_robustness_experiment()
    acquisition_symmetric_experiment()
    grid_refinement_experiment()
    print("Done.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Run a single named study, e.g. `python general_sum_equilibrium.py
        # acquisition_symmetric_experiment`, instead of the full suite.
        _ensure_results_dir()
        globals()[sys.argv[1]]()
    else:
        main()
