"""Sequential auction in the intractable regime + learned-BR calibration (paper 2,
phase 4).

The benchmark (``sequential_benchmark.py``) lives where exact CFR and exact
exploitability still run. This file crosses into the regime where they cannot: a
sequential auction with many private signals per bidder AND several rounds, whose
information sets grow as ``num_values ** num_signals`` and whose history count
grows as ``num_values ** (1 + 2 * num_signals)`` times the bidding subtree, so the
tree cannot be tabulated. Two studies:

1. Intractable regime: train PPO and PPG self-play without tabulating (weights are
   tail-averaged) and score them with learned-best-response approximate
   exploitability, against a uniform-random anchor and a naive no-shading bidder.
   Shows the learners drive exploitability down where the exact solvers cannot run.

2. Estimator calibration: on ENUMERABLE multi-signal sequential games (k=1,2, where
   exact exploitability is ground truth) build policies of known exploitability by
   mixing the CFR equilibrium with uniform play, and compare the learned-BR estimate
   to the exact value. Validates that the estimator tracks exact exploitability on
   the sequential multi-signal information-set structure, so the intractable-regime
   numbers can be read as lower bounds and not artifacts.

Run one study:
    python experiments/sequential_intractable.py seq_calibration_study
"""

from __future__ import annotations

import csv
import math
import os
import statistics
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyspiel
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import cfr, exploitability

import torch

import dealgame  # noqa: F401  (registers games)
from dealgame.intractable import (UniformPolicy, approx_exploitability,
                                  approx_exploitability_detailed,
                                  train_intractable)


class SequentialNaiveBidder:
    """Naive no-shading bidder for the ASCENDING auction.

    Like ``intractable.NaiveBidder`` it bids its Bayesian posterior mean of W with
    no winner's-curse shading, but respects the ascending protocol: it can only
    raise above the standing bid or pass. If its posterior mean is at or above the
    cheapest legal raise it takes the legal raise nearest the mean; if the standing
    bid already exceeds its mean it passes rather than overpay. This is the
    economically exploitable reference a shading best-responder should beat, adapted
    so it never assigns probability to an illegal (below-standing) bid.
    """

    def __init__(self, game):
        self._n = game.num_values
        self._num_bids = game.num_bids
        self._noise = game.signal_noise_0

    def _posterior_mean_index(self, signals):
        logpost = []
        for w in range(self._n):
            lp = 0.0
            for s in signals:
                p = self._noise / self._n + (1.0 - self._noise if s == w else 0.0)
                lp += math.log(max(p, 1e-12))
            logpost.append(lp)
        m = max(logpost)
        weights = [math.exp(x - m) for x in logpost]
        z = sum(weights)
        post_mean = sum((w + 1) * weights[w] / z for w in range(self._n))
        return min(range(self._num_bids), key=lambda b: abs(b - post_mean))

    def action_probabilities(self, state, player_id=None):
        player = state.current_player()
        target = self._posterior_mean_index(state._signals[player])
        legal = state.legal_actions(player)
        pass_action = max(legal)              # PASS == num_bids, the largest action
        raises = [a for a in legal if a != pass_action]
        if not raises or target < min(raises):
            choice = pass_action              # mean below cheapest raise: don't overpay
        else:
            choice = min(raises, key=lambda b: abs(b - target))
        return {a: (1.0 if a == choice else 0.0) for a in legal}

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
GAME = "dealgame_sequential_takeover"
WORKERS = min(32, os.cpu_count() or 8)

# Intractable instance: multi-round AND multi-signal, so both the bidding tree and
# the signal information sets are beyond enumeration.
INTRACTABLE_PARAMS = {"num_values": 3, "num_bids": 6, "num_rounds": 3,
                      "num_signals": 6}
INTRACTABLE_EPISODES = 300000
INTRACTABLE_EVAL_POINTS = 1
INTRACTABLE_SEEDS = [0, 1, 2]
# BR strength matched to the paper-1 intractable study: strong enough that a
# "PPO/PPG approx 0" reading means near-unexploitable, not a weak best response.
INTRACTABLE_APPROX_KW = dict(br_batches=250, br_batch_episodes=256, mc_episodes=20000)

# Calibration on enumerable sequential games. Exact exploitability is the ground
# truth; we mix the CFR equilibrium with uniform to build known-exploitability
# policies and check the learned-BR estimate tracks them.
CALIB_BASE = {"num_values": 3, "num_bids": 4, "num_rounds": 2}
CALIB_KS = [1, 2]
CALIB_CFR_ITERS = {1: 400, 2: 300}
CALIB_MIX = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0]   # uniform-mix fraction
CALIB_SEEDS = [0, 1, 2]
CALIB_APPROX_KW = dict(br_batches=250, br_batch_episodes=256, mc_episodes=20000)


def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _mean_std(values):
    m = statistics.mean(values)
    s = statistics.pstdev(values) if len(values) > 1 else 0.0
    return m, s


def _load(params):
    return pyspiel.load_game(GAME, params)


def _seq_infoset_and_history_scale(params):
    """Order-of-magnitude (infosets/bidder, terminal histories) for the intractable
    instance, to document why exact methods cannot run. The signal layer dominates:
    infosets scale as num_values**num_signals and histories at least as
    num_values**(1 + 2*num_signals) times the bidding subtree (>= num_bids)."""
    nv = params["num_values"]
    k = params["num_signals"]
    infosets = nv ** k
    histories = nv ** (1 + 2 * k) * params["num_bids"]
    return infosets, histories


# --- process-pool workers ---------------------------------------------------

def _intractable_job(args):
    kind, params, seed, episodes, eval_points, approx_kw = args
    torch.set_num_threads(1)
    curve = train_intractable(_load(params), kind=kind, episodes=episodes,
                              eval_points=eval_points, seed=seed, approx_kw=approx_kw)
    return kind, seed, curve


def _anchor_job(args):
    name, params, approx_kw = args
    torch.set_num_threads(1)
    game = _load(params)
    pol = UniformPolicy(game) if name == "Random" else SequentialNaiveBidder(game)
    approx, _, _ = approx_exploitability(game, pol, seed=0, **approx_kw)
    return name, approx


def _calib_job(args):
    """One (k, mix, seed) learned-BR estimate at a mixed policy of known
    exploitability. The CFR equilibrium array is passed in (solved once per k in the
    parent) so each worker only builds the mixed policy and runs the estimator."""
    k, params, eps, seed, eq_arr, approx_kw = args
    torch.set_num_threads(1)
    game = _load(params)
    mixed = policy_lib.TabularPolicy(game)
    uni_arr = policy_lib.TabularPolicy(game).action_probability_array
    mixed.action_probability_array[:] = (1.0 - eps) * eq_arr + eps * uni_arr
    d = approx_exploitability_detailed(game, mixed, seed=seed, **approx_kw)
    return k, eps, seed, d["approx"]


def _pool(fn, arglist):
    if len(arglist) <= 1:
        return [fn(a) for a in arglist]
    with ProcessPoolExecutor(max_workers=min(WORKERS, len(arglist))) as ex:
        return list(ex.map(fn, arglist))


def seq_intractable_experiment():
    print("== Sequential intractable regime: approximate exploitability ==")
    params = INTRACTABLE_PARAMS
    infosets, histories = _seq_infoset_and_history_scale(params)
    print(f"  instance {params}: ~{histories:.2e} histories, "
          f"~{infosets:.2e} signal infosets/bidder (exact methods infeasible)")

    anchors = dict(_pool(_anchor_job, [(name, params, INTRACTABLE_APPROX_KW)
                                       for name in ("Random", "Naive")]))
    for name, a in anchors.items():
        print(f"  {name} anchor: approx exploitability {a:.4f}")

    specs = [(kind, params, s, INTRACTABLE_EPISODES, INTRACTABLE_EVAL_POINTS,
              INTRACTABLE_APPROX_KW)
             for kind in ("ppo", "ppg") for s in INTRACTABLE_SEEDS]
    tagged = _pool(_intractable_job, specs)
    finals = {}
    for kind in ("ppo", "ppg"):
        vals = [curve[-1]["approx_expl"] for k, s, curve in tagged if k == kind]
        finals[kind] = _mean_std(vals)
        print(f"  {kind.upper()}: final approx exploitability "
              f"{finals[kind][0]:.4f}+/-{finals[kind][1]:.4f}")

    labels = ["Random", "Naive", "PPO", "PPG"]
    vals = [anchors["Random"], anchors["Naive"], finals["ppo"][0], finals["ppg"][0]]
    errs = [0.0, 0.0, finals["ppo"][1], finals["ppg"][1]]
    plt.figure(figsize=(6, 4.5))
    plt.bar(labels, vals, yerr=errs, capsize=4,
            color=["#888888", "#E8884A", "#C94A2A", "#4A6FC9"])
    plt.ylabel("approximate exploitability (learned BR)")
    plt.title(f"Intractable sequential auction "
              f"(k={params['num_signals']} signals, {params['num_rounds']} rounds;\n"
              f"~{histories:.0e} histories, exact methods cannot run)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "seq_intractable.png"), dpi=130,
                bbox_inches="tight")
    plt.close()

    with open(os.path.join(RESULTS_DIR, "seq_intractable.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instance_histories", f"{histories:.3e}"])
        w.writerow(["instance_signal_infosets_per_bidder", f"{infosets:.3e}"])
        w.writerow(["method", "approx_expl_mean", "approx_expl_std"])
        w.writerow(["Random", f"{anchors['Random']:.6f}", "0.0"])
        w.writerow(["Naive", f"{anchors['Naive']:.6f}", "0.0"])
        for kind in ("ppo", "ppg"):
            w.writerow([kind.upper(), f"{finals[kind][0]:.6f}",
                        f"{finals[kind][1]:.6f}"])
    print("  wrote results/seq_intractable.{png,csv}")


def seq_calibration_study():
    """Validate the learned-BR estimator against exact exploitability on enumerable
    multi-signal sequential games (k=1,2)."""
    print("== Sequential estimator calibration: learned BR vs exact (k=1,2) ==")
    all_rows = []
    for k in CALIB_KS:
        params = dict(CALIB_BASE, num_signals=k)
        game = _load(params)
        iters = CALIB_CFR_ITERS.get(k, 300)
        print(f"  k={k}: solving CFR ({iters} iters) for the reference equilibrium ...",
              flush=True)
        solver = cfr.CFRSolver(game)
        for _ in range(iters):
            solver.evaluate_and_update_policy()
        eq_arr = solver.average_policy().action_probability_array

        # Exact exploitability of each mixed policy (parent; cheap tree pass).
        exact_by_eps = {}
        for eps in CALIB_MIX:
            mixed = policy_lib.TabularPolicy(game)
            uni_arr = policy_lib.TabularPolicy(game).action_probability_array
            mixed.action_probability_array[:] = (1.0 - eps) * eq_arr + eps * uni_arr
            exact_by_eps[eps] = exploitability.exploitability(game, mixed)

        # Learned-BR estimates over (mix, seed), pooled.
        specs = [(k, params, eps, sd, eq_arr, CALIB_APPROX_KW)
                 for eps in CALIB_MIX for sd in CALIB_SEEDS]
        est = _pool(_calib_job, specs)
        for eps in CALIB_MIX:
            approxes = [a for (kk, e, sd, a) in est if kk == k and e == eps]
            m, s = _mean_std(approxes)
            all_rows.append({"k": k, "mix": eps, "exact": exact_by_eps[eps],
                             "approx_mean": m, "approx_std": s})
            print(f"    k={k} mix={eps:.2f}  exact={exact_by_eps[eps]:.4f}  "
                  f"approx={m:.4f}+/-{s:.4f}", flush=True)

    ks = sorted({r["k"] for r in all_rows})
    fig, axes = plt.subplots(1, len(ks), figsize=(5 * len(ks), 4.5), squeeze=False)
    for ax, k in zip(axes[0], ks):
        rows = [r for r in all_rows if r["k"] == k]
        exact_v = [r["exact"] for r in rows]
        mean_v = [r["approx_mean"] for r in rows]
        std_v = [r["approx_std"] for r in rows]
        lim = max(max(exact_v), max(mean_v)) * 1.05
        ax.plot([0, lim], [0, lim], "--", color="#888888", label="y = x")
        ax.errorbar(exact_v, mean_v, yerr=std_v, fmt="o-", color="#C94A2A",
                    capsize=3, label=f"learned BR ({len(CALIB_SEEDS)} seeds)")
        ax.set_xlabel("exact exploitability (ground truth)")
        ax.set_ylabel("approximate exploitability (learned BR)")
        ax.set_title(f"k = {k} signal" + ("s" if k != 1 else ""))
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Sequential multi-signal calibration: the estimator tracks exact")
    fig.savefig(os.path.join(RESULTS_DIR, "seq_calibration.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(RESULTS_DIR, "seq_calibration.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["num_signals", "mix_fraction", "exact_expl",
                    "approx_mean", "approx_std"])
        for r in all_rows:
            w.writerow([r["k"], f"{r['mix']:.3f}", f"{r['exact']:.6f}",
                        f"{r['approx_mean']:.6f}", f"{r['approx_std']:.6f}"])
    print("  wrote results/seq_calibration.{png,csv}")


def main():
    _ensure_results_dir()
    seq_calibration_study()
    seq_intractable_experiment()
    print("All sequential intractable-regime studies complete.")


if __name__ == "__main__":
    import sys
    _ensure_results_dir()
    if len(sys.argv) > 1:
        globals()[sys.argv[1]]()
    else:
        main()
