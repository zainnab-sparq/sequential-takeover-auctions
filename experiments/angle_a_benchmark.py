"""Angle A benchmark: the arXiv-grade experiment suite.

Three studies, all on commodity CPU with no frontier-model spend:

1. Deep self-play (the seed paper's regime): generic deep policy gradient vs
   NFSP on a scaled-up common-value takeover auction, exploitability tracked
   against BOTH episodes and wall-clock seconds (a compute-normalized view).

2. Cross-game table: every method's final exploitability across heterogeneous
   games (common-value small/large, common-value with a toehold, and a
   structurally different private-value auction).

3. Economic finding: in the common-value auction, a bidder's equilibrium bid
   rises with its toehold, recovering the takeover-auction literature's
   prediction that toeholds make bidders more aggressive.

Run:
    docker run --rm -v "<repo>:/work" -w /work imperfect-info:latest \
        python experiments/angle_a_benchmark.py
"""

from __future__ import annotations

import csv
import os
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyspiel
from open_spiel.python.algorithms import cfr, exploitability

import torch

import dealgame  # noqa: F401  (registers games)
from dealgame.deep_solving import (train_deep_cfr, train_nfsp,
                                   train_policy_gradient, train_psro)
from dealgame.ppo_solving import train_ppg, train_ppo
from dealgame.solving import ReinforcePolicyGradient, run_cfr, run_mmd

# Pin the CPU thread count so multithreaded float reductions are bounded and
# run-to-run variation is controlled (matches OMP_NUM_THREADS set in the image).
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

HEADLINE_GAME = ("dealgame_takeover_auction", {"num_values": 5, "num_bids": 6})
HEADLINE_EPISODES = 300000
HEADLINE_EVAL_EVERY = 15000

CROSS_GAMES = {
    "CV-small": ("dealgame_takeover_auction", {"num_values": 3, "num_bids": 4}),
    "CV-large": ("dealgame_takeover_auction", {"num_values": 5, "num_bids": 6}),
    "CV-toehold": ("dealgame_takeover_auction",
                   {"num_values": 5, "num_bids": 6, "toehold": 0.4}),
    "PV": ("dealgame_private_value_auction", {"num_values": 4, "num_bids": 4}),
}
CROSS_DEEP_EPISODES = 150000
TOEHOLD_GRID = [0.0, 0.2, 0.4, 0.6]
SEEDS = [0, 1, 2, 3, 4]  # deep methods are stochastic; report mean +/- std over seeds

# Scaling study: common-value auctions of growing size. Exact solvers traverse
# the whole game tree every iteration, so their per-iteration cost grows with the
# tree; the sampling/function-approximation methods do not. We measure wall-clock
# to reach a target exploitability to see whether the learning methods overtake
# the exact solvers as the game grows.
SCALING_GAMES = [
    ("S4", {"num_values": 4, "num_bids": 5}),
    ("S6", {"num_values": 6, "num_bids": 8}),
    ("S8", {"num_values": 8, "num_bids": 10}),
    ("S12", {"num_values": 12, "num_bids": 14}),
]
SCALING_TARGET = 0.05          # exploitability threshold for the wall-clock race
SCALING_PPO_EPISODES = 600000
SCALING_PPO_EVAL_EVERY = 20000

# Intractable regime: the multi-signal common-value auction. With k signals per
# bidder the strategy space (num_values**k) and game tree are far too large to
# tabulate or enumerate, so exact CFR/exploitability cannot run; we evaluate with
# learned-best-response approximate exploitability instead.
INTRACTABLE_GAME = ("dealgame_takeover_auction",
                    {"num_values": 6, "num_bids": 6, "num_signals": 8})
INTRACTABLE_EPISODES = 300000
INTRACTABLE_EVAL_POINTS = 1
INTRACTABLE_SEEDS = [0, 1, 2]
# BR strength: a diagnostic showed br_batches 250 and 600 (64- vs 128-wide) give the
# same approximate exploitability for both the uniform and naive references, so the
# learned BR is at its power ceiling here and 250 is sufficient (a stronger BR finds
# no more), making a "PPO/PPG approx 0" reading credible rather than BR-limited.
INTRACTABLE_APPROX_KW = dict(br_batches=250, br_batch_episodes=256,
                             mc_episodes=20000)

# Estimator calibration: on the tractable CV-large game (where exact exploitability
# is available as ground truth) we build policies of KNOWN exploitability by mixing
# the CFR equilibrium with uniform play, then compare the learned-BR approximate
# exploitability to the exact value. This calibrates the estimator used in the
# intractable regime: it shows the range where the estimate is tight and the
# Monte-Carlo resolution floor below which it reads zero, so a "PPO/PPG approx 0"
# reading can be interpreted correctly rather than dismissed as a weak BR.
CALIB_GAME = ("dealgame_takeover_auction", {"num_values": 5, "num_bids": 6})
CALIB_CFR_ITERS = 400
CALIB_MIX = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0]  # uniform-mix fraction
CALIB_APPROX_KW = dict(br_batches=250, br_batch_episodes=256, mc_episodes=20000)
CALIB_SEEDS = [0, 1, 2, 3, 4]  # BR seeds per mix point: report across-seed spread, not
                               # a single-seed estimate, so estimator tightness is robust

# Multi-signal calibration: the headline intractable result applies the learned-BR
# estimator to a k=8 multi-signal auction, but the single-signal calibration above
# only validates it at k=1. Here we repeat the CFR-equilibrium-mixed-with-uniform
# calibration on enumerable multi-signal games (k=1 as a bridge, then k=2 ~1.1e5 and
# k=3 ~2.8e6 histories, all still exactly solvable), five seeds per mix point, to show
# the estimate tracks exact exploitability tightly on the multi-signal information-set
# structure and that tightness is preserved as k grows toward the intractable regime.
MULTISIG_CALIB_K = [1, 2, 3]
MULTISIG_CALIB_BASE = {"num_values": 5, "num_bids": 6}
MULTISIG_CALIB_SEEDS = [0, 1, 2, 3, 4]
MULTISIG_CALIB_CFR_ITERS = {1: 400, 2: 400, 3: 200}  # k=3 tree is large; fewer iters
MULTISIG_CALIB_MIX = CALIB_MIX                        # same 7 mix points
MULTISIG_CALIB_APPROX_KW = CALIB_APPROX_KW            # same BR/MC settings

# Floor-vs-k: the headline k=8 result extrapolates the estimator's resolution floor
# past the k=1,2,3 calibration above. On the nv=5 family k>=4 is no longer exactly
# solvable (k=4 ~7e7 histories: CFR + exact exploitability time out), so we cannot
# extend the SAME-game calibration. Instead we measure how the floor GROWS with k on a
# SMALLER game (nv=3, nb=4) that stays enumerable through k=5 (k=5 ~2.8e6 histories).
# Floor magnitudes are not comparable across game families; the transferable property
# is the trend -- whether the floor explodes or grows slowly as the info-set count rises.
# k=1..4 here: k=4 (3^9*16 ~ 3.1e5 histories) already extends a full step beyond the
# k=1,2,3 calibration on the headline family. k=5 (~2.8e6 histories) is enumerable in
# principle but vanilla Python CFR on that tree is impractically slow even at a handful
# of iterations, so we stop at k=4 -- itself an illustration of the scaling thesis.
FLOOR_K = [1, 2, 3, 4]
FLOOR_BASE = {"num_values": 3, "num_bids": 4}
FLOOR_MIX = [0.0, 0.1]                  # equilibrium (the floor) plus one tracking check
FLOOR_SEEDS = [0, 1, 2]
# The floor metric is the Monte-Carlo SE, which does not need a tightly converged
# equilibrium; vanilla Python CFR is slow on the larger trees, so we use few iterations
# (enough for a low-exploitability anchor) -- the SE-based floor is unaffected.
FLOOR_CFR_ITERS = {1: 100, 2: 100, 3: 80, 4: 50}
FLOOR_APPROX_KW = CALIB_APPROX_KW

# Weight- vs tabular-averaging validation. In the intractable regime we tail-average
# network WEIGHTS (the policy cannot be tabulated), but the last-iterate-cycles /
# time-average-converges argument is about averaging the induced POLICY. Weight
# (Polyak) averaging of a nonlinear net is not the same object, so we validate it on
# the tractable CV-large game: run one PPO trajectory, average it both ways from the
# same post-burn-in iterates, and compare exact exploitability. If the weight average
# matches the tabular time-average, weight-averaging is a sound stand-in.
WAVG_GAME = ("dealgame_takeover_auction", {"num_values": 5, "num_bids": 6})
WAVG_EPISODES = 300000
WAVG_BATCH = 256
WAVG_SEEDS = [0, 1, 2]

# Stable deep policy-gradient configuration (see deep_solving defaults).
PG_KW = dict(loss_str="rpg", hidden=(64,), pi_lr=0.005, critic_lr=0.05)
PPO_KW = dict(batch_episodes=256, hidden=(64,), lr=3e-3, clip=0.2,
              epochs=4, minibatches=4, ent_coef=0.01)
PPG_KW = dict(batch_episodes=256, hidden=(64,), lr=3e-3, clip=0.2,
              epochs=4, minibatches=4, ent_coef=0.01, n_policy=8, aux_epochs=6)
DEEPCFR_KW = dict(num_iterations=80, num_traversals=100,
                  advantage_train_steps=600, policy_train_steps=600)
PSRO_ITERS = 15

# Column order for the cross-game table: exact/tabular solvers, then the
# function-approximation (deep) methods.
CROSS_METHODS = ["CFR", "MMD", "PSRO", "REINFORCE",
                 "PPO", "PPG", "DeepPG", "DeepCFR", "NFSP"]


def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _mean_std(values):
    m = statistics.mean(values)
    s = statistics.pstdev(values) if len(values) > 1 else 0.0
    return m, s


def _agg_curves(curves, key):
    """Aggregate same-schedule curves: per episode index where all have ``key``,
    return ``{episode, seconds, mean, std}`` (seconds averaged over seeds)."""
    out = []
    for rows in zip(*curves):
        if not all(key in r for r in rows):
            continue
        m, s = _mean_std([r[key] for r in rows])
        out.append({"episode": rows[0]["episode"],
                    "seconds": statistics.mean(r["seconds"] for r in rows),
                    "mean": m, "std": s})
    return out


def _timed_cfr_final(game, iterations=400):
    t0 = time.time()
    solver = cfr.CFRSolver(game)
    for _ in range(iterations):
        solver.evaluate_and_update_policy()
    expl = exploitability.exploitability(game, solver.average_policy())
    return expl, time.time() - t0, solver.average_policy()


def _timed_cfr_curve(game, iterations, eval_every):
    """CFR checkpoints with cumulative training wall-clock (eval time excluded)."""
    solver = cfr.CFRSolver(game)
    curve = []
    train_seconds = 0.0
    for it in range(1, iterations + 1):
        t0 = time.time()
        solver.evaluate_and_update_policy()
        train_seconds += time.time() - t0
        if it % eval_every == 0 or it == 1:
            expl = exploitability.exploitability(game, solver.average_policy())
            curve.append({"iteration": it, "seconds": train_seconds, "expl": expl})
    return curve


def _plot_band(ax, agg, xkey, **kw):
    xs = [a[xkey] for a in agg]
    ms = [a["mean"] for a in agg]
    lo = [max(a["mean"] - a["std"], 1e-6) for a in agg]
    hi = [a["mean"] + a["std"] for a in agg]
    line, = ax.plot(xs, ms, **kw)
    ax.fill_between(xs, lo, hi, color=line.get_color(), alpha=0.15)


def headline_deep_experiment():
    print("== Headline: deep self-play (PPO vs PG vs NFSP), %d seeds ==" % len(SEEDS))
    game = pyspiel.load_game(*HEADLINE_GAME)
    cfr_expl, cfr_t, _ = _timed_cfr_final(game)
    print(f"  CFR reference: exploitability={cfr_expl:.4f} in {cfr_t:.1f}s")
    psro_curves = [train_psro(game, iterations=PSRO_ITERS, eval_every=PSRO_ITERS,
                              seed=s) for s in SEEDS]
    psro_m, psro_s = _mean_std([c[-1]["psro"] for c in psro_curves])
    psro_t = statistics.mean(c[-1]["seconds"] for c in psro_curves)
    print(f"  PSRO reference: {psro_m:.4f}+/-{psro_s:.4f} in {psro_t:.1f}s")
    dcfr_curves = [train_deep_cfr(game, seed=s, **DEEPCFR_KW) for s in SEEDS]
    dcfr_m, dcfr_s = _mean_std([c[0]["deepcfr"] for c in dcfr_curves])
    dcfr_t = statistics.mean(c[0]["seconds"] for c in dcfr_curves)
    print(f"  Deep CFR reference: {dcfr_m:.4f}+/-{dcfr_s:.4f} in {dcfr_t:.1f}s")

    print("  training PPO over seeds ...")
    ppo_runs = [train_ppo(game, episodes=HEADLINE_EPISODES,
                          eval_every=HEADLINE_EVAL_EVERY, seed=s, **PPO_KW)
                for s in SEEDS]
    print("  training PPG over seeds ...")
    ppg_runs = [train_ppg(game, episodes=HEADLINE_EPISODES,
                          eval_every=HEADLINE_EVAL_EVERY, seed=s, **PPG_KW)
                for s in SEEDS]
    print("  training deep PG (rpg) over seeds ...")
    pg_runs = [train_policy_gradient(
        game, episodes=HEADLINE_EPISODES, eval_every=HEADLINE_EVAL_EVERY,
        seed=s, **PG_KW) for s in SEEDS]
    print("  training NFSP over seeds ...")
    nfsp_runs = [train_nfsp(
        game, episodes=HEADLINE_EPISODES, eval_every=HEADLINE_EVAL_EVERY,
        hidden=(64,), batch_size=128, seed=s) for s in SEEDS]

    ppo_avg = _agg_curves(ppo_runs, "avg")
    ppg_avg = _agg_curves(ppg_runs, "avg")
    pg_avg = _agg_curves(pg_runs, "avg")
    nfsp_agg = _agg_curves(nfsp_runs, "nfsp")
    print(f"  final (mean+/-std): PPO tail-avg={ppo_avg[-1]['mean']:.4f}+/-{ppo_avg[-1]['std']:.4f}"
          f"  PPG tail-avg={ppg_avg[-1]['mean']:.4f}+/-{ppg_avg[-1]['std']:.4f}"
          f"  PG tail-avg={pg_avg[-1]['mean']:.4f}+/-{pg_avg[-1]['std']:.4f}"
          f"  NFSP={nfsp_agg[-1]['mean']:.4f}+/-{nfsp_agg[-1]['std']:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, xkey, xlabel in [(ax1, "episode", "episodes"),
                             (ax2, "seconds", "wall-clock seconds")]:
        _plot_band(ax, ppo_avg, xkey, marker="^", ms=4, label="PPO (tail-average)")
        _plot_band(ax, ppg_avg, xkey, marker="v", ms=4, label="PPG (tail-average)")
        _plot_band(ax, pg_avg, xkey, marker="o", ms=3,
                   label="Deep PG / RPG (tail-average)")
        _plot_band(ax, nfsp_agg, xkey, marker="s", ms=3, label="NFSP")
        ax.axhline(cfr_expl, color="gray", ls="--", lw=1,
                   label=f"CFR (exact, {cfr_t:.0f}s)")
        ax.axhline(psro_m, color="purple", ls="-.", lw=1,
                   label=f"PSRO (exact-BR, {psro_t:.0f}s)")
        ax.axhline(dcfr_m, color="brown", ls=":", lw=1,
                   label=f"Deep CFR ({dcfr_t:.0f}s)")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("exploitability (log)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Deep self-play on the common-value takeover auction "
                 f"(mean +/- std over {len(SEEDS)} seeds)")
    fig.savefig(os.path.join(RESULTS_DIR, "deep_convergence.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    pg_last = _agg_curves(pg_runs, "last")
    ppo_last = _agg_curves(ppo_runs, "last")
    ppg_last = _agg_curves(ppg_runs, "last")
    with open(os.path.join(RESULTS_DIR, "deep_convergence.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "episodes", "seconds", "expl_mean", "expl_std"])
        for label, agg in [("PPO_tailavg", ppo_avg), ("PPO_last", ppo_last),
                           ("PPG_tailavg", ppg_avg), ("PPG_last", ppg_last),
                           ("DeepPG_RPG_tailavg", pg_avg),
                           ("DeepPG_RPG_last", pg_last), ("NFSP", nfsp_agg)]:
            for a in agg:
                w.writerow([label, a["episode"], f"{a['seconds']:.2f}",
                            f"{a['mean']:.6f}", f"{a['std']:.6f}"])
        w.writerow(["CFR", "", f"{cfr_t:.2f}", f"{cfr_expl:.6f}", "0.0"])
        w.writerow(["PSRO", "", f"{psro_t:.2f}", f"{psro_m:.6f}", f"{psro_s:.6f}"])
        w.writerow(["DeepCFR", "", f"{dcfr_t:.2f}", f"{dcfr_m:.6f}", f"{dcfr_s:.6f}"])
    print("  wrote results/deep_convergence.{png,csv}")


def _seed_finals(fn):
    """mean +/- std of fn(seed) over the configured seeds."""
    return _mean_std([fn(s) for s in SEEDS])


def compute_cross_game_row(name):
    """Compute every method's final exploitability for one game.

    Writes a one-row file ``results/cross_game_<name>.csv`` so games can be run in
    separate processes and merged by :func:`merge_cross_game`. Deep methods report
    mean +/- std over seeds; exact/tabular solvers are (near-)deterministic.
    """
    gname, params = CROSS_GAMES[name]
    game = pyspiel.load_game(gname, params)
    entry = {"game": name}
    entry["CFR"] = (run_cfr(game, 400, 400)[-1][1], 0.0)
    entry["MMD"] = (run_mmd(game, 400, 400, alpha=0.0, stepsize=1.0)[-1][1], 0.0)
    entry["PSRO"] = _seed_finals(
        lambda s: train_psro(game, iterations=PSRO_ITERS, eval_every=PSRO_ITERS,
                             seed=s)[-1]["psro"])
    entry["REINFORCE"] = (
        ReinforcePolicyGradient(game, lr=0.5, batch=256, seed=0).run(200, 200)[-1][1],
        0.0)
    entry["PPO"] = _seed_finals(
        lambda s: train_ppo(game, episodes=CROSS_DEEP_EPISODES,
                            eval_every=CROSS_DEEP_EPISODES, seed=s, **PPO_KW)[-1]["avg"])
    entry["PPG"] = _seed_finals(
        lambda s: train_ppg(game, episodes=CROSS_DEEP_EPISODES,
                            eval_every=CROSS_DEEP_EPISODES, seed=s, **PPG_KW)[-1]["avg"])
    entry["DeepPG"] = _seed_finals(
        lambda s: train_policy_gradient(game, episodes=CROSS_DEEP_EPISODES,
                                       eval_every=CROSS_DEEP_EPISODES, seed=s,
                                       **PG_KW)[-1]["avg"])
    entry["DeepCFR"] = _seed_finals(
        lambda s: train_deep_cfr(game, seed=s, **DEEPCFR_KW)[0]["deepcfr"])
    entry["NFSP"] = _seed_finals(
        lambda s: train_nfsp(game, episodes=CROSS_DEEP_EPISODES,
                            eval_every=CROSS_DEEP_EPISODES, seed=s)[-1]["nfsp"])
    print(f"  {name}: " + "  ".join(
        f"{m}={entry[m][0]:.4f}+/-{entry[m][1]:.4f}" for m in CROSS_METHODS))

    path = os.path.join(RESULTS_DIR, f"cross_game_{name}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        header = ["game"]
        for m in CROSS_METHODS:
            header += [f"{m}_mean", f"{m}_std"]
        w.writerow(header)
        row = [name]
        for m in CROSS_METHODS:
            row += [f"{entry[m][0]:.6f}", f"{entry[m][1]:.6f}"]
        w.writerow(row)
    print(f"  wrote {path}")
    return entry


def merge_cross_game():
    """Combine per-game rows into results/cross_game.csv (skips missing games)."""
    header = ["game"]
    for m in CROSS_METHODS:
        header += [f"{m}_mean", f"{m}_std"]
    out_rows = []
    for name in CROSS_GAMES:
        path = os.path.join(RESULTS_DIR, f"cross_game_{name}.csv")
        if not os.path.exists(path):
            print(f"  (missing {path}, skipping)")
            continue
        with open(path, newline="") as fh:
            rows = list(csv.reader(fh))
        out_rows.append(rows[1])  # the single data row
    with open(os.path.join(RESULTS_DIR, "cross_game.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(out_rows)
    print("  wrote results/cross_game.csv")


def cross_game_table():
    print("== Cross-game final exploitability (deep: %d seeds) ==" % len(SEEDS))
    for name in CROSS_GAMES:
        compute_cross_game_row(name)
    merge_cross_game()


def _expected_player0_bid(game, pol):
    """Reach-weighted expected bid of bidder 0 under a joint policy."""
    def rec(state, reach):
        if state.is_terminal():
            return 0.0
        if state.is_chance_node():
            return sum(rec(state.child(a), reach * p)
                       for a, p in state.chance_outcomes())
        probs = pol.action_probabilities(state)
        if state.current_player() == 0:
            contrib = sum(reach * p * a for a, p in probs.items())  # bid value == action index
            return contrib + sum(rec(state.child(a), reach * p)
                                 for a, p in probs.items())
        return sum(rec(state.child(a), reach * p) for a, p in probs.items())
    return rec(game.new_initial_state(), 1.0)


def _p0_ownprofit_br_bid(game, opp_pol):
    """Bidder 0's expected bid under the bid that maximizes its OWN profit.

    Holds bidder 1 fixed at ``opp_pol`` and, for each bidder-0 signal, picks the
    bid maximizing bidder 0's own (general-sum) profit, then returns the
    signal-weighted expected bid. This isolates the toehold's own-profit channel
    from the rivalry term that the zero-sum profit-difference scoring injects.
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
    return sum(mass[s0] * max(actions, key=actions.get)
               for s0, actions in q.items())


def toehold_finding():
    print("== Economic finding: toehold raises aggressiveness ==")
    # Fix bidder 1 at the base-game (theta=0) zero-sum equilibrium so the
    # own-profit best-response curve varies only with bidder 0's toehold.
    base_game = pyspiel.load_game(
        "dealgame_takeover_auction", {"num_values": 5, "num_bids": 6})
    _, _, base_opp = _timed_cfr_final(base_game, iterations=600)
    rows = []
    for theta in TOEHOLD_GRID:
        game = pyspiel.load_game(
            "dealgame_takeover_auction",
            {"num_values": 5, "num_bids": 6, "toehold": theta})
        _, _, avg = _timed_cfr_final(game, iterations=600)
        zs_bid = _expected_player0_bid(game, avg)        # zero-sum equilibrium
        op_bid = _p0_ownprofit_br_bid(game, base_opp)    # own-profit best response
        rows.append((theta, zs_bid, op_bid))
        print(f"  toehold={theta:.1f}  zero-sum eq bid={zs_bid:.3f}  "
              f"own-profit BR bid={op_bid:.3f}")

    plt.figure(figsize=(7, 5))
    plt.plot([r[0] for r in rows], [r[1] for r in rows], marker="o",
             label="zero-sum equilibrium (CFR)")
    plt.plot([r[0] for r in rows], [r[2] for r in rows], marker="s",
             label="own-profit best response (fixed opponent)")
    plt.xlabel("bidder 0 toehold (fraction of target already owned)")
    plt.ylabel("bidder 0 expected bid")
    plt.title("Toeholds make bidders more aggressive")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "toehold.png"), dpi=130, bbox_inches="tight")
    plt.close()
    with open(os.path.join(RESULTS_DIR, "toehold.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["toehold", "zerosum_eq_bid", "ownprofit_br_bid"])
        for theta, zs_bid, op_bid in rows:
            w.writerow([theta, f"{zs_bid:.6f}", f"{op_bid:.6f}"])
    print("  wrote results/toehold.{png,csv}")


def _wallclock_to_target(curve, key, target, time_key="seconds"):
    """First wall-clock second at which ``curve[key]`` drops to ``target``.

    Returns ``None`` if the target is never reached (so the table can show a miss
    honestly rather than silently truncating)."""
    for row in curve:
        if key in row and row[key] <= target:
            return row[time_key]
    return None


def scaling_study():
    """Wall-clock to reach a target exploitability as the game grows.

    Exact CFR/PSRO traverse the whole tree each iteration; PPO samples a fixed
    number of episodes per update regardless of tree size. We report, per game
    size, the wall-clock each method needs to reach exploitability
    ``SCALING_TARGET`` (or that it never did), to see whether the learning method
    overtakes the exact solvers as the game scales. Exact exploitability is still
    computed by tree enumeration; genuinely intractable instances (continuous or
    many-round) needing approximate exploitability are left to future work.
    """
    print("== Scaling study: wall-clock to exploitability %.3f ==" % SCALING_TARGET)
    rows = []
    for label, params in SCALING_GAMES:
        game = pyspiel.load_game("dealgame_takeover_auction", params)
        n_states = sum(1 for _ in _iter_states(game.new_initial_state()))
        cfr_curve = _timed_cfr_curve(game, 1000, 10)
        cfr_t = _wallclock_to_target(cfr_curve, "expl", SCALING_TARGET)
        ppo_ts = []
        ppo_bests = []  # best (lowest) tail-average exploitability reached per seed
        for s in SEEDS[:3]:  # 3 seeds is enough for a wall-clock crossover
            curve = train_ppo(game, episodes=SCALING_PPO_EPISODES,
                              eval_every=SCALING_PPO_EVAL_EVERY, seed=s, **PPO_KW)
            ppo_ts.append(_wallclock_to_target(curve, "avg", SCALING_TARGET))
            avgs = [row["avg"] for row in curve if "avg" in row]
            if avgs:
                ppo_bests.append(min(avgs))
        reached = [t for t in ppo_ts if t is not None]
        ppo_t, ppo_t_std = _mean_std(reached) if reached else (None, None)
        ppo_best_m, ppo_best_s = _mean_std(ppo_bests) if ppo_bests else (None, None)
        rows.append((label, n_states, cfr_t, ppo_t, ppo_t_std, len(reached),
                     len(ppo_ts), ppo_best_m, ppo_best_s))
        print(f"  {label} (~{n_states} states): CFR={cfr_t}  "
              f"PPO={ppo_t} ({len(reached)}/{len(ppo_ts)} seeds reached), "
              f"PPO best expl={ppo_best_m}")

    with open(os.path.join(RESULTS_DIR, "scaling.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["game", "states", "cfr_seconds_to_target",
                    "ppo_seconds_to_target", "ppo_seconds_std", "ppo_seeds_reached",
                    "ppo_seeds", "ppo_best_expl_mean", "ppo_best_expl_std"])
        for label, n, cfr_t, ppo_t, ppo_t_std, hit, tot, best_m, best_s in rows:
            w.writerow([label, n,
                        "" if cfr_t is None else f"{cfr_t:.2f}",
                        "" if ppo_t is None else f"{ppo_t:.2f}",
                        "" if ppo_t_std is None else f"{ppo_t_std:.2f}", hit, tot,
                        "" if best_m is None else f"{best_m:.4f}",
                        "" if best_s is None else f"{best_s:.4f}"])
    print("  wrote results/scaling.csv")


def _iter_states(state):
    """Yield every decision/terminal state in the game tree (for sizing)."""
    yield state
    if state.is_terminal():
        return
    if state.is_chance_node():
        for a, _ in state.chance_outcomes():
            yield from _iter_states(state.child(a))
    else:
        for a in state.legal_actions():
            yield from _iter_states(state.child(a))


def intractable_experiment():
    """Approximate exploitability on a game too large to enumerate.

    On the multi-signal common-value auction (exact CFR/exploitability infeasible),
    train PPO and PPG without tabulating and track learned-best-response approximate
    exploitability, against a uniform-random anchor. Shows the learning methods drive
    exploitability down in a regime where the exact solvers cannot run."""
    from dealgame.intractable import (NaiveBidder, UniformPolicy,
                                      approx_exploitability, train_intractable,
                                      tree_and_strategy_size)
    print("== Intractable regime: approximate exploitability ==")
    gname, params = INTRACTABLE_GAME
    game = pyspiel.load_game(gname, params)
    terminals, infosets = tree_and_strategy_size(
        params["num_values"], params["num_bids"], params["num_signals"])
    print(f"  instance {params}: ~{terminals:.2e} histories, "
          f"{infosets:.2e} infosets/player (exact methods infeasible)")

    # Anchors: uniform play (highly exploitable) and a naive no-shading bidder
    # (economically exploitable). The same BR must beat these for "PPO/PPG approx 0"
    # to mean "near-unexploitable" rather than "BR too weak".
    anchors = {}
    for name, pol in [("Random", UniformPolicy(game)), ("Naive", NaiveBidder(game))]:
        a, _, _ = approx_exploitability(game, pol, seed=0, **INTRACTABLE_APPROX_KW)
        anchors[name] = a
        print(f"  {name} anchor: approx exploitability {a:.4f}")

    finals = {}
    for kind in ["ppo", "ppg"]:
        seed_finals = [train_intractable(
            game, kind=kind, episodes=INTRACTABLE_EPISODES,
            eval_points=INTRACTABLE_EVAL_POINTS, seed=s,
            approx_kw=INTRACTABLE_APPROX_KW)[-1]["approx_expl"]
            for s in INTRACTABLE_SEEDS]
        finals[kind] = _mean_std(seed_finals)
        print(f"  {kind.upper()}: final approx exploitability "
              f"{finals[kind][0]:.4f}+/-{finals[kind][1]:.4f}")

    labels = ["Random", "Naive", "PPO", "PPG"]
    vals = [anchors["Random"], anchors["Naive"], finals["ppo"][0], finals["ppg"][0]]
    errs = [0.0, 0.0, finals["ppo"][1], finals["ppg"][1]]
    plt.figure(figsize=(6, 4.5))
    plt.bar(labels, vals, yerr=errs, capsize=4,
            color=["#888888", "#E8884A", "#C94A2A", "#4A6FC9"])
    plt.ylabel("approximate exploitability (learned BR)")
    plt.title(f"Intractable multi-signal auction (k={params['num_signals']};\n"
              f"~{terminals:.0e} histories, exact methods cannot run)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "intractable.png"),
                dpi=130, bbox_inches="tight")
    plt.close()

    with open(os.path.join(RESULTS_DIR, "intractable.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instance_histories", f"{terminals:.3e}"])
        w.writerow(["instance_infosets_per_player", f"{infosets:.3e}"])
        w.writerow(["method", "approx_expl_mean", "approx_expl_std"])
        w.writerow(["Random", f"{anchors['Random']:.6f}", "0.0"])
        w.writerow(["Naive", f"{anchors['Naive']:.6f}", "0.0"])
        for kind in ["ppo", "ppg"]:
            w.writerow([kind.upper(), f"{finals[kind][0]:.6f}",
                        f"{finals[kind][1]:.6f}"])
    print("  wrote results/intractable.{png,csv}")


def calibration_study():
    """Validate the learned-BR estimator against exact exploitability.

    On CV-large (small enough to solve exactly) we build policies of known
    exploitability by mixing the CFR equilibrium with uniform play, then compare the
    learned-BR approximate exploitability to the exact value. This calibrates the
    estimator used in the intractable regime: it shows the range where the estimate
    is tight and the Monte-Carlo resolution floor below which it reads zero."""
    from open_spiel.python import policy as policy_lib
    from dealgame.intractable import approx_exploitability_detailed
    print("== Estimator calibration: learned BR vs exact exploitability ==")
    gname, params = CALIB_GAME
    game = pyspiel.load_game(gname, params)

    solver = cfr.CFRSolver(game)
    for _ in range(CALIB_CFR_ITERS):
        solver.evaluate_and_update_policy()
    eq_arr = solver.average_policy().action_probability_array
    uni_arr = policy_lib.TabularPolicy(game).action_probability_array

    rows = []
    for eps in CALIB_MIX:
        mixed = policy_lib.TabularPolicy(game)
        mixed.action_probability_array[:] = (1.0 - eps) * eq_arr + eps * uni_arr
        exact = exploitability.exploitability(game, mixed)
        ds = [approx_exploitability_detailed(game, mixed, seed=sd, **CALIB_APPROX_KW)
              for sd in CALIB_SEEDS]
        approx_mean = statistics.mean(d["approx"] for d in ds)
        approx_std = statistics.pstdev(d["approx"] for d in ds)
        se_mean = statistics.mean(d["se"] for d in ds)
        rows.append({"mix": eps, "exact": exact, "approx": approx_mean,
                     "approx_std": approx_std, "se": se_mean})
        print(f"  mix={eps:.2f}  exact={exact:.4f}  "
              f"approx={approx_mean:.4f}+/-{approx_std:.4f}  (mc se {se_mean:.4f})")

    exact_v = [r["exact"] for r in rows]
    approx_v = [r["approx"] for r in rows]
    std_v = [r["approx_std"] for r in rows]
    floor = statistics.mean(r["se"] for r in rows)
    plt.figure(figsize=(6, 4.5))
    lim = max(max(exact_v), max(approx_v)) * 1.05
    plt.plot([0, lim], [0, lim], "--", color="#888888", label="y = x (perfect estimate)")
    plt.axhline(floor, color="#4A6FC9", ls=":", label=f"MC resolution floor (~{floor:.3f})")
    plt.errorbar(exact_v, approx_v, yerr=std_v, fmt="o-", color="#C94A2A",
                 capsize=3, label=f"learned BR ({len(CALIB_SEEDS)} seeds)")
    plt.xlabel("exact exploitability (ground truth)")
    plt.ylabel("approximate exploitability (learned BR)")
    plt.title("Learned-BR calibration on CV-large\n(CFR equilibrium mixed with uniform)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "calibration.png"),
                dpi=130, bbox_inches="tight")
    plt.close()

    with open(os.path.join(RESULTS_DIR, "calibration.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mix_fraction", "exact_expl", "approx_expl",
                    "approx_std", "mc_se"])
        for r in rows:
            w.writerow([f"{r['mix']:.3f}", f"{r['exact']:.6f}",
                        f"{r['approx']:.6f}", f"{r['approx_std']:.6f}",
                        f"{r['se']:.6f}"])
    print("  wrote results/calibration.{png,csv}")


def _write_multisig_calibration(all_rows):
    """Write the multi-signal calibration CSV and a one-panel-per-k figure.

    Called after each k completes so partial results survive a long k=3 run."""
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
                    capsize=3, label="learned BR (5 seeds)")
        ax.set_xlabel("exact exploitability (ground truth)")
        ax.set_ylabel("approximate exploitability (learned BR)")
        ax.set_title(f"k = {k} signal" + ("s" if k != 1 else ""))
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Multi-signal calibration: the estimator stays tight as k grows")
    fig.savefig(os.path.join(RESULTS_DIR, "calibration_multisig.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(RESULTS_DIR, "calibration_multisig.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["num_signals", "mix_fraction", "exact_expl",
                    "approx_mean", "approx_std"])
        for r in all_rows:
            w.writerow([r["k"], f"{r['mix']:.3f}", f"{r['exact']:.6f}",
                        f"{r['approx_mean']:.6f}", f"{r['approx_std']:.6f}"])


def multisignal_calibration_study():
    """Calibrate the learned-BR estimator on enumerable MULTI-signal games.

    The single-signal calibration validates the estimator at k=1; the headline
    intractable result applies it at k=8. This bridges the gap: it repeats the
    CFR-equilibrium-mixed-with-uniform calibration on k=1,2,3 (all still exactly
    solvable), five seeds per mix point (mean +/- std), to show the estimate tracks
    exact exploitability tightly on the multi-signal information-set structure and
    that tightness is preserved as k grows toward the intractable regime."""
    from open_spiel.python import policy as policy_lib
    from dealgame.intractable import approx_exploitability_detailed
    print("== Multi-signal estimator calibration (k=1,2,3; 5 seeds per mix point) ==")
    all_rows = []
    for k in MULTISIG_CALIB_K:
        params = dict(MULTISIG_CALIB_BASE, num_signals=k)
        game = pyspiel.load_game("dealgame_takeover_auction", params)
        cfr_iters = MULTISIG_CALIB_CFR_ITERS.get(k, 200)
        print(f"  k={k}: solving CFR ({cfr_iters} iters) for the reference equilibrium ...")
        solver = cfr.CFRSolver(game)
        for _ in range(cfr_iters):
            solver.evaluate_and_update_policy()
        eq_arr = solver.average_policy().action_probability_array
        uni_arr = policy_lib.TabularPolicy(game).action_probability_array
        for eps in MULTISIG_CALIB_MIX:
            mixed = policy_lib.TabularPolicy(game)
            mixed.action_probability_array[:] = (1.0 - eps) * eq_arr + eps * uni_arr
            exact = exploitability.exploitability(game, mixed)
            approx_seeds = [approx_exploitability_detailed(
                game, mixed, seed=sd, **MULTISIG_CALIB_APPROX_KW)["approx"]
                for sd in MULTISIG_CALIB_SEEDS]
            mean = statistics.mean(approx_seeds)
            std = statistics.pstdev(approx_seeds)
            all_rows.append({"k": k, "mix": eps, "exact": exact,
                             "approx_mean": mean, "approx_std": std})
            print(f"    k={k} mix={eps:.2f}  exact={exact:.4f}  "
                  f"approx={mean:.4f}+/-{std:.4f}")
        _write_multisig_calibration(all_rows)  # incremental save after each k
    print("  wrote results/calibration_multisig.{png,csv}")


def _write_floor_vs_k(rows):
    """Write the floor-vs-k CSV and figure. Called after each k so a long run is
    recoverable from partial results."""
    floor_rows = [r for r in rows if r["mix"] == 0.0]
    plt.figure(figsize=(6, 4.5))
    plt.plot([r["k"] for r in floor_rows], [r["se_mean"] for r in floor_rows],
             "o-", color="#C94A2A", label="MC resolution floor (SE at equilibrium)")
    plt.plot([r["k"] for r in floor_rows], [r["exact"] for r in floor_rows],
             "s--", color="#4A6FC9", label="exact exploitability of CFR eq")
    plt.xlabel("number of diligence signals $k$ (info-sets/bidder $=3^k$)")
    plt.ylabel("exploitability scale")
    plt.title("Estimator resolution floor grows slowly with $k$ (nv=3 game)")
    plt.xticks(sorted({r["k"] for r in floor_rows}))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "floor_vs_k.png"), dpi=130, bbox_inches="tight")
    plt.close()

    with open(os.path.join(RESULTS_DIR, "floor_vs_k.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["num_values", "num_bids", "num_signals", "infosets_per_bidder",
                    "mix_fraction", "exact_expl", "approx_mean", "approx_std", "mc_se_mean"])
        for r in rows:
            w.writerow([FLOOR_BASE["num_values"], FLOOR_BASE["num_bids"], r["k"],
                        r["infosets"], f"{r['mix']:.3f}", f"{r['exact']:.6f}",
                        f"{r['approx_mean']:.6f}", f"{r['approx_std']:.6f}",
                        f"{r['se_mean']:.6f}"])


def floor_vs_k_study():
    """Measure how the learned-BR resolution floor grows with k on an enumerable game.

    The intractable k=8 result extrapolates the estimator's floor past the k<=3
    same-game calibration (k>=4 on nv=5 is no longer solvable). On a small game
    (nv=3, nb=4) that stays enumerable through k=5 we track, per k: the exact
    exploitability of the CFR equilibrium (near zero), the learned-BR estimate of it
    (the floor: what the estimator reads when there is almost nothing to find) with its
    Monte-Carlo standard error, and the estimate at a mixed policy of known
    exploitability (a tracking check). If the floor grows slowly with k here, the k=8
    extrapolation on the headline family is the same qualitative behavior, not a leap."""
    from open_spiel.python import policy as policy_lib
    from dealgame.intractable import approx_exploitability_detailed
    print("== Floor-vs-k: resolution-floor growth on a small enumerable game (nv=3) ==")
    rows = []
    for k in FLOOR_K:
        params = dict(FLOOR_BASE, num_signals=k)
        game = pyspiel.load_game("dealgame_takeover_auction", params)
        cfr_iters = FLOOR_CFR_ITERS.get(k, 100)
        infosets = FLOOR_BASE["num_values"] ** k
        print(f"  k={k} (infosets/bidder={infosets}): solving CFR ({cfr_iters} iters) ...",
              flush=True)
        solver = cfr.CFRSolver(game)
        for _ in range(cfr_iters):
            solver.evaluate_and_update_policy()
        eq_arr = solver.average_policy().action_probability_array
        uni_arr = policy_lib.TabularPolicy(game).action_probability_array
        for eps in FLOOR_MIX:
            mixed = policy_lib.TabularPolicy(game)
            mixed.action_probability_array[:] = (1.0 - eps) * eq_arr + eps * uni_arr
            exact = exploitability.exploitability(game, mixed)
            ds = [approx_exploitability_detailed(game, mixed, seed=sd, **FLOOR_APPROX_KW)
                  for sd in FLOOR_SEEDS]
            approx_mean = statistics.mean(d["approx"] for d in ds)
            approx_std = statistics.pstdev(d["approx"] for d in ds)
            se_mean = statistics.mean(d["se"] for d in ds)
            rows.append({"k": k, "infosets": infosets, "mix": eps, "exact": exact,
                         "approx_mean": approx_mean, "approx_std": approx_std,
                         "se_mean": se_mean})
            print(f"    k={k} mix={eps:.2f}  exact={exact:.4f}  "
                  f"approx={approx_mean:.4f}+/-{approx_std:.4f}  se={se_mean:.4f}",
                  flush=True)
        _write_floor_vs_k(rows)  # incremental save after each k
    print("  wrote results/floor_vs_k.{png,csv}")


def weight_averaging_validation():
    """Validate weight tail-averaging against the tabular time-average on CV-large.

    Run one PPO self-play trajectory and average it two ways from the same
    post-burn-in iterates: the tabular time-average (the convergent object in
    zero-sum games) and the weight (Polyak) tail-average used in the intractable
    regime. Compare exact exploitability of the last iterate, the tabular average,
    and the weight average. Matching tabular and weight averages shows the weight
    average is a sound stand-in where the policy cannot be tabulated."""
    from open_spiel.python.algorithms import exploitability as expl_lib
    from dealgame.ppo_solving import PPOAgent, _collect
    from dealgame.deep_solving import RunningAverageTabularPolicy, RLPolicies
    from dealgame.intractable import _averaged_policy
    import torch
    import numpy as np
    print("== Weight- vs tabular-averaging validation (CV-large, exact ground truth) ==")
    gname, params = WAVG_GAME
    rows = []
    for seed in WAVG_SEEDS:
        game = pyspiel.load_game(gname, params)
        torch.manual_seed(seed)
        np.random.seed(seed)
        info_size = game.information_state_tensor_size()
        num_actions = game.num_distinct_actions()
        agents = [PPOAgent(p, info_size, num_actions, hidden=PPO_KW["hidden"],
                           lr=PPO_KW["lr"], clip=PPO_KW["clip"],
                           epochs=PPO_KW["epochs"], minibatches=PPO_KW["minibatches"],
                           ent_coef=PPO_KW["ent_coef"]) for p in range(2)]
        tabular = RunningAverageTabularPolicy(game, agents)
        burn_in = int(0.5 * WAVG_EPISODES)
        wsum = [None, None]
        wcount = 0
        done = 0
        while done < WAVG_EPISODES:
            batch = _collect(game, agents, WAVG_BATCH)
            for p in range(2):
                agents[p].update(batch[p])
            done += WAVG_BATCH
            if done >= burn_in:
                tabular.snapshot()
                for p in range(2):
                    sd = {k: v.clone() for k, v in agents[p].net.state_dict().items()}
                    if wsum[p] is None:
                        wsum[p] = sd
                    else:
                        for k in wsum[p]:
                            wsum[p][k] += sd[k]
                wcount += 1
        wavg = [{k: v / wcount for k, v in wsum[p].items()} for p in range(2)]
        weight_pol = _averaged_policy(game, agents, wavg)
        e_last = expl_lib.exploitability(game, RLPolicies(game, agents))
        e_tab = expl_lib.exploitability(game, tabular)
        e_wgt = expl_lib.exploitability(game, weight_pol)
        rows.append({"seed": seed, "last": e_last, "tabular": e_tab, "weight": e_wgt})
        print(f"  seed {seed}: last={e_last:.4f}  tabular-avg={e_tab:.4f}  "
              f"weight-avg={e_wgt:.4f}")

    last_m = _mean_std([r["last"] for r in rows])
    tab_m = _mean_std([r["tabular"] for r in rows])
    wgt_m = _mean_std([r["weight"] for r in rows])
    print(f"  MEAN: last={last_m[0]:.4f}+/-{last_m[1]:.4f}  "
          f"tabular={tab_m[0]:.4f}+/-{tab_m[1]:.4f}  "
          f"weight={wgt_m[0]:.4f}+/-{wgt_m[1]:.4f}")
    with open(os.path.join(RESULTS_DIR, "weight_averaging.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "last_iterate", "tabular_tailavg", "weight_tailavg"])
        for r in rows:
            w.writerow([r["seed"], f"{r['last']:.6f}", f"{r['tabular']:.6f}",
                        f"{r['weight']:.6f}"])
        w.writerow(["mean", f"{last_m[0]:.6f}", f"{tab_m[0]:.6f}", f"{wgt_m[0]:.6f}"])
        w.writerow(["std", f"{last_m[1]:.6f}", f"{tab_m[1]:.6f}", f"{wgt_m[1]:.6f}"])
    print("  wrote results/weight_averaging.csv")


# Burn-in fairness. PPO/PPG/DeepPG are scored on a post-burn-in tail-average (the
# random-init transient discarded), while NFSP and Deep CFR use their native from-start
# averages. To check the headline ordering is not an artifact of that burn-in, we re-run
# the PG family on the headline game with NO burn-in (burn_in_frac=0.0, a full average
# that includes the early transient, the same treatment NFSP/Deep CFR get) and compare.
# If the PG family still beats NFSP/Deep CFR with no burn-in, the gap is a method effect,
# not a burn-in head start.
BURNIN_SEEDS = [0, 1, 2]


def burnin_fairness_study():
    """PG family scored with NO burn-in (full average) vs the native deep averages."""
    print("== Burn-in fairness: PG family full-average (no burn-in) vs native averages ==")
    game = pyspiel.load_game(*HEADLINE_GAME)
    rows = []
    pg_specs = [
        ("PPO", lambda s: train_ppo(
            game, episodes=HEADLINE_EPISODES, eval_every=HEADLINE_EPISODES,
            seed=s, burn_in_frac=0.0, **PPO_KW)),
        ("PPG", lambda s: train_ppg(
            game, episodes=HEADLINE_EPISODES, eval_every=HEADLINE_EPISODES,
            seed=s, burn_in_frac=0.0, **PPG_KW)),
        ("DeepPG", lambda s: train_policy_gradient(
            game, episodes=HEADLINE_EPISODES, eval_every=HEADLINE_EPISODES,
            seed=s, burn_in_frac=0.0, **PG_KW)),
    ]
    for name, fn in pg_specs:
        avgs, lasts = [], []
        for s in BURNIN_SEEDS:
            curve = fn(s)
            avgs.append(curve[-1]["avg"])
            lasts.append(curve[-1]["last"])
        am, asd = _mean_std(avgs)
        lm, lsd = _mean_std(lasts)
        rows.append((name, "full-average (no burn-in)", am, asd))
        rows.append((name, "last iterate", lm, lsd))
        print(f"  {name}: full-avg {am:.4f}+/-{asd:.4f}  last {lm:.4f}+/-{lsd:.4f}")

    nfsp_finals = [train_nfsp(game, episodes=HEADLINE_EPISODES,
                              eval_every=HEADLINE_EPISODES, hidden=(64,),
                              batch_size=128, seed=s)[-1]["nfsp"] for s in BURNIN_SEEDS]
    nm, nsd = _mean_std(nfsp_finals)
    rows.append(("NFSP", "native average", nm, nsd))
    print(f"  NFSP native average: {nm:.4f}+/-{nsd:.4f}")
    dcfr_finals = [train_deep_cfr(game, seed=s, **DEEPCFR_KW)[0]["deepcfr"]
                   for s in BURNIN_SEEDS]
    dm, dsd = _mean_std(dcfr_finals)
    rows.append(("DeepCFR", "native average", dm, dsd))
    print(f"  Deep CFR native average: {dm:.4f}+/-{dsd:.4f}")

    with open(os.path.join(RESULTS_DIR, "burnin_matched.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "scoring", "expl_mean", "expl_std"])
        for name, scoring, m, sd in rows:
            w.writerow([name, scoring, f"{m:.6f}", f"{sd:.6f}"])
    print("  wrote results/burnin_matched.csv")


def main():
    _ensure_results_dir()
    headline_deep_experiment()
    cross_game_table()
    toehold_finding()
    scaling_study()
    intractable_experiment()
    calibration_study()
    multisignal_calibration_study()
    weight_averaging_validation()
    burnin_fairness_study()
    print("All benchmark studies complete.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Run a single named study, e.g. `python angle_a_benchmark.py
        # multisignal_calibration_study`, instead of the full suite.
        _ensure_results_dir()
        globals()[sys.argv[1]]()
    else:
        main()
