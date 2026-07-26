"""Sequential-auction benchmark: the deep-vs-exact solver comparison on genuinely
multi-round games (paper 2's computational leg).

Paper 1 benchmarked the solvers on a SEALED (single-decision) auction. The open
question this file answers is whether the same ranking holds once the game is
genuinely sequential, which is the home turf of CFR-style solvers and the hardest
setting for generic policy-gradient methods (the Rudolph et al. 2026 thesis). Two
studies:

1. Deep self-play on a small multi-round ascending auction: PPO/PPG vs
   CFR/PSRO/NFSP/DeepCFR/DeepPG, exploitability tracked against BOTH episodes and
   wall-clock seconds, mean +/- std over seeds.

2. Scaling in the number of rounds: how exact-solver cost and reachable
   exploitability move as the contest deepens from 1 to R rounds. Because the
   ascending-bid ceiling caps the tree once the bid grid is exhausted, the bid
   grid is set wide enough that added rounds genuinely deepen the tree; the state
   count is reported per round so any saturation is visible rather than hidden.

Run (full suite):
    docker run --rm -v "<repo>:/work" -w /work imperfect-info:latest \
        python experiments/sequential_benchmark.py
Run one study:
    python experiments/sequential_benchmark.py seq_scaling_experiment
"""

from __future__ import annotations

import csv
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor

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

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
GAME = "dealgame_sequential_takeover"

# The independent training runs (one per seed, method, and round count) are
# dispatched across a process pool. Each worker pins itself to a single thread so
# many concurrent rollouts do not oversubscribe the cores; the rollout loop is the
# bottleneck, not the tiny-net matmuls, so 1-thread-per-worker x many workers wins.
WORKERS = min(32, os.cpu_count() or 8)

# Headline: a genuinely three-round ascending auction, small enough that exact CFR
# and exact exploitability are still a valid reference. The bid grid (num_bids) is
# wide enough that three rounds of alternating raises are not capped by the ceiling.
HEADLINE_PARAMS = {"num_values": 3, "num_bids": 6, "num_rounds": 3}
HEADLINE_EPISODES = 300000
HEADLINE_EVAL_EVERY = 15000
SEEDS = [0, 1, 2, 3, 4]

# Scaling in the number of rounds. Same bid grid throughout so the only thing that
# changes is contest depth; state counts are recorded so saturation is visible.
SCALING_BASE = {"num_values": 3, "num_bids": 8}
SCALING_ROUNDS = [1, 2, 3, 4]
SCALING_TARGET = 0.05           # exploitability threshold for the wall-clock race
SCALING_CFR_ITERS = 1000
SCALING_CFR_EVAL_EVERY = 10
SCALING_PPO_EPISODES = 300000
SCALING_PPO_EVAL_EVERY = 20000
SCALING_SEEDS = [0, 1, 2]       # 3 seeds is enough for a wall-clock crossover

PPO_KW = dict(batch_episodes=256, hidden=(64,), lr=3e-3, clip=0.2,
              epochs=4, minibatches=4, ent_coef=0.01)
PPG_KW = dict(batch_episodes=256, hidden=(64,), lr=3e-3, clip=0.2,
              epochs=4, minibatches=4, ent_coef=0.01, n_policy=8, aux_epochs=6)
PG_KW = dict(loss_str="rpg", hidden=(64,), pi_lr=0.005, critic_lr=0.05)
DEEPCFR_KW = dict(num_iterations=80, num_traversals=100,
                  advantage_train_steps=600, policy_train_steps=600)
PSRO_ITERS = 15


# --- process-pool workers ---------------------------------------------------
# Each worker is a module-level function (picklable), builds its own game from
# params (pyspiel games do not pickle reliably), and single-threads itself. It
# returns a plain curve/list of dicts, which pickles cleanly back to the parent.

def _load(params):
    return pyspiel.load_game(GAME, params)


def _ppo_job(args):
    params, seed, episodes, eval_every = args
    torch.set_num_threads(1)
    return train_ppo(_load(params), episodes=episodes, eval_every=eval_every,
                     seed=seed, **PPO_KW)


def _deep_job(args):
    """Unified deep-method worker for the headline, so every (method, seed) run is
    dispatched into one pool at once. Returns ``(kind, seed, curve)`` so the parent
    regroups robustly regardless of scheduling."""
    kind, params, seed, episodes, eval_every = args
    torch.set_num_threads(1)
    game = _load(params)
    if kind == "ppo":
        curve = train_ppo(game, episodes=episodes, eval_every=eval_every,
                          seed=seed, **PPO_KW)
    elif kind == "ppg":
        curve = train_ppg(game, episodes=episodes, eval_every=eval_every,
                          seed=seed, **PPG_KW)
    elif kind == "pg":
        curve = train_policy_gradient(game, episodes=episodes,
                                      eval_every=eval_every, seed=seed, **PG_KW)
    elif kind == "nfsp":
        curve = train_nfsp(game, episodes=episodes, eval_every=eval_every,
                           hidden=(64,), batch_size=128, seed=seed)
    else:
        raise ValueError(f"unknown deep method {kind!r}")
    return kind, seed, curve


def _psro_job(args):
    params, seed = args
    torch.set_num_threads(1)
    return train_psro(_load(params), iterations=PSRO_ITERS, eval_every=PSRO_ITERS,
                      seed=seed)


def _deepcfr_job(args):
    params, seed = args
    torch.set_num_threads(1)
    return train_deep_cfr(_load(params), seed=seed, **DEEPCFR_KW)


def _cfr_curve_job(args):
    params, iterations, eval_every = args
    torch.set_num_threads(1)
    return _timed_cfr_curve(_load(params), iterations, eval_every)


def _pool(fn, arglist):
    """Run ``fn`` over ``arglist`` concurrently, preserving order. Falls back to a
    serial call for a single item so smoke runs stay simple to trace."""
    if len(arglist) <= 1:
        return [fn(a) for a in arglist]
    with ProcessPoolExecutor(max_workers=min(WORKERS, len(arglist))) as ex:
        return list(ex.map(fn, arglist))


def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _mean_std(values):
    m = statistics.mean(values)
    s = statistics.pstdev(values) if len(values) > 1 else 0.0
    return m, s


def _agg_curves(curves, key):
    """Aggregate same-schedule curves at each index where all have ``key``."""
    out = []
    for rows in zip(*curves):
        if not all(key in r for r in rows):
            continue
        m, s = _mean_std([r[key] for r in rows])
        out.append({"episode": rows[0]["episode"],
                    "seconds": statistics.mean(r["seconds"] for r in rows),
                    "mean": m, "std": s})
    return out


def _plot_band(ax, agg, xkey, **kw):
    xs = [a[xkey] for a in agg]
    ms = [a["mean"] for a in agg]
    lo = [max(a["mean"] - a["std"], 1e-6) for a in agg]
    hi = [a["mean"] + a["std"] for a in agg]
    line, = ax.plot(xs, ms, **kw)
    ax.fill_between(xs, lo, hi, color=line.get_color(), alpha=0.15)


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


def _wallclock_to_target(curve, key, target, time_key="seconds"):
    """First wall-clock second at which ``curve[key]`` reaches ``target``.

    Returns ``None`` if never reached, so a miss shows honestly."""
    for row in curve:
        if key in row and row[key] <= target:
            return row[time_key]
    return None


def _count_states(state):
    stack = [state]
    n = 0
    while stack:
        s = stack.pop()
        n += 1
        if s.is_terminal():
            continue
        if s.is_chance_node():
            stack.extend(s.child(a) for a, _ in s.chance_outcomes())
        else:
            stack.extend(s.child(a) for a in s.legal_actions())
    return n


def seq_headline_experiment():
    print("== Sequential headline: deep self-play, %d seeds ==" % len(SEEDS))
    print(f"  game {HEADLINE_PARAMS}")
    game = pyspiel.load_game(GAME, HEADLINE_PARAMS)
    print(f"  ~{_count_states(game.new_initial_state())} states")

    cfr_expl, cfr_t, _ = _timed_cfr_final(game)
    print(f"  CFR reference: exploitability={cfr_expl:.4f} in {cfr_t:.1f}s")
    psro_curves = _pool(_psro_job, [(HEADLINE_PARAMS, s) for s in SEEDS])
    psro_m, psro_s = _mean_std([c[-1]["psro"] for c in psro_curves])
    psro_t = statistics.mean(c[-1]["seconds"] for c in psro_curves)
    print(f"  PSRO reference: {psro_m:.4f}+/-{psro_s:.4f} in {psro_t:.1f}s")
    dcfr_curves = _pool(_deepcfr_job, [(HEADLINE_PARAMS, s) for s in SEEDS])
    dcfr_m, dcfr_s = _mean_std([c[0]["deepcfr"] for c in dcfr_curves])
    dcfr_t = statistics.mean(c[0]["seconds"] for c in dcfr_curves)
    print(f"  Deep CFR reference: {dcfr_m:.4f}+/-{dcfr_s:.4f} in {dcfr_t:.1f}s")

    print("  training PPO/PPG/DeepPG/NFSP over seeds (pooled) ...")
    specs = [(kind, HEADLINE_PARAMS, s, HEADLINE_EPISODES, HEADLINE_EVAL_EVERY)
             for kind in ("ppo", "ppg", "pg", "nfsp") for s in SEEDS]
    tagged = _pool(_deep_job, specs)
    runs = {"ppo": [], "ppg": [], "pg": [], "nfsp": []}
    for kind, _seed, curve in tagged:
        runs[kind].append(curve)
    ppo_runs, ppg_runs, pg_runs, nfsp_runs = (
        runs["ppo"], runs["ppg"], runs["pg"], runs["nfsp"])

    ppo_avg = _agg_curves(ppo_runs, "avg")
    ppg_avg = _agg_curves(ppg_runs, "avg")
    pg_avg = _agg_curves(pg_runs, "avg")
    nfsp_agg = _agg_curves(nfsp_runs, "nfsp")
    print(f"  final (mean+/-std): PPO={ppo_avg[-1]['mean']:.4f}+/-{ppo_avg[-1]['std']:.4f}"
          f"  PPG={ppg_avg[-1]['mean']:.4f}+/-{ppg_avg[-1]['std']:.4f}"
          f"  PG={pg_avg[-1]['mean']:.4f}+/-{pg_avg[-1]['std']:.4f}"
          f"  NFSP={nfsp_agg[-1]['mean']:.4f}+/-{nfsp_agg[-1]['std']:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, xkey, xlabel in [(ax1, "episode", "episodes"),
                             (ax2, "seconds", "wall-clock seconds")]:
        _plot_band(ax, ppo_avg, xkey, marker="^", ms=4, label="PPO (tail-average)")
        _plot_band(ax, ppg_avg, xkey, marker="v", ms=4, label="PPG (tail-average)")
        _plot_band(ax, pg_avg, xkey, marker="o", ms=3, label="Deep PG / RPG (tail-average)")
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
    fig.suptitle(f"Deep self-play on the sequential takeover auction "
                 f"({HEADLINE_PARAMS['num_rounds']} rounds, "
                 f"mean +/- std over {len(SEEDS)} seeds)")
    fig.savefig(os.path.join(RESULTS_DIR, "seq_deep_convergence.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    ppo_last = _agg_curves(ppo_runs, "last")
    ppg_last = _agg_curves(ppg_runs, "last")
    pg_last = _agg_curves(pg_runs, "last")
    with open(os.path.join(RESULTS_DIR, "seq_deep_convergence.csv"), "w",
              newline="") as fh:
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
    print("  wrote results/seq_deep_convergence.{png,csv}")


def seq_scaling_experiment():
    """Wall-clock to a target exploitability as the contest deepens in rounds.

    Exact CFR traverses the whole tree each iteration, so its cost grows with the
    tree the added rounds create; PPO samples a fixed number of episodes per update
    regardless. We report, per round count, the state count, the exact CFR
    wall-clock to reach ``SCALING_TARGET``, and the PPO wall-clock (or that it never
    did). Recording the state count makes any ascending-bid saturation explicit."""
    print("== Sequential scaling: wall-clock to exploitability %.3f vs rounds =="
          % SCALING_TARGET)
    params_by_R = {R: dict(SCALING_BASE, num_rounds=R) for R in SCALING_ROUNDS}
    states_by_R = {R: _count_states(pyspiel.load_game(GAME, params_by_R[R])
                                    .new_initial_state()) for R in SCALING_ROUNDS}

    # Exact CFR curves (one per round count) and PPO runs (one per round x seed),
    # all dispatched concurrently; ex.map preserves input order for regrouping.
    cfr_curves = _pool(_cfr_curve_job,
                       [(params_by_R[R], SCALING_CFR_ITERS, SCALING_CFR_EVAL_EVERY)
                        for R in SCALING_ROUNDS])
    cfr_by_R = dict(zip(SCALING_ROUNDS, cfr_curves))
    ppo_specs = [(params_by_R[R], s, SCALING_PPO_EPISODES, SCALING_PPO_EVAL_EVERY)
                 for R in SCALING_ROUNDS for s in SCALING_SEEDS]
    ppo_flat = _pool(_ppo_job, ppo_specs)
    ppo_by_R = {R: ppo_flat[i * len(SCALING_SEEDS):(i + 1) * len(SCALING_SEEDS)]
                for i, R in enumerate(SCALING_ROUNDS)}

    rows = []
    for R in SCALING_ROUNDS:
        n_states = states_by_R[R]
        cfr_curve = cfr_by_R[R]
        cfr_t = _wallclock_to_target(cfr_curve, "expl", SCALING_TARGET)
        cfr_best = min((r["expl"] for r in cfr_curve), default=None)
        ppo_ts, ppo_bests = [], []
        for curve in ppo_by_R[R]:
            ppo_ts.append(_wallclock_to_target(curve, "avg", SCALING_TARGET))
            avgs = [row["avg"] for row in curve if "avg" in row]
            if avgs:
                ppo_bests.append(min(avgs))
        reached = [t for t in ppo_ts if t is not None]
        ppo_t, ppo_t_std = _mean_std(reached) if reached else (None, None)
        ppo_best_m, ppo_best_s = _mean_std(ppo_bests) if ppo_bests else (None, None)
        rows.append((R, n_states, cfr_t, cfr_best, ppo_t, ppo_t_std,
                     len(reached), len(ppo_ts), ppo_best_m, ppo_best_s))
        print(f"  R={R} (~{n_states} states): CFR->target={cfr_t} (best {cfr_best:.4f})  "
              f"PPO->target={ppo_t} ({len(reached)}/{len(ppo_ts)} seeds), "
              f"PPO best expl={ppo_best_m}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    Rs = [r[0] for r in rows]
    ax1.plot(Rs, [r[1] for r in rows], "o-", color="#4A6FC9")
    ax1.set_xlabel("number of rounds R")
    ax1.set_ylabel("game-tree states")
    ax1.set_yscale("log")
    ax1.set_title("Tree size vs rounds")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_xticks(Rs)
    cfr_ts = [r[2] for r in rows]
    ppo_ts_plot = [r[4] for r in rows]
    ax2.plot(Rs, [t if t is not None else float("nan") for t in cfr_ts],
             "s--", color="gray", label="CFR (exact)")
    ax2.plot(Rs, [t if t is not None else float("nan") for t in ppo_ts_plot],
             "^-", color="#C94A2A", label="PPO (tail-average)")
    ax2.set_xlabel("number of rounds R")
    ax2.set_ylabel(f"wall-clock seconds to expl {SCALING_TARGET}")
    ax2.set_title("Cost to target vs rounds")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(Rs)
    ax2.legend(fontsize=8)
    fig.suptitle("Sequential auction: cost grows with contest depth")
    fig.savefig(os.path.join(RESULTS_DIR, "seq_scaling.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(RESULTS_DIR, "seq_scaling.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rounds", "states", "cfr_seconds_to_target", "cfr_best_expl",
                    "ppo_seconds_to_target", "ppo_seconds_std", "ppo_seeds_reached",
                    "ppo_seeds", "ppo_best_expl_mean", "ppo_best_expl_std"])
        for R, n, cfr_t, cfr_best, ppo_t, ppo_t_std, hit, tot, best_m, best_s in rows:
            w.writerow([R, n,
                        "" if cfr_t is None else f"{cfr_t:.2f}",
                        "" if cfr_best is None else f"{cfr_best:.6f}",
                        "" if ppo_t is None else f"{ppo_t:.2f}",
                        "" if ppo_t_std is None else f"{ppo_t_std:.2f}", hit, tot,
                        "" if best_m is None else f"{best_m:.4f}",
                        "" if best_s is None else f"{best_s:.4f}"])
    print("  wrote results/seq_scaling.{png,csv}")


def main():
    _ensure_results_dir()
    seq_headline_experiment()
    seq_scaling_experiment()
    print("All sequential-benchmark studies complete.")


if __name__ == "__main__":
    import sys
    _ensure_results_dir()
    if len(sys.argv) > 1:
        globals()[sys.argv[1]]()
    else:
        main()
