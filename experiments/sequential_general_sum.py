"""The preemption channel: what a toehold buys once bidding is sequential.

Paper 1 solved a *sealed* common-value takeover auction and found that a toehold
raises its holder's profit but leaves its equilibrium bid essentially flat: the
aggressiveness channel is competed away. It asserted, but could not show, that the
channel lives in *sequential* contests, where a bidder can jump above the standing
bid to preempt a rival (Fishman 1988; Bulow-Huang-Klemperer 1999). These studies
make that the result.

Everything is read off the auction's own-profit Bayes-Nash equilibrium (see
:mod:`dealgame.sequential_general_sum`), never off the zero-sum rendering, which
would inject a rivalry term the auction does not have.

Five studies, in the order a sceptic would demand them:

1. ``toehold_preemption_experiment`` -- the headline, and it is not the one we went
   looking for. Solved from several starting policies, each toehold yields equilibria
   that agree closely on bidder 0's *profit* (to four decimals at toeholds 0 and 0.15,
   to within 6e-5 at 0.05) and disagree completely on its *conduct*: at a toehold of
   0.05, restarts certified to eps 4e-6 and 8e-5 deter the rival with probability 0.002
   and 0.333. So the toehold fixes what bidder 0
   earns, not how it bids, and the deterrence-versus-toehold curve one would like to draw
   does not exist. Paper 1's sealed auction is solved on the *same* bid grid and overlaid:
   its bid is flat in the toehold and it never preempts, so a sequential contest can at
   least *express* an aggressiveness channel that a sealed one cannot. Preemption is
   sustainable, not implied.

2. ``preemption_incentive_experiment`` -- why both conducts are equilibria. Force bidder
   0's opening bid, best-respond in the continuation, and read off what preempting is
   worth. The forced-opening best response is exact; the two equilibrium rivals it is run
   against are solver output, so only the control is fully solver-free. Against a rival
   that folds, a jump bid earns a large premium; against one that will not fold it earns
   nothing, which is what makes each conduct a best reply to the other. Note that the
   least-deterring *equilibrium* is often not a non-folding rival -- at several toeholds
   it still folds a fifth to a third of the time -- so ``is_passive`` records which rows
   actually test the contrast. The control is a rival bidding at random: it cannot be
   scared off, and the premium against it is exactly zero at every toehold.

3. ``rounds_compounding_experiment`` -- the two-move artifact. A toehold in a two-move
   contest is already known (Dodonova 2012); what is not is what happens as the contest
   runs longer. Sweeps the number of rounds. The name is historical: the project was
   founded on the guess that a toehold's value *compounds* across rounds, and the sweep
   showed the deterrence channel evaporates instead, saturating by the second round.

4. ``equilibrium_selection_experiment`` -- the check that decides whether any of this is
   real. A general-sum game can carry several equilibria and fictitious play from a
   uniform start finds one of them. Restarts from random policies; if the economics
   moves, the headline is an artifact of equilibrium selection.

5. ``bid_grid_refinement_experiment`` -- the same price range at three resolutions. A
   jump bid is measured in money, so a coarse grid could manufacture (or hide) one.

Reproducing the paper's numbers
-------------------------------
The paper's Tables 1 and 2 and Figure 1 are *certified* artifacts: they only mean what
they say when the compute budget, not the stopping rule, ended each solve. That is the
``certify`` entry point, and it is the one to run:

    docker run --rm -v "<repo>:/work" imperfect-info:latest \
        python experiments/sequential_general_sum.py certify

Expect roughly 20 hours; solves cache under ``results/.solve_cache`` and are reused.

The bare command below runs every study at the default 300k/1e-4 budget. That budget
stops *at* the tolerance, so its equilibria are certified only to ~1e-4 and its
certificate omits the toehold 0.05 row entirely (0.05 is not in the default sweep).
Those numbers are not the paper's, so a default run writes the three certified artifacts
to ``*_loose.csv`` siblings and leaves the certified files alone:

    docker run --rm -v "<repo>:/work" imperfect-info:latest \
        python experiments/sequential_general_sum.py

Two cheap checks need no solving at all, because they only re-read committed data:

    python experiments/sequential_general_sum.py certificate_from_selection_csv
    python experiments/sequential_general_sum.py replot_preemption_incentive_from_csv

Run with an unknown argument to list every entry point and the artifact it writes.
See REPRODUCING.md for the full entry-point-to-table map.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dealgame.general_sum import (EnumeratedAuction, expected_bid,
                                  own_profit_fictitious_play as sealed_fictitious_play)
from dealgame.sequential_general_sum import (SequentialAuctionTree,
                                             equilibrium_statistics,
                                             opening_bid_profit_curve,
                                             own_profit_fictitious_play)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
# Solved equilibria, keyed by (parameters, restart, budget). Reused across studies so a
# 500k-round solve is never paid for twice. Safe to delete; it only costs time.
CACHE_DIR = os.environ.get(
    "SEQ_SOLVE_CACHE", os.path.join(RESULTS_DIR, ".solve_cache"))

# The common value is 1..3, so the bid grid is made to span exactly [0, 3]: every level
# is economically live, and there is room for each bidder to raise four times. Bidding
# above the highest possible value is dominated, so widening the grid instead of
# refining it would only pad the tree with branches no bidder would ever take.
NUM_VALUES = 3
NUM_BIDS = 9
BID_STEP = 3.0 / (NUM_BIDS - 1)   # 0.375; grid is 0, 0.375, ..., 3.0
NOISE = 0.5
ROUNDS = 3                        # turns per bidder in the base game
BASE = {"num_values": NUM_VALUES, "num_bids": NUM_BIDS, "bid_step": BID_STEP,
        "num_rounds": ROUNDS, "signal_noise_0": NOISE, "signal_noise_1": NOISE}

TOEHOLD_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
ROUNDS_GRID = [1, 2, 3, 4]
ROUNDS_TOEHOLDS = [0.0, 0.15, 0.30]

# Fictitious play's NashConv falls off like C/t. C grows with the tree, so a sequential
# solve needs a far larger budget than the sealed tensor solver did. The budget is set
# from the measured decay rather than guessed: on the base game NashConv reaches ~2.5e-5
# by 300k rounds and ~1e-5 by 625k. 300k buys an equilibrium certified to about 0.01% of a
# bidder's payoff, which is far inside anything the economics turns on, and it keeps a
# 144-solve sweep to an overnight run rather than two days. Every solve still reports its
# own NashConv, and no comparative static is ever read off a solve that did not converge.
MAX_ITERATIONS = 300000
TOLERANCE = 1e-4

# A certify run sets the tolerance below anything the budget can reach, so every solve is
# ended by the budget and ``own_profit_fictitious_play`` reports converged=False even at
# NashConv ~1e-6. A budget-terminated solve with small NashConv is still an eps-Nash
# equilibrium, and the certificate is the achieved eps, not the reached-tolerance flag.
# When ``ACCEPT_EPS`` is set, a restart counts as an eps-equilibrium if its NashConv is at
# or below it (certify uses this); when it is None the studies fall back to the flag, which
# for a default run means the solve reached ``TOLERANCE``. This keeps the default behaviour
# identical (a converged default solve has NashConv <= TOLERANCE = the acceptance bound) and
# stops a tight, budget-terminated certify run from discarding genuine ~1e-6 equilibria.
ACCEPT_EPS = None

# Restarts are not a footnote in this paper, they are the result: the toehold pins down
# bidder 0's *value* but not its *conduct*, so every economic claim is made across a
# spread of starts rather than from the one equilibrium a uniform start happens to find.
NUM_RESTARTS = 6
RESTART_TOEHOLDS = [0.0, 0.15, 0.30]
RESTART_SEED = 20260713

# Probe-precision certification. The default 300k/1e-4 run exits at the stopping rule, so
# the committed certificate reports NashConv ~= 1e-4 and understates how tight the
# equilibria actually are. The ``certify`` entry point re-runs the selection study with the
# tolerance set far below anything the run will reach and a budget large enough to drive
# NashConv to ~1e-6, so the compute *budget* ends each solve rather than the stopping rule.
# A referee then reproduces the ~1e-6 figures the paper cites (Table 1) from released code
# instead of trusting uncommitted probe logs. The toehold set is the three rows of the
# multiplicity table (0.05 is the headline eps ~= 4e-6 row, absent from the default sweep).
# Solves cache by iteration budget, so the 2e6 certify solves never collide with the 300k
# solves the other studies reuse: certify computes its own tighter equilibria and leaves
# the rest of the cache untouched.
CERTIFY_ITERATIONS = 2_000_000
CERTIFY_TOLERANCE = 1e-8
CERTIFY_TOEHOLDS = [0.0, 0.05, 0.15]
# The two-move-artifact table (paper Table 2) reports deterrence at one, two, and three
# rounds; R=4 appears in no table, and at three-plus rounds the higher-toehold cells do not
# converge even at this budget (that non-convergence is itself a reported finding), so the
# certify rounds sweep stops at three. The R=3 cells are shared with the multiplicity
# certificate through the solve cache, so certifying both tables barely costs more than one.
CERTIFY_ROUNDS = [1, 2, 3]
# Toeholds for the certified mechanism figure. The full nine-point TOEHOLD_GRID at 2e6
# iterations would dominate the run, and the figure only needs enough of the range to show
# the premium collapsing against a rival that will not fold.
CERTIFY_INCENTIVE_TOEHOLDS = [0.0, 0.05, 0.10, 0.15, 0.20]
# A rival that folds no more often than this is fairly described as "does not fold". The
# least-deterring *equilibrium* rival often clears this bar and often does not, and which
# is which is the difference between the mechanism figure showing what it claims and
# showing its opposite, so it is recorded per row rather than assumed.
PASSIVE_MAX_DETERRENCE = 0.05
# A budget-terminated certify solve at or below this NashConv counts as an eps-equilibrium.
# It matches the default run's tolerance, so certify accepts exactly the equilibria a default
# run would have (its NashConv is far tighter), while rejecting genuine non-converged basins
# (e.g. a restart that stalls at ~2e-4 on a different value). The certificate still reports
# each profile's achieved eps, so nothing is rounded up or hidden.
CERTIFY_ACCEPT_EPS = 1e-4

# Picking the two representative profiles for the certificate by argmax/argmin deterrence
# alone lets numerical noise choose them. At toehold 0.15 five restarts sit within 2e-5 of
# each other in deterrence, and a bare argmax picked the one whose NashConv was 3.2e-5
# rather than the one at 5.3e-6, inflating the certified eps five-fold and opening a 2.2e-4
# gap in value0 that made the "same value to four decimals" claim false for that row. The
# profiles are economically the same conduct, so which one represents it should be decided
# by certification quality, not by the fifth decimal of a fold probability. Restarts within
# this band of the extreme are treated as tied on conduct and the tightest-eps one wins.
DETERRENCE_TIE = 1e-3

# The paper's Tables 1 and 2 are certified artifacts: they are only meaningful when the
# budget, not the stopping rule, ended each solve. A default 300k/1e-4 run produces the
# same three CSVs with eps ~= 1e-4 in every row and, for the certificate, without the
# toehold 0.05 row at all (0.05 is not in the default RESTART_TOEHOLDS). Silently
# overwriting the certified files with those would leave a referee holding data that
# contradicts the paper, so a non-certify run writes them under a ``_loose`` suffix
# instead. Only ``certify`` (and the cheap CSV re-derivation) write the real names.
CERTIFY_RUN = False

# The three artifacts that back paper Tables 1 and 2 and must come from a certify run.
CERTIFIED_ARTIFACTS = frozenset({
    "seq_multiplicity_certificate.csv",
    "seq_equilibrium_selection.csv",
    "seq_rounds_compounding.csv",
})

# Solves are independent, and a converged restart sweep is hours serially. One process per
# solve, each pinned to a single thread (set OMP_NUM_THREADS=1 in the container).
WORKERS = min(32, (os.cpu_count() or 4))

# Grid refinement: the same price range [0, 3] at three resolutions. 9 levels is the
# base; 7 is coarser, 13 finer. All keep 2*ROUNDS <= num_bids so the round cap, not the
# grid, is what limits the contest.
REFINE_BIDS = [7, 9, 13]
REFINE_TOEHOLDS = [0.0, 0.15, 0.30]
# The one refinement cell worth more compute, and how much more. See ``certify_deep``.
DEEP_CELL_BIDS = 13
DEEP_CELL_TOEHOLD = 0.15
DEEP_EXTRA_RESTARTS = 12


def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _results_path(name: str) -> str:
    """Where a study writes ``name``, protecting the certified artifacts.

    Outside a certify run the three files backing paper Tables 1 and 2 are diverted to a
    ``_loose`` sibling, so the documented default command can be run freely without
    destroying the certified data (see the CERTIFY_RUN note).
    """
    if name in CERTIFIED_ARTIFACTS and not CERTIFY_RUN:
        stem, ext = os.path.splitext(name)
        name = f"{stem}_loose{ext}"
    return os.path.join(RESULTS_DIR, name)


def _selection_row(stat: dict) -> dict:
    """One restart, rounded to the precision the selection CSV records.

    The certificate is built from these rounded rows rather than from raw floats so that
    the two ways of producing it -- a full ``certify`` solve and
    :func:`certificate_from_selection_csv` reading the committed per-restart CSV -- give
    byte-identical output. Without this the cheap path would differ from the expensive one
    in the seventh decimal and a referee could not tell a real disagreement from rounding.
    """
    return {
        "restart": int(stat["restart"]),
        "opening_bid": round(float(stat["expected_opening_bid0"]), 6),
        "p_rival_deterred": round(float(stat["p_rival_deterred"]), 6),
        "value0": round(float(stat["value0"]), 6),
        "value1": round(float(stat["value1"]), 6),
        "nashconv": round(float(stat["nashconv"]), 8),
        "reached_tolerance": int(stat["converged"]),
    }


def _build_certificate(rows: list[dict], toehold: float, accept_eps: float,
                       tolerance: float, iterations: int) -> dict | None:
    """The multiplicity certificate for one toehold, from that toehold's restart rows.

    ``rows`` are :func:`_selection_row` outputs. A restart is admitted as an
    eps-equilibrium by the magnitude of its NashConv (``accept_eps``), never by the
    reached-tolerance flag, which is False by construction in a budget-terminated run.

    The two representatives are the most- and least-deterring admitted profiles, with ties
    inside ``DETERRENCE_TIE`` broken toward the tightest NashConv (see the note there). The
    reported ``eps`` is the worse of the pair, so it is an upper bound on both.
    """
    good = [r for r in rows if r["nashconv"] <= accept_eps]
    if len(good) < 2:
        return None

    top = max(r["p_rival_deterred"] for r in good)
    bottom = min(r["p_rival_deterred"] for r in good)
    if top - bottom <= DETERRENCE_TIE:
        # Every admitted restart is the same conduct; there is no multiplicity to certify.
        return None
    deterring, passive = _representative_rivals(good)
    return {
        "toehold": toehold,
        "eps": max(deterring["nashconv"], passive["nashconv"]),
        "eps_deterring": deterring["nashconv"],
        "eps_passive": passive["nashconv"],
        "value0_deterring": deterring["value0"],
        "value0_passive": passive["value0"],
        "value0_gap": round(abs(deterring["value0"] - passive["value0"]), 8),
        "deterrence_deterring": deterring["p_rival_deterred"],
        "deterrence_passive": passive["p_rival_deterred"],
        "deterrence_gap": round(deterring["p_rival_deterred"] - passive["p_rival_deterred"], 8),
        "restart_deterring": deterring["restart"],
        "restart_passive": passive["restart"],
        "n_accepted": len(good),
        "accept_eps": accept_eps,
        "tolerance": tolerance,
        "iterations": iterations,
    }


def _representative_rivals(good: list[dict]) -> tuple[dict, dict]:
    """Pick the most- and least-deterring profiles, breaking ties toward the tightest solve.

    Shared by the certificate (Table 1) and the preemption figure (Figure 1) because they
    read the same solves and must not disagree about which two profiles represent a
    toehold. A plain argmax over deterrence lets numerical noise choose: several restarts
    sit within ~1e-5 of each other, and the one that overshoots is the one that has not
    settled, so it also carries the loosest NashConv. At toehold 0.15 that picked a
    profile at eps 3.0e-05 over three at eps ~3.3e-06 and moved the reported jump premium
    by a factor of six.
    """
    top = max(r["p_rival_deterred"] for r in good)
    bottom = min(r["p_rival_deterred"] for r in good)
    deterring = min((r for r in good if r["p_rival_deterred"] >= top - DETERRENCE_TIE),
                    key=lambda r: r["nashconv"])
    passive = min((r for r in good if r["p_rival_deterred"] <= bottom + DETERRENCE_TIE),
                  key=lambda r: r["nashconv"])
    return deterring, passive


def _write_certificate(certificates: list[dict], path: str | None = None) -> None:
    cert_csv = path or _results_path("seq_multiplicity_certificate.csv")
    with open(cert_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(certificates[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(certificates)
    print("  wrote results/seq_multiplicity_certificate.csv")


def certificate_from_selection_csv(out_path: str | None = None):
    """Rebuild the certificate from the committed per-restart CSV, in about a second.

    The certificate is a pure selection over ``results/seq_equilibrium_selection.csv``: it
    picks two of the restarts already in that file and copies their numbers. Regenerating
    it therefore does not need the 2e6-iteration solves back -- only the committed rows.
    This exists so a referee can check the certificate against the per-restart data without
    the ~20h ``certify`` run, and so the expensive path can be verified against a cheap one.
    Produces byte-identical output to ``certify`` (both build from :func:`_selection_row`).
    """
    global CERTIFY_RUN
    CERTIFY_RUN = True
    src = os.path.join(RESULTS_DIR, "seq_equilibrium_selection.csv")
    print(f"== Rebuilding the certificate from {src} ==", flush=True)
    by_toehold: dict[float, list[dict]] = {}
    with open(src, newline="") as fh:
        for raw in csv.DictReader(fh):
            row = {
                "restart": int(raw["restart"]),
                "opening_bid": float(raw["opening_bid"]),
                "p_rival_deterred": float(raw["p_rival_deterred"]),
                "value0": float(raw["value0"]),
                "value1": float(raw["value1"]),
                "nashconv": float(raw["nashconv"]),
            }
            by_toehold.setdefault(float(raw["toehold"]), []).append(row)

    certificates = []
    for theta in sorted(by_toehold):
        cert = _build_certificate(by_toehold[theta], theta, CERTIFY_ACCEPT_EPS,
                                  CERTIFY_TOLERANCE, CERTIFY_ITERATIONS)
        if cert is None:
            print(f"  toehold={theta:.2f}: fewer than two distinct admitted conducts; "
                  f"no certificate")
            continue
        certificates.append(cert)
        print(f"  toehold={theta:.2f}: restarts {cert['restart_deterring']} and "
              f"{cert['restart_passive']}; eps <= {cert['eps']:.2e}; "
              f"value gap {cert['value0_gap']:.2e}; "
              f"deterrence {cert['deterrence_passive']:.3f} vs "
              f"{cert['deterrence_deterring']:.3f}", flush=True)
    if certificates:
        _write_certificate(certificates, out_path)
    return certificates


def _restart_policies(tree: SequentialAuctionTree, restart: int):
    """Starting policies for a restart. Restart 0 is the uniform start; the rest are
    random full-support policies, so no information set starts out unreachable."""
    if restart == 0:
        return None
    rng = np.random.default_rng(RESTART_SEED + restart)
    return [tree.random_policy(0, rng), tree.random_policy(1, rng)]


def _accepted(stat: dict) -> bool:
    """Is this solve usable as an eps-equilibrium for the summaries and the certificate?

    Falls back to the reached-tolerance convergence flag unless ``ACCEPT_EPS`` is set, in
    which case a budget-terminated solve whose NashConv is at or below ``ACCEPT_EPS`` is
    accepted (see the ACCEPT_EPS note above).
    """
    if ACCEPT_EPS is None:
        return bool(stat["converged"])
    return stat["nashconv"] <= ACCEPT_EPS


def _solve_job(job: tuple) -> dict:
    """One (parameters, restart) solve. Module-level so a process pool can pickle it.

    Returns the economics plus the rival's policy, which the preemption-incentive study
    needs in order to ask what a jump bid is worth against *this particular* equilibrium.
    """
    params, restart, iterations, tolerance = job
    import pyspiel
    import dealgame  # noqa: F401

    game = pyspiel.load_game("dealgame_sequential_takeover", params)
    tree = SequentialAuctionTree(game)
    # One checkpoint per solve, beside its cache entry and keyed the same way, so a
    # machine that reboots mid-study resumes each cell instead of restarting it. The
    # 13-level certify cells are ~100h apiece; without this a reboot on day four costs
    # the whole run.
    result = own_profit_fictitious_play(
        tree, max_iterations=iterations, tolerance=tolerance,
        initial_policies=_restart_policies(tree, restart),
        checkpoint_path=os.path.join(CACHE_DIR, f"{_cache_key(job)}.ckpt"))
    # Certify the tightest profile the run visited, not the one it happened to stop on.
    # These are the same object on a solve that converged; they diverge by orders of
    # magnitude on one that cycled, which is precisely the case the old code reported
    # wrongly. ``wander`` is that gap, kept so a reader can tell the two apart.
    stats = equilibrium_statistics(tree, result["best_policy0"], result["best_policy1"])
    stats.update(nashconv=result["best_nashconv"], converged=result["converged"],
                 iterations=result["iterations"], restart=restart,
                 params=params, policy1=result["best_policy1"],
                 best_iteration=result["best_iteration"],
                 final_nashconv=result["nashconv"],
                 wander=result["nashconv"] / max(result["best_nashconv"], 1e-300))
    return stats


# Bump whenever _solve_job changes what a cached entry *means*, not just what it costs.
# v2: solves certify the best iterate rather than the final one, so a v1 entry carries a
# different (looser, and on a cycling cell wildly looser) NashConv for the same key. The
# cache directory is gitignored, so without this a stale entry would be served silently
# and the paper would mix two definitions of its own certificate.
_SOLVE_SCHEMA_VERSION = 2


def _cache_key(job: tuple) -> str:
    # Every input that changes the solve has to be in the key, tolerance included. It was
    # once omitted, which was harmless only because the two shipped configurations happen
    # to differ in budget too; tightening the tolerance alone would have silently served
    # the looser cached solve, and the cache directory is gitignored so nothing would show.
    params, restart, iterations, tolerance = job
    payload = json.dumps([sorted(params.items()), restart, iterations, repr(tolerance),
                          _SOLVE_SCHEMA_VERSION], sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _cache_result(path: str, stats: dict) -> None:
    """Publish one finished solve into the shared cache, atomically.

    Write-then-rename, with a temp file private to this writer. Two studies run
    concurrently share this cache and can land on the same cell, and the progress reader
    scans it live, so a reader must never catch a half-written entry and a writer must
    never rename away a file another writer still holds open.
    """
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".",
                               prefix=f"{os.path.basename(path)}.", suffix=".tmp")
    with os.fdopen(fd, "wb") as handle:
        pickle.dump(stats, handle)
    os.replace(tmp, path)


def _solve_many(jobs: list[tuple]) -> list[dict]:
    """Run solves across processes, reusing any that were already done.

    The studies overlap heavily: the headline and the incentive study want the same 54
    equilibria, and the selection study wants a subset of them. Each solve is half an hour
    of CPU, so recomputing them is a day of wasted compute. Results are keyed by the exact
    (parameters, restart, budget, tolerance) tuple, so a cache hit is the same solve by
    construction and changing any of them invalidates it automatically.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    os.makedirs(CACHE_DIR, exist_ok=True)
    paths = [os.path.join(CACHE_DIR, f"{_cache_key(job)}.pkl") for job in jobs]

    todo = [(job, path) for job, path in zip(jobs, paths) if not os.path.exists(path)]
    if len(todo) < len(jobs):
        print(f"  reusing {len(jobs) - len(todo)} cached solve(s)", flush=True)

    # Longest-processing-time-first. With a fixed pool the makespan is decided by when the
    # most expensive solves *start*, and cost runs away in the tree: a 13-level, 3-round
    # cell is ~25x a 7-level one. Left in natural order the refinement grid dispatches all
    # its cheap cells first and the expensive ones trail off the end, which cost several
    # hours of wall clock on the previous run. Results are keyed by job rather than by
    # position, so reordering changes nothing about what gets computed.
    todo.sort(key=lambda item: -(item[0][2] * item[0][0]["num_bids"]
                                 ** item[0][0]["num_rounds"]))

    if todo:
        # Consume results as they finish, not in submission order. The jobs are dispatched
        # longest-first, so submission order means nothing reaches disk until the slowest
        # cell in the batch returns: on the refinement grid that held 18 finished solves in
        # memory for days, invisible to the progress reader and to the sibling study that
        # wanted the same cells, and all of them lost if the container died.
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(_solve_job, job): path for job, path in todo}
            for future in as_completed(futures):
                _cache_result(futures[future], future.result())

    results = []
    for path in paths:
        with open(path, "rb") as handle:
            results.append(pickle.load(handle))
    return results


def _solve_with_tree(params: dict, max_iterations: int | None = None,
                     initial_policies=None) -> tuple[SequentialAuctionTree, dict, dict]:
    """Solve one sequential auction, keeping the cached tree for further queries.

    The iteration budget is resolved at call time, not bound as a default, so that
    lowering ``MAX_ITERATIONS`` actually lowers it. Binding it as a default silently
    ignores the override and runs the full budget, which looks exactly like a fast
    solve that converged.
    """
    import pyspiel
    import dealgame  # noqa: F401  (registers the game)

    budget = MAX_ITERATIONS if max_iterations is None else max_iterations
    game = pyspiel.load_game("dealgame_sequential_takeover", params)
    tree = SequentialAuctionTree(game)
    result = own_profit_fictitious_play(
        tree, max_iterations=budget, tolerance=TOLERANCE,
        initial_policies=initial_policies)
    # Same rule as _solve_job: certify the tightest profile the solve visited. Nothing
    # calls this helper today, and keeping it consistent is the point. Left reading the
    # final iterate, it would hand a future study a quietly different definition of the
    # paper's certificate, which is exactly how the two got mixed in the first place.
    stats = equilibrium_statistics(tree, result["best_policy0"], result["best_policy1"])
    stats["nashconv"] = result["best_nashconv"]
    stats["converged"] = result["converged"]
    stats["iterations"] = result["iterations"]
    stats["final_nashconv"] = result["nashconv"]
    stats["wander"] = result["nashconv"] / max(result["best_nashconv"], 1e-300)
    return tree, result, stats


def _solve(params: dict, max_iterations: int | None = None,
           initial_policies=None) -> tuple[dict, dict]:
    """Solve one sequential auction and return (equilibrium, economic statistics)."""
    _, result, stats = _solve_with_tree(params, max_iterations, initial_policies)
    return result, stats


def _sealed_expected_bid(toehold: float) -> tuple[float, float]:
    """Paper 1's sealed auction on the same bid grid: (expected bid, own profit).

    Solved here rather than cited so the sealed/sequential contrast is like-for-like:
    same values, same noise, same money grid, same equilibrium concept.
    """
    auction = EnumeratedAuction(
        num_values=NUM_VALUES, num_bids=NUM_BIDS, noise_0=NOISE, noise_1=NOISE,
        toehold=toehold, bid_values=np.linspace(0.0, NUM_VALUES, NUM_BIDS))
    result = sealed_fictitious_play(auction, max_iterations=400000)
    return expected_bid(auction, result["policy0"], 0), result["value0"]


def toehold_preemption_experiment():
    """Headline: the toehold pins down bidder 0's profit, but not its conduct.

    The obvious version of this study solves each toehold once and draws a deterrence
    curve. That curve does not exist. Solving each toehold from several starting policies
    finds equilibria that agree on bidder 0's value to four decimals and disagree wildly
    on whether it preempts at all: at a toehold of 0.05, restarts certified to within
    1e-5 of equilibrium deter the rival with probability 0.000 and 0.333. Both are
    self-consistent (see ``preemption_incentive_experiment``), so preemption is
    *sustainable*, not *implied*.

    What survives, and what the sealed overlay is for: paper 1's sealed auction on the
    identical bid grid has an equilibrium bid flat in the toehold and no preemption at
    all. So a sequential contest can express an aggressiveness channel that a sealed one
    cannot. That is paper 1's asserted claim, demonstrated. What cannot be claimed is a
    number for how much deterrence a given toehold buys.
    """
    print("== Toehold: value is pinned, conduct is not ==")
    jobs = [(dict(BASE, toehold=theta), restart, MAX_ITERATIONS, TOLERANCE)
            for theta in TOEHOLD_GRID for restart in range(NUM_RESTARTS)]
    print(f"  {len(jobs)} solves across {WORKERS} workers "
          f"({MAX_ITERATIONS} rounds each)", flush=True)
    solved = _solve_many(jobs)

    rows, summary = [], []
    for theta in TOEHOLD_GRID:
        at_theta = [s for s in solved if s["params"]["toehold"] == theta]
        rows.extend(at_theta)
        # An unconverged solve is not an equilibrium and cannot speak to what the
        # equilibria look like. Report how many were dropped rather than hiding it.
        good = [s for s in at_theta if s["converged"]]
        dropped = len(at_theta) - len(good)
        if not good:
            print(f"  toehold={theta:.2f}  no restart converged; nothing to report")
            continue

        sealed_bid, sealed_value = _sealed_expected_bid(theta)
        deter = [s["p_rival_deterred"] for s in good]
        values = [s["value0"] for s in good]
        summary.append({
            "toehold": theta, "n_converged": len(good), "n_dropped": dropped,
            "value0_min": min(values), "value0_max": max(values),
            "deter_min": min(deter), "deter_max": max(deter),
            "sealed_bid": sealed_bid, "sealed_value0": sealed_value,
        })
        note = f"  [{dropped} restart(s) did not converge]" if dropped else ""
        print(f"  toehold={theta:.2f}  value0 in [{min(values):.4f}, {max(values):.4f}] "
              f"(spread {max(values) - min(values):.5f})   "
              f"deterrence in [{min(deter):.3f}, {max(deter):.3f}] "
              f"(spread {max(deter) - min(deter):.3f}){note}", flush=True)

    # Write the per-restart record before anything can go wrong with the summary: at large
    # toeholds no restart converges at all, and that is a finding, not a crash.
    out_csv = os.path.join(RESULTS_DIR, "seq_toehold_preemption.csv")
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["toehold", "restart", "opening_bid", "p_rival_deterred",
                         "p_open_wait", "expected_price_given_sale", "p_win0", "p_win1",
                         "p_no_sale", "value0", "value1", "nashconv", "converged",
                         "iterations"])
        for s in rows:
            writer.writerow([
                s["params"]["toehold"], s["restart"],
                f"{s['expected_opening_bid0']:.6f}", f"{s['p_rival_deterred']:.6f}",
                f"{s['p_open_wait0']:.6f}", f"{s['expected_price_given_sale']:.6f}",
                f"{s['p_win0']:.6f}", f"{s['p_win1']:.6f}", f"{s['p_no_sale']:.6f}",
                f"{s['value0']:.6f}", f"{s['value1']:.6f}", f"{s['nashconv']:.8f}",
                int(s["converged"]), s["iterations"]])

    if not summary:
        print("  no toehold had a converged restart; wrote the per-restart record only")
        return

    thetas = [s["toehold"] for s in summary]
    fig, (ax_value, ax_deter) = plt.subplots(1, 2, figsize=(12, 5))

    ax_value.fill_between(thetas, [s["value0_min"] for s in summary],
                          [s["value0_max"] for s in summary], color="C0", alpha=0.35,
                          label="sequential: range across restarts")
    ax_value.plot(thetas, [s["value0_min"] for s in summary], marker="o", color="C0")
    ax_value.plot(thetas, [s["sealed_value0"] for s in summary], marker="s", color="C7",
                  ls="--", label="sealed (paper 1)")
    ax_value.set_xlabel("bidder 0 toehold $\\theta$")
    ax_value.set_ylabel("bidder 0 equilibrium profit")
    ax_value.set_title("Profit is pinned down\n(the band is there, it is just too thin to see)")
    ax_value.legend(fontsize=8)
    ax_value.grid(True, alpha=0.3)

    # Every converged restart as its own point. The scatter is the result: at a given
    # toehold the equilibria disagree, so there is no curve to draw through them.
    for theta in TOEHOLD_GRID:
        good = [s for s in solved
                if s["params"]["toehold"] == theta and s["converged"]]
        if good:
            ax_deter.scatter([theta] * len(good), [s["p_rival_deterred"] for s in good],
                             color="C3", alpha=0.65, zorder=3)
    ax_deter.set_xlabel("bidder 0 toehold $\\theta$")
    ax_deter.set_ylabel("P(rival concedes immediately)")
    ax_deter.set_title("Conduct is not\n(each dot is one converged equilibrium)")
    ax_deter.set_ylim(-0.03, None)
    ax_deter.grid(True, alpha=0.3)

    fig.suptitle("A toehold fixes what bidder 0 earns, not how it bids")
    fig.savefig(os.path.join(RESULTS_DIR, "seq_toehold_preemption.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    sum_csv = os.path.join(RESULTS_DIR, "seq_toehold_summary.csv")
    with open(sum_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print("  wrote results/seq_toehold_preemption.{png,csv} and seq_toehold_summary.csv")


def rounds_compounding_experiment():
    """The novelty qualifier: does the toehold's edge compound as the contest runs on?

    A toehold in a two-move contest is already understood. The claim this paper rests on
    is that its value keeps growing once information is revealed between several rounds,
    which is what separates it from a one-shot jump. If the toehold's effect were flat in
    the number of rounds, the contribution would collapse back onto the known two-move
    case, so this study is the one that can falsify the paper.

    It survives the multiplicity result intact, and that is not luck. The claim is about
    the toehold's *value*, and value is precisely the thing the equilibria agree on. The
    profit gain from holding a toehold is therefore a well-posed comparative static even
    though the deterrence probability is not, so the gain is plotted as the finding and
    deterrence only as a spread across restarts.
    """
    print("== Does the toehold's edge compound across rounds? ==")
    jobs = [(dict(BASE, num_rounds=rounds, toehold=theta), restart, MAX_ITERATIONS, TOLERANCE)
            for rounds in ROUNDS_GRID for theta in ROUNDS_TOEHOLDS
            for restart in range(NUM_RESTARTS)]
    print(f"  {len(jobs)} solves across {WORKERS} workers", flush=True)
    solved = _solve_many(jobs)

    def converged_at(rounds, theta):
        return [s for s in solved
                if s["params"]["num_rounds"] == rounds
                and s["params"]["toehold"] == theta and _accepted(s)]

    cells = {}
    for rounds in ROUNDS_GRID:
        for theta in ROUNDS_TOEHOLDS:
            good = converged_at(rounds, theta)
            if not good:
                print(f"  rounds={rounds}  toehold={theta:.2f}  no restart converged")
                continue
            values = [s["value0"] for s in good]
            deter = [s["p_rival_deterred"] for s in good]
            cells[(rounds, theta)] = {
                "value0_min": min(values), "value0_max": max(values),
                "deter_min": min(deter), "deter_max": max(deter), "n": len(good),
            }
            print(f"  rounds={rounds}  toehold={theta:.2f}  "
                  f"value0 in [{min(values):.4f}, {max(values):.4f}]  "
                  f"deterrence in [{min(deter):.3f}, {max(deter):.3f}]  "
                  f"({len(good)}/{NUM_RESTARTS} converged)", flush=True)

    fig, (ax_gain, ax_deter) = plt.subplots(1, 2, figsize=(12, 5))

    # The identified claim. The toehold's value gain is a difference of equilibrium
    # values, and the restarts agree on values, so this comparative static is well posed.
    for theta in [t for t in ROUNDS_TOEHOLDS if t > 0]:
        xs, gains = [], []
        for rounds in ROUNDS_GRID:
            cell, base = cells.get((rounds, theta)), cells.get((rounds, 0.0))
            if cell and base:
                xs.append(rounds)
                gains.append(cell["value0_min"] - base["value0_min"])
        ax_gain.plot(xs, gains, marker="s", label=f"$\\theta$={theta:.2f}")
    ax_gain.set_xlabel("rounds (turns per bidder)")
    ax_gain.set_ylabel("own-profit gain from the toehold")
    ax_gain.set_title("The toehold's value against contest length\n(identified: restarts agree on value)")
    ax_gain.set_xticks(ROUNDS_GRID)
    ax_gain.legend(fontsize=8)
    ax_gain.grid(True, alpha=0.3)

    # The unidentified one, shown as the band it really is.
    for theta in ROUNDS_TOEHOLDS:
        xs = [r for r in ROUNDS_GRID if (r, theta) in cells]
        if not xs:
            continue
        lo = [cells[(r, theta)]["deter_min"] for r in xs]
        hi = [cells[(r, theta)]["deter_max"] for r in xs]
        ax_deter.fill_between(xs, lo, hi, alpha=0.3, label=f"$\\theta$={theta:.2f}")
    ax_deter.set_xlabel("rounds (turns per bidder)")
    ax_deter.set_ylabel("P(rival concedes immediately)")
    ax_deter.set_title("Deterrence across restarts\n(not identified: the band is the answer)")
    ax_deter.set_xticks(ROUNDS_GRID)
    ax_deter.legend(fontsize=8)
    ax_deter.grid(True, alpha=0.3)

    fig.suptitle("The toehold's profit gain (identified) and its deterrence band (not identified) by contest length")
    fig.savefig(os.path.join(RESULTS_DIR, "seq_rounds_compounding.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    out_csv = _results_path("seq_rounds_compounding.csv")
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["num_rounds", "toehold", "restart", "opening_bid",
                         "p_rival_deterred", "expected_price_given_sale", "value0",
                         "value1", "nashconv", "reached_tolerance", "accepted",
                         "tolerance", "iterations"])
        accept_eps = CERTIFY_ACCEPT_EPS if ACCEPT_EPS is None else ACCEPT_EPS
        for s in solved:
            writer.writerow([
                s["params"]["num_rounds"], s["params"]["toehold"], s["restart"],
                f"{s['expected_opening_bid0']:.6f}", f"{s['p_rival_deterred']:.6f}",
                f"{s['expected_price_given_sale']:.6f}", f"{s['value0']:.6f}",
                f"{s['value1']:.6f}", f"{s['nashconv']:.8f}", int(s["converged"]),
                int(round(float(s["nashconv"]), 8) <= accept_eps),
                f"{TOLERANCE:g}", MAX_ITERATIONS])
    print("  wrote results/seq_rounds_compounding.{png,csv}")


def equilibrium_selection_experiment():
    """The certificate. Exhibit two equilibria that agree on profit and disagree on conduct.

    This is the study the paper's central claim is made from, so it is stated in the form
    a sceptic can check. For each toehold it pulls out the most- and least-deterring
    converged restarts and prints both with their NashConv, which is an exact
    best-response gap: a profile with NashConv eps is an eps-Nash equilibrium, meaning no
    bidder gains more than eps by deviating.

    Note what is and is not claimed. A small NashConv certifies an *eps*-equilibrium, and
    in a game this flat an eps-equilibrium can sit far from any exact one, so we do not
    claim two exact Nash equilibria exist. The claim is the checkable one: there are two
    profiles, each within eps of equilibrium for eps on the order of 1e-5 (about 0.004% of
    a bidder's payoff), that give bidder 0 the same profit to four decimals while deterring
    the rival with probability 0.00 and 0.33. That alone is enough to establish that
    deterrence is not identified at any precision we or a referee can reach.
    """
    print("== Equilibrium selection: same profit, different conduct ==")
    jobs = [(dict(BASE, toehold=theta), restart, MAX_ITERATIONS, TOLERANCE)
            for theta in RESTART_TOEHOLDS for restart in range(NUM_RESTARTS)]
    print(f"  {len(jobs)} solves across {WORKERS} workers", flush=True)
    solved = _solve_many(jobs)

    accept_eps = CERTIFY_ACCEPT_EPS if ACCEPT_EPS is None else ACCEPT_EPS
    certificates = []
    for theta in RESTART_TOEHOLDS:
        rows = [_selection_row(s) for s in
                sorted([s for s in solved if s["params"]["toehold"] == theta],
                       key=lambda s: s["restart"])]
        for row in rows:
            flag = "" if row["nashconv"] <= accept_eps else "  [excluded: eps too loose]"
            print(f"  toehold={theta:.2f}  restart={row['restart']}  "
                  f"opening bid={row['opening_bid']:.3f}  "
                  f"rival folds={row['p_rival_deterred']:.3f}  "
                  f"value0={row['value0']:+.4f}  "
                  f"NashConv={row['nashconv']:.2e}{flag}", flush=True)

        cert = _build_certificate(rows, theta, accept_eps, TOLERANCE, MAX_ITERATIONS)
        if cert is None:
            print(f"    -> toehold={theta:.2f}: fewer than two admitted restarts on "
                  f"distinct conducts; no certificate")
            continue
        certificates.append(cert)
        print(f"    -> toehold={theta:.2f}: two eps-equilibria with eps <= "
              f"{cert['eps']:.1e} (restarts {cert['restart_deterring']} and "
              f"{cert['restart_passive']}); profit differs by {cert['value0_gap']:.5f} "
              f"but deterrence differs by {cert['deterrence_gap']:.3f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for theta in RESTART_TOEHOLDS:
        series = [s for s in solved
                  if s["params"]["toehold"] == theta and _accepted(s)]
        if series:
            ax.scatter([theta] * len(series), [s["p_rival_deterred"] for s in series],
                       alpha=0.75, s=55, label=f"$\\theta$={theta:.2f}")
    ax.set_xlabel("bidder 0 toehold $\\theta$")
    ax.set_ylabel("P(rival concedes immediately)")
    ax.set_ylim(-0.03, None)
    ax.set_title(f"Each dot is a converged equilibrium from one of {NUM_RESTARTS} starts.\n"
                 "They agree on profit and disagree on conduct.")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(RESULTS_DIR, "seq_equilibrium_selection.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    out_csv = _results_path("seq_equilibrium_selection.csv")
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["toehold", "restart", "opening_bid", "p_rival_deterred",
                         "value0", "value1", "nashconv", "reached_tolerance", "accepted",
                         "tolerance", "iterations"])
        for s in solved:
            row = _selection_row(s)
            writer.writerow([
                s["params"]["toehold"], row["restart"],
                f"{row['opening_bid']:.6f}", f"{row['p_rival_deterred']:.6f}",
                f"{row['value0']:.6f}", f"{row['value1']:.6f}", f"{row['nashconv']:.8f}",
                row["reached_tolerance"], int(row["nashconv"] <= accept_eps),
                f"{TOLERANCE:g}", MAX_ITERATIONS])

    if certificates:
        _write_certificate(certificates)
    print("  wrote results/seq_equilibrium_selection.{png,csv}")


def bid_grid_refinement_experiment():
    """Is any of this an artifact of a coarse price grid?

    A jump is measured in money, so grid resolution is not a detail: too coarse a grid can
    manufacture a jump (the smallest legal raise is already large) or hide one. The price
    range is held at [0, 3] and only the number of levels changes.

    With multiplicity established, this study carries a second and more important job. If
    the spread in deterrence across restarts narrowed as the grid refined, the multiplicity
    would be a discretisation artifact and the honest conclusion would be that a finer grid
    pins conduct down after all. So the deterrence *band* is plotted at each resolution: it
    is the multiplicity itself that is under test here, not just the jump.
    """
    print("== Bid-grid refinement: same price range, three resolutions ==")
    jobs = []
    for num_bids in REFINE_BIDS:
        step = float(NUM_VALUES) / (num_bids - 1)
        for theta in REFINE_TOEHOLDS:
            for restart in range(NUM_RESTARTS):
                jobs.append((dict(BASE, num_bids=num_bids, bid_step=step, toehold=theta),
                             restart, MAX_ITERATIONS, TOLERANCE))
    print(f"  {len(jobs)} solves across {WORKERS} workers", flush=True)
    solved = _solve_many(jobs)

    cells = {}
    for num_bids in REFINE_BIDS:
        for theta in REFINE_TOEHOLDS:
            good = [s for s in solved
                    if s["params"]["num_bids"] == num_bids
                    and s["params"]["toehold"] == theta and _accepted(s)]
            if not good:
                print(f"  levels={num_bids}  toehold={theta:.2f}  no restart converged")
                continue
            values = [s["value0"] for s in good]
            deter = [s["p_rival_deterred"] for s in good]
            cells[(num_bids, theta)] = {
                "value0_min": min(values), "value0_max": max(values),
                "deter_min": min(deter), "deter_max": max(deter), "n": len(good)}
            print(f"  levels={num_bids} (step={good[0]['params']['bid_step']:.3f})  "
                  f"toehold={theta:.2f}  value0 in [{min(values):.4f}, {max(values):.4f}]  "
                  f"deterrence in [{min(deter):.3f}, {max(deter):.3f}]  "
                  f"({len(good)}/{NUM_RESTARTS} converged)", flush=True)


    fig, (ax_value, ax_deter) = plt.subplots(1, 2, figsize=(12, 5))
    for num_bids in REFINE_BIDS:
        xs = [t for t in REFINE_TOEHOLDS if (num_bids, t) in cells]
        if not xs:
            continue
        step = float(NUM_VALUES) / (num_bids - 1)
        label = f"{num_bids} levels (step {step:.2f})"
        # Pin the colour to the resolution. fill_between and vlines advance the property
        # cycle differently, so leaving it implicit gave the thirteen-level interval the
        # same blue as the seven-level band and made the legend say two things at once.
        colour = f"C{REFINE_BIDS.index(num_bids)}"
        ax_value.plot(xs, [cells[(num_bids, t)]["value0_min"] for t in xs],
                      marker="o", color=colour, label=label)
        # A resolution that certifies only one toehold has no band to fill between, and
        # at thirteen levels that is the only cell that converges. Drawn as an explicit
        # interval so the finest grid, which is the one the refinement question is about,
        # does not silently vanish from the panel while keeping its legend entry.
        lo = [cells[(num_bids, t)]["deter_min"] for t in xs]
        hi = [cells[(num_bids, t)]["deter_max"] for t in xs]
        if len(xs) > 1:
            ax_deter.fill_between(xs, lo, hi, color=colour, alpha=0.3, label=label)
        else:
            ax_deter.vlines(xs, lo, hi, color=colour, linewidth=6, alpha=0.5, label=label)
        ax_deter.plot(xs, lo, marker="_", ls="none", color=colour, markersize=12)
        ax_deter.plot(xs, hi, marker="_", ls="none", color=colour, markersize=12)
    ax_value.set_xlabel("bidder 0 toehold $\\theta$")
    ax_value.set_ylabel("bidder 0 equilibrium profit")
    # Not "profit is the same at every resolution": it is not. Changing the grid changes
    # the game, so the equilibrium value moves with it (0.208, 0.184 and 0.174 at a zero
    # toehold), and a title claiming otherwise would be contradicted by its own panel.
    ax_value.set_title("Profit moves with the grid\n"
                       "(a different grid is a different game)")
    ax_value.legend(fontsize=8)
    ax_value.grid(True, alpha=0.3)
    ax_deter.set_xlabel("bidder 0 toehold $\\theta$")
    ax_deter.set_ylabel("P(rival concedes immediately)")
    ax_deter.set_title("Conduct does not settle as the grid refines\n"
                       "(the low-deterrence branch is absent only at 7 levels)")
    ax_deter.legend(fontsize=8)
    ax_deter.grid(True, alpha=0.3)
    fig.suptitle("Refining the price grid does not pin conduct down", y=1.02)
    fig.savefig(os.path.join(RESULTS_DIR, "seq_grid_refinement.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    # ``nashconv`` is the certified eps: the tightest profile the solve visited, which is
    # the profile every other column on the row describes. ``final_nashconv`` is where the
    # run happened to stop and ``wander`` is the ratio. A wander of 1.0 means the solve
    # settled and the two coincide; a large one means it left its basin, which at these
    # toeholds it does, and is the only column that tells the two cases apart.
    out_csv = os.path.join(RESULTS_DIR, "seq_grid_refinement.csv")
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["num_bids", "bid_step", "toehold", "restart", "opening_bid",
                         "p_rival_deterred", "value0", "nashconv", "reached_tolerance",
                         "final_nashconv", "wander", "best_iteration", "iterations",
                         "accepted"])
        for s in sorted(solved, key=lambda r: (r["params"]["num_bids"],
                                               r["params"]["toehold"], r["restart"])):
            writer.writerow([
                s["params"]["num_bids"], f"{s['params']['bid_step']:.6f}",
                s["params"]["toehold"], s["restart"],
                f"{s['expected_opening_bid0']:.6f}", f"{s['p_rival_deterred']:.6f}",
                f"{s['value0']:.6f}", f"{s['nashconv']:.10f}", int(s["converged"]),
                f"{s['final_nashconv']:.10f}", f"{s['wander']:.3f}",
                s["best_iteration"], s["iterations"], int(_accepted(s))])
    print("  wrote results/seq_grid_refinement.{png,csv}")


def preemption_incentive_experiment():
    """What is a jump bid actually worth? Measured exactly, without trusting the solver.

    The other studies read behaviour off a fictitious-play solve. That is circular
    where the solve is itself in doubt, and at a large toehold it is: play cycles
    instead of settling. So this study asks the question that does not depend on the
    solver converging. Hold the rival fixed, force bidder 0 to open at each level of
    the money grid, and let every later decision best-respond exactly. The profit
    curve that comes back is an exact object.

    Its shape is the finding. A curve with a clear interior peak means preempting
    strictly pays, and the opening bid is pinned down. A flat curve means the bidder
    is indifferent across openings, so nothing pins the opening down, and a solver
    that wanders there is reporting the game's own flatness rather than failing.

    The curve against a uniformly random rival is the control: a rival that cannot be
    scared off is a rival not worth paying a premium to scare off, so preemption
    should earn nothing against it.
    """
    print("== What a jump bid is worth: exact profit vs a forced opening bid ==")
    import pyspiel
    import dealgame  # noqa: F401

    jobs = [(dict(BASE, toehold=theta), restart, MAX_ITERATIONS, TOLERANCE)
            for theta in TOEHOLD_GRID for restart in range(NUM_RESTARTS)]
    print(f"  {len(jobs)} solves across {WORKERS} workers", flush=True)
    solved = _solve_many(jobs)

    rows, curves = [], {}
    for theta in TOEHOLD_GRID:
        # Accept by NashConv magnitude, not the reached-tolerance flag: under ``certify``
        # every solve is budget-terminated and the flag is False even at 1e-6, so gating
        # on it here would silently drop every rival and leave the study with nothing.
        good = [s for s in solved
                if s["params"]["toehold"] == theta and _accepted(s)]
        if not good:
            print(f"  toehold={theta:.2f}  no restart admitted; skipping")
            continue

        tree = SequentialAuctionTree(pyspiel.load_game(
            "dealgame_sequential_takeover", dict(BASE, toehold=theta)))

        # The whole point: hold up the most- and least-deterring equilibria side by side
        # and ask what preempting is worth against each. If jump-bidding pays against the
        # rival who folds and does not pay against the rival who does not, then both
        # conducts are self-consistent and neither is "the" prediction.
        #
        # The labels say most/least *deterring*, not "passive", because the least-deterring
        # equilibrium is frequently not passive: at several toeholds it still folds a fifth
        # to a third of the time, and calling it "the rival who does not fold" turned the
        # middle panel into a near-copy of the left one under a caption saying the opposite.
        # ``is_passive`` records which ones actually earn that description.
        # Every certified rival, not a chosen pair. Selecting one most-deterring and one
        # least-deterring profile made the reported premium a property of the selection
        # rule rather than of the game: profiles certified at the same eps, deterring with
        # the same probability to five decimals, give premia eighteen-fold apart (2.7% and
        # 49.0% at toehold 0.10). Deterrence is one scalar summary of a strategy defined
        # over many information sets, and this curve probes exactly the off-path behaviour
        # that scalar discards. The interval across certified rivals is what is identified.
        rivals = [(f"eqm-restart-{s['restart']}", s["policy1"], s)
                  for s in sorted(good, key=lambda s: s["restart"])]
        # Control: a rival bidding at random cannot be scared off, so paying a premium to
        # scare it off should earn exactly nothing.
        rivals.append(("uniform-rival", tree.uniform_policy(1), None))

        for label, rival, source in rivals:
            bids, profits = opening_bid_profit_curve(tree, rival)
            # Drop the trailing "wait" option: it names no price, so it is not a point
            # on a curve over bid levels.
            bids, profits = bids[:-1], profits[:-1]
            peak = int(np.argmax(profits))
            # The economics: committing to the best opening, versus opening at zero.
            premium = float(profits[peak] - profits[0])
            relative = premium / abs(profits[peak]) if profits[peak] else float("nan")
            # The paper phrases the premium as "over opening at the bottom of the grid",
            # which is this one, not ``relative_premium`` (whose base is the peak). Both
            # are written out so a quoted percentage names an unambiguous column.
            over_zero = premium / abs(profits[0]) if profits[0] else float("nan")
            deterrence = source["p_rival_deterred"] if source else float("nan")
            rows.append({
                "toehold": theta, "rival": label,
                "restart": ("" if source is None else source["restart"]),
                "rival_deterrence": deterrence,
                "is_passive": ("" if source is None
                               else int(deterrence <= PASSIVE_MAX_DETERRENCE)),
                "nashconv": (source["nashconv"] if source else float("nan")),
                "peak_bid": float(bids[peak]), "peak_profit": float(profits[peak]),
                "profit_opening_at_zero": float(profits[0]),
                "preemption_premium": premium,
                "relative_premium": relative,
                "premium_over_opening_at_zero": over_zero,
            })
            curves[(theta, label)] = (bids, profits)
            print(f"  toehold {theta:.2f} vs {label:<16s} peak bid {bids[peak]:.3f} "
                  f"premium {premium:+.4f} ({over_zero:+.1%} over opening at zero)",
                  flush=True)

        # The interval, which is the quantity the paper is entitled to quote.
        spread = [r["premium_over_opening_at_zero"] for r in rows
                  if r["toehold"] == theta and r["rival"] != "uniform-rival"]
        print(f"  toehold {theta:.2f} RANGE over {len(spread)} certified rivals: "
              f"{min(spread):+.1%} to {max(spread):+.1%}", flush=True)

    _ensure_results_dir()
    if not rows:
        print("  no toehold had a converged restart; nothing to measure against")
        return

    path = os.path.join(RESULTS_DIR, "seq_preemption_incentive.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    # The summary above reports the peak and the open-at-zero point, which cannot tell a
    # flat curve from a steeply falling one. Both readings are economically different, so
    # write out every point and let the curve be checked rather than trusted.
    curve_path = os.path.join(RESULTS_DIR, "seq_preemption_curves.csv")
    with open(curve_path, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["toehold", "rival", "opening_bid", "own_profit"])
        for (theta, label), (bids, profits) in curves.items():
            for bid, profit in zip(bids, profits):
                writer.writerow([theta, label, f"{bid:.4f}", f"{profit:.6f}"])

    _plot_preemption_panels(curves, rows)
    print(f"  wrote {path}")


def _plot_preemption_panels(curves: dict, rows: list[dict]) -> None:
    """Draw the three forced-opening panels (paper Figure 1).

    The panels used to be one hand-picked rival each. That presented the premium as a
    number when it is an interval: certified profiles that deter identically give premia
    an order of magnitude apart, so any single curve is a statement about the selection
    rule. The middle panel is now that interval, which is the quantity the data pins down.
    """
    thetas = sorted({theta for theta, _ in curves})
    shade = {theta: plt.cm.viridis(i / max(len(thetas) - 1, 1))
             for i, theta in enumerate(thetas)}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    for (theta, label), (bids, profits) in sorted(curves.items()):
        if label == "uniform-rival":
            continue
        axes[0].plot(bids, profits, color=shade[theta], alpha=0.8, linewidth=1.1,
                     marker="o", markersize=2.5)
    axes[0].set_title("vs every certified equilibrium rival\n"
                      "(one line per accepted restart)", fontsize=10)
    axes[0].set_ylabel("bidder 0 own profit (continuation optimal)")
    axes[0].set_xlabel("opening bid bidder 0 commits to")

    equilibria = [r for r in rows if r["rival"] != "uniform-rival"]
    for theta in thetas:
        premia = [float(r["premium_over_opening_at_zero"]) * 100 for r in equilibria
                  if float(r["toehold"]) == theta]
        if not premia:
            continue
        axes[1].vlines(theta, min(premia), max(premia), color=shade[theta], linewidth=2.5)
        axes[1].plot([theta] * len(premia), premia, "o", color=shade[theta],
                     markersize=5, markeredgecolor="white", markeredgewidth=0.5)
    axes[1].set_title("how much the jump is worth is not pinned down\n"
                      "(every certified rival, same game, same eps)", fontsize=10)
    axes[1].set_ylabel("premium over opening at the bottom of the grid (%)")
    axes[1].set_xlabel("toehold")
    axes[1].set_xticks(thetas)
    axes[1].set_xticklabels([f"{theta:.2f}" for theta in thetas])
    axes[1].set_xlim(min(thetas) - 0.02, max(thetas) + 0.02)

    for theta in thetas:
        if (theta, "uniform-rival") not in curves:
            continue
        bids, profits = curves[(theta, "uniform-rival")]
        axes[2].plot(bids, profits, color=shade[theta], marker="o", markersize=3,
                     label=f"toehold {theta:.2f}")
    axes[2].set_title("vs a rival bidding at random\n"
                      "(cannot be deterred: control)", fontsize=10)
    axes[2].set_xlabel("opening bid bidder 0 commits to")
    axes[2].legend(fontsize=7, ncol=2)

    for axis in axes:
        axis.axhline(0.0, color="0.8", linewidth=0.8, zorder=0)
        axis.grid(True, alpha=0.3)
    fig.suptitle("A jump bid pays only against a rival that folds, and how much it pays "
                 "is not pinned down by equilibrium")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "seq_preemption_incentive.png"), dpi=160)
    plt.close(fig)


def replot_preemption_incentive_from_csv():
    """Redraw paper Figure 1 from the committed curve CSV, without re-solving.

    Every point in the figure is a row of ``results/seq_preemption_curves.csv``, so the
    panel can be corrected (or checked) in a second rather than by repeating the solves
    that produced the rival policies.
    """
    curve_path = os.path.join(RESULTS_DIR, "seq_preemption_curves.csv")
    inc_path = os.path.join(RESULTS_DIR, "seq_preemption_incentive.csv")
    print(f"== Redrawing paper Figure 1 from {curve_path} ==", flush=True)

    curves: dict = {}
    with open(curve_path, newline="") as fh:
        for raw in csv.DictReader(fh):
            key = (float(raw["toehold"]), raw["rival"])
            bids, profits = curves.setdefault(key, ([], []))
            bids.append(float(raw["opening_bid"]))
            profits.append(float(raw["own_profit"]))

    with open(inc_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    _plot_preemption_panels(curves, rows)
    print("  wrote results/seq_preemption_incentive.png")


def certify():
    """Regenerate the multiplicity certificate at probe precision (NashConv ~1e-6).

    The default :func:`equilibrium_selection_experiment` stops at ``TOLERANCE`` = 1e-4, so
    the committed ``seq_multiplicity_certificate.csv`` reports eps ~= 1e-4 and understates
    the certification. This entry point drives the same equilibria far past the default
    stopping rule (tolerance 1e-8, budget 2e6), so NashConv reaches ~1e-6 and the budget,
    not the stopping rule, ends each solve. It regenerates both tables the paper cites at
    probe precision: the multiplicity certificate (Table 1) over the three headline toeholds,
    and the two-move-artifact rounds sweep (Table 2) over one to three rounds. A referee then
    reproduces the ~1e-6 figures from released code, rather than trusting uncommitted probe
    logs. The R=3 solves are shared between the two passes through the solve cache.

    Run inside the container (expect several hours; the 2e6-round solves are ~6-7x the
    default budget, run in parallel across the process pool):
        docker run --rm -v "<repo>:/work" imperfect-info:latest \\
            python experiments/sequential_general_sum.py certify
    """
    global MAX_ITERATIONS, TOLERANCE, RESTART_TOEHOLDS, ROUNDS_GRID, ACCEPT_EPS
    global TOEHOLD_GRID, CERTIFY_RUN
    MAX_ITERATIONS = CERTIFY_ITERATIONS
    TOLERANCE = CERTIFY_TOLERANCE
    RESTART_TOEHOLDS = CERTIFY_TOEHOLDS
    ROUNDS_GRID = CERTIFY_ROUNDS
    ACCEPT_EPS = CERTIFY_ACCEPT_EPS
    CERTIFY_RUN = True
    print(f"== CERTIFY: {CERTIFY_ITERATIONS} rounds, tolerance {CERTIFY_TOLERANCE:.0e} ==",
          flush=True)
    print(f"  Table 1 (multiplicity) toeholds {CERTIFY_TOEHOLDS}; "
          f"Table 2 (two-move artifact) rounds {CERTIFY_ROUNDS}", flush=True)
    print("  the budget, not the stopping rule, ends each solve; "
          "NashConv should reach ~1e-6 where the game is not flat", flush=True)
    equilibrium_selection_experiment()
    rounds_compounding_experiment()
    # The mechanism figure (paper Figure 1) reads the forced-opening profit curve against
    # equilibrium rival policies. The forced-opening best response is exact, but those
    # rivals are fictitious-play output, so at the default budget they stop *at* the 1e-4
    # tolerance -- the "let go, not converged" state the paper disqualifies elsewhere.
    # Certifying the figure means re-solving its rivals at probe precision too.
    TOEHOLD_GRID = CERTIFY_INCENTIVE_TOEHOLDS
    preemption_incentive_experiment()


def certify_deep():
    """Add restarts to the one refinement cell where the basin may simply be unsampled.

    At thirteen levels and a toehold of 0.15 no restart is certified, and the natural
    reading is that the cell needs a bigger budget. The diagnostic says otherwise. Every
    restart close to the acceptance bound is close because it cycled and the best-iterate
    rule caught a favourable snapshot (the nearest has ``wander`` 121, meaning its final
    iterate was 121x worse than its certified one), while every cleanly converged restart
    sits 46x or further away. More iterations buy more cycling, not more precision.

    What has *not* been tested is whether the interesting basin was ever sampled. At nine
    levels the low-deterrence branch appears in exactly ONE of six restarts, so at thirteen
    it may be missing by luck rather than by refinement. This adds twelve fresh starts at
    the same budget, which is the failure mode this paper's own methodological caution
    names: a sample of restarts that agree proves nothing, because the disagreeing basin
    may not be in the sample.

    Reports the whole cell, original restarts included, so the answer is a spread over
    eighteen starts rather than over the twelve new ones alone.

    Run detached; it is about four days on twelve cores:
        docker run -d --name seq_certify_deep --restart on-failure \\
            -v "<repo>:/work" -w /work imperfect-info:latest \\
            python experiments/sequential_general_sum.py certify_deep
    """
    global MAX_ITERATIONS, TOLERANCE, ACCEPT_EPS, CERTIFY_RUN
    MAX_ITERATIONS = CERTIFY_ITERATIONS
    TOLERANCE = CERTIFY_TOLERANCE
    ACCEPT_EPS = CERTIFY_ACCEPT_EPS
    CERTIFY_RUN = True

    step = float(NUM_VALUES) / (DEEP_CELL_BIDS - 1)
    params = dict(BASE, num_bids=DEEP_CELL_BIDS, bid_step=step,
                  toehold=DEEP_CELL_TOEHOLD)
    total = NUM_RESTARTS + DEEP_EXTRA_RESTARTS
    print(f"== DEEP RESTARTS: {DEEP_CELL_BIDS} levels, toehold "
          f"{DEEP_CELL_TOEHOLD}, restarts 0..{total - 1} ==", flush=True)
    print(f"  budget {CERTIFY_ITERATIONS} at tolerance {CERTIFY_TOLERANCE:.0e}, "
          f"accepting NashConv <= {CERTIFY_ACCEPT_EPS:.0e}", flush=True)
    print("  under test: is the low-deterrence branch absent at this resolution, or "
          "was it never sampled?", flush=True)

    jobs = [(params, restart, MAX_ITERATIONS, TOLERANCE) for restart in range(total)]
    solved = _solve_many(jobs)

    good = [s for s in solved if _accepted(s)]
    print(f"  {len(good)}/{total} certified at NashConv <= {CERTIFY_ACCEPT_EPS:.0e}",
          flush=True)
    for stat in sorted(solved, key=lambda s: s["restart"]):
        mark = "  " if _accepted(stat) else " x"
        print(f"  {mark} restart {stat['restart']:>2}  eps {stat['nashconv']:.2e}  "
              f"wander {stat['wander']:>7.1f}  deterrence {stat['p_rival_deterred']:.4f}  "
              f"value0 {stat['value0']:.4f}", flush=True)

    _ensure_results_dir()
    path = _results_path("seq_grid_deep_restarts.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, lineterminator="\n",
            fieldnames=["num_bids", "bid_step", "toehold", "restart", "opening_bid",
                        "p_rival_deterred", "value0", "nashconv", "final_nashconv",
                        "wander", "best_iteration", "iterations", "accepted"])
        writer.writeheader()
        for stat in sorted(solved, key=lambda s: s["restart"]):
            writer.writerow({
                "num_bids": DEEP_CELL_BIDS, "bid_step": f"{step:.6f}",
                "toehold": DEEP_CELL_TOEHOLD, "restart": stat["restart"],
                "opening_bid": f"{stat['expected_opening_bid0']:.6f}",
                "p_rival_deterred": f"{stat['p_rival_deterred']:.6f}",
                "value0": f"{stat['value0']:.6f}",
                "nashconv": f"{stat['nashconv']:.10f}",
                "final_nashconv": f"{stat['final_nashconv']:.10f}",
                "wander": f"{stat['wander']:.3f}",
                "best_iteration": stat["best_iteration"],
                "iterations": stat["iterations"],
                "accepted": int(_accepted(stat))})
    print(f"  wrote {path}", flush=True)

    if not good:
        print("  VERDICT  still nothing certified here; the cell does not converge at "
              "this budget and the paper's limitation stands as written.", flush=True)
        return
    spread = [s["p_rival_deterred"] for s in good]
    print(f"  VERDICT  deterrence over {len(good)} certified restart(s): "
          f"{min(spread):.4f} to {max(spread):.4f}", flush=True)
    if max(spread) - min(spread) > DETERRENCE_TIE:
        print("           the multiplicity IS present at this resolution and a positive "
              "toehold; earlier absence was an unsampled basin.", flush=True)
    else:
        print("           one conduct only, so the added starts found no second basin.",
              flush=True)


ENTRY_POINTS = {
    "certify": "the certified run behind paper Tables 1, 2 and Figure 1 (~20h)",
    "certify_deep":
        "12 extra restarts at 13 levels / toehold 0.15 (~4d): unsampled basin or not",
    "certificate_from_selection_csv":
        "rebuild the certificate from the committed per-restart CSV (~1s)",
    "toehold_preemption_experiment": "toehold sweep: profit pinned, conduct not",
    "preemption_incentive_experiment": "forced-opening profit curves (paper Figure 1)",
    "replot_preemption_incentive_from_csv":
        "redraw paper Figure 1 from the committed curve CSV (~1s)",
    "rounds_compounding_experiment": "rounds sweep (paper Table 2)",
    "equilibrium_selection_experiment": "restart spread + certificate (paper Table 1)",
    "bid_grid_refinement_experiment": "the same price range at three resolutions",
    "certify_grid": "the refinement study at the certify budget (~4d; 13-level trees)",
    "main": "every study at the default 300k/1e-4 budget",
}


def certify_grid():
    """Re-run the bid-grid refinement at probe precision, closing the fine-grid gap.

    At the default budget the refinement study converges 6/6 restarts at 9 levels but only
    3/6 at 7 and 1/6 at 13: a finer grid means a bigger tree and fictitious play converges
    more slowly, so the 13-level cells carry almost no evidence. The committed figure
    therefore shows the deterrence band surviving from 7 to 9 levels and says nothing at 13,
    which is exactly the resolution a referee would press on, since the whole point of the
    study is whether multiplicity is a discretisation artifact.

    This runs all three resolutions at the certify budget so the comparison is like-for-like
    (a band measured at 2e6 iterations against one measured at 3e5 would not be), and
    accepts by achieved NashConv rather than the reached-tolerance flag, for the reason
    given in the ACCEPT_EPS note.

    Read the outcome as follows. If the deterrence band still fails to close at 13 levels,
    the multiplicity is confirmed not to be a discretisation artifact and the paper's
    limitation about the fine grid can be dropped. If the band *does* close, the multiplicity
    is partly a coarse-grid effect at fine resolution and the paper's central claim needs
    restating, not deleting: it would then hold on the grids studied but not in the limit.

    This is the slowest study in the suite (54 solves, the 13-level trees being much the
    largest). Expect well over a day of wall-clock. Run it detached:
        docker run --rm -v "<repo>:/work" imperfect-info:latest \\
            python experiments/sequential_general_sum.py certify_grid
    """
    global MAX_ITERATIONS, TOLERANCE, ACCEPT_EPS
    MAX_ITERATIONS = CERTIFY_ITERATIONS
    TOLERANCE = CERTIFY_TOLERANCE
    ACCEPT_EPS = CERTIFY_ACCEPT_EPS
    print(f"== CERTIFY GRID: {CERTIFY_ITERATIONS} rounds, tolerance {CERTIFY_TOLERANCE:.0e} ==",
          flush=True)
    print(f"  resolutions {REFINE_BIDS} x toeholds {REFINE_TOEHOLDS} "
          f"x {NUM_RESTARTS} restarts, all at the same budget", flush=True)
    print("  under test: does the deterrence band close as the grid refines?", flush=True)
    bid_grid_refinement_experiment()


def main():
    _ensure_results_dir()
    print("== default run: 300k rounds, tolerance 1e-4 ==", flush=True)
    print("  This is NOT the certified configuration. The three artifacts behind paper",
          flush=True)
    print("  Tables 1 and 2 will be written with a '_loose' suffix so the certified",
          flush=True)
    print("  files are left intact. For the paper's numbers run:", flush=True)
    print("      python experiments/sequential_general_sum.py certify", flush=True)
    toehold_preemption_experiment()
    preemption_incentive_experiment()
    rounds_compounding_experiment()
    equilibrium_selection_experiment()
    bid_grid_refinement_experiment()
    print("Done.")


if __name__ == "__main__":
    import sys
    _ensure_results_dir()
    if len(sys.argv) > 1:
        name = sys.argv[1]
        if name not in ENTRY_POINTS:
            width = max(len(k) for k in ENTRY_POINTS)
            print(f"unknown entry point {name!r}. Available:")
            for key, blurb in ENTRY_POINTS.items():
                print(f"  {key:<{width}}  {blurb}")
            raise SystemExit(2)
        globals()[name]()
    else:
        main()
