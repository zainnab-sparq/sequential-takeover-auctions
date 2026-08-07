"""The multiplicity certificate's selection rule (paper Table 1).

These are pure-data tests over ``experiments/sequential_general_sum._build_certificate``:
no game, no solver, no pyspiel. They exist because the rule was wrong in a way nothing
caught. Picking the two representative profiles by argmax/argmin deterrence alone let
numerical noise choose them: at toehold 0.15 five restarts sat within 2e-5 of each other
in deterrence, the argmax landed on the one whose NashConv was 3.2e-5 rather than the one
at 5.3e-6, and the certificate came out with a five-fold looser eps and a 2.2e-4 gap in
value0 -- which made the paper's "same value to four decimals" claim false for that row.
Every test below fails against the old rule.
"""

from __future__ import annotations

import csv
import os

import pytest

from experiments.sequential_general_sum import (DETERRENCE_TIE, _build_certificate,
                                                _representative_rivals,
                                                certificate_from_selection_csv)

ACCEPT = 1e-4
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def row(restart, deterrence, value0, nashconv):
    return {"restart": restart, "opening_bid": 0.0, "p_rival_deterred": deterrence,
            "value0": value0, "value1": 0.0, "nashconv": nashconv,
            "reached_tolerance": 0}


def test_ties_on_conduct_are_broken_toward_the_tightest_certificate():
    """Two restarts on the same conduct: the better-certified one must represent it."""
    rows = [
        row(0, 0.333331, 0.455435, 6.02e-06),
        row(1, 0.333329, 0.455422, 5.25e-06),
        # Wins a bare argmax by 1.3e-05 in deterrence -- noise -- at 6x the NashConv.
        row(4, 0.333344, 0.455646, 3.21e-05),
        row(3, 0.220494, 0.455430, 6.39e-06),
    ]
    cert = _build_certificate(rows, 0.15, ACCEPT, 1e-8, 2_000_000)

    assert cert["restart_deterring"] == 1, "argmax deterrence must not beat a tighter eps"
    assert cert["eps"] == pytest.approx(6.39e-06)
    assert cert["value0_gap"] < 1e-4, "the pair must agree to four decimals"


def test_eps_is_the_worse_of_the_pair_and_both_are_reported():
    rows = [row(0, 0.333, 0.2715, 4.30e-06), row(1, 0.002, 0.2716, 7.56e-05)]
    cert = _build_certificate(rows, 0.05, ACCEPT, 1e-8, 2_000_000)

    assert cert["eps_deterring"] == pytest.approx(4.30e-06)
    assert cert["eps_passive"] == pytest.approx(7.56e-05)
    # The headline eps is an upper bound on both, never the flattering one.
    assert cert["eps"] == pytest.approx(7.56e-05)


def test_restarts_above_the_acceptance_bound_are_excluded():
    rows = [row(0, 0.333, 0.455, 6.0e-06),
            row(1, 0.220, 0.455, 6.4e-06),
            row(2, 0.900, 0.594, 2.23e-04)]   # a different value, not an equilibrium
    cert = _build_certificate(rows, 0.15, ACCEPT, 1e-8, 2_000_000)

    assert cert["n_accepted"] == 2
    assert cert["deterrence_deterring"] == pytest.approx(0.333)


def test_no_certificate_when_every_admitted_restart_is_the_same_conduct():
    """A spread of 1e-5 in deterrence is not multiplicity; it must not be certified."""
    rows = [row(0, 0.333331, 0.6564, 6e-06), row(1, 0.333330, 0.6577, 7e-06)]

    assert _build_certificate(rows, 0.20, ACCEPT, 1e-8, 2_000_000) is None


def test_no_certificate_from_fewer_than_two_admitted_restarts():
    rows = [row(0, 0.333, 0.455, 6e-06), row(1, 0.220, 0.455, 5e-03)]

    assert _build_certificate(rows, 0.15, ACCEPT, 1e-8, 2_000_000) is None


def test_tie_band_is_wide_enough_to_cover_solver_noise_and_narrow_enough_to_keep_0_15():
    # The real toehold-0.15 pair is 0.333 vs 0.220: a genuine conduct difference that the
    # band must never swallow, while still absorbing the 1e-5 wobble between restarts.
    assert 2e-5 < DETERRENCE_TIE < 0.112


def test_committed_certificate_matches_the_committed_per_restart_rows(tmp_path):
    """The released certificate must be exactly what its own selection rule produces.

    This is the check that would have caught the bad rule in the released artifact rather
    than only in the code: it re-derives the certificate from the committed per-restart
    CSV and compares against the committed certificate cell by cell.
    """
    cert_path = os.path.join(RESULTS, "seq_multiplicity_certificate.csv")
    with open(cert_path, newline="") as fh:
        committed = list(csv.DictReader(fh))

    # Write to a scratch path: a test must not mutate a committed artifact.
    scratch = str(tmp_path / "rebuilt.csv")
    certificate_from_selection_csv(scratch)
    with open(scratch, newline="") as fh:
        rebuilt = list(csv.DictReader(fh))

    assert rebuilt == committed
    assert b"\r\n" not in open(scratch, "rb").read(), "artifacts are eol=lf per .gitattributes"


def test_every_advertised_entry_point_actually_resolves():
    """The CLI's own help must not list a command that fails.

    ``certify_grid`` was defined but dropped from ``ENTRY_POINTS`` by a merge, so the
    documented command exited 2 with "unknown entry point" while the function sat right
    below the registry. Nothing caught it because the failure only shows up when someone
    runs a multi-day study. Both directions are checked: every advertised name resolves
    to something callable, and every study function is advertised.
    """
    from experiments import sequential_general_sum as module

    for name in module.ENTRY_POINTS:
        target = getattr(module, name, None)
        assert callable(target), f"{name} is advertised but not callable"

    studies = [name for name, value in vars(module).items()
               if callable(value) and not name.startswith("_")
               and getattr(value, "__module__", None) == module.__name__
               and (name.endswith("_experiment") or name.startswith("certif")
                    or name == "main")]
    unregistered = [name for name in studies if name not in module.ENTRY_POINTS]
    assert not unregistered, f"study functions missing from ENTRY_POINTS: {unregistered}"


def test_representative_rivals_prefer_the_tightest_solve_among_deterrence_ties():
    """The two representative profiles must be picked the same way everywhere.

    Numerical noise puts several restarts within ~1e-5 of each other in deterrence, so a
    plain argmax lets noise choose, and it reliably chooses badly: the restart that
    overshoots the true value by noise is the one that has not settled, so it also carries
    the loosest NashConv. Observed at toehold 0.15, where argmax preferred a profile at
    deterrence 0.333344 with eps 3.0e-05 over three profiles at 0.33333 with eps ~3.3e-06.
    """
    good = [
        {"restart": 0, "p_rival_deterred": 0.333331, "nashconv": 3.32e-06, "value0": 0.4554},
        {"restart": 1, "p_rival_deterred": 0.333329, "nashconv": 3.28e-06, "value0": 0.4554},
        {"restart": 3, "p_rival_deterred": 0.220576, "nashconv": 3.51e-06, "value0": 0.4554},
        {"restart": 4, "p_rival_deterred": 0.333344, "nashconv": 2.96e-05, "value0": 0.4557},
    ]

    deterring, passive = _representative_rivals(good)

    assert deterring["restart"] == 1, (
        "argmax deterrence picked the noisiest profile instead of the tightest one "
        "within the tie band")
    assert passive["restart"] == 3


def _incentive_rows():
    incentive = os.path.join(RESULTS, "seq_preemption_incentive.csv")
    if not os.path.exists(incentive):
        pytest.skip("released CSVs not present")
    with open(incentive) as fh:
        return list(csv.DictReader(fh))


def test_the_figure_measures_every_certified_rival_not_a_chosen_pair():
    """Figure 1 must report all accepted restarts, because the premium is not identified.

    It used to select one most-deterring and one least-deterring profile per toehold. That
    is a statement about the selection rule rather than about the game: profiles certified
    at the same eps, deterring with the same probability to five decimals, give premia that
    differ by up to eighteen-fold (2.7% and 49.0% at toehold 0.10). Deterrence is one
    scalar summary of a strategy defined over many information sets, and the forced-opening
    curve probes exactly the off-path behaviour that scalar discards. So every certified
    rival goes in the released data and the paper quotes the interval.
    """
    selection = os.path.join(RESULTS, "seq_equilibrium_selection.csv")
    if not os.path.exists(selection):
        pytest.skip("released CSVs not present")

    accepted = {}
    with open(selection) as fh:
        for r in csv.DictReader(fh):
            if float(r["nashconv"]) <= ACCEPT:
                accepted.setdefault(float(r["toehold"]), set()).add(int(r["restart"]))

    measured = {}
    for row in _incentive_rows():
        if row["rival"] == "uniform-rival":
            continue
        measured.setdefault(float(row["toehold"]), set()).add(int(row["restart"]))

    checked = 0
    for theta, restarts in accepted.items():
        if theta not in measured:
            continue
        assert measured[theta] == restarts, (
            f"toehold {theta}: the figure measured restarts {sorted(measured[theta])} "
            f"but {sorted(restarts)} were certified")
        checked += 1
    assert checked, "no overlapping toeholds were actually compared"


def test_the_unbluffable_control_earns_no_premium_at_any_toehold():
    """The control is the claim's own falsifier, so it must hold everywhere.

    A jump bid buys deterrence. Against a rival bidding at random, which cannot be
    deterred, it must buy nothing: the curve slopes down and the best opening is the
    cheapest one. If this ever showed a premium, the premium would not be deterrence.
    """
    control = [r for r in _incentive_rows() if r["rival"] == "uniform-rival"]
    assert control, "the control rival is missing from the released data"
    for row in control:
        assert float(row["peak_bid"]) == 0.0, (
            f"toehold {row['toehold']}: the control peaked at a jump of "
            f"{row['peak_bid']}, so the premium is not deterrence")
        assert float(row["preemption_premium"]) == pytest.approx(0.0, abs=1e-12)


def test_extra_restarts_really_are_new_draws():
    """Deepening a cell only tests anything if the added starts differ from the first six.

    ``_restart_policies`` keys its generator on ``RESTART_SEED + restart``, so this holds
    by construction; it is pinned because the failure is silent. Reusing the seeds would
    recompute the same six equilibria under twelve different cache keys, burn four days,
    and return "no new basin found", which reads exactly like a real negative result.
    """
    import numpy as np
    import pyspiel

    import dealgame  # noqa: F401  (registers the game)
    from dealgame.sequential_general_sum import SequentialAuctionTree
    from experiments.sequential_general_sum import (BASE, DEEP_EXTRA_RESTARTS,
                                                    NUM_RESTARTS, _restart_policies)

    tree = SequentialAuctionTree(pyspiel.load_game(
        "dealgame_sequential_takeover", dict(BASE, num_values=2, num_bids=3,
                                             num_rounds=2)))
    draws = {}
    for restart in range(NUM_RESTARTS + DEEP_EXTRA_RESTARTS):
        policies = _restart_policies(tree, restart)
        if policies is None:            # restart 0 is the uniform start
            continue
        key = tuple(np.round(np.concatenate([p.ravel() for p in policies[0]]), 12))
        assert key not in draws, (
            f"restart {restart} repeats restart {draws[key]}; the added starts are not "
            "new draws and the deepened cell would re-solve the same basins")
        draws[key] = restart
    assert len(draws) == NUM_RESTARTS + DEEP_EXTRA_RESTARTS - 1
