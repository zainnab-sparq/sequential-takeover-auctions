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
