"""Concurrency properties of the solve cache and the checkpoint writer.

Both failures below were observed on the certification runs, not imagined. Two study
containers share one cache directory on purpose, because the studies overlap and a solve
is hours of CPU, so the cache is the thing that stops the same equilibrium being computed
twice. Sharing it means two processes can be working the same cell at the same moment,
and every write into that directory has to be safe under that.
"""

from __future__ import annotations

import os
import pickle
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

import pytest

import pyspiel

import dealgame  # noqa: F401  (registers the game)
from dealgame import sequential_general_sum as solver
from experiments import sequential_general_sum as study


def _small_tree():
    return solver.SequentialAuctionTree(pyspiel.load_game(
        "dealgame_sequential_takeover",
        {"num_values": 2, "num_bids": 3, "num_rounds": 2, "toehold": 0.0}))


def test_two_processes_checkpointing_one_cell_do_not_consume_each_others_temp(tmp_path,
                                                                             monkeypatch):
    """A checkpoint's temp file must belong to the writer that made it.

    Observed 2026-07-29: the certify container died 23.6 hours in with
    ``FileNotFoundError: ...ckpt.tmp -> ...ckpt``. Two containers were solving the same
    cell, both wrote the one shared temp name, and the first rename carried away the file
    the second was about to rename. Losing the checkpoint would be survivable; the raise
    propagated out of the worker and killed the whole study.
    """
    path = str(tmp_path / "cell.ckpt")
    renamed = []
    real_replace = os.replace
    monkeypatch.setattr(solver.os, "replace",
                        lambda src, dst: (renamed.append(src), real_replace(src, dst))[1])

    monkeypatch.setattr(solver.os, "getpid", lambda: 4001)
    solver._write_checkpoint(path, {"who": "first"})
    monkeypatch.setattr(solver.os, "getpid", lambda: 4002)
    solver._write_checkpoint(path, {"who": "second"})

    assert len(set(renamed)) == 2, f"both writers used the same temp file: {renamed}"


def test_a_checkpoint_write_survives_a_sibling_publishing_the_same_cell_mid_write(
        tmp_path, monkeypatch):
    """The crash, staged deterministically: the sibling renames while we are still writing.

    This is the interleaving that killed the run. The second writer's rename carries away
    the temp file the first writer still holds open, so the first writer's own rename
    finds nothing to move. Reproduced by letting the sibling run at the exact moment the
    first writer has finished dumping and has not yet renamed.
    """
    path = str(tmp_path / "cell.ckpt")
    real_dump = solver.pickle.dump

    def dump_then_let_the_sibling_finish(state, handle, **kwargs):
        real_dump(state, handle, **kwargs)
        if state.get("who") == "first":
            solver._write_checkpoint(path, {"who": "second"})

    monkeypatch.setattr(solver.pickle, "dump", dump_then_let_the_sibling_finish)

    solver._write_checkpoint(path, {"who": "first"})

    with open(path, "rb") as handle:
        assert pickle.load(handle) == {"who": "first"}


class _GatedPool(ThreadPoolExecutor):
    """Stands in for the process pool so a test can pin the completion order."""

    def __init__(self, max_workers=None):
        super().__init__(max_workers=2)


_GATE = threading.Event()


def _gated_solve(job: tuple) -> dict:
    """The expensive job blocks until the cheap one has finished, deterministically."""
    if job[0]["num_bids"] == 13:
        assert _GATE.wait(30), "the cheap job never released the gate"
        return {"tag": "expensive", "nashconv": 1.0}
    _GATE.set()
    return {"tag": "cheap", "nashconv": 1.0}


def test_each_solve_is_cached_when_it_finishes_not_when_the_slowest_one_does(tmp_path,
                                                                            monkeypatch):
    """A finished solve belongs on disk immediately, whatever else is still running.

    The jobs are dispatched longest-first, so the 13-level cell is submitted before the
    7-level one and finishes long after it. Consuming results in submission order holds
    every finished solve in memory until the slowest returns: on the refinement grid that
    is 18 completed cells kept unwritten for days, all of them lost if the container dies,
    and none of them visible to the progress reader or to the sibling study that could
    have reused them.
    """
    _GATE.clear()
    monkeypatch.setattr(study, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(study, "WORKERS", 2)
    monkeypatch.setattr(study, "_solve_job", _gated_solve)
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _GatedPool)

    written = []
    real_cache_result = study._cache_result
    monkeypatch.setattr(study, "_cache_result",
                        lambda path, stats: (written.append(stats["tag"]),
                                             real_cache_result(path, stats))[1])

    cheap = ({"num_bids": 7, "num_rounds": 3}, 0, 10, 1e-8)
    expensive = ({"num_bids": 13, "num_rounds": 3}, 0, 10, 1e-8)
    results = study._solve_many([cheap, expensive])

    assert written == ["cheap", "expensive"], (
        "results were cached in submission order, so the finished cheap solve sat "
        f"unwritten until the slow one returned: {written}")
    assert [r["tag"] for r in results] == ["cheap", "expensive"], (
        "_solve_many must return results in the caller's job order regardless of "
        "the order they completed in")


def test_a_cached_result_is_published_atomically(tmp_path, monkeypatch):
    """The progress reader scans this directory live, so no reader may see a torn file."""
    monkeypatch.setattr(study, "CACHE_DIR", str(tmp_path))
    path = os.path.join(str(tmp_path), "cell.pkl")
    seen = []
    real_replace = os.replace
    monkeypatch.setattr(study.os, "replace",
                        lambda src, dst: (seen.append((src, dst)), real_replace(src, dst))[1])

    study._cache_result(path, {"tag": "first"})
    study._cache_result(path, {"tag": "second"})

    assert seen and all(dst == path for _, dst in seen), (
        "the result was not published by a rename onto the cache path")
    assert len({src for src, _ in seen}) == 2, (
        f"two writers of one cell shared a temp file: {seen}")
    with open(path, "rb") as handle:
        assert pickle.load(handle) == {"tag": "second"}


def test_a_failed_checkpoint_write_does_not_kill_the_solve(tmp_path, monkeypatch):
    """A checkpoint is an optimisation for resume, so losing one must never be fatal.

    It was. The rename propagated out of the worker, through the process pool, and killed
    the whole study twice: FileNotFoundError on 2026-07-29 and PermissionError on
    2026-08-02, the second one after four days of a 54-cell run. Losing a checkpoint costs
    at most ``checkpoint_every`` iterations of resume granularity. Losing the study costs
    days, and on the second occasion it destroyed eighteen finished solves outright. The
    repository also lives inside a syncing folder, so a rename can fail for reasons no
    amount of care inside this process can prevent.
    """
    tree = _small_tree()
    path = str(tmp_path / "cell.ckpt")

    def refuse(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(solver.os, "replace", refuse)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = solver.own_profit_fictitious_play(
            tree, max_iterations=40, tolerance=0.0,
            checkpoint_path=path, checkpoint_every=10)

    assert result["nashconv"] >= 0.0, "the solve did not finish"
    assert not os.path.exists(path), "no checkpoint should have been published"
    assert any("checkpoint" in str(w.message).lower() for w in caught), (
        "the failure was swallowed silently; a run that cannot checkpoint must say so")


def test_a_solve_that_cannot_checkpoint_still_returns_the_same_answer(tmp_path,
                                                                     monkeypatch):
    """Degrading to no-checkpointing must not change the mathematics."""
    tree = _small_tree()
    expected = solver.own_profit_fictitious_play(tree, max_iterations=40, tolerance=0.0)

    monkeypatch.setattr(solver.os, "replace",
                        lambda *a, **k: (_ for _ in ()).throw(OSError(28, "No space")))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        degraded = solver.own_profit_fictitious_play(
            tree, max_iterations=40, tolerance=0.0,
            checkpoint_path=str(tmp_path / "cell.ckpt"), checkpoint_every=10)

    assert degraded["best_nashconv"] == pytest.approx(expected["best_nashconv"])
    assert degraded["best_iteration"] == expected["best_iteration"]
