"""Report the state of a long detached experiment container.

The long certification runs (``certify``, ``certify_grid``) are launched with
``docker run -d`` so they survive the shell, the terminal, and any assistant session
that started them. That decoupling is deliberate: nothing about the run should depend
on a window staying open. The cost is that nothing announces when the run ends, so
this script is the thing you poll instead.

It exists mainly to answer one question correctly: *is silence a hang?* For these runs
the answer is almost always no. A study collects every solve before printing any cell,
so the container emits nothing at all between its header and its final results, and it
stops writing to the solve cache once the last batch is dispatched. A run can sit
completely silent for many hours while working perfectly. Judging it by log output
alone leads to killing a healthy multi-day job.

So the verdict below is driven by CPU, not by logs or cache mtimes, which are reported
only as supporting detail.

    python scripts/check_long_run.py [container_name]
"""

import json
import os
import subprocess
import sys
import time

DEFAULT_CONTAINER = "seq_certify_grid"
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "results", ".solve_cache")
# Below this, the container is not doing arithmetic and something is genuinely wrong.
# One busy worker is ~100%, and these runs fan out across the pool, so a healthy run
# sits in the hundreds or thousands. Anything under a single core is the alarm.
_ALIVE_CPU_PERCENT = 50.0
# Cell lines and the writer's confirmation; the noisy per-solve warnings are dropped.
_RESULT_MARKERS = ("levels=", "no restart converged", "wrote results", "toehold=")


def _docker(*args: str) -> tuple[int, str]:
    try:
        p = subprocess.run(("docker",) + args, capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        return 127, "docker not on PATH"
    except subprocess.TimeoutExpired:
        return 124, "docker timed out"
    return p.returncode, (p.stdout or p.returncode and p.stderr or "").strip()


def _inspect(name: str) -> dict | None:
    code, out = _docker("inspect", name)
    if code != 0 or not out:
        return None
    try:
        return json.loads(out)[0]
    except (ValueError, IndexError, KeyError):
        return None


def _cpu_percent(name: str) -> float | None:
    code, out = _docker("stats", "--no-stream", "--format", "{{.CPUPerc}}", name)
    if code != 0 or not out:
        return None
    try:
        return float(out.rstrip("%"))
    except ValueError:
        return None


def _cache_progress() -> tuple[int, float | None]:
    if not os.path.isdir(CACHE_DIR):
        return 0, None
    newest = None
    count = 0
    for entry in os.scandir(CACHE_DIR):
        if not entry.is_file():
            continue
        count += 1
        mtime = entry.stat().st_mtime
        if newest is None or mtime > newest:
            newest = mtime
    return count, newest


def _hours_since(ts: float | None) -> str:
    if ts is None:
        return "never"
    return f"{(time.time() - ts) / 3600:.1f}h ago"


def _results(name: str) -> list[str]:
    code, out = _docker("logs", "--tail", "200", name)
    if code != 0:
        return []
    return [ln for ln in out.splitlines() if any(m in ln for m in _RESULT_MARKERS)]


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONTAINER
    info = _inspect(name)
    if info is None:
        print(f"{name}: NOT FOUND (never started, or removed after finishing)")
        print("  If it finished and was removed, its output is gone; check results/ mtimes.")
        return 2

    state = info.get("State", {})
    running = bool(state.get("Running"))
    started = state.get("StartedAt", "?")
    cached, newest = _cache_progress()

    if not running:
        exit_code = state.get("ExitCode")
        print(f"{name}: FINISHED (exit {exit_code}) at {state.get('FinishedAt', '?')}")
        lines = _results(name)
        if lines:
            print("\n".join(f"  {ln}" for ln in lines))
        else:
            print("  No result lines in the last 200 log lines; run: "
                  f"docker logs {name}")
        return 0 if exit_code == 0 else 1

    cpu = _cpu_percent(name)
    print(f"{name}: RUNNING since {started}")
    print(f"  cpu           {cpu if cpu is None else f'{cpu:.0f}%'}"
          f"   (healthy is hundreds or thousands; the pool fans out)")
    print(f"  solve cache   {cached} files, last write {_hours_since(newest)}")

    if cpu is None:
        print("  VERDICT       cannot read CPU; check `docker stats` by hand")
        return 3
    if cpu < _ALIVE_CPU_PERCENT:
        print("  VERDICT       STUCK. CPU is near zero, so it is not solving.")
        return 3
    print("  VERDICT       ALIVE and working. Silence is expected: the study prints "
          "nothing")
    print("                until every solve returns, and stops writing cache once the")
    print("                last batch is dispatched. Do NOT kill it on a quiet log.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
