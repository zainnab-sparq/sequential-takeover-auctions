# Does a Toehold Make a Bidder Bolder?<br>Preemption and Multiplicity in Multi-Round Takeover Auctions

[![arXiv](https://img.shields.io/badge/arXiv-pending-b31b1b.svg)](paper/sequential.pdf) [![paper](https://img.shields.io/badge/paper-PDF-blue)](paper/sequential.pdf) [![Python](https://img.shields.io/badge/python-3.10%2B-blue)](requirements.txt) [![OpenSpiel](https://img.shields.io/badge/built%20on-OpenSpiel-green)](https://github.com/google-deepmind/open_spiel) [![compute](https://img.shields.io/badge/compute-laptop%20CPU-lightgrey)](#reproducing-the-paper) [![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## TL;DR

Code, games, and experiments for the paper *Does a Toehold Make a Bidder Bolder? Preemption and Multiplicity in Multi-Round Takeover Auctions*. A **toehold** is a stake in a target quietly bought before a takeover contest, and theory has long said it should make a bidder bid harder and scare rivals off. Toeholds are nevertheless rare in practice, which is known as the toehold puzzle. We model a takeover fight that actually unfolds over several rounds of bidding, rather than the one or two moves earlier models assume, and find that the predicted scare-off effect does not survive. It is an artifact of stopping the model too early. The compiled paper is in [`paper/sequential.pdf`](paper/sequential.pdf).

> **New to auction theory or takeover contests?** Two buyers want the same company. Nobody knows what it is really worth, so each does private homework and forms its own noisy estimate. They then bid against each other in rounds, each seeing the other's bids but not the other's homework. One of them already owns a slice of the target: its *toehold*. The intuition everyone has is that owning a slice should make that buyer bid more boldly, because it profits either way, and that boldness should frighten the rival into quitting. This repo turns that situation into a game a computer can solve exactly, then checks whether the frightening-the-rival part actually happens.

<details>
<summary><b>Under the hood: the research behind this repo</b></summary>

The paper's central result is **equilibrium multiplicity**. In the multi-round ascending auction the equilibrium *value* to the toehold-holder is pinned down, but its *conduct* is not: independent solves converge to the same own profit to four decimals while deterring the rival with wildly different probability (for example 0.002 versus 0.333 at a toehold of 0.05). Both are certified epsilon-Nash equilibria. Deterrence is therefore not identified by the toehold.

Two consequences follow. First, preemption occurs even at a **zero** toehold, so preemption is not something a toehold buys; the toehold-buys-boldness thesis fails as a causal claim. Second, the clean "bigger toehold, more deterrence" relationship that classical two-move models predict is real at exactly one round, saturates at two, and dissolves by three. It is an artifact of the two-move approximation. That rationalizes the toehold puzzle without appealing to hidden costs: the bidding edge that would justify taking a toehold is not identified in a contest with a genuine third round.

Methodologically, the paper leans on a forced-opening best-response construction that measures what a preemptive jump bid is worth *without* trusting any single solver run, which is what breaks the circularity of reading economics off a solve that is itself in doubt. It also reports a cautionary negative result: a single fictitious-play solve manufactures confident artifacts, so every economic claim here is made across a spread of independent restarts. On the computational side, a generic policy-gradient method (PPG) reaches a near-exact equilibrium on the sequential game while a standard imperfect-information baseline (NFSP) fails to converge on the same budget, and the analysis extends into a multi-round multi-signal regime too large to enumerate, where exact solvers cannot run.

Full tables, figures, and caveats are in the [paper](paper/sequential.pdf).

</details>

## Companion paper

This builds on *How Much Due Diligence Before You Bid? Learning in Intractable Takeover Auctions* ([repo](https://github.com/zainnab-sparq/imperfect-information-deal-games)), which studied **sealed** takeover auctions, where each bidder submits one bid and cannot react to the other. That paper asserted, but could not show, that the toehold's aggressiveness channel lives in sequential contests. This paper makes the sequential contest the object of study, and turns the assertion into a result, though not the one we expected. The two repos share the underlying `DealGame` library.

## What's in here

| Path | Contents |
|------|----------|
| [`src/dealgame/`](src/dealgame) | The reusable `DealGame` abstraction, the sequential auction game, the solver wrappers, and the extensive-form own-profit equilibrium solver. |
| [`experiments/`](experiments) | The studies that produce every figure and table in the paper. |
| [`results/`](results) | Generated CSVs and figures. Sequential-paper artifacts are prefixed `seq_`. |
| [`tests/`](tests) | Game-validity and solver-convergence tests. |
| [`paper/`](paper) | LaTeX source and the compiled PDF. |

## Quick start

OpenSpiel has no Windows wheels, so everything runs in a pinned Docker image. Build it once:

```bash
docker build -t imperfect-info:latest .
```

Run the tests to confirm the games and solvers behave:

```bash
docker run --rm -v "$(pwd):/work" -w /work imperfect-info:latest python -m pytest tests/ -q
```

> **On Windows in Git Bash:** prefix the `docker run` commands with `MSYS_NO_PATHCONV=1`, and if `$(pwd)` does not expand cleanly, substitute the absolute repository path for the volume mount.

## Reproducing the paper

Three scripts run the full suites and write their CSVs and figures into `results/`:

```bash
# the economics: multiplicity, preemption, the rounds sweep
docker run --rm -v "$(pwd):/work" -w /work imperfect-info:latest python experiments/sequential_general_sum.py

# solver benchmark and scaling in the number of rounds
docker run --rm -v "$(pwd):/work" -w /work imperfect-info:latest python experiments/sequential_benchmark.py

# the intractable regime and the estimator calibration
docker run --rm -v "$(pwd):/work" -w /work imperfect-info:latest python experiments/sequential_intractable.py
```

Each script also accepts a single study name, so you can regenerate one figure without rerunning everything:

```bash
docker run --rm -v "$(pwd):/work" -w /work imperfect-info:latest \
  python experiments/sequential_general_sum.py equilibrium_selection_experiment
```

### Reproducing the epsilon-equilibrium certificate

The paper certifies its equilibria to epsilon on the order of 1e-6. A default run stops at its tolerance of 1e-4, which understates how tight the equilibria actually are, so the tight numbers get their own entry point:

```bash
docker run --rm -v "$(pwd):/work" -w /work imperfect-info:latest \
  python experiments/sequential_general_sum.py certify
```

This re-solves the paper's two headline tables with the tolerance set far below anything the run can reach and a budget large enough to drive NashConv to roughly 1e-6, so the compute budget rather than the stopping rule ends each solve. **Expect this to take many hours** (it was about 20 on a commodity many-core CPU). Solves are cached under `results/.solve_cache/` keyed by parameters, restart, and iteration budget, so an interrupted run resumes rather than restarting.

You do not need those hours to check the certificate. The certificate is a pure selection over the committed per-restart data, so it rebuilds in about a second, byte-identically:

```bash
docker run --rm -v "$(pwd):/work" -w /work imperfect-info:latest \
  python experiments/sequential_general_sum.py certificate_from_selection_csv
```

**Do not run the script with no arguments if you want to keep the certified numbers.** A bare `python experiments/sequential_general_sum.py` runs the full default suite at the loose budget and overwrites the certificate with rows at the 1e-4 stopping tolerance. See [`REPRODUCING.md`](REPRODUCING.md) for the full map of which command regenerates which artifact, and at what precision.

### Figure and table map

| Paper artifact | Script | Single-study name |
|----------------|--------|-------------------|
| Equilibrium multiplicity (Table 1) | `sequential_general_sum.py` | `equilibrium_selection_experiment` |
| Two-move artifact, deterrence by rounds (Table 2) | `sequential_general_sum.py` | `rounds_compounding_experiment` |
| Toehold and preemption | `sequential_general_sum.py` | `toehold_preemption_experiment` |
| Forced-opening mechanism panel | `sequential_general_sum.py` | `preemption_incentive_experiment` |
| Bid-grid refinement robustness | `sequential_general_sum.py` | `bid_grid_refinement_experiment` |
| Both certified tables at probe precision | `sequential_general_sum.py` | `certify` |
| Deep self-play benchmark (headline) | `sequential_benchmark.py` | `seq_headline_experiment` |
| Scaling in the number of rounds | `sequential_benchmark.py` | `seq_scaling_experiment` |
| Intractable regime | `sequential_intractable.py` | `seq_intractable_experiment` |
| Estimator calibration | `sequential_intractable.py` | `seq_calibration_study` |

## Code layout

| File | Role |
|------|------|
| [`src/dealgame/sequential_takeover.py`](src/dealgame/sequential_takeover.py) | The multi-round ascending takeover auction with a toehold, as an OpenSpiel game. Bid history is public, diligence signals are private. |
| [`src/dealgame/sequential_general_sum.py`](src/dealgame/sequential_general_sum.py) | Extensive-form own-profit fictitious play, plus the forced-opening best-response construction that reads the value of a jump bid without trusting a solve. |
| [`src/dealgame/base.py`](src/dealgame/base.py) | Reusable deal-game primitives: information-set construction and the payoff contracts. |
| [`src/dealgame/takeover.py`](src/dealgame/takeover.py) | The sealed common-value auction from the companion paper, kept as the like-for-like comparison. |
| [`src/dealgame/solving.py`](src/dealgame/solving.py) | Solver wrappers (CFR, MMD, PSRO) and the from-scratch tabular policy gradient. |
| [`src/dealgame/deep_solving.py`](src/dealgame/deep_solving.py) | The deep solvers (PPO, PPG, generic deep policy gradient, Deep CFR, NFSP). |
| [`src/dealgame/ppo_solving.py`](src/dealgame/ppo_solving.py) | The PPO/PPG self-play implementation, multi-decision aware for sequential games. |
| [`src/dealgame/intractable.py`](src/dealgame/intractable.py) | The multi-signal regime and the learned-best-response exploitability estimator. |

## A note on what the code will and will not tell you

The headline finding is a *negative identification* result, and the repo is built to keep it honest. Every economic claim is made across `NUM_RESTARTS` independent solves and reported as a spread, not from the one equilibrium a uniform start happens to find. Solves that fail to converge are excluded from equilibrium claims and the dropped count is printed. If you run a single solve and read economics off it, you will get a confident, plausible, and wrong answer; that failure mode is itself one of the paper's results.

## Citation

```bibtex
@misc{naboulsi2026toehold,
  title  = {Does a Toehold Make a Bidder Bolder? Preemption and Multiplicity in Multi-Round Takeover Auctions},
  author = {Naboulsi, Zain},
  year   = {2026},
  note   = {arXiv preprint; identifier to be added upon posting}
}
```
