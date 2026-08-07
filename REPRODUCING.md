# Reproducing the results

Two papers share this repository.

- **Paper 1** (sealed takeover auctions): `experiments/angle_a_benchmark.py`,
  `experiments/general_sum_equilibrium.py`, `experiments/angle_c_toy.py`.
  Artifacts are the `results/*.csv` files without a `seq_` prefix.
- **Paper 2** (`paper/sequential.tex`, multi-round takeover auctions with a toehold):
  `experiments/sequential_*.py`. Artifacts are `results/seq_*.{csv,png}`.

This document covers paper 2.

## Environment

```bash
docker build -t imperfect-info:latest .
docker run --rm -v "$PWD:/work" imperfect-info:latest pytest -q
```

Outside Docker: Python 3.12, `pip install -r requirements.txt`, and
`export PYTHONPATH=$PWD/src`. `torch` is in `requirements.txt` but resolves to the
default index; for the CPU-only wheel use
`pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu`.

Thread counts are pinned in the image (`OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`).
Solves are deterministic: restart 0 is the uniform policy and restarts 1–5 come from
`numpy.random.default_rng(RESTART_SEED + restart)` with `RESTART_SEED = 20260713`.

## Certified vs default runs

The economics results are **certified** artifacts. Fictitious play exits the moment
NashConv crosses `TOLERANCE`, so a run that stops *at* its tolerance has not converged, it has been let go, and in a flat game a 1e-4 equilibrium can sit far from any
equilibrium. Everything the paper's Tables 1 and 2 and Figure 1 rest on is therefore
re-run with the tolerance set to 1e-8, far below what the budget can reach, so that the
**budget** ends each solve. `certify` is that configuration.

A default run (no argument) uses 300k iterations at tolerance 1e-4. Its numbers are not
the paper's: eps lands at ~1e-4 in every row, and the certificate loses the toehold 0.05
row because 0.05 is not in the default sweep. So a default run writes the three certified
artifacts to `*_loose.csv` siblings and leaves the certified files untouched.

Acceptance is by NashConv magnitude (`nashconv <= 1e-4`), never by the
`reached_tolerance` flag, which is `0` in every row of a certify run **by construction**.
Each row carries its own `accepted`, `tolerance` and `iterations` so it is self-describing.

## Entry-point → artifact map

`python experiments/sequential_general_sum.py <entry-point>`

| Entry point | Writes | Backs | Cost |
|---|---|---|---|
| `certify` | `seq_multiplicity_certificate.csv`, `seq_equilibrium_selection.{csv,png}`, `seq_rounds_compounding.{csv,png}`, `seq_preemption_incentive.{csv,png}`, `seq_preemption_curves.csv` | **Tables 1, 2 and Figure 1** | ~20 h |
| `certificate_from_selection_csv` | `seq_multiplicity_certificate.csv` | Table 1 | ~1 s |
| `replot_preemption_incentive_from_csv` | `seq_preemption_incentive.png` | Figure 1 | ~1 s |
| `toehold_preemption_experiment` | `seq_toehold_preemption.csv`, `seq_toehold_summary.csv`, `seq_toehold_preemption.png` | §5 toehold sweep | ~8 h |
| `preemption_incentive_experiment` | `seq_preemption_incentive.{csv,png}`, `seq_preemption_curves.csv` | Figure 1 | ~8 h |
| `rounds_compounding_experiment` | `seq_rounds_compounding.{csv,png}` | Table 2 | ~10 h |
| `equilibrium_selection_experiment` | `seq_equilibrium_selection.{csv,png}`, `seq_multiplicity_certificate.csv` | Table 1 | ~6 h |
| `bid_grid_refinement_experiment` | `seq_grid_refinement.{csv,png}` | §9 limitations | ~40 h |
| `certify_grid` | `seq_grid_refinement.{csv,png}` at probe precision | §6 grid refinement | ~7 days |
| `certify_deep` | `seq_grid_deep_restarts.csv` | §9 limitations | ~4 days |
| `main` | everything above at the default budget | nothing in the paper | ~30 h |

`certify_grid` and `certify_deep` are the two that will not finish overnight. Both solve
13-level trees, which cost ~100 h per solve on one core, and both run their cells
concurrently across a process pool, so wall-clock is roughly one solve rather than the
sum. Both checkpoint into `results/.solve_cache/` and resume where they stopped, so an
interrupted run is not a lost one. `certify_deep` runs eighteen restarts of the single
cell at 13 levels and toehold 0.15, which `certify_grid` leaves uncertified at six.

`python experiments/sequential_benchmark.py <entry-point>`

| Entry point | Writes | Backs |
|---|---|---|
| `seq_headline_experiment` | `seq_deep_convergence.{csv,png}` | §7 benchmark, Figure 2 |
| `seq_scaling_experiment` | `seq_scaling.{csv,png}` | §7 scaling, Figure 3 |

`python experiments/sequential_intractable.py <entry-point>`

| Entry point | Writes | Backs |
|---|---|---|
| `seq_intractable_experiment` | `seq_intractable.{csv,png}` | §8, Figure 4 |
| `seq_calibration_study` | `seq_calibration.{csv,png}` | §8, Figure 5 |

Running any of these scripts with an unrecognised argument prints the available entry
points.

## The two one-second checks

Neither needs a solver, because both only re-read committed data. Use them to check the
released artifacts against each other without paying for the solves:

```bash
python experiments/sequential_general_sum.py certificate_from_selection_csv
python experiments/sequential_general_sum.py replot_preemption_incentive_from_csv
```

The certificate is a pure selection over `seq_equilibrium_selection.csv`, it picks two
of the restarts already in that file and copies their numbers, so rebuilding it from the
committed per-restart rows is exact, and produces byte-identical output to `certify`.
Figure 1 is a plot of `seq_preemption_curves.csv`, likewise.

## Checking a certificate instead of trusting it

Every ε in this repo is the solver grading its own work. To re-derive one from the game
tree instead:

```bash
python scripts/verify_certified_profile.py
```

It takes the one certified profile of `seq_grid_deep_restarts.csv`, rebuilds bidder 0's
exact best response against the stored rival, and reports both the exploitability it
measures and whether the published deterrence and opening bid reproduce. It needs the
solve cache, so run it after `certify_deep` rather than on a fresh clone. One limit it
states rather than hides: the cache stores the rival's policy but not bidder 0's, so only
bidder 0's half of NashConv is independently checkable this way.

## Instances

The economics game (Tables 1 and 2, Figure 1) is **not** the benchmark game. Parameters:

| | economics | benchmark | scaling | intractable | calibration |
|---|---|---|---|---|---|
| `num_values` | 3 | 3 | 3 | 3 | 3 |
| `num_bids` | 9 | 6 | 8 | 6 | 4 |
| `bid_step` | 0.375 | default | default | default | default |
| `num_signals` | 1 | 1 | 1 | 6 | 1, 2 |
| `signal_noise` | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| `num_rounds` | 1–3 | 3 | 1–4 | 3 | 2 |
| restarts / seeds | 6 restarts | 5 seeds | 3 seeds | 3 seeds | 3 seeds |

## Known gaps

- **Grid refinement is underpowered.** Acceptance in `seq_grid_refinement.csv` is 6/18
  solves at 7 levels, 11/18 at 9, and 2/18 at 13, so it is not monotone in resolution and
  the coarsest grid is not the easiest to solve. The multiplicity is exhibited at the base
  grid and at 13 levels only at a zero toehold. The headline does not rest on this study.
- **The fine grid at a positive toehold certifies one conduct, not two.**
  `seq_grid_deep_restarts.csv` re-runs the 13-level, toehold-0.15 cell with eighteen
  restarts instead of six. One clears the bound, at ε 3.78e-05 and deterrence 0.333. Four
  of the twelve added restarts do settle on low-deterrence conduct, but the tightest of
  those is 27× from the bound and three are still improving when the budget ends, so the
  low branch is reached at this resolution and not certified. Read that as "not resolved
  at this budget", not as "absent".
- **Figure 1's rival policies are default-budget solves in the committed data.** Every
  row of the committed `seq_preemption_incentive.csv` has `nashconv` ≈ 1e-4, i.e. those
  rivals stopped at the stopping rule. `certify` now re-runs this study at probe
  precision; the committed version predates that and the paper says so.
- **No passive equilibrium rival was found at toeholds 0.05 and 0.20** at the default
  budget: the least-deterring accepted restart there still folds with probability 0.25
  and 0.33. `is_passive` in `seq_preemption_incentive.csv` marks the rows where the
  "rival that will not fold" description actually holds (only toehold 0 does).
