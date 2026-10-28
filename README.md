# Elliptic Curve Intelligence Lab

How much of the rank of an elliptic curve over Q can a model see from cheap arithmetic data,
such as the point counts a_p mod small primes? And does what it learns on small conductors
carry over to large ones? A second module benchmarks elliptic-curve and post-quantum
signature schemes using established implementations.

## The two tasks

| Task | Curves | Label |
|---|---|---|
| 1 | all | rank > 0 |
| 2 | root number +1, i.e. rank 0 or 2 | rank = 2 |

Task 1 is the obvious question, but it is partly answered by parity. The root number
w = (-1)^rank is a cheap local computation from the bad primes, so a model can score well by
rediscovering parity. Task 2 holds the root number fixed, so any skill must come from
elsewhere. It is the headline task. See Section 0 of the [lecture notes](notes/lectures.pdf).

The model never sees quantities computed from the rank or from L(E, 1), such as the
regulator, generators, the analytic order of Sha or the root number on Task 1. Section 2
of the notes audits every feature.

## Status

| Step | What | Status |
|---|---|---|
| 0 | Problem statement, repo, tests, CI; notes Section 0 | done |
| 1 | Load and validate 3.8M curves | done |
| 2 | Features (invariants, a_p) and leakage audit | done |
| 3 | Isogeny-grouped and conductor-holdout splits | done |
| 4 | Baselines: majority, logistic regression | done |
| 5 | Gradient boosting and feature ablation | done |
| 6 | Small neural network, calibration | planned |
| 7 | Out-of-distribution test, murmurations | planned |
| 8 | Ed25519 / ECDSA P-256 benchmark | planned |
| 9 | ML-DSA benchmark | planned |
| 10 | Reproducibility, report | planned |

## Setup

Requires Python 3.11 or newer.

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

On macOS/Linux, use `python3.11 -m venv .venv` and `source .venv/bin/activate`.

## Data

The curves come from an export of the LMFDB elliptic-curve tables: 3,824,372 curves over Q, one
per line, as `label|[a1,a2,a3,a4,a6]|rank|[torsion]|conductor|sha`. The file is about
180 MB and is not committed. To build the table:

```bash
python -m eclab.data.build --source path/to/ecdata_v2.txt
```

The default source is `$ECLAB_ECDATA`, or `data/raw/ecdata_v2.txt` if that is unset. The build
writes `data/curves.parquet` and `data/validation.json`. It runs 10 consistency checks;
for example, the discriminant of each minimal model must have exactly the conductor's
primes. The build fails if any curve fails a check. All 3,824,372 curves pass, and the run
takes about 80 seconds.

What the data looks like (from `scripts/figs_step01.py`):

- **Two parts:** 3,064,705 curves with conductor ≤ 500,000 (Cremona's complete table), plus
  759,667 above it. Every curve above 500,000 has a prime conductor (727,585) or a 7-smooth
  conductor (32,082).
- **Rank 2 share depends on the kind of conductor:** in Task 2 it is 22.9% for the complete
  range but 37.0% and 39.1% for prime conductors below and above 500,000. So the large-conductor
  test is mainly a shift in which conductors appear, and Step 3 compares prime with prime.
- **The `sha` column leaks the label:** Sha > 1 for 20.4% of rank-0 curves but only 0.1% of
  rank-2 curves. It is never used as a feature.

## Features

```bash
python -m eclab.features
```

This writes `data/features.parquet` in about 4 minutes. It has 15 invariants (conductor,
discriminant, j-invariant, torsion, reduction types) and a_p for the 25 primes below 100,
computed by vectorized point counting. `EXCLUDED` in `src/eclab/features.py` lists every
quantity that is never a feature and why.

PARI does not build on Windows, so the a_p are checked against facts they must satisfy
(from `scripts/figs_step02.py`):

- They agree with brute-force counts and with the modular forms of level 11 and 37.
- They satisfy the Hasse bound; the largest |a_p|/2√p is 0.995.
- They are identical across all 672,443 multi-curve isogeny classes.
- For all 1,513,650 semistable curves, they give the right parity of the rank.

The parameter-free sum S = Σ_{p<100} −a_p log p / p, which comes from the Birch and
Swinnerton-Dyer heuristic, reaches ROC-AUC **0.965 on Task 2** and 0.714 on Task 1 with no
training. Every learned model has to beat it.

## Splits

`src/eclab/splits.py` keeps each isogeny class on one side of every split, assigning folds by
a keyed hash of the class name. Isogenous curves share every a_p, so this matters: with a
curve-level split, 44.2% of Task 2 test curves have an exact a_p twin in training, and
copying its label is 100% accurate. With a class-level split the figure is 0%.

| Split | Train → test | Task 2 rank-2 share (train / test) |
|---|---|---|
| random | N ≤ 500k, folds 1–4 → fold 0 | 22.9% / 22.9% |
| conductor | N ≤ 400k → 400k < N ≤ 500k | 22.3% / 25.7% |
| prime | prime N ≤ 10⁸ → prime N > 2·10⁸ | 39.4% / 38.9% |
| family | N ≤ 500k → prime N > 500k | 22.9% / 39.1% |

## Notes

`notes/lectures.tex` is written alongside the code, with one section per step. To rebuild
the PDF:

```bash
cd notes && latexmk -pdf lectures.tex
```

## Layout

```
src/eclab/      the package
  tasks.py        the two prediction tasks
  curves.py       exact discriminants and factoring (Python integers)
  features.py     invariants, a_p by point counting, and the leakage audit
  splits.py       isogeny-class folds and the four train/test splits
  dataset.py      the table every model reads, and task_split(frame, task, split)
  evaluate.py     metrics, calibration, bootstrap over isogeny classes
  models.py       majority, logistic regression, gradient boosting
  data/           parsing (ecdata.py), validation (validate.py), build entry point (build.py)
scripts/        figs_stepNN.py: every number and figure in the notes
results/        stepNN_*.csv: the metrics behind every table
tests/          pytest suite, run in CI on Python 3.11 and 3.12;
                tests marked `data` need data/curves.parquet and skip without it
notes/          lecture notes (LaTeX source and PDF)
```
