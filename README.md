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
| 1 | Load and validate 3.8M curves | planned |
| 2 | Features (invariants, a_p) and leakage audit | planned |
| 3 | Isogeny-grouped and conductor-holdout splits | planned |
| 4 | Baselines: majority, logistic regression | planned |
| 5 | Gradient boosting and feature ablation | planned |
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
180 MB and is not committed. Step 1 adds the loader and documents where to put it.

## Notes

`notes/lectures.tex` is written alongside the code, with one section per step. To rebuild
the PDF:

```bash
cd notes && latexmk -pdf lectures.tex
```

## Layout

```
src/eclab/      the package (tasks.py defines the two prediction tasks)
tests/          pytest suite, run in CI on Python 3.11 and 3.12
notes/          lecture notes (LaTeX source and PDF)
```
