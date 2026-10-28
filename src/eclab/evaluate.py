"""Metrics, calibration, and bootstrap intervals over isogeny classes.

Curves in one isogeny class share their features and their label, so they are not independent
test cases. The bootstrap here resamples whole classes: each replicate draws classes with
replacement and gives every curve its class's draw count as a weight. Every model is scored
on the same replicates, so differences between models get paired intervals.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

N_BOOT = 200
EPS = 1e-6


class WeightedAuc:
    """ROC-AUC under many weight vectors for one fixed score; sorts once, then O(n) per call.

    AUC = sum over (positive, negative) pairs of w_i w_j [s_i > s_j] (ties count 1/2),
    divided by the total positive weight times the total negative weight.
    """

    def __init__(self, y: np.ndarray, s: np.ndarray):
        order = np.argsort(s, kind="mergesort")
        s_sorted = s[order]
        self.order = order
        self.group = np.cumsum(np.r_[True, s_sorted[1:] != s_sorted[:-1]]) - 1  # ties share a group
        self.y = y[order].astype(np.float64)

    def __call__(self, w: np.ndarray) -> float:
        w = w[self.order]
        pos = np.bincount(self.group, weights=w * self.y)
        neg = np.bincount(self.group, weights=w * (1 - self.y))
        below = np.cumsum(neg) - neg
        return float((pos * (below + neg / 2)).sum() / (pos.sum() * neg.sum()))


def bootstrap_weights(codes: np.ndarray, n_boot: int = N_BOOT, seed: int = 0):
    """One weight vector per replicate: each curve gets the number of times its class was drawn."""
    rng = np.random.default_rng(seed)
    k = codes.max() + 1
    for _ in range(n_boot):
        yield np.bincount(rng.integers(0, k, k), minlength=k)[codes].astype(np.float64)


def auc_replicates(y: np.ndarray, scores: dict, codes: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """ROC-AUC of every model on the same class-bootstrap replicates (one row per replicate)."""
    aucs = {name: WeightedAuc(y, s) for name, s in scores.items()}
    return pd.DataFrame([{name: f(w) for name, f in aucs.items()}
                         for w in bootstrap_weights(codes, n_boot, seed)])


def interval(reps: pd.Series, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(reps.quantile(a)), float(reps.quantile(1 - a))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error with equal-width bins: sum over bins of share * |mean p - frequency|."""
    b = np.minimum((p * bins).astype(int), bins - 1)
    n = np.bincount(b, minlength=bins)
    gap = np.abs(np.bincount(b, weights=p, minlength=bins) - np.bincount(b, weights=y, minlength=bins))
    return float(gap.sum() / n.sum())


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 15) -> pd.DataFrame:
    """Per bin: mean predicted probability, observed frequency of label 1, number of curves."""
    b = np.minimum((p * bins).astype(int), bins - 1)
    df = pd.DataFrame({"bin": b, "p": p, "y": y}).groupby("bin").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    return df.reset_index()


def shift_prior(p: np.ndarray, pi_from: float, pi_to: float) -> np.ndarray:
    """Correct probabilities for label shift: multiply the odds by the ratio of prior odds."""
    odds = p / (1 - p) * (pi_to / (1 - pi_to)) / (pi_from / (1 - pi_from))
    return odds / (1 + odds)


def estimate_prior(p: np.ndarray, pi_train: float, iters: int = 100, tol: float = 1e-8) -> float:
    """The share of label 1 on an unlabelled test set, by EM (Saerens, Latinne and Decaestecker, 2002).

    Assumes label shift: p(x | label) is the same as in training. Alternates between shifting the
    predictions to the current prior and setting the prior to the mean shifted prediction.
    """
    pi = pi_train
    for _ in range(iters):
        new = float(shift_prior(p, pi_train, pi).mean())
        if abs(new - pi) < tol:
            break
        pi = new
    return pi


def report(y: np.ndarray, preds: dict, codes: np.ndarray, baseline: str | None = None,
           n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """One row per model: ROC-AUC with a 95% class-bootstrap interval, PR-AUC, log-loss, ECE,
    and (if a baseline is named) the paired difference in ROC-AUC from it with its interval."""
    reps = auc_replicates(y, preds, codes, n_boot, seed)
    rows = []
    for name, p in preds.items():
        q = np.clip(p, EPS, 1 - EPS)
        lo, hi = interval(reps[name])
        row = {"model": name, "roc_auc": roc_auc_score(y, p), "roc_lo": lo, "roc_hi": hi,
               "pr_auc": average_precision_score(y, p), "log_loss": log_loss(y, q, labels=[0, 1]),
               "ece": ece(y, q)}
        if baseline is not None:
            d = reps[name] - reps[baseline]
            row |= {"diff": row["roc_auc"] - roc_auc_score(y, preds[baseline]),
                    "diff_lo": interval(d)[0], "diff_hi": interval(d)[1]}
        rows.append(row)
    return pd.DataFrame(rows)
