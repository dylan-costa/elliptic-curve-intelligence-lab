import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from eclab.evaluate import WeightedAuc, auc_replicates, bootstrap_weights, ece, interval, report, shift_prior
from eclab.features import PRIMES, mestre_nagao_sum


@pytest.fixture
def scored():
    rng = np.random.default_rng(0)
    codes = np.repeat(np.arange(3000), rng.integers(1, 4, 3000))
    y_class = rng.random(3000) < 0.3
    y = y_class[codes].astype(int)
    s = np.round(y + rng.normal(0, 1, len(y)), 1)  # rounded, so there are many ties
    return y, s, codes


def test_weighted_auc_matches_sklearn(scored):
    y, s, codes = scored
    w = next(bootstrap_weights(codes))
    assert WeightedAuc(y, s)(np.ones(len(y))) == pytest.approx(roc_auc_score(y, s))
    assert WeightedAuc(y, s)(w) == pytest.approx(roc_auc_score(y, s, sample_weight=w))


def test_bootstrap_resamples_whole_classes(scored):
    _, _, codes = scored
    w = next(bootstrap_weights(codes))
    per_class = pd.Series(w).groupby(codes).nunique()
    assert (per_class == 1).all()  # every curve in a class gets the same count
    assert w.sum() == pytest.approx(len(codes), rel=0.1)


def test_paired_difference_of_a_model_with_itself_is_zero(scored):
    y, s, codes = scored
    out = report(y, {"a": 1 / (1 + np.exp(-s)), "b": 1 / (1 + np.exp(-s))}, codes, baseline="a", n_boot=20)
    assert (out[["diff", "diff_lo", "diff_hi"]].abs() < 1e-12).all().all()


def test_interval_contains_point_estimate(scored):
    y, s, codes = scored
    reps = auc_replicates(y, {"s": s}, codes, n_boot=100)
    lo, hi = interval(reps["s"])
    assert lo < roc_auc_score(y, s) < hi


def test_ece_is_zero_when_calibrated_and_large_when_not():
    rng = np.random.default_rng(1)
    p = rng.random(200_000)
    y = (rng.random(len(p)) < p).astype(int)
    assert ece(y, p) < 0.01
    assert ece(y, np.full(len(p), 0.9)) == pytest.approx(0.4, abs=0.01)


def test_shift_prior_moves_the_base_rate():
    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(shift_prior(p, 0.3, 0.3), p)
    q = shift_prior(p, 0.2, 0.5)  # prior odds x4
    assert np.allclose(q / (1 - q), 4 * p / (1 - p))


def test_mestre_nagao_sum_skips_bad_primes():
    ap = np.array([[-2] * len(PRIMES), [-2] * len(PRIMES)])
    s = mestre_nagao_sum(ap, np.array([1, 11]))  # conductor 1 is a stand-in for "no bad prime < 100"
    full = 2 * sum(np.log(p) / p for p in PRIMES)
    assert s[0] == pytest.approx(full)
    assert s[1] == pytest.approx(full - 2 * np.log(11) / 11)

