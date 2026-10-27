from pathlib import Path

import numpy as np
import pytest

from eclab.curves import b_invariants, factor
from eclab.data.ecdata import ainvs, read_ecdata
from eclab.features import (AP_COLUMNS, DEFAULT_FEATURES, EXCLUDED, GROUPS, PRIMES, ap_brute_force,
                            build_features, feature_columns, load_features)

FIXTURE = Path(__file__).parent / "fixtures" / "ecdata_sample.txt"


@pytest.fixture(scope="module")
def sample():
    curves = read_ecdata(FIXTURE)
    return curves, build_features(curves)


def test_ap_matches_brute_force(sample):
    curves, feats = sample
    for i, row in curves.iterrows():
        for p in PRIMES:
            assert feats.loc[i, f"ap_{p}"] == ap_brute_force(ainvs(row), p), (row.label, p)


@pytest.mark.parametrize("label, expected", [
    # coefficients of the modular forms of level 11 and 37
    ("11.a3", {2: -2, 3: -1, 5: 1, 7: -2, 11: 1, 13: 4}),
    ("37.a1", {2: -2, 3: -3, 5: -2, 7: -1, 37: -1}),
])
def test_known_q_expansions(sample, label, expected):
    curves, feats = sample
    row = feats[curves.label == label].iloc[0]
    assert {p: row[f"ap_{p}"] for p in expected} == expected


def test_hasse_bound(sample):
    _, feats = sample
    for p in PRIMES:
        assert (feats[f"ap_{p}"].abs() <= 2 * p**0.5).all()


def test_ap_constant_on_isogeny_classes(sample):
    curves, feats = sample
    assert (feats[AP_COLUMNS].groupby(curves.iso_class).nunique() == 1).all().all()


def test_split_criterion_matches_point_count(sample):
    """For multiplicative p >= 3, a_p = (-c6 / p); features.py relies on this for p >= 100."""
    curves, feats = sample
    checked = 0
    for i, row in curves.iterrows():
        b2, b4, b6, _ = b_invariants(*ainvs(row))
        c6 = -b2**3 + 36 * b2 * b4 - 216 * b6
        for p, e in factor(row.conductor).items():
            if e == 1 and 3 <= p < 100:
                legendre = 1 if pow(-c6 % p, (p - 1) // 2, p) == 1 else -1
                assert feats.loc[i, f"ap_{p}"] == legendre
                checked += 1
    assert checked >= 10


def test_semistable_parity(sample):
    """Exercise 0.2: for semistable E, w = (-1)^(omega(N)+1) prod a_p, and w = (-1)^rank."""
    curves, feats = sample
    s = feats.semistable
    w = (-1) ** (feats.n_bad_primes[s] + 1 + feats.n_nonsplit[s])
    assert (w == (-1) ** curves["rank"][s]).all()


def test_invariants_of_389a(sample):
    curves, feats = sample
    row = feats[curves.label == "389.a1"].iloc[0]
    assert row.conductor_is_prime and row.semistable and row.n_split == 1 and row.szpiro_ratio == 1.0


def test_no_excluded_column_is_a_feature():
    cols = feature_columns(*GROUPS)
    assert not set(cols) & set(EXCLUDED)
    assert "sha" in EXCLUDED and "rank" in EXCLUDED


def test_feature_columns_exist(sample):
    _, feats = sample
    assert set(feature_columns("invariants", "ap")) <= set(feats.columns)


@pytest.mark.data
def test_full_features():
    if not Path(DEFAULT_FEATURES).exists():
        pytest.skip("data/features.parquet not built")
    feats = load_features()
    assert len(feats) == 3_824_372
    bound = np.array([2 * p**0.5 for p in PRIMES])
    assert (np.abs(feats[AP_COLUMNS].to_numpy()) <= bound).all()
