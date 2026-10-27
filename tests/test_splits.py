import numpy as np
import pandas as pd
import pytest

from eclab.splits import SPLITS, class_fold, split


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    n_classes = 20_000
    conductor = rng.integers(11, 3 * 10**8, n_classes)
    conductor[: n_classes // 2] = rng.integers(11, 500_001, n_classes // 2)
    size = rng.integers(1, 4, n_classes)
    df = pd.DataFrame({
        "iso_class": np.repeat([f"{c}.{i}" for i, c in enumerate(conductor)], size),
        "conductor": np.repeat(conductor, size),
    })
    df["conductor_is_prime"] = df.conductor % 2 == 1  # stand-in; only the masks' logic is tested
    return df.sample(frac=1, random_state=1).reset_index(drop=True)


def test_folds_are_deterministic_and_order_free(frame):
    folds = class_fold(frame.iso_class)
    shuffled = frame.sample(frac=1, random_state=2)
    assert (class_fold(shuffled.iso_class) == folds[shuffled.index]).all()


def test_folds_are_balanced(frame):
    share = np.bincount(class_fold(frame.drop_duplicates("iso_class").iso_class)) / frame.iso_class.nunique()
    assert np.allclose(share, 0.2, atol=0.02)


@pytest.mark.parametrize("name", SPLITS)
def test_no_class_on_both_sides(frame, name):
    train, test = split(frame, name)
    assert not (train & test).any()
    assert not set(frame.iso_class[train]) & set(frame.iso_class[test])
    assert train.any() and test.any()


def test_split_definitions(frame):
    n, prime = frame.conductor, frame.conductor_is_prime
    train, test = split(frame, "prime")
    assert prime[train | test].all() and n[train].max() <= 10**8 and n[test].min() > 2 * 10**8
    train, test = split(frame, "family")
    assert n[train].max() <= 500_000 and n[test].min() > 500_000
    with pytest.raises(ValueError):
        split(frame, "nope")
