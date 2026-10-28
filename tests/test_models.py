import numpy as np
import pandas as pd
import pytest

from eclab.features import EXCLUDED, feature_columns
from eclab.models import Linear, Majority


@pytest.fixture
def frame():
    """Synthetic curves whose label is the parity of n_bad_primes + n_nonsplit, as on Task 1."""
    rng = np.random.default_rng(0)
    n = 20_000
    df = pd.DataFrame(rng.normal(size=(n, len(feature_columns("invariants", "ap")))),
                      columns=feature_columns("invariants", "ap"))
    df["n_bad_primes"] = rng.integers(1, 4, n)
    df["n_nonsplit"] = rng.integers(0, 3, n)
    df["iso_class"] = [f"c{i}" for i in range(n)]
    df["rank"] = 9  # an excluded column with a value no model may use
    y = ((df.n_bad_primes + df.n_nonsplit) % 2).to_numpy()
    return df, y


def test_models_never_read_excluded_columns(frame):
    df, y = frame
    assert not set(Linear("invariants", "ap").columns) & set(EXCLUDED)


def test_majority_predicts_the_base_rate(frame):
    df, y = frame
    assert np.allclose(Majority().fit(df, y).predict(df[:5]), y.mean())

