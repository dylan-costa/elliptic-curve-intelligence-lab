import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from eclab.features import EXCLUDED, feature_columns
from eclab.models import Boosting, Linear, Majority
from eclab.splits import validation_mask


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
    for model in (Linear("invariants", "ap"), Boosting("invariants", "ap")):
        assert not set(model.columns) & set(EXCLUDED)


def test_majority_predicts_the_base_rate(frame):
    df, y = frame
    assert np.allclose(Majority().fit(df, y).predict(df[:5]), y.mean())


def test_boosting_learns_parity_that_logistic_cannot(frame):
    df, y = frame
    train, test = df.index < 15_000, df.index >= 15_000
    val = validation_mask(df[train])
    lin = Linear("invariants").fit(df[train], y[train])
    gb = Boosting("invariants", max_iter=200).fit(df[train], y[train], val)
    assert roc_auc_score(y[test], lin.predict(df[test])) < 0.6
    assert roc_auc_score(y[test], gb.predict(df[test])) > 0.99
    assert gb.n_trees < 200  # stopped early on the validation classes

