"""The models of Sections 4-6. Each has fit(train, y, val=None) and predict(frame) -> P(label = 1).

``train`` and ``frame`` are rows of the table from eclab.dataset; a model picks its own columns
through feature_columns, so it can only see audited features. ``val`` is a boolean mask over
the training rows (whole isogeny classes, from splits.validation_mask) for models that stop early.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eclab.features import feature_columns


class Majority:
    """Predicts the training share of label 1 for every curve."""

    def fit(self, train, y, val=None):
        self.p = float(y.mean())
        return self

    def predict(self, frame):
        return np.full(len(frame), self.p)


class Linear:
    """Logistic regression on standardized features from the named groups."""

    def __init__(self, *groups: str, C: float = 1.0):
        self.columns = feature_columns(*groups)
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))

    def fit(self, train, y, val=None):
        self.model.fit(train[self.columns].to_numpy(np.float64), y)
        return self

    def predict(self, frame):
        return self.model.predict_proba(frame[self.columns].to_numpy(np.float64))[:, 1]

    def coefficients(self) -> dict:
        """Weight per feature, in units of one standard deviation of that feature."""
        return dict(zip(self.columns, self.model[-1].coef_[0]))


class Boosting:
    """Histogram gradient boosting on the named groups, stopped early on the validation classes.

    sklearn's own early stopping holds out random rows, which would put isogenous twins on both
    sides; here the held-out rows are whole classes.
    """

    def __init__(self, *groups: str, learning_rate: float = 0.1, max_leaf_nodes: int = 63,
                 max_iter: int = 2000, seed: int = 0):
        self.columns = feature_columns(*groups)
        self.model = HistGradientBoostingClassifier(
            learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes, max_iter=max_iter,
            early_stopping=True, n_iter_no_change=30, scoring="loss", random_state=seed)

    def fit(self, train, y, val):
        X = train[self.columns].to_numpy(np.float32)
        self.model.fit(X[~val], y[~val], X_val=X[val], y_val=y[val])
        return self

    def predict(self, frame):
        return self.model.predict_proba(frame[self.columns].to_numpy(np.float32))[:, 1]

    @property
    def n_trees(self) -> int:
        return self.model.n_iter_
