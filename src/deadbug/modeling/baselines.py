"""Baseline feature extractors and estimators for the model ladder.

The ladder runs dumbest to smartest, and the rung that matters is the third:

    1. majority           absolute floor
    2. RF on flatten      weak and slow -- a strawman
    3. RF on summary      6 statistics per channel  <- THE BASELINE TO BEAT
    4. LITEMV             the model

Beating RF(flatten) proves nothing. Flattening a multivariate series into one
long vector destroys the temporal structure, so it is easy to beat and beating
it is not evidence that a time-series model was needed.
"""

from __future__ import annotations

import numpy as np


def summary_features(X: np.ndarray) -> np.ndarray:
    """``(n, c, t) -> (n, c*6)``: mean, std, min, max, mean |diff|, std diff.

    Cheap, strong, and usually close to a deep model on small data -- which is
    exactly why it is the honest baseline.
    """
    X = np.asarray(X, dtype=np.float64)
    d = np.diff(X, axis=2)
    return np.concatenate(
        [X.mean(2), X.std(2), X.min(2), X.max(2), np.abs(d).mean(2), d.std(2)],
        axis=1,
    )


def flatten_features(X: np.ndarray) -> np.ndarray:
    """``(n, c, t) -> (n, c*t)``. Included only as the strawman rung."""
    X = np.asarray(X, dtype=np.float64)
    return X.reshape(X.shape[0], -1)


def make_majority():
    from sklearn.dummy import DummyClassifier

    return DummyClassifier(strategy="most_frequent")


def make_rf(n_estimators: int = 300, random_state: int = 0, n_jobs: int = -1):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs
    )
