"""Baseline feature extractors and estimators for the model ladder.

The ladder runs dumbest to smartest:

    1. majority           absolute floor
    2. RF on flatten      strawman -- flattening destroys temporal structure
    3. RF on summary      6 statistics per channel
    4. MiniRocket         <- THE BASELINE TO BEAT
    5. LITEMV             the model

Beating RF(flatten) proves nothing. MiniRocket is the honest bar: a random
convolutional transform plus a linear classifier, seconds to fit, and on small
time-series datasets it is routinely competitive with deep models. If LITEMV
cannot beat it, that is the finding.

**Class weighting is a correctness fix here, not tuning.** These datasets are
strongly imbalanced -- KERAAL_clf_mc_CTK is 108/77/49/51 -- and unweighted
models collapse onto the majority classes and never predict the rare ones at
all. Run 1 showed exactly that: pooled per-class F1 of C=.772 E1=.582 E2=.000
E3=.000, with macro-F1 averaging those zeros in at full weight.

Note the asymmetry: RandomForest and MiniRocket both accept ``class_weight``;
aeon does not expose it on ``LITETimeClassifier``. Report that rather than
hiding it -- it is part of why the comparison lands where it does.
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


def make_rf(
    n_estimators: int = 300,
    random_state: int = 0,
    n_jobs: int = -1,
    class_weight: str | None = "balanced",
):
    """Random forest, class-weighted by default.

    ``class_weight="balanced"`` is not a tuning knob here, it is a correctness
    fix. These datasets are strongly imbalanced -- KERAAL_clf_mc_CTK is
    108/77/49/51 -- and without it the forest collapses onto the majority
    classes and never predicts the rare ones at all. Run 1 showed exactly that:
    pooled per-class F1 of C=.772 E1=.582 E2=.000 E3=.000. Macro-F1 averages
    those zeros in at full weight.
    """
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight=class_weight,
    )


def make_minirocket(
    n_kernels: int = 10000,
    random_state: int = 0,
    n_jobs: int = -1,
    class_weight: str | None = "balanced",
):
    """MiniRocket -- random convolutional kernels + a linear classifier.

    Takes the multivariate series directly, so unlike the RF rungs it needs no
    hand-designed feature step and keeps the temporal structure. Fits in seconds
    on datasets this size, which is what makes running it across all 39
    benchmark problems practical.
    """
    from aeon.classification.convolution_based import MiniRocketClassifier

    return MiniRocketClassifier(
        n_kernels=n_kernels,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight=class_weight,
    )
