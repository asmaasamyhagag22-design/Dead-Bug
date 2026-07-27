"""fit/predict entry points, one per rung of the ladder.

Every function has the same signature ``(X_train, y_train, X_test) -> y_pred``
so :func:`deadbug.modeling.evaluate.eval_folds` can drive any of them.
"""

from __future__ import annotations

import numpy as np

from . import baselines


def fit_predict_majority(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, **_) -> np.ndarray:
    model = baselines.make_majority()
    model.fit(baselines.flatten_features(x_tr), y_tr)
    return model.predict(baselines.flatten_features(x_te))


def fit_predict_rf_flatten(
    x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray,
    n_estimators: int = 300, random_state: int = 0, n_jobs: int = -1, **_,
) -> np.ndarray:
    model = baselines.make_rf(n_estimators, random_state, n_jobs)
    model.fit(baselines.flatten_features(x_tr), y_tr)
    return model.predict(baselines.flatten_features(x_te))


def fit_predict_rf_summary(
    x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray,
    n_estimators: int = 300, random_state: int = 0, n_jobs: int = -1, **_,
) -> np.ndarray:
    model = baselines.make_rf(n_estimators, random_state, n_jobs)
    model.fit(baselines.summary_features(x_tr), y_tr)
    return model.predict(baselines.summary_features(x_te))


def fit_predict_litemv(
    x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray,
    use_litemv: bool = True, n_classifiers: int = 1, n_epochs: int = 300,
    batch_size: int = 64, random_state: int = 0, verbose: bool = False, **_,
) -> np.ndarray:
    """LITEMV -- ~10k parameters, built for exactly this sample size.

    ``use_litemv=True`` is mandatory for multivariate skeleton data. The aeon
    default of ``n_epochs=1500`` is far more than this needs.

    The import is inside the function on purpose: ``LITETimeClassifier`` builds
    Keras objects in ``__init__``, so it raises at *construction* when
    TensorFlow is missing. A module-level import would make this whole file
    unimportable without TF and take the sklearn baselines down with it.
    """
    from aeon.classification.deep_learning import LITETimeClassifier

    model = LITETimeClassifier(
        use_litemv=use_litemv,
        n_classifiers=n_classifiers,
        n_epochs=n_epochs,
        batch_size=batch_size,
        random_state=random_state,
        verbose=verbose,
    )
    model.fit(x_tr, y_tr)
    return model.predict(x_te)


FIT_PREDICT = {
    "majority": fit_predict_majority,
    "rf_flatten": fit_predict_rf_flatten,
    "rf_summary": fit_predict_rf_summary,
    "litemv": fit_predict_litemv,
}
