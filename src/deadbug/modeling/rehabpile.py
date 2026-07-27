"""Rehab-Pile / KIMORE loading for Track A.

Track A exists to prove the training and evaluation code is correct against a
published benchmark before any of it is pointed at Dead Bug data. Its folds are
subject-wise by construction, so **do not** build a group splitter here -- the
benchmark already guarantees no subject spans train and test.

Requires ``venv-a`` (aeon + tensorflow). The MediaPipe venv cannot import this.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

KIMORE_BINARY = [
    "KIMORE_clf_bn_LA",   # lifting arms
    "KIMORE_clf_bn_LT",   # lateral tilt
    "KIMORE_clf_bn_PR",   # pelvis rotation
    "KIMORE_clf_bn_Sq",   # squat
    "KIMORE_clf_bn_TR",   # trunk rotation
]

#: Multi-class = classifying the *type* of error, the closest published
#: analogue to what this project is ultimately trying to do.
KERAAL_MULTICLASS = ["KERAAL_clf_mc_CTK", "KERAAL_clf_mc_ELK", "KERAAL_clf_mc_RTK"]


def ensure_registry() -> None:
    """Populate aeon's dataset-name list without depending on the network.

    ``load_rehab_pile_classification_datasets()`` scrapes the dataset host on
    first call. When that fails it returns an empty list, and the *next*
    ``load_rehab_pile_dataset`` call then raises a misleading "Dataset not
    found" that sends you hunting for a typo in a name that was correct.

    ``REHABPILE_FOLDS`` is a hardcoded dict in the same module, so the offline
    fallback is exact rather than a guess. Call this at import time.
    """
    import aeon.datasets.rehabpile_loader as R

    try:
        found = R.load_rehab_pile_classification_datasets()
    except Exception:  # noqa: BLE001 -- any network/parse failure is the same to us
        found = []
    if not found:
        R._rehabpile_classification_datasets = sorted(R.REHABPILE_FOLDS["classification"])
        R._rehabpile_regression_datasets = sorted(R.REHABPILE_FOLDS["regression"])


def list_datasets(task: str = "classification") -> list[str]:
    import aeon.datasets.rehabpile_loader as R

    ensure_registry()
    return sorted(R.REHABPILE_FOLDS[task])


def n_folds(name: str, task: str = "classification") -> int:
    """Fold count for a dataset, read from the registry rather than assumed.

    KIMORE datasets have 5; KERAAL have 6; UI-PRMD range from 5 to 8.
    """
    import aeon.datasets.rehabpile_loader as R

    folds = R.REHABPILE_FOLDS[task]
    if name not in folds:
        raise KeyError(f"unknown dataset {name!r}; try list_datasets({task!r})")
    return int(folds[name])


def load_fold(
    name: str, fold: int, extract_path: str | Path | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one subject-wise fold.

    Returns ``(X_train, y_train, X_test, y_test)`` with X shaped
    ``(n_cases, n_channels, n_timepoints)``.
    """
    from aeon.datasets import load_rehab_pile_dataset

    ensure_registry()
    path = str(extract_path) if extract_path is not None else None
    x_tr, y_tr = load_rehab_pile_dataset(name, split="train", fold=fold, extract_path=path)
    x_te, y_te = load_rehab_pile_dataset(name, split="test", fold=fold, extract_path=path)
    # The loader hands back (n, 1) column vectors; sklearn warns and aeon is
    # happier with 1-D targets.
    return x_tr, np.asarray(y_tr).ravel(), x_te, np.asarray(y_te).ravel()


def describe(name: str, fold: int = 0, extract_path: str | Path | None = None) -> dict[str, Any]:
    """Shape and class balance of a dataset -- run this before trusting a number."""
    x_tr, y_tr, x_te, y_te = load_fold(name, fold, extract_path)
    classes, counts = np.unique(np.concatenate([y_tr, y_te]), return_counts=True)
    return {
        "name": name,
        "n_folds": n_folds(name),
        "n_train": int(x_tr.shape[0]),
        "n_test": int(x_te.shape[0]),
        "n_channels": int(x_tr.shape[1]),
        "n_timepoints": int(x_tr.shape[2]),
        "classes": classes.tolist(),
        "class_counts": counts.tolist(),
    }
