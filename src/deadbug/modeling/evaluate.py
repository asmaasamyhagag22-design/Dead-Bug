"""Metrics and cross-validation protocol, shared by both tracks.

This module is deliberately numpy-only -- no sklearn, no aeon. Track A runs in
``venv-a`` and Track B in the MediaPipe venv, and this is the code they have in
common: the same macro-F1, the same leave-one-subject-out splitter, the same
leakage assertion. That shared use is what makes the two tracks one project
rather than two.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence

import numpy as np


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(cm, labels)`` with ``cm[i, j]`` = true ``i`` predicted ``j``."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)

    index = {v: i for i, v in enumerate(labels)}
    cm = np.zeros((labels.size, labels.size), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[index[t], index[p]] += 1
    return cm, labels


def per_class_f1(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class F1 and the label order it corresponds to."""
    cm, labels = confusion_matrix(y_true, y_pred, labels)
    tp = np.diag(cm).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    actual = cm.sum(axis=1).astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(actual > 0, tp / actual, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    return f1, labels


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence | None = None) -> float:
    """Unweighted mean of per-class F1 -- the headline metric for both tracks."""
    f1, _ = per_class_f1(y_true, y_pred, labels)
    return float(f1.mean())


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true).ravel() == np.asarray(y_pred).ravel()))


def wilson_interval(
    successes: int, n: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Used for the false-alarm rate, which must never be reported as a bare
    percentage. "0/48 reps flagged, 95% CI [0, 7.4%]" is defensible at a small
    subject count; "0% false alarms" from the same data is not, because the
    normal approximation gives a zero-width interval exactly when the estimate
    is least trustworthy.
    """
    if n <= 0:
        return (0.0, 1.0)
    z = _z_for(alpha)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _z_for(alpha: float) -> float:
    """Two-sided normal quantile. Avoids a scipy import in a shared module."""
    common = {0.10: 1.6448536269514722, 0.05: 1.959963984540054, 0.01: 2.5758293035489004}
    if alpha in common:
        return common[alpha]
    # Acklam's inverse-normal approximation; plenty accurate for a CI.
    p = 1.0 - alpha / 2.0
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------


def loso_folds(groups: Sequence) -> Iterator[tuple[np.ndarray, np.ndarray, object]]:
    """Leave-one-subject-out splits.

    Yields ``(train_idx, test_idx, held_out_group)``. The held-out identity is
    returned because the result must be reported per subject, not pooled -- at a
    small subject count the spread across subjects carries more information than
    the mean.
    """
    g = np.asarray(groups)
    for subject in _stable_unique(g):
        test = np.flatnonzero(g == subject)
        train = np.flatnonzero(g != subject)
        yield train, test, subject


def _stable_unique(a: np.ndarray) -> list:
    seen, out = set(), []
    for v in a.tolist():
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def assert_no_group_leak(train_groups: Sequence, test_groups: Sequence) -> None:
    """Raise if any subject appears on both sides of a split.

    Raises rather than warns, on purpose: subject leakage inflates every metric
    and produces no visible symptom, so it has to be impossible to ignore.
    """
    overlap = set(np.asarray(train_groups).tolist()) & set(np.asarray(test_groups).tolist())
    if overlap:
        raise AssertionError(
            f"subject leakage: {sorted(overlap)} appear in both train and test"
        )


def random_split_folds(
    n: int, n_splits: int = 5, seed: int = 0
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Group-blind K-fold, for the leakage sanity check only.

    Reporting this alongside the subject-wise number quantifies how much the
    protocol is worth. Never use it as a headline result.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    for test in np.array_split(order, n_splits):
        yield np.setdiff1d(order, test), test


# --------------------------------------------------------------------------
# Fold aggregation
# --------------------------------------------------------------------------


def eval_folds(
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    load_fold: Callable[[int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    n_folds: int,
) -> dict:
    """Run a model across every fold and aggregate.

    The reported number is the **mean over folds with its standard deviation**.
    The pooled confusion matrix is for display only -- the macro-F1 computed
    from it is a different quantity and must not be quoted as the result.
    A large std is itself a finding: it measures how much the score depends on
    which subjects happened to be held out.
    """
    per_fold, all_true, all_pred = [], [], []
    for fold in range(n_folds):
        x_tr, y_tr, x_te, y_te = load_fold(fold)
        y_hat = fit_predict(x_tr, y_tr, x_te)
        per_fold.append(macro_f1(y_te, y_hat))
        all_true.append(np.asarray(y_te).ravel())
        all_pred.append(np.asarray(y_hat).ravel())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    cm, labels = confusion_matrix(y_true, y_pred)
    f1, _ = per_class_f1(y_true, y_pred, labels)

    return {
        "per_fold": per_fold,
        "mean": float(np.mean(per_fold)),
        "std": float(np.std(per_fold)),
        "cm_pooled": cm,
        "labels": labels,
        "per_class_f1_pooled": f1,
        "n_folds": n_folds,
    }


def results_table(results: dict[str, dict]) -> str:
    """Render the model ladder as a markdown table."""
    lines = ["| model | macro-F1 (mean ± std) | per fold |",
             "|---|---|---|"]
    for name, r in results.items():
        folds = " ".join(f"{v:.3f}" for v in r["per_fold"])
        lines.append(f"| {name} | {r['mean']:.3f} ± {r['std']:.3f} | {folds} |")
    return "\n".join(lines)
