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
    """Return ``(cm, labels)`` with ``cm[i, j]`` = true ``i`` predicted ``j``.

    The matrix always spans the union of true and predicted labels. Restricting
    it to a subset would have to silently drop samples whose prediction falls
    outside that subset, which flatters recall: a model that answers "E3" on a
    fold containing no E3 would look like it had not answered at all.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    union = np.unique(np.concatenate([y_true, y_pred]))
    if labels is not None:
        union = np.unique(np.concatenate([np.asarray(labels), union]))

    index = {v: i for i, v in enumerate(union.tolist())}
    cm = np.zeros((union.size, union.size), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[index[t], index[p]] += 1
    return cm, union


def per_class_f1(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class F1 over the union of true and predicted labels."""
    cm, union = confusion_matrix(y_true, y_pred, labels)
    tp = np.diag(cm).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    actual = cm.sum(axis=1).astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(actual > 0, tp / actual, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    return f1, union


def macro_f1(
    y_true: np.ndarray, y_pred: np.ndarray, average_over: Sequence | None = None
) -> float:
    """Unweighted mean of per-class F1.

    ``average_over`` restricts which classes are *averaged*, not which samples
    are counted. Predicting a class outside that set still costs the true class
    its recall -- it simply does not add a zero-F1 term of its own.
    """
    f1, union = per_class_f1(y_true, y_pred)
    if average_over is None:
        return float(f1.mean())
    keep = np.isin(union, np.asarray(average_over))
    return float(f1[keep].mean()) if keep.any() else float("nan")


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true).ravel() == np.asarray(y_pred).ravel()))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean per-class recall over the classes actually present in ``y_true``."""
    cm, union = confusion_matrix(y_true, y_pred)
    actual = cm.sum(axis=1).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(actual > 0, np.diag(cm) / actual, np.nan)
    keep = np.isin(union, np.unique(y_true))
    return float(np.nanmean(recall[keep])) if keep.any() else float("nan")


def macro_f1_present(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 over the classes present in ``y_true`` only.

    Why this exists alongside :func:`macro_f1`: these benchmarks use very small
    subject-wise test folds that often do not contain every class. On
    ``KERAAL_clf_mc_CTK`` a fold holds 13-14 samples of a 4-class problem, and
    three of the six folds contain only two classes. Averaging over the union of
    true and predicted labels then charges the model a zero for any class that
    was never testable -- ``E2`` and ``E3`` appear in 4 and 6 test samples in
    total across the whole dataset -- which caps the achievable score near 0.5
    regardless of how good the model is.

    Neither variant is "the" right answer, so report both and say which is which.
    The union version is the stricter reading; this one measures performance on
    what the fold could actually assess.
    """
    return macro_f1(y_true, y_pred, average_over=np.unique(y_true))


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
    METRICS = {
        "macro_f1": macro_f1,
        "macro_f1_present": macro_f1_present,
        "balanced_accuracy": balanced_accuracy,
        "accuracy": accuracy,
    }

    per_fold_metrics: dict[str, list[float]] = {k: [] for k in METRICS}
    all_true, all_pred = [], []
    for fold in range(n_folds):
        x_tr, y_tr, x_te, y_te = load_fold(fold)
        y_hat = fit_predict(x_tr, y_tr, x_te)
        for name, fn in METRICS.items():
            per_fold_metrics[name].append(fn(y_te, y_hat))
        all_true.append(np.asarray(y_te).ravel())
        all_pred.append(np.asarray(y_hat).ravel())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    cm, labels = confusion_matrix(y_true, y_pred)
    f1, _ = per_class_f1(y_true, y_pred, labels)

    out = {
        "per_fold": per_fold_metrics["macro_f1"],
        "mean": float(np.mean(per_fold_metrics["macro_f1"])),
        "std": float(np.std(per_fold_metrics["macro_f1"])),
        "cm_pooled": cm,
        "labels": labels,
        "per_class_f1_pooled": f1,
        "n_folds": n_folds,
        # Predictions are kept so any metric can be recomputed without retraining.
        # Re-running LITEMV across both datasets costs ~80 minutes; recomputing a
        # metric from stored predictions costs milliseconds, and the choice of
        # metric is exactly the kind of thing that gets revised late.
        "y_true": y_true,
        "y_pred": y_pred,
        "fold_sizes": [len(a) for a in all_true],
    }
    for name, values in per_fold_metrics.items():
        out[f"{name}_per_fold"] = values
        out[f"{name}_mean"] = float(np.mean(values))
        out[f"{name}_std"] = float(np.std(values))
    return out


def rank_table(
    by_dataset: dict[str, dict[str, dict]], metric: str = "macro_f1_mean"
) -> dict:
    """Aggregate the ladder across datasets by **mean rank**.

    This is the headline result, and it exists because no single dataset here
    can carry one. Test folds run to 6-14 samples; a per-fold score can swing
    0.14 to 1.00 on sampling alone. Averaging *scores* across datasets is also
    misleading, because a dataset where every model scores 0.9 would dominate
    one where they all score 0.4.

    Ranking sidesteps both: on each dataset the models are ordered 1..k (ties
    share the average rank), and the reported figure is each model's mean rank
    over every dataset it ran on. It is the standard presentation in the
    time-series-classification literature for exactly this reason.

    Only datasets where a model actually ran count toward its mean, so a model
    evaluated on a subset is not silently credited or penalised -- but the
    ``n_datasets`` column makes the uneven coverage visible.
    """
    ranks: dict[str, list[float]] = {}
    wins: dict[str, int] = {}
    scores: dict[str, list[float]] = {}

    for results in by_dataset.values():
        usable = {m: r[metric] for m, r in results.items() if np.isfinite(r.get(metric, np.nan))}
        if len(usable) < 2:
            continue
        models = list(usable)
        values = np.array([usable[m] for m in models], dtype=np.float64)
        order = _average_ranks(-values)          # negate: higher score -> better rank
        best = float(values.max())
        for model, rank, value in zip(models, order, values):
            ranks.setdefault(model, []).append(float(rank))
            scores.setdefault(model, []).append(float(value))
            wins[model] = wins.get(model, 0) + int(value >= best - 1e-12)

    rows = []
    for model, values in ranks.items():
        rows.append({
            "model": model,
            "mean_rank": float(np.mean(values)),
            "n_datasets": len(values),
            "wins": wins.get(model, 0),
            "mean_score": float(np.mean(scores[model])),
            "median_score": float(np.median(scores[model])),
        })
    rows.sort(key=lambda r: r["mean_rank"])
    return {"rows": rows, "metric": metric, "n_datasets": len(by_dataset)}


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks starting at 1, with ties sharing their average."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    for value in np.unique(values):
        tied = values == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


def render_rank_table(summary: dict) -> str:
    lines = [
        f"| model | mean rank | wins | mean {summary['metric']} | median | datasets |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary["rows"]:
        lines.append(
            f"| {r['model']} | **{r['mean_rank']:.2f}** | {r['wins']} "
            f"| {r['mean_score']:.3f} | {r['median_score']:.3f} | {r['n_datasets']} |"
        )
    return "\n".join(lines)


def family_table(
    by_dataset: dict[str, dict[str, dict]], metric: str = "macro_f1_mean"
) -> str:
    """Per-source-collection means -- IRDS, KIMORE, KERAAL, UI-PRMD and so on.

    Worth separating because the collections differ in exercise, capture rig and
    difficulty, and a model that only wins on one family is a different claim
    from one that wins broadly.
    """
    families: dict[str, dict[str, list[float]]] = {}
    for name, results in by_dataset.items():
        family = name.split("_clf")[0]
        for model, r in results.items():
            value = r.get(metric)
            if value is not None and np.isfinite(value):
                families.setdefault(family, {}).setdefault(model, []).append(value)

    models = sorted({m for f in families.values() for m in f})
    lines = ["| family | n | " + " | ".join(models) + " |",
             "|---" * (len(models) + 2) + "|"]
    for family in sorted(families):
        counts = max(len(v) for v in families[family].values())
        cells = []
        for model in models:
            values = families[family].get(model)
            cells.append(f"{np.mean(values):.3f}" if values else "--")
        lines.append(f"| {family} | {counts} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def results_table(results: dict[str, dict]) -> str:
    """Render the model ladder as a markdown table.

    Four metrics, because on folds this small no single one is trustworthy
    alone -- see :func:`macro_f1_present` for why the two macro-F1 columns can
    differ so much.
    """
    lines = [
        "| model | macro-F1 (union) | macro-F1 (present) | balanced acc | accuracy | per fold (macro-F1 union) |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        folds = " ".join(f"{v:.3f}" for v in r["per_fold"])
        lines.append(
            f"| {name} "
            f"| {r['macro_f1_mean']:.3f} ± {r['macro_f1_std']:.3f} "
            f"| {r['macro_f1_present_mean']:.3f} ± {r['macro_f1_present_std']:.3f} "
            f"| {r['balanced_accuracy_mean']:.3f} ± {r['balanced_accuracy_std']:.3f} "
            f"| {r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f} "
            f"| {folds} |"
        )
    return "\n".join(lines)
