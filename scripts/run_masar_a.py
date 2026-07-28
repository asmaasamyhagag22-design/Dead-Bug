"""Track A driver: the Rehab-Pile model ladder.

A script rather than a notebook -- it runs unattended in the background, reruns
cleanly, and ``venv-a`` has no jupyter.

    venv-a/Scripts/python.exe scripts/run_masar_a.py --sanity
    venv-a/Scripts/python.exe scripts/run_masar_a.py

The sanity run exists only to prove nothing raises. Do not read its number.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deadbug.config import cfg_get, load_config, resolve_path, seed_everything  # noqa: E402
from deadbug.modeling import evaluate, rehabpile, train  # noqa: E402


def benchmark_order() -> list[str]:
    """All 39 classification datasets, most relevant to this project first.

    Relevance order rather than size order on purpose: a run that dies partway
    should have finished the datasets the report actually leans on. Multi-class
    problems are error-*type* classification, which is what Dead Bug ultimately
    is, so they come before the binary ones.
    """
    import aeon.datasets.rehabpile_loader as R

    everything = sorted(R.REHABPILE_FOLDS["classification"])
    head = (
        [d for d in everything if d.startswith("KIMORE")]
        + [d for d in everything if d.startswith("KERAAL") and "_mc_" in d]
        + [d for d in everything if d.startswith("UCDHE") and "_mc_" in d]
        + [d for d in everything if d.startswith("KERAAL") and "_bn_" in d]
    )
    return head + [d for d in everything if d not in head]


def load_existing(csv_path: Path) -> dict[str, dict[str, dict]]:
    """Re-read a previous run so the benchmark is resumable.

    Only the aggregate metrics come back -- predictions are not reloaded -- but
    that is enough for the rank table, and it means a 10-hour run that dies at
    hour 8 does not start over.
    """
    if not csv_path.exists():
        return {}
    out: dict[str, dict[str, dict]] = {}
    lines = [l for l in csv_path.read_text(encoding="utf-8").splitlines() if l and not l.startswith("#")]
    if not lines:
        return {}
    header = lines[0].split(",")
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < len(header):
            continue
        row = dict(zip(header, parts))
        record: dict = {}
        for key, value in row.items():
            if key in ("dataset", "model", "per_fold"):
                continue
            try:
                record[key] = float(value)
            except ValueError:
                pass
        record["per_fold"] = [float(v) for v in row.get("per_fold", "").split() if v]
        record["mean"] = record.get("macro_f1_mean", float("nan"))
        record["std"] = record.get("macro_f1_std", float("nan"))
        record["n_folds"] = int(record.get("n_folds", 0))
        out.setdefault(row["dataset"], {})[row["model"]] = record
    return out


def run_dataset(name: str, cfg: dict, models: list[str], folds: int | None, sanity: bool) -> dict:
    extract_path = resolve_path(cfg, "masar_a.extract_path")
    extract_path.mkdir(parents=True, exist_ok=True)

    total_folds = rehabpile.n_folds(name)
    n = 1 if sanity else (folds or total_folds)
    n = min(n, total_folds)

    print(f"\n=== {name} ===")
    info = rehabpile.describe(name, fold=0, extract_path=extract_path)
    print(
        f"  shape (n, c, t) = ({info['n_train']}+{info['n_test']}, "
        f"{info['n_channels']}, {info['n_timepoints']})  "
        f"classes={info['classes']} counts={info['class_counts']}  folds={total_folds}"
    )
    if n < total_folds:
        print(f"  NOTE: running {n} of {total_folds} folds -- not the reportable number")

    def load(fold: int):
        return rehabpile.load_fold(name, fold, extract_path)

    lite_kwargs = dict(cfg_get(cfg, "masar_a.litemv"))
    if sanity:
        lite_kwargs["n_epochs"] = cfg_get(cfg, "masar_a.sanity.n_epochs")
    kwargs_for = {
        "litemv": lite_kwargs,
        "minirocket": dict(cfg_get(cfg, "masar_a.minirocket")),
    }
    rf_kwargs = dict(cfg_get(cfg, "masar_a.rf"))

    results: dict[str, dict] = {}
    for model in models:
        fn = train.FIT_PREDICT[model]
        kwargs = kwargs_for.get(model, rf_kwargs)
        t0 = time.time()
        try:
            res = evaluate.eval_folds(
                lambda a, b, c, _fn=fn, _kw=kwargs: _fn(a, b, c, **_kw), load, n
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  {model:12s} FAILED: {type(exc).__name__}: {exc}")
            continue
        res["seconds"] = time.time() - t0
        results[model] = res
        print(
            f"  {model:12s} macro-F1 {res['mean']:.3f} +/- {res['std']:.3f}"
            f"   ({res['seconds']:.0f}s)"
        )
    return results


def write_reports(all_results: dict[str, dict[str, dict]], cfg: dict, sanity: bool) -> None:
    reports = resolve_path(cfg, "paths.reports")
    figures = resolve_path(cfg, "paths.figures")
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    suffix = "_sanity" if sanity else ""

    header = (
        "dataset,model,macro_f1_mean,macro_f1_std,macro_f1_present_mean,"
        "macro_f1_present_std,balanced_accuracy_mean,balanced_accuracy_std,"
        "accuracy_mean,accuracy_std,n_folds,per_fold,seconds"
    )
    rows = [header]
    for dataset, results in all_results.items():
        for model, r in results.items():
            per_fold = " ".join(f"{v:.4f}" for v in r["per_fold"])
            rows.append(
                f"{dataset},{model},"
                f"{r['macro_f1_mean']:.4f},{r['macro_f1_std']:.4f},"
                f"{r['macro_f1_present_mean']:.4f},{r['macro_f1_present_std']:.4f},"
                f"{r['balanced_accuracy_mean']:.4f},{r['balanced_accuracy_std']:.4f},"
                f"{r['accuracy_mean']:.4f},{r['accuracy_std']:.4f},"
                f"{r['n_folds']},{per_fold},{r.get('seconds', 0):.1f}"
            )
    csv_path = reports / f"masar_a_results{suffix}.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"\nwrote {csv_path}")

    # Persist predictions so metrics can be revised without retraining. Records
    # restored from a previous run's CSV carry metrics only, so skip those.
    preds = reports / f"masar_a_predictions{suffix}.npz"
    payload = {}
    for dataset, results in all_results.items():
        for model, r in results.items():
            if "y_true" not in r:
                continue
            payload[f"{dataset}__{model}__true"] = r["y_true"]
            payload[f"{dataset}__{model}__pred"] = r["y_pred"]
            payload[f"{dataset}__{model}__folds"] = np.asarray(r["fold_sizes"])
    if payload:
        np.savez_compressed(preds, **payload)
        print(f"wrote {preds}")

    md = ["# Track A -- Rehab-Pile benchmark", ""]
    if sanity:
        md += ["> SANITY RUN -- one fold, reduced epochs. Not a reportable number.", ""]

    if len(all_results) > 1:
        summary = evaluate.rank_table(all_results)
        md += [
            f"## Headline: mean rank over {summary['n_datasets']} datasets",
            "",
            evaluate.render_rank_table(summary),
            "",
            "Mean rank, not mean score. Test folds here hold 6-14 samples, so a "
            "per-fold score can swing on sampling alone, and averaging raw scores "
            "would let a dataset where everything scores 0.9 outweigh one where "
            "everything scores 0.4. Ranking is the standard presentation in the "
            "time-series-classification literature for exactly this reason. "
            "`wins` counts datasets where a model tied or took the top score.",
            "",
            "`class_weight='balanced'` is set on RandomForest and MiniRocket. aeon "
            "does not expose it for LITETimeClassifier, so LITEMV runs unweighted -- "
            "relevant wherever the classes are imbalanced.",
            "",
            "## Per collection",
            "",
            evaluate.family_table(all_results),
            "",
            "## Per dataset",
            "",
        ]

    for dataset, results in all_results.items():
        md += [f"## {dataset}", "", evaluate.results_table(results), ""]
        best = max(results, key=lambda m: results[m].get("mean", float("-inf")), default=None)
        if best and "labels" in results[best]:
            r = results[best]
            md += [
                f"Pooled per-class F1 for `{best}`: "
                + ", ".join(
                    f"{lab}={v:.3f}"
                    for lab, v in zip(r["labels"], r["per_class_f1_pooled"])
                ),
                "",
                "The reported number is the **mean over folds with its standard "
                "deviation**. The pooled confusion matrix below is for display only; "
                "the macro-F1 computed from it is a different quantity. A large std "
                "measures how much the score depends on which subjects were held out.",
                "",
            ]
    md_path = reports / f"masar_a_results{suffix}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {md_path}")

    _plot_confusions(all_results, figures, suffix)


def _plot_confusions(all_results: dict, figures: Path, suffix: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    have_cm = [
        (d, m, r)
        for d, res in all_results.items()
        for m, r in res.items()
        if "cm_pooled" in r
    ]
    panels = [p for p in have_cm if p[1] == "litemv"][:2] or have_cm[:2]
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.8), squeeze=False)
    for ax, (dataset, model, r) in zip(axes[0], panels):
        cm = r["cm_pooled"]
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(r["labels"])), r["labels"], rotation=45, ha="right")
        ax.set_yticks(range(len(r["labels"])), r["labels"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{dataset}\n{model}  F1 {r['mean']:.3f} ± {r['std']:.3f}", fontsize=9)
    fig.suptitle("Pooled over folds -- display only; its F1 is not the reported mean", fontsize=8)
    fig.tight_layout()
    out = figures / f"fig1_confusion_matrix{suffix}.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--dataset", action="append", help="repeatable; defaults to config")
    ap.add_argument("--all", action="store_true", help="every classification dataset")
    ap.add_argument("--models", help="comma-separated subset of the ladder")
    ap.add_argument("--cheap", action="store_true", help="skip LITEMV (no TensorFlow)")
    ap.add_argument("--folds", type=int, help="cap the fold count (debugging only)")
    ap.add_argument("--sanity", action="store_true", help="1 fold, reduced epochs")
    ap.add_argument("--fresh", action="store_true", help="ignore previous results")
    ap.add_argument("--litemv-classifiers", type=int,
                    help="override the ensemble size (cost scales linearly)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.litemv_classifiers:
        cfg["masar_a"]["litemv"]["n_classifiers"] = args.litemv_classifiers
    seed_everything(cfg_get(cfg, "seed"))
    rehabpile.ensure_registry()

    if args.all:
        datasets = benchmark_order()
    else:
        datasets = args.dataset or cfg_get(cfg, "masar_a.datasets")

    if args.models:
        models = args.models.split(",")
    elif args.cheap:
        models = list(train.CHEAP_MODELS)
    else:
        models = list(cfg_get(cfg, "masar_a.models"))

    if args.sanity:
        print("SANITY RUN -- the goal is that nothing raises. Ignore the numbers.")

    suffix = "_sanity" if args.sanity else ""
    csv_path = resolve_path(cfg, "paths.reports") / f"masar_a_results{suffix}.csv"
    all_results = {} if args.fresh else load_existing(csv_path)
    if all_results:
        print(f"resuming: {len(all_results)} dataset(s) already in {csv_path.name}")

    for i, name in enumerate(datasets, 1):
        done = set(all_results.get(name, {}))
        todo = [m for m in models if m not in done]
        if not todo:
            print(f"[{i}/{len(datasets)}] {name}: already complete, skipping")
            continue

        print(f"\n[{i}/{len(datasets)}]", end=" ")
        try:
            res = run_dataset(name, cfg, todo, args.folds, args.sanity)
        except Exception as exc:  # noqa: BLE001 -- one bad dataset must not end the run
            print(f"  {name} FAILED to load: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if res:
            all_results.setdefault(name, {}).update(res)
            # Write after EVERY dataset. A ten-hour run that dies at hour eight
            # must not throw away work that only ever existed in memory.
            write_reports(all_results, cfg, args.sanity)

    if not all_results:
        print("\nno results -- every model failed", file=sys.stderr)
        return 1

    if len(all_results) > 1:
        print("\n" + evaluate.render_rank_table(evaluate.rank_table(all_results)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
