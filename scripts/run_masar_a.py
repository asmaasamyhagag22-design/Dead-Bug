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

    rf_kwargs = {
        "n_estimators": cfg_get(cfg, "masar_a.rf.n_estimators"),
        "random_state": cfg_get(cfg, "masar_a.rf.random_state"),
        "n_jobs": cfg_get(cfg, "masar_a.rf.n_jobs"),
    }
    lite_kwargs = dict(cfg_get(cfg, "masar_a.litemv"))
    if sanity:
        lite_kwargs["n_epochs"] = cfg_get(cfg, "masar_a.sanity.n_epochs")

    results: dict[str, dict] = {}
    for model in models:
        fn = train.FIT_PREDICT[model]
        kwargs = lite_kwargs if model == "litemv" else rf_kwargs
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

    rows = ["dataset,model,macro_f1_mean,macro_f1_std,n_folds,per_fold,seconds"]
    for dataset, results in all_results.items():
        for model, r in results.items():
            per_fold = " ".join(f"{v:.4f}" for v in r["per_fold"])
            rows.append(
                f"{dataset},{model},{r['mean']:.4f},{r['std']:.4f},"
                f"{r['n_folds']},{per_fold},{r.get('seconds', 0):.1f}"
            )
    csv_path = reports / f"masar_a_results{suffix}.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"\nwrote {csv_path}")

    md = ["# Track A -- Rehab-Pile model ladder", ""]
    if sanity:
        md += ["> SANITY RUN -- one fold, reduced epochs. Not a reportable number.", ""]
    for dataset, results in all_results.items():
        md += [f"## {dataset}", "", evaluate.results_table(results), ""]
        best = max(results, key=lambda m: results[m]["mean"], default=None)
        if best:
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

    panels = [
        (d, m, r) for d, res in all_results.items() for m, r in res.items() if m == "litemv"
    ] or [(d, m, r) for d, res in all_results.items() for m, r in res.items()][:1]
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
    ap.add_argument("--models", help="comma-separated subset of the ladder")
    ap.add_argument("--folds", type=int, help="cap the fold count (debugging only)")
    ap.add_argument("--sanity", action="store_true", help="1 fold, reduced epochs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg_get(cfg, "seed"))
    rehabpile.ensure_registry()

    datasets = args.dataset or cfg_get(cfg, "masar_a.datasets")
    models = (
        args.models.split(",") if args.models else list(cfg_get(cfg, "masar_a.models"))
    )
    if args.sanity:
        print("SANITY RUN -- the goal is that nothing raises. Ignore the numbers.")

    all_results = {}
    for name in datasets:
        res = run_dataset(name, cfg, models, args.folds, args.sanity)
        if res:
            all_results[name] = res
            # Write after EVERY dataset, not once at the end. A long multi-dataset
            # run that dies partway (OOM, a killed terminal) would otherwise throw
            # away hours of finished work that only ever existed in memory.
            write_reports(all_results, cfg, args.sanity)

    if not all_results:
        print("\nno results -- every model failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
