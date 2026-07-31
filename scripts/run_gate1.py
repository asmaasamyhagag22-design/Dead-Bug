"""Gate 1: does the lumbar signal separate correct reps from arched ones?

    python scripts/run_gate1.py

Produces `reports/figures/gate1_lumbar_vs_excursion.png` and prints the numbers
that go with it. Run it after `deadbug build`; it reads the reps table and
nothing else.

**Written before the footage exists, on purpose.** The analysis is decided in
advance -- which axes, which statistic, which threshold counts as a pass -- so
that the figure cannot be steered once the data arrives. Everything here runs
today and reports honestly that there is nothing to plot yet.

The plot is `lumbar_gap_peak` against `excursion_peak`, not against time.
Excursion is how far the rep actually reached, so a rep that scores well by
never extending far enough to load the back sits at the left of the axis where
that is visible, rather than blending into a time series.

**Pass criterion, fixed in advance:** the `arched` reps sit above the correct
band's upper control limit more often than the correct reps do, by a margin
whose confidence interval excludes zero. Anything weaker is a trend, not a gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deadbug.config import cfg_get, load_config, resolve_path  # noqa: E402
from deadbug.dataset.build import read_reps  # noqa: E402
from deadbug.dataset.normative import (  # noqa: E402
    band_loso, fit_band, score_rep, supported_mask,
)
from deadbug.modeling.evaluate import wilson_interval  # noqa: E402

CONDITION_STYLE = {
    "correct": ("#1a7f37", "o", "correct"),
    "arched": ("#cf222e", "^", "arched"),
    "fast": ("#9a6700", "s", "fast"),
    "rotated": ("#8250df", "D", "rotated"),
}


def load_table(cfg: dict):
    processed = resolve_path(cfg, "paths.processed")
    for name in ("reps.parquet", "reps.csv"):
        if (processed / name).exists():
            return read_reps(processed / name)
    raise SystemExit("no reps table -- run `deadbug build` first")


def exceed_rate(reps, band, signal="lumbar_gap_peak") -> tuple[int, int, tuple[float, float]]:
    """How many of these reps break the band's upper limit, with a Wilson interval."""
    n_exceed = 0
    n_scored = 0
    for _, row in reps.iterrows():
        verdict = score_rep(band, row["excursion_peak"], row[signal])
        if not np.isfinite(verdict["z"]):
            continue                      # unsupported bin abstains
        n_scored += 1
        n_exceed += bool(verdict["exceeds"])
    lo, hi = wilson_interval(n_exceed, n_scored) if n_scored else (float("nan"),) * 2
    return n_exceed, n_scored, (lo, hi)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--signal", default="lumbar_gap_peak")
    args = ap.parse_args()

    cfg = load_config(args.config)
    reps = load_table(cfg)
    views = cfg_get(cfg, "dataset.band.views")
    reps = reps[reps["view"].isin(views)]

    conditions = sorted(reps["condition"].unique())
    print(f"{len(reps)} reps in views {views}")
    for condition in conditions:
        subset = reps[reps["condition"] == condition]
        print(f"  {condition:9s} {len(subset):3d} reps, "
              f"{subset['person_id'].nunique()} subject(s)")

    correct = reps[reps["condition"] == "correct"]
    if correct["person_id"].nunique() < 2:
        print(
            "\nGate 1 cannot run yet.\n"
            f"  correct reps in a side view: {len(correct)} "
            f"from {correct['person_id'].nunique()} subject(s)\n"
            "  a leave-one-subject-out band needs at least 2\n"
            "  and there are no faulted reps to test against.\n"
            "See FILMING.md -- one shoot supplies both.",
            file=sys.stderr,
        )
        return 1

    sigma = cfg_get(cfg, "dataset.control_limit.sigma")
    n_bins = cfg_get(cfg, "dataset.excursion_bins")
    pooled = fit_band(
        correct["excursion_peak"].to_numpy(), correct[args.signal].to_numpy(),
        correct["person_id"].to_numpy(), n_bins=n_bins, sigma=sigma,
    )
    bands = band_loso(
        correct["excursion_peak"].to_numpy(), correct[args.signal].to_numpy(),
        correct["person_id"].to_numpy(), n_bins=n_bins, sigma=sigma,
    )

    print(f"\nband: {pooled.n_reps} correct reps, {pooled.n_persons} subjects, "
          f"{n_bins} excursion bins, limit = mean + {sigma}*sd")
    print("\nreps above the upper control limit:")
    rows = []
    for condition in conditions:
        subset = reps[reps["condition"] == condition]
        # Correct reps are scored against the band fitted WITHOUT them; faulted
        # reps against the pooled correct band, which never saw any of them.
        if condition == "correct":
            n_exceed = n_scored = 0
            for person, group in subset.groupby("person_id"):
                held = bands.get(str(person))
                if held is None:
                    continue
                e, s, _ = exceed_rate(group, held, args.signal)
                n_exceed += e
                n_scored += s
            lo, hi = wilson_interval(n_exceed, n_scored) if n_scored else (float("nan"),) * 2
        else:
            n_exceed, n_scored, (lo, hi) = exceed_rate(subset, pooled, args.signal)
        rate = n_exceed / n_scored if n_scored else float("nan")
        rows.append((condition, n_exceed, n_scored, rate, lo, hi))
        print(f"  {condition:9s} {n_exceed:3d}/{n_scored:<3d} = {rate:5.1%}  "
              f"95% CI [{lo:.1%}, {hi:.1%}]")

    correct_row = next((r for r in rows if r[0] == "correct"), None)
    arched_row = next((r for r in rows if r[0] == "arched"), None)
    if correct_row and arched_row:
        print(
            f"\nGate 1: arched {arched_row[3]:.1%} vs correct {correct_row[3]:.1%}. "
            f"{'PASS' if arched_row[4] > correct_row[5] else 'NOT SEPARATED'} "
            "(pass = the arched CI sits entirely above the correct CI)."
        )
    else:
        print("\nGate 1: no `arched` reps present -- nothing to separate.", file=sys.stderr)

    _plot(reps, pooled, args.signal, cfg)
    return 0


def _plot(reps, band, signal: str, cfg: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centres = 0.5 * (np.asarray(band.edges[:-1]) + np.asarray(band.edges[1:]))
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    # Only draw the limit where it means something. fit_band interpolates the
    # per-bin statistics so the curve is continuous, but score_rep abstains in
    # bins below MIN_BIN_SUPPORT -- drawing a confident line through them would
    # show a decision boundary the system will not actually apply.
    supported = supported_mask(band)
    lower = np.asarray(band.mean) - band.sigma * np.asarray(band.std)
    upper = band.upper()
    shown_lower = np.where(supported, lower, np.nan)
    shown_upper = np.where(supported, upper, np.nan)

    ax.fill_between(centres, shown_lower, shown_upper, color="#1a7f37", alpha=0.12,
                    label=f"correct, mean ± {band.sigma}sd")
    ax.plot(centres, np.where(supported, band.mean, np.nan),
            color="#1a7f37", lw=1.2, alpha=0.7)
    ax.plot(centres, shown_upper, color="#1a7f37", lw=1.0, ls="--",
            label="upper control limit")
    if not supported.all():
        ax.plot(centres, upper, color="#8c959f", lw=0.8, ls=":", alpha=0.6,
                label=f"interpolated, no verdict ({int((~supported).sum())}/"
                      f"{len(supported)} bins)")

    for condition, (colour, marker, label) in CONDITION_STYLE.items():
        subset = reps[reps["condition"] == condition]
        if subset.empty:
            continue
        ax.scatter(subset["excursion_peak"], subset[signal], s=26, c=colour,
                   marker=marker, alpha=0.75, edgecolors="none",
                   label=f"{label} (n={len(subset)})")

    ax.set_xlabel("excursion at peak  (wrist-to-opposite-ankle, torso lengths)")
    ax.set_ylabel(f"{signal}  (silhouette area / torso_len²)")
    ax.set_title(
        f"Gate 1 — lumbar gap against excursion\n"
        f"band from {band.n_reps} correct reps, {band.n_persons} subjects, "
        f"leave-one-subject-out",
        fontsize=10,
    )
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.15)
    fig.tight_layout()

    out = resolve_path(cfg, "paths.figures") / "gate1_lumbar_vs_excursion.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
