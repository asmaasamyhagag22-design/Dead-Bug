"""Measure every clip, and pre-fill what can honestly be pre-filled in clips.csv.

    python scripts/run_triage.py                # measure, write reports/triage.csv
    python scripts/run_triage.py --write-manifest   # also draft data/clips.csv

What it measures per clip: detection rate, median ``view_score`` (in **pixel**
space -- the normalized ratio is distorted by the aspect ratio), torso-length CV,
core-joint visibility, and how much of the clip is actually exercise.

What it fills in ``data/clips.csv``: ``view`` (from the measured score),
``dedup_group`` (from the dedup report), and the diagnostics columns.

What it deliberately leaves blank: ``person_id``. Two clips scoring 0.00
similarity are *different footage*, which is not the same claim as *different
people* -- the same coach appears in more than one video, and assuming
otherwise is the direction that leaks a subject across a LOSO split. The draft
therefore assigns one provisional id per dedup group and marks every row
``VERIFY person_id``, so the file does not validate until a human has looked.

``condition`` is left as ``correct`` for the existing footage, which is what all
of it is, and is the reason a fault detector cannot be evaluated on it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deadbug.config import cfg_get, load_config, resolve_path  # noqa: E402
from deadbug.dataset.build import _clip_id, extract_or_load  # noqa: E402
from deadbug.dataset.schema import ClipRecord, write_clips  # noqa: E402
from deadbug.geometry.normalize import classify_view, torso_len_cv, view_score  # noqa: E402
from deadbug.qc.report import core_visibility  # noqa: E402
from deadbug.segment.activity import find_segments, summarise  # noqa: E402


def measure(video: Path, cfg: dict, force: bool = False) -> dict:
    clip_id = _clip_id(video)
    data = extract_or_load(video, clip_id, cfg, force=force)

    kpts = data["kpts_raw"].astype(np.float64)
    fps = float(data["fps"])
    frame_size = tuple(int(v) for v in data["frame_size"])

    score = float(np.nanmedian(view_score(kpts, frame_size)))
    segments = find_segments(kpts, fps, frame_size)
    activity = summarise(segments, kpts.shape[0] / fps)

    return {
        "clip_id": clip_id,
        "file": str(video.relative_to(REPO_ROOT)).replace("\\", "/"),
        "n_frames": int(kpts.shape[0]),
        "fps": round(fps, 3),
        "duration_s": round(kpts.shape[0] / fps, 1),
        "detection_rate": round(
            float(np.mean(np.isfinite(kpts[:, :, :2]).any(axis=(1, 2)))), 3
        ),
        "mean_visibility_core": round(core_visibility(kpts), 3),
        "view_score": round(score, 3),
        "view": classify_view(
            score,
            side_max=cfg_get(cfg, "triage.view_score_side_max"),
            oblique_max=cfg_get(cfg, "triage.view_score_oblique_max"),
        ),
        "torso_len_cv": round(torso_len_cv(kpts), 3),
        "n_reps": activity["total_reps"],
        "exercise_fraction": activity["exercise_fraction"],
    }


def read_dedup_groups(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return {row["file"]: row.get("dedup_group", "") for row in csv.DictReader(fh)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--force", action="store_true", help="re-extract instead of using the cache")
    ap.add_argument("--write-manifest", action="store_true",
                    help="also draft data/clips.csv (will not validate until reviewed)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    clips_dir = resolve_path(cfg, "paths.clips")
    videos = sorted(clips_dir.glob("*.mp4"))
    if not videos:
        print(f"no .mp4 under {clips_dir}", file=sys.stderr)
        return 1

    groups = read_dedup_groups(resolve_path(cfg, "paths.reports") / "dedup.csv")

    rows = []
    for video in videos:
        print(f"measuring {video.name} ...", flush=True)
        try:
            row = measure(video, cfg, force=args.force)
        except Exception as exc:  # noqa: BLE001 - one bad clip must not stop the batch
            print(f"  FAILED: {exc}", file=sys.stderr)
            rows.append({"clip_id": _clip_id(video), "file": video.name, "error": str(exc)})
            continue
        row["dedup_group"] = groups.get(video.name, "")
        rows.append(row)
        print(
            f"  det={row['detection_rate']:.3f} view={row['view_score']:.3f}"
            f" ({row['view']}) torsoCV={row['torso_len_cv']:.3f}"
            f" reps={row['n_reps']} exercise={row['exercise_fraction']:.0%}"
        )

    out = resolve_path(cfg, "paths.reports") / "triage.csv"
    fields = sorted({k for row in rows for k in row})
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")

    (resolve_path(cfg, "paths.reports") / "triage.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    if args.write_manifest:
        manifest = resolve_path(cfg, "paths.clips_manifest")
        records = []
        for row in rows:
            if "error" in row:
                continue
            group = row.get("dedup_group", "")
            records.append(ClipRecord(
                clip_id=row["clip_id"], file=row["file"],
                # Provisional: one id per dedup group. See the module docstring.
                person_id=f"src_{group}" if group else "",
                condition="correct", view=row["view"],
                dedup_group=group,
                detection_rate=row["detection_rate"],
                view_score=row["view_score"], torso_len_cv=row["torso_len_cv"],
                notes="VERIFY person_id -- provisional, one id per dedup group",
            ))
        write_clips(records, manifest)
        print(f"wrote {manifest} ({len(records)} rows)")
        print("Review person_id before running `deadbug build`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
