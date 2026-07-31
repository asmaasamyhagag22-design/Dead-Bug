"""Turn a folder of clips into the reps table.

One pass per clip:

    video -> pose+mask (cached .npz) -> floor -> normalize -> segments
          -> reps -> per-rep signal summaries -> rows

Three decisions worth stating, because each of them was arrived at the
expensive way:

**The npz cache is the unit of work, not the whole pipeline.** Pose extraction
is the only slow step -- minutes per clip against milliseconds for everything
downstream. Caching it in ``data/interim`` means a change to the lumbar window
or the prominence rule is re-run in seconds, so those parameters actually get
examined instead of being left at whatever was tried first.

**Reps are cut inside detected segments, never across the whole file.** An
instructional clip is mostly talking, and on ``videoplayback (3)`` what the
segmenter does find is a coach drifting between demonstrations rather than a set
(see LIMITATIONS.md section 9). Running the segmenter over the full timeline
mixes the coach walking around into the floor estimate and the personal
baseline. So
:mod:`deadbug.segment.activity` finds the exercise first, and a clip that
contains no set produces no rows -- which is the correct answer, not a failure.

**A clip that fails QC produces no rows.** Not a flag on the rows, no rows.
A rejected clip is one whose floor estimate or detection rate is not trustworthy,
and a per-rep number computed on top of an untrustworthy floor is not a
degraded measurement, it is a wrong one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from ..config import cfg_get, resolve_path, write_manifest
from ..geometry import floor as FL
from ..geometry.normalize import normalize, torso_len_cv, view_score, classify_view
from ..pose import skeleton as sk
from ..qc.report import core_visibility
from ..segment.activity import find_segments, summarise
from ..segment.reps import Rep, segment_from_config
from ..signals import ribcage as RIB
from ..signals import rotation as ROT
from ..signals import smoothness as SM
from ..signals import lumbar as LU
from .schema import ClipRecord, RepRecord, read_clips, write_clips

#: Written next to the reps table so a later run can tell what produced it.
REPS_BASENAME = "reps"


# --------------------------------------------------------------------------
# clips.csv bootstrap
# --------------------------------------------------------------------------


def clips_template(
    clips_dir: str | Path,
    dedup_csv: str | Path | None = None,
    out: str | Path | None = None,
) -> list[ClipRecord]:
    """Generate a ``clips.csv`` skeleton with ``person_id`` and ``condition`` blank.

    Deliberately not filled in. ``person_id`` is an identity claim and
    ``condition`` is a statement about intent; both come from the person who was
    there. What this does supply is the ``dedup_group``, so that the human
    assigning identities is looking at the evidence while they do it.

    The result will not pass :func:`~deadbug.dataset.schema.validate_clips`
    until those two columns are filled -- that is the point.
    """
    clips_dir = Path(clips_dir)
    groups = _read_dedup_groups(dedup_csv) if dedup_csv else {}

    records = []
    for path in sorted(clips_dir.glob("*.mp4")):
        records.append(
            ClipRecord(
                clip_id=_clip_id(path),
                file=str(path.relative_to(clips_dir.parent.parent))
                if clips_dir.is_absolute() else str(path),
                person_id="",
                condition="",
                view="",
                dedup_group=groups.get(path.name, ""),
                notes="FILL IN person_id, condition, view",
            )
        )
    if out is not None:
        write_clips(records, out)
    return records


def _read_dedup_groups(dedup_csv: str | Path) -> dict[str, str]:
    import csv

    path = Path(dedup_csv)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return {row["file"]: row.get("dedup_group", "") for row in csv.DictReader(fh)}


def _clip_id(path: Path) -> str:
    """A filesystem-safe, stable id from the filename."""
    stem = path.stem.lower()
    return "".join(ch if ch.isalnum() else "_" for ch in stem).strip("_")


# --------------------------------------------------------------------------
# Extraction cache
# --------------------------------------------------------------------------


def extract_or_load(
    video: str | Path, clip_id: str, cfg: dict, force: bool = False
) -> dict:
    """Return ``{"kpts_raw", "mask", "fps", "frame_size"}``, extracting if needed.

    The cache key is the clip id alone. That is safe because the config hash is
    written into the interim manifest and :func:`build_dataset` refuses a cache
    built under a different pose config.
    """
    interim = resolve_path(cfg, "paths.interim")
    npz_path = interim / f"{clip_id}.npz"

    if force or not npz_path.exists():
        from ..pose.mediapipe_backbone import extract_clip

        preview = None
        if cfg_get(cfg, "pose.write_preview", False):
            preview = resolve_path(cfg, "paths.preview") / f"{clip_id}.mp4"
            preview.parent.mkdir(parents=True, exist_ok=True)
        extract_clip(video, npz_path, cfg, preview_path=preview)

    with np.load(npz_path) as data:
        out = {k: data[k] for k in data.files}
    if "mask" not in out:
        raise RuntimeError(
            f"{npz_path} has no segmentation mask. The lumbar signal is read off "
            "the mask, so an extraction without one is unusable -- re-extract "
            "with pose.output_segmentation_masks: true."
        )

    # Hard guard, not a warning. A cache written before masks were padded has
    # one entry per *detected* frame rather than per frame, so mask t belongs to
    # some earlier moment than keypoints t. Every lumbar reading derived from it
    # is wrong and none of them look wrong.
    n_kpts, n_masks = out["kpts_raw"].shape[0], out["mask"].shape[0]
    if n_masks != n_kpts:
        raise RuntimeError(
            f"{npz_path} is a stale cache: {n_kpts} keypoint frames against "
            f"{n_masks} masks. Masks used to be appended only when present, which "
            "misaligns them with the keypoints. Re-extract with --force."
        )
    if "mask_valid" not in out:
        out["mask_valid"] = np.ones(n_kpts, dtype=bool)
    return out


# --------------------------------------------------------------------------
# Per-clip build
# --------------------------------------------------------------------------


def _rep_window(signal: np.ndarray, rep: Rep) -> np.ndarray:
    return np.asarray(signal, dtype=np.float64)[rep.start : rep.end + 1]


def _finite_max(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else float("nan")


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def build_clip(clip: ClipRecord, cfg: dict, force: bool = False) -> tuple[list[RepRecord], dict]:
    """Build every rep row for one clip, plus the clip's diagnostics.

    Returns ``(rows, diagnostics)``. ``rows`` is empty when the clip contains no
    detected exercise segment or when the floor estimate fails -- both are
    reported in ``diagnostics`` rather than raised, so one bad clip does not
    abort a batch.
    """
    video = resolve_path(cfg, "paths.clips").parent.parent / clip.file
    if not video.exists():
        video = Path(clip.file)
    data = extract_or_load(video, clip.clip_id, cfg, force=force)

    kpts_raw = data["kpts_raw"].astype(np.float64)
    masks = data["mask"]
    mask_valid = np.asarray(data["mask_valid"], dtype=bool)
    fps = float(data["fps"])
    frame_size = tuple(int(v) for v in data["frame_size"])

    # Everything that measures a LENGTH works from this, never from kpts_raw.
    # MediaPipe returns [0, 1] coordinates whose x and y have different pixel
    # scales, and a supine subject is the worst case -- the torso lies along x
    # while the limbs swing through y, so a raw distance is skewed by the
    # frame's aspect ratio. Measured here: the wrist-to-opposite-ankle distance
    # is distorted by up to 1.82x within a single clip, which does not cancel in
    # the auto-scaled prominence and changed videoplayback (6) from 13 reps to
    # 17. configs/base.yaml states the rule as `triage.view_space: pixel`;
    # activity.py and run_live.py both apply it. This is the third place.
    kpts_px_frame = kpts_raw.copy()
    kpts_px_frame[..., 0] *= frame_size[0]
    kpts_px_frame[..., 1] *= frame_size[1]

    diagnostics: dict = {
        "clip_id": clip.clip_id,
        "n_frames": int(kpts_raw.shape[0]),
        "fps": fps,
        "frame_size": list(frame_size),
        "detection_rate": float(np.mean(np.isfinite(kpts_raw[:, :, :2]).any(axis=(1, 2)))),
        "mean_visibility_core": core_visibility(kpts_raw),
        "view_score": float(np.nanmedian(view_score(kpts_raw, frame_size))),
        # A ratio of lengths, so it needs the same pixel correction.
        "torso_len_cv": torso_len_cv(kpts_px_frame),
    }
    diagnostics["view"] = classify_view(
        diagnostics["view_score"],
        side_max=cfg_get(cfg, "triage.view_score_side_max"),
        oblique_max=cfg_get(cfg, "triage.view_score_oblique_max"),
    )

    segments = find_segments(kpts_raw, fps, frame_size)
    diagnostics["activity"] = summarise(segments, kpts_raw.shape[0] / fps)
    if not segments:
        diagnostics["reason"] = "no exercise segment detected"
        diagnostics["n_reps"] = 0
        return [], diagnostics

    diagnostics["mask_rate"] = float(mask_valid.mean())
    if not mask_valid.any():
        diagnostics["reason"] = "no segmentation mask on any frame"
        diagnostics["n_reps"] = 0
        return [], diagnostics

    # Fit the floor to real masks only. A padded zero mask has no lower
    # boundary, so including them would just be noise -- but they must stay in
    # the array so the indices keep matching the keypoints.
    floor = FL.estimate_floor_from_config(masks[mask_valid], kpts_raw[mask_valid], cfg)
    diagnostics["floor"] = {k: _jsonable(v) for k, v in floor.items()}
    if not np.isfinite(floor["b"]):
        diagnostics["reason"] = "floor estimate failed"
        diagnostics["n_reps"] = 0
        return [], diagnostics
    if floor["inlier_ratio"] < cfg_get(cfg, "qc.min_floor_inlier_ratio"):
        diagnostics["reason"] = (
            f"floor inlier ratio {floor['inlier_ratio']:.2f} below "
            f"{cfg_get(cfg, 'qc.min_floor_inlier_ratio')}"
        )
        diagnostics["n_reps"] = 0
        return [], diagnostics

    # Signals are computed once over the whole clip and sliced per rep. Slicing
    # first would give each rep its own rotation baseline, and a per-rep
    # baseline cannot see a pelvis that is rotated for the entire set.
    kpts_px = LU.to_mask_pixels(kpts_raw, masks.shape[1:3])
    lumbar = LU.lumbar_gap(
        masks, kpts_px, floor,
        start_frac=cfg_get(cfg, "signals.lumbar.window_start_frac"),
        end_frac=cfg_get(cfg, "signals.lumbar.window_end_frac"),
    )["lumbar_gap"]
    rib = RIB.rib_gap(
        masks, kpts_px, floor,
        start_frac=cfg_get(cfg, "signals.ribcage.window_start_frac"),
        end_frac=cfg_get(cfg, "signals.ribcage.window_end_frac"),
    )["rib_gap"]
    rotation = ROT.rot_deviation_from_config(kpts_raw, cfg, frame_size=frame_size)["rot_dev"]

    # A padded frame has an empty silhouette, which the band code reads as a gap
    # of zero -- indistinguishable from a back pressed flat to the floor. NaN is
    # the honest value, and the per-rep summaries skip it.
    lumbar = np.where(mask_valid, lumbar, np.nan)
    rib = np.where(mask_valid, rib, np.nan)

    # Pixel-space input, so the rep signal is in real torso lengths and matches
    # what find_segments already decided on. Feeding kpts_raw here would have
    # find_segments and find_reps disagreeing about the units of the same
    # quantity.
    kpts_norm, _ = normalize(
        kpts_px_frame, stat=cfg_get(cfg, "geometry.normalize.torso_len_ref")
    )

    sparc_cfg = cfg_get(cfg, "signals.smoothness.sparc")
    rows: list[RepRecord] = []
    for segment in segments:
        lo = int(round(segment.start_s * fps))
        hi = min(kpts_norm.shape[0], int(round(segment.end_s * fps)))
        reps, _info = segment_from_config(kpts_norm[lo:hi], fps, cfg)
        for rep in reps:
            # Shift back onto the clip timeline so start_s means what it says.
            rep.start += lo
            rep.peak += lo
            rep.end += lo
            rows.append(
                _rep_row(clip, rep, len(rows), lumbar, rib, rotation, kpts_norm, fps, sparc_cfg)
            )

    diagnostics["n_reps"] = len(rows)
    # Derived from the rows actually emitted, so it covers the whole clip
    # without any cross-segment bookkeeping. None rather than True when there
    # are no reps: check_alternation returns intact vacuously for 0 or 1 rep,
    # and a degenerate clip must not silently pass the gate.
    diagnostics["alternation_intact"] = (
        None if not rows
        else not any("alternation_broken" in r.flags.split(";") for r in rows)
    )
    return rows, diagnostics


def _rep_row(
    clip: ClipRecord, rep: Rep, index: int,
    lumbar: np.ndarray, rib: np.ndarray, rotation: np.ndarray,
    kpts_norm: np.ndarray, fps: float, sparc_cfg: dict,
) -> RepRecord:
    lumbar_win = _rep_window(lumbar, rep)
    peak_offset = min(max(rep.peak - rep.start, 0), max(lumbar_win.size - 1, 0))
    at_peak = float(lumbar_win[peak_offset]) if lumbar_win.size else float("nan")

    smooth = SM.sparc_per_rep(
        kpts_norm, rep, fps,
        padlevel=sparc_cfg["padlevel"], fc=sparc_cfg["fc"], amp_th=sparc_cfg["amp_th"],
    )
    return RepRecord(
        rep_id=f"{clip.clip_id}__{index:03d}{rep.side}",
        clip_id=clip.clip_id,
        person_id=clip.person_id,
        condition=clip.condition,
        view=clip.view,
        side=rep.side,
        rep_index=index,
        start_s=rep.start / fps,
        end_s=rep.end / fps,
        peak_s=rep.peak / fps,
        duration_s=rep.duration_s,
        t_extend_s=rep.t_extend_s,
        t_return_s=rep.t_return_s,
        dwell_s=rep.dwell_s,
        overlap_s=rep.overlap_s,
        excursion_peak=rep.excursion_peak,
        lumbar_gap_mean=_finite_mean(lumbar_win),
        lumbar_gap_peak=_finite_max(lumbar_win),
        lumbar_gap_at_excursion_peak=at_peak,
        rib_gap_peak=_finite_max(_rep_window(rib, rep)),
        rot_dev_peak=ROT.rot_dev_peak(_rep_window(rotation, rep)),
        sparc_wrist=smooth["sparc_wrist"],
        sparc_ankle=smooth["sparc_ankle"],
        flags=";".join(rep.flags),
    )


# --------------------------------------------------------------------------
# Whole-dataset build
# --------------------------------------------------------------------------


def build_dataset(cfg: dict, force: bool = False, clips: Iterable[ClipRecord] | None = None) -> dict:
    """Build the reps table for every clip in the manifest.

    Writes ``data/processed/reps.parquet`` (or ``.csv`` when pyarrow is absent
    and ``dataset.csv_fallback`` is set) plus a manifest recording the config
    hash and git sha that produced it.
    """
    records = list(clips) if clips is not None else read_clips(
        resolve_path(cfg, "paths.clips_manifest")
    )

    rows: list[RepRecord] = []
    diagnostics = []
    for clip in records:
        if clip.qc_status == "reject":
            diagnostics.append({"clip_id": clip.clip_id, "reason": "qc_status=reject"})
            continue
        clip_rows, clip_diag = build_clip(clip, cfg, force=force)
        rows.extend(clip_rows)
        diagnostics.append(clip_diag)

    processed = resolve_path(cfg, "paths.processed")
    out_path = write_reps(rows, processed, cfg)
    write_manifest(
        processed, cfg, "dataset.build",
        n_clips=len(records), n_reps=len(rows), reps_table=str(out_path),
    )
    return {
        "n_clips": len(records),
        "n_reps": len(rows),
        "n_persons": len({r.person_id for r in rows}),
        "by_condition": _count(rows, "condition"),
        "path": str(out_path),
        "diagnostics": diagnostics,
    }


def write_reps(rows: list[RepRecord], out_dir: str | Path, cfg: dict) -> Path:
    """Write the reps table, preferring parquet and falling back to CSV."""
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([r.as_row() for r in rows], columns=RepRecord.field_names())

    engine = cfg_get(cfg, "dataset.parquet_engine", "pyarrow")
    parquet_path = out_dir / f"{REPS_BASENAME}.parquet"
    try:
        frame.to_parquet(parquet_path, engine=engine, index=False)
        return parquet_path
    except (ImportError, ValueError) as exc:
        if not cfg_get(cfg, "dataset.csv_fallback", True):
            raise
        csv_path = out_dir / f"{REPS_BASENAME}.csv"
        frame.to_csv(csv_path, index=False)
        csv_path.with_suffix(".csv.note").write_text(
            f"parquet unavailable ({exc}); wrote CSV instead\n", encoding="utf-8"
        )
        return csv_path


def read_reps(path: str | Path):
    """Load a reps table written by :func:`write_reps`."""
    import pandas as pd

    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _count(rows: list[RepRecord], attr: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        key = getattr(row, attr)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
