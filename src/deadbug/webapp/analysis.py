"""Run a whole video file through the coaching engine, with progress.

Same :class:`~deadbug.live.engine.CoachEngine` the camera uses and the same one
``scripts/run_live.py`` drives, on media time. A file uploaded through the web
UI and the same file played through the CLI produce identical numbers, which is
the property that lets the demo stand in for evidence.

Alongside the coaching report this returns the **triage** measurements --
detection rate, view score, torso-length CV -- because the honest answer to
"why did it say my form was fine" is sometimes "it could not see you". A UI that
reports reps without reporting whether the camera angle was usable is
overclaiming, and this project's whole posture is against that.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..config import cfg_get
from ..geometry.normalize import classify_view, torso_len_cv, view_score
from ..ingest.video_source import VideoSource
from ..live.engine import CoachEngine
from ..qc.report import core_visibility

ProgressFn = Callable[[float, str], None]


@dataclass
class Triage:
    """Was this footage usable at all? Reported next to every result."""

    detection_rate: float = float("nan")
    mean_visibility_core: float = float("nan")
    view_score: float = float("nan")
    view: str = "unknown"
    torso_len_cv: float = float("nan")
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: _clean(v) for k, v in asdict(self).items()}


def assess(kpts: np.ndarray, frame_size: tuple[int, int], cfg: dict) -> Triage:
    """Triage a clip's keypoints against the configured thresholds."""
    if kpts.size == 0:
        return Triage(warnings=["no frames were read from this video"])

    kpts_px = kpts.copy()
    kpts_px[..., 0] *= frame_size[0]
    kpts_px[..., 1] *= frame_size[1]

    score = float(np.nanmedian(view_score(kpts, frame_size)))
    triage = Triage(
        detection_rate=float(np.mean(np.isfinite(kpts[:, :, :2]).any(axis=(1, 2)))),
        mean_visibility_core=core_visibility(kpts),
        view_score=score,
        view=classify_view(
            score,
            side_max=cfg_get(cfg, "triage.view_score_side_max"),
            oblique_max=cfg_get(cfg, "triage.view_score_oblique_max"),
        ),
        torso_len_cv=torso_len_cv(kpts_px),
    )

    min_detection = cfg_get(cfg, "triage.min_detection_rate")
    if triage.detection_rate < min_detection:
        triage.warnings.append(
            f"the person was found in only {triage.detection_rate:.0%} of frames "
            f"(want {min_detection:.0%}+) - try better lighting or a clearer background"
        )
    if triage.view != "side":
        triage.warnings.append(
            f"camera angle reads as '{triage.view}' (view score {triage.view_score:.3f}). "
            "The lower-back signal is only defined from a true side view, so the "
            "back-arch check is unreliable here - the rep count is not affected"
        )
    max_cv = cfg_get(cfg, "triage.max_torso_len_cv")
    if np.isfinite(triage.torso_len_cv) and triage.torso_len_cv > max_cv:
        triage.warnings.append(
            f"the camera moved or zoomed during the clip (torso-length CV "
            f"{triage.torso_len_cv:.2f}, want under {max_cv:.2f}) - distances are "
            "measured in torso lengths, so this inflates them"
        )
    return triage


def analyse_video(
    path: str | Path,
    cfg: dict,
    setup_seconds: float = 3.0,
    baseline_reps: int = 3,
    progress: ProgressFn | None = None,
    annotate_to: str | Path | None = None,
    max_seconds: float | None = None,
) -> dict:
    """Coach a whole file. Returns the report plus triage and a per-rep timeline.

    Args:
        annotate_to: write an overlaid mp4 here. Costs roughly a third again on
            top of inference, so the caller decides.
        max_seconds: stop early. Extraction is linear in duration and an hour
            of video is an hour of MediaPipe.
    """
    from ..live import ui
    from ..pose.draw import draw_skeleton

    source = VideoSource(path, snap_to=cfg_get(cfg, "ingest.snap_dims_to"))
    size = (source.meta.out_width, source.meta.out_height)
    fps = source.meta.fps
    total = source.meta.n_frames or 0
    if max_seconds:
        total = min(total, int(max_seconds * fps)) if total else int(max_seconds * fps)

    writer = None
    if annotate_to is not None:
        Path(annotate_to).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(annotate_to), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
        )
        if not writer.isOpened():
            writer = None

    all_kpts: list[np.ndarray] = []
    timeline: list[dict] = []
    seen_reps = 0

    engine = CoachEngine(
        cfg, fps=fps, frame_size=size,
        setup_seconds=setup_seconds, baseline_reps=baseline_reps,
    )
    try:
        for index, frame in source:
            if max_seconds and index / fps > max_seconds:
                break
            # Media time, never wall time -- see CoachEngine.process.
            result = engine.process(frame, index, index / fps)
            all_kpts.append(result.kpts if result.kpts is not None else np.full((33, 4), np.nan))

            state = result.state
            if state.reps_done > seen_reps and state.last_verdict is not None:
                v = state.last_verdict
                timeline.append({
                    "rep": v.rep_index, "side": v.side, "ok": bool(v.ok),
                    "errors": list(v.errors), "message": v.message,
                    "at_s": round(index / fps, 2),
                    "detail": {k: _clean(x) for k, x in v.detail.items()},
                })
                seen_reps = state.reps_done

            if writer is not None:
                if result.detected:
                    draw_skeleton(frame, result.kpts, vis_threshold=0.5)
                ui.render(frame, state, kpts=result.kpts,
                          mask_shape=result.mask_shape, fps=fps)
                writer.write(frame)

            if progress and total and index % 15 == 0:
                progress(
                    min(0.99, index / total),
                    f"{state.phase.value}: {state.reps_done} rep(s) so far",
                )
        engine.finish()
        report = engine.report()
    finally:
        engine.close()
        if writer is not None:
            writer.release()

    kpts = np.stack(all_kpts) if all_kpts else np.zeros((0, 33, 4))
    report["triage"] = assess(kpts, size, cfg).as_dict()
    report["timeline"] = timeline
    report["duration_s"] = round(kpts.shape[0] / fps, 1) if fps else 0.0
    report["source"] = Path(path).name
    report["annotated"] = str(annotate_to) if (annotate_to and writer is not None) else None
    report = _sanitise(report)

    if not report["total_reps"]:
        report["explanation"] = (
            f"No rep was judged. The counter saw {report.get('counted_reps', 0)} rep(s), "
            f"but the first {setup_seconds:.0f}s go to floor calibration and the next "
            f"{baseline_reps} rep(s) set your personal baseline - only reps after that "
            "are scored. On a short clip that can leave nothing over."
        )
    if progress:
        progress(1.0, "done")
    return report


def _clean(value):
    """JSON has no NaN. Report it as null rather than as a number."""
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if math.isnan(f) or math.isinf(f) else round(f, 6)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_clean(v) for v in value.tolist()]
    return value


def _sanitise(obj):
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v) for v in obj]
    return _clean(obj)
