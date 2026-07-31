"""One coaching loop, used by every entry point.

``scripts/run_live.py``, the web app's camera socket and the web app's file
analyser all drive this class. That is deliberate and it is the project's main
honesty guarantee: the thing demonstrated live is byte-for-byte the thing that
produced the offline numbers. A second copy of this loop would drift within a
week, and the drift would be invisible -- both copies would keep producing
plausible rep counts.

What it owns, in the order a session needs them:

1. **The backbone.** One :class:`MediaPipeBackbone` per engine, never shared.
   ``RunningMode.VIDEO`` carries tracking state and requires strictly
   increasing timestamps, so reusing a detector across sources crashes natively.
2. **The floor.** Accumulated over the first frames while the subject lies
   still, then frozen. This is the calibration a recorded clip never provides
   and the reason the live path is strictly better than offline.
3. **The rotation baseline.** Same idea: the pelvis reading is only meaningful
   as a deviation from where this person's pelvis started.
4. **The session.** Phases, per-rep judgement, the report.

The caller supplies the clock. For a camera that is wall time; for a file it
must be ``frame_index / fps``, or every duration measures how fast the machine
runs MediaPipe rather than how fast the subject moved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..config import cfg_get
from ..geometry import floor as FL
from ..pose import skeleton as sk
from ..pose.mediapipe_backbone import MediaPipeBackbone
from ..signals import lumbar as LU
from ..signals import rotation as ROT
from .session import LiveSession, SessionConfig, SessionState

#: Frames of silhouette needed before the floor line is fitted and frozen.
SETUP_MIN_MASKS = 15


def contralateral(kpts: np.ndarray, frame_size: tuple[int, int]) -> dict[str, float]:
    """Wrist-to-opposite-ankle distance **in torso lengths**, per side.

    Two corrections that both matter and are easy to miss:

    - Convert to pixels first. MediaPipe's normalized coordinates give x and y
      different pixel scales, and a supine subject has the torso along x with
      the limbs swinging through y, so the raw distance is skewed by the
      frame's aspect ratio.
    - Divide by the current torso length, because the counter's ``min_rise``
      floor is expressed in torso lengths. Feeding it raw normalized units
      silently mismatches the scale and the counter simply never fires -- which
      is exactly what happened on the first end-to-end run.
    """
    k = np.asarray(kpts, dtype=np.float64).copy()
    k[:, 0] *= frame_size[0]
    k[:, 1] *= frame_size[1]

    hip_c = 0.5 * (k[sk.L_HIP, :2] + k[sk.R_HIP, :2])
    sho_c = 0.5 * (k[sk.L_SHOULDER, :2] + k[sk.R_SHOULDER, :2])
    torso = float(np.linalg.norm(sho_c - hip_c))
    if not np.isfinite(torso) or torso < 1e-6:
        return {}

    out = {}
    for side, (wrist, ankle) in sk.CONTRALATERAL_MP33.items():
        a, b = k[wrist, :2], k[ankle, :2]
        out[side] = (
            float(np.linalg.norm(a - b) / torso)
            if np.isfinite(a).all() and np.isfinite(b).all() else float("nan")
        )
    return out


@dataclass
class FrameResult:
    """Everything one processed frame produced, for rendering or serialising."""

    frame_index: int
    now: float
    detected: bool
    state: SessionState
    kpts: np.ndarray | None = None            # (33, 4) normalized
    mask_shape: tuple[int, int] | None = None
    lumbar_gap: float = float("nan")
    rot_dev: float = float("nan")
    floor_ready: bool = False
    latency_ms: float = float("nan")
    extra: dict[str, Any] = field(default_factory=dict)


class CoachEngine:
    """Stateful per-session coaching loop. One engine per source, never reused."""

    def __init__(
        self,
        cfg: dict,
        fps: float,
        frame_size: tuple[int, int],
        setup_seconds: float = 3.0,
        baseline_reps: int = 3,
    ) -> None:
        self.cfg = cfg
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.backbone = MediaPipeBackbone.from_config(cfg)
        self.session = LiveSession(
            fps=self.fps,
            config=SessionConfig(setup_seconds=setup_seconds, baseline_reps=baseline_reps),
        )

        self._setup_masks: list[np.ndarray] = []
        self._setup_kpts: list[np.ndarray] = []
        self._floor: dict | None = None
        self._rot_calib: float | None = None
        self._rot_history: list[float] = []
        self._last_ts_ms = -1

    # ------------------------------------------------------------------

    @property
    def floor(self) -> dict | None:
        return self._floor

    def process(self, frame_bgr: np.ndarray, frame_index: int, now: float) -> FrameResult:
        """Run one frame. ``now`` is seconds on the caller's chosen clock."""
        # Strictly increasing integer milliseconds, derived from the clock the
        # caller chose. detect_for_video rejects anything else.
        ts_ms = max(self._last_ts_ms + 1, int(now * 1000))
        self._last_ts_ms = ts_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.backbone.infer(rgb, ts_ms)
        kpts = result.kpts.astype(np.float64)
        mask = result.mask
        detected = bool(np.isfinite(kpts[:, :2]).any())

        lumbar_gap = float("nan")
        rot_dev = float("nan")
        mask_shape = mask.shape[:2] if mask is not None else None

        if detected and mask is not None:
            if self._floor is None:
                # Both appended together, always -- appending one without the
                # other is the desync class of bug that already cost this
                # project a day.
                self._setup_masks.append(mask)
                self._setup_kpts.append(kpts)
                if len(self._setup_masks) >= SETUP_MIN_MASKS:
                    self._floor = FL.estimate_floor_from_config(
                        np.stack(self._setup_masks), np.stack(self._setup_kpts), self.cfg
                    )
                    self._setup_masks.clear()
                    self._setup_kpts.clear()
            if self._floor is not None and np.isfinite(self._floor["b"]):
                kpx = LU.to_mask_pixels(kpts[None, ...], mask_shape)
                lumbar_gap = float(
                    LU.lumbar_gap(mask[None, ...], kpx, self._floor)["lumbar_gap"][0]
                )

            proxy = float(ROT.rot_proxy(kpts[None, ...], frame_size=self.frame_size)[0])
            if np.isfinite(proxy):
                if self._rot_calib is None:
                    self._rot_history.append(proxy)
                    if len(self._rot_history) >= SETUP_MIN_MASKS:
                        self._rot_calib = float(np.median(self._rot_history))
                else:
                    rot_dev = proxy - self._rot_calib

        state = self.session.update(
            contralateral(kpts, self.frame_size) if detected else {},
            lumbar_gap, rot_dev, frame_index, now,
        )
        return FrameResult(
            frame_index=frame_index, now=now, detected=detected, state=state,
            kpts=kpts if detected else None, mask_shape=mask_shape,
            lumbar_gap=lumbar_gap, rot_dev=rot_dev,
            floor_ready=self._floor is not None,
            latency_ms=result.latency_ms,
        )

    def finish(self) -> None:
        self.session.finish()

    def report(self) -> dict:
        report = self.session.report()
        report["counted_reps"] = self.session.counter.count
        report["floor"] = (
            {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
             for k, v in self._floor.items()}
            if self._floor else None
        )
        report["fps"] = self.fps
        report["frame_size"] = list(self.frame_size)
        return report

    def close(self) -> None:
        backbone = getattr(self, "backbone", None)
        if backbone is not None:
            backbone.close()
            self.backbone = None

    def __enter__(self) -> "CoachEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def engine_for_source(
    cfg: dict, fps: float, frame_size: tuple[int, int], **kwargs
) -> CoachEngine:
    """Construct an engine with the config's dimension snapping already applied."""
    from ..ingest.video_source import snap

    snap_to = cfg_get(cfg, "ingest.snap_dims_to")
    size = (snap(frame_size[0], snap_to), snap(frame_size[1], snap_to))
    return CoachEngine(cfg, fps, size, **kwargs)
