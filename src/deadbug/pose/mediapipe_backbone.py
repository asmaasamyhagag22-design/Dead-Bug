"""MediaPipe Pose Landmarker backbone (Tasks API).

Settings are carried over verbatim from ``scripts/run_pose.py``, which was
already correct for this MediaPipe version. Notes that cost time to rediscover:

- ``mp.solutions`` and ``mediapipe.framework`` no longer exist in 0.10.3x. Only
  ``mp.Image``, ``mp.ImageFormat`` and ``mp.tasks`` are importable, so every
  tutorial using ``mp.solutions.pose`` raises ``AttributeError``. Skeleton
  drawing is manual OpenCV -- see :mod:`deadbug.pose.draw`.
- ``RunningMode.VIDEO`` carries tracking state between calls, which is what
  keeps jitter down. It also means **one detector instance per clip**: reusing
  one across clips, with timestamps restarting and the frame size changing,
  crashes natively.
- Timestamps must be strictly increasing integer milliseconds derived from the
  frame index, never the wall clock.
- Reading the segmentation mask aborts the process on frames whose width is not
  4-byte aligned. :class:`deadbug.ingest.video_source.VideoSource` snaps the
  dimensions before we get here.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from ..config import REPO_ROOT
from .backbone import BackboneResult, PoseBackbone

N_MP33 = 33


class MediaPipeBackbone(PoseBackbone):
    """MediaPipe Pose Landmarker in VIDEO mode, with segmentation masks."""

    LAYOUT = "mp33"
    N_JOINTS = N_MP33

    def __init__(
        self,
        model_path: str | Path,
        num_poses: int = 1,
        output_segmentation_masks: bool = True,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        mask_binarize_threshold: float = 0.5,
        mask_downscale: int = 2,
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self.mask_binarize_threshold = mask_binarize_threshold
        self.mask_downscale = max(1, int(mask_downscale))

        path = Path(model_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"pose model not found: {path}")

        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
            output_segmentation_masks=output_segmentation_masks,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector = vision.PoseLandmarker.create_from_options(options)

    @classmethod
    def from_config(cls, cfg: dict) -> "MediaPipeBackbone":
        from ..config import cfg_get

        variant = cfg_get(cfg, "pose.variant")
        return cls(
            model_path=cfg_get(cfg, f"pose.model_paths.{variant}"),
            num_poses=cfg_get(cfg, "pose.num_poses"),
            output_segmentation_masks=cfg_get(cfg, "pose.output_segmentation_masks"),
            min_pose_detection_confidence=cfg_get(cfg, "pose.min_pose_detection_confidence"),
            min_pose_presence_confidence=cfg_get(cfg, "pose.min_pose_presence_confidence"),
            min_tracking_confidence=cfg_get(cfg, "pose.min_tracking_confidence"),
            mask_binarize_threshold=cfg_get(cfg, "pose.mask_binarize_threshold"),
            mask_downscale=cfg_get(cfg, "pose.mask_downscale"),
        )

    def infer(self, frame_rgb: np.ndarray, ts_ms: int) -> BackboneResult:
        mp = self._mp
        t0 = time.perf_counter()
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_rgb),
        )
        result = self._detector.detect_for_video(image, ts_ms)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            kpts = np.array(
                [[p.x, p.y, p.z, p.visibility] for p in lm], dtype=np.float32
            )
        else:
            kpts = self.empty(N_MP33)

        mask = None
        if result.segmentation_masks:
            raw = result.segmentation_masks[0].numpy_view()
            binary = (raw > self.mask_binarize_threshold).astype(np.uint8) * 255
            if self.mask_downscale > 1:
                h, w = binary.shape[:2]
                binary = cv2.resize(
                    binary,
                    (w // self.mask_downscale, h // self.mask_downscale),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask = binary

        return BackboneResult(kpts=kpts, mask=mask, latency_ms=latency_ms)

    def close(self) -> None:
        detector = getattr(self, "_detector", None)
        if detector is not None:
            detector.close()
            self._detector = None


def extract_clip(
    video_path: str | Path,
    out_npz: str | Path,
    cfg: dict,
    start_s: float | None = None,
    end_s: float | None = None,
    preview_path: str | Path | None = None,
) -> dict:
    """Run a whole clip through the backbone and persist the per-clip npz.

    A fresh backbone is constructed here, per clip, deliberately -- see the
    module docstring.

    Writes ``kpts_raw (T, 33, 4)``, ``mask (T, H', W')``, ``fps``,
    ``frame_size``, ``latency_ms (T,)`` and returns a summary dict for QC.
    """
    import cv2 as _cv2

    from ..config import cfg_get
    from ..ingest.video_source import VideoSource
    from .draw import PreviewWriter, draw_mask_contour, draw_skeleton

    source = VideoSource(
        video_path,
        snap_to=cfg_get(cfg, "ingest.snap_dims_to"),
        start_s=start_s,
        end_s=end_s,
    )
    draw_cfg = cfg_get(cfg, "pose.draw")

    all_kpts: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    latencies: list[float] = []
    n_detected = 0

    writer = PreviewWriter(
        preview_path,
        source.meta.fps,
        (source.meta.out_width, source.meta.out_height),
    )
    with MediaPipeBackbone.from_config(cfg) as backbone, writer:
        for index, frame in source:
            rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
            result = backbone.infer(rgb, source.ts_ms(index))

            all_kpts.append(result.kpts)
            latencies.append(result.latency_ms)
            if np.isfinite(result.kpts[:, :2]).any():
                n_detected += 1
            if result.mask is not None:
                all_masks.append(result.mask)

            if preview_path is not None:
                draw_skeleton(frame, result.kpts, vis_threshold=draw_cfg["vis_threshold"])
                draw_mask_contour(
                    frame, result.mask,
                    canny_lo=draw_cfg["canny_lo"], canny_hi=draw_cfg["canny_hi"],
                )
                writer.write(frame)

    if not all_kpts:
        raise RuntimeError(f"no frames read from {video_path}")

    kpts = np.stack(all_kpts).astype(np.float32)
    payload = {
        "kpts_raw": kpts,
        "fps": np.float32(source.meta.fps),
        "frame_size": np.array(
            [source.meta.out_width, source.meta.out_height], dtype=np.int32
        ),
        "latency_ms": np.array(latencies, dtype=np.float32),
    }
    if all_masks:
        payload["mask"] = np.stack(all_masks).astype(np.uint8)

    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **payload)

    return {
        "video": str(video_path),
        "npz": str(out_npz),
        "n_frames": int(kpts.shape[0]),
        "detection_rate": float(n_detected / max(1, kpts.shape[0])),
        "fps": float(source.meta.fps),
        "frame_size": [source.meta.out_width, source.meta.out_height],
        "source_size": [source.meta.width, source.meta.height],
        "snapped": source.meta.needs_resize,
        "median_latency_ms": float(np.median(latencies)) if latencies else float("nan"),
    }
