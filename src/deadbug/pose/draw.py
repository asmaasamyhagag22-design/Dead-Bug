"""Manual OpenCV overlays.

MediaPipe 0.10.3x removed ``mp.solutions.drawing_utils`` along with the rest of
``mp.solutions``, so every skeleton is drawn by hand. Carried over from
``scripts/run_pose.py``, which already did this correctly.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import skeleton as sk

BONE_COLOR = (0, 200, 120)
JOINT_COLOR = (30, 60, 240)
MASK_COLOR = (255, 160, 0)


def draw_skeleton(
    frame: np.ndarray,
    kpts: np.ndarray,
    vis_threshold: float = 0.5,
    joint_radius: int = 3,
    bone_thickness: int = 2,
    bone_color: tuple[int, int, int] = BONE_COLOR,
    joint_color: tuple[int, int, int] = JOINT_COLOR,
) -> np.ndarray:
    """Draw a MediaPipe-33 skeleton in place. ``kpts`` is normalized ``(33, 4)``."""
    if kpts is None or not np.isfinite(kpts[:, :2]).any():
        return frame

    h, w = frame.shape[:2]
    pts = np.stack([kpts[:, 0] * w, kpts[:, 1] * h], axis=1)
    ok = np.isfinite(pts).all(axis=1) & (kpts[:, 3] > vis_threshold)
    pts_i = np.where(np.isfinite(pts), pts, 0).astype(int)

    for a, b in sk.MP33_CONNECTIONS:
        if ok[a] and ok[b]:
            cv2.line(frame, tuple(pts_i[a]), tuple(pts_i[b]), bone_color, bone_thickness)
    for i in range(kpts.shape[0]):
        if ok[i]:
            cv2.circle(frame, tuple(pts_i[i]), joint_radius, joint_color, -1)
    return frame


def draw_mask_contour(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = MASK_COLOR,
    canny_lo: int = 50,
    canny_hi: int = 150,
) -> np.ndarray:
    """Outline the silhouette. The mask may be at a lower resolution."""
    if mask is None:
        return frame
    h, w = frame.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    frame[cv2.Canny(mask, canny_lo, canny_hi) > 0] = color
    return frame


def draw_floor_line(
    frame: np.ndarray, a: float, b: float, scale: float = 1.0,
    color: tuple[int, int, int] = (0, 0, 255), thickness: int = 1,
) -> np.ndarray:
    """Draw the fitted floor ``y = a*x + b``. ``scale`` maps mask px to frame px."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return frame
    h, w = frame.shape[:2]
    y0 = int(round(b * scale))
    y1 = int(round((a * (w / scale) + b) * scale))
    cv2.line(frame, (0, y0), (w - 1, y1), color, thickness)
    return frame


def put_hud(frame: np.ndarray, lines: list[str], origin: tuple[int, int] = (8, 20)) -> np.ndarray:
    """Small text overlay for live signal values in the preview video."""
    x, y = origin
    for i, text in enumerate(lines):
        pos = (x, y + i * 18)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


class PreviewWriter:
    """mp4 writer for annotated previews. A no-op when ``path`` is None."""

    def __init__(self, path: str | Path | None, fps: float, size: tuple[int, int]) -> None:
        self._writer = None
        if path is None:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
        )

    def write(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "PreviewWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
