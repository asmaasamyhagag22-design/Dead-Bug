"""The backbone interface every pose estimator implements.

A backbone returns its **native** layout. It does not convert to COCO-17.
Conversion lives in :func:`deadbug.pose.skeleton.to_coco17`, a pure function
that can be tested without a video or an inference session, and it is applied
in exactly two places: the feature builder and the backbone comparison. The
signal modules index MediaPipe-33 directly, because ``visibility`` is a QC gate
and ``z`` is needed for pelvis rotation, and neither survives the projection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class BackboneResult:
    """One frame of inference.

    Attributes:
        kpts: ``(N_JOINTS, 4)`` float32 -- x, y normalized to [0, 1], z, and
            visibility. All-NaN when nothing was detected, so a dropped frame
            is a hole rather than a silently repeated pose.
        mask: ``(H, W)`` uint8 in {0, 255}, possibly downscaled, or None.
        latency_ms: wall-clock inference time for this frame.
    """

    kpts: np.ndarray
    mask: np.ndarray | None
    latency_ms: float


class PoseBackbone(ABC):
    """Common interface. Subclasses declare the layout they emit."""

    LAYOUT: str = "mp33"
    N_JOINTS: int = 33

    @abstractmethod
    def infer(self, frame_rgb: np.ndarray, ts_ms: int) -> BackboneResult:
        """Run one frame. ``ts_ms`` must increase strictly across calls."""

    def close(self) -> None:
        """Release native resources."""

    def __enter__(self) -> "PoseBackbone":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @staticmethod
    def empty(n_joints: int) -> np.ndarray:
        return np.full((n_joints, 4), np.nan, dtype=np.float32)
