"""Frame iteration, with the two fixes that let us skip ffmpeg entirely.

**Dimension snapping.** MediaPipe 0.10.35 aborts the *process* -- a native
``Check failed: 1 == ChannelSize() (1 vs. 4)`` with no Python traceback -- when
``segmentation_masks[0].numpy_view()`` is called on a frame whose width is not
4-byte aligned. Width 1006 gives 4024 bytes per row; every clip in this dataset
whose width is a multiple of 4 works, and 1006 kills the interpreter. Snapping
to a multiple of 16 before inference removes the crash. Verified on the clip
that reproduced it.

**Sub-clipping by seek.** Trimming with ffmpeg would mean re-encoding every
source to a constant frame rate. Instead we seek, carry the true fps, and let
``geometry.filters.resample_to_fps`` put the *signals* on a common rate. One
fewer dependency, one fewer CLI stage, and no generational quality loss.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def snap(value: int, to: int = 16) -> int:
    """Nearest multiple of ``to``, never below ``to``."""
    if to <= 1:
        return int(value)
    return max(to, int(round(value / to)) * to)


@dataclass(frozen=True)
class VideoMeta:
    path: Path
    width: int          # as stored in the file
    height: int
    fps: float
    n_frames: int
    out_width: int      # after snapping -- what the backbone actually sees
    out_height: int

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0

    @property
    def needs_resize(self) -> bool:
        return (self.out_width, self.out_height) != (self.width, self.height)


def probe(path: str | Path, snap_to: int = 16) -> VideoMeta:
    """Read container metadata. No ffprobe needed -- OpenCV reports fps correctly."""
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video: {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return VideoMeta(
        path=path, width=width, height=height, fps=fps, n_frames=n_frames,
        out_width=snap(width, snap_to), out_height=snap(height, snap_to),
    )


class VideoSource:
    """Iterate ``(frame_index, frame_bgr)`` over a clip or a sub-range of one.

    ``frame_index`` is absolute within the source file, so timestamps stay
    strictly increasing across a seek -- which ``detect_for_video`` requires.
    """

    def __init__(
        self,
        path: str | Path,
        snap_to: int = 16,
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> None:
        self.meta = probe(path, snap_to)
        self.start_s = start_s
        self.end_s = end_s
        self._size = (self.meta.out_width, self.meta.out_height)

    @property
    def start_frame(self) -> int:
        return 0 if self.start_s is None else int(round(self.start_s * self.meta.fps))

    @property
    def end_frame(self) -> int:
        if self.end_s is None:
            return self.meta.n_frames
        return min(self.meta.n_frames, int(round(self.end_s * self.meta.fps)))

    def ts_ms(self, frame_index: int) -> int:
        """Milliseconds for the VIDEO-mode API.

        Derived from the frame index and the file's fps, never from the wall
        clock -- the values must be strictly increasing integers.
        """
        return int(frame_index / self.meta.fps * 1000)

    def __len__(self) -> int:
        return max(0, self.end_frame - self.start_frame)

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        cap = cv2.VideoCapture(str(self.meta.path))
        if not cap.isOpened():
            raise OSError(f"cannot open video: {self.meta.path}")
        try:
            start, end = self.start_frame, self.end_frame
            if start:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            i = start
            while i < end:
                ok, frame = cap.read()
                if not ok:
                    break
                if self.meta.needs_resize:
                    frame = cv2.resize(frame, self._size, interpolation=cv2.INTER_AREA)
                yield i, frame
                i += 1
        finally:
            cap.release()
