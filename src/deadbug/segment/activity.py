"""Find the parts of a video where the exercise is actually happening.

An instructional clip is mostly not exercise. Measured on ``videoplayback (3)``
(97.9 s): the detections that survive the tempo filter have 3.6-6.0 s extension
times against a coached tempo of 1-2 s, and their alternation string is
``LRLLRLRLR`` where a Dead Bug alternates strictly -- a coach drifting between
demonstrations rather than a set. Feeding that whole span to the pipeline
corrupts three things at once:

- the floor estimate, which assumes the subject is lying down,
- the personal baseline, which assumes the first reps are reps,
- every rep-derived statistic, which gets slow body drift mixed in.

Trimming by hand solves it, but does not scale and cannot run on a video the
user just picked. So the segments are found from the signal itself: reps cluster
in time, and everything else does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..pose import skeleton as sk
from .reps import Rep, find_reps

#: A dead bug extension takes roughly one to four seconds. Anything slower is
#: the subject moving, not exercising. Same physiological prior the live
#: counter uses -- see live/counter.py::max_extend_s.
MIN_EXTEND_S = 0.4
MAX_EXTEND_S = 6.0

#: Reps further apart than this belong to different sets.
MAX_GAP_S = 12.0

#: A segment needs at least this many reps to be worth keeping.
MIN_REPS = 3


@dataclass
class Segment:
    start_s: float
    end_s: float
    n_reps: int
    sides: str

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def as_row(self) -> dict:
        return {
            "start_s": round(self.start_s, 1),
            "end_s": round(self.end_s, 1),
            "n_reps": self.n_reps,
            "duration_s": round(self.duration_s, 1),
        }


def contralateral_normalized(kpts: np.ndarray, frame_size: tuple[int, int]) -> dict[str, np.ndarray]:
    """Wrist-to-opposite-ankle distance in torso lengths, per side.

    Converts to pixels first: MediaPipe's normalized coordinates give x and y
    different pixel scales, and a supine subject lies along x while the limbs
    swing through y, so the raw distance is skewed by the frame's aspect ratio.
    """
    k = np.asarray(kpts, dtype=np.float64).copy()
    k[..., 0] *= frame_size[0]
    k[..., 1] *= frame_size[1]

    hip = 0.5 * (k[:, sk.L_HIP, :2] + k[:, sk.R_HIP, :2])
    sho = 0.5 * (k[:, sk.L_SHOULDER, :2] + k[:, sk.R_SHOULDER, :2])
    torso = np.linalg.norm(sho - hip, axis=-1)

    out = {}
    for side, (wrist, ankle) in sk.CONTRALATERAL_MP33.items():
        dist = np.linalg.norm(k[:, wrist, :2] - k[:, ankle, :2], axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[side] = np.where(torso > 1e-6, dist / torso, np.nan)
    return out


def plausible_reps(
    kpts: np.ndarray, fps: float, frame_size: tuple[int, int],
    min_extend_s: float = MIN_EXTEND_S, max_extend_s: float = MAX_EXTEND_S,
) -> list[Rep]:
    """Detected reps, filtered to those with a physiologically plausible tempo."""
    signals = contralateral_normalized(kpts, frame_size)
    kept: list[Rep] = []
    for side, signal in signals.items():
        for rep in find_reps(signal, fps, side=side):
            extend_s = (rep.peak - rep.start) / fps
            if min_extend_s <= extend_s <= max_extend_s:
                kept.append(rep)
    return sorted(kept, key=lambda r: r.start)


def find_segments(
    kpts: np.ndarray, fps: float, frame_size: tuple[int, int],
    max_gap_s: float = MAX_GAP_S, min_reps: int = MIN_REPS, margin_s: float = 1.5,
) -> list[Segment]:
    """Group plausible reps into contiguous exercise segments.

    Returns segments in time order. An empty list means the clip contains no
    recognisable set -- which is a legitimate and useful answer for an
    instructional video, not a failure.
    """
    reps = plausible_reps(kpts, fps, frame_size)
    if not reps:
        return []

    clusters: list[list[Rep]] = [[reps[0]]]
    for rep in reps[1:]:
        previous = clusters[-1][-1]
        if (rep.start - previous.end) / fps > max_gap_s:
            clusters.append([rep])
        else:
            clusters[-1].append(rep)

    segments = []
    duration_s = kpts.shape[0] / fps
    for cluster in clusters:
        if len(cluster) < min_reps:
            continue
        start = max(0.0, cluster[0].start / fps - margin_s)
        end = min(duration_s, cluster[-1].end / fps + margin_s)
        segments.append(
            Segment(start, end, len(cluster), "".join(r.side for r in cluster))
        )
    return segments


def summarise(segments: list[Segment], total_s: float) -> dict:
    """How much of the clip is actually exercise."""
    covered = sum(s.duration_s for s in segments)
    return {
        "n_segments": len(segments),
        "total_reps": sum(s.n_reps for s in segments),
        "exercise_s": round(covered, 1),
        "clip_s": round(total_s, 1),
        "exercise_fraction": round(covered / total_s, 3) if total_s else 0.0,
        "segments": [s.as_row() for s in segments],
    }
