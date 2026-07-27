"""Skeleton normalization: translate, scale, rotate -- in that order.

The order matters and so does the scale statistic. Scaling per frame injects
hip and shoulder jitter into every downstream signal, so the divisor is the
**clip-level median** torso length.

Only the ``x, y`` channels are transformed. ``z`` and ``visibility`` pass
through untouched: ``signals/rotation.py`` reads raw ``z``, and the QC gate
reads raw ``visibility``.
"""

from __future__ import annotations

import numpy as np

from ..pose import skeleton as sk

MIN_TORSO_LEN = 1e-6


def hip_center(kpts: np.ndarray) -> np.ndarray:
    """``(..., 33, C) -> (..., 2)`` midpoint of the two hips."""
    return 0.5 * (kpts[..., sk.L_HIP, :2] + kpts[..., sk.R_HIP, :2])


def shoulder_center(kpts: np.ndarray) -> np.ndarray:
    """``(..., 33, C) -> (..., 2)`` midpoint of the two shoulders."""
    return 0.5 * (kpts[..., sk.L_SHOULDER, :2] + kpts[..., sk.R_SHOULDER, :2])


def torso_vector(kpts: np.ndarray) -> np.ndarray:
    """``(..., 2)`` hip centre -> shoulder centre."""
    return shoulder_center(kpts) - hip_center(kpts)


def torso_len(kpts: np.ndarray) -> np.ndarray:
    """Per-frame torso length. Use :func:`torso_len_ref` for the scale divisor."""
    return np.linalg.norm(torso_vector(kpts), axis=-1)


def torso_len_ref(kpts: np.ndarray, stat: str = "median") -> float:
    """The clip-level reference torso length used as the scale divisor.

    Median by default, and deliberately so -- a handful of frames where the
    detector misplaces a shoulder must not move the reference.
    """
    lengths = torso_len(kpts)
    if stat == "median":
        value = float(np.nanmedian(lengths))
    elif stat == "mean":
        value = float(np.nanmean(lengths))
    else:
        raise ValueError(f"unknown torso_len_ref stat: {stat!r}")
    if not np.isfinite(value) or value < MIN_TORSO_LEN:
        raise ValueError(
            f"torso length reference is degenerate ({value!r}); "
            "the clip has no usable hip/shoulder detections"
        )
    return value


def torso_len_cv(kpts: np.ndarray) -> float:
    """Coefficient of variation of the torso length across the clip.

    A high value means the camera moved or zoomed -- QC flags above 0.10.
    """
    lengths = torso_len(kpts)
    mean = float(np.nanmean(lengths))
    if not np.isfinite(mean) or abs(mean) < MIN_TORSO_LEN:
        return float("nan")
    return float(np.nanstd(lengths) / mean)


def normalize(
    kpts: np.ndarray,
    stat: str = "median",
    rotate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a clip of keypoints.

    1. **Translate** -- subtract the per-frame hip centre.
    2. **Scale** -- divide by the clip-median torso length.
    3. **Rotate** -- align the torso axis to +x.

    Args:
        kpts: ``(T, 33, C)`` with ``C >= 2``. Any coordinate space, as long as
            x and y share a scale -- see :func:`view_score` for why that matters.
        stat: scale statistic, ``"median"`` (correct) or ``"mean"``.
        rotate: set False to ship translate+scale only.

    Returns:
        ``(kpts_norm, rot_angle)``. ``rot_angle`` is the per-frame torso
        orientation in radians *before* rotation; its variance across the clip
        is itself a stability feature, which is why it is returned rather than
        discarded.
    """
    if kpts.ndim != 3:
        raise ValueError(f"expected (T, J, C), got shape {kpts.shape}")

    out = np.asarray(kpts, dtype=np.float64).copy()
    xy = out[..., :2]

    # 1. translate
    xy -= hip_center(out)[:, None, :]

    # 2. scale by the clip median, not per frame
    xy /= torso_len_ref(out, stat=stat)

    # 3. rotate the torso onto +x
    torso = shoulder_center(out) - hip_center(out)
    angle = np.arctan2(torso[:, 1], torso[:, 0])
    if rotate:
        cos, sin = np.cos(angle), np.sin(angle)
        x, y = xy[..., 0].copy(), xy[..., 1].copy()
        # R(-angle) applied per frame
        xy[..., 0] = cos[:, None] * x + sin[:, None] * y
        xy[..., 1] = -sin[:, None] * x + cos[:, None] * y

    out[..., :2] = xy
    return out, angle


def view_score(kpts: np.ndarray, frame_size: tuple[float, float] | None = None) -> np.ndarray:
    """Camera-obliqueness proxy: hip separation over torso length.

    In a true side view the hip axis is parallel to the viewing direction, so it
    projects to nearly zero. Calibration from the design doc: 0 deg -> 0.00,
    15 deg -> 0.08, 30 deg -> 0.15, 45 deg -> 0.21, 90 deg -> 0.30.

    **Pass ``frame_size`` whenever the input is in normalized [0, 1] coords.**
    Normalized x and y have different pixel scales, so the ratio is distorted by
    the aspect ratio: measured on one real clip, 0.240 normalized against 0.131
    in pixels -- enough to flip a side/oblique verdict. Supine subjects are the
    worst case, because the torso runs along x while the hips separate along y.

    Args:
        kpts: ``(T, 33, C)``.
        frame_size: ``(width, height)`` in pixels; omit only if ``kpts`` is
            already in pixel coordinates.

    Returns:
        ``(T,)`` view score, NaN where the torso length is degenerate.
    """
    k = np.asarray(kpts, dtype=np.float64)
    if frame_size is not None:
        k = k.copy()
        k[..., 0] *= float(frame_size[0])
        k[..., 1] *= float(frame_size[1])

    hip_sep = np.linalg.norm(k[..., sk.L_HIP, :2] - k[..., sk.R_HIP, :2], axis=-1)
    torso = torso_len(k)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = hip_sep / torso
    return np.where(torso > MIN_TORSO_LEN, score, np.nan)


def classify_view(score: float, side_max: float = 0.12, oblique_max: float = 0.21) -> str:
    """Map a view score onto ``"side"`` / ``"oblique45"`` / ``"other"``.

    Only ``"side"`` clips may build the normative band; ``"oblique45"`` is
    test-only. Recalibrate the thresholds against footage shot at known angles
    rather than trusting the geometric table -- body proportions vary.
    """
    if not np.isfinite(score):
        return "other"
    if score < side_max:
        return "side"
    if score < oblique_max:
        return "oblique45"
    return "other"
