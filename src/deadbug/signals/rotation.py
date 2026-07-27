"""Pelvic rotation — reported as deviation, never as an absolute value.

The raw quantity is the hip separation over torso length:

    rot_proxy(t) = ‖ kpts[L_HIP] − kpts[R_HIP] ‖ / torso_len

**That is numerically the same thing as ``view_score``.** Read absolutely, it
measures where the camera is standing, not what the pelvis is doing: a subject
filmed 30° off-axis reads high for the entire clip while rocking not at all.

So everything here is expressed as a **deviation from the clip's own calibration
median**, taken over the first N frames. The camera angle is constant within a
clip, so it cancels, and what survives is the part that actually changes —
the rocking.
"""

from __future__ import annotations

import numpy as np

from ..geometry.normalize import torso_len
from ..pose import skeleton as sk

EPS = 1e-9


def rot_proxy(kpts: np.ndarray, frame_size: tuple[float, float] | None = None) -> np.ndarray:
    """``(T,)`` hip separation over torso length.

    Pass ``frame_size`` for normalized-coordinate input, for the same
    aspect-ratio reason as ``view_score``: x and y have different pixel scales,
    and a supine subject has the torso along x with the hips separating along y
    -- the worst case.
    """
    k = np.asarray(kpts, dtype=np.float64)
    if frame_size is not None:
        k = k.copy()
        k[..., 0] *= float(frame_size[0])
        k[..., 1] *= float(frame_size[1])

    sep = np.linalg.norm(k[..., sk.L_HIP, :2] - k[..., sk.R_HIP, :2], axis=-1)
    length = torso_len(k)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(length > EPS, sep / length, np.nan)


def depth_proxy(kpts: np.ndarray) -> np.ndarray:
    """``(T,)`` absolute difference of the two hips' z.

    A second view of the same rotation, from the channel a COCO-17 projection
    would have thrown away. Independent enough from the in-plane separation to
    disambiguate rotation from a change in camera angle.
    """
    k = np.asarray(kpts, dtype=np.float64)
    if k.shape[-1] < 3:
        return np.full(k.shape[0], np.nan)
    return np.abs(k[..., sk.L_HIP, 2] - k[..., sk.R_HIP, 2])


def calibration_median(signal: np.ndarray, n_frames: int = 30) -> float:
    """Median of the first ``n_frames`` finite samples -- the clip's own baseline."""
    head = np.asarray(signal, dtype=np.float64)[:n_frames]
    head = head[np.isfinite(head)]
    return float(np.median(head)) if head.size else float("nan")


def rot_deviation(
    kpts: np.ndarray,
    calib_frames: int = 30,
    use_z: bool = True,
    frame_size: tuple[float, float] | None = None,
) -> dict[str, np.ndarray]:
    """Pelvic rotation as deviation from the clip's baseline.

    Returns the raw series alongside the deviations so QC can see both, but
    **only the ``_dev`` series belong in any feature vector or threshold.**
    """
    proxy = rot_proxy(kpts, frame_size)
    baseline = calibration_median(proxy, calib_frames)
    out = {
        "rot_proxy": proxy,
        "rot_dev": proxy - baseline,
        "rot_calib": np.float64(baseline),
    }
    if use_z:
        depth = depth_proxy(kpts)
        out["depth_proxy"] = depth
        out["depth_dev"] = depth - calibration_median(depth, calib_frames)
    return out


def rot_dev_peak(deviation: np.ndarray) -> float:
    """Largest absolute deviation -- the per-rep summary for ``reps.parquet``."""
    d = np.asarray(deviation, dtype=np.float64)
    d = d[np.isfinite(d)]
    return float(np.max(np.abs(d))) if d.size else float("nan")


def rot_deviation_from_config(kpts: np.ndarray, cfg: dict, frame_size=None) -> dict:
    from ..config import cfg_get

    return rot_deviation(
        kpts,
        calib_frames=cfg_get(cfg, "signals.rotation.calib_frames"),
        use_z=cfg_get(cfg, "signals.rotation.use_z"),
        frame_size=frame_size,
    )
