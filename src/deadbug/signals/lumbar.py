"""The lumbar gap -- the project's primary signal.

The error that matters in a Dead Bug is the lower back lifting off the floor,
and it is **invisible in the skeleton**: MediaPipe has no landmark anywhere on
the spine. So the signal is read off the silhouette instead, as the area
enclosed between the body's lower boundary and the floor line over a window
spanning the lumbar region.

    u              = unit vector hip_centre -> shoulder_centre
    window         = x in [0.00, 0.35] * torso_len_px along u
    gap(x)         = max(0, floor_y(x) - bottom(x))
    lumbar_gap(t)  = sum gap(x) / torso_len_px**2      dimensionless, scale-free
    lumbar_gap_max = max  gap(x) / torso_len_px        back height in torso-lengths

Dividing by ``torso_len_px**2`` is what makes the signal independent of camera
distance and body size: an area over a length squared.

There is no fixed correct range of motion, so nothing here thresholds an
absolute distance to the floor. The correct ROM is per person -- the excursion
at which *their* gap starts to rise -- which is why the control-limit curve
plots this signal against excursion rather than time.
"""

from __future__ import annotations

import numpy as np

from ..geometry.floor import floor_y, lower_boundary
from ..pose import skeleton as sk

EPS = 1e-9


def to_mask_pixels(kpts: np.ndarray, mask_shape: tuple[int, int]) -> np.ndarray:
    """Scale normalized [0, 1] keypoints into mask pixel coordinates.

    The mask is stored downscaled, so this is the only place the two coordinate
    spaces meet. Everything else stays in normalized or mask-pixel space and
    never mixes them.
    """
    h, w = mask_shape[:2]
    out = np.asarray(kpts, dtype=np.float64).copy()
    out[..., 0] *= w
    out[..., 1] *= h
    return out


def torso_frame(kpts_px: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(hip_c, u, torso_len)`` in whatever space ``kpts_px`` is in.

    ``u`` is the unit hip->shoulder direction. Its sign is what makes the window
    follow a subject facing either way without a special case.
    """
    hip_c = 0.5 * (kpts_px[..., sk.L_HIP, :2] + kpts_px[..., sk.R_HIP, :2])
    sho_c = 0.5 * (kpts_px[..., sk.L_SHOULDER, :2] + kpts_px[..., sk.R_SHOULDER, :2])
    vec = sho_c - hip_c
    length = np.linalg.norm(vec, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = vec / np.where(length[..., None] > EPS, length[..., None], np.nan)
    return hip_c, u, length


def band_window(
    hip_c: np.ndarray, u: np.ndarray, torso_len: float,
    start_frac: float, end_frac: float, width: int,
) -> tuple[int, int]:
    """Column range of a signal band, clipped to the frame.

    Returns a half-open ``[x0, x1)``. Empty when the torso is degenerate.
    """
    if not np.isfinite(torso_len) or torso_len <= EPS or not np.isfinite(u[0]):
        return (0, 0)
    a = hip_c[0] + start_frac * torso_len * u[0]
    b = hip_c[0] + end_frac * torso_len * u[0]
    x0, x1 = (a, b) if a <= b else (b, a)
    x0 = max(0, int(round(x0)))
    x1 = min(width, int(round(x1)))
    return (x0, max(x0, x1))


def gap_profile(mask: np.ndarray, a: float, b: float, x0: int, x1: int) -> np.ndarray:
    """``max(0, floor_y(x) - bottom(x))`` over ``[x0, x1)``.

    Columns with no silhouette contribute zero -- an absent body is not a gap.
    """
    if x1 <= x0:
        return np.zeros(0)
    bottom = lower_boundary(mask)[x0:x1]
    ground = floor_y(a, b, np.arange(x0, x1))
    gap = ground - bottom
    return np.where(np.isfinite(gap), np.maximum(0.0, gap), 0.0)


def band_signal(
    masks: np.ndarray,
    kpts_px: np.ndarray,
    floor: dict,
    start_frac: float,
    end_frac: float,
) -> dict[str, np.ndarray]:
    """Compute a silhouette band signal over a clip.

    Shared by the lumbar band ``[0.00, 0.35]`` and the rib band ``[0.45, 0.75]``
    -- they differ only in the window, so they must not differ in code.

    Returns ``{"area": (T,), "peak": (T,), "width_px": (T,)}`` where ``area`` is
    normalized by ``torso_len**2`` and ``peak`` by ``torso_len``.
    """
    masks = np.asarray(masks)
    n = len(masks)
    hip_c, u, torso = torso_frame(np.asarray(kpts_px, dtype=np.float64))
    a, b = floor["a"], floor["b"]

    area = np.full(n, np.nan)
    peak = np.full(n, np.nan)
    widths = np.zeros(n, dtype=np.int32)

    height, width = masks.shape[1], masks.shape[2]
    del height
    for t in range(n):
        length = float(torso[t])
        if not np.isfinite(length) or length <= EPS:
            continue
        x0, x1 = band_window(hip_c[t], u[t], length, start_frac, end_frac, width)
        widths[t] = x1 - x0
        if x1 <= x0:
            area[t], peak[t] = 0.0, 0.0
            continue
        gap = gap_profile(masks[t], a, b, x0, x1)
        area[t] = float(gap.sum()) / (length * length)
        peak[t] = float(gap.max()) / length
    return {"area": area, "peak": peak, "width_px": widths}


def lumbar_gap(
    masks: np.ndarray,
    kpts_px: np.ndarray,
    floor: dict,
    start_frac: float = 0.00,
    end_frac: float = 0.35,
) -> dict[str, np.ndarray]:
    """The primary signal. See the module docstring for the formula."""
    out = band_signal(masks, kpts_px, floor, start_frac, end_frac)
    return {
        "lumbar_gap": out["area"],
        "lumbar_gap_max": out["peak"],
        "window_px": out["width_px"],
    }


def geometric_signal(kpts: np.ndarray) -> np.ndarray:
    """Backbone-agnostic fallback: pelvis offset from the shoulder-knee line.

    Weak compared to the silhouette, but free, mask-independent, and it works on
    any backbone -- which makes it the cross-check for validity test V4. If the
    two signals are uncorrelated, one of them is measuring noise.

    Positive means the pelvis sits on the side the back arches toward. Returned
    in torso lengths.
    """
    k = np.asarray(kpts, dtype=np.float64)
    hip_c = 0.5 * (k[..., sk.L_HIP, :2] + k[..., sk.R_HIP, :2])
    sho_c = 0.5 * (k[..., sk.L_SHOULDER, :2] + k[..., sk.R_SHOULDER, :2])
    knee_c = 0.5 * (k[..., sk.L_KNEE, :2] + k[..., sk.R_KNEE, :2])

    axis = knee_c - sho_c
    length = np.linalg.norm(axis, axis=-1)
    rel = hip_c - sho_c
    cross = axis[..., 0] * rel[..., 1] - axis[..., 1] * rel[..., 0]

    torso = np.linalg.norm(sho_c - hip_c, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        distance = cross / np.where(length > EPS, length, np.nan)
        return distance / np.where(torso > EPS, torso, np.nan)


def perturb_mask(
    mask: np.ndarray,
    kpts_px_frame: np.ndarray,
    delta_px: float,
    start_frac: float = 0.00,
    end_frac: float = 0.35,
) -> np.ndarray:
    """Lift the silhouette's lumbar region by ``delta_px`` -- validity test V1.

    This is the synthetic arch: it injects a known deviation of known size, so
    the measured response can be checked against the closed-form geometric
    prediction. It is what lets instrument validity be established from
    correct-only footage, with no faulty reps at all.
    """
    out = np.asarray(mask).copy()
    if delta_px <= 0:
        return out

    hip_c, u, torso = torso_frame(np.asarray(kpts_px_frame, dtype=np.float64))
    x0, x1 = band_window(hip_c, u, float(torso), start_frac, end_frac, out.shape[1])
    if x1 <= x0:
        return out

    lift = int(round(delta_px))
    for x in range(x0, x1):
        column = np.flatnonzero(out[:, x] > 0)
        if column.size:
            out[max(0, column[-1] - lift + 1) : column[-1] + 1, x] = 0
    return out


def expected_v1_slope(window_px: int, torso_len_px: float) -> float:
    """Closed-form ``d(lumbar_gap) / d(delta_px)`` for a flat synthetic lift.

    Lifting ``n`` columns by ``delta`` adds exactly ``n * delta`` to the enclosed
    area, so after normalization the response is linear with slope
    ``n / torso_len**2``. V1 asserts the measured slope against this.
    """
    return window_px / (torso_len_px * torso_len_px)
