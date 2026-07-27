"""Floor-line estimation from the silhouette.

The original protocol assumed a three-second static hold at the start of each
recording, from which the floor could be read directly. YouTube clips have no
such hold, so estimation has to be calibration-free.

Two methods, in the order to try them:

1. **percentile** -- three lines, no dependencies, works on any clip. The floor
   is the 95th percentile of the silhouette's lowest points across the clip.
2. **ransac** -- fits a tilted line ``y = a*x + b``, which matters when the
   camera is not level. Hand-rolled rather than pulled from sklearn: the
   MediaPipe venv has no sklearn, and adding it to dodge twenty lines is a
   worse trade than writing them.

Both report an inlier ratio. Below 0.80 the clip is a QC **reject**, flagged
for manual review rather than silently carried forward.
"""

from __future__ import annotations

import numpy as np

MIN_POINTS = 8


def lower_boundary(mask: np.ndarray) -> np.ndarray:
    """``(H, W) -> (W,)`` the lowest silhouette pixel per column, NaN if empty.

    "Lowest" means largest row index -- image coordinates run downward, and the
    floor is at the bottom of the frame.
    """
    occupied = mask > 0
    any_col = occupied.any(axis=0)
    # argmax on the reversed rows finds the last True; convert back to a row index.
    last = mask.shape[0] - 1 - np.argmax(occupied[::-1, :], axis=0)
    return np.where(any_col, last.astype(np.float64), np.nan)


def static_frames(kpts: np.ndarray, threshold: float = 0.005) -> np.ndarray:
    """``(T,)`` bool -- frames where the subject is barely moving.

    Uses normalized-coordinate displacement between consecutive frames, so the
    threshold is resolution independent. When a clip has a setup hold, these are
    the frames to calibrate on.
    """
    xy = np.asarray(kpts, dtype=np.float64)[..., :2]
    disp = np.full(xy.shape[0], np.inf)
    delta = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
    with np.errstate(invalid="ignore"):
        disp[1:] = np.nanmean(delta, axis=-1)
    disp[0] = disp[1] if disp.size > 1 else 0.0
    return np.nan_to_num(disp, nan=np.inf) < threshold


def _collect_points(
    masks: np.ndarray, frame_indices: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Gather ``(x, bottom(x))`` samples from the chosen frames."""
    xs, ys = [], []
    indices = range(len(masks)) if frame_indices is None else frame_indices
    for t in indices:
        bottom = lower_boundary(masks[t])
        finite = np.flatnonzero(np.isfinite(bottom))
        if finite.size:
            xs.append(finite.astype(np.float64))
            ys.append(bottom[finite])
    if not xs:
        return np.empty(0), np.empty(0)
    return np.concatenate(xs), np.concatenate(ys)


def fit_floor_ransac(
    x: np.ndarray,
    y: np.ndarray,
    residual_threshold: float = 3.0,
    max_trials: int = 200,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Fit ``y = a*x + b`` robustly. Returns ``(a, b, inlier_ratio)``."""
    n = x.size
    if n < MIN_POINTS:
        return (float("nan"), float("nan"), 0.0)

    rng = np.random.default_rng(seed)
    best_a, best_b, best_inliers = 0.0, float(np.median(y)), None
    best_count = int(np.sum(np.abs(y - best_b) <= residual_threshold))

    for _ in range(max_trials):
        i, j = rng.integers(0, n, size=2)
        if x[i] == x[j]:
            continue
        a = (y[j] - y[i]) / (x[j] - x[i])
        b = y[i] - a * x[i]
        inliers = np.abs(y - (a * x + b)) <= residual_threshold
        count = int(inliers.sum())
        if count > best_count:
            best_a, best_b, best_count, best_inliers = a, b, count, inliers

    # Least-squares refit on the consensus set.
    if best_inliers is not None and best_inliers.sum() >= MIN_POINTS:
        coef = np.polyfit(x[best_inliers], y[best_inliers], 1)
        best_a, best_b = float(coef[0]), float(coef[1])
        best_count = int(np.sum(np.abs(y - (best_a * x + best_b)) <= residual_threshold))

    return (float(best_a), float(best_b), best_count / n)


def estimate_floor(
    masks: np.ndarray,
    kpts: np.ndarray | None = None,
    method: str = "percentile",
    percentile: float = 95.0,
    calib_static_frames: int = 30,
    static_disp_threshold: float = 0.005,
    calib_fallback_percentile: float = 20.0,
    ransac_residual_threshold_px: float = 3.0,
    ransac_max_trials: int = 200,
    seed: int = 0,
) -> dict:
    """Estimate the floor line for a clip.

    Returns ``{"a", "b", "inlier_ratio", "method", "n_points"}`` where the line
    is ``y = a*x + b`` in **mask pixel coordinates**.
    """
    masks = np.asarray(masks)
    if masks.ndim != 3:
        raise ValueError(f"expected masks (T, H, W), got {masks.shape}")

    calib = None
    if kpts is not None:
        still = static_frames(kpts, static_disp_threshold)
        if still.any():
            calib = np.flatnonzero(still)[:calib_static_frames]

    x, y = _collect_points(masks, calib)
    if x.size < MIN_POINTS:
        x, y = _collect_points(masks)          # no setup hold -- use everything

    # Keep only the points plausibly *on* the floor: the lowest slice of the
    # silhouette (largest y). This is not an optimisation, it is required for
    # correctness. In a Dead Bug the limbs are deliberately in the air, so most
    # columns never touch the ground; fitting to all of them drags the line up,
    # and measuring inliers over all of them makes `inlier_ratio` count "how
    # much of the body is on the floor" -- a quantity that is *anti*-correlated
    # with correct form. The 0.80 QC gate would then reject exactly the clips
    # it is meant to keep.
    if x.size >= MIN_POINTS:
        cutoff = np.percentile(y, 100.0 - calib_fallback_percentile)
        keep = y >= cutoff
        if keep.sum() >= MIN_POINTS:
            x, y = x[keep], y[keep]

    if x.size < MIN_POINTS:
        return {"a": float("nan"), "b": float("nan"), "inlier_ratio": 0.0,
                "method": method, "n_points": int(x.size)}

    if method == "ransac":
        a, b, ratio = fit_floor_ransac(
            x, y, ransac_residual_threshold_px, ransac_max_trials, seed
        )
    elif method == "percentile":
        a = 0.0
        b = float(np.percentile(y, percentile))
        ratio = float(np.mean(np.abs(y - b) <= ransac_residual_threshold_px))
    else:
        raise ValueError(f"unknown floor method: {method!r}")

    return {"a": a, "b": b, "inlier_ratio": ratio, "method": method,
            "n_points": int(x.size)}


def floor_y(a: float, b: float, x: np.ndarray) -> np.ndarray:
    """Evaluate the floor line at the given columns."""
    return a * np.asarray(x, dtype=np.float64) + b


def estimate_floor_from_config(masks: np.ndarray, kpts: np.ndarray, cfg: dict) -> dict:
    from ..config import cfg_get

    return estimate_floor(
        masks, kpts,
        method=cfg_get(cfg, "geometry.floor.method"),
        percentile=cfg_get(cfg, "geometry.floor.percentile"),
        calib_static_frames=cfg_get(cfg, "geometry.floor.calib_static_frames"),
        static_disp_threshold=cfg_get(cfg, "geometry.floor.static_disp_threshold_norm"),
        calib_fallback_percentile=cfg_get(cfg, "geometry.floor.calib_fallback_percentile"),
        ransac_residual_threshold_px=cfg_get(cfg, "geometry.floor.ransac_residual_threshold_px"),
        ransac_max_trials=cfg_get(cfg, "geometry.floor.ransac_max_trials"),
        seed=cfg_get(cfg, "seed"),
    )
