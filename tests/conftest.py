"""Synthetic fixtures for the whole suite.

Every fixture here is generated, not loaded. No test in this project needs a
video file, a MediaPipe session, or a network call -- which is what lets Gate 0
run to completion while the Track A dependencies are still downloading.
"""

from __future__ import annotations

import numpy as np
import pytest

from deadbug.pose import skeleton as sk

# Geometry of the synthetic supine subject, in pixels (image coords, y down).
HIP_CX, HIP_CY = 400.0, 300.0
TORSO_LEN = 100.0          # hip_center -> shoulder_center, pointing +x (head right)
HALF_HIP_SEP = 5.0         # view_score = 10 / 100 = 0.10, i.e. a clean side view
FLOOR_Y = 340.0


def _base_pose() -> np.ndarray:
    """One supine frame in the tabletop position, ``(33, 4)`` in pixels.

    Head to the right, arms and thighs vertical (toward the ceiling, so smaller
    y in the image), shins horizontal.
    """
    k = np.zeros((33, 4), dtype=np.float64)
    k[:, 3] = 1.0                                   # visibility
    sho_x = HIP_CX + TORSO_LEN

    def put(i: int, x: float, y: float) -> None:
        k[i, 0], k[i, 1] = x, y

    put(0, sho_x + 40, HIP_CY)                      # nose
    for i, dy in zip(range(1, 7), (-4, -5, -6, 4, 5, 6)):
        put(i, sho_x + 34, HIP_CY + dy)             # eyes
    put(7, sho_x + 25, HIP_CY - 5)                  # left ear
    put(8, sho_x + 25, HIP_CY + 5)                  # right ear
    put(9, sho_x + 35, HIP_CY - 2)                  # mouth left
    put(10, sho_x + 35, HIP_CY + 2)                 # mouth right

    put(sk.L_SHOULDER, sho_x, HIP_CY - 8)
    put(sk.R_SHOULDER, sho_x, HIP_CY + 8)
    put(13, sho_x, HIP_CY - 8 - 45)                 # left elbow
    put(14, sho_x, HIP_CY + 8 - 45)                 # right elbow
    put(sk.L_WRIST, sho_x, HIP_CY - 8 - 90)
    put(sk.R_WRIST, sho_x, HIP_CY + 8 - 90)
    for i, base in ((17, sk.L_WRIST), (19, sk.L_WRIST), (21, sk.L_WRIST)):
        put(i, k[base, 0] + 4, k[base, 1] - 6)      # left hand
    for i, base in ((18, sk.R_WRIST), (20, sk.R_WRIST), (22, sk.R_WRIST)):
        put(i, k[base, 0] + 4, k[base, 1] - 6)      # right hand

    put(sk.L_HIP, HIP_CX, HIP_CY - HALF_HIP_SEP)
    put(sk.R_HIP, HIP_CX, HIP_CY + HALF_HIP_SEP)
    put(sk.L_KNEE, HIP_CX, HIP_CY - HALF_HIP_SEP - 70)
    put(sk.R_KNEE, HIP_CX, HIP_CY + HALF_HIP_SEP - 70)
    put(sk.L_ANKLE, HIP_CX + 60, HIP_CY - HALF_HIP_SEP - 70)
    put(sk.R_ANKLE, HIP_CX + 60, HIP_CY + HALF_HIP_SEP - 70)
    for i, base in ((29, sk.L_ANKLE), (31, sk.L_ANKLE)):
        put(i, k[base, 0] + 8, k[base, 1] + 4)
    for i, base in ((30, sk.R_ANKLE), (32, sk.R_ANKLE)):
        put(i, k[base, 0] + 8, k[base, 1] + 4)
    return k


@pytest.fixture
def base_pose() -> np.ndarray:
    """A single supine frame, ``(33, 4)`` float64 pixels."""
    return _base_pose()


@pytest.fixture
def synth_geometry() -> dict:
    """The constants the synthetic pose and mask are both built from.

    The lumbar tests need these to compute the closed-form expected response,
    so they live in one place rather than being restated per test.
    """
    return {
        "hip_cx": HIP_CX,
        "hip_cy": HIP_CY,
        "torso_len": TORSO_LEN,
        "half_hip_sep": HALF_HIP_SEP,
        "floor_y": FLOOR_Y,
        "mask_h": 400,
        "mask_w": 640,
        "body_top": 250.0,
    }


@pytest.fixture
def scaled_pose():
    """Callable ``(scale) -> (33, 4)``: the base pose at a different camera distance."""

    def _make(scale: float = 1.0) -> np.ndarray:
        k = _base_pose()
        k[:, :2] *= scale
        return k

    return _make


@pytest.fixture
def synth_kpts33() -> np.ndarray:
    """``(T, 33, 4)`` of a supine subject performing alternating extensions.

    Right wrist and left ankle extend together, then the mirror pair, so the
    contralateral segmentation signals have a genuine alternating structure.
    """
    return synth_clip(n_reps=4, fps=30.0)


def synth_clip(n_reps: int = 4, fps: float = 30.0, rep_s: float = 2.0) -> np.ndarray:
    """Build a synthetic clip with ``n_reps`` single-side extensions."""
    per_rep = int(round(rep_s * fps))
    T = per_rep * n_reps
    out = np.repeat(_base_pose()[None, :, :], T, axis=0)

    # Raised-cosine excursion, one bump per rep, alternating sides.
    phase = np.arange(per_rep) / per_rep
    bump = 0.5 * (1.0 - np.cos(2.0 * np.pi * phase))

    for r in range(n_reps):
        sl = slice(r * per_rep, (r + 1) * per_rep)
        wrist, ankle = (
            (sk.R_WRIST, sk.L_ANKLE) if r % 2 == 0 else (sk.L_WRIST, sk.R_ANKLE)
        )
        elbow = 14 if wrist == sk.R_WRIST else 13
        knee = sk.L_KNEE if ankle == sk.L_ANKLE else sk.R_KNEE
        # Arm sweeps overhead (+x), leg extends away from the head (-x).
        out[sl, wrist, 0] += 70.0 * bump
        out[sl, wrist, 1] += 60.0 * bump
        out[sl, elbow, 0] += 35.0 * bump
        out[sl, elbow, 1] += 30.0 * bump
        out[sl, ankle, 0] -= 90.0 * bump
        out[sl, ankle, 1] += 55.0 * bump
        out[sl, knee, 1] += 40.0 * bump
    return out


@pytest.fixture
def synth_mask_with_bump():
    """Callable ``(delta_px, ...) -> (H, W) uint8`` binary silhouette.

    A supine body resting on a flat floor. The lumbar window is lifted by
    ``delta_px``, so the enclosed area -- and therefore ``lumbar_gap`` -- has a
    closed-form value the V1 monotonicity test can check its slope against:

        lumbar_gap = (window_width_px * delta_px) / torso_len_px ** 2
    """

    def _make(
        delta_px: float,
        torso_len_px: float = TORSO_LEN,
        hip_cx: float = HIP_CX,
        floor_y: float = FLOOR_Y,
        window: tuple[float, float] = (0.0, 0.35),
        h: int = 400,
        w: int = 640,
        body_top: float = 250.0,
    ) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        x0 = int(round(hip_cx - 0.2 * torso_len_px))
        x1 = int(round(hip_cx + 1.4 * torso_len_px))
        mask[int(body_top) : int(floor_y), x0:x1] = 255

        lo = int(round(hip_cx + window[0] * torso_len_px))
        hi = int(round(hip_cx + window[1] * torso_len_px))
        if delta_px > 0:
            mask[int(floor_y - delta_px) : int(floor_y), lo:hi] = 0
        return mask

    return _make


@pytest.fixture
def minjerk_speed():
    """Callable ``(n, ...) -> (n,)`` speed profile of a minimum-jerk reach.

    Position ``10t^3 - 15t^4 + 6t^5`` differentiates to ``30t^2(1-t)^2``. SPARC
    must score this above any noisier or multi-peaked profile of equal duration.
    """

    def _make(n: int = 128, amplitude: float = 1.0) -> np.ndarray:
        t = np.linspace(0.0, 1.0, n)
        return amplitude * 30.0 * t**2 * (1.0 - t) ** 2

    return _make


@pytest.fixture
def alternating_signal():
    """Callable ``(n_reps, fps) -> (sig_R, sig_L, boundaries)``.

    ``boundaries`` is the ground-truth list of ``(start, peak, end, side)`` the
    segmentation test compares against.
    """

    def _make(n_reps: int = 6, fps: float = 30.0, rep_s: float = 2.0):
        per_rep = int(round(rep_s * fps))
        T = per_rep * n_reps
        sig_r = np.full(T, 0.5)
        sig_l = np.full(T, 0.5)
        phase = np.arange(per_rep) / per_rep
        bump = 0.5 * (1.0 - np.cos(2.0 * np.pi * phase))

        boundaries = []
        for r in range(n_reps):
            start = r * per_rep
            sig = sig_r if r % 2 == 0 else sig_l
            sig[start : start + per_rep] += bump
            boundaries.append(
                (start, start + per_rep // 2, start + per_rep - 1, "R" if r % 2 == 0 else "L")
            )
        return sig_r, sig_l, boundaries

    return _make
