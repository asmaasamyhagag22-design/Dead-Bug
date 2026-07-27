"""Skeleton layout constants and pure-geometry helpers.

This module is a dependency-free leaf: numpy only, no OpenCV, no MediaPipe. That
is deliberate -- it makes the MediaPipe-33 -> COCO-17 mapping and the horizontal
flip testable in milliseconds without a video or an inference session.

Storage format for the whole project is **native MediaPipe-33**, normalized
[0, 1], shape ``(T, 33, 4)`` = x, y, z, visibility. COCO-17 is a *view* produced
by :func:`to_coco17`, used only by the feature builder and the backbone
comparison. Signals and geometry index mp33 directly, because the visibility
channel is a QC gate, ``z`` is needed for pelvis rotation, and neither survives
a COCO-17 projection.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# MediaPipe Pose Landmarker, 33 landmarks
# --------------------------------------------------------------------------

MP33_NAMES: list[str] = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Named indices the signal and geometry modules use directly.
L_SHOULDER, R_SHOULDER = 11, 12
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

#: Joints whose visibility the QC gate averages (shoulders + hips).
CORE_JOINTS_MP33 = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)

#: Contralateral pairs defining a rep: right wrist <-> left ankle, and mirror.
CONTRALATERAL_MP33 = {"R": (R_WRIST, L_ANKLE), "L": (L_WRIST, R_ANKLE)}

# Lifted verbatim from scripts/run_pose.py, which drew these correctly.
MP33_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

#: 16 left/right pairs; index 0 (nose) is unpaired. 1 + 2*16 == 33.
MP33_FLIP_PAIRS: list[tuple[int, int]] = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
    (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30),
    (31, 32),
]

# --------------------------------------------------------------------------
# COCO-17
# --------------------------------------------------------------------------

COCO17_NAMES: list[str] = [
    "nose",
    "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

#: ``coco[i] = mp33[MP33_TO_COCO17[i]]``.
MP33_TO_COCO17: np.ndarray = np.array(
    [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28],
    dtype=np.int64,
)

#: 8 left/right pairs; index 0 (nose) unpaired.
COCO17_FLIP_PAIRS: list[tuple[int, int]] = [
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
]

#: Kinematic tree rooted at the nose. ``COCO17_PARENTS[0] == 0`` marks the root.
#: The tree is flip-symmetric -- ``mirror(parent[c]) == parent[mirror(c)]`` for
#: every joint -- which is what lets bone and angle channels permute correctly
#: under a horizontal flip. ``test_skeleton.py`` asserts this.
COCO17_PARENTS: np.ndarray = np.array(
    [0, 0, 0, 1, 2, 0, 0, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14],
    dtype=np.int64,
)

#: 16 ``(child, parent)`` edges; bone ``k`` has child ``k + 1``.
COCO17_BONES: list[tuple[int, int]] = [
    (c, int(COCO17_PARENTS[c])) for c in range(1, 17)
]

#: Bone ``k`` mirrors bone ``k ^ 1`` because children mirror pairwise.
BONE_FLIP_PAIRS: list[tuple[int, int]] = [
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
]

#: ``(a, vertex, c)`` on COCO-17 indices; the angle is taken at ``vertex``.
#: Chosen to be strictly bilaterally symmetric in four mirror pairs so the
#: flip augmentation can permute the angle channels. Channels 4-7 double as
#: the ``hip_angle_start`` / ``knee_angle_start`` setup-validation fields.
ANGLE_TRIPLETS: list[tuple[int, int, int]] = [
    (6, 8, 10),    # 0 right elbow
    (5, 7, 9),     # 1 left  elbow
    (12, 6, 8),    # 2 right shoulder
    (11, 5, 7),    # 3 left  shoulder
    (6, 12, 14),   # 4 right hip
    (5, 11, 13),   # 5 left  hip
    (12, 14, 16),  # 6 right knee
    (11, 13, 15),  # 7 left  knee
]

#: Angles permute under a flip but do not change sign -- they are unsigned.
ANGLE_FLIP_PAIRS: list[tuple[int, int]] = [(0, 1), (2, 3), (4, 5), (6, 7)]

N_COCO17 = 17
N_BONES = 16
N_ANGLES = 8

#: joints 17*2 + bones 16*2 + velocity 17*2 + angles 8 + lumbar 1
FEATURE_DIM = N_COCO17 * 2 + N_BONES * 2 + N_COCO17 * 2 + N_ANGLES + 1

_FLIP_PAIRS = {"mp33": MP33_FLIP_PAIRS, "coco17": COCO17_FLIP_PAIRS}
_N_JOINTS = {"mp33": 33, "coco17": 17}


# --------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------


def to_coco17(kpts: np.ndarray, layout: str = "mp33") -> np.ndarray:
    """Project keypoints onto the COCO-17 layout.

    Idempotent for ``layout="coco17"``. Only the feature builder and the
    backbone comparison should call this; signals index mp33 directly.

    Args:
        kpts: ``(..., J, C)`` with ``J`` matching ``layout``.
        layout: ``"mp33"`` or ``"coco17"``.
    """
    if layout == "coco17":
        _check_joints(kpts, 17, layout)
        return kpts
    if layout != "mp33":
        raise ValueError(f"unknown layout: {layout!r}")
    _check_joints(kpts, 33, layout)
    return kpts[..., MP33_TO_COCO17, :]


def flip_horizontal(kpts: np.ndarray, width: float, layout: str = "mp33") -> np.ndarray:
    """Mirror keypoints about the vertical axis, in **pixel** coordinates.

    Two steps, and omitting the second is the most dangerous silent bug in the
    project: the model would learn contradictory left/right labels while the
    loss descended normally.

    Only valid in pixel space. After :func:`~deadbug.geometry.normalize.normalize`
    has rotated the torso onto +x, negating x would mirror the torso axis itself
    -- so flip first, then re-normalize.

    Args:
        kpts: ``(..., J, C)`` with C >= 1; column 0 is x.
        width: frame width in pixels (or 1.0 for normalized coordinates).
        layout: ``"mp33"`` or ``"coco17"``.
    """
    if layout not in _FLIP_PAIRS:
        raise ValueError(f"unknown layout: {layout!r}")
    _check_joints(kpts, _N_JOINTS[layout], layout)

    out = kpts.copy()
    out[..., 0] = width - 1 - out[..., 0]          # 1. mirror x
    for a, b in _FLIP_PAIRS[layout]:                # 2. swap left/right
        out[..., [a, b], :] = out[..., [b, a], :]
    return out


def bone_vectors(kpts17: np.ndarray) -> np.ndarray:
    """``(..., 17, C) -> (..., 16, C)`` child-minus-parent vectors."""
    _check_joints(kpts17, 17, "coco17")
    children = np.arange(1, 17)
    return kpts17[..., children, :] - kpts17[..., COCO17_PARENTS[children], :]


def joint_angles(kpts17: np.ndarray) -> np.ndarray:
    """``(..., 17, >=2) -> (..., 8)`` unsigned joint angles in radians.

    Uses ``arctan2`` of the cross and dot products rather than ``arccos`` of a
    normalized dot, which loses precision and can go NaN for near-collinear
    limbs -- exactly the straight-arm case this exercise spends most of its time in.
    """
    _check_joints(kpts17, 17, "coco17")
    xy = kpts17[..., :2]
    idx_a = [t[0] for t in ANGLE_TRIPLETS]
    idx_v = [t[1] for t in ANGLE_TRIPLETS]
    idx_c = [t[2] for t in ANGLE_TRIPLETS]

    v1 = xy[..., idx_a, :] - xy[..., idx_v, :]
    v2 = xy[..., idx_c, :] - xy[..., idx_v, :]
    cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    dot = (v1 * v2).sum(-1)
    return np.abs(np.arctan2(cross, dot))


def mirror_index(joint: int, layout: str = "coco17") -> int:
    """The left/right counterpart of ``joint``; unpaired joints map to themselves."""
    for a, b in _FLIP_PAIRS[layout]:
        if joint == a:
            return b
        if joint == b:
            return a
    return joint


def _check_joints(kpts: np.ndarray, expected: int, layout: str) -> None:
    if kpts.ndim < 2 or kpts.shape[-2] != expected:
        raise ValueError(
            f"layout {layout!r} expects {expected} joints, "
            f"got array of shape {kpts.shape}"
        )
