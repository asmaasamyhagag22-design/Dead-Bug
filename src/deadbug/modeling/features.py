"""The D=109 per-frame feature vector, and its behaviour under a mirror flip.

    joints    17 * 2 = 34    normalized x, y
    bones     16 * 2 = 32    child minus parent
    velocity  17 * 2 = 34    first difference, per second
    angles          =  8     unsigned, radians
    lumbar          =  1     the silhouette signal
                      ---
                      109

Layout is COCO-17, not MediaPipe-33. The 16 extra MediaPipe landmarks are face
and hand detail that adds 32 channels of noise to an exercise that is about the
trunk. Signals still index mp33 directly -- ``z`` and ``visibility`` do not
survive the projection, and the rotation signal and the QC gate need them.

The reason this module owns :func:`flip_features` rather than leaving flipping
to the augmentation code: once features are stacked, a horizontal mirror is a
**channel permutation with a sign flip on x**, and getting it wrong is the
project's most dangerous silent bug. The left/right labels become contradictory,
the network learns the average of two mutually exclusive targets, and the loss
curve looks completely normal while it happens. Doing it in one place, next to
the layout that defines it, is what makes it testable in milliseconds.
"""

from __future__ import annotations

import numpy as np

from ..geometry.filters import resample_to_length
from ..pose import skeleton as sk

N_JOINT_CH = sk.N_COCO17 * 2      # 34
N_BONE_CH = sk.N_BONES * 2        # 32
N_VEL_CH = sk.N_COCO17 * 2        # 34
N_ANGLE_CH = sk.N_ANGLES          # 8
N_LUMBAR_CH = 1

#: Half-open channel ranges, in stack order.
CHANNEL_SLICES: dict[str, slice] = {
    "joints": slice(0, N_JOINT_CH),
    "bones": slice(N_JOINT_CH, N_JOINT_CH + N_BONE_CH),
    "velocity": slice(N_JOINT_CH + N_BONE_CH, N_JOINT_CH + N_BONE_CH + N_VEL_CH),
    "angles": slice(
        N_JOINT_CH + N_BONE_CH + N_VEL_CH,
        N_JOINT_CH + N_BONE_CH + N_VEL_CH + N_ANGLE_CH,
    ),
    "lumbar": slice(
        N_JOINT_CH + N_BONE_CH + N_VEL_CH + N_ANGLE_CH,
        sk.FEATURE_DIM,
    ),
}

FEATURE_DIM = sk.FEATURE_DIM      # 109


def channel_names() -> list[str]:
    """Human-readable name per channel, for feature-importance plots."""
    names: list[str] = []
    for joint in sk.COCO17_NAMES:
        names += [f"joint.{joint}.x", f"joint.{joint}.y"]
    for k, (child, parent) in enumerate(sk.COCO17_BONES):
        tag = f"{sk.COCO17_NAMES[child]}<-{sk.COCO17_NAMES[parent]}"
        names += [f"bone.{k}.{tag}.x", f"bone.{k}.{tag}.y"]
    for joint in sk.COCO17_NAMES:
        names += [f"vel.{joint}.x", f"vel.{joint}.y"]
    for k, (a, v, c) in enumerate(sk.ANGLE_TRIPLETS):
        names.append(f"angle.{k}.{sk.COCO17_NAMES[v]}")
    names.append("lumbar_gap")
    if len(names) != FEATURE_DIM:
        raise AssertionError(f"channel_names produced {len(names)}, expected {FEATURE_DIM}")
    return names


def stack_features(
    kpts_norm: np.ndarray,
    lumbar: np.ndarray | None = None,
    fps: float = 30.0,
    layout: str = "mp33",
) -> np.ndarray:
    """``(T, J, C) -> (T, 109)``.

    Args:
        kpts_norm: keypoints **after**
            :func:`deadbug.geometry.normalize.normalize`. Raw pixel input would
            make every channel depend on camera distance.
        lumbar: ``(T,)`` lumbar gap, or None to leave the channel at zero. None
            is the mask-free path -- a skeleton-only backbone has no silhouette,
            and the comparison against it is only fair if the channel is
            *absent* rather than filled with a plausible-looking constant.
        fps: needed for the velocity channels to be per-second rather than
            per-frame; clips here run at four different rates.
    """
    kpts17 = sk.to_coco17(np.asarray(kpts_norm, dtype=np.float64), layout=layout)
    n_frames = kpts17.shape[0]

    joints = kpts17[..., :2].reshape(n_frames, -1)
    bones = sk.bone_vectors(kpts17)[..., :2].reshape(n_frames, -1)

    # Velocity is padded at the front, not the back: the first frame has no
    # predecessor, and padding at the end would shift every rep's velocity one
    # frame earlier relative to its position channels.
    diff = np.diff(kpts17[..., :2], axis=0) * fps
    velocity = np.concatenate([np.zeros((1,) + diff.shape[1:]), diff], axis=0)
    velocity = velocity.reshape(n_frames, -1)

    angles = sk.joint_angles(kpts17)

    if lumbar is None:
        lumbar_ch = np.zeros((n_frames, 1))
    else:
        lumbar_ch = np.asarray(lumbar, dtype=np.float64).reshape(-1, 1)
        if lumbar_ch.shape[0] != n_frames:
            raise ValueError(
                f"lumbar has {lumbar_ch.shape[0]} frames, keypoints have {n_frames}"
            )

    out = np.concatenate([joints, bones, velocity, angles, lumbar_ch], axis=1)
    if out.shape[1] != FEATURE_DIM:
        raise AssertionError(f"built {out.shape[1]} channels, expected {FEATURE_DIM}")
    return out


def _permutation(n_items: int, pairs, base: int, order, sign, stride: int = 2) -> None:
    """Fill in the mirror mapping for one ``(x, y)``-interleaved block.

    ``flipped[:, i] = sign[i] * original[:, order[i]]``.

    **The sign goes on y, not x.** In pixel space a mirror negates x, but these
    features are built *after* :func:`~deadbug.geometry.normalize.normalize`,
    which has already rotated the torso onto +x. Negating x there would mirror
    the torso axis itself and leave the frame un-normalized. Working the
    composition through -- mirror, then re-normalize, which changes the torso
    angle from ``a`` to ``pi - a`` -- the two rotations cancel on x and compose
    to a negation on y:

        p'_norm.x = p_norm.x        p'_norm.y = -p_norm.y

    So in the normalized frame the sagittal mirror is a reflection about the
    torso axis. ``tests/test_schema.py`` asserts this against the pixel-space
    route rather than taking the derivation on trust.
    """
    for a, b in pairs:
        for axis in (0, 1):
            order[base + a * stride + axis] = base + b * stride + axis
            order[base + b * stride + axis] = base + a * stride + axis
    for j in range(n_items):
        sign[base + j * stride + 1] = -1.0


def _build_flip() -> tuple[np.ndarray, np.ndarray]:
    order = np.arange(FEATURE_DIM)
    sign = np.ones(FEATURE_DIM)

    _permutation(sk.N_COCO17, sk.COCO17_FLIP_PAIRS,
                 CHANNEL_SLICES["joints"].start, order, sign)
    _permutation(sk.N_BONES, sk.BONE_FLIP_PAIRS,
                 CHANNEL_SLICES["bones"].start, order, sign)
    _permutation(sk.N_COCO17, sk.COCO17_FLIP_PAIRS,
                 CHANNEL_SLICES["velocity"].start, order, sign)

    base = CHANNEL_SLICES["angles"].start
    for a, b in sk.ANGLE_FLIP_PAIRS:
        order[base + a] = base + b
        order[base + b] = base + a
    # Angles are unsigned; the lumbar channel is a scalar area. Neither flips.
    return order, sign


FLIP_ORDER, FLIP_SIGN = _build_flip()


def flip_features(x: np.ndarray) -> np.ndarray:
    """Mirror a stacked feature array in place of re-extracting a flipped clip.

    Equivalent to flipping the keypoints in pixel space and re-running
    :func:`stack_features` -- ``tests/test_schema.py`` asserts the two agree.
    Doing it here is far cheaper, which is what makes flip augmentation
    affordable at all.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape[-1] != FEATURE_DIM:
        raise ValueError(f"expected {FEATURE_DIM} channels, got {arr.shape[-1]}")
    return arr[..., FLIP_ORDER] * FLIP_SIGN


def rep_tensor(
    features: np.ndarray, start: int, end: int, n_out: int = 32
) -> np.ndarray:
    """Slice one rep out of a clip's features and put it on a fixed phase axis.

    ``n_out`` frames of *normalized time*, so reps of different durations are
    comparable. This throws the duration away, which is why the timing features
    live in the reps table as scalars and are never recovered from this tensor.
    """
    window = np.asarray(features)[start : end + 1]
    if window.shape[0] < 2:
        raise ValueError(f"rep window [{start}, {end}] has fewer than 2 frames")
    return resample_to_length(window, n_out)


def build_rep_tensors(
    features: np.ndarray, reps, n_out: int = 32
) -> tuple[np.ndarray, list]:
    """``(N, n_out, 109)`` for a clip's reps, plus the reps actually used.

    A rep too short to resample is dropped and reported rather than padded --
    a padded rep is a fabricated observation.
    """
    tensors, kept = [], []
    for rep in reps:
        try:
            tensors.append(rep_tensor(features, rep.start, rep.end, n_out))
            kept.append(rep)
        except ValueError:
            continue
    if not tensors:
        return np.zeros((0, n_out, FEATURE_DIM)), []
    return np.stack(tensors), kept


def summary_stats(tensor: np.ndarray) -> np.ndarray:
    """``(N, T, D) -> (N, D*5)`` mean, std, min, max, range per channel.

    The same reduction :mod:`deadbug.modeling.baselines` uses for
    ``rf_summary`` on Track A, kept here so Track B's tabular baseline is the
    identical transform and the two are comparable.
    """
    arr = np.asarray(tensor, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        parts = [
            np.nanmean(arr, axis=1), np.nanstd(arr, axis=1),
            np.nanmin(arr, axis=1), np.nanmax(arr, axis=1),
        ]
        parts.append(parts[3] - parts[2])
    return np.concatenate(parts, axis=1)


def features_from_config(kpts_norm: np.ndarray, lumbar, fps: float, cfg: dict) -> np.ndarray:
    from ..config import cfg_get

    dim = cfg_get(cfg, "features.dim")
    out = stack_features(kpts_norm, lumbar, fps, layout=cfg_get(cfg, "skeleton.storage_layout"))
    if out.shape[1] != dim:
        raise AssertionError(f"config says features.dim={dim}, builder produced {out.shape[1]}")
    return out
