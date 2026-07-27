"""Gate 0 -- normalization invariances.

If these fail, stop the pipeline. Every downstream signal assumes the skeleton
has been made independent of where the subject stands, how far the camera is,
and how the frame is oriented.
"""

from __future__ import annotations

import numpy as np
import pytest

from deadbug.geometry import normalize as G
from deadbug.pose import skeleton as sk

TOL_TRANSLATION = 1e-6
TOL_SCALE = 5e-2
TOL_ROTATION = 1e-3


# --------------------------------------------------------------------------
# The three invariances
# --------------------------------------------------------------------------


def test_translation_invariance(synth_kpts33):
    """Shifting the subject in frame must not change anything."""
    shifted = synth_kpts33.copy()
    shifted[..., 0] += 120.0
    shifted[..., 1] += -80.0

    a, _ = G.normalize(synth_kpts33)
    b, _ = G.normalize(shifted)
    np.testing.assert_allclose(a[..., :2], b[..., :2], atol=TOL_TRANSLATION)


def test_scale_invariance(synth_kpts33):
    """Camera distance must not change anything -- scaling about *any* origin."""
    origin = np.array([73.0, -19.0])
    scaled = synth_kpts33.copy()
    scaled[..., :2] = (scaled[..., :2] - origin) * 1.7 + origin

    a, _ = G.normalize(synth_kpts33)
    b, _ = G.normalize(scaled)
    np.testing.assert_allclose(a[..., :2], b[..., :2], atol=TOL_SCALE)


def test_rotation_alignment(synth_kpts33):
    """After normalization the torso axis must lie along +x."""
    out, _ = G.normalize(synth_kpts33)
    torso = G.shoulder_center(out) - G.hip_center(out)

    assert np.all(torso[:, 0] > 0), "torso points along -x; the sign convention is wrong"
    np.testing.assert_allclose(torso[:, 1], 0.0, atol=TOL_ROTATION)


def test_rotation_invariance(synth_kpts33):
    """A rotated camera must produce the same normalized skeleton."""
    theta = np.deg2rad(37.0)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    turned = synth_kpts33.copy()
    turned[..., :2] = turned[..., :2] @ rot.T

    a, _ = G.normalize(synth_kpts33)
    b, _ = G.normalize(turned)
    np.testing.assert_allclose(a[..., :2], b[..., :2], atol=TOL_SCALE)


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------


def test_normalize_preserves_shape_and_passthrough_channels(synth_kpts33):
    out, angle = G.normalize(synth_kpts33)
    assert out.shape == synth_kpts33.shape
    assert angle.shape == (synth_kpts33.shape[0],)
    # z and visibility are carried through untouched -- rotation.py reads raw z,
    # and the QC gate reads raw visibility.
    np.testing.assert_array_equal(out[..., 2:], synth_kpts33[..., 2:])


def test_normalize_puts_the_hip_centre_at_the_origin(synth_kpts33):
    out, _ = G.normalize(synth_kpts33)
    np.testing.assert_allclose(G.hip_center(out), 0.0, atol=TOL_TRANSLATION)


def test_torso_scale_is_about_one(synth_kpts33):
    """Scaling by the clip median puts the torso near unit length."""
    out, _ = G.normalize(synth_kpts33)
    torso = np.linalg.norm(G.shoulder_center(out) - G.hip_center(out), axis=-1)
    assert np.median(torso) == pytest.approx(1.0, abs=1e-9)


def test_scale_uses_the_clip_median_not_the_frame(synth_kpts33):
    """A few bad frames must not move the reference length.

    Per-frame scaling injects hip and shoulder jitter into every signal, which
    is why the config pins this to the clip median.
    """
    corrupted = synth_kpts33.copy()
    corrupted[:3, sk.L_SHOULDER, 0] += 900.0     # 3 wild frames out of many
    corrupted[:3, sk.R_SHOULDER, 0] += 900.0

    assert G.torso_len_ref(corrupted) == pytest.approx(
        G.torso_len_ref(synth_kpts33), rel=1e-9
    )


def test_normalize_tolerates_nan_frames(synth_kpts33):
    holed = synth_kpts33.copy()
    holed[5:9] = np.nan
    out, angle = G.normalize(holed)
    assert np.isnan(out[5:9, :, :2]).all()
    assert np.isfinite(out[10:, :, :2]).all()
    assert np.isnan(angle[5:9]).all()


def test_rotation_angle_is_recorded(synth_kpts33):
    """The per-frame torso angle is stored because its variance is a feature."""
    theta = np.deg2rad(20.0)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    turned = synth_kpts33.copy()
    turned[..., :2] = turned[..., :2] @ rot.T

    _, a0 = G.normalize(synth_kpts33)
    _, a1 = G.normalize(turned)
    np.testing.assert_allclose(a1 - a0, theta, atol=1e-9)


# --------------------------------------------------------------------------
# view_score
# --------------------------------------------------------------------------


def test_view_score_on_a_clean_side_view(base_pose):
    """The synthetic pose has hip_sep 10 px over a 100 px torso."""
    vs = G.view_score(base_pose[None, ...])
    assert vs[0] == pytest.approx(0.10, abs=1e-9)
    assert vs[0] < 0.12, "should qualify as a side view"


def test_view_score_must_be_computed_in_pixel_space():
    """Normalized [0,1] coords have different x and y pixel scales, so the
    hip-sep/torso ratio is distorted by the aspect ratio. Measured on a real
    clip: 0.240 normalized vs 0.131 in pixels -- enough to flip the verdict."""
    w, h = 1280.0, 720.0
    k = np.zeros((1, 33, 4))
    k[:, :, 3] = 1.0
    # Torso runs along x, hips separate along y -- the worst case for the
    # aspect-ratio error, and exactly the supine geometry we care about.
    k[0, sk.L_HIP] = [0.40, 0.50 - 20.0 / h, 0, 1]
    k[0, sk.R_HIP] = [0.40, 0.50 + 20.0 / h, 0, 1]
    k[0, sk.L_SHOULDER] = [0.40 + 200.0 / w, 0.50, 0, 1]
    k[0, sk.R_SHOULDER] = [0.40 + 200.0 / w, 0.50, 0, 1]

    in_pixels = G.view_score(k, frame_size=(w, h))
    naive = G.view_score(k)
    assert in_pixels[0] == pytest.approx(40.0 / 200.0, abs=1e-9)
    assert not np.isclose(naive[0], in_pixels[0], atol=1e-2)


def test_view_score_is_scale_invariant(base_pose):
    big = base_pose.copy()
    big[..., :2] *= 3.3
    np.testing.assert_allclose(
        G.view_score(big[None, ...]), G.view_score(base_pose[None, ...]), atol=1e-9
    )
