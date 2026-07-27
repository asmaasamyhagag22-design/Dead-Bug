"""Floor estimation and the lumbar gap.

**This file is validity tests V1 and V3.** The design docs frame both as
notebook figures, but they are assertions with closed-form expected values, so
they belong here where they run on every commit:

  V1  response to a synthetic perturbation -- lifting the silhouette's lumbar
      region by a known delta must produce a monotone response whose slope
      matches the geometric prediction. This is what establishes that the
      signal measures back height rather than some artifact, using no faulty
      footage at all.
  V3  invariance to scale and translation -- the same body filmed closer or
      shifted in frame must yield the same gap.
"""

from __future__ import annotations

import numpy as np
import pytest

from deadbug.geometry import floor as FL
from deadbug.signals import lumbar as LU

DELTAS = [0, 2, 4, 8, 16]


def _scene(make_mask, make_pose, geom, delta, scale=1.0):
    """Build ``(mask, kpts_px, floor)`` at a given scale, with a known lift."""
    mask = make_mask(
        delta_px=delta * scale,
        torso_len_px=geom["torso_len"] * scale,
        hip_cx=geom["hip_cx"] * scale,
        floor_y=geom["floor_y"] * scale,
        h=int(geom["mask_h"] * scale),
        w=int(geom["mask_w"] * scale),
        body_top=geom["body_top"] * scale,
    )
    return mask, make_pose(scale), {"a": 0.0, "b": geom["floor_y"] * scale - 1.0}


# --------------------------------------------------------------------------
# Floor
# --------------------------------------------------------------------------


def test_lower_boundary_finds_the_bottom_row():
    mask = np.zeros((10, 6), dtype=np.uint8)
    mask[3:8, 1:4] = 255
    bottom = FL.lower_boundary(mask)

    np.testing.assert_array_equal(bottom[1:4], 7.0)      # last occupied row
    assert np.isnan(bottom[0]) and np.isnan(bottom[4])   # empty columns


def test_estimate_floor_recovers_a_flat_floor(synth_mask_with_bump, scaled_pose, synth_geometry):
    masks = np.stack([synth_mask_with_bump(0.0) for _ in range(5)])
    result = FL.estimate_floor(masks, method="percentile")

    assert result["b"] == pytest.approx(synth_geometry["floor_y"] - 1, abs=1.0)
    assert result["a"] == 0.0
    assert result["inlier_ratio"] > 0.9


def test_ransac_recovers_a_tilted_floor():
    rng = np.random.default_rng(0)
    x = np.arange(0, 300, dtype=float)
    y = 0.15 * x + 120.0 + rng.normal(0, 0.5, x.size)
    x = np.concatenate([x, rng.uniform(0, 300, 40)])      # 12% gross outliers
    y = np.concatenate([y, rng.uniform(0, 100, 40)])

    a, b, ratio = FL.fit_floor_ransac(x, y, residual_threshold=3.0, seed=0)
    assert a == pytest.approx(0.15, abs=0.02)
    assert b == pytest.approx(120.0, abs=3.0)
    assert 0.8 < ratio < 1.0


def test_degenerate_floor_reports_zero_inliers():
    result = FL.estimate_floor(np.zeros((3, 40, 40), dtype=np.uint8), method="percentile")
    assert result["inlier_ratio"] == 0.0
    assert np.isnan(result["b"])


# --------------------------------------------------------------------------
# V1 -- response to a synthetic perturbation
# --------------------------------------------------------------------------


def test_v1_response_is_monotone_in_delta(synth_mask_with_bump, scaled_pose, synth_geometry):
    """Lift the lumbar region further, get a larger gap. Every time."""
    values = []
    for delta in DELTAS:
        mask, kpts, floor = _scene(synth_mask_with_bump, scaled_pose, synth_geometry, delta)
        values.append(LU.lumbar_gap(mask[None], kpts[None], floor)["lumbar_gap"][0])

    assert values[0] == pytest.approx(0.0, abs=1e-9), "a flat back must read zero"
    assert np.all(np.diff(values) > 0), f"not monotone: {values}"


def test_v1_slope_matches_the_geometric_prediction(synth_mask_with_bump, scaled_pose, synth_geometry):
    """The measured response must match the closed form, not merely rise.

    Lifting ``n`` columns by ``delta`` adds ``n * delta`` to the enclosed area,
    so after dividing by ``torso_len**2`` the slope is ``n / torso_len**2``.
    Monotonicity alone would also be satisfied by a signal responding to the
    wrong thing; the slope is what pins it to the geometry.
    """
    deltas = np.array(DELTAS, dtype=float)
    values, window_px = [], None
    for delta in deltas:
        mask, kpts, floor = _scene(synth_mask_with_bump, scaled_pose, synth_geometry, delta)
        out = LU.lumbar_gap(mask[None], kpts[None], floor)
        values.append(out["lumbar_gap"][0])
        window_px = int(out["window_px"][0])

    measured = np.polyfit(deltas, values, 1)[0]
    expected = LU.expected_v1_slope(window_px, synth_geometry["torso_len"])

    assert measured == pytest.approx(expected, rel=0.05), (
        f"measured {measured:.6f} vs geometric {expected:.6f}"
    )


def test_v1_peak_reads_in_torso_lengths(synth_mask_with_bump, scaled_pose, synth_geometry):
    """``lumbar_gap_max`` is the interpretable one: how high, in torso lengths."""
    mask, kpts, floor = _scene(synth_mask_with_bump, scaled_pose, synth_geometry, 16)
    peak = LU.lumbar_gap(mask[None], kpts[None], floor)["lumbar_gap_max"][0]
    assert peak == pytest.approx(16.0 / synth_geometry["torso_len"], rel=0.1)


# --------------------------------------------------------------------------
# V3 -- invariance to scale and translation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("delta", [4, 16])
def test_v3_scale_invariance(synth_mask_with_bump, scaled_pose, synth_geometry, delta):
    """The same arch filmed 1.5x closer must read the same."""
    m0, k0, f0 = _scene(synth_mask_with_bump, scaled_pose, synth_geometry, delta, scale=1.0)
    m1, k1, f1 = _scene(synth_mask_with_bump, scaled_pose, synth_geometry, delta, scale=1.5)

    a = LU.lumbar_gap(m0[None], k0[None], f0)["lumbar_gap"][0]
    b = LU.lumbar_gap(m1[None], k1[None], f1)["lumbar_gap"][0]
    assert b == pytest.approx(a, rel=0.10), f"{a:.6f} vs {b:.6f}"


def test_v3_translation_invariance(synth_mask_with_bump, scaled_pose, synth_geometry):
    """Shifting the subject within the frame must not change the gap."""
    mask, kpts, floor = _scene(synth_mask_with_bump, scaled_pose, synth_geometry, 8)
    shift = 60
    shifted_mask = np.roll(mask, shift, axis=1)
    shifted_kpts = kpts.copy()
    shifted_kpts[:, 0] += shift

    a = LU.lumbar_gap(mask[None], kpts[None], floor)["lumbar_gap"][0]
    b = LU.lumbar_gap(shifted_mask[None], shifted_kpts[None], floor)["lumbar_gap"][0]
    assert b == pytest.approx(a, rel=1e-6)


# --------------------------------------------------------------------------
# Windowing and the geometric cross-check
# --------------------------------------------------------------------------


def test_window_follows_the_direction_the_subject_faces(synth_geometry):
    """The sign of u handles left- and right-facing subjects with no branch."""
    hip = np.array([400.0, 300.0])
    torso = synth_geometry["torso_len"]

    right = LU.band_window(hip, np.array([1.0, 0.0]), torso, 0.0, 0.35, 640)
    left = LU.band_window(hip, np.array([-1.0, 0.0]), torso, 0.0, 0.35, 640)

    assert right == (400, 435)
    assert left == (365, 400)
    assert (right[1] - right[0]) == (left[1] - left[0])


def test_window_is_empty_for_a_degenerate_torso(synth_geometry):
    hip = np.array([400.0, 300.0])
    assert LU.band_window(hip, np.array([np.nan, np.nan]), 0.0, 0.0, 0.35, 640) == (0, 0)


def test_rib_band_sits_above_the_lumbar_band(synth_mask_with_bump, scaled_pose, synth_geometry):
    """Rib flare uses the same formula over [0.45, 0.75] -- non-overlapping."""
    mask, kpts, floor = _scene(synth_mask_with_bump, scaled_pose, synth_geometry, 8)
    lumbar = LU.band_signal(mask[None], kpts[None], floor, 0.00, 0.35)
    ribs = LU.band_signal(mask[None], kpts[None], floor, 0.45, 0.75)

    assert lumbar["area"][0] > 0
    assert ribs["area"][0] == pytest.approx(0.0, abs=1e-9), "the lift was lumbar-only"


def test_geometric_signal_is_zero_for_a_straight_body():
    """Shoulders, hips and knees collinear -> no pelvic offset."""
    k = np.zeros((1, 33, 4))
    from deadbug.pose import skeleton as sk

    for idx, x in ((sk.L_SHOULDER, 0.0), (sk.R_SHOULDER, 0.0),
                   (sk.L_HIP, 1.0), (sk.R_HIP, 1.0),
                   (sk.L_KNEE, 2.0), (sk.R_KNEE, 2.0)):
        k[0, idx, :2] = [x, 0.0]
    assert LU.geometric_signal(k)[0] == pytest.approx(0.0, abs=1e-12)


def test_geometric_signal_responds_to_a_pelvic_offset():
    from deadbug.pose import skeleton as sk

    k = np.zeros((1, 33, 4))
    for idx, xy in ((sk.L_SHOULDER, [0.0, 0.0]), (sk.R_SHOULDER, [0.0, 0.0]),
                    (sk.L_HIP, [1.0, -0.3]), (sk.R_HIP, [1.0, -0.3]),
                    (sk.L_KNEE, [2.0, 0.0]), (sk.R_KNEE, [2.0, 0.0])):
        k[0, idx, :2] = xy
    assert abs(LU.geometric_signal(k)[0]) > 0.1


def test_perturb_mask_creates_a_measurable_arch(synth_mask_with_bump, scaled_pose, synth_geometry):
    """V1's perturbation helper must agree with a directly-built bump."""
    flat, kpts, floor = _scene(synth_mask_with_bump, scaled_pose, synth_geometry, 0)
    lifted = LU.perturb_mask(flat, kpts, delta_px=10)

    before = LU.lumbar_gap(flat[None], kpts[None], floor)["lumbar_gap"][0]
    after = LU.lumbar_gap(lifted[None], kpts[None], floor)["lumbar_gap"][0]
    assert before == pytest.approx(0.0, abs=1e-9)
    assert after > before
