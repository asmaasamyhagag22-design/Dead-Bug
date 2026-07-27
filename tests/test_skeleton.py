"""Gate 0 -- skeleton layout invariants.

These guard the single most dangerous silent bug in the project: a horizontal
flip that mirrors x without swapping left/right joint indices. The loss would
descend normally while the model learned contradictory labels.
"""

from __future__ import annotations

import numpy as np
import pytest

from deadbug.pose import skeleton as sk


# --------------------------------------------------------------------------
# Layout bookkeeping
# --------------------------------------------------------------------------


def test_layout_sizes_are_consistent():
    assert len(sk.MP33_NAMES) == 33
    assert len(sk.COCO17_NAMES) == 17
    # Every joint is either the unpaired root or a member of exactly one pair.
    assert 1 + 2 * len(sk.MP33_FLIP_PAIRS) == 33
    assert 1 + 2 * len(sk.COCO17_FLIP_PAIRS) == 17
    assert len(sk.COCO17_BONES) == sk.N_BONES == 16
    assert len(sk.ANGLE_TRIPLETS) == sk.N_ANGLES == 8


def test_feature_dim_is_109():
    """joints 34 + bones 32 + velocity 34 + angles 8 + lumbar 1."""
    assert sk.FEATURE_DIM == 17 * 2 + 16 * 2 + 17 * 2 + 8 + 1 == 109


@pytest.mark.parametrize("layout", ["mp33", "coco17"])
def test_flip_pairs_are_disjoint(layout):
    pairs = {"mp33": sk.MP33_FLIP_PAIRS, "coco17": sk.COCO17_FLIP_PAIRS}[layout]
    seen = [i for pair in pairs for i in pair]
    assert len(seen) == len(set(seen)), "a joint appears in two flip pairs"


def test_flip_pairs_match_left_right_names():
    """The index pairs must agree with the human-readable names."""
    for names, pairs in (
        (sk.MP33_NAMES, sk.MP33_FLIP_PAIRS),
        (sk.COCO17_NAMES, sk.COCO17_FLIP_PAIRS),
    ):
        for a, b in pairs:
            assert names[a].replace("left", "@") == names[b].replace("right", "@"), (
                f"{names[a]!r} is not the mirror of {names[b]!r}"
            )


def test_kinematic_tree_is_flip_symmetric():
    """``mirror(parent[c]) == parent[mirror(c)]`` for every joint.

    This is what guarantees the bone and angle channels permute consistently
    under a flip -- the second instance of the left/right bug, one layer deeper
    than the joints themselves, where nobody looks.
    """
    for c in range(sk.N_COCO17):
        parent = int(sk.COCO17_PARENTS[c])
        assert sk.mirror_index(parent) == int(sk.COCO17_PARENTS[sk.mirror_index(c)]), (
            f"tree is not flip-symmetric at joint {c} ({sk.COCO17_NAMES[c]})"
        )


def test_bone_flip_pairs_follow_from_the_tree():
    """Bone ``k`` has child ``k+1``, so bones mirror exactly as children do."""
    derived = set()
    for k in range(sk.N_BONES):
        child = k + 1
        derived.add(tuple(sorted((k, sk.mirror_index(child) - 1))))
    derived.discard(())
    assert derived == {tuple(sorted(p)) for p in sk.BONE_FLIP_PAIRS}


# --------------------------------------------------------------------------
# to_coco17
# --------------------------------------------------------------------------


def test_to_coco17_maps_the_joints_we_actually_use(base_pose):
    coco = sk.to_coco17(base_pose)
    assert coco.shape == (17, 4)
    for coco_idx, mp_idx in (
        (0, 0),                     # nose
        (5, sk.L_SHOULDER),
        (6, sk.R_SHOULDER),
        (9, sk.L_WRIST),
        (10, sk.R_WRIST),
        (11, sk.L_HIP),
        (12, sk.R_HIP),
        (15, sk.L_ANKLE),
        (16, sk.R_ANKLE),
    ):
        np.testing.assert_array_equal(coco[coco_idx], base_pose[mp_idx])


def test_to_coco17_names_line_up():
    for i, src in enumerate(sk.MP33_TO_COCO17):
        assert sk.MP33_NAMES[src] == sk.COCO17_NAMES[i]


def test_to_coco17_is_idempotent(base_pose):
    coco = sk.to_coco17(base_pose)
    np.testing.assert_array_equal(sk.to_coco17(coco, layout="coco17"), coco)


def test_to_coco17_rejects_the_wrong_joint_count(base_pose):
    with pytest.raises(ValueError, match="33 joints"):
        sk.to_coco17(base_pose[:17])


def test_to_coco17_preserves_leading_dims(synth_kpts33):
    assert sk.to_coco17(synth_kpts33).shape == (synth_kpts33.shape[0], 17, 4)


# --------------------------------------------------------------------------
# flip_horizontal -- Gate 0
# --------------------------------------------------------------------------


@pytest.mark.parametrize("layout", ["mp33", "coco17"])
def test_flip_is_involutive(base_pose, layout):
    """flip(flip(k)) == k, exactly. One of the four Gate 0 tests."""
    k = base_pose if layout == "mp33" else sk.to_coco17(base_pose)
    once = sk.flip_horizontal(k, width=640, layout=layout)
    twice = sk.flip_horizontal(once, width=640, layout=layout)
    np.testing.assert_array_equal(twice, k)


def test_flip_actually_swaps_left_and_right(base_pose):
    """A flip that mirrors x but forgets step 2 would pass an x-only check."""
    flipped = sk.flip_horizontal(base_pose, width=640)
    # y and visibility of the left shoulder must now hold the right shoulder's.
    np.testing.assert_array_equal(
        flipped[sk.L_SHOULDER, 1:], base_pose[sk.R_SHOULDER, 1:]
    )
    np.testing.assert_array_equal(flipped[sk.L_HIP, 1:], base_pose[sk.R_HIP, 1:])
    assert flipped[sk.L_SHOULDER, 0] == 640 - 1 - base_pose[sk.R_SHOULDER, 0]


def test_flip_preserves_inter_joint_distances(base_pose):
    """A mirror is an isometry -- but between *corresponding* joints.

    After the flip, slot ``a`` holds what used to be at ``mirror(a)``, so the
    distance to compare against is the one between the mirrored indices. Any
    index bug breaks this; asserting against the unmirrored pair instead would
    only hold for a bilaterally symmetric pose and is not a real invariant.
    """
    flipped = sk.flip_horizontal(base_pose, width=640)
    for a, b in ((sk.L_HIP, sk.R_HIP), (sk.L_SHOULDER, sk.L_WRIST), (0, sk.R_ANKLE)):
        ma, mb = sk.mirror_index(a, "mp33"), sk.mirror_index(b, "mp33")
        d0 = np.linalg.norm(base_pose[ma, :2] - base_pose[mb, :2])
        d1 = np.linalg.norm(flipped[a, :2] - flipped[b, :2])
        assert d1 == pytest.approx(d0, abs=1e-9)


# --------------------------------------------------------------------------
# Derived channels
# --------------------------------------------------------------------------


def test_bone_vectors_are_child_minus_parent(base_pose):
    coco = sk.to_coco17(base_pose)
    bones = sk.bone_vectors(coco)
    assert bones.shape == (16, 4)
    for k, (child, parent) in enumerate(sk.COCO17_BONES):
        np.testing.assert_allclose(bones[k], coco[child] - coco[parent])


def test_joint_angles_match_hand_computed_geometry(base_pose):
    """The synthetic pose is built with a right angle at the knee and a
    straight arm, so two of the eight channels have known values."""
    angles = sk.joint_angles(sk.to_coco17(base_pose))
    assert angles.shape == (8,)
    assert angles[6] == pytest.approx(np.pi / 2, abs=1e-9)   # right knee, tabletop
    assert angles[7] == pytest.approx(np.pi / 2, abs=1e-9)   # left knee
    assert angles[0] == pytest.approx(np.pi, abs=1e-9)       # right elbow, straight
    assert angles[1] == pytest.approx(np.pi, abs=1e-9)       # left elbow, straight


def test_joint_angles_are_in_range(synth_kpts33):
    angles = sk.joint_angles(sk.to_coco17(synth_kpts33))
    assert angles.shape == (synth_kpts33.shape[0], 8)
    assert np.all(angles >= 0.0) and np.all(angles <= np.pi + 1e-12)
    assert np.isfinite(angles).all()


def test_angles_permute_under_flip_without_changing_sign(synth_kpts33):
    """Unsigned angles are mirror-invariant in magnitude, so a flip must
    permute the eight channels by ANGLE_FLIP_PAIRS and nothing more."""
    coco = sk.to_coco17(synth_kpts33)
    flipped = sk.flip_horizontal(coco, width=640, layout="coco17")

    a0 = sk.joint_angles(coco)
    a1 = sk.joint_angles(flipped)
    for i, j in sk.ANGLE_FLIP_PAIRS:
        np.testing.assert_allclose(a1[..., i], a0[..., j], atol=1e-9)
        np.testing.assert_allclose(a1[..., j], a0[..., i], atol=1e-9)


def test_bones_permute_under_flip_with_x_negated(synth_kpts33):
    coco = sk.to_coco17(synth_kpts33)[..., :2]
    flipped = sk.flip_horizontal(coco, width=640, layout="coco17")

    b0 = sk.bone_vectors(coco)
    b1 = sk.bone_vectors(flipped)
    for i, j in sk.BONE_FLIP_PAIRS:
        np.testing.assert_allclose(b1[..., i, 0], -b0[..., j, 0], atol=1e-9)
        np.testing.assert_allclose(b1[..., i, 1], b0[..., j, 1], atol=1e-9)
