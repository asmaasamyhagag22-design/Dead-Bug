"""Schema rules, the feature stack, and the normative band.

The schema tests are all negative tests. Every one of them describes a table
that would produce a plausible-looking number rather than an error, which is
what makes them worth having: a duplicate subject silently inflates a LOSO
score, and nothing downstream would notice.

The feature test that matters most is
:func:`test_flip_features_matches_pixel_space_route`. It checks the mirror
permutation against the long way round -- flip the raw pixels, re-normalize,
re-stack -- because a wrong flip does not crash. It teaches the model
contradictory left/right labels while the loss descends normally.
"""

from __future__ import annotations

import numpy as np
import pytest

from deadbug.dataset import normative as NB
from deadbug.dataset.schema import (
    ClipRecord,
    RepRecord,
    SchemaError,
    assert_not_circular,
    read_clips,
    validate_clips,
    validate_reps,
)
from deadbug.geometry.normalize import normalize
from deadbug.modeling import features as FT
from deadbug.pose import skeleton as sk


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _clip(clip_id="c1", person="p1", condition="correct", **kwargs) -> ClipRecord:
    base = dict(clip_id=clip_id, file=f"data/clips/{clip_id}.mp4", person_id=person,
                condition=condition, view="side")
    base.update(kwargs)
    return ClipRecord(**base)


def _rep(clip: ClipRecord, index=0, side="R", **kwargs) -> RepRecord:
    base = dict(
        rep_id=f"{clip.clip_id}__{index:03d}{side}", clip_id=clip.clip_id,
        person_id=clip.person_id, condition=clip.condition, view=clip.view,
        side=side, rep_index=index, start_s=0.0, end_s=2.0, peak_s=1.0,
        duration_s=2.0, t_extend_s=1.0, t_return_s=1.0, dwell_s=0.2,
        overlap_s=0.5, excursion_peak=1.4,
    )
    base.update(kwargs)
    return RepRecord(**base)


@pytest.fixture
def supine_clip() -> np.ndarray:
    """A synthetic supine subject in pixel coordinates, 40 frames.

    Built rather than loaded so the flip test does not depend on a video file,
    and jittered so the flip check is not accidentally satisfied by symmetry.
    """
    rng = np.random.default_rng(0)
    n_frames = 40
    kpts = np.zeros((n_frames, 33, 4), dtype=np.float64)

    # Torso along +x, hips separated along y -- the supine geometry.
    layout = {
        sk.L_SHOULDER: (620.0, 240.0), sk.R_SHOULDER: (620.0, 300.0),
        sk.L_HIP: (400.0, 250.0), sk.R_HIP: (400.0, 290.0),
        sk.L_KNEE: (330.0, 200.0), sk.R_KNEE: (330.0, 340.0),
        sk.L_ANKLE: (270.0, 180.0), sk.R_ANKLE: (270.0, 360.0),
        sk.L_WRIST: (700.0, 190.0), sk.R_WRIST: (700.0, 350.0),
    }
    for joint in range(33):
        x, y = layout.get(joint, (500.0 + 3.0 * joint, 270.0 + 1.5 * joint))
        kpts[:, joint, 0] = x
        kpts[:, joint, 1] = y
    # A slow limb swing plus noise, so no two joints move together.
    t = np.linspace(0.0, 2.0 * np.pi, n_frames)
    kpts[:, sk.R_WRIST, 0] += 60.0 * np.sin(t)
    kpts[:, sk.L_ANKLE, 1] -= 40.0 * np.sin(t)
    kpts[:, :, :2] += rng.normal(0.0, 1.5, size=(n_frames, 33, 2))
    kpts[:, :, 3] = 0.99
    return kpts


# --------------------------------------------------------------------------
# clips.csv
# --------------------------------------------------------------------------


def test_valid_manifest_round_trips():
    clips = validate_clips([_clip("c1", "p1"), _clip("c2", "p2")])
    assert [c.clip_id for c in clips] == ["c1", "c2"]


def test_duplicate_clip_id_rejected():
    with pytest.raises(SchemaError, match="duplicate clip_id"):
        validate_clips([_clip("c1"), _clip("c1", person="p2")])


def test_blank_person_id_rejected():
    with pytest.raises(SchemaError, match="person_id is blank"):
        validate_clips([_clip(person="")])


def test_unknown_condition_rejected():
    with pytest.raises(SchemaError, match="condition"):
        validate_clips([_clip(condition="sloppy")])


def test_label_from_signal_rejected():
    """The circularity guard, at the manifest level."""
    with pytest.raises(SchemaError, match="circular"):
        validate_clips([_clip(label_source="lumbar_gap_threshold")])


def test_same_dedup_group_must_be_same_person():
    """clip.mp4 and videoplayback (4).mp4 are byte-identical -- one subject, not two."""
    with pytest.raises(SchemaError, match="dedup group"):
        validate_clips([
            _clip("clip", person="p1", dedup_group="G06"),
            _clip("videoplayback_4", person="p2", dedup_group="G06"),
        ])


def test_same_dedup_group_same_person_is_fine():
    clips = validate_clips([
        _clip("clip", person="p1", dedup_group="G06"),
        _clip("videoplayback_4", person="p1", dedup_group="G06"),
    ])
    assert len(clips) == 2


def test_unknown_column_rejected():
    with pytest.raises(SchemaError, match="unknown column"):
        ClipRecord.from_row({"clip_id": "c1", "file": "f", "person_id": "p",
                             "condition": "correct", "view": "side", "nonsense": "1"})


# --------------------------------------------------------------------------
# reps table
# --------------------------------------------------------------------------


def test_reps_must_agree_with_their_clip():
    clip = _clip()
    rogue = _rep(clip)
    rogue.condition = "arched"          # the clip says correct
    with pytest.raises(SchemaError, match="condition"):
        validate_reps([rogue], [clip])


def test_rep_pointing_at_unknown_clip_rejected():
    clip = _clip()
    orphan = _rep(clip)
    orphan.clip_id = "nope"
    with pytest.raises(SchemaError, match="not in the manifest"):
        validate_reps([orphan], [clip])


def test_zero_length_rep_rejected():
    clip = _clip()
    with pytest.raises(SchemaError, match="end_s"):
        validate_reps([_rep(clip, end_s=0.0)], [clip])


def test_lumbar_label_with_lumbar_feature_is_circular():
    with pytest.raises(SchemaError):
        assert_not_circular(["lumbar_gap_peak", "duration_s"], "lumbar_gap_peak")
    with pytest.raises(SchemaError):
        assert_not_circular(["duration_s"], "lumbar_gap_mean")
    assert_not_circular(["lumbar_gap_peak", "duration_s"], "condition") is None


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------


def test_feature_dim_is_109():
    assert FT.FEATURE_DIM == 109
    assert len(FT.channel_names()) == 109


def test_channel_slices_tile_the_vector():
    covered = sorted(
        i for s in FT.CHANNEL_SLICES.values() for i in range(s.start, s.stop)
    )
    assert covered == list(range(FT.FEATURE_DIM))


def test_stack_features_shape(supine_clip):
    kpts_norm, _ = normalize(supine_clip)
    lumbar = np.linspace(0.0, 0.01, supine_clip.shape[0])
    stacked = FT.stack_features(kpts_norm, lumbar, fps=30.0)
    assert stacked.shape == (supine_clip.shape[0], 109)
    assert np.allclose(stacked[:, FT.CHANNEL_SLICES["lumbar"]].ravel(), lumbar)


def test_flip_is_an_involution(supine_clip):
    kpts_norm, _ = normalize(supine_clip)
    stacked = FT.stack_features(kpts_norm, None, fps=30.0)
    assert np.allclose(FT.flip_features(FT.flip_features(stacked)), stacked)


def test_flip_features_matches_pixel_space_route(supine_clip):
    """The permutation must equal flip-in-pixels then re-normalize.

    This is the test the derivation in ``features._permutation`` is written
    against. If the sign ends up on x instead of y, this fails and nothing else
    does.
    """
    width = 1280.0
    lumbar = np.linspace(0.0, 0.01, supine_clip.shape[0])

    # Long way: mirror the raw pixels, then re-normalize and re-stack.
    flipped_px = sk.flip_horizontal(supine_clip, width, layout="mp33")
    flipped_norm, _ = normalize(flipped_px)
    expected = FT.stack_features(flipped_norm, lumbar, fps=30.0)

    # Short way: stack once, permute channels.
    kpts_norm, _ = normalize(supine_clip)
    actual = FT.flip_features(FT.stack_features(kpts_norm, lumbar, fps=30.0))

    assert np.allclose(actual, expected, atol=1e-9)


def test_rep_tensor_puts_reps_on_a_common_axis(supine_clip):
    kpts_norm, _ = normalize(supine_clip)
    stacked = FT.stack_features(kpts_norm, None, fps=30.0)
    short = FT.rep_tensor(stacked, 0, 9, n_out=32)
    long = FT.rep_tensor(stacked, 0, 30, n_out=32)
    assert short.shape == long.shape == (32, 109)


def test_rep_tensor_refuses_a_single_frame(supine_clip):
    kpts_norm, _ = normalize(supine_clip)
    stacked = FT.stack_features(kpts_norm, None, fps=30.0)
    with pytest.raises(ValueError, match="fewer than 2 frames"):
        FT.rep_tensor(stacked, 5, 5)


# --------------------------------------------------------------------------
# normative band
# --------------------------------------------------------------------------


def _band_inputs(n_per_person: int = 30, n_people: int = 4):
    rng = np.random.default_rng(1)
    excursion, signal, persons = [], [], []
    for p in range(n_people):
        e = rng.uniform(0.3, 1.6, n_per_person)
        excursion += list(e)
        # Correct reps: the gap grows mildly with excursion, plus per-person offset.
        signal += list(0.002 * e + 0.0005 * p + rng.normal(0, 1e-4, n_per_person))
        persons += [f"p{p}"] * n_per_person
    return np.array(excursion), np.array(signal), np.array(persons, dtype=object)


def test_band_bins_span_the_observed_excursion():
    exc, sig, people = _band_inputs()
    band = NB.fit_band(exc, sig, people, n_bins=10)
    assert len(band.mean) == 10
    assert band.edges[0] <= exc.min() and band.edges[-1] >= exc.max()
    assert band.n_persons == 4


def test_loso_band_excludes_the_held_out_subject():
    exc, sig, people = _band_inputs()
    bands = NB.band_loso(exc, sig, people, n_bins=10)
    assert set(bands) == {"p0", "p1", "p2", "p3"}
    for person, band in bands.items():
        assert band.held_out == person
        assert band.n_persons == 3          # fitted without them


def test_band_refuses_to_fit_on_nothing():
    with pytest.raises(ValueError, match="filming blocker"):
        NB.fit_band([0.5], [0.001], ["p0"])


def test_score_rep_flags_an_excessive_gap():
    exc, sig, people = _band_inputs()
    band = NB.fit_band(exc, sig, people, n_bins=10, sigma=2.0)
    bin_index = band.bin_of(1.0)
    huge = band.mean[bin_index] + 50 * max(band.std[bin_index], 1e-9)
    assert NB.score_rep(band, 1.0, huge)["exceeds"] is True
    assert NB.score_rep(band, 1.0, band.mean[bin_index])["exceeds"] is False


def test_score_rep_abstains_when_the_bin_has_no_spread():
    """An unsupported bin must abstain rather than pass everything."""
    band = NB.Band(
        edges=[0.0, 1.0], mean=[0.001], std=[float("nan")], p05=[0.0], p95=[0.0],
        count=[1], sigma=2.0, n_reps=1, n_persons=1,
    )
    verdict = NB.score_rep(band, 0.5, 0.5)
    assert verdict["exceeds"] is False
    assert "needs 2" in verdict["reason"]


def test_empty_bins_abstain_through_the_real_fit_band_path():
    """The abstention contract must hold for a band built by fit_band.

    Regression test for a defect the hand-built-Band test above could not see.
    ``fit_band`` interpolates the per-bin statistics across empty bins so the
    plotted curve is continuous, which leaves a bin with *zero* observations
    carrying a finite, entirely fabricated std. Keying abstention on "is the std
    NaN" therefore never fired: on the project's own data 13 of 20 bins were
    empty and all 20 returned a confident verdict.
    """
    # Deliberately clustered, so the middle of the range is unvisited.
    rng = np.random.default_rng(7)
    excursion, signal, persons = [], [], []
    for p in range(3):
        for centre in (0.30, 1.50):
            e = rng.normal(centre, 0.01, 8)
            excursion += list(e)
            signal += list(0.002 * e + rng.normal(0, 1e-5, 8))
            persons += [f"p{p}"] * 8

    band = NB.fit_band(excursion, signal, persons, n_bins=20)
    supported = NB.supported_mask(band)
    assert not supported.all(), "fixture failed to leave any bin empty"

    # Interpolation still produced a drawable curve...
    assert np.isfinite(band.std).all()
    # ...but every unsupported bin must refuse to decide.
    for b in np.flatnonzero(~supported):
        centre = 0.5 * (band.edges[b] + band.edges[b + 1])
        verdict = NB.score_rep(band, centre, 1e6)   # absurdly large signal
        assert verdict["exceeds"] is False, f"bin {b} issued a verdict on {verdict['support']} obs"
        assert not np.isfinite(verdict["z"])
        assert verdict["support"] < NB.MIN_BIN_SUPPORT

    # And a supported bin still works, or the guard has disabled everything.
    b = int(np.flatnonzero(supported)[0])
    centre = 0.5 * (band.edges[b] + band.edges[b + 1])
    assert NB.score_rep(band, centre, 1e6)["exceeds"] is True


def test_blank_label_source_is_not_silently_promoted_to_intent(tmp_path):
    """A blank provenance cell must not become the claim that intent was recorded."""
    header = "clip_id,file,person_id,condition,view,label_source,qc_status\n"
    path = tmp_path / "clips.csv"

    path.write_text(header + "c1,f.mp4,p1,correct,side,,ok\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="label_source"):
        read_clips(path)

    path.write_text(header + "c1,f.mp4,p1,correct,side,intent,\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="qc_status"):
        read_clips(path)

    # The fully-specified row still loads.
    path.write_text(header + "c1,f.mp4,p1,correct,side,intent,ok\n", encoding="utf-8")
    assert read_clips(path)[0].clip_id == "c1"


def test_qc_flag_gates_are_wired_to_keys_build_actually_produces():
    """Every flag gate must name a key build_clip emits.

    A gate reading a key nobody produces reports ':missing' on every clip and
    silently stops being a gate. Two of the four did exactly that.
    """
    from deadbug.qc.report import assert_gates_wired

    produced = {
        "clip_id", "n_frames", "fps", "frame_size", "detection_rate",
        "mean_visibility_core", "view_score", "torso_len_cv", "view",
        "activity", "mask_rate", "floor", "n_reps", "alternation_intact",
    }
    assert assert_gates_wired({k: None for k in produced}) == []


def test_qc_row_does_not_let_a_measurement_overwrite_the_verdict():
    from deadbug.qc.report import ClipQC

    qc = ClipQC(clip_id="c1", status="reject", measures={"status": "ok", "n_reps": 3})
    row = qc.as_row()
    assert row["status"] == "reject"
    assert row["measure_status"] == "ok"
    assert row["n_reps"] == 3


def test_band_round_trips_through_json(tmp_path):
    exc, sig, people = _band_inputs()
    bands = NB.band_loso(exc, sig, people, n_bins=8)
    path = NB.save_band(bands, tmp_path / "band.json")
    loaded = NB.load_band(path)
    assert set(loaded) == set(bands)
    assert loaded["p0"].mean == bands["p0"].mean
