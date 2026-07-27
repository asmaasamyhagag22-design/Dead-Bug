"""SPARC — spectral arc length.

The defining property: a pure minimum-jerk profile is the smoothness ceiling.
Nothing may score above it, and anything jerkier must score clearly below.
"""

from __future__ import annotations

import numpy as np
import pytest

from deadbug.signals import smoothness as S

FPS = 30.0


def test_minimum_jerk_is_the_smoothest(minjerk_speed):
    """The reference profile must beat every perturbation of it."""
    smooth = S.sparc(minjerk_speed(128), FPS)
    assert np.isfinite(smooth)

    rng = np.random.default_rng(0)
    noisy = S.sparc(minjerk_speed(128) + np.abs(rng.normal(0, 0.3, 128)), FPS)
    # Two-peaked: a rep that stalls and restarts.
    t = np.linspace(0, 1, 128)
    stuttered = S.sparc(
        30 * t**2 * (1 - t) ** 2 * (1 + 0.8 * np.sin(12 * np.pi * t)), FPS
    )

    assert smooth > noisy, f"noisy scored smoother: {smooth:.3f} vs {noisy:.3f}"
    assert smooth > stuttered, f"stuttered scored smoother: {smooth:.3f} vs {stuttered:.3f}"


def test_sparc_is_negative(minjerk_speed):
    assert S.sparc(minjerk_speed(128), FPS) < 0


def test_sparc_is_amplitude_invariant(minjerk_speed):
    """Reaching further is not the same as reaching more smoothly."""
    a = S.sparc(minjerk_speed(128, amplitude=1.0), FPS)
    b = S.sparc(minjerk_speed(128, amplitude=7.5), FPS)
    assert a == pytest.approx(b, rel=1e-9)


def test_sparc_is_broadly_insensitive_to_duration(minjerk_speed):
    """The whole reason for choosing SPARC over LDLJ.

    A time-domain jerk metric is dominated by duration, which would make
    "jerky" collinear with the separate "moving too fast" feature. SPARC must
    rank the same shape similarly whether it took 2 s or 6 s.
    """
    short = S.sparc(minjerk_speed(60), FPS)
    long = S.sparc(minjerk_speed(180), FPS)
    assert abs(short - long) < 0.6 * abs(short)


def test_degenerate_inputs_return_nan():
    assert np.isnan(S.sparc(np.array([1.0, 2.0]), FPS))
    assert np.isnan(S.sparc(np.zeros(128), FPS))
    assert np.isnan(S.sparc(np.full(128, np.nan), FPS))


def test_speed_profile_matches_a_known_velocity():
    """Constant 0.01 units/frame at 30 fps is 0.3 units/second."""
    kpts = np.zeros((10, 33, 4))
    kpts[:, 15, 0] = np.arange(10) * 0.01
    speed = S.speed_profile(kpts, 15, FPS)

    assert speed.shape == (9,)
    np.testing.assert_allclose(speed, 0.3, atol=1e-12)


def test_sparc_per_rep_reports_wrist_and_ankle_separately(synth_kpts33):
    """An arm can stay controlled while the leg drops -- averaging hides that."""
    from deadbug.segment.reps import segment_reps

    reps, _ = segment_reps(synth_kpts33, FPS)
    assert reps, "no reps to measure"

    out = S.sparc_per_rep(synth_kpts33, reps[0], FPS)
    assert set(out) == {"sparc_wrist", "sparc_ankle"}
    assert all(np.isfinite(v) or np.isnan(v) for v in out.values())


def test_minimum_jerk_helper_peaks_in_the_middle():
    speed = S.minimum_jerk_speed(101)
    assert int(np.argmax(speed)) == 50
    assert speed[0] == pytest.approx(0.0) and speed[-1] == pytest.approx(0.0)
