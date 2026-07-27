"""One Euro filter, gap interpolation and resampling."""

from __future__ import annotations

import numpy as np
import pytest

from deadbug.geometry import filters as F

FPS = 30.0


# --------------------------------------------------------------------------
# One Euro
# --------------------------------------------------------------------------


def test_constant_signal_passes_through():
    x = np.full(50, 7.5)
    np.testing.assert_allclose(F.one_euro_array(x, FPS), 7.5, atol=1e-9)


def test_step_response_settles_without_overshoot():
    """A low-pass must approach the step monotonically and never exceed it."""
    x = np.concatenate([np.zeros(30), np.ones(120)])
    y = F.one_euro_array(x, FPS)

    assert np.all(y <= 1.0 + 1e-12), "overshoot: this is not a low-pass"
    assert np.all(np.diff(y[30:]) >= -1e-12), "non-monotone settling"
    assert y[35] < 1.0, "no smoothing applied at all"
    assert y[-1] == pytest.approx(1.0, abs=1e-2), "failed to settle"


def test_filter_reduces_noise():
    """On a stationary signal the filter must cut the jitter substantially.

    Measured against a *slow* reference on purpose: comparing a filtered fast
    sine to its clean original measures the filter's phase lag, not its noise
    rejection, and lag is the thing the peak-preservation test below bounds.
    """
    rng = np.random.default_rng(0)
    noisy = rng.normal(0.0, 0.05, 400)
    filtered = F.one_euro_array(noisy, FPS)

    assert np.std(filtered) < 0.5 * np.std(noisy)


def test_filter_preserves_peak_location():
    """The reason for One Euro over a moving average: peaks must not migrate.

    Rep segmentation finds peaks in the contralateral distance signal, so a
    filter that shifts them would corrupt every rep boundary downstream.
    """
    t = np.linspace(0, 1, 120)
    clean = np.exp(-(((t - 0.5) / 0.08) ** 2))
    rng = np.random.default_rng(1)
    filtered = F.one_euro_array(clean + rng.normal(0, 0.02, t.size), FPS)

    assert abs(int(np.argmax(filtered)) - int(np.argmax(clean))) <= 3


def test_nan_does_not_smear():
    x = np.arange(20, dtype=float)
    x[10] = np.nan
    y = F.one_euro_array(x, FPS)

    assert np.isnan(y[10])
    assert np.isfinite(y[:10]).all() and np.isfinite(y[11:]).all()


def test_filter_handles_multichannel_arrays():
    x = np.zeros((40, 33, 4))
    x[:, :, 0] = np.linspace(0, 1, 40)[:, None]
    y = F.one_euro_array(x, FPS)
    assert y.shape == x.shape


# --------------------------------------------------------------------------
# Gap interpolation
# --------------------------------------------------------------------------


def test_short_gaps_are_filled_linearly():
    k = np.zeros((10, 33, 4))
    k[:, :, 0] = np.arange(10)[:, None]
    k[4:7, 5, 0] = np.nan          # a 3-frame gap, under the limit

    filled, valid = F.interpolate_gaps(k, max_gap=5)
    np.testing.assert_allclose(filled[4:7, 5, 0], [4.0, 5.0, 6.0], atol=1e-9)
    assert valid.all()


def test_long_gaps_stay_nan_and_invalidate_the_frame():
    k = np.zeros((20, 33, 4))
    k[:, :, 0] = np.arange(20)[:, None]
    k[5:14, 5, 0] = np.nan         # a 9-frame gap, over the limit

    filled, valid = F.interpolate_gaps(k, max_gap=5)
    assert np.isnan(filled[5:14, 5, 0]).all()
    assert not valid[5:14].any()
    assert valid[:5].all() and valid[14:].all()


def test_leading_and_trailing_gaps_are_never_extrapolated():
    k = np.zeros((12, 33, 4))
    k[:, :, 0] = np.arange(12)[:, None]
    k[:2, 5, 0] = np.nan
    k[-2:, 5, 0] = np.nan

    filled, valid = F.interpolate_gaps(k, max_gap=5)
    assert np.isnan(filled[:2, 5, 0]).all()
    assert np.isnan(filled[-2:, 5, 0]).all()
    assert not valid[:2].any() and not valid[-2:].any()


def test_gap_at_exactly_the_limit_is_filled():
    k = np.zeros((15, 33, 4))
    k[:, :, 0] = np.arange(15)[:, None]
    k[5:10, 5, 0] = np.nan         # exactly 5

    filled, _ = F.interpolate_gaps(k, max_gap=5)
    assert np.isfinite(filled[5:10, 5, 0]).all()


def test_interpolation_leaves_clean_data_untouched(synth_kpts33):
    filled, valid = F.interpolate_gaps(synth_kpts33, max_gap=5)
    np.testing.assert_array_equal(filled, synth_kpts33)
    assert valid.all()


# --------------------------------------------------------------------------
# Resampling -- this is what replaces ffmpeg
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fps_in", [23.976, 25.0, 29.97, 30.0])
def test_resample_preserves_duration_in_seconds(fps_in):
    n_in = 300
    x = np.linspace(0.0, 1.0, n_in)
    y = F.resample_to_fps(x, fps_in, 30.0)

    assert (y.shape[0] - 1) / 30.0 == pytest.approx((n_in - 1) / fps_in, abs=1 / 30.0)


def test_resample_preserves_a_linear_ramp():
    """Values must land at the right *times*, not merely span the same range."""
    n_in, fps_in = 240, 24.0
    x = np.linspace(0.0, 10.0, n_in)
    y = F.resample_to_fps(x, fps_in, 30.0)

    duration = (n_in - 1) / fps_in
    t_out = np.arange(y.size) / 30.0
    np.testing.assert_allclose(y, 10.0 * t_out / duration, atol=1e-9)


def test_resample_never_extrapolates_past_the_input():
    """Rounding n_out up would clamp and silently repeat the last frame."""
    n_in, fps_in = 240, 24.0
    y = F.resample_to_fps(np.linspace(0.0, 10.0, n_in), fps_in, 30.0)
    assert (y.size - 1) / 30.0 <= (n_in - 1) / fps_in + 1e-12
    assert y[-1] != pytest.approx(y[-2]), "tail was clamped, not interpolated"


def test_resample_is_a_noop_at_the_same_rate():
    x = np.linspace(0.0, 1.0, 50)
    np.testing.assert_array_equal(F.resample_to_fps(x, 30.0, 30.0), x)


def test_resample_keeps_trailing_dims():
    x = np.zeros((100, 33, 4))
    assert F.resample_to_fps(x, 25.0, 30.0).shape[1:] == (33, 4)


def test_resample_to_length_puts_a_rep_on_the_phase_axis():
    x = np.linspace(0.0, 1.0, 47)
    y = F.resample_to_length(x, 32)
    assert y.shape == (32,)
    assert y[0] == pytest.approx(0.0) and y[-1] == pytest.approx(1.0)
