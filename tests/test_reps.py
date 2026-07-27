"""Rep segmentation — Gate 3 (counting error <= 2%)."""

from __future__ import annotations

import numpy as np
import pytest

from deadbug.pose import skeleton as sk
from deadbug.segment import reps as R

FPS = 30.0


def test_contralateral_pairs_are_the_diagonals():
    """Right wrist pairs with LEFT ankle. Getting this backwards would still
    produce plausible-looking peaks, so it is pinned explicitly."""
    assert sk.CONTRALATERAL_MP33["R"] == (sk.R_WRIST, sk.L_ANKLE)
    assert sk.CONTRALATERAL_MP33["L"] == (sk.L_WRIST, sk.R_ANKLE)


def test_signals_peak_when_the_diagonal_extends(synth_kpts33):
    signals = R.contralateral_signals(synth_kpts33)
    assert set(signals) == {"R", "L"}
    for sig in signals.values():
        assert sig.shape == (synth_kpts33.shape[0],)
        assert np.ptp(sig) > 0, "signal is flat -- no extension detected"


# --------------------------------------------------------------------------
# Counting -- Gate 3
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_reps", [4, 6, 10])
def test_counts_reps_exactly_on_a_clean_signal(alternating_signal, n_reps):
    sig_r, sig_l, truth = alternating_signal(n_reps=n_reps, fps=FPS)
    found = R.find_reps(sig_r, FPS, side="R") + R.find_reps(sig_l, FPS, side="L")

    assert len(found) == n_reps
    assert R.count_error(len(found), n_reps) == 0.0


def test_gate_3_tolerance_on_a_noisy_signal(alternating_signal):
    """Gate 3: automatic count within 2% of the truth."""
    rng = np.random.default_rng(0)
    sig_r, sig_l, _ = alternating_signal(n_reps=10, fps=FPS)
    sig_r = sig_r + rng.normal(0, 0.02, sig_r.size)
    sig_l = sig_l + rng.normal(0, 0.02, sig_l.size)

    found = R.find_reps(sig_r, FPS, side="R") + R.find_reps(sig_l, FPS, side="L")
    assert R.count_error(len(found), 10) <= 0.02


def test_peaks_land_near_the_true_peaks(alternating_signal):
    sig_r, _, truth = alternating_signal(n_reps=6, fps=FPS)
    found = R.find_reps(sig_r, FPS, side="R")
    expected = [t[1] for t in truth if t[3] == "R"]

    assert len(found) == len(expected)
    for rep, peak in zip(found, expected):
        assert abs(rep.peak - peak) <= 3


def test_boundaries_bracket_the_peak(alternating_signal):
    sig_r, _, _ = alternating_signal(n_reps=4, fps=FPS)
    for rep in R.find_reps(sig_r, FPS, side="R"):
        assert rep.start < rep.peak < rep.end


def test_min_distance_suppresses_double_counting(alternating_signal):
    """Two bumps closer than min_distance_s must not both be counted."""
    sig = np.full(120, 0.5)
    for centre in (30, 36):                      # 0.2 s apart at 30 fps
        idx = np.arange(120)
        sig = sig + 0.5 * np.exp(-(((idx - centre) / 3.0) ** 2))

    assert len(R.find_reps(sig, FPS, min_distance_s=0.8)) == 1


def test_flat_signal_yields_no_reps():
    assert R.find_reps(np.full(200, 0.5), FPS) == []


def test_all_nan_signal_yields_no_reps():
    assert R.find_reps(np.full(120, np.nan), FPS) == []


def test_auto_prominence_scales_with_amplitude():
    """A fixed threshold would miss small movers and over-split large ones."""
    small = np.tile(np.concatenate([np.zeros(20), np.ones(20) * 0.1]), 5)
    large = small * 10.0
    assert R.auto_prominence(large) > R.auto_prominence(small)
    assert R.auto_prominence(np.zeros(50)) == pytest.approx(0.02)


# --------------------------------------------------------------------------
# Alternation and timing
# --------------------------------------------------------------------------


def test_alternation_intact_is_detected(alternating_signal):
    sig_r, sig_l, _ = alternating_signal(n_reps=6, fps=FPS)
    found = R.find_reps(sig_r, FPS, side="R") + R.find_reps(sig_l, FPS, side="L")
    intact, ordered = R.check_alternation(found)

    assert intact
    assert [r.side for r in ordered] == ["R", "L"] * 3


def test_broken_alternation_is_flagged_not_dropped():
    """A repeated side usually means a missed rep -- a QC signal, not noise."""
    reps = [
        R.Rep(side="R", start=0, peak=10, end=20, fps=FPS),
        R.Rep(side="R", start=30, peak=40, end=50, fps=FPS),
        R.Rep(side="L", start=60, peak=70, end=80, fps=FPS),
    ]
    intact, ordered = R.check_alternation(reps)

    assert not intact
    assert len(ordered) == 3, "a flagged rep must survive"
    assert "alternation_broken" in ordered[1].flags


def test_overlap_is_negative_when_reps_run_together():
    """`overlap_s < 0` IS the definition of moving too fast."""
    reps = [
        R.Rep(side="R", start=0, peak=15, end=40, fps=FPS),
        R.Rep(side="L", start=30, peak=45, end=70, fps=FPS),   # starts before #1 ends
    ]
    ordered = R.rep_timing(reps, {})

    assert ordered[0].overlap_s == pytest.approx((30 - 40) / FPS)
    assert ordered[0].overlap_s < 0
    assert "overlaps_next" in ordered[0].flags


def test_overlap_is_positive_with_a_clean_pause():
    reps = [
        R.Rep(side="R", start=0, peak=15, end=30, fps=FPS),
        R.Rep(side="L", start=45, peak=60, end=75, fps=FPS),
    ]
    ordered = R.rep_timing(reps, {})

    assert ordered[0].overlap_s == pytest.approx(0.5)
    assert "overlaps_next" not in ordered[0].flags


def test_timing_splits_extend_and_return():
    rep = R.Rep(side="R", start=0, peak=30, end=90, fps=FPS)
    ordered = R.rep_timing([rep], {})[0]

    assert ordered.duration_s == pytest.approx(3.0)
    assert ordered.t_extend_s == pytest.approx(1.0)
    assert ordered.t_return_s == pytest.approx(2.0)


def test_dwell_measures_the_pause_at_the_top():
    """A held peak must register more dwell than a sharp one of equal duration."""
    n = 90
    held = np.concatenate([np.linspace(0, 1, 30), np.ones(30), np.linspace(1, 0, 30)])
    sharp = np.concatenate([np.linspace(0, 1, 45), np.linspace(1, 0, 45)])
    rep = R.Rep(side="R", start=0, peak=45, end=n - 1, fps=FPS)

    dwell_held = R.rep_timing([rep], {"R": held})[0].dwell_s
    rep2 = R.Rep(side="R", start=0, peak=45, end=n - 1, fps=FPS)
    dwell_sharp = R.rep_timing([rep2], {"R": sharp})[0].dwell_s

    assert dwell_held > dwell_sharp


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_segment_reps_end_to_end(synth_kpts33):
    reps, info = R.segment_reps(synth_kpts33, FPS)

    assert info["n_reps"] == len(reps) > 0
    assert info["n_right"] + info["n_left"] == info["n_reps"]
    assert all(r.start < r.peak < r.end for r in reps)
    assert [r.start for r in reps] == sorted(r.start for r in reps)
    assert all(np.isfinite(r.duration_s) for r in reps)


def test_count_error_helper():
    assert R.count_error(10, 10) == 0.0
    assert R.count_error(49, 50) == pytest.approx(0.02)
    assert np.isnan(R.count_error(5, 0))
