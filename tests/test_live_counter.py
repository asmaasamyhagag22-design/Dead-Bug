"""Causal rep counting — the live counterpart of tests/test_reps.py.

The offline segmenter may look at the whole signal. This one may not, so the
properties worth pinning are different: it must not count a rep before the
movement has actually come back down, it must adapt to how far a given person
reaches, and it must never emit a rep twice.
"""

from __future__ import annotations

import numpy as np
import pytest

from deadbug.live.counter import DualCounter, OnlineRepCounter, State

FPS = 30.0


def feed(counter, signal, fps=FPS, start_frame=0):
    """Push a whole signal through one sample at a time."""
    out = []
    for i, value in enumerate(signal):
        rep = counter.update(float(value), start_frame + i, (start_frame + i) / fps)
        if rep is not None:
            out.append(rep)
    return out


def bumps(n, per_rep=60, baseline=0.5, amplitude=1.0):
    """``n`` raised-cosine extensions back to back."""
    phase = np.arange(per_rep) / per_rep
    one = baseline + amplitude * 0.5 * (1 - np.cos(2 * np.pi * phase))
    return np.tile(one, n)


# --------------------------------------------------------------------------
# Counting
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 3, 8])
def test_counts_clean_reps(n):
    assert len(feed(OnlineRepCounter("R", FPS), bumps(n))) == n


def test_counts_are_not_duplicated():
    """Each extension is emitted exactly once, never re-fired while descending."""
    reps = feed(OnlineRepCounter("R", FPS), bumps(5))
    assert len(reps) == 5
    peaks = [r.peak_frame for r in reps]
    assert len(set(peaks)) == 5
    assert peaks == sorted(peaks)


def test_flat_signal_produces_nothing():
    assert feed(OnlineRepCounter("R", FPS), np.full(300, 0.5)) == []


def test_noise_alone_does_not_produce_reps():
    """min_range stops sensor jitter from being counted as movement."""
    rng = np.random.default_rng(0)
    signal = 0.5 + rng.normal(0, 0.004, 400)
    assert feed(OnlineRepCounter("R", FPS), signal) == []


def test_nan_frames_are_ignored():
    signal = bumps(3)
    signal[40:46] = np.nan          # a dropped detection mid-rep
    assert len(feed(OnlineRepCounter("R", FPS), signal)) == 3


# --------------------------------------------------------------------------
# Causality -- the property that separates this from the offline segmenter
# --------------------------------------------------------------------------


def test_rep_is_not_emitted_before_the_descent():
    """At the top of the movement the counter must stay silent.

    Nothing seen so far distinguishes a peak from a pause partway up, so
    emitting here would mean guessing.
    """
    counter = OnlineRepCounter("R", FPS)
    rising = bumps(1)[:31]          # up to and including the peak
    assert feed(counter, rising) == []
    assert counter.state is State.RISING


def test_rep_is_emitted_shortly_after_the_peak():
    """Confirmation costs a bounded delay, and the bound is worth pinning."""
    counter = OnlineRepCounter("R", FPS)
    reps = feed(counter, bumps(1))
    assert len(reps) == 1
    rep = reps[0]
    lag_frames = rep.end_frame - rep.peak_frame
    assert 0 < lag_frames <= 20, f"confirmation lag {lag_frames} frames is too large"


def test_peak_frame_is_close_to_the_true_peak():
    reps = feed(OnlineRepCounter("R", FPS), bumps(3, per_rep=60))
    for i, rep in enumerate(reps):
        assert abs(rep.peak_frame - (i * 60 + 30)) <= 3


# --------------------------------------------------------------------------
# Adaptation
# --------------------------------------------------------------------------


def test_threshold_adapts_to_reach():
    """A short person and a tall one produce different amplitudes; a fixed
    threshold would miss one and over-split the other."""
    small = OnlineRepCounter("R", FPS)
    large = OnlineRepCounter("R", FPS)
    feed(small, bumps(4, amplitude=0.2))
    feed(large, bumps(4, amplitude=2.0))
    assert large.threshold() > small.threshold()


def test_small_amplitude_reps_are_still_counted():
    assert len(feed(OnlineRepCounter("R", FPS), bumps(4, amplitude=0.25))) == 4


def test_refractory_period_blocks_a_double_peak():
    """One movement that wobbles at the top is one rep, not two."""
    idx = np.arange(120)
    signal = 0.5 + 0.6 * np.exp(-(((idx - 40) / 8.0) ** 2)) \
                 + 0.6 * np.exp(-(((idx - 52) / 8.0) ** 2))   # 0.4 s apart
    counter = OnlineRepCounter("R", FPS, min_rep_s=0.8)
    assert len(feed(counter, signal)) == 1


# --------------------------------------------------------------------------
# Both sides
# --------------------------------------------------------------------------


def test_dual_counter_alternates():
    dual = DualCounter(fps=FPS)
    per_rep, n = 60, 6
    sig = bumps(1, per_rep=per_rep)
    flat = np.full(per_rep, 0.5)

    for r in range(n):
        active, idle = ("R", "L") if r % 2 == 0 else ("L", "R")
        for i in range(per_rep):
            dual.update({active: sig[i], idle: flat[i]}, r * per_rep + i,
                        (r * per_rep + i) / FPS)

    assert dual.count == n
    assert [rep.side for rep in dual.reps] == ["R", "L"] * (n // 2)
    assert not any(r.metrics.get("alternation_broken") for r in dual.reps)


def test_same_side_twice_is_flagged():
    dual = DualCounter(fps=FPS)
    per_rep = 60
    sig = bumps(1, per_rep=per_rep)
    flat = np.full(per_rep, 0.5)

    for r in range(2):                      # both on R
        for i in range(per_rep):
            dual.update({"R": sig[i], "L": flat[i]}, r * per_rep + i,
                        (r * per_rep + i) / FPS)

    assert dual.count == 2
    assert dual.reps[1].metrics.get("alternation_broken") is True
