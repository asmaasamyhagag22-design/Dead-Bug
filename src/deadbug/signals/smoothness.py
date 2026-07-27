"""Movement smoothness via SPARC (spectral arc length).

**SPARC, not LDLJ or any time-domain jerk metric.** Jerk-based measures are
dominated by movement *duration*, which would make "jerky" almost collinear
with "moving too fast" -- and those are two separate errors this project has to
tell apart. SPARC responds to the shape of the speed profile and is comparatively
insensitive to how long the movement took.

Reference: Balasubramanian et al., "On the analysis of movement smoothness",
J NeuroEng Rehabil 2015.

Values are negative; **closer to zero means smoother**. A pure minimum-jerk
reach scores highest, which is what ``tests/test_smoothness.py`` asserts.
"""

from __future__ import annotations

import numpy as np


def sparc(
    speed: np.ndarray,
    fs: float,
    padlevel: int = 4,
    fc: float = 10.0,
    amp_th: float = 0.05,
) -> float:
    """Spectral arc length of a speed profile.

    Args:
        speed: 1-D speed magnitude over time.
        fs: sampling rate in Hz.
        padlevel: zero-padding exponent; more padding, finer frequency grid.
        fc: cutoff frequency -- everything above is treated as noise.
        amp_th: normalized amplitude below which the spectrum is trimmed.

    Returns:
        Negative arc length, or NaN if the profile is too short or flat.
    """
    v = np.asarray(speed, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 4:
        return float("nan")

    nfft = int(2 ** (np.ceil(np.log2(v.size)) + padlevel))
    freq = np.arange(0, fs, fs / nfft)
    spectrum = np.abs(np.fft.fft(v, nfft))
    peak = spectrum.max()
    if peak <= 0:
        return float("nan")
    spectrum = spectrum / peak

    keep = freq <= fc
    freq, spectrum = freq[keep], spectrum[keep]
    above = np.nonzero(spectrum >= amp_th)[0]
    if above.size < 2:
        return float("nan")
    freq = freq[above[0] : above[-1] + 1]
    spectrum = spectrum[above[0] : above[-1] + 1]

    span = freq[-1] - freq[0]
    if span <= 0:
        return float("nan")
    return float(
        -np.sum(np.sqrt((np.diff(freq) / span) ** 2 + np.diff(spectrum) ** 2))
    )


def speed_profile(kpts: np.ndarray, joint: int, fps: float) -> np.ndarray:
    """Speed magnitude of one joint, in normalized units per second."""
    xy = np.asarray(kpts, dtype=np.float64)[:, joint, :2]
    return np.linalg.norm(np.diff(xy, axis=0), axis=-1) * fps


def sparc_per_rep(
    kpts_norm: np.ndarray,
    rep,
    fps: float,
    padlevel: int = 4,
    fc: float = 10.0,
    amp_th: float = 0.05,
) -> dict[str, float]:
    """SPARC for the moving wrist and the moving ankle of one rep.

    The two are reported separately: an arm can be controlled while the leg
    drops, and averaging them would hide exactly that.
    """
    from ..pose import skeleton as sk

    wrist, ankle = sk.CONTRALATERAL_MP33[rep.side]
    window = slice(rep.start, rep.end + 1)
    return {
        "sparc_wrist": sparc(speed_profile(kpts_norm[window], wrist, fps), fps,
                             padlevel, fc, amp_th),
        "sparc_ankle": sparc(speed_profile(kpts_norm[window], ankle, fps), fps,
                             padlevel, fc, amp_th),
    }


def minimum_jerk_speed(n: int, amplitude: float = 1.0) -> np.ndarray:
    """Speed profile of an ideal minimum-jerk reach.

    Position ``10t³ − 15t⁴ + 6t⁵`` differentiates to ``30t²(1−t)²``. This is the
    smoothness ceiling: nothing should score above it.
    """
    t = np.linspace(0.0, 1.0, n)
    return amplitude * 30.0 * t**2 * (1.0 - t) ** 2
