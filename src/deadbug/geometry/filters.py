"""One Euro filtering, gap interpolation and frame-rate resampling.

Detector jitter is the noise floor of the whole project. A moving average would
lag and flatten exactly the peaks rep segmentation depends on, so we use the
One Euro filter (Casiez, Roussel & Vogel, CHI 2012), which adapts its cutoff to
the speed of the signal: heavy smoothing when still, light smoothing when fast.
"""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (TWO_PI * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Scalar One Euro filter.

    Args:
        freq: nominal sample rate in Hz.
        min_cutoff: cutoff at zero speed. Lower = smoother but laggier.
        beta: speed coefficient. Higher = less lag when moving fast.
        d_cutoff: cutoff of the derivative's own low-pass.
    """

    def __init__(
        self,
        freq: float,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        if freq <= 0:
            raise ValueError("freq must be positive")
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0

    def __call__(self, x: float, dt: float | None = None) -> float:
        dt = 1.0 / self.freq if dt is None else float(dt)
        if dt <= 0:
            raise ValueError("dt must be positive")

        if self._x_prev is None or not np.isfinite(self._x_prev):
            # First finite sample seeds the filter; smoothing a NaN history
            # would poison every subsequent output.
            self._x_prev = float(x)
            self._dx_prev = 0.0
            return float(x)

        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev, self._dx_prev = x_hat, dx_hat
        return float(x_hat)


def one_euro_array(
    x: np.ndarray,
    fps: float,
    min_cutoff: float = 1.0,
    beta: float = 0.007,
    d_cutoff: float = 1.0,
) -> np.ndarray:
    """Filter along axis 0, independently per trailing coordinate.

    NaNs pass through as NaN and do not corrupt the filter state, so a dropped
    detection leaves a hole rather than a smear.
    """
    arr = np.asarray(x, dtype=np.float64)
    flat = arr.reshape(arr.shape[0], -1)
    out = np.empty_like(flat)

    for c in range(flat.shape[1]):
        f = OneEuroFilter(fps, min_cutoff, beta, d_cutoff)
        series = flat[:, c]
        for t, value in enumerate(series):
            if np.isnan(value):
                out[t, c] = np.nan
                f.reset()
            else:
                out[t, c] = f(value)
    return out.reshape(arr.shape)


def interpolate_gaps(
    kpts: np.ndarray, max_gap: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly fill short NaN runs; leave long ones as holes.

    A gap of up to ``max_gap`` frames is a dropped detection worth bridging.
    Anything longer is a real loss of track and must stay visible to QC rather
    than being invented.

    Leading and trailing runs are never filled -- there is nothing to
    interpolate between, and extrapolating a limb position is worse than a NaN.

    Returns:
        ``(filled, valid)`` where ``valid`` is ``(T,)`` and False for any frame
        still carrying a NaN in x or y.
    """
    arr = np.asarray(kpts, dtype=np.float64).copy()
    flat = arr.reshape(arr.shape[0], -1)
    n = flat.shape[0]

    for c in range(flat.shape[1]):
        series = flat[:, c]
        bad = np.isnan(series)
        if not bad.any() or bad.all():
            continue
        for start, stop in _runs(bad):
            if start == 0 or stop == n:      # unbracketed: nothing to span
                continue
            if stop - start > max_gap:
                continue
            lo, hi = series[start - 1], series[stop]
            series[start:stop] = np.linspace(lo, hi, stop - start + 2)[1:-1]

    filled = flat.reshape(arr.shape)
    valid = ~np.isnan(filled[..., :2]).any(axis=(1, 2))
    return filled, valid


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``[start, stop)`` index ranges of consecutive True values."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2], edges[1::2]))


def resample_to_fps(x: np.ndarray, fps_in: float, fps_out: float = 30.0) -> np.ndarray:
    """Resample along axis 0 from ``fps_in`` to ``fps_out``, preserving duration.

    This is what lets us skip re-encoding the source videos. Clips arrive at
    23.98 / 25 / 29.97 / 30 fps; rather than run every file through ffmpeg to
    force a constant rate, we carry the true fps and resample the *signals*.
    """
    if fps_in <= 0 or fps_out <= 0:
        raise ValueError("frame rates must be positive")
    arr = np.asarray(x, dtype=np.float64)
    n_in = arr.shape[0]
    if n_in < 2 or fps_in == fps_out:
        return arr.copy()

    duration = (n_in - 1) / fps_in
    # floor, not round: rounding up would push the last output sample past the
    # end of the input, where np.interp clamps and silently repeats the final
    # value. Better to end up to one frame short than to invent a frame.
    n_out = int(np.floor(duration * fps_out)) + 1
    t_in = np.arange(n_in) / fps_in
    t_out = np.arange(n_out) / fps_out

    flat = arr.reshape(n_in, -1)
    out = np.empty((n_out, flat.shape[1]), dtype=np.float64)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(t_out, t_in, flat[:, c])
    return out.reshape((n_out,) + arr.shape[1:])


def resample_to_length(x: np.ndarray, n_out: int) -> np.ndarray:
    """Resample along axis 0 onto ``n_out`` samples of normalized time.

    Used to put every rep on the same 32-point phase axis. The unresampled
    original must be kept alongside -- this destroys the duration information
    the timing features depend on.
    """
    arr = np.asarray(x, dtype=np.float64)
    n_in = arr.shape[0]
    if n_in < 2:
        return np.repeat(arr[:1], n_out, axis=0)

    t_in = np.linspace(0.0, 1.0, n_in)
    t_out = np.linspace(0.0, 1.0, n_out)
    flat = arr.reshape(n_in, -1)
    out = np.empty((n_out, flat.shape[1]), dtype=np.float64)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(t_out, t_in, flat[:, c])
    return out.reshape((n_out,) + arr.shape[1:])
