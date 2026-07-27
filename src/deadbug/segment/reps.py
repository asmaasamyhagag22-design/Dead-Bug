"""Rep segmentation from the contralateral extension signals.

A **rep is one single-side extension**, not a left+right pair. That doubles the
sample count, and more importantly it makes the labels precise: one side can be
performed correctly while the other fails, and a paired definition would smear
those together.

Two signals, one per side:

    sig_R = ‖ kpts_norm[R_wrist] − kpts_norm[L_ankle] ‖
    sig_L = ‖ kpts_norm[L_wrist] − kpts_norm[R_ankle] ‖

Both peak when that diagonal is fully extended. Peaks are found with an
**auto-tuned prominence** — ``0.25 × (p95 − p05)`` of the signal itself — rather
than a fixed threshold, because the amplitude varies with body proportions and
how far each person actually reaches.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

from ..geometry.filters import one_euro_array
from ..pose import skeleton as sk


@dataclass
class Rep:
    """One single-side extension."""

    side: str          # "R" or "L"
    start: int         # frame index
    peak: int
    end: int
    fps: float
    excursion_peak: float = float("nan")
    duration_s: float = float("nan")
    t_extend_s: float = float("nan")
    t_return_s: float = float("nan")
    dwell_s: float = float("nan")
    overlap_s: float = float("nan")
    flags: list[str] = field(default_factory=list)

    @property
    def start_s(self) -> float:
        return self.start / self.fps

    @property
    def end_s(self) -> float:
        return self.end / self.fps


def contralateral_signals(kpts_norm: np.ndarray) -> dict[str, np.ndarray]:
    """``(T, 33, C) -> {"R": (T,), "L": (T,)}`` wrist-to-opposite-ankle distance."""
    out = {}
    for side, (wrist, ankle) in sk.CONTRALATERAL_MP33.items():
        delta = kpts_norm[:, wrist, :2] - kpts_norm[:, ankle, :2]
        out[side] = np.linalg.norm(delta, axis=-1)
    return out


def auto_prominence(
    signal: np.ndarray, frac: float = 0.25, floor: float = 0.02,
    lo: float = 5.0, hi: float = 95.0,
) -> float:
    """Prominence scaled to the signal's own range.

    Percentiles rather than min/max so one bad frame cannot set the threshold.
    """
    finite = signal[np.isfinite(signal)]
    if finite.size < 2:
        return floor
    spread = float(np.percentile(finite, hi) - np.percentile(finite, lo))
    return max(floor, frac * spread)


def _walk_to_minima(signal: np.ndarray, peak: int) -> tuple[int, int]:
    """From a peak, walk outward while the signal keeps falling."""
    n = signal.size
    start = peak
    while start > 0 and signal[start - 1] <= signal[start]:
        start -= 1
    end = peak
    while end < n - 1 and signal[end + 1] <= signal[end]:
        end += 1
    return start, end


def find_reps(
    signal: np.ndarray,
    fps: float,
    side: str = "R",
    min_distance_s: float = 0.8,
    prominence_frac: float = 0.25,
    prominence_floor: float = 0.02,
    smooth: bool = True,
    one_euro_kwargs: dict | None = None,
) -> list[Rep]:
    """Locate every extension in one side's signal."""
    x = np.asarray(signal, dtype=np.float64)
    if smooth:
        # zero_phase: rep boundaries are timing measurements, and a causal
        # filter would shift every one of them later by the same amount.
        kwargs = {"zero_phase": True, **(one_euro_kwargs or {})}
        x = one_euro_array(x, fps, **kwargs)

    # find_peaks cannot handle NaN; interpolate for detection only, and keep the
    # original values for the reported amplitudes.
    filled = x.copy()
    bad = ~np.isfinite(filled)
    if bad.all():
        return []
    if bad.any():
        idx = np.arange(filled.size)
        filled[bad] = np.interp(idx[bad], idx[~bad], filled[~bad])

    peaks, _ = find_peaks(
        filled,
        distance=max(1, int(round(min_distance_s * fps))),
        prominence=auto_prominence(filled, prominence_frac, prominence_floor),
    )

    reps = []
    for peak in peaks:
        start, end = _walk_to_minima(filled, int(peak))
        if end <= start:
            continue
        reps.append(
            Rep(side=side, start=start, peak=int(peak), end=end, fps=fps,
                excursion_peak=float(filled[peak]))
        )
    return reps


def rep_timing(reps: list[Rep], signals: dict[str, np.ndarray], dwell_frac: float = 0.05) -> list[Rep]:
    """Fill in the per-rep timing features, in place.

    ``dwell_s`` is the coached pause: frames within ``dwell_frac`` of the peak
    amplitude.

    ``overlap_s = start(next) − end(this)``. **Negative means the next rep began
    before this one finished** — that is the operational definition of "moving
    too fast". It is preferred over a duration threshold because it needs no
    per-person calibration: a naturally quick mover with clean separation is not
    penalised, and a slow mover who runs their reps together still is.
    """
    ordered = sorted(reps, key=lambda r: r.start)
    for i, rep in enumerate(ordered):
        rep.duration_s = (rep.end - rep.start) / rep.fps
        rep.t_extend_s = (rep.peak - rep.start) / rep.fps
        rep.t_return_s = (rep.end - rep.peak) / rep.fps

        sig = signals.get(rep.side)
        if sig is not None and rep.end > rep.start:
            window = np.asarray(sig, dtype=np.float64)[rep.start : rep.end + 1]
            amplitude = np.nanmax(window) if np.isfinite(window).any() else np.nan
            if np.isfinite(amplitude):
                near = window >= amplitude * (1.0 - dwell_frac)
                rep.dwell_s = float(np.count_nonzero(near)) / rep.fps

        if i + 1 < len(ordered):
            rep.overlap_s = (ordered[i + 1].start - rep.end) / rep.fps
            if rep.overlap_s < 0:
                rep.flags.append("overlaps_next")
    return ordered


def check_alternation(reps: list[Rep]) -> tuple[bool, list[Rep]]:
    """Reps should alternate R, L, R, L… Flag every break rather than dropping it.

    A broken run is usually a missed rep or a double-counted one, which is a QC
    signal about the clip -- not something to quietly discard.
    """
    ordered = sorted(reps, key=lambda r: r.start)
    intact = True
    for prev, cur in zip(ordered, ordered[1:]):
        if prev.side == cur.side:
            cur.flags.append("alternation_broken")
            intact = False
    return intact, ordered


def segment_reps(
    kpts_norm: np.ndarray,
    fps: float,
    min_distance_s: float = 0.8,
    prominence_frac: float = 0.25,
    prominence_floor: float = 0.02,
    dwell_frac: float = 0.05,
    smooth: bool = True,
) -> tuple[list[Rep], dict]:
    """Full segmentation for a clip.

    Returns ``(reps, info)`` with reps in time order and ``info`` carrying the
    per-side counts and the alternation verdict for QC.
    """
    signals = contralateral_signals(kpts_norm)
    reps: list[Rep] = []
    for side, sig in signals.items():
        reps += find_reps(
            sig, fps, side=side, min_distance_s=min_distance_s,
            prominence_frac=prominence_frac, prominence_floor=prominence_floor,
            smooth=smooth,
        )

    intact, reps = check_alternation(reps)
    reps = rep_timing(reps, signals, dwell_frac=dwell_frac)
    for index, rep in enumerate(reps):
        rep.rep_index_in_clip = index  # type: ignore[attr-defined]

    info = {
        "n_reps": len(reps),
        "n_right": sum(r.side == "R" for r in reps),
        "n_left": sum(r.side == "L" for r in reps),
        "alternation_intact": intact,
        "n_overlapping": sum("overlaps_next" in r.flags for r in reps),
    }
    return reps, info


def segment_from_config(kpts_norm: np.ndarray, fps: float, cfg: dict) -> tuple[list[Rep], dict]:
    from ..config import cfg_get

    return segment_reps(
        kpts_norm, fps,
        min_distance_s=cfg_get(cfg, "segment.min_distance_s"),
        prominence_frac=cfg_get(cfg, "segment.prominence_frac"),
        prominence_floor=cfg_get(cfg, "segment.prominence_floor"),
        dwell_frac=cfg_get(cfg, "segment.dwell_amplitude_frac"),
        smooth=cfg_get(cfg, "segment.smooth_with_one_euro"),
    )


def count_error(n_auto: int, n_manual: int) -> float:
    """Relative counting error -- Gate 3 requires <= 0.02."""
    if n_manual <= 0:
        return float("nan")
    return abs(n_auto - n_manual) / n_manual
