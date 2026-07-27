"""Rib flare — the early-warning band.

Identical formulation to :mod:`deadbug.signals.lumbar`, over a window further up
the torso (``[0.45, 0.75] × torso_len`` instead of ``[0.00, 0.35]``). Sharing
one implementation is deliberate: two bands that differ only in their window
must not differ in code, or they will drift apart.

The point of measuring it separately is timing. **Rib flare precedes visible
lumbar arching** — the ribcage lifts away from the pelvis before the lower back
actually leaves the floor — so ``rib_lead`` turns positive early in a rep that
is going to fail. That makes it useful for coaching in a way a lagging
indicator is not.
"""

from __future__ import annotations

import numpy as np

from .lumbar import band_signal

RIB_WINDOW = (0.45, 0.75)


def rib_gap(
    masks: np.ndarray,
    kpts_px: np.ndarray,
    floor: dict,
    start_frac: float = RIB_WINDOW[0],
    end_frac: float = RIB_WINDOW[1],
) -> dict[str, np.ndarray]:
    """Silhouette gap over the rib band."""
    out = band_signal(masks, kpts_px, floor, start_frac, end_frac)
    return {
        "rib_gap": out["area"],
        "rib_gap_max": out["peak"],
        "rib_window_px": out["width_px"],
    }


def rib_lead(rib: np.ndarray, lumbar: np.ndarray) -> np.ndarray:
    """``rib_gap − lumbar_gap``.

    Positive early in a failing rep, because the ribs lift first. Both inputs
    are already normalized by ``torso_len**2``, so the difference is meaningful.
    """
    return np.asarray(rib, dtype=np.float64) - np.asarray(lumbar, dtype=np.float64)


def lead_time_s(rib: np.ndarray, lumbar: np.ndarray, fps: float, k: float = 2.0) -> float:
    """How many seconds the rib signal crosses its baseline before the lumbar one.

    Each signal gets its own baseline from the first fifth of the rep, and the
    crossing is at ``baseline + k * std``. NaN when either never crosses --
    which is the expected outcome for a correct rep, and must not be read as
    "zero lead".
    """
    rib = np.asarray(rib, dtype=np.float64)
    lumbar = np.asarray(lumbar, dtype=np.float64)
    if rib.size != lumbar.size or rib.size < 10:
        return float("nan")

    rib_cross = _first_crossing(rib, k)
    lumbar_cross = _first_crossing(lumbar, k)
    if rib_cross is None or lumbar_cross is None:
        return float("nan")
    return (lumbar_cross - rib_cross) / fps


def _first_crossing(signal: np.ndarray, k: float) -> int | None:
    calib = signal[: max(3, signal.size // 5)]
    calib = calib[np.isfinite(calib)]
    if calib.size < 2:
        return None
    threshold = float(np.mean(calib) + k * np.std(calib))
    above = np.flatnonzero(np.isfinite(signal) & (signal > threshold))
    return int(above[0]) if above.size else None
