"""The normative band: what ``lumbar_gap`` does during a correct rep.

There is no universal threshold for "the back is off the floor". Body
proportions, camera distance and how far a person actually reaches all move the
number, and the exercise has no fixed correct range of motion. So the reference
is not a constant -- it is a **curve of lumbar gap against excursion**, built
from correct reps only, and a rep is judged against the band at the excursion it
actually reached.

Binning on excursion rather than on time is the whole idea. Two people doing the
same correct rep at different speeds are at the same place on this curve; two
people at the same *timestamp* are not. It also removes the loophole where a rep
scores well by never extending far enough to load the back.

**The band is built leave-one-subject-out.** A band that saw subject S is not a
reference for subject S -- comparing S against a curve S helped define
understates their deviation by construction, and with a handful of subjects the
effect is large, not academic. :func:`band_loso` builds one band per held-out
subject; :func:`fit_band` is the pooled version, for deployment only, where the
user was never in the training data at all.

Only ``view == "side"`` clips may contribute. An oblique camera foreshortens the
silhouette gap, so pooling views would widen the band by an amount that has
nothing to do with movement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np

#: Excursion is wrist-to-opposite-ankle distance in torso lengths. Reps below
#: this never loaded the back, so they carry no information about whether it
#: stayed down and would only pull the band's lower bins toward zero.
MIN_EXCURSION = 0.20


@dataclass
class Band:
    """A normative curve: per-excursion-bin mean and spread of ``lumbar_gap``."""

    edges: list[float]          # len = n_bins + 1
    mean: list[float]
    std: list[float]
    p05: list[float]
    p95: list[float]
    count: list[int]
    sigma: float                # control limit multiplier
    n_reps: int
    n_persons: int
    held_out: str | None = None
    signal: str = "lumbar_gap_peak"
    excursion: str = "excursion_peak"

    def upper(self) -> np.ndarray:
        """The control limit: ``mean + sigma * std`` per bin."""
        return np.asarray(self.mean) + self.sigma * np.asarray(self.std)

    def bin_of(self, excursion: float) -> int:
        """Index of the bin an excursion falls in, clamped to the ends.

        Clamped rather than rejected: a rep that extends further than anything
        in the training data is judged against the furthest bin available, and
        :func:`score_rep` reports that the value was extrapolated.
        """
        edges = np.asarray(self.edges)
        if not np.isfinite(excursion):
            return -1
        return int(np.clip(np.searchsorted(edges, excursion, side="right") - 1,
                           0, len(self.mean) - 1))

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "Band":
        return cls(**payload)


def _binned_stats(
    excursion: np.ndarray, signal: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(edges) - 1
    mean = np.full(n, np.nan)
    std = np.full(n, np.nan)
    p05 = np.full(n, np.nan)
    p95 = np.full(n, np.nan)
    count = np.zeros(n, dtype=int)

    index = np.clip(np.searchsorted(edges, excursion, side="right") - 1, 0, n - 1)
    for b in range(n):
        values = signal[(index == b) & np.isfinite(signal)]
        count[b] = values.size
        if values.size:
            mean[b] = values.mean()
            # ddof=1 needs two samples; a one-sample bin has no spread to report
            # and must not silently contribute a zero-width limit.
            std[b] = values.std(ddof=1) if values.size > 1 else np.nan
            p05[b] = np.percentile(values, 5)
            p95[b] = np.percentile(values, 95)
    return mean, std, p05, p95, count


def _interpolate_gaps(values: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Fill empty bins from their neighbours, **for display only**.

    With a handful of subjects some excursion bins are simply unvisited, and a
    curve full of holes is unreadable. So the gaps are interpolated -- but the
    filled values are fabricated, and :func:`score_rep` refuses to decide
    anything from them. ``Band.count`` is the record of what was actually
    observed, and it is the field that governs, not the presence of a finite
    std. See :data:`MIN_BIN_SUPPORT`.
    """
    out = np.asarray(values, dtype=np.float64).copy()
    known = np.flatnonzero(np.isfinite(out) & (count > 0))
    if known.size == 0:
        return out
    all_idx = np.arange(out.size)
    return np.interp(all_idx, known, out[known])


def fit_band(
    excursion: Sequence[float],
    signal: Sequence[float],
    persons: Sequence[str],
    n_bins: int = 20,
    sigma: float = 2.0,
    min_excursion: float = MIN_EXCURSION,
    held_out: str | None = None,
) -> Band:
    """Fit a pooled band from correct reps.

    The caller is responsible for having filtered to ``condition == "correct"``
    and ``view == "side"`` -- this function does not know about conditions, and
    silently filtering here would hide a mistake in the caller.
    """
    exc = np.asarray(excursion, dtype=np.float64)
    sig = np.asarray(signal, dtype=np.float64)
    people = np.asarray(persons, dtype=object)

    keep = np.isfinite(exc) & np.isfinite(sig) & (exc >= min_excursion)
    exc, sig, people = exc[keep], sig[keep], people[keep]
    if exc.size < 2:
        raise ValueError(
            f"only {exc.size} usable correct rep(s) above excursion {min_excursion}; "
            "a band cannot be fitted. This is the filming blocker, not a code bug."
        )

    edges = np.linspace(exc.min(), exc.max(), n_bins + 1)
    mean, std, p05, p95, count = _binned_stats(exc, sig, edges)

    return Band(
        edges=[float(e) for e in edges],
        mean=[float(v) for v in _interpolate_gaps(mean, count)],
        std=[float(v) for v in _interpolate_gaps(std, count)],
        p05=[float(v) for v in _interpolate_gaps(p05, count)],
        p95=[float(v) for v in _interpolate_gaps(p95, count)],
        count=[int(c) for c in count],
        sigma=float(sigma),
        n_reps=int(exc.size),
        n_persons=int(len(set(people.tolist()))),
        held_out=held_out,
    )


def band_loso(
    excursion: Sequence[float],
    signal: Sequence[float],
    persons: Sequence[str],
    n_bins: int = 20,
    sigma: float = 2.0,
    min_excursion: float = MIN_EXCURSION,
) -> dict[str, Band]:
    """One band per held-out subject. See the module docstring for why.

    A subject whose removal leaves fewer than two usable reps is skipped rather
    than fitted on nothing; the returned dict simply has no entry for them.
    """
    people = np.asarray(persons, dtype=object)
    bands: dict[str, Band] = {}
    for person in sorted(set(people.tolist())):
        mask = people != person
        try:
            bands[str(person)] = fit_band(
                np.asarray(excursion)[mask], np.asarray(signal)[mask], people[mask],
                n_bins=n_bins, sigma=sigma, min_excursion=min_excursion,
                held_out=str(person),
            )
        except ValueError:
            continue
    return bands


#: A bin needs this many observations before it may issue a verdict. Two,
#: because a ddof=1 standard deviation is undefined below it.
MIN_BIN_SUPPORT = 2


def score_rep(band: Band, excursion: float, signal: float) -> dict:
    """Judge one rep against a band.

    ``z`` is ``(signal − mean) / std`` in the rep's excursion bin, and
    ``exceeds`` is ``signal > mean + sigma*std``. Both are NaN / False when the
    bin has too little evidence -- an unsupported bin must abstain, not pass.

    **Abstention is keyed on ``count``, not on the std being NaN.**
    :func:`fit_band` interpolates the per-bin statistics across empty bins so
    the plotted curve is continuous, which means a bin with zero observations
    still carries a finite, entirely fabricated std. Measured on this project's
    own 9 side-view reps: 13 of 20 bins are empty, and every one of them would
    otherwise return a confident z. The interpolated values stay -- they are
    what makes the figure readable -- but they may not decide anything.
    """
    b = band.bin_of(excursion)
    if b < 0 or not np.isfinite(signal):
        return {"bin": b, "z": float("nan"), "limit": float("nan"), "exceeds": False,
                "support": 0, "extrapolated": False, "reason": "no bin"}

    support = int(band.count[b])
    mean = band.mean[b]
    std = band.std[b]
    usable = support >= MIN_BIN_SUPPORT and np.isfinite(std) and std > 0

    limit = mean + band.sigma * std if usable else float("nan")
    z = (signal - mean) / std if usable else float("nan")

    edges = band.edges
    extrapolated = excursion < edges[0] or excursion > edges[-1]
    if usable:
        reason = ""
    elif support < MIN_BIN_SUPPORT:
        reason = f"bin has {support} observation(s), needs {MIN_BIN_SUPPORT}"
    else:
        reason = "bin has no spread"
    return {
        "bin": b,
        "z": float(z),
        "limit": float(limit),
        "exceeds": bool(usable and signal > limit),
        "support": support,
        "extrapolated": bool(extrapolated),
        "reason": reason,
    }


def supported_mask(band: Band) -> np.ndarray:
    """``(n_bins,)`` bool -- which bins have enough evidence to decide anything.

    Plotting code should use this to stop drawing the control limit across bins
    that were interpolated out of nothing.
    """
    return np.asarray(band.count) >= MIN_BIN_SUPPORT


def save_band(band: Band | dict[str, Band], path: str | Path) -> Path:
    """Persist a band, or a full LOSO set of them, as JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(band, Band):
        payload = {"kind": "pooled", "band": band.as_dict()}
    else:
        payload = {"kind": "loso", "bands": {k: v.as_dict() for k, v in band.items()}}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def load_band(path: str | Path) -> Band | dict[str, Band]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("kind") == "pooled":
        return Band.from_dict(payload["band"])
    return {k: Band.from_dict(v) for k, v in payload["bands"].items()}


def build_from_reps(reps, cfg: dict) -> dict[str, Band]:
    """Fit the LOSO bands from a reps table, applying the config's filters.

    ``reps`` is the DataFrame from :func:`deadbug.dataset.build.read_reps`.
    """
    from ..config import cfg_get

    conditions = cfg_get(cfg, "dataset.band.conditions")
    views = cfg_get(cfg, "dataset.band.views")
    subset = reps[reps["condition"].isin(conditions) & reps["view"].isin(views)]
    if subset.empty:
        raise ValueError(
            f"no reps with condition in {conditions} and view in {views}. "
            "Check data/clips.csv -- the band cannot be built from an empty set."
        )
    return band_loso(
        subset["excursion_peak"].to_numpy(),
        subset["lumbar_gap_peak"].to_numpy(),
        subset["person_id"].to_numpy(),
        n_bins=cfg_get(cfg, "dataset.excursion_bins"),
        sigma=cfg_get(cfg, "dataset.control_limit.sigma"),
    )
