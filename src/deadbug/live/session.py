"""Live coaching session: calibrate on this person, then judge each rep.

The reference is **the user's own body**, not a population. That is not a
shortcut around having few subjects -- it is the correct design for a coaching
tool, and it is what the project's own analysis concluded: there is no fixed
correct range of motion. The correct range is the excursion at which *this*
person's lower back starts to leave the floor, and that differs with hip
mobility, limb length and core strength.

So a session runs in phases:

    SETUP     lie still  -> where is the floor, and what does this back look
                           like when it is flat?
    BASELINE  3 easy reps -> what are this person's normal gap, rotation and
                           tempo when they are not yet fatigued?
    ACTIVE    every rep judged against that personal baseline
    DONE      session report

A consequence worth stating plainly at the viva: the app needs no dataset to
*function*, only to be *validated*. Those are different claims and conflating
them is what makes a demo look like evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .counter import DualCounter, LiveRep


class Phase(Enum):
    SETUP = "setup"
    BASELINE = "baseline"
    ACTIVE = "active"
    DONE = "done"


#: Error -> (message, body region to highlight). Regions map onto the silhouette
#: bands the signals are computed over, so the highlight marks the place the
#: measurement actually came from rather than a generic body part.
ERRORS = {
    "lumbar_arch": ("Lower back is lifting - press it into the floor", "lumbar"),
    "rotation": ("Hips are rocking - keep the pelvis still", "pelvis"),
    "too_fast": ("Slow down - control the movement out", None),
}


@dataclass
class Verdict:
    """The judgement on one rep."""

    rep_index: int
    side: str
    ok: bool
    errors: list[str] = field(default_factory=list)
    region: str | None = None
    detail: dict = field(default_factory=dict)

    @property
    def message(self) -> str:
        if self.ok:
            return "Good rep"
        return ERRORS.get(self.errors[0], ("Form error", None))[0]


@dataclass
class SessionConfig:
    setup_seconds: float = 3.0
    baseline_reps: int = 3
    #: How many baseline standard deviations count as a deviation. 2.5 rather
    #: than 2.0 because a live session has far fewer baseline samples than an
    #: offline band, so the estimate of sigma is itself noisy -- a tighter
    #: threshold would produce false alarms on rep four.
    k_sigma: float = 2.5
    #: Absolute floor on the trigger, in torso-length^2 units. Without it, a
    #: subject whose baseline reps were unusually consistent gets a near-zero
    #: sigma and every subsequent rep trips the threshold.
    min_lumbar_trigger: float = 0.004
    min_rotation_trigger: float = 0.04
    #: An extension quicker than this fraction of the baseline median is "too
    #: fast". Measured on the extension phase, not the whole rep -- see
    #: LiveRep.t_extend_s for why the full duration is biased online.
    fast_duration_frac: float = 0.60
    max_reps: int = 100


@dataclass
class SessionState:
    phase: Phase
    reps_done: int = 0
    reps_correct: int = 0
    progress: float = 0.0          # 0..1 within the current phase
    prompt: str = ""
    last_verdict: Verdict | None = None
    highlight: str | None = None
    live_lumbar: float = float("nan")
    live_z: float = float("nan")   # deviation from baseline, in sigmas


class LiveSession:
    """Drives one coaching session. UI- and capture-agnostic.

    Feed it per-frame signals; it returns the state to render and speak.
    """

    def __init__(self, fps: float = 30.0, config: SessionConfig | None = None) -> None:
        self.fps = fps
        self.cfg = config or SessionConfig()
        self.counter = DualCounter(fps=fps)
        self.phase = Phase.SETUP
        self.verdicts: list[Verdict] = []

        self._t0: float | None = None
        self._setup_lumbar: list[float] = []
        self._setup_rot: list[float] = []
        self._frame_lumbar: dict[int, float] = {}
        self._frame_rot: dict[int, float] = {}
        self._baseline: dict[str, float] = {}
        self._prev_rep_end: float | None = None

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------

    def update(
        self,
        contralateral: dict[str, float],
        lumbar_gap: float,
        rot_dev: float,
        frame: int,
        now: float,
    ) -> SessionState:
        if self._t0 is None:
            self._t0 = now

        if np.isfinite(lumbar_gap):
            self._frame_lumbar[frame] = lumbar_gap
        if np.isfinite(rot_dev):
            self._frame_rot[frame] = rot_dev

        if self.phase is Phase.SETUP:
            return self._do_setup(lumbar_gap, rot_dev, now)
        if self.phase is Phase.DONE:
            return self._state(prompt="Session complete")

        finished = self.counter.update(contralateral, frame, now)
        for rep in finished:
            self._on_rep(rep)

        if self.phase is Phase.BASELINE:
            done = self.counter.count
            if done >= self.cfg.baseline_reps:
                self._compute_baseline()
                self.phase = Phase.ACTIVE
            return self._state(
                progress=min(1.0, done / max(1, self.cfg.baseline_reps)),
                prompt=f"Warm-up: {done}/{self.cfg.baseline_reps} easy reps",
            )

        return self._state(prompt="Go", live_lumbar=lumbar_gap)

    # ------------------------------------------------------------------

    def _do_setup(self, lumbar_gap: float, rot_dev: float, now: float) -> SessionState:
        elapsed = now - (self._t0 or now)
        if np.isfinite(lumbar_gap):
            self._setup_lumbar.append(lumbar_gap)
        if np.isfinite(rot_dev):
            self._setup_rot.append(rot_dev)

        if elapsed >= self.cfg.setup_seconds and len(self._setup_lumbar) >= 10:
            self.phase = Phase.BASELINE
            self.counter.reset()
            return self._state(prompt="Now do 3 slow reps")

        remaining = max(0.0, self.cfg.setup_seconds - elapsed)
        return self._state(
            progress=min(1.0, elapsed / self.cfg.setup_seconds),
            prompt=f"Lie still... {remaining:.0f}",
        )

    def _on_rep(self, rep: LiveRep) -> None:
        """Attach the signal summary for the frames this rep spanned."""
        span = range(rep.start_frame, rep.end_frame + 1)
        lumbar = [self._frame_lumbar[f] for f in span if f in self._frame_lumbar]
        rot = [abs(self._frame_rot[f]) for f in span if f in self._frame_rot]
        rep.metrics["lumbar_peak"] = float(np.max(lumbar)) if lumbar else float("nan")
        rep.metrics["lumbar_mean"] = float(np.mean(lumbar)) if lumbar else float("nan")
        rep.metrics["rot_peak"] = float(np.max(rot)) if rot else float("nan")
        rep.metrics["duration_s"] = rep.duration_s
        rep.metrics["t_extend_s"] = rep.t_extend_s

        if self.phase is Phase.ACTIVE:
            self.verdicts.append(self._judge(rep))
            if len(self.verdicts) >= self.cfg.max_reps:
                self.phase = Phase.DONE

    def _compute_baseline(self) -> None:
        """Freeze this person's normal from their warm-up reps."""
        warmup = self.counter.reps[: self.cfg.baseline_reps]

        def stat(key: str) -> tuple[float, float]:
            values = [r.metrics.get(key, np.nan) for r in warmup]
            values = [v for v in values if np.isfinite(v)]
            if not values:
                return float("nan"), float("nan")
            return float(np.mean(values)), float(np.std(values))

        lm, ls = stat("lumbar_peak")
        rm, rs = stat("rot_peak")
        durations = [r.metrics.get("t_extend_s", np.nan) for r in warmup]
        durations = [d for d in durations if np.isfinite(d)]

        rest = float(np.median(self._setup_lumbar)) if self._setup_lumbar else float("nan")
        self._baseline = {
            "lumbar_rest": rest,
            "lumbar_mean": lm, "lumbar_std": ls,
            "rot_mean": rm, "rot_std": rs,
            "duration_median": float(np.median(durations)) if durations else float("nan"),
        }

    def _trigger(self, mean: float, std: float, minimum: float) -> float:
        if not np.isfinite(mean):
            return float("inf")
        spread = std if np.isfinite(std) else 0.0
        return max(mean + self.cfg.k_sigma * spread, mean + minimum)

    def _judge(self, rep: LiveRep) -> Verdict:
        base = self._baseline
        errors: list[str] = []
        detail: dict = {}

        lumbar_peak = rep.metrics.get("lumbar_peak", np.nan)
        trigger = self._trigger(
            base.get("lumbar_mean", np.nan), base.get("lumbar_std", np.nan),
            self.cfg.min_lumbar_trigger,
        )
        detail["lumbar_peak"] = lumbar_peak
        detail["lumbar_trigger"] = trigger
        if np.isfinite(lumbar_peak) and lumbar_peak > trigger:
            errors.append("lumbar_arch")

        rot_peak = rep.metrics.get("rot_peak", np.nan)
        rot_trigger = self._trigger(
            base.get("rot_mean", np.nan), base.get("rot_std", np.nan),
            self.cfg.min_rotation_trigger,
        )
        detail["rot_peak"] = rot_peak
        if np.isfinite(rot_peak) and rot_peak > rot_trigger:
            errors.append("rotation")

        # Tempo is judged on the extension phase only. Trough and peak are both
        # observed directly, so the measure carries no confirmation lag; judging
        # the full rep duration instead flagged every rep of a clean clip,
        # because the lag inflates it by a constant.
        median = base.get("duration_median", np.nan)
        extend = rep.metrics.get("t_extend_s", np.nan)
        detail["t_extend_s"] = extend
        detail["t_extend_baseline"] = median
        if (
            np.isfinite(median) and np.isfinite(extend)
            and extend < median * self.cfg.fast_duration_frac
        ):
            errors.append("too_fast")

        region = ERRORS[errors[0]][1] if errors else None
        return Verdict(
            rep_index=len(self.verdicts) + 1, side=rep.side,
            ok=not errors, errors=errors, region=region, detail=detail,
        )

    def _state(self, **kwargs) -> SessionState:
        correct = sum(v.ok for v in self.verdicts)
        last = self.verdicts[-1] if self.verdicts else None
        live = kwargs.pop("live_lumbar", float("nan"))

        z = float("nan")
        base = self._baseline
        if np.isfinite(live) and np.isfinite(base.get("lumbar_mean", np.nan)):
            std = base.get("lumbar_std", 0.0) or self.cfg.min_lumbar_trigger
            z = (live - base["lumbar_mean"]) / max(std, 1e-9)

        return SessionState(
            phase=self.phase,
            reps_done=len(self.verdicts),
            reps_correct=correct,
            last_verdict=last,
            highlight=last.region if last and not last.ok else None,
            live_lumbar=live,
            live_z=z,
            **kwargs,
        )

    def finish(self) -> None:
        self.phase = Phase.DONE

    # ------------------------------------------------------------------

    def report(self) -> dict:
        """End-of-session summary: what was right, what was wrong, and how often."""
        total = len(self.verdicts)
        correct = sum(v.ok for v in self.verdicts)
        counts: dict[str, int] = {}
        for verdict in self.verdicts:
            for error in verdict.errors:
                counts[error] = counts.get(error, 0) + 1

        return {
            "total_reps": total,
            "correct_reps": correct,
            "accuracy": correct / total if total else float("nan"),
            "error_counts": counts,
            "per_rep": [
                {
                    "rep": v.rep_index, "side": v.side, "ok": v.ok,
                    "errors": v.errors, "detail": v.detail,
                }
                for v in self.verdicts
            ],
            "baseline": dict(self._baseline),
        }
