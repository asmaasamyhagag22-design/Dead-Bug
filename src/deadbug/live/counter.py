"""Causal rep counting — detects reps as they happen, with no lookahead.

The offline segmenter in :mod:`deadbug.segment.reps` runs ``find_peaks`` over a
whole recorded signal and then walks outward to the minima on either side of
each peak. None of that is available live: at the moment a rep is at its peak,
the code has not yet seen the descent that proves it *was* a peak.

So this is a state machine instead:

    WAITING ──rise > prominence──▶ RISING ──fall > confirm_drop──▶ rep emitted

A rep is confirmed only once the signal has come back down by a fraction of its
own rise. That costs a small, unavoidable delay -- roughly the time it takes to
descend 40% of the way -- which is why feedback lands just after the top of the
movement rather than at it. Any "instant" alternative would have to guess, and
would miscount every time a person paused at the top.

The amplitude threshold adapts from a rolling window rather than being fixed,
because reach varies with limb length and with how far a given person actually
extends. It is the online counterpart of ``segment.reps.auto_prominence``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class State(Enum):
    WAITING = "waiting"
    RISING = "rising"


@dataclass
class LiveRep:
    """One completed extension, as observed live."""

    side: str
    start_frame: int
    peak_frame: int
    end_frame: int
    start_time: float
    peak_time: float
    end_time: float
    amplitude: float
    metrics: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        """Trough to confirmation.

        **Biased late** and unsuitable for judging tempo: confirmation only
        happens once the signal has fallen far enough to prove the peak passed,
        so this always overstates the movement by the confirmation delay. Use
        :attr:`t_extend_s` for anything the user is scored on.
        """
        return self.end_time - self.start_time

    @property
    def t_extend_s(self) -> float:
        """Trough to peak -- the extension phase.

        Both endpoints are observed directly rather than inferred, so this
        carries no confirmation lag. It is the honest tempo measure online.
        """
        return self.peak_time - self.start_time


class OnlineRepCounter:
    """Counts extensions in one side's contralateral distance signal.

    Args:
        window_s: rolling window used to estimate the signal's range.
        prominence_frac: rise, as a fraction of the observed range, that starts
            a rep. Mirrors the offline ``0.25 * (p95 - p05)`` rule.
        confirm_frac: fall, as a fraction of the rep's own rise, that confirms
            the peak has passed.
        min_rep_s: refractory period; a second peak sooner than this is the same
            movement wobbling, not a new rep.
        min_rise: absolute floor on the trigger, in normalized torso lengths.
            This bounds the *threshold*, not the estimated range -- flooring the
            range instead lets ``prominence_frac`` scale it back down, and a
            motionless subject then generates reps out of detector jitter. A
            real extension moves the wrist-to-ankle distance far more than this,
            so the floor never suppresses a genuine rep.
    """

    def __init__(
        self,
        side: str,
        fps: float = 30.0,
        window_s: float = 12.0,
        prominence_frac: float = 0.25,
        confirm_frac: float = 0.40,
        min_rep_s: float = 0.8,
        min_rise: float = 0.05,
        trough_window_s: float = 4.0,
        max_extend_s: float = 6.0,
    ) -> None:
        self.side = side
        self.fps = fps
        self.prominence_frac = prominence_frac
        self.confirm_frac = confirm_frac
        self.min_rep_s = min_rep_s
        self.min_rise = min_rise
        #: A dead bug extension takes roughly one to four seconds. A rise slower
        #: than this is not a rep -- it is the subject repositioning, sitting up
        #: to talk, or drifting between sets. Without the bound, an instructional
        #: video (mostly explanation, a handful of actual reps) yields "reps"
        #: with 30-45 second extension phases, and every tempo judgement built on
        #: them is meaningless. This is a physiological prior, not a tuned knob.
        self.max_extend_s = max_extend_s

        self.window_s = window_s
        self.trough_window_s = trough_window_s

        # Both windows are trimmed by ELAPSED TIME, not by a sample count.
        #
        # Sizing them as `seconds * fps` assumes one sample arrives per source
        # frame, which is true only when the loop keeps up with the video. It
        # does not: measured here, MediaPipe heavy runs at ~12 fps against a
        # 24 fps clip, so a deque holding `4.0 * 24 = 96` samples spans 8 real
        # seconds, not 4. Every duration the counter derives is then wrong by
        # the ratio of nominal to achieved frame rate -- and `max_extend_s`
        # rejects reps on exactly such a duration, which is how a clip with 4
        # reps counted 0.
        #
        # maxlen stays as a memory bound: at most `seconds * fps` samples can
        # arrive within the window, and fewer when the loop is behind.
        self._history: deque[tuple[float, float]] = deque(maxlen=max(8, int(window_s * fps)))
        #: Recent samples used to anchor the trough. Tracking a running minimum
        #: since the last rep instead looks correct until the subject spends a
        #: while not exercising -- an instructional video is mostly talking --
        #: and then the trough stays pinned to a moment tens of seconds in the
        #: past, so the next rep reports an extension phase lasting most of the
        #: clip. Measured 45 s for a 3 s movement before this was bounded.
        self._recent: deque[tuple[float, int, float]] = deque(
            maxlen=max(4, int(trough_window_s * fps))
        )
        self._state = State.WAITING
        self._trough = float("inf")
        self._trough_frame = 0
        self._trough_time = 0.0
        self._peak = float("-inf")
        self._peak_frame = 0
        self._peak_time = 0.0
        self._last_rep_time = float("-inf")

    @property
    def state(self) -> State:
        return self._state

    def signal_range(self) -> float:
        """Rolling p5-p95 range, the basis for the adaptive threshold."""
        if len(self._history) < 8:
            return 0.0
        ordered = sorted(value for value, _t in self._history)
        lo = ordered[int(0.05 * (len(ordered) - 1))]
        hi = ordered[int(0.95 * (len(ordered) - 1))]
        return hi - lo

    def _trim(self, now: float) -> None:
        """Drop samples older than each window, measured in seconds."""
        while self._history and now - self._history[0][1] > self.window_s:
            self._history.popleft()
        while self._recent and now - self._recent[0][2] > self.trough_window_s:
            self._recent.popleft()

    def threshold(self) -> float:
        """Rise required to start a rep: adaptive, but never below ``min_rise``."""
        return max(self.min_rise, self.signal_range() * self.prominence_frac)

    def update(self, value: float, frame: int, now: float) -> LiveRep | None:
        """Feed one sample. Returns a rep at the moment one is confirmed."""
        if value != value:                       # NaN -- detection dropped out
            return None
        self._history.append((value, now))
        self._trim(now)
        threshold = self.threshold()

        if self._state is State.WAITING:
            self._recent.append((value, frame, now))
            # Anchor the trough to the lowest point in the recent window, not to
            # the lowest since the last rep. A rep's extension phase starts when
            # the limbs were last down, which is a recent event by definition.
            self._trough, self._trough_frame, self._trough_time = min(self._recent)
            if value - self._trough >= threshold:
                self._state = State.RISING
                self._peak, self._peak_frame, self._peak_time = value, frame, now
            return None

        # RISING
        if value > self._peak:
            self._peak, self._peak_frame, self._peak_time = value, frame, now
            return None

        rise = self._peak - self._trough
        if rise <= 0 or (self._peak - value) < rise * self.confirm_frac:
            return None

        rep = None
        too_slow = (self._peak_time - self._trough_time) > self.max_extend_s
        if not too_slow and now - self._last_rep_time >= self.min_rep_s:
            rep = LiveRep(
                side=self.side,
                start_frame=self._trough_frame, peak_frame=self._peak_frame,
                end_frame=frame,
                start_time=self._trough_time, peak_time=self._peak_time,
                end_time=now, amplitude=rise,
            )
            self._last_rep_time = now

        self._state = State.WAITING
        self._recent.clear()
        self._recent.append((value, frame, now))
        self._trough, self._trough_frame, self._trough_time = value, frame, now
        return rep

    def reset(self) -> None:
        self._state = State.WAITING
        self._recent.clear()
        self._trough = float("inf")
        self._peak = float("-inf")


class DualCounter:
    """Both sides at once, with the alternation check the exercise implies.

    Dead Bug alternates right-arm/left-leg, then the mirror. Two reps in a row
    on the same side usually means one was missed, so it is surfaced rather than
    silently accepted.
    """

    def __init__(self, fps: float = 30.0, **kwargs) -> None:
        self.counters = {
            "R": OnlineRepCounter("R", fps=fps, **kwargs),
            "L": OnlineRepCounter("L", fps=fps, **kwargs),
        }
        self.reps: list[LiveRep] = []
        self._last_side: str | None = None

    def update(self, values: dict[str, float], frame: int, now: float) -> list[LiveRep]:
        """Feed one sample per side; returns any reps confirmed this frame."""
        confirmed = []
        for side, counter in self.counters.items():
            rep = counter.update(values.get(side, float("nan")), frame, now)
            if rep is not None:
                if self._last_side == side:
                    rep.metrics["alternation_broken"] = True
                self._last_side = side
                self.reps.append(rep)
                confirmed.append(rep)
        return confirmed

    @property
    def count(self) -> int:
        return len(self.reps)

    def reset(self) -> None:
        for counter in self.counters.values():
            counter.reset()
        self.reps.clear()
        self._last_side = None
