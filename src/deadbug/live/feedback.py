"""Spoken coaching cues.

Two constraints shape this:

**It must never stall the camera loop.** Speech synthesis takes hundreds of
milliseconds; blocking on it would drop frames, and dropped frames break rep
detection. So synthesis runs on a worker thread and the capture loop only ever
puts a string on a queue.

**It must not talk over itself.** A coach who says "lower back" on every rep of
a set is noise the user stops hearing. Each distinct cue has a cooldown, and
if the queue is already backed up the newest message is dropped rather than
queued behind stale ones -- late advice about rep 4 is worse than silence
during rep 7.
"""

from __future__ import annotations

import queue
import threading
import time


class Speaker:
    """Non-blocking text-to-speech with per-message cooldown.

    Falls back to printing when no TTS engine is available, so the demo still
    runs on a machine without one.
    """

    def __init__(self, cooldown_s: float = 4.0, enabled: bool = True, max_queue: int = 2) -> None:
        self.cooldown_s = cooldown_s
        self.enabled = enabled
        self._queue: queue.Queue[str] = queue.Queue(maxsize=max_queue)
        self._last_said: dict[str, float] = {}
        self._stop = threading.Event()
        self._engine_ok = False
        self._thread: threading.Thread | None = None
        if enabled:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def _worker(self) -> None:
        engine = None
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            self._engine_ok = True
        except Exception:  # noqa: BLE001 -- any TTS failure degrades to printing
            self._engine_ok = False

        while not self._stop.is_set():
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if engine is not None:
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception:  # noqa: BLE001
                    print(f"[voice] {text}")
            else:
                print(f"[voice] {text}")

    def say(self, text: str, key: str | None = None, force: bool = False) -> bool:
        """Queue a cue. Returns False if it was suppressed.

        ``key`` groups messages for cooldown purposes, so rewording the same
        advice does not bypass it.
        """
        if not self.enabled or not text:
            return False
        key = key or text
        now = time.monotonic()
        if not force and now - self._last_said.get(key, float("-inf")) < self.cooldown_s:
            return False
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            return False               # drop rather than fall behind
        self._last_said[key] = now
        return True

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


class CoachPolicy:
    """Decides what to say and when, from the session state.

    Kept apart from the session logic so the judging rules and the talking
    rules can be reasoned about -- and changed -- independently.
    """

    def __init__(self, speaker: Speaker) -> None:
        self.speaker = speaker
        self._last_phase = None
        self._last_count = 0
        self._streak = 0

    def on_state(self, state) -> None:
        from .session import Phase

        if state.phase is not self._last_phase:
            self._announce_phase(state.phase)
            self._last_phase = state.phase

        if state.reps_done == self._last_count:
            return
        self._last_count = state.reps_done

        verdict = state.last_verdict
        if verdict is None:
            return

        if verdict.ok:
            self._streak += 1
            # Count aloud so the user can keep their eyes closed and still
            # follow along; praise only occasionally, so it stays meaningful.
            self.speaker.say(str(state.reps_correct), key="count", force=True)
            if self._streak in (5, 10, 15, 20):
                self.speaker.say("Good, keep that form", key="praise")
        else:
            self._streak = 0
            self.speaker.say(verdict.message, key=verdict.errors[0])

    def _announce_phase(self, phase) -> None:
        from .session import Phase

        messages = {
            Phase.SETUP: "Lie on your back and stay still",
            Phase.BASELINE: "Now do three slow reps",
            Phase.ACTIVE: "Start",
            Phase.DONE: "Session complete",
        }
        if phase in messages:
            self.speaker.say(messages[phase], key=f"phase:{phase.value}", force=True)
