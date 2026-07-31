"""The shared engine's contract, and the web app's wiring.

None of these load MediaPipe. Constructing a backbone needs the 30 MB model file
and several seconds, which would make the suite something nobody runs. What is
tested here is the part that was actually wrong: the clock, the window sizing,
and whether the app's plumbing agrees with the engine's shape.

The regression that motivates most of this file: ``run_live`` drove the counter
with ``time.monotonic()`` for every source while sizing the counter's windows in
source frames. On a machine slower than the video, every duration measured long,
``max_extend_s`` rejected real reps, and a clip with 4 reps counted 0.

Two of these were checked by reverting the fix and confirming they fail --
``test_counter_windows_are_trimmed_by_time_not_sample_count`` and
``test_session_leaves_setup_when_the_clock_starts_at_exactly_zero``. The rest
are contract tests, not regression tests, and are labelled as such rather than
dressed up as coverage they do not provide.
"""

from __future__ import annotations

import numpy as np
import pytest

from deadbug.live.counter import DualCounter, OnlineRepCounter
from deadbug.live.engine import contralateral
from deadbug.live.session import LiveSession, Phase, SessionConfig
from deadbug.pose import skeleton as sk

FPS = 24.0


def _pose(wrist_ankle_px: float, frame=(640, 352)) -> np.ndarray:
    """A supine pose with a chosen right-wrist-to-left-ankle distance, normalized."""
    k = np.zeros((33, 4))
    k[:, 3] = 0.99
    torso_px = 100.0
    hip = np.array([300.0, 176.0])
    sho = hip + np.array([torso_px, 0.0])
    for j, p in {
        sk.L_HIP: hip + [0, -12], sk.R_HIP: hip + [0, 12],
        sk.L_SHOULDER: sho + [0, -12], sk.R_SHOULDER: sho + [0, 12],
        sk.L_ANKLE: hip - [80, 0], sk.R_ANKLE: hip - [80, 30],
        sk.L_WRIST: sho + [60, -20], sk.R_WRIST: hip - [80, 0] + [wrist_ankle_px, 0],
    }.items():
        k[j, :2] = p
    k[:, 0] /= frame[0]
    k[:, 1] /= frame[1]
    return k


# ----------------------------------------------------------------------
# contralateral: the scale bug that stopped the counter firing
# ----------------------------------------------------------------------


def test_contralateral_is_in_torso_lengths_not_normalized_units():
    """Pixel conversion first, then divide by torso length.

    Feeding raw normalized units silently mismatches the counter's `min_rise`
    floor, which is expressed in torso lengths, and the counter simply never
    fires.
    """
    frame = (640, 352)
    out = contralateral(_pose(200.0, frame), frame)
    # R wrist sits 200 px from L ankle along x; torso is 100 px. So 2 torso lengths.
    assert out["R"] == pytest.approx(2.0, rel=1e-6)


def test_contralateral_is_aspect_corrected():
    """A pure aspect change must not change a distance measured in torso lengths."""
    k = _pose(200.0, (640, 352))
    wide = contralateral(k, (640, 352))
    # Re-encode the same geometry into a frame of different aspect: undo the old
    # normalization and apply a new one.
    k2 = k.copy()
    k2[:, 0] *= 640 / 1280
    k2[:, 1] *= 352 / 720
    tall = contralateral(k2, (1280, 720))
    assert tall["R"] == pytest.approx(wide["R"], rel=1e-9)


def test_contralateral_returns_empty_on_a_degenerate_torso():
    k = np.zeros((33, 4))
    assert contralateral(k, (640, 352)) == {}


# ----------------------------------------------------------------------
# the clock
# ----------------------------------------------------------------------


def _drive(counter: DualCounter, n_reps: int, sample_hz: float, drift: float = 0.0):
    """Run ``n_reps`` 2 s reps, sampled at ``sample_hz`` samples/second.

    ``now`` is always honest seconds. What varies is how many samples arrive per
    second -- exactly what changes when the loop cannot keep up with the source.

    ``drift`` slowly raises the resting level between reps. That is what makes
    the trough window's *length* observable: a window that reaches too far back
    anchors the trough to a stale, lower resting level and reports a larger rise
    than actually happened. A perfectly periodic signal hides the difference,
    which is why this fixture is not periodic.
    """
    n = int(sample_hz * 2)                     # samples in one 2 s rep
    for rep in range(n_reps):
        for i in range(n):
            k = rep * n + i
            t = k / sample_hz                  # real seconds, always
            reach = 1.0 + drift * t + 1.2 * np.sin(np.pi * (i / n)) ** 2
            counter.update({"R": reach, "L": 1.0}, k, t)
    return counter.count


def test_counter_counts_clean_reps_at_a_low_sample_rate():
    """A sanity floor: the counter works when samples arrive slowly.

    MediaPipe heavy runs at ~12 fps here, so the counter is routinely fed far
    fewer samples per second than the source's frame rate. It must still count.
    """
    assert _drive(DualCounter(fps=6.0), 4, sample_hz=6.0) == 4


def test_counter_windows_are_trimmed_by_time_not_sample_count():
    c = OnlineRepCounter(side="R", fps=FPS, window_s=2.0, trough_window_s=1.0)
    # 200 samples spanning 10 s: far more samples than 2 s * 24 fps = 48.
    for i in range(200):
        c.update(1.0 + 0.001 * i, i, i * 0.05)
    span = c._history[-1][1] - c._history[0][1]
    assert span <= 2.0 + 0.05, f"history spans {span:.2f}s, window is 2.0s"


def test_session_leaves_setup_when_the_clock_starts_at_exactly_zero():
    """Regression: `now - (self._t0 or now)` pinned elapsed at 0.0.

    t0 is a timestamp and 0.0 is a valid one, but Python treats it as falsy. A
    file replay counts media time from frame 0, so t0 was exactly 0.0 and the
    session never left SETUP. time.monotonic() is never exactly zero, which is
    why it stayed hidden.
    """
    session = LiveSession(fps=FPS, config=SessionConfig(setup_seconds=1.0, baseline_reps=2))
    for frame in range(int(FPS * 3)):
        state = session.update({"R": 1.0, "L": 1.0}, 0.01, 0.0, frame, frame / FPS)
    assert state.phase is not Phase.SETUP, "session never left SETUP on a zero-based clock"


# ----------------------------------------------------------------------
# web app wiring
# ----------------------------------------------------------------------


def test_frame_payload_is_json_safe():
    """NaN is not JSON. The socket must emit null, not a bare NaN token."""
    import json

    from deadbug.webapp.server import _sanitise

    payload = _sanitise({
        "a": float("nan"), "b": np.float32(1.5), "c": np.int64(3),
        "d": np.array([1.0, float("inf")]), "e": {"f": float("-inf")},
    })
    text = json.dumps(payload)          # raises ValueError on NaN with allow_nan=False
    json.loads(text)
    assert payload["a"] is None
    assert payload["b"] == 1.5
    assert payload["c"] == 3
    assert payload["d"] == [1.0, None]
    assert payload["e"]["f"] is None


def test_decode_rejects_junk_instead_of_raising():
    from deadbug.webapp.server import _decode

    assert _decode("") is None
    assert _decode("data:image/jpeg;base64,!!!not-base64!!!") is None
    assert _decode("data:image/jpeg;base64,") is None


def test_decode_round_trips_a_real_jpeg():
    import base64

    import cv2

    from deadbug.webapp.server import _decode

    img = np.full((32, 48, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    url = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    out = _decode(url)
    assert out is not None and out.shape == (32, 48, 3)


def test_triage_warns_about_an_oblique_camera():
    """The app must say when the back-arch check is not trustworthy."""
    from deadbug.config import load_config
    from deadbug.webapp.analysis import assess

    cfg = load_config()
    frame = (640, 352)
    # Hips well separated in y -> a high view score -> not a side view.
    k = np.stack([_pose(200.0, frame) for _ in range(30)])
    k[:, sk.L_HIP, 1] = 130 / frame[1]
    k[:, sk.R_HIP, 1] = 222 / frame[1]

    triage = assess(k, frame, cfg)
    assert triage.view != "side"
    assert any("camera angle" in w for w in triage.warnings)


def test_triage_reports_a_clean_side_view_without_warnings():
    from deadbug.config import load_config
    from deadbug.webapp.analysis import assess

    cfg = load_config()
    frame = (640, 352)
    k = np.stack([_pose(200.0, frame) for _ in range(30)])
    # Hips nearly coincident in projection: a true side view.
    k[:, sk.L_HIP, 1] = 176 / frame[1]
    k[:, sk.R_HIP, 1] = 178 / frame[1]

    triage = assess(k, frame, cfg)
    assert triage.view == "side"
    assert triage.detection_rate == pytest.approx(1.0)
    assert triage.warnings == []
