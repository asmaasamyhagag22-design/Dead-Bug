"""On-screen coaching overlay.

Design rule: the user is lying on the floor looking sideways at a screen, so
everything that matters has to be readable at a glance and from an angle. That
means a full-frame colour wash for the verdict rather than a small icon, a rep
count large enough to read across a room, and the offending body region marked
*on the body* rather than named in text.
"""

from __future__ import annotations

import cv2
import numpy as np

GREEN = (80, 220, 100)
RED = (60, 60, 235)
AMBER = (0, 190, 250)
GREY = (170, 170, 170)
DARK = (35, 35, 35)
WHITE = (255, 255, 255)


def _band(frame_w: int, mask_w: int, x0: int, x1: int) -> tuple[int, int]:
    """Scale a mask-space column range up to frame space."""
    scale = frame_w / max(1, mask_w)
    return int(x0 * scale), int(x1 * scale)


def draw_verdict_border(frame: np.ndarray, colour, thickness: int = 14) -> np.ndarray:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), colour, thickness)
    return frame


def tint(frame: np.ndarray, colour, alpha: float = 0.16) -> np.ndarray:
    """Wash the whole frame -- visible in peripheral vision from the floor."""
    layer = np.full_like(frame, colour, dtype=np.uint8)
    cv2.addWeighted(layer, alpha, frame, 1 - alpha, 0, dst=frame)
    return frame


def highlight_region(
    frame: np.ndarray, kpts: np.ndarray, mask_shape: tuple[int, int],
    region: str, colour=RED, alpha: float = 0.38,
) -> np.ndarray:
    """Mark the body band the failing measurement came from.

    The window is the same one the signal is computed over, so the highlight
    is literally where the number came from -- not an illustration of it.
    """
    from ..signals.lumbar import band_window, to_mask_pixels, torso_frame

    windows = {"lumbar": (0.00, 0.35), "pelvis": (-0.15, 0.15), "ribs": (0.45, 0.75)}
    if region not in windows:
        return frame

    kpx = to_mask_pixels(kpts[None, ...], mask_shape)[0]
    hip_c, u, torso = torso_frame(kpx)
    if not np.isfinite(torso) or torso <= 0:
        return frame

    start, end = windows[region]
    x0, x1 = band_window(hip_c, u, float(torso), start, end, mask_shape[1])
    if x1 <= x0:
        return frame

    fx0, fx1 = _band(frame.shape[1], mask_shape[1], x0, x1)
    overlay = frame.copy()
    cv2.rectangle(overlay, (fx0, 0), (fx1, frame.shape[0]), colour, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)
    cv2.line(frame, (fx0, 0), (fx0, frame.shape[0]), colour, 2)
    cv2.line(frame, (fx1, 0), (fx1, frame.shape[0]), colour, 2)
    return frame


def _text(frame, text, org, scale=0.8, colour=WHITE, thickness=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                thickness, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, state, fps: float = 0.0) -> np.ndarray:
    """Counter, phase prompt and the last verdict."""
    from .session import Phase

    h, w = frame.shape[:2]
    panel = frame[0:96, 0:w]
    cv2.addWeighted(np.full_like(panel, DARK, np.uint8), 0.55, panel, 0.45, 0, dst=panel)

    if state.phase in (Phase.ACTIVE, Phase.DONE):
        _text(frame, f"{state.reps_correct}", (20, 74), scale=2.0, colour=GREEN, thickness=4)
        width = cv2.getTextSize(f"{state.reps_correct}", cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)[0][0]
        _text(frame, f"/ {state.reps_done}", (28 + width, 74), scale=1.0, colour=GREY)
        _text(frame, "correct", (24, 26), scale=0.5, colour=GREY, thickness=1)
    else:
        _text(frame, state.prompt, (20, 62), scale=1.1, colour=AMBER, thickness=3)
        if state.progress > 0:
            bar_w = int((w - 40) * min(1.0, state.progress))
            cv2.rectangle(frame, (20, 78), (20 + bar_w, 86), AMBER, -1)
            cv2.rectangle(frame, (20, 78), (w - 20, 86), GREY, 1)

    verdict = state.last_verdict
    if verdict is not None and state.phase is Phase.ACTIVE:
        colour = GREEN if verdict.ok else RED
        label = "GOOD" if verdict.ok else verdict.message
        _text(frame, label, (20, h - 28), scale=0.95, colour=colour, thickness=2)

    if fps:
        _text(frame, f"{fps:4.1f} fps", (w - 130, 30), scale=0.55, colour=GREY, thickness=1)
    return frame


def draw_gauge(frame: np.ndarray, state, x: int | None = None) -> np.ndarray:
    """A live bar for how far the lower back is from its own baseline.

    Shown in sigmas rather than raw units: the number the user needs is "how
    unusual is this for me", and the raw gap is meaningless without the baseline.
    """
    if not np.isfinite(state.live_z):
        return frame
    h, w = frame.shape[:2]
    x = x if x is not None else w - 60
    top, bottom = 120, h - 120
    cv2.rectangle(frame, (x, top), (x + 26, bottom), GREY, 1)

    frac = float(np.clip((state.live_z + 1.0) / 5.0, 0.0, 1.0))
    fill_top = int(bottom - frac * (bottom - top))
    colour = GREEN if state.live_z < 2.5 else RED
    cv2.rectangle(frame, (x + 1, fill_top), (x + 25, bottom - 1), colour, -1)

    limit = int(bottom - (3.5 / 5.0) * (bottom - top))
    cv2.line(frame, (x - 6, limit), (x + 32, limit), AMBER, 2)
    _text(frame, "back", (x - 12, top - 12), scale=0.5, colour=GREY, thickness=1)
    return frame


def render(
    frame: np.ndarray, state, kpts=None, mask_shape=None, fps: float = 0.0
) -> np.ndarray:
    """Compose the full overlay for one frame."""
    from .session import Phase

    verdict = state.last_verdict
    if state.phase is Phase.ACTIVE and verdict is not None:
        colour = GREEN if verdict.ok else RED
        tint(frame, colour, alpha=0.10 if verdict.ok else 0.18)
        draw_verdict_border(frame, colour)
        if not verdict.ok and verdict.region and kpts is not None and mask_shape:
            highlight_region(frame, kpts, mask_shape, verdict.region)

    draw_hud(frame, state, fps)
    if state.phase is Phase.ACTIVE:
        draw_gauge(frame, state)
    return frame


def render_report(report: dict, size: tuple[int, int] = (720, 1000)) -> np.ndarray:
    """A closing summary card: how many were right, and what went wrong."""
    h, w = size
    card = np.full((h, w, 3), 28, np.uint8)
    _text(card, "Session report", (40, 70), scale=1.2, thickness=3)

    total = report["total_reps"]
    correct = report["correct_reps"]
    _text(card, f"{correct} / {total} correct", (40, 150), scale=1.5,
          colour=GREEN if total and correct == total else AMBER, thickness=3)
    if total:
        _text(card, f"{100 * correct / total:.0f}%", (40, 205), scale=0.9, colour=GREY)

    y = 280
    _text(card, "What went wrong", (40, y), scale=0.8, colour=GREY, thickness=2)
    y += 46
    from .session import ERRORS

    if not report["error_counts"]:
        _text(card, "Nothing - every rep was clean", (56, y), scale=0.7, colour=GREEN)
    else:
        for error, count in sorted(
            report["error_counts"].items(), key=lambda kv: -kv[1]
        ):
            label = ERRORS.get(error, (error, None))[0]
            _text(card, f"{count:>3}x  {label}", (56, y), scale=0.62, colour=RED, thickness=1)
            y += 40

    y = min(h - 60, y + 40)
    _text(card, "Not a medical device. Detects deviation from your own baseline.",
          (40, h - 40), scale=0.5, colour=GREY, thickness=1)
    return card
