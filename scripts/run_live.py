"""Live Dead Bug coach.

    python scripts/run_live.py                      # default camera
    python scripts/run_live.py --source 1           # a different camera
    python scripts/run_live.py --source clip.mp4    # a video file

The source is deliberately interchangeable: a webcam, a phone connected as a
webcam, or a recorded file all take the identical path through the pipeline.
That is not only convenient -- it means the demo can be rehearsed and debugged
without a camera present, and it means the thing shown live is the same code
that produced the offline numbers.

Keys:  q quit and show the report   r restart the session   space pause
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deadbug.config import cfg_get, load_config  # noqa: E402
from deadbug.geometry import floor as FL  # noqa: E402
from deadbug.ingest.video_source import snap  # noqa: E402
from deadbug.live import ui  # noqa: E402
from deadbug.live.feedback import CoachPolicy, Speaker  # noqa: E402
from deadbug.live.session import LiveSession, Phase, SessionConfig  # noqa: E402
from deadbug.pose import skeleton as sk  # noqa: E402
from deadbug.pose.draw import draw_skeleton  # noqa: E402
from deadbug.pose.mediapipe_backbone import MediaPipeBackbone  # noqa: E402
from deadbug.signals import lumbar as LU  # noqa: E402
from deadbug.signals import rotation as ROT  # noqa: E402

SETUP_MIN_MASKS = 15


def open_source(source: str, snap_to: int = 16):
    """Open a camera index or a file path, returning (cap, size, fps, is_live)."""
    is_live = source.isdigit()
    cap = cv2.VideoCapture(int(source) if is_live else source)
    if not cap.isOpened():
        raise SystemExit(
            f"cannot open source {source!r}. "
            "No camera on this machine? Pass a video file instead, e.g. "
            "--source 'data/clips/videoplayback (3).mp4'"
        )
    if is_live:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    # Snap dimensions: MediaPipe aborts the process reading a mask when the
    # frame width is not 4-byte aligned. See ingest/video_source.py.
    return cap, (snap(width, snap_to), snap(height, snap_to)), fps, is_live


def contralateral(kpts: np.ndarray, frame_size: tuple[int, int]) -> dict[str, float]:
    """Wrist-to-opposite-ankle distance **in torso lengths**.

    Two corrections that both matter and are easy to miss:

    - Convert to pixels first. MediaPipe's normalized coordinates give x and y
      different pixel scales, and a supine subject has the torso along x with
      the limbs swinging through y, so the raw distance is skewed by the frame's
      aspect ratio.
    - Divide by the current torso length. The counter's ``min_rise`` floor is
      expressed in torso lengths so that it means the same thing for a subject
      filling the frame and one filmed from across a room. Feeding it raw
      normalized units silently mismatches the scale, and the counter simply
      never fires -- which is exactly what happened on the first end-to-end run.
    """
    k = np.asarray(kpts, dtype=np.float64).copy()
    k[:, 0] *= frame_size[0]
    k[:, 1] *= frame_size[1]

    hip_c = 0.5 * (k[sk.L_HIP, :2] + k[sk.R_HIP, :2])
    sho_c = 0.5 * (k[sk.L_SHOULDER, :2] + k[sk.R_SHOULDER, :2])
    torso = float(np.linalg.norm(sho_c - hip_c))
    if not np.isfinite(torso) or torso < 1e-6:
        return {}

    out = {}
    for side, (wrist, ankle) in sk.CONTRALATERAL_MP33.items():
        a, b = k[wrist, :2], k[ankle, :2]
        out[side] = (
            float(np.linalg.norm(a - b) / torso)
            if np.isfinite(a).all() and np.isfinite(b).all() else float("nan")
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="0", help="camera index or video path")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--setup-seconds", type=float, default=3.0)
    ap.add_argument("--baseline-reps", type=int, default=3)
    ap.add_argument("--report", default="reports/session_report.json")
    ap.add_argument("--headless", action="store_true",
                    help="no window; for testing the pipeline without a display")
    ap.add_argument("--max-frames", type=int, default=0, help="stop early (testing)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cap, size, fps, is_live = open_source(args.source, cfg_get(cfg, "ingest.snap_dims_to"))
    print(f"source={args.source}  {size[0]}x{size[1]} @ {fps:.1f} fps  live={is_live}")

    session = LiveSession(
        fps=fps,
        config=SessionConfig(
            setup_seconds=args.setup_seconds, baseline_reps=args.baseline_reps
        ),
    )
    speaker = Speaker(enabled=not args.no_voice)
    coach = CoachPolicy(speaker)

    setup_masks: list[np.ndarray] = []
    setup_kpts: list[np.ndarray] = []
    floor: dict | None = None
    rot_calib: float | None = None
    rot_history: list[float] = []

    frame_index = 0
    t_start = time.monotonic()
    last_ts = -1
    smoothed_fps = 0.0
    paused = False
    state = None

    window = "Dead Bug Coach"
    if not args.headless:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, size[0], size[1])

    with MediaPipeBackbone.from_config(cfg) as backbone:
        while True:
            if args.max_frames and frame_index >= args.max_frames:
                break
            key = (cv2.waitKey(1) & 0xFF) if not args.headless else 255
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
            if key == ord("r"):
                session = LiveSession(fps=fps, config=session.cfg)
                coach = CoachPolicy(speaker)
                setup_masks.clear(); setup_kpts.clear()
                floor = None; rot_calib = None; rot_history.clear()
                t_start = time.monotonic()
            if paused:
                continue

            ok, frame = cap.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            if is_live:
                frame = cv2.flip(frame, 1)          # mirror, so the user's left is left

            t0 = time.perf_counter()
            now = time.monotonic() - t_start
            ts_ms = max(last_ts + 1, int(now * 1000))   # must be strictly increasing
            last_ts = ts_ms

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = backbone.infer(rgb, ts_ms)
            kpts, mask = result.kpts.astype(np.float64), result.mask
            detected = np.isfinite(kpts[:, :2]).any()

            lumbar_gap = float("nan")
            rot_dev = float("nan")
            mask_shape = mask.shape[:2] if mask is not None else None

            if detected and mask is not None:
                # Floor: accumulate during setup, then freeze. A live session can
                # ask the user to lie still, which is the calibration the YouTube
                # clips never had -- so this path is strictly better than offline.
                if floor is None:
                    setup_masks.append(mask)
                    setup_kpts.append(kpts)
                    if len(setup_masks) >= SETUP_MIN_MASKS:
                        floor = FL.estimate_floor_from_config(
                            np.stack(setup_masks), np.stack(setup_kpts), cfg
                        )
                if floor is not None and np.isfinite(floor["b"]):
                    kpx = LU.to_mask_pixels(kpts[None, ...], mask_shape)
                    lumbar_gap = float(
                        LU.lumbar_gap(mask[None, ...], kpx, floor)["lumbar_gap"][0]
                    )

                proxy = float(ROT.rot_proxy(kpts[None, ...], frame_size=size)[0])
                if np.isfinite(proxy):
                    if rot_calib is None:
                        rot_history.append(proxy)
                        if len(rot_history) >= SETUP_MIN_MASKS:
                            rot_calib = float(np.median(rot_history))
                    else:
                        rot_dev = proxy - rot_calib

            state = session.update(
                contralateral(kpts, size) if detected else {},
                lumbar_gap, rot_dev, frame_index, now,
            )
            coach.on_state(state)

            if detected:
                draw_skeleton(frame, result.kpts, vis_threshold=0.5)
            dt = time.perf_counter() - t0
            smoothed_fps = 0.9 * smoothed_fps + 0.1 * (1.0 / max(dt, 1e-6))

            if not detected:
                ui._text(frame, "no person detected", (20, size[1] // 2),
                         scale=0.9, colour=ui.AMBER)
            ui.render(frame, state, kpts=kpts if detected else None,
                      mask_shape=mask_shape, fps=smoothed_fps)
            if not args.headless:
                cv2.imshow(window, frame)
            elif frame_index % 60 == 0:
                print(f"  f{frame_index:5d}  {state.phase.value:9s} "
                      f"reps={state.reps_done:2d} ok={state.reps_correct:2d} "
                      f"gap={lumbar_gap:.5f}  {smoothed_fps:4.1f} fps")
            frame_index += 1

    cap.release()
    session.finish()
    speaker.close()

    report = session.report()
    out = REPO_ROOT / args.report
    out.parent.mkdir(parents=True, exist_ok=True)
    import json

    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{report['correct_reps']} / {report['total_reps']} reps correct")
    for error, count in sorted(report["error_counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {count:>3}x  {error}")
    print(f"wrote {out}")

    if report["total_reps"]:
        card = ui.render_report(report)
        cv2.imwrite(str(out.with_suffix(".png")), card)
        print(f"wrote {out.with_suffix('.png')}")
        if not args.headless:
            cv2.imshow("Session report", card)
            cv2.waitKey(0)
    if not args.headless:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
