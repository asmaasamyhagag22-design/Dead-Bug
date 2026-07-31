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
from deadbug.ingest.video_source import snap  # noqa: E402
from deadbug.live import ui  # noqa: E402
from deadbug.live.engine import CoachEngine  # noqa: E402
from deadbug.live.feedback import CoachPolicy, Speaker  # noqa: E402
from deadbug.pose.draw import draw_skeleton  # noqa: E402


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

    def new_engine() -> CoachEngine:
        return CoachEngine(
            cfg, fps=fps, frame_size=size,
            setup_seconds=args.setup_seconds, baseline_reps=args.baseline_reps,
        )

    speaker = Speaker(enabled=not args.no_voice)
    coach = CoachPolicy(speaker)
    engine = new_engine()

    frame_index = 0
    t_start = time.monotonic()
    smoothed_fps = 0.0
    paused = False
    state = None

    window = "Dead Bug Coach"
    if not args.headless:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, size[0], size[1])

    try:
        while True:
            if args.max_frames and frame_index >= args.max_frames:
                break
            key = (cv2.waitKey(1) & 0xFF) if not args.headless else 255
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
            if key == ord("r"):
                engine.close()
                engine = new_engine()
                coach = CoachPolicy(speaker)
                frame_index = 0
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
            # THE CLOCK THE COUNTER IS JUDGED ON.
            #
            # For a camera the wall clock is the truth. For a file it is not:
            # the elapsed wall time depends on how fast this machine runs
            # MediaPipe, not on the movement. Measured here, heavy runs at
            # ~12 fps against a 24 fps clip, so every extension phase measured
            # twice its real length, max_extend_s = 6.0 rejected reps that took
            # three seconds, and videoplayback (4) counted 0 where the offline
            # pipeline finds 4.
            #
            # Media time also makes a file replay deterministic, which is what
            # the module docstring promises: the demo can be rehearsed and
            # debugged without a camera and give the same answer every time.
            now = (time.monotonic() - t_start) if is_live else (frame_index / fps)

            result = engine.process(frame, frame_index, now)
            state = result.state
            coach.on_state(state)

            if result.detected:
                draw_skeleton(frame, result.kpts, vis_threshold=0.5)
            dt = time.perf_counter() - t0
            smoothed_fps = 0.9 * smoothed_fps + 0.1 * (1.0 / max(dt, 1e-6))

            if not result.detected:
                ui._text(frame, "no person detected", (20, size[1] // 2),
                         scale=0.9, colour=ui.AMBER)
            ui.render(frame, state, kpts=result.kpts,
                      mask_shape=result.mask_shape, fps=smoothed_fps)
            if not args.headless:
                cv2.imshow(window, frame)
            elif frame_index % 60 == 0:
                print(f"  f{frame_index:5d}  {state.phase.value:9s} "
                      f"reps={state.reps_done:2d} ok={state.reps_correct:2d} "
                      f"gap={result.lumbar_gap:.5f}  {smoothed_fps:4.1f} fps")
            frame_index += 1
    finally:
        cap.release()

    engine.finish()
    speaker.close()

    session = engine.session
    report = engine.report()
    out = REPO_ROOT / args.report
    out.parent.mkdir(parents=True, exist_ok=True)
    import json

    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{report['correct_reps']} / {report['total_reps']} reps correct")
    for error, count in sorted(report["error_counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {count:>3}x  {error}")

    if not report["total_reps"]:
        # Distinguish "nothing happened" from "the protocol ate the clip". Only
        # reps performed after calibration AND after the personal baseline are
        # judged, so a short clip can legitimately produce no verdicts while the
        # counter was working perfectly.
        counted = session.counter.count
        print(
            f"\nNo rep was judged. The session reached phase '{state.phase.value}' "
            f"and the counter saw {counted} rep(s).\n"
            f"  {args.setup_seconds:.0f}s went to floor calibration and the next "
            f"{args.baseline_reps} rep(s) set your personal baseline;\n"
            f"  only reps after that are scored. On a clip this short that can "
            f"leave nothing over.\n"
            f"  Use a longer clip, or lower --setup-seconds / --baseline-reps."
        )
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
