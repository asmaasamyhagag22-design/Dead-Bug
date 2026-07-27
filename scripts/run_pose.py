import sys
import os
import time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL = "pose_landmarker_heavy.task"

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


def draw(frame, kpts, vis_thresh=0.5):
    h, w = frame.shape[:2]
    pts = np.stack([kpts[:, 0] * w, kpts[:, 1] * h], axis=1).astype(int)
    ok = kpts[:, 3] > vis_thresh
    for a, b in CONNECTIONS:
        if ok[a] and ok[b]:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (0, 200, 120), 2)
    for i in range(33):
        if ok[i]:
            cv2.circle(frame, tuple(pts[i]), 3, (30, 60, 240), -1)
    return frame


def main(video_path):
    if not os.path.exists(MODEL):
        sys.exit(f"model not found: {MODEL}")

    stem = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs("out", exist_ok=True)

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        output_segmentation_masks=True,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{W}x{H} @ {fps:.1f} fps")

    writer = cv2.VideoWriter(
        f"out/{stem}_annotated.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)
    )

    all_kpts, all_masks = [], []
    n_found, i, t0 = 0, 0, time.time()

    with vision.PoseLandmarker.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(i / fps * 1000)
            res = detector.detect_for_video(mp_img, ts_ms)

            if res.pose_landmarks:
                lm = res.pose_landmarks[0]
                kpts = np.array(
                    [[p.x, p.y, p.z, p.visibility] for p in lm],
                    dtype=np.float32
                )
                n_found += 1
                draw(frame, kpts)
            else:
                kpts = np.full((33, 4), np.nan, dtype=np.float32)

            all_kpts.append(kpts)

            if res.segmentation_masks:
                m = res.segmentation_masks[0].numpy_view()
                m = (m > 0.5).astype(np.uint8) * 255
                full = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                all_masks.append(
                    cv2.resize(m, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
                )
                edge = cv2.Canny(full, 50, 150)
                frame[edge > 0] = (255, 160, 0)

            writer.write(frame)
            i += 1
            if i % 60 == 0:
                print(f"  {i} frames...", end="\r")

    cap.release()
    writer.release()

    kpts_arr = np.stack(all_kpts)
    np.save(f"out/{stem}_kpts.npy", kpts_arr)
    if all_masks:
        masks_arr = np.stack(all_masks)
        np.save(f"out/{stem}_mask.npy", masks_arr)

    dt = time.time() - t0
    rate = 100 * n_found / max(i, 1)
    print("\n" + "=" * 46)
    print(f"frames    : {i}")
    print(f"detected  : {n_found}  ({rate:.1f}%)")
    print(f"speed     : {i / dt:.1f} fps")
    print(f"kpts      : {kpts_arr.shape}  -> out/{stem}_kpts.npy")
    if all_masks:
        print(f"mask      : {masks_arr.shape}  -> out/{stem}_mask.npy")
    print(f"video     : out/{stem}_annotated.mp4")
    print("=" * 46)

    if rate < 90:
        print("\nWARNING: detection rate below 90% - check camera angle or lighting.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python run_pose.py <video.mp4>")
    main(sys.argv[1])