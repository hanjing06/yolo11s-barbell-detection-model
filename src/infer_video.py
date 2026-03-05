from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Configurations

WEIGHTS = Path("weights/best.pt")
VIDEO_PATH = Path("squat.mov")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONF = 0.25
TARGET_FPS = 10  # sample rate (e.g., 10 fps). Lower = faster, smoother later with filtering.

BARBELL_CLS = 0

# kp0 = front plate, kp1 = side plate (swap if your dataset is opposite)
KP_FRONT = 0
KP_SIDE = 1

def pick_barbell_detection(result, barbell_cls=0):
    """
    Pick the best barbell detection from a YOLO pose result.
    Returns (keypoints_xy, conf) or (None, None).
    keypoints_xy shape: (K, 2) in pixel coordinates.
    """
    if result.boxes is None or len(result.boxes) == 0:
        return None, None

    boxes = result.boxes
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()

    # Filter to barbell class detections
    idxs = np.where(cls == barbell_cls)[0]
    if idxs.size == 0:
        return None, None

    # Choose highest-confidence barbell
    best_i = idxs[np.argmax(conf[idxs])]

    if result.keypoints is None:
        return None, None

    kpts = result.keypoints.xy.cpu().numpy()  # (N, K, 2)
    return kpts[best_i], float(conf[best_i])

def main():
    model = YOLO(str(WEIGHTS))

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(round(native_fps / TARGET_FPS)))

    frame_idx = 0
    kept_idx = 0

    # Store bar path points: list of dicts with frame#, time, x, y, confidence
    path = []

    # Optional: output annotated video
    out_video_path = OUT_DIR / "annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_video_path), fourcc, TARGET_FPS, (width, height))

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        # Inference
        results = model.predict(frame_bgr, conf=CONF, verbose=False)
        r0 = results[0]

        # Find barbell + keypoints
        kpts_xy, det_conf = pick_barbell_detection(r0, BARBELL_CLS)

        if kpts_xy is not None:
            # Compute barbell centroid from the 2 plate keypoints
            front = kpts_xy[KP_FRONT]
            side = kpts_xy[KP_SIDE]
            center = (front + side) / 2.0
            cx, cy = float(center[0]), float(center[1])

            t = frame_idx / native_fps
            path.append({"frame": frame_idx, "time": t, "x": cx, "y": cy, "conf": det_conf})

        # Save annotated frame to video
        annotated_bgr = r0.plot()  # BGR image with boxes + kpts drawn
        writer.write(annotated_bgr)

        kept_idx += 1
        frame_idx += 1

    cap.release()
    writer.release()

    # Save bar path to CSV
    import csv
    csv_path = OUT_DIR / "bar_path.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "time", "x", "y", "conf"])
        w.writeheader()
        w.writerows(path)

    print(f"Annotated video: {out_video_path}")
    print(f"Bar path CSV:     {csv_path}")
    print(f"Points collected: {len(path)}")


if __name__ == "__main__":
    main()