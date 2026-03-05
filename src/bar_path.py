from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = Path("squat.MOV")
WEIGHTS = Path("weights/best.pt")

CONF = 0.25
TARGET_FPS = 15  # lower for faster, higher for smoother

# Your dataset: 3 classes, barbell class index assumed 0
BARBELL_CLS = 0

KP0 = 0  # front plate
KP1 = 1  # side plate

# Trail settings
MAX_POINTS = 5000          # keep last N points (large is fine)
DRAW_EVERY_N_POINTS = 1    # draw every point or subsample the trail


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
MAGENTA = (255, 0, 255)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)

def text_bg_outline(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font=cv2.FONT_HERSHEY_PLAIN,
    scale: float = 2,
    text_color: Tuple[int, int, int] = GREEN,
    thickness: int = 1,
    bg_color: Tuple[int, int, int] = WHITE,
    pad: int = 4,
) -> None:
    """Draw text with a filled background + outline box (similar spirit to repo). :contentReference[oaicite:2]{index=2}"""
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    # background rect
    cv2.rectangle(img, (x - pad, y + pad), (x + w + pad, y - h - pad), bg_color, -1)
    # outline rect
    cv2.rectangle(img, (x - pad, y + pad), (x + w + pad, y - h - pad), text_color, 2, cv2.LINE_AA)
    # text
    cv2.putText(img, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)

def draw_trail(
    img: np.ndarray,
    pts: Deque[Tuple[int, int]],
    color: Tuple[int, int, int] = RED,
    thickness: int = 3,
    alpha: float = 0.65,
) -> None:
    """Draw a polyline trail with transparency."""
    if len(pts) < 2:
        return
    overlay = img.copy()
    arr = np.array(list(pts)[::DRAW_EVERY_N_POINTS], dtype=np.int32)
    cv2.polylines(overlay, [arr], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def pick_best_barbell(result, barbell_cls: int) -> Optional[int]:
    """Return index of best barbell detection in this frame, else None."""
    if result.boxes is None or len(result.boxes) == 0:
        return None
    cls = result.boxes.cls.detach().cpu().numpy().astype(int)
    conf = result.boxes.conf.detach().cpu().numpy()
    idxs = np.where(cls == barbell_cls)[0]
    if idxs.size == 0:
        return None
    return int(idxs[np.argmax(conf[idxs])])

def get_center_from_kpts(result, det_i: int, kp0: int, kp1: int) -> Optional[Tuple[int, int]]:
    """Return centroid (int x,y) from two keypoints for detection det_i."""
    if result.keypoints is None:
        return None
    kxy = result.keypoints.xy.detach().cpu().numpy()  # (N, K, 2)
    if det_i >= kxy.shape[0]:
        return None
    if kxy.shape[1] <= max(kp0, kp1):
        return None

    p0 = kxy[det_i, kp0]
    p1 = kxy[det_i, kp1]

    # If a keypoint is missing/invalid, YOLO usually puts 0,0 or something weird.
    # Basic sanity check:
    if (p0 <= 1).all() or (p1 <= 1).all():
        return None

    c = (p0 + p1) / 2.0
    return int(c[0]), int(c[1])

def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS}")

    model = YOLO(str(WEIGHTS))

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(native_fps / TARGET_FPS)))

    pts: Deque[Tuple[int, int]] = deque(maxlen=MAX_POINTS)

    draw_line = True
    show_model_ann = False  # toggle whether to show YOLO's own plotted boxes/kpts
    paused = False

    frame_i = 0
    last_time = time.time()

    print("Controls:")
    print("  l = toggle line trail")
    print("  a = toggle YOLO annotations (boxes/kpts)")
    print("  c = clear trail")
    print("  space = pause/resume")
    print("  q or esc = quit")

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_i += 1

            if frame_i % step == 0:
                # Inference
                res = model.predict(frame, conf=CONF, verbose=False)[0]

                det_i = pick_best_barbell(res, BARBELL_CLS)
                if det_i is not None:
                    center = get_center_from_kpts(res, det_i, KP0, KP1)
                    if center is not None:
                        pts.append(center)

                # Choose display frame
                disp = res.plot() if show_model_ann else frame.copy()
            else:
                disp = frame.copy()
        else:
            # if paused, just keep showing the same frame
            # (OpenCV will still respond to keypress)
            # NOTE: disp must exist; handle initial pause edge-case
            disp = disp if "disp" in locals() else np.zeros((480, 640, 3), dtype=np.uint8)

        # Draw trail
        if draw_line:
            draw_trail(disp, pts, color=RED, thickness=3, alpha=0.70)

        # HUD
        now = time.time()
        fps_est = 1.0 / max(1e-6, (now - last_time))
        last_time = now

        status = f"trail={'ON' if draw_line else 'OFF'} | ann={'ON' if show_model_ann else 'OFF'} | pts={len(pts)} | (press 'l' 'a' 'c' space 'q')"
        text_bg_outline(disp, status, (10, 30), text_color=MAGENTA if draw_line else ORANGE, bg_color=WHITE)

        cv2.imshow("Bar Path Live", disp)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):  # q or ESC
            break
        elif key == ord('l'):
            draw_line = not draw_line
        elif key == ord('a'):
            show_model_ann = not show_model_ann
        elif key == ord('c'):
            pts.clear()
        elif key == ord(' '):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
