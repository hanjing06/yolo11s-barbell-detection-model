# yolo11s-barbell-detection-model
how to train and deploy an ultralytics yolo model using the yolo11s

A computer vision system by @mleis4 and @hanjing06 that analyzes barbell trajectory during a squat using a the **YOLOv8 pose model** and **OpenCV-based video processing**. The system detects the barbell plates, calculates the barbell centroid, and visualizes the bar path across frames.

This project explores **real-time computer vision pipelines**, combining deep learning inference with classical computer vision techniques to analyze athletic movement (currently focusing on just squatting).

# System Architecture

The system follows a real-time computer vision pipeline:
Video input -> Frame Extraction using OpenCV -> YOLO Pose Model Inference -> Keypoint extraction via front and side plate -> Calculates barbell's centroid -> Trajectory tracking -> Visualization + data

# Model

The system uses a **YOLOv8 pose model** trained to detect:

| Class | Description |
|-----|-----|
| 0 | Barbell |
| 1 | Front Plate |
| 2 | Side Plate |

Each barbell detection contains **two keypoints** corresponding to the centers of the plates.

The barbell position is estimated using the centroid of these keypoints:

centroid = (plate_1 + plate_2) / 2

This centroid represents the effective bar position for trajectory analysis.

# Core Components

## 1. Video Processing

OpenCV handles video ingestion, playback, and rendering.

cv2.VideoCapture()
cv2.imshow()
cv2.waitKey()


Frames may optionally be **downsampled to a target FPS** to reduce inference cost.

## 2. Model Inference

Ultralytics YOLO is used for pose detection.

```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")
results = model.predict(frame)
```

Our model returns:
- bounding boxes
- keypoints
- detection confidence

## 3. Barbell Centroid Calculation

```python
front_plate = keypoints[0]
side_plate  = keypoints[1]

center = (front_plate + side_plate) / 2
```

## 4. Trajectory Tracking

```python
deque[(x, y)]
```

## 5. Visualization

- bounding box (wip)
- keypoints
- bar path trajectory
- gui (wip)

# Math :-p (for later)

drift = max(|x_i - x_start|)

`x_i` =  x bar poisiton at frame i
`x_start` = initial x position

path linearity is measured by
x = a*y + b

mean squared deviation represents path irregularity
linearity_error = mean((x_i - (a*y_i + b))²)

bar velocity is estimated from frame to frame position changes
v_i = sqrt((x_i - x_{i-1})² + (y_i - y_{i-1})²) / Δt
- concentric velocity
- eccentric velocity
- sticking points

squat depth detection
depth = max(y_i)

lift phase segmentation
descent: v_y > 0
bottom:  v_y ≈ 0
ascent:  v_y < 0

# Getting started

Dependencies:
ultralytics
opencv-python
numpy

```
pip install ultralytics opencv-python numpy
```

from this project root:
```
python src/bar_path.py
```

the system will:
1. Load the trained YOLO model

2. Parse the input video

3. Detect barbell keypoints

4. Track the bar trajectory

5. Display a live visualization window

## currently working on:
- trajectory smoothing

- velocity heatmaps

- rep counting

- automated coaching feedback

- graphical control dashboard

- multi-camera analysis



