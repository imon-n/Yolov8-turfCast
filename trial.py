import cv2
import base64
import numpy as np
import time
from ultralytics import YOLO
import socketio

# ================= CONFIG =================
DETECT_EVERY = 6            # YOLO runs every N frames (CPU friendly)
SOCKET_EVERY = 6
EMA_ALPHA = 0.2
SWITCH_COOLDOWN = 4

# --- Detection & display sizes ---
DETECT_SIZE = (960, 540)     # Used for YOLO (faster, reliable)
DISPLAY_SIZE = (640, 360)    # Only for visualization

# --- Confidence thresholds ---
NORMAL_CONF = 0.25
LOW_CONF = 0.15              # Used only when ball is lost

# --- Switching thresholds ---
NORMAL_SWITCH_THRESHOLD = 1.1
LOW_SWITCH_THRESHOLD = 1.02

NO_BALL_EPS = 0.02
JPEG_QUALITY = 90
# =========================================

# Load COCO classes
with open("utils/coco.txt", "r") as f:
    class_list = f.read().split("\n")

SPORTS_BALL_ID = class_list.index("sports ball")

# Load YOLO (CPU)
model = YOLO("weights/yolov8n.pt")

# Open videos
# caps = [
#     cv2.VideoCapture("inference/videos/trial1_cam11.mp4"),
#     cv2.VideoCapture("inference/videos/trial1_cam2.mp4"),
# ]

# caps = [
#     cv2.VideoCapture("inference/videos/cam11.mp4"),
#     cv2.VideoCapture("inference/videos/cam22.mp4"),
# ]

caps = [
    cv2.VideoCapture("inference/videos/cam1.mp4"),
    cv2.VideoCapture("inference/videos/cam2.mp4"),
]

# Open videos
# caps = [
#     cv2.VideoCapture("inference/videos/0t1.mp4"),
#     cv2.VideoCapture("inference/videos/0t2.mp4"),
# ]


if not all(c.isOpened() for c in caps):
    print(" Video not found")
    exit()

# Socket.IO
sio = socketio.Client()
sio.connect("http://localhost:5000")
print("Connected to server")

def send_best_frame(frame):
    _, buffer = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    sio.emit("bestFrame", base64.b64encode(buffer).decode())

# ================= STATE =================
frame_id = 0
ema_scores = [0.0] * len(caps)
last_boxes = [None] * len(caps)
active_cam = 0
last_switch_time = time.time()
# =========================================

while True:
    frames = []

    # Detect if ball is lost (from previous frame)
    no_ball_detected = all(score < NO_BALL_EPS for score in ema_scores)
    current_conf = LOW_CONF if no_ball_detected else NORMAL_CONF

    for i, cap in enumerate(caps):
        ret, frame = cap.read()

        if not ret:
            frame = np.zeros((DETECT_SIZE[1], DETECT_SIZE[0], 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, DETECT_SIZE)

        # -------- YOLO DETECTION --------
        if frame_id % DETECT_EVERY == 0:
            results = model.predict(
                frame,
                conf=current_conf,
                verbose=False
            )

            boxes = results[0].boxes
            best_area = 0
            best_box = None

            if boxes is not None:
                for box in boxes:
                    if int(box.cls[0]) == SPORTS_BALL_ID:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area = (x2 - x1) * (y2 - y1)

                        if area > best_area:
                            best_area = area
                            best_box = (x1, y1, x2, y2)

            ema_scores[i] = (
                EMA_ALPHA * best_area +
                (1 - EMA_ALPHA) * ema_scores[i]
            )
            last_boxes[i] = best_box

        # Draw last known box
        display = frame.copy()
        if last_boxes[i]:
            x1, y1, x2, y2 = last_boxes[i]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        frames.append(cv2.resize(display, DISPLAY_SIZE))

    frame_id += 1

    # -------- CAMERA SWITCH LOGIC --------
    best_idx = int(np.argmax(ema_scores))
    now = time.time()

    current_switch_threshold = (
        LOW_SWITCH_THRESHOLD if no_ball_detected
        else NORMAL_SWITCH_THRESHOLD
    )

    if (
        best_idx != active_cam and
        ema_scores[best_idx] > ema_scores[active_cam] * current_switch_threshold and
        now - last_switch_time > SWITCH_COOLDOWN
    ):
        active_cam = best_idx
        last_switch_time = now

    # -------- DISPLAY INPUT --------
    cv2.imshow("Input Videos", np.hstack(frames))

    # -------- BEST CAMERA VIEW --------
    best_frame = frames[active_cam].copy()

    status = "SEARCHING" if no_ball_detected else "TRACKING"
    color = (0, 255, 255) if no_ball_detected else (0, 255, 0)

    cv2.putText(
        best_frame,
        status,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    cv2.putText(
        best_frame,
        f"BEST CAMERA: {active_cam + 1}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.imshow("Best Camera View", best_frame)

    # -------- SOCKET SEND --------
    if frame_id % SOCKET_EVERY == 0:
        send_best_frame(best_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
for cap in caps:
    cap.release()

cv2.destroyAllWindows()
sio.disconnect()
