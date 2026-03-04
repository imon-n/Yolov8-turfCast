import cv2
import base64
import numpy as np
import time
import threading
from ultralytics import YOLO
import socketio

# ================= CONFIG =================
DETECT_EVERY = 1
YOLO_FPS = 8
SOCKET_EVERY = 6

EMA_ALPHA = 0.2
SWITCH_COOLDOWN = 3

DETECT_SIZE = (960, 540)
DISPLAY_SIZE = (640, 360)

NORMAL_CONF = 0.25
LOW_CONF = 0.18

NORMAL_SWITCH_THRESHOLD = 1.1
NO_BALL_EPS = 0.02

DEFAULT_CAMERA = 0
JPEG_QUALITY = 90
NAMESPACE = "/"
# =========================================

# ================= LOAD COCO =================
with open("utils/coco.txt", "r") as f:
    class_list = f.read().split("\n")

SPORTS_BALL_ID = class_list.index("sports ball")

# ================= LOAD YOLO =================
model = YOLO("weights/yolov8n.pt")

# ================= VIDEO INPUTS (4) =================
caps = [
    cv2.VideoCapture("inference/videos/rc1.mp4"),
    cv2.VideoCapture("inference/videos/rc2.mp4"),
    cv2.VideoCapture("inference/videos/rc3.mp4"),
    cv2.VideoCapture("inference/videos/rc4.mp4"),
]

if not all(c.isOpened() for c in caps):
    print("One or more videos not found")
    exit()

# ================= SOCKET (POLLING SAFE) =================
sio = socketio.Client()

@sio.event
def connect():
    print("Socket connected (polling)")

@sio.event
def disconnect():
    print("Socket disconnected")

sio.connect(
    "http://localhost:5000",
    namespaces=[NAMESPACE]
)

def send_best_frame(frame):
    if not sio.connected:
        return

    _, buffer = cv2.imencode(
        ".jpg", frame,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )

    sio.emit(
        "bestFrame",
        base64.b64encode(buffer).decode(),
        namespace=NAMESPACE
    )

# ================= SHARED STATE =================
N = len(caps)
latest_frames = [None] * N
ema_scores = [0.0] * N
last_boxes = [None] * N

active_cam = DEFAULT_CAMERA
last_switch_time = time.time()

lock = threading.Lock()
running = True
# ==============================================

# ================= CAMERA THREAD =================
def camera_reader(idx, cap):
    global running

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = 1.0 / fps if fps > 0 else 1 / 30

    while running:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            running = False
            break

        frame = cv2.resize(frame, DETECT_SIZE)

        with lock:
            latest_frames[idx] = frame

        sleep = delay - (time.time() - start)
        if sleep > 0:
            time.sleep(sleep)

# ================= YOLO THREAD =================
def yolo_worker():
    global running
    yolo_delay = 1.0 / YOLO_FPS
    frame_id = 0

    while running:
        start = time.time()

        with lock:
            frames = latest_frames.copy()
            scores = ema_scores.copy()

        no_ball = all(s < NO_BALL_EPS for s in scores)
        conf = LOW_CONF if no_ball else NORMAL_CONF

        for i, frame in enumerate(frames):
            if frame is None:
                continue

            if frame_id % DETECT_EVERY == 0:
                results = model.predict(frame, conf=conf, verbose=False)
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

                with lock:
                    ema_scores[i] = EMA_ALPHA * best_area + (1 - EMA_ALPHA) * ema_scores[i]
                    last_boxes[i] = best_box

        frame_id += 1
        sleep = yolo_delay - (time.time() - start)
        if sleep > 0:
            time.sleep(sleep)

# ================= START THREADS =================
for i, cap in enumerate(caps):
    threading.Thread(target=camera_reader, args=(i, cap), daemon=True).start()

threading.Thread(target=yolo_worker, daemon=True).start()

# ================= MAIN LOOP =================
frame_id = 0

while running:
    with lock:
        frames_raw = latest_frames.copy()
        scores = ema_scores.copy()
        boxes = last_boxes.copy()

    no_ball = all(s < NO_BALL_EPS for s in scores)

    # -------- CAMERA SELECTION --------
    if no_ball:
        active_cam = DEFAULT_CAMERA
    else:
        best_idx = int(np.argmax(scores))
        now = time.time()

        if (
            best_idx != active_cam and
            scores[best_idx] > scores[active_cam] * NORMAL_SWITCH_THRESHOLD and
            now - last_switch_time > SWITCH_COOLDOWN
        ):
            active_cam = best_idx
            last_switch_time = now

    # -------- DRAW INPUT FRAMES --------
    displays = []
    for i in range(N):
        frame = frames_raw[i]
        disp = frame.copy() if frame is not None else np.zeros(
            (DETECT_SIZE[1], DETECT_SIZE[0], 3), np.uint8
        )

        if boxes[i]:
            x1, y1, x2, y2 = boxes[i]
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(disp, f"CAM {i+1}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        displays.append(cv2.resize(disp, DISPLAY_SIZE))

    # -------- 2x2 GRID --------
    top = np.hstack(displays[:2])
    bottom = np.hstack(displays[2:])
    cv2.imshow("Input Videos", np.vstack([top, bottom]))

    # -------- BEST CAMERA --------
    best_frame = displays[active_cam].copy()
    cv2.putText(best_frame, f"BEST CAMERA: {active_cam+1}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

    cv2.imshow("Best Camera View", best_frame)

    # -------- SOCKET SEND --------
    if frame_id % SOCKET_EVERY == 0:
        send_best_frame(best_frame)

    frame_id += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        running = False
        break

# ================= CLEANUP =================
for cap in caps:
    cap.release()

cv2.destroyAllWindows()
sio.disconnect()