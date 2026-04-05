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

#  Reduced for live RTSP
DETECT_SIZE = (640, 360)
DISPLAY_SIZE = (640, 360)

NORMAL_CONF = 0.20
LOW_CONF = 0.15

NORMAL_SWITCH_THRESHOLD = 1.1
NO_BALL_EPS = 0.02

DEFAULT_CAMERA = 0
JPEG_QUALITY = 85
# =========================================

# Load COCO classes
with open("utils/coco.txt", "r") as f:
    class_list = f.read().split("\n")

SPORTS_BALL_ID = class_list.index("sports ball")

# Load YOLO
model = YOLO("weights/yolov8n.pt")

# ================= RTSP SOURCES =================
rtsp_urls = [
    "rtsp://admin:Stellar11@192.168.0.201:554/Streaming/Channels/101",
    "rtsp://admin:Stellar11@192.168.0.202:554/Streaming/Channels/101",
]

caps = []

for url in rtsp_urls:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)  # FFMPEG
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # Disable buffering
    caps.append(cap)

if not all(c.isOpened() for c in caps):
    print("RTSP stream not found")
    exit()

# ================= SOCKET =================
sio = socketio.Client()
sio.connect("http://localhost:5000")
print("Connected to server")

def send_best_frame(frame):
    _, buffer = cv2.imencode(
        ".jpg", frame,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    sio.emit("bestFrame", base64.b64encode(buffer).decode())

# ================= SHARED STATE =================
latest_frames = [None] * len(caps)
ema_scores = [0.0] * len(caps)
last_boxes = [None] * len(caps)

active_cam = DEFAULT_CAMERA
last_switch_time = time.time()

lock = threading.Lock()
running = True

# ================= CAMERA THREAD =================
def camera_reader(idx, cap):
    global running

    while running:
        start = time.time()

        # Flush old frames (low latency trick)
        for _ in range(2):
            cap.grab()

        ret, frame = cap.retrieve()

        if not ret:
            continue  # don't stop system for RTSP

        frame = cv2.resize(frame, DETECT_SIZE)

        with lock:
            latest_frames[idx] = frame

        # Small sleep to reduce CPU overload
        time.sleep(0.01)

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
    threading.Thread(
        target=camera_reader,
        args=(i, cap),
        daemon=True
    ).start()

threading.Thread(target=yolo_worker, daemon=True).start()

# ================= MAIN LOOP =================
frame_id = 0

while running:
    frames = []
    with lock:
        scores = ema_scores.copy()
        boxes = last_boxes.copy()
        frames_raw = latest_frames.copy()

    no_ball = all(s < NO_BALL_EPS for s in scores)

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

    for i in range(len(caps)):
        frame = frames_raw[i]

        if frame is None:
            display = np.zeros((DETECT_SIZE[1], DETECT_SIZE[0], 3), np.uint8)
        else:
            display = frame.copy()
            if boxes[i]:
                x1, y1, x2, y2 = boxes[i]
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        frames.append(cv2.resize(display, DISPLAY_SIZE))

    cv2.imshow("Input Videos", np.hstack(frames))

    best_frame = frames[active_cam].copy()
    status = "DEFAULT VIEW (NO BALL)" if no_ball else "TRACKING BALL"

    cv2.putText(best_frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255) if no_ball else (0, 255, 0), 2)

    cv2.putText(best_frame, f"CAMERA: {active_cam + 1}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 0, 0), 2)

    cv2.imshow("Best Camera View", best_frame)

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