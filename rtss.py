import cv2
import base64
import numpy as np
import time
import threading
from ultralytics import YOLO
import socketio

# ================= CONFIG =================
YOLO_FPS = 8                  # TOTAL YOLO calls/sec
EMA_ALPHA = 0.35
SWITCH_THRESHOLD = 1.05
SWITCH_COOLDOWN = 1.5
SOCKET_EVERY = 8

FRAME_SIZE = (640, 360)
NORMAL_CONF = 0.35
LOW_CONF = 0.2
NO_BALL_EPS = 50

DEFAULT_CAMERA = 0
JPEG_QUALITY = 85
# =========================================

# ---------- COCO ----------
with open("utils/coco.txt") as f:
    classes = f.read().split("\n")

SPORTS_BALL_ID = classes.index("sports ball")

# ---------- YOLO ----------
model = YOLO("weights/yolov8n.pt")
model.fuse()

# ---------- RTSP URLS ----------
RTSP_URLS = [
    "rtsp://admin:Stellar11@192.168.0.201:554/Streaming/Channels/101",
    "rtsp://admin:cam2-2025@192.168.0.200:554/Streaming/Channels/101",
]

# ---------- CAPTURES ----------
caps = []
for url in RTSP_URLS:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    caps.append(cap)

if not all(c.isOpened() for c in caps):
    print("❌ RTSP open failed")
    exit()

# ---------- SOCKET ----------
sio = socketio.Client()
sio.connect("http://localhost:5000")
print("✅ Socket connected")

def send_best_frame(frame):
    _, buf = cv2.imencode(".jpg", frame,
                          [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    sio.emit("bestFrame", base64.b64encode(buf).decode())

# ================= SHARED STATE =================
latest_frames = [None] * len(caps)
ema_scores = [0.0] * len(caps)
last_boxes = [None] * len(caps)

active_cam = DEFAULT_CAMERA
last_switch_time = time.time()

rr_index = 0
lock = threading.Lock()
running = True
# ================================================

# ================= CAMERA THREAD =================
def camera_reader(idx, cap):
    global running
    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, FRAME_SIZE)
        with lock:
            latest_frames[idx] = frame

# ================= ROUND ROBIN YOLO =================
def yolo_worker():
    global rr_index, running

    delay = 1.0 / YOLO_FPS

    while running:
        start = time.time()

        with lock:
            frame = latest_frames[rr_index]
            prev_score = ema_scores[rr_index]

        conf = LOW_CONF if prev_score < NO_BALL_EPS else NORMAL_CONF

        if frame is not None:
            results = model.predict(
                frame,
                conf=conf,
                device="cpu",
                verbose=False
            )

            best_area = 0
            best_box = None
            boxes = results[0].boxes

            if boxes is not None:
                for box in boxes:
                    if int(box.cls[0]) == SPORTS_BALL_ID:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area = (x2 - x1) * (y2 - y1)
                        if area > best_area:
                            best_area = area
                            best_box = (x1, y1, x2, y2)

            with lock:
                ema_scores[rr_index] = (
                    EMA_ALPHA * best_area +
                    (1 - EMA_ALPHA) * ema_scores[rr_index]
                )
                last_boxes[rr_index] = best_box

        rr_index = (rr_index + 1) % len(caps)

        sleep = delay - (time.time() - start)
        if sleep > 0:
            time.sleep(sleep)

# ================= START THREADS =================
for i, cap in enumerate(caps):
    threading.Thread(target=camera_reader, args=(i, cap), daemon=True).start()

threading.Thread(target=yolo_worker, daemon=True).start()

# ================= MAIN LOOP =================
frame_id = 0

while True:
    with lock:
        frames = latest_frames.copy()
        scores = ema_scores.copy()
        boxes = last_boxes.copy()

    no_ball = all(s < NO_BALL_EPS for s in scores)

    # ---- CAMERA SWITCH ----
    if no_ball:
        active_cam = DEFAULT_CAMERA
    else:
        best_idx = int(np.argmax(scores))
        now = time.time()

        if (
            best_idx != active_cam
            and scores[best_idx] > scores[active_cam] * SWITCH_THRESHOLD
            and now - last_switch_time > SWITCH_COOLDOWN
        ):
            active_cam = best_idx
            last_switch_time = now

    # ---- DRAW INPUTS ----
    views = []
    for i in range(len(frames)):
        f = frames[i]
        if f is None:
            disp = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), np.uint8)
        else:
            disp = f.copy()
            if boxes[i]:
                x1, y1, x2, y2 = boxes[i]
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        views.append(disp)

    cv2.imshow("Input Cameras", np.hstack(views))

    # ---- BEST VIEW ----
    best_frame = views[active_cam].copy()
    status = "NO BALL → DEFAULT" if no_ball else "TRACKING BALL"

    cv2.putText(best_frame, status, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255) if no_ball else (0, 255, 0), 2)

    cv2.putText(best_frame, f"CAMERA {active_cam + 1}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Best Camera View", best_frame)

    if frame_id % SOCKET_EVERY == 0:
        send_best_frame(best_frame)

    frame_id += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ================= CLEANUP =================
running = False
for cap in caps:
    cap.release()

cv2.destroyAllWindows()
sio.disconnect()