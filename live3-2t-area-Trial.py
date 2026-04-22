import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO


# ================= CONFIG =================
DETECT_EVERY = 1
YOLO_FPS = 8
EMA_ALPHA = 0.2
SWITCH_COOLDOWN = 1   # reduced

DETECT_SIZE = (960, 540)
DISPLAY_SIZE = (640, 360)

NORMAL_CONF = 0.12
LOW_CONF = 0.08

NORMAL_SWITCH_THRESHOLD = 1.1

# UPDATED THRESHOLD LOGIC
NO_BALL_EPS = 500

DEFAULT_CAMERA = 0

# FRAME-BASED NO BALL DETECTION
NO_BALL_FRAME_LIMIT = 5
# =========================================


# Load COCO classes
with open("utils/coco.txt", "r") as f:
    class_list = f.read().split("\n")

SPORTS_BALL_ID = class_list.index("sports ball")

# Load YOLO
model = YOLO("weights/yolov8n.pt")


# ================= RTSP SOURCES (3 CAM) =================
rtsp_urls = [
    "rtsp://admin:Stellar11@192.168.0.201:554/Streaming/Channels/101",
    "rtsp://admin:cam2-2025@192.168.0.200:554/Streaming/Channels/101",
    "rtsp://admin:Stellar12@192.168.0.100:554/Streaming/Channels/101",
]

caps = []

for url in rtsp_urls:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    caps.append(cap)

if not all(c.isOpened() for c in caps):
    print("RTSP stream not found")
    exit()


# ================= SHARED STATE =================
latest_frames = [None] * len(caps)
ema_scores = [0.0] * len(caps)
last_boxes = [None] * len(caps)

# NEW
no_ball_frames = [0] * len(caps)

active_cam = DEFAULT_CAMERA
last_switch_time = time.time()

lock = threading.Lock()
running = True


# ================= CAMERA THREAD =================
def camera_reader(idx, cap):
    global running

    while running:
        for _ in range(2):
            cap.grab()

        ret, frame = cap.retrieve()

        if not ret:
            continue

        frame = cv2.resize(frame, DETECT_SIZE)

        with lock:
            latest_frames[idx] = frame

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

        no_ball_global = all(f > NO_BALL_FRAME_LIMIT for f in no_ball_frames)
        conf = LOW_CONF if no_ball_global else NORMAL_CONF

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
                    # FRAME COUNTER LOGIC
                    if best_box is None:
                        no_ball_frames[i] += 1
                        ema_scores[i] *= 0.5   # 🔥 FAST DECAY
                    else:
                        no_ball_frames[i] = 0
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
        no_ball_local = no_ball_frames.copy()

    # NEW NO BALL LOGIC
    no_ball = all(f > NO_BALL_FRAME_LIMIT for f in no_ball_local)

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


    # ================= GRID DISPLAY =================
    h, w = DISPLAY_SIZE[1], DISPLAY_SIZE[0]
    blank = np.zeros((h, w, 3), dtype=np.uint8)

    grid_frames = frames.copy()
    while len(grid_frames) < 4:
        grid_frames.append(blank)

    top_row = np.hstack((grid_frames[0], grid_frames[1]))
    bottom_row = np.hstack((grid_frames[2], grid_frames[3]))
    grid = np.vstack((top_row, bottom_row))

    cv2.imshow("Input Videos", grid)


    # ================= BEST CAMERA =================
    best_frame = frames[active_cam].copy()

    status = "DEFAULT VIEW (NO BALL)" if no_ball else "TRACKING BALL"

    cv2.putText(best_frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255) if no_ball else (0, 255, 0), 2)

    cv2.putText(best_frame,
                f"CAMERA: {active_cam + 1}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2)

    cv2.imshow("Best Camera View", best_frame)

    frame_id += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        running = False
        break

# ================= CLEANUP =================
for cap in caps:
    cap.release()

cv2.destroyAllWindows()