import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO

DETECT_EVERY = 1
YOLO_FPS = 8
EMA_ALPHA = 0.2
SWITCH_COOLDOWN = 3

DETECT_SIZE = (960, 540)
DISPLAY_SIZE = (640, 360)

NORMAL_CONF = 0.25
LOW_CONF = 0.18

NORMAL_SWITCH_THRESHOLD = 1.1
NO_BALL_EPS = 0.02

DEFAULT_CAMERA = 0

# Load COCO classes
with open("utils/coco.txt", "r") as f:
    class_list = f.read().split("\n")

SPORTS_BALL_ID = class_list.index("sports ball")

# Load YOLO
model = YOLO("weights/yolov8n.pt")

# Video sources
caps = [
    cv2.VideoCapture("inference/videos/cam1.mp4"),
    cv2.VideoCapture("inference/videos/cam2.mp4"),
]

if not all(c.isOpened() for c in caps):
    print("Video not found")
    exit()

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

# ================= VIDEO WRITER (FIXED) =================
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('best_output.avi', fourcc, 20.0, DISPLAY_SIZE)

if not out.isOpened():
    print("❌ VideoWriter failed to open")
    exit()
else:
    print("✅ Video recording started...")

# ================= MAIN LOOP =================
while running:
    frames = []

    with lock:
        scores = ema_scores.copy()
        boxes = last_boxes.copy()
        frames_raw = latest_frames.copy()

    no_ball = all(s < NO_BALL_EPS for s in scores)

    # -------- SWITCH LOGIC --------
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

    # -------- DRAW --------
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

    # -------- SHOW INPUT --------
    cv2.imshow("Input Videos", np.hstack(frames))

    # -------- BEST FRAME --------
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

    # ===== SAVE VIDEO (FIXED) =====
    best_frame = cv2.resize(best_frame, DISPLAY_SIZE)
    out.write(best_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        running = False
        break

# ================= CLEANUP =================
for cap in caps:
    cap.release()

out.release()
cv2.destroyAllWindows()