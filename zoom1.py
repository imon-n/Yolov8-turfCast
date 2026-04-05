import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO

# ================= SETTINGS =================
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

# -------- CINEMATIC ZOOM SETTINGS --------
ZOOM_AREA_THRESHOLD = 1200
ZOOM_FACTOR = 1.25      # Max 25% zoom
ZOOM_MIN_DURATION = 4
ZOOM_COOLDOWN = 8

# The "Crawl" Control
ZOOM_STEP = 0.005       # 0.5% zoom change per frame
# -----------------------------------------------

# Load YOLO
model = YOLO("weights/yolov8n.pt")
SPORTS_BALL_ID = 32 

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

# -------- ZOOM STATE --------
zoom_active = False
zoom_start_time = 0
last_zoom_time = 0

current_zoom = 1.0
target_zoom = 1.0

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
            if frame is None: continue

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
while running:
    with lock:
        scores = ema_scores.copy()
        boxes = last_boxes.copy()
        frames_raw = latest_frames.copy()

    no_ball = all(s < NO_BALL_EPS for s in scores)

    # -------- CAMERA SWITCH --------
    if no_ball:
        active_cam = DEFAULT_CAMERA
    else:
        best_idx = int(np.argmax(scores))
        now = time.time()
        if (not zoom_active and best_idx != active_cam and 
            scores[best_idx] > scores[active_cam] * NORMAL_SWITCH_THRESHOLD and 
            now - last_switch_time > SWITCH_COOLDOWN):
            active_cam = best_idx
            last_switch_time = now

    frame_raw = frames_raw[active_cam]
    if frame_raw is None:
        continue
    
    best_frame = frame_raw.copy()
    box = boxes[active_cam]

    # -------- ZOOM TRIGGER & HOLD --------
    now = time.time()
    if box:
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if (area < ZOOM_AREA_THRESHOLD and not zoom_active and 
            now - last_zoom_time > ZOOM_COOLDOWN):
            zoom_active = True
            zoom_start_time = now
            last_zoom_time = now

    if zoom_active:
        if now - zoom_start_time < ZOOM_MIN_DURATION:
            target_zoom = ZOOM_FACTOR
        else:
            zoom_active = False
            target_zoom = 1.0
    else:
        target_zoom = 1.0

    # -------- CINEMATIC ZOOM STEP (The "Crawl") --------
    if abs(current_zoom - target_zoom) > 0.001:
        if current_zoom < target_zoom:
            current_zoom = min(current_zoom + ZOOM_STEP, target_zoom)
        else:
            current_zoom = max(current_zoom - ZOOM_STEP, target_zoom)

    # -------- APPLY ZOOM (Centered Default) --------
    if current_zoom > 1.001:
        h, w = best_frame.shape[:2]
        new_w, new_h = int(w / current_zoom), int(h / current_zoom)

        cx, cy = w // 2, h // 2
        x1_crop = int(cx - new_w // 2)
        y1_crop = int(cy - new_h // 2)
        x2_crop = x1_crop + new_w
        y2_crop = y1_crop + new_h

        crop = best_frame[y1_crop:y2_crop, x1_crop:x2_crop]
        best_frame = cv2.resize(crop, (w, h))

    # -------- ZOOM OVERLAY --------
    # This displays the 1.0x, 1.1x... text on the frame
    zoom_text = f"Zoom: {current_zoom:.2f}x"
    cv2.putText(best_frame, zoom_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # -------- DISPLAY --------
    cv2.imshow("Best Camera View", cv2.resize(best_frame, DISPLAY_SIZE))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        running = False
        break

# Cleanup
for cap in caps: cap.release()
cv2.destroyAllWindows()