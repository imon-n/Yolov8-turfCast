import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO

# ================= CONFIG =================
DETECT_EVERY = 1
YOLO_FPS = 8   #  ---------
EMA_ALPHA = 0.2
SWITCH_COOLDOWN = 3   # ---------)

DETECT_SIZE = (960, 540)
DISPLAY_SIZE = (640, 360)

NORMAL_CONF = 0.12   # was 0.
LOW_CONF = 0.08      # was 0.

EDGE_THRESHOLD = 0.34
NO_BALL_EPS = 0.02

DEFAULT_CAMERA = 0

STABLE_FRAME_LIMIT = 3   # ---------
BETTER_RATIO = 1.3
# =========================================


# Load COCO classes
with open("utils/coco.txt", "r") as f:
    class_list = f.read().split("\n")

SPORTS_BALL_ID = class_list.index("sports ball")

# Load YOLO
model = YOLO("weights/yolov8n.pt")


# ================= RTSP =================
rtsp_urls = [
    "rtsp://admin:Stellar11@192.168.0.201:554/Streaming/Channels/101",
    "rtsp://admin:cam2-2025@192.168.0.200:554/Streaming/Channels/101",
    "rtsp://admin:Stellar12@192.168.0.100:554/Streaming/Channels/101",
]


def create_capture(url):
    cap = cv2.VideoCapture(f"{url}?rtsp_transport=tcp", cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# ================= SHARED STATE =================
latest_frames = [None] * len(rtsp_urls)
ema_scores = [0.0] * len(rtsp_urls)
last_boxes = [None] * len(rtsp_urls)
stable_counts = [0] * len(rtsp_urls)

active_cam = DEFAULT_CAMERA
last_switch_time = time.time()

lock = threading.Lock()
running = True


# ================= CAMERA THREAD =================
def camera_reader(idx, url):
    global running

    while running:
        cap = create_capture(url)

        if not cap.isOpened():
            time.sleep(1)
            continue

        while running:
            # reduce lag
            for _ in range(1):   # 🔥 was 3 (less delay)
                cap.grab()

            ret, frame = cap.retrieve()

            if not ret:
                break  # reconnect

            frame = cv2.resize(frame, DETECT_SIZE)

            with lock:
                latest_frames[idx] = frame

            time.sleep(0.01)

        cap.release()
        time.sleep(1)


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
                try:
                    results = model.predict(frame, conf=conf, verbose=False)
                except:
                    continue

                boxes = results[0].boxes

                H, W = DETECT_SIZE[1], DETECT_SIZE[0]

                best_score = 0
                best_box = None

                if boxes is not None:
                    for box in boxes:
                        if int(box.cls[0]) == SPORTS_BALL_ID:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            cx = (x1 + x2) / 2
                            cy = (y1 + y2) / 2

                            dx = abs(cx - W/2) / (W/2)
                            dy = abs(cy - H/2) / (H/2)

                            dist = (dx + dy) / 2
                            score = 1 - dist

                            if score > best_score:
                                best_score = score
                                best_box = (x1, y1, x2, y2)

                with lock:
                    if best_score < NO_BALL_EPS:
                        stable_counts[i] = 0
                    else:
                        stable_counts[i] += 1

                    ema_scores[i] = (
                        EMA_ALPHA * best_score +
                        (1 - EMA_ALPHA) * ema_scores[i]
                    )

                    last_boxes[i] = best_box

        frame_id += 1

        sleep = yolo_delay - (time.time() - start)
        if sleep > 0:
            time.sleep(sleep)


# ================= START THREADS =================
for i, url in enumerate(rtsp_urls):
    threading.Thread(
        target=camera_reader,
        args=(i, url),
        daemon=True
    ).start()

threading.Thread(target=yolo_worker, daemon=True).start()


# ================= MAIN LOOP =================
while running:
    frames = []

    with lock:
        scores = ema_scores.copy()
        boxes = last_boxes.copy()
        frames_raw = latest_frames.copy()
        stable_local = stable_counts.copy()

    no_ball = all(s < NO_BALL_EPS for s in scores)

    # -------- SMART SWITCH --------
    if no_ball:
        active_cam = DEFAULT_CAMERA
    else:
        best_idx = int(np.argmax(scores))
        now = time.time()

        current_score = scores[active_cam]
        best_score = scores[best_idx]

        current_has_ball = current_score > NO_BALL_EPS
        best_has_ball = best_score > NO_BALL_EPS

        if not current_has_ball and best_has_ball:
            active_cam = best_idx
            last_switch_time = now

        elif active_cam == DEFAULT_CAMERA:
            if best_has_ball and stable_local[best_idx] >= STABLE_FRAME_LIMIT:
                active_cam = best_idx
                last_switch_time = now

        elif current_has_ball and best_has_ball:
            if (
                best_idx != active_cam and
                best_score > current_score * BETTER_RATIO and
                now - last_switch_time > SWITCH_COOLDOWN
            ):
                active_cam = best_idx
                last_switch_time = now

    # -------- DRAW --------
    for i in range(len(rtsp_urls)):
        frame = frames_raw[i]

        if frame is None:
            display = np.zeros((DETECT_SIZE[1], DETECT_SIZE[0], 3), np.uint8)
        else:
            display = frame.copy()
            if boxes[i]:
                x1, y1, x2, y2 = boxes[i]
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        frames.append(cv2.resize(display, DISPLAY_SIZE))

    # ================= GRID =================
    h, w = DISPLAY_SIZE[1], DISPLAY_SIZE[0]
    blank = np.zeros((h, w, 3), dtype=np.uint8)

    grid_frames = []

    for i in range(3):
        if i < len(frames) and frames[i] is not None:
            frame = frames[i]
        else:
            frame = blank.copy()

        cv2.putText(frame, f"CAM {i+1}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 0), 2)

        grid_frames.append(frame)

    grid_frames.append(blank.copy())

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

    cv2.putText(best_frame, f"CAMERA: {active_cam + 1}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0), 2)

    cv2.imshow("Best Camera View", best_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        running = False
        break


# ================= CLEANUP =================
cv2.destroyAllWindows()