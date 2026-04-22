import cv2
import numpy as np
import time
import threading
import os
from ultralytics import YOLO

# ================= CONFIG =================
DETECT_EVERY = 1
YOLO_FPS = 8
EMA_ALPHA = 0.2
SWITCH_COOLDOWN = 1

DETECT_SIZE = (960, 540)
DISPLAY_SIZE = (640, 360)

NORMAL_CONF = 0.12
LOW_CONF = 0.08

NORMAL_SWITCH_THRESHOLD = 1.1

NO_BALL_EPS = 500
NO_BALL_FRAME_LIMIT = 5

DEFAULT_CAMERA = 0

# Speed control: increase to slow down output video (1.0 = normal)
OUTPUT_SLOW_FACTOR = 1.3

# OUTPUT
OUTPUT_DIR = "output"
BEST_CAM_OUTPUT = os.path.join(OUTPUT_DIR, "best_camera.avi")
GRID_OUTPUT = os.path.join(OUTPUT_DIR, "grid_view.avi")

os.makedirs(OUTPUT_DIR, exist_ok=True)
# =========================================

# Load COCO classes
with open("utils/coco.txt", "r") as f:
    class_list = f.read().split("\n")

SPORTS_BALL_ID = class_list.index("sports ball")

model = YOLO("weights/yolov8n.pt")

# ================= VIDEO INPUT =================
caps = [
    cv2.VideoCapture("inference/videos/md.mp4"),
    cv2.VideoCapture("inference/videos/lf.mp4"),
    cv2.VideoCapture("inference/videos/rt.mp4"),
]

if not all(c.isOpened() for c in caps):
    print("Video not found")
    exit()

source_fps = caps[0].get(cv2.CAP_PROP_FPS)
if source_fps <= 0:
    source_fps = 30.0

FRAME_DELAY = 1.0 / source_fps
OUTPUT_FPS = source_fps / OUTPUT_SLOW_FACTOR

print(f"[INFO] Source FPS: {source_fps:.2f} | Output FPS: {OUTPUT_FPS:.2f} | Slow Factor: {OUTPUT_SLOW_FACTOR}x")

# ================= STATE =================
latest_frames = [None] * len(caps)
ema_scores = [0.0] * len(caps)
last_boxes = [None] * len(caps)
no_ball_frames = [0] * len(caps)

active_cam = DEFAULT_CAMERA
last_switch_time = time.time()

lock = threading.Lock()
running = True

best_writer = None
grid_writer = None


# ================= CAMERA THREAD =================
def camera_reader(idx, cap):
    global running

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = 1.0 / fps if fps > 0 else 1.0 / 30.0

    while running:
        t_start = time.time()

        ret, frame = cap.read()
        if not ret:
            running = False
            break

        frame = cv2.resize(frame, DETECT_SIZE)

        with lock:
            latest_frames[idx] = frame

        elapsed = time.time() - t_start
        sleep_time = delay - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


# ================= YOLO THREAD =================
def yolo_worker():
    global running

    yolo_delay = 1.0 / YOLO_FPS
    frame_id = 0

    while running:
        start = time.time()

        with lock:
            frames = latest_frames.copy()

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
                    if best_box is None:
                        no_ball_frames[i] += 1
                        ema_scores[i] *= 0.5
                    else:
                        no_ball_frames[i] = 0
                        ema_scores[i] = EMA_ALPHA * best_area + (1 - EMA_ALPHA) * ema_scores[i]

                    last_boxes[i] = best_box

        frame_id += 1

        sleep = yolo_delay - (time.time() - start)
        if sleep > 0:
            time.sleep(sleep)


# # ================= OVERLAY FUNCTION =================
# def draw_overlay(frame, cam_idx, is_default, is_tracking):
#     overlay = frame.copy()
#     # Semi-transparent top bar
#     cv2.rectangle(overlay, (0, 0), (frame.shape[1], 75), (0, 0, 0), -1)
#     cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
#     # CAM: 1 / 2 / 3
#     cv2.putText(frame,
#                 f"CAM: {cam_idx + 1}",
#                 (10, 25),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7,
#                 (255, 255, 0), 2)
#     # Status
#     if is_tracking:
#         status_text = "TRACKING BALL"
#         status_color = (0, 255, 0)      # green
#     else:
#         status_text = "DEFAULT VIEW"
#         status_color = (0, 165, 255)    # orange

#     cv2.putText(frame,
#                 status_text,
#                 (10, 55),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7,
#                 status_color, 2)

#     # NO BALL — top right
#     if not is_tracking:
#         cv2.putText(frame,
#                     "NO BALL",
#                     (frame.shape[1] - 120, 25),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.65,
#                     (0, 0, 255), 2)

#     return frame


def draw_overlay(frame, cam_idx, is_default, is_tracking):
    # CAM: 1 / 2 / 3
    cv2.putText(frame,
                f"CAM: {cam_idx + 1}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 0), 2)

    # Status
    if is_tracking:
        status_text = "TRACKING BALL"
        status_color = (0, 255, 0)
    else:
        status_text = "DEFAULT VIEW"
        status_color = (0, 165, 255)

    cv2.putText(frame,
                status_text,
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                status_color, 2)

    # NO BALL — top right
    if not is_tracking:
        cv2.putText(frame,
                    "NO BALL",
                    (frame.shape[1] - 120, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 0, 255), 2)

    return frame

# ================= START THREADS =================
for i, cap in enumerate(caps):
    threading.Thread(target=camera_reader, args=(i, cap), daemon=True).start()

threading.Thread(target=yolo_worker, daemon=True).start()


# ================= MAIN LOOP =================
while running:

    loop_start = time.time()

    with lock:
        scores = ema_scores.copy()
        boxes = last_boxes.copy()
        frames_raw = latest_frames.copy()
        no_ball_local = no_ball_frames.copy()

    no_ball = all(f > NO_BALL_FRAME_LIMIT for f in no_ball_local)

    # SWITCH LOGIC
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

    is_tracking = not no_ball
    is_default = (active_cam == DEFAULT_CAMERA)

    # DRAW FRAMES
    frames = []
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

    # GRID VIEW
    blank = np.zeros_like(frames[0])
    grid_frames = frames.copy()
    while len(grid_frames) < 4:
        grid_frames.append(blank)

    top = np.hstack((grid_frames[0], grid_frames[1]))
    bottom = np.hstack((grid_frames[2], grid_frames[3]))
    grid = np.vstack((top, bottom))

    cv2.imshow("Grid", grid)

    # BEST FRAME with overlay
    best_frame = frames[active_cam].copy()
    best_frame = draw_overlay(best_frame, active_cam, is_default, is_tracking)

    cv2.imshow("Best", best_frame)

    # INIT WRITERS
    if best_writer is None:
        h, w = best_frame.shape[:2]
        best_writer = cv2.VideoWriter(
            BEST_CAM_OUTPUT,
            cv2.VideoWriter_fourcc(*"XVID"),
            OUTPUT_FPS, (w, h)
        )
        print(f"[INFO] Writer started -> {BEST_CAM_OUTPUT}")

    if grid_writer is None:
        h, w = grid.shape[:2]
        grid_writer = cv2.VideoWriter(
            GRID_OUTPUT,
            cv2.VideoWriter_fourcc(*"XVID"),
            OUTPUT_FPS, (w, h)
        )
        print(f"[INFO] Writer started -> {GRID_OUTPUT}")

    # WRITE
    best_writer.write(best_frame)
    grid_writer.write(grid)

    # SPEED CONTROL
    elapsed = time.time() - loop_start
    sleep_time = FRAME_DELAY - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        running = False
        break


# CLEANUP
for cap in caps:
    cap.release()

if best_writer:
    best_writer.release()
    print(f"[INFO] Saved -> {BEST_CAM_OUTPUT}")

if grid_writer:
    grid_writer.release()
    print(f"[INFO] Saved -> {GRID_OUTPUT}")

cv2.destroyAllWindows()
print("[INFO] Done")