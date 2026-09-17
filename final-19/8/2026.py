import cv2
import numpy as np
import time
import threading
import os
import math
from collections import deque
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

NORMAL_SWITCH_THRESHOLD = 1.1  # kept for reference, no longer used by the new switching logic

NO_BALL_EPS = 500
NO_BALL_FRAME_LIMIT = 5

DEFAULT_CAMERA = 0

# Speed control: increase to slow down output video (1.0 = normal)
OUTPUT_SLOW_FACTOR = 1.3

# ---- New: multi-factor switching logic tuning ----
EMA_ALPHA_CONF = 0.3            # smoothing for detection confidence (separate from ball area EMA)
CONSISTENCY_WINDOW = 10         # detection cycles used to compute the "consistency" score
RECENCY_HALF_LIFE = 4           # cycles for the recency score to decay (ball-loss grace period)
HYSTERESIS_SWITCH_MARGIN = 0.15 # candidate must beat active by at least this much (0-1 scale)
CANDIDATE_PERSISTENCE = 5       # candidate must stay ahead for this many CONSECUTIVE cycles
SWITCH_PENALTY = 0.08           # score penalty applied to non-active cameras (biases toward staying)

# score component weights (sum to 1.0)
WEIGHT_SIZE = 0.35
WEIGHT_CONF = 0.25
WEIGHT_CONSISTENCY = 0.20
WEIGHT_RECENCY = 0.20

os.makedirs("output", exist_ok=True)
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
last_boxes = [None] * len(caps)
no_ball_frames = [0] * len(caps)

# --- New switching-related state (per camera) ---
ema_area = [0.0] * len(caps)                 # smoothed ball bounding-box area
ema_conf = [0.0] * len(caps)                 # smoothed detection confidence
consistency_window = [deque(maxlen=CONSISTENCY_WINDOW) for _ in range(len(caps))]
cycles_since_detection = [CONSISTENCY_WINDOW] * len(caps)
persistence_count = [0] * len(caps)          # consecutive cycles this camera has beaten active
final_scores = [0.0] * len(caps)             # last computed normalized score (for debug overlay)

active_cam = DEFAULT_CAMERA
candidate_cam = None
last_switch_time = time.time()
switch_reason = "Initial camera (default)"

lock = threading.Lock()
running = True


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


# ================= SCORING =================
def compute_camera_score(i):
    """
    Combines ball size, confidence, detection consistency and recency into
    a single normalized (0-1) score for camera i. Must be called with
    `lock` already held (reads/writes shared state).
    """
    frame_area = DETECT_SIZE[0] * DETECT_SIZE[1]

    # ball size score: a ball filling ~6% of the frame counts as a full-score close-up
    size_ref = frame_area * 0.06
    size_score = min(ema_area[i] / size_ref, 1.0) if size_ref > 0 else 0.0

    # confidence score, rescaled from [LOW_CONF, 1.0] -> [0, 1]
    if ema_conf[i] <= 0:
        conf_score = 0.0
    else:
        conf_score = max(0.0, min((ema_conf[i] - LOW_CONF) / (1.0 - LOW_CONF), 1.0))

    # consistency score: fraction of recent cycles where the ball was detected
    window = consistency_window[i]
    consistency_score = (sum(window) / len(window)) if window else 0.0

    # recency score: smooth exponential decay since the ball was last seen
    # (this is what implements the ball-loss "grace period" - a 1-3 cycle
    # gap barely dents the score instead of collapsing it instantly)
    recency_score = math.exp(-cycles_since_detection[i] / RECENCY_HALF_LIFE) if RECENCY_HALF_LIFE > 0 else 0.0

    score = (
        WEIGHT_SIZE * size_score
        + WEIGHT_CONF * conf_score
        + WEIGHT_CONSISTENCY * consistency_score
        + WEIGHT_RECENCY * recency_score
    )

    final_scores[i] = score
    return score


# ================= SWITCHING DECISION =================
def update_switching():
    """
    Runs once per YOLO detection cycle (called from yolo_worker, under `lock`).
    Replaces the old argmax + 1.1x-threshold logic with:
      - a global no-ball fallback to DEFAULT_CAMERA
      - a switch penalty on non-active cameras (bias to stay)
      - a hysteresis margin a candidate must clear before it even counts
      - a persistence requirement (N consecutive cycles) before switching
      - the existing real-time SWITCH_COOLDOWN
    This is what removes the flicker/bounce of the old system while still
    reacting to genuine, sustained changes in the action.
    """
    global active_cam, last_switch_time, candidate_cam, switch_reason

    scores = [compute_camera_score(i) for i in range(len(caps))]

    # ---- global no-ball fallback ----
    no_ball_global = all(f > NO_BALL_FRAME_LIMIT for f in no_ball_frames)
    if no_ball_global:
        if active_cam != DEFAULT_CAMERA:
            active_cam = DEFAULT_CAMERA
            last_switch_time = time.time()
            switch_reason = "No ball detected on any camera -> default"
        candidate_cam = None
        for i in range(len(caps)):
            persistence_count[i] = 0
        return

    # ---- apply switch penalty to non-active cameras ----
    adjusted = [
        scores[i] if i == active_cam else max(0.0, scores[i] - SWITCH_PENALTY)
        for i in range(len(caps))
    ]
    active_score = adjusted[active_cam]

    # ---- find the strongest non-active candidate ----
    best_i, best_s = None, -1.0
    for i in range(len(caps)):
        if i == active_cam:
            continue
        if adjusted[i] > best_s:
            best_i, best_s = i, adjusted[i]
    candidate_cam = best_i

    # ---- persistence bookkeeping (resets on any dip below the margin,
    #      which is what stops rapid alternation / single-frame spikes) ----
    for i in range(len(caps)):
        if i == active_cam:
            persistence_count[i] = 0
            continue
        if adjusted[i] > active_score + HYSTERESIS_SWITCH_MARGIN:
            persistence_count[i] += 1
        else:
            persistence_count[i] = 0

    # ---- final decision: persistence satisfied AND cooldown elapsed ----
    now = time.time()
    cooldown_ok = now - last_switch_time > SWITCH_COOLDOWN
    persistence_ok = best_i is not None and persistence_count[best_i] >= CANDIDATE_PERSISTENCE

    if persistence_ok and cooldown_ok:
        switch_reason = (
            f"CAM{best_i + 1} beat CAM{active_cam + 1} by >{HYSTERESIS_SWITCH_MARGIN:.2f} "
            f"for {persistence_count[best_i]} consecutive cycles"
        )
        active_cam = best_i
        last_switch_time = now
        for i in range(len(caps)):
            persistence_count[i] = 0


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
                best_conf = 0.0

                if boxes is not None:
                    for box in boxes:
                        if int(box.cls[0]) == SPORTS_BALL_ID:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            area = (x2 - x1) * (y2 - y1)
                            if area > best_area:
                                best_area = area
                                best_box = (x1, y1, x2, y2)
                                best_conf = float(box.conf[0])

                with lock:
                    if best_box is None:
                        no_ball_frames[i] += 1
                        cycles_since_detection[i] += 1
                        # decay gradually instead of an instant *0.5 cut -
                        # this is what gives the active camera a grace period
                        ema_area[i] *= (1 - EMA_ALPHA * 0.5)
                        ema_conf[i] *= (1 - EMA_ALPHA_CONF * 0.5)
                    else:
                        no_ball_frames[i] = 0
                        cycles_since_detection[i] = 0
                        ema_area[i] = EMA_ALPHA * best_area + (1 - EMA_ALPHA) * ema_area[i]
                        ema_conf[i] = EMA_ALPHA_CONF * best_conf + (1 - EMA_ALPHA_CONF) * ema_conf[i]

                    consistency_window[i].append(0 if best_box is None else 1)
                    last_boxes[i] = best_box

        # run the switching decision once per detection cycle, after every
        # camera's score has been updated for this cycle
        with lock:
            update_switching()

        frame_id += 1

        sleep = yolo_delay - (time.time() - start)
        if sleep > 0:
            time.sleep(sleep)


# ================= OVERLAY FUNCTION =================

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


def draw_switch_debug(frame, cur_active, cur_candidate, cur_scores, cur_persist, cur_reason):
    """Extra debug HUD: candidate camera, scores, persistence, and last switch reason."""
    y = frame.shape[0] - 55
    for i in range(len(cur_scores)):
        tag = "ACTIVE" if i == cur_active else ("CAND" if i == cur_candidate else "")
        cv2.putText(frame, f"CAM{i+1} score={cur_scores[i]:.2f} {tag}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        y += 15
    if cur_candidate is not None and cur_candidate != cur_active:
        cv2.putText(frame,
                    f"candidate persistence: {cur_persist[cur_candidate]}/{CANDIDATE_PERSISTENCE}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1)
    return frame


# ================= START THREADS =================
for i, cap in enumerate(caps):
    threading.Thread(target=camera_reader, args=(i, cap), daemon=True).start()

threading.Thread(target=yolo_worker, daemon=True).start()


# ================= MAIN LOOP =================
while running:

    loop_start = time.time()

    with lock:
        boxes = last_boxes.copy()
        frames_raw = latest_frames.copy()
        no_ball_local = no_ball_frames.copy()
        cur_active = active_cam
        cur_candidate = candidate_cam
        cur_scores = final_scores.copy()
        cur_persist = persistence_count.copy()
        cur_reason = switch_reason

    # active_cam is now decided inside yolo_worker (once per detection cycle,
    # via update_switching()) - the main loop just reads it here.
    no_ball = all(f > NO_BALL_FRAME_LIMIT for f in no_ball_local)
    is_tracking = not no_ball
    is_default = (cur_active == DEFAULT_CAMERA)

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
    best_frame = frames[cur_active].copy()
    best_frame = draw_overlay(best_frame, cur_active, is_default, is_tracking)
    best_frame = draw_switch_debug(best_frame, cur_active, cur_candidate, cur_scores, cur_persist, cur_reason)

    cv2.imshow("Best", best_frame)

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

cv2.destroyAllWindows()
print("[INFO] Done")