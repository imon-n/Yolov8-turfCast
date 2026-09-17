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

# Detection confidence
NORMAL_CONF = 0.18
LOW_CONF = 0.12

NORMAL_SWITCH_THRESHOLD = 1.1

NO_BALL_EPS = 500
NO_BALL_FRAME_LIMIT = 5

DEFAULT_CAMERA = 0

# Speed control
OUTPUT_SLOW_FACTOR = 1.3

# ================= BALL FILTER =================

# Minimum and maximum ball bounding-box size
MIN_BALL_WIDTH = 15
MIN_BALL_HEIGHT = 15

MAX_BALL_WIDTH = 100
MAX_BALL_HEIGHT = 100

# Minimum bounding-box area
MIN_BALL_AREA = 225
MAX_BALL_AREA = 8000

# Ball aspect ratio
MIN_ASPECT_RATIO = 0.50
MAX_ASPECT_RATIO = 1.80

# Maximum movement between consecutive detections
# Set to None if you do not want position filtering
MAX_BALL_MOVEMENT = 250

# High confidence detection can bypass movement filter
HIGH_CONF_BYPASS = 0.45

# ================= OUTPUT =================
OUTPUT_DIR = "output"

BEST_CAM_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "best_camera.avi"
)

GRID_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "grid_view.avi"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================
# LOAD COCO CLASSES
# =========================================

with open("utils/coco.txt", "r") as f:
    class_list = f.read().split("\n")

SPORTS_BALL_ID = class_list.index("sports ball")

# ================= MODEL =================

model = YOLO("weights/yolov8n.pt")

# ================= VIDEO INPUT =================

caps = [
    cv2.VideoCapture("inference/videos/md.mp4"),
    cv2.VideoCapture("inference/videos/lf.mp4"),
    cv2.VideoCapture("inference/videos/rt.mp4"),
]

if not all(c.isOpened() for c in caps):
    print("[ERROR] Video not found")
    exit()

source_fps = caps[0].get(cv2.CAP_PROP_FPS)

if source_fps <= 0:
    source_fps = 30.0

FRAME_DELAY = 1.0 / source_fps
OUTPUT_FPS = source_fps / OUTPUT_SLOW_FACTOR

print(
    f"[INFO] Source FPS: {source_fps:.2f} | "
    f"Output FPS: {OUTPUT_FPS:.2f} | "
    f"Slow Factor: {OUTPUT_SLOW_FACTOR}x"
)

# ================= STATE =================

latest_frames = [None] * len(caps)

ema_scores = [0.0] * len(caps)

last_boxes = [None] * len(caps)

last_centers = [None] * len(caps)

no_ball_frames = [0] * len(caps)

active_cam = DEFAULT_CAMERA

last_switch_time = time.time()

lock = threading.Lock()

running = True

best_writer = None
grid_writer = None


# ============================================================
# BALL VALIDATION
# ============================================================

def validate_ball(box, confidence, previous_center=None):
    """
    Validate YOLO sports-ball detection using:
    1. Confidence
    2. Bounding box size
    3. Area
    4. Aspect ratio
    5. Temporal movement
    """

    x1, y1, x2, y2 = map(int, box)

    width = x2 - x1
    height = y2 - y1

    # -----------------------------
    # Size filter
    # -----------------------------

    if width < MIN_BALL_WIDTH:
        return False

    if height < MIN_BALL_HEIGHT:
        return False

    if width > MAX_BALL_WIDTH:
        return False

    if height > MAX_BALL_HEIGHT:
        return False

    # -----------------------------
    # Area filter
    # -----------------------------

    area = width * height

    if area < MIN_BALL_AREA:
        return False

    if area > MAX_BALL_AREA:
        return False

    # -----------------------------
    # Aspect ratio filter
    # -----------------------------

    aspect_ratio = width / float(height)

    if aspect_ratio < MIN_ASPECT_RATIO:
        return False

    if aspect_ratio > MAX_ASPECT_RATIO:
        return False

    # -----------------------------
    # Temporal movement filter
    # -----------------------------

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    if previous_center is not None:

        previous_x, previous_y = previous_center

        movement = np.sqrt(
            (center_x - previous_x) ** 2 +
            (center_y - previous_y) ** 2
        )

        # Very confident detections are allowed
        # to move farther.
        if (
            movement > MAX_BALL_MOVEMENT
            and confidence < HIGH_CONF_BYPASS
        ):
            return False

    return True


# ============================================================
# CAMERA THREAD
# ============================================================

def camera_reader(idx, cap):
    global running

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    delay = 1.0 / fps

    while running:

        t_start = time.time()

        ret, frame = cap.read()

        if not ret:
            running = False
            break

        frame = cv2.resize(
            frame,
            DETECT_SIZE
        )

        with lock:
            latest_frames[idx] = frame

        elapsed = time.time() - t_start

        sleep_time = delay - elapsed

        if sleep_time > 0:
            time.sleep(sleep_time)


# ============================================================
# YOLO THREAD
# ============================================================

def yolo_worker():
    global running

    yolo_delay = 1.0 / YOLO_FPS

    frame_id = 0

    while running:

        start = time.time()

        with lock:
            frames = latest_frames.copy()
            previous_centers = last_centers.copy()

        # -----------------------------------------
        # Global no-ball state
        # -----------------------------------------

        no_ball_global = all(
            f > NO_BALL_FRAME_LIMIT
            for f in no_ball_frames
        )

        conf = LOW_CONF if no_ball_global else NORMAL_CONF

        # =========================================
        # PROCESS EACH CAMERA
        # =========================================

        for i, frame in enumerate(frames):

            if frame is None:
                continue

            if frame_id % DETECT_EVERY != 0:
                continue

            # -------------------------------------
            # YOLO
            # -------------------------------------

            results = model.predict(
                frame,
                conf=conf,
                imgsz=960,
                verbose=False
            )

            boxes = results[0].boxes

            candidates = []

            # -------------------------------------
            # Find sports-ball candidates
            # -------------------------------------

            if boxes is not None:

                for box in boxes:

                    class_id = int(
                        box.cls[0]
                    )

                    if class_id != SPORTS_BALL_ID:
                        continue

                    confidence = float(
                        box.conf[0]
                    )

                    xyxy = box.xyxy[0]

                    x1, y1, x2, y2 = map(
                        int,
                        xyxy
                    )

                    candidate_box = (
                        x1,
                        y1,
                        x2,
                        y2
                    )

                    # -----------------------------
                    # Validate candidate
                    # -----------------------------

                    valid = validate_ball(
                        candidate_box,
                        confidence,
                        previous_centers[i]
                    )

                    if not valid:
                        continue

                    width = x2 - x1
                    height = y2 - y1

                    area = width * height

                    center_x = (
                        x1 + x2
                    ) // 2

                    center_y = (
                        y1 + y2
                    ) // 2

                    # ---------------------------------
                    # Candidate score
                    # ---------------------------------

                    score = area * confidence

                    # Slight preference for
                    # detections near previous position
                    if previous_centers[i] is not None:

                        px, py = previous_centers[i]

                        distance = np.sqrt(
                            (center_x - px) ** 2 +
                            (center_y - py) ** 2
                        )

                        score *= (
                            1.0 /
                            (1.0 + distance / 200.0)
                        )

                    candidates.append(
                        (
                            score,
                            area,
                            confidence,
                            candidate_box,
                            (center_x, center_y)
                        )
                    )

            # =====================================
            # SELECT BEST VALID BALL
            # =====================================

            best_box = None
            best_area = 0
            best_center = None

            if candidates:

                candidates.sort(
                    key=lambda x: x[0],
                    reverse=True
                )

                (
                    _,
                    best_area,
                    best_confidence,
                    best_box,
                    best_center
                ) = candidates[0]

            # =====================================
            # UPDATE STATE
            # =====================================

            with lock:

                if best_box is None:

                    no_ball_frames[i] += 1

                    # Gradually reduce score
                    ema_scores[i] *= 0.5

                    # Keep last center temporarily
                    # for short tracking gaps

                    if no_ball_frames[i] > 2:
                        last_centers[i] = None

                else:

                    no_ball_frames[i] = 0

                    ema_scores[i] = (
                        EMA_ALPHA * best_area
                        +
                        (1 - EMA_ALPHA)
                        * ema_scores[i]
                    )

                    last_centers[i] = best_center

                last_boxes[i] = best_box

        frame_id += 1

        # =========================================
        # CONTROL YOLO FPS
        # =========================================

        elapsed = time.time() - start

        sleep = yolo_delay - elapsed

        if sleep > 0:
            time.sleep(sleep)


# ============================================================
# OVERLAY
# ============================================================

def draw_overlay(
    frame,
    cam_idx,
    is_default,
    is_tracking
):

    # -----------------------------
    # Camera number
    # -----------------------------

    cv2.putText(
        frame,
        f"CAM: {cam_idx + 1}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    # -----------------------------
    # Status
    # -----------------------------

    if is_tracking:

        status_text = "TRACKING BALL"
        status_color = (0, 255, 0)

    else:

        status_text = "DEFAULT VIEW"
        status_color = (0, 165, 255)

    cv2.putText(
        frame,
        status_text,
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2
    )

    # -----------------------------
    # NO BALL
    # -----------------------------

    if not is_tracking:

        cv2.putText(
            frame,
            "NO BALL",
            (
                frame.shape[1] - 120,
                25
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )

    return frame


# ============================================================
# START THREADS
# ============================================================

for i, cap in enumerate(caps):

    threading.Thread(
        target=camera_reader,
        args=(i, cap),
        daemon=True
    ).start()


threading.Thread(
    target=yolo_worker,
    daemon=True
).start()


# ============================================================
# MAIN LOOP
# ============================================================

while running:

    loop_start = time.time()

    # -----------------------------------------
    # Read shared state
    # -----------------------------------------

    with lock:

        scores = ema_scores.copy()

        boxes = last_boxes.copy()

        frames_raw = latest_frames.copy()

        no_ball_local = no_ball_frames.copy()

    # -----------------------------------------
    # Check global ball state
    # -----------------------------------------

    no_ball = all(
        f > NO_BALL_FRAME_LIMIT
        for f in no_ball_local
    )

    # ========================================================
    # SWITCH LOGIC
    # ========================================================

    if no_ball:

        active_cam = DEFAULT_CAMERA

    else:

        best_idx = int(
            np.argmax(scores)
        )

        now = time.time()

        current_score = scores[active_cam]

        best_score = scores[best_idx]

        # Avoid zero-score division problem
        if current_score <= 0:
            switch_condition = True
        else:
            switch_condition = (
                best_score
                >
                current_score
                * NORMAL_SWITCH_THRESHOLD
            )

        if (
            best_idx != active_cam
            and switch_condition
            and now - last_switch_time
            > SWITCH_COOLDOWN
        ):

            active_cam = best_idx

            last_switch_time = now

    # -----------------------------------------
    # Status
    # -----------------------------------------

    is_tracking = not no_ball

    is_default = (
        active_cam == DEFAULT_CAMERA
    )

    # ========================================================
    # DRAW FRAMES
    # ========================================================

    frames = []

    for i in range(len(caps)):

        frame = frames_raw[i]

        if frame is None:

            display = np.zeros(
                (
                    DETECT_SIZE[1],
                    DETECT_SIZE[0],
                    3
                ),
                np.uint8
            )

        else:

            display = frame.copy()

            # ---------------------------------
            # Draw validated ball only
            # ---------------------------------

            if boxes[i] is not None:

                x1, y1, x2, y2 = boxes[i]

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

        frames.append(
            cv2.resize(
                display,
                DISPLAY_SIZE
            )
        )

    # ========================================================
    # GRID VIEW
    # ========================================================

    blank = np.zeros_like(
        frames[0]
    )

    grid_frames = frames.copy()

    while len(grid_frames) < 4:

        grid_frames.append(
            blank.copy()
        )

    top = np.hstack(
        (
            grid_frames[0],
            grid_frames[1]
        )
    )

    bottom = np.hstack(
        (
            grid_frames[2],
            grid_frames[3]
        )
    )

    grid = np.vstack(
        (
            top,
            bottom
        )
    )

    cv2.imshow(
        "Grid",
        grid
    )

    # ========================================================
    # BEST FRAME
    # ========================================================

    best_frame = frames[
        active_cam
    ].copy()

    best_frame = draw_overlay(
        best_frame,
        active_cam,
        is_default,
        is_tracking
    )

    cv2.imshow(
        "Best",
        best_frame
    )

    # ========================================================
    # INIT WRITERS
    # ========================================================

    if best_writer is None:

        h, w = best_frame.shape[:2]

        best_writer = cv2.VideoWriter(
            BEST_CAM_OUTPUT,
            cv2.VideoWriter_fourcc(
                *"XVID"
            ),
            OUTPUT_FPS,
            (w, h)
        )

        print(
            f"[INFO] Writer started -> "
            f"{BEST_CAM_OUTPUT}"
        )

    if grid_writer is None:

        h, w = grid.shape[:2]

        grid_writer = cv2.VideoWriter(
            GRID_OUTPUT,
            cv2.VideoWriter_fourcc(
                *"XVID"
            ),
            OUTPUT_FPS,
            (w, h)
        )

        print(
            f"[INFO] Writer started -> "
            f"{GRID_OUTPUT}"
        )

    # ========================================================
    # WRITE OUTPUT
    # ========================================================

    best_writer.write(
        best_frame
    )

    grid_writer.write(
        grid
    )

    # ========================================================
    # SPEED CONTROL
    # ========================================================

    elapsed = (
        time.time()
        - loop_start
    )

    sleep_time = (
        FRAME_DELAY
        - elapsed
    )

    if sleep_time > 0:

        time.sleep(
            sleep_time
        )

    # ========================================================
    # QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        running = False

        break


# ============================================================
# CLEANUP
# ============================================================

running = False

for cap in caps:

    cap.release()

if best_writer:

    best_writer.release()

    print(
        f"[INFO] Saved -> "
        f"{BEST_CAM_OUTPUT}"
    )

if grid_writer:

    grid_writer.release()

    print(
        f"[INFO] Saved -> "
        f"{GRID_OUTPUT}"
    )

cv2.destroyAllWindows()

print("[INFO] Done")