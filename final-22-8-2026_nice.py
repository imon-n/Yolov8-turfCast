import cv2
import numpy as np
import time
import threading
import os
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

# ================= CONFIG =================
DETECT_EVERY = 1
YOLO_FPS = 8
EMA_ALPHA = 0.2
DETECT_SIZE = (960, 540)
DISPLAY_SIZE = (640, 360)
NORMAL_CONF = 0.12
LOW_CONF = 0.08
MIN_BALL_AREA = 40
MAX_BALL_AREA = 20000
MAX_POSITION_JUMP = 300
MAX_PREDICTION_FRAMES = 2
DEFAULT_CAMERA = 0
DEFAULT_CAM_HOLD_SECONDS = 4.0
OUTPUT_SLOW_FACTOR = 1.3
OUTPUT_DIR = "output"
BEST_CAM_OUTPUT = os.path.join(OUTPUT_DIR, "best_camera.avi")
GRID_OUTPUT = os.path.join(OUTPUT_DIR, "grid_view.avi")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= LOAD COCO =================
with open("utils/coco.txt", "r") as f:
    class_list = f.read().split("\n")

SPORTS_BALL_ID = class_list.index("sports ball")

# ================= YOLO =================
model = YOLO("weights/yolov8n.pt")

# ================= VIDEO INPUT =================
caps = [
    cv2.VideoCapture("inference/videos/md.mp4"),
    cv2.VideoCapture("inference/videos/lf.mp4"),
    cv2.VideoCapture("inference/videos/rt.mp4")
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
    f"YOLO FPS: {YOLO_FPS}"
)

# ================= STATE =================
latest_frames = [None] * len(caps)
camera_scores = [0.0] * len(caps)
last_boxes = [None] * len(caps)
last_confidences = [0.0] * len(caps)
last_ball_centers = [None] * len(caps)
last_ball_areas = [0.0] * len(caps)
no_ball_frames = [0] * len(caps)
prediction_frames = [0] * len(caps)
real_detection = [False] * len(caps)

active_cam = DEFAULT_CAMERA

# Camera that was tracking the ball before
# the ball disappeared.
last_tracking_camera = DEFAULT_CAMERA

# Time when all cameras first lost the ball.
tracking_lost_time = None

lock = threading.Lock()
running = True

best_writer = None
grid_writer = None

# ================= KALMAN FILTER =================
kalman_filters = [None] * len(caps)
kalman_initialized = [False] * len(caps)


def create_kalman():
    kf = KalmanFilter(dim_x=4, dim_z=2)

    dt = 1.0 / YOLO_FPS

    kf.F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=float)

    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=float)

    kf.R = np.eye(2) * 5.0
    kf.Q = np.eye(4) * 0.03
    kf.P = np.eye(4) * 100.0

    kf.x = np.zeros(
        (4, 1),
        dtype=float
    )

    return kf


def reset_kalman(i):
    kalman_filters[i] = create_kalman()
    kalman_initialized[i] = False
    prediction_frames[i] = 0


def update_kalman(i, center):
    if kalman_filters[i] is None:
        reset_kalman(i)

    kf = kalman_filters[i]

    measurement = np.array([
        [float(center[0])],
        [float(center[1])]
    ])

    if not kalman_initialized[i]:
        kf.x[0, 0] = measurement[0, 0]
        kf.x[1, 0] = measurement[1, 0]
        kf.x[2, 0] = 0.0
        kf.x[3, 0] = 0.0

        kalman_initialized[i] = True

    else:
        kf.predict()
        kf.update(measurement)


def get_expected_position(i):
    if not kalman_initialized[i]:
        return None

    if kalman_filters[i] is None:
        return None

    kf = kalman_filters[i]

    dt = 1.0 / YOLO_FPS

    return (
        float(kf.x[0, 0] + kf.x[2, 0] * dt),
        float(kf.x[1, 0] + kf.x[3, 0] * dt)
    )


def predict_kalman(i):
    if not kalman_initialized[i]:
        return None

    if kalman_filters[i] is None:
        return None

    kf = kalman_filters[i]

    kf.predict()

    return (
        float(kf.x[0, 0]),
        float(kf.x[1, 0]),
        float(kf.x[2, 0]),
        float(kf.x[3, 0])
    )


# ================= CAMERA SCORE =================
def calculate_camera_score(i, frame_shape):
    box = last_boxes[i]

    if box is None:
        return 0.0

    h, w = frame_shape[:2]

    x1, y1, x2, y2 = box

    bw = max(
        1,
        x2 - x1
    )

    bh = max(
        1,
        y2 - y1
    )

    area = float(
        bw * bh
    )

    frame_area = float(
        w * h
    )

    # Area = 30%
    area_score = min(
        1.0,
        area / max(
            1.0,
            frame_area * 0.08
        )
    )

    # Center proximity = 30%
    cx = (
        x1 + x2
    ) / 2.0

    cy = (
        y1 + y2
    ) / 2.0

    dx = (
        cx - w / 2.0
    ) / max(
        1.0,
        w / 2.0
    )

    dy = (
        cy - h / 2.0
    ) / max(
        1.0,
        h / 2.0
    )

    center_distance = min(
        1.0,
        np.sqrt(
            dx * dx +
            dy * dy
        )
    )

    center_score = (
        1.0 -
        center_distance
    )

    # Velocity = 20%
    velocity_score = 0.5

    if (
        kalman_initialized[i]
        and kalman_filters[i] is not None
    ):
        vx = float(
            kalman_filters[i].x[2, 0]
        )

        vy = float(
            kalman_filters[i].x[3, 0]
        )

        speed = np.hypot(
            vx,
            vy
        )

        if speed > 0.5:
            velocity_score = min(
                1.0,
                speed / 30.0
            )

    # Confidence = 20%
    confidence_score = min(
        1.0,
        max(
            0.0,
            last_confidences[i]
        )
    )

    # Edge penalty
    edge_margin = min(
        cx,
        w - cx,
        cy,
        h - cy
    )

    edge_score = min(
        1.0,
        max(
            0.0,
            edge_margin /
            (min(w, h) / 2.0)
        )
    )

    score = (
        0.30 * area_score +
        0.30 * center_score +
        0.20 * velocity_score +
        0.20 * confidence_score
    )

    score *= (
        0.60 +
        0.40 * edge_score
    )

    return max(
        0.001,
        score
    )


# ================= CAMERA SELECTION =================
def select_best_camera(
    scores,
    boxes,
    real_detection,
    prediction_counts
):
    global active_cam
    global last_tracking_camera
    global tracking_lost_time

    detected = []
    predicted = []

    for i in range(len(scores)):
        if boxes[i] is None:
            continue

        if real_detection[i]:
            detected.append(i)

        elif prediction_counts[i] > 0:
            predicted.append(i)

    # ================= REAL YOLO DETECTION =================
    if detected:
        best_camera = max(
            detected,
            key=lambda i: scores[i]
        )

        active_cam = best_camera
        last_tracking_camera = best_camera
        tracking_lost_time = None

        return best_camera

    # ================= KALMAN BACKUP =================
    if predicted:
        best_camera = max(
            predicted,
            key=lambda i: scores[i]
        )

        active_cam = best_camera
        last_tracking_camera = best_camera
        tracking_lost_time = None

        return best_camera

    # ================= NO BALL =================
    if tracking_lost_time is None:
        tracking_lost_time = time.time()

    lost_duration = (
        time.time() -
        tracking_lost_time
    )

    if lost_duration < DEFAULT_CAM_HOLD_SECONDS:
        return last_tracking_camera

    active_cam = DEFAULT_CAMERA

    return DEFAULT_CAMERA


# ================= CAMERA THREAD =================
def camera_reader(idx, cap):
    global running

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    delay = (
        1.0 / fps
        if fps > 0
        else 1.0 / 30.0
    )

    while running:
        start = time.time()

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

        elapsed = (
            time.time() -
            start
        )

        sleep_time = (
            delay -
            elapsed
        )

        if sleep_time > 0:
            time.sleep(
                sleep_time
            )


# ================= YOLO THREAD =================
def yolo_worker():
    global running

    yolo_delay = 1.0 / YOLO_FPS
    frame_id = 0

    while running:
        start = time.time()

        with lock:
            frames = latest_frames.copy()

        for i, frame in enumerate(frames):

            if frame is None:
                continue

            if frame_id % DETECT_EVERY != 0:
                continue

            results = model.predict(
                frame,
                conf=NORMAL_CONF,
                imgsz=DETECT_SIZE[0],
                verbose=False
            )

            boxes = results[0].boxes
            detections = []

            if boxes is not None:
                for box in boxes:

                    if int(
                        box.cls[0]
                    ) != SPORTS_BALL_ID:
                        continue

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    bw = x2 - x1
                    bh = y2 - y1

                    if bw <= 0 or bh <= 0:
                        continue

                    area = float(
                        bw * bh
                    )

                    if area < MIN_BALL_AREA:
                        continue

                    if area > MAX_BALL_AREA:
                        continue

                    confidence = float(
                        box.conf[0]
                    )

                    cx = (
                        x1 + x2
                    ) / 2.0

                    cy = (
                        y1 + y2
                    ) / 2.0

                    detections.append({
                        "box": (
                            x1,
                            y1,
                            x2,
                            y2
                        ),
                        "center": (
                            cx,
                            cy
                        ),
                        "area": area,
                        "confidence": confidence
                    })

            best_detection = None

            # ================= REAL YOLO DETECTION =================
            if detections:

                expected = (
                    get_expected_position(i)
                )

                if expected is not None:

                    px, py = expected
                    valid = []

                    for detection in detections:

                        cx, cy = detection[
                            "center"
                        ]

                        distance = np.hypot(
                            cx - px,
                            cy - py
                        )

                        if (
                            distance
                            <= MAX_POSITION_JUMP
                        ):
                            detection[
                                "distance"
                            ] = distance

                            valid.append(
                                detection
                            )

                    if valid:
                        best_detection = max(
                            valid,
                            key=lambda d: (
                                d["confidence"],
                                d["area"],
                                -d.get(
                                    "distance",
                                    0
                                )
                            )
                        )

                    else:
                        best_detection = max(
                            detections,
                            key=lambda d: (
                                d["confidence"],
                                d["area"]
                            )
                        )

                else:
                    best_detection = max(
                        detections,
                        key=lambda d: (
                            d["confidence"],
                            d["area"]
                        )
                    )

            with lock:

                # ================= BALL FOUND =================
                if best_detection is not None:

                    box = best_detection[
                        "box"
                    ]

                    center = best_detection[
                        "center"
                    ]

                    area = best_detection[
                        "area"
                    ]

                    confidence = best_detection[
                        "confidence"
                    ]

                    update_kalman(
                        i,
                        center
                    )

                    last_boxes[i] = box

                    last_ball_centers[i] = (
                        center
                    )

                    last_ball_areas[i] = (
                        area
                    )

                    last_confidences[i] = (
                        confidence
                    )

                    no_ball_frames[i] = 0

                    prediction_frames[i] = 0

                    real_detection[i] = True

                    raw_score = (
                        calculate_camera_score(
                            i,
                            frame.shape
                        )
                    )

                    camera_scores[i] = max(
                        0.001,
                        EMA_ALPHA *
                        raw_score +
                        (1.0 - EMA_ALPHA) *
                        camera_scores[i]
                    )

                # ================= BALL NOT FOUND =================
                else:

                    real_detection[i] = False

                    no_ball_frames[i] += 1

                    prediction_frames[i] += 1

                    # ================= SHORT KALMAN BACKUP =================
                    if (
                        kalman_initialized[i]
                        and prediction_frames[i]
                        <= MAX_PREDICTION_FRAMES
                    ):

                        prediction = (
                            predict_kalman(i)
                        )

                        if prediction is not None:

                            px, py, vx, vy = (
                                prediction
                            )

                            h, w = (
                                frame.shape[:2]
                            )

                            # Reject prediction outside frame
                            if (
                                px < 0
                                or px >= w
                                or py < 0
                                or py >= h
                            ):
                                last_boxes[i] = None
                                camera_scores[i] = 0.0
                                continue

                            area = max(
                                MIN_BALL_AREA,
                                last_ball_areas[i]
                            )

                            side = max(
                                4.0,
                                min(
                                    np.sqrt(area),
                                    min(w, h) *
                                    0.25
                                )
                            )

                            x1 = int(
                                max(
                                    0,
                                    px -
                                    side / 2
                                )
                            )

                            y1 = int(
                                max(
                                    0,
                                    py -
                                    side / 2
                                )
                            )

                            x2 = int(
                                min(
                                    w - 1,
                                    px +
                                    side / 2
                                )
                            )

                            y2 = int(
                                min(
                                    h - 1,
                                    py +
                                    side / 2
                                )
                            )

                            last_boxes[i] = (
                                x1,
                                y1,
                                x2,
                                y2
                            )

                            # Kalman prediction is weaker
                            # than real YOLO detection.
                            camera_scores[i] *= 0.70

                    else:

                        last_boxes[i] = None

                        last_confidences[i] = 0.0

                        camera_scores[i] = 0.0

                        real_detection[i] = False

                        if (
                            prediction_frames[i]
                            > MAX_PREDICTION_FRAMES
                        ):
                            reset_kalman(i)

        frame_id += 1

        sleep = (
            yolo_delay -
            (
                time.time() -
                start
            )
        )

        if sleep > 0:
            time.sleep(
                sleep
            )


# ================= OVERLAY =================
def draw_overlay(
    frame,
    cam_idx,
    is_default,
    is_tracking
):

    cv2.putText(
        frame,
        f"CAM: {cam_idx + 1}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

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


# ================= START THREADS =================
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


# ================= MAIN LOOP =================
while running:

    loop_start = time.time()

    with lock:

        scores = camera_scores.copy()

        boxes = last_boxes.copy()

        frames_raw = latest_frames.copy()

        prediction_counts = (
            prediction_frames.copy()
        )

        real_detection_local = (
            real_detection.copy()
        )

    # ================= CAMERA SWITCH =================
    active_cam = select_best_camera(
        scores,
        boxes,
        real_detection_local,
        prediction_counts
    )

    # ================= TRACKING STATUS =================
    is_tracking = (
        any(real_detection_local)
        or any(
            prediction_counts[i] > 0
            and boxes[i] is not None
            for i in range(len(caps))
        )
    )

    is_default = (
        active_cam == DEFAULT_CAMERA
        and not is_tracking
    )

    # ================= DRAW 3 CAMERAS =================
    frames = []

    for i in range(len(caps)):

        frame = frames_raw[i]

        if frame is None:

            display = np.zeros(
                (
                    DISPLAY_SIZE[1],
                    DISPLAY_SIZE[0],
                    3
                ),
                dtype=np.uint8
            )

        else:

            display = frame.copy()

            # ================= BALL BOX =================
            if boxes[i] is not None:

                x1, y1, x2, y2 = (
                    boxes[i]
                )

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

        display = cv2.resize(
            display,
            DISPLAY_SIZE
        )

        # ================= ACTIVE CAMERA BORDER =================
        if i == active_cam:

            cv2.rectangle(
                display,
                (0, 0),
                (
                    DISPLAY_SIZE[0] - 1,
                    DISPLAY_SIZE[1] - 1
                ),
                (0, 255, 0),
                6
            )

            cv2.putText(
                display,
                f"CAM {i + 1} [ACTIVE]",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

        else:

            cv2.rectangle(
                display,
                (0, 0),
                (
                    DISPLAY_SIZE[0] - 1,
                    DISPLAY_SIZE[1] - 1
                ),
                (80, 80, 80),
                2
            )

            cv2.putText(
                display,
                f"CAM {i + 1}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

        frames.append(
            display
        )

    # ================= GRID DISPLAY =================
    # CAM 1 | CAM 2
    # CAM 3 | BLANK

    blank = np.zeros_like(
        frames[0]
    )

    top = np.hstack((
        frames[0],
        frames[1]
    ))

    bottom = np.hstack((
        frames[2],
        blank
    ))

    grid = np.vstack((
        top,
        bottom
    ))

    cv2.imshow(
        "TurfCast - 3 Camera Grid",
        grid
    )

    # ================= BEST CAMERA =================
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
        "TurfCast - Best Camera",
        best_frame
    )

    # ================= SAVE BEST CAMERA =================
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

    best_writer.write(
        best_frame
    )

    # ================= SAVE GRID VIEW =================
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

    grid_writer.write(
        grid
    )

    # ================= SPEED CONTROL =================
    elapsed = (
        time.time() -
        loop_start
    )

    sleep_time = (
        FRAME_DELAY -
        elapsed
    )

    if sleep_time > 0:
        time.sleep(
            sleep_time
        )

    # ================= QUIT =================
    if cv2.waitKey(1) & 0xFF == ord("q"):

        running = False

        break


# ================= CLEANUP =================
running = False

for cap in caps:
    cap.release()

if best_writer is not None:

    best_writer.release()

    print(
        f"[INFO] Saved -> "
        f"{BEST_CAM_OUTPUT}"
    )

if grid_writer is not None:

    grid_writer.release()

    print(
        f"[INFO] Saved -> "
        f"{GRID_OUTPUT}"
    )

cv2.destroyAllWindows()

print("[INFO] Done")
