import cv2, random, base64, numpy as np, time
from ultralytics import YOLO
import socketio

# ---------------- CONFIG ----------------
DETECT_EVERY = 6           # run YOLO every N frames
EMA_ALPHA = 0.3            # smoothing factor
SWITCH_THRESHOLD = 1.3     # new camera must be 30% better
SWITCH_COOLDOWN = 0.5      # seconds
IMG_SIZE = (640, 360)
CONF_THRESH = 0.5
# ----------------------------------------

# Load COCO classes
with open("utils/coco.txt", "r") as f:
    class_list = f.read().split("\n")

SPORTS_BALL_ID = class_list.index("sports ball")

# Load YOLO
model = YOLO("weights/yolov8n.pt")

# Open videos
caps = [
    cv2.VideoCapture("inference/videos/0t1.mp4"),
    cv2.VideoCapture("inference/videos/0t2.mp4"),
]

if not all(c.isOpened() for c in caps):
    print("Video not found")
    exit()

# Socket.IO
sio = socketio.Client()
sio.connect("http://localhost:5000")
print("Connected to server")

def send_best_frame(frame):
    frame = cv2.resize(frame, IMG_SIZE)
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    sio.emit("bestFrame", base64.b64encode(buffer).decode())

# ---------------- STATE ----------------
frame_id = 0
ema_scores = [0.0] * len(caps)
last_boxes = [None] * len(caps)
active_cam = 0
last_switch_time = time.time()
# --------------------------------------

while True:
    frames = []
    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((360, 640, 3), dtype=np.uint8)

        frame = cv2.resize(frame, IMG_SIZE)
        frames.append(frame)

        # -------- YOLO DETECTION --------
        if frame_id % DETECT_EVERY == 0:
            results = model.predict(frame, conf=CONF_THRESH, verbose=False)
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

            # EMA smoothing (🔥 key logic)
            ema_scores[i] = EMA_ALPHA * best_area + (1 - EMA_ALPHA) * ema_scores[i]
            last_boxes[i] = best_box

        # Draw last known box
        if last_boxes[i]:
            x1, y1, x2, y2 = last_boxes[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    frame_id += 1

    # -------- CAMERA SWITCH LOGIC --------
    best_idx = int(np.argmax(ema_scores))
    now = time.time()

    if (
        best_idx != active_cam and
        ema_scores[best_idx] > ema_scores[active_cam] * SWITCH_THRESHOLD and
        now - last_switch_time > SWITCH_COOLDOWN
    ):
        active_cam = best_idx
        last_switch_time = now

    # -------- DISPLAY INPUT VIDEOS --------
    cv2.imshow("Input Videos", np.hstack(frames))

    # -------- BEST CAMERA DISPLAY --------
    best_frame = frames[active_cam].copy()

    if last_boxes[active_cam]:
        x1, y1, x2, y2 = last_boxes[active_cam]
        area = (x2 - x1) * (y2 - y1)

        cv2.rectangle(best_frame, (x1, y1), (x2, y2), (0,255,0), 3)
        cv2.putText(
            best_frame,
            f"Ball Area: {area}",
            (x1, max(y1 - 12, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

    cv2.putText(
        best_frame,
        f"BEST CAMERA: {active_cam + 1}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,0,255),
        2
    )

    cv2.imshow("Best Camera View", best_frame)

    # -------- SEND SOCKET FRAME --------
    send_best_frame(best_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
for cap in caps:
    cap.release()

cv2.destroyAllWindows()
sio.disconnect()