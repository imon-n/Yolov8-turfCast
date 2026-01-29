import cv2
import random
import base64
import numpy as np
import socketio
import threading
import time
from ultralytics import YOLO

# ---------------- CONFIG ----------------
VIDEO_SIZE = (416, 256)
DISPLAY_SIZE = (640, 360)
SKIP_RATE = 14
CONF_THRESH = 0.5
SOCKET_SEND_EVERY = 1
# ---------------------------------------

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

for cap in caps:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not all([cap.isOpened() for cap in caps]):
    print("❌ Video not found")
    exit()

# Socket.IO
sio = socketio.Client()
sio.connect("http://localhost:5000")
print("✅ Connected to server")

# ---------------- SHARED DATA ----------------
frames = [None, None]
ball_areas = [0, 0]
best_boxes = [None, None]
prev_centers = [None, None]

lock = threading.Lock()
frame_counts = [0, 0]
detect_counter = 0

# ---------------- BALL TRACKER ----------------
def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def distance(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))

# ---------------- YOLO THREAD ----------------
def yolo_worker():
    global detect_counter

    while True:
        with lock:
            local_frames = frames.copy()

        for i, frame in enumerate(local_frames):
            if frame is None:
                continue

            frame_counts[i] += 1
            if frame_counts[i] % SKIP_RATE != 0:
                continue

            resized = cv2.resize(frame, VIDEO_SIZE)
            candidates = []

            results = model(resized, conf=CONF_THRESH, verbose=False)
            boxes = results[0].boxes

            if boxes is not None:
                for box in boxes:
                    clsID = int(box.cls[0])
                    if clsID == SPORTS_BALL_ID:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area = (x2 - x1) * (y2 - y1)
                        center = box_center((x1, y1, x2, y2))
                        candidates.append((area, (x1, y1, x2, y2), center))

            best_box = None
            best_area = 0

            if candidates:
                # 🔥 TRACKING LOGIC
                if prev_centers[i] is None:
                    best = max(candidates, key=lambda x: x[0])
                else:
                    best = min(
                        candidates,
                        key=lambda x: distance(x[2], prev_centers[i]) - x[0] * 0.0005
                    )

                best_area, best_box, best_center = best
                prev_centers[i] = best_center

            with lock:
                ball_areas[i] = best_area
                best_boxes[i] = best_box

        detect_counter += 1
        time.sleep(0.005)

threading.Thread(target=yolo_worker, daemon=True).start()

# ---------------- SOCKET SEND ----------------
def send_best_frame(frame):
    resized = cv2.resize(frame, DISPLAY_SIZE)
    _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 60])
    sio.emit("bestFrame", base64.b64encode(buf).decode())

# ---------------- MAIN LOOP ----------------
while True:
    disp_frames = []

    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((DISPLAY_SIZE[1], DISPLAY_SIZE[0], 3), dtype=np.uint8)

        with lock:
            frames[i] = frame

        disp_frames.append(cv2.resize(frame, DISPLAY_SIZE))

    cv2.imshow("Input Videos (Smooth)", np.hstack(disp_frames))

    with lock:
        best_idx = int(np.argmax(ball_areas))
        best_frame = frames[best_idx]
        best_box = best_boxes[best_idx]

    if best_frame is not None:
        show = cv2.resize(best_frame, DISPLAY_SIZE)

        if best_box is not None:
            x1, y1, x2, y2 = best_box

            sx = DISPLAY_SIZE[0] / VIDEO_SIZE[0]
            sy = DISPLAY_SIZE[1] / VIDEO_SIZE[1]

            x1, x2 = int(x1 * sx), int(x2 * sx)
            y1, y2 = int(y1 * sy), int(y2 * sy)

            area = (x2 - x1) * (y2 - y1)

            cv2.rectangle(show, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(show, f"Ball Area: {area}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Best Frame (Tracked Ball)", show)

        if detect_counter % SOCKET_SEND_EVERY == 0:
            send_best_frame(best_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
sio.disconnect()