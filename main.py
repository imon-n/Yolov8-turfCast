import cv2, random, base64, numpy as np
from ultralytics import YOLO
import socketio

# --- Load COCO classes ---
with open("utils/coco.txt", "r") as f:
    class_list = f.read().split("\n")

SPORTS_BALL_ID = class_list.index("sports ball")

detection_colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for _ in class_list]

# Load YOLOv8 model (CPU fallback if no GPU)
model = YOLO("weights/yolov8n.pt")

# Open 2 videos
caps = [
    cv2.VideoCapture("inference/videos/0t1.mp4"),
    cv2.VideoCapture("inference/videos/0t2.mp4"),
]

if not all([cap.isOpened() for cap in caps]):
    print("One or both video files not found!")
    exit()

# --- Socket.IO client ---
sio = socketio.Client()

try:
    sio.connect("http://localhost:5000")
    print("✅ Connected to Node.js server")
except Exception as e:
    print("Could not connect to server:", e)
    exit()

def send_best_frame(frame):
    # Resize before sending → কম data ট্রান্সফার
    resized = cv2.resize(frame, (640, 360))
    _, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 60])
    jpg_as_text = base64.b64encode(buffer).decode("utf-8")

    sio.emit("bestFrame", jpg_as_text)
    print("📤 Best frame sent")

frame_count = 0
skip_rate = 4   # detect (speed boost)

while True:
    frames, ball_sizes = [], []

    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            frames.append(None)
            ball_sizes.append(0)
            continue

        # Smaller frame → faster inference
        resized = cv2.resize(frame, (640, 360))

        frame_count += 1
        max_ball_area = 0

        # detect
        if frame_count % skip_rate == 0:
            results = model.predict(resized, conf=0.5, verbose=False)
            boxes = results[0].boxes

            if boxes is not None:
                for box in boxes:
                    clsID = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    bb = box.xyxy[0].cpu().numpy()

                    if clsID == SPORTS_BALL_ID:
                        x1, y1, x2, y2 = map(int, bb)
                        area = (x2 - x1) * (y2 - y1)
                        if area > max_ball_area:
                            max_ball_area = area

                        cv2.rectangle(resized, (x1, y1), (x2, y2), detection_colors[clsID], 2)
                        cv2.putText(resized, f"Ball {round(conf*100,1)}%", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        frames.append(resized)
        ball_sizes.append(max_ball_area)

    if frame_count % skip_rate != 0:
        continue

    best_idx = int(np.argmax(ball_sizes))
    best_frame = frames[best_idx]

    if best_frame is not None:
        print(f"✅ Best Frame from Video {best_idx+1} | Ball Area: {ball_sizes[best_idx]}")
        cv2.imshow("Best Frame", best_frame)
        send_best_frame(best_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
sio.disconnect()
