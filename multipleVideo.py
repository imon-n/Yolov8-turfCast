import cv2
import random
import numpy as np
from ultralytics import YOLO

# --- Load COCO classes ---
with open("utils/coco.txt", "r") as f:
    class_list = f.read().split("\n")

# Sports Ball ID (from coco.txt list)
SPORTS_BALL_ID = class_list.index("sports ball")

# Random colors for visualization
detection_colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for _ in class_list]

# Load YOLOv8 model
model = YOLO("weights/yolov8n.pt")

# Open multiple video feeds
caps = [
    cv2.VideoCapture("inference/videos/turf1.mp4"),
    cv2.VideoCapture("inference/videos/turf2.mp4"),
    # cv2.VideoCapture("inference/videos/turf.mp4")
]

if not all([cap.isOpened() for cap in caps]):
    print("One or more video files not found!")
    exit()

while True:
    frames = []
    ball_sizes = []

    # Read frames from all videos
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            frames.append(None)
            ball_sizes.append(0)
            continue

        # Run YOLO detection
        results = model.predict(source=[frame], conf=0.5, verbose=False)
        boxes = results[0].boxes

        max_ball_area = 0
        if boxes is not None:
            for box in boxes:
                clsID = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                bb = box.xyxy[0].cpu().numpy()

                if clsID == SPORTS_BALL_ID:  # Only detect sports ball
                    x1, y1, x2, y2 = map(int, bb)
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_ball_area:
                        max_ball_area = area

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), detection_colors[clsID], 3)
                    cv2.putText(frame, f"Ball {round(conf*100,1)}%", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        frames.append(frame)
        ball_sizes.append(max_ball_area)

    # Select frame with largest detected ball
    best_idx = int(np.argmax(ball_sizes))
    best_frame = frames[best_idx]

    if best_frame is not None:
        cv2.imshow("Best Camera View", best_frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release all video feeds
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
