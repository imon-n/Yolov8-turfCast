import cv2
from ultralytics import YOLO

# --- Load YOLO Model ---
model = YOLO("weights/yolov8n.pt")   # or "yolov8s.pt", etc.

# --- Load Image ---
image_path = "inference/images/bird.jpeg"   
img = cv2.imread(image_path)

if img is None:
    print("Image not found!")
    exit()

# --- Run YOLO inference ---
results = model.predict(source=img, conf=0.5, verbose=False)

# Get class names
with open("utils/coco.txt", "r") as f:
    class_list = f.read().split("\n")

# --- Draw bounding boxes on image ---
boxes = results[0].boxes

if boxes is not None:
    for box in boxes:
        clsID = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Draw label
        label = f"{class_list[clsID]} {round(conf*100, 1)}%"
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# --- Show Result ---
cv2.imshow("Object Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()