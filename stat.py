import cv2
import numpy as np
import time
import threading
import os
import sys

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
OUTPUT_SLOW_FACTOR = 1.3
# =========================================

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
print(f"[INFO] Source FPS: {source_fps:.2f} | Slow Factor: {OUTPUT_SLOW_FACTOR}x")

# ================= STATE & STATISTICS =================
latest_frames = [None] * len(caps)
ema_scores = [0.0] * len(caps)
last_boxes = [None] * len(caps)
no_ball_frames = [0] * len(caps)

active_cam = DEFAULT_CAMERA
last_switch_time = time.time()

lock = threading.Lock()
running = True

# --- PERFORMANCE EVALUATION METRICS ---
total_frames = 0
detected_frames = 0

camera_total_frames = [0, 0, 0]
camera_detected_frames = [0, 0, 0]
camera_selected = [1, 0, 0] 

total_switches = 0
switch_intervals = []
last_switch_timestamp = time.time()

# YOLO Metrics (YOLO Thread থেকে ট্র্যাক করা হবে)
total_yolo_time = 0.0
yolo_call_count = 0

# Display Metrics (Main Loop থেকে ট্র্যাক করা হবে)
fps_sum = 0.0
fps_count = 0


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
    global total_frames, detected_frames, camera_total_frames, camera_detected_frames
    global total_yolo_time, yolo_call_count

    yolo_delay = 1.0 / YOLO_FPS
    frame_id = 0

    while running:
        start = time.time()
        with lock:
            frames = latest_frames.copy()

        no_ball_global = all(f > NO_BALL_FRAME_LIMIT for f in no_ball_frames)
        conf = LOW_CONF if no_ball_global else NORMAL_CONF

        # এই ইটারেশনে মোট ৩টি ক্যামেরার টোটাল প্রসেসিং টাইম ট্র্যাক করার জন্য
        batch_start = time.time()
        active_batch = False

        for i, frame in enumerate(frames):
            if frame is None:
                continue

            if frame_id % DETECT_EVERY == 0:
                active_batch = True
                
                results = model.predict(frame, conf=conf, verbose=False)
                boxes = results[0].boxes

                camera_total_frames[i] += 1
                total_frames += 1

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
                        
                        camera_detected_frames[i] += 1
                        detected_frames += 1

                    last_boxes[i] = best_box

        # ৩টি ফ্রেমের ব্যাচ প্রসেস হতে মোট কত সময় লাগল (Inference Time)
        if active_batch:
            batch_elapsed = (time.time() - batch_start) * 1000  # ms
            total_yolo_time += batch_elapsed
            yolo_call_count += 1

        frame_id += 1
        sleep = yolo_delay - (time.time() - start)
        if sleep > 0:
            time.sleep(sleep)


# ================= OVERLAY FUNCTION =================
def draw_overlay(frame, cam_idx, is_default, is_tracking):
    cv2.putText(frame, f"CAM: {cam_idx + 1}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    if is_tracking:
        status_text, status_color = "TRACKING BALL", (0, 255, 0)
    else:
        status_text, status_color = "DEFAULT VIEW", (0, 165, 255)

    cv2.putText(frame, status_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    if not is_tracking:
        cv2.putText(frame, "NO BALL", (frame.shape[1] - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    return frame


# ================= START THREADS =================
for i, cap in enumerate(caps):
    threading.Thread(target=camera_reader, args=(i, cap), daemon=True).start()

threading.Thread(target=yolo_worker, daemon=True).start()

os.system('cls' if os.name == 'nt' else 'clear')

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
        if active_cam != DEFAULT_CAMERA:
            total_switches += 1
            switch_intervals.append(time.time() - last_switch_timestamp)
            last_switch_timestamp = time.time()
            active_cam = DEFAULT_CAMERA
            last_switch_time = time.time()
            camera_selected[active_cam] += 1
    else:
        best_idx = int(np.argmax(scores))
        now = time.time()

        if (
            best_idx != active_cam and
            scores[best_idx] > scores[active_cam] * NORMAL_SWITCH_THRESHOLD and
            now - last_switch_time > SWITCH_COOLDOWN
        ):
            total_switches += 1
            switch_intervals.append(now - last_switch_timestamp)
            last_switch_timestamp = now
            active_cam = best_idx
            last_switch_time = now
            camera_selected[active_cam] += 1

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

    best_frame = frames[active_cam].copy()
    best_frame = draw_overlay(best_frame, active_cam, is_default, is_tracking)
    cv2.imshow("Best", best_frame)

    # Calculate Main Loop Render FPS
    elapsed = time.time() - loop_start
    loop_fps = 1 / max(elapsed, 1e-6)
    fps_sum += loop_fps
    fps_count += 1

    # === LIVE TERMINAL DASHBOARD (CORRECTED LOGIC) ===
    current_render_fps = fps_sum / fps_count if fps_count else 0
    
    # ইনফারেন্স টাইম এবং বাস্তবসম্মত YOLO FPS ক্যালকুলেশন
    current_avg_inference = total_yolo_time / yolo_call_count if yolo_call_count else 0
    current_yolo_fps = 1000 / current_avg_inference if current_avg_inference > 0 else 0
    
    current_overall_rate = (detected_frames / total_frames * 100) if total_frames else 0
    current_avg_switch = sum(switch_intervals) / len(switch_intervals) if switch_intervals else 0

    sys.stdout.write("\033[H")
    stats_output = (
        "========== Live Performance Evaluation ==========\n"
        f"Total Frames              : {total_frames}\n"
        f"Detected Frames           : {detected_frames}\n"
        f"Overall Detection Rate    : {current_overall_rate:.2f}%\n"
        f"Display Render FPS        : {current_render_fps:.2f}\n\n"
        f"YOLO Execution FPS        : {current_yolo_fps:.2f}\n"
        f"Average Inference Time    : {current_avg_inference:.2f} ms\n\n"
        f"Total Camera Switches     : {total_switches}\n"
        f"Camera 1 Selected         : {camera_selected[0]} times\n"
        f"Camera 2 Selected         : {camera_selected[1]} times\n"
        f"Camera 3 Selected         : {camera_selected[2]} times\n"
        f"Average Switch Interval   : {current_avg_switch:.2f} sec\n"
        "=================================================\n"
    )
    sys.stdout.write(stats_output)
    sys.stdout.flush()

    # SPEED CONTROL
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
print("\n[INFO] Processing Stopped. Final statistics are displayed above.")