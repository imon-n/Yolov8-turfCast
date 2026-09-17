import cv2
import numpy as np
import time
import threading
import os
from collections import deque
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

# ================= CONFIG =================
DETECT_EVERY = 3              # Run YOLO every N frames (Kalman fills gaps)
YOLO_FPS = 10
EMA_ALPHA = 0.3

# Transition timing
MIN_SHOT_DURATION = 2.0       # Seconds before allowed to switch
TRANSITION_FRAMES = 12        # Frames for crossfade (~0.4s at 30fps)
COOLDOWN_AFTER_SWITCH = 1.5   # Seconds lock after transition completes

# Scoring weights
W_AREA = 0.30
W_CENTER = 0.30
W_VELOCITY = 0.20
W_CONF = 0.20
W_EDGE = 0.40

DETECT_SIZE = (960, 540)
DISPLAY_SIZE = (1280, 720)    # Output at higher res

NORMAL_CONF = 0.20
LOW_CONF = 0.10

NO_BALL_FRAME_LIMIT = 8       # More tolerant before dropping to default

DEFAULT_CAMERA = 1            # Camera index for "default/wide" view

# Speed control: same as Code 1 -> declared writer FPS = source_fps / OUTPUT_SLOW_FACTOR
# This compensates for processing lag (YOLO/Kalman/resize) so the saved
# video plays back at the correct (normal) speed instead of looking sped up.
OUTPUT_SLOW_FACTOR = 1.3

OUTPUT_DIR = "output"
BEST_CAM_OUTPUT = os.path.join(OUTPUT_DIR, "best_camera_smooth.avi")
GRID_OUTPUT = os.path.join(OUTPUT_DIR, "grid_view.avi")

os.makedirs(OUTPUT_DIR, exist_ok=True)
# =========================================

# ================= KALMAN FILTER FOR BALL =================
class BallTracker:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        self.kf.R *= 10
        self.kf.P *= 1000
        self.kf.Q *= 0.01
        self.initialized = False
        self.missed_frames = 0
        self.max_missed = 15
        
    def update(self, cx, cy):
        if not self.initialized:
            self.kf.x[:2] = np.array([[cx], [cy]])
            self.initialized = True
        else:
            self.kf.predict()
            self.kf.update(np.array([[cx], [cy]]))
        self.missed_frames = 0
        return self.kf.x[0, 0], self.kf.x[1, 0]
    
    def predict(self):
        if not self.initialized or self.missed_frames > self.max_missed:
            return None, None
        self.kf.predict()
        self.missed_frames += 1
        return self.kf.x[0, 0], self.kf.x[1, 0]
    
    def reset(self):
        self.initialized = False
        self.missed_frames = 0

# ================= CAMERA SCORER =================
class CameraScorer:
    def __init__(self, cam_id, frame_w, frame_h):
        self.cam_id = cam_id
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.cx = frame_w / 2
        self.cy = frame_h / 2
        self.ema_score = 0.0
        self.last_box = None
        self.no_ball_frames = 0
        self.tracker = BallTracker()
        self.ball_velocity = (0, 0)  # vx, vy
        self.last_center = None
        
    def compute_score(self, box, conf):
        """
        box: (x1, y1, x2, y2)
        Returns: score 0.0 - 1.0
        """
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        max_area = self.frame_w * self.frame_h
        norm_area = min(area / (max_area * 0.15), 1.0)  # Cap at 15% frame area
        
        # Center proximity (gaussian falloff)
        bx, by = (x1 + x2) / 2, (y1 + y2) / 2
        dx = (bx - self.cx) / (self.frame_w / 2)
        dy = (by - self.cy) / (self.frame_h / 2)
        center_dist = np.sqrt(dx**2 + dy**2)
        center_score = np.exp(-2.0 * center_dist)  # 1.0 at center, ~0.13 at edge
        
        # Edge penalty (if ball is leaving frame)
        margin = 80
        edge_penalty = 0.0
        if x1 < margin or x2 > self.frame_w - margin or y1 < margin or y2 > self.frame_h - margin:
            edge_penalty = 0.5
        
        # Velocity alignment (prefer camera ball is moving TOWARD center of)
        if self.last_center is not None:
            vx = bx - self.last_center[0]
            vy = by - self.last_center[1]
            # If moving toward center, boost score
            to_center_x = self.cx - bx
            to_center_y = self.cy - by
            dot = vx * to_center_x + vy * to_center_y
            vel_score = 0.5 + 0.5 * np.tanh(dot / 50)  # -0.5 to 1.5 range, clamped
        else:
            vel_score = 0.5
            
        self.last_center = (bx, by)
        
        # Update Kalman
        self.tracker.update(bx, by)
        
        # Combine
        score = (
            W_AREA * norm_area +
            W_CENTER * center_score +
            W_VELOCITY * vel_score +
            W_CONF * conf -
            W_EDGE * edge_penalty
        )
        
        # EMA smoothing for temporal stability
        self.ema_score = EMA_ALPHA * score + (1 - EMA_ALPHA) * self.ema_score
        self.no_ball_frames = 0
        self.last_box = box
        
        return max(self.ema_score, 0.0)
    
    def miss(self):
        self.no_ball_frames += 1
        self.ema_score *= 0.7  # Decay score when ball lost
        self.last_box = None
        # Try Kalman prediction
        pred = self.tracker.predict()
        if pred[0] is not None:
            # Create predicted box around predicted center
            px, py = pred
            self.last_box = (int(px)-20, int(py)-20, int(px)+20, int(py)+20)
        return max(self.ema_score, 0.0)
    
    def is_lost(self):
        return self.no_ball_frames > NO_BALL_FRAME_LIMIT and not self.tracker.initialized

# ================= BROADCAST DIRECTOR =================
class BroadcastDirector:
    def __init__(self, num_cams, default_cam, source_fps):
        self.num_cams = num_cams
        self.default_cam = default_cam
        self.source_fps = source_fps
        
        self.current_cam = default_cam
        self.target_cam = default_cam
        self.state = "HOLD"  # HOLD, TRANSITION
        self.transition_frame = 0
        self.last_switch_time = time.time()
        self.min_shot_frames = int(MIN_SHOT_DURATION * source_fps)
        self.cooldown_frames = int(COOLDOWN_AFTER_SWITCH * source_fps)
        self.time_since_switch = 9999
        
        # Shot history to prevent ping-pong
        self.shot_history = deque(maxlen=5)
        
    def can_switch(self):
        return self.time_since_switch > (self.min_shot_frames + self.cooldown_frames)
    
    def request_switch(self, best_cam):
        if best_cam == self.current_cam:
            return False
        if not self.can_switch():
            return False
        # Anti ping-pong: don't go back to previous camera too fast
        if len(self.shot_history) > 0 and best_cam == self.shot_history[-1]:
            if self.time_since_switch < self.min_shot_frames * 2:
                return False
        
        self.target_cam = best_cam
        self.state = "TRANSITION"
        self.transition_frame = 0
        self.shot_history.append(self.current_cam)
        return True
    
    def update(self):
        self.time_since_switch += 1
        
        if self.state == "TRANSITION":
            self.transition_frame += 1
            alpha = self.transition_frame / TRANSITION_FRAMES
            
            if alpha >= 1.0:
                self.current_cam = self.target_cam
                self.state = "HOLD"
                self.last_switch_time = time.time()
                self.time_since_switch = 0
                return 1.0, self.current_cam, self.target_cam  # Done
            
            return alpha, self.current_cam, self.target_cam
        
        return 0.0, self.current_cam, self.current_cam
    
    def get_active_cam(self):
        return self.current_cam if self.state == "HOLD" else self.target_cam

# ================= OVERLAY =================
def draw_overlay(frame, cam_idx, is_tracking, score, blend_alpha=1.0):
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    # Top bar
    cv2.rectangle(overlay, (0, 0), (w, 55), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # Camera label
    color = (0, 255, 255) if is_tracking else (128, 128, 128)
    cv2.putText(frame, f"CAM {cam_idx + 1}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Status
    if is_tracking:
        status = "LIVE TRACK"
        status_color = (0, 255, 0)
    else:
        status = "SEARCHING..."
        status_color = (0, 165, 255)
    
    cv2.putText(frame, status, (150, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Score bar
    bar_w = int(200 * score)
    cv2.rectangle(frame, (w - 220, 15), (w - 20, 35), (50, 50, 50), -1)
    cv2.rectangle(frame, (w - 220, 15), (w - 220 + bar_w, 35), (0, 255, 0), -1)
    cv2.putText(frame, f"{score:.2f}", (w - 215, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

# ================= MAIN SYSTEM =================
class MultiCameraSystem:
    def __init__(self, video_paths):
        self.caps = [cv2.VideoCapture(p) for p in video_paths]
        if not all(c.isOpened() for c in self.caps):
            raise ValueError("Some videos failed to open")
        
        self.source_fps = self.caps[0].get(cv2.CAP_PROP_FPS) or 30.0
        # FPS used for the saved output files (compensated like Code 1)
        self.output_fps = self.source_fps / OUTPUT_SLOW_FACTOR
        print(f"[INFO] Source FPS: {self.source_fps:.2f} | Output FPS: "
              f"{self.output_fps:.2f} | Slow Factor: {OUTPUT_SLOW_FACTOR}x")
        self.num_cams = len(self.caps)
        
        # Use fine-tuned model if available, else yolov8n
        model_path = "weights/yolov8n.pt"  # Change to your fine-tuned model
        self.model = YOLO(model_path)
        
        # self.scorers = [CameraScorer(i, DETECT_SIZE[0], DETECT_SIZE[1]) 
        #                for i in range(self.num_cams)]
        # self.director = BroadcastDirector(self.num_cams, DEFAULT_CAMERA, self.source_fps)

        self.scorers = [CameraScorer(i, DETECT_SIZE[0], DETECT_SIZE[1]) 
                for i in range(self.num_cams)]

        self.default_cam = DEFAULT_CAMERA
        self.director = BroadcastDirector(
            self.num_cams,
            self.default_cam,
            self.source_fps
        )
        
        self.latest_frames = [None] * self.num_cams
        self.display_frames = [None] * self.num_cams
        
        self.lock = threading.Lock()
        self.running = True
        self.frame_count = 0
        
        # Writers
        self.best_writer = None
        self.grid_writer = None
        
    def camera_reader(self, idx):
        cap = self.caps[idx]
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        delay = 1.0 / fps
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.running = False
                break
            
            frame = cv2.resize(frame, DETECT_SIZE)
            
            with self.lock:
                self.latest_frames[idx] = frame.copy()
            
            time.sleep(max(0, delay - 0.001))
    
    def detection_worker(self):
        yolo_delay = 1.0 / YOLO_FPS
        
        while self.running:
            start = time.time()
            
            with self.lock:
                frames = [f.copy() if f is not None else None 
                         for f in self.latest_frames]
            
            # Global no-ball check for confidence tuning
            all_lost = all(s.is_lost() for s in self.scorers)
            conf = LOW_CONF if all_lost else NORMAL_CONF
            
            for i, frame in enumerate(frames):
                if frame is None:
                    continue
                
                if self.frame_count % DETECT_EVERY == 0:
                    results = self.model.predict(frame, conf=conf, verbose=False, 
                                                 classes=[32])  # COCO sports ball
                    boxes = results[0].boxes
                    
                    best_score = 0
                    best_box = None
                    
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            confidence = float(box.conf[0])
                            area = (x2 - x1) * (y2 - y1)
                            
                            # Filter tiny detections
                            if area < 100:
                                continue
                                
                            score = self.scorers[i].compute_score((x1, y1, x2, y2), confidence)
                            if score > best_score:
                                best_score = score
                                best_box = (x1, y1, x2, y2)
                    
                    if best_box is None:
                        self.scorers[i].miss()
                    else:
                        # Draw box on display frame
                        display = frame.copy()
                        cv2.rectangle(display, (best_box[0], best_box[1]), 
                                    (best_box[2], best_box[3]), (0, 255, 0), 2)
                        with self.lock:
                            self.display_frames[i] = display
                else:
                    # Kalman prediction frame - just predict
                    self.scorers[i].miss()  # Will use Kalman internally
            
            self.frame_count += 1
            sleep_time = yolo_delay - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def get_best_camera(self):
        """Decide which camera should be active based on scores"""
        scores = [s.ema_score for s in self.scorers]
        
        # If all lost, return default
        if all(s.is_lost() for s in self.scorers):
            return self.default_cam, scores, False
        
        best = int(np.argmax(scores))
        return best, scores, True
    
    def blend_frames(self, frame_a, frame_b, alpha):
        """Crossfade between two frames"""
        if frame_a is None:
            return frame_b
        if frame_b is None:
            return frame_a
        return cv2.addWeighted(frame_a, 1 - alpha, frame_b, alpha, 0)

    @staticmethod
    def _pick(display_frame, raw_frame):
        """
        Safely choose display_frame if it exists, otherwise raw_frame.
        NOTE: frames are numpy arrays, so plain `a or b` is invalid
        (raises "truth value of an array is ambiguous"). Must use
        explicit `is not None` checks instead.
        """
        return display_frame if display_frame is not None else raw_frame
    
    def run(self):
        # Start threads
        for i in range(self.num_cams):
            threading.Thread(target=self.camera_reader, args=(i,), 
                           daemon=True).start()
        threading.Thread(target=self.detection_worker, daemon=True).start()
        
        # Wait for first frames
        while any(f is None for f in self.latest_frames):
            time.sleep(0.05)
        
        frame_delay = 1.0 / self.source_fps
        
        while self.running:
            loop_start = time.time()
            
            with self.lock:
                raw_frames = [f.copy() if f is not None else None 
                             for f in self.latest_frames]
                disp_frames = [f.copy() if f is not None else None 
                              for f in self.display_frames]
            
            # Get scores and decide
            best_cam, scores, is_tracking = self.get_best_camera()
            
            # Director logic
            if is_tracking and best_cam != self.director.current_cam:
                # Check if worth switching (significant improvement)
                current_score = scores[self.director.current_cam]
                if scores[best_cam] > current_score * 1.2:  # 20% better
                    self.director.request_switch(best_cam)
            
            alpha, cam_from, cam_to = self.director.update()
            
            # Build output frame
            if alpha > 0 and cam_from != cam_to:
                # Transition: blend between cameras
                f_from = self._pick(disp_frames[cam_from], raw_frames[cam_from])
                f_to = self._pick(disp_frames[cam_to], raw_frames[cam_to])
                
                if f_from is not None and f_to is not None:
                    # Resize to output size
                    f_from = cv2.resize(f_from, DISPLAY_SIZE)
                    f_to = cv2.resize(f_to, DISPLAY_SIZE)
                    best_frame = self.blend_frames(f_from, f_to, alpha)
                    active_idx = cam_to  # Show target info during fade
                else:
                    fallback = raw_frames[self.director.current_cam]
                    if fallback is None:
                        fallback = np.zeros((DETECT_SIZE[1], DETECT_SIZE[0], 3), np.uint8)
                    best_frame = cv2.resize(fallback, DISPLAY_SIZE)
                    active_idx = self.director.current_cam
            else:
                # Hold shot
                src = self._pick(disp_frames[self.director.current_cam], raw_frames[self.director.current_cam])
                if src is not None:
                    best_frame = cv2.resize(src, DISPLAY_SIZE)
                else:
                    best_frame = np.zeros((DISPLAY_SIZE[1], DISPLAY_SIZE[0], 3), np.uint8)
                active_idx = self.director.current_cam
            
            # Overlay
            best_frame = draw_overlay(best_frame, active_idx, is_tracking, 
                                     scores[active_idx] if active_idx < len(scores) else 0)
            
            # Grid view
            grid_parts = []
            for i in range(self.num_cams):
                src = self._pick(disp_frames[i], raw_frames[i])
                if src is not None:
                    small = cv2.resize(src, (DISPLAY_SIZE[0]//2, DISPLAY_SIZE[1]//2))
                    # Highlight active camera in grid
                    if i == active_idx:
                        cv2.rectangle(small, (0,0), (small.shape[1]-1, small.shape[0]-1), 
                                    (0, 255, 0), 3)
                else:
                    small = np.zeros((DISPLAY_SIZE[1]//2, DISPLAY_SIZE[0]//2, 3), np.uint8)
                grid_parts.append(small)
            
            # Pad to 4 if needed
            while len(grid_parts) < 4:
                grid_parts.append(np.zeros_like(grid_parts[0]))
            
            top = np.hstack((grid_parts[0], grid_parts[1]))
            bottom = np.hstack((grid_parts[2], grid_parts[3]))
            grid = np.vstack((top, bottom))
            
            # Show
            cv2.imshow("Best Camera (Smooth)", best_frame)
            cv2.imshow("Grid View", grid)
            
            # Init writers
            if self.best_writer is None:
                h, w = best_frame.shape[:2]
                self.best_writer = cv2.VideoWriter(
                    BEST_CAM_OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"),
                    self.output_fps, (w, h))
                self.grid_writer = cv2.VideoWriter(
                    GRID_OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"),
                    self.output_fps, (grid.shape[1], grid.shape[0]))
                print(f"[INFO] Recording started")
            
            self.best_writer.write(best_frame)
            self.grid_writer.write(grid)
            
            # FPS control
            elapsed = time.time() - loop_start
            sleep_time = frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
        
        self.cleanup()
    
    def cleanup(self):
        self.running = False
        for cap in self.caps:
            cap.release()
        if self.best_writer:
            self.best_writer.release()
            print(f"[INFO] Saved: {BEST_CAM_OUTPUT}")
        if self.grid_writer:
            self.grid_writer.release()
            print(f"[INFO] Saved: {GRID_OUTPUT}")
        cv2.destroyAllWindows()

# ================= RUN =================
if __name__ == "__main__":
    video_paths = [
        "inference/videos/md.mp4",
        "inference/videos/lf.mp4", 
        "inference/videos/rt.mp4",
    ]
    
    system = MultiCameraSystem(video_paths)
    system.run()