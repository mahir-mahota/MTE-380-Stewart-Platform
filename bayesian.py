import cv2
import numpy as np
import serial
import time
import json
import traceback
import random
from datetime import datetime
from balancer_controller.Balancer import PositionBalancer
from bayes_opt import BayesianOptimization

class BallTracker:
    """Ball tracker with persistent camera/serial, and resettable balancer for optimization."""

    def __init__(self, config_path="config.json", serial_port="COM7", baud_rate=115200):
        # Load config
        with open(config_path, "r") as f:
            cfg = json.load(f)

        self.frame_w = cfg["camera"]["frame_width"]
        self.frame_h = cfg["camera"]["frame_height"]
        self.frame_center = np.array([self.frame_w // 2, self.frame_h // 2])

        # HSV for ball detection
        self.lower_hsv = np.array(cfg["ball_detection"]["lower_hsv"], dtype=np.uint8)
        self.upper_hsv = np.array(cfg["ball_detection"]["upper_hsv"], dtype=np.uint8)

        # Persistent camera
        self.cap = cv2.VideoCapture(cfg["camera"]["index"], cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_h)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera.")

        # Persistent serial
        try:
            self.ser = serial.Serial(serial_port, baud_rate, timeout=1)
            time.sleep(2)
            print(f"[INFO] Serial connected on {serial_port} at {baud_rate}")
        except Exception as e:
            print(f"[WARN] Could not open serial: {e}")
            self.ser = None

        # Fixed platform points (relative to centroid)
        self.platform_points_rel = [
            np.array([206.0, -122.0]),    # P1
            np.array([-204.0, -119.0]), # P2
            np.array([-8.0, 241]) # P3
        ]

        cv2.namedWindow("Ball Tracker")
        self.ball_pos = None
        self.balancer = None

    # ===========================
    # Ball detection
    # ===========================
    def detect_ball(self, frame):
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            c = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(c)
            if radius < 5:
                return None
            rel_x = x - self.frame_center[0]
            rel_y = y - self.frame_center[1]
            return np.array([rel_x, rel_y])
        except Exception:
            traceback.print_exc()
            return None

    # ===========================
    # Draw overlay
    # ===========================
    def draw_overlay(self, frame):
        try:
            cv2.drawMarker(frame, tuple(self.frame_center.astype(int)), (255, 255, 255),
                           cv2.MARKER_CROSS, 15, 2)
            if self.ball_pos is not None:
                abs_pos = (self.ball_pos + self.frame_center).astype(int)
                cv2.circle(frame, tuple(abs_pos), 8, (0, 255, 255), 2)
                cv2.circle(frame, tuple(abs_pos), 3, (0, 255, 255), -1)
            if self.balancer:
                gains_text = f"Kp:{self.balancer.pid.Kp:.5f} Ki:{self.balancer.pid.Ki:.5f} Kd:{self.balancer.pid.Kd:.5f}"
                cv2.putText(frame, gains_text, (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                des_text = f"Desired: {self.balancer.desired_pos}"
                cv2.putText(frame, des_text, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            return frame
        except Exception:
            traceback.print_exc()
            return frame

    # ===========================
    # Serial send
    # ===========================
    def send_serial(self, data):
        if self.ser is None:
            return
        try:
            msg = ",".join(f"{int(v)}" for v in np.ravel(data)) + "\n"
            self.ser.write(msg.encode("utf-8"))
        except Exception:
            traceback.print_exc()

    # ===========================
    # Run single PID evaluation
    # ===========================
    def run_pid(self, Kp, Ki, Kd, duration=10, target_range=90):
        """Run balancer with given PID for `duration` seconds, moving target randomly."""
        self.balancer = PositionBalancer(self.platform_points_rel)
        self.balancer.pid.Kp = Kp
        self.balancer.pid.Ki = Ki
        self.balancer.pid.Kd = Kd
        self.balancer.desired_pos = np.array([0.0, 0.0])

        total_error = 0.0
        start_time = time.time()
        last_target_change = start_time

        while time.time() - start_time < duration:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue

            # Ball detection
            ball = self.detect_ball(frame)
            self.ball_pos = ball

            # Random target every second
            if time.time() - last_target_change > 1.0:
                self.balancer.desired_pos = np.array([
                    random.uniform(-target_range, target_range),
                    random.uniform(-target_range, target_range)
                ])
                last_target_change = time.time()

            # Step balancer
            dt = 0.01
            output = self.balancer.step(ball, dt)
            self.send_serial(output)

            # Accumulate error
            if self.balancer.curr_error is not None:
                total_error += np.linalg.norm(self.balancer.curr_error[:2])

            # Display
            cv2.imshow("Ball Tracker", self.draw_overlay(frame))
            cv2.waitKey(1)

        return -total_error  # negative because Bayesian optimizer maximizes by default


if __name__ == "__main__":
    tracker = BallTracker("config.json", serial_port="COM7", baud_rate=115200)

    # ===============================
    # Bayesian Optimization setup
    # ===============================
    def eval_pid(Kp, Ki, Kd):
        return tracker.run_pid(Kp, Ki, Kd, duration=10, target_range=90)

    pbounds = {
        'Kp': (0.00001, 0.001),
        'Ki': (0.00001, 0.001),
        'Kd': (0.00001, 0.001)
    }

    optimizer = BayesianOptimization(
        f=eval_pid,
        pbounds=pbounds,
        verbose=2,
        random_state=42
    )

    optimizer.maximize(init_points=3, n_iter=10)

    print("Best PID found:", optimizer.max)
