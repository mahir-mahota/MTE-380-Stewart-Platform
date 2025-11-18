import cv2
import numpy as np
import json
from datetime import datetime
from balancer_controller.Balancer import PositionBalancer
import time
import serial
import traceback
import threading


class BallTracker:
    """Ball tracking system with PID sliders + clicking to set target position."""

    def __init__(self, config_path="config.json", serial_port="COM7", baud_rate=115200):
        # Load config
        with open(config_path, "r") as f:
            cfg = json.load(f)

        self.lower_hsv = np.array(cfg["ball_detection"]["lower_hsv"], dtype=np.uint8)
        self.upper_hsv = np.array(cfg["ball_detection"]["upper_hsv"], dtype=np.uint8)
        self.frame_w = cfg["camera"]["frame_width"]
        self.frame_h = cfg["camera"]["frame_height"]
        self.cam_index = cfg["camera"]["index"]

        # initial camera center (replaced after 3 clicks)
        self.frame_center = np.array([self.frame_w // 2, self.frame_h // 2])

        self.platform_points_abs = []
        self.platform_points_rel = []
        self.selected_points = []

        self.ball_pos = None
        self.balancer = None
        self.prev_time = None
        self.last_alive_time = time.time()

        self.tracking_active = False  # <--- NEW: detect when tracking mode begins

        # Serial setup
        try:
            self.ser = serial.Serial(serial_port, baud_rate, timeout=1)
            time.sleep(2)
            print(f"[INFO] Serial connected to {serial_port} at {baud_rate} baud.")
        except Exception as e:
            print(f"[WARN] Could not open serial port {serial_port}: {e}")
            self.ser = None

        # Camera setup
        self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_h)

        if not self.cap.isOpened():
            print("[ERROR] Failed to open camera.")

        cv2.namedWindow("Ball Tracker")
        cv2.setMouseCallback("Ball Tracker", self.mouse_callback)

        print("[INFO] BallTracker initialized.")
        print("Click 3 platform points, press SPACE. Then click anywhere to set the target position.")
        print("Press 'q' to quit.")

        self.start_pid_input_thread()
        self.create_pid_trackbars()

    # ============================================================
    #       PID TRACKBARS
    # ============================================================
    def create_pid_trackbars(self):
        cv2.namedWindow("PID Tuners")
        cv2.createTrackbar("Kp x1e-5", "PID Tuners", 1, 100, self.on_trackbar)
        cv2.createTrackbar("Ki x1e-5", "PID Tuners", 1, 100, self.on_trackbar)
        cv2.createTrackbar("Kd x1e-5", "PID Tuners", 1, 100, self.on_trackbar)

    def on_trackbar(self, value):
        if not self.balancer:
            return

        Kp_raw = cv2.getTrackbarPos("Kp x1e-5", "PID Tuners") / 100000.0
        Ki_raw = cv2.getTrackbarPos("Ki x1e-5", "PID Tuners") / 100000.0
        Kd_raw = cv2.getTrackbarPos("Kd x1e-5", "PID Tuners") / 100000.0

        self.balancer.pid.Kp = Kp_raw
        self.balancer.pid.Ki = Ki_raw
        self.balancer.pid.Kd = Kd_raw

        print(f"[SLIDER] PID -> Kp={Kp_raw:.6f}, Ki={Ki_raw:.6f}, Kd={Kd_raw:.6f}")

    # ============================================================
    #       PID INPUT THREAD
    # ============================================================
    def update_pid_gains_via_input(self):
        try:
            inp = input("Enter PID gains (Kp,Ki,Kd): ").strip()
            if inp:
                Kp, Ki, Kd = map(float, inp.split(","))
                if self.balancer:
                    self.balancer.pid.Kp = Kp
                    self.balancer.pid.Ki = Ki
                    self.balancer.pid.Kd = Kd
                    print("[INFO] Updated PID via typing.")
        except:
            traceback.print_exc()

    def start_pid_input_thread(self):
        t = threading.Thread(target=lambda: [self.update_pid_gains_via_input()], daemon=True)
        t.start()

    # ============================================================
    #       MOUSE INPUT (MODIFIED)
    # ============================================================
    def mouse_callback(self, event, x, y, flags, param):
        # === FIRST: selecting 3 platform points ===
        if event == cv2.EVENT_LBUTTONDOWN and not self.tracking_active:
            if len(self.platform_points_abs) < 3:
                abs_pt = np.array([x, y], dtype=float)
                self.platform_points_abs.append(abs_pt)
                self.selected_points.append((x, y))
                print(f"[POINT] Platform point {len(self.platform_points_abs)} selected at ABS {abs_pt}")
            return

        # === AFTER tracking begins: click sets desired_pos ===
        if event == cv2.EVENT_LBUTTONDOWN and self.tracking_active:
            rel_x = x - self.frame_center[0]
            rel_y = y - self.frame_center[1]

            target = np.array([rel_x, rel_y])
            self.balancer.desired_pos = target

            print(f"[TARGET] Clicked => Desired position set to {target}")

    # ============================================================
    #       DETECTION
    # ============================================================
    def detect_ball(self, frame):
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
            mask = cv2.erode(mask, None, 2)
            mask = cv2.dilate(mask, None, 2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            c = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(c)

            if radius < 5:
                return None

            rel_x = x - self.frame_center[0]
            rel_y = y - self.frame_center[1]

            return (rel_x, rel_y, int(radius))

        except:
            traceback.print_exc()
            return None

    # ============================================================
    #       DRAW OVERLAY
    # ============================================================
    def draw_overlay(self, frame):
        try:
            cv2.drawMarker(frame, tuple(self.frame_center.astype(int)), (255, 255, 255),
                           cv2.MARKER_CROSS, 15, 2)

            for i, p in enumerate(self.selected_points):
                cv2.circle(frame, p, 6, (0, 255, 0), -1)
                cv2.putText(frame, f"P{i+1}", (p[0]+10, p[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if self.ball_pos is not None:
                rel_x, rel_y, radius = self.ball_pos
                ax = int(rel_x + self.frame_center[0])
                ay = int(rel_y + self.frame_center[1])
                cv2.circle(frame, (ax, ay), radius, (0, 255, 255), 2)

            if self.balancer:
                t = self.balancer.desired_pos
                cv2.putText(frame, f"TGT: {t[0]:.1f}, {t[1]:.1f}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 200, 0), 2)
            return frame

        except:
            traceback.print_exc()
            return frame

    # ============================================================
    #       SERIAL OUT
    # ============================================================
    def send_serial(self, data):
        if not self.ser:
            return
        try:
            msg = ",".join(str(int(v)) for v in data) + "\n"
            self.ser.write(msg.encode())
        except:
            traceback.print_exc()

    # ============================================================
    #       MAIN LOOP
    # ============================================================
    def run(self):
        collecting_points = True
        period = 0.05
        last = time.time()

        while True:
            try:
                ok, frame = self.cap.read()
                if not ok:
                    continue

                if collecting_points:
                    display = self.draw_overlay(frame.copy())
                    cv2.imshow("Ball Tracker", display)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(" "):
                        if len(self.platform_points_abs) == 3:
                            arr = np.array(self.platform_points_abs)
                            centroid = np.mean(arr, axis=0)
                            self.frame_center = centroid.copy()

                            self.platform_points_rel = [
                                np.array([p[0] - centroid[0],
                                          p[1] - centroid[1], 0.0])
                                for p in arr
                            ]

                            self.balancer = PositionBalancer(self.platform_points_rel)

                            # Default target at (0,0)
                            self.balancer.desired_pos = np.array([0.0, 0.0])

                            self.tracking_active = True  # <--- NOW CLICKS CHANGE TARGET
                            collecting_points = False

                            print("[INFO] Tracking started.")
                        else:
                            print("[WARN] Select all 3 points first.")
                    elif key == ord("q"):
                        break

                    continue

                # === Tracking mode ===
                ball = self.detect_ball(frame)
                self.ball_pos = ball
                ball_np = np.array([ball[0], ball[1], 0.0]) if ball else None

                now = time.time()
                if now - last >= period:
                    dt = now - last
                    last = now

                    output = self.balancer.step(ball_np, dt)
                    self.send_serial(output)

                display = self.draw_overlay(frame.copy())
                cv2.imshow("Ball Tracker", display)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            except:
                traceback.print_exc()

        self.cap.release()
        if self.ser:
            self.ser.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = BallTracker("config.json", "COM7", 115200)
    tracker.run()
