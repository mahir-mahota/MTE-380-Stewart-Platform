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
    """Ball tracking system with live PID typing, fixed-rate control, and debug logs."""

    def __init__(self, config_path="config.json", serial_port="COM7", baud_rate=115200):
        # Load config
        with open(config_path, "r") as f:
            cfg = json.load(f)

        self.lower_hsv = np.array(cfg["ball_detection"]["lower_hsv"], dtype=np.uint8)
        self.upper_hsv = np.array(cfg["ball_detection"]["upper_hsv"], dtype=np.uint8)
        self.frame_w = cfg["camera"]["frame_width"]
        self.frame_h = cfg["camera"]["frame_height"]
        self.cam_index = cfg["camera"]["index"]

        # initial camera center (will be REPLACED by point centroid)
        self.frame_center = np.array([self.frame_w // 2, self.frame_h // 2])

        self.platform_points_abs = []   # store absolute pixel coordinates
        self.platform_points_rel = []   # will be computed later
        self.selected_points = []
        self.ball_pos = None
        self.balancer = None
        self.prev_time = None
        self.last_alive_time = time.time()

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
            print("[ERROR] Failed to open camera. Check cam index or permissions.")

        cv2.namedWindow("Ball Tracker")
        cv2.setMouseCallback("Ball Tracker", self.mouse_callback)

        print("[INFO] BallTracker initialized.")
        print("Click 3 platform points, then press 'SPACE' to start tracking.")
        print("Press 'q' to quit.")

        # Start PID input thread
        self.start_pid_input_thread()

    # ---------------- PID TYPING ----------------
    def update_pid_gains_via_input(self):
        """Allow user to type Kp, Ki, Kd values live in the console."""
        try:
            inp = input("Enter PID gains as Kp,Ki,Kd (or empty to skip): ").strip()
            if inp:
                parts = inp.split(",")
                if len(parts) == 3:
                    Kp, Ki, Kd = map(float, parts)
                    if self.balancer:
                        self.balancer.pid.Kp = Kp
                        self.balancer.pid.Ki = Ki
                        self.balancer.pid.Kd = Kd
                        print(f"[INFO] Updated PID gains -> Kp: {Kp}, Ki: {Ki}, Kd: {Kd}")
                else:
                    print("[WARN] Please enter 3 comma-separated values.")
        except Exception:
            print("[ERROR] Failed to parse PID input:")
            traceback.print_exc()

    def start_pid_input_thread(self):
        """Run PID typing input in a background thread."""
        def loop():
            while True:
                self.update_pid_gains_via_input()
        t = threading.Thread(target=loop, daemon=True)
        t.start()

    # ---------------- LOGGING ----------------
    def log_heartbeat(self, tag):
        now = time.time()
        dt = now - self.last_alive_time
        self.last_alive_time = now
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] OK -> {tag} (+{dt*1000:.1f} ms)")

    # ---------------- MOUSE ----------------
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.platform_points_abs) < 3:
            abs_pt = np.array([x, y], dtype=float)
            self.platform_points_abs.append(abs_pt)
            self.selected_points.append((x, y))
            print(f"[POINT] Platform point {len(self.platform_points_abs)} selected at ABS {abs_pt}")

    # ---------------- DETECTION ----------------
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
            return (rel_x, rel_y, int(radius))
        except Exception:
            print("[ERROR] detect_ball crashed:")
            traceback.print_exc()
            return None

    # ---------------- DRAWING ----------------
    def draw_overlay(self, frame):
        try:
            # draw the NEW computed center
            cv2.drawMarker(frame, tuple(self.frame_center.astype(int)), (255, 255, 255),
                           cv2.MARKER_CROSS, 15, 2)

            # draw selected calibration points
            for i, p in enumerate(self.selected_points):
                cv2.circle(frame, p, 6, (0, 255, 0), -1)
                cv2.putText(frame, f"P{i+1}", (p[0]+8, p[1]-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # draw ball
            if self.ball_pos is not None:
                rel_x, rel_y, radius = self.ball_pos
                abs_x = int(rel_x + self.frame_center[0])
                abs_y = int(rel_y + self.frame_center[1])
                cv2.circle(frame, (abs_x, abs_y), radius, (0, 255, 255), 2)
                cv2.circle(frame, (abs_x, abs_y), 3, (0, 255, 255), -1)

            # Display PID gains
            if self.balancer:
                gains_text = f"Kp: {self.balancer.pid.Kp:.5f}  Ki: {self.balancer.pid.Ki:.5f}  Kd: {self.balancer.pid.Kd:.5f}"
                cv2.putText(frame, gains_text, (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            return frame
        except Exception:
            print("[ERROR] draw_overlay failed:")
            traceback.print_exc()
            return frame

    # ---------------- SERIAL ----------------
    def send_serial(self, data):
        if self.ser is None:
            return
        try:
            msg = ",".join(f"{int(v)}" for v in np.ravel(data)) + "\n"
            self.ser.write(msg.encode("utf-8"))
            self.log_heartbeat("serial_sent")
        except Exception:
            print("[ERROR] Serial send failed:")
            traceback.print_exc()

    # ---------------- MAIN LOOP ----------------
    def run(self):
        collecting_points = True
        control_period = 0.05  # 100ms loop
        last_control_time = time.time()

        while True:
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("[WARN] Empty frame from camera.")
                    continue
                self.log_heartbeat("frame_read")

                if collecting_points:
                    display = self.draw_overlay(frame.copy())
                    cv2.putText(display, f"Select {3 - len(self.platform_points_abs)} more points.",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow("Ball Tracker", display)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord(' '):
                        if len(self.platform_points_abs) == 3:

                            # ---- Compute centroid in ABSOLUTE coordinates ----
                            arr = np.array(self.platform_points_abs)  # Nx2
                            centroid = np.mean(arr, axis=0)           # (cx, cy)

                            print(f"[INFO] Platform centroid (ABS): {centroid}")

                            # update frame center
                            self.frame_center = centroid.copy()

                            # convert to REL coordinates (centered at centroid)
                            rel_points = []
                            for p in arr:
                                rel = np.array([p[0] - centroid[0],
                                                p[1] - centroid[1],
                                                0.0])
                                rel_points.append(rel)

                            self.platform_points_rel = rel_points

                            print("[INFO] Relative platform points:", self.platform_points_rel)

                            # create balancer
                            self.balancer = PositionBalancer(self.platform_points_rel)

                            self.prev_time = time.time()
                            last_control_time = self.prev_time

                            collecting_points = False
                            print("[INFO] Platform points fixed. Starting tracking...")

                        else:
                            print("[WARN] Please select exactly 3 points first.")

                    elif key == ord('q'):
                        break
                    continue

                # --- Ball detection ---
                ball = self.detect_ball(frame)
                self.ball_pos = ball
                self.log_heartbeat("ball_detected")
                ball_pos_np = np.array([ball[0], ball[1], 0]) if ball else None

                # --- Control loop ---
                now = time.time()
                if now - last_control_time >= control_period:
                    dt = now - self.prev_time
                    self.prev_time = now
                    last_control_time = now

                    output = self.balancer.step(ball_pos_np, dt)
                    self.log_heartbeat("balancer_step")

                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] control output: {output}")
                    self.send_serial(output)

                # --- Display ---
                display = self.draw_overlay(frame)
                cv2.imshow("Ball Tracker", display)
                self.log_heartbeat("frame_displayed")

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            except Exception:
                print("[ERROR] main loop iteration crashed:")
                traceback.print_exc()
                time.sleep(0.1)

        # cleanup
        self.cap.release()
        if self.ser:
            self.ser.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
        tracker = BallTracker("config.json", serial_port="COM7", baud_rate=115200)
        tracker.run()
