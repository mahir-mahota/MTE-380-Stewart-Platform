import cv2
import numpy as np
import json
from datetime import datetime
from balancer_controller.Balancer import PositionBalancer
import time
import serial
import traceback
import threading
from pid_optimizer import PIDOptimizer, OptimizationResult
from typing import List, Tuple, Optional


class BallTrackerWithOptimizer:
    """Enhanced ball tracking system with integrated PID optimization."""

    def __init__(self, config_path="config.json", serial_port="COM7", baud_rate=115200):
        # Load config
        with open(config_path, "r") as f:
            cfg = json.load(f)

        self.lower_hsv = np.array(cfg["ball_detection"]["lower_hsv"], dtype=np.uint8)
        self.upper_hsv = np.array(cfg["ball_detection"]["upper_hsv"], dtype=np.uint8)
        self.frame_w = cfg["camera"]["frame_width"]
        self.frame_h = cfg["camera"]["frame_height"]
        self.cam_index = cfg["camera"]["index"]

        # Initial camera center (replaced after 3 clicks)
        self.frame_center = np.array([self.frame_w // 2, self.frame_h // 2])

        self.platform_points_abs = []
        self.platform_points_rel = []
        self.selected_points = []

        self.ball_pos = None
        self.balancer = None
        self.prev_time = None
        self.last_alive_time = time.time()

        self.tracking_active = False

        # Optimization mode
        self.optimization_mode = False
        self.optimizer = PIDOptimizer({'Kp': 0.00016, 'Ki': 0.00001, 'Kd': 0.00013})
        self.current_test_trajectory = []
        self.current_test_errors = []
        self.test_start_time = None
        self.test_duration = 10.0  # seconds per test
        self.optimization_queue = []
        self.current_optimization_params = None
        
        # Performance tracking (always active)
        self.performance_tracker = {
            'trajectory': [],
            'errors': [],
            'start_time': None,
            'recording': False
        }

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

        print("[INFO] BallTrackerWithOptimizer initialized.")
        print("Click 3 platform points, press SPACE. Then:")
        print("  - Click anywhere to set target position")
        print("  - Press 'o' to start optimization mode")
        print("  - Press 'g' for grid search optimization")
        print("  - Press 'r' for random search optimization")
        print("  - Press 'a' for adaptive search optimization")
        print("  - Press 's' to save optimization results")
        print("  - Press 'p' to plot optimization results")
        print("  - Press 'e' to evaluate current performance")
        print("  - Press 'q' to quit")

        self.create_pid_trackbars()

    # ============================================================
    #       PID TRACKBARS
    # ============================================================
    def create_pid_trackbars(self):
        cv2.namedWindow("PID Tuners")
        cv2.createTrackbar("Kp x1e-5", "PID Tuners", 16, 100, self.on_trackbar)
        cv2.createTrackbar("Ki x1e-5", "PID Tuners", 1, 100, self.on_trackbar)
        cv2.createTrackbar("Kd x1e-5", "PID Tuners", 13, 100, self.on_trackbar)

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
    #       MOUSE INPUT
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

            # Reset performance tracking when target changes
            self.reset_performance_tracking()

            print(f"[TARGET] Clicked => Desired position set to {target}")

    # ============================================================
    #       PERFORMANCE TRACKING
    # ============================================================
    def reset_performance_tracking(self):
        """Reset performance tracking for new target or test."""
        self.performance_tracker = {
            'trajectory': [],
            'errors': [],
            'start_time': time.time(),
            'recording': True
        }
        
    def update_performance_tracking(self, ball_pos: Optional[Tuple[float, float]], 
                                  target: np.ndarray):
        """Update performance tracking data."""
        if not self.performance_tracker['recording']:
            return
            
        if ball_pos is not None:
            pos = np.array([ball_pos[0], ball_pos[1]])
            error = np.linalg.norm(target - pos)
            
            self.performance_tracker['trajectory'].append((pos[0], pos[1]))
            self.performance_tracker['errors'].append(error)

    def evaluate_current_performance(self):
        """Evaluate and display current performance metrics."""
        if not self.performance_tracker['trajectory']:
            print("[PERFORMANCE] No data to evaluate")
            return
            
        metrics = self.optimizer.evaluate_performance(
            self.performance_tracker['trajectory'],
            self.performance_tracker['errors'],
            self.balancer.desired_pos if self.balancer else np.array([0, 0])
        )
        
        score = self.optimizer.calculate_score(metrics)
        
        print("\n" + "="*50)
        print("[PERFORMANCE EVALUATION]")
        print(f"  Current PID: Kp={self.balancer.pid.Kp:.6f}, Ki={self.balancer.pid.Ki:.6f}, Kd={self.balancer.pid.Kd:.6f}")
        print(f"  Settling Time: {metrics['settling_time']:.2f} s")
        print(f"  Overshoot: {metrics['overshoot']*100:.1f}%")
        print(f"  Steady-State Error: {metrics['steady_state_error']:.2f} pixels")
        print(f"  Rise Time: {metrics['rise_time']:.2f} s")
        print(f"  Oscillations: {metrics['oscillation_count']}")
        print(f"  Average Error: {metrics['total_error']:.2f} pixels")
        print(f"  Overall Score: {score:.4f} (lower is better)")
        print("="*50 + "\n")

    # ============================================================
    #       OPTIMIZATION METHODS
    # ============================================================
    def test_pid_parameters(self, kp: float, ki: float, kd: float) -> Tuple[List, List]:
        """
        Test a set of PID parameters and collect performance data.
        
        Args:
            kp, ki, kd: PID parameters to test
            
        Returns:
            Tuple of (trajectory, errors)
        """
        print(f"[TEST] Testing Kp={kp:.6f}, Ki={ki:.6f}, Kd={kd:.6f}")
        
        # Set the PID parameters
        if self.balancer:
            self.balancer.pid.Kp = kp
            self.balancer.pid.Ki = ki
            self.balancer.pid.Kd = kd
            self.balancer.pid.reset()  # Reset integral and derivative
        
        # Reset tracking
        trajectory = []
        errors = []
        start_time = time.time()
        
        # Run test for specified duration
        while time.time() - start_time < self.test_duration:
            ok, frame = self.cap.read()
            if not ok:
                continue
            
            ball = self.detect_ball(frame)
            
            if ball:
                pos = np.array([ball[0], ball[1]])
                error = np.linalg.norm(self.balancer.desired_pos - pos)
                trajectory.append((pos[0], pos[1]))
                errors.append(error)
                
                # Update balancer
                ball_np = np.array([ball[0], ball[1], 0.0])
                dt = 0.05  # Fixed dt for testing
                output = self.balancer.step(ball_np, dt)
                self.send_serial(output)
            
            # Display
            display = self.draw_overlay(frame.copy())
            cv2.putText(display, f"TESTING: {time.time() - start_time:.1f}s / {self.test_duration}s",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Ball Tracker", display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        return trajectory, errors

    def run_grid_search_optimization(self):
        """Run grid search optimization."""
        if not self.balancer:
            print("[ERROR] Initialize tracking first (select 3 points and press SPACE)")
            return
        
        # Define search ranges (centered around current values)
        current_kp = self.balancer.pid.Kp
        current_ki = self.balancer.pid.Ki
        current_kd = self.balancer.pid.Kd
        
        param_ranges = {
            'Kp': (current_kp * 0.5, current_kp * 2.0, 5),  # 5 steps
            'Ki': (current_ki * 0.5, current_ki * 2.0, 3),  # 3 steps
            'Kd': (current_kd * 0.5, current_kd * 2.0, 4)   # 4 steps
        }
        
        print("\n[OPTIMIZATION] Starting Grid Search...")
        best_result = self.optimizer.grid_search(param_ranges, self.test_pid_parameters)
        
        if best_result:
            print(f"\n[OPTIMIZATION] Best parameters found:")
            print(f"  Kp={best_result.Kp:.6f}")
            print(f"  Ki={best_result.Ki:.6f}")
            print(f"  Kd={best_result.Kd:.6f}")
            print(f"  Score={best_result.score:.4f}")
            
            # Apply best parameters
            self.balancer.pid.Kp = best_result.Kp
            self.balancer.pid.Ki = best_result.Ki
            self.balancer.pid.Kd = best_result.Kd
            
            # Update trackbars
            cv2.setTrackbarPos("Kp x1e-5", "PID Tuners", int(best_result.Kp * 100000))
            cv2.setTrackbarPos("Ki x1e-5", "PID Tuners", int(best_result.Ki * 100000))
            cv2.setTrackbarPos("Kd x1e-5", "PID Tuners", int(best_result.Kd * 100000))

    def run_random_search_optimization(self):
        """Run random search optimization."""
        if not self.balancer:
            print("[ERROR] Initialize tracking first (select 3 points and press SPACE)")
            return
        
        current_kp = self.balancer.pid.Kp
        current_ki = self.balancer.pid.Ki
        current_kd = self.balancer.pid.Kd
        
        param_ranges = {
            'Kp': (current_kp * 0.3, current_kp * 3.0),
            'Ki': (current_ki * 0.3, current_ki * 3.0),
            'Kd': (current_kd * 0.3, current_kd * 3.0)
        }
        
        print("\n[OPTIMIZATION] Starting Random Search...")
        best_result = self.optimizer.random_search(param_ranges, 20, self.test_pid_parameters)
        
        if best_result:
            print(f"\n[OPTIMIZATION] Best parameters found:")
            print(f"  Kp={best_result.Kp:.6f}")
            print(f"  Ki={best_result.Ki:.6f}")
            print(f"  Kd={best_result.Kd:.6f}")
            print(f"  Score={best_result.score:.4f}")
            
            # Apply best parameters
            self.balancer.pid.Kp = best_result.Kp
            self.balancer.pid.Ki = best_result.Ki
            self.balancer.pid.Kd = best_result.Kd
            
            # Update trackbars
            cv2.setTrackbarPos("Kp x1e-5", "PID Tuners", int(best_result.Kp * 100000))
            cv2.setTrackbarPos("Ki x1e-5", "PID Tuners", int(best_result.Ki * 100000))
            cv2.setTrackbarPos("Kd x1e-5", "PID Tuners", int(best_result.Kd * 100000))

    def run_adaptive_search_optimization(self):
        """Run adaptive search optimization."""
        if not self.balancer:
            print("[ERROR] Initialize tracking first (select 3 points and press SPACE)")
            return
        
        current_kp = self.balancer.pid.Kp
        current_ki = self.balancer.pid.Ki
        current_kd = self.balancer.pid.Kd
        
        param_ranges = {
            'Kp': (current_kp * 0.2, current_kp * 5.0),
            'Ki': (current_ki * 0.2, current_ki * 5.0),
            'Kd': (current_kd * 0.2, current_kd * 5.0)
        }
        
        print("\n[OPTIMIZATION] Starting Adaptive Search...")
        best_result = self.optimizer.adaptive_search(param_ranges, self.test_pid_parameters, 30)
        
        if best_result:
            print(f"\n[OPTIMIZATION] Best parameters found:")
            print(f"  Kp={best_result.Kp:.6f}")
            print(f"  Ki={best_result.Ki:.6f}")
            print(f"  Kd={best_result.Kd:.6f}")
            print(f"  Score={best_result.score:.4f}")
            
            # Apply best parameters
            self.balancer.pid.Kp = best_result.Kp
            self.balancer.pid.Ki = best_result.Ki
            self.balancer.pid.Kd = best_result.Kd
            
            # Update trackbars
            cv2.setTrackbarPos("Kp x1e-5", "PID Tuners", int(best_result.Kp * 100000))
            cv2.setTrackbarPos("Ki x1e-5", "PID Tuners", int(best_result.Ki * 100000))
            cv2.setTrackbarPos("Kd x1e-5", "PID Tuners", int(best_result.Kd * 100000))

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
                
                # Show current position
                cv2.putText(frame, f"Ball: ({rel_x:.1f}, {rel_y:.1f})",
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           (0, 255, 255), 2)

            if self.balancer:
                t = self.balancer.desired_pos
                tx_abs = int(t[0] + self.frame_center[0])
                ty_abs = int(t[1] + self.frame_center[1])
                
                # Draw target crosshair
                cv2.drawMarker(frame, (tx_abs, ty_abs), (255, 0, 255),
                              cv2.MARKER_CROSS, 20, 2)
                
                cv2.putText(frame, f"Target: ({t[0]:.1f}, {t[1]:.1f})",
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           (255, 200, 0), 2)
                
                # Show current error
                if self.ball_pos:
                    error = np.linalg.norm(t - np.array([self.ball_pos[0], self.ball_pos[1]]))
                    cv2.putText(frame, f"Error: {error:.1f} px",
                               (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                               (0, 255, 0) if error < 20 else (0, 165, 255), 2)
                
                # Show PID values
                cv2.putText(frame, f"Kp={self.balancer.pid.Kp:.6f}",
                           (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (200, 200, 200), 1)
                cv2.putText(frame, f"Ki={self.balancer.pid.Ki:.6f}",
                           (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (200, 200, 200), 1)
                cv2.putText(frame, f"Kd={self.balancer.pid.Kd:.6f}",
                           (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (200, 200, 200), 1)
            
            # Show optimization status
            if self.optimization_mode:
                cv2.putText(frame, "OPTIMIZATION MODE",
                           (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           (0, 255, 0), 2)
            
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

                            self.tracking_active = True
                            collecting_points = False
                            
                            # Start performance tracking
                            self.reset_performance_tracking()

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

                # Update performance tracking
                if ball:
                    self.update_performance_tracking(ball, self.balancer.desired_pos)

                now = time.time()
                if now - last >= period:
                    dt = now - last
                    last = now

                    output = self.balancer.step(ball_np, dt)
                    self.send_serial(output)

                display = self.draw_overlay(frame.copy())
                cv2.imshow("Ball Tracker", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("g"):
                    self.run_grid_search_optimization()
                elif key == ord("r"):
                    self.run_random_search_optimization()
                elif key == ord("a"):
                    self.run_adaptive_search_optimization()
                elif key == ord("s"):
                    self.optimizer.save_results()
                elif key == ord("p"):
                    self.optimizer.plot_results()
                elif key == ord("e"):
                    self.evaluate_current_performance()
                elif key == ord("o"):
                    self.optimization_mode = not self.optimization_mode
                    print(f"[INFO] Optimization mode: {self.optimization_mode}")

            except:
                traceback.print_exc()

        self.cap.release()
        if self.ser:
            self.ser.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = BallTrackerWithOptimizer("config.json", "COM7", 115200)
    tracker.run()
