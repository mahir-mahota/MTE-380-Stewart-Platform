# Simple Auto Calibration System for Ball Tracking
# Interactive calibration tool for color detection
# Generates config.json file for use with ball tracking

import cv2
import numpy as np
import json
import time
from collections import deque
from datetime import datetime

# Set matplotlib backend before importing pyplot
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARNING] matplotlib not installed. Graph will not be available.")
    print("Install with: pip install matplotlib")

class SimpleAutoCalibrator:
    """Interactive calibration system for ball tracking."""
    
    def __init__(self):
        """Initialize calibration parameters and default values."""
        # Camera configuration
        self.CAM_INDEX = 0  # Default camera index
        self.FRAME_W, self.FRAME_H = 640, 480  # Frame dimensions
        
        # Calibration state tracking
        self.current_frame = None  # Current video frame
        self.phase = "color"  # Current phase: "color", "complete"
        
        # Color calibration data
        self.hsv_samples = []  # Collected HSV color samples
        self.lower_hsv = None  # Lower HSV bound for ball detection
        self.upper_hsv = None  # Upper HSV bound for ball detection
        
        # Motion tracking data
        self.prev_pos = None  # Previous ball position (x, y)
        self.smoothed_pos = None  # Smoothed position to reduce noise
        self.prev_velocity = None  # Previous velocity (vx, vy) in pixels per frame
        self.current_velocity = None  # Current velocity (vx, vy) in pixels per frame
        self.smoothed_velocity = None  # Smoothed velocity
        self.acceleration = None  # Current acceleration (ax, ay) in pixels per frame^2
        self.smoothed_accel = None  # Smoothed acceleration for display
        self.accel_moving_avg = deque(maxlen=5)  # Moving average buffer for extra smoothing
        self.frame_count = 0  # Frame counter for frame-based calculations
        self.fps = 30.0  # Assumed frame rate for conversion to px/s
        
        # Graph data for acceleration magnitude history
        self.accel_history = deque(maxlen=150)  # Store last 5 seconds at 30fps
        self.time_history = deque(maxlen=150)
        self.graph_update_counter = 0
        self.graph_update_interval = 1  # Update graph every frame for smoother display

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse click events for interactive calibration.
        
        Args:
            event: OpenCV mouse event type
            x, y: Mouse click coordinates
            flags: Additional event flags
            param: User data (unused)
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.phase == "color":
                # Color sampling phase - collect HSV samples at click point
                self.sample_color(x, y)

    def sample_color(self, x, y):
        """Sample HSV color values in a 5x5 region around click point.
        
        Args:
            x, y: Center coordinates for color sampling
        """
        if self.current_frame is None:
            return
        
        # Convert frame to HSV color space
        hsv = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)
        
        # Sample 5x5 region around click point
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                px, py = x + dx, y + dy
                # Check bounds and collect valid samples
                if 0 <= px < hsv.shape[1] and 0 <= py < hsv.shape[0]:
                    self.hsv_samples.append(hsv[py, px])
        
        # Update HSV bounds based on collected samples
        if self.hsv_samples:
            samples = np.array(self.hsv_samples)
            
            # Calculate adaptive margins for each HSV channel
            h_margin = max(5, (np.max(samples[:, 0]) - np.min(samples[:, 0])) * 0.1)
            s_margin = max(10, (np.max(samples[:, 1]) - np.min(samples[:, 1])) * 0.15)
            v_margin = max(10, (np.max(samples[:, 2]) - np.min(samples[:, 2])) * 0.15)
            
            # Set lower bounds with margin
            self.lower_hsv = [
                max(0, np.min(samples[:, 0]) - h_margin),
                max(0, np.min(samples[:, 1]) - s_margin),
                max(0, np.min(samples[:, 2]) - v_margin)
            ]
            
            # Set upper bounds with margin
            self.upper_hsv = [
                min(179, np.max(samples[:, 0]) + h_margin),
                min(255, np.max(samples[:, 1]) + s_margin),
                min(255, np.max(samples[:, 2]) + v_margin)
            ]
            
            print(f"[COLOR] Samples: {len(self.hsv_samples)}")

    def detect_ball(self, frame):
        """Detect ball in frame and return pixel coordinates and radius.
        
        Args:
            frame: Input BGR image frame
            
        Returns:
            tuple or None: (x, y, radius) in pixels, None if not detected
        """
        if not self.lower_hsv:
            return None
        
        # Convert to HSV and create color mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array(self.lower_hsv, dtype=np.uint8)
        upper = np.array(self.upper_hsv, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        
        # Clean up mask with morphological operations
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # Find contours in mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # Get largest contour (assumed to be ball)
        largest = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest)
        
        # Filter out very small detections
        if radius < 5:
            return None
        
        return (int(x), int(y), int(radius))

    def update_motion(self, current_pos):
        """Update velocity and acceleration based on current ball position.
        Uses frame-based calculations with position smoothing to reduce noise.
        
        Args:
            current_pos: Current ball position (x, y) or None if not detected
        """
        if current_pos is None:
            # Reset tracking if ball is lost
            self.prev_pos = None
            self.smoothed_pos = None
            self.prev_velocity = None
            self.current_velocity = None
            self.smoothed_velocity = None
            self.acceleration = None
            self.smoothed_accel = None
            self.accel_moving_avg.clear()
            self.frame_count = 0
            return
        
        x, y = current_pos[0], current_pos[1]
        
        # Smooth position using exponential moving average with aggressive smoothing
        # Current method: Exponential Moving Average (EMA)
        # Formula: smoothed = alpha * new + (1-alpha) * old
        # Lower alpha = more smoothing (more weight on old values)
        if self.smoothed_pos is None:
            self.smoothed_pos = (float(x), float(y))
        else:
            alpha_pos = 0.3  # Very low alpha = much more smoothing (was 0.5)
            self.smoothed_pos = (
                alpha_pos * x + (1 - alpha_pos) * self.smoothed_pos[0],
                alpha_pos * y + (1 - alpha_pos) * self.smoothed_pos[1]
            )
        
        # Use frame-based calculations (pixels per frame) to avoid division by tiny time deltas
        if self.prev_pos is not None:
            # Calculate velocity in pixels per frame
            vx = self.smoothed_pos[0] - self.prev_pos[0]
            vy = self.smoothed_pos[1] - self.prev_pos[1]
            
            self.current_velocity = (vx, vy)
            
            # Smooth velocity with exponential moving average
            if self.smoothed_velocity is None:
                self.smoothed_velocity = (vx, vy)
            else:
                alpha_vel = 0.3  # Smooth velocity too
                self.smoothed_velocity = (
                    alpha_vel * vx + (1 - alpha_vel) * self.smoothed_velocity[0],
                    alpha_vel * vy + (1 - alpha_vel) * self.smoothed_velocity[1]
                )
            
            # Calculate acceleration if we have previous velocity
            if self.prev_velocity is not None:
                # Use smoothed velocity for acceleration calculation
                ax = self.smoothed_velocity[0] - self.prev_velocity[0]
                ay = self.smoothed_velocity[1] - self.prev_velocity[1]
                self.acceleration = (ax, ay)
                
                # MULTI-LAYER SMOOTHING for acceleration:
                # Layer 1: Exponential Moving Average (EMA)
                if self.smoothed_accel is None:
                    self.smoothed_accel = (ax, ay)
                else:
                    alpha_accel = 0.2  # Very aggressive smoothing (was 0.4)
                    self.smoothed_accel = (
                        alpha_accel * ax + (1 - alpha_accel) * self.smoothed_accel[0],
                        alpha_accel * ay + (1 - alpha_accel) * self.smoothed_accel[1]
                    )
                
                # Layer 2: Moving Average on top of EMA (additional smoothing)
                self.accel_moving_avg.append(self.smoothed_accel)
                if len(self.accel_moving_avg) >= 3:  # Need at least 3 samples
                    # Calculate average of last N samples
                    avg_ax = np.mean([a[0] for a in self.accel_moving_avg])
                    avg_ay = np.mean([a[1] for a in self.accel_moving_avg])
                    # Use weighted combination: 70% moving avg, 30% current EMA
                    final_ax = 0.7 * avg_ax + 0.3 * self.smoothed_accel[0]
                    final_ay = 0.7 * avg_ay + 0.3 * self.smoothed_accel[1]
                    self.smoothed_accel = (final_ax, final_ay)
                
                # Store acceleration magnitude in history
                accel_mag = np.sqrt(self.smoothed_accel[0]**2 + self.smoothed_accel[1]**2)
                accel_mag_ps2 = accel_mag * (self.fps ** 2)  # Convert to px/s^2
                current_time = time.time()
                
                # Remove old data points (older than 5 seconds)
                while self.time_history and (current_time - self.time_history[0]) > 5.0:
                    self.time_history.popleft()
                    self.accel_history.popleft()
                
                # Add new data point
                self.time_history.append(current_time)
                self.accel_history.append(accel_mag_ps2)
            else:
                self.smoothed_accel = None
            
            # Update previous velocity (use smoothed velocity)
            self.prev_velocity = self.smoothed_velocity
        else:
            # First detection, no velocity/acceleration yet
            self.current_velocity = None
            self.acceleration = None
            self.smoothed_accel = None
        
        # Update previous position
        self.prev_pos = self.smoothed_pos
        self.frame_count += 1

    def save_config(self):
        """Save all calibration results to config.json file."""
        config = {
            "timestamp": datetime.now().isoformat(),
            "camera": {
                "index": int(self.CAM_INDEX),
                "frame_width": int(self.FRAME_W),
                "frame_height": int(self.FRAME_H)
            },
            "ball_detection": {
                "lower_hsv": [float(x) for x in self.lower_hsv] if self.lower_hsv else None,
                "upper_hsv": [float(x) for x in self.upper_hsv] if self.upper_hsv else None
            }
        }
        
        # Write configuration to JSON file
        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)
        print("[SAVE] Configuration saved to config.json")

    def draw_overlay(self, frame):
        """Draw calibration status and instructions overlay on frame.
        
        Args:
            frame: Input BGR image frame
            
        Returns:
            numpy.ndarray: Frame with overlay graphics and text
        """
        overlay = frame.copy()
        
        # Phase-specific instruction text
        phase_text = {
            "color": "Click on ball to sample colors. Press 'c' when done.",
            "complete": "Calibration complete! Press 's' to save"
        }
        
        # Draw current phase and instructions
        cv2.putText(overlay, f"Phase: {self.phase}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay, phase_text[self.phase], (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show color calibration progress
        if self.hsv_samples:
            cv2.putText(overlay, f"Color samples: {len(self.hsv_samples)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Show real-time ball detection if color calibration is complete
        if self.lower_hsv:
            ball = self.detect_ball(frame)
            if ball:
                x, y, radius = ball
                # Update motion tracking
                self.update_motion((x, y))
                
                    # Draw detection circle
                cv2.circle(overlay, (x, y), radius, (0, 255, 255), 2)
                cv2.circle(overlay, (x, y), 3, (0, 255, 255), -1)
                
                # Show pixel coordinates
                print(f"Ball detected at position: ({x}, {y})")
                cv2.putText(overlay, f"Pos: ({x}, {y})",
                           (x + 20, y - 40),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
                # Show velocity if available
                if self.current_velocity:
                    vx, vy = self.current_velocity
                    speed = np.sqrt(vx**2 + vy**2)
                    # cv2.putText(overlay, f"Vel: ({vx:.1f}, {vy:.1f}) px/s",
                    #            (x + 20, y - 20),
                    #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    # cv2.putText(overlay, f"Speed: {speed:.1f} px/s",
                    #            (x + 20, y),
                    #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Show acceleration if available (using smoothed values)
                if self.smoothed_accel:
                    ax, ay = self.smoothed_accel
                    # Convert from pixels per frame^2 to pixels per second^2
                    ax_ps2 = ax * (self.fps ** 2)
                    ay_ps2 = ay * (self.fps ** 2)
                    accel_magnitude = np.sqrt(ax_ps2**2 + ay_ps2**2)
                    cv2.putText(overlay, f"Accel: ({ax_ps2:.1f}, {ay_ps2:.1f}) px/s^2",
                               (x + 20, y + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                    cv2.putText(overlay, f"|a|: {accel_magnitude:.1f} px/s^2",
                               (x + 20, y + 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                    print(f"Acceleration: ({ax_ps2:.2f}, {ay_ps2:.2f}) px/s^2, Magnitude: {accel_magnitude:.2f} px/s^2")
            else:
                # Ball not detected, reset motion tracking
                self.update_motion(None)
        
        return overlay

    def init_graph(self):
        """Initialize the matplotlib graph for acceleration magnitude."""
        if not MATPLOTLIB_AVAILABLE:
            print("[GRAPH] matplotlib not available, cannot show graph")
            return
        
        try:
            plt.ion()  # Turn on interactive mode
            self.fig, self.ax = plt.subplots(figsize=(10, 5))
            self.ax.set_xlabel('Time (seconds ago)', fontsize=10)
            self.ax.set_ylabel('Acceleration Magnitude (px/s²)', fontsize=10)
            self.ax.set_title('Acceleration Magnitude (Last 5 seconds)', fontsize=12, fontweight='bold')
            self.ax.grid(True, alpha=0.3)
            
            # Initialize with empty data
            self.line, = self.ax.plot([], [], 'b-', linewidth=2, label='Acceleration')
            self.ax.legend()
            self.ax.set_xlim(5, 0)  # Show last 5 seconds (reversed so 0 is most recent)
            self.ax.set_ylim(0, 1000)  # Initial y-axis limit, will auto-adjust
            
            # Add text to show it's waiting for data
            self.waiting_text = self.ax.text(2.5, 500, 'Waiting for acceleration data...', 
                                            ha='center', va='center', fontsize=12, 
                                            style='italic', color='gray')
            
            plt.tight_layout()
            
            # Make sure window is visible
            self.fig.canvas.manager.set_window_title('Acceleration Graph')
            
            # Try to bring window to front (platform dependent)
            try:
                mngr = self.fig.canvas.manager
                if hasattr(mngr, 'window'):
                    mngr.window.wm_attributes('-topmost', 1)
                    mngr.window.wm_attributes('-topmost', 0)
            except:
                pass
            
            self.fig.show()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            print("[GRAPH] Acceleration graph window opened!")
            print("[GRAPH] Look for a window titled 'Acceleration Graph'")
        except Exception as e:
            print(f"[GRAPH] Error initializing graph: {e}")
            import traceback
            traceback.print_exc()

    def update_graph(self):
        """Update the acceleration magnitude graph with current data."""
        if not MATPLOTLIB_AVAILABLE:
            return
        if not hasattr(self, 'fig') or not hasattr(self, 'ax'):
            return
        
        if len(self.time_history) < 2:
            return
        
        try:
            current_time = time.time()
            
            # Calculate relative times (seconds ago)
            relative_times = [current_time - t for t in self.time_history]
            
            # Remove waiting text if it exists
            if hasattr(self, 'waiting_text') and self.waiting_text:
                self.waiting_text.remove()
                self.waiting_text = None
            
            # Update the plot
            self.line.set_data(relative_times, list(self.accel_history))
            
            # Auto-adjust y-axis based on data range
            if self.accel_history:
                max_accel = max(self.accel_history)
                min_accel = min(self.accel_history)
                margin = (max_accel - min_accel) * 0.1 if max_accel > min_accel else 50
                self.ax.set_ylim(max(0, min_accel - margin), max_accel + margin)
            
            # Auto-adjust x-axis to show last 5 seconds
            if relative_times:
                max_time = max(relative_times)
                self.ax.set_xlim(max(5, max_time + 0.5), -0.5)
            
            # Redraw
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception as e:
            print(f"[GRAPH] Error updating graph: {e}")

    def run(self):
        """Main calibration loop with interactive GUI."""
        # Initialize camera capture
        self.cap = cv2.VideoCapture(self.CAM_INDEX, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_H)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
        
        # Setup OpenCV window and mouse callback
        cv2.namedWindow("Auto Calibration")
        cv2.setMouseCallback("Auto Calibration", self.mouse_callback)
        
        # Initialize graph immediately (before calibration)
        if MATPLOTLIB_AVAILABLE:
            self.init_graph()
            graph_initialized = True
        else:
            graph_initialized = False
        
        # Display instructions
        print("[INFO] Simple Auto Calibration")
        print("Click on ball to sample colors, press 'c' when done")
        print("Press 's' to save, 'q' to quit")
        print("\n[SMOOTHING INFO]")
        print("Current smoothing method: Multi-layer smoothing")
        print("  1. Position: Exponential Moving Average (alpha=0.3)")
        print("  2. Velocity: Exponential Moving Average (alpha=0.3)")
        print("  3. Acceleration: EMA (alpha=0.2) + Moving Average (5 samples)")
        print("  Lower alpha = more smoothing (slower response to changes)")
        
        # Main calibration loop
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            self.current_frame = frame
            
            # Draw overlay and display frame
            display = self.draw_overlay(frame)
            cv2.imshow("Auto Calibration", display)
            
            # Initialize graph when calibration is complete
            if self.phase == "complete" and not graph_initialized:
                self.init_graph()
                graph_initialized = True
            
            # Update graph periodically (every few frames for performance)
            # if graph_initialized and self.phase == "complete":
            #     self.graph_update_counter += 1
            #     if self.graph_update_counter >= self.graph_update_interval:
            #         self.update_graph()
            #         self.graph_update_counter = 0
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Quit calibration
                break
            elif key == ord('c') and self.phase == "color":
                # Complete color calibration phase
                if self.hsv_samples:
                    self.phase = "complete"
                print("[INFO] Color calibration complete. Press 's' to save.")
            elif key == ord('s') and self.phase == "complete":
                # Save configuration and exit
                self.save_config()
                break
        
        # Clean up resources
        self.cap.release()
        cv2.destroyAllWindows()
        if hasattr(self, 'fig'):
            plt.close(self.fig)

if __name__ == "__main__":
    """Run calibration when script is executed directly."""
    calibrator = SimpleAutoCalibrator()
    calibrator.run()