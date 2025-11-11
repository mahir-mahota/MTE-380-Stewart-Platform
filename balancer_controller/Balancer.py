import numpy as np
from PID import PID

class PositionBalancer:
    def __init__(self, platform_points):
        self.platform_points = np.array(platform_points)  # Nx3
        self.balancer = Balancer(platform_points)

        # Single PID for both axes
        self.pid = PID(Kp=0.1, Ki=0.0, Kd=0.0)

        self.desired_pos = np.array([0.0, 0.0])  # target ball position
        self.prev_heights = np.zeros(len(platform_points))  # last servo heights

    def step(self, meas_pos, dt):
        if meas_pos is None or np.any(np.isnan(meas_pos)):
            return np.zeros(len(self.platform_points)).tolist()

        meas_pos_np = np.array(meas_pos[:2])
        pos_error = self.desired_pos - meas_pos_np
        print("Position error:", pos_error)

        # --- PID outputs a 2D tilt vector directly ---
        desired_tilt = self.pid.compute(pos_error, dt)
        desired_tilt = list(desired_tilt)
        tmp = desired_tilt[1]
        desired_tilt[1] = -desired_tilt[0]
        desired_tilt[0] = tmp
        desired_tilt = np.array(desired_tilt)
        print("Desired tilt:", desired_tilt)

        # --- Compute absolute servo heights for that tilt ---
        servo_heights = np.array(self.balancer.compute_servo_heights(desired_tilt))

        # --- Convert to delta heights relative to previous ---
        delta_h = servo_heights - self.prev_heights

        self.prev_heights = servo_heights

        return delta_h.tolist()


class Balancer:
    def __init__(self, platform_points):
        self.platform_points = np.array(platform_points)
        self.A = self._build_A_matrix(platform_points)

    def _build_A_matrix(self, platform_points):
        A = []
        for (x, y, z) in platform_points:
            A.append([-y, x])
        return np.array(A)

    def compute_servo_heights(self, desired_tilt):
        h = self.A @ desired_tilt
        return h.tolist()
