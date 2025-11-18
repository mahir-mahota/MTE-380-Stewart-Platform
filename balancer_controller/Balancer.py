import numpy as np
from PID import PID


class PositionBalancer:
    def __init__(self, platform_points):
        self.platform_points = np.array(platform_points)
        self.balancer = Balancer(self.platform_points)

        # 2D PID (x,y) → tilt commands
        self.pid = PID(Kp=0.00016, Ki=0.00001, Kd=0.00013, integral_limit=10000)

        self.desired_pos = np.array([0.0, 0.0])
        self.prev_heights = np.zeros(len(platform_points))
        self.filtered_tilt = np.array([0.0, 0.0])


    def step(self, meas_pos, dt):
        if meas_pos is None or np.any(np.isnan(meas_pos)):
            return [0]*len(self.platform_points)

        ball = np.array(meas_pos[:2])
        error = self.desired_pos - ball
        if np.linalg.norm(error) < 20: error -= error

        raw_tilt = np.array(self.pid.compute(error, dt))

        # --- IMPORTANT AXIS SWAP ---
        # PID(x,y) → tilt(x,y)
        # tilt_x controls Y
        # tilt_y controls X
        tilt_x = raw_tilt[1]
        tilt_y = -raw_tilt[0]
        raw_tilt = np.array([tilt_x, tilt_y])

        # smooth the tilt


        # convert to servo heights
        servo_heights = np.array(self.balancer.compute_servo_heights(raw_tilt))
        print(f"NEED: {servo_heights}")
        servo_heights += 15
        servo_heights = np.clip(servo_heights, 0, 30)
        print(self.prev_heights)
        servo_heights = np.round(servo_heights)
        # output deltas
        delta = servo_heights-self.prev_heights
        # delta = np.clip(delta, -1, 1)
        print(delta)
        self.prev_heights += delta

        return delta.tolist()



class Balancer:
    def __init__(self, platform_points):
        self.platform_points = np.array(platform_points)
        self.A = self._build_A_matrix(platform_points)

    def _build_A_matrix(self, pts):
        A = []
        for (x, y, z) in pts:
            # do NOT flip these
            A.append([-y, x])
        return np.array(A)

    def compute_servo_heights(self, tilt_xy):
        return (self.A @ tilt_xy).tolist()
