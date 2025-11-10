import numpy as np
from PID import PID

class PositionBalancer:
    Kp = 1.0
    Ki = 0.0
    Kd = 0.1

    def __init__(self, platform_points):
        self.platform_points = platform_points
        self.platform = Balancer(platform_points)
        self.position_pids = [PID(self.Kp, self.Ki, self.Kd) for _ in range(2)]
        
        self.desired_pos = np.array([0.0, 0.0, 0.0])

    def step(self, meas_pos, dt):
        if meas_pos is not None:
            meas_pos_np = np.array([meas_pos[0], meas_pos[1], 0.0])
            pos_error = self.desired_pos - meas_pos_np

            desired_accel = []
            for i in range(2):
                accel_cmd = self.position_pids[i].compute(pos_error[i], dt)
                desired_accel.append(accel_cmd)
            
            self.platform.desired_accel = desired_accel + [0.0]
        return self.platform.step(meas_pos, dt) 


class Balancer:
    Kp = 1.0
    Ki = 0.0
    Kd = 0.1

    def __init__(self, platform_points):
        self.platform_points = platform_points
        self.platform_pids = [PID(self.Kp, self.Ki, self.Kd) for _ in platform_points]

        self.prev_pos = None
        self.pos = None

        self.prev_vel = None
        self.vel = None

        self.accel = None

        self.desired_accel = [0, 0, 0]

    def update(self, meas_pos, dt):
        if self.pos is not None:
            self.prev_pos = self.pos    
            self.prev_vel = self.vel
        
        self.pos = meas_pos

        if self.prev_pos is not None and self.pos is not None:
            self.vel = (self.pos - self.prev_pos)/dt
        
        if self.prev_vel is not None and self.pos is not None:
            self.accel = (self.vel - self.prev_vel)/dt
            
    def step(self, meas_pos, dt):
        self.update(meas_pos, dt)
        delta_list = self._project_control()

        for i, pid in enumerate(self.platform_pids):
            delta_list[i] = pid.compute(delta_list[i], dt)
        
        return delta_list

    def _project_control(self):
        accel_des = np.array(self.desired_accel)
        if self.accel is None: return [0 for _ in self.platform_points]
        accel = np.array(self.accel)

        accel_diff = accel_des - accel
        
        accel_mags = []
        for point in self.platform_points:
            dir = -np.array(point)
            dir = dir / np.linalg.norm(dir)

            mag = np.dot(dir, accel_diff)
            accel_mags.append(mag)

        return accel_mags
