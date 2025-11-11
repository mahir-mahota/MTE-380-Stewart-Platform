import numpy as np

class PID:
    def __init__(self, Kp, Ki, Kd, integral_limit=None):
        """
        PID controller with optional integral windup protection.

        Args:
            Kp, Ki, Kd: PID gains.
            integral_limit: scalar or array specifying max magnitude for integral term.
                            e.g., 100 or np.array([100, 100])
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_limit = integral_limit

        self.prev_error = None
        self.integral = None

    def compute(self, error, dt):
        err = np.array(error, dtype=float)

        # Lazy init
        if self.prev_error is None:
            self.prev_error = np.zeros_like(err)
        if self.integral is None:
            self.integral = np.zeros_like(err)

        # Update integral
        self.integral += err * dt

        # --- Anti-windup clamp ---
        if self.integral_limit is not None:
            limit = np.array(self.integral_limit, dtype=float)
            self.integral = np.clip(self.integral, -limit, limit)

        # Derivative
        derivative = (err - self.prev_error) / dt if dt > 0 else np.zeros_like(err)

        # Output
        output = (self.Kp * err) + (self.Ki * self.integral) + (self.Kd * derivative)
        self.prev_error = err

        return float(output) if output.size == 1 else output

    def reset(self):
        """Reset the integral and derivative history."""
        self.prev_error = None
        self.integral = None
