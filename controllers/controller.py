import carla


class FixedSpeedPIDController:
    def __init__(
        self,
        fixed_throttle: float = 0.04,
        kp: float = 0.005,
        ki: float = 0.0000,
        kd: float = 0.01,
        steer_limit: float = 0.85,
        max_steer_step: float = 0.08,
        integral_limit: float = 3000.0,
    ):
        self.fixed_throttle = fixed_throttle
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.steer_limit = steer_limit
        self.max_steer_step = max_steer_step
        self.integral_limit = integral_limit

        self.prev_error = 0.0
        self.integral = 0.0
        self.last_steer = 0.0

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_steer = 0.0

    def update(self, error: float, brake: float = 0.0, reverse: bool = False) -> carla.VehicleControl:
        self.integral += error
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))

        derivative = error - self.prev_error
        self.prev_error = error

        steer = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        steer = max(-self.steer_limit, min(self.steer_limit, steer))

        delta = steer - self.last_steer
        if delta > self.max_steer_step:
            steer = self.last_steer + self.max_steer_step
        elif delta < -self.max_steer_step:
            steer = self.last_steer - self.max_steer_step

        self.last_steer = steer

        return carla.VehicleControl(
            throttle=self.fixed_throttle,
            steer=steer,
            brake=brake,
            reverse=reverse,
            hand_brake=False,
        )