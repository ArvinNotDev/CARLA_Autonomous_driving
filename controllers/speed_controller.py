from __future__ import annotations

import math
import time


class SpeedPIDController:
    """
    Longitudinal PID controller.

    The requested speed is clamped to the configured moving-speed interval.
    A requested speed of exactly zero is an explicit stop and always bypasses
    the minimum-speed floor. Throttle is generated here for every autonomous
    movement mode; steering controllers and maneuver managers only own steer.
    """

    KP = 0.90
    KI = 0.15
    KD = 0.08
    MAX_DECELERATION_MPS2 = 3.5
    MIN_FORWARD_THROTTLE = 0.14
    LAUNCH_THROTTLE = 0.36
    LAUNCH_SPEED_KMH = 2.0
    MAX_THROTTLE = 0.78
    THROTTLE_PER_MPS2 = 0.22
    STOP_BRAKE = 1.0
    OVERSPEED_DEADBAND_MPS = 0.20

    def __init__(
        self,
        *,
        acceleration_mps2: float = 2.5,
        min_speed_kmh: float = 8.0,
        max_speed_kmh: float = 25.0,
    ) -> None:
        self.acceleration_mps2 = 2.5
        self.min_speed_kmh = 8.0
        self.max_speed_kmh = 25.0
        self._integral = 0.0
        self._previous_error_mps = 0.0
        self._previous_speed_mps: float | None = None
        self._last_update_at: float | None = None
        self.configure(
            acceleration_mps2=acceleration_mps2,
            min_speed_kmh=min_speed_kmh,
            max_speed_kmh=max_speed_kmh,
        )

    def configure(
        self,
        *,
        acceleration_mps2: float | None = None,
        min_speed_kmh: float | None = None,
        max_speed_kmh: float | None = None,
    ) -> None:
        acceleration = (
            self.acceleration_mps2
            if acceleration_mps2 is None
            else float(acceleration_mps2)
        )
        minimum = self.min_speed_kmh if min_speed_kmh is None else float(min_speed_kmh)
        maximum = self.max_speed_kmh if max_speed_kmh is None else float(max_speed_kmh)

        if not all(math.isfinite(value) for value in (acceleration, minimum, maximum)):
            raise ValueError("Speed-controller settings must be finite.")
        if acceleration <= 0.0:
            raise ValueError("Acceleration must be greater than zero.")
        if minimum <= 0.0:
            raise ValueError("Minimum moving speed must be greater than zero.")
        if maximum <= minimum:
            raise ValueError("Maximum speed must be greater than minimum speed.")

        self.acceleration_mps2 = min(8.0, max(0.2, acceleration))
        self.min_speed_kmh = min(60.0, max(1.0, minimum))
        self.max_speed_kmh = min(120.0, max(self.min_speed_kmh + 0.5, maximum))

    def reset(self) -> None:
        self._integral = 0.0
        self._previous_error_mps = 0.0
        self._previous_speed_mps = None
        self._last_update_at = None

    def normalize_target_speed(self, target_speed_kmh: float) -> float:
        target = float(target_speed_kmh)
        if not math.isfinite(target):
            raise ValueError("Target speed must be finite.")
        # Zero is an intentional stop command. Every positive request is a
        # moving request and therefore respects the configured minimum speed.
        if target <= 0.0:
            return 0.0
        return min(self.max_speed_kmh, max(self.min_speed_kmh, target))

    def update(
        self,
        current_speed_kmh: float,
        target_speed_kmh: float,
        *,
        dt: float | None = None,
    ) -> tuple[float, float]:
        """Return (throttle, brake) while respecting the active speed target."""
        current = max(0.0, float(current_speed_kmh))
        if not math.isfinite(current):
            current = 0.0
        target = self.normalize_target_speed(target_speed_kmh)

        now = time.perf_counter()
        if dt is None:
            dt = (
                0.025
                if self._last_update_at is None
                else now - self._last_update_at
            )
        self._last_update_at = now
        dt = min(0.1, max(0.005, float(dt)))

        if target <= 0.0:
            self.reset()
            return 0.0, self.STOP_BRAKE

        current_mps = current / 3.6
        target_mps = target / 3.6
        error_mps = target_mps - current_mps

        measured_acceleration = 0.0
        if self._previous_speed_mps is not None:
            measured_acceleration = (
                current_mps - self._previous_speed_mps
            ) / dt
        self._previous_speed_mps = current_mps

        if error_mps < -self.OVERSPEED_DEADBAND_MPS:
            self._integral = 0.0
            overspeed = -error_mps - self.OVERSPEED_DEADBAND_MPS
            brake = min(0.55, max(0.08, overspeed * 0.30))
            self._previous_error_mps = error_mps
            return 0.0, brake

        candidate_integral = min(
            4.0,
            max(-1.0, self._integral + error_mps * dt),
        )
        acceleration_command = (
            self.KP * error_mps
            + self.KI * candidate_integral
            - self.KD * measured_acceleration
        )
        acceleration_command = min(
            self.acceleration_mps2,
            max(0.0, acceleration_command),
        )

        # Do not wind the integral up while the requested acceleration is
        # saturated. This keeps recovery predictable after a speed-limit change.
        if acceleration_command < self.acceleration_mps2 - 1e-6:
            self._integral = candidate_integral

        throttle = (
            self.MIN_FORWARD_THROTTLE
            + acceleration_command * self.THROTTLE_PER_MPS2
        )
        throttle = min(self.MAX_THROTTLE, max(self.MIN_FORWARD_THROTTLE, throttle))

        # CARLA vehicles may need meaningful torque to overcome standstill
        # resistance. This launch floor is active only while the vehicle is
        # nearly stationary; it does not force a minimum throttle at cruising speed.
        if current < self.LAUNCH_SPEED_KMH and current + 0.15 < target:
            throttle = max(throttle, self.LAUNCH_THROTTLE)

        self._previous_error_mps = error_mps
        return float(throttle), 0.0
