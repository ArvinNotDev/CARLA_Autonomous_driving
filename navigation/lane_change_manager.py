from __future__ import annotations

import threading
import time
from typing import Optional, Dict

import config_city as conf


class LaneChangeManager:
    """
    Nonblocking lane-change state machine.

    Planner queries may run asynchronously, but vehicle control always stays in
    the main control thread. No sleep() or apply_control() occurs here.
    """

    def __init__(
        self,
        vehicle,
        planner,
        lane_change_debounce_seconds: float = 1.5,
    ) -> None:
        self.vehicle = vehicle
        self.planner = planner

        self.lane_change_debounce_seconds = float(lane_change_debounce_seconds)
        self.lane_change_debounce_until = 0.0
        self.lane_change_attempts = 0
        self.line_angles: Dict[str, Optional[float]] = {"left": None, "right": None}
        self.angle_cooldown_until = 0.0

        self.phase = "IDLE"  # IDLE / VERIFY / CHANGE
        self.direction: Optional[str] = None
        self._segment_start_location = None
        self._state_lock = threading.Lock()

        self._planner_query_active = False
        self._last_planner_query_at = 0.0
        self._request_id = 0
        self._latest_planner_result: Optional[tuple[int, str]] = None

    def _movement_active(self) -> bool:
        with self._state_lock:
            return self.phase in {"VERIFY", "CHANGE"}

    def movement_active(self) -> bool:
        return self._movement_active()

    def _angle_guard_triggered(self) -> bool:
        threshold = float(getattr(conf, "LANE_CHANGE_LINE_ANGLE_THRESHOLD_DEG", 20.0))
        return any(
            angle is not None and abs(float(angle)) > threshold
            for angle in self.line_angles.values()
        )

    def _apply_planner_result(self, current_location) -> None:
        with self._state_lock:
            result = self._latest_planner_result
            if result is None or self.phase != "IDLE":
                return
            self._latest_planner_result = None
            _, direction = result
            self.direction = direction
            self.phase = "VERIFY"
            self._segment_start_location = current_location

    def update(
        self,
        current_location,
        goal_location,
        line_angles: Optional[Dict[str, Optional[float]]] = None,
    ) -> None:
        if self.vehicle is None:
            return

        if line_angles is not None:
            self.line_angles = {
                "left": line_angles.get("left"),
                "right": line_angles.get("right"),
            }

        self._apply_planner_result(current_location)

        with self._state_lock:
            phase = self.phase
            direction = self.direction
            segment_start = self._segment_start_location

        now = time.monotonic()
        if phase == "VERIFY" and segment_start is not None:
            try:
                traveled = float(current_location.distance(segment_start))
            except Exception:
                traveled = 0.0

            if traveled >= 1.0:
                with self._state_lock:
                    self.phase = "CHANGE"
                    self._segment_start_location = current_location
            return

        if phase == "CHANGE" and segment_start is not None:
            try:
                traveled = float(current_location.distance(segment_start))
            except Exception:
                traveled = 0.0
            if traveled >= 2.0:
                self.lane_change_attempts += 1
                if (
                    self.lane_change_attempts
                    % max(1, int(getattr(conf, "LANE_CHANGE_ANGLE_CHECK_EVERY", 2)))
                    == 0
                    and self._angle_guard_triggered()
                ):
                    self.angle_cooldown_until = now + float(
                        getattr(conf, "LANE_CHANGE_ANGLE_COOLDOWN_SECONDS", 5.0)
                    )
                self._finish()
            return

        if phase != "IDLE":
            return

        if now < self.angle_cooldown_until or time.time() < self.lane_change_debounce_until:
            return

        query_interval = float(
            getattr(conf, "LANE_CHANGE_PLANNER_CHECK_INTERVAL_SECONDS", 0.50)
        )
        with self._state_lock:
            if self._planner_query_active or (
                now - self._last_planner_query_at < query_interval
            ):
                return
            self._planner_query_active = True
            self._last_planner_query_at = now
            self._request_id += 1
            request_id = self._request_id

        def query_worker() -> None:
            direction = ""
            try:
                text = self.planner.get_next_maneuver_text(current_location, goal_location)
                if text == "CHANGE_LANE_LEFT":
                    direction = "left"
                elif text == "CHANGE_LANE_RIGHT":
                    direction = "right"
                if direction:
                    with self._state_lock:
                        if request_id == self._request_id and self.phase == "IDLE":
                            self._latest_planner_result = (request_id, direction)
                            self.lane_change_debounce_until = (
                                time.time() + self.lane_change_debounce_seconds
                            )
            except Exception:
                pass
            finally:
                with self._state_lock:
                    self._planner_query_active = False

        threading.Thread(
            target=query_worker,
            name="carla-lane-planner-query",
            daemon=True,
        ).start()

    def control_override(self):
        """Return the special lane-change control, or None."""
        import carla

        with self._state_lock:
            phase = self.phase
            direction = self.direction

        if phase == "VERIFY":
            return carla.VehicleControl(
                throttle=0.08,
                steer=0.0,
                brake=0.0,
                reverse=False,
                hand_brake=False,
                manual_gear_shift=False,
            )
        if phase == "CHANGE":
            steer = -0.2 if direction == "left" else 0.2
            return carla.VehicleControl(
                throttle=0.10,
                steer=steer,
                brake=0.0,
                reverse=False,
                hand_brake=False,
                manual_gear_shift=False,
            )
        return None

    def _finish(self) -> None:
        with self._state_lock:
            self.phase = "IDLE"
            self.direction = None
            self._segment_start_location = None
