from __future__ import annotations

import threading
import time
from typing import Optional


class LaneChangeManager:
    """
    Nonblocking lane-change state machine.

    The route planner (CARLA GPS) owns lane-change completion. Camera lane
    angles are deliberately not used to release steering back to lane-follow.
    While the planner keeps requesting the same lane change, this manager owns
    the steering command and cycles through two steered attempts followed by
    two straight attempts.
    """

    STEER_ATTEMPTS_PER_CYCLE = 2
    STRAIGHT_ATTEMPTS_PER_CYCLE = 2

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

        # IDLE / VERIFY / CHANGE / STRAIGHT / CHECK
        self.phase = "IDLE"
        self.direction: Optional[str] = None
        self._segment_start_location = None
        self._pending_next_phase: Optional[str] = None
        self._steer_attempts = 0
        self._straight_attempts = 0
        self._state_lock = threading.Lock()

        self._planner_query_active = False
        self._planner_query_kind: Optional[str] = None
        self._last_planner_query_at = 0.0
        self._request_id = 0
        self._latest_planner_result: Optional[tuple[int, str]] = None
        self._latest_active_planner_result: Optional[tuple[int, Optional[str]]] = None

    @staticmethod
    def _direction_from_instruction(instruction: object) -> Optional[str]:
        value = str(instruction or "").strip().upper()
        if value == "CHANGE_LANE_LEFT":
            return "left"
        if value == "CHANGE_LANE_RIGHT":
            return "right"
        return None

    def _movement_active(self) -> bool:
        with self._state_lock:
            return self.phase in {"VERIFY", "CHANGE", "STRAIGHT", "CHECK"}

    def movement_active(self) -> bool:
        return self._movement_active()

    def phase_name(self) -> str:
        with self._state_lock:
            return self.phase

    def _reset_attempt_cycle(self) -> None:
        self._steer_attempts = 0
        self._straight_attempts = 0
        self._pending_next_phase = None

    def _apply_planner_result(self, current_location) -> None:
        if current_location is None:
            return
        with self._state_lock:
            result = self._latest_planner_result
            if result is None or self.phase != "IDLE":
                return
            self._latest_planner_result = None
            _, direction = result
            self.direction = direction
            self.phase = "VERIFY"
            self._segment_start_location = current_location
            self._reset_attempt_cycle()

    def _start_planner_query(
        self,
        current_location,
        goal_location,
        *,
        active_check: bool,
        force: bool = False,
    ) -> bool:
        if self.planner is None or current_location is None:
            return False

        now = time.monotonic()
        interval = max(
            0.05,
            float(getattr(conf, "LANE_CHANGE_PLANNER_CHECK_INTERVAL_SECONDS", 0.50)),
        )
        with self._state_lock:
            if self._planner_query_active:
                return False
            if not force and now - self._last_planner_query_at < interval:
                return False

            self._planner_query_active = True
            self._planner_query_kind = "active" if active_check else "idle"
            self._last_planner_query_at = now
            self._request_id += 1
            request_id = self._request_id
            query_kind = self._planner_query_kind

        def query_worker() -> None:
            try:
                instruction = self.planner.get_next_maneuver_text(
                    current_location,
                    goal_location,
                )
                direction = self._direction_from_instruction(instruction)
                with self._state_lock:
                    if request_id != self._request_id:
                        return
                    if query_kind == "active" and self.phase == "CHECK":
                        # None means the GPS no longer requests this lane change.
                        self._latest_active_planner_result = (request_id, direction)
                    elif query_kind == "idle" and self.phase == "IDLE" and direction:
                        self._latest_planner_result = (request_id, direction)
                        self.lane_change_debounce_until = (
                            time.time() + self.lane_change_debounce_seconds
                        )
            except Exception:
                # A failed GPS query is not evidence that the lane change ended.
                # CHECK will retry while retaining steering ownership.
                pass
            finally:
                with self._state_lock:
                    if request_id == self._request_id:
                        self._planner_query_active = False
                        self._planner_query_kind = None

        threading.Thread(
            target=query_worker,
            name="carla-lane-planner-query",
            daemon=True,
        ).start()
        return True

    def _next_phase_after_attempt(self, completed_phase: str) -> str:
        self.lane_change_attempts += 1
        if completed_phase == "CHANGE":
            self._steer_attempts += 1
            if self._steer_attempts < self.STEER_ATTEMPTS_PER_CYCLE:
                return "CHANGE"
            return "STRAIGHT"

        self._straight_attempts += 1
        if self._straight_attempts < self.STRAIGHT_ATTEMPTS_PER_CYCLE:
            return "STRAIGHT"

        # Two straight attempts did not satisfy the GPS. Start a fresh cycle.
        self._reset_attempt_cycle()
        return "CHANGE"

    def _consume_active_planner_result(self, current_location) -> bool:
        if current_location is None:
            return False

        with self._state_lock:
            result = self._latest_active_planner_result
            if result is None or self.phase != "CHECK":
                return False
            self._latest_active_planner_result = None
            _, direction = result

            if direction == self.direction:
                self.phase = self._pending_next_phase or "CHANGE"
                self._segment_start_location = current_location
                self._pending_next_phase = None
                return True

            if direction is not None:
                # The current request ended and GPS now wants the opposite
                # lane. Keep manager ownership and begin that new maneuver.
                self.direction = direction
                self.phase = "VERIFY"
                self._segment_start_location = current_location
                self._reset_attempt_cycle()
                return True

            # GPS no longer requests a lane change. Only now may lane-follow
            # regain control on the next command calculation.
            self._finish_locked()
            return True

    def update(self, current_location, goal_location) -> None:
        if self.vehicle is None:
            return

        self._apply_planner_result(current_location)

        with self._state_lock:
            phase = self.phase
            segment_start = self._segment_start_location

        now = time.monotonic()

        if phase == "VERIFY":
            if segment_start is not None and current_location is not None:
                try:
                    traveled = float(current_location.distance(segment_start))
                except Exception:
                    traveled = 0.0
                verify_distance = max(
                    0.1, float(getattr(conf, "LANE_CHANGE_VERIFY_DISTANCE_M", 1.0))
                )
                if traveled >= verify_distance:
                    with self._state_lock:
                        if self.phase == "VERIFY":
                            self.phase = "CHANGE"
                            self._segment_start_location = current_location
            return

        if phase in {"CHANGE", "STRAIGHT"}:
            if segment_start is None or current_location is None:
                return
            try:
                traveled = float(current_location.distance(segment_start))
            except Exception:
                traveled = 0.0
            attempt_distance = max(
                0.25, float(getattr(conf, "LANE_CHANGE_STEER_DISTANCE_M", 2.0))
            )
            if traveled >= attempt_distance:
                with self._state_lock:
                    if self.phase != phase:
                        return
                    self._pending_next_phase = self._next_phase_after_attempt(phase)
                    self.phase = "CHECK"
                    self._segment_start_location = None
                # Query immediately at the end of each movement attempt.
                self._start_planner_query(
                    current_location,
                    goal_location,
                    active_check=True,
                    force=True,
                )
            return

        if phase == "CHECK":
            if self._consume_active_planner_result(current_location):
                return
            with self._state_lock:
                query_active = self._planner_query_active
            if not query_active:
                self._start_planner_query(
                    current_location,
                    goal_location,
                    active_check=True,
                )
            return

        if phase != "IDLE":
            return

        if time.time() < self.lane_change_debounce_until:
            return
        self._start_planner_query(
            current_location,
            goal_location,
            active_check=False,
        )

    def control_override(self):
        """Keep lane-follow out until GPS no longer requests this lane change."""
        import carla

        with self._state_lock:
            phase = self.phase
            direction = self.direction

        if phase == "VERIFY":
            throttle = 0.08
            steer = 0.0
        elif phase == "CHANGE":
            throttle = 0.10
            magnitude = max(
                0.0,
                min(1.0, float(getattr(conf, "LANE_CHANGE_STEER_VALUE", 0.2))),
            )
            steer = -magnitude if direction == "left" else magnitude
        elif phase in {"STRAIGHT", "CHECK"}:
            # STRAIGHT is an intentional attempt; CHECK keeps the vehicle
            # straight while the asynchronous GPS query is being resolved.
            throttle = 0.10
            steer = 0.0
        else:
            return None

        return carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=0.0,
            reverse=False,
            hand_brake=False,
            manual_gear_shift=False,
        )

    def _finish_locked(self) -> None:
        self.phase = "IDLE"
        self.direction = None
        self._segment_start_location = None
        self._reset_attempt_cycle()
        self._latest_active_planner_result = None

    def _finish(self) -> None:
        with self._state_lock:
            self._finish_locked()
