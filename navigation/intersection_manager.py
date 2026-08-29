from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Optional, Tuple

import config_city as conf


@dataclass(frozen=True)
class JunctionPlan:
    request_id: int
    maneuver: str
    segments: tuple[tuple[float, float, bool, float, float], ...]




@dataclass(frozen=True)
class StaticControlFallback:
    throttle: float
    steer: float
    brake: float = 0.0
    reverse: bool = False
    hand_brake: bool = False
    manual_gear_shift: bool = False


class IntersectionManager:
    """
    Nonblocking junction state machine.

    Planner lookup may run asynchronously, but no worker thread ever calls
    vehicle.apply_control. The control loop owns all vehicle commands.

    State:
        IDLE -> PLANNING -> CROSSWALK -> STATIC -> IDLE

    CROSSWALK is an explicit nonblocking brake/pause phase. The main control loop
    owns the configurable trajectory takeover after STATIC.
    """

    def __init__(self, vehicle, planner) -> None:
        self.vehicle = vehicle
        self.planner = planner

        self.next_maneuver: Optional[str] = None
        self.phase = "IDLE"
        self._state_lock = threading.Lock()
        self._planning_thread: Optional[threading.Thread] = None
        self._planning_request_id = 0
        self._latest_plan: Optional[JunctionPlan] = None
        self._last_plan_error: Optional[str] = None

        self._active_plan: Optional[JunctionPlan] = None
        self._segment_index = 0
        self._segment_start_location = None
        self._segment_started_at = 0.0
        self._crosswalk_started_at = 0.0

    @staticmethod
    def _segments_for_maneuver(
        maneuver: Optional[str],
    ) -> list[Tuple[float, float, bool, float, float]]:
        throttle = float(getattr(conf, "JUNCTION_STATIC_THROTTLE", 0.2))
        timeout = float(getattr(conf, "JUNCTION_STATIC_TIMEOUT_SECONDS", 20.0))
        configured = getattr(conf, "JUNCTION_MOVEMENT_SEQUENCES", {})
        sequence = configured.get(str(maneuver or "").upper(), [])
        segments: list[Tuple[float, float, bool, float, float]] = []
        for action in sequence if isinstance(sequence, list) else []:
            if not isinstance(action, dict):
                continue
            try:
                distance_m = max(0.0, float(action["distance_m"]))
                steering_value = max(-1.0, min(1.0, float(action["steering_value"])))
            except (KeyError, TypeError, ValueError):
                continue
            if distance_m <= 0.0:
                continue
            segments.append((distance_m, steering_value, True, throttle, timeout))
        return segments

    def movement_active(self) -> bool:
        with self._state_lock:
            return self.phase in {"PLANNING", "CROSSWALK", "STATIC"}

    def maneuver_active(self) -> bool:
        with self._state_lock:
            return self.phase in {"CROSSWALK", "STATIC"}

    def start_for_intersection(self, current_location, goal_location) -> bool:
        if self.vehicle is None or self.planner is None:
            return False

        with self._state_lock:
            if self.phase != "IDLE":
                return False
            self._planning_request_id += 1
            request_id = self._planning_request_id
            self.phase = "PLANNING"
            self.next_maneuver = None
            self._latest_plan = None
            self._active_plan = None
            self._last_plan_error = None

        def worker() -> None:
            try:
                distance = self.planner.distance_to_next_maneuver(
                    current_location,
                    goal_location,
                )
                entry_trigger = float(getattr(conf, "JUNCTION_ENTRY_DISTANCE_M", 11.0))
                if distance is None or float(distance) > entry_trigger:
                    plan = None
                else:
                    maneuver = str(
                        self.planner.get_next_maneuver_text(
                            current_location,
                            goal_location,
                        )
                        or ""
                    )
                    segments = self._segments_for_maneuver(maneuver)
                    plan = (
                        JunctionPlan(
                            request_id=request_id,
                            maneuver=maneuver,
                            segments=tuple(segments),
                        )
                        if segments
                        else None
                    )

                with self._state_lock:
                    if request_id != self._planning_request_id:
                        return
                    self._latest_plan = plan
                    self.next_maneuver = plan.maneuver if plan else None
                    if plan is None:
                        self.phase = "IDLE"
            except Exception as exc:
                with self._state_lock:
                    if request_id == self._planning_request_id:
                        self._latest_plan = None
                        self._last_plan_error = str(exc)
                        self.phase = "IDLE"

        thread = threading.Thread(
            target=worker,
            name="carla-junction-planner",
            daemon=True,
        )
        with self._state_lock:
            self._planning_thread = thread
        thread.start()
        return True

    def _activate_plan_if_ready(self, current_location) -> None:
        with self._state_lock:
            plan = self._latest_plan
            if self.phase != "PLANNING" or plan is None:
                return
            self._latest_plan = None
            self._active_plan = plan
            self.phase = "CROSSWALK"
            self._segment_index = 0
            self._segment_start_location = None
            self._segment_started_at = 0.0
            self._crosswalk_started_at = time.monotonic()
            self.next_maneuver = plan.maneuver

    def update(self, current_location) -> None:
        """Advance the state machine without sleeping or blocking."""
        self._activate_plan_if_ready(current_location)

        with self._state_lock:
            phase = self.phase
            plan = self._active_plan
            crosswalk_started_at = self._crosswalk_started_at

        if phase == "CROSSWALK":
            pause = max(0.0, float(getattr(conf, "CROSSWALK_SLEEP", 3.0)))
            if time.monotonic() - crosswalk_started_at >= pause:
                with self._state_lock:
                    if self.phase != "CROSSWALK":
                        return
                    self.phase = "STATIC"
                    self._segment_index = 0
                    self._segment_start_location = current_location
                    self._segment_started_at = time.monotonic()
            return

        with self._state_lock:
            if self.phase != "STATIC":
                return
            plan = self._active_plan
            idx = self._segment_index
            start = self._segment_start_location
            segment_started_at = self._segment_started_at

        if plan is None or start is None or current_location is None:
            return

        distance_m = float(current_location.distance(start))
        target_m, _, _, _, timeout = plan.segments[idx]
        segment_elapsed = max(0.0, time.monotonic() - segment_started_at)

        if distance_m >= float(target_m) or segment_elapsed >= float(timeout):
            with self._state_lock:
                if self.phase != "STATIC":
                    return
                self._segment_index += 1
                if self._segment_index >= len(plan.segments):
                    self.phase = "IDLE"
                    self._active_plan = None
                    self._segment_start_location = None
                    self._segment_started_at = 0.0
                    return
                self._segment_start_location = current_location
                self._segment_started_at = time.monotonic()

    def static_control(self, current_location):
        """Return current crosswalk brake or static lead-in control."""
        with self._state_lock:
            phase = self.phase
            active = self._active_plan
            idx = self._segment_index
            segment = active.segments[idx] if active is not None and idx < len(active.segments) else None

        if phase == "CROSSWALK":
            values = dict(throttle=0.0, steer=0.0, brake=1.0, reverse=False, hand_brake=False, manual_gear_shift=False)
        elif phase == "STATIC" and segment is not None:
            _, steer, forward, throttle, _ = segment
            values = dict(
                throttle=max(0.0, min(1.0, float(throttle))),
                steer=max(-1.0, min(1.0, float(steer))),
                brake=0.0,
                reverse=not bool(forward),
                hand_brake=False,
                manual_gear_shift=False,
            )
        else:
            return None
        try:
            import carla
            return carla.VehicleControl(**values)
        except ImportError:
            return StaticControlFallback(**values)

    def phase_name(self) -> str:
        with self._state_lock:
            return self.phase

    def last_plan_error(self) -> Optional[str]:
        with self._state_lock:
            return self._last_plan_error

    def stop(self) -> None:
        with self._state_lock:
            self._planning_request_id += 1
            self.phase = "IDLE"
            self._latest_plan = None
            self._active_plan = None
            self._segment_start_location = None
