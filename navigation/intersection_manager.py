from __future__ import annotations

import threading
from typing import Optional, Sequence, Tuple

import config_city as conf
from navigation.navigate import move_vehicle_for_distance
from utils.vehicle_utils import make_stop_control


class IntersectionManager:
    """Runs the short fixed-distance junction lead-in off the render thread."""

    def __init__(self, vehicle, planner) -> None:
        self.vehicle = vehicle
        self.planner = planner

        self.movement_thread: Optional[threading.Thread] = None
        self.turning_intersection = False
        self.next_maneuver: Optional[str] = None
        self._planning_active = False
        self._state_lock = threading.Lock()

    def _movement_active(self) -> bool:
        with self._state_lock:
            return bool(
                self._planning_active
                or (
                    self.movement_thread is not None
                    and self.movement_thread.is_alive()
                )
            )

    def movement_active(self) -> bool:
        return self._movement_active()

    @staticmethod
    def _segments_for_maneuver(
        maneuver: Optional[str],
    ) -> list[Tuple[float, float, bool, float, float]]:
        throttle = float(getattr(conf, "JUNCTION_STATIC_THROTTLE", 0.2))
        timeout = float(getattr(conf, "JUNCTION_STATIC_TIMEOUT_SECONDS", 20.0))
        entry = float(getattr(conf, "JUNCTION_ENTRY_DISTANCE_M", 11.0))
        right_turn = float(getattr(conf, "JUNCTION_RIGHT_TURN_DISTANCE_M", 6.0))
        left_straight = float(
            getattr(conf, "JUNCTION_LEFT_STRAIGHT_DISTANCE_M", 10.0)
        )
        left_turn = float(getattr(conf, "JUNCTION_LEFT_TURN_DISTANCE_M", 4.0))
        straight = float(getattr(conf, "JUNCTION_STRAIGHT_DISTANCE_M", 3.0))

        if maneuver == "RIGHT":
            return [
                (entry, 0.0, True, throttle, timeout),
                (right_turn, 0.65, True, throttle, timeout),
            ]
        if maneuver == "LEFT":
            return [
                (left_straight, 0.0, True, throttle, timeout),
                (left_turn, -0.5, True, throttle, timeout),
            ]
        if maneuver == "STRAIGHT":
            return [(straight, 0.0, True, throttle, timeout)]
        return []

    def start_for_intersection(self, current_location, goal_location) -> bool:
        """Schedule route lookup and the static lead-in without blocking main."""
        if self.vehicle is None or self.planner is None:
            return False

        with self._state_lock:
            if self._planning_active or (
                self.movement_thread is not None
                and self.movement_thread.is_alive()
            ):
                return False
            self._planning_active = True
            self.next_maneuver = None

        def worker() -> None:
            try:
                distance = self.planner.distance_to_next_maneuver(
                    current_location,
                    goal_location,
                )
                entry_trigger = float(
                    getattr(conf, "JUNCTION_ENTRY_DISTANCE_M", 11.0)
                )
                if distance is None or distance > entry_trigger:
                    return

                maneuver = self.planner.get_next_maneuver_text(
                    current_location,
                    goal_location,
                )
                self.next_maneuver = maneuver
                segments = self._segments_for_maneuver(maneuver)
                if not segments:
                    return

                self.turning_intersection = True
                for distance_m, steer, forward, throttle, timeout in segments:
                    move_vehicle_for_distance(
                        self.vehicle,
                        distance_m,
                        steer,
                        forward,
                        throttle,
                        timeout,
                        blocking=True,
                    )
            except Exception:
                self.next_maneuver = None
            finally:
                try:
                    if self.vehicle is not None:
                        self.vehicle.apply_control(make_stop_control())
                except Exception:
                    pass
                with self._state_lock:
                    self.turning_intersection = False
                    self._planning_active = False
                    self.movement_thread = None

        thread = threading.Thread(
            target=worker,
            name="carla-junction-lead-in",
            daemon=True,
        )
        with self._state_lock:
            self.movement_thread = thread
        thread.start()
        return True

    def update(self, is_intersection: bool, current_location, goal_location) -> bool:
        if not is_intersection:
            return False
        return self.start_for_intersection(current_location, goal_location)
