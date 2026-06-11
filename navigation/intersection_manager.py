from __future__ import annotations

import threading
from typing import Optional, Sequence, Tuple, Any

from navigation.navigate import move_vehicle_for_distance
from utils.vehicle_utils import make_stop_control


class IntersectionManager:
    def __init__(self, vehicle, planner) -> None:
        self.vehicle = vehicle
        self.planner = planner

        self.movement_thread: Optional[threading.Thread] = None
        self.turning_intersection = False
        self.next_maneuver: Optional[str] = None

    def _movement_active(self) -> bool:
        return self.movement_thread is not None and self.movement_thread.is_alive()

    def movement_active(self) -> bool:
        return self._movement_active()

    def _start_movement_sequence(self, segments: Sequence[Tuple[float, float, bool, float, float]]) -> None:
        if self.vehicle is None or self._movement_active():
            return

        def worker() -> None:
            try:
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
            finally:
                try:
                    if self.vehicle is not None:
                        self.vehicle.apply_control(make_stop_control())
                except Exception:
                    pass
                self.turning_intersection = False

        self.turning_intersection = True
        self.movement_thread = threading.Thread(target=worker, daemon=True)
        self.movement_thread.start()

    def update(self, is_intersection: bool, current_location, goal_location) -> None:
        if not is_intersection:
            return

        dist_next_maneuver = self.planner.distance_to_next_maneuver(
            current_location,
            goal_location,
        )

        if dist_next_maneuver is None:
            self.next_maneuver = None
            return

        if dist_next_maneuver > 6:
            return

        self.next_maneuver = self.planner.get_next_maneuver_text(
            current_location,
            goal_location,
        )

        if self.next_maneuver == "RIGHT":
            self._start_movement_sequence([
                (3, 0.04, True, 0.17, 20.0),
                (4, 0.08, True, 0.17, 20.0),
                (11, 0.16, True, 0.17, 20.0),
                (3, 0.6, True, 0.17, 20.0),
            ])

        elif self.next_maneuver == "LEFT":
            self._start_movement_sequence([
                (18.5, 0, True, 0.2, 20.0),
                (18, -0.23, True, 0.2, 20.0),
                (4.5, 0, True, 0.2, 20.0),
            ])

        elif self.next_maneuver == "STRAIGHT":
            self._start_movement_sequence([
                (30, 0.0, True, 0.2, 20.0),
            ])