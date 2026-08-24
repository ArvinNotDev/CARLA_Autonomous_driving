from __future__ import annotations

import threading
import time
from typing import Callable, Optional, Dict, Any

import config_city as conf
from navigation.navigate import move_vehicle_for_distance


class LaneChangeManager:
    def __init__(
        self,
        vehicle,
        lane_side_model,
        planner,
        get_latest_rgb: Callable[[], Optional[Any]],
        lane_change_debounce_seconds: float = 1.5,
    ) -> None:
        self.vehicle = vehicle
        self.lane_side = lane_side_model
        self.planner = planner
        self.get_latest_rgb = get_latest_rgb

        self.lane_change_debounce_seconds = float(lane_change_debounce_seconds)
        self.lane_change_debounce_until = 0.0
        self.lane_change_attempts = 0
        self.line_angles: Dict[str, Optional[float]] = {"left": None, "right": None}
        self.angle_cooldown_until = 0.0

        self.movement_thread: Optional[threading.Thread] = None
        self.turning_intersection = False

    def _movement_active(self) -> bool:
        return self.movement_thread is not None and self.movement_thread.is_alive()

    def movement_active(self) -> bool:
        return self._movement_active()

    def _get_lane_state(self, rgb_frame: Optional[Any]) -> Dict[str, bool]:
        if rgb_frame is None:
            return {
                "left_lane": False,
                "right_lane": False,
                "out_from_right": False,
                "out_from_left": False,
            }

        return {
            "left_lane": self.lane_side.is_left_lane(rgb_frame),
            "right_lane": self.lane_side.is_right_lane(rgb_frame),
            "out_from_right": self.lane_side.is_out_from_right(rgb_frame),
            "out_from_left": self.lane_side.is_out_from_left(rgb_frame),
        }

    def _should_change_lane(self, direction: str, lane_state: Dict[str, bool]) -> bool:
        if direction == "left":
            if lane_state["left_lane"]:
                return False
            if lane_state["out_from_left"]:
                return False
            return True

        if direction == "right":
            if lane_state["right_lane"]:
                return False
            if lane_state["out_from_right"]:
                return False
            return True

        return False

    def _perform_verified_lane_change(self, direction: str) -> None:
        if self.vehicle is None:
            return
    
        move_vehicle_for_distance(
            self.vehicle,
            1.0,
            0.0,
            True,
            0.08,
            8.0,
            blocking=True,
        )

        time.sleep(0.15)

        rgb_frame = self.get_latest_rgb()
        lane_state = self._get_lane_state(rgb_frame)

        if not self._should_change_lane(direction, lane_state):
            return

        if direction == "left":
            move_vehicle_for_distance(
                self.vehicle,
                2.0,
                -0.2,
                True,
                0.1,
                20.0,
                blocking=True,
            )

        elif direction == "right":
            move_vehicle_for_distance(
                self.vehicle,
                2.0,
                0.2,
                True,
                0.1,
                20.0,
                blocking=True,
            )

    def _angle_guard_triggered(self) -> bool:
        threshold = float(
            getattr(conf, "LANE_CHANGE_LINE_ANGLE_THRESHOLD_DEG", 20.0)
        )
        return any(
            angle is not None and abs(float(angle)) > threshold
            for angle in self.line_angles.values()
        )

    def _start_verified_lane_change(self, direction: str) -> None:
        if self.vehicle is None or self._movement_active():
            return

        def worker() -> None:
            try:
                self._perform_verified_lane_change(direction)
                self.lane_change_attempts += 1
                check_every = int(
                    getattr(conf, "LANE_CHANGE_ANGLE_CHECK_EVERY", 2)
                )
                if (
                    check_every > 0
                    and self.lane_change_attempts % check_every == 0
                    and self._angle_guard_triggered()
                ):
                    cooldown = float(
                        getattr(
                            conf,
                            "LANE_CHANGE_ANGLE_COOLDOWN_SECONDS",
                            5.0,
                        )
                    )
                    self.angle_cooldown_until = time.time() + cooldown
            finally:
                self.turning_intersection = False

        self.turning_intersection = True
        self.movement_thread = threading.Thread(target=worker, daemon=True)
        self.movement_thread.start()

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

        if time.time() < self.angle_cooldown_until:
            return

        if time.time() < self.lane_change_debounce_until:
            return

        next_maneuver_text = self.planner.get_next_maneuver_text(
            current_location,
            goal_location,
        )

        if next_maneuver_text == "CHANGE_LANE_LEFT":
            self._start_verified_lane_change("left")
            self.lane_change_debounce_until = time.time() + self.lane_change_debounce_seconds

        elif next_maneuver_text == "CHANGE_LANE_RIGHT":
            self._start_verified_lane_change("right")
            self.lane_change_debounce_until = time.time() + self.lane_change_debounce_seconds
