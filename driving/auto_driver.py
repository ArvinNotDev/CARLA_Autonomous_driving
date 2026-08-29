from __future__ import annotations

from typing import Optional, Any

import carla
import numpy as np

from utils.vehicle_utils import blank_frame, make_stop_control


class AutoDriver:
    """Compute autonomous control without performing CARLA RPCs."""

    def __init__(
        self,
        vehicle,
        world,
        controller,
        intersection_model,
        lane_change_manager,
        intersection_manager,
    ) -> None:
        self.vehicle = vehicle
        self.world = world
        self.controller = controller
        self.intersection_model = intersection_model
        self.lane_change_manager = lane_change_manager
        self.intersection_manager = intersection_manager

        self.is_intersection = False
        self.current_location = None
        self.goal_location = None

    def movement_active(self) -> bool:
        lane_active = (
            self.lane_change_manager is not None
            and self.lane_change_manager.movement_active()
        )
        inter_active = (
            self.intersection_manager is not None
            and self.intersection_manager.movement_active()
        )
        return lane_active or inter_active

    def update(
        self,
        rgb_frame: Optional[np.ndarray],
        vision_result: Optional[dict],
        current_location,
        goal_location,
        intersection_sequence,
    ) -> tuple[bool, np.ndarray, carla.VehicleControl]:
        self.current_location = current_location
        self.goal_location = goal_location

        screen = (
            rgb_frame.copy()
            if rgb_frame is not None
            else blank_frame()
        )

        if vision_result is None:
            return False, screen, make_stop_control()

        error = float(vision_result.get("error", 0.0) or 0.0)
        debug = vision_result.get("debug", {})
        combined = debug.get("combined")
        if isinstance(combined, np.ndarray):
            screen = combined.copy()

        if intersection_sequence:
            return True, screen, self.controller.update(error=error)

        if self.lane_change_manager is not None:
            self.lane_change_manager.update(
                current_location,
                goal_location,
                line_angles={
                    "left": debug.get("left_line_angle_deg"),
                    "right": debug.get("right_line_angle_deg"),
                },
            )
            override = self.lane_change_manager.control_override()
            if override is not None:
                return True, screen, override

        control = self.controller.update(error=error)
        return True, screen, control
