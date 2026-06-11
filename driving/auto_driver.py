from __future__ import annotations

from typing import Optional, Any

import carla
import numpy as np

from utils.vehicle_utils import blank_frame, make_stop_control


class AutoDriver:
    def __init__(
        self,
        vehicle,
        world,
        vision_processor,
        controller,
        intersection_model,
        lane_change_manager,
        intersection_manager,
    ) -> None:
        self.vehicle = vehicle
        self.world = world
        self.vision_processor = vision_processor
        self.controller = controller
        self.intersection_model = intersection_model
        self.lane_change_manager = lane_change_manager
        self.intersection_manager = intersection_manager

        self.is_intersection = False
        self.current_location = None
        self.goal_location = None

    def movement_active(self) -> bool:
        lane_active = False
        inter_active = False

        if self.lane_change_manager is not None:
            lane_active = self.lane_change_manager.movement_active()

        if self.intersection_manager is not None:
            inter_active = self.intersection_manager.movement_active()

        return lane_active or inter_active

    def update(
        self,
        rgb_frame: Optional[np.ndarray],
        semantic_frame: Optional[np.ndarray],
        current_location,
        goal_location,
    ) -> np.ndarray:
        self.current_location = current_location
        self.goal_location = goal_location

        self.is_intersection = self.intersection_model.is_intersection_ahead(rgb_frame)

        if semantic_frame is None:
            screen = rgb_frame.copy() if rgb_frame is not None else blank_frame()
            if self.vehicle is not None:
                self.vehicle.apply_control(make_stop_control())
            return screen

        result = self.vision_processor.detect(semantic_frame, rgb_frame)
        error = float(result.get("error", 0.0))
        debug = result.get("debug", {})

        screen = debug.get("combined", semantic_frame).copy()

        control = self.controller.update(error=error)
        if self.vehicle is not None:
            self.vehicle.apply_control(control)

        if self.lane_change_manager is not None:
            self.lane_change_manager.update(current_location, goal_location)

        if self.intersection_manager is not None:
            self.intersection_manager.update(
                self.is_intersection,
                current_location,
                goal_location,
            )

        return screen