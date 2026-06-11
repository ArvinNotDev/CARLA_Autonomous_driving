from __future__ import annotations

import threading
import time
from typing import Any, Optional

import carla
import cv2
import numpy as np

import config_city as conf
from carla_manager import CarlaManager
from controllers.controller import FixedSpeedPIDController
from controllers.input_manager import InputManager
from driving.auto_driver import AutoDriver
from navigation.intersection_detector import IntersectionModel
from navigation.intersection_manager import IntersectionManager
from navigation.global_planner import RoutePlanner
from navigation.lane_change_manager import LaneChangeManager
from navigation.lane_side import LaneSideModel
from sensors.camera_manager import CameraManager
from stream import start_stream
from ui.renderer import Renderer
from utils.vehicle_utils import blank_frame, make_stop_control
from vision.city_vision_processing import VisionProcessor
from vision.color_extractor import HSVColorThresholdExtractor
from vision.drivable_area_debugger import DrivableAreaDebugger


def cfg(name: str, default: Any) -> Any:
    return getattr(conf, name, default)


class CarlaLaneDrivingApp:
    """
    Main CARLA app for lane following and manual/auto mode switching.
    """

    def __init__(self) -> None:
        self.carla_manager = CarlaManager()
        self.world = None
        self.vehicle = None

        self.camera_manager: Optional[CameraManager] = None
        self.vision_processor: Optional[VisionProcessor] = None
        self.controller: Optional[FixedSpeedPIDController] = None
        self.input_manager: Optional[InputManager] = None
        self.renderer = Renderer()
        self.drivable_debugger: Optional[DrivableAreaDebugger] = None

        self.auto_driver: Optional[AutoDriver] = None
        self.lane_change_manager: Optional[LaneChangeManager] = None
        self.intersection_manager: Optional[IntersectionManager] = None

        self.auto_mode = bool(cfg("AUTO_MODE_DEFAULT", True))
        self.running = True

        self.intersection_model = IntersectionModel(
            "models_and_datasets/models/junction_model_resnet18.pt"
        )
        self.lane_side = LaneSideModel(
            "models_and_datasets/models/lane_side_model_resnet18.pt"
        )

        self.planner: Optional[RoutePlanner] = None
        self.goal_location = None
        self.current_location = None
        self.vehicle_wp = None

    def _get_latest_rgb_frame(self) -> Optional[np.ndarray]:
        if self.camera_manager is None:
            return None
        return self.camera_manager.get_latest_rgb()

    def _get_latest_semantic_frame(self) -> Optional[np.ndarray]:
        if self.camera_manager is None:
            return None
        return self.camera_manager.get_latest_semantic()

    def _show_frame(self, screen: np.ndarray) -> None:
        self.renderer.show_frame(screen)

    def _show_drivable_area(
        self,
        rgb_frame: Optional[np.ndarray],
        semantic_frame: Optional[np.ndarray],
    ) -> None:
        if self.drivable_debugger is None:
            cv2.imshow("drivable area", blank_frame())
            cv2.waitKey(1)
            return

        self.drivable_debugger.show(rgb_frame, semantic_frame)

    def setup(self) -> None:
        self.world = self.carla_manager.connect()
        self.vehicle = self.carla_manager.spawn_vehicle()

        self.camera_manager = CameraManager(self.world, self.vehicle)
        self.camera_manager.start()

        HSVColorThresholdExtractor(
            morph_kernel_size=getattr(conf, "MORPH_KERNEL_SIZE", 1)
        )

        self.planner = RoutePlanner(self.world)
        self.goal_location = carla.Location(x=20, y=24.74, z=0)
        self.current_location = self.vehicle.get_location()
        self.vehicle_wp = self.world.get_map().get_waypoint(self.current_location)

        self.vision_processor = VisionProcessor()

        self.controller = FixedSpeedPIDController(
            fixed_throttle=conf.FIXED_THROTTLE,
            kp=conf.KP,
            ki=conf.KI,
            kd=conf.KD,
            steer_limit=conf.STEER_LIMIT,
            max_steer_step=conf.MAX_STEER_STEP,
        )

        self.input_manager = InputManager()

        self.lane_change_manager = LaneChangeManager(
            vehicle=self.vehicle,
            lane_side_model=self.lane_side,
            planner=self.planner,
            get_latest_rgb=self._get_latest_rgb_frame,
            lane_change_debounce_seconds=float(cfg("LANE_CHANGE_DEBOUNCE_SECONDS", 1.5)),
        )

        self.intersection_manager = IntersectionManager(
            vehicle=self.vehicle,
            planner=self.planner,
        )

        self.auto_driver = AutoDriver(
            vehicle=self.vehicle,
            world=self.world,
            vision_processor=self.vision_processor,
            controller=self.controller,
            intersection_model=self.intersection_model,
            lane_change_manager=self.lane_change_manager,
            intersection_manager=self.intersection_manager,
        )

        self.drivable_debugger = DrivableAreaDebugger(self.vision_processor)

        stream_thread = threading.Thread(
            target=start_stream,
            kwargs={"host": conf.STREAM_HOST, "port": conf.STREAM_PORT},
            daemon=True,
        )
        stream_thread.start()
        time.sleep(1.0)

    def shutdown(self) -> None:
        try:
            if self.vehicle is not None:
                self.vehicle.apply_control(make_stop_control())
        except Exception:
            pass

        if self.input_manager is not None:
            try:
                self.input_manager.shutdown()
            except Exception:
                pass

        try:
            if self.camera_manager is not None:
                self.camera_manager.cleanup()
        except Exception:
            pass

        try:
            if self.carla_manager is not None:
                self.carla_manager.cleanup()
        except Exception:
            pass

        if cfg("SHOW_OPENCV_WINDOW", False):
            cv2.destroyAllWindows()

    def _render_mode_overlay(self, screen: np.ndarray, mode: str, error: float) -> None:
        self.renderer.render_mode_overlay(screen, mode, error)

    def _movement_active(self) -> bool:
        if self.auto_driver is None:
            return False
        return self.auto_driver.movement_active()

    def run(self) -> None:
        self.setup()
        try:
            while self.running:
                if (
                    self.input_manager is None
                    or self.controller is None
                    or self.vision_processor is None
                    or self.auto_driver is None
                ):
                    break

                self.running, toggle_auto, manual_control = self.input_manager.poll()

                if toggle_auto:
                    self.auto_mode = not self.auto_mode
                    self.controller.reset()
                    if self.vehicle is not None:
                        self.vehicle.apply_control(make_stop_control())

                rgb_frame = (
                    self.camera_manager.get_latest_rgb()
                    if self.camera_manager is not None
                    else None
                )
                semantic_frame = (
                    self.camera_manager.get_latest_semantic()
                    if self.camera_manager is not None
                    else None
                )

                if rgb_frame is None and semantic_frame is None:
                    screen = blank_frame()
                    self._show_frame(screen)
                    self._show_drivable_area(rgb_frame, semantic_frame)
                    time.sleep(0.01)
                    continue

                self._show_drivable_area(rgb_frame, semantic_frame)

                movement_active = self._movement_active()
                if movement_active:
                    screen = rgb_frame.copy() if rgb_frame is not None else blank_frame()
                    self._show_frame(screen)
                    time.sleep(0.01)
                    continue

                if self.auto_mode:
                    self.current_location = self.vehicle.get_location() if self.vehicle is not None else None
                    screen = self.auto_driver.update(
                        rgb_frame,
                        semantic_frame,
                        self.current_location,
                        self.goal_location,
                    )
                    self._show_frame(screen)

                else:
                    screen = rgb_frame.copy() if rgb_frame is not None else blank_frame()
                    self._render_mode_overlay(screen, "MANUAL", 0.0)

                    if manual_control is None:
                        manual_control = make_stop_control()

                    if self.vehicle is not None:
                        self.vehicle.apply_control(manual_control)

                    self._show_frame(screen)

                time.sleep(0.01)

        finally:
            self.shutdown()


def main() -> None:
    app = CarlaLaneDrivingApp()
    app.run()


if __name__ == "__main__":
    main()