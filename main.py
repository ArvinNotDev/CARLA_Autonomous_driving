from __future__ import annotations

import threading
import time
from typing import Any, Optional, Tuple

import carla
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

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
from ui.spawn_goal_picker import SpawnGoalPicker
from utils.vehicle_utils import blank_frame, make_stop_control
from vision.city_vision_processing import VisionProcessor
from vision.color_extractor import HSVColorThresholdExtractor
from vision.drivable_area_debugger import DrivableAreaDebugger
from trajectory.steering_agent import TrajectorySteeringAgent
from trajectory.visualize import draw_waypoints
from runtime_settings import RuntimeSettings
from ui.control_panel import ControlPanel


def cfg(name: str, default: Any) -> Any:
    return getattr(conf, name, default)


class CarlaLaneDrivingApp:
    """
    Main CARLA app for lane following and manual/auto mode switching.

    Perception is owned by main:
        main -> vision_processor.detect()
        main -> auto_driver.update(..., vision_result, ...)
        main -> drivable_debugger.show(vision_result)
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
        self.trajectory_steering_agent: Optional[TrajectorySteeringAgent] = None

        self.auto_mode = bool(cfg("AUTO_MODE_DEFAULT", True))
        self.running = True

        self.intersection_sequence = False
        self.intersec_once = False
        self.is_intersection = False
        self.seq_timeout: float = 0.0
        self.next_maneuver: Optional[str] = None

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

        self.out_checker = False
        self.out_checker_started_at: Optional[float] = None
        self.out_checker_window_seconds = float(cfg("OUT_CHECKER_WINDOW_SECONDS", 10.0))
        self.out_checker_error_threshold = float(cfg("OUT_CHECKER_ERROR_THRESHOLD", 20.0))

        self._last_trajectory_inference_at = 0.0
        self._cached_trajectory_steer = 0.0
        self._cached_trajectory_pred = None
        self._last_intersection_check_at = 0.0
        self._last_loop_at = time.perf_counter()
        self._fps = 0.0
        self._startup_drive_until = 0.0
        self.settings = RuntimeSettings()
        self.control_panel: Optional[ControlPanel] = None

    def _get_latest_rgb_frame(self) -> Optional[np.ndarray]:
        if self.camera_manager is None:
            return None
        return self.camera_manager.get_latest_rgb()

    def _get_latest_semantic_frame(self) -> Optional[np.ndarray]:
        if self.camera_manager is None:
            return None
        return self.camera_manager.get_latest_semantic()

    def _show_frame(self, screen: np.ndarray) -> None:
        lines = []
        if cfg("DEBUG_SHOW_FPS", True):
            lines.append(f"FPS: {self._fps:.1f}")
        if cfg("DEBUG_SHOW_GPS", True) and self.vehicle is not None:
            try:
                loc = self.vehicle.get_location()
                lines.append(f"GPS: {loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f}")
            except Exception:
                pass
        self.renderer.show_frame(screen, {"lines": lines})

    def _on_settings_changed(self, values: dict) -> None:
        self.settings.apply(values)
        if self.controller is not None:
            self.controller.fixed_throttle = float(cfg("FIXED_THROTTLE", self.controller.fixed_throttle))
            self.controller.kp = float(cfg("KP", self.controller.kp))
            self.controller.ki = float(cfg("KI", self.controller.ki))
            self.controller.kd = float(cfg("KD", self.controller.kd))
            self.controller.steer_limit = float(cfg("STEER_LIMIT", self.controller.steer_limit))
            self.controller.max_steer_step = float(cfg("MAX_STEER_STEP", self.controller.max_steer_step))
        if self.vision_processor is not None:
            self.vision_processor.debug = bool(cfg("VISION_DEBUG", False))
            self.vision_processor.lane_center_alpha = float(
                cfg("LANE_CENTER_SMOOTH_ALPHA", self.vision_processor.lane_center_alpha)
            )
        if self.trajectory_steering_agent is not None:
            self.trajectory_steering_agent.cfg.target_speed_kmh = float(
                cfg("TARGET_SPEED_KMH", self.trajectory_steering_agent.cfg.target_speed_kmh)
            )

    def _request_car_reset(self) -> None:
        if self.control_panel is None:
            return
        QMessageBox.information(
            self.control_panel,
            "Car reset required",
            "This setting changes a CARLA sensor/model resource. Reset the car (or restart the run) to apply it.",
        )

    def _render_mode_overlay(self, screen: np.ndarray, mode: str, error: float) -> None:
        self.renderer.render_mode_overlay(screen, mode, error)

    def _movement_active(self) -> bool:
        if self.auto_driver is None:
            return False
        return self.auto_driver.movement_active()

    def _safe_vehicle_apply_control(self, control: carla.VehicleControl) -> None:
        if self.vehicle is None:
            return
        try:
            self.vehicle.apply_control(control)
        except Exception:
            pass

    def _force_left_correction(self) -> None:
        steer_limit = float(cfg("STEER_LIMIT", 0.35))
        fixed_throttle = float(cfg("FIXED_THROTTLE", 0.3))

        control = carla.VehicleControl(
            throttle=fixed_throttle,
            steer=-abs(steer_limit),
            brake=0.0,
        )
        self._safe_vehicle_apply_control(control)
        time.sleep(1)

    def _extract_error(self, drivable_area_result: Any) -> float:
        if isinstance(drivable_area_result, dict):
            try:
                return float(drivable_area_result.get("error", 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    def _update_out_checker_logic(
        self,
        candidate_out_checker: bool,
        drivable_area_result: Any,
    ) -> None:
        """
        Start a checking window when AutoDriver requests it.
        During that window, if error > threshold, steer left.
        """
        error = self._extract_error(drivable_area_result)
        now = time.monotonic()

        if self.out_checker_started_at is None:
            if candidate_out_checker:
                self.out_checker_started_at = now
                self.out_checker = True
            else:
                self.out_checker = False
            return

        self.out_checker = True

        if error > self.out_checker_error_threshold:
            self._force_left_correction()

        if now - self.out_checker_started_at >= self.out_checker_window_seconds:
            self.out_checker = False
            self.out_checker_started_at = None

    def _apply_goal_to_planner(self) -> None:
        """
        Best-effort integration point for planners that support an explicit goal setter.
        This keeps the code working even if your RoutePlanner API differs.
        """
        if self.planner is None or self.goal_location is None:
            return

        for method_name in (
            "set_goal_location",
            "set_goal",
            "set_destination",
            "set_target_location",
        ):
            method = getattr(self.planner, method_name, None)
            if callable(method):
                try:
                    method(self.goal_location)
                    return
                except TypeError:
                    try:
                        method(goal_location=self.goal_location)
                        return
                    except Exception:
                        pass
                except Exception:
                    pass

    def _update_intersection_manager(self) -> None:
        """
        Best-effort call into IntersectionManager so the app does not crash
        if its API differs slightly.
        """
        if self.intersection_manager is None:
            return

        candidates = [
            (self.intersection_sequence, self.current_location, self.goal_location),
            (),
        ]

        for args in candidates:
            try:
                self.intersection_manager.update(*args)
                return
            except TypeError:
                continue
            except Exception:
                return

    def _get_intersection_input_frame(
        self,
        rgb_frame: Optional[np.ndarray],
        semantic_frame: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        return rgb_frame if rgb_frame is not None else semantic_frame

    def setup(self) -> None:
        self.world = self.carla_manager.connect()
        if self.world is None:
            raise RuntimeError("Failed to connect to CARLA world.")

        picker = SpawnGoalPicker(self.world)
        if picker.exec() != QDialog.DialogCode.Accepted:
            self.running = False
            return

        spawn_tf = picker.get_spawn_transform()
        self.goal_location = picker.get_goal_location()

        self.vehicle = self.carla_manager.spawn_vehicle(spawn_transform=spawn_tf)
        if self.vehicle is None:
            raise RuntimeError("Failed to spawn vehicle.")

        self.camera_manager = CameraManager(self.world, self.vehicle)
        self.camera_manager.start()

        HSVColorThresholdExtractor(
            morph_kernel_size=getattr(conf, "MORPH_KERNEL_SIZE", 1)
        )

        self.planner = RoutePlanner(self.world)
        self._apply_goal_to_planner()

        self.current_location = self.vehicle.get_location()
        self.vehicle_wp = self.world.get_map().get_waypoint(self.current_location)

        self.vision_processor = VisionProcessor("onnx")

        self.controller = FixedSpeedPIDController(
            fixed_throttle=float(cfg("FIXED_THROTTLE", 0.3)),
            kp=float(cfg("KP", 0.0)),
            ki=float(cfg("KI", 0.0)),
            kd=float(cfg("KD", 0.0)),
            steer_limit=float(cfg("STEER_LIMIT", 0.35)),
            max_steer_step=float(cfg("MAX_STEER_STEP", 0.05)),
        )

        self.input_manager = InputManager()

        self.lane_change_manager = LaneChangeManager(
            vehicle=self.vehicle,
            lane_side_model=self.lane_side,
            planner=self.planner,
            get_latest_rgb=self._get_latest_rgb_frame,
            lane_change_debounce_seconds=float(
                cfg("LANE_CHANGE_DEBOUNCE_SECONDS", 4)
            ),
        )

        self.intersection_manager = IntersectionManager(
            vehicle=self.vehicle,
            planner=self.planner,
        )

        self.auto_driver = AutoDriver(
            vehicle=self.vehicle,
            world=self.world,
            controller=self.controller,
            intersection_model=self.intersection_model,
            lane_change_manager=self.lane_change_manager,
            intersection_manager=self.intersection_manager,
        )

        self.trajectory_steering_agent = TrajectorySteeringAgent(
            "models_and_datasets/models/trajectory_modelV2.pt",
            conf,
            "LANE_FOLLOW",
        )

        self.drivable_debugger = DrivableAreaDebugger()
        self._startup_drive_until = time.monotonic() + 3.0
        self.control_panel = ControlPanel(self.settings)
        self.control_panel.settings_changed.connect(self._on_settings_changed)
        self.control_panel.reset_requested.connect(self._request_car_reset)
        self.control_panel.show()

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

    def run(self) -> None:
        self.setup()
        try:
            while self.running:
                QApplication.processEvents()
                tick_now = time.perf_counter()
                dt = tick_now - self._last_loop_at
                self._last_loop_at = tick_now
                if dt > 0:
                    instant_fps = 1.0 / dt
                    self._fps = instant_fps if self._fps <= 0 else 0.9 * self._fps + 0.1 * instant_fps
                if (
                    self.input_manager is None
                    or self.controller is None
                    or self.vision_processor is None
                    or self.auto_driver is None
                    or self.drivable_debugger is None
                    or self.trajectory_steering_agent is None
                ):
                    break

                self.running, toggle_auto, manual_control = self.input_manager.poll()

                if toggle_auto:
                    self.auto_mode = not self.auto_mode
                    self.controller.reset()
                    self._safe_vehicle_apply_control(make_stop_control())

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
                _alt_frame = (
                    self.camera_manager.get_latest_alt()
                    if self.camera_manager is not None
                    else None
                )

                if rgb_frame is None and semantic_frame is None:
                    screen = blank_frame()
                    if self.auto_mode and time.monotonic() < self._startup_drive_until:
                        self._safe_vehicle_apply_control(carla.VehicleControl(throttle=0.18))
                    self._show_frame(screen)
                    time.sleep(0.01)
                    continue

                vision_result = None
                if semantic_frame is not None:
                    try:
                        vision_result = self.vision_processor.detect(
                            semantic_frame,
                            rgb_frame,
                        )
                    except Exception:
                        vision_result = None

                try:
                    drivable_area_result = self.drivable_debugger.show(vision_result)
                except Exception:
                    drivable_area_result = None

                if self._movement_active() and cfg("JUNCTION_CONTROL_MODE", "trajectory") == "static_meters":
                    screen = rgb_frame.copy() if rgb_frame is not None else blank_frame()
                    self._show_frame(screen)
                    time.sleep(0.01)
                    continue

                now = time.perf_counter()
                if self.trajectory_steering_agent is not None:
                    command = "LANE_FOLLOW"
                    if self.is_intersection and self.next_maneuver in getattr(conf, "COMMANDS", ()):
                        command = self.next_maneuver
                    self.trajectory_steering_agent.command_name = command
                trajectory_interval = float(
                    cfg("TRAJECTORY_INFERENCE_INTERVAL_SECONDS", 0.05)
                )
                if (
                    rgb_frame is not None
                    and (
                        self._cached_trajectory_pred is None
                        or now - self._last_trajectory_inference_at
                        >= trajectory_interval
                    )
                ):
                    try:
                        self._cached_trajectory_steer, self._cached_trajectory_pred = (
                            self.trajectory_steering_agent.get_steering_and_pred(
                                rgb_frame
                            )
                        )
                        self._last_trajectory_inference_at = now
                    except Exception:
                        self._cached_trajectory_steer = 0.0
                        self._cached_trajectory_pred = None
                steer = self._cached_trajectory_steer
                pred = self._cached_trajectory_pred

                if self.auto_mode:
                    if self.vehicle is not None:
                        self.current_location = self.vehicle.get_location()
                    else:
                        self.current_location = None
                    if self.intersection_sequence and (time.time() - self.seq_timeout < 10.0):
                        self.next_maneuver = None
                        if self.planner is not None and self.current_location is not None:
                            try:
                                self.next_maneuver = self.planner.get_next_maneuver_text(
                                    self.current_location,
                                    self.goal_location,
                                )
                            except Exception:
                                self.next_maneuver = None

                        if self.intersection_manager is not None and self.intersec_once:
                            self._update_intersection_manager()
                            self.intersec_once = False

                        if cfg("JUNCTION_CONTROL_MODE", "trajectory") == "trajectory":
                            speed = 0.0
                            try:
                                vel = self.vehicle.get_velocity()
                                speed = float((vel.x**2 + vel.y**2 + vel.z**2) ** 0.5) * 3.6
                            except Exception:
                                pass
                            throttle, brake = self.trajectory_steering_agent.throttle_brake_from_speed(
                                speed, float(cfg("TARGET_SPEED_KMH", 25.0))
                            )
                            self._safe_vehicle_apply_control(
                                carla.VehicleControl(throttle=throttle, steer=float(steer), brake=brake)
                            )
                        else:
                            self._safe_vehicle_apply_control(
                                carla.VehicleControl(throttle=0.15, steer=float(steer), brake=0.0)
                            )
                    else:
                        self.seq_timeout = 0.0
                        self.intersection_sequence = False
                        self.intersec_once = False

                    intersection_input = self._get_intersection_input_frame(
                        rgb_frame,
                        semantic_frame,
                    )
                    self.is_intersection = False
                    intersection_interval = float(
                        cfg("INTERSECTION_CHECK_INTERVAL_SECONDS", 0.25)
                    )
                    now = time.perf_counter()
                    if (
                        not self.intersection_sequence
                        and intersection_input is not None
                        and now - self._last_intersection_check_at
                        >= intersection_interval
                    ):
                        try:
                            self.is_intersection = bool(
                                self.intersection_model.is_intersection_ahead(
                                    intersection_input
                                )
                            )
                        except Exception:
                            self.is_intersection = False
                        self._last_intersection_check_at = now

                    if self.is_intersection and cfg("JUNCTION_CONTROL_MODE", "trajectory") == "static_meters":
                        self.intersection_sequence = True
                        self.seq_timeout = time.time()
                        self.intersec_once = True
                    elif self.is_intersection and cfg("JUNCTION_CONTROL_MODE", "trajectory") == "trajectory":
                        self.intersection_sequence = True
                        self.seq_timeout = time.time()
                        self.intersec_once = False
                        if self.planner is not None and self.current_location is not None:
                            try:
                                maneuver = self.planner.get_next_maneuver_text(
                                    self.current_location, self.goal_location
                                )
                                if maneuver in getattr(conf, "COMMANDS", ()):
                                    self.next_maneuver = maneuver
                                    self.trajectory_steering_agent.command_name = maneuver
                            except Exception:
                                pass

                    try:
                        candidate_out_checker, screen = self.auto_driver.update(
                            rgb_frame=rgb_frame,
                            vision_result=vision_result,
                            current_location=self.current_location,
                            goal_location=self.goal_location,
                            intersection_sequence=self.intersection_sequence,
                        )
                    except TypeError:
                        candidate_out_checker, screen = self.auto_driver.update(
                            rgb_frame,
                            vision_result,
                            self.current_location,
                            self.goal_location,
                            self.intersection_sequence,
                        )
                    except Exception:
                        candidate_out_checker = False
                        screen = rgb_frame.copy() if rgb_frame is not None else blank_frame()

                    self._update_out_checker_logic(
                        candidate_out_checker=candidate_out_checker,
                        drivable_area_result=drivable_area_result,
                    )

                    traj_visualized_scrn = draw_waypoints(screen, pred) if cfg("DEBUG_SHOW_TRAJECTORY", True) else screen
                    self._show_frame(traj_visualized_scrn)

                else:
                    screen = rgb_frame.copy() if rgb_frame is not None else blank_frame()
                    self._render_mode_overlay(screen, "MANUAL", 0.0)

                    if manual_control is None:
                        manual_control = make_stop_control()

                    self._safe_vehicle_apply_control(manual_control)

                    traj_visualized_scrn = draw_waypoints(screen, pred) if cfg("DEBUG_SHOW_TRAJECTORY", True) else screen
                    self._show_frame(traj_visualized_scrn)

                time.sleep(0.01)

        finally:
            self.shutdown()


def main() -> None:
    qt_app = QApplication.instance()
    if qt_app is None:
        qt_app = QApplication([])

    app = CarlaLaneDrivingApp()
    app.run()


if __name__ == "__main__":
    main()
