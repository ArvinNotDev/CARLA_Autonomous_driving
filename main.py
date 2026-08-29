from __future__ import annotations

import threading
import time
from typing import Any, Optional

import carla
import numpy as np
from PySide6.QtCore import QMetaObject, Qt
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
from sensors.camera_manager import CameraManager
from stream import start_stream
from ui.renderer import Renderer
from ui.spawn_goal_picker import SpawnGoalPicker
from utils.vehicle_utils import blank_frame, make_stop_control
from vision.city_vision_processing import VisionProcessor
from vision.drivable_area_debugger import DrivableAreaDebugger
from trajectory.steering_agent import TrajectorySteeringAgent
from trajectory.visualize import draw_waypoints
from trajectory.worker import InferenceWorker
from runtime_metrics import ControlMetrics, DisplayMetrics
from runtime_settings import RuntimeSettings
from ui.control_panel import ControlPanel
from ui.frame_store import DebugFrameStore


def cfg(name: str, default: Any) -> Any:
    return getattr(conf, name, default)


class CarlaLaneDrivingApp:
    """
    CARLA control runtime.

    Qt's main thread is reserved for the PySide6 event loop. Vehicle control
    runs on one fixed-cadence worker thread. All ML inference is serialized in
    one dedicated inference worker, and debug frames are transferred through a
    latest-frame store consumed by a Qt timer.
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
        self.drivable_debugger = DrivableAreaDebugger()

        self.auto_driver: Optional[AutoDriver] = None
        self.lane_change_manager: Optional[LaneChangeManager] = None
        self.intersection_manager: Optional[IntersectionManager] = None
        self.intersection_model = None
        self.trajectory_steering_agent: Optional[TrajectorySteeringAgent] = None
        self.inference_worker: Optional[InferenceWorker] = None

        self.auto_mode = bool(cfg("AUTO_MODE_DEFAULT", True))
        self.running = True
        self._stop_event = threading.Event()
        self._control_thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()

        self.intersection_sequence = False
        self.is_intersection = False
        self.seq_timeout: float = 0.0
        self.trajectory_takeover_until = 0.0
        self.next_maneuver: Optional[str] = None

        self.planner: Optional[RoutePlanner] = None
        self.goal_location = None
        self.current_location = None

        self._last_trajectory_submit_at = 0.0
        self._last_trajectory_frame_id = -1
        self._last_vision_submit_at = 0.0
        self._last_vision_frame_id = -1
        self._last_intersection_submit_at = 0.0
        self._last_intersection_frame_id = -1

        self._accepted_trajectory_request = 0
        self._accepted_vision_request = 0
        self._accepted_intersection_request = 0
        self._last_traj_command = "LANE_FOLLOW"

        self._vision_result = None
        self._intersection_result = False
        self._intersection_result_completed_at = 0.0
        self._intersection_result_source_frame_id = -1

        self.control_metrics = ControlMetrics(window_seconds=1.0)
        self.display_metrics = DisplayMetrics(window_seconds=1.0)
        self.frame_store = DebugFrameStore()

        self._startup_drive_until = 0.0
        self._last_recovery_notice_at = 0.0
        self._out_checker = False
        self._out_checker_started_at: Optional[float] = None
        self._last_drivable_info: dict[str, Any] = {}

        self.settings = RuntimeSettings()
        self.control_panel: Optional[ControlPanel] = None

        # These models are created before the inference worker, but after this
        # point each model is accessed by only one execution thread.
        self.intersection_model = IntersectionModel(
            "models_and_datasets/models/junction_model_resnet18.pt"
        )

    def _metrics_for_panel(self) -> dict[str, Any]:
        control = self.control_metrics.snapshot()
        data: dict[str, Any] = {
            "control_fps": control.fps,
            "trajectory_fps": 0.0,
            "vision_fps": 0.0,
            "trajectory_age_s": None,
            "trajectory_busy": False,
            "vision_busy": False,
            "junction_phase": (
                self.intersection_manager.phase_name()
                if self.intersection_manager is not None
                else "IDLE"
            ),
        }

        if self.inference_worker is not None:
            traj, traj_busy, _, _ = self.inference_worker.trajectory_snapshot()
            _, vis_busy, _, _ = self.inference_worker.vision_snapshot()
            traj_metrics = self.inference_worker.metrics["trajectory"].snapshot()
            vis_metrics = self.inference_worker.metrics["vision"].snapshot()
            data["trajectory_fps"] = traj_metrics.fps
            data["vision_fps"] = vis_metrics.fps
            data["trajectory_busy"] = traj_busy
            data["vision_busy"] = vis_busy
            if traj is not None and traj.valid:
                data["trajectory_age_s"] = max(
                    0.0, time.perf_counter() - traj.completed_at
                )
        return data

    def _submit_inference(self, rgb_packet, semantic_packet, now: float) -> None:
        if self.inference_worker is None or rgb_packet is None:
            return

        frame_id = int(rgb_packet.frame_id)

        vision_interval = max(
            0.03, float(cfg("VISION_INFERENCE_INTERVAL_SECONDS", 0.05))
        )
        if frame_id != self._last_vision_frame_id and (
            now - self._last_vision_submit_at >= vision_interval
        ):
            semantic_frame = (
                semantic_packet.bgr
                if semantic_packet is not None
                else None
            )
            request_id = self.inference_worker.submit_vision(
                source_frame_id=frame_id,
                semantic_frame=semantic_frame,
                rgb_frame=rgb_packet.bgr,
            )
            if request_id:
                self._last_vision_submit_at = now
                self._last_vision_frame_id = frame_id

        command = self.next_maneuver if self.intersection_sequence else "LANE_FOLLOW"
        if command not in getattr(conf, "COMMANDS", ("LANE_FOLLOW",)):
            command = "LANE_FOLLOW"
        self._last_traj_command = command

        trajectory_interval = max(
            0.03, float(cfg("TRAJECTORY_INFERENCE_INTERVAL_SECONDS", 0.10))
        )
        if frame_id != self._last_trajectory_frame_id and (
            now - self._last_trajectory_submit_at >= trajectory_interval
        ):
            request_id = self.inference_worker.submit_trajectory(
                source_frame_id=frame_id,
                bgr_frame=rgb_packet.bgr,
                command=command,
            )
            if request_id:
                self._last_trajectory_submit_at = now
                self._last_trajectory_frame_id = frame_id

        intersection_interval = max(
            0.05, float(cfg("INTERSECTION_CHECK_INTERVAL_SECONDS", 0.25))
        )
        if frame_id != self._last_intersection_frame_id and (
            now - self._last_intersection_submit_at >= intersection_interval
        ):
            request_id = self.inference_worker.submit_intersection(
                source_frame_id=frame_id,
                frame_bgr=rgb_packet.bgr,
            )
            if request_id:
                self._last_intersection_submit_at = now
                self._last_intersection_frame_id = frame_id

    def _accept_inference_results(self) -> tuple[Any, Any]:
        trajectory = None
        if self.inference_worker is not None:
            traj, _, _, _ = self.inference_worker.trajectory_snapshot()
            if (
                traj is not None
                and traj.valid
                and traj.request_id > self._accepted_trajectory_request
            ):
                # Immutable result object; no worker can mutate this after publish.
                self._accepted_trajectory_request = traj.request_id
                trajectory = traj

            vision, _, _, _ = self.inference_worker.vision_snapshot()
            if (
                vision is not None
                and vision.valid
                and vision.request_id > self._accepted_vision_request
            ):
                self._accepted_vision_request = vision.request_id
                self._vision_result = vision.result

            inter, _, _, _ = self.inference_worker.intersection_snapshot()
            if (
                inter is not None
                and inter.valid
                and inter.request_id > self._accepted_intersection_request
            ):
                self._accepted_intersection_request = inter.request_id
                self._intersection_result = bool(inter.is_intersection)
                self._intersection_result_completed_at = inter.completed_at
                self._intersection_result_source_frame_id = inter.source_frame_id

        return trajectory, self._vision_result

    def _force_left_correction(self) -> None:
        control = carla.VehicleControl(
            throttle=float(cfg("DRIVABLE_RECOVERY_THROTTLE", cfg("FIXED_THROTTLE", 0.3))),
            steer=-abs(float(cfg("DRIVABLE_RECOVERY_STEER", cfg("STEER_LIMIT", 0.35)))),
            brake=0.0,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
        )
        self._safe_vehicle_apply_control(control)

    @staticmethod
    def _extract_drivable_error(drive_info: Any) -> float:
        if isinstance(drive_info, dict):
            try:
                return float(drive_info.get("error", 0.0) or 0.0)
            except (TypeError, ValueError):
                pass
        return 0.0

    def _update_out_checker_logic(self, candidate_out_checker: bool, drive_info: Any) -> None:
        """Exact trajectory-branch recovery behavior.

        A valid AutoDriver result starts a ten-second checking window. During
        that window, a positive drivable-area error forces the same left
        correction used by the trajectory branch.
        """
        if not bool(cfg("DRIVABLE_RECOVERY_ENABLED", True)):
            self._out_checker = False
            self._out_checker_started_at = None
            return

        error = self._extract_drivable_error(drive_info)
        now = time.monotonic()
        if self._out_checker_started_at is None:
            if candidate_out_checker:
                self._out_checker_started_at = now
                self._out_checker = True
            else:
                self._out_checker = False
            return

        self._out_checker = True
        if error > float(cfg("DRIVABLE_RECOVERY_ERROR_THRESHOLD", 20.0)):
            self._force_left_correction()

        if now - self._out_checker_started_at >= float(
            cfg("DRIVABLE_RECOVERY_WINDOW_SECONDS", 10.0)
        ):
            self._out_checker = False
            self._out_checker_started_at = None

    def _trajectory_for_control(self, current_command: str):
        if self.inference_worker is None:
            return None
        result, _, _, _ = self.inference_worker.trajectory_snapshot()
        if result is None or not result.valid:
            return None
        if result.command != current_command:
            return None
        return result

    def _publish_debug_frame(self, screen: np.ndarray) -> None:
        if screen is None:
            return

        lines = []
        if cfg("DEBUG_SHOW_FPS", True):
            lines.append(
                f"Display FPS: {self.display_metrics.fps():.1f} | "
                f"Control FPS: {self.control_metrics.snapshot().fps:.1f}"
            )

        if self.inference_worker is not None:
            traj, traj_busy, traj_error, _ = self.inference_worker.trajectory_snapshot()
            _, vis_busy, vis_error, _ = self.inference_worker.vision_snapshot()
            traj_metrics = self.inference_worker.metrics["trajectory"].snapshot()
            vis_metrics = self.inference_worker.metrics["vision"].snapshot()

            age = (
                max(0.0, time.perf_counter() - traj.completed_at)
                if traj is not None and traj.valid
                else None
            )
            lines.extend(
                [
                    f"Trajectory FPS: {traj_metrics.fps:.1f} | age: {'—' if age is None else f'{age:.2f}s'}",
                    f"Vision FPS: {vis_metrics.fps:.1f}",
                    f"Junction: {self.intersection_manager.phase_name() if self.intersection_manager else 'IDLE'}",
                    f"Workers: trajectory={'busy' if traj_busy else 'idle'} vision={'busy' if vis_busy else 'idle'}",
                ]
            )
            if traj_error:
                lines.append("Trajectory worker: retaining last valid result")
            if vis_error:
                lines.append("Vision worker: retaining last valid result")

        if cfg("DEBUG_SHOW_GPS", True) and self.vehicle is not None:
            try:
                loc = self.current_location
                if loc is None:
                    loc = self.vehicle.get_location()
                lines.append(f"GPS: {loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f}")
            except Exception:
                pass

        drive_info = self._last_drivable_info
        if drive_info.get("error") is not None:
            lines.append(f"Drivable recovery error: {float(drive_info['error']):.1f}px")

        if self.intersection_manager is not None:
            phase = self.intersection_manager.phase_name()
        else:
            phase = "IDLE"
        if phase == "STATIC":
            lines.append("Junction phase: static lead-in")
        elif self.intersection_sequence:
            lines.append(
                f"Junction phase: trajectory ({self.next_maneuver or 'LANE_FOLLOW'})"
            )

        composed = self.renderer.compose(screen, {"lines": lines})
        self.frame_store.publish(composed)

    def _on_settings_changed(self, values: dict) -> None:
        # All settings are applied by RuntimeSettings in the Qt thread. The
        # control loop only reads scalar config values at its next tick.
        if "AUTO_MODE_DEFAULT" in values:
            with self._state_lock:
                self.auto_mode = bool(values["AUTO_MODE_DEFAULT"])

    def _request_car_reset(self) -> None:
        if self.control_panel is None:
            return
        QMessageBox.information(
            self.control_panel,
            "Car reset required",
            "This setting changes a CARLA sensor/model resource. Restart the run to apply it.",
        )

    def _safe_vehicle_apply_control(self, control: carla.VehicleControl) -> None:
        if self.vehicle is None or control is None:
            return
        try:
            self.vehicle.apply_control(control)
        except Exception:
            pass

    def _apply_goal_to_planner(self) -> None:
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

        self.planner = RoutePlanner(self.world)
        self._apply_goal_to_planner()

        self.current_location = self.vehicle.get_location()
        self.vision_processor = VisionProcessor(str(cfg("VISION_MODE", "onnx")))

        self.controller = FixedSpeedPIDController(
            fixed_throttle=float(cfg("FIXED_THROTTLE", 0.13)),
            kp=float(cfg("KP", 0.015)),
            ki=float(cfg("KI", 0.0)),
            kd=float(cfg("KD", 0.002)),
            steer_limit=float(cfg("STEER_LIMIT", 0.75)),
            max_steer_step=float(cfg("MAX_STEER_STEP", 0.10)),
        )

        self.input_manager = InputManager()

        self.lane_change_manager = LaneChangeManager(
            vehicle=self.vehicle,
            planner=self.planner,
            lane_change_debounce_seconds=float(
                cfg("LANE_CHANGE_DEBOUNCE_SECONDS", 2.0)
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
            intersection_model=None,
            lane_change_manager=self.lane_change_manager,
            intersection_manager=self.intersection_manager,
        )

        self.trajectory_steering_agent = TrajectorySteeringAgent(
            "models_and_datasets/models/trajectory_modelV2.pt",
            conf,
            "LANE_FOLLOW",
        )

        self.inference_worker = InferenceWorker(
            vision_processor=self.vision_processor,
            trajectory_agent=self.trajectory_steering_agent,
            intersection_model=self.intersection_model,
        )
        self.inference_worker.start()

        self._startup_drive_until = time.monotonic() + 3.0

        self.control_panel = ControlPanel(
            self.settings,
            display_metrics=self.display_metrics,
            frame_store=self.frame_store,
            metrics_provider=self._metrics_for_panel,
        )
        self.control_panel.settings_changed.connect(self._on_settings_changed)
        self.control_panel.reset_requested.connect(self._request_car_reset)
        self.control_panel.show()

        # No sleep here: stream startup must not stall the Qt UI.
        try:
            threading.Thread(
                target=start_stream,
                kwargs={"host": conf.STREAM_HOST, "port": conf.STREAM_PORT},
                name="carla-debug-stream",
                daemon=True,
            ).start()
        except Exception:
            pass

    def _control_loop(self) -> None:
        configured_hz = float(cfg("CONTROL_LOOP_HZ", 30.0))
        period = 1.0 / max(10.0, min(60.0, configured_hz))
        next_tick = time.perf_counter()

        while self.running and not self._stop_event.is_set():
            tick_started = time.perf_counter()
            try:
                self._control_iteration(tick_started)
            except Exception:
                try:
                    self._safe_vehicle_apply_control(make_stop_control())
                except Exception:
                    pass

            duration = time.perf_counter() - tick_started
            self.control_metrics.record(duration, time.perf_counter())

            next_tick += period
            now = time.perf_counter()
            if next_tick < now:
                next_tick = now
            self._stop_event.wait(max(0.0, next_tick - now))

        app = QApplication.instance()
        if app is not None:
            try:
                QMetaObject.invokeMethod(
                    app,
                    "quit",
                    Qt.ConnectionType.QueuedConnection,
                )
            except Exception:
                pass

    def _control_iteration(self, now: float) -> None:
        if self.input_manager is None or self.camera_manager is None:
            return

        running, toggle_auto, manual_control = self.input_manager.poll()
        self.running = bool(running)
        if toggle_auto:
            with self._state_lock:
                self.auto_mode = not self.auto_mode
            if self.controller is not None:
                self.controller.reset()

        with self._state_lock:
            auto_mode = bool(self.auto_mode)

        rgb_packet, semantic_packet, _ = self.camera_manager.snapshot()

        if rgb_packet is None:
            control = (
                carla.VehicleControl(throttle=0.18)
                if auto_mode and time.monotonic() < self._startup_drive_until
                else (manual_control if not auto_mode else make_stop_control())
            )
            self._safe_vehicle_apply_control(control)
            self._publish_debug_frame(blank_frame())
            return

        # Manual driving must not wait for vision, trajectory, or junction
        # inference. Those workers can be busy when E is pressed, which used
        # to make keyboard control and the debug panel appear frozen.
        if not auto_mode:
            control = manual_control or make_stop_control()
            screen = rgb_packet.bgr.copy()
            self._render_mode_overlay(screen, "MANUAL", 0.0)
            self._safe_vehicle_apply_control(control)
            self._publish_debug_frame(screen)
            return

        self._submit_inference(rgb_packet, semantic_packet, now)
        _, vision_result = self._accept_inference_results()
        try:
            drive_info = self.drivable_debugger.show(vision_result)
        except Exception:
            drive_info = {}
        self._last_drivable_info = drive_info if isinstance(drive_info, dict) else {}
        if isinstance(vision_result, dict):
            vision_result.setdefault("debug", {})["drivable_info"] = self._last_drivable_info

        try:
            self.current_location = self.vehicle.get_location()
        except Exception:
            self.current_location = None

        command_changed = False
        if self.intersection_manager is not None:
            phase_before = self.intersection_manager.phase_name()
            self.intersection_manager.update(self.current_location)

            if phase_before == "STATIC" and self.intersection_manager.phase_name() == "IDLE":
                maneuver = self.intersection_manager.next_maneuver
                if maneuver in getattr(conf, "COMMANDS", ()):
                    self.next_maneuver = maneuver
                    self.intersection_sequence = True
                    self.seq_timeout = time.monotonic()
                    self.trajectory_takeover_until = (
                        self.seq_timeout
                        + float(cfg("JUNCTION_TRAJECTORY_WINDOW_SECONDS", 6.0))
                    )
                    command_changed = True

        if command_changed and self.inference_worker is not None:
            # Do not wait for the normal inference interval after a junction
            # command transition. The new command receives the newest frame now.
            self._last_trajectory_frame_id = -1
            self._last_trajectory_submit_at = 0.0
            self._submit_inference(rgb_packet, semantic_packet, now)

        # Accept a fresh intersection classifier result without making it part
        # of the control-critical inference path.
        if (
            self._intersection_result
            and time.perf_counter() - self._intersection_result_completed_at
            <= float(cfg("INTERSECTION_RESULT_MAX_AGE_SECONDS", 0.75))
            and not self.intersection_sequence
            and self.intersection_manager is not None
        ):
            started = self.intersection_manager.start_for_intersection(
                self.current_location,
                self.goal_location,
            )
            if started:
                self._intersection_result = False

        current_command = (
            self.next_maneuver if self.intersection_sequence else "LANE_FOLLOW"
        )
        if current_command not in getattr(conf, "COMMANDS", ()):
            current_command = "LANE_FOLLOW"

        matching_trajectory = self._trajectory_for_control(current_command)

        control: carla.VehicleControl
        screen = (
            vision_result.get("debug", {}).get("combined")
            if isinstance(vision_result, dict)
            else None
        )
        if not isinstance(screen, np.ndarray):
            screen = rgb_packet.bgr

        phase = (
            self.intersection_manager.phase_name()
            if self.intersection_manager is not None
            else "IDLE"
        )
        candidate_out_checker = False

        if auto_mode:
            # Static junction lead-in always owns the control command.
            static_control = (
                self.intersection_manager.static_control(self.current_location)
                if self.intersection_manager is not None
                and phase in {"CROSSWALK", "STATIC"}
                else None
            )
            if static_control is not None:
                control = static_control
                self.intersection_sequence = False
            elif (
                self.intersection_sequence
                and time.monotonic() < self.trajectory_takeover_until
            ):
                speed_kmh = 0.0
                try:
                    vel = self.vehicle.get_velocity()
                    speed_kmh = (
                        float((vel.x**2 + vel.y**2 + vel.z**2) ** 0.5) * 3.6
                    )
                except Exception:
                    pass

                if matching_trajectory is not None and matching_trajectory.steer is not None:
                    control = carla.VehicleControl(
                        throttle=0.15,
                        steer=float(matching_trajectory.steer),
                        brake=0.0,
                        hand_brake=False,
                        reverse=False,
                        manual_gear_shift=False,
                    )
                    candidate_out_checker = vision_result is not None
                else:
                    # The last valid trajectory remains visible, but while a new
                    # command has no matching result, use the stable lane controller
                    # rather than mixing commands from different requests.
                    candidate_out_checker, screen, control = self.auto_driver.update(
                        rgb_packet.bgr,
                        vision_result,
                        self.current_location,
                        self.goal_location,
                        False,
                    )
            else:
                if self.intersection_sequence:
                    self.intersection_sequence = False
                    self.next_maneuver = None
                    self.trajectory_takeover_until = 0.0
                candidate_out_checker, screen, control = self.auto_driver.update(
                    rgb_packet.bgr,
                    vision_result,
                    self.current_location,
                    self.goal_location,
                    False,
                )

        else:
            self._render_mode_overlay(screen, "MANUAL", 0.0)
            control = manual_control or make_stop_control()

        self._safe_vehicle_apply_control(control)
        if auto_mode and static_control is None:
            self._update_out_checker_logic(
                candidate_out_checker=candidate_out_checker,
                drive_info=self._last_drivable_info,
            )

        if (
            cfg("DEBUG_SHOW_TRAJECTORY", True)
            and self.inference_worker is not None
        ):
            traj_snapshot, _, _, _ = self.inference_worker.trajectory_snapshot()
            if traj_snapshot is not None and traj_snapshot.valid and traj_snapshot.prediction is not None:
                # Retain the last valid prediction during worker busy/error states.
                screen = draw_waypoints(
                    screen,
                    traj_snapshot.prediction,
                    scale=float(cfg("TRAJECTORY_DEBUG_SCALE", 2.2)),
                )

        self._publish_debug_frame(screen)

    def _render_mode_overlay(self, screen: np.ndarray, mode: str, error: float) -> None:
        self.renderer.render_mode_overlay(screen, mode, error)

    def shutdown(self) -> None:
        self.running = False
        self._stop_event.set()

        if self._control_thread is not None and self._control_thread.is_alive():
            self._control_thread.join(timeout=2.0)

        if self.inference_worker is not None:
            self.inference_worker.stop()
            self.inference_worker = None

        if self.intersection_manager is not None:
            try:
                self.intersection_manager.stop()
            except Exception:
                pass

        if self.control_panel is not None:
            try:
                self.control_panel.close()
            except Exception:
                pass

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


    def run(self) -> None:
        self.setup()
        if not self.running:
            self.shutdown()
            return

        self._control_thread = threading.Thread(
            target=self._control_loop,
            name="carla-control-loop",
            daemon=True,
        )
        self._control_thread.start()

        qt_app = QApplication.instance()
        if qt_app is not None:
            qt_app.exec()

        self.shutdown()


def main() -> None:
    qt_app = QApplication.instance()
    if qt_app is None:
        qt_app = QApplication([])

    app = CarlaLaneDrivingApp()
    app.run()


if __name__ == "__main__":
    main()
