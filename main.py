import threading
import time
from typing import Any, Optional, Tuple

import carla
import cv2
import numpy as np

import config_city as conf
from carla_manager import CarlaManager
from controllers.controller import FixedSpeedPIDController
from controllers.input_manager import InputManager
from sensors.camera_manager import CameraManager
from stream import start_stream
from vision.city_vision_processing import VisionProcessor
from vision.color_extractor import HSVColorThresholdExtractor
from navigation.intersection_detector import IntersectionModel
from navigation.global_planner import RoutePlanner
from navigation.navigate import move_vehicle_for_distance
from navigation.lane_side import LaneSideModel


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def cfg(name: str, default: Any) -> Any:
    return getattr(conf, name, default)


def overlay_text(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    scale: float = 0.7,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> None:
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def make_stop_control() -> carla.VehicleControl:
    return carla.VehicleControl(
        throttle=0.0,
        steer=0.0,
        brake=1.0,
        reverse=False,
        hand_brake=False,
        manual_gear_shift=False,
    )


def blank_frame() -> np.ndarray:
    return np.zeros(
        (
            cfg("CAMERA_IMAGE_HEIGHT", 720),
            cfg("CAMERA_IMAGE_WIDTH", 1280),
            3,
        ),
        dtype=np.uint8,
    )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class CarlaLaneDrivingApp:
    """
    Main CARLA app for lane following and manual/auto mode switching.

    Auto mode:
      - uses semantic frame only
      - extracts green + dark purple lane markings
      - computes lane center from the extracted line masks

    Manual mode:
      - uses RGB frame for display
      - applies user input control

    Kept:
      - camera
      - HSV color extraction
      - lane vision processing
      - PID steering controller
      - input manager
      - streaming thread
    """

    def __init__(self) -> None:
        self.carla_manager = CarlaManager()
        self.world = None
        self.vehicle = None

        self.camera_manager: Optional[CameraManager] = None
        self.vision_processor: Optional[VisionProcessor] = None
        self.controller: Optional[FixedSpeedPIDController] = None
        self.input_manager: Optional[InputManager] = None

        self.auto_mode = bool(cfg("AUTO_MODE_DEFAULT", True))
        self.running = True

        self.turning_intersection = False
        self.intersection_model = IntersectionModel(
            "models_and_datasets/models/junction_model_resnet18.pt"
        )

        self.lane_change_debounce_seconds = float(cfg("LANE_CHANGE_DEBOUNCE_SECONDS", 1.5))
        self.lane_change_debounce_until = 0.0

        self.movement_thread: Optional[threading.Thread] = None
        self.next_maneuver: Optional[str] = None
        self.is_intersection = False
        self.lane_side = LaneSideModel("models_and_datasets/models/lane_side_model_resnet18.pt")

    def _movement_active(self) -> bool:
        return self.movement_thread is not None and self.movement_thread.is_alive()

    def _get_latest_rgb_frame(self) -> Optional[np.ndarray]:
        if self.camera_manager is None:
            return None
        return self.camera_manager.get_latest_rgb()

    def _show_frame(self, screen: np.ndarray) -> None:
        conf.debug_frame_buffer = screen
        if cfg("SHOW_OPENCV_WINDOW", False):
            cv2.imshow("CARLA", cv2.resize(screen, (1280, 720)))
            cv2.waitKey(1)

    def _get_lane_state(self, rgb_frame: Optional[np.ndarray]) -> dict:
        """
        Read all lane-side checks from the same frame so the decision is consistent.
        """
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

    def _should_change_lane(self, direction: str, lane_state: dict) -> bool:
        """
        Decide whether a lane change should happen at all.
        """
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
        """
        Lane change flow:
        1) small settling move
        2) read latest RGB frame
        3) check lane-side state
        4) only then perform the actual lane shift
        """
        if self.vehicle is None:
            return

        # Small move before checking lane-side state.
        move_vehicle_for_distance(
            self.vehicle,
            1.0,
            0.0,
            True,
            0.10,
            8.0,
            blocking=True,
        )

        time.sleep(0.15)

        rgb_frame = self._get_latest_rgb_frame()
        lane_state = self._get_lane_state(rgb_frame)

        if not self._should_change_lane(direction, lane_state):
            return

        # Actual lane shift.
        if direction == "left":
            move_vehicle_for_distance(
                self.vehicle,
                3.5,
                -0.2,
                True,
                0.08,
                20.0,
                blocking=True,
            )

        elif direction == "right":
            move_vehicle_for_distance(
                self.vehicle,
                3.5,
                0.2,
                True,
                0.08,
                20.0,
                blocking=True,
            )

    def _start_movement_sequence(self, segments) -> None:
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

    def _start_verified_lane_change(self, direction: str) -> None:
        """
        Start a lane change in a worker thread, but only after:
        - a short move
        - a fresh lane-side check
        - the actual lane shift
        """
        if self.vehicle is None or self._movement_active():
            return

        def worker() -> None:
            try:
                self._perform_verified_lane_change(direction)
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

    def setup(self) -> None:
        self.world = self.carla_manager.connect()
        self.vehicle = self.carla_manager.spawn_vehicle()

        self.camera_manager = CameraManager(self.world, self.vehicle)
        self.camera_manager.start()

        HSVColorThresholdExtractor(
            morph_kernel_size=getattr(conf, "MORPH_KERNEL_SIZE", 1)
        )

        self.planner = RoutePlanner(self.world)
        self.goal_location = carla.Location(x=-114.91, y=-44.27, z=0)
        self.current_location = self.vehicle.get_location()

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
        overlay_text(screen, f"MODE: {mode}", (20, 30), 0.9, (255, 255, 0), 2)
        overlay_text(screen, f"error: {error:.2f}", (20, 60), 0.7, (255, 255, 255), 2)

    def _handle_lane_change_requests(self) -> None:
        # Lane change requests are checked only when debounce allows it.
        if time.time() < self.lane_change_debounce_until:
            return

        next_maneuver_text = self.planner.get_next_maneuver_text(
            self.current_location,
            self.goal_location,
        )

        if next_maneuver_text == "CHANGE_LANE_LEFT":
            self._start_verified_lane_change("left")
            self.lane_change_debounce_until = time.time() + self.lane_change_debounce_seconds

        if next_maneuver_text == "CHANGE_LANE_RIGHT":
            self._start_verified_lane_change("right")
            self.lane_change_debounce_until = time.time() + self.lane_change_debounce_seconds

    def _handle_intersection_requests(self) -> None:
        if not self.is_intersection:
            return

        dist_next_maneuver = self.planner.distance_to_next_maneuver(
            self.current_location,
            self.goal_location,
        )

        if dist_next_maneuver is None:
            self.next_maneuver = None
            return

        if dist_next_maneuver > 6:
            return

        self.next_maneuver = self.planner.get_next_maneuver_text(
            self.current_location,
            self.goal_location,
        )

        if self.next_maneuver == "RIGHT":
            self._start_movement_sequence([
                (8, 0.05, True, 0.2, 20.0),
                (7, 0.1, True, 0.2, 20.0),
                (6.9, 0.4, True, 0.2, 20.0),
            ])

        if self.next_maneuver == "LEFT":
            self._start_movement_sequence([
                (19.5, 0, True, 0.2, 20.0),
                (17.5, -0.23, True, 0.2, 20.0),
                (4.5, 0, True, 0.2, 20.0),
            ])

        if self.next_maneuver == "STRAIGHT":
            self._start_movement_sequence([
                (30, 0.0, True, 0.2, 20.0),
            ])

    def _run_auto_mode(self, rgb_frame: Optional[np.ndarray], semantic_frame: Optional[np.ndarray]) -> np.ndarray:
        # 1) Detect whether there is an intersection ahead.
        self.is_intersection = self.intersection_model.is_intersection_ahead(rgb_frame)

        # 2) If the semantic frame is missing, stop and keep showing RGB.
        if semantic_frame is None:
            screen = rgb_frame.copy() if rgb_frame is not None else blank_frame()
            if self.vehicle is not None:
                self.vehicle.apply_control(make_stop_control())
            return screen

        # 3) Run vision on the semantic frame and get steering error.
        result = self.vision_processor.detect(semantic_frame, rgb_frame)
        error = float(result.get("error", 0.0))
        debug = result.get("debug", {})
        cv2.imshow("drivable area", debug.get("drivable_mask", blank_frame()))

        screen = debug.get("combined", semantic_frame).copy()

        # 4) Update PID steering control.
        control = self.controller.update(error=error)
        if self.vehicle is not None:
            self.vehicle.apply_control(control)

        # 5) Update navigation state from the current vehicle position.
        self.current_location = self.vehicle.get_location()
        self.vehicle_wp = self.world.get_map().get_waypoint(self.vehicle.get_location())

        # 6) Lane changes are handled separately from intersection turns.
        self._handle_lane_change_requests()

        # 7) Intersection maneuver logic.
        self._handle_intersection_requests()

        return screen

    def run(self) -> None:
        self.setup()
        try:
            while self.running:
                if self.input_manager is None or self.controller is None or self.vision_processor is None:
                    break

                self.running, toggle_auto, manual_control = self.input_manager.poll()

                if toggle_auto:
                    self.auto_mode = not self.auto_mode
                    self.controller.reset()
                    if self.vehicle is not None:
                        self.vehicle.apply_control(make_stop_control())

                rgb_frame = self.camera_manager.get_latest_rgb() if self.camera_manager is not None else None
                semantic_frame = self.camera_manager.get_latest_semantic() if self.camera_manager is not None else None

                if self.movement_thread is not None and not self.movement_thread.is_alive():
                    self.movement_thread = None

                movement_active = self._movement_active()
                self.turning_intersection = movement_active

                if rgb_frame is None and semantic_frame is None:
                    screen = blank_frame()
                    self._show_frame(screen)
                    time.sleep(0.01)
                    continue

                if movement_active:
                    screen = rgb_frame.copy() if rgb_frame is not None else blank_frame()
                    self._show_frame(screen)
                    time.sleep(0.01)
                    continue

                if self.auto_mode:
                    screen = self._run_auto_mode(rgb_frame, semantic_frame)
                    self._show_frame(screen)

                else:
                    # Manual mode: show RGB frame and apply user input directly.
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