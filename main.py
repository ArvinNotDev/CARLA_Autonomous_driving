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
from navigation.turning_options import get_turn_options, get_intersection_options
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
        self.intersection_model = IntersectionModel("models_and_datasets/models/junction_model_resnet18.pt")
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

    def _get_lane_state(self, rgb_frame: Optional[np.ndarray]) -> dict:
        """
        Reads all lane-side checks together so we can make one decision
        from one frame.
        """
        if rgb_frame is None:
            return {
                "left_lane": False,
                "right_lane": False,
                "out_from_right": False,
                "out_from_left": False,
            }

        left_lane = self.lane_side.is_left_lane(rgb_frame)
        right_lane = self.lane_side.is_right_lane(rgb_frame)
        out_from_right = self.lane_side.is_out_from_right(rgb_frame)
        out_from_left = self.lane_side.is_out_from_left(rgb_frame)

        return {
            "left_lane": left_lane,
            "right_lane": right_lane,
            "out_from_right": out_from_right,
            "out_from_left": out_from_left,
        }

    def _should_change_lane(self, direction: str, lane_state: dict) -> bool:
        """
        Decide whether the requested lane change should actually be executed.

        This is the place where the lane-side model is used to double-check the car
        before we do the final lane change move.
        """
        if direction == "left":
            if lane_state["left_lane"]:
                return False

            # If the car is already drifting too far left, do not keep pushing left.
            if lane_state["out_from_left"]:
                return False

            # Left change is allowed if we are still on the right side or clearly not in left lane.
            return True

        if direction == "right":
            if lane_state["right_lane"]:
                return False

            # If the car is already drifting too far right, do not keep pushing right.
            if lane_state["out_from_right"]:
                return False

            return True

        return False

    def _perform_verified_lane_change(self, direction: str) -> None:
        """
        This is the lane-change flow you asked for.

        Flow:
        1) move the vehicle a little first
        2) read the latest RGB frame
        3) check where the car is using:
             - self.lane_side.is_left_lane(rgb_frame)
             - self.lane_side.is_right_lane(rgb_frame)
             - self.lane_side.is_out_from_right(rgb_frame)
             - self.lane_side.is_out_from_left(rgb_frame)
        4) only then do the actual lane-change movement
        """
        if self.vehicle is None:
            return

        # -------------------------------------------------------------------
        # TUNE HERE:
        # This is the small move before the lane-side verification.
        # Change the distance / steer / throttle / timeout if your camera
        # needs more or less motion before it can correctly detect the lane side.
        # -------------------------------------------------------------------
        move_vehicle_for_distance(
            self.vehicle,
            1.0,          # small settling distance before the lane check
            0.0,          # keep straight here so the frame is easier to verify
            True,
            0.10,         # slow throttle for a clean settle
            8.0,          # timeout for this short move
            blocking=True,
        )

        time.sleep(0.15)

        rgb_frame = self._get_latest_rgb_frame()
        lane_state = self._get_lane_state(rgb_frame)

        # If the frame says we are already in the target lane, skip the lane shift.
        if not self._should_change_lane(direction, lane_state):
            return

        # -------------------------------------------------------------------
        # TUNE HERE:
        # This is the actual lane-shift movement.
        # Adjust steer, throttle, and distance to fit your CARLA vehicle setup.
        # -------------------------------------------------------------------
        if direction == "left":
            move_vehicle_for_distance(
                self.vehicle,
                3.5,       # lane change distance
                -0.2,      # steer left
                True,
                0.08,      # throttle
                20.0,      # timeout
                blocking=True,
            )

        elif direction == "right":
            move_vehicle_for_distance(
                self.vehicle,
                3.5,       # lane change distance
                0.2,       # steer right
                True,
                0.08,      # throttle
                20.0,      # timeout
                blocking=True,
            )

    def _start_movement_sequence(self, segments) -> None:
        if self.vehicle is None or self._movement_active():
            return

        def worker() -> None:
            try:
                for distance_m, steer, forward, throttle, timeout in segments:
                    # -------------------------------------------------------------------
                    # TUNE HERE:
                    # Each tuple below controls one movement block.
                    # distance_m -> how far this block moves
                    # steer       -> steering amount for this block
                    # forward      -> forward / reverse direction flag
                    # throttle    -> throttle value used by move_vehicle_for_distance
                    # timeout     -> max seconds allowed for this block
                    # -------------------------------------------------------------------
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
        Starts a lane change in a worker thread, but only after:
        - a short move
        - a fresh lane-side check
        - then the actual lane shift
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

        color_extractor = HSVColorThresholdExtractor(
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
                    blank = np.zeros(
                        (
                            cfg("CAMERA_IMAGE_HEIGHT", 720),
                            cfg("CAMERA_IMAGE_WIDTH", 1280),
                            3,
                        ),
                        dtype=np.uint8,
                    )
                    conf.debug_frame_buffer = blank

                    if cfg("SHOW_OPENCV_WINDOW", False):
                        cv2.imshow("CARLA", cv2.resize(blank, (1280, 720)))
                        cv2.waitKey(1)

                    time.sleep(0.01)
                    continue

                if movement_active:
                    screen = rgb_frame.copy() if rgb_frame is not None else np.zeros(
                        (
                            cfg("CAMERA_IMAGE_HEIGHT", 720),
                            cfg("CAMERA_IMAGE_WIDTH", 1280),
                            3,
                        ),
                        dtype=np.uint8,
                    )
                    conf.debug_frame_buffer = screen

                    if cfg("SHOW_OPENCV_WINDOW", False):
                        cv2.imshow("CARLA", cv2.resize(screen, (1280, 720)))
                        cv2.waitKey(1)

                    time.sleep(0.01)
                    continue

                if self.auto_mode:
                    self.is_intersection = self.intersection_model.is_intersection_ahead(rgb_frame)

                    if semantic_frame is None:
                        screen = rgb_frame.copy() if rgb_frame is not None else np.zeros(
                            (
                                cfg("CAMERA_IMAGE_HEIGHT", 720),
                                cfg("CAMERA_IMAGE_WIDTH", 1280),
                                3,
                            ),
                            dtype=np.uint8,
                        )

                        if self.vehicle is not None:
                            self.vehicle.apply_control(make_stop_control())

                    else:
                        result = self.vision_processor.detect(semantic_frame, rgb_frame)
                        error = float(result.get("error", 0.0))
                        debug = result.get("debug", {})
                        
                        screen = debug.get("combined", semantic_frame).copy()

                        control = self.controller.update(error=error)
                        if self.vehicle is not None:
                            self.vehicle.apply_control(control)

                        ########################
                        # waypoint informations#
                        ########################
                        self.current_location = self.vehicle.get_location()
                        self.vehicle_wp = self.world.get_map().get_waypoint(self.vehicle.get_location())

                        if time.time() >= self.lane_change_debounce_until:
                            next_maneuver_text = self.planner.get_next_maneuver_text(
                                self.current_location,
                                self.goal_location,
                            )

                            # -------------------------------------------------------------------
                            # Lane change requests:
                            # First do a short move, then check lane-side status, then shift lanes.
                            # -------------------------------------------------------------------
                            if next_maneuver_text == "CHANGE_LANE_LEFT":
                                self._start_verified_lane_change("left")
                                self.lane_change_debounce_until = time.time() + self.lane_change_debounce_seconds

                            if next_maneuver_text == "CHANGE_LANE_RIGHT":
                                self._start_verified_lane_change("right")
                                self.lane_change_debounce_until = time.time() + self.lane_change_debounce_seconds

                        if self.is_intersection:
                            dist_next_maneuver = self.planner.distance_to_next_maneuver(
                                self.current_location,
                                self.goal_location,
                            )
                            if dist_next_maneuver is not None:
                                if dist_next_maneuver <= 6:
                                    self.next_maneuver = self.planner.get_next_maneuver_text(
                                        self.current_location,
                                        self.goal_location,
                                    )

                                    if self.next_maneuver == "RIGHT":
                                        # -------------------------------------------------------------------
                                        # TUNE HERE:
                                        # Intersection right turn sequence.
                                        # Each movement block can be changed independently.
                                        # -------------------------------------------------------------------
                                        self._start_movement_sequence([
                                            (8, 0.05, True, 0.2, 20.0),
                                            (7, 0.1, True, 0.2, 20.0),
                                            (6.9, 0.4, True, 0.2, 20.0),
                                        ])

                                    if self.next_maneuver == "LEFT":
                                        # -------------------------------------------------------------------
                                        # TUNE HERE:
                                        # Intersection left turn sequence.
                                        # Each movement block can be changed independently.
                                        # -------------------------------------------------------------------
                                        self._start_movement_sequence([
                                            (19.5, 0, True, 0.2, 20.0),
                                            (17.5, -0.23, True, 0.2, 20.0),
                                            (4.5, 0, True, 0.2, 20.0),
                                        ])

                                    if self.next_maneuver == "STRAIGHT":
                                        # -------------------------------------------------------------------
                                        # TUNE HERE:
                                        # Straight intersection sequence.
                                        # -------------------------------------------------------------------
                                        self._start_movement_sequence([
                                            (30, 0.0, True, 0.2, 20.0),
                                        ])
                            else:
                                self.next_maneuver = None

                        if self.turning_intersection == True:
                            pass

                    conf.debug_frame_buffer = screen

                    if cfg("SHOW_OPENCV_WINDOW", False):
                        cv2.imshow("CARLA", cv2.resize(screen, (1280, 720)))
                        cv2.waitKey(1)

                else:
                    screen = rgb_frame.copy() if rgb_frame is not None else np.zeros(
                        (
                            cfg("CAMERA_IMAGE_HEIGHT", 720),
                            cfg("CAMERA_IMAGE_WIDTH", 1280),
                            3,
                        ),
                        dtype=np.uint8,
                    )
                    self._render_mode_overlay(screen, "MANUAL", 0.0)

                    if manual_control is None:
                        manual_control = make_stop_control()

                    if self.vehicle is not None:
                        self.vehicle.apply_control(manual_control)

                    conf.debug_frame_buffer = screen

                    if cfg("SHOW_OPENCV_WINDOW", False):
                        cv2.imshow("CARLA", cv2.resize(screen, (1280, 720)))
                        cv2.waitKey(1)

                time.sleep(0.01)

        finally:
            self.shutdown()


def main() -> None:
    app = CarlaLaneDrivingApp()
    app.run()


if __name__ == "__main__":
    main()