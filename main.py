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
from navigation.intersection_detector import is_four_way_intersection_ahead, is_intersection_ahead
from navigation.turning_options import get_turn_options, get_intersection_options
from navigation.global_planner import RoutePlanner
from navigation.navigate import move_toward_target
import time
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

    def setup(self) -> None:
        self.world = self.carla_manager.connect()
        self.vehicle = self.carla_manager.spawn_vehicle()

        self.camera_manager = CameraManager(self.world, self.vehicle)
        self.camera_manager.start()

        color_extractor = HSVColorThresholdExtractor(
            morph_kernel_size=getattr(conf, "MORPH_KERNEL_SIZE", 1)
        )


        self.planner = RoutePlanner(self.world)
        self.goal_location = carla.Location(x=89.91, y= -56.27, z=0)
        self.current_location = self.vehicle.get_location()



        self.vision_processor = VisionProcessor(color_extractor=color_extractor)

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

                if self.auto_mode:
                    if semantic_frame is None:
                        screen = rgb_frame.copy() if rgb_frame is not None else np.zeros(
                            (
                                cfg("CAMERA_IMAGE_HEIGHT", 720),
                                cfg("CAMERA_IMAGE_WIDTH", 1280),
                                3,
                            ),
                            dtype=np.uint8,
                        )
                        # self._render_mode_overlay(screen, "AUTO (NO SEMANTIC FRAME)", 0.0)

                        if self.vehicle is not None:
                            self.vehicle.apply_control(make_stop_control())

                    else:





                        

                        cv2.imshow("sem", semantic_frame)
                        result = self.vision_processor.detect(semantic_frame, rgb_frame)
                        error = float(result.get("error", 0.0))
                        debug = result.get("debug", {})

                        screen = debug.get("combined", semantic_frame).copy()
                        # self._render_mode_overlay(screen, "AUTO", error)

                        control = self.controller.update(error=error)
                        if self.vehicle is not None:
                            self.vehicle.apply_control(control)

                        ########################
                        # waypoint informations#
                        ########################
                        self.current_location = self.vehicle.get_location()
                        self.vehicle_wp = self.world.get_map().get_waypoint(self.vehicle.get_location())
                        
                        if is_intersection_ahead(self.vehicle_wp, distance=30):
                            dist_next_maneuver = self.planner.distance_to_next_maneuver(self.current_location, self.goal_location)
                            if dist_next_maneuver is not None:
                                if dist_next_maneuver <= 20:
                                    self.next_maneuver = self.planner.get_next_maneuver_text(self.current_location, self.goal_location)
                                    route = self.planner.get_route(self.current_location,self.goal_location)
                                    target_wp = route[min(10, len(route)-1)][0]
                                    target_location = target_wp.transform.location # after implementing the readin
                            else:
                                self.next_maneuver = None
                        else:
                            self.options = []
                        
                        crosswalk_info = result["is_crosswalk"]
                        if self.crosswalk_timeout == False:
                            if crosswalk_info[0] == True:
                                # target_location = None
                                self.turning_intersection = True
                                self.vehicle.apply_control(
                                    carla.VehicleControl(
                                        throttle=0.0,
                                        steer=0.0,
                                        brake=1.0
                                    )
                                )
                                time.sleep(5)

                        if self.turning_intersection == True:
                            self.turning_intersection = False
                            self.crosswalk_timeout = True
                            self.vehicle.apply_control(
                                carla.VehicleControl(
                                    throttle=0.2,
                                    steer=0.0,
                                    brake=0.0
                                )
                            )

                        
                        #####################################

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