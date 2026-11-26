from __future__ import annotations

import argparse
from collections import deque
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Optional


class AvisFpsMonitor:
    """Rolling-window FPS meter for AVIS frame, inference and control rates."""

    def __init__(self, window_seconds: float = 2.0) -> None:
        self.window_seconds = max(0.25, float(window_seconds))
        self._timestamps: deque[float] = deque()

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def record(self, timestamp: float | None = None) -> None:
        sample = time.perf_counter() if timestamp is None else float(timestamp)
        if not math.isfinite(sample):
            return
        if self._timestamps and sample <= self._timestamps[-1]:
            return
        self._timestamps.append(sample)
        self._prune(sample)

    def fps(self, now: float | None = None) -> float:
        current = time.perf_counter() if now is None else float(now)
        if not math.isfinite(current):
            return 0.0
        self._prune(current)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0.0:
            return 0.0
        return float((len(self._timestamps) - 1) / elapsed)


PROJECT_ROOT = Path(__file__).resolve().parent
AVIS_PROFILE_NAME = "AVIS_StableV2"

# Only settings consumed by the AVIS lane-following runtime are copied from
# StableV2. CARLA sensors, map navigation, junction, and LiDAR settings are
# intentionally excluded.
AVIS_PROFILE_FIELDS = (
    "AUTO_MODE_DEFAULT",
    "KP",
    "KI",
    "KD",
    "STEER_LIMIT",
    "MAX_STEER_STEP",
    "CONTROL_LOOP_HZ",
    "VISION_INFERENCE_INTERVAL_SECONDS",
    "LANE_PROB_THRESHOLD",
    "LANE_CENTER_SMOOTH_ALPHA",
    "DRIVABLE_PROB_THRESHOLD",
    "RL_TOP_ROI",
    "RL_BOTTOM_ROI",
    "RL_LEFT_ROI",
    "RL_RIGHT_ROI",
    "LL_TOP_ROI",
    "LL_BOTTOM_ROI",
    "LL_LEFT_ROI",
    "LL_RIGHT_ROI",
    "VISION_DEBUG",
    "DEBUG_SHOW_ROIS",
    "DEBUG_SHOW_LANE_MASK",
    "DEBUG_SHOW_DRIVABLE_AREA",
    "DEBUG_SHOW_TEXT",
    "DEBUG_PANEL_HZ",
    "MODEL_PATH",
    "MODEL_IMAGE_SIZE",
    "ACCELERATION_MPS2",
    "MIN_SPEED_KMH",
    "MAX_SPEED_KMH",
)


def load_avis_profile(path: Path | str | None = None) -> dict[str, Any]:
    profile_path = Path(path) if path is not None else PROJECT_ROOT / "runtime_profiles.json"
    if not profile_path.is_absolute():
        profile_path = PROJECT_ROOT / profile_path

    try:
        profiles = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read runtime profiles: {profile_path}") from exc

    profile = profiles.get(AVIS_PROFILE_NAME) if isinstance(profiles, dict) else None
    if not isinstance(profile, dict):
        raise RuntimeError(
            f"Runtime profile '{AVIS_PROFILE_NAME}' is missing from {profile_path}."
        )

    selected = {key: profile[key] for key in AVIS_PROFILE_FIELDS if key in profile}
    missing = [key for key in AVIS_PROFILE_FIELDS if key not in selected]
    if missing:
        raise RuntimeError(
            f"Profile '{AVIS_PROFILE_NAME}' is incomplete. Missing: {', '.join(missing)}"
        )
    return selected


def apply_avis_profile(profile: dict[str, Any], config_module: ModuleType) -> None:
    for key in AVIS_PROFILE_FIELDS:
        if key in profile:
            setattr(config_module, key, profile[key])


def normalize_avis_controls(throttle: float, steering: float) -> tuple[int, int]:
    """Convert autonomous throttle [0, 1] and steering [-1, 1] to AVIS commands."""
    throttle_value = float(throttle)
    steering_value = float(steering)
    if not math.isfinite(throttle_value) or not math.isfinite(steering_value):
        raise ValueError("AVIS controls must be finite numbers.")
    speed_command = int(round(min(1.0, max(0.0, throttle_value)) * 100.0))
    steering_command = int(round(min(1.0, max(-1.0, steering_value)) * 100.0))
    return speed_command, steering_command


def normalize_avis_manual_controls(
    throttle: float,
    steering: float,
) -> tuple[int, int]:
    """Convert bidirectional manual throttle and steering to AVIS API commands."""
    throttle_value = float(throttle)
    steering_value = float(steering)
    if not math.isfinite(throttle_value) or not math.isfinite(steering_value):
        raise ValueError("AVIS manual controls must be finite numbers.")
    return (
        int(round(min(1.0, max(-1.0, throttle_value)) * 100.0)),
        int(round(min(1.0, max(-1.0, steering_value)) * 100.0)),
    )


def select_avis_target_speed(
    steering: float,
    minimum_kmh: float,
    maximum_kmh: float,
) -> float:
    """Keep StableV2's steering-demand speed reduction for AVIS lane following."""
    minimum = float(minimum_kmh)
    maximum = float(maximum_kmh)
    steering_value = float(steering)
    if not all(math.isfinite(value) for value in (minimum, maximum, steering_value)):
        raise ValueError("Steering and speed settings must be finite.")
    if minimum <= 0.0 or maximum <= minimum:
        raise ValueError("AVIS speed limits must satisfy 0 < minimum < maximum.")

    steering_demand = min(1.0, max(0.0, abs(steering_value)))
    turn_factor = min(1.0, max(0.0, (steering_demand - 0.15) / 0.55))
    return maximum - ((maximum - minimum) * turn_factor)



class AvisSteeringController:
    """StableV2 steering PID/rate-limit logic without CARLA-specific types."""

    def __init__(
        self,
        kp: float = 0.015,
        ki: float = 0.0,
        kd: float = 0.002,
        steer_limit: float = 0.75,
        max_steer_step: float = 0.10,
        integral_limit: float = 3000.0,
    ) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.steer_limit = min(1.0, max(0.0, float(steer_limit)))
        self.max_steer_step = min(1.0, max(0.0, float(max_steer_step)))
        self.integral_limit = abs(float(integral_limit))
        self.reset()

    def configure(
        self,
        *,
        kp: float,
        ki: float,
        kd: float,
        steer_limit: float,
        max_steer_step: float,
    ) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.steer_limit = min(1.0, max(0.0, float(steer_limit)))
        self.max_steer_step = min(1.0, max(0.0, float(max_steer_step)))
        self.reset()

    def reset(self) -> None:
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_steer = 0.0

    def update(self, error: float) -> float:
        error_value = float(error)
        if not math.isfinite(error_value):
            raise ValueError("Lane-centre error must be finite.")

        self.integral += error_value
        self.integral = min(
            self.integral_limit,
            max(-self.integral_limit, self.integral),
        )
        derivative = error_value - self.prev_error
        self.prev_error = error_value

        steer = (
            self.kp * error_value
            + self.ki * self.integral
            + self.kd * derivative
        )
        steer = min(self.steer_limit, max(-self.steer_limit, steer))

        delta = steer - self.last_steer
        if delta > self.max_steer_step:
            steer = self.last_steer + self.max_steer_step
        elif delta < -self.max_steer_step:
            steer = self.last_steer - self.max_steer_step

        self.last_steer = steer
        return float(steer)


class AvisObstacleLaneChangeController:
    """AVIS-only obstacle lane-change state machine without GPS dependencies."""

    RIGHT_LANE = "RIGHT_LANE"
    STEER_LEFT = "STEER_LEFT"
    ACQUIRE_LEFT_LINE = "ACQUIRE_LEFT_LINE"
    LEFT_LANE = "LEFT_LANE"
    STEER_RIGHT = "STEER_RIGHT"
    ACQUIRE_RIGHT_LINE = "ACQUIRE_RIGHT_LINE"
    FAILED = "FAILED"

    def __init__(self) -> None:
        self.phase = self.RIGHT_LANE
        self.phase_distance_m = 0.0
        self.left_lane_distance_m = 0.0
        self.last_obstacle_distance_m: float | None = None
        self.left_lane_target_distance_m: float | None = None
        self.return_reason = "DISTANCE"
        self.obstacle_confirm_frames = 0
        self.sensor_clear_frames = 0
        self.obstacle_latched = False
        self.last_obstacle_sensor: str | None = None
        self.line_confirm_frames = 0
        self.last_line_observation_id = None

    @staticmethod
    def front_sensor_distance_cm(sensors: Any) -> float | None:
        """AVIS sensor order is left, middle/front, right; all values are cm."""
        try:
            value = float(sensors[1])
        except (TypeError, ValueError, IndexError, KeyError):
            return None
        if not math.isfinite(value) or value < 0.0:
            return None
        return value

    @staticmethod
    def sensor_distances_cm(sensors: Any) -> tuple[float | None, float | None, float | None]:
        """Return AVIS side/front/side distances in centimetres."""
        values: list[float | None] = []
        for index in range(3):
            try:
                value = float(sensors[index])
            except (TypeError, ValueError, IndexError, KeyError):
                value = float("nan")
            values.append(value if math.isfinite(value) and value >= 0.0 else None)
        return values[0], values[1], values[2]

    @staticmethod
    def _positive_setting(settings: dict[str, Any], key: str, default: float) -> float:
        try:
            value = float(settings.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) and value > 0.0 else default

    def _target_lane(self) -> str:
        if self.phase in {self.STEER_LEFT, self.ACQUIRE_LEFT_LINE, self.LEFT_LANE}:
            return "left"
        return "right"

    def snapshot(self, sensors: Any = None) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "lane_side": "left" if self.phase in {
                self.LEFT_LANE, self.STEER_RIGHT, self.ACQUIRE_RIGHT_LINE
            } else "right",
            "target_lane": self._target_lane(),
            "left_lane_distance_m": float(self.left_lane_distance_m),
            "left_lane_target_distance_m": self.left_lane_target_distance_m,
            "last_obstacle_distance_m": self.last_obstacle_distance_m,
            "return_reason": self.return_reason,
            "phase_distance_m": float(self.phase_distance_m),
            "front_sensor_cm": self.front_sensor_distance_cm(sensors),
            "sensor_distances_cm": list(self.sensor_distances_cm(sensors)),
            "obstacle_trigger_sensor": self.last_obstacle_sensor,
            "active": self.phase in {
                self.STEER_LEFT, self.ACQUIRE_LEFT_LINE,
                self.STEER_RIGHT, self.ACQUIRE_RIGHT_LINE,
            },
            "failed": self.phase == self.FAILED,
            "steering_override": None,
            "safe_stop": self.phase == self.FAILED,
            "pid_reset": False,
        }

    def _begin_right_return(self, reason: str) -> None:
        self.phase = self.STEER_RIGHT
        self.phase_distance_m = 0.0
        self.line_confirm_frames = 0
        self.last_line_observation_id = None
        self.return_reason = reason

    def update(
        self,
        *,
        sensors: Any,
        speed_kmh: float,
        dt: float,
        left_x: int | float | None,
        right_x: int | float | None,
        settings: dict[str, Any],
        allow_new_maneuver: bool = True,
        observation_id: Any = None,
    ) -> dict[str, Any]:
        sensor_distances = self.sensor_distances_cm(sensors)
        try:
            speed = float(speed_kmh)
            elapsed = float(dt)
        except (TypeError, ValueError):
            speed, elapsed = 0.0, 0.0
        if not math.isfinite(speed):
            speed = 0.0
        if not math.isfinite(elapsed):
            elapsed = 0.0
        travelled_m = max(0.0, speed) / 3.6 * min(0.25, max(0.0, elapsed))

        if self.phase in {
            self.STEER_LEFT, self.ACQUIRE_LEFT_LINE,
            self.STEER_RIGHT, self.ACQUIRE_RIGHT_LINE,
        }:
            self.phase_distance_m += travelled_m
        elif self.phase == self.LEFT_LANE:
            self.left_lane_distance_m += travelled_m

        trigger_cm = self._positive_setting(
            settings, "obstacle_front_threshold_cm", 950.0
        )
        configured_clear_cm = self._positive_setting(
            settings, "obstacle_sensor_clear_threshold_cm", 900.0
        )
        # Keep sensor hysteresis valid automatically; users should not have to
        # tune two coupled fields just to change the obstacle trigger distance.
        clear_cm = max(configured_clear_cm, trigger_cm + 1.0)
        trigger_frames = max(
            1, int(self._positive_setting(settings, "obstacle_trigger_confirm_frames", 3))
        )
        clear_frames_required = max(
            1, int(self._positive_setting(
                settings, "obstacle_sensor_clear_confirm_frames", 5
            ))
        )

        # Clear the latch only after every distance sensor reports a clear
        # path; one side sensor still seeing an obstacle must keep it latched.
        all_sensors_clear = all(
            distance is not None and distance >= clear_cm
            for distance in sensor_distances
        )
        if all_sensors_clear:
            self.sensor_clear_frames += 1
            if self.sensor_clear_frames >= clear_frames_required:
                self.obstacle_latched = False
                self.obstacle_confirm_frames = 0
        else:
            self.sensor_clear_frames = 0

        pid_reset = False
        detection_enabled = bool(settings.get("obstacle_lane_change_enabled", True))
        # Any of the left, front, or right sensors can request the lane change.
        # Distance sensing stays active even when lane perception is missing.
        can_check_obstacle = self.phase in {self.RIGHT_LANE, self.LEFT_LANE}
        triggered_sensors = [
            (index, distance)
            for index, distance in enumerate(sensor_distances)
            if distance is not None and distance <= trigger_cm
        ]
        if detection_enabled and can_check_obstacle and not self.obstacle_latched:
            if triggered_sensors:
                self.obstacle_confirm_frames += 1
            else:
                self.obstacle_confirm_frames = 0

            if self.obstacle_confirm_frames >= trigger_frames:
                self.obstacle_latched = True
                self.obstacle_confirm_frames = 0
                sensor_names = ("LEFT", "FRONT", "RIGHT")
                sensor_index, obstacle_distance = min(
                    triggered_sensors, key=lambda item: item[1]
                )
                self.last_obstacle_sensor = sensor_names[sensor_index]
                self.last_obstacle_distance_m = obstacle_distance / 100.0
                clearance_m = self._positive_setting(
                    settings, "obstacle_clearance_after_pass_m", 5.0
                )
                self.left_lane_target_distance_m = (
                    self.last_obstacle_distance_m + clearance_m
                )

                if self.phase == self.RIGHT_LANE:
                    self.phase = self.STEER_LEFT
                    self.phase_distance_m = 0.0
                    self.line_confirm_frames = 0
                    self.last_line_observation_id = None
                    self.return_reason = "OBSTACLE"
                    pid_reset = True
                elif self.phase == self.LEFT_LANE:
                    # Any new obstacle sensor trigger while in the left lane
                    # overrides the previous pass-distance target and returns
                    # the vehicle to the right lane.
                    self._begin_right_return("OBSTACLE_IN_LEFT_LANE")
                    pid_reset = True

        steer_distance = self._positive_setting(
            settings, "obstacle_lane_change_steer_distance_m", 3.0
        )
        left_steer = min(
            1.0,
            self._positive_setting(settings, "obstacle_lane_change_left_steer", 0.20),
        )
        right_steer = min(
            1.0,
            self._positive_setting(settings, "obstacle_lane_change_right_steer", 0.20),
        )
        line_frames_required = max(
            1, int(self._positive_setting(
                settings, "obstacle_lane_line_confirm_frames", 3
            ))
        )
        line_search_max_m = self._positive_setting(
            settings, "obstacle_lane_line_search_max_distance_m", 40.0
        )
        if self.phase == self.LEFT_LANE:
            target_distance = self.left_lane_target_distance_m
            if target_distance is not None and self.left_lane_distance_m >= target_distance:
                self._begin_right_return("DISTANCE")
                pid_reset = True

        # Count only new inference results, not repeated control-loop reads.
        if self.phase == self.ACQUIRE_LEFT_LINE and observation_id is not None:
            if observation_id != self.last_line_observation_id:
                self.last_line_observation_id = observation_id
                self.line_confirm_frames = (
                    self.line_confirm_frames + 1 if left_x is not None else 0
                )
        elif self.phase == self.ACQUIRE_RIGHT_LINE and observation_id is not None:
            if observation_id != self.last_line_observation_id:
                self.last_line_observation_id = observation_id
                self.line_confirm_frames = (
                    self.line_confirm_frames + 1 if right_x is not None else 0
                )

        steering_override: float | None = None
        safe_stop = False

        if self.phase == self.STEER_LEFT:
            if self.phase_distance_m >= steer_distance:
                self.phase = self.ACQUIRE_LEFT_LINE
                self.phase_distance_m = 0.0
                self.line_confirm_frames = 0
                self.last_line_observation_id = observation_id
                steering_override = 0.0
            else:
                steering_override = -left_steer

        elif self.phase == self.ACQUIRE_LEFT_LINE:
            steering_override = 0.0
            if self.line_confirm_frames >= line_frames_required:
                self.phase = self.LEFT_LANE
                self.phase_distance_m = 0.0
                self.left_lane_distance_m = 0.0
                self.line_confirm_frames = 0
                self.last_line_observation_id = None
                pid_reset = True
        
        elif self.phase == self.STEER_RIGHT:
            if self.phase_distance_m >= steer_distance:
                self.phase = self.ACQUIRE_RIGHT_LINE
                self.phase_distance_m = 0.0
                self.line_confirm_frames = 0
                self.last_line_observation_id = observation_id
                steering_override = 0.0
            else:
                steering_override = right_steer

        elif self.phase == self.ACQUIRE_RIGHT_LINE:
            steering_override = 0.0
            if self.line_confirm_frames >= line_frames_required:
                self.phase = self.RIGHT_LANE
                self.phase_distance_m = 0.0
                self.left_lane_distance_m = 0.0
                self.left_lane_target_distance_m = None
                self.last_obstacle_distance_m = None
                self.line_confirm_frames = 0
                self.last_line_observation_id = None
                self.return_reason = "DISTANCE"
                pid_reset = True
        
        elif self.phase == self.FAILED:
            safe_stop = True
            steering_override = 0.0

        # Evaluate after phase transitions so telemetry reflects the actual
        # active line-acquisition phase for this control update.
        search_limit_reached = (
            self.phase in {self.ACQUIRE_LEFT_LINE, self.ACQUIRE_RIGHT_LINE}
            and self.phase_distance_m >= line_search_max_m
        )
        result = self.snapshot(sensors)
        result.update({
            "steering_override": steering_override,
            "safe_stop": safe_stop or self.phase == self.FAILED,
            "pid_reset": pid_reset,
            "obstacle_latched": self.obstacle_latched,
            "line_confirm_frames": self.line_confirm_frames,
            "search_limit_reached": search_limit_reached,
        })
        return result


def _candidate_api_directories(api_path: str | None) -> list[Path]:
    candidates: list[Path] = []
    if api_path:
        supplied = Path(api_path).expanduser()
        candidates.append(supplied.parent if supplied.is_file() else supplied)

    environment_path = os.environ.get("AVIS_ENGINE_API_PATH")
    if environment_path:
        supplied = Path(environment_path).expanduser()
        candidates.append(supplied.parent if supplied.is_file() else supplied)

    candidates.extend(
        (
            PROJECT_ROOT / "AVIS-Engine-Python-API",
            PROJECT_ROOT / "avis-engine-python-api",
            PROJECT_ROOT.parent / "AVIS-Engine-Python-API",
        )
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def load_avis_api(api_path: str | None = None) -> ModuleType:
    """Load the vendor API while isolating its top-level utility module."""
    engine_file: Optional[Path] = None
    for directory in _candidate_api_directories(api_path):
        candidate = directory / "avisengine.py"
        if candidate.is_file():
            engine_file = candidate.resolve()
            break

    if engine_file is None:
        try:
            discovered = importlib.util.find_spec("avisengine")
        except (ImportError, ValueError):
            discovered = None
        if discovered is not None and discovered.origin:
            engine_file = Path(discovered.origin).resolve()

    if engine_file is None:
        raise RuntimeError(
            "The AVIS Python API was not found. Clone the 1.2.4-compatible "
            "FIRA-ACL branch into the project folder, or pass --avis-api-path:\n"
            "git clone --branch FIRA-ACL --single-branch "
            "https://github.com/AVISEngine/AVIS-Engine-Python-API.git"
        )

    api_utils_file = engine_file.with_name("utils.py")
    previous_utils = sys.modules.get("utils")
    previous_engine = sys.modules.get("avisengine")
    try:
        if api_utils_file.is_file():
            utils_spec = importlib.util.spec_from_file_location("utils", api_utils_file)
            if utils_spec is None or utils_spec.loader is None:
                raise ImportError(f"Could not load AVIS utility module: {api_utils_file}")
            utils_module = importlib.util.module_from_spec(utils_spec)
            sys.modules["utils"] = utils_module
            utils_spec.loader.exec_module(utils_module)

        engine_spec = importlib.util.spec_from_file_location("avisengine", engine_file)
        if engine_spec is None or engine_spec.loader is None:
            raise ImportError(f"Could not load AVIS API module: {engine_file}")
        engine_module = importlib.util.module_from_spec(engine_spec)
        sys.modules["avisengine"] = engine_module
        engine_spec.loader.exec_module(engine_module)
    finally:
        if previous_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = previous_utils
        if previous_engine is None:
            sys.modules.pop("avisengine", None)
        else:
            sys.modules["avisengine"] = previous_engine

    if not callable(getattr(engine_module, "Car", None)):
        raise RuntimeError(f"The AVIS API at {engine_file} does not expose Car().")
    return engine_module


class AvisEngineClient:
    """Defensive adapter around the vendor's synchronous TCP client."""

    def __init__(self, api_module: ModuleType, host: str, port: int) -> None:
        self.car = api_module.Car()
        self._connected = False
        self._closed = False

        # The vendor destructor calls stop() even after explicit shutdown.
        # This adapter owns shutdown and makes that destructor path harmless.
        self.car.stop = lambda: None
        try:
            self._connected = bool(self.car.connect(host, int(port)))
        except Exception:
            self._close_socket()
            self._closed = True
            raise

        if not self._connected:
            self._close_socket()
            self._closed = True
            raise ConnectionError(
                f"Could not connect to AVIS Engine at {host}:{int(port)}. "
                "Check that the simulator server is running and its IP/port match."
            )

    def set_sensor_angle(self, angle: int) -> None:
        if self._closed:
            raise RuntimeError("The AVIS connection is already closed.")
        self.car.setSensorAngle(int(angle))

    def read_data(self) -> tuple[Any, float, Any]:
        if self._closed or not self._connected:
            raise RuntimeError("The AVIS connection is not active.")
        self.car.getData()
        image = self.car.getImage()
        speed = self.car.getSpeed()
        sensors = self.car.getSensors()
        try:
            speed_kmh = float(speed)
        except (TypeError, ValueError):
            speed_kmh = 0.0
        if not math.isfinite(speed_kmh):
            speed_kmh = 0.0
        return image, max(0.0, speed_kmh), sensors

    def send_control(self, throttle: float, steering: float) -> None:
        if self._closed or not self._connected:
            raise RuntimeError("The AVIS connection is not active.")
        speed_command, steering_command = normalize_avis_controls(throttle, steering)
        self.car.setSpeed(speed_command)
        self.car.setSteering(steering_command)

    def send_manual_control(self, throttle: float, steering: float) -> None:
        if self._closed or not self._connected:
            raise RuntimeError("The AVIS connection is not active.")
        speed_command, steering_command = normalize_avis_manual_controls(
            throttle, steering
        )
        self.car.setSpeed(speed_command)
        self.car.setSteering(steering_command)

    def _close_socket(self) -> None:
        sock = getattr(self.car, "sock", None)
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._connected:
            try:
                self.car.setSpeed(0)
            except Exception:
                pass
            try:
                self.car.setSteering(0)
            except Exception:
                pass
            sock = getattr(self.car, "sock", None)
            if sock is not None:
                try:
                    sock.sendall(b"stop")
                except OSError:
                    pass
        self._close_socket()

    def __enter__(self) -> "AvisEngineClient":
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the existing StableV2 vision/PID driving pipeline on AVIS Engine."
    )
    parser.add_argument("--avis", action="store_true", help="Select the AVIS Engine runtime.")
    parser.add_argument(
        "--avis-host",
        default=os.environ.get("AVIS_HOST", "127.0.0.1"),
        help="AVIS server IP address (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--avis-port",
        type=int,
        default=int(os.environ.get("AVIS_PORT", "25001")),
        help="AVIS server TCP port (default: 25001).",
    )
    parser.add_argument(
        "--avis-api-path",
        default=None,
        help="Path to the AVIS Python API repository or avisengine.py file.",
    )
    parser.add_argument(
        "--avis-sensor-angle",
        type=int,
        default=40,
        help="Angle between the three AVIS distance sensors (default: 40 degrees).",
    )
    parser.add_argument(
        "--avis-startup-delay",
        type=float,
        default=3.0,
        help="Seconds to wait after connecting before reading camera data.",
    )
    parser.add_argument(
        "--avis-headless",
        action="store_true",
        help="Disable the local OpenCV preview window; Flask remains enabled.",
    )
    parser.add_argument(
        "--avis-web-host",
        default=os.environ.get("AVIS_WEB_HOST", "127.0.0.1"),
        help="AVIS dashboard bind address (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--avis-web-port",
        type=int,
        default=int(os.environ.get("AVIS_WEB_PORT", "5001")),
        help="AVIS dashboard port (default: 5001).",
    )
    parser.add_argument(
        "--avis-no-web",
        action="store_true",
        help="Disable the AVIS-only Flask dashboard.",
    )
    return parser


AVIS_ROI_CONFIG_KEYS = {
    "ll_top_roi": "LL_TOP_ROI",
    "ll_bottom_roi": "LL_BOTTOM_ROI",
    "ll_left_roi": "LL_LEFT_ROI",
    "ll_right_roi": "LL_RIGHT_ROI",
    "rl_top_roi": "RL_TOP_ROI",
    "rl_bottom_roi": "RL_BOTTOM_ROI",
    "rl_left_roi": "RL_LEFT_ROI",
    "rl_right_roi": "RL_RIGHT_ROI",
}

AVIS_SHARED_VISION_CONFIG_KEYS = {
    "min_side_pixels": "MIN_SIDE_PIXELS",
    "lane_offset_ratio": "FALLBACK_LANE_OFFSET_RATIO",
    "center_smoothing": "LANE_CENTER_SMOOTH_ALPHA",
}


def _apply_avis_roi_settings(conf: ModuleType, settings: dict[str, Any]) -> None:
    """Apply AVIS-only ROI and lane geometry values to this process's config."""
    for mapping in (AVIS_ROI_CONFIG_KEYS, AVIS_SHARED_VISION_CONFIG_KEYS):
        for setting_key, config_key in mapping.items():
            if setting_key in settings:
                setattr(conf, config_key, float(settings[setting_key]))


def _build_avis_settings(conf: ModuleType, sensor_angle: int) -> dict[str, Any]:
    return {
        "vision_backend": "ultralytics",
        "control_loop_hz": float(conf.CONTROL_LOOP_HZ),
        "inference_interval_ms": max(
            10.0, float(conf.VISION_INFERENCE_INTERVAL_SECONDS) * 1000.0
        ),
        "sensor_angle": int(sensor_angle),
        "min_speed_kmh": float(conf.MIN_SPEED_KMH),
        "max_speed_kmh": float(conf.MAX_SPEED_KMH),
        "acceleration_mps2": float(conf.ACCELERATION_MPS2),
        "kp": float(conf.KP),
        "ki": float(conf.KI),
        "kd": float(conf.KD),
        "steer_limit": float(conf.STEER_LIMIT),
        "max_steer_step": float(conf.MAX_STEER_STEP),
        "min_side_pixels": int(conf.MIN_SIDE_PIXELS),
        "lane_offset_ratio": float(conf.FALLBACK_LANE_OFFSET_RATIO),
        "center_smoothing": float(conf.LANE_CENTER_SMOOTH_ALPHA),
        "ll_top_roi": float(conf.LL_TOP_ROI),
        "ll_bottom_roi": float(conf.LL_BOTTOM_ROI),
        "ll_left_roi": float(conf.LL_LEFT_ROI),
        "ll_right_roi": float(conf.LL_RIGHT_ROI),
        "rl_top_roi": float(conf.RL_TOP_ROI),
        "rl_bottom_roi": float(conf.RL_BOTTOM_ROI),
        "rl_left_roi": float(conf.RL_LEFT_ROI),
        "rl_right_roi": float(conf.RL_RIGHT_ROI),
    }


def _create_avis_vision_processor(
    backend: str,
    settings: dict[str, Any],
    adaptive_roi_controller=None,
):
    if backend == "classical":
        from avis_classical_vision import AvisClassicalVisionProcessor

        return AvisClassicalVisionProcessor(
            settings,
            adaptive_roi_controller=adaptive_roi_controller,
        )

    from vision.city_vision_processing import VisionProcessor

    class AvisVisionProcessor(VisionProcessor):
        """AVIS model adapter; vehicle segmentation is optional and never drawn."""

        def detect(self, semantic_frame, rgb_frame=None):
            # The shared processor reads ROI coordinates from config_city.
            # Override them only inside this AVIS-specific adapter and only
            # for the current inference; no shared CARLA code is modified.
            if self.adaptive_roi_controller is not None:
                import config_city as avis_conf

                _apply_avis_roi_settings(
                    avis_conf,
                    self.adaptive_roi_controller.active_settings(),
                )
            result = super().detect(semantic_frame, rgb_frame=rgb_frame)
            frame = rgb_frame if rgb_frame is not None else semantic_frame
            if (
                self.adaptive_roi_controller is not None
                and frame is not None
                and getattr(frame, "ndim", 0) >= 2
            ):
                debug = result.get("debug")
                if isinstance(debug, dict):
                    self.adaptive_roi_controller.observe(
                        debug.get("left_x"),
                        debug.get("right_x"),
                        int(frame.shape[1]),
                    )
                    debug["adaptive_roi"] = self.adaptive_roi_controller.snapshot()
            return result

        @staticmethod
        def _class_ids_from_names(
            class_names: dict[int, str],
        ) -> tuple[int, int, int | None]:
            # Lane and drivable-area classes are required by AVIS lane following.
            # Vehicle segmentation is not used by AVIS, so checkpoints without
            # that class must remain valid.
            required_ids = []
            for class_name, label in (
                ("lane_lines", "lane"),
                ("ego_drivable_area", "drivable-area"),
            ):
                matches = [
                    class_id
                    for class_id, name in class_names.items()
                    if name == class_name
                ]
                if len(matches) != 1:
                    raise RuntimeError(
                        "The AVIS Ultralytics checkpoint must contain exactly "
                        f"one '{class_name}' class for {label} masks; "
                        f"found names {class_names!r}"
                    )
                required_ids.append(matches[0])

            vehicle_ids = [
                class_id
                for class_id, name in class_names.items()
                if name == "vehicles"
            ]
            if len(vehicle_ids) > 1:
                raise RuntimeError(
                    "The AVIS Ultralytics checkpoint may contain at most one "
                    f"'vehicles' class; found names {class_names!r}"
                )
            vehicle_class_id = vehicle_ids[0] if vehicle_ids else None
            return required_ids[0], required_ids[1], vehicle_class_id

        @staticmethod
        def _semantic_map_to_masks(
            semantic_map,
            lane_class_id: int,
            drivable_class_id: int,
            vehicles_class_id: int | None,
            frame_shape: tuple[int, int],
        ):
            if vehicles_class_id is not None:
                return VisionProcessor._semantic_map_to_masks(
                    semantic_map,
                    lane_class_id,
                    drivable_class_id,
                    vehicles_class_id,
                    frame_shape,
                )

            # Reuse the shared mask decoder for the required classes, then
            # explicitly report that no vehicle mask exists for this model.
            drivable_mask, lane_mask, _ = VisionProcessor._semantic_map_to_masks(
                semantic_map,
                lane_class_id,
                drivable_class_id,
                -1,
                frame_shape,
            )
            return drivable_mask, lane_mask, None

        def __init__(self, mode: str):
            super().__init__(mode)
            self.adaptive_roi_controller = adaptive_roi_controller
            self.last_lane_half_width: float | None = None
            if getattr(self, "vehicles_class_id", None) is None:
                print(
                    "[AVIS] Model has no 'vehicles' class; "
                    "continuing without vehicle masks."
                )

        def _pick_sticky_lane(
            self,
            left_x,
            right_x,
            frame_center,
            lane_offset_px,
        ):
            """Use AVIS midpoint centring for model and classical lane masks."""
            if left_x is not None and right_x is not None and right_x > left_x:
                self.last_lane_half_width = (right_x - left_x) / 2.0
                self.active_side = None
                return "both", left_x + self.last_lane_half_width

            # If one lane marking disappears temporarily, infer the target
            # from the most recently measured lane half-width. Only use the
            # configured fixed ratio before any complete lane width is known.
            offset_px = (
                self.last_lane_half_width
                if self.last_lane_half_width is not None
                else lane_offset_px
            )

            if self.active_side == "right":
                if right_x is not None:
                    return "only_right", right_x - offset_px
                if left_x is not None:
                    self.active_side = "left"
                    return "only_left", left_x + offset_px
                return (
                    "none",
                    self.last_lane_center
                    if self.last_lane_center is not None
                    else frame_center,
                )

            if self.active_side == "left":
                if left_x is not None:
                    return "only_left", left_x + offset_px
                if right_x is not None:
                    self.active_side = "right"
                    return "only_right", right_x - offset_px
                return (
                    "none",
                    self.last_lane_center
                    if self.last_lane_center is not None
                    else frame_center,
                )

            if right_x is not None:
                self.active_side = "right"
                return "only_right", right_x - offset_px
            if left_x is not None:
                self.active_side = "left"
                return "only_left", left_x + offset_px
            return (
                "none",
                self.last_lane_center
                if self.last_lane_center is not None
                else frame_center,
            )

        def _draw_debug(
            self,
            frame,
            line_mask,
            drivable_mask,
            left_x,
            right_x,
            lane_center,
            lane_type,
            vehicles_mask=None,
        ):
            return super()._draw_debug(
                frame=frame,
                line_mask=line_mask,
                drivable_mask=drivable_mask,
                left_x=left_x,
                right_x=right_x,
                lane_center=lane_center,
                lane_type=lane_type,
                vehicles_mask=None,
            )

    return AvisVisionProcessor("ultralytics")


def _encode_avis_preview(cv2_module, image) -> bytes | None:
    if image is None:
        return None
    try:
        ok, encoded = cv2_module.imencode(
            ".jpg",
            image,
            [int(cv2_module.IMWRITE_JPEG_QUALITY), 78],
        )
    except Exception:
        return None
    return encoded.tobytes() if ok else None


def run_avis(args: argparse.Namespace) -> int:
    client: Optional[AvisEngineClient] = None
    inference_worker = None
    dashboard_server = None
    cv2_module = None
    dashboard_state = None

    try:
        import config_city as conf

        from avis_dashboard import AvisDashboardServer, AvisDashboardState

        profile = load_avis_profile()
        apply_avis_profile(profile, conf)

        # Load the vendor API before project modules that may use a local
        # package with the same top-level utility-module name.
        avis_api = load_avis_api(args.avis_api_path)

        import cv2
        import numpy as np
        from controllers.speed_controller import SpeedPIDController
        from trajectory.worker import InferenceWorker

        cv2_module = cv2
        print(f"[AVIS] Loading AVIS-specific profile: {AVIS_PROFILE_NAME}")

        dashboard_state = AvisDashboardState(
            initial_settings=_build_avis_settings(conf, args.avis_sensor_angle),
            settings_path=PROJECT_ROOT / "avis_settings.json",
        )
        settings, settings_revision = dashboard_state.settings_snapshot()
        _apply_avis_roi_settings(conf, settings)

        from avis_adaptive_roi import AvisAdaptiveROIController

        adaptive_roi_controller = AvisAdaptiveROIController(settings)
        try:
            vision_processor = _create_avis_vision_processor(
                settings["vision_backend"],
                settings,
                adaptive_roi_controller=adaptive_roi_controller,
            )
        except (FileNotFoundError, ImportError) as exc:
            if settings["vision_backend"] != "ultralytics":
                raise
            print(
                "[AVIS] Ultralytics model is unavailable; switching this AVIS "
                f"session to classical lane detection: {exc}"
            )
            settings = dashboard_state.update_settings(
                {"vision_backend": "classical"}
            )
            settings_revision = dashboard_state.settings_snapshot()[1]
            adaptive_roi_controller.configure(settings)
            vision_processor = _create_avis_vision_processor(
                "classical",
                settings,
                adaptive_roi_controller=adaptive_roi_controller,
            )

        inference_worker = InferenceWorker(vision_processor=vision_processor)
        inference_worker.start()

        client = AvisEngineClient(avis_api, args.avis_host, args.avis_port)
        client.send_control(0.0, 0.0)
        client.set_sensor_angle(int(settings["sensor_angle"]))

        steering_controller = AvisSteeringController(
            kp=float(settings["kp"]),
            ki=float(settings["ki"]),
            kd=float(settings["kd"]),
            steer_limit=float(settings["steer_limit"]),
            max_steer_step=float(settings["max_steer_step"]),
        )
        speed_controller = SpeedPIDController(
            acceleration_mps2=float(settings["acceleration_mps2"]),
            min_speed_kmh=float(settings["min_speed_kmh"]),
            max_speed_kmh=float(settings["max_speed_kmh"]),
        )
        obstacle_lane_change = AvisObstacleLaneChangeController()

        if not bool(conf.AUTO_MODE_DEFAULT):
            dashboard_state.set_mode("stopped")

        if not args.avis_no_web:
            dashboard_server = AvisDashboardServer(
                dashboard_state,
                host=args.avis_web_host,
                port=args.avis_web_port,
            )
            dashboard_server.start()
            dashboard_address = (
                "127.0.0.1"
                if args.avis_web_host in {"0.0.0.0", "::"}
                else args.avis_web_host
            )
            print(
                f"[AVIS WEB] Dashboard: http://{dashboard_address}:{args.avis_web_port}"
            )
            if args.avis_web_host not in {"127.0.0.1", "localhost", "::1"}:
                print(
                    "[AVIS WEB] WARNING: the dashboard has no authentication. "
                    "Do not expose it to untrusted networks."
                )

        print(f"[AVIS] Connected to {args.avis_host}:{args.avis_port}.")
        print(
            f"[AVIS] Waiting {max(0.0, args.avis_startup_delay):.1f}s "
            "for the simulator camera stream..."
        )
        time.sleep(max(0.0, float(args.avis_startup_delay)))

        control_period = 1.0 / float(settings["control_loop_hz"])
        inference_interval = float(settings["inference_interval_ms"]) / 1000.0
        active_backend = str(settings["vision_backend"])
        active_sensor_angle = int(settings["sensor_angle"])
        last_inference_submit = 0.0
        last_vision_result = None
        last_dashboard_publish = 0.0
        frame_id = 0
        last_error_message = None
        consecutive_connection_errors = 0
        last_control_time = time.perf_counter()
        recovery_steering = 0.0
        next_tick = last_control_time
        frame_read_fps = AvisFpsMonitor(window_seconds=2.0)
        inference_fps = AvisFpsMonitor(window_seconds=2.0)
        control_loop_fps = AvisFpsMonitor(window_seconds=2.0)
        last_inference_fps_timestamp: float | None = None

        print("[AVIS] AVIS runtime is ready.")
        print("[AVIS] Use the web dashboard for auto/manual mode and live tuning.")
        print("[AVIS] Press Q or Esc in the local preview window to stop.")

        while True:
            control_loop_fps.record()
            try:
                frame, current_speed, sensors = client.read_data()
                if isinstance(frame, np.ndarray) and frame.size > 0:
                    frame_read_fps.record()
                consecutive_connection_errors = 0
            except Exception as exc:
                consecutive_connection_errors += 1
                client.send_control(0.0, 0.0)
                signature = f"{type(exc).__name__}: {exc}"
                if signature != last_error_message:
                    print(f"[AVIS] Data receive failed: {signature}")
                    last_error_message = signature
                if consecutive_connection_errors >= 3:
                    raise ConnectionError(
                        "AVIS data reception failed three consecutive times."
                    ) from exc
                continue

            current_settings, current_revision = dashboard_state.settings_snapshot()
            if current_revision != settings_revision:
                try:
                    adaptive_roi_controller.configure(current_settings)
                    _apply_avis_roi_settings(conf, current_settings)
                    if int(current_settings["sensor_angle"]) != active_sensor_angle:
                        client.set_sensor_angle(int(current_settings["sensor_angle"]))
                        active_sensor_angle = int(current_settings["sensor_angle"])

                    steering_controller.configure(
                        kp=float(current_settings["kp"]),
                        ki=float(current_settings["ki"]),
                        kd=float(current_settings["kd"]),
                        steer_limit=float(current_settings["steer_limit"]),
                        max_steer_step=float(current_settings["max_steer_step"]),
                    )
                    speed_controller.configure(
                        acceleration_mps2=float(current_settings["acceleration_mps2"]),
                        min_speed_kmh=float(current_settings["min_speed_kmh"]),
                        max_speed_kmh=float(current_settings["max_speed_kmh"]),
                    )

                    selected_backend = str(current_settings["vision_backend"])
                    if selected_backend != active_backend:
                        replacement = _create_avis_vision_processor(
                            selected_backend,
                            current_settings,
                            adaptive_roi_controller=adaptive_roi_controller,
                        )
                        inference_worker.set_vision_processor(replacement)
                        vision_processor = replacement
                        active_backend = selected_backend
                        print(f"[AVIS] Vision backend changed to {active_backend}.")
                    elif active_backend == "classical":
                        vision_processor.update_settings(current_settings)
                    else:
                        vision_processor.min_side_pixels = int(
                            current_settings["min_side_pixels"]
                        )
                        vision_processor.fallback_lane_offset_ratio = float(
                            current_settings["lane_offset_ratio"]
                        )
                        vision_processor.lane_center_alpha = float(
                            current_settings["center_smoothing"]
                        )

                    if any(
                        settings.get(key) != current_settings.get(key)
                        for key in (
                            *AVIS_ROI_CONFIG_KEYS,
                            *AVIS_SHARED_VISION_CONFIG_KEYS,
                        )
                    ) and active_backend == "ultralytics":
                        # Reset only the AVIS-side sticky lane state after
                        # editing shared ROI/centre geometry settings.
                        vision_processor.last_lane_center = None
                        vision_processor.active_side = None

                    control_period = 1.0 / float(current_settings["control_loop_hz"])
                    inference_interval = (
                        float(current_settings["inference_interval_ms"]) / 1000.0
                    )
                    last_vision_result = None
                    last_inference_submit = 0.0
                    settings_revision = current_revision
                    settings = current_settings
                except Exception as exc:
                    error_message = (
                        f"Could not apply AVIS settings: {type(exc).__name__}: {exc}"
                    )
                    print(f"[AVIS] {error_message}")
                    previous_backend = active_backend
                    dashboard_state.update_settings(
                        {"vision_backend": previous_backend}, persist=True
                    )
                    settings, settings_revision = dashboard_state.settings_snapshot()

            valid_frame = (
                isinstance(frame, np.ndarray)
                and frame.ndim == 3
                and frame.shape[2] >= 3
                and frame.size > 0
            )
            vision_error = None
            vision_data: dict[str, Any] = {}
            fps_metrics = {
                "frame_read_fps": frame_read_fps.fps(),
                "inference_fps": inference_fps.fps(),
                "control_loop_fps": control_loop_fps.fps(),
            }
            if valid_frame:
                frame_id += 1
                frame_bgr = np.ascontiguousarray(frame[:, :, :3])
                now = time.perf_counter()

                if now - last_inference_submit >= inference_interval:
                    inference_worker.submit_vision(
                        source_frame_id=frame_id,
                        semantic_frame=None,
                        rgb_frame=frame_bgr,
                    )
                    last_inference_submit = now

                vision_snapshot, vision_busy, vision_error, _ = (
                    inference_worker.vision_snapshot()
                )
                if vision_snapshot is not None and vision_snapshot.valid:
                    last_vision_result = vision_snapshot
                    completed_at = float(vision_snapshot.completed_at)
                    if completed_at != last_inference_fps_timestamp:
                        inference_fps.record(completed_at)
                        last_inference_fps_timestamp = completed_at
                if last_vision_result is not None:
                    vision_data = last_vision_result.result or {}
                fps_metrics = {
                    "frame_read_fps": frame_read_fps.fps(),
                    "inference_fps": inference_fps.fps(),
                    "control_loop_fps": control_loop_fps.fps(),
                }

                drive_mode, manual_throttle, manual_steering, manual_active = (
                    dashboard_state.drive_state()
                )
                throttle = 0.0
                steering = 0.0
                target_speed = 0.0
                coasting = False
                lane_valid = False
                lane_error = 0.0
                status = "WAITING FOR FIRST PREDICTION"

                debug_data = vision_data.get("debug", {})
                if not isinstance(debug_data, dict):
                    debug_data = {}
                lane_change_state = obstacle_lane_change.snapshot(sensors)

                if drive_mode == "manual":
                    throttle = manual_throttle if manual_active else 0.0
                    steering = manual_steering if manual_active else 0.0
                    client.send_manual_control(throttle, steering)
                    status = (
                        "MANUAL"
                        if manual_active
                        else "MANUAL / INPUT TIMED OUT - ZERO THROTTLE"
                    )
                elif drive_mode == "stopped":
                    client.send_control(0.0, 0.0)
                    status = "STOPPED"
                else:
                    result_fresh = False
                    if last_vision_result is not None:
                        result_age = max(
                            0.0,
                            time.perf_counter() - last_vision_result.completed_at,
                        )
                        result_fresh = result_age <= max(
                            0.75, inference_interval * 8.0
                        )

                    lane_valid = bool(
                        result_fresh
                        and vision_data.get(
                            "lane_valid",
                            vision_data.get("lane_type", "none") != "none",
                        )
                    )
                    lane_error = (
                        float(vision_data.get("error", 0.0) or 0.0)
                        if result_fresh else 0.0
                    )
                    control_dt = min(
                        0.1,
                        max(0.005, time.perf_counter() - last_control_time),
                    )
                    lane_change_state = obstacle_lane_change.update(
                        sensors=sensors,
                        speed_kmh=current_speed,
                        dt=control_dt,
                        left_x=debug_data.get("left_x") if result_fresh else None,
                        right_x=debug_data.get("right_x") if result_fresh else None,
                        settings=current_settings,
                        # The distance sensor is independent of lane inference:
                        # do not suppress a confirmed obstacle response just
                        # because the lane marking temporarily disappeared.
                        allow_new_maneuver=True,
                        observation_id=(
                            last_vision_result.completed_at if result_fresh else None
                        ),
                    )
                    if lane_change_state.get("pid_reset"):
                        steering_controller.reset()

                    minimum_speed = float(current_settings["min_speed_kmh"])
                    steering_override = lane_change_state.get("steering_override")
                    if lane_change_state.get("safe_stop"):
                        # Reserved for an explicit controller failure; ordinary
                        # lane loss and line-search timeout never enter this path.
                        throttle = 0.0
                        steering = 0.0
                        status = "LANE CHANGE CONTROLLER FAILURE"
                    elif steering_override is not None:
                        # Steering and target-line acquisition phases all run
                        # at the configured minimum moving speed.
                        steering = float(steering_override)
                        recovery_steering = steering
                        target_speed = minimum_speed
                        throttle, requested_brake = speed_controller.update(
                            current_speed, target_speed, dt=control_dt
                        )
                        coasting = requested_brake > 0.0
                        if coasting:
                            throttle = 0.0
                        if lane_change_state["phase"] == "STEER_LEFT":
                            status = "OBSTACLE AVOIDANCE / STEERING LEFT"
                        elif lane_change_state["phase"] == "ACQUIRE_LEFT_LINE":
                            status = (
                                "OBSTACLE AVOIDANCE / LOW-SPEED LEFT ROI SEARCH"
                                if lane_change_state.get("search_limit_reached")
                                else "OBSTACLE AVOIDANCE / WAITING FOR LEFT ROI"
                            )
                        elif lane_change_state["phase"] == "STEER_RIGHT":
                            status = "OBSTACLE AVOIDANCE / STEERING RIGHT"
                        elif lane_change_state["phase"] == "ACQUIRE_RIGHT_LINE":
                            status = (
                                "OBSTACLE AVOIDANCE / LOW-SPEED RIGHT ROI SEARCH"
                                if lane_change_state.get("search_limit_reached")
                                else "OBSTACLE AVOIDANCE / WAITING FOR RIGHT ROI"
                            )
                        else:
                            status = "OBSTACLE AVOIDANCE"
                    elif not result_fresh:
                        recovery_steering = max(
                            -0.25, min(0.25, recovery_steering * 0.85)
                        )
                        steering = recovery_steering
                        target_speed = minimum_speed
                        throttle, requested_brake = speed_controller.update(
                            current_speed, target_speed, dt=control_dt
                        )
                        coasting = requested_brake > 0.0
                        if coasting:
                            throttle = 0.0
                        status = "VISION RESULT STALE / LOW-SPEED FORWARD"
                    elif not lane_valid:
                        # Keep a gentle forward crawl while lane perception recovers.
                        # Decay the last steering command instead of holding a sharp turn.
                        recovery_steering = max(
                            -0.25, min(0.25, recovery_steering * 0.85)
                        )
                        steering = recovery_steering
                        target_speed = minimum_speed
                        throttle, requested_brake = speed_controller.update(
                            current_speed, target_speed, dt=control_dt
                        )
                        coasting = requested_brake > 0.0
                        if coasting:
                            throttle = 0.0
                        status = "LANE NOT DETECTED / LOW-SPEED FORWARD"
                    else:
                        steering = steering_controller.update(lane_error)
                        recovery_steering = steering
                        base_target_speed = select_avis_target_speed(
                            steering,
                            minimum_speed,
                            float(current_settings["max_speed_kmh"]),
                        )
                        target_speed = base_target_speed
                        throttle, requested_brake = speed_controller.update(
                            current_speed, target_speed, dt=control_dt
                        )
                        coasting = requested_brake > 0.0
                        if coasting:
                            throttle = 0.0
                        status = (
                            "AVOIDANCE / LEFT LANE"
                            if lane_change_state["phase"] == "LEFT_LANE"
                            else "AUTO / COAST" if coasting else "AUTO"
                        )
                    client.send_control(throttle, steering)

                
                if not isinstance(debug_data, dict):
                    debug_data = {}
                combined = debug_data.get("combined")
                line_mask = debug_data.get("lane_mask")
                drivable_mask = debug_data.get("drivable_mask")

                display = (
                    np.ascontiguousarray(combined).copy()
                    if isinstance(combined, np.ndarray) and combined.size
                    else frame_bgr.copy()
                )
                cv2.putText(
                    display,
                    f"AVIS | {status}",
                    (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 230, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    (
                        f"Speed: {current_speed:.1f} km/h | "
                        f"Target: {target_speed:.1f} km/h | "
                        f"Throttle: {throttle * 100:.0f}%"
                    ),
                    (12, 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    f"Steering: {steering:+.3f} ({steering * 100:+.0f})",
                    (12, 76),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                if isinstance(sensors, (list, tuple)) and len(sensors) >= 3:
                    cv2.putText(
                        display,
                        f"Distance sensors: L {sensors[0]} | C {sensors[1]} | R {sensors[2]} cm",
                        (12, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (220, 220, 220),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.putText(
                    display,
                    (
                        f"FPS | Read: {fps_metrics['frame_read_fps']:.1f} | "
                        f"Inference: {fps_metrics['inference_fps']:.1f} | "
                        f"Control: {fps_metrics['control_loop_fps']:.1f}"
                    ),
                    (12, 124),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (100, 235, 255),
                    1,
                    cv2.LINE_AA,
                )

                now = time.perf_counter()
                if now - last_dashboard_publish >= 0.10:
                    black_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
                    dashboard_state.publish_images(
                        {
                            "camera": _encode_avis_preview(cv2, frame_bgr),
                            "combined": _encode_avis_preview(cv2, display),
                            "line-mask": _encode_avis_preview(
                                cv2,
                                line_mask if isinstance(line_mask, np.ndarray) else black_mask,
                            ),
                            "drivable-mask": _encode_avis_preview(
                                cv2,
                                drivable_mask
                                if isinstance(drivable_mask, np.ndarray)
                                else black_mask,
                            ),
                        }
                    )
                    last_dashboard_publish = now

                dashboard_state.publish_telemetry(
                    {
                        "connected": True,
                        "status": status,
                        "speed_kmh": round(float(current_speed), 2),
                        "target_speed_kmh": round(float(target_speed), 2),
                        "throttle": round(float(throttle), 3),
                        "steering": round(float(steering), 3),
                        "lane_error_px": round(float(lane_error), 2),
                        "lane_valid": lane_valid,
                        "sensors_cm": (
                            list(sensors[:3])
                            if isinstance(sensors, (list, tuple)) and len(sensors) >= 3
                            else None
                        ),
                        "front_sensor_cm": lane_change_state.get("front_sensor_cm"),
                        "sensor_distances_cm": lane_change_state.get("sensor_distances_cm"),
                        "obstacle_trigger_sensor": lane_change_state.get("obstacle_trigger_sensor"),
                        "lane_change_phase": lane_change_state.get("phase", "RIGHT_LANE"),
                        "lane_change_side": lane_change_state.get("lane_side", "right"),
                        "lane_change_target": lane_change_state.get("target_lane", "right"),
                        "lane_change_distance_m": round(
                            float(lane_change_state.get("left_lane_distance_m", 0.0)), 2
                        ),
                        "lane_change_target_distance_m": lane_change_state.get(
                            "left_lane_target_distance_m"
                        ),
                        "lane_change_obstacle_distance_m": lane_change_state.get(
                            "last_obstacle_distance_m"
                        ),
                        "lane_change_return_reason": lane_change_state.get(
                            "return_reason", "DISTANCE"
                        ),
                        "lane_change_segment_distance_m": round(
                            float(lane_change_state.get("phase_distance_m", 0.0)), 2
                        ),
                        "adaptive_roi": vision_data.get("debug", {}).get(
                            "adaptive_roi", adaptive_roi_controller.snapshot()
                        ),
                        **fps_metrics,
                        "vision_error": (
                            str(vision_error) if vision_error else None
                        ),
                    }
                )
            else:
                client.send_control(0.0, 0.0)
                display = np.zeros((360, 720, 3), dtype=np.uint8)
                cv2.putText(
                    display,
                    "WAITING FOR AVIS CAMERA...",
                    (28, 185),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
                dashboard_state.publish_telemetry(
                    {
                        "connected": True,
                        "status": "Waiting for camera",
                        "speed_kmh": float(current_speed),
                        "target_speed_kmh": 0.0,
                        "throttle": 0.0,
                        "steering": 0.0,
                        "lane_error_px": 0.0,
                        "lane_valid": False,
                        "sensors_cm": (
                            list(sensors[:3])
                            if isinstance(sensors, (list, tuple)) and len(sensors) >= 3
                            else None
                        ),
                        **fps_metrics,
                        "vision_error": "No valid AVIS camera frame.",
                    }
                )

            last_control_time = time.perf_counter()

            if not args.avis_headless:
                try:
                    cv2.imshow("AVIS Engine | AVIS Runtime", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                except cv2.error as exc:
                    print(
                        f"[AVIS] OpenCV preview unavailable; continuing headless: {exc}"
                    )
                    args.avis_headless = True

            next_tick += control_period
            now = time.perf_counter()
            if next_tick < now:
                next_tick = now
            time.sleep(max(0.0, next_tick - now))

        return 0

    except KeyboardInterrupt:
        print("\n[AVIS] Stop requested.")
        return 0
    except Exception as exc:
        print(f"[AVIS] Runtime failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        if dashboard_state is not None:
            dashboard_state.set_mode("stopped")
        if client is not None:
            client.close()
        if dashboard_server is not None:
            dashboard_server.stop()
        if inference_worker is not None:
            try:
                inference_worker.stop()
            except Exception as exc:
                print(f"[AVIS] Inference worker shutdown warning: {exc}")
        if cv2_module is not None:
            try:
                cv2_module.destroyAllWindows()
            except Exception:
                pass


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.avis:
        print("AVIS mode requires --avis.")
        return 2
    if not 1 <= args.avis_port <= 65535:
        print("[AVIS] Port must be between 1 and 65535.")
        return 2
    if not 10 <= args.avis_sensor_angle <= 90:
        print("[AVIS] Sensor angle must be between 10 and 90 degrees.")
        return 2
    if not 1 <= args.avis_web_port <= 65535:
        print("[AVIS] Web dashboard port must be between 1 and 65535.")
        return 2
    return run_avis(args)


if __name__ == "__main__":
    raise SystemExit(main())
