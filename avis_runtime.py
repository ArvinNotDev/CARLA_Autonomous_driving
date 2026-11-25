from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Optional


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


def _apply_avis_roi_settings(conf: ModuleType, settings: dict[str, Any]) -> None:
    """Apply AVIS-local ROI values to the in-memory vision config only."""
    for setting_key, config_key in AVIS_ROI_CONFIG_KEYS.items():
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
):
    if backend == "classical":
        from avis_classical_vision import AvisClassicalVisionProcessor

        return AvisClassicalVisionProcessor(settings)

    from vision.city_vision_processing import VisionProcessor

    return VisionProcessor("ultralytics")


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

        try:
            vision_processor = _create_avis_vision_processor(
                settings["vision_backend"], settings
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
            vision_processor = _create_avis_vision_processor("classical", settings)

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
        next_tick = last_control_time

        print("[AVIS] AVIS runtime is ready.")
        print("[AVIS] Use the web dashboard for auto/manual mode and live tuning.")
        print("[AVIS] Press Q or Esc in the local preview window to stop.")

        while True:
            try:
                frame, current_speed, sensors = client.read_data()
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
                            selected_backend, current_settings
                        )
                        inference_worker.set_vision_processor(replacement)
                        vision_processor = replacement
                        active_backend = selected_backend
                        print(f"[AVIS] Vision backend changed to {active_backend}.")
                    elif active_backend == "classical":
                        vision_processor.update_settings(current_settings)
                    elif any(
                        settings.get(key) != current_settings.get(key)
                        for key in AVIS_ROI_CONFIG_KEYS
                    ):
                        # VisionProcessor reads these in-memory ROI values
                        # on each frame; reset only its AVIS-side sticky state.
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
                if last_vision_result is not None:
                    vision_data = last_vision_result.result or {}

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

                    if not result_fresh:
                        status = "VISION RESULT STALE - STOPPED"
                    else:
                        lane_valid = bool(
                            vision_data.get(
                                "lane_valid",
                                vision_data.get("lane_type", "none") != "none",
                            )
                        )
                        lane_error = float(vision_data.get("error", 0.0) or 0.0)
                        if not lane_valid:
                            status = "LANE NOT DETECTED - STOPPED"
                        else:
                            steering = steering_controller.update(lane_error)
                            target_speed = select_avis_target_speed(
                                steering,
                                float(current_settings["min_speed_kmh"]),
                                float(current_settings["max_speed_kmh"]),
                            )
                            dt = min(
                                0.1,
                                max(
                                    0.005,
                                    time.perf_counter() - last_control_time,
                                ),
                            )
                            throttle, requested_brake = speed_controller.update(
                                current_speed,
                                target_speed,
                                dt=dt,
                            )
                            # AVIS 1.2.4 has no independent service-brake command.
                            coasting = requested_brake > 0.0
                            if coasting:
                                throttle = 0.0
                            status = "AUTO / COAST" if coasting else "AUTO"
                    client.send_control(throttle, steering)

                debug_data = vision_data.get("debug", {})
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

                lane_valid = bool(
                    vision_data.get(
                        "lane_valid",
                        vision_data.get("lane_type", "none") != "none",
                    )
                )
                if drive_mode == "auto":
                    lane_error = float(vision_data.get("error", 0.0) or 0.0)
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
