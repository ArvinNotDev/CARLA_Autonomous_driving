from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import config_city as conf


PROFILE_PATH = Path("runtime_profiles.json")


SETTING_SPECS = [
    ("JUNCTION_CONTROL_MODE", "Junction control", "choice", ("trajectory", "static_meters"), True),
    ("AUTO_MODE_DEFAULT", "Auto mode", "bool", None, True),
    ("FIXED_THROTTLE", "Fixed throttle", "float", (0.0, 1.0, 0.01), True),
    ("KP", "PID Kp", "float", (0.0, 2.0, 0.001), True),
    ("KI", "PID Ki", "float", (0.0, 2.0, 0.001), True),
    ("KD", "PID Kd", "float", (0.0, 2.0, 0.001), True),
    ("STEER_LIMIT", "Steer limit", "float", (0.0, 1.0, 0.01), True),
    ("MAX_STEER_STEP", "Max steer step", "float", (0.0, 1.0, 0.01), True),
    ("TARGET_SPEED_KMH", "Target speed (km/h)", "float", (0.0, 120.0, 1.0), True),
    ("TRAJECTORY_INFERENCE_INTERVAL_SECONDS", "Trajectory interval (s)", "float", (0.01, 2.0, 0.01), True),
    ("TRAJECTORY_STEER_GAIN", "Trajectory steer gain", "float", (0.0, 3.0, 0.05), True),
    ("TRAJECTORY_MAX_STEER", "Trajectory max steer", "float", (0.0, 1.0, 0.01), True),
    ("LANE_PROB_THRESHOLD", "Lane probability threshold", "float", (0.0, 1.0, 0.01), True),
    ("LANE_CENTER_SMOOTH_ALPHA", "Lane smoothing", "float", (0.0, 1.0, 0.01), True),
    ("VISION_DEBUG", "Vision debug overlays", "bool", None, True),
    ("DEBUG_SHOW_ROIS", "Show ROIs", "bool", None, True),
    ("DEBUG_SHOW_LANE_MASK", "Show lane mask", "bool", None, True),
    ("DEBUG_SHOW_TRAJECTORY", "Show trajectory", "bool", None, True),
    ("DEBUG_SHOW_FPS", "Show FPS", "bool", None, True),
    ("DEBUG_SHOW_GPS", "Show GPS/navigation", "bool", None, True),
    ("SHOW_OPENCV_WINDOW", "OpenCV debug window", "bool", None, True),
    ("CAMERA_IMAGE_WIDTH", "Camera width", "int", (160, 1920, 16), False),
    ("CAMERA_IMAGE_HEIGHT", "Camera height", "int", (120, 1080, 16), False),
    ("MODEL_PATH", "ONNX model path", "str", None, False),
]


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class RuntimeSettings:
    def __init__(self, path: Path = PROFILE_PATH) -> None:
        self.path = path
        self.profiles: dict[str, dict[str, Any]] = {}
        self.load()
        if "default" not in self.profiles:
            self.profiles["default"] = self.snapshot()
            self.save()
        active = self.profiles.get("_active_profile")
        if isinstance(active, str) and active in self.profiles:
            self.apply(self.profiles[active])

    def snapshot(self) -> dict[str, Any]:
        return {key: _json_value(getattr(conf, key, None)) for key, *_ in SETTING_SPECS}

    def apply(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            if any(spec[0] == key for spec in SETTING_SPECS):
                setattr(conf, key, value)

    def load(self) -> None:
        try:
            self.profiles = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(self.profiles, dict):
                self.profiles = {}
        except (OSError, ValueError):
            self.profiles = {}

    def save(self) -> None:
        self.path.write_text(json.dumps(self.profiles, indent=2), encoding="utf-8")

    def save_profile(self, name: str, values: dict[str, Any]) -> None:
        self.profiles[name] = dict(values)
        self.save()
