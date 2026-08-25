from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import config_city as conf


PROFILE_PATH = Path("runtime_profiles.json")


SETTING_SPECS = [
    ("AUTO_MODE_DEFAULT", "Auto mode", "bool", None, True),
    ("FIXED_THROTTLE", "Fixed throttle", "float", (0.0, 1.0, 0.01), True),
    ("KP", "PID Kp", "float", (0.0, 2.0, 0.001), True),
    ("KI", "PID Ki", "float", (0.0, 2.0, 0.001), True),
    ("KD", "PID Kd", "float", (0.0, 2.0, 0.001), True),
    ("STEER_LIMIT", "Steer limit", "float", (0.0, 1.0, 0.01), True),
    ("MAX_STEER_STEP", "Max steer step", "float", (0.0, 1.0, 0.01), True),
    ("TARGET_SPEED_KMH", "Target speed (km/h)", "float", (0.0, 120.0, 1.0), True),
    ("TRAJECTORY_INFERENCE_INTERVAL_SECONDS", "Trajectory inference interval (s)", "float", (0.03, 2.0, 0.01), True),
    ("VISION_INFERENCE_INTERVAL_SECONDS", "Line model interval (s)", "float", (0.03, 2.0, 0.01), True),
    ("TRAJECTORY_STEER_GAIN", "Trajectory steer gain", "float", (0.0, 3.0, 0.05), True),
    ("TRAJECTORY_MAX_STEER", "Trajectory max steer", "float", (0.0, 1.0, 0.01), True),
    ("TRAJECTORY_DEBUG_SCALE", "Trajectory debug scale", "float", (0.1, 10.0, 0.1), True),
    ("JUNCTION_STATIC_THROTTLE", "Junction lead-in throttle", "float", (0.0, 1.0, 0.01), True),
    ("JUNCTION_STATIC_TIMEOUT_SECONDS", "Static segment timeout (s)", "float", (1.0, 60.0, 1.0), True),
    ("JUNCTION_ENTRY_DISTANCE_M", "Junction entry distance (m)", "float", (0.0, 50.0, 0.5), True),
    ("JUNCTION_RIGHT_TURN_DISTANCE_M", "Right turn distance (m)", "float", (0.0, 30.0, 0.5), True),
    ("JUNCTION_LEFT_STRAIGHT_DISTANCE_M", "Left straight distance (m)", "float", (0.0, 40.0, 0.5), True),
    ("JUNCTION_LEFT_TURN_DISTANCE_M", "Left turn distance (m)", "float", (0.0, 30.0, 0.5), True),
    ("JUNCTION_STRAIGHT_DISTANCE_M", "Straight distance (m)", "float", (0.0, 30.0, 0.5), True),
    ("JUNCTION_TRAJECTORY_WINDOW_SECONDS", "Trajectory takeover duration (s)", "float", (0.5, 60.0, 0.5), True),
    ("DRIVABLE_RECOVERY_ENABLED", "Drivable-area recovery", "bool", None, True),
    ("DRIVABLE_RECOVERY_ERROR_THRESHOLD", "Recovery error threshold (px)", "float", (1.0, 300.0, 1.0), True),
    ("DRIVABLE_RECOVERY_WINDOW_SECONDS", "Recovery window (s)", "float", (0.5, 30.0, 0.5), True),
    ("DRIVABLE_RECOVERY_STEER", "Recovery steering", "float", (0.05, 1.0, 0.05), True),
    ("DRIVABLE_RECOVERY_THROTTLE", "Recovery throttle", "float", (0.0, 1.0, 0.01), True),
    ("INTERSECTION_CHECK_INTERVAL_SECONDS", "Junction detector interval (s)", "float", (0.05, 5.0, 0.05), True),
    ("LANE_CHANGE_DEBOUNCE_SECONDS", "Lane-change debounce (s)", "float", (0.0, 30.0, 0.5), True),
    ("LANE_CHANGE_LINE_ANGLE_THRESHOLD_DEG", "Lane-change angle threshold", "float", (0.0, 90.0, 1.0), True),
    ("LANE_CHANGE_PLANNER_CHECK_INTERVAL_SECONDS", "Lane planner interval (s)", "float", (0.05, 5.0, 0.05), True),
    ("CROSSWALK_SLEEP", "Crosswalk pause (s)", "float", (0.0, 30.0, 0.5), True),
    ("LANE_THRESHOLD", "Lane mask threshold", "int", (0, 255, 1), True),
    ("CROSSWALK_THRESHOLD", "Crosswalk threshold", "int", (0, 255, 1), True),
    ("LANE_PROB_THRESHOLD", "Lane probability threshold", "float", (0.0, 1.0, 0.01), True),
    ("LANE_CENTER_SMOOTH_ALPHA", "Lane smoothing", "float", (0.0, 1.0, 0.01), True),
    ("DRIVABLE_PROB_THRESHOLD", "Drivable probability threshold", "float", (0.0, 1.0, 0.01), True),
    ("RL_TOP_ROI", "Right ROI top", "float", (0.0, 1.0, 0.01), True),
    ("RL_BOTTOM_ROI", "Right ROI bottom", "float", (0.0, 1.0, 0.01), True),
    ("RL_LEFT_ROI", "Right ROI left", "float", (0.0, 1.0, 0.01), True),
    ("RL_RIGHT_ROI", "Right ROI right", "float", (0.0, 1.0, 0.01), True),
    ("LL_TOP_ROI", "Left ROI top", "float", (0.0, 1.0, 0.01), True),
    ("LL_BOTTOM_ROI", "Left ROI bottom", "float", (0.0, 1.0, 0.01), True),
    ("LL_LEFT_ROI", "Left ROI left", "float", (0.0, 1.0, 0.01), True),
    ("LL_RIGHT_ROI", "Left ROI right", "float", (0.0, 1.0, 0.01), True),
    ("CW_TOP_ROI", "Drivable/crosswalk ROI top", "float", (0.0, 1.0, 0.01), True),
    ("CW_BOTTOM_ROI", "Drivable/crosswalk ROI bottom", "float", (0.0, 1.0, 0.01), True),
    ("CW_LEFT_ROI", "Drivable/crosswalk ROI left", "float", (0.0, 1.0, 0.01), True),
    ("CW_RIGHT_ROI", "Drivable/crosswalk ROI right", "float", (0.0, 1.0, 0.01), True),
    ("INPUT_WIDTH", "Model input width", "int", (160, 1920, 16), False),
    ("INPUT_HEIGHT", "Model input height", "int", (96, 1080, 16), False),
    ("VISION_DEBUG", "Vision debug overlays", "bool", None, True),
    ("DEBUG_SHOW_ROIS", "Show ROIs", "bool", None, True),
    ("DEBUG_SHOW_LANE_MASK", "Show lane mask", "bool", None, True),
    ("DEBUG_SHOW_DRIVABLE_AREA", "Show drivable area", "bool", None, True),
    ("DEBUG_SHOW_TRAJECTORY", "Show trajectory", "bool", None, True),
    ("DEBUG_SHOW_FPS", "Show FPS", "bool", None, True),
    ("DEBUG_SHOW_GPS", "Show GPS/navigation", "bool", None, True),
    ("CAMERA_IMAGE_WIDTH", "Camera width", "int", (160, 1920, 16), False),
    ("CAMERA_IMAGE_HEIGHT", "Camera height", "int", (120, 1080, 16), False),
    ("CONTROL_LOOP_HZ", "Control loop frequency (Hz)", "float", (10.0, 60.0, 1.0), False),
    ("DEBUG_PANEL_HZ", "Debug panel refresh (Hz)", "float", (5.0, 30.0, 1.0), True),
    ("ENABLE_SEMANTIC_CAMERA", "Enable semantic camera", "bool", None, False),
    ("ENABLE_ALT_CAMERA", "Enable alternate camera", "bool", None, False),
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
