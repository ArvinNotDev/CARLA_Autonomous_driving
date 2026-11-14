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
    ("TRAJECTORY_MODEL_ENABLED", "Enable trajectory model", "bool", None, False),
    ("TRAJECTORY_INFERENCE_INTERVAL_SECONDS", "Trajectory inference interval (s)", "float", (0.03, 2.0, 0.01), True),
    ("VISION_INFERENCE_INTERVAL_SECONDS", "Line model interval (s)", "float", (0.01, 2.0, 0.005), True),
    ("TRAJECTORY_STEER_GAIN", "Trajectory steer gain", "float", (0.0, 3.0, 0.05), True),
    ("TRAJECTORY_MAX_STEER", "Trajectory max steer", "float", (0.0, 1.0, 0.01), True),
    ("TRAJECTORY_DEBUG_SCALE", "Trajectory debug scale", "float", (0.1, 10.0, 0.1), True),
    ("JUNCTION_STATIC_THROTTLE", "Junction lead-in throttle", "float", (0.0, 1.0, 0.01), True),
    ("JUNCTION_STATIC_TIMEOUT_SECONDS", "Static segment timeout (s)", "float", (1.0, 60.0, 1.0), True),
    ("JUNCTION_ENTRY_DISTANCE_M", "Junction entry distance (m)", "float", (0.0, 50.0, 0.5), True),
    ("JUNCTION_MOVEMENT_SEQUENCES", "Junction movement sequences", "sequence", None, True),
    ("JUNCTION_TRAJECTORY_WINDOW_SECONDS", "Trajectory takeover duration (s)", "float", (0.5, 60.0, 0.5), True),
    ("JUNCTION_DETECTION_MODE", "Junction detection method", "choice", ("model", "crosswalk_roi"), False),
    ("JUNCTION_ROI_MIN_LINE_LENGTH_RATIO", "ROI line minimum width ratio", "float", (0.10, 0.95, 0.05), True),
    ("JUNCTION_ROI_MAX_LINE_ANGLE_DEG", "ROI line angle tolerance (deg)", "float", (1.0, 30.0, 1.0), True),
    ("JUNCTION_ROI_CONFIRM_FRAMES", "ROI line confirmation frames", "int", (1, 3, 1), True),
    ("JUNCTION_ROI_REARM_CLEAR_SECONDS", "Clear time before next junction (s)", "float", (0.2, 5.0, 0.1), True),
    ("INTERSECTION_CHECK_INTERVAL_SECONDS", "Junction detector interval (s)", "float", (0.05, 5.0, 0.05), True),
    ("LANE_CHANGE_DEBOUNCE_SECONDS", "Lane-change debounce (s)", "float", (0.0, 30.0, 0.5), True),
    ("LANE_CHANGE_PLANNER_CHECK_INTERVAL_SECONDS", "Lane planner interval (s)", "float", (0.05, 5.0, 0.05), True),
    ("LANE_CHANGE_VERIFY_DISTANCE_M", "Lane-change approach distance (m)", "float", (0.1, 10.0, 0.1), True),
    ("LANE_CHANGE_STEER_DISTANCE_M", "Distance per lane-change attempt (m)", "float", (0.25, 15.0, 0.25), True),
    ("LANE_CHANGE_STEER_VALUE", "Lane-change steering magnitude", "float", (0.01, 1.0, 0.01), True),
    ("POST_JUNCTION_LANE_CHANGE_DELAY_SECONDS", "Wait after junction before lane change (s)", "float", (0.0, 30.0, 0.5), True),
    ("CROSSWALK_SLEEP", "Crosswalk pause (s)", "float", (0.0, 30.0, 0.5), True),
    ("LANE_THRESHOLD", "Lane mask threshold", "int", (0, 255, 1), True),
    ("CROSSWALK_THRESHOLD", "Crosswalk threshold", "int", (0, 255, 1), True),
    ("LANE_PROB_THRESHOLD", "Lane probability threshold", "float", (0.0, 1.0, 0.01), True),
    ("LANE_CENTER_SMOOTH_ALPHA", "Lane smoothing", "float", (0.0, 1.0, 0.01), True),
    ("DRIVABLE_PROB_THRESHOLD", "Drivable probability threshold", "float", (0.0, 1.0, 0.01), True),
    ("DRIVABLE_RECOVERY_ENABLED", "Enable drivable recovery", "bool", None, True),
    ("DRIVABLE_RECOVERY_ERROR_THRESHOLD", "Recovery error threshold (px)", "float", (0.0, 512.0, 1.0), True),
    ("DRIVABLE_RECOVERY_WINDOW_SECONDS", "Recovery window (s)", "float", (0.1, 60.0, 0.5), True),
    ("DRIVABLE_RECOVERY_STEER", "Recovery steering", "float", (0.0, 1.0, 0.01), True),
    ("DRIVABLE_RECOVERY_THROTTLE", "Recovery throttle", "float", (0.0, 1.0, 0.01), True),
    ("DRIVABLE_RECOVERY_LIDAR_GATE_ENABLED", "Enable LiDAR obstacle suppression", "bool", None, True),
    ("DRIVABLE_RECOVERY_LIDAR_MIN_HEIGHT_M", "LiDAR minimum object height (m)", "float", (0.0, 5.0, 0.05), True),
    ("DRIVABLE_RECOVERY_LIDAR_MAX_HEIGHT_M", "LiDAR maximum object height (m)", "float", (0.1, 20.0, 0.1), True),
    ("DRIVABLE_RECOVERY_LIDAR_MIN_SIDE_DISTANCE_M", "LiDAR minimum side distance (m)", "float", (0.0, 10.0, 0.1), True),
    ("DRIVABLE_RECOVERY_LIDAR_MAX_SIDE_DISTANCE_M", "LiDAR maximum side distance (m)", "float", (0.1, 30.0, 0.5), True),
    ("DRIVABLE_RECOVERY_LIDAR_REAR_EXTENT_M", "LiDAR side scan reach behind vehicle (m)", "float", (0.0, 50.0, 0.5), True),
    ("DRIVABLE_RECOVERY_LIDAR_MAX_FORWARD_DISTANCE_M", "LiDAR side scan reach ahead (m)", "float", (1.0, 100.0, 1.0), True),
    ("DRIVABLE_RECOVERY_LIDAR_MIN_POINTS", "LiDAR minimum object points", "int", (1, 1000, 1), True),
    ("DRIVABLE_RECOVERY_LIDAR_MAX_SCAN_AGE_SECONDS", "LiDAR maximum scan age (s)", "float", (0.05, 2.0, 0.05), True),
    ("LIDAR_ENABLED", "Enable LiDAR sensor", "bool", None, True),
    ("LIDAR_DISPLAY_RANGE_M", "LiDAR radar display range (m)", "float", (5.0, 200.0, 5.0), True),
    ("LIDAR_RANGE_M", "LiDAR sensor range (restart required)", "float", (10.0, 200.0, 5.0), False),
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
    ("DW_TOP_ROI", "Road-width ROI top", "float", (0.0, 1.0, 0.01), True),
    ("DW_BOTTOM_ROI", "Road-width ROI bottom", "float", (0.0, 1.0, 0.01), True),
    ("DW_LEFT_ROI", "Road-width ROI left", "float", (0.0, 1.0, 0.01), True),
    ("DW_RIGHT_ROI", "Road-width ROI right", "float", (0.0, 1.0, 0.01), True),
    ("VISION_DEBUG", "Vision debug overlays", "bool", None, True),
    ("DEBUG_SHOW_ROIS", "Show ROIs", "bool", None, True),
    ("DEBUG_SHOW_LANE_MASK", "Show lane mask", "bool", None, True),
    ("DEBUG_SHOW_DRIVABLE_AREA", "Show drivable area", "bool", None, True),
    ("DEBUG_SHOW_TRAJECTORY", "Show trajectory", "bool", None, True),
    ("CAMERA_X", "Camera X", "float", (-20.0, 20.0, 0.1), True),
    ("CAMERA_Y", "Camera Y", "float", (-20.0, 20.0, 0.1), True),
    ("CAMERA_Z", "Camera Z", "float", (-20.0, 20.0, 0.1), True),
    ("CAMERA_FOV", "Camera FOV", "float", (10.0, 179.0, 1.0), False),
    ("CAMERA_IMAGE_WIDTH", "Camera width", "int", (160, 1920, 16), False),
    ("CAMERA_IMAGE_HEIGHT", "Camera height", "int", (120, 1080, 16), False),
    ("CAMERA_SENSOR_TICK", "RGB camera interval (s)", "float", (0.01, 1.0, 0.005), False),
    ("CONTROL_LOOP_HZ", "Control loop frequency (Hz)", "float", (10.0, 60.0, 1.0), True),
    ("DEBUG_PANEL_HZ", "Debug panel refresh (Hz)", "float", (5.0, 60.0, 1.0), True),
    ("ENABLE_SEMANTIC_CAMERA", "Enable semantic camera", "bool", None, False),
    ("ENABLE_ALT_CAMERA", "Enable alternate camera", "bool", None, False),
    ("MODEL_PATH", "YOLO26s-sem ONNX Model Path", "str", None, False),
]


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
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

    @classmethod
    def persist_values(cls, values: dict[str, Any], path: Path = PROFILE_PATH) -> None:
        """Persist live changes into the active profile without reloading config."""
        try:
            profiles = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(profiles, dict):
                profiles = {}
        except (OSError, ValueError):
            profiles = {}

        active = profiles.get("_active_profile")
        profile_name = (
            active
            if isinstance(active, str) and active and active in profiles
            else "default"
        )
        profile = profiles.get(profile_name)
        if not isinstance(profile, dict):
            profile = {}
        profile.update({key: _json_value(value) for key, value in values.items()})
        profiles[profile_name] = profile
        path.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
