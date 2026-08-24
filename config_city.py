from __future__ import annotations
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


# -------------------------------------------------
# Optional JSON config override
# -------------------------------------------------
CHANGE_WITH_JSON = True

# The editor saves ROI here.
ROI_CONFIG_PATH = "Forza.json"

# Optional fallback if you still have older config files.
LEGACY_CONFIG_PATHS = ["carla.json"]

def _apply_dict_overrides(data: dict) -> None:
    for key, value in data.items():
        globals()[key] = value

def apply_json_override() -> None:
    if not CHANGE_WITH_JSON:
        return

    paths_to_try = [ROI_CONFIG_PATH] + LEGACY_CONFIG_PATHS
    for path in paths_to_try:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                _apply_dict_overrides(data)
            return
        except Exception:
            continue


# -------------------------------------------------
# Runtime / CARLA
# -------------------------------------------------
HOST = "localhost"
PORT = 2000
TIMEOUT = 20.0

VEHICLE_BLUEPRINT = "vehicle.tesla.model3"

# -------------------------------------------------
# Camera
# -------------------------------------------------
CAMERA_IMAGE_WIDTH = 350
CAMERA_IMAGE_HEIGHT = 210
CAMERA_FOV = 120
CAMERA_SENSOR_TICK = 0.03

CAMERA_X = 1.4
CAMERA_Y = 0.0
CAMERA_Z = 2.2
CAMERA_PITCH_DEG = -25.0
CAMERA_YAW_DEG = 0.0
CAMERA_ROLL_DEG = 0.0

ALT_CAMERA_X = 1.4
ALT_CAMERA_Y = 0.0
ALT_CAMERA_Z = 2.2
ALT_CAMERA_PITCH_DEG = -25.0


# -------------------------------------------------
# UI / stream
# -------------------------------------------------
AUTO_MODE_DEFAULT = True
SHOW_OPENCV_WINDOW = True
STREAM_HOST = "0.0.0.0"
STREAM_PORT = 5000

debug_frame_buffer = None

# -------------------------------------------------
# PID controller
# -------------------------------------------------
FIXED_THROTTLE = 0.13
KP = 0.015
KI = 0.0000
KD = 0.002
STEER_LIMIT = 0.75
MAX_STEER_STEP = 0.10

# -------------------------------------------------
# Vision model
# -------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolopv2.onnx")
INPUT_WIDTH = 320
INPUT_HEIGHT = 192
DRIVABLE_OUTPUT_INDEX = 5

LANE_PROB_THRESHOLD = 0.50
LANE_PROB_THRESHOLD_FALLBACK = 0.35
LANE_PROB_THRESHOLD_MIN = 0.22

MIN_SIDE_PIXELS = 12
FALLBACK_LANE_OFFSET_RATIO = 0.16
LANE_CENTER_SMOOTH_ALPHA = 0.30

# -------------------------------------------------
# ROI values used by the editor
# normalized [0, 1]
# -------------------------------------------------
RL_TOP_ROI = 0.00
RL_BOTTOM_ROI = 1.00
RL_LEFT_ROI = 0.50
RL_RIGHT_ROI = 1.00

LL_TOP_ROI = 0.00
LL_BOTTOM_ROI = 1.00
LL_LEFT_ROI = 0.00
LL_RIGHT_ROI = 0.50

CW_TOP_ROI = 0.00
CW_BOTTOM_ROI = 1.00
CW_LEFT_ROI = 0.00
CW_RIGHT_ROI = 1.00

# -------------------------------------------------
# Advanced settings used by the editor
# -------------------------------------------------
LANE_THRESHOLD = 180
CROSSWALK_THRESHOLD = 180
CROSSWALK_SLEEP = 3.0
CROSSWALK_THRESH_SPEND = 8.0
RUN_LVL = "MOVE"
LANE_CHANGE_DEBOUNCE_SECONDS = 2
LANE_CHANGE_ANGLE_CHECK_EVERY = 2
LANE_CHANGE_LINE_ANGLE_THRESHOLD_DEG = 20.0
LANE_CHANGE_ANGLE_COOLDOWN_SECONDS = 5.0


# -------------------------------------------------
# Trajectory
# -------------------------------------------------

COMMANDS = (
    "LANE_FOLLOW",
    "LEFT",
    "RIGHT",
    "STRAIGHT",
)

COMMAND_TO_IDX = {name: i for i, name in enumerate(COMMANDS)}

model_image_size = (224, 224)
n_waypoints: int = 5



@dataclass(frozen=True)
class PipelineConfig:
    # CARLA connection
    host: str = "127.0.0.1"
    port: int = 2000
    timeout_s: float = 20.0

    # Towns
    town_names: Tuple[str, ...] = (
        "Town01",
        "Town02",
        "Town03",
        "Town04",
        "Town05",
        "Town10HD",
    )

    # Deterministic collection
    fixed_delta_seconds: float = 0.03
    sensor_tick: float = 0.03
    weather_presets: Tuple[str, ...] = (
        "ClearNoon",
        "CloudyNoon",
        "WetNoon",
        "WetCloudyNoon",
        "SoftRainNoon",
        "HardRainNoon",
    )
    # CAMERA_X = 1.4
    # CAMERA_Y = 0.0
    # CAMERA_Z = 2.2
    # CAMERA_PITCH_DEG = -25.0
    # CAMERA_YAW_DEG = 0.0
    # CAMERA_ROLL_DEG = 0.0
    # Camera: rear third-person view
    image_width: int = 320
    image_height: int = 192
    camera_fov: float = 100.0
    camera_x: float = 1.4
    camera_y: float = 0.0
    camera_z: float = 2.2
    camera_pitch: float = -25.0

    # Training image size
    model_image_size: Tuple[int, int] = (224, 224)

    # Trajectory targets
    waypoint_distances_m: Tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 25.0)
    n_waypoints: int = 5
    target_horizon_m: float = 25.0

    # Collection windows
    pre_junction_start_m: float = 60.0
    post_junction_clear_m: float = 70.0
    lane_follow_min_next_junction_m: float = 120.0
    lane_follow_duration_s: float = 90.0

    # Collection tuning
    episode_duration_s: float = 120.0
    record_every_n_ticks: int = 2
    min_record_distance_m: float = 1.0

    # Maneuver inference thresholds
    straight_yaw_thresh_deg: float = 10.0
    straight_lateral_thresh_m: float = 1.5
    turn_yaw_thresh_deg: float = 15.0
    turn_lateral_thresh_m: float = 2.0

    # Control
    target_speed_kmh: float = 25.0
    junction_lookahead_step_m: float = 1.0
    junction_scan_max_m: float = 250.0

    # Dataset / training
    batch_size: int = 32
    num_workers: int = 0
    lr: float = 3e-4
    weight_decay: float = 1e-5
    epochs: int = 100
    val_split: float = 0.15
    grad_clip_norm: float = 1.0
    seed: int = 42

    # Output directories
    root_dir: Path = Path("carla_pipeline_data_new")

    @property
    def raw_dir(self) -> Path:
        return self.root_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.root_dir / "processed"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root_dir / "checkpoints"

    @property
    def debug_dir(self) -> Path:
        return self.root_dir / "debug"
    
apply_json_override()
