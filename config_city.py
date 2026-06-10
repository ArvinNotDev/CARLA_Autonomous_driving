import json
import os

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
CAMERA_IMAGE_WIDTH = 320
CAMERA_IMAGE_HEIGHT = 192
CAMERA_FOV = 90
CAMERA_SENSOR_TICK = 0.02

CAMERA_X = 1.4
CAMERA_Y = 0.0
CAMERA_Z = 1.8
CAMERA_PITCH_DEG = -20.0
CAMERA_YAW_DEG = 0.0
CAMERA_ROLL_DEG = 0.0

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
FIXED_THROTTLE = 0.15
KP = 0.008
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


apply_json_override()