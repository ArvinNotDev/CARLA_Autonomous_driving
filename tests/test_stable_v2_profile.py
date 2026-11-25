import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stable_v2_is_the_default_runtime_profile_without_mutating_its_values():
    profiles = json.loads((ROOT / "runtime_profiles.json").read_text(encoding="utf-8"))

    assert profiles["_active_profile"] == "StableV2"
    assert "CARLA_V1_LowGPU" not in profiles

    stable = profiles["StableV2"]
    assert stable["CAMERA_IMAGE_WIDTH"] == 512
    assert stable["CAMERA_IMAGE_HEIGHT"] == 320
    assert stable["CAMERA_SENSOR_TICK"] == 0.025
    assert stable["ENABLE_SEMANTIC_CAMERA"] is True
    assert stable["ENABLE_ALT_CAMERA"] is True
    assert stable["CONTROL_LOOP_HZ"] == 40
    assert stable["DEBUG_PANEL_HZ"] == 40
    assert stable["ACCELERATION_MPS2"] == 2.5
    assert stable["MIN_SPEED_KMH"] == 8.0
    assert stable["MAX_SPEED_KMH"] == 25.0
    assert stable["MIN_SPEED_KMH"] < stable["MAX_SPEED_KMH"]
    assert "FIXED_THROTTLE" not in stable
    assert "TARGET_SPEED_KMH" not in stable
    assert stable["JUNCTION_DETECTION_MODE"] == "crosswalk_roi"
