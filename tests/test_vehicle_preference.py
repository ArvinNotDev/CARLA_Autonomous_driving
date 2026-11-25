from __future__ import annotations

import json

from runtime_settings import RuntimeSettings
from utils.vehicle_selection import vehicle_spawn_candidates


def test_last_vehicle_is_loaded_and_persisted_without_losing_profiles(tmp_path):
    path = tmp_path / "runtime_profiles.json"
    original = {
        "_active_profile": "StableV2",
        "StableV2": {"KP": 0.015, "CAMERA_IMAGE_WIDTH": 512},
    }
    path.write_text(json.dumps(original), encoding="utf-8")

    assert RuntimeSettings.load_last_vehicle(path=path) == "vehicle.tesla.model3"

    RuntimeSettings.persist_last_vehicle("vehicle.audi.tt", path=path)
    saved = json.loads(path.read_text(encoding="utf-8"))

    assert saved["_last_vehicle_blueprint"] == "vehicle.audi.tt"
    assert saved["_active_profile"] == "StableV2"
    assert saved["StableV2"]["VEHICLE_BLUEPRINT"] == "vehicle.audi.tt"
    assert saved["StableV2"]["KP"] == 0.015


def test_last_vehicle_load_ignores_missing_or_invalid_preference(tmp_path):
    missing = tmp_path / "missing.json"
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{not-json", encoding="utf-8")

    assert RuntimeSettings.load_last_vehicle(path=missing) == "vehicle.tesla.model3"
    assert RuntimeSettings.load_last_vehicle("vehicle.lincoln.mkz", path=invalid) == "vehicle.lincoln.mkz"


def test_vehicle_spawn_prefers_last_selection_then_tesla_fallback():
    available = [
        ("vehicle.audi.tt", "Audi"),
        ("vehicle.tesla.model3", "Tesla"),
        ("vehicle.lincoln.mkz", "Lincoln"),
    ]
    assert vehicle_spawn_candidates(
        available, "vehicle.audi.tt"
    ) == [
        "vehicle.audi.tt", "vehicle.tesla.model3", "vehicle.lincoln.mkz"
    ]
    assert vehicle_spawn_candidates(
        available, "vehicle.not.available"
    ) == [
        "vehicle.tesla.model3", "vehicle.audi.tt", "vehicle.lincoln.mkz"
    ]


def test_vehicle_spawn_uses_first_available_if_tesla_is_absent():
    available = [
        ("vehicle.audi.tt", "Audi"),
        ("vehicle.lincoln.mkz", "Lincoln"),
    ]
    assert vehicle_spawn_candidates(
        available, "vehicle.not.available"
    ) == ["vehicle.audi.tt", "vehicle.lincoln.mkz"]
