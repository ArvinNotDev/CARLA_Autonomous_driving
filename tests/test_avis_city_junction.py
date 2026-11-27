from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from avis_city_junction import (
    AvisCityJunctionController,
    AvisJunctionSignOnnxClassifier,
    DEFAULT_CITY_MANEUVER_SEQUENCES,
    DEFAULT_CITY_SIGN_CLASS_MAP,
    DEFAULT_CITY_SIGN_CLASSES,
    parse_city_maneuver_sequences,
    parse_city_sign_class_map,
)
from avis_dashboard import (
    AvisDashboardState,
    DASHBOARD_HTML,
    DEFAULT_AVIS_SETTINGS,
    validate_avis_settings,
)
from avis_runtime import AvisCombinedVisionProcessor


def _settings(**overrides):
    result = {
        "city_crosswalk_stop_seconds": 5.0,
        "city_junction_sign_confidence_threshold": 0.55,
        "city_junction_sign_confirm_frames": 2,
        "city_junction_sign_max_age_seconds": 20.0,
        "city_junction_sign_rearm_clear_frames": 2,
        "city_maneuver_segment_timeout_seconds": 20.0,
        "city_maneuver_sequences_json": json.dumps(
            DEFAULT_CITY_MANEUVER_SEQUENCES
        ),
    }
    result.update(overrides)
    return result


def _sign_prediction(label: str, observed_at: float, confidence: float = 0.95):
    return {
        "valid": True,
        "observed_at": observed_at,
        "class_id": DEFAULT_CITY_SIGN_CLASSES.index(label),
        "class_name": label,
        "confidence": confidence,
        "detections": [
            {
                "class_id": DEFAULT_CITY_SIGN_CLASSES.index(label),
                "class_name": label,
                "confidence": confidence,
            }
        ],
        "model_classes": list(DEFAULT_CITY_SIGN_CLASSES),
    }


def _confirm_sign(controller, label: str, settings):
    controller.update(
        None,
        _sign_prediction(label, 1.0),
        enabled=True,
        settings=settings,
        current_speed_kmh=8.0,
        dt=0.05,
        now=1.0,
    )
    return controller.update(
        None,
        _sign_prediction(label, 1.1),
        enabled=True,
        settings=settings,
        current_speed_kmh=8.0,
        dt=0.05,
        now=1.1,
    )


def _crosswalk():
    return {
        "line_present": True,
        "detected": True,
        "line": (10, 100, 310, 104),
        "length_ratio": 0.78,
    }


def _trigger_crosswalk(controller, settings, *, speed=8.0, now=2.0):
    return controller.update(
        _crosswalk(),
        None,
        enabled=True,
        settings=settings,
        current_speed_kmh=speed,
        dt=0.05,
        observation_id=10,
        observation_timestamp=now,
        now=now,
    )


def test_six_sign_classes_are_mapped_without_conflating_normal_and_90_degree_turns():
    mapping = parse_city_sign_class_map(DEFAULT_CITY_SIGN_CLASS_MAP)
    assert mapping["RIGHT"] == {"right"}
    assert mapping["LEFT"] == {"left"}
    assert mapping["STRAIGHT"] == {"straight"}
    assert mapping["STOP"] == {"stop", "stop_sign", "stop_signs"}
    assert "90r" in mapping["RIGHT_90"]
    assert "90l" in mapping["LEFT_90"]
    assert "right" not in mapping["RIGHT_90"]
    assert "left" not in mapping["LEFT_90"]


def test_onnx_metadata_class_names_support_label_to_index_dictionary():
    names = AvisJunctionSignOnnxClassifier._parse_names(
        {"90l": 5, "90r": 4, "stop": 3, "straight": 2, "left": 1, "right": 0}
    )
    assert names == ["right", "left", "straight", "stop", "90r", "90l"]


def test_sign_class_order_validation_requires_exactly_the_six_model_labels():
    settings = validate_avis_settings(
        dict(DEFAULT_AVIS_SETTINGS),
        {
            "city_junction_sign_class_order_json": json.dumps(
                ["90l", "90r", "stop", "straight", "left", "right"]
            )
        },
    )
    assert json.loads(settings["city_junction_sign_class_order_json"]) == [
        "90l", "90r", "stop", "straight", "left", "right"
    ]

    with pytest.raises(ValueError, match="exactly once"):
        validate_avis_settings(
            dict(DEFAULT_AVIS_SETTINGS),
            {
                "city_junction_sign_class_order_json": json.dumps(
                    ["right", "left", "straight", "stop", "90r", "90r"]
                )
            },
        )


def test_avis_default_normal_turn_sequences_match_active_carla_stable_v2_profile():
    root = Path(__file__).resolve().parents[1]
    carla_profiles = json.loads(
        (root / "runtime_profiles.json").read_text(encoding="utf-8")
    )
    carla_sequences = carla_profiles["StableV2"]["JUNCTION_MOVEMENT_SEQUENCES"]
    avis_sequences = json.loads(DEFAULT_AVIS_SETTINGS["city_maneuver_sequences_json"])

    for route in ("LEFT", "RIGHT", "STRAIGHT"):
        assert len(avis_sequences[route]) == len(carla_sequences[route])
        for avis_step, carla_step in zip(avis_sequences[route], carla_sequences[route]):
            assert avis_step["distance_m"] == pytest.approx(carla_step["distance_m"])
            assert avis_step["steering_value"] == pytest.approx(carla_step["steering_value"])

    # 90-degree sign classes are independently editable but start from the
    # matching CARLA left/right route until calibrated separately.
    assert avis_sequences["LEFT_90"] == avis_sequences["LEFT"]
    assert avis_sequences["RIGHT_90"] == avis_sequences["RIGHT"]


def test_maneuver_json_accepts_independent_sequences_for_all_turn_types():
    sequences = {
        "STRAIGHT": [{"distance_m": 4.0, "steering_value": 0.0}],
        "LEFT": [{"distance_m": 5.0, "steering_value": -0.25}],
        "RIGHT": [{"distance_m": 6.0, "steering_value": 0.25}],
        "LEFT_90": [{"distance_m": 7.0, "steering_value": -0.8}],
        "RIGHT_90": [{"distance_m": 8.0, "steering_value": 0.8}],
    }
    parsed = parse_city_maneuver_sequences(json.dumps(sequences))
    assert parsed["LEFT"][0]["steering_value"] == pytest.approx(-0.25)
    assert parsed["RIGHT_90"][0]["distance_m"] == pytest.approx(8.0)

    validated = validate_avis_settings(
        dict(DEFAULT_AVIS_SETTINGS),
        {"city_maneuver_sequences_json": json.dumps(sequences)},
    )
    assert json.loads(validated["city_maneuver_sequences_json"]) == sequences

    with pytest.raises(ValueError, match="distance_m"):
        validate_avis_settings(
            dict(DEFAULT_AVIS_SETTINGS),
            {
                "city_maneuver_sequences_json": json.dumps(
                    {"LEFT": [{"distance_m": 0, "steering_value": -0.2}]}
                )
            },
        )


def test_right_left_and_straight_wait_five_seconds_after_the_vehicle_stops():
    for label, expected_route, expected_segments in (
        ("right", "RIGHT", 3),
        ("left", "LEFT", 4),
        ("straight", "STRAIGHT", 1),
    ):
        settings = _settings()
        controller = AvisCityJunctionController()
        _confirm_sign(controller, label, settings)

        braking = _trigger_crosswalk(controller, settings)
        assert braking["phase"] == "BRAKING"
        assert braking["route_type"] == expected_route
        assert braking["braking"] is True

        dwell = controller.update(
            _crosswalk(),
            None,
            enabled=True,
            settings=settings,
            current_speed_kmh=0.2,
            dt=0.05,
            observation_id=11,
            observation_timestamp=2.1,
            now=2.1,
        )
        assert dwell["phase"] == "DWELL"
        assert dwell["dwell_remaining_seconds"] == pytest.approx(5.0)

        still_waiting = controller.update(
            None, None,
            enabled=True,
            settings=settings,
            current_speed_kmh=0.0,
            dt=0.05,
            now=6.9,
        )
        assert still_waiting["phase"] == "DWELL"
        assert still_waiting["dwell_remaining_seconds"] == pytest.approx(0.2)

        maneuver = controller.update(
            None, None,
            enabled=True,
            settings=settings,
            current_speed_kmh=0.0,
            dt=0.05,
            now=7.1,
        )
        assert maneuver["phase"] == "MANEUVER"
        assert maneuver["maneuver_active"] is True
        assert maneuver["segment_count"] == expected_segments


@pytest.mark.parametrize(
    ("label", "expected_route", "expected_steer"),
    [
        ("90r", "RIGHT_90", 0.65),
        ("90l", "LEFT_90", -0.5),
    ],
)
def test_90_degree_sign_skips_five_second_dwell_and_starts_turn_when_stopped(
    label, expected_route, expected_steer
):
    settings = _settings(
        city_maneuver_sequences_json=json.dumps(
            {
                "RIGHT_90": [{"distance_m": 0.5, "steering_value": 0.65}],
                "LEFT_90": [{"distance_m": 0.5, "steering_value": -0.5}],
            }
        )
    )
    controller = AvisCityJunctionController()
    _confirm_sign(controller, label, settings)

    braking = _trigger_crosswalk(controller, settings, speed=8.0)
    assert braking["phase"] == "BRAKING"
    assert braking["waiting_for_stop"] is True

    stopped = controller.update(
        None, None,
        enabled=True,
        settings=settings,
        current_speed_kmh=0.2,
        dt=0.05,
        now=2.1,
    )
    assert stopped["phase"] == "MANEUVER"
    assert stopped["dwell_active"] is False
    assert stopped["route_type"] == expected_route
    assert stopped["steering_override"] == pytest.approx(expected_steer)


def test_stop_sign_holds_vehicle_after_dwell_and_never_starts_a_maneuver():
    settings = _settings()
    controller = AvisCityJunctionController()
    _confirm_sign(controller, "stop", settings)

    braking = _trigger_crosswalk(controller, settings)
    assert braking["phase"] == "BRAKING"
    assert braking["route_type"] == "STOP"

    dwell = controller.update(
        None, None,
        enabled=True,
        settings=settings,
        current_speed_kmh=0.2,
        dt=0.05,
        now=2.1,
    )
    assert dwell["phase"] == "DWELL"

    terminal = controller.update(
        None, None,
        enabled=True,
        settings=settings,
        current_speed_kmh=0.0,
        dt=0.05,
        now=7.1,
    )
    assert terminal["phase"] == "STOPPED"
    assert terminal["braking"] is True
    assert terminal["active"] is True
    assert terminal["maneuver_active"] is False
    assert terminal["reason"] == "stop_sign_terminal_stop"

    held = controller.update(
        None, None,
        enabled=True,
        settings=settings,
        current_speed_kmh=0.0,
        dt=0.05,
        now=30.0,
    )
    assert held["phase"] == "STOPPED"
    assert held["braking"] is True


def test_maneuver_segments_advance_by_distance_and_apply_the_next_steering_value():
    settings = _settings(
        city_maneuver_sequences_json=json.dumps(
            {"LEFT": [
                {"distance_m": 1.0, "steering_value": -0.3},
                {"distance_m": 0.5, "steering_value": 0.2},
            ]}
        )
    )
    controller = AvisCityJunctionController()
    _confirm_sign(controller, "left", settings)
    _trigger_crosswalk(controller, settings)
    controller.update(
        None, None,
        enabled=True,
        settings=settings,
        current_speed_kmh=0.2,
        dt=0.05,
        now=2.1,
    )
    first = controller.update(
        None, None,
        enabled=True,
        settings=settings,
        current_speed_kmh=0.0,
        dt=0.05,
        now=7.1,
    )
    assert first["phase"] == "MANEUVER"
    assert first["steering_override"] == pytest.approx(-0.3)

    halfway = controller.update(
        None, None,
        enabled=True,
        settings=settings,
        current_speed_kmh=7.2,
        dt=0.25,
        now=7.35,
    )
    assert halfway["segment_index"] == 1

    second = controller.update(
        None, None,
        enabled=True,
        settings=settings,
        current_speed_kmh=7.2,
        dt=0.25,
        now=7.60,
    )
    assert second["phase"] == "MANEUVER"
    assert second["segment_index"] == 2
    assert second["steering_override"] == pytest.approx(0.2)


def test_combined_vision_processor_publishes_timestamped_junction_sign_output():
    class LaneProcessor:
        def detect(self, semantic_frame, rgb_frame=None):
            return {"debug": {}, "lane_valid": True}

    class SignClassifier:
        def predict(self, frame, *, confidence_threshold):
            assert confidence_threshold == pytest.approx(0.6)
            return {
                "valid": True,
                "class_name": "90r",
                "confidence": 0.91,
                "detections": [{"class_name": "90r", "confidence": 0.91}],
            }

    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    processor = AvisCombinedVisionProcessor(
        LaneProcessor(),
        junction_sign_classifier=SignClassifier(),
        junction_sign_confidence_threshold=0.6,
    )
    result = processor.detect(frame, rgb_frame=frame)
    assert result["junction_sign"]["class_name"] == "90r"
    assert result["junction_sign"]["observed_at"] > 0.0
    assert result["debug"]["junction_sign"]["confidence"] == pytest.approx(0.91)


def test_resnet18_onnx_classifier_uses_class_order_and_normalized_probabilities(
    tmp_path, monkeypatch
):
    model_path = tmp_path / "signs.onnx"
    model_path.write_bytes(b"fake-onnx-placeholder")

    class FakeSession:
        def __init__(self):
            self.logits = np.asarray([[0, 0, 0, 0, 8, 0]], dtype=np.float32)

        def get_inputs(self):
            return [SimpleNamespace(
                name="images", type="tensor(float)", shape=[1, 3, 32, 32]
            )]

        def get_outputs(self):
            return [SimpleNamespace(
                name="logits", type="tensor(float)", shape=[1, 6]
            )]

        def get_modelmeta(self):
            return SimpleNamespace(custom_metadata_map={})

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def run(self, output_names, feeds):
            assert output_names == ["logits"]
            tensor = feeds["images"]
            assert tensor.shape == (1, 3, 32, 32)
            assert tensor.dtype == np.float32
            return [self.logits]

    fake_session = FakeSession()
    fake_ort = SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
        InferenceSession=lambda path, providers: fake_session,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    classifier = AvisJunctionSignOnnxClassifier(
        model_path,
        class_order=["right", "left", "straight", "stop", "90r", "90l"],
        input_size=32,
    )
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    prediction = classifier.predict(frame, confidence_threshold=0.55)

    assert prediction["valid"] is True
    assert prediction["class_id"] == 4
    assert prediction["class_name"] == "90r"
    assert prediction["confidence"] > 0.9
    assert prediction["detections"][0]["class_name"] == "90r"
    assert prediction["providers"] == ["CPUExecutionProvider"]


def test_city_profile_enables_sign_model_and_maneuver_configuration():
    from avis_runtime import load_avis_mode_profile

    city = load_avis_mode_profile("city")
    settings = validate_avis_settings(
        dict(DEFAULT_AVIS_SETTINGS),
        city["settings"],
    )
    assert settings["city_junction_sign_enabled"] is True
    assert settings["junction_sign_model_path"].endswith("sign_detector_deadlock.onnx")
    sequences = json.loads(settings["city_maneuver_sequences_json"])
    assert {"STRAIGHT", "LEFT", "RIGHT", "LEFT_90", "RIGHT_90"} <= set(sequences)


def test_dashboard_exposes_sign_model_and_editable_maneuver_sequences():
    assert "city_junction_sign_enabled" in DASHBOARD_HTML
    assert "junction_sign_model_path" in DASHBOARD_HTML
    assert "city_junction_sign_class_order_json" in DASHBOARD_HTML
    assert "city_maneuver_sequences_json" in DASHBOARD_HTML
    assert "RIGHT_90" in DEFAULT_AVIS_SETTINGS["city_maneuver_sequences_json"]

def test_maneuver_sequences_are_persisted_and_restored_per_profile(tmp_path):
    settings_path = tmp_path / "avis_settings.json"
    profile_path = tmp_path / "avis_profiles.json"
    state = AvisDashboardState(
        settings_path=settings_path,
        profile_path=profile_path,
        default_profile_name="AVIS_StableV2",
    )

    city_sequences = {
        "LEFT": [{"distance_m": 13.0, "steering_value": -0.7}],
        "RIGHT": [{"distance_m": 7.0, "steering_value": 0.6}],
    }
    state.update_settings(
        {"city_maneuver_sequences_json": json.dumps(city_sequences)}
    )
    city_profile = state.save_profile("AVIS_StableV2-city")
    assert city_profile["maneuvers"]["LEFT"] == city_sequences["LEFT"]
    assert json.loads(
        city_profile["settings"]["city_maneuver_sequences_json"]
    ) == city_profile["maneuvers"]

    road_sequences = {
        "LEFT": [{"distance_m": 4.0, "steering_value": -0.25}],
        "RIGHT": [{"distance_m": 16.0, "steering_value": 0.9}],
    }
    state.update_settings(
        {"city_maneuver_sequences_json": json.dumps(road_sequences)}
    )
    road_profile = state.save_profile("AVIS_StableV2-road")
    assert road_profile["maneuvers"]["RIGHT"] == road_sequences["RIGHT"]
    assert road_profile["maneuvers"]["LEFT"] != city_profile["maneuvers"]["LEFT"]

    loaded_city = state.load_profile("AVIS_StableV2-city")
    assert json.loads(loaded_city["city_maneuver_sequences_json"]) == city_profile["maneuvers"]
    assert state.get_profile("AVIS_StableV2-road")["maneuvers"] == road_profile["maneuvers"]

    # The explicit structured field is canonical when importing a profile.
    imported = state.import_profile(
        {
            "schema_version": 1,
            "name": "AVIS_StableV2-imported",
            "settings": {
                "city_maneuver_sequences_json": json.dumps(
                    {"LEFT": [{"distance_m": 99.0, "steering_value": -1.0}]}
                )
            },
            "maneuvers": {
                "LEFT": [{"distance_m": 8.0, "steering_value": -0.4}]
            },
        }
    )
    assert imported["maneuvers"]["LEFT"] == [
        {"distance_m": 8.0, "steering_value": -0.4}
    ]
    assert json.loads(
        imported["settings"]["city_maneuver_sequences_json"]
    ) == imported["maneuvers"]

    # Re-open the persisted profile store to verify the sequences survive restart.
    reloaded = AvisDashboardState(
        settings_path=settings_path,
        profile_path=profile_path,
        default_profile_name="AVIS_StableV2",
    )
    assert reloaded.get_profile("AVIS_StableV2-city")["maneuvers"] == city_profile["maneuvers"]
    assert reloaded.get_profile("AVIS_StableV2-road")["maneuvers"] == road_profile["maneuvers"]


def test_every_checked_in_stable_v2_profile_contains_its_own_maneuver_sequences():
    root = Path(__file__).resolve().parents[1]
    profile_paths = sorted((root / "avis_profiles").glob("AVIS_StableV2*.json"))
    assert profile_paths
    for profile_path in profile_paths:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        assert isinstance(profile.get("maneuvers"), dict), profile_path.name
        assert "city_maneuver_sequences_json" in profile["settings"], profile_path.name
        stored_settings_sequences = json.loads(
            profile["settings"]["city_maneuver_sequences_json"]
        )
        assert profile["maneuvers"] == stored_settings_sequences, profile_path.name

