import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from avis_runtime import (
    AVIS_PROFILE_FIELDS,
    AVIS_PROFILE_NAME,
    AvisEngineClient,
    AvisFpsMonitor,
    AvisSteeringController,
    AvisObstacleLaneChangeController,
    AvisModelObstacleLaneChangeController,
    AvisObstacleLaneClassifier,
    apply_avis_profile,
    load_avis_profile,
    normalize_avis_controls,
    normalize_avis_manual_controls,
    select_avis_target_speed,
    _create_avis_vision_processor,
)

from avis_classical_vision import AvisClassicalVisionProcessor
from avis_adaptive_roi import AvisAdaptiveROIController
from avis_dashboard import (
    DEFAULT_AVIS_SETTINGS,
    AvisDashboardState,
    create_avis_dashboard_app,
    validate_avis_settings,
)


ROOT = Path(__file__).resolve().parents[1]


def test_avis_fps_monitor_calculates_rolling_rate_and_expires_stale_samples():
    meter = AvisFpsMonitor(window_seconds=2.0)
    meter.record(10.0)
    meter.record(10.1)
    meter.record(10.2)
    assert meter.fps(now=10.2) == pytest.approx(10.0)
    assert meter.fps(now=13.0) == pytest.approx(0.0)


def test_avis_fps_monitor_ignores_non_monotonic_timestamps():
    meter = AvisFpsMonitor(window_seconds=2.0)
    meter.record(5.0)
    meter.record(4.0)
    meter.record(5.1)
    assert meter.fps(now=5.1) == pytest.approx(10.0)


def test_avis_profile_is_a_stable_v2_subset_without_carla_only_settings():
    profiles = json.loads(
        (ROOT / "runtime_profiles.json").read_text(encoding="utf-8")
    )
    stable = profiles["StableV2"]
    avis = profiles[AVIS_PROFILE_NAME]

    assert profiles["_active_profile"] == "StableV2"
    assert set(avis) == set(AVIS_PROFILE_FIELDS)
    assert avis["KP"] == stable["KP"]
    assert avis["KI"] == stable["KI"]
    assert avis["KD"] == stable["KD"]
    assert avis["STEER_LIMIT"] == stable["STEER_LIMIT"]
    assert avis["MAX_STEER_STEP"] == stable["MAX_STEER_STEP"]
    assert avis["CONTROL_LOOP_HZ"] == stable["CONTROL_LOOP_HZ"]
    assert stable["MODEL_PATH"] == "models_and_datasets/lane_and_drivable-area_model.pt"
    assert avis["MODEL_PATH"] == "models_and_datasets/Avis_models/lane_and_drivable-area_model_Avis.pt"
    assert (ROOT / avis["MODEL_PATH"]).is_file()
    assert AvisObstacleLaneClassifier.DEFAULT_PATH == (
        ROOT / "models_and_datasets" / "Avis_models" / "obstacle_and_lane_Avis_model.onnx"
    )
    assert AvisObstacleLaneClassifier.DEFAULT_PATH.is_file()
    assert avis["MIN_SPEED_KMH"] == stable["MIN_SPEED_KMH"]
    assert avis["MAX_SPEED_KMH"] == stable["MAX_SPEED_KMH"]
    assert "LIDAR_ENABLED" not in avis
    assert "CAMERA_X" not in avis
    assert "JUNCTION_MOVEMENT_SEQUENCES" not in avis
    assert "TRAJECTORY_MODEL_ENABLED" not in avis


def test_load_avis_profile_rejects_incomplete_profile(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps({AVIS_PROFILE_NAME: {"KP": 0.015}}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="incomplete"):
        load_avis_profile(path)


def test_apply_avis_profile_only_applies_supported_keys():
    config = SimpleNamespace(
        KP=0.0,
        MODEL_PATH="models_and_datasets/lane_and_drivable-area_model.pt",
        LIDAR_ENABLED=True,
    )
    apply_avis_profile(
        {
            "KP": 0.02,
            "MODEL_PATH": "models_and_datasets/Avis_models/lane_and_drivable-area_model_Avis.pt",
            "LIDAR_ENABLED": False,
        },
        config,
    )

    assert config.KP == 0.02
    assert config.MODEL_PATH == "models_and_datasets/Avis_models/lane_and_drivable-area_model_Avis.pt"
    assert config.LIDAR_ENABLED is True


@pytest.mark.parametrize(
    ("throttle", "steering", "expected"),
    [
        (0.5, 0.25, (50, 25)),
        (2.0, -2.0, (100, -100)),
        (-1.0, 0.0, (0, 0)),
        (0.0, 0.755, (0, 76)),
    ],
)
def test_normalize_avis_controls(throttle, steering, expected):
    assert normalize_avis_controls(throttle, steering) == expected


def test_normalize_avis_controls_rejects_non_finite_values():
    with pytest.raises(ValueError):
        normalize_avis_controls(float("nan"), 0.0)


def test_avis_steering_controller_preserves_stable_v2_rate_limit():
    controller = AvisSteeringController(
        kp=0.015,
        ki=0.0,
        kd=0.002,
        steer_limit=0.75,
        max_steer_step=0.10,
    )

    first = controller.update(100.0)
    second = controller.update(100.0)

    assert first == pytest.approx(0.10)
    assert second == pytest.approx(0.20)
    assert abs(controller.update(10000.0)) <= 0.75


def test_avis_speed_target_matches_stable_v2_turn_reduction():
    assert select_avis_target_speed(0.0, 8.0, 25.0) == pytest.approx(25.0)
    assert select_avis_target_speed(0.15, 8.0, 25.0) == pytest.approx(25.0)
    assert select_avis_target_speed(0.70, 8.0, 25.0) == pytest.approx(8.0)


class FakeSocket:
    def __init__(self):
        self.sent = []
        self.closed = False

    def sendall(self, payload):
        if self.closed:
            raise OSError("socket is closed")
        self.sent.append(payload)

    def close(self):
        self.closed = True


class FakeCar:
    def __init__(self):
        self.sock = FakeSocket()
        self.speed_commands = []
        self.steering_commands = []
        self.sensor_angle = None
        self.stopped = False

    def connect(self, host, port):
        self.host = host
        self.port = port
        return True

    def setSpeed(self, value):
        self.speed_commands.append(value)

    def setSteering(self, value):
        self.steering_commands.append(value)

    def setSensorAngle(self, value):
        self.sensor_angle = value

    def getData(self):
        return None

    def getImage(self):
        return None

    def getSpeed(self):
        return 0

    def getSensors(self):
        return [100, 100, 100]

    def stop(self):
        self.stopped = True


def test_avis_client_maps_controls_and_closes_socket_once():
    api = SimpleNamespace(Car=FakeCar)
    client = AvisEngineClient(api, "127.0.0.1", 25001)

    client.set_sensor_angle(40)
    client.send_control(0.5, -0.25)
    fake_car = client.car
    client.close()
    client.close()

    assert fake_car.sensor_angle == 40
    assert fake_car.speed_commands == [50, 0]
    assert fake_car.steering_commands == [-25, 0]
    assert fake_car.sock.sent[-1] == b"stop"
    assert fake_car.sock.closed is True
    assert fake_car.stopped is False


def test_avis_client_rejects_failed_connection():
    class DisconnectedCar(FakeCar):
        def connect(self, host, port):
            return False

    with pytest.raises(ConnectionError, match="Could not connect"):
        AvisEngineClient(SimpleNamespace(Car=DisconnectedCar), "127.0.0.1", 25001)



def test_normalize_avis_manual_controls_supports_forward_and_reverse():
    assert normalize_avis_manual_controls(0.5, -0.25) == (50, -25)
    assert normalize_avis_manual_controls(-0.4, 0.6) == (-40, 60)


def test_avis_dashboard_persists_only_avis_settings_and_validates_updates(tmp_path):
    settings_path = tmp_path / "avis_settings.json"
    state = AvisDashboardState(settings_path=settings_path)

    updated = state.update_settings(
        {"vision_backend": "classical", "canny_low": 45, "canny_high": 135}
    )
    assert updated["vision_backend"] == "classical"
    assert json.loads(settings_path.read_text(encoding="utf-8"))["canny_low"] == 45

    restored = AvisDashboardState(settings_path=settings_path)
    restored_settings, _ = restored.settings_snapshot()
    assert restored_settings["vision_backend"] == "classical"
    assert restored_settings["canny_high"] == 135

    with pytest.raises(ValueError, match="Maximum speed"):
        state.update_settings(
            {"min_speed_kmh": 30, "max_speed_kmh": 20},
            persist=False,
        )


def test_avis_dashboard_manual_api_accepts_reverse_and_emergency_stop(tmp_path):
    state = AvisDashboardState(settings_path=tmp_path / "avis_settings.json")
    client = create_avis_dashboard_app(state).test_client()

    assert client.get("/").status_code == 200
    assert client.post("/api/mode", json={"mode": "manual"}).status_code == 200

    response = client.post(
        "/api/manual",
        json={"throttle": -0.35, "steering": 0.25},
    )
    assert response.status_code == 200
    mode, throttle, steering, active = state.drive_state()
    assert mode == "manual"
    assert throttle == pytest.approx(-0.35)
    assert steering == pytest.approx(0.25)
    assert active is True

    assert client.post("/api/stop", json={}).status_code == 200
    assert client.post(
        "/api/manual",
        json={"throttle": 0.5, "steering": 0.0},
    ).status_code == 409
    assert state.drive_state() == ("stopped", 0.0, 0.0, False)


def test_avis_manual_control_deadman_expires_without_dashboard_heartbeat(
    tmp_path, monkeypatch
):
    import avis_dashboard

    now = [100.0]
    monkeypatch.setattr(avis_dashboard.time, "monotonic", lambda: now[0])
    state = AvisDashboardState(
        settings_path=tmp_path / "avis_settings.json",
        manual_timeout_seconds=0.2,
    )
    state.set_mode("manual")
    state.set_manual_controls(0.4, -0.2)
    assert state.drive_state() == ("manual", 0.4, -0.2, True)

    now[0] += 0.3
    assert state.drive_state() == ("manual", 0.0, 0.0, False)


def test_avis_classical_vision_uses_the_profile_ll_and_rl_rectangles():
    import cv2
    import numpy as np

    frame = np.zeros((192, 320, 3), dtype=np.uint8)
    cv2.line(frame, (108, 109), (100, 132), (255, 255, 255), 5)
    cv2.line(frame, (212, 109), (220, 132), (255, 255, 255), 5)

    processor = AvisClassicalVisionProcessor()
    result = processor.detect(None, frame)
    debug = result["debug"]

    assert result["lane_valid"] is True
    # Both detected boundaries must be used, regardless of sticky-side state.
    assert result["lane_type"] == "both"
    assert debug["lane_center"] == pytest.approx(
        (debug["left_x"] + debug["right_x"]) / 2.0
    )
    assert abs((debug["lane_center"] - debug["left_x"]) -
               (debug["right_x"] - debug["lane_center"])) < 1.0
    assert abs(result["error"]) < 5
    assert debug["lane_half_width_px"] == pytest.approx(
        (debug["right_x"] - debug["left_x"]) / 2.0
    )
    assert debug["lane_mask"].shape == frame.shape[:2]
    assert debug["drivable_mask"] is None
    assert np.count_nonzero(debug["lane_mask"]) > 0
    assert debug["combined"].shape == frame.shape

    allowed = np.zeros(frame.shape[:2], dtype=bool)
    ll = DEFAULT_AVIS_SETTINGS
    allowed[
        round(ll["ll_top_roi"] * frame.shape[0]):round(ll["ll_bottom_roi"] * frame.shape[0]),
        round(ll["ll_left_roi"] * frame.shape[1]):round(ll["ll_right_roi"] * frame.shape[1]),
    ] = True
    allowed[
        round(ll["rl_top_roi"] * frame.shape[0]):round(ll["rl_bottom_roi"] * frame.shape[0]),
        round(ll["rl_left_roi"] * frame.shape[1]):round(ll["rl_right_roi"] * frame.shape[1]),
    ] = True
    assert np.count_nonzero(debug["lane_mask"][~allowed]) == 0


@pytest.mark.parametrize("active_side", [None, "left", "right"])
def test_avis_classical_lane_center_uses_midpoint_when_both_boundaries_exist(active_side):
    processor = AvisClassicalVisionProcessor()
    processor.active_side = active_side

    lane_type, lane_center = processor._pick_sticky_lane(
        left_x=100,
        right_x=220,
        frame_center=160.0,
        lane_offset_px=20.0,
    )

    assert lane_type == "both"
    assert lane_center == pytest.approx(160.0)
    assert processor.last_lane_half_width == pytest.approx(60.0)
    assert processor.active_side is None


def test_avis_classical_single_boundary_uses_last_measured_half_lane_width():
    processor = AvisClassicalVisionProcessor()

    # Learn the lane width from a frame where both boundaries are visible.
    processor._pick_sticky_lane(100, 220, 160.0, 20.0)

    lane_type, left_only_center = processor._pick_sticky_lane(
        left_x=100,
        right_x=None,
        frame_center=160.0,
        lane_offset_px=20.0,
    )
    assert lane_type == "only_left"
    assert left_only_center == pytest.approx(160.0)

    processor.active_side = "right"
    lane_type, right_only_center = processor._pick_sticky_lane(
        left_x=None,
        right_x=220,
        frame_center=160.0,
        lane_offset_px=20.0,
    )
    assert lane_type == "only_right"
    assert right_only_center == pytest.approx(160.0)


def test_avis_classical_hough_receives_only_cropped_left_and_right_rois(monkeypatch):
    import cv2
    import numpy as np

    settings = {
        "ll_top_roi": 0.20,
        "ll_bottom_roi": 0.80,
        "ll_left_roi": 0.10,
        "ll_right_roi": 0.30,
        "rl_top_roi": 0.25,
        "rl_bottom_roi": 0.75,
        "rl_left_roi": 0.70,
        "rl_right_roi": 0.90,
    }
    processor = AvisClassicalVisionProcessor(settings)
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    seen_shapes = []

    def fake_hough(binary, **_kwargs):
        seen_shapes.append(binary.shape)
        return np.array([[[10, 0, 10, binary.shape[0] - 1]]], dtype=np.int32)

    monkeypatch.setattr(cv2, "HoughLinesP", fake_hough)
    result = processor.detect(None, frame)
    debug = result["debug"]

    assert seen_shapes == [(120, 64), (100, 64)]
    assert debug["left_hough_segments"] == 1
    assert debug["right_hough_segments"] == 1

    allowed = np.zeros(frame.shape[:2], dtype=bool)
    allowed[40:160, 32:96] = True
    allowed[50:150, 224:288] = True
    assert np.count_nonzero(debug["lane_mask"][~allowed]) == 0



@pytest.mark.parametrize(
    ("roi_side", "expected_x"),
    [("ll", 30), ("rl", 6)],
)
def test_avis_lane_line_selector_groups_fragments_and_picks_inner_candidate(
    monkeypatch, roi_side, expected_x
):
    import avis_lane_line_selection as selector

    def fake_hough(binary, **_kwargs):
        height = binary.shape[0]
        return np.array(
            [
                [[6, 0, 6, 9]],
                [[6, 10, 6, height - 1]],
                [[30, 0, 30, height - 1]],
            ],
            dtype=np.int32,
        )

    monkeypatch.setattr(selector.cv2, "HoughLinesP", fake_hough)
    mask = np.full((40, 40), 255, dtype=np.uint8)
    selected, segments, candidates, selected_x = selector.select_lane_line_mask(
        mask,
        roi_side,
        {
            "hough_threshold": 2,
            "hough_min_line_length_ratio": 0.05,
            "hough_max_line_gap": 2,
            "min_lane_angle_deg": 0.0,
            "max_lane_angle_deg": 80.0,
        },
    )

    assert segments == 3
    assert candidates == 2
    assert selected_x == pytest.approx(expected_x)
    _, xs = np.where(selected > 0)
    assert len(xs) > 0
    assert abs(float(np.median(xs)) - expected_x) <= 1.0


def test_avis_classical_vision_selects_rightmost_ll_and_leftmost_rl_lines(monkeypatch):
    import cv2

    settings = {
        **DEFAULT_AVIS_SETTINGS,
        "ll_top_roi": 0.20,
        "ll_bottom_roi": 0.80,
        "ll_left_roi": 0.10,
        "ll_right_roi": 0.30,
        "rl_top_roi": 0.20,
        "rl_bottom_roi": 0.80,
        "rl_left_roi": 0.70,
        "rl_right_roi": 0.90,
        "hough_threshold": 2,
        "hough_min_line_length_ratio": 0.05,
        "hough_max_line_gap": 2,
    }
    processor = AvisClassicalVisionProcessor(settings)

    def fake_threshold(roi, _settings):
        return np.zeros(roi.shape[:2], dtype=np.uint8)

    def fake_hough(binary, **_kwargs):
        height = binary.shape[0]
        return np.array(
            [
                [[6, 0, 6, height - 1]],
                [[30, 0, 30, height - 1]],
            ],
            dtype=np.int32,
        )

    monkeypatch.setattr(
        AvisClassicalVisionProcessor,
        "_threshold_roi",
        staticmethod(fake_threshold),
    )
    monkeypatch.setattr(cv2, "HoughLinesP", fake_hough)
    result = processor.detect(None, np.zeros((100, 200, 3), dtype=np.uint8))
    debug = result["debug"]

    assert debug["left_x"] == 20 + 30
    assert debug["right_x"] == 140 + 6
    assert debug["left_hough_candidates"] == 2
    assert debug["right_hough_candidates"] == 2
    left_roi_mask = debug["lane_mask"][20:80, 20:60]
    right_roi_mask = debug["lane_mask"][20:80, 140:180]
    assert np.median(np.where(left_roi_mask > 0)[1]) == pytest.approx(30)
    assert np.median(np.where(right_roi_mask > 0)[1]) == pytest.approx(6)


def test_avis_model_backend_uses_the_same_inner_line_selection(monkeypatch):
    import config_city
    import vision.city_vision_processing as shared_vision
    import avis_lane_line_selection as selector

    monkeypatch.setattr(config_city, "LL_TOP_ROI", 0.20, raising=False)
    monkeypatch.setattr(config_city, "LL_BOTTOM_ROI", 0.80, raising=False)
    monkeypatch.setattr(config_city, "LL_LEFT_ROI", 0.10, raising=False)
    monkeypatch.setattr(config_city, "LL_RIGHT_ROI", 0.30, raising=False)
    monkeypatch.setattr(config_city, "RL_TOP_ROI", 0.20, raising=False)
    monkeypatch.setattr(config_city, "RL_BOTTOM_ROI", 0.80, raising=False)
    monkeypatch.setattr(config_city, "RL_LEFT_ROI", 0.70, raising=False)
    monkeypatch.setattr(config_city, "RL_RIGHT_ROI", 0.90, raising=False)

    class FakeVisionProcessor:
        def __init__(self, mode):
            self.mode = mode
            self.min_side_pixels = 1
            self.vehicles_class_id = None

        def _boundary_from_roi(self, full_mask, top, bottom, left, right):
            roi = full_mask[top:bottom, left:right]
            _, xs = np.where(roi > 0)
            return left + int(np.median(xs)) if len(xs) else None

        def detect(self, semantic_frame, rgb_frame=None):
            frame = rgb_frame if rgb_frame is not None else semantic_frame
            height, width = frame.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            ll = (
                int(config_city.LL_TOP_ROI * height),
                int(config_city.LL_BOTTOM_ROI * height),
                int(config_city.LL_LEFT_ROI * width),
                int(config_city.LL_RIGHT_ROI * width),
            )
            rl = (
                int(config_city.RL_TOP_ROI * height),
                int(config_city.RL_BOTTOM_ROI * height),
                int(config_city.RL_LEFT_ROI * width),
                int(config_city.RL_RIGHT_ROI * width),
            )
            for top, bottom, left, right in (ll, rl):
                mask[top:bottom, left + 6] = 255
                mask[top:bottom, left + 30] = 255
            left_x = self._boundary_from_roi(mask, *ll)
            right_x = self._boundary_from_roi(mask, *rl)
            return {
                "error": 0.0,
                "lane_type": "both",
                "debug": {
                    "lane_mask": mask,
                    "left_x": left_x,
                    "right_x": right_x,
                },
            }

    settings = {
        **DEFAULT_AVIS_SETTINGS,
        "hough_threshold": 2,
        "hough_min_line_length_ratio": 0.05,
        "hough_max_line_gap": 2,
    }

    monkeypatch.setattr(shared_vision, "VisionProcessor", FakeVisionProcessor)
    monkeypatch.setattr(
        selector.cv2,
        "HoughLinesP",
        lambda binary, **_kwargs: np.array(
            [
                [[6, 0, 6, binary.shape[0] - 1]],
                [[30, 0, 30, binary.shape[0] - 1]],
            ],
            dtype=np.int32,
        ),
    )
    processor = _create_avis_vision_processor("ultralytics", settings)
    result = processor.detect(None, np.zeros((100, 200, 3), dtype=np.uint8))
    debug = result["debug"]

    assert debug["left_x"] == 20 + 30
    assert debug["right_x"] == 140 + 6
    assert debug["left_hough_candidates"] == 2
    assert debug["right_hough_candidates"] == 2



@pytest.mark.parametrize(
    ("roi_side", "transient_x", "expected_x"),
    [
        ("ll", 36, 30),
        ("rl", 0, 6),
    ],
)
def test_temporal_lane_selector_ignores_one_frame_extreme_outlier(
    monkeypatch, roi_side, transient_x, expected_x
):
    import avis_lane_line_selection as selector

    def lines(*xs):
        return np.array([[[x, 0, x, 39]] for x in xs], dtype=np.int32)

    responses = iter([
        lines(6, 30),
        lines(6, 30, transient_x),
        lines(6, 30),
    ])
    monkeypatch.setattr(
        selector.cv2,
        "HoughLinesP",
        lambda *_args, **_kwargs: next(responses),
    )
    tracker = selector.TemporalLaneLineSelector(roi_side)
    settings = {
        "hough_threshold": 2,
        "hough_min_line_length_ratio": 0.05,
        "hough_max_line_gap": 2,
        "min_lane_angle_deg": 0.0,
        "max_lane_angle_deg": 80.0,
    }
    mask = np.full((40, 50), 255, dtype=np.uint8)

    first = tracker.select(mask, settings)
    second = tracker.select(mask, settings)
    third = tracker.select(mask, settings)

    assert first.selected_x == pytest.approx(expected_x)
    assert second.selected_x == pytest.approx(expected_x)
    assert second.pending_switch_frames == 1
    assert third.selected_x == pytest.approx(expected_x)
    assert third.switched is False


@pytest.mark.parametrize(
    ("roi_side", "new_edge_x", "expected_initial_x"),
    [
        ("ll", 36, 30),
        ("rl", 0, 6),
    ],
)
def test_temporal_lane_selector_switches_after_new_edge_is_stable(
    monkeypatch, roi_side, new_edge_x, expected_initial_x
):
    import avis_lane_line_selection as selector

    def lines(*xs):
        return np.array([[[x, 0, x, 39]] for x in xs], dtype=np.int32)

    responses = iter([
        lines(6, 30),
        lines(6, 30, new_edge_x),
        lines(6, 30, new_edge_x),
        lines(6, 30, new_edge_x),
    ])
    monkeypatch.setattr(
        selector.cv2,
        "HoughLinesP",
        lambda *_args, **_kwargs: next(responses),
    )
    tracker = selector.TemporalLaneLineSelector(roi_side)
    settings = {
        "hough_threshold": 2,
        "hough_min_line_length_ratio": 0.05,
        "hough_max_line_gap": 2,
        "min_lane_angle_deg": 0.0,
        "max_lane_angle_deg": 80.0,
    }
    mask = np.full((40, 50), 255, dtype=np.uint8)

    assert tracker.select(mask, settings).selected_x == pytest.approx(
        expected_initial_x
    )
    second = tracker.select(mask, settings)
    third = tracker.select(mask, settings)
    fourth = tracker.select(mask, settings)

    assert second.switched is False
    assert third.switched is False
    assert fourth.switched is True
    assert fourth.selected_x == pytest.approx(new_edge_x)


def test_temporal_lane_selector_holds_hough_dropout_and_tracks_absolute_x(
    monkeypatch,
):
    import avis_lane_line_selection as selector

    responses = iter([
        np.array([[[12, 0, 12, 39]]], dtype=np.int32),
        None,
        np.array([[[17, 0, 17, 39]]], dtype=np.int32),
    ])
    monkeypatch.setattr(
        selector.cv2,
        "HoughLinesP",
        lambda *_args, **_kwargs: next(responses),
    )
    tracker = selector.TemporalLaneLineSelector("ll")
    settings = {
        "hough_threshold": 2,
        "hough_min_line_length_ratio": 0.05,
        "hough_max_line_gap": 2,
        "min_lane_angle_deg": 0.0,
        "max_lane_angle_deg": 80.0,
    }
    mask = np.full((40, 50), 255, dtype=np.uint8)

    first = tracker.select(mask, settings, x_offset=100.0)
    dropout = tracker.select(mask, settings, x_offset=95.0)
    recovered = tracker.select(mask, settings, x_offset=95.0)

    assert first.selected_x == pytest.approx(112.0)
    assert dropout.selected_x == pytest.approx(112.0)
    assert dropout.predicted is True
    assert np.count_nonzero(dropout.mask) > 0
    assert recovered.selected_x == pytest.approx(112.0)
    assert recovered.missed_frames == 0


def test_temporal_lane_selector_reacquires_when_the_previous_track_disappears(
    monkeypatch,
):
    import avis_lane_line_selection as selector

    responses = iter([
        np.array([[[30, 0, 30, 39]]], dtype=np.int32),
        np.array([[[6, 0, 6, 39]]], dtype=np.int32),
        np.array([[[6, 0, 6, 39]]], dtype=np.int32),
    ])
    monkeypatch.setattr(
        selector.cv2,
        "HoughLinesP",
        lambda *_args, **_kwargs: next(responses),
    )
    tracker = selector.TemporalLaneLineSelector("rl")
    settings = {
        "hough_threshold": 2,
        "hough_min_line_length_ratio": 0.05,
        "hough_max_line_gap": 2,
        "min_lane_angle_deg": 0.0,
        "max_lane_angle_deg": 80.0,
    }
    mask = np.full((40, 50), 255, dtype=np.uint8)

    first = tracker.select(mask, settings)
    missing = tracker.select(mask, settings)
    reacquired = tracker.select(mask, settings)

    assert first.selected_x == pytest.approx(30.0)
    assert missing.selected_x == pytest.approx(30.0)
    assert missing.predicted is True
    assert reacquired.selected_x == pytest.approx(6.0)
    assert reacquired.switched is True




def _obstacle_lane_change_test_settings(**overrides):
    settings = dict(DEFAULT_AVIS_SETTINGS)
    settings.update({
        "obstacle_lane_change_enabled": True,
        "obstacle_front_threshold_cm": 700,
        "obstacle_trigger_confirm_frames": 2,
        "obstacle_sensor_clear_threshold_cm": 900,
        "obstacle_sensor_clear_confirm_frames": 2,
        "obstacle_lane_change_steer_distance_m": 2.0,
        "obstacle_lane_change_left_steer": 0.30,
        "obstacle_lane_change_right_steer": 0.30,
        "obstacle_lane_line_confirm_frames": 2,
        "obstacle_lane_line_search_max_distance_m": 20.0,
        "obstacle_clearance_after_pass_m": 5.0,
    })
    settings.update(overrides)
    return settings


@pytest.mark.parametrize(
    ("target_side", "old_x", "new_x", "expected_phase", "steering_sign"),
    [
        ("left", 100.0, 130.0, AvisObstacleLaneChangeController.LEFT_LANE, -1.0),
        ("right", 220.0, 190.0, AvisObstacleLaneChangeController.RIGHT_LANE, 1.0),
    ],
)
def test_lane_change_repeats_short_steering_pulses_and_checks_target_roi_between_them(
    target_side, old_x, new_x, expected_phase, steering_sign
):
    controller = AvisObstacleLaneChangeController()
    settings = _obstacle_lane_change_test_settings(
        obstacle_trigger_confirm_frames=1,
        obstacle_lane_change_steer_distance_m=0.20,
        obstacle_lane_change_left_steer=0.30,
        obstacle_lane_change_right_steer=0.30,
        obstacle_lane_line_confirm_frames=2,
    )
    if target_side == "right":
        controller.phase = AvisObstacleLaneChangeController.LEFT_LANE
        controller.left_lane_target_distance_m = 1000.0

    def update_frame(observation_id, line_x, *, speed=0.0, trigger=False):
        left_x = line_x if target_side == "left" else 100
        right_x = line_x if target_side == "right" else 220
        return controller.update(
            sensors=(
                [1500, 600, 1500]
                if trigger else [1500, 1500, 1500]
            ),
            speed_kmh=speed,
            dt=0.1,
            left_x=left_x,
            right_x=right_x,
            settings=settings,
            observation_id=observation_id,
            frame_width=320,
        )

    # Build a target-ROI baseline from two independent inference frames.
    update_frame(1, old_x)
    update_frame(2, old_x)
    state = update_frame(3, old_x, trigger=True)
    expected_steer_phase = (
        AvisObstacleLaneChangeController.STEER_LEFT
        if target_side == "left"
        else AvisObstacleLaneChangeController.STEER_RIGHT
    )
    expected_check_phase = (
        AvisObstacleLaneChangeController.ACQUIRE_LEFT_LINE
        if target_side == "left"
        else AvisObstacleLaneChangeController.ACQUIRE_RIGHT_LINE
    )
    assert state["phase"] == expected_steer_phase
    assert state["steering_pulse_number"] == 1
    assert state["steering_override"] == pytest.approx(steering_sign * 0.30)

    # The first pulse uses the configured 0.30 steering amount.
    for observation_id in (4, 5):
        state = update_frame(observation_id, old_x, speed=3.6)
    assert state["phase"] == expected_check_phase
    assert state["steering_override"] == pytest.approx(0.0)

    # Short straight-check interval, then a second, weaker pulse.
    state = update_frame(6, old_x, speed=3.6)
    assert state["phase"] == expected_check_phase
    assert state["steering_override"] == pytest.approx(0.0)
    state = update_frame(7, old_x, speed=3.6)
    assert state["phase"] == expected_steer_phase
    assert state["steering_pulse_number"] == 2
    assert state["steering_override"] == pytest.approx(steering_sign * 0.20)

    for observation_id in (8, 9):
        state = update_frame(observation_id, old_x, speed=3.6)
        assert state["steering_override"] == pytest.approx(steering_sign * 0.20)
    state = update_frame(10, old_x, speed=3.6)
    assert state["phase"] == expected_check_phase
    assert state["steering_override"] == pytest.approx(0.0)

    # A further checked interval starts the next pulse immediately.
    update_frame(11, old_x, speed=3.6)
    state = update_frame(12, old_x, speed=3.6)
    assert state["steering_pulse_number"] == 3
    assert state["phase"] == expected_steer_phase
    assert state["steering_override"] == pytest.approx(steering_sign * 0.20)

    # Even after the old marking disappears, short corrections must continue.
    state = update_frame(13, None, speed=3.6)
    assert state["phase"] == expected_steer_phase
    assert state["steering_override"] == pytest.approx(steering_sign * 0.20)
    state = update_frame(14, None, speed=3.6)
    assert state["phase"] == expected_steer_phase
    assert state["target_line_old_cleared"] is True
    assert state["target_line_confirmed"] is False
    assert state["steering_override"] == pytest.approx(steering_sign * 0.20)
    state = update_frame(15, None, speed=3.6)
    assert state["phase"] == expected_check_phase
    assert state["steering_override"] == pytest.approx(0.0)

    # Only a spatially new line confirmed across the configured frames stops pulses.
    state = update_frame(16, new_x, speed=3.6)
    assert state["phase"] == expected_check_phase
    state = update_frame(17, new_x, speed=3.6)
    assert state["phase"] == expected_phase
    assert state["steering_pulse_number"] == 0



@pytest.mark.parametrize(
    ("target_side", "old_x", "sign"),
    [("left", 100.0, -1.0), ("right", 220.0, 1.0)],
)
def test_lane_change_keeps_pulsing_after_old_line_is_lost_until_new_line_is_confirmed(
    target_side, old_x, sign
):
    controller = AvisObstacleLaneChangeController()
    settings = _obstacle_lane_change_test_settings(
        obstacle_trigger_confirm_frames=1,
        obstacle_lane_change_steer_distance_m=0.20,
        obstacle_lane_line_confirm_frames=3,
    )
    if target_side == "right":
        controller.phase = AvisObstacleLaneChangeController.LEFT_LANE
        controller.left_lane_target_distance_m = 1000.0

    def update_frame(observation_id, line_x, speed=3.6, trigger=False):
        return controller.update(
            sensors=[1500, 600 if trigger else 1500, 1500],
            speed_kmh=speed,
            dt=0.1,
            left_x=line_x if target_side == "left" else 100.0,
            right_x=line_x if target_side == "right" else 220.0,
            settings=settings,
            observation_id=observation_id,
            frame_width=320,
        )

    update_frame(1, old_x, speed=0.0)
    update_frame(2, old_x, speed=0.0)
    state = update_frame(3, old_x, trigger=True, speed=0.0)
    assert state["steering_override"] == pytest.approx(sign * 0.30)

    # End the first pulse and enter the target-ROI check interval.
    for observation_id in (4, 5):
        state = update_frame(observation_id, old_x)
    assert state["phase"] in {
        AvisObstacleLaneChangeController.ACQUIRE_LEFT_LINE,
        AvisObstacleLaneChangeController.ACQUIRE_RIGHT_LINE,
    }

    # Let the old line disappear. It is not enough to stop the correction.
    state = update_frame(6, None, speed=3.6)
    state = update_frame(7, None, speed=3.6)
    assert state["target_line_old_cleared"] is True
    assert state["target_line_confirmed"] is False
    assert state["steering_override"] == pytest.approx(sign * 0.20)

    # The next pulse must remain active while the target marking is missing.
    for observation_id in (8, 9):
        state = update_frame(observation_id, None, speed=3.6)
        assert state["phase"] in {
            AvisObstacleLaneChangeController.STEER_LEFT,
            AvisObstacleLaneChangeController.STEER_RIGHT,
        }
        assert state["steering_override"] == pytest.approx(sign * 0.20)

    # A destination marking must be stable for all required fresh frames.
    for observation_id in (10, 11, 12):
        state = update_frame(observation_id, old_x + (30.0 * sign * -1), speed=3.6)
    assert state["phase"] in {
        AvisObstacleLaneChangeController.ACQUIRE_LEFT_LINE,
        AvisObstacleLaneChangeController.ACQUIRE_RIGHT_LINE,
        AvisObstacleLaneChangeController.LEFT_LANE,
        AvisObstacleLaneChangeController.RIGHT_LANE,
    }


def test_avis_obstacle_lane_change_default_trigger_is_950_cm():
    settings = dict(DEFAULT_AVIS_SETTINGS)
    settings["obstacle_trigger_confirm_frames"] = 1
    controller = AvisObstacleLaneChangeController()

    state = controller.update(
        sensors=[1500, 951, 1500], speed_kmh=8.0, dt=0.1,
        left_x=100, right_x=220, settings=settings,
        allow_new_maneuver=True, observation_id=1,
    )
    assert state["phase"] == "RIGHT_LANE"

    state = controller.update(
        sensors=[1500, 950, 1500], speed_kmh=8.0, dt=0.1,
        left_x=100, right_x=220, settings=settings,
        allow_new_maneuver=True, observation_id=2,
    )
    assert state["phase"] == "STEER_LEFT"


@pytest.mark.parametrize(
    ("sensor_index", "sensor_name"),
    [(0, "LEFT"), (1, "FRONT"), (2, "RIGHT")],
)
def test_avis_any_of_three_distance_sensors_can_trigger_lane_change(
    sensor_index, sensor_name
):
    controller = AvisObstacleLaneChangeController()
    settings = _obstacle_lane_change_test_settings(
        obstacle_front_threshold_cm=700,
        obstacle_trigger_confirm_frames=2,
    )
    sensors = [1500, 1500, 1500]
    sensors[sensor_index] = 650

    first = controller.update(
        sensors=sensors, speed_kmh=8.0, dt=0.1,
        left_x=None, right_x=None, settings=settings,
        allow_new_maneuver=False, observation_id=1,
    )
    assert first["phase"] == "RIGHT_LANE"

    second = controller.update(
        sensors=sensors, speed_kmh=8.0, dt=0.1,
        left_x=None, right_x=None, settings=settings,
        allow_new_maneuver=False, observation_id=2,
    )
    assert second["phase"] == "STEER_LEFT"
    assert second["obstacle_trigger_sensor"] == sensor_name
    assert second["last_obstacle_distance_m"] == pytest.approx(6.5)
    assert second["steering_override"] == pytest.approx(-0.30)


@pytest.mark.parametrize(
    ("sensor_index", "sensor_name"),
    [(0, "LEFT"), (1, "FRONT"), (2, "RIGHT")],
)
def test_any_distance_sensor_triggered_in_left_lane_forces_return_right(
    sensor_index, sensor_name
):
    controller = AvisObstacleLaneChangeController()
    controller.phase = AvisObstacleLaneChangeController.LEFT_LANE
    controller.left_lane_target_distance_m = 30.0
    settings = _obstacle_lane_change_test_settings(
        obstacle_trigger_confirm_frames=2,
    )
    sensors = [1500, 1500, 1500]
    sensors[sensor_index] = 450

    for observation_id in (1, 2):
        state = controller.update(
            sensors=sensors, speed_kmh=8.0, dt=0.0,
            left_x=100, right_x=220, settings=settings,
            allow_new_maneuver=False, observation_id=observation_id,
        )

    assert state["phase"] == "STEER_RIGHT"
    assert state["return_reason"] == "OBSTACLE_IN_LEFT_LANE"
    assert state["obstacle_trigger_sensor"] == sensor_name
    assert state["steering_override"] == pytest.approx(0.30)


def test_obstacle_latch_clears_only_after_all_three_sensors_are_clear():
    controller = AvisObstacleLaneChangeController()
    settings = _obstacle_lane_change_test_settings(
        obstacle_trigger_confirm_frames=2,
        obstacle_sensor_clear_confirm_frames=2,
    )
    blocked_left = [650, 1500, 1500]
    for observation_id in (1, 2):
        state = controller.update(
            sensors=blocked_left, speed_kmh=0.0, dt=0.0,
            left_x=None, right_x=None, settings=settings,
            allow_new_maneuver=False, observation_id=observation_id,
        )
    assert state["obstacle_latched"] is True

    state = controller.update(
        sensors=[650, 1500, 1500], speed_kmh=0.0, dt=0.0,
        left_x=None, right_x=None, settings=settings, observation_id=3,
    )
    assert state["obstacle_latched"] is True

    for observation_id in (4, 5):
        state = controller.update(
            sensors=[1500, 1500, 1500], speed_kmh=0.0, dt=0.0,
            left_x=None, right_x=None, settings=settings,
            observation_id=observation_id,
        )
    assert state["obstacle_latched"] is False


def test_avis_obstacle_lane_change_uses_middle_sensor_and_confirms_obstacle():
    controller = AvisObstacleLaneChangeController()
    settings = _obstacle_lane_change_test_settings()

    first = controller.update(
        sensors=[1500, 650, 1500], speed_kmh=0.0, dt=0.1,
        left_x=100, right_x=220, settings=settings,
        allow_new_maneuver=True, observation_id=1,
    )
    assert first["phase"] == "RIGHT_LANE"

    second = controller.update(
        sensors=[1500, 650, 1500], speed_kmh=0.0, dt=0.1,
        left_x=100, right_x=220, settings=settings,
        allow_new_maneuver=True, observation_id=2,
    )
    assert second["phase"] == "STEER_LEFT"
    assert second["steering_override"] == pytest.approx(-0.30)
    assert second["obstacle_latched"] is True
    assert second["last_obstacle_distance_m"] == pytest.approx(6.5)
    assert second["left_lane_target_distance_m"] == pytest.approx(11.5)


def test_avis_obstacle_lane_change_switches_to_straight_then_waits_for_left_roi():
    controller = AvisObstacleLaneChangeController()
    settings = _obstacle_lane_change_test_settings()

    for observation_id in (1, 2):
        state = controller.update(
            sensors=[1500, 600, 1500], speed_kmh=0.0, dt=0.0,
            left_x=100, right_x=220, settings=settings,
            observation_id=observation_id,
        )
    assert state["phase"] == "STEER_LEFT"

    for observation_id in (3, 4):
        state = controller.update(
            sensors=[1500, 600, 1500], speed_kmh=36.0, dt=0.1,
            left_x=None, right_x=220, settings=settings,
            observation_id=observation_id,
        )
    assert state["phase"] == "ACQUIRE_LEFT_LINE"
    assert state["steering_override"] == pytest.approx(0.0)

    state = controller.update(
        sensors=[1500, 600, 1500], speed_kmh=0.0, dt=0.0,
        left_x=None, right_x=220, settings=settings, observation_id=5,
    )
    assert state["phase"] == "ACQUIRE_LEFT_LINE"
    assert state["steering_override"] == pytest.approx(0.0)

    for observation_id in (6, 7):
        state = controller.update(
            sensors=[1500, 600, 1500], speed_kmh=0.0, dt=0.0,
            left_x=110, right_x=None, settings=settings,
            observation_id=observation_id,
        )
    assert state["phase"] == "LEFT_LANE"
    assert state["pid_reset"] is True
    assert state["left_lane_distance_m"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("target_side", "old_x", "new_x", "expected_phase"),
    [
        ("left", 100.0, 130.0, AvisObstacleLaneChangeController.LEFT_LANE),
        ("right", 220.0, 190.0, AvisObstacleLaneChangeController.RIGHT_LANE),
    ],
)
def test_lane_change_ignores_previous_target_roi_line_until_new_line_is_confirmed(
    target_side, old_x, new_x, expected_phase
):
    controller = AvisObstacleLaneChangeController()
    settings = _obstacle_lane_change_test_settings(
        obstacle_trigger_confirm_frames=2,
        obstacle_lane_line_confirm_frames=2,
        obstacle_lane_change_steer_distance_m=2.0,
    )
    controller.phase = (
        AvisObstacleLaneChangeController.RIGHT_LANE
        if target_side == "left"
        else AvisObstacleLaneChangeController.LEFT_LANE
    )
    controller.left_lane_target_distance_m = 1000.0 if target_side == "right" else None

    def update_frame(observation_id, line_x, speed=0.0, sensors=None):
        distances = sensors or [1500, 1500, 1500]
        left_x = line_x if target_side == "left" else 100
        right_x = line_x if target_side == "right" else 220
        return controller.update(
            sensors=distances,
            speed_kmh=speed,
            dt=0.1,
            left_x=left_x,
            right_x=right_x,
            settings=settings,
            observation_id=observation_id,
            frame_width=320,
        )

    if target_side == "left":
        # Two pre-manoeuvre frames establish the old line in the LL history.
        update_frame(1, old_x, sensors=[1500, 600, 1500])
        state = update_frame(2, old_x, sensors=[1500, 600, 1500])
    else:
        # While driving in the left lane, build history for the return-side RL.
        update_frame(1, old_x)
        update_frame(2, old_x)
        controller.left_lane_target_distance_m = 0.0
        state = update_frame(3, old_x)

    if target_side == "left":
        assert state["phase"] == "STEER_LEFT"
    else:
        assert state["phase"] == "STEER_RIGHT"
    assert state["target_line_baseline_x"] == pytest.approx(old_x)

    # The old marking remains visible through the steering pulse and must be
    # tracked, not accepted as proof that the destination lane was reached.
    start_id = 2 if target_side == "left" else 3
    for observation_id in (start_id + 1, start_id + 2):
        state = update_frame(observation_id, old_x, speed=36.0)
    if target_side == "left":
        assert state["phase"] == "ACQUIRE_LEFT_LINE"
    else:
        assert state["phase"] == "ACQUIRE_RIGHT_LINE"

    for observation_id in range(start_id + 3, start_id + 6):
        state = update_frame(observation_id, old_x)
        assert state["phase"] != expected_phase
        assert state["line_confirm_frames"] == 0
        assert state["target_line_old_cleared"] is False

    # Once the old line has left the target ROI, a distinct new marking must
    # remain stable across the configured confirmation frames.
    for observation_id in (start_id + 6, start_id + 7):
        state = update_frame(observation_id, None)
    assert state["target_line_old_cleared"] is True
    state = update_frame(start_id + 8, new_x)
    assert state["phase"] != expected_phase
    assert state["line_confirm_frames"] == 1
    state = update_frame(start_id + 9, new_x)
    assert state["phase"] == expected_phase


def test_avis_obstacle_lane_change_returns_right_after_configured_distance():
    controller = AvisObstacleLaneChangeController()
    settings = _obstacle_lane_change_test_settings()

    for observation_id in (1, 2):
        state = controller.update(
            sensors=[1500, 600, 1500], speed_kmh=0.0, dt=0.0,
            left_x=100, right_x=220, settings=settings,
            observation_id=observation_id,
        )
    for observation_id in (3, 4):
        state = controller.update(
            sensors=[1500, 600, 1500], speed_kmh=36.0, dt=0.1,
            left_x=None, right_x=220, settings=settings,
            observation_id=observation_id,
        )
    # The old LL marking can reappear after a short detection gap. It must
    # still be ignored rather than being mistaken for the destination-lane line.
    for observation_id in (5, 6):
        state = controller.update(
            sensors=[1500, 600, 1500], speed_kmh=0.0, dt=0.0,
            left_x=100, right_x=None, settings=settings,
            observation_id=observation_id,
        )
    assert state["phase"] == "ACQUIRE_LEFT_LINE"
    assert state["line_confirm_frames"] == 0
    assert state["target_line_old_cleared"] is False

    # A spatially distinct LL marking, stable across two inference frames,
    # confirms entry into the new lane.
    for observation_id in (7, 8):
        state = controller.update(
            sensors=[1500, 600, 1500], speed_kmh=0.0, dt=0.0,
            left_x=110, right_x=None, settings=settings,
            observation_id=observation_id,
        )
    assert state["phase"] == "LEFT_LANE"

    # The first obstacle was detected at 6m; with 5m extra clearance,
    # the controller should return after 11m in the left lane (not a fixed 750m).
    for observation_id in range(9, 20):
        state = controller.update(
            sensors=[1500, 1500, 1500], speed_kmh=36.0, dt=0.1,
            left_x=100, right_x=220, settings=settings,
            observation_id=observation_id,
        )
    assert state["left_lane_target_distance_m"] == pytest.approx(11.0)
    assert state["left_lane_distance_m"] == pytest.approx(11.0)
    assert state["phase"] == "STEER_RIGHT"
    assert state["steering_override"] == pytest.approx(0.30)

    for observation_id in (10, 11):
        state = controller.update(
            sensors=[1500, 1500, 1500], speed_kmh=36.0, dt=0.1,
            left_x=100, right_x=220, settings=settings,
            observation_id=observation_id,
        )
    assert state["phase"] == "ACQUIRE_RIGHT_LINE"
    assert state["steering_override"] == pytest.approx(0.0)

    for observation_id in (12, 13):
        state = controller.update(
            sensors=[1500, 1500, 1500], speed_kmh=0.0, dt=0.0,
            left_x=None, right_x=210, settings=settings,
            observation_id=observation_id,
        )
    assert state["phase"] == "RIGHT_LANE"
    assert state["pid_reset"] is True


def test_avis_obstacle_in_left_lane_forces_immediate_return_before_old_target():
    controller = AvisObstacleLaneChangeController()
    settings = _obstacle_lane_change_test_settings(
        obstacle_trigger_confirm_frames=2,
        obstacle_sensor_clear_confirm_frames=2,
        obstacle_clearance_after_pass_m=5.0,
    )

    # First obstacle starts the right-to-left change at 6m sensor distance.
    for observation_id in (1, 2):
        state = controller.update(
            sensors=[1500, 600, 1500], speed_kmh=0.0, dt=0.0,
            left_x=100, right_x=220, settings=settings,
            observation_id=observation_id,
        )
    for observation_id in (3, 4):
        state = controller.update(
            sensors=[1500, 600, 1500], speed_kmh=36.0, dt=0.1,
            left_x=None, right_x=220, settings=settings,
            observation_id=observation_id,
        )
    for observation_id in (5, 6):
        state = controller.update(
            sensors=[1500, 600, 1500], speed_kmh=0.0, dt=0.0,
            left_x=110, right_x=None, settings=settings,
            observation_id=observation_id,
        )
    assert state["phase"] == "LEFT_LANE"
    assert state["left_lane_target_distance_m"] == pytest.approx(11.0)

    # Clear the first obstacle's latch; the original 11m travel target is not met.
    for observation_id in (7, 8):
        state = controller.update(
            sensors=[1500, 1500, 1500], speed_kmh=0.0, dt=0.0,
            left_x=110, right_x=None, settings=settings,
            observation_id=observation_id,
        )
    assert state["left_lane_distance_m"] == pytest.approx(0.0)

    # A second obstacle in front of the left lane must override the old target.
    for observation_id in (9, 10):
        state = controller.update(
            sensors=[1500, 450, 1500], speed_kmh=0.0, dt=0.0,
            left_x=110, right_x=None, settings=settings,
            allow_new_maneuver=False,
            observation_id=observation_id,
        )
    assert state["phase"] == "STEER_RIGHT"
    assert state["return_reason"] == "OBSTACLE_IN_LEFT_LANE"
    assert state["steering_override"] == pytest.approx(0.30)
    assert state["left_lane_distance_m"] < state["left_lane_target_distance_m"]


def test_avis_obstacle_lane_change_keeps_moving_when_target_roi_line_is_missing():
    controller = AvisObstacleLaneChangeController()
    settings = _obstacle_lane_change_test_settings(
        obstacle_lane_line_search_max_distance_m=1.0,
    )
    for observation_id in (1, 2):
        state = controller.update(
            sensors=[1500, 600, 1500], speed_kmh=0.0, dt=0.0,
            left_x=100, right_x=220, settings=settings,
            observation_id=observation_id,
        )
    for observation_id in (3, 4):
        state = controller.update(
            sensors=[1500, 600, 1500], speed_kmh=36.0, dt=0.1,
            left_x=None, right_x=220, settings=settings,
            observation_id=observation_id,
        )
    assert state["phase"] == "ACQUIRE_LEFT_LINE"
    state = controller.update(
        sensors=[1500, 600, 1500], speed_kmh=36.0, dt=0.1,
        left_x=None, right_x=220, settings=settings,
        observation_id=5,
    )
    assert state["phase"] == "STEER_LEFT"
    assert state["steering_override"] == pytest.approx(-0.20)
    assert state["steering_pulse_number"] == 2
    assert state["search_limit_reached"] is True
    assert state["safe_stop"] is False


def test_avis_dashboard_lane_change_settings_are_persistent_and_validated(tmp_path):
    state = AvisDashboardState(settings_path=tmp_path / "avis_settings.json")
    updated = state.update_settings({
        "obstacle_lane_change_enabled": True,
        "obstacle_front_threshold_cm": 950,
        "obstacle_clearance_after_pass_m": 6.5,
        "obstacle_lane_change_left_steer": 0.24,
    })
    assert updated["obstacle_front_threshold_cm"] == 950
    assert updated["obstacle_clearance_after_pass_m"] == pytest.approx(6.5)
    assert updated["obstacle_lane_change_left_steer"] == pytest.approx(0.24)

    restored = AvisDashboardState(settings_path=tmp_path / "avis_settings.json")
    restored_settings, _ = restored.settings_snapshot()
    assert restored_settings["obstacle_clearance_after_pass_m"] == pytest.approx(6.5)

    # The front threshold may exceed the configured clear threshold; runtime
    # applies hysteresis automatically instead of rejecting the form.
    updated = state.update_settings({
        "obstacle_front_threshold_cm": 1200,
        "obstacle_sensor_clear_threshold_cm": 900,
    }, persist=False)
    assert updated["obstacle_front_threshold_cm"] == 1200


def test_avis_dashboard_accepts_decimal_obstacle_tuning_without_step_restrictions():
    updated = validate_avis_settings(
        DEFAULT_AVIS_SETTINGS,
        {
            "obstacle_lane_change_steer_distance_m": 1.37,
            "obstacle_lane_change_left_steer": 0.237,
            "obstacle_lane_change_right_steer": 0.193,
            "obstacle_clearance_after_pass_m": 0.0,
            "obstacle_front_threshold_cm": 1499,
            "obstacle_sensor_clear_threshold_cm": 500,
        },
    )
    assert updated["obstacle_lane_change_steer_distance_m"] == pytest.approx(1.37)
    assert updated["obstacle_lane_change_left_steer"] == pytest.approx(0.237)
    assert updated["obstacle_lane_change_right_steer"] == pytest.approx(0.193)
    assert updated["obstacle_clearance_after_pass_m"] == pytest.approx(0.0)


def test_avis_dashboard_migrates_legacy_fixed_lane_return_distance(tmp_path):
    settings_path = tmp_path / "avis_settings.json"
    settings_path.write_text(
        json.dumps({
            "obstacle_return_distance_m": 750,
            "obstacle_slowdown_start_cm": 1600,
            "obstacle_front_threshold_cm": 850,
            "obstacle_clearance_after_pass_m": 7.0,
        }),
        encoding="utf-8",
    )
    state = AvisDashboardState(settings_path=settings_path)
    settings, _ = state.settings_snapshot()
    assert settings["obstacle_front_threshold_cm"] == 850
    assert settings["obstacle_clearance_after_pass_m"] == pytest.approx(7.0)
    assert "obstacle_return_distance_m" not in settings
    assert "obstacle_slowdown_start_cm" not in settings


def test_avis_import_profile_migrates_removed_obstacle_slowdown_setting(tmp_path):
    state = AvisDashboardState(
        settings_path=tmp_path / "avis_settings.json",
        profile_path=tmp_path / "avis_profiles.json",
    )
    old_settings = dict(DEFAULT_AVIS_SETTINGS)
    old_settings["obstacle_slowdown_start_cm"] = 1600

    imported = state.import_profile({
        "schema_version": 1,
        "name": "Older Speed Profile",
        "settings": old_settings,
    })

    assert imported["settings"]["obstacle_front_threshold_cm"] == 950
    assert "obstacle_slowdown_start_cm" not in imported["settings"]


def test_avis_dashboard_exposes_obstacle_lane_change_settings(tmp_path):
    state = AvisDashboardState(settings_path=tmp_path / "avis_settings.json")
    html = create_avis_dashboard_app(state).test_client().get("/").get_data(as_text=True)
    assert 'id="obstacle_lane_change_enabled"' in html
    assert 'id="obstacle_model_confidence_threshold"' in html
    assert 'id="obstacle_lane_confidence_threshold"' in html
    assert 'id="obstacle_pass_sensor_angle"' in html
    assert 'id="obstacle_pass_sensor_presence_threshold_cm"' in html
    assert 'id="obstacle_pass_sensor_clear_threshold_cm"' in html
    assert 'id="obstacle_return_delay_seconds"' in html
    assert 'id="obstacle_slowdown_start_cm"' not in html
    assert 'id="obstacle_clearance_after_pass_m"' not in html
    assert 'id="obstacle_return_distance_m"' not in html
    assert 'id="obstacle_front_threshold_cm"' not in html
    assert 'id="obstacle_pass_sensor_presence_threshold_cm" data-setting="obstacle_pass_sensor_presence_threshold_cm" type="number" min="20" max="1499" step="1"' in html
    assert 'id="obstacle_return_delay_seconds" data-setting="obstacle_return_delay_seconds" type="number" min="0" max="60" step=".1"' in html
    assert "right-side sensor" in html
    assert 'id="obstacle_pass_sensor_angle" data-setting="obstacle_pass_sensor_angle" type="number" min="10" max="90" step="1"' in html
    assert 'id="obstacle_lane_line_search_max_distance_m"' in html
    assert 'id="settingsForm" novalidate' in html
    assert 'id="obstacle_lane_change_steer_distance_m" data-setting="obstacle_lane_change_steer_distance_m" type="number" min=".1" max="100" step="any"' in html
    assert 'id="obstacle_lane_change_left_steer" data-setting="obstacle_lane_change_left_steer" type="number" min=".001" max="1" step="any"' in html
    assert 'id="obstacle_pass_sensor_clear_threshold_cm" data-setting="obstacle_pass_sensor_clear_threshold_cm" type="number" min="21" max="1500" step="1"' in html
    assert 'id="obstacle_pass_sensor_clear_frames" data-setting="obstacle_pass_sensor_clear_frames" type="number" min="1" max="30" step="1"' in html
    assert 'id="frameReadFpsValue"' in html
    assert 'id="inferenceFpsValue"' in html
    assert 'id="controlLoopFpsValue"' in html


def test_avis_dashboard_roi_settings_validate_each_rectangle():
    updated = validate_avis_settings(
        DEFAULT_AVIS_SETTINGS,
        {
            "ll_top_roi": 0.50,
            "ll_bottom_roi": 0.75,
            "rl_left_roi": 0.60,
            "rl_right_roi": 0.85,
        },
    )
    assert updated["ll_top_roi"] == pytest.approx(0.50)
    assert updated["rl_right_roi"] == pytest.approx(0.85)

    with pytest.raises(ValueError, match="LL ROI bottom"):
        validate_avis_settings(
            DEFAULT_AVIS_SETTINGS,
            {"ll_top_roi": 0.75, "ll_bottom_roi": 0.70},
        )

    with pytest.raises(ValueError, match="RL ROI right edge"):
        validate_avis_settings(
            DEFAULT_AVIS_SETTINGS,
            {"rl_left_roi": 0.80, "rl_right_roi": 0.70},
        )


def test_avis_dashboard_roi_editor_is_available_for_both_vision_backends(tmp_path):
    state = AvisDashboardState(settings_path=tmp_path / "avis_settings.json")
    client = create_avis_dashboard_app(state).test_client()
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'id="roiCanvas"' in html
    assert 'id="ll_top_roi"' in html
    assert 'id="rl_top_roi"' in html
    assert "shared by both AVIS vision backends" in html
    assert 'data-panel="live"' in html
    assert 'data-panel="settings"' in html
    assert 'data-panel="roi"' in html
    assert 'id="drivableView"' not in html
    assert 'src="/stream/drivable-mask"' not in html
    assert "Drivable-area and vehicle masks are not calculated or displayed" in html
    assert client.get("/stream/drivable-mask").status_code == 404


def test_avis_classical_vision_stops_when_no_lane_is_detected():
    import numpy as np

    processor = AvisClassicalVisionProcessor()
    result = processor.detect(None, np.zeros((192, 320, 3), dtype=np.uint8))

    assert result["lane_valid"] is False
    assert result["error"] == 0.0
    assert result["debug"]["drivable_mask"] is None



def test_avis_client_sends_reverse_speed_only_for_manual_controls():
    api = SimpleNamespace(Car=FakeCar)
    client = AvisEngineClient(api, "127.0.0.1", 25001)
    fake_car = client.car

    client.send_manual_control(-0.3, -0.2)
    client.close()

    assert fake_car.speed_commands == [-30, 0]
    assert fake_car.steering_commands == [-20, 0]



def test_avis_model_extracts_only_lane_lines_and_does_not_require_other_classes(monkeypatch):
    import numpy as np
    import vision.city_vision_processing as shared_vision

    class FakeVisionProcessor:
        def __init__(self, mode):
            self.mode = mode

    monkeypatch.setattr(shared_vision, "VisionProcessor", FakeVisionProcessor)
    processor = _create_avis_vision_processor("ultralytics", {})

    # Both a lane-only checkpoint and an older multi-class checkpoint are valid.
    assert processor._class_ids_from_names({
        0: "background",
        2: "lane_lines",
    }) == (2, None, None)
    assert processor._class_ids_from_names({
        0: "background",
        1: "ego_drivable_area",
        2: "lane_lines",
        3: "vehicles",
    }) == (2, None, None)

    processor.lane_class_id = 2
    semantic_map = np.array([[0, 1, 2]], dtype=np.uint8)
    processor._predict_ultralytics = lambda _frame: [
        SimpleNamespace(semantic_mask=SimpleNamespace(data=semantic_map))
    ]

    drivable_mask, lane_mask, vehicles_mask = processor._extract_masks_from_ultralytics(
        np.zeros((1, 3, 3), dtype=np.uint8)
    )

    assert drivable_mask is None
    assert lane_mask.tolist() == [[0, 0, 255]]
    assert vehicles_mask is None


@pytest.mark.parametrize("active_side", [None, "left", "right"])
def test_avis_model_lane_center_uses_midpoint_regardless_of_sticky_side(
    monkeypatch, active_side
):
    import vision.city_vision_processing as shared_vision

    class FakeVisionProcessor:
        def __init__(self, mode):
            self.mode = mode
            self.active_side = None
            self.last_lane_center = None
            self.vehicles_class_id = None

    monkeypatch.setattr(shared_vision, "VisionProcessor", FakeVisionProcessor)
    processor = _create_avis_vision_processor("ultralytics", {})
    processor.active_side = active_side

    lane_type, lane_center = processor._pick_sticky_lane(
        left_x=100,
        right_x=220,
        frame_center=160.0,
        lane_offset_px=20.0,
    )

    assert lane_type == "both"
    assert lane_center == pytest.approx(160.0)
    assert processor.last_lane_half_width == pytest.approx(60.0)
    assert processor.active_side is None


def test_avis_model_single_boundary_uses_last_measured_lane_half_width(monkeypatch):
    import vision.city_vision_processing as shared_vision

    class FakeVisionProcessor:
        def __init__(self, mode):
            self.mode = mode
            self.active_side = None
            self.last_lane_center = None
            self.vehicles_class_id = None

    monkeypatch.setattr(shared_vision, "VisionProcessor", FakeVisionProcessor)
    processor = _create_avis_vision_processor("ultralytics", {})

    processor._pick_sticky_lane(100, 220, 160.0, 20.0)

    lane_type, left_only_center = processor._pick_sticky_lane(
        left_x=100,
        right_x=None,
        frame_center=160.0,
        lane_offset_px=20.0,
    )
    assert lane_type == "only_left"
    assert left_only_center == pytest.approx(160.0)

    processor.active_side = "right"
    lane_type, right_only_center = processor._pick_sticky_lane(
        left_x=None,
        right_x=220,
        frame_center=160.0,
        lane_offset_px=20.0,
    )
    assert lane_type == "only_right"
    assert right_only_center == pytest.approx(160.0)


def test_avis_model_debug_uses_pure_red_lane_pixels_and_suppresses_other_masks(monkeypatch):
    import numpy as np
    import vision.city_vision_processing as shared_vision

    received = {}

    class FakeVisionProcessor:
        def __init__(self, mode):
            received["mode"] = mode

        def _draw_debug(self, **kwargs):
            received["line_mask"] = kwargs.get("line_mask")
            received["drivable_mask"] = kwargs.get("drivable_mask")
            received["vehicles_mask"] = kwargs.get("vehicles_mask")
            return np.ascontiguousarray(kwargs["frame"]).copy()

    monkeypatch.setattr(shared_vision, "VisionProcessor", FakeVisionProcessor)
    processor = _create_avis_vision_processor("ultralytics", {})
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    lane_mask = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    result = processor._draw_debug(
        frame=frame,
        line_mask=lane_mask,
        drivable_mask=np.full((2, 3), 255, dtype=np.uint8),
        left_x=None,
        right_x=None,
        lane_center=1.5,
        lane_type="none",
        vehicles_mask=np.full((2, 3), 255, dtype=np.uint8),
    )

    assert received["mode"] == "ultralytics"
    assert received["line_mask"] is None
    assert received["drivable_mask"] is None
    assert received["vehicles_mask"] is None
    assert tuple(result[0, 0]) == (0, 0, 255)
    assert tuple(result[0, 1]) == (0, 0, 0)


def test_avis_lane_mask_preview_is_black_with_pure_red_selected_pixels():
    import cv2
    import numpy as np
    from avis_runtime import _colorize_avis_lane_mask

    mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    preview = _colorize_avis_lane_mask(cv2, np, mask, mask.shape)

    assert preview.shape == (2, 2, 3)
    assert tuple(preview[0, 0]) == (0, 0, 0)
    assert tuple(preview[0, 1]) == (0, 0, 255)
    assert tuple(preview[1, 0]) == (0, 0, 255)


def test_avis_adaptive_roi_uses_last_seven_frames_to_expand_toward_escape():
    settings = dict(DEFAULT_AVIS_SETTINGS)
    adaptive = AvisAdaptiveROIController(settings)
    frame_width = 1000
    left, right = settings["ll_left_roi"], settings["ll_right_roi"]
    positions = (0.68, 0.54, 0.39, 0.25, 0.17, 0.11, 0.08)
    for ratio in positions:
        x = (left + (right - left) * ratio) * frame_width
        adaptive.observe(x, None, frame_width)

    snapshot = adaptive.snapshot()
    assert snapshot["ll"]["history_frames"] == 7
    assert snapshot["ll"]["expanded_left"] > 0
    assert snapshot["ll"]["expanded_right"] == 0
    assert snapshot["ll"]["left"] < left


def test_avis_adaptive_roi_restores_slowly_after_line_returns_to_base_roi_center():
    settings = dict(DEFAULT_AVIS_SETTINGS)
    adaptive = AvisAdaptiveROIController(settings)
    frame_width = 1000
    left, right = settings["ll_left_roi"], settings["ll_right_roi"]
    for ratio in (0.60, 0.42, 0.24, 0.16, 0.10):
        adaptive.observe((left + (right - left) * ratio) * frame_width, None, frame_width)

    before = adaptive.snapshot()["ll"]["expanded_left"]
    assert before > 0
    center_x = (left + (right - left) * 0.50) * frame_width
    for _ in range(7):
        adaptive.observe(center_x, None, frame_width)

    after = adaptive.snapshot()["ll"]["expanded_left"]
    assert 0 <= after < before
    for _ in range(20):
        adaptive.observe(center_x, None, frame_width)
    assert adaptive.snapshot()["ll"]["expanded_left"] == 0


def test_avis_adaptive_roi_expands_the_correct_edge_of_each_independent_roi():
    settings = dict(DEFAULT_AVIS_SETTINGS)
    adaptive = AvisAdaptiveROIController(settings)
    frame_width = 1000
    rl_left, rl_right = settings["rl_left_roi"], settings["rl_right_roi"]
    for ratio in (0.36, 0.50, 0.68, 0.82, 0.91):
        x = (rl_left + (rl_right - rl_left) * ratio) * frame_width
        adaptive.observe(None, x, frame_width)

    snapshot = adaptive.snapshot()
    assert snapshot["rl"]["expanded_right"] > 0
    assert snapshot["rl"]["expanded_left"] == 0
    assert snapshot["ll"]["expanded_left"] == 0
    assert snapshot["ll"]["expanded_right"] == 0


def test_avis_adaptive_roi_never_expands_the_inner_edges_toward_the_other_lane():
    settings = dict(DEFAULT_AVIS_SETTINGS)
    adaptive = AvisAdaptiveROIController(settings)
    frame_width = 1000

    # RL marking escapes toward the left, but its left edge is the inner edge.
    rl_left, rl_right = settings["rl_left_roi"], settings["rl_right_roi"]
    for ratio in (0.70, 0.54, 0.38, 0.22, 0.12, 0.08):
        adaptive.observe(
            None,
            (rl_left + (rl_right - rl_left) * ratio) * frame_width,
            frame_width,
        )

    # LL marking escapes toward the right, but its right edge is the inner edge.
    ll_left, ll_right = settings["ll_left_roi"], settings["ll_right_roi"]
    for ratio in (0.30, 0.48, 0.64, 0.80, 0.92, 0.96):
        adaptive.observe(
            (ll_left + (ll_right - ll_left) * ratio) * frame_width,
            None,
            frame_width,
        )

    snapshot = adaptive.snapshot()
    active = adaptive.active_settings()
    assert snapshot["rl"]["expanded_left"] == 0.0
    assert snapshot["ll"]["expanded_right"] == 0.0
    assert active["rl_left_roi"] == pytest.approx(rl_left)
    assert active["ll_right_roi"] == pytest.approx(ll_right)


def test_avis_adaptive_roi_can_be_disabled_without_changing_base_rectangles():
    settings = dict(DEFAULT_AVIS_SETTINGS)
    settings["adaptive_roi_enabled"] = False
    adaptive = AvisAdaptiveROIController(settings)
    frame_width = 1000
    left, right = settings["ll_left_roi"], settings["ll_right_roi"]
    for ratio in (0.60, 0.42, 0.24, 0.16, 0.10):
        adaptive.observe((left + (right - left) * ratio) * frame_width, None, frame_width)

    active = adaptive.active_settings()
    assert active["ll_left_roi"] == pytest.approx(left)
    assert active["ll_right_roi"] == pytest.approx(right)


def test_avis_named_profiles_save_load_import_export_and_persist(tmp_path):
    settings_path = tmp_path / "avis_settings.json"
    profile_path = tmp_path / "avis_profiles.json"
    state = AvisDashboardState(
        settings_path=settings_path,
        profile_path=profile_path,
    )
    state.update_settings(
        {
            "vision_backend": "classical",
            "min_speed_kmh": 7.5,
            "adaptive_roi_max_expansion_ratio": 0.18,
        }
    )
    saved = state.save_profile("Night Track")
    assert saved["name"] == "Night Track"
    assert saved["settings"]["vision_backend"] == "classical"

    state.update_settings({"vision_backend": "ultralytics", "min_speed_kmh": 10.0})
    loaded = state.load_profile("Night Track")
    assert loaded["vision_backend"] == "classical"
    assert loaded["min_speed_kmh"] == pytest.approx(7.5)
    assert loaded["adaptive_roi_max_expansion_ratio"] == pytest.approx(0.18)

    exported = state.export_profile("Night Track")
    restored = AvisDashboardState(
        settings_path=tmp_path / "other_settings.json",
        profile_path=tmp_path / "other_profiles.json",
    )
    imported = restored.import_profile(exported)
    assert imported["settings"]["min_speed_kmh"] == pytest.approx(7.5)
    assert restored.load_profile("Night Track")["vision_backend"] == "classical"

    reloaded = AvisDashboardState(
        settings_path=tmp_path / "reloaded_settings.json",
        profile_path=profile_path,
    )
    assert any(
        profile["name"] == "Night Track"
        for profile in reloaded.profiles_snapshot()
    )
    reloaded.delete_profile("Night Track")
    assert all(
        profile["name"] != "Night Track"
        for profile in reloaded.profiles_snapshot()
    )


def test_avis_named_profiles_reject_invalid_names_and_protect_baseline(tmp_path):
    state = AvisDashboardState(
        settings_path=tmp_path / "avis_settings.json",
        profile_path=tmp_path / "avis_profiles.json",
    )
    for name in ("", "../escape", "bad/name", "x" * 49):
        with pytest.raises(ValueError):
            state.save_profile(name)
    with pytest.raises(ValueError, match="cannot be deleted"):
        state.delete_profile("AVIS_StableV2")


def test_avis_profile_api_import_export_and_listing(tmp_path):
    state = AvisDashboardState(
        settings_path=tmp_path / "avis_settings.json",
        profile_path=tmp_path / "avis_profiles.json",
    )
    client = create_avis_dashboard_app(state).test_client()
    assert client.get("/api/profiles").status_code == 200
    imported = client.post(
        "/api/profiles/import",
        json={
            "schema_version": 1,
            "name": "Imported CV",
            "settings": {"vision_backend": "classical", "min_speed_kmh": 6.5},
        },
    )
    assert imported.status_code == 200
    assert imported.get_json()["settings"]["vision_backend"] == "classical"
    exported = client.get("/api/profiles/export?name=Imported%20CV")
    assert exported.status_code == 200
    document = json.loads(exported.get_data(as_text=True))
    assert document["name"] == "Imported CV"
    assert document["settings"]["min_speed_kmh"] == pytest.approx(6.5)


def test_avis_adaptive_roi_reconfiguration_resets_stale_expansion():
    settings = dict(DEFAULT_AVIS_SETTINGS)
    adaptive = AvisAdaptiveROIController(settings)
    width = 1000
    left, right = settings["ll_left_roi"], settings["ll_right_roi"]
    for ratio in (0.60, 0.42, 0.24, 0.16, 0.10):
        adaptive.observe((left + (right-left)*ratio)*width, None, width)
    assert adaptive.snapshot()["ll"]["expanded_left"] > 0

    adaptive.configure({"adaptive_roi_enabled": False})
    assert adaptive.snapshot()["ll"]["expanded_left"] == 0
    adaptive.configure({"adaptive_roi_enabled": True})
    assert adaptive.snapshot()["ll"]["expanded_left"] == 0


def test_avis_classical_vision_runs_hough_inside_temporarily_expanded_roi(monkeypatch):
    settings = dict(DEFAULT_AVIS_SETTINGS)
    adaptive = AvisAdaptiveROIController(settings)
    frame_width = 200
    left, right = settings["ll_left_roi"], settings["ll_right_roi"]
    for ratio in (0.60, 0.42, 0.24, 0.16, 0.10):
        adaptive.observe(
            (left + (right - left) * ratio) * frame_width,
            None,
            frame_width,
        )
    expected_width = int(round(
        (right - left + adaptive.snapshot()["ll"]["expanded_left"]) * frame_width
    ))
    assert adaptive.snapshot()["ll"]["expanded_left"] > 0

    processor = AvisClassicalVisionProcessor(
        settings,
        adaptive_roi_controller=adaptive,
    )
    observed_shapes = []
    def fake_threshold(roi_frame, _settings):
        observed_shapes.append(roi_frame.shape[:2])
        return np.zeros(roi_frame.shape[:2], dtype=np.uint8)

    monkeypatch.setattr(
        AvisClassicalVisionProcessor,
        "_threshold_roi",
        staticmethod(fake_threshold),
    )
    monkeypatch.setattr(
        "avis_classical_vision.cv2.HoughLinesP",
        lambda *args, **kwargs: None,
    )
    frame = np.zeros((100, frame_width, 3), dtype=np.uint8)
    processor.detect(None, rgb_frame=frame)

    base_width = int(round((right - left) * frame_width))
    assert expected_width > base_width
    assert observed_shapes[0][1] == expected_width


def test_avis_github_profile_paths_encode_spaces_once():
    from avis_profiles import AvisGitHubProfileSync

    sync = AvisGitHubProfileSync()
    assert sync._filename("Night Track") == "Night Track.json"
    assert sync._url("avis_profiles/Night Track.json").endswith(
        "/avis_profiles/Night%20Track.json?ref=main"
    )
    assert sync._url("avis_profiles/پروفایل.json").endswith(
        "/avis_profiles/%D9%BE%D8%B1%D9%88%D9%81%D8%A7%DB%8C%D9%84.json?ref=main"
    )



def _model_obstacle_prediction(
    obstacle_yes=0.95,
    lane_left=0.90,
):
    return {
        "valid": True,
        "obstacle_ahead": {
            "label": "yes" if obstacle_yes >= 0.5 else "no",
            "confidence": max(obstacle_yes, 1.0 - obstacle_yes),
            "probabilities": {"no": 1.0 - obstacle_yes, "yes": obstacle_yes},
        },
        "lane": {
            "label": "left" if lane_left >= 0.5 else "right",
            "confidence": max(lane_left, 1.0 - lane_left),
            "probabilities": {"right": 1.0 - lane_left, "left": lane_left},
        },
    }


def test_model_based_obstacle_trigger_requires_yes_and_left_lane_consensus():
    controller = AvisModelObstacleLaneChangeController()
    settings = dict(DEFAULT_AVIS_SETTINGS)
    settings["obstacle_model_confirm_frames"] = 2
    prediction = _model_obstacle_prediction()

    first = controller.update(
        sensors=[1500, 1500, 1500],
        speed_kmh=8.0, dt=0.1,
        left_x=100, right_x=220, settings=settings,
        classifier_output=prediction,
        observation_id=1, frame_width=320, now=10.0,
    )
    assert first["phase"] == controller.RIGHT_LANE
    assert first["obstacle_model_confirm_frames"] == 1

    second = controller.update(
        sensors=[1500, 1500, 1500],
        speed_kmh=8.0, dt=0.1,
        left_x=100, right_x=220, settings=settings,
        classifier_output=prediction,
        observation_id=2, frame_width=320, now=10.1,
    )
    assert second["phase"] == controller.STEER_LEFT
    assert second["obstacle_trigger_sensor"] == "MODEL"
    assert second["steering_override"] < 0


def test_model_based_obstacle_trigger_rejects_right_lane_prediction():
    controller = AvisModelObstacleLaneChangeController()
    settings = dict(DEFAULT_AVIS_SETTINGS)
    settings["obstacle_model_confirm_frames"] = 1
    prediction = _model_obstacle_prediction(lane_left=0.10)

    state = controller.update(
        sensors=[100, 100, 100],
        speed_kmh=8.0, dt=0.1,
        left_x=100, right_x=220, settings=settings,
        classifier_output=prediction,
        observation_id=1, frame_width=320, now=10.0,
    )
    assert state["phase"] == controller.RIGHT_LANE
    assert state["obstacle_model_confirm_frames"] == 0


def test_right_sensor_must_see_then_clear_obstacle_before_return_delay_starts():
    controller = AvisModelObstacleLaneChangeController()
    settings = dict(DEFAULT_AVIS_SETTINGS)
    settings.update({
        "obstacle_pass_sensor_presence_frames": 2,
        "obstacle_pass_sensor_clear_frames": 3,
        "obstacle_return_delay_seconds": 2.0,
        "obstacle_pass_sensor_timeout_seconds": 30.0,
    })
    controller.phase = controller.WAIT_OBSTACLE_PASS
    controller.pass_sensor_started_at = 100.0

    def tick(sensor_cm, observation_id, now):
        return controller.update(
            sensors=[1500, 1500, sensor_cm],
            speed_kmh=8.0, dt=0.1,
            left_x=100, right_x=220, settings=settings,
            classifier_output=None,
            observation_id=observation_id,
            frame_width=320,
            now=now,
        )

    first_seen = tick(200, 1, 100.0)
    assert first_seen["phase"] == controller.WAIT_OBSTACLE_PASS
    assert first_seen["pass_sensor_stage"] == "WAITING_FOR_OBSTACLE"

    confirmed_seen = tick(200, 2, 100.1)
    assert confirmed_seen["pass_sensor_stage"] == "OBSTACLE_SEEN_ON_RIGHT"

    tick(800, 3, 100.2)
    tick(800, 4, 100.3)
    cleared = tick(800, 5, 100.4)
    assert cleared["phase"] == controller.RETURN_DELAY
    assert cleared["pass_sensor_stage"] == "OBSTACLE_CLEARED"
    assert cleared["return_delay_remaining_s"] == pytest.approx(2.0)

    waiting = tick(800, 6, 102.3)
    assert waiting["phase"] == controller.RETURN_DELAY
    assert waiting["return_delay_remaining_s"] == pytest.approx(0.1)

    returned = tick(800, 7, 102.5)
    assert returned["phase"] == controller.STEER_RIGHT
    assert returned["return_reason"] == "RIGHT_SENSOR_CONFIRMED_PASS"
    assert returned["sensor_angle_override"] == settings["sensor_angle"]


def test_pass_sensor_timeout_fails_safe_without_blind_return():
    controller = AvisModelObstacleLaneChangeController()
    settings = dict(DEFAULT_AVIS_SETTINGS)
    settings["obstacle_pass_sensor_timeout_seconds"] = 5.0
    controller.phase = controller.WAIT_OBSTACLE_PASS
    controller.pass_sensor_started_at = 100.0

    state = controller.update(
        sensors=[1500, 1500, 1500],
        speed_kmh=8.0, dt=0.1,
        left_x=100, right_x=220, settings=settings,
        classifier_output=None,
        observation_id=1, frame_width=320, now=106.0,
    )
    assert state["phase"] == controller.FAILED
    assert state["safe_stop"] is True
    assert state["pass_sensor_timeout"] is True



def test_avis_onnx_classifier_uses_roadforge_task_metadata_and_training_preprocess(
    tmp_path, monkeypatch
):
    tasks = [
        {
            "key": "obstacle_ahead",
            "name": "Obstacle ahead",
            "classes": ["no", "yes"],
            "output_name": "logits_obstacle_ahead",
            "output_index": 0,
        },
        {
            "key": "lane",
            "name": "Lane",
            "classes": ["right", "left"],
            "output_name": "logits_lane",
            "output_index": 1,
        },
    ]
    metadata = {
        "task": "classification",
        "roadforge_format": "avis_multitask_classification_v1",
        "backbone": "resnet18",
        "input_size": "224",
        "normalization_mean": json.dumps([0.485, 0.456, 0.406]),
        "normalization_std": json.dumps([0.229, 0.224, 0.225]),
        "tasks": json.dumps(tasks),
    }

    class FakeSession:
        def __init__(self):
            self.feed = None

        def get_modelmeta(self):
            return SimpleNamespace(custom_metadata_map=metadata)

        def get_inputs(self):
            return [
                SimpleNamespace(
                    name="images",
                    type="tensor(float)",
                    shape=["batch", 3, 224, 224],
                )
            ]

        def get_outputs(self):
            return [
                SimpleNamespace(name="logits_obstacle_ahead"),
                SimpleNamespace(name="logits_lane"),
            ]

        def get_providers(self):
            return self.providers

        def run(self, output_names, feeds):
            assert output_names == ["logits_obstacle_ahead", "logits_lane"]
            self.feed = feeds["images"]
            return [
                np.asarray([[0.0, 2.0]], dtype=np.float32),
                np.asarray([[2.0, 0.0]], dtype=np.float32),
            ]

    fake_session = FakeSession()
    dll_events = []

    def create_session(_path, providers):
        dll_events.append("session")
        fake_session.providers = list(providers)
        return fake_session

    fake_ort = SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        preload_dlls=lambda: dll_events.append("preload"),
        InferenceSession=create_session,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    checkpoint = tmp_path / "avis_classifier.onnx"
    checkpoint.write_bytes(b"mock ONNX placeholder")

    classifier = AvisObstacleLaneClassifier(checkpoint)
    assert dll_events == ["preload", "session"]
    assert classifier.providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    frame_bgr = np.empty((32, 48, 3), dtype=np.uint8)
    frame_bgr[:] = [10, 128, 240]  # BGR input must become RGB [240, 128, 10].
    prediction = classifier.predict(frame_bgr)

    assert prediction["valid"] is True
    assert prediction["backbone"] == "resnet18"
    assert prediction["obstacle_ahead"]["label"] == "yes"
    assert prediction["lane"]["label"] == "right"
    assert prediction["obstacle_ahead"]["probabilities"]["yes"] > 0.85
    assert prediction["lane"]["probabilities"]["right"] > 0.85

    assert fake_session.feed.shape == (1, 3, 224, 224)
    assert fake_session.feed.dtype == np.float32
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    rgb_values = np.asarray([240.0, 128.0, 10.0], dtype=np.float32) / 255.0
    expected = (rgb_values - mean) / std
    for channel in range(3):
        np.testing.assert_allclose(
            fake_session.feed[0, channel],
            expected[channel],
            rtol=0.0,
            atol=1e-6,
        )
