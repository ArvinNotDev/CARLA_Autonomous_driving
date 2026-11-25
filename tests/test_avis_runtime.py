import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from avis_runtime import (
    AVIS_PROFILE_FIELDS,
    AVIS_PROFILE_NAME,
    AvisEngineClient,
    AvisSteeringController,
    apply_avis_profile,
    load_avis_profile,
    normalize_avis_controls,
    normalize_avis_manual_controls,
    select_avis_target_speed,
    _create_avis_vision_processor,
)

from avis_classical_vision import AvisClassicalVisionProcessor
from avis_dashboard import (
    DEFAULT_AVIS_SETTINGS,
    AvisDashboardState,
    create_avis_dashboard_app,
    validate_avis_settings,
)


ROOT = Path(__file__).resolve().parents[1]


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
    assert avis["MODEL_PATH"] == stable["MODEL_PATH"]
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
    config = SimpleNamespace(KP=0.0, LIDAR_ENABLED=True)
    apply_avis_profile(
        {"KP": 0.02, "LIDAR_ENABLED": False},
        config,
    )

    assert config.KP == 0.02
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
    assert 'id="drivableView"' in html
    assert "does not construct or calculate a drivable-area mask" in html


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



def test_avis_model_debug_suppresses_vehicle_mask_without_mutating_shared_processor(monkeypatch):
    import vision.city_vision_processing as shared_vision

    received = {}

    class FakeVisionProcessor:
        def __init__(self, mode):
            received["mode"] = mode

        def _draw_debug(self, **kwargs):
            received["vehicles_mask"] = kwargs.get("vehicles_mask")
            return "debug-frame"

    monkeypatch.setattr(shared_vision, "VisionProcessor", FakeVisionProcessor)
    processor = _create_avis_vision_processor("ultralytics", {})
    result = processor._draw_debug(
        frame=None,
        line_mask=None,
        drivable_mask=None,
        left_x=None,
        right_x=None,
        lane_center=0.0,
        lane_type="none",
        vehicles_mask=object(),
    )

    assert result == "debug-frame"
    assert received["mode"] == "ultralytics"
    assert received["vehicles_mask"] is None
