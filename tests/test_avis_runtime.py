import json
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
    apply_avis_profile,
    load_avis_profile,
    normalize_avis_controls,
    normalize_avis_manual_controls,
    select_avis_target_speed,
    select_avis_obstacle_approach_speed,
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


@pytest.mark.parametrize(
    ("distance_cm", "expected_speed"),
    [
        (2000, 12.0),
        (1500, 12.0),
        (1225, 10.0),
        (950, 8.0),
        (500, 8.0),
        (None, 12.0),
    ],
)
def test_avis_obstacle_approach_speed_tapers_from_1500_to_950_cm(
    distance_cm, expected_speed
):
    assert select_avis_obstacle_approach_speed(
        12.0,
        distance_cm,
        8.0,
        slowdown_start_cm=1500.0,
        lane_change_trigger_cm=950.0,
    ) == pytest.approx(expected_speed)


def test_avis_obstacle_approach_speed_requires_slowdown_start_beyond_trigger():
    with pytest.raises(ValueError, match="slowdown start above trigger"):
        select_avis_obstacle_approach_speed(
            12.0, 1200.0, 8.0,
            slowdown_start_cm=950.0,
            lane_change_trigger_cm=950.0,
        )


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


def _obstacle_lane_change_test_settings(**overrides):
    settings = dict(DEFAULT_AVIS_SETTINGS)
    settings.update({
        "obstacle_lane_change_enabled": True,
        "obstacle_front_threshold_cm": 700,
        "obstacle_trigger_confirm_frames": 2,
        "obstacle_sensor_clear_threshold_cm": 900,
        "obstacle_sensor_clear_confirm_frames": 2,
        "obstacle_lane_change_steer_distance_m": 2.0,
        "obstacle_lane_change_left_steer": 0.20,
        "obstacle_lane_change_right_steer": 0.20,
        "obstacle_lane_line_confirm_frames": 2,
        "obstacle_lane_line_search_max_distance_m": 20.0,
        "obstacle_clearance_after_pass_m": 5.0,
    })
    settings.update(overrides)
    return settings


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
    assert second["steering_override"] == pytest.approx(-0.20)
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
    for observation_id in (5, 6):
        state = controller.update(
            sensors=[1500, 600, 1500], speed_kmh=0.0, dt=0.0,
            left_x=100, right_x=None, settings=settings,
            observation_id=observation_id,
        )
    assert state["phase"] == "LEFT_LANE"

    # The first obstacle was detected at 6m; with 5m extra clearance,
    # the controller should return after 11m in the left lane (not a fixed 750m).
    for observation_id in range(7, 18):
        state = controller.update(
            sensors=[1500, 1500, 1500], speed_kmh=36.0, dt=0.1,
            left_x=100, right_x=220, settings=settings,
            observation_id=observation_id,
        )
    assert state["left_lane_target_distance_m"] == pytest.approx(11.0)
    assert state["left_lane_distance_m"] == pytest.approx(11.0)
    assert state["phase"] == "STEER_RIGHT"
    assert state["steering_override"] == pytest.approx(0.20)

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
    assert state["steering_override"] == pytest.approx(0.20)
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
    assert state["phase"] == "ACQUIRE_LEFT_LINE"
    assert state["search_limit_reached"] is True
    assert state["safe_stop"] is False


def test_avis_dashboard_lane_change_settings_are_persistent_and_validated(tmp_path):
    state = AvisDashboardState(settings_path=tmp_path / "avis_settings.json")
    updated = state.update_settings({
        "obstacle_lane_change_enabled": True,
        "obstacle_slowdown_start_cm": 1600,
        "obstacle_front_threshold_cm": 950,
        "obstacle_clearance_after_pass_m": 6.5,
        "obstacle_lane_change_left_steer": 0.24,
    })
    assert updated["obstacle_slowdown_start_cm"] == 1600
    assert updated["obstacle_front_threshold_cm"] == 950
    assert updated["obstacle_clearance_after_pass_m"] == pytest.approx(6.5)
    assert updated["obstacle_lane_change_left_steer"] == pytest.approx(0.24)

    restored = AvisDashboardState(settings_path=tmp_path / "avis_settings.json")
    restored_settings, _ = restored.settings_snapshot()
    assert restored_settings["obstacle_clearance_after_pass_m"] == pytest.approx(6.5)
    assert restored_settings["obstacle_slowdown_start_cm"] == 1600

    with pytest.raises(ValueError, match="slowdown start must be greater"):
        state.update_settings({
            "obstacle_slowdown_start_cm": 1499,
            "obstacle_front_threshold_cm": 1499,
        }, persist=False)

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


def test_avis_dashboard_exposes_obstacle_lane_change_settings(tmp_path):
    state = AvisDashboardState(settings_path=tmp_path / "avis_settings.json")
    html = create_avis_dashboard_app(state).test_client().get("/").get_data(as_text=True)
    assert 'id="obstacle_lane_change_enabled"' in html
    assert 'id="obstacle_front_threshold_cm"' in html
    assert 'id="obstacle_slowdown_start_cm"' in html
    assert 'id="obstacle_clearance_after_pass_m"' in html
    assert 'id="obstacle_return_distance_m"' not in html
    assert 'id="obstacle_slowdown_start_cm" data-setting="obstacle_slowdown_start_cm" type="number" min="951" max="6000" step="1"' in html
    assert 'id="obstacle_front_threshold_cm" data-setting="obstacle_front_threshold_cm" type="number" min="50" max="1499" step="1"' in html
    assert 'id="obstacle_sensor_clear_threshold_cm" data-setting="obstacle_sensor_clear_threshold_cm" type="number" min="51" max="1500" step="1"' in html
    assert 'id="obstacle_lane_line_search_max_distance_m"' in html
    assert 'id="settingsForm" novalidate' in html
    assert 'id="obstacle_lane_change_steer_distance_m" data-setting="obstacle_lane_change_steer_distance_m" type="number" min=".1" max="100" step="any"' in html
    assert 'id="obstacle_lane_change_left_steer" data-setting="obstacle_lane_change_left_steer" type="number" min=".001" max="1" step="any"' in html
    assert 'id="obstacle_clearance_after_pass_m" data-setting="obstacle_clearance_after_pass_m" type="number" min="0" max="1000" step="any"' in html
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



def test_avis_model_allows_missing_vehicle_class_without_changing_shared_processor(monkeypatch):
    import numpy as np
    import vision.city_vision_processing as shared_vision

    class FakeVisionProcessor:
        def __init__(self, mode):
            self.mode = mode
            self.vehicles_class_id = None

        @staticmethod
        def _semantic_map_to_masks(
            semantic_map,
            lane_class_id,
            drivable_class_id,
            vehicles_class_id,
            frame_shape,
        ):
            class_map = np.asarray(semantic_map).squeeze()
            if class_map.shape != frame_shape:
                class_map = np.resize(class_map, frame_shape)
            drivable = (class_map == drivable_class_id).astype(np.uint8) * 255
            lane = (class_map == lane_class_id).astype(np.uint8) * 255
            vehicles = (class_map == vehicles_class_id).astype(np.uint8) * 255
            return drivable, lane, vehicles

    monkeypatch.setattr(shared_vision, "VisionProcessor", FakeVisionProcessor)
    processor = _create_avis_vision_processor("ultralytics", {})

    class_names = {
        0: "background",
        1: "ego_drivable_area",
        2: "lane_lines",
    }
    lane_id, drivable_id, vehicles_id = processor._class_ids_from_names(class_names)
    assert (lane_id, drivable_id, vehicles_id) == (2, 1, None)

    semantic_map = np.array([[0, 1, 2]], dtype=np.uint8)
    drivable_mask, lane_mask, vehicles_mask = processor._semantic_map_to_masks(
        semantic_map,
        lane_class_id=lane_id,
        drivable_class_id=drivable_id,
        vehicles_class_id=vehicles_id,
        frame_shape=(1, 3),
    )

    assert drivable_mask.tolist() == [[0, 255, 0]]
    assert lane_mask.tolist() == [[0, 0, 255]]
    assert vehicles_mask is None


def test_avis_model_still_uses_vehicle_class_when_checkpoint_has_it(monkeypatch):
    import numpy as np
    import vision.city_vision_processing as shared_vision

    class FakeVisionProcessor:
        def __init__(self, mode):
            self.mode = mode
            self.vehicles_class_id = 3

        @staticmethod
        def _semantic_map_to_masks(
            semantic_map,
            lane_class_id,
            drivable_class_id,
            vehicles_class_id,
            frame_shape,
        ):
            class_map = np.asarray(semantic_map).reshape(frame_shape)
            return (
                (class_map == drivable_class_id).astype(np.uint8) * 255,
                (class_map == lane_class_id).astype(np.uint8) * 255,
                (class_map == vehicles_class_id).astype(np.uint8) * 255,
            )

    monkeypatch.setattr(shared_vision, "VisionProcessor", FakeVisionProcessor)
    processor = _create_avis_vision_processor("ultralytics", {})

    assert processor._class_ids_from_names({
        0: "background",
        1: "ego_drivable_area",
        2: "lane_lines",
        3: "vehicles",
    }) == (2, 1, 3)

    _, _, vehicles_mask = processor._semantic_map_to_masks(
        np.array([[3]], dtype=np.uint8),
        lane_class_id=2,
        drivable_class_id=1,
        vehicles_class_id=3,
        frame_shape=(1, 1),
    )
    assert vehicles_mask.tolist() == [[255]]


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
