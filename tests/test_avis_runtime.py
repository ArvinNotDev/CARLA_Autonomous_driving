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
)

from avis_classical_vision import AvisClassicalVisionProcessor
from avis_dashboard import AvisDashboardState, create_avis_dashboard_app


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


def test_avis_classical_vision_finds_synthetic_lane_lines_and_masks():
    import cv2
    import numpy as np

    frame = np.zeros((192, 320, 3), dtype=np.uint8)
    cv2.line(frame, (42, 188), (145, 105), (255, 255, 255), 6)
    cv2.line(frame, (178, 105), (278, 188), (255, 255, 255), 6)

    processor = AvisClassicalVisionProcessor()
    result = processor.detect(None, frame)
    debug = result["debug"]

    assert result["lane_valid"] is True
    assert result["lane_type"] == "both_sides"
    assert abs(result["error"]) < 20
    assert debug["lane_mask"].shape == frame.shape[:2]
    assert debug["drivable_mask"].shape == frame.shape[:2]
    assert np.count_nonzero(debug["lane_mask"]) > 0
    assert np.count_nonzero(debug["drivable_mask"]) > 0
    assert debug["combined"].shape == frame.shape


def test_avis_classical_vision_stops_when_no_lane_is_detected():
    import numpy as np

    processor = AvisClassicalVisionProcessor()
    result = processor.detect(None, np.zeros((192, 320, 3), dtype=np.uint8))

    assert result["lane_valid"] is False
    assert result["error"] == 0.0
    assert np.count_nonzero(result["debug"]["drivable_mask"]) == 0



def test_avis_client_sends_reverse_speed_only_for_manual_controls():
    api = SimpleNamespace(Car=FakeCar)
    client = AvisEngineClient(api, "127.0.0.1", 25001)
    fake_car = client.car

    client.send_manual_control(-0.3, -0.2)
    client.close()

    assert fake_car.speed_commands == [-30, 0]
    assert fake_car.steering_commands == [-20, 0]
