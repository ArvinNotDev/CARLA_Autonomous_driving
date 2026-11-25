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
    select_avis_target_speed,
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
