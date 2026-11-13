import sys
import types

import pytest

import config_city as conf
from navigation.lane_change_manager import LaneChangeManager


class Location:
    def __init__(self, x):
        self.x = float(x)

    def distance(self, other):
        return abs(self.x - other.x)


class VehicleControl:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@pytest.fixture(autouse=True)
def carla_stub(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "carla",
        types.SimpleNamespace(VehicleControl=VehicleControl),
    )
    monkeypatch.setattr(conf, "LANE_CHANGE_VERIFY_DISTANCE_M", 1.0, raising=False)
    monkeypatch.setattr(conf, "LANE_CHANGE_STEER_DISTANCE_M", 2.0, raising=False)
    monkeypatch.setattr(conf, "LANE_CHANGE_ANGLE_ENTER_DEG", 10.0, raising=False)
    monkeypatch.setattr(conf, "LANE_CHANGE_ANGLE_SETTLE_DEG", 6.0, raising=False)
    monkeypatch.setattr(conf, "LANE_CHANGE_CONFIRM_FRAMES", 3, raising=False)
    monkeypatch.setattr(conf, "LANE_CHANGE_SETTLE_STEER", 0.08, raising=False)


def make_manager(direction="left"):
    manager = LaneChangeManager(vehicle=object(), planner=object())
    manager.phase = "CHANGE"
    manager.direction = direction
    manager._segment_start_location = Location(0)
    manager._baseline_line_angles = {"left": -15.0, "right": 15.0}
    return manager


def test_lane_change_keeps_steering_override_after_initial_distance():
    manager = make_manager("left")

    manager.update(
        Location(2.1),
        None,
        line_angles={"left": -13.0, "right": 14.0},
        lane_type="only_left",
        source_frame_id=10,
    )

    assert manager.phase_name() == "SETTLE"
    assert manager.movement_active()
    control = manager.control_override()
    assert control is not None
    assert control.steer == pytest.approx(-0.08)


def test_lane_change_releases_only_after_angle_transition_and_stable_frames():
    manager = make_manager("right")

    manager.update(
        Location(0.5),
        None,
        line_angles={"left": -27.0, "right": 16.0},
        lane_type="only_right",
        source_frame_id=10,
    )
    assert manager.phase_name() == "CHANGE"
    assert manager._angle_transition_seen

    manager.update(
        Location(2.0),
        None,
        line_angles={"left": -18.0, "right": 17.0},
        lane_type="only_right",
        source_frame_id=11,
    )
    assert manager.phase_name() == "SETTLE"
    assert manager._stable_angle_frames == 1

    # Repeated control ticks with the same vision frame must not count twice.
    manager.update(
        Location(2.1),
        None,
        line_angles={"left": -15.0, "right": 15.0},
        lane_type="only_right",
        source_frame_id=11,
    )
    assert manager._stable_angle_frames == 1

    manager.update(
        Location(2.2),
        None,
        line_angles={"left": -15.0, "right": 15.0},
        lane_type="only_right",
        source_frame_id=12,
    )
    assert manager.phase_name() == "SETTLE"
    assert manager.control_override() is not None

    manager.update(
        Location(2.3),
        None,
        line_angles={"left": -15.0, "right": 15.0},
        lane_type="only_right",
        source_frame_id=13,
    )
    assert manager.phase_name() == "IDLE"
    assert not manager.movement_active()
    assert manager.control_override() is None


def test_lane_change_does_not_confirm_without_lane_markings():
    manager = make_manager("left")

    for frame_id, distance in enumerate((0.5, 2.0, 2.2, 2.4, 2.6), start=1):
        manager.update(
            Location(distance),
            None,
            line_angles={"left": -15.0, "right": 15.0},
            lane_type="none",
            source_frame_id=frame_id,
        )

    assert manager.phase_name() == "SETTLE"
    assert manager.control_override() is not None


def test_lane_change_does_not_confirm_from_angle_without_transition():
    manager = make_manager("left")

    for frame_id, distance in enumerate((0.5, 2.0, 2.2, 2.4, 2.6), start=1):
        manager.update(
            Location(distance),
            None,
            line_angles={"left": -15.0, "right": 15.0},
            lane_type="only_left",
            source_frame_id=frame_id,
        )

    assert manager.phase_name() == "SETTLE"
    assert manager._angle_transition_seen is False
    assert manager.control_override() is not None


def test_unconfirmed_lane_change_fails_safe_to_full_brake(monkeypatch):
    monkeypatch.setattr(conf, "LANE_CHANGE_CONFIRM_TIMEOUT_SECONDS", 1.0, raising=False)
    manager = make_manager("left")
    manager.phase = "SETTLE"
    manager._steering_distance_reached = True
    manager._angle_transition_seen = True
    manager._settle_started_at = __import__("time").monotonic() - 2.0

    manager.update(
        Location(2.1),
        None,
        line_angles={"left": -13.0, "right": 14.0},
        lane_type="only_left",
        source_frame_id=50,
    )

    control = manager.control_override()
    assert manager.phase_name() == "SETTLE"
    assert manager.movement_active()
    assert control.throttle == 0.0
    assert control.brake == 1.0


def test_auto_driver_does_not_return_steering_to_lane_pid_during_settle(monkeypatch):
    # AutoDriver is intentionally loaded with lightweight CARLA/utility stubs.
    uv = types.ModuleType("utils.vehicle_utils")
    uv.blank_frame = lambda: __import__("numpy").zeros((8, 8, 3), dtype="uint8")
    uv.make_stop_control = lambda: VehicleControl(throttle=0.0, brake=1.0)
    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = []
    utils_pkg.vehicle_utils = uv
    monkeypatch.setitem(sys.modules, "utils", utils_pkg)
    monkeypatch.setitem(sys.modules, "utils.vehicle_utils", uv)

    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "driving" / "auto_driver.py"
    spec = importlib.util.spec_from_file_location("auto_driver_lane_change_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    manager = make_manager("left")
    manager.phase = "SETTLE"
    manager._segment_start_location = Location(2.0)
    manager._steering_distance_reached = True
    manager._angle_transition_seen = True
    manager._stable_angle_frames = 0

    class Controller:
        calls = 0

        def update(self, error):
            self.calls += 1
            return VehicleControl(throttle=0.2, steer=0.0)

    controller = Controller()
    driver = module.AutoDriver(
        vehicle=object(),
        world=object(),
        controller=controller,
        intersection_model=None,
        lane_change_manager=manager,
        intersection_manager=None,
    )

    import numpy as np
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    for frame_id in (20, 21):
        _, _, control = driver.update(
            frame,
            {
                "error": 10.0,
                "lane_type": "only_left",
                "_source_frame_id": frame_id,
                "debug": {
                    "combined": None,
                    "left_line_angle_deg": -15.0,
                    "right_line_angle_deg": 15.0,
                },
            },
            Location(2.0),
            None,
            False,
        )
        assert control.steer == pytest.approx(-0.08)
        assert controller.calls == 0

    # The third distinct stable frame confirms the destination lane. Only
    # after that does AutoDriver hand steering back to the lane-follow PID.
    _, _, control = driver.update(
        frame,
        {
            "error": 10.0,
            "lane_type": "only_left",
            "_source_frame_id": 22,
            "debug": {
                "combined": None,
                "left_line_angle_deg": 2.0,
                "right_line_angle_deg": -2.0,
            },
        },
        Location(2.0),
        None,
        False,
    )
    assert manager.phase_name() == "IDLE"
    assert controller.calls == 1
    assert control.steer == 0.0
