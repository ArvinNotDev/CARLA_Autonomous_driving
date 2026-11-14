import sys
import threading
import time
import types

import numpy as np
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


class FakePlanner:
    def __init__(self, instruction="CHANGE_LANE_LEFT"):
        self.instruction = instruction

    def get_next_maneuver_text(self, current, goal):
        return self.instruction


@pytest.fixture(autouse=True)
def carla_stub(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "carla",
        types.SimpleNamespace(VehicleControl=VehicleControl),
    )
    monkeypatch.setattr(conf, "LANE_CHANGE_VERIFY_DISTANCE_M", 1.0, raising=False)
    monkeypatch.setattr(conf, "LANE_CHANGE_STEER_DISTANCE_M", 2.0, raising=False)
    monkeypatch.setattr(conf, "LANE_CHANGE_STEER_VALUE", 0.2, raising=False)
    monkeypatch.setattr(conf, "LANE_CHANGE_PLANNER_CHECK_INTERVAL_SECONDS", 0.05, raising=False)


def make_active_manager(instruction="CHANGE_LANE_LEFT"):
    planner = FakePlanner(instruction)
    manager = LaneChangeManager(
        vehicle=object(),
        planner=planner,
        lane_change_debounce_seconds=0.0,
    )
    manager._latest_planner_result = (1, "left")
    manager.update(Location(0.0), Location(100.0))
    assert manager.phase_name() == "VERIFY"
    manager.update(Location(1.0), Location(100.0))
    assert manager.phase_name() == "CHANGE"
    return manager, planner


def finish_attempt_and_wait_for_gps(manager, x, goal=Location(100.0)):
    manager.update(Location(x), goal)
    assert manager.phase_name() == "CHECK"

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        with manager._state_lock:
            ready = manager._latest_active_planner_result is not None
        if ready:
            break
        time.sleep(0.005)
    else:
        pytest.fail("Active lane-change GPS query did not return")

    manager.update(Location(x + 0.05), goal)


def test_lane_change_uses_two_steered_then_two_straight_attempts_and_repeats():
    manager, planner = make_active_manager("CHANGE_LANE_LEFT")

    control = manager.control_override()
    assert control.steer == pytest.approx(-0.2)

    # First steered attempt: GPS still requests the same lane change.
    finish_attempt_and_wait_for_gps(manager, 3.0)
    assert manager.phase_name() == "CHANGE"
    assert manager.control_override().steer == pytest.approx(-0.2)

    # Second steered attempt: switch to straight attempts.
    finish_attempt_and_wait_for_gps(manager, 5.1)
    assert manager.phase_name() == "STRAIGHT"
    assert manager.control_override().steer == pytest.approx(0.0)

    # First straight attempt: remain straight.
    finish_attempt_and_wait_for_gps(manager, 7.2)
    assert manager.phase_name() == "STRAIGHT"
    assert manager.control_override().steer == pytest.approx(0.0)

    # Second straight attempt: start a new two-steer/two-straight cycle.
    finish_attempt_and_wait_for_gps(manager, 9.3)
    assert manager.phase_name() == "CHANGE"
    assert manager.control_override().steer == pytest.approx(-0.2)

    # On the next GPS check, the planner no longer requests a lane change.
    planner.instruction = "LANEFOLLOW"
    finish_attempt_and_wait_for_gps(manager, 11.4)
    assert manager.phase_name() == "IDLE"
    assert not manager.movement_active()
    assert manager.control_override() is None


def test_opposite_gps_lane_request_keeps_lane_change_control_owned():
    manager, planner = make_active_manager("CHANGE_LANE_LEFT")
    finish_attempt_and_wait_for_gps(manager, 3.0)
    planner.instruction = "CHANGE_LANE_RIGHT"

    # Complete another attempt; a changed direction starts a new VERIFY phase
    # without a lane-follow/PID handoff in between.
    finish_attempt_and_wait_for_gps(manager, 5.1)
    assert manager.phase_name() == "VERIFY"
    assert manager.direction == "right"
    assert manager.movement_active()
    assert manager.control_override().steer == pytest.approx(0.0)


def test_lane_follow_pid_is_not_called_while_gps_lane_change_is_active(monkeypatch):
    # Load AutoDriver with lightweight CARLA/utility stubs.
    uv = types.ModuleType("utils.vehicle_utils")
    uv.blank_frame = lambda: np.zeros((8, 8, 3), dtype=np.uint8)
    uv.make_stop_control = lambda: VehicleControl(throttle=0.0, brake=1.0)
    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = []
    utils_pkg.vehicle_utils = uv
    monkeypatch.setitem(sys.modules, "utils", utils_pkg)
    monkeypatch.setitem(sys.modules, "utils.vehicle_utils", uv)

    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "driving" / "auto_driver.py"
    spec = importlib.util.spec_from_file_location("auto_driver_gps_lane_change_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    manager, _ = make_active_manager()
    manager.phase = "STRAIGHT"
    manager._segment_start_location = Location(1.0)

    class Controller:
        calls = 0

        def update(self, error):
            self.calls += 1
            return VehicleControl(throttle=0.2, steer=0.6)

    controller = Controller()
    driver = module.AutoDriver(
        vehicle=object(),
        world=object(),
        controller=controller,
        intersection_model=None,
        lane_change_manager=manager,
        intersection_manager=None,
    )

    _, _, control = driver.update(
        np.zeros((8, 8, 3), dtype=np.uint8),
        {
            "error": 10.0,
            "lane_type": "only_left",
            "debug": {
                "combined": None,
                "left_line_angle_deg": 45.0,
                "right_line_angle_deg": -45.0,
            },
        },
        Location(1.2),
        Location(100.0),
        False,
    )

    assert manager.phase_name() == "STRAIGHT"
    assert control.steer == pytest.approx(0.0)
    assert controller.calls == 0
