import time
import unittest

from navigation.intersection_manager import IntersectionManager


class FakeLocation:
    def __init__(self, x):
        self.x = x

    def distance(self, other):
        return abs(self.x - other.x)


class FakePlanner:
    def distance_to_next_maneuver(self, current, goal):
        return 0.0

    def get_next_maneuver_text(self, current, goal):
        return "STRAIGHT"


class TestJunctionManager(unittest.TestCase):
    def test_nonblocking_static_state_machine(self):
        manager = IntersectionManager(object(), FakePlanner())
        started = manager.start_for_intersection(FakeLocation(0), FakeLocation(100))
        self.assertTrue(started)

        deadline = time.time() + 1.0
        while time.time() < deadline and manager.phase_name() == "PLANNING":
            manager.update(FakeLocation(0))
            time.sleep(0.005)

        self.assertEqual(manager.phase_name(), "STATIC")
        control = manager.static_control(FakeLocation(0))
        self.assertIsNotNone(control)
        # Progress the configured straight segment beyond its target.
        for x in (4.0, 8.0, 12.0):
            manager.update(FakeLocation(x))
        self.assertEqual(manager.phase_name(), "IDLE")

    def test_static_control_does_not_sleep(self):
        manager = IntersectionManager(object(), FakePlanner())
        manager._active_plan = type(
            "P", (), {
                "segments": ((2.0, 0.0, True, 0.2, 20.0),),
                "maneuver": "STRAIGHT",
            }
        )()
        manager._segment_index = 0
        manager._segment_start_location = FakeLocation(0)
        manager._segment_started_at = time.monotonic()
        manager.phase = "STATIC"

        started = time.perf_counter()
        manager.static_control(FakeLocation(0.1))
        self.assertLess(time.perf_counter() - started, 0.02)


if __name__ == "__main__":
    unittest.main()
