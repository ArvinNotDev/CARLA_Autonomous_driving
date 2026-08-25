import unittest

from runtime_metrics import RollingFps


class TestRollingFps(unittest.TestCase):
    def test_event_based_fps(self):
        meter = RollingFps(window_seconds=1.0)
        meter.record(0.0)
        meter.record(0.1)
        meter.record(0.2)
        self.assertAlmostEqual(meter.value(0.2), 10.0, places=6)

    def test_old_events_are_dropped_by_time_window(self):
        meter = RollingFps(window_seconds=1.0)
        for t in (0.0, 0.1, 0.2, 1.3):
            meter.record(t)
        self.assertAlmostEqual(meter.value(1.3), 0.0, places=6)
