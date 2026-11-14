import importlib.util
import os
import unittest

from runtime_metrics import DisplayMetrics
from ui.frame_store import DebugFrameStore


@unittest.skipUnless(importlib.util.find_spec("PySide6"), "PySide6 is not installed")
class TestPanelConstruction(unittest.TestCase):
    def test_panel_constructs(self):
        from PySide6.QtWidgets import QApplication
        from runtime_settings import RuntimeSettings
        from ui.control_panel import ControlPanel

        app = QApplication.instance() or QApplication([])
        settings = RuntimeSettings(path=__import__("pathlib").Path("test_runtime_profiles.json"))
        panel = ControlPanel(
            settings,
            display_metrics=DisplayMetrics(),
            frame_store=DebugFrameStore(),
        )
        self.assertIsNotNone(panel.debug_frame)
        self.assertIsNotNone(panel.workspace_splitter)
        self.assertIn("display", panel.metric_values)
        self.assertIsNotNone(panel.settings_search)
        self.assertFalse(panel.stream_toggle.isChecked())
        self.assertGreaterEqual(panel.tabs.count(), 10)

        target_speed = panel.widgets["TARGET_SPEED_KMH"]
        pid_gain = panel.widgets["KP"]
        panel.settings_search.setText("TARGET_SPEED_KMH")
        self.assertFalse(target_speed.isHidden())
        self.assertTrue(pid_gain.isHidden())
        panel.settings_search.clear()
        self.assertFalse(pid_gain.isHidden())
        panel.close()
        try:
            os.remove("test_runtime_profiles.json")
        except OSError:
            pass


if __name__ == "__main__":
    unittest.main()
