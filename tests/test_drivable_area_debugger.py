import unittest
from unittest.mock import patch

import numpy as np

import config_city as conf
from vision.drivable_area_debugger import DrivableAreaDebugger


class TestDrivableAreaWidthIndex(unittest.TestCase):
    def test_width_index_uses_dw_roi_and_median_row_span(self):
        debugger = DrivableAreaDebugger()
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[20:80, 65:135] = 255

        with (
            patch.object(conf, "DW_TOP_ROI", 0.10),
            patch.object(conf, "DW_BOTTOM_ROI", 0.90),
            patch.object(conf, "DW_LEFT_ROI", 0.25),
            patch.object(conf, "DW_RIGHT_ROI", 0.75),
            patch.object(conf, "CW_TOP_ROI", 0.0),
            patch.object(conf, "CW_BOTTOM_ROI", 1.0),
            patch.object(conf, "CW_LEFT_ROI", 0.0),
            patch.object(conf, "CW_RIGHT_ROI", 1.0),
        ):
            result = debugger.show({"debug": {"drivable_mask": mask}})

        self.assertEqual(result["roi"], (50, 10, 150, 90))
        self.assertEqual(result["road_width_index_px"], 70)
        self.assertAlmostEqual(result["road_width_ratio"], 0.35)
        self.assertEqual(result["road_valid_rows"], 60)
        self.assertEqual(result["left_x"], 65)
        self.assertEqual(result["right_x"], 134)

    def test_empty_roi_mask_has_no_width_index(self):
        debugger = DrivableAreaDebugger()
        mask = np.zeros((100, 200), dtype=np.uint8)
        result = debugger.show({"debug": {"drivable_mask": mask}})
        self.assertIsNone(result["road_width_index_px"])
        self.assertEqual(result["road_valid_rows"], 0)

