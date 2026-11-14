from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

import config_city as conf


class DrivableAreaDebugger:
    """Extract compact road-geometry measurements from the configured DW ROI."""

    def __init__(self, window_name: str = "drivable area") -> None:
        self.window_name = window_name

    def apply_center_roi(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Return the configured road-width ROI as image-space slice bounds."""
        height, width = mask.shape[:2]

        x1 = int(float(getattr(conf, "DW_LEFT_ROI", 0.10)) * width)
        x2 = int(float(getattr(conf, "DW_RIGHT_ROI", 0.90)) * width)
        y1 = int(float(getattr(conf, "DW_TOP_ROI", 0.60)) * height)
        y2 = int(float(getattr(conf, "DW_BOTTOM_ROI", 0.72)) * height)

        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        return x1, y1, x2, y2

    def show(
        self,
        vision_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Measure the median horizontal drivable-mask span inside DW ROI.

        road_width_index_px is a relative comparison index in pixels, not a
        physical road width in metres. The median across valid rows makes the
        value less sensitive to isolated segmentation noise.
        """
        empty_result = {
            "error": None,
            "center_x": None,
            "image_center_x": None,
            "left_x": None,
            "right_x": None,
            "roi": None,
            "roi_width_px": None,
            "roi_height_px": None,
            "road_width_index_px": None,
            "road_width_ratio": None,
            "road_valid_rows": 0,
        }

        if not isinstance(vision_result, dict):
            return empty_result

        try:
            debug = vision_result.get("debug", {})
            drivable_mask = debug.get("drivable_mask") if isinstance(debug, dict) else None
            if drivable_mask is None:
                return empty_result

            mask = np.asarray(drivable_mask)
            if mask.size == 0:
                return empty_result
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2GRAY)
            if mask.ndim != 2:
                return empty_result

            mask_float = mask.astype(np.float32, copy=False)
            max_val = float(mask_float.max()) if mask_float.size else 0.0
            if max_val <= 1.0:
                display_mask = mask_float > (1.0 / 255.0)
            else:
                display_mask = mask_float > 1.0

            height, width = display_mask.shape
            image_center_x = width // 2
            x1, y1, x2, y2 = self.apply_center_roi(display_mask)
            roi_mask = display_mask[y1:y2, x1:x2]
            if roi_mask.size == 0:
                return {
                    **empty_result,
                    "image_center_x": image_center_x,
                    "roi": (x1, y1, x2, y2),
                    "roi_width_px": max(0, x2 - x1),
                    "roi_height_px": max(0, y2 - y1),
                }

            valid_rows = np.flatnonzero(np.any(roi_mask, axis=1))
            road_width_index_px = None
            road_width_ratio = None
            left_x = None
            right_x = None
            center_x = image_center_x

            if valid_rows.size:
                valid_mask_rows = roi_mask[valid_rows]
                roi_width = valid_mask_rows.shape[1]
                row_lefts = np.argmax(valid_mask_rows, axis=1)
                row_rights = roi_width - 1 - np.argmax(valid_mask_rows[:, ::-1], axis=1)
                row_spans = row_rights - row_lefts + 1

                median_span = float(np.median(row_spans))
                road_width_index_px = int(round(median_span))
                road_width_ratio = float(median_span / max(1, width))

                _, all_x = np.where(roi_mask)
                left_x = x1 + int(all_x.min())
                right_x = x1 + int(all_x.max())
                center_x = int(round((left_x + right_x) / 2.0))

            error = image_center_x - center_x
            return {
                "error": error,
                "center_x": center_x,
                "image_center_x": image_center_x,
                "left_x": left_x,
                "right_x": right_x,
                "roi": (x1, y1, x2, y2),
                "roi_width_px": x2 - x1,
                "roi_height_px": y2 - y1,
                "road_width_index_px": road_width_index_px,
                "road_width_ratio": road_width_ratio,
                "road_valid_rows": int(valid_rows.size),
            }
        except Exception:
            return empty_result

    def get_error(self, vision_result: Optional[Dict[str, Any]]) -> Optional[int]:
        """Return only the lateral error used by the recovery controller."""
        result = self.show(vision_result)
        return result["error"]
