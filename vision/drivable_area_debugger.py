from __future__ import annotations

from typing import Any, Optional, Tuple, Dict

import cv2
import numpy as np

import config_city as conf
from utils.vehicle_utils import blank_frame


class DrivableAreaDebugger:
    def __init__(self, window_name: str = "drivable area") -> None:
        self.window_name = window_name

    def apply_center_roi(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        h, w = mask.shape[:2]

        x1 = int(conf.CW_LEFT_ROI * w)
        x2 = int(conf.CW_RIGHT_ROI * w)
        y1 = int(conf.CW_TOP_ROI * h)
        y2 = int(conf.CW_BOTTOM_ROI * h)

        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        return x1, y1, x2, y2

    def show(
        self,
        vision_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Displays the drivable-area debug view and returns the computed values.

        Returns a dictionary with:
          - error
          - center_x
          - image_center_x
          - left_x
          - right_x
          - roi
        """
        empty_result = {
            "error": None,
            "center_x": None,
            "image_center_x": None,
            "left_x": None,
            "right_x": None,
            "roi": None,
        }

        if vision_result is None:
            if getattr(conf, "SHOW_OPENCV_WINDOW", False):
                cv2.imshow(self.window_name, blank_frame())
                cv2.waitKey(1)
            return empty_result

        try:
            debug = vision_result.get("debug", {})
            drivable_mask = debug.get("drivable_mask", None)

            if drivable_mask is None:
                if getattr(conf, "SHOW_OPENCV_WINDOW", False):
                    cv2.imshow(self.window_name, blank_frame())
                    cv2.waitKey(1)
                return empty_result

            if len(drivable_mask.shape) == 3:
                gray = cv2.cvtColor(drivable_mask, cv2.COLOR_BGR2GRAY)
            else:
                gray = drivable_mask.copy()

            gray = gray.astype(np.float32)
            max_val = float(gray.max()) if gray.size > 0 else 0.0

            if max_val <= 1.0:
                display_mask = (gray * 255.0).astype(np.uint8)
            else:
                display_mask = np.clip(gray, 0, 255).astype(np.uint8)

            _, binary = cv2.threshold(display_mask, 1, 255, cv2.THRESH_BINARY)

            h, w = binary.shape
            image_center_x = w // 2

            x1, y1, x2, y2 = self.apply_center_roi(binary)
            roi_binary = binary[y1:y2, x1:x2]
            _, xs = np.where(roi_binary > 0)

            if len(xs) == 0:
                center_x = image_center_x
                left_x = None
                right_x = None
            else:
                left_x = x1 + int(xs.min())
                right_x = x1 + int(xs.max())
                center_x = int((left_x + right_x) / 2)

            error = image_center_x - center_x

            vis = cv2.cvtColor(display_mask, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.line(vis, (center_x, 0), (center_x, h - 1), (0, 0, 255), 2)
            cv2.line(vis, (image_center_x, 0), (image_center_x, h - 1), (255, 0, 0), 2)

            scan_y = y2 - 1
            if scan_y < 0:
                scan_y = h // 2

            cv2.line(vis, (x1, scan_y), (x2, scan_y), (0, 255, 255), 1)

            if left_x is not None:
                cv2.circle(vis, (left_x, scan_y), 6, (0, 255, 0), -1)
            if right_x is not None:
                cv2.circle(vis, (right_x, scan_y), 6, (0, 255, 0), -1)

            cv2.putText(vis, f"max={max_val:.2f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis, f"center_x={center_x}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis, f"img_center={image_center_x}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis, f"error={error}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # cv2.imshow(self.window_name, vis)
            # cv2.waitKey(1)

            return {
                "error": error,
                "center_x": center_x,
                "image_center_x": image_center_x,
                "left_x": left_x,
                "right_x": right_x,
                "roi": (x1, y1, x2, y2),
            }

        except Exception:
            # cv2.imshow(self.window_name, blank_frame())
            # cv2.waitKey(1)
            return empty_result

    def get_error(self, vision_result: Optional[Dict[str, Any]]) -> Optional[int]:
        """
        Runs the debugger and returns only the computed error.
        """
        result = self.show(vision_result)
        return result["error"]
