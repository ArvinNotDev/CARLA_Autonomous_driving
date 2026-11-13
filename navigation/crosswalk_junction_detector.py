from __future__ import annotations

from collections import deque
from typing import Any

import cv2
import numpy as np

import config_city as conf


class CrosswalkRoiJunctionDetector:
    """Detect a long, near-horizontal image line inside the Flask CW ROI.

    The detector deliberately operates on the RGB camera frame, not on LiDAR
    or the lane-model output. A short temporal vote rejects one-frame texture
    noise while retaining a fast response to a real crosswalk marking.
    """

    def __init__(self) -> None:
        self._recent_hits: deque[bool] = deque(maxlen=3)

    @staticmethod
    def _roi_bounds(width: int, height: int) -> tuple[int, int, int, int]:
        left = float(getattr(conf, "CW_LEFT_ROI", 0.0))
        right = float(getattr(conf, "CW_RIGHT_ROI", 1.0))
        top = float(getattr(conf, "CW_TOP_ROI", 0.0))
        bottom = float(getattr(conf, "CW_BOTTOM_ROI", 1.0))

        x1 = max(0, min(width, int(round(left * width))))
        x2 = max(0, min(width, int(round(right * width))))
        y1 = max(0, min(height, int(round(top * height))))
        y2 = max(0, min(height, int(round(bottom * height))))
        return x1, y1, x2, y2

    def detect(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        """Return raw line evidence and a temporally confirmed detection."""
        if (
            not isinstance(frame_bgr, np.ndarray)
            or frame_bgr.ndim != 3
            or frame_bgr.shape[2] < 3
            or frame_bgr.size == 0
        ):
            self._recent_hits.append(False)
            return {"line_present": False, "detected": False, "line": None, "length_ratio": 0.0}

        height, width = frame_bgr.shape[:2]
        x1, y1, x2, y2 = self._roi_bounds(width, height)
        roi_width = x2 - x1
        roi_height = y2 - y1
        if roi_width < 24 or roi_height < 12:
            self._recent_hits.append(False)
            return {"line_present": False, "detected": False, "line": None, "length_ratio": 0.0}

        roi = frame_bgr[y1:y2, x1:x2, :3]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 50, 140, L2gradient=True)

        min_ratio = float(getattr(conf, "JUNCTION_ROI_MIN_LINE_LENGTH_RATIO", 0.35))
        min_ratio = max(0.10, min(0.95, min_ratio))
        max_angle = float(getattr(conf, "JUNCTION_ROI_MAX_LINE_ANGLE_DEG", 12.0))
        max_angle = max(1.0, min(30.0, max_angle))
        min_length = max(20, int(round(roi_width * min_ratio)))
        max_gap = max(5, int(round(roi_width * 0.06)))

        segments = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=max(12, int(round(roi_width * 0.08))),
            minLineLength=min_length,
            maxLineGap=max_gap,
        )

        best_line = None
        best_length = 0.0
        if segments is not None:
            for segment in segments.reshape(-1, 4):
                sx1, sy1, sx2, sy2 = (int(value) for value in segment)
                dx = float(sx2 - sx1)
                dy = float(sy2 - sy1)
                length = float(np.hypot(dx, dy))
                if length < min_length or abs(dx) < 1.0:
                    continue
                angle = abs(float(np.degrees(np.arctan2(dy, dx))))
                angle = min(angle, abs(180.0 - angle))
                if angle <= max_angle and length > best_length:
                    best_length = length
                    best_line = (
                        x1 + sx1,
                        y1 + sy1,
                        x1 + sx2,
                        y1 + sy2,
                    )

        line_present = best_line is not None
        self._recent_hits.append(line_present)
        confirm_frames = int(getattr(conf, "JUNCTION_ROI_CONFIRM_FRAMES", 2))
        confirm_frames = max(1, min(3, confirm_frames))
        detected = sum(self._recent_hits) >= confirm_frames

        return {
            "line_present": line_present,
            "detected": detected,
            "line": best_line,
            "length_ratio": min(1.0, best_length / max(1.0, float(roi_width))),
        }
