from __future__ import annotations

import math
import threading
import time
from typing import Any

import cv2
import numpy as np


DEFAULT_CLASSICAL_SETTINGS: dict[str, Any] = {
    "blur_kernel": 5,
    "canny_low": 55,
    "canny_high": 150,
    "hough_threshold": 12,
    "hough_min_line_length_ratio": 0.035,
    "hough_max_line_gap": 24,
    "ll_top_roi": 0.57,
    "ll_bottom_roi": 0.69,
    "ll_left_roi": 0.24,
    "ll_right_roi": 0.45,
    "rl_top_roi": 0.57,
    "rl_bottom_roi": 0.69,
    "rl_left_roi": 0.56,
    "rl_right_roi": 0.78,
    "white_l_min": 165,
    "white_s_max": 115,
    "yellow_enabled": True,
    "yellow_h_min": 12,
    "yellow_h_max": 39,
    "yellow_s_min": 65,
    "yellow_l_min": 75,
    "morphology_kernel": 3,
    "min_lane_angle_deg": 0.0,
    "max_lane_angle_deg": 80.0,
    "min_side_pixels": 12,
    "lane_offset_ratio": 0.16,
    "center_smoothing": 0.30,
}


class AvisClassicalVisionProcessor:
    """AVIS-only classical mask extraction with the project's exact ROI flow.

    HLS/Canny/Hough processing is performed independently inside LL and RL
    rectangles. The resulting binary lane mask then goes through the same
    median-boundary, angle, sticky-side, offset and smoothing procedure used
    by the project's model-backed VisionProcessor.

    The drivable mask is a visual corridor estimate; it is not semantic
    road-surface segmentation and is not used as a second steering signal.
    """

    ROI_KEYS = (
        "ll_top_roi",
        "ll_bottom_roi",
        "ll_left_roi",
        "ll_right_roi",
        "rl_top_roi",
        "rl_bottom_roi",
        "rl_left_roi",
        "rl_right_roi",
    )

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        self._lock = threading.RLock()
        self._settings = dict(DEFAULT_CLASSICAL_SETTINGS)
        if settings:
            self.update_settings(settings)
        self.last_lane_center: float | None = None
        self.active_side: str | None = None

    def update_settings(self, settings: dict[str, Any]) -> None:
        with self._lock:
            previous = self._settings.copy()
            for key, value in settings.items():
                if key in self._settings:
                    self._settings[key] = value
            if any(previous.get(key) != self._settings.get(key) for key in self.ROI_KEYS):
                self.last_lane_center = None
                self.active_side = None

    def settings_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._settings)

    @staticmethod
    def _odd_kernel(value: int, maximum: int = 31) -> int:
        kernel = max(1, min(maximum, int(value)))
        return kernel if kernel % 2 else min(maximum, kernel + 1)

    @staticmethod
    def _roi_bounds(
        frame_shape: tuple[int, int],
        settings: dict[str, Any],
        prefix: str,
    ) -> tuple[int, int, int, int]:
        height, width = frame_shape
        top = max(0, min(height, int(round(float(settings[f"{prefix}_top_roi"]) * height))))
        bottom = max(0, min(height, int(round(float(settings[f"{prefix}_bottom_roi"]) * height))))
        left = max(0, min(width, int(round(float(settings[f"{prefix}_left_roi"]) * width))))
        right = max(0, min(width, int(round(float(settings[f"{prefix}_right_roi"]) * width))))
        return top, bottom, left, right

    @staticmethod
    def _threshold_roi(
        roi_frame: np.ndarray,
        settings: dict[str, Any],
    ) -> np.ndarray:
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        blur_kernel = AvisClassicalVisionProcessor._odd_kernel(
            settings["blur_kernel"]
        )
        blurred = (
            cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
            if blur_kernel > 1
            else gray
        )
        edges = cv2.Canny(
            blurred,
            int(settings["canny_low"]),
            int(settings["canny_high"]),
            L2gradient=True,
        )

        hls = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HLS)
        white_mask = cv2.inRange(
            hls,
            np.array([0, int(settings["white_l_min"]), 0], dtype=np.uint8),
            np.array([179, 255, int(settings["white_s_max"])], dtype=np.uint8),
        )
        color_mask = white_mask
        if bool(settings["yellow_enabled"]):
            yellow_mask = cv2.inRange(
                hls,
                np.array(
                    [
                        int(settings["yellow_h_min"]),
                        int(settings["yellow_l_min"]),
                        int(settings["yellow_s_min"]),
                    ],
                    dtype=np.uint8,
                ),
                np.array(
                    [int(settings["yellow_h_max"]), 255, 255],
                    dtype=np.uint8,
                ),
            )
            color_mask = cv2.bitwise_or(color_mask, yellow_mask)

        binary = cv2.bitwise_or(edges, color_mask)
        morphology_size = max(1, min(9, int(settings["morphology_kernel"])))
        if morphology_size > 1:
            if morphology_size % 2 == 0:
                morphology_size = min(9, morphology_size + 1)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (morphology_size, morphology_size),
            )
            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=1,
            )
        return binary

    @staticmethod
    def _valid_segment(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        min_angle: float,
        max_angle: float,
    ) -> bool:
        dy = abs(y2 - y1)
        dx = abs(x2 - x1)
        if dy < 2:
            return False
        angle = abs(math.degrees(math.atan2(float(dx), float(dy))))
        return min_angle <= angle <= max_angle

    def _hough_roi_mask(
        self,
        frame: np.ndarray,
        prefix: str,
        settings: dict[str, Any],
        line_mask: np.ndarray,
        threshold_mask: np.ndarray,
    ) -> tuple[int | None, int | None, float | None]:
        height, width = frame.shape[:2]
        top, bottom, left, right = self._roi_bounds(
            (height, width), settings, prefix
        )
        if bottom <= top or right <= left:
            return None, None, None

        # This is intentionally a crop, not a post-Hough polygon mask:
        # blur, threshold, Canny and HoughLinesP never see pixels outside
        # the selected side's ROI.
        roi = frame[top:bottom, left:right]
        binary = self._threshold_roi(roi, settings)
        threshold_mask[top:bottom, left:right] = binary

        roi_height, roi_width = roi.shape[:2]
        min_line_length = max(
            5,
            int(round(roi_width * float(settings["hough_min_line_length_ratio"]))),
        )
        segments = cv2.HoughLinesP(
            binary,
            rho=1,
            theta=np.pi / 180.0,
            threshold=int(settings["hough_threshold"]),
            minLineLength=min_line_length,
            maxLineGap=int(settings["hough_max_line_gap"]),
        )

        min_angle = float(settings["min_lane_angle_deg"])
        max_angle = float(settings["max_lane_angle_deg"])
        valid_segments = 0
        if segments is not None:
            for raw_segment in segments:
                x1, y1, x2, y2 = (
                    int(value) for value in raw_segment.reshape(-1)[:4]
                )
                if not self._valid_segment(
                    x1, y1, x2, y2, min_angle, max_angle
                ):
                    continue

                start = (left + x1, top + y1)
                end = (left + x2, top + y2)
                cv2.line(
                    line_mask,
                    start,
                    end,
                    255,
                    thickness=3,
                    lineType=cv2.LINE_AA,
                )
                valid_segments += 1

        return top, bottom, float(valid_segments)

    def _boundary_from_roi(
        self,
        full_mask: np.ndarray,
        top: int,
        bottom: int,
        left: int,
        right: int,
        min_side_pixels: int,
    ) -> int | None:
        height, width = full_mask.shape[:2]
        top = max(0, min(height, top))
        bottom = max(0, min(height, bottom))
        left = max(0, min(width, left))
        right = max(0, min(width, right))
        if bottom <= top or right <= left:
            return None

        roi = full_mask[top:bottom, left:right]
        ys, xs = np.where(roi > 0)
        if len(xs) < min_side_pixels:
            return None
        return left + int(np.median(xs))

    def _line_angle_from_roi(
        self,
        full_mask: np.ndarray,
        top: int,
        bottom: int,
        left: int,
        right: int,
        min_side_pixels: int,
    ) -> float | None:
        height, width = full_mask.shape[:2]
        top = max(0, min(height, top))
        bottom = max(0, min(height, bottom))
        left = max(0, min(width, left))
        right = max(0, min(width, right))
        if bottom <= top or right <= left:
            return None

        roi = full_mask[top:bottom, left:right]
        ys, xs = np.where(roi > 0)
        if len(xs) < min_side_pixels:
            return None

        ys = ys.astype(np.float32) + float(top)
        xs = xs.astype(np.float32) + float(left)
        if np.ptp(ys) < 1.0:
            return None

        slope, _ = np.polyfit(ys, xs, 1)
        return float(np.degrees(np.arctan(slope)))

    def _pick_sticky_lane(
        self,
        left_x: int | None,
        right_x: int | None,
        frame_center: float,
        lane_offset_px: float,
    ) -> tuple[str, float]:
        lane_type = "none"

        if self.active_side == "right":
            if right_x is not None:
                return "only_right", right_x - lane_offset_px
            if left_x is not None:
                self.active_side = "left"
                return "only_left", left_x + lane_offset_px
            return (
                lane_type,
                self.last_lane_center
                if self.last_lane_center is not None
                else frame_center,
            )

        if self.active_side == "left":
            if left_x is not None:
                return "only_left", left_x + lane_offset_px
            if right_x is not None:
                self.active_side = "right"
                return "only_right", right_x - lane_offset_px
            return (
                lane_type,
                self.last_lane_center
                if self.last_lane_center is not None
                else frame_center,
            )

        if right_x is not None:
            self.active_side = "right"
            return "only_right", right_x - lane_offset_px
        if left_x is not None:
            self.active_side = "left"
            return "only_left", left_x + lane_offset_px
        return (
            lane_type,
            self.last_lane_center
            if self.last_lane_center is not None
            else frame_center,
        )

    @staticmethod
    def _draw_roi(
        overlay: np.ndarray,
        bounds: tuple[int, int, int, int],
        label: str,
        color: tuple[int, int, int],
    ) -> None:
        top, bottom, left, right = bounds
        if bottom <= top or right <= left:
            return
        cv2.rectangle(
            overlay,
            (left, top),
            (right - 1, bottom - 1),
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            label,
            (left + 3, max(14, top - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )

    def _estimated_drivable_mask(
        self,
        frame_shape: tuple[int, int],
        settings: dict[str, Any],
        lane_center: float,
        lane_valid: bool,
    ) -> np.ndarray:
        height, width = frame_shape
        mask = np.zeros((height, width), dtype=np.uint8)
        if not lane_valid:
            return mask

        ll_top, ll_bottom, _, _ = self._roi_bounds(
            frame_shape, settings, "ll"
        )
        rl_top, rl_bottom, _, _ = self._roi_bounds(
            frame_shape, settings, "rl"
        )
        top = max(0, min(ll_top, rl_top))
        bottom = min(height, max(ll_bottom, rl_bottom))
        if bottom <= top:
            return mask

        half_width = max(2.0, width * float(settings["lane_offset_ratio"]))
        middle_y = (top + bottom) * 0.5
        points = np.array(
            [[
                (int(round(lane_center - half_width * 0.72)), top),
                (int(round(lane_center + half_width * 0.72)), top),
                (int(round(lane_center + half_width)), bottom - 1),
                (int(round(lane_center - half_width)), bottom - 1),
            ]],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, points, 255)
        return mask

    def detect(
        self,
        semantic_frame: np.ndarray | None,
        rgb_frame: np.ndarray | None = None,
    ) -> dict[str, Any]:
        frame = rgb_frame if rgb_frame is not None else semantic_frame
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("AVIS classical vision requires a non-empty BGR frame.")
        if frame.ndim != 3 or frame.shape[2] < 3 or frame.dtype != np.uint8:
            raise ValueError(
                f"Expected a uint8 BGR image with three channels; received {frame.shape!r}."
            )

        started = time.perf_counter()
        frame = np.ascontiguousarray(frame[:, :, :3])
        height, width = frame.shape[:2]
        settings = self.settings_snapshot()
        ll_bounds = self._roi_bounds((height, width), settings, "ll")
        rl_bounds = self._roi_bounds((height, width), settings, "rl")

        line_mask = np.zeros((height, width), dtype=np.uint8)
        threshold_mask = np.zeros((height, width), dtype=np.uint8)
        _, _, ll_segments = self._hough_roi_mask(
            frame, "ll", settings, line_mask, threshold_mask
        )
        _, _, rl_segments = self._hough_roi_mask(
            frame, "rl", settings, line_mask, threshold_mask
        )

        min_side_pixels = max(1, int(settings["min_side_pixels"]))
        ll_top, ll_bottom, ll_left, ll_right = ll_bounds
        rl_top, rl_bottom, rl_left, rl_right = rl_bounds
        left_x = self._boundary_from_roi(
            line_mask,
            ll_top,
            ll_bottom,
            ll_left,
            ll_right,
            min_side_pixels,
        )
        right_x = self._boundary_from_roi(
            line_mask,
            rl_top,
            rl_bottom,
            rl_left,
            rl_right,
            min_side_pixels,
        )
        left_angle = self._line_angle_from_roi(
            line_mask,
            ll_top,
            ll_bottom,
            ll_left,
            ll_right,
            min_side_pixels,
        )
        right_angle = self._line_angle_from_roi(
            line_mask,
            rl_top,
            rl_bottom,
            rl_left,
            rl_right,
            min_side_pixels,
        )

        frame_center = width / 2.0
        lane_offset_px = width * float(settings["lane_offset_ratio"])
        lane_type, lane_center = self._pick_sticky_lane(
            left_x,
            right_x,
            frame_center,
            lane_offset_px,
        )
        lane_valid = lane_type != "none"
        if lane_valid:
            alpha = min(1.0, max(0.0, float(settings["center_smoothing"])))
            lane_center = (
                lane_center
                if self.last_lane_center is None
                else alpha * lane_center
                + (1.0 - alpha) * self.last_lane_center
            )
            self.last_lane_center = lane_center
        error = float(lane_center - frame_center) if lane_valid else 0.0

        drivable_mask = self._estimated_drivable_mask(
            (height, width),
            settings,
            lane_center,
            lane_valid,
        )
        overlay = frame.copy()
        if lane_valid and np.any(drivable_mask):
            green = overlay.copy()
            green[drivable_mask > 0] = (45, 150, 75)
            overlay = cv2.addWeighted(overlay, 0.78, green, 0.22, 0)

        if np.any(line_mask):
            line_overlay = overlay.copy()
            line_overlay[line_mask > 0] = (0, 235, 255)
            overlay = cv2.addWeighted(overlay, 0.62, line_overlay, 0.38, 0)

        self._draw_roi(overlay, ll_bounds, "LL ROI", (70, 220, 145))
        self._draw_roi(overlay, rl_bounds, "RL ROI", (40, 185, 255))
        cv2.line(
            overlay,
            (int(round(frame_center)), 0),
            (int(round(frame_center)), height - 1),
            (40, 40, 230),
            1,
            cv2.LINE_AA,
        )
        if lane_valid:
            cv2.line(
                overlay,
                (int(round(lane_center)), 0),
                (int(round(lane_center)), height - 1),
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return {
            "error": error,
            "lane_type": lane_type,
            "lane_valid": lane_valid,
            "processing_ms": (time.perf_counter() - started) * 1000.0,
            "debug": {
                "combined": overlay,
                "drivable_mask": drivable_mask,
                "vehicles_mask": None,
                "lane_mask": line_mask,
                "threshold_mask": threshold_mask,
                "left_x": left_x,
                "right_x": right_x,
                "lane_center": float(lane_center),
                "left_line_angle_deg": left_angle,
                "right_line_angle_deg": right_angle,
                "left_hough_segments": int(ll_segments or 0),
                "right_hough_segments": int(rl_segments or 0),
                "lane_valid": lane_valid,
            },
        }
