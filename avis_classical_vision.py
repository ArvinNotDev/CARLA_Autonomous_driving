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
    "hough_threshold": 22,
    "hough_min_line_length_ratio": 0.035,
    "hough_max_line_gap": 24,
    "roi_top_ratio": 0.54,
    "roi_bottom_ratio": 0.98,
    "roi_top_half_width_ratio": 0.20,
    "roi_bottom_half_width_ratio": 0.48,
    "white_l_min": 165,
    "white_s_max": 115,
    "yellow_enabled": True,
    "yellow_h_min": 12,
    "yellow_h_max": 39,
    "yellow_s_min": 65,
    "yellow_l_min": 75,
    "morphology_kernel": 3,
    "min_lane_angle_deg": 0.0,
    "max_lane_angle_deg": 78.0,
    "lane_width_ratio": 0.42,
    "center_smoothing": 0.35,
}


class AvisClassicalVisionProcessor:
    """AVIS-only lane detector using HLS thresholds, Canny and HoughLinesP.

    The returned drivable mask is an estimated corridor between detected or
    inferred lane boundaries, not a semantic road-surface classification.
    """

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        self._lock = threading.RLock()
        self._settings = dict(DEFAULT_CLASSICAL_SETTINGS)
        if settings:
            self.update_settings(settings)
        self._last_lane_center: float | None = None
        self._last_lane_width: float | None = None

    def update_settings(self, settings: dict[str, Any]) -> None:
        with self._lock:
            for key, value in settings.items():
                if key in DEFAULT_CLASSICAL_SETTINGS:
                    self._settings[key] = value

    def settings_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._settings)

    @staticmethod
    def _weighted_line(
        segments: list[tuple[float, float, float, float, float]],
        y_top: int,
        y_bottom: int,
    ) -> tuple[float, float] | None:
        if not segments:
            return None
        weights = np.asarray([segment[4] for segment in segments], dtype=np.float64)
        slopes = np.asarray(
            [
                (segment[2] - segment[0]) / max(1.0, segment[3] - segment[1])
                for segment in segments
            ],
            dtype=np.float64,
        )
        intercepts = np.asarray(
            [segment[0] - slope * segment[1] for segment, slope in zip(segments, slopes)],
            dtype=np.float64,
        )
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            return None
        slope = float(np.average(slopes, weights=weights))
        intercept = float(np.average(intercepts, weights=weights))
        return (
            slope * y_top + intercept,
            slope * y_bottom + intercept,
        )

    @staticmethod
    def _odd_kernel(value: int) -> int:
        kernel = max(1, min(31, int(value)))
        return kernel if kernel % 2 else kernel + 1

    def _extract_binary(self, frame: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_kernel = self._odd_kernel(settings["blur_kernel"])
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

        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        white_mask = cv2.inRange(
            hls,
            np.array([0, int(settings["white_l_min"]), 0], dtype=np.uint8),
            np.array([179, 255, int(settings["white_s_max"])], dtype=np.uint8),
        )
        color_mask = white_mask
        if settings["yellow_enabled"]:
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

        # Color thresholds preserve painted markings; Canny also catches
        # faded, shadowed, or low-saturation lane boundaries.
        binary = cv2.bitwise_or(edges, color_mask)
        morphology_size = max(1, min(9, int(settings["morphology_kernel"])))
        if morphology_size > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (morphology_size, morphology_size)
            )
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        return binary

    @staticmethod
    def _roi_mask(
        shape: tuple[int, int],
        settings: dict[str, Any],
    ) -> tuple[np.ndarray, int, int]:
        height, width = shape
        top_y = max(0, min(height - 2, int(height * settings["roi_top_ratio"])))
        bottom_y = max(top_y + 1, min(height - 1, int(height * settings["roi_bottom_ratio"])))
        center_x = width // 2
        top_half = max(2, int(width * settings["roi_top_half_width_ratio"]))
        bottom_half = max(top_half, int(width * settings["roi_bottom_half_width_ratio"]))
        polygon = np.array(
            [[
                (max(0, center_x - bottom_half), bottom_y),
                (max(0, center_x - top_half), top_y),
                (min(width - 1, center_x + top_half), top_y),
                (min(width - 1, center_x + bottom_half), bottom_y),
            ]],
            dtype=np.int32,
        )
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, polygon, 255)
        return mask, top_y, bottom_y

    @staticmethod
    def _collect_lane_segments(
        lines: np.ndarray | None,
        center_x: float,
        top_y: int,
        bottom_y: int,
        settings: dict[str, Any],
    ) -> tuple[
        list[tuple[float, float, float, float, float]],
        list[tuple[float, float, float, float, float]],
    ]:
        left_segments: list[tuple[float, float, float, float, float]] = []
        right_segments: list[tuple[float, float, float, float, float]] = []
        if lines is None:
            return left_segments, right_segments

        min_angle = float(settings["min_lane_angle_deg"])
        max_angle = float(settings["max_lane_angle_deg"])
        for raw_line in lines:
            x1, y1, x2, y2 = (int(value) for value in raw_line.reshape(-1)[:4])
            if y1 == y2:
                continue
            if y1 < y2:
                x_top, y_top, x_bottom, y_bottom = x1, y1, x2, y2
            else:
                x_top, y_top, x_bottom, y_bottom = x2, y2, x1, y1

            dy = float(y_bottom - y_top)
            dx = float(x_bottom - x_top)
            length = math.hypot(dx, dy)
            if length < 1.0:
                continue

            angle = abs(math.degrees(math.atan2(dx, dy)))
            if angle < min_angle or angle > max_angle:
                continue

            midpoint_x = (x_top + x_bottom) * 0.5
            segment = (
                float(x_top),
                float(y_top),
                float(x_bottom),
                float(y_bottom),
                float(length),
            )
            if midpoint_x < center_x and dx <= 0.04 * dy:
                left_segments.append(segment)
            elif midpoint_x > center_x and dx >= -0.04 * dy:
                right_segments.append(segment)

        return left_segments, right_segments

    def detect(self, semantic_frame: np.ndarray | None, rgb_frame: np.ndarray | None = None) -> dict[str, Any]:
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
        roi_mask, top_y, bottom_y = self._roi_mask((height, width), settings)

        threshold_mask = self._extract_binary(frame, settings)
        threshold_mask = cv2.bitwise_and(threshold_mask, roi_mask)
        min_line_length = max(
            8,
            int(width * float(settings["hough_min_line_length_ratio"])),
        )
        lines = cv2.HoughLinesP(
            threshold_mask,
            rho=1,
            theta=np.pi / 180.0,
            threshold=int(settings["hough_threshold"]),
            minLineLength=min_line_length,
            maxLineGap=int(settings["hough_max_line_gap"]),
        )
        center_x = width * 0.5
        left_segments, right_segments = self._collect_lane_segments(
            lines, center_x, top_y, bottom_y, settings
        )
        left_line = self._weighted_line(left_segments, top_y, bottom_y)
        right_line = self._weighted_line(right_segments, top_y, bottom_y)

        lane_width = self._last_lane_width
        fallback_width = width * float(settings["lane_width_ratio"])
        if left_line is not None and right_line is not None:
            measured_width = right_line[1] - left_line[1]
            if width * 0.15 <= measured_width <= width * 0.90:
                lane_width = (
                    measured_width
                    if lane_width is None
                    else 0.25 * measured_width + 0.75 * lane_width
                )
                self._last_lane_width = lane_width
        if lane_width is None:
            lane_width = fallback_width

        lane_valid = left_line is not None or right_line is not None
        if left_line is not None and right_line is None:
            right_line = (
                left_line[0] + lane_width,
                left_line[1] + lane_width,
            )
        elif right_line is not None and left_line is None:
            left_line = (
                right_line[0] - lane_width,
                right_line[1] - lane_width,
            )

        lane_mask = np.zeros((height, width), dtype=np.uint8)
        drivable_mask = np.zeros((height, width), dtype=np.uint8)
        overlay = frame.copy()
        left_bottom = right_bottom = None
        lane_center = self._last_lane_center if self._last_lane_center is not None else center_x

        if lane_valid and left_line is not None and right_line is not None:
            left_top_x, left_bottom_x = left_line
            right_top_x, right_bottom_x = right_line
            bottom_width = right_bottom_x - left_bottom_x
            top_width = right_top_x - left_top_x
            plausible_width = (
                width * 0.15 <= bottom_width <= width * 0.90
                and width * 0.08 <= top_width <= width * 0.90
            )
            lane_valid = bool(plausible_width)
            if lane_valid:
                lane_center_raw = (left_bottom_x + right_bottom_x) * 0.5
                smoothing = float(settings["center_smoothing"])
                lane_center = (
                    lane_center_raw
                    if self._last_lane_center is None
                    else smoothing * lane_center_raw
                    + (1.0 - smoothing) * self._last_lane_center
                )
                self._last_lane_center = lane_center

                left_top_pt = (int(round(left_top_x)), top_y)
                right_top_pt = (int(round(right_top_x)), top_y)
                left_bottom_pt = (int(round(left_bottom_x)), bottom_y)
                right_bottom_pt = (int(round(right_bottom_x)), bottom_y)
                corridor = np.array(
                    [[left_top_pt, right_top_pt, right_bottom_pt, left_bottom_pt]],
                    dtype=np.int32,
                )
                cv2.fillPoly(drivable_mask, corridor, 255)
                cv2.line(lane_mask, left_top_pt, left_bottom_pt, 255, 5, cv2.LINE_AA)
                cv2.line(lane_mask, right_top_pt, right_bottom_pt, 255, 5, cv2.LINE_AA)
                left_bottom = int(round(left_bottom_x))
                right_bottom = int(round(right_bottom_x))

                green_overlay = overlay.copy()
                green_overlay[drivable_mask > 0] = (40, 170, 80)
                overlay = cv2.addWeighted(overlay, 0.72, green_overlay, 0.28, 0)
                cv2.line(overlay, left_top_pt, left_bottom_pt, (255, 150, 0), 3, cv2.LINE_AA)
                cv2.line(overlay, right_top_pt, right_bottom_pt, (0, 220, 255), 3, cv2.LINE_AA)
                cv2.line(
                    overlay,
                    (int(round(lane_center)), bottom_y),
                    (int(round(center_x)), bottom_y),
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                lane_center = (
                    self._last_lane_center
                    if self._last_lane_center is not None
                    else center_x
                )

        error = float(lane_center - center_x) if lane_valid else 0.0
        cv2.polylines(
            overlay,
            [
                np.array(
                    [[
                        (max(0, int(center_x - width * settings["roi_bottom_half_width_ratio"])), bottom_y),
                        (max(0, int(center_x - width * settings["roi_top_half_width_ratio"])), top_y),
                        (min(width - 1, int(center_x + width * settings["roi_top_half_width_ratio"])), top_y),
                        (min(width - 1, int(center_x + width * settings["roi_bottom_half_width_ratio"])), bottom_y),
                    ]],
                    dtype=np.int32,
                )
            ],
            True,
            (180, 180, 0),
            1,
        )
        cv2.line(
            overlay,
            (int(center_x), 0),
            (int(center_x), height - 1),
            (0, 0, 220),
            1,
            cv2.LINE_AA,
        )
        if lane_valid:
            cv2.circle(
                overlay,
                (int(round(lane_center)), bottom_y),
                5,
                (255, 0, 255),
                -1,
                cv2.LINE_AA,
            )

        return {
            "error": error,
            "lane_type": (
                "both_sides"
                if left_segments and right_segments
                else "left_only"
                if left_segments
                else "right_only"
                if right_segments
                else "none"
            ),
            "lane_valid": lane_valid,
            "processing_ms": (time.perf_counter() - started) * 1000.0,
            "debug": {
                "combined": overlay,
                "drivable_mask": drivable_mask,
                "vehicles_mask": None,
                "lane_mask": lane_mask,
                "threshold_mask": threshold_mask,
                "left_x": left_bottom,
                "right_x": right_bottom,
                "lane_center": float(lane_center),
                "lane_valid": lane_valid,
                "hough_segments": int(0 if lines is None else len(lines)),
            },
        }
