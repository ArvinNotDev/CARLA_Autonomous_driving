from __future__ import annotations

import math
import threading
import time
from typing import Any

import cv2
import numpy as np

from avis_lane_line_selection import TemporalLaneLineSelector


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

    Classical mode only estimates lane boundaries and the target lane centre.
    It deliberately does not build a drivable-area mask.
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

    def __init__(
        self,
        settings: dict[str, Any] | None = None,
        adaptive_roi_controller=None,
    ) -> None:
        self._lock = threading.RLock()
        self.adaptive_roi_controller = adaptive_roi_controller
        self._settings = dict(DEFAULT_CLASSICAL_SETTINGS)
        self._lane_line_selectors = {
            "ll": TemporalLaneLineSelector("ll"),
            "rl": TemporalLaneLineSelector("rl"),
        }
        if settings:
            self.update_settings(settings)
        self.last_lane_center: float | None = None
        self.last_lane_half_width: float | None = None
        self.active_side: str | None = None

    def update_settings(self, settings: dict[str, Any]) -> None:
        with self._lock:
            previous = self._settings.copy()
            for key, value in settings.items():
                if key in self._settings:
                    self._settings[key] = value
            selector_keys = (
                "hough_threshold",
                "hough_min_line_length_ratio",
                "hough_max_line_gap",
                "min_lane_angle_deg",
                "max_lane_angle_deg",
            )
            selection_changed = any(
                previous.get(key) != self._settings.get(key)
                for key in (*self.ROI_KEYS, *selector_keys)
            )
            if selection_changed:
                for selector in self._lane_line_selectors.values():
                    selector.reset()
            if any(previous.get(key) != self._settings.get(key) for key in self.ROI_KEYS):
                self.last_lane_center = None
                self.last_lane_half_width = None
                self.active_side = None
            if self.adaptive_roi_controller is not None:
                self.adaptive_roi_controller.configure(self._settings)

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
    ):
        height, width = frame.shape[:2]
        top, bottom, left, right = self._roi_bounds(
            (height, width), settings, prefix
        )
        if bottom <= top or right <= left:
            selection = self._lane_line_selectors[prefix].select(
                np.zeros((0, 0), dtype=np.uint8), settings
            )
            return top, bottom, selection

        roi = frame[top:bottom, left:right]
        binary = self._threshold_roi(roi, settings)
        threshold_mask[top:bottom, left:right] = binary
        selection = self._lane_line_selectors[prefix].select(
            binary,
            settings,
            x_offset=float(left),
        )
        if selection.mask.size:
            line_mask[top:bottom, left:right] = selection.mask
        return top, bottom, selection

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
        # When both lane boundaries are visible, always target their true
        # midpoint. The sticky-side fallback must never override this case.
        if left_x is not None and right_x is not None and right_x > left_x:
            self.last_lane_half_width = (right_x - left_x) / 2.0
            self.active_side = None
            return "both", left_x + self.last_lane_half_width

        # If one boundary temporarily disappears, reuse half of the most
        # recently measured lane width instead of a fixed image-width offset.
        # The configured ratio is only the cold-start fallback before a full
        # lane width has ever been observed.
        offset_px = (
            self.last_lane_half_width
            if self.last_lane_half_width is not None
            else lane_offset_px
        )

        if self.active_side == "right":
            if right_x is not None:
                return "only_right", right_x - offset_px
            if left_x is not None:
                self.active_side = "left"
                return "only_left", left_x + offset_px
            return (
                "none",
                self.last_lane_center
                if self.last_lane_center is not None
                else frame_center,
            )

        if self.active_side == "left":
            if left_x is not None:
                return "only_left", left_x + offset_px
            if right_x is not None:
                self.active_side = "right"
                return "only_right", right_x - offset_px
            return (
                "none",
                self.last_lane_center
                if self.last_lane_center is not None
                else frame_center,
            )

        if right_x is not None:
            self.active_side = "right"
            return "only_right", right_x - offset_px
        if left_x is not None:
            self.active_side = "left"
            return "only_left", left_x + offset_px
        return (
            "none",
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

    def detect_roi_fallbacks(
        self,
        frame: np.ndarray,
        sides: list[str] | tuple[str, ...],
        *,
        roi_settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the standalone Hough pipeline only inside the requested ROIs."""
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("AVIS Hough fallback requires a non-empty BGR frame.")
        if frame.ndim != 3 or frame.shape[2] < 3 or frame.dtype != np.uint8:
            raise ValueError(
                f"Expected a uint8 BGR image with three channels; received {frame.shape!r}."
            )

        requested_sides = tuple(dict.fromkeys(
            str(side).strip().lower() for side in sides
        ))
        invalid_sides = sorted(set(requested_sides) - {"left", "right"})
        if invalid_sides:
            raise ValueError(
                f"Unsupported Hough fallback ROI side(s): {', '.join(invalid_sides)}."
            )

        frame = np.ascontiguousarray(frame[:, :, :3])
        height, width = frame.shape[:2]
        settings = self.settings_snapshot()
        if isinstance(roi_settings, dict):
            for key in self.ROI_KEYS:
                if key in roi_settings:
                    settings[key] = roi_settings[key]

        line_mask = np.zeros((height, width), dtype=np.uint8)
        threshold_mask = np.zeros((height, width), dtype=np.uint8)
        bounds_by_side: dict[str, tuple[int, int, int, int]] = {}
        selections: dict[str, Any] = {}
        minimum_pixels = max(1, int(settings["min_side_pixels"]))
        coordinates: dict[str, int | None] = {"left_x": None, "right_x": None}
        angles: dict[str, float | None] = {
            "left_line_angle_deg": None,
            "right_line_angle_deg": None,
        }

        for side in requested_sides:
            prefix = "ll" if side == "left" else "rl"
            top, bottom, left, right = self._roi_bounds(
                (height, width), settings, prefix
            )
            bounds_by_side[side] = (top, bottom, left, right)
            selected_top, selected_bottom, selection = self._hough_roi_mask(
                frame,
                prefix,
                settings,
                line_mask,
                threshold_mask,
            )
            selections[side] = selection
            if selected_bottom <= selected_top or right <= left:
                continue

            coordinates[f"{side}_x"] = self._boundary_from_roi(
                line_mask,
                selected_top,
                selected_bottom,
                left,
                right,
                minimum_pixels,
            )
            angles[f"{side}_line_angle_deg"] = self._line_angle_from_roi(
                line_mask,
                selected_top,
                selected_bottom,
                left,
                right,
                minimum_pixels,
            )

        return {
            "lane_mask": line_mask,
            "threshold_mask": threshold_mask,
            "roi_bounds": bounds_by_side,
            "selections": selections,
            **coordinates,
            **angles,
        }

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
        roi_settings = (
            self.adaptive_roi_controller.active_settings()
            if self.adaptive_roi_controller is not None
            else settings
        )
        ll_bounds = self._roi_bounds((height, width), roi_settings, "ll")
        rl_bounds = self._roi_bounds((height, width), roi_settings, "rl")

        line_mask = np.zeros((height, width), dtype=np.uint8)
        threshold_mask = np.zeros((height, width), dtype=np.uint8)
        _, _, ll_selection = self._hough_roi_mask(
            frame, "ll", roi_settings, line_mask, threshold_mask
        )
        _, _, rl_selection = self._hough_roi_mask(
            frame, "rl", roi_settings, line_mask, threshold_mask
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

        if self.adaptive_roi_controller is not None:
            self.adaptive_roi_controller.observe(left_x, right_x, width)

        frame_center = width / 2.0
        lane_offset_px = width * float(settings["lane_offset_ratio"])
        lane_type, lane_center = self._pick_sticky_lane(
            left_x,
            right_x,
            frame_center,
            lane_offset_px,
        )
        lane_valid = lane_type != "none"
        # Match VisionProcessor._smooth_lane_center exactly, including the
        # no-detection frame path where the previous centre is retained.
        alpha = min(1.0, max(0.0, float(settings["center_smoothing"])))
        lane_center = (
            lane_center
            if self.last_lane_center is None
            else alpha * lane_center
            + (1.0 - alpha) * self.last_lane_center
        )
        self.last_lane_center = lane_center
        error = float(lane_center - frame_center)

        # Hough mode is lane-boundary-only: do not infer a road polygon or
        # allocate a synthetic drivable-area mask.
        overlay = frame.copy()

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
                "drivable_mask": None,
                "vehicles_mask": None,
                "lane_mask": line_mask,
                "threshold_mask": threshold_mask,
                "left_x": left_x,
                "right_x": right_x,
                "lane_center": float(lane_center),
                "lane_half_width_px": (
                    float(self.last_lane_half_width)
                    if self.last_lane_half_width is not None
                    else None
                ),
                "left_line_angle_deg": left_angle,
                "right_line_angle_deg": right_angle,
                "left_hough_segments": int(ll_selection.segment_count),
                "right_hough_segments": int(rl_selection.segment_count),
                "left_hough_candidates": int(ll_selection.candidate_count),
                "right_hough_candidates": int(rl_selection.candidate_count),
                "left_selected_line_x": ll_selection.selected_x,
                "right_selected_line_x": rl_selection.selected_x,
                "left_selected_line_quality": ll_selection.quality,
                "right_selected_line_quality": rl_selection.quality,
                "left_selected_line_stable_frames": ll_selection.stable_frames,
                "right_selected_line_stable_frames": rl_selection.stable_frames,
                "left_selected_line_missed_frames": ll_selection.missed_frames,
                "right_selected_line_missed_frames": rl_selection.missed_frames,
                "left_pending_switch_frames": ll_selection.pending_switch_frames,
                "right_pending_switch_frames": rl_selection.pending_switch_frames,
                "left_selected_line_switched": ll_selection.switched,
                "right_selected_line_switched": rl_selection.switched,
                "left_selected_line_predicted": ll_selection.predicted,
                "right_selected_line_predicted": rl_selection.predicted,
                "lane_valid": lane_valid,
                "adaptive_roi": (
                    self.adaptive_roi_controller.snapshot()
                    if self.adaptive_roi_controller is not None
                    else None
                ),
            },
        }
