import cv2
import numpy as np

import config_city as conf
from vision.color_extractor import HSVColorThresholdExtractor


class VisionProcessor:
    def __init__(self, color_extractor: HSVColorThresholdExtractor | None = None):
        self.extractor = color_extractor or HSVColorThresholdExtractor(
            morph_kernel_size=getattr(conf, "MORPH_KERNEL_SIZE", 1)
        )

        # Smoothing / fallback
        self.last_lane_center = None
        self.lane_center_alpha = getattr(conf, "LANE_CENTER_SMOOTH_ALPHA", 0.35)
        self.fallback_lane_offset_ratio = getattr(conf, "FALLBACK_LANE_OFFSET_RATIO", 0.18)
        self.min_side_pixels = getattr(conf, "MIN_SIDE_PIXELS", 80)

        self.active_side = None

    def _boundary_from_roi(self, full_mask, top, bottom, left, right):
        h, w = full_mask.shape[:2]

        top = max(0, min(h, top))
        bottom = max(0, min(h, bottom))
        left = max(0, min(w, left))
        right = max(0, min(w, right))

        if bottom <= top or right <= left:
            return None

        roi = full_mask[top:bottom, left:right]
        ys, xs = np.where(roi > 0)

        if len(xs) < self.min_side_pixels:
            return None

        return left + int(np.median(xs))

    def _draw_debug(self, frame, line_mask, left_x, right_x, lane_center, lane_type):
        vis = frame.copy()
        overlay = vis.copy()

        overlay[line_mask > 0] = (0, 255, 0)
        vis = cv2.addWeighted(vis, 0.75, overlay, 0.25, 0)

        h, w = vis.shape[:2]
        frame_center = w / 2.0

        ll_top_roi = getattr(conf, "LL_TOP_ROI", 0.55)
        ll_bottom_roi = getattr(conf, "LL_BOTTOM_ROI", 0.98)
        ll_left_roi = getattr(conf, "LL_LEFT_ROI", 0.05)
        ll_right_roi = getattr(conf, "LL_RIGHT_ROI", 0.50)

        rl_top_roi = getattr(conf, "RL_TOP_ROI", 0.55)
        rl_bottom_roi = getattr(conf, "RL_BOTTOM_ROI", 0.98)
        rl_left_roi = getattr(conf, "RL_LEFT_ROI", 0.50)
        rl_right_roi = getattr(conf, "RL_RIGHT_ROI", 0.95)

        ll_top = int(ll_top_roi * h)
        ll_bottom = int(ll_bottom_roi * h)
        ll_left = int(ll_left_roi * w)
        ll_right = int(ll_right_roi * w)

        rl_top = int(rl_top_roi * h)
        rl_bottom = int(rl_bottom_roi * h)
        rl_left = int(rl_left_roi * w)
        rl_right = int(rl_right_roi * w)

        cv2.rectangle(vis, (ll_left, ll_top), (ll_right, ll_bottom), (0, 255, 0), 2)
        cv2.rectangle(vis, (rl_left, rl_top), (rl_right, rl_bottom), (0, 255, 0), 2)

        cv2.line(vis, (int(frame_center), 0), (int(frame_center), h), (0, 0, 255), 2)
        cv2.line(vis, (int(lane_center), 0), (int(lane_center), h), (255, 0, 255), 2)

        if left_x is not None:
            cv2.circle(vis, (int(left_x), int(h * 0.90)), 6, (255, 0, 0), -1)
        if right_x is not None:
            cv2.circle(vis, (int(right_x), int(h * 0.90)), 6, (0, 255, 255), -1)

        cv2.putText(
            vis,
            f"lane_type: {lane_type}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"error: {lane_center - frame_center:.1f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return vis

    def _pick_lane_side(self, left_x, right_x):
        """
        Sticky side logic:
        - If we're already on left, stay on left while left exists.
        - If we're already on right, stay on right while right exists.
        - If the active side disappears, switch to the other side if available.
        - If no side is active yet, prefer left first when both are visible.
        """

        offset = None
        lane_type = "none"

        if self.active_side == "left":
            if left_x is not None:
                lane_type = "only_left"
                offset = left_x + self._lane_offset_px()
                return lane_type, offset

            if right_x is not None:
                self.active_side = "right"
                lane_type = "only_right"
                offset = right_x - self._lane_offset_px()
                return lane_type, offset

            return lane_type, offset

        if self.active_side == "right":
            if right_x is not None:
                lane_type = "only_right"
                offset = right_x - self._lane_offset_px()
                return lane_type, offset

            if left_x is not None:
                self.active_side = "left"
                lane_type = "only_left"
                offset = left_x + self._lane_offset_px()
                return lane_type, offset

            return lane_type, offset

        # No active side yet:
        # choose left first if both are available, so right does not have priority.
        if left_x is not None:
            self.active_side = "left"
            lane_type = "only_left"
            offset = left_x + self._lane_offset_px()
            return lane_type, offset

        if right_x is not None:
            self.active_side = "right"
            lane_type = "only_right"
            offset = right_x - self._lane_offset_px()
            return lane_type, offset

        return lane_type, offset

    def _lane_offset_px(self):
        return 0.0

    def detect(self, semantic_frame):
        if semantic_frame is None or semantic_frame.size == 0:
            return {
                "error": 0.0,
                "lane_type": "none",
                "debug": {
                    "combined": None,
                    "lane_mask": None,
                    "green_mask": None,
                    "dark_purple_mask": None,
                    "left_x": None,
                    "right_x": None,
                    "lane_center": None,
                },
            }

        height, width = semantic_frame.shape[:2]
        frame_center = width / 2.0
        lane_offset_px = width * self.fallback_lane_offset_ratio

        masks = self.extractor.extract(semantic_frame)
        green_mask = masks["green"]
        dark_purple_mask = masks["dark_purple"]
        line_mask = green_mask

        ll_top_roi = getattr(conf, "LL_TOP_ROI", 0.55)
        ll_bottom_roi = getattr(conf, "LL_BOTTOM_ROI", 0.98)
        ll_left_roi = getattr(conf, "LL_LEFT_ROI", 0.05)
        ll_right_roi = getattr(conf, "LL_RIGHT_ROI", 0.50)

        rl_top_roi = getattr(conf, "RL_TOP_ROI", 0.55)
        rl_bottom_roi = getattr(conf, "RL_BOTTOM_ROI", 0.98)
        rl_left_roi = getattr(conf, "RL_LEFT_ROI", 0.50)
        rl_right_roi = getattr(conf, "RL_RIGHT_ROI", 0.95)

        ll_top = int(ll_top_roi * height)
        ll_bottom = int(ll_bottom_roi * height)
        ll_left = int(ll_left_roi * width)
        ll_right = int(ll_right_roi * width)

        rl_top = int(rl_top_roi * height)
        rl_bottom = int(rl_bottom_roi * height)
        rl_left = int(rl_left_roi * width)
        rl_right = int(rl_right_roi * width)

        left_x = self._boundary_from_roi(line_mask, ll_top, ll_bottom, ll_left, ll_right)
        right_x = self._boundary_from_roi(line_mask, rl_top, rl_bottom, rl_left, rl_right)

        if self.active_side == "left":
            if left_x is not None:
                lane_type = "only_left"
                lane_center = left_x + lane_offset_px + 40
            elif right_x is not None:
                self.active_side = "right"
                lane_type = "only_right"
                lane_center = right_x - lane_offset_px
            else:
                lane_type = "none"
                lane_center = self.last_lane_center if self.last_lane_center is not None else frame_center

        elif self.active_side == "right":
            if right_x is not None:
                lane_type = "only_right"
                lane_center = right_x - lane_offset_px - 40
            elif left_x is not None:
                self.active_side = "left"
                lane_type = "only_left"
                lane_center = left_x + lane_offset_px
            else:
                lane_type = "none"
                lane_center = self.last_lane_center if self.last_lane_center is not None else frame_center

        else:
            # No side chosen yet: prefer left first, then right.
            if left_x is not None:
                self.active_side = "left"
                lane_type = "only_left"
                lane_center = left_x + lane_offset_px
            elif right_x is not None:
                self.active_side = "right"
                lane_type = "only_right"
                lane_center = right_x - lane_offset_px
            else:
                lane_type = "none"
                lane_center = self.last_lane_center if self.last_lane_center is not None else frame_center

        if self.last_lane_center is None:
            smoothed_lane_center = lane_center
        else:
            smoothed_lane_center = (
                self.lane_center_alpha * lane_center
                + (1.0 - self.lane_center_alpha) * self.last_lane_center
            )

        self.last_lane_center = smoothed_lane_center
        error = smoothed_lane_center - frame_center

        combined = self._draw_debug(
            frame=semantic_frame,
            line_mask=line_mask,
            left_x=left_x,
            right_x=right_x,
            lane_center=smoothed_lane_center,
            lane_type=lane_type,
        )

        try:
            conf.debug_frame_buffer = combined
        except Exception:
            pass

        return {
            "error": error,
            "lane_type": lane_type,
            "debug": {
                "combined": combined,
                "lane_mask": line_mask,
                "green_mask": green_mask,
                "dark_purple_mask": dark_purple_mask,
                "left_x": left_x,
                "right_x": right_x,
                "lane_center": smoothed_lane_center,
            },
        }