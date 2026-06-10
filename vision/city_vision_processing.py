import os

import cv2
import numpy as np
import onnxruntime as ort

import config_city as conf
from utils import utils_onnx


class VisionProcessor:
    def __init__(self, mode: str = "onnx"):
        self.mode = "onnx"

        self.debug = True  # True = draw debug overlays, False = faster

        self.last_lane_center = None
        self.lane_center_alpha = getattr(conf, "LANE_CENTER_SMOOTH_ALPHA", 0.35)
        self.fallback_lane_offset_ratio = getattr(conf, "FALLBACK_LANE_OFFSET_RATIO", 0.18)
        self.min_side_pixels = getattr(conf, "MIN_SIDE_PIXELS", 80)

        self.active_side = None

        self.session = None
        self.input_name = None
        self.input_width = getattr(conf, "INPUT_WIDTH", 416)
        self.input_height = getattr(conf, "INPUT_HEIGHT", 416)

        model_path = conf.MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        print(self.session.get_providers())
        self.input_name = self.session.get_inputs()[0].name

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

        if line_mask is not None:
            overlay = vis.copy()
            overlay[line_mask > 0] = (0, 255, 0)
            vis = cv2.addWeighted(vis, 0.85, overlay, 0.15, 0)

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

        return vis
    
    def _smooth_lane_center(self, lane_center):
        if self.last_lane_center is None:
            smoothed_lane_center = lane_center
        else:
            smoothed_lane_center = (
                self.lane_center_alpha * lane_center
                + (1.0 - self.lane_center_alpha) * self.last_lane_center
            )
        self.last_lane_center = smoothed_lane_center
        return smoothed_lane_center

    def _preprocess_onnx(self, frame):
        img = cv2.resize(
            frame,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, 0)

    def _sigmoid(self, x):
        x = np.clip(np.asarray(x, dtype=np.float32), -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-x))

    def _extract_prob_map(self, outputs, output_index):
        if not outputs:
            raise RuntimeError("Model returned no outputs")

        if output_index < 0 or output_index >= len(outputs):
            raise IndexError(f"Invalid output index: {output_index}")

        out = np.squeeze(np.asarray(outputs[output_index])).astype(np.float32)
        if out.ndim == 3:
            out = out[0]

        if out.min() < 0.0 or out.max() > 1.0:
            out = self._sigmoid(out)

        return out

    def _predict_mask(self, prob_map, threshold):
        mask = (prob_map > threshold).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask
    
    def _extract_masks_from_onnx(self, rgb_frame):
        outputs = self.session.run(None, {self.input_name: self._preprocess_onnx(rgb_frame)})

        # YOLOPv2 standard indices:
        # 4 = driving area
        # 5 = lane line
        lane_prob = self._extract_prob_map(outputs, 5)
        drivable = utils_onnx.driving_area_mask(outputs[4])
        
        lane_mask_small = self._predict_mask(lane_prob, getattr(conf, "LANE_PROB_THRESHOLD", 0.50), )

        return drivable, lane_mask_small

    def _pick_sticky_lane(self, left_x, right_x, frame_center, lane_offset_px):
        """
        Sticky-side logic:
        - First visible side becomes active.
        - Keep using that side as long as it exists.
        - If it disappears, switch to the other side if available.
        """
        lane_type = "none"

        if self.active_side == "right":
            if right_x is not None:
                return "only_right", right_x - lane_offset_px - 25

            if left_x is not None:
                self.active_side = "left"
                return "only_left", left_x + lane_offset_px + 25

            return lane_type, self.last_lane_center if self.last_lane_center is not None else frame_center

        if self.active_side == "left":
            if left_x is not None:
                return "only_left", left_x + lane_offset_px + 25

            if right_x is not None:
                self.active_side = "right"
                return "only_right", right_x - lane_offset_px - 25

            return lane_type, self.last_lane_center if self.last_lane_center is not None else frame_center

        # No active side yet: prefer right first, then left
        if right_x is not None:
            self.active_side = "right"
            return "only_right", right_x - lane_offset_px - 25

        if left_x is not None:
            self.active_side = "left"
            return "only_left", left_x + lane_offset_px + 25

        return lane_type, self.last_lane_center if self.last_lane_center is not None else frame_center

    def detect(self, semantic_frame, rgb_frame=None):
        """
        ONNX-only processing.
        - rgb_frame is used for inference and debug display when available
        - semantic_frame is kept only as a fallback if rgb_frame is not provided
        """
        base_frame = rgb_frame if rgb_frame is not None else semantic_frame

        if base_frame is None or base_frame.size == 0:
            return {
                "error": 0.0,
                "lane_type": "none",
                "debug": {
                    "combined": None,
                    "drivable_mask": None,
                    "lane_mask": None,
                    "green_mask": None,
                    "dark_purple_mask": None,
                    "left_x": None,
                    "right_x": None,
                    "lane_center": None,
                },
            }

        if rgb_frame is None or rgb_frame.size == 0:
            rgb_frame = base_frame

        height, width = base_frame.shape[:2]
        frame_center = width / 2.0
        lane_offset_px = width * self.fallback_lane_offset_ratio

        drivable_mask_small, line_mask_small = self._extract_masks_from_onnx(rgb_frame)

        drivable_mask = cv2.resize(
            drivable_mask_small,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

        line_mask = cv2.resize(
            line_mask_small,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

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

        lane_type, lane_center = self._pick_sticky_lane(
            left_x=left_x,
            right_x=right_x,
            frame_center=frame_center,
            lane_offset_px=lane_offset_px,
        )

        smoothed_lane_center = self._smooth_lane_center(lane_center)
        error = smoothed_lane_center - frame_center

        if self.debug:
            combined = self._draw_debug(
                frame=base_frame,   # always RGB debug
                line_mask=line_mask,
                left_x=left_x,
                right_x=right_x,
                lane_center=smoothed_lane_center,
                lane_type=lane_type,
            )
        else:
            combined = base_frame

        try:
            conf.debug_frame_buffer = combined
        except Exception:
            pass

        return {
            "error": error,
            "lane_type": lane_type,
            "debug": {
                "combined": combined,
                "drivable_mask": drivable_mask,
                "lane_mask": line_mask,
                "green_mask": None,
                "dark_purple_mask": None,
                "left_x": left_x,
                "right_x": right_x,
                "lane_center": smoothed_lane_center,
            },
        }