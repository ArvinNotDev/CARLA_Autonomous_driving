import os

import cv2
import numpy as np
import onnxruntime as ort

import config_city as conf
from utils import utils_onnx
from vision.color_extractor import HSVColorThresholdExtractor


class VisionProcessor:
    def __init__(
        self,
        mode: str = "onnx",
        color_extractor: HSVColorThresholdExtractor | None = None,
    ):
        self.mode = mode.lower().strip()
        if self.mode not in {"segmentation", "onnx"}:
            raise ValueError("mode must be either 'segmentation' or 'onnx'")

        self.debug = bool(getattr(conf, "VISION_DEBUG", False))

        self.extractor = color_extractor or HSVColorThresholdExtractor(
            morph_kernel_size=getattr(conf, "MORPH_KERNEL_SIZE", 1)
        )

        self.last_lane_center = None
        self.lane_center_alpha = getattr(conf, "LANE_CENTER_SMOOTH_ALPHA", 0.35)
        self.fallback_lane_offset_ratio = getattr(conf, "FALLBACK_LANE_OFFSET_RATIO", 0.18)
        self.min_side_pixels = getattr(conf, "MIN_SIDE_PIXELS", 80)

        self.active_side = None

        self.session = None
        self.input_name = None
        self.input_width = getattr(conf, "INPUT_WIDTH", 416)
        self.input_height = getattr(conf, "INPUT_HEIGHT", 416)

        if self.mode == "onnx":
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

    def _line_angle_from_roi(self, full_mask, top, bottom, left, right):
        """Return the absolute line angle from vertical in degrees."""
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

        # Fit x as a function of y because lane lines are predominantly
        # vertical in the camera image. The angle is measured from vertical.
        ys = ys.astype(np.float32) + float(top)
        xs = xs.astype(np.float32) + float(left)
        if np.ptp(ys) < 1.0:
            return None

        slope, _ = np.polyfit(ys, xs, 1)
        return float(np.degrees(np.arctan(slope)))

    def _draw_debug(self, frame, line_mask, drivable_mask, left_x, right_x, lane_center, lane_type):
        # OpenCV convention throughout the vision/control path is BGR.
        vis = np.ascontiguousarray(frame[:, :, :3]).copy()
        h, w = vis.shape[:2]

        if drivable_mask is not None and getattr(conf, "DEBUG_SHOW_DRIVABLE_AREA", True):
            drive = np.asarray(drivable_mask)
            if drive.ndim == 3:
                drive = cv2.cvtColor(drive, cv2.COLOR_BGR2GRAY)
            drive = np.where(drive > 0, 255, 0).astype(np.uint8)
            if drive.shape[:2] != (h, w):
                drive = cv2.resize(drive, (w, h), interpolation=cv2.INTER_NEAREST)
            overlay = vis.copy()
            overlay[drive > 0] = (50, 170, 50)
            vis = cv2.addWeighted(vis, 0.75, overlay, 0.25, 0)

        if line_mask is not None and getattr(conf, "DEBUG_SHOW_LANE_MASK", False):
            lane = np.asarray(line_mask)
            if lane.ndim == 3:
                lane = cv2.cvtColor(lane, cv2.COLOR_BGR2GRAY)
            if lane.shape[:2] != (h, w):
                lane = cv2.resize(lane, (w, h), interpolation=cv2.INTER_NEAREST)
            overlay = vis.copy()
            overlay[lane > 0] = (0, 255, 255)
            vis = cv2.addWeighted(vis, 0.82, overlay, 0.18, 0)

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

        if getattr(conf, "DEBUG_SHOW_ROIS", True):
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
        drivable = utils_onnx.driving_area_mask(outputs[int(getattr(conf, "DRIVABLE_OUTPUT_INDEX", 4))])

        lane_mask_small = self._predict_mask(lane_prob, getattr(conf, "LANE_PROB_THRESHOLD", 0.50))

        drivable_arr = np.asarray(drivable)
        if drivable_arr.ndim > 2:
            drivable_arr = np.squeeze(drivable_arr)
        if drivable_arr.dtype != np.uint8 or (drivable_arr.size and float(drivable_arr.max()) <= 1.0):
            drivable_arr = (np.asarray(drivable_arr, dtype=np.float32) >= float(getattr(conf, "DRIVABLE_PROB_THRESHOLD", 0.50))).astype(np.uint8) * 255
        else:
            drivable_arr = (drivable_arr > 0).astype(np.uint8) * 255

        return drivable_arr, lane_mask_small

    def _extract_masks_from_segmentation(self, semantic_frame):
        masks = self.extractor.extract(semantic_frame)
        green_mask = masks["green"]
        dark_purple_mask = masks["dark_purple"]
        line_mask = green_mask
        return line_mask, green_mask, dark_purple_mask

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
                return "only_right", right_x - lane_offset_px - 0

            if left_x is not None:
                self.active_side = "left"
                return "only_left", left_x + lane_offset_px + 0

            return lane_type, self.last_lane_center if self.last_lane_center is not None else frame_center

        if self.active_side == "left":
            if left_x is not None:
                return "only_left", left_x + lane_offset_px + 0

            if right_x is not None:
                self.active_side = "right"
                return "only_right", right_x - lane_offset_px - 0

            return lane_type, self.last_lane_center if self.last_lane_center is not None else frame_center

        # No active side yet: prefer right first, then left
        if right_x is not None:
            self.active_side = "right"
            return "only_right", right_x - lane_offset_px - 0

        if left_x is not None:
            self.active_side = "left"
            return "only_left", left_x + lane_offset_px + 0

        return lane_type, self.last_lane_center if self.last_lane_center is not None else frame_center

    def detect(self, semantic_frame, rgb_frame=None):
        """
        mode == 'segmentation':
            - semantic_frame is used for mask extraction
            - rgb_frame is used for debug display

        mode == 'onnx':
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

        if self.mode == "segmentation":
            if semantic_frame is None or semantic_frame.size == 0:
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
        else:
            if rgb_frame is None or rgb_frame.size == 0:
                rgb_frame = base_frame

        height, width = base_frame.shape[:2]
        frame_center = width / 2.0
        lane_offset_px = width * self.fallback_lane_offset_ratio

        if self.mode == "onnx":
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
            green_mask = None
            dark_purple_mask = None
        else:
            line_mask, green_mask, dark_purple_mask = self._extract_masks_from_segmentation(semantic_frame)
            drivable_mask = None

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
        left_line_angle_deg = self._line_angle_from_roi(
            line_mask, ll_top, ll_bottom, ll_left, ll_right
        )
        right_line_angle_deg = self._line_angle_from_roi(
            line_mask, rl_top, rl_bottom, rl_left, rl_right
        )

        lane_type, lane_center = self._pick_sticky_lane(
            left_x=left_x,
            right_x=right_x,
            frame_center=frame_center,
            lane_offset_px=lane_offset_px,
        )

        smoothed_lane_center = self._smooth_lane_center(lane_center)
        error = smoothed_lane_center - frame_center

        debug_requested = any((
            bool(getattr(conf, "VISION_DEBUG", False)),
            bool(getattr(conf, "DEBUG_SHOW_ROIS", True)),
            bool(getattr(conf, "DEBUG_SHOW_LANE_MASK", False)),
            bool(getattr(conf, "DEBUG_SHOW_DRIVABLE_AREA", True)),
        ))
        if debug_requested:
            combined = self._draw_debug(
                frame=np.ascontiguousarray(base_frame[:, :, :3]),
                line_mask=line_mask,
                drivable_mask=drivable_mask,
                left_x=left_x,
                right_x=right_x,
                lane_center=smoothed_lane_center,
                lane_type=lane_type,
            )
        else:
            combined = np.ascontiguousarray(base_frame[:, :, :3]).copy()

        return {
            "error": error,
            "lane_type": lane_type,
            "debug": {
                "combined": combined,
                "drivable_mask": drivable_mask,
                "lane_mask": line_mask,
                "green_mask": green_mask,
                "dark_purple_mask": dark_purple_mask,
                "left_x": left_x,
                "right_x": right_x,
                "lane_center": smoothed_lane_center,
                "left_line_angle_deg": left_line_angle_deg,
                "right_line_angle_deg": right_line_angle_deg,
            },
        }
