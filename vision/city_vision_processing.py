import os

import cv2
import numpy as np
import onnxruntime as ort

import config_city as conf
from vision.color_extractor import HSVColorThresholdExtractor


class VisionProcessor:
    def __init__(
        self,
        mode: str = "onnx",  # "segmentation" or "onnx"
        color_extractor: HSVColorThresholdExtractor | None = None,
    ):
        self.mode = mode.lower().strip()
        if self.mode not in {"segmentation", "onnx"}:
            raise ValueError("mode must be either 'segmentation' or 'onnx'")

        self.extractor = color_extractor or HSVColorThresholdExtractor(
            morph_kernel_size=getattr(conf, "MORPH_KERNEL_SIZE", 1)
        )

        # Smoothing / fallback
        self.last_lane_center = None
        self.lane_center_alpha = getattr(conf, "LANE_CENTER_SMOOTH_ALPHA", 0.35)
        self.fallback_lane_offset_ratio = getattr(conf, "FALLBACK_LANE_OFFSET_RATIO", 0.18)
        self.min_side_pixels = getattr(conf, "MIN_SIDE_PIXELS", 80)
        self.active_side = None

        # ONNX setup
        self.session = None
        self.input_name = None
        self.input_width = getattr(conf, "INPUT_WIDTH", 416)
        self.input_height = getattr(conf, "INPUT_HEIGHT", 416)

        if self.mode == "onnx":
            model_path = conf.MODEL_PATH
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"ONNX model not found: {model_path}")

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            opts.inter_op_num_threads = 1

            self.session = ort.InferenceSession(
                model_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
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

    def _lane_offset_px(self):
        return 0.0

    def _draw_debug(self, frame, line_mask, left_x, right_x, lane_center, lane_type):
        vis = frame.copy()
        overlay = vis.copy()

        if line_mask is not None:
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
            f"mode: {self.mode}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"lane_type: {lane_type}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"error: {lane_center - frame_center:.1f}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

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

    # -------------------------------------------------------------------------
    # Segmentation mode
    # -------------------------------------------------------------------------

    def _extract_masks_from_segmentation(self, semantic_frame):
        masks = self.extractor.extract(semantic_frame)
        green_mask = masks["green"]
        dark_purple_mask = masks["dark_purple"]
        line_mask = green_mask
        return line_mask, green_mask, dark_purple_mask

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

        # No active side yet: choose left first if both are available
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

    # -------------------------------------------------------------------------
    # ONNX mode
    # -------------------------------------------------------------------------

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

    def _extract_prob_map(self, outputs):
        if not outputs:
            raise RuntimeError("Model returned no outputs")

        idx = getattr(conf, "DRIVABLE_OUTPUT_INDEX", -1)
        if idx < 0 or idx >= len(outputs):
            idx = len(outputs) - 1

        out = np.squeeze(np.asarray(outputs[idx])).astype(np.float32)
        if out.ndim == 3:
            out = out[0]

        if out.min() < 0.0 or out.max() > 1.0:
            out = self._sigmoid(out)
        return out

    def _predict_lane_mask(self, outputs):
        lane_prob = self._extract_prob_map(outputs)

        for threshold in (
            getattr(conf, "LANE_PROB_THRESHOLD", 0.50),
            getattr(conf, "LANE_PROB_THRESHOLD_FALLBACK", 0.35),
            max(
                getattr(conf, "LANE_PROB_THRESHOLD_MIN", 0.20),
                float(np.percentile(lane_prob, 80)),
            ),
        ):
            lane_mask = (lane_prob > threshold).astype(np.uint8) * 255
            if cv2.countNonZero(lane_mask) >= 200:
                break

        kernel = np.ones((3, 3), np.uint8)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return lane_mask

    def _extract_masks_from_onnx(self, rgb_frame):
        outputs = self.session.run(None, {self.input_name: self._preprocess_onnx(rgb_frame)})
        lane_mask_small = self._predict_lane_mask(outputs)
        return lane_mask_small, None, None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def detect(self, semantic_frame, rgb_frame=None):
        """
        mode == 'segmentation':
            - semantic_frame is used for mask extraction
            - rgb_frame is used only for debug display if provided

        mode == 'onnx':
            - rgb_frame is used for ONNX inference and debug display
            - semantic_frame is ignored
        """
        if self.mode == "onnx":
            if rgb_frame is None or rgb_frame.size == 0:
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
            base_frame = rgb_frame
        else:
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
            base_frame = rgb_frame if rgb_frame is not None else semantic_frame

        if base_frame is None or base_frame.size == 0:
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

        height, width = base_frame.shape[:2]
        frame_center = width / 2.0
        lane_offset_px = width * self.fallback_lane_offset_ratio

        # ---------------------------------------------------------------------
        # Extract lane mask depending on mode
        # ---------------------------------------------------------------------
        if self.mode == "onnx":
            line_mask_small, green_mask, dark_purple_mask = self._extract_masks_from_onnx(rgb_frame)
            line_mask = cv2.resize(
                line_mask_small,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            line_mask, green_mask, dark_purple_mask = self._extract_masks_from_segmentation(semantic_frame)

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

        # ---------------------------------------------------------------------
        # Lane center logic
        # - segmentation: sticky side logic
        # - onnx: simple left/right/both logic
        # ---------------------------------------------------------------------
        if self.mode == "segmentation":
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

        else:
            # ONNX mode doesn't have no sticky side
            if left_x is not None and right_x is not None:
                lane_type = "both"
                lane_center = (left_x + right_x) / 2.0
            elif right_x is not None:
                lane_type = "only_right"
                lane_center = right_x - lane_offset_px - 40
            elif left_x is not None:
                lane_type = "only_left"
                lane_center = left_x + lane_offset_px + 40
            else:
                lane_type = "none"
                lane_center = self.last_lane_center if self.last_lane_center is not None else frame_center

        smoothed_lane_center = self._smooth_lane_center(lane_center)
        error = smoothed_lane_center - frame_center

        combined = self._draw_debug(
            frame=base_frame,   # always RGB debug
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