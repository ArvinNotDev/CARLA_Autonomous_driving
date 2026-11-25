import ast
from pathlib import Path

import cv2
import numpy as np

import config_city as conf
from vision.color_extractor import HSVColorThresholdExtractor
from navigation.crosswalk_junction_detector import CrosswalkRoiJunctionDetector


class VisionProcessor:
    def __init__(
        self,
        mode: str = "ultralytics",
        color_extractor: HSVColorThresholdExtractor | None = None,
    ):
        self.mode = mode.lower().strip()
        if self.mode not in {"segmentation", "onnx", "ultralytics"}:
            raise ValueError(
                "mode must be 'segmentation', 'onnx', or 'ultralytics'"
            )

        self.debug = bool(getattr(conf, "VISION_DEBUG", False))

        self.extractor = color_extractor or HSVColorThresholdExtractor(
            morph_kernel_size=getattr(conf, "MORPH_KERNEL_SIZE", 1)
        )

        self.last_lane_center = None
        self.lane_center_alpha = getattr(conf, "LANE_CENTER_SMOOTH_ALPHA", 0.35)
        self.fallback_lane_offset_ratio = getattr(conf, "FALLBACK_LANE_OFFSET_RATIO", 0.18)
        self.min_side_pixels = getattr(conf, "MIN_SIDE_PIXELS", 80)

        self.active_side = None
        self.crosswalk_junction_enabled = (
            str(getattr(conf, "JUNCTION_DETECTION_MODE", "model")).strip().lower()
            == "crosswalk_roi"
        )
        self.crosswalk_junction_detector = CrosswalkRoiJunctionDetector()

        self.session = None
        self.model = None
        self.torch = None
        self.device = "cpu"
        self.model_image_size = max(
            160, int(getattr(conf, "MODEL_IMAGE_SIZE", 512))
        )
        self.input_name = None
        self.output_name = None
        self.input_width = None
        self.input_height = None
        self.output_width = None
        self.output_height = None
        self.lane_class_id = None
        self.drivable_class_id = None
        self.vehicles_class_id = None
        self.class_names = {}

        if self.mode in {"onnx", "ultralytics"}:
            model_path = self._resolve_model_path()
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"Vision model not found: {model_path}. "
                    "Place the configured .pt checkpoint at this path."
                )
            if self.mode == "onnx":
                try:
                    import onnxruntime as ort
                except ImportError as exc:
                    raise ImportError(
                        "Legacy ONNX mode requires onnxruntime. "
                        "The default Ultralytics .pt mode does not."
                    ) from exc
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                opts.intra_op_num_threads = 4
                opts.inter_op_num_threads = 1
                self.session = ort.InferenceSession(
                    str(model_path),
                    sess_options=opts,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                print(
                    "ONNX Runtime providers: "
                    f"{self.session.get_providers()}"
                )
                self._configure_onnx_model()
            else:
                self._configure_ultralytics_model()

    @staticmethod
    def _resolve_model_path() -> Path:
        project_root = Path(__file__).resolve().parents[1]
        model_path = Path(str(conf.MODEL_PATH)).expanduser()
        if not model_path.is_absolute():
            model_path = project_root / model_path
        return model_path.resolve()

    @staticmethod
    def select_ultralytics_device(torch_module) -> str:
        """Return Ultralytics' explicit CUDA device, or CPU when unavailable."""
        try:
            if (
                torch_module.cuda.is_available()
                and torch_module.cuda.device_count() > 0
            ):
                return "0"
        except Exception:
            pass
        return "cpu"

    @staticmethod
    def _normalize_class_names(raw_names) -> dict[int, str]:
        if isinstance(raw_names, dict):
            pairs = raw_names.items()
        elif isinstance(raw_names, (list, tuple)):
            pairs = enumerate(raw_names)
        else:
            raise RuntimeError(
                "Ultralytics checkpoint does not expose a class-name mapping."
            )
        names = {}
        for class_id, label in pairs:
            try:
                normalized_id = int(class_id)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Invalid class ID in Ultralytics names: {class_id!r}"
                ) from exc
            if not isinstance(label, str) or not label.strip():
                raise RuntimeError(
                    f"Invalid class name for class ID {normalized_id}: {label!r}"
                )
            names[normalized_id] = label.strip().lower().replace(" ", "_")
        return names

    @staticmethod
    def _class_ids_from_names(class_names: dict[int, str]) -> tuple[int, int, int]:
        required = (
            ("lane_lines", "lane"),
            ("ego_drivable_area", "drivable-area"),
            ("vehicles", "vehicle"),
        )
        ids = []
        for class_name, label in required:
            matches = [
                class_id
                for class_id, name in class_names.items()
                if name == class_name
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "The Ultralytics semantic checkpoint must contain exactly "
                    f"one '{class_name}' class for {label} masks; "
                    f"found names {class_names!r}"
                )
            ids.append(matches[0])
        return tuple(ids)

    @staticmethod
    def _semantic_map_to_masks(
        semantic_map,
        lane_class_id: int,
        drivable_class_id: int,
        vehicles_class_id: int,
        frame_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        class_map = np.asarray(semantic_map)
        class_map = np.squeeze(class_map)
        if class_map.ndim != 2:
            raise RuntimeError(
                "Ultralytics semantic output must be a two-dimensional "
                f"class-ID map; received shape {class_map.shape!r}"
            )
        frame_height, frame_width = frame_shape
        if class_map.shape != (frame_height, frame_width):
            class_map = cv2.resize(
                class_map.astype(np.float32),
                (frame_width, frame_height),
                interpolation=cv2.INTER_NEAREST,
            )
        lane_mask = (
            (class_map == lane_class_id).astype(np.uint8) * 255
        )
        drivable_mask = (
            (class_map == drivable_class_id).astype(np.uint8) * 255
        )
        vehicles_mask = (
            (class_map == vehicles_class_id).astype(np.uint8) * 255
        )
        return drivable_mask, lane_mask, vehicles_mask

    def _configure_ultralytics_model(self) -> None:
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics vision mode requires PyTorch and ultralytics. "
                "Install the project requirements before starting CARLA."
            ) from exc

        self.torch = torch
        self.device = self.select_ultralytics_device(torch)
        self.model = YOLO(str(self._resolve_model_path()))
        self.class_names = self._normalize_class_names(
            getattr(self.model, "names", None)
        )
        (
            self.lane_class_id,
            self.drivable_class_id,
            self.vehicles_class_id,
        ) = self._class_ids_from_names(self.class_names)

        if self.device == "cpu":
            print("Ultralytics device: CPU (CUDA is not available)")
        else:
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = "CUDA device 0"
            print(f"Ultralytics device: CUDA:0 ({gpu_name})")
        print(
            "Ultralytics semantic classes: "
            f"{self.class_names}"
        )

    def _predict_ultralytics(self, frame):
        if self.model is None:
            raise RuntimeError("Ultralytics vision model is not loaded.")

        def predict(device):
            return self.model.predict(
                source=frame,
                imgsz=self.model_image_size,
                device=device,
                verbose=False,
            )

        try:
            return predict(self.device)
        except Exception as exc:
            if self.device == "cpu":
                raise
            print(
                "Ultralytics CUDA inference failed; retrying on CPU: "
                f"{exc}"
            )
            try:
                self.torch.cuda.empty_cache()
            except Exception:
                pass
            self.device = "cpu"
            return predict("cpu")

    def _extract_masks_from_ultralytics(self, camera_frame):
        predictions = self._predict_ultralytics(camera_frame)
        if not predictions:
            raise RuntimeError("Ultralytics returned no prediction result.")
        result = predictions[0]
        semantic_output = getattr(result, "semantic_mask", None)
        if semantic_output is None or getattr(semantic_output, "data", None) is None:
            raise RuntimeError(
                "The configured .pt checkpoint did not return semantic_mask. "
                "Use an Ultralytics YOLO26-sem semantic-segmentation checkpoint "
                "trained with lane_lines, ego_drivable_area, and vehicles classes."
            )
        semantic_data = semantic_output.data
        if hasattr(semantic_data, "detach"):
            semantic_data = semantic_data.detach()
        if hasattr(semantic_data, "cpu"):
            semantic_data = semantic_data.cpu()
        if hasattr(semantic_data, "numpy"):
            semantic_data = semantic_data.numpy()
        return self._semantic_map_to_masks(
            semantic_data,
            lane_class_id=self.lane_class_id,
            drivable_class_id=self.drivable_class_id,
            vehicles_class_id=self.vehicles_class_id,
            frame_shape=np.asarray(camera_frame).shape[:2],
        )

    @staticmethod
    def _static_dimension(value, tensor_name, axis):
        if not isinstance(value, int) or value <= 0:
            raise RuntimeError(
                f"ONNX tensor '{tensor_name}' must have a positive static "
                f"dimension at axis {axis}; got {value!r}"
            )
        return value

    def _configure_onnx_model(self):
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise RuntimeError(
                f"YOLO26s-sem ONNX expects one input tensor; found {len(inputs)}"
            )

        model_input = inputs[0]
        if model_input.type != "tensor(float)":
            raise RuntimeError(
                f"Unsupported ONNX input type {model_input.type!r}; "
                "expected tensor(float)"
            )
        if len(model_input.shape) != 4:
            raise RuntimeError(
                f"Unsupported ONNX input shape {model_input.shape!r}; "
                "expected NCHW [1, 3, height, width]"
            )

        batch, channels, height, width = (
            self._static_dimension(value, model_input.name, axis)
            for axis, value in enumerate(model_input.shape)
        )
        if batch != 1 or channels != 3:
            raise RuntimeError(
                f"Unsupported ONNX input shape {model_input.shape!r}; "
                "expected a single NCHW RGB image"
            )
        self.input_name = model_input.name
        self.input_height = height
        self.input_width = width

        outputs = self.session.get_outputs()
        if len(outputs) != 1:
            raise RuntimeError(
                f"YOLO26s-sem ONNX expects one semantic output tensor; "
                f"found {len(outputs)}"
            )
        model_output = outputs[0]
        if model_output.type != "tensor(uint8)":
            raise RuntimeError(
                f"Unsupported ONNX output type {model_output.type!r}; "
                "expected decoded tensor(uint8) class IDs"
            )
        if len(model_output.shape) != 3:
            raise RuntimeError(
                f"Unsupported ONNX output shape {model_output.shape!r}; "
                "expected [1, height, width] decoded class IDs"
            )
        output_batch, output_height, output_width = (
            self._static_dimension(value, model_output.name, axis)
            for axis, value in enumerate(model_output.shape)
        )
        if (
            output_batch != 1
            or output_height != self.input_height
            or output_width != self.input_width
        ):
            raise RuntimeError(
                f"Unsupported ONNX output shape {model_output.shape!r}; "
                f"expected [1, {self.input_height}, {self.input_width}]"
            )
        self.output_name = model_output.name
        self.output_height = output_height
        self.output_width = output_width

        metadata = self.session.get_modelmeta().custom_metadata_map or {}
        raw_names = metadata.get("names")
        if raw_names is None:
            raise RuntimeError(
                "YOLO26s-sem ONNX metadata does not contain class names"
            )
        try:
            parsed_names = ast.literal_eval(raw_names)
            if isinstance(parsed_names, dict):
                class_names = {
                    int(class_id): name for class_id, name in parsed_names.items()
                }
            elif isinstance(parsed_names, (list, tuple)):
                class_names = dict(enumerate(parsed_names))
            else:
                raise ValueError("class names must be a mapping or sequence")
        except (SyntaxError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Unable to parse ONNX class-name metadata: {raw_names!r}"
            ) from exc

        if (
            any(not isinstance(name, str) for name in class_names.values())
            or len(set(class_names.values())) != len(class_names)
        ):
            raise RuntimeError(
                "Expected unique semantic class names; "
                f"found {class_names!r}"
            )
        lane_ids = [
            class_id
            for class_id, name in class_names.items()
            if name == "lane_lines"
        ]
        drivable_ids = [
            class_id
            for class_id, name in class_names.items()
            if name == "ego_drivable_area"
        ]
        vehicles_ids = [
            class_id
            for class_id, name in class_names.items()
            if name == "vehicles"
        ]
        if (
            len(lane_ids) != 1
            or len(drivable_ids) != 1
            or len(vehicles_ids) != 1
        ):
            raise RuntimeError(
                "ONNX class metadata must identify exactly one 'lane_lines' "
                "'ego_drivable_area', and 'vehicles' class; "
                f"found {class_names!r}"
            )
        self.class_names = class_names
        self.lane_class_id = lane_ids[0]
        self.drivable_class_id = drivable_ids[0]
        self.vehicles_class_id = vehicles_ids[0]

    def _preprocess_onnx(self, frame):
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Expected a three-channel BGR camera frame; got shape {frame.shape!r}"
            )
        if frame.dtype != np.uint8:
            raise ValueError(
                f"Expected a uint8 camera frame; got dtype {frame.dtype}"
            )

        frame_height, frame_width = frame.shape[:2]
        if frame_height <= 0 or frame_width <= 0:
            raise ValueError(f"Invalid camera frame dimensions: {frame.shape!r}")

        scale = min(
            self.input_width / frame_width,
            self.input_height / frame_height,
        )
        resized_width = max(1, int(round(frame_width * scale)))
        resized_height = max(1, int(round(frame_height * scale)))
        if (resized_width, resized_height) == (frame_width, frame_height):
            resized = frame
        else:
            resized = cv2.resize(
                frame,
                (resized_width, resized_height),
                interpolation=cv2.INTER_LINEAR,
            )

        pad_width = self.input_width - resized_width
        pad_height = self.input_height - resized_height
        left = int(round(pad_width / 2.0 - 0.1))
        top = int(round(pad_height / 2.0 - 0.1))
        if pad_width == 0 and pad_height == 0:
            letterboxed = resized
        else:
            letterboxed = np.full(
                (self.input_height, self.input_width, 3), 114, dtype=np.uint8
            )
            letterboxed[
                top : top + resized_height,
                left : left + resized_width,
            ] = resized

        image = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image)
        transform = (top, left, resized_height, resized_width)
        return image, transform

    def _extract_masks_from_onnx(self, camera_frame):
        input_tensor, transform = self._preprocess_onnx(camera_frame)
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor},
        )
        if len(outputs) != 1 or outputs[0] is None:
            raise RuntimeError("YOLO26s-sem ONNX returned no semantic output")

        output = np.asarray(outputs[0])
        expected_shape = (1, self.output_height, self.output_width)
        if output.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected semantic output shape {output.shape!r}; "
                f"expected {expected_shape!r}"
            )
        if output.dtype != np.uint8:
            raise RuntimeError(
                f"Unexpected semantic output dtype {output.dtype}; expected uint8"
            )

        semantic_mask = np.squeeze(output)
        if semantic_mask.ndim != 2:
            raise RuntimeError(
                "Decoded semantic class map must be two-dimensional after "
                f"squeezing; got shape {semantic_mask.shape!r}"
            )

        top, left, resized_height, resized_width = transform
        semantic_mask = semantic_mask[
            top : top + resized_height,
            left : left + resized_width,
        ]
        if semantic_mask.shape != (resized_height, resized_width):
            raise RuntimeError(
                "Letterbox crop did not match the resized camera frame: "
                f"got {semantic_mask.shape!r}, expected "
                f"{(resized_height, resized_width)!r}"
            )

        lane_mask = (semantic_mask == self.lane_class_id).astype(np.uint8) * 255
        drivable_mask = (
            (semantic_mask == self.drivable_class_id).astype(np.uint8) * 255
        )
        vehicles_mask = (
            (semantic_mask == self.vehicles_class_id).astype(np.uint8) * 255
        )
        frame_height, frame_width = np.asarray(camera_frame).shape[:2]
        if lane_mask.shape != (frame_height, frame_width):
            lane_mask = cv2.resize(
                lane_mask,
                (frame_width, frame_height),
                interpolation=cv2.INTER_NEAREST,
            )
        if drivable_mask.shape != (frame_height, frame_width):
            drivable_mask = cv2.resize(
                drivable_mask,
                (frame_width, frame_height),
                interpolation=cv2.INTER_NEAREST,
            )
        if vehicles_mask.shape != (frame_height, frame_width):
            vehicles_mask = cv2.resize(
                vehicles_mask,
                (frame_width, frame_height),
                interpolation=cv2.INTER_NEAREST,
            )
        return drivable_mask, lane_mask, vehicles_mask

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

    def _draw_debug(
        self, frame, line_mask, drivable_mask, left_x, right_x, lane_center,
        lane_type, vehicles_mask=None,
    ):
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

        if vehicles_mask is not None and getattr(conf, "DEBUG_SHOW_VEHICLES", True):
            vehicles = np.asarray(vehicles_mask)
            if vehicles.ndim == 3:
                vehicles = cv2.cvtColor(vehicles, cv2.COLOR_BGR2GRAY)
            if vehicles.shape[:2] != (h, w):
                vehicles = cv2.resize(
                    vehicles, (w, h), interpolation=cv2.INTER_NEAREST
                )
            vehicle_pixels = vehicles > 0
            if np.any(vehicle_pixels):
                overlay = vis.copy()
                overlay[vehicle_pixels] = (0, 0, 255)
                vis = cv2.addWeighted(vis, 0.55, overlay, 0.45, 0)
                cv2.putText(
                    vis, "Vehicles", (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 255), 2, cv2.LINE_AA,
                )

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

        mode == 'onnx' or 'ultralytics':
            - rgb_frame is used for inference and debug display when available
            - semantic_frame is kept only as a fallback if rgb_frame is not provided
        """
        base_frame = rgb_frame if rgb_frame is not None else semantic_frame

        if base_frame is None or base_frame.size == 0:
            return {
                "error": 0.0,
                "lane_type": "none",
                "junction_roi": {
                    "line_present": False, "detected": False,
                    "line": None, "length_ratio": 0.0,
                },
                "debug": {
                    "combined": None,
                    "drivable_mask": None,
                    "vehicles_mask": None,
                    "lane_mask": None,
                    "green_mask": None,
                    "dark_purple_mask": None,
                    "left_x": None,
                    "right_x": None,
                    "lane_center": None,
                },
            }

        junction_roi_info = None
        if self.crosswalk_junction_enabled:
            junction_roi_info = self.crosswalk_junction_detector.detect(base_frame)

        if self.mode == "segmentation":
            if semantic_frame is None or semantic_frame.size == 0:
                return {
                    "error": 0.0,
                    "lane_type": "none",
                    "junction_roi": junction_roi_info,
                    "debug": {
                        "combined": None,
                        "drivable_mask": None,
                        "vehicles_mask": None,
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
            drivable_mask, line_mask, vehicles_mask = self._extract_masks_from_onnx(rgb_frame)
            green_mask = None
            dark_purple_mask = None
        elif self.mode == "ultralytics":
            drivable_mask, line_mask, vehicles_mask = (
                self._extract_masks_from_ultralytics(rgb_frame)
            )
            green_mask = None
            dark_purple_mask = None
        else:
            line_mask, green_mask, dark_purple_mask = self._extract_masks_from_segmentation(semantic_frame)
            drivable_mask = None
            vehicles_mask = None

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
            bool(getattr(conf, "DEBUG_SHOW_VEHICLES", True)),
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
                vehicles_mask=vehicles_mask,
            )
        else:
            combined = np.ascontiguousarray(base_frame[:, :, :3]).copy()

        if junction_roi_info is not None:
            line = junction_roi_info.get("line")
            if line is not None:
                cv2.line(
                    combined,
                    (int(line[0]), int(line[1])),
                    (int(line[2]), int(line[3])),
                    (0, 200, 255),
                    3,
                    cv2.LINE_AA,
                )

        return {
            "error": error,
            "lane_type": lane_type,
            "junction_roi": junction_roi_info,
            "debug": {
                "combined": combined,
                "drivable_mask": drivable_mask,
                "vehicles_mask": vehicles_mask,
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
