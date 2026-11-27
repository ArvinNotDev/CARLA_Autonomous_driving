from __future__ import annotations

from collections import deque
import json
import math
import time
from pathlib import Path
from typing import Any

from navigation.junction_trigger import JunctionTriggerLatch


DEFAULT_CITY_SIGN_CLASSES = ["right", "left", "straight", "stop", "90r", "90l"]
DEFAULT_CITY_SIGN_CLASS_ORDER_JSON = json.dumps(DEFAULT_CITY_SIGN_CLASSES)
DEFAULT_CITY_SIGN_CLASS_MAP = {
    "RIGHT": ["right"],
    "LEFT": ["left"],
    "STRAIGHT": ["straight"],
    "STOP": ["stop", "stop_sign", "stop_signs"],
    "RIGHT_90": ["90r", "right_90", "right90", "right_turn_90", "turn_right_90", "90_right", "junction_right"],
    "LEFT_90": ["90l", "left_90", "left90", "left_turn_90", "turn_left_90", "90_left", "junction_left"],
}

DEFAULT_CITY_SIGN_CLASS_MAP_JSON = json.dumps(
    DEFAULT_CITY_SIGN_CLASS_MAP, indent=2
)


DEFAULT_CITY_MANEUVER_SEQUENCES = {
    "STRAIGHT": [
        {"distance_m": 6.0, "steering_value": 0.0},
    ],
    "LEFT": [
        {"distance_m": 22.0, "steering_value": 0.0},
        {"distance_m": 5.0, "steering_value": -0.4},
        {"distance_m": 3.5, "steering_value": -0.5},
        {"distance_m": 2.5, "steering_value": 0.0},
    ],
    "RIGHT": [
        {"distance_m": 9.5, "steering_value": 0.0},
        {"distance_m": 8.0, "steering_value": 0.25},
        {"distance_m": 3.0, "steering_value": 0.7},
    ],
    # AVIS exposes independent 90-degree sign classes. These start from the
    # corresponding CARLA left/right route and can be tuned separately.
    "LEFT_90": [
        {"distance_m": 22.0, "steering_value": 0.0},
        {"distance_m": 5.0, "steering_value": -0.4},
        {"distance_m": 3.5, "steering_value": -0.5},
        {"distance_m": 2.5, "steering_value": 0.0},
    ],
    "RIGHT_90": [
        {"distance_m": 9.5, "steering_value": 0.0},
        {"distance_m": 8.0, "steering_value": 0.25},
        {"distance_m": 3.0, "steering_value": 0.7},
    ],
}

DEFAULT_CITY_MANEUVER_SEQUENCES_JSON = json.dumps(
    DEFAULT_CITY_MANEUVER_SEQUENCES, indent=2
)


def _normalise_class_name(value: Any) -> str:
    text = str(value or "").strip().casefold()
    for character in (" ", "-", "/", "\\"):
        text = text.replace(character, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def parse_city_sign_class_map(value: Any) -> dict[str, set[str]]:
    """Parse class aliases for the sign model's canonical route decisions."""
    raw = json.loads(value) if isinstance(value, str) else value
    if not isinstance(raw, dict):
        raise ValueError("Junction sign class mapping must be a JSON object.")

    result: dict[str, set[str]] = {}
    for canonical in ("RIGHT", "LEFT", "STRAIGHT", "STOP", "RIGHT_90", "LEFT_90"):
        aliases = raw.get(canonical, [])
        if isinstance(aliases, str):
            aliases = [aliases]
        if not isinstance(aliases, list):
            raise ValueError(f"{canonical} aliases must be a string or list.")
        normalized = {
            _normalise_class_name(alias)
            for alias in aliases
            if _normalise_class_name(alias)
        }
        result[canonical] = normalized

    if not any(result.values()):
        raise ValueError("At least one junction sign class alias must be configured.")
    return result


def parse_city_maneuver_sequences(value: Any) -> dict[str, list[dict[str, float]]]:
    """Validate the CARLA-compatible distance/steering sequence format."""
    raw = json.loads(value) if isinstance(value, str) else value
    if not isinstance(raw, dict):
        raise ValueError("Junction maneuver sequences must be a JSON object.")

    result: dict[str, list[dict[str, float]]] = {}
    for key, actions in raw.items():
        maneuver = str(key).strip().upper()
        if not maneuver or maneuver == "STOP":
            raise ValueError("Sequence keys must name a maneuver, not STOP.")
        if not isinstance(actions, list):
            raise ValueError(f"{maneuver} sequence must be a list.")
        normalized_actions: list[dict[str, float]] = []
        for index, action in enumerate(actions):
            if not isinstance(action, dict):
                raise ValueError(f"{maneuver}[{index}] must be an object.")
            try:
                distance = float(action["distance_m"])
                steering = float(action["steering_value"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{maneuver}[{index}] needs numeric distance_m and steering_value."
                ) from exc
            if not math.isfinite(distance) or distance <= 0.0:
                raise ValueError(f"{maneuver}[{index}].distance_m must be greater than zero.")
            if not math.isfinite(steering) or not -1.0 <= steering <= 1.0:
                raise ValueError(f"{maneuver}[{index}].steering_value must be between -1 and 1.")
            normalized_actions.append(
                {"distance_m": distance, "steering_value": steering}
            )
        result[maneuver] = normalized_actions
    return result





class AvisJunctionSignOnnxClassifier:
    """ONNX Runtime image classifier for the six AVIS City junction-sign classes.

    The graph's embedded class names take precedence. For exports without class
    metadata, the explicit class-order setting is used; no class IDs are guessed.
    """

    DEFAULT_CLASSES = tuple(DEFAULT_CITY_SIGN_CLASSES)
    DEFAULT_MEAN = (0.485, 0.456, 0.406)
    DEFAULT_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        model_path: Path | str,
        *,
        class_order: list[str] | tuple[str, ...] | None = None,
        input_size: int = 224,
    ) -> None:
        import ast
        import numpy as np
        import onnxruntime as ort

        self.np = np
        self.ort = ort
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"AVIS junction-sign ONNX model was not found: {self.model_path}"
            )

        configured = [
            _normalise_class_name(item)
            for item in (class_order or self.DEFAULT_CLASSES)
        ]
        if len(configured) != 6 or len(set(configured)) != 6:
            raise ValueError(
                "Junction-sign class order must contain six unique labels: "
                "right, left, straight, stop, 90r, 90l."
            )
        if set(configured) != set(self.DEFAULT_CLASSES):
            raise ValueError(
                "Junction-sign class order must contain exactly "
                "right, left, straight, stop, 90r and 90l."
            )

        self.providers = self._preferred_providers()
        self.session = self._create_session(self.providers)
        metadata = self._metadata()
        metadata_names = self._parse_names(
            metadata.get("names")
            or metadata.get("class_names")
            or metadata.get("classes")
        )
        self.class_names = metadata_names or configured
        if len(self.class_names) != 6 or set(self.class_names) != set(self.DEFAULT_CLASSES):
            raise RuntimeError(
                "The sign ONNX model must expose exactly the six AVIS classes "
                f"(right, left, straight, stop, 90r, 90l); found {self.class_names!r}."
            )

        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(float)":
            raise RuntimeError(
                "Junction-sign ONNX must expose exactly one float32 image input."
            )
        shape = inputs[0].shape
        if len(shape) != 4 or shape[1] not in (3, "3"):
            raise RuntimeError(
                f"Junction-sign ONNX input must be NCHW RGB; received {shape!r}."
            )
        height, width = shape[2], shape[3]
        static_height = height if isinstance(height, int) and height > 0 else None
        static_width = width if isinstance(width, int) and width > 0 else None
        if static_height and static_width and static_height != static_width:
            raise RuntimeError(
                "Junction-sign ONNX expects a non-square input, which is not "
                "supported by the ResNet18 AVIS preprocessing pipeline."
            )
        self.input_size = int(static_height or static_width or input_size)
        if not 32 <= self.input_size <= 2048:
            raise RuntimeError(f"Invalid junction-sign input size: {self.input_size}.")

        self.input_name = inputs[0].name
        outputs = self.session.get_outputs()
        if len(outputs) != 1:
            raise RuntimeError(
                f"Junction-sign ONNX must expose one classification output; found {len(outputs)}."
            )
        self.output_name = outputs[0].name
        output_shape = outputs[0].shape
        if len(output_shape) != 2:
            raise RuntimeError(
                f"Junction-sign output must have shape [1, 6]; got {output_shape!r}."
            )
        output_classes = output_shape[1]
        if isinstance(output_classes, int) and output_classes != 6:
            raise RuntimeError(
                f"Junction-sign output has {output_classes} classes; expected six."
            )

        self.mean = self._metadata_vector(
            metadata.get("normalization_mean") or metadata.get("mean"), self.DEFAULT_MEAN
        )
        self.std = self._metadata_vector(
            metadata.get("normalization_std") or metadata.get("std"), self.DEFAULT_STD
        )
        if not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.std)):
            raise RuntimeError("Junction-sign normalization values must be finite.")
        if np.any(self.std <= 0):
            raise RuntimeError("Junction-sign normalization std values must be positive.")

        self.providers = list(self.session.get_providers())
        print(
            "[AVIS] Junction-sign ResNet18 ONNX loaded: "
            f"{self.model_path} · input={self.input_size} · "
            f"classes={self.class_names} · providers={self.providers}"
        )

    @staticmethod
    def _parse_names(value: Any) -> list[str]:
        if value is None:
            return []
        parsed = value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                try:
                    import ast
                    parsed = ast.literal_eval(value)
                except (SyntaxError, ValueError):
                    return []
        if isinstance(parsed, dict):
            try:
                items = sorted(parsed.items(), key=lambda pair: int(pair[0]))
                parsed = [item[1] for item in items]
            except (TypeError, ValueError):
                try:
                    items = sorted(parsed.items(), key=lambda pair: int(pair[1]))
                    parsed = [item[0] for item in items]
                except (TypeError, ValueError):
                    return []
        if not isinstance(parsed, (list, tuple)):
            return []
        return [_normalise_class_name(item) for item in parsed]

    @staticmethod
    def _metadata_vector(value: Any, default: tuple[float, float, float]):
        import numpy as np
        if value is None:
            return np.asarray(default, dtype=np.float32).reshape(1, 1, 3)
        try:
            values = json.loads(value) if isinstance(value, str) else value
            result = np.asarray(values, dtype=np.float32)
        except (TypeError, ValueError, json.JSONDecodeError):
            return np.asarray(default, dtype=np.float32).reshape(1, 1, 3)
        if result.shape != (3,):
            return np.asarray(default, dtype=np.float32).reshape(1, 1, 3)
        return result.reshape(1, 1, 3)

    def _metadata(self) -> dict[str, Any]:
        try:
            return dict(self.session.get_modelmeta().custom_metadata_map or {})
        except Exception:
            return {}

    def _preferred_providers(self) -> list[str]:
        available = set(self.ort.get_available_providers())
        if "CUDAExecutionProvider" not in available:
            return ["CPUExecutionProvider"]
        preload_dlls = getattr(self.ort, "preload_dlls", None)
        if callable(preload_dlls):
            try:
                preload_dlls()
            except Exception as exc:
                print(
                    "[AVIS] Junction-sign CUDA DLL preload failed; "
                    f"using CPU: {type(exc).__name__}: {exc}"
                )
                return ["CPUExecutionProvider"]
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def _create_session(self, providers: list[str]):
        try:
            return self.ort.InferenceSession(
                str(self.model_path), providers=providers
            )
        except Exception as exc:
            if providers == ["CPUExecutionProvider"]:
                raise
            print(
                "[AVIS] Junction-sign CUDA session initialization failed; "
                f"retrying on CPU: {type(exc).__name__}: {exc}"
            )
            return self.ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )

    def _preprocess(self, frame_bgr: Any):
        from PIL import Image

        frame = self.np.asarray(frame_bgr)
        if frame.ndim != 3 or frame.shape[2] < 3 or frame.size == 0:
            raise ValueError(
                f"Junction-sign classifier expects a BGR image; got {frame.shape!r}."
            )
        if frame.dtype != self.np.uint8:
            raise ValueError(
                f"Junction-sign classifier expects uint8 pixels; got {frame.dtype}."
            )
        rgb = self.np.ascontiguousarray(frame[:, :, :3][:, :, ::-1])
        image = Image.fromarray(rgb, mode="RGB").resize(
            (self.input_size, self.input_size),
            resample=Image.Resampling.BILINEAR,
        )
        tensor = self.np.asarray(image, dtype=self.np.float32) / 255.0
        tensor = (tensor - self.mean) / self.std
        tensor = self.np.transpose(tensor, (2, 0, 1))[None, ...]
        return self.np.ascontiguousarray(tensor, dtype=self.np.float32)

    def predict(
        self,
        frame_bgr: Any,
        *,
        confidence_threshold: float = 0.55,
    ) -> dict[str, Any]:
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            return {
                "valid": False, "class_id": None, "class_name": None,
                "confidence": 0.0, "detections": [],
                "model_classes": list(self.class_names),
                "providers": list(self.providers),
            }
        tensor = self._preprocess(frame_bgr)
        try:
            outputs = self.session.run(
                [self.output_name], {self.input_name: tensor}
            )
        except Exception as exc:
            if self.providers == ["CPUExecutionProvider"]:
                raise
            print(
                "[AVIS] Junction-sign CUDA inference failed; switching to CPU: "
                f"{type(exc).__name__}: {exc}"
            )
            self.session = self._create_session(["CPUExecutionProvider"])
            self.providers = list(self.session.get_providers())
            outputs = self.session.run(
                [self.output_name], {self.input_name: tensor}
            )

        values = self.np.asarray(outputs[0], dtype=self.np.float32)
        if values.shape != (1, 6) or not self.np.all(self.np.isfinite(values)):
            raise RuntimeError(
                f"Junction-sign ONNX returned invalid logits with shape {values.shape!r}; "
                "expected finite [1, 6] values."
            )
        scores = values[0]
        if (
            self.np.all(scores >= 0.0)
            and self.np.all(scores <= 1.0)
            and abs(float(scores.sum()) - 1.0) <= 0.02
        ):
            probabilities = scores / max(float(scores.sum()), 1e-12)
        else:
            shifted = scores - float(scores.max())
            exponentials = self.np.exp(shifted)
            probabilities = exponentials / max(float(exponentials.sum()), 1e-12)
        class_id = int(self.np.argmax(probabilities))
        confidence = float(probabilities[class_id])
        class_name = self.class_names[class_id]
        try:
            threshold = min(0.99, max(0.01, float(confidence_threshold)))
        except (TypeError, ValueError):
            threshold = 0.55
        detections = (
            [{"class_id": class_id, "class_name": class_name, "confidence": confidence}]
            if confidence >= threshold else []
        )
        return {
            "valid": True,
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "probabilities": {
                name: float(probabilities[index])
                for index, name in enumerate(self.class_names)
            },
            "detections": detections,
            "model_classes": list(self.class_names),
            "providers": list(self.providers),
        }


class AvisJunctionSignDetector:
    """Ultralytics detector/classifier for the upcoming AVIS City junction sign."""

    def __init__(self, model_path: Path | str) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"AVIS City junction sign model was not found: {self.model_path}"
            )

        import torch
        from ultralytics import YOLO

        self.torch = torch
        self.device: str | int = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(self.model_path))
        self.class_names = self._normalise_names(getattr(self.model, "names", None))
        if not self.class_names:
            raise RuntimeError(
                "The AVIS junction sign checkpoint does not expose class names."
            )

    @staticmethod
    def _normalise_names(names: Any) -> dict[int, str]:
        if isinstance(names, dict):
            items = names.items()
        elif isinstance(names, (list, tuple)):
            items = enumerate(names)
        else:
            return {}
        result: dict[int, str] = {}
        for key, value in items:
            try:
                class_id = int(key)
            except (TypeError, ValueError):
                continue
            name = str(value).strip()
            if name:
                result[class_id] = name
        return result

    @staticmethod
    def _as_list(values: Any) -> list[Any]:
        if values is None:
            return []
        if hasattr(values, "detach"):
            values = values.detach().cpu()
        if hasattr(values, "tolist"):
            values = values.tolist()
        if isinstance(values, (list, tuple)):
            return list(values)
        return [values]

    def predict(
        self,
        frame_bgr: Any,
        *,
        confidence_threshold: float = 0.55,
    ) -> dict[str, Any]:
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            return {
                "valid": False,
                "class_id": None,
                "class_name": None,
                "confidence": 0.0,
                "detections": [],
                "model_classes": list(self.class_names.values()),
            }

        threshold = min(0.99, max(0.01, float(confidence_threshold)))

        def run(device: str | int):
            return self.model.predict(
                source=frame_bgr,
                conf=threshold,
                device=device,
                verbose=False,
            )

        try:
            results = run(self.device)
        except Exception:
            if self.device == "cpu":
                raise
            try:
                self.torch.cuda.empty_cache()
            except Exception:
                pass
            self.device = "cpu"
            results = run("cpu")

        if not results:
            return {
                "valid": True,
                "class_id": None,
                "class_name": None,
                "confidence": 0.0,
                "detections": [],
                "model_classes": list(self.class_names.values()),
            }

        result = results[0]
        detections: list[dict[str, Any]] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            class_ids = self._as_list(getattr(boxes, "cls", None))
            confidences = self._as_list(getattr(boxes, "conf", None))
            for class_id_value, confidence_value in zip(class_ids, confidences):
                try:
                    class_id = int(class_id_value)
                    confidence = float(confidence_value)
                except (TypeError, ValueError):
                    continue
                class_name = self.class_names.get(class_id, str(class_id))
                if math.isfinite(confidence) and confidence >= threshold:
                    detections.append({
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                    })

        # Also support an Ultralytics image-classification checkpoint.
        if not detections:
            probabilities = getattr(result, "probs", None)
            if probabilities is not None:
                try:
                    class_id = int(probabilities.top1)
                    confidence = float(probabilities.top1conf)
                except (AttributeError, TypeError, ValueError):
                    class_id = -1
                    confidence = 0.0
                if class_id >= 0 and confidence >= threshold:
                    detections.append({
                        "class_id": class_id,
                        "class_name": self.class_names.get(class_id, str(class_id)),
                        "confidence": confidence,
                    })

        detections.sort(key=lambda item: item["confidence"], reverse=True)
        best = detections[0] if detections else {}
        return {
            "valid": True,
            "class_id": best.get("class_id"),
            "class_name": best.get("class_name"),
            "confidence": float(best.get("confidence", 0.0)),
            "detections": detections,
            "model_classes": list(self.class_names.values()),
        }


class AvisCityJunctionController:
    """Crosswalk stop, upcoming sign latch and configurable post-stop maneuvers."""

    STOP_SPEED_KMH = 0.3
    RESTART_COOLDOWN_SECONDS = 5.0

    def __init__(self) -> None:
        self._trigger = JunctionTriggerLatch(clear_duration=1.0)
        self.reset()

    def reset(self) -> None:
        self._trigger = JunctionTriggerLatch(clear_duration=1.0)
        self._phase = "IDLE"
        self._pending_sign_type: str | None = None
        self._pending_sign_confidence = 0.0
        self._pending_sign_at: float | None = None
        self._sign_votes: deque[str] = deque(maxlen=8)
        self._last_sign_observed_at: float | None = None
        self._sign_locked_until_clear = False
        self._sign_clear_frames = 0
        self._last_crosswalk_observation_id: Any = None
        self._line_present = False
        self._crosswalk_detected = False
        self._crosswalk_length_ratio = 0.0
        self._active_route_type: str | None = None
        self._last_route_type: str | None = None
        self._dwell_started_at: float | None = None
        self._cooldown_until: float | None = None
        self._sequence: list[dict[str, float]] = []
        self._segment_index = 0
        self._segment_distance_m = 0.0
        self._segment_started_at: float | None = None
        self._last_reason = "disabled"
        self._just_activated = False
        self._just_started_maneuver = False
        self._just_completed = False
        self._last_sign_prediction: dict[str, Any] | None = None

    @staticmethod
    def _map_prediction(
        prediction: dict[str, Any] | None,
        class_map: dict[str, set[str]],
        confidence_threshold: float,
    ) -> tuple[str | None, float]:
        if not isinstance(prediction, dict) or not prediction.get("valid", False):
            return None, 0.0
        detections = prediction.get("detections")
        if not isinstance(detections, list):
            detections = []
        mapped: list[tuple[str, float]] = []
        for item in detections:
            if not isinstance(item, dict):
                continue
            name = _normalise_class_name(item.get("class_name"))
            try:
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(confidence) or confidence < confidence_threshold:
                continue
            for canonical, aliases in class_map.items():
                if name in aliases:
                    mapped.append((canonical, confidence))
                    break
        if not mapped:
            return None, 0.0

        # STOP is safety-dominant if it is visible alongside another sign.
        stop_confidences = [confidence for kind, confidence in mapped if kind == "STOP"]
        if stop_confidences:
            return "STOP", max(stop_confidences)
        return max(mapped, key=lambda item: item[1])

    def _observe_sign(
        self,
        prediction: dict[str, Any] | None,
        *,
        settings: dict[str, Any],
        now: float,
    ) -> None:
        if not isinstance(prediction, dict):
            return
        try:
            observed_at = float(prediction.get("observed_at"))
        except (TypeError, ValueError):
            return
        if not math.isfinite(observed_at) or (
            self._last_sign_observed_at is not None
            and observed_at <= self._last_sign_observed_at
        ):
            return
        self._last_sign_observed_at = observed_at
        self._last_sign_prediction = dict(prediction)

        max_age = max(0.5, float(settings.get("city_junction_sign_max_age_seconds", 20.0)))
        if self._pending_sign_at is not None and now - self._pending_sign_at > max_age:
            self._pending_sign_type = None
            self._pending_sign_confidence = 0.0
            self._pending_sign_at = None
            self._sign_votes.clear()

        try:
            class_map = parse_city_sign_class_map(
                settings.get(
                    "city_junction_sign_class_map_json",
                    DEFAULT_CITY_SIGN_CLASS_MAP_JSON,
                )
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            class_map = parse_city_sign_class_map(DEFAULT_CITY_SIGN_CLASS_MAP)

        threshold = float(settings.get("city_junction_sign_confidence_threshold", 0.55))
        canonical, confidence = self._map_prediction(prediction, class_map, threshold)

        if self._sign_locked_until_clear:
            if canonical is None:
                self._sign_clear_frames += 1
                required_clear = max(
                    1, int(settings.get("city_junction_sign_rearm_clear_frames", 2))
                )
                if self._sign_clear_frames >= required_clear:
                    self._sign_locked_until_clear = False
                    self._sign_clear_frames = 0
                    self._sign_votes.clear()
            else:
                self._sign_clear_frames = 0
            return

        if canonical is None:
            self._sign_votes.clear()
            return

        confirm_frames = max(
            1, min(10, int(settings.get("city_junction_sign_confirm_frames", 2)))
        )
        self._sign_votes = deque(self._sign_votes, maxlen=confirm_frames)
        self._sign_votes.append(canonical)
        if len(self._sign_votes) >= confirm_frames and all(
            vote == canonical for vote in self._sign_votes
        ):
            self._pending_sign_type = canonical
            self._pending_sign_confidence = float(confidence)
            self._pending_sign_at = observed_at

    def _start_maneuver(
        self,
        *,
        settings: dict[str, Any],
        now: float,
    ) -> None:
        try:
            sequences = parse_city_maneuver_sequences(
                settings.get(
                    "city_maneuver_sequences_json",
                    DEFAULT_CITY_MANEUVER_SEQUENCES_JSON,
                )
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            sequences = parse_city_maneuver_sequences(
                DEFAULT_CITY_MANEUVER_SEQUENCES
            )

        sequence_key = {
            "STRAIGHT": "STRAIGHT",
            "LEFT": "LEFT",
            "RIGHT": "RIGHT",
            "LEFT_90": "LEFT_90",
            "RIGHT_90": "RIGHT_90",
        }.get(self._active_route_type or "STRAIGHT", "STRAIGHT")
        self._sequence = list(sequences.get(sequence_key, []))
        self._segment_index = 0
        self._segment_distance_m = 0.0
        self._segment_started_at = now
        self._dwell_started_at = None
        self._phase = "MANEUVER"
        self._just_started_maneuver = True
        self._last_reason = f"junction_{sequence_key.lower()}_maneuver_started"
        if not self._sequence:
            self._finish(now)

    def _finish(self, now: float) -> None:
        self._last_route_type = self._active_route_type
        self._active_route_type = None
        self._phase = "IDLE"
        self._dwell_started_at = None
        self._segment_started_at = None
        self._sequence = []
        self._segment_index = 0
        self._segment_distance_m = 0.0
        self._cooldown_until = now + self.RESTART_COOLDOWN_SECONDS
        self._just_completed = True
        self._last_reason = "junction_event_complete"

    def update(
        self,
        junction_roi: dict[str, Any] | None,
        sign_prediction: dict[str, Any] | None,
        *,
        enabled: bool,
        settings: dict[str, Any],
        current_speed_kmh: float,
        dt: float,
        observation_id: Any = None,
        observation_timestamp: float | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        current = time.perf_counter() if now is None else float(now)
        if not math.isfinite(current):
            current = time.perf_counter()
        self._just_activated = False
        self._just_started_maneuver = False
        self._just_completed = False

        if not enabled:
            self.reset()
            return self.snapshot(now=current, settings=settings)

        cooldown_active = (
            self._cooldown_until is not None and current < self._cooldown_until
        )
        if self._phase == "IDLE" and not self._sign_locked_until_clear:
            self._observe_sign(sign_prediction, settings=settings, now=current)
        elif self._sign_locked_until_clear:
            self._observe_sign(sign_prediction, settings=settings, now=current)

        if observation_id is not None and observation_id != self._last_crosswalk_observation_id:
            self._last_crosswalk_observation_id = observation_id
            info = junction_roi if isinstance(junction_roi, dict) else {}
            self._line_present = bool(info.get("line_present", False))
            self._crosswalk_detected = bool(info.get("detected", False))
            try:
                self._crosswalk_length_ratio = max(
                    0.0, min(1.0, float(info.get("length_ratio", 0.0)))
                )
            except (TypeError, ValueError):
                self._crosswalk_length_ratio = 0.0
            timestamp = current
            try:
                if observation_timestamp is not None:
                    timestamp = float(observation_timestamp)
            except (TypeError, ValueError):
                timestamp = current
            if not math.isfinite(timestamp):
                timestamp = current
            pending = self._trigger.observe(
                detected=self._crosswalk_detected and self._line_present,
                evidence_present=self._line_present,
                timestamp=timestamp,
                manager_idle=self._phase == "IDLE" and not cooldown_active,
            )
            if pending and self._phase == "IDLE" and not cooldown_active:
                self._trigger.acknowledge()
                self._active_route_type = self._pending_sign_type or "STRAIGHT"
                self._pending_sign_type = None
                self._pending_sign_confidence = 0.0
                self._pending_sign_at = None
                self._sign_locked_until_clear = True
                self._sign_clear_frames = 0
                self._phase = "BRAKING"
                self._dwell_started_at = None
                self._just_activated = True
                self._last_reason = "crosswalk_confirmed_braking_started"

        try:
            speed = max(0.0, float(current_speed_kmh))
        except (TypeError, ValueError):
            speed = float("inf")
        if not math.isfinite(speed):
            speed = float("inf")

        if self._phase == "BRAKING" and speed <= self.STOP_SPEED_KMH:
            if self._active_route_type in {"LEFT_90", "RIGHT_90"}:
                self._start_maneuver(settings=settings, now=current)
            else:
                self._phase = "DWELL"
                self._dwell_started_at = current
                self._last_reason = "crosswalk_stationary_dwell_started"

        elif self._phase == "DWELL":
            stop_seconds = max(0.0, float(settings.get("city_crosswalk_stop_seconds", 5.0)))
            elapsed = (
                max(0.0, current - self._dwell_started_at)
                if self._dwell_started_at is not None else 0.0
            )
            if elapsed >= stop_seconds:
                if self._active_route_type == "STOP":
                    # STOP is a terminal route instruction: after the same
                    # configured stop dwell, keep the vehicle held until the
                    # operator explicitly switches mode or resets the run.
                    self._last_route_type = "STOP"
                    self._phase = "STOPPED"
                    self._dwell_started_at = None
                    self._sequence = []
                    self._segment_index = 0
                    self._segment_distance_m = 0.0
                    self._segment_started_at = None
                    self._just_completed = True
                    self._last_reason = "stop_sign_terminal_stop"
                else:
                    self._start_maneuver(settings=settings, now=current)
            else:
                self._last_reason = "crosswalk_stationary_dwell_in_progress"

        elif self._phase == "MANEUVER":
            try:
                delta = min(0.25, max(0.0, float(dt)))
            except (TypeError, ValueError):
                delta = 0.0
            self._segment_distance_m += (speed / 3.6) * delta
            if self._segment_index < len(self._sequence):
                action = self._sequence[self._segment_index]
                target_distance = float(action["distance_m"])
                segment_elapsed = (
                    max(0.0, current - self._segment_started_at)
                    if self._segment_started_at is not None else 0.0
                )
                timeout = max(
                    1.0,
                    float(settings.get("city_maneuver_segment_timeout_seconds", 20.0)),
                )
                if self._segment_distance_m >= target_distance or segment_elapsed >= timeout:
                    self._segment_index += 1
                    self._segment_distance_m = 0.0
                    self._segment_started_at = current
                    if self._segment_index >= len(self._sequence):
                        self._finish(current)
                    else:
                        self._last_reason = "junction_maneuver_next_segment"

        if self._phase == "IDLE" and not cooldown_active and self._cooldown_until is not None:
            self._cooldown_until = None
            self._last_reason = "junction_cooldown_complete"

        return self.snapshot(now=current, settings=settings)

    def snapshot(
        self,
        *,
        now: float | None = None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        current = time.perf_counter() if now is None else float(now)
        options = settings or {}
        cooldown_remaining = (
            max(0.0, self._cooldown_until - current)
            if self._cooldown_until is not None else 0.0
        )
        dwell_elapsed = (
            max(0.0, current - self._dwell_started_at)
            if self._dwell_started_at is not None else 0.0
        )
        dwell_required = max(0.0, float(options.get("city_crosswalk_stop_seconds", 5.0)))
        dwell_remaining = max(0.0, dwell_required - dwell_elapsed) if self._phase == "DWELL" else 0.0
        maneuver_active = self._phase == "MANEUVER"
        action = (
            self._sequence[self._segment_index]
            if maneuver_active and self._segment_index < len(self._sequence)
            else None
        )
        remaining = (
            max(0.0, float(action["distance_m"]) - self._segment_distance_m)
            if action is not None else 0.0
        )
        prediction = self._last_sign_prediction or {}
        return {
            "phase": self._phase,
            "active": self._phase in {"BRAKING", "DWELL", "MANEUVER", "STOPPED"},
            "braking": self._phase in {"BRAKING", "DWELL", "STOPPED"},
            "waiting_for_stop": self._phase == "BRAKING",
            "dwell_active": self._phase == "DWELL",
            "dwell_remaining_seconds": float(dwell_remaining),
            "dwell_elapsed_seconds": float(dwell_elapsed if self._phase == "DWELL" else 0.0),
            "remaining_seconds": float(dwell_remaining),
            "elapsed_seconds": float(dwell_elapsed if self._phase == "DWELL" else 0.0),
            "maneuver_active": bool(maneuver_active),
            "route_type": self._active_route_type or self._last_route_type,
            "last_route_type": self._last_route_type,
            "steering_override": (
                float(action["steering_value"]) if action is not None else None
            ),
            "segment_index": int(self._segment_index + 1) if action is not None else 0,
            "segment_count": len(self._sequence),
            "segment_distance_remaining_m": float(remaining),
            "line_present": bool(self._line_present),
            "crosswalk_detected": bool(self._crosswalk_detected),
            "detected": bool(self._crosswalk_detected),
            "length_ratio": float(self._crosswalk_length_ratio),
            "pending_sign_type": (
                self._pending_sign_type
                or self._active_route_type
                or ("STRAIGHT" if self._phase == "IDLE" else "UNKNOWN")
            ),
            "pending_sign_confidence": float(self._pending_sign_confidence),
            "sign_class": prediction.get("class_name"),
            "sign_confidence": prediction.get("confidence"),
            "sign_model_classes": prediction.get("model_classes", []),
            "just_activated": bool(self._just_activated),
            "just_started_maneuver": bool(self._just_started_maneuver),
            "just_completed": bool(self._just_completed),
            "cooldown_active": cooldown_remaining > 0.0,
            "cooldown_remaining_seconds": float(cooldown_remaining),
            "reason": self._last_reason,
        }
