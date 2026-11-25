from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vision.city_vision_processing import VisionProcessor


def test_ultralytics_class_ids_match_the_required_semantic_classes():
    names = {
        0: "ego_drivable_area",
        1: "lane_lines",
        2: "vehicles",
    }
    assert VisionProcessor._class_ids_from_names(names) == (1, 0, 2)


def test_ultralytics_rejects_checkpoints_without_the_required_class_names():
    names = {0: "road", 1: "lane_lines", 2: "vehicles"}
    try:
        VisionProcessor._class_ids_from_names(names)
    except RuntimeError as exc:
        assert "ego_drivable_area" in str(exc)
    else:
        raise AssertionError("Missing semantic class should fail validation")


def test_semantic_class_map_becomes_full_resolution_binary_masks():
    semantic = np.array(
        [
            [0, 1],
            [2, 0],
        ],
        dtype=np.uint8,
    )
    drivable, lane, vehicles = VisionProcessor._semantic_map_to_masks(
        semantic,
        lane_class_id=1,
        drivable_class_id=0,
        vehicles_class_id=2,
        frame_shape=(2, 2),
    )
    assert drivable.dtype == np.uint8
    assert lane.dtype == np.uint8
    assert vehicles.dtype == np.uint8
    np.testing.assert_array_equal(
        drivable, np.array([[255, 0], [0, 255]], dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        lane, np.array([[0, 255], [0, 0]], dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        vehicles, np.array([[0, 0], [255, 0]], dtype=np.uint8)
    )


def test_semantic_mask_is_resized_to_the_original_camera_dimensions():
    semantic = np.array([[0, 1], [2, 0]], dtype=np.uint8)
    masks = VisionProcessor._semantic_map_to_masks(
        semantic, lane_class_id=1, drivable_class_id=0,
        vehicles_class_id=2, frame_shape=(8, 8)
    )
    assert all(mask.shape == (8, 8) for mask in masks)


def test_ultralytics_extraction_reads_semantic_mask_data():
    processor = object.__new__(VisionProcessor)
    processor.model = object()
    processor.lane_class_id = 1
    processor.drivable_class_id = 0
    processor.vehicles_class_id = 2
    semantic = np.array([[0, 1], [2, 0]], dtype=np.uint8)
    prediction = SimpleNamespace(
        semantic_mask=SimpleNamespace(data=semantic)
    )
    processor._predict_ultralytics = lambda _frame: [prediction]

    drivable, lane, vehicles = processor._extract_masks_from_ultralytics(
        np.zeros((2, 2, 3), dtype=np.uint8)
    )
    assert drivable[0, 0] == 255
    assert lane[0, 1] == 255
    assert vehicles[1, 0] == 255


def test_ultralytics_explicitly_selects_cuda_when_available():
    cuda_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
        )
    )
    cpu_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
        )
    )
    assert VisionProcessor.select_ultralytics_device(cuda_torch) == "0"
    assert VisionProcessor.select_ultralytics_device(cpu_torch) == "cpu"


def test_ultralytics_retries_on_cpu_if_cuda_prediction_fails():
    calls = []

    class FakeCuda:
        @staticmethod
        def empty_cache():
            return None

    class FakeModel:
        @staticmethod
        def predict(**kwargs):
            calls.append(kwargs["device"])
            if kwargs["device"] == "0":
                raise RuntimeError("synthetic CUDA failure")
            return ["cpu-result"]

    processor = object.__new__(VisionProcessor)
    processor.model = FakeModel()
    processor.torch = SimpleNamespace(cuda=FakeCuda())
    processor.device = "0"
    processor.model_image_size = 512

    result = processor._predict_ultralytics(np.zeros((4, 4, 3), dtype=np.uint8))
    assert result == ["cpu-result"]
    assert calls == ["0", "cpu"]
    assert processor.device == "cpu"

def test_inference_worker_accepts_vision_model_after_async_loading():
    from trajectory.worker import InferenceWorker

    worker = InferenceWorker()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    assert worker.submit_vision(
        source_frame_id=1, semantic_frame=None, rgb_frame=frame
    ) == 0

    processor = SimpleNamespace(detect=lambda _semantic, _rgb: {})
    worker.set_vision_processor(processor)
    assert worker.submit_vision(
        source_frame_id=2, semantic_frame=None, rgb_frame=frame
    ) > 0
    worker.stop()

