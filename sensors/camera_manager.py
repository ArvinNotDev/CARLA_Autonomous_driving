from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Optional

import carla
import numpy as np

import config_city as conf


@dataclass(frozen=True)
class CameraFrame:
    frame_id: int
    captured_at: float
    bgr: np.ndarray


class CameraManager:
    """
    CARLA camera ownership and latest-frame storage.

    Each callback copies the CARLA buffer exactly once because raw_data is only
    valid for the callback lifetime. Readers receive the immutable owned array
    without another copy.
    """

    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle

        self.rgb_camera = None
        self.semantic_camera = None
        self.alt_camera = None

        self._latest_rgb: Optional[CameraFrame] = None
        self._latest_semantic: Optional[CameraFrame] = None
        self._latest_drivable: Optional[CameraFrame] = None
        self._latest_drivable_mask: Optional[CameraFrame] = None
        self._latest_alt: Optional[CameraFrame] = None

        self.rgb_lock = threading.Lock()
        self.semantic_lock = threading.Lock()
        self.alt_lock = threading.Lock()

    @staticmethod
    def _copy_bgra_bgr(image) -> np.ndarray:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            (image.height, image.width, 4)
        )
        frame = arr[:, :, :3].copy()
        frame.setflags(write=False)
        return frame

    def start(self):
        bp_lib = self.world.get_blueprint_library()

        camera_transform = carla.Transform(
            carla.Location(
                x=conf.CAMERA_X,
                y=conf.CAMERA_Y,
                z=conf.CAMERA_Z,
            ),
            carla.Rotation(
                pitch=conf.CAMERA_PITCH_DEG,
                yaw=conf.CAMERA_YAW_DEG,
                roll=conf.CAMERA_ROLL_DEG,
            ),
        )

        rgb_bp = bp_lib.find("sensor.camera.rgb")
        rgb_bp.set_attribute("image_size_x", str(conf.CAMERA_IMAGE_WIDTH))
        rgb_bp.set_attribute("image_size_y", str(conf.CAMERA_IMAGE_HEIGHT))
        rgb_bp.set_attribute("fov", str(conf.CAMERA_FOV))
        rgb_bp.set_attribute("sensor_tick", str(conf.CAMERA_SENSOR_TICK))

        self.rgb_camera = self.world.spawn_actor(
            rgb_bp,
            camera_transform,
            attach_to=self.vehicle,
        )
        self.rgb_camera.listen(self._rgb_callback)

        # The current vision pipeline is YOLOPv2/ONNX and does not consume
        # semantic frames. Keep the sensor optional for legacy segmentation mode.
        use_semantic = bool(
            getattr(conf, "ENABLE_SEMANTIC_CAMERA", False)
            or str(getattr(conf, "VISION_MODE", "onnx")).lower() == "segmentation"
        )
        if use_semantic:
            sem_bp = bp_lib.find("sensor.camera.semantic_segmentation")
            sem_bp.set_attribute("image_size_x", str(conf.CAMERA_IMAGE_WIDTH))
            sem_bp.set_attribute("image_size_y", str(conf.CAMERA_IMAGE_HEIGHT))
            sem_bp.set_attribute("fov", str(conf.CAMERA_FOV))
            sem_bp.set_attribute("sensor_tick", str(conf.CAMERA_SENSOR_TICK))

            self.semantic_camera = self.world.spawn_actor(
                sem_bp,
                camera_transform,
                attach_to=self.vehicle,
            )
            self.semantic_camera.listen(self._semantic_callback)

        # This camera was previously spawned unconditionally but is not used by
        # the runtime control path. Keep it opt-in to avoid an unnecessary sensor
        # callback and memory copy.
        if bool(getattr(conf, "ENABLE_ALT_CAMERA", False)):
            alt_transform = carla.Transform(
                carla.Location(
                    x=conf.ALT_CAMERA_X,
                    y=conf.ALT_CAMERA_Y,
                    z=conf.ALT_CAMERA_Z,
                ),
                carla.Rotation(
                    pitch=conf.ALT_CAMERA_PITCH_DEG,
                    yaw=0.0,
                    roll=0.0,
                ),
            )

            alt_bp = bp_lib.find("sensor.camera.rgb")
            alt_bp.set_attribute("image_size_x", str(conf.model_image_size[0]))
            alt_bp.set_attribute("image_size_y", str(conf.model_image_size[1]))
            alt_bp.set_attribute("fov", str(conf.CAMERA_FOV))
            alt_bp.set_attribute("sensor_tick", str(conf.CAMERA_SENSOR_TICK))

            self.alt_camera = self.world.spawn_actor(
                alt_bp,
                alt_transform,
                attach_to=self.vehicle,
            )
            self.alt_camera.listen(self._alt_callback)

    def _rgb_callback(self, image):
        frame = self._copy_bgra_bgr(image)
        packet = CameraFrame(int(image.frame), time.perf_counter(), frame)
        with self.rgb_lock:
            self._latest_rgb = packet

    def _semantic_callback(self, image):
        raw = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            (image.height, image.width, 4)
        )
        class_map = raw[:, :, 2]
        drivable_mask = (class_map == 1).astype(np.uint8) * 255
        drivable_mask.setflags(write=False)

        image.convert(carla.ColorConverter.CityScapesPalette)
        visual = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            (image.height, image.width, 4)
        )[:, :, :3].copy()
        visual.setflags(write=False)

        drivable_view = np.zeros_like(visual)
        drivable_view[:, :, 1] = drivable_mask
        drivable_view.setflags(write=False)

        captured_at = time.perf_counter()
        with self.semantic_lock:
            self._latest_semantic = CameraFrame(
                int(image.frame), captured_at, visual
            )
            self._latest_drivable = CameraFrame(
                int(image.frame), captured_at, drivable_view
            )
            self._latest_drivable_mask = CameraFrame(
                int(image.frame), captured_at, drivable_mask
            )

    def _alt_callback(self, image):
        frame = self._copy_bgra_bgr(image)
        packet = CameraFrame(int(image.frame), time.perf_counter(), frame)
        with self.alt_lock:
            self._latest_alt = packet

    def snapshot(
        self,
    ) -> tuple[
        Optional[CameraFrame],
        Optional[CameraFrame],
        Optional[CameraFrame],
    ]:
        with self.rgb_lock, self.semantic_lock, self.alt_lock:
            return self._latest_rgb, self._latest_semantic, self._latest_alt

    def get_latest_rgb(self):
        packet, _, _ = self.snapshot()
        return packet.bgr if packet is not None else None

    def get_latest_semantic(self):
        _, packet, _ = self.snapshot()
        return packet.bgr if packet is not None else None

    def get_latest_drivable(self):
        with self.semantic_lock:
            return self._latest_drivable.bgr if self._latest_drivable is not None else None

    def get_latest_drivable_mask(self):
        with self.semantic_lock:
            return (
                self._latest_drivable_mask.bgr
                if self._latest_drivable_mask is not None
                else None
            )

    def get_latest_alt(self):
        _, _, packet = self.snapshot()
        return packet.bgr if packet is not None else None

    def cleanup(self):
        for attr in ("rgb_camera", "semantic_camera", "alt_camera"):
            actor = getattr(self, attr)
            if actor is None:
                continue
            try:
                actor.stop()
                actor.destroy()
            except Exception:
                pass
            setattr(self, attr, None)
