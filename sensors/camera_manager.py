import threading

import carla
import numpy as np

import config_city as conf


class CameraManager:
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle

        self.rgb_camera = None
        self.semantic_camera = None
        self.alt_camera = None

        self.latest_rgb = None
        self.latest_semantic = None
        self.latest_drivable = None
        self.latest_drivable_mask = None
        self.latest_alt = None

        self.rgb_lock = threading.Lock()
        self.semantic_lock = threading.Lock()
        self.alt_lock = threading.Lock()

    # =====================================================
    # START CAMERAS
    # =====================================================
    def start(self):
        bp_lib = self.world.get_blueprint_library()

        # -------------------------
        # Common transform
        # -------------------------
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

        # =====================================================
        # RGB CAMERA
        # =====================================================
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

        # =====================================================
        # SEMANTIC SEGMENTATION CAMERA
        # =====================================================
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

        # =====================================================
        # ALT CAMERA
        # =====================================================
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

    # =====================================================
    # RGB CALLBACK
    # =====================================================
    def _rgb_callback(self, image):
        img = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = img.reshape((image.height, image.width, 4))
        frame = img[:, :, :3].copy()

        with self.rgb_lock:
            self.latest_rgb = frame

    # =====================================================
    # SEMANTIC CALLBACK
    # =====================================================
    def _semantic_callback(self, image):
        # Keep raw data for mask extraction before any visualization conversion
        raw = np.frombuffer(image.raw_data, dtype=np.uint8)
        raw = raw.reshape((image.height, image.width, 4))

        # CARLA semantic class id is usually in channel 2
        class_map = raw[:, :, 2]

        # Road/drivable class is usually 1
        drivable_mask = (class_map == 1).astype(np.uint8) * 255

        # Convert only for visualization
        image.convert(carla.ColorConverter.CityScapesPalette)
        vis = np.frombuffer(image.raw_data, dtype=np.uint8)
        vis = vis.reshape((image.height, image.width, 4))
        visual = vis[:, :, :3].copy()

        # Green overlay for drivable area
        drivable_view = np.zeros_like(visual)
        drivable_view[:, :, 1] = drivable_mask

        with self.semantic_lock:
            self.latest_semantic = visual
            self.latest_drivable = drivable_view
            self.latest_drivable_mask = drivable_mask

    # =====================================================
    # ALT CALLBACK
    # =====================================================
    def _alt_callback(self, image):
        img = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = img.reshape((image.height, image.width, 4))
        frame = img[:, :, :3].copy()

        with self.alt_lock:
            self.latest_alt = frame

    # =====================================================
    # GETTERS
    # =====================================================
    def get_latest_rgb(self):
        with self.rgb_lock:
            if self.latest_rgb is None:
                return None
            return self.latest_rgb.copy()

    def get_latest_semantic(self):
        with self.semantic_lock:
            if self.latest_semantic is None:
                return None
            return self.latest_semantic.copy()

    def get_latest_drivable(self):
        with self.semantic_lock:
            if self.latest_drivable is None:
                return None
            return self.latest_drivable.copy()

    def get_latest_drivable_mask(self):
        with self.semantic_lock:
            if self.latest_drivable_mask is None:
                return None
            return self.latest_drivable_mask.copy()

    def get_latest_alt(self):
        with self.alt_lock:
            if self.latest_alt is None:
                return None
            return self.latest_alt.copy()

    # =====================================================
    # CLEANUP
    # =====================================================
    def cleanup(self):
        if self.rgb_camera is not None:
            try:
                self.rgb_camera.stop()
                self.rgb_camera.destroy()
            except Exception:
                pass
            self.rgb_camera = None

        if self.semantic_camera is not None:
            try:
                self.semantic_camera.stop()
                self.semantic_camera.destroy()
            except Exception:
                pass
            self.semantic_camera = None

        if self.alt_camera is not None:
            try:
                self.alt_camera.stop()
                self.alt_camera.destroy()
            except Exception:
                pass
            self.alt_camera = None