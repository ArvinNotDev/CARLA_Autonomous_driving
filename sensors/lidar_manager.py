from __future__ import annotations

import math
import threading
import time
from typing import Optional

import carla
import numpy as np

import config_city as conf


class LidarManager:
    """Collect optional roof LiDAR sweeps for radar display and obstacle checks."""

    def __init__(self, world, vehicle) -> None:
        self.world = world
        self.vehicle = vehicle
        self.sensor = None
        self.enabled = False

        self._lock = threading.Lock()
        self._scan_chunks: list[np.ndarray] = []
        self._latest_complete_scan: Optional[np.ndarray] = None  # downsampled GUI scan
        self._latest_complete_scan_raw: Optional[np.ndarray] = None  # full-resolution control scan
        self._latest_complete_received_at: Optional[float] = None
        self._scan_started_at: Optional[float] = None
        self._scan_reference_inverse: Optional[np.ndarray] = None
        self._road_height_below_sensor_m = 2.0
        self._heading_deg = 0.0

    def start(self) -> None:
        if self.sensor is not None:
            self.enabled = True
            return

        blueprint = self.world.get_blueprint_library().find(
            "sensor.lidar.ray_cast"
        )
        lidar_range_m = min(
            float(conf.LIDAR_MAX_RANGE_M),
            max(1.0, float(conf.LIDAR_RANGE_M)),
        )
        attributes = {
            "channels": int(conf.LIDAR_CHANNELS),
            "points_per_second": int(conf.LIDAR_POINTS_PER_SECOND),
            "rotation_frequency": float(conf.LIDAR_ROTATION_FREQUENCY_HZ),
            "horizontal_fov": float(conf.LIDAR_HORIZONTAL_FOV_DEG),
            "upper_fov": float(conf.LIDAR_UPPER_FOV_DEG),
            "lower_fov": float(conf.LIDAR_LOWER_FOV_DEG),
            "range": lidar_range_m,
            "noise_stddev": float(conf.LIDAR_NOISE_STDDEV_M),
            # Capture every simulation step to assemble one rotation locally.
            "sensor_tick": 0.0,
        }
        for name, value in attributes.items():
            blueprint.set_attribute(name, str(value))

        bounds = self.vehicle.bounding_box
        mount_x = float(bounds.location.x)
        mount_y = float(bounds.location.y)
        mount_z = float(
            bounds.location.z + bounds.extent.z + conf.LIDAR_ROOF_MARGIN_M
        )
        self._road_height_below_sensor_m = max(
            0.1,
            mount_z - float(bounds.location.z - bounds.extent.z),
        )
        transform = carla.Transform(
            carla.Location(x=mount_x, y=mount_y, z=mount_z),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        )
        sensor = self.world.spawn_actor(
            blueprint,
            transform,
            attach_to=self.vehicle,
        )
        self.sensor = sensor
        self.enabled = True
        try:
            sensor.listen(self._on_measurement)
        except Exception:
            self.enabled = False
            self.sensor = None
            try:
                sensor.destroy()
            except Exception:
                pass
            raise

    @staticmethod
    def _sample_for_display(points: np.ndarray) -> np.ndarray:
        maximum = max(1, int(conf.LIDAR_MAX_DISPLAY_POINTS))
        if len(points) > maximum:
            stride = int(math.ceil(len(points) / maximum))
            points = points[::stride]
        points = np.ascontiguousarray(points, dtype=np.float32)
        points.setflags(write=False)
        return points

    def _on_measurement(self, measurement) -> None:
        if not self.enabled:
            return
        raw = np.frombuffer(measurement.raw_data, dtype=np.float32)
        if raw.size % 4:
            return

        xyz = raw.reshape(-1, 4)[:, :3]
        try:
            matrix = np.asarray(
                measurement.transform.get_matrix(), dtype=np.float32
            )
            heading_deg = float(measurement.transform.rotation.yaw) % 360.0
            timestamp = float(measurement.timestamp)
        except (AttributeError, TypeError, ValueError):
            return

        with self._lock:
            if not self.enabled:
                return
            if self._scan_started_at is None:
                self._scan_started_at = timestamp
                self._scan_reference_inverse = np.linalg.inv(matrix).astype(
                    np.float32, copy=False
                )
            relative_matrix = self._scan_reference_inverse @ matrix

        reference_xyz = (
            xyz @ relative_matrix[:3, :3].T + relative_matrix[:3, 3]
        )
        points = np.column_stack(
            (
                reference_xyz[:, 0],
                reference_xyz[:, 1],
                reference_xyz[:, 2] + self._road_height_below_sensor_m,
            )
        ).astype(np.float32, copy=False)

        with self._lock:
            if not self.enabled:
                return
            revolution_seconds = 1.0 / float(conf.LIDAR_ROTATION_FREQUENCY_HZ)
            completed_revolution = (
                timestamp - self._scan_started_at >= revolution_seconds * 0.95
            )
            self._heading_deg = heading_deg

            if completed_revolution and self._scan_chunks:
                raw_complete_scan = np.ascontiguousarray(
                    np.concatenate((*self._scan_chunks, points), axis=0),
                    dtype=np.float32,
                )
                raw_complete_scan.setflags(write=False)
                self._latest_complete_scan_raw = raw_complete_scan
                self._latest_complete_scan = self._sample_for_display(raw_complete_scan)
                self._latest_complete_received_at = time.monotonic()
                self._scan_chunks = []
                self._scan_started_at = timestamp
                self._scan_reference_inverse = np.linalg.inv(matrix).astype(
                    np.float32, copy=False
                )
            else:
                self._scan_chunks.append(points)

    def snapshot(self) -> tuple[Optional[np.ndarray], float, bool]:
        with self._lock:
            if self._latest_complete_scan is not None:
                return self._latest_complete_scan, self._heading_deg, True
            if not self._scan_chunks:
                return None, self._heading_deg, False
            partial = self._sample_for_display(
                np.concatenate(self._scan_chunks, axis=0)
            )
            return partial, self._heading_deg, False

    def complete_snapshot(self) -> tuple[Optional[np.ndarray], Optional[float]]:
        """Return the full-resolution 360-degree scan and its monotonic receive time.

        The GUI consumes a separately downsampled copy through snapshot(); recovery
        uses the complete scan so small objects are not lost during display sampling.
        """
        with self._lock:
            return self._latest_complete_scan_raw, self._latest_complete_received_at

    def set_enabled(self, enabled: bool) -> None:
        """Start or fully stop the CARLA LiDAR actor."""
        if bool(enabled):
            self.start()
        else:
            self.cleanup()

    def cleanup(self) -> None:
        self.enabled = False
        sensor = self.sensor
        self.sensor = None
        if sensor is not None:
            try:
                sensor.stop()
            except Exception:
                pass
            try:
                sensor.destroy()
            except Exception:
                pass

        with self._lock:
            self._scan_chunks.clear()
            self._latest_complete_scan = None
            self._latest_complete_scan_raw = None
            self._latest_complete_received_at = None
            self._scan_started_at = None
            self._scan_reference_inverse = None
            self._heading_deg = 0.0
