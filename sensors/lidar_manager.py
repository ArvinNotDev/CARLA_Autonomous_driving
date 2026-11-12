from __future__ import annotations

import math
import threading
from typing import Optional

import carla
import numpy as np

import config_city as conf


class LidarManager:
    """Own a visualization-only roof LiDAR and publish one full scan at a time."""

    def __init__(self, world, vehicle) -> None:
        self.world = world
        self.vehicle = vehicle
        self.sensor = None

        self._lock = threading.Lock()
        self._scan_chunks: list[np.ndarray] = []
        self._latest_complete_scan: Optional[np.ndarray] = None
        self._scan_started_at: Optional[float] = None
        self._road_height_below_sensor_m = 2.0
        self._heading_deg = 0.0

    def start(self) -> None:
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
            # Capture every simulation step so the client can accumulate each
            # 10 Hz rotation from its per-step partial sweeps.
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
        self.sensor = self.world.spawn_actor(
            blueprint,
            transform,
            attach_to=self.vehicle,
        )
        self.sensor.listen(self._on_measurement)

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
        raw = np.frombuffer(measurement.raw_data, dtype=np.float32)
        if raw.size % 4:
            return

        xyz = raw.reshape(-1, 4)[:, :3]
        try:
            matrix = np.asarray(measurement.transform.get_matrix(), dtype=np.float32)
            world_xyz = xyz @ matrix[:3, :3].T + matrix[:3, 3]
            road_z = (
                float(measurement.transform.location.z)
                - self._road_height_below_sensor_m * float(matrix[2, 2])
            )
            height_above_road = world_xyz[:, 2] - road_z
            points = np.column_stack((xyz[:, 0], xyz[:, 1], height_above_road))
            heading_deg = float(measurement.transform.rotation.yaw) % 360.0
            timestamp = float(measurement.timestamp)
        except (AttributeError, TypeError, ValueError):
            return

        points = points.astype(np.float32, copy=False)
        with self._lock:
            if self._scan_started_at is None:
                self._scan_started_at = timestamp
            revolution_seconds = 1.0 / float(conf.LIDAR_ROTATION_FREQUENCY_HZ)
            completed_revolution = (
                timestamp - self._scan_started_at >= revolution_seconds * 0.95
            )
            self._heading_deg = heading_deg

            if completed_revolution and self._scan_chunks:
                complete_scan = self._sample_for_display(
                    np.concatenate((*self._scan_chunks, points), axis=0)
                )
                self._latest_complete_scan = complete_scan
                self._scan_chunks = []
                self._scan_started_at = timestamp
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

    def cleanup(self) -> None:
        sensor = self.sensor
        self.sensor = None
        if sensor is None:
            return
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
            self._scan_started_at = None
