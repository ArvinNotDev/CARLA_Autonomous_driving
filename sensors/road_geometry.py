from __future__ import annotations

import numpy as np


class RoadGeometryAnalyzer:
    """Extract only the two visible road-edge polylines from one LiDAR sweep."""

    def __init__(self) -> None:
        self.x_min_m = 3.0
        self.x_max_m = 35.0
        self.y_limit_m = 12.0
        self.station_step_m = 1.0
        self.station_half_width_m = 0.5
        self.lateral_resolution_m = 0.25
        self.road_start_distance_m = 1.25
        self.road_surface_tolerance_m = 0.22
        self.edge_step_m = 0.13
        self.min_ground_height_m = -4.0
        self.max_ground_height_m = 4.0
        self._y_count = int(
            np.ceil((2.0 * self.y_limit_m) / self.lateral_resolution_m)
        )
        self._station_positions = np.arange(
            self.x_min_m + 1.0,
            self.x_max_m - 1.0,
            self.station_step_m,
            dtype=np.float32,
        )

    def analyze(self, points: np.ndarray | None) -> dict[str, np.ndarray]:
        empty = self._empty_result()
        if points is None:
            return empty

        cloud = np.asarray(points, dtype=np.float32)
        if cloud.ndim != 2 or cloud.shape[1] < 3 or cloud.size == 0:
            return empty

        x, y, z = cloud[:, 0], cloud[:, 1], cloud[:, 2]
        valid = (
            np.isfinite(x)
            & np.isfinite(y)
            & np.isfinite(z)
            & (x >= self.x_min_m)
            & (x <= self.x_max_m)
            & (np.abs(y) <= self.y_limit_m)
            & (z >= self.min_ground_height_m)
            & (z <= self.max_ground_height_m)
        )
        if np.count_nonzero(valid) < 100:
            return empty

        x, y, z = x[valid], y[valid], z[valid]
        station_index = np.floor(
            (x - (self.x_min_m + 1.0 - self.station_half_width_m))
            / self.station_step_m
        ).astype(np.int32)
        lateral_index = np.floor(
            (y + self.y_limit_m) / self.lateral_resolution_m
        ).astype(np.int32)

        station_count = len(self._station_positions)
        in_grid = (
            (station_index >= 0)
            & (station_index < station_count)
            & (lateral_index >= 0)
            & (lateral_index < self._y_count)
        )
        station_index = station_index[in_grid]
        lateral_index = lateral_index[in_grid]
        z = z[in_grid]
        if z.size < 100:
            return empty

        profile = np.full(
            (station_count, self._y_count), np.inf, dtype=np.float32
        )
        np.minimum.at(profile, (station_index, lateral_index), z)
        counts = np.bincount(
            station_index * self._y_count + lateral_index,
            minlength=station_count * self._y_count,
        ).reshape(station_count, self._y_count)
        profile[counts == 0] = np.nan

        y_centers = (
            -self.y_limit_m
            + (np.arange(self._y_count, dtype=np.float32) + 0.5)
            * self.lateral_resolution_m
        )
        left_edges: list[tuple[float, float]] = []
        right_edges: list[tuple[float, float]] = []

        for index, station_x in enumerate(self._station_positions):
            lateral_profile = profile[index]
            center_values = lateral_profile[
                np.isfinite(lateral_profile) & (np.abs(y_centers) <= 1.5)
            ]
            if center_values.size < 2:
                continue

            baseline_z = float(np.median(center_values))
            if center_values.size >= 4:
                spread = float(
                    np.percentile(center_values, 90)
                    - np.percentile(center_values, 10)
                )
                if spread > 0.35:
                    continue

            left = self._trace_edge(
                lateral_profile, y_centers, baseline_z, side=-1
            )
            right = self._trace_edge(
                lateral_profile, y_centers, baseline_z, side=1
            )
            if left is not None:
                left_edges.append((float(station_x), -left))
            if right is not None:
                right_edges.append((float(station_x), right))

        return {
            "left_edge_points": self._readonly_points(left_edges),
            "right_edge_points": self._readonly_points(right_edges),
        }

    def _trace_edge(
        self,
        profile: np.ndarray,
        lateral_centers: np.ndarray,
        baseline_z: float,
        *,
        side: int,
    ) -> float | None:
        indices = np.flatnonzero(
            lateral_centers * side >= self.road_start_distance_m
        )
        if side < 0:
            indices = indices[::-1]
        if indices.size < 5:
            return None

        previous_distance: float | None = None
        previous_z = baseline_z
        missing = 0

        for position, index in enumerate(indices):
            distance = float(lateral_centers[index] * side)
            current_z = float(profile[index])

            if not np.isfinite(current_z):
                missing += 1
                if missing >= 3:
                    return None
                continue

            missing = 0
            if previous_distance is None:
                previous_distance = distance
                previous_z = current_z
                continue

            local_step = abs(current_z - previous_z)
            total_offset = abs(current_z - baseline_z)
            has_transition = (
                local_step >= self.edge_step_m
                or (
                    total_offset > self.road_surface_tolerance_m
                    and local_step >= 0.08
                )
            )
            if has_transition and self._stable_after_step(
                profile, indices, position, current_z
            ):
                return float((previous_distance + distance) * 0.5)

            previous_distance = distance
            previous_z = current_z

        return None

    @staticmethod
    def _stable_after_step(
        profile: np.ndarray,
        ordered_indices: np.ndarray,
        position: int,
        first_height: float,
    ) -> bool:
        next_indices = ordered_indices[position + 1 : position + 5]
        if next_indices.size < 3:
            return False
        values = profile[next_indices]
        values = values[np.isfinite(values)]
        if values.size < 3:
            return False
        return abs(float(np.median(values[:3])) - first_height) <= 0.16

    @staticmethod
    def _readonly_points(
        points: list[tuple[float, float]],
    ) -> np.ndarray:
        result = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        result.setflags(write=False)
        return result

    def _empty_result(self) -> dict[str, np.ndarray]:
        empty = np.empty((0, 2), dtype=np.float32)
        empty.setflags(write=False)
        return {
            "left_edge_points": empty,
            "right_edge_points": empty,
        }
