from __future__ import annotations

import numpy as np

import config_city as conf


class RoadGeometryAnalyzer:
    """Extract left, right, and front road-edge polylines from one LiDAR sweep."""

    _SETTING_DEFAULTS = {
        "LIDAR_BOUNDARY_ENABLED": True,
        "LIDAR_BOUNDARY_MIN_FORWARD_M": 0.5,
        "LIDAR_BOUNDARY_FORWARD_RANGE_M": 80.0,
        "LIDAR_BOUNDARY_SIDE_RANGE_M": 20.0,
        "LIDAR_BOUNDARY_MIN_SIDE_M": 0.75,
        "LIDAR_BOUNDARY_HEIGHT_STEP_M": 0.10,
        "LIDAR_BOUNDARY_SURFACE_TOLERANCE_M": 0.22,
        "LIDAR_BOUNDARY_MIN_FRONT_SPAN_M": 3.0,
        "LIDAR_BOUNDARY_MIN_FRONT_POINTS": 8,
    }

    def __init__(self) -> None:
        self.station_step_m = 0.75
        self.lateral_resolution_m = 0.25
        self.min_ground_height_m = -4.0
        self.max_ground_height_m = 4.0
        self.update_settings({
            key: getattr(conf, key, default)
            for key, default in self._SETTING_DEFAULTS.items()
        })

    def update_settings(self, values: dict) -> None:
        """Apply live GUI settings and rebuild the compact analysis grid."""
        current = {
            "LIDAR_BOUNDARY_ENABLED": self.enabled if hasattr(self, "enabled") else True,
            "LIDAR_BOUNDARY_MIN_FORWARD_M": getattr(self, "x_min_m", 0.5),
            "LIDAR_BOUNDARY_FORWARD_RANGE_M": getattr(self, "x_max_m", 80.0),
            "LIDAR_BOUNDARY_SIDE_RANGE_M": getattr(self, "y_limit_m", 20.0),
            "LIDAR_BOUNDARY_MIN_SIDE_M": getattr(self, "road_start_distance_m", 0.75),
            "LIDAR_BOUNDARY_HEIGHT_STEP_M": getattr(self, "edge_step_m", 0.10),
            "LIDAR_BOUNDARY_SURFACE_TOLERANCE_M": getattr(self, "road_surface_tolerance_m", 0.22),
            "LIDAR_BOUNDARY_MIN_FRONT_SPAN_M": getattr(self, "front_min_span_m", 3.0),
            "LIDAR_BOUNDARY_MIN_FRONT_POINTS": getattr(self, "front_min_points", 8),
        }
        current.update(values or {})

        self.enabled = bool(current["LIDAR_BOUNDARY_ENABLED"])
        self.x_min_m = max(0.1, min(10.0, float(current["LIDAR_BOUNDARY_MIN_FORWARD_M"])))
        sensor_range_m = min(
            float(getattr(conf, "LIDAR_MAX_RANGE_M", 200.0)),
            max(1.0, float(getattr(conf, "LIDAR_RANGE_M", 100.0))),
        )
        self.x_max_m = max(
            self.x_min_m + 8.0,
            min(sensor_range_m, float(current["LIDAR_BOUNDARY_FORWARD_RANGE_M"])),
        )
        self.y_limit_m = max(3.0, min(50.0, float(current["LIDAR_BOUNDARY_SIDE_RANGE_M"])))
        self.road_start_distance_m = max(
            0.1, min(self.y_limit_m - 0.5, float(current["LIDAR_BOUNDARY_MIN_SIDE_M"]))
        )
        self.edge_step_m = max(0.03, min(1.0, float(current["LIDAR_BOUNDARY_HEIGHT_STEP_M"])))
        self.road_surface_tolerance_m = max(
            0.05, min(1.5, float(current["LIDAR_BOUNDARY_SURFACE_TOLERANCE_M"]))
        )
        self.front_min_span_m = max(
            1.0, min(self.y_limit_m * 2.0, float(current["LIDAR_BOUNDARY_MIN_FRONT_SPAN_M"]))
        )
        self.front_min_points = max(
            3, min(128, int(current["LIDAR_BOUNDARY_MIN_FRONT_POINTS"]))
        )

        self._y_count = int(np.ceil((2.0 * self.y_limit_m) / self.lateral_resolution_m))
        station_count = max(
            1, int(np.ceil((self.x_max_m - self.x_min_m) / self.station_step_m))
        )
        self._station_positions = (
            self.x_min_m
            + (np.arange(station_count, dtype=np.float32) + 0.5) * self.station_step_m
        )

    def analyze(self, points: np.ndarray | None) -> dict[str, np.ndarray]:
        empty = self._empty_result()
        if not self.enabled or points is None:
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
            (x - self.x_min_m) / self.station_step_m
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

            left = self._trace_side_edge(
                lateral_profile, y_centers, baseline_z, side=-1
            )
            right = self._trace_side_edge(
                lateral_profile, y_centers, baseline_z, side=1
            )
            if left is not None:
                left_edges.append((float(station_x), -left))
            if right is not None:
                right_edges.append((float(station_x), right))

        front_edges = self._trace_front_edge(profile, y_centers)
        return {
            "left_edge_points": self._readonly_points(left_edges),
            "right_edge_points": self._readonly_points(right_edges),
            "front_edge_points": self._readonly_points(front_edges),
        }

    def _trace_side_edge(
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
                if missing >= 6:
                    return None
                continue

            missing = 0
            if previous_distance is None:
                if (
                    abs(current_z - baseline_z) > self.road_surface_tolerance_m
                    and self._stable_after_step(
                        profile, indices, position, current_z
                    )
                ):
                    return float(
                        max(
                            self.road_start_distance_m,
                            distance - self.lateral_resolution_m * 0.5,
                        )
                    )
                previous_distance = distance
                previous_z = current_z
                continue

            local_step = abs(current_z - previous_z)
            total_offset = abs(current_z - baseline_z)
            has_transition = (
                local_step >= self.edge_step_m
                or (
                    total_offset > self.road_surface_tolerance_m
                    and local_step >= max(0.05, self.edge_step_m * 0.5)
                )
            )
            if has_transition and self._stable_after_step(
                profile, indices, position, current_z
            ):
                return float((previous_distance + distance) * 0.5)

            previous_distance = distance
            previous_z = current_z

        return None

    def _trace_front_edge(
        self,
        profile: np.ndarray,
        lateral_centers: np.ndarray,
    ) -> list[tuple[float, float]]:
        """Find a coherent height transition across the forward LiDAR view."""
        candidates: list[tuple[float, float]] = []

        for lateral_index, lateral_y in enumerate(lateral_centers):
            line_profile = profile[:, lateral_index]
            valid_indices = np.flatnonzero(np.isfinite(line_profile))
            if valid_indices.size < 6:
                continue

            baseline_indices = valid_indices[:2]
            baseline_values = line_profile[baseline_indices]
            if float(np.ptp(baseline_values)) > max(
                0.35, self.road_surface_tolerance_m * 1.5
            ):
                continue
            baseline_z = float(np.median(baseline_values))
            previous_index = int(valid_indices[1])
            previous_z = float(line_profile[previous_index])

            for position in range(2, valid_indices.size):
                current_index = int(valid_indices[position])
                current_z = float(line_profile[current_index])
                if current_index - previous_index > 2:
                    previous_index = current_index
                    previous_z = current_z
                    continue

                local_step = abs(current_z - previous_z)
                total_offset = abs(current_z - baseline_z)
                has_transition = (
                    local_step >= self.edge_step_m
                    or (
                        total_offset > self.road_surface_tolerance_m
                        and local_step >= max(0.05, self.edge_step_m * 0.5)
                    )
                )
                if has_transition and self._stable_forward_step(
                    line_profile, valid_indices, position, current_z
                ):
                    edge_x = (
                        float(self._station_positions[previous_index])
                        + float(self._station_positions[current_index])
                    ) * 0.5
                    candidates.append((edge_x, float(lateral_y)))
                    break

                previous_index = current_index
                previous_z = current_z

        if len(candidates) < self.front_min_points:
            return []

        groups: list[list[tuple[float, float]]] = []
        max_y_gap = self.lateral_resolution_m * 2.5
        max_x_step = max(1.25, self.station_step_m * 2.0)
        for candidate in candidates:
            if not groups:
                groups.append([candidate])
                continue
            previous_x, previous_y = groups[-1][-1]
            if (
                candidate[1] - previous_y <= max_y_gap
                and abs(candidate[0] - previous_x) <= max_x_step
            ):
                groups[-1].append(candidate)
            else:
                groups.append([candidate])

        eligible = [
            group for group in groups
            if len(group) >= self.front_min_points
            and group[-1][1] - group[0][1] >= self.front_min_span_m
        ]
        if not eligible:
            return []

        selected = max(
            eligible,
            key=lambda group: (
                group[-1][1] - group[0][1],
                len(group),
                -float(np.median([point[0] for point in group])),
            ),
        )
        smoothed_x = []
        for index, point in enumerate(selected):
            start = max(0, index - 1)
            end = min(len(selected), index + 2)
            smoothed_x.append(
                float(np.median([item[0] for item in selected[start:end]]))
            )
        return [
            (smoothed_x[index], point[1])
            for index, point in enumerate(selected)
        ]

    def _stable_after_step(
        self,
        profile: np.ndarray,
        ordered_indices: np.ndarray,
        position: int,
        first_height: float,
    ) -> bool:
        next_indices = ordered_indices[position + 1 : position + 5]
        if next_indices.size < 3:
            return False
        if np.any(np.abs(np.diff(next_indices[:3])) > 2):
            return False
        values = profile[next_indices]
        values = values[np.isfinite(values)]
        if values.size < 3:
            return False
        tolerance = max(0.16, self.edge_step_m * 1.5)
        return abs(float(np.median(values[:3])) - first_height) <= tolerance

    def _stable_forward_step(
        self,
        profile: np.ndarray,
        valid_indices: np.ndarray,
        position: int,
        first_height: float,
    ) -> bool:
        next_indices = valid_indices[position + 1 : position + 4]
        if next_indices.size < 3:
            return False
        if np.any(np.diff(next_indices) > 2):
            return False
        values = profile[next_indices]
        return abs(float(np.median(values)) - first_height) <= max(
            0.16, self.edge_step_m * 1.5
        )

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
            "front_edge_points": empty,
        }
