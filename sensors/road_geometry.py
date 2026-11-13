from __future__ import annotations

import numpy as np


class RoadGeometryAnalyzer:
    """Reconstruct locally visible road and shoulder surfaces from LiDAR geometry.

    The detector relies on persistent lateral surface transitions. It does not
    infer material classes, so a coplanar road and shoulder without a visible
    geometric boundary cannot be separated from LiDAR alone.
    """

    def __init__(self) -> None:
        self.resolution_m = 0.25
        self.x_min_m = 3.0
        self.x_max_m = 35.0
        self.y_limit_m = 12.0
        self.station_step_m = 1.0
        self.road_start_distance_m = 1.25
        self.road_surface_tolerance_m = 0.22
        self.edge_step_m = 0.13
        self.min_ground_height_m = -4.0
        self.max_ground_height_m = 4.0
        self.max_shoulder_width_m = 5.0

    def analyze(self, points: np.ndarray | None) -> dict:
        empty = self._empty_result()
        if points is None:
            return empty

        cloud = np.asarray(points, dtype=np.float32)
        if cloud.ndim != 2 or cloud.shape[1] < 3 or cloud.size == 0:
            return empty

        finite = np.isfinite(cloud[:, :3]).all(axis=1)
        in_roi = (
            finite
            & (cloud[:, 0] >= self.x_min_m)
            & (cloud[:, 0] <= self.x_max_m)
            & (np.abs(cloud[:, 1]) <= self.y_limit_m)
            & (cloud[:, 2] >= self.min_ground_height_m)
            & (cloud[:, 2] <= self.max_ground_height_m)
        )
        cloud = cloud[in_roi, :3]
        if cloud.shape[0] < 100:
            empty["status"] = "insufficient_points"
            empty["ground_point_count"] = int(cloud.shape[0])
            return empty

        x_count = int(np.ceil((self.x_max_m - self.x_min_m) / self.resolution_m))
        y_count = int(np.ceil((2.0 * self.y_limit_m) / self.resolution_m))
        x_centers = self.x_min_m + (
            np.arange(x_count, dtype=np.float32) + 0.5
        ) * self.resolution_m
        y_centers = -self.y_limit_m + (
            np.arange(y_count, dtype=np.float32) + 0.5
        ) * self.resolution_m

        x_indices = np.floor(
            (cloud[:, 0] - self.x_min_m) / self.resolution_m
        ).astype(np.int32)
        y_indices = np.floor(
            (cloud[:, 1] + self.y_limit_m) / self.resolution_m
        ).astype(np.int32)
        valid_indices = (
            (x_indices >= 0)
            & (x_indices < x_count)
            & (y_indices >= 0)
            & (y_indices < y_count)
        )
        x_indices = x_indices[valid_indices]
        y_indices = y_indices[valid_indices]
        ground_z = cloud[valid_indices, 2]

        height_grid = np.full((x_count, y_count), np.inf, dtype=np.float32)
        point_counts = np.zeros((x_count, y_count), dtype=np.uint16)
        np.minimum.at(height_grid, (x_indices, y_indices), ground_z)
        np.add.at(point_counts, (x_indices, y_indices), 1)
        height_grid[point_counts == 0] = np.nan

        occupied = point_counts > 0
        connected_occupancy = self._fill_single_cell_holes(occupied)
        surface_labels = np.zeros((x_count, y_count), dtype=np.uint8)
        stations: list[dict] = []
        left_edges: list[tuple[float, float]] = []
        right_edges: list[tuple[float, float]] = []
        left_shoulder_edges: list[tuple[float, float]] = []
        right_shoulder_edges: list[tuple[float, float]] = []

        first_station = self.x_min_m + 1.0
        last_station = self.x_max_m - 1.0
        station_positions = np.arange(
            first_station,
            last_station,
            self.station_step_m,
            dtype=np.float32,
        )

        for station_x in station_positions:
            row_start = max(
                0,
                int(np.floor(
                    (station_x - 0.5 - self.x_min_m) / self.resolution_m
                )),
            )
            row_end = min(
                x_count,
                int(np.ceil(
                    (station_x + 0.5 - self.x_min_m) / self.resolution_m
                )),
            )
            if row_end <= row_start:
                continue

            window_z = height_grid[row_start:row_end]
            window_counts = point_counts[row_start:row_end]
            lateral_profile = np.full(y_count, np.nan, dtype=np.float32)
            lateral_counts = window_counts.sum(axis=0)

            for lateral_index in np.flatnonzero(lateral_counts > 0):
                values = window_z[:, lateral_index]
                values = values[np.isfinite(values)]
                if values.size:
                    lateral_profile[lateral_index] = float(np.median(values))

            center_values = lateral_profile[
                np.isfinite(lateral_profile) & (np.abs(y_centers) <= 1.5)
            ]
            if center_values.size < 2:
                center_values = lateral_profile[
                    np.isfinite(lateral_profile) & (np.abs(y_centers) <= 2.5)
                ]
            if center_values.size == 0:
                continue

            baseline_z = float(np.median(center_values))
            if center_values.size >= 4:
                spread = float(
                    np.percentile(center_values, 90)
                    - np.percentile(center_values, 10)
                )
                if spread > 0.35:
                    continue

            left = self._trace_side(
                lateral_profile, y_centers, baseline_z, side=-1
            )
            right = self._trace_side(
                lateral_profile, y_centers, baseline_z, side=1
            )
            if left is None and right is None:
                continue

            station = {
                "x": float(station_x),
                "baseline_z": baseline_z,
                "left": left,
                "right": right,
            }
            stations.append(station)

            if left is not None:
                left_edges.append((float(station_x), -left["edge_m"]))
                if left["outer_m"] is not None:
                    left_shoulder_edges.append(
                        (float(station_x), -left["outer_m"])
                    )
            if right is not None:
                right_edges.append((float(station_x), right["edge_m"]))
                if right["outer_m"] is not None:
                    right_shoulder_edges.append(
                        (float(station_x), right["outer_m"])
                    )

            y_grid = y_centers
            valid_cells = connected_occupancy[row_start:row_end]
            if left is not None and right is not None:
                road_mask = (
                    (y_grid >= -left["edge_m"])
                    & (y_grid <= right["edge_m"])
                )
                surface_labels[row_start:row_end, :] = np.where(
                    valid_cells & road_mask[None, :],
                    1,
                    surface_labels[row_start:row_end, :],
                )

            if left is not None and left["outer_m"] is not None:
                shoulder_mask = (
                    (y_grid >= -left["outer_m"])
                    & (y_grid < -left["edge_m"])
                )
                surface_labels[row_start:row_end, :] = np.where(
                    valid_cells & shoulder_mask[None, :],
                    2,
                    surface_labels[row_start:row_end, :],
                )

            if right is not None and right["outer_m"] is not None:
                shoulder_mask = (
                    (y_grid > right["edge_m"])
                    & (y_grid <= right["outer_m"])
                )
                surface_labels[row_start:row_end, :] = np.where(
                    valid_cells & shoulder_mask[None, :],
                    2,
                    surface_labels[row_start:row_end, :],
                )

        both_sides = [
            station
            for station in stations
            if station["left"] is not None and station["right"] is not None
        ]
        road_widths = [
            station["left"]["edge_m"] + station["right"]["edge_m"]
            for station in both_sides
        ]
        left_widths = [
            station["left"]["width_m"]
            for station in stations
            if station["left"] is not None
            and station["left"]["width_m"] is not None
        ]
        right_widths = [
            station["right"]["width_m"]
            for station in stations
            if station["right"] is not None
            and station["right"]["width_m"] is not None
        ]
        left_heights = [
            station["left"]["height_m"]
            for station in stations
            if station["left"] is not None
            and station["left"]["height_m"] is not None
        ]
        right_heights = [
            station["right"]["height_m"]
            for station in stations
            if station["right"] is not None
            and station["right"]["height_m"] is not None
        ]

        required_road_stations = max(
            6, int(np.ceil(len(station_positions) * 0.20))
        )
        road_width_candidate = (
            float(np.median(road_widths)) if road_widths else None
        )
        road_width_spread = (
            float(np.percentile(road_widths, 90) - np.percentile(road_widths, 10))
            if len(road_widths) >= 4
            else None
        )
        road_width = (
            road_width_candidate
            if len(road_widths) >= required_road_stations
            and road_width_candidate is not None
            and 2.5 <= road_width_candidate <= 24.0
            and road_width_spread is not None
            and road_width_spread <= 2.0
            else None
        )
        required_side_stations = max(
            4, int(np.ceil(len(station_positions) * 0.12))
        )
        left_width = (
            float(np.median(left_widths))
            if len(left_widths) >= required_side_stations
            else None
        )
        right_width = (
            float(np.median(right_widths))
            if len(right_widths) >= required_side_stations
            else None
        )
        left_height = (
            float(np.median(left_heights))
            if len(left_heights) >= required_side_stations
            else None
        )
        right_height = (
            float(np.median(right_heights))
            if len(right_heights) >= required_side_stations
            else None
        )

        left_length = self._longest_station_run(
            [
                station["x"]
                for station in stations
                if station["left"] is not None
                and station["left"]["width_m"] is not None
            ]
        )
        right_length = self._longest_station_run(
            [
                station["x"]
                for station in stations
                if station["right"] is not None
                and station["right"]["width_m"] is not None
            ]
        )
        left_length = (
            left_length
            if left_length is not None and left_length >= 4.0
            else None
        )
        right_length = (
            right_length
            if right_length is not None and right_length >= 4.0
            else None
        )
        shoulder_lengths = [
            value for value in (left_length, right_length) if value is not None
        ]
        shoulder_length = max(shoulder_lengths) if shoulder_lengths else None

        cell_x, cell_y = np.nonzero(surface_labels)
        if cell_x.size:
            surface_cells = np.column_stack(
                (
                    x_centers[cell_x],
                    y_centers[cell_y],
                    surface_labels[cell_x, cell_y].astype(np.float32),
                )
            ).astype(np.float32, copy=False)
        else:
            surface_cells = np.empty((0, 3), dtype=np.float32)
        surface_cells.setflags(write=False)

        if road_width is not None:
            status = "detected"
        elif stations:
            status = "partial"
        else:
            status = "no_reliable_edge"

        result = {
            "status": status,
            "road_width_m": road_width,
            "shoulder_width_left_m": left_width,
            "shoulder_width_right_m": right_width,
            "shoulder_height_left_m": left_height,
            "shoulder_height_right_m": right_height,
            "shoulder_length_m": shoulder_length,
            "shoulder_length_left_m": left_length,
            "shoulder_length_right_m": right_length,
            "edge_station_count": len(both_sides),
            "station_count": len(station_positions),
            "road_width_confidence": (
                float(len(road_widths) / len(station_positions))
                if len(station_positions) else 0.0
            ),
            "shoulder_left_confidence": float(
                len(left_widths) / len(station_positions)
            ) if len(station_positions) else 0.0,
            "shoulder_right_confidence": float(
                len(right_widths) / len(station_positions)
            ) if len(station_positions) else 0.0,
            "ground_point_count": int(cloud.shape[0]),
            "surface_resolution_m": self.resolution_m,
            "surface_cells": surface_cells,
            "left_edge_points": self._readonly_points(left_edges),
            "right_edge_points": self._readonly_points(right_edges),
            "left_shoulder_outer_points": self._readonly_points(
                left_shoulder_edges
            ),
            "right_shoulder_outer_points": self._readonly_points(
                right_shoulder_edges
            ),
        }
        return result

    def _trace_side(
        self,
        profile: np.ndarray,
        lateral_centers: np.ndarray,
        baseline_z: float,
        *,
        side: int,
    ) -> dict | None:
        indices = np.flatnonzero(lateral_centers * side >= self.road_start_distance_m)
        if side < 0:
            indices = indices[::-1]
        if indices.size < 5:
            return None

        road_last: float | None = None
        previous_z = baseline_z
        missing = 0
        edge_distance: float | None = None
        shoulder_start_position: int | None = None

        for order_position, index in enumerate(indices):
            coordinate = float(lateral_centers[index] * side)
            z_value = float(profile[index])

            if not np.isfinite(z_value):
                missing += 1
                if missing >= 3:
                    return None
                continue

            missing = 0
            if road_last is None:
                if abs(z_value - baseline_z) > self.road_surface_tolerance_m:
                    if self._stable_after_step(
                        profile, indices, order_position, z_value
                    ):
                        edge_distance = max(
                            self.road_start_distance_m,
                            coordinate - self.resolution_m * 0.5,
                        )
                        shoulder_start_position = order_position
                        break
                road_last = coordinate
                previous_z = z_value
                continue

            local_step = abs(z_value - previous_z)
            total_offset = abs(z_value - baseline_z)
            if (
                local_step >= self.edge_step_m
                or (
                    total_offset > self.road_surface_tolerance_m
                    and local_step >= 0.08
                )
            ):
                if self._stable_after_step(
                    profile, indices, order_position, z_value
                ):
                    edge_distance = (road_last + coordinate) * 0.5
                    shoulder_start_position = order_position
                    break
                continue

            road_last = coordinate
            previous_z = z_value

        if edge_distance is None:
            return None

        width, height, outer = self._measure_shoulder(
            profile,
            lateral_centers,
            indices,
            shoulder_start_position,
            edge_distance,
            baseline_z,
        )
        return {
            "edge_m": float(edge_distance),
            "width_m": width,
            "height_m": height,
            "outer_m": outer,
        }

    @staticmethod
    def _stable_after_step(
        profile: np.ndarray,
        ordered_indices: np.ndarray,
        order_position: int,
        first_height: float,
    ) -> bool:
        next_values = []
        for next_position in range(order_position + 1, min(
            len(ordered_indices), order_position + 5
        )):
            value = float(profile[ordered_indices[next_position]])
            if np.isfinite(value):
                next_values.append(value)
        if len(next_values) < 3:
            return False
        return abs(float(np.median(next_values[:3])) - first_height) <= 0.16

    def _measure_shoulder(
        self,
        profile: np.ndarray,
        lateral_centers: np.ndarray,
        ordered_indices: np.ndarray,
        start_position: int | None,
        edge_distance: float,
        baseline_z: float,
    ) -> tuple[float | None, float | None, float | None]:
        if start_position is None:
            return None, None, None

        heights: list[float] = []
        last_coordinate: float | None = None
        gap_count = 0
        reference_height: float | None = None

        for position in range(start_position, len(ordered_indices)):
            index = ordered_indices[position]
            coordinate = float(lateral_centers[index] * np.sign(
                lateral_centers[ordered_indices[0]]
            ))
            coordinate = abs(coordinate)
            distance_from_edge = coordinate - edge_distance
            if distance_from_edge > self.max_shoulder_width_m:
                break

            z_value = float(profile[index])
            if not np.isfinite(z_value):
                gap_count += 1
                if gap_count >= 3:
                    break
                continue

            gap_count = 0
            if abs(z_value - baseline_z) > 1.2:
                break
            if reference_height is not None and abs(
                z_value - reference_height
            ) > 0.18:
                break

            heights.append(z_value)
            reference_height = float(np.median(heights))
            last_coordinate = coordinate

        if len(heights) < 2 or last_coordinate is None:
            return None, None, None

        width = max(
            0.0,
            last_coordinate - edge_distance + self.resolution_m * 0.5,
        )
        if width < 0.45:
            return None, None, None

        shoulder_height = float(np.median(heights) - baseline_z)
        outer_edge = float(edge_distance + width)
        return width, shoulder_height, outer_edge

    def _fill_single_cell_holes(self, occupancy: np.ndarray) -> np.ndarray:
        padded = np.pad(occupancy.astype(np.uint8), 1, mode="constant")
        neighbor_count = np.zeros_like(occupancy, dtype=np.uint8)
        for dx in range(3):
            for dy in range(3):
                if dx == 1 and dy == 1:
                    continue
                neighbor_count += padded[
                    dx : dx + occupancy.shape[0],
                    dy : dy + occupancy.shape[1],
                ]
        return occupancy | (neighbor_count >= 6)

    def _longest_station_run(self, positions: list[float]) -> float | None:
        if not positions:
            return None
        ordered = np.sort(np.asarray(positions, dtype=np.float32))
        longest = self.station_step_m
        current = self.station_step_m
        for difference in np.diff(ordered):
            if difference <= self.station_step_m * 1.6:
                current += float(difference)
            else:
                longest = max(longest, current)
                current = self.station_step_m
        return float(max(longest, current))

    @staticmethod
    def _readonly_points(points: list[tuple[float, float]]) -> np.ndarray:
        array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        array.setflags(write=False)
        return array

    def _empty_result(self) -> dict:
        empty_points = np.empty((0, 2), dtype=np.float32)
        empty_cells = np.empty((0, 3), dtype=np.float32)
        empty_points.setflags(write=False)
        empty_cells.setflags(write=False)
        return {
            "status": "waiting_for_scan",
            "road_width_m": None,
            "shoulder_width_left_m": None,
            "shoulder_width_right_m": None,
            "shoulder_height_left_m": None,
            "shoulder_height_right_m": None,
            "shoulder_length_m": None,
            "shoulder_length_left_m": None,
            "shoulder_length_right_m": None,
            "edge_station_count": 0,
            "station_count": 0,
            "road_width_confidence": 0.0,
            "shoulder_left_confidence": 0.0,
            "shoulder_right_confidence": 0.0,
            "ground_point_count": 0,
            "surface_resolution_m": self.resolution_m,
            "surface_cells": empty_cells,
            "left_edge_points": empty_points,
            "right_edge_points": empty_points,
            "left_shoulder_outer_points": empty_points,
            "right_shoulder_outer_points": empty_points,
        }
