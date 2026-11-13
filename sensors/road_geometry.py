from __future__ import annotations

import numpy as np

import config_city as conf


class RoadGeometryAnalyzer:
    """Track road boundaries using local LiDAR elevation profiles.

    Each cross-section is perpendicular to the tracked road tangent. Edge
    candidates are validated by height discontinuity, spatial support, local
    surface consistency, and continuity with the previously tracked boundary.
    """

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
        self.station_step_m = 0.90
        self.lateral_resolution_m = 0.20
        self.min_ground_height_m = -3.0
        self.max_ground_height_m = 3.5
        self.max_curb_step_m = 0.60
        self.max_cell_vertical_span_m = 0.55
        self.max_edge_track_step_m = 1.35
        self.max_interpolation_gap_m = 6.0
        self._default_half_width_m = 3.0
        self.update_settings(
            {
                key: getattr(conf, key, default)
                for key, default in self._SETTING_DEFAULTS.items()
            }
        )

    def update_settings(self, values: dict) -> None:
        """Apply live GUI settings and rebuild the analysis grid."""
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
        sensor_range_m = min(
            float(getattr(conf, "LIDAR_MAX_RANGE_M", 200.0)),
            max(1.0, float(getattr(conf, "LIDAR_RANGE_M", 100.0))),
        )
        requested_min_forward = float(current["LIDAR_BOUNDARY_MIN_FORWARD_M"])
        self.x_min_m = max(
            0.1, min(10.0, sensor_range_m - 8.0, requested_min_forward)
        )
        self.x_max_m = max(
            self.x_min_m + 8.0,
            min(sensor_range_m, float(current["LIDAR_BOUNDARY_FORWARD_RANGE_M"])),
        )
        self.y_limit_m = max(
            3.0, min(50.0, float(current["LIDAR_BOUNDARY_SIDE_RANGE_M"]))
        )
        self.road_start_distance_m = max(
            0.5,
            min(self.y_limit_m - 1.0, float(current["LIDAR_BOUNDARY_MIN_SIDE_M"])),
        )
        self.edge_step_m = max(
            0.04, min(1.0, float(current["LIDAR_BOUNDARY_HEIGHT_STEP_M"]))
        )
        self.road_surface_tolerance_m = max(
            0.05,
            min(1.5, float(current["LIDAR_BOUNDARY_SURFACE_TOLERANCE_M"])),
        )
        self.front_min_span_m = max(
            1.0,
            min(
                self.y_limit_m * 2.0,
                float(current["LIDAR_BOUNDARY_MIN_FRONT_SPAN_M"]),
            ),
        )
        self.front_min_points = max(
            3, min(128, int(current["LIDAR_BOUNDARY_MIN_FRONT_POINTS"]))
        )

        self._y_count = int(
            np.ceil((2.0 * self.y_limit_m) / self.lateral_resolution_m)
        )
        station_count = max(
            1,
            int(np.ceil((self.x_max_m - self.x_min_m) / self.station_step_m)),
        )
        self._station_positions = (
            self.x_min_m
            + (np.arange(station_count, dtype=np.float32) + 0.5) * self.station_step_m
        )
        self._y_centers = (
            -self.y_limit_m
            + (np.arange(self._y_count, dtype=np.float32) + 0.5)
            * self.lateral_resolution_m
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
            & (x < self.x_max_m)
            & (np.abs(y) <= self.y_limit_m)
            & (z >= self.min_ground_height_m)
            & (z <= self.max_ground_height_m)
        )
        if np.count_nonzero(valid) < 100:
            return empty

        x, y, z = x[valid], y[valid], z[valid]
        sort_order = np.argsort(x)
        x_sorted = x[sort_order]
        y_sorted = y[sort_order]
        z_sorted = z[sort_order]
        cross_section_half_length = 1.8

        left_points: list[tuple[float, float] | None] = []
        right_points: list[tuple[float, float] | None] = []
        center_y = 0.0
        center_velocity = 0.0
        half_width = self._default_half_width_m
        previous_x: float | None = None
        last_observation_x: float | None = None
        paired_edge_observations = 0
        width_confirmed = False

        for station_x in self._station_positions:
            if previous_x is None:
                delta_x = self.station_step_m
                predicted_center = center_y
            else:
                delta_x = float(station_x - previous_x)
                predicted_center = center_y + center_velocity * delta_x

            # Project the local window onto the normal of the tracked road tangent.
            window_start = int(
                np.searchsorted(
                    x_sorted,
                    station_x - cross_section_half_length,
                    side="left",
                )
            )
            window_end = int(
                np.searchsorted(
                    x_sorted,
                    station_x + cross_section_half_length,
                    side="right",
                )
            )
            local_x = x_sorted[window_start:window_end] - float(station_x)
            local_y = y_sorted[window_start:window_end] - predicted_center
            local_z = z_sorted[window_start:window_end]
            tangent_scale = float(np.hypot(1.0, center_velocity))
            local_lateral = (
                local_y - center_velocity * local_x
            ) / tangent_scale
            local_bins = np.floor(
                (local_lateral + self.y_limit_m) / self.lateral_resolution_m
            ).astype(np.int32)
            in_lateral_grid = (
                (local_bins >= 0)
                & (local_bins < self._y_count)
                & (local_z >= self.min_ground_height_m)
                & (local_z <= self.max_ground_height_m)
            )
            local_bins = local_bins[in_lateral_grid]
            local_z = local_z[in_lateral_grid]

            profile = np.full(self._y_count, np.inf, dtype=np.float32)
            high_profile = np.full(self._y_count, -np.inf, dtype=np.float32)
            count_profile = np.bincount(
                local_bins, minlength=self._y_count
            ).astype(np.int32, copy=False)
            if local_z.size:
                np.minimum.at(profile, local_bins, local_z)
                np.maximum.at(high_profile, local_bins, local_z)
            profile[count_profile == 0] = np.nan
            high_profile[count_profile == 0] = np.nan
            span_profile = high_profile - profile

            if np.count_nonzero(np.isfinite(profile)) < 8:
                left_points.append(None)
                right_points.append(None)
                center_y = predicted_center
                previous_x = float(station_x)
                continue

            predicted_left = -half_width
            predicted_right = half_width
            observation_gap = (
                delta_x
                if last_observation_x is None
                else max(delta_x, float(station_x - last_observation_x))
            )
            continuity_limit = min(
                4.0,
                self.max_edge_track_step_m
                + max(0.0, observation_gap - self.station_step_m) * 0.22,
            )
            left_candidates = self._find_edge_candidates(
                profile,
                span_profile,
                count_profile,
                0.0,
                side=-1,
                predicted_edge=predicted_left,
                expected_offset=half_width,
                continuity_limit=continuity_limit,
            )
            right_candidates = self._find_edge_candidates(
                profile,
                span_profile,
                count_profile,
                0.0,
                side=1,
                predicted_edge=predicted_right,
                expected_offset=half_width,
                continuity_limit=continuity_limit,
            )

            left = left_candidates[0] if left_candidates else None
            right = right_candidates[0] if right_candidates else None

            if left is not None and right is not None:
                measured_width = right[0] - left[0]
                center_candidate = 0.5 * (left[0] + right[0])
                if 2.8 <= measured_width <= min(16.0, self.y_limit_m * 1.8):
                    half_width = 0.90 * half_width + 0.10 * (measured_width * 0.5)
                    paired_edge_observations = min(20, paired_edge_observations + 1)
                    width_confirmed = paired_edge_observations >= 3
                    new_center = predicted_center + center_candidate / tangent_scale
                else:
                    left_error = abs(left[0] - predicted_left)
                    right_error = abs(right[0] - predicted_right)
                    if left_error <= right_error:
                        right = None
                        paired_edge_observations = 0
                        width_confirmed = False
                        new_center = predicted_center + (left[0] + half_width) / tangent_scale
                    else:
                        left = None
                        paired_edge_observations = 0
                        width_confirmed = False
                        new_center = predicted_center + (right[0] - half_width) / tangent_scale
            elif left is not None:
                new_center = predicted_center + (left[0] + half_width) / tangent_scale
            elif right is not None:
                new_center = predicted_center + (right[0] - half_width) / tangent_scale
            else:
                new_center = predicted_center

            # Infer an occluded opposite edge only after road width was observed
            # on both sides repeatedly, rather than assuming width immediately.
            if width_confirmed:
                if left is not None and right is None:
                    right = (left[0] + 2.0 * half_width, left[1])
                elif right is not None and left is None:
                    left = (right[0] - 2.0 * half_width, right[1])

            max_center_shift = max(0.5, min(3.0, observation_gap * 1.15))
            center_delta = float(
                np.clip(
                    new_center - predicted_center,
                    -max_center_shift,
                    max_center_shift,
                )
            )
            updated_center = predicted_center + center_delta
            observed_velocity = center_delta / max(observation_gap, 1e-3)
            center_velocity = float(
                np.clip(
                    0.65 * center_velocity + 0.35 * observed_velocity,
                    -1.35,
                    1.35,
                )
            )
            center_y = updated_center

            if left is not None or right is not None:
                last_observation_x = float(station_x)

            # Convert the normal-offset estimate back into sensor-local XY.
            if left is not None:
                left_offset = float(left[0]) / tangent_scale
                left_points.append(
                    (
                        float(station_x - center_velocity * left_offset),
                        float(predicted_center + left_offset),
                    )
                )
            else:
                left_points.append(None)

            if right is not None:
                right_offset = float(right[0]) / tangent_scale
                right_points.append(
                    (
                        float(station_x - center_velocity * right_offset),
                        float(predicted_center + right_offset),
                    )
                )
            else:
                right_points.append(None)

            previous_x = float(station_x)

        return {
            "left_edge_points": self._pack_track(left_points),
            "right_edge_points": self._pack_track(right_points),
            "front_edge_points": self._readonly_points([]),
        }

    def _find_edge_candidates(
        self,
        profile: np.ndarray,
        span_profile: np.ndarray,
        count_profile: np.ndarray,
        center_y: float,
        *,
        side: int,
        predicted_edge: float,
        expected_offset: float,
        continuity_limit: float,
    ) -> list[tuple[float, float]]:
        signed_distance = (self._y_centers - center_y) * side
        ordered = np.flatnonzero(
            (signed_distance >= self.road_start_distance_m)
            & (signed_distance <= self.y_limit_m * 0.95)
        )
        if side < 0:
            ordered = ordered[::-1]
        if ordered.size < 7:
            return []

        heights = profile[ordered]
        spans = span_profile[ordered]
        counts = count_profile[ordered].astype(np.float32, copy=False)
        positions = np.arange(3, ordered.size - 6)
        if positions.size == 0:
            return []

        def window_median(
            values: np.ndarray,
            window_positions: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            matrix = np.column_stack(
                [values[window_positions + offset] for offset in range(3)]
            )
            is_valid = np.isfinite(matrix)
            valid_count = np.sum(is_valid, axis=1)
            ordered_values = np.sort(np.where(is_valid, matrix, np.inf), axis=1)
            median = np.full(len(window_positions), np.nan, dtype=np.float32)
            two = valid_count == 2
            three = valid_count == 3
            one = valid_count == 1
            median[two] = 0.5 * (ordered_values[two, 0] + ordered_values[two, 1])
            median[three] = ordered_values[three, 1]
            median[one] = ordered_values[one, 0]
            return median, valid_count

        inner_z, inner_valid = window_median(heights, positions - 3)
        outer_z, outer_valid = window_median(heights, positions)
        stable_z, stable_valid = window_median(heights, positions + 3)
        inner_span, _ = window_median(spans, positions - 3)
        outer_span, _ = window_median(spans, positions)
        stable_span, _ = window_median(spans, positions + 3)

        inner_support = np.sum(
            np.column_stack(
                [counts[positions - 3 + offset] for offset in range(3)]
            ),
            axis=1,
        )
        outer_support = np.sum(
            np.column_stack(
                [counts[positions + offset] for offset in range(3)]
            ),
            axis=1,
        )
        stable_support = np.sum(
            np.column_stack(
                [counts[positions + 3 + offset] for offset in range(3)]
            ),
            axis=1,
        )

        step = np.abs(outer_z - inner_z)
        edge_y = 0.5 * (
            self._y_centers[ordered[positions - 1]]
            + self._y_centers[ordered[positions]]
        )
        offset = np.abs(edge_y - center_y)
        continuity_error = np.abs(edge_y - predicted_edge)

        stable_check = (
            (stable_valid >= 1)
            & (stable_support >= 1)
            & (stable_span <= self.max_cell_vertical_span_m)
            & (
                np.abs(stable_z - outer_z)
                <= max(0.18, self.road_surface_tolerance_m)
            )
        )
        valid = (
            (inner_valid >= 2)
            & (outer_valid >= 2)
            & (inner_support >= 2)
            & (outer_support >= 2)
            & (inner_span <= self.max_cell_vertical_span_m)
            & (outer_span <= self.max_cell_vertical_span_m)
            & (step >= self.edge_step_m)
            & (step <= self.max_curb_step_m)
            & stable_check
            & (continuity_error <= continuity_limit)
        )
        candidate_ids = np.flatnonzero(valid)
        if candidate_ids.size == 0:
            return []

        support_score = np.minimum(
            1.0,
            (inner_support + outer_support + stable_support) / 12.0,
        )
        step_score = 1.0 - np.minimum(1.0, np.abs(step - 0.18) / 0.45)
        expected_score = np.abs(offset - expected_offset) / max(
            1.0, expected_offset
        )
        continuity_score = continuity_error / max(continuity_limit, 1e-3)
        scores = (
            1.15 * continuity_score
            + 0.75 * expected_score
            - 0.25 * support_score
            - 0.15 * step_score
        )
        ranked = candidate_ids[np.argsort(scores[candidate_ids])]
        return [
            (float(edge_y[index]), float(scores[index]))
            for index in ranked
        ]

    def _pack_track(
        self,
        points: list[tuple[float, float] | None],
    ) -> np.ndarray:
        anchors = [point for point in points if point is not None]
        if len(anchors) < 2:
            return self._readonly_points(anchors)

        segments: list[list[tuple[float, float]]] = []
        current: list[tuple[float, float]] = [anchors[0]]
        for index in range(1, len(anchors)):
            previous = anchors[index - 1]
            following = anchors[index]
            gap = following[0] - previous[0]
            if 0.0 < gap <= self.max_interpolation_gap_m:
                current.extend(self._interpolate_curve(anchors, index - 1, index))
                current.append(following)
            else:
                if len(current) >= 2:
                    segments.append(current)
                current = [following]
        if len(current) >= 2:
            segments.append(current)

        result: list[tuple[float, float]] = []
        for segment in segments:
            if result:
                result.append((float("nan"), float("nan")))
            result.extend(self._smooth_segment(segment))
        return self._readonly_points(result)

    def _interpolate_curve(
        self,
        anchors: list[tuple[float, float]],
        first_index: int,
        second_index: int,
    ) -> list[tuple[float, float]]:
        first = anchors[first_index]
        second = anchors[second_index]
        gap = second[0] - first[0]
        if gap <= self.station_step_m * 1.5:
            return []

        if first_index > 0:
            previous = anchors[first_index - 1]
            slope_start = (second[1] - previous[1]) / max(
                second[0] - previous[0], 1e-3
            )
        else:
            slope_start = (second[1] - first[1]) / gap

        if second_index + 1 < len(anchors):
            following = anchors[second_index + 1]
            slope_end = (following[1] - first[1]) / max(
                following[0] - first[0], 1e-3
            )
        else:
            slope_end = (second[1] - first[1]) / gap

        secant = (second[1] - first[1]) / gap
        max_slope = max(0.25, abs(secant) * 2.5 + 0.12)
        slope_start = float(np.clip(slope_start, -max_slope, max_slope))
        slope_end = float(np.clip(slope_end, -max_slope, max_slope))

        sample_count = max(
            1, int(np.floor(gap / self.station_step_m)) - 1
        )
        interpolated: list[tuple[float, float]] = []
        for sample in range(1, sample_count + 1):
            x = first[0] + min(gap - 1e-4, sample * self.station_step_m)
            t = (x - first[0]) / gap
            h00 = 2 * t**3 - 3 * t**2 + 1
            h10 = t**3 - 2 * t**2 + t
            h01 = -2 * t**3 + 3 * t**2
            h11 = t**3 - t**2
            y = (
                h00 * first[1]
                + h10 * gap * slope_start
                + h01 * second[1]
                + h11 * gap * slope_end
            )
            interpolated.append((float(x), float(y)))
        return interpolated

    @staticmethod
    def _smooth_segment(
        points: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        if len(points) < 5:
            return points
        values = np.asarray(points, dtype=np.float32)
        y_values = values[:, 1]
        padded = np.pad(y_values, (2, 2), mode="edge")
        windows = np.column_stack(
            [padded[offset:offset + len(y_values)] for offset in range(5)]
        )
        smoothed_y = np.median(windows, axis=1)
        return [
            (float(values[index, 0]), float(smoothed_y[index]))
            for index in range(len(values))
        ]

    @staticmethod
    def _readonly_points(points: list[tuple[float, float]]) -> np.ndarray:
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
