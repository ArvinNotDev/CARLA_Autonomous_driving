from __future__ import annotations

import unittest

import numpy as np

from sensors.road_geometry import RoadGeometryAnalyzer


class RoadGeometryAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = RoadGeometryAnalyzer()
        self.analyzer.update_settings({
            "LIDAR_BOUNDARY_ENABLED": True,
            "LIDAR_BOUNDARY_MIN_FORWARD_M": 0.5,
            "LIDAR_BOUNDARY_FORWARD_RANGE_M": 35.0,
            "LIDAR_BOUNDARY_SIDE_RANGE_M": 10.0,
            "LIDAR_BOUNDARY_MIN_SIDE_M": 0.75,
            "LIDAR_BOUNDARY_HEIGHT_STEP_M": 0.08,
            "LIDAR_BOUNDARY_SURFACE_TOLERANCE_M": 0.22,
        })

    @staticmethod
    def make_road(
        *, curved: bool, facade: bool = True, length: float = 34.0
    ) -> tuple[np.ndarray, object]:
        rng = np.random.default_rng(17)
        x_values = np.arange(0.7, length, 0.15, dtype=np.float32)
        y_values = np.arange(-10.0, 10.0, 0.15, dtype=np.float32)
        xx, yy = np.meshgrid(x_values, y_values, indexing="ij")
        centerline = 0.003 * xx**2 if curved else np.zeros_like(xx)
        lateral_offset = yy - centerline
        grade = 0.004 * xx + 0.008 * lateral_offset
        z = np.where(np.abs(lateral_offset) <= 3.0, grade, grade + 0.16)
        points = np.column_stack((xx.ravel(), yy.ravel(), z.ravel()))

        if facade:
            walls = []
            wall_z = np.arange(0.35, 3.0, 0.18, dtype=np.float32)
            wall_x, wall_height = np.meshgrid(x_values, wall_z)
            for side in (-1.0, 1.0):
                wall_center = 0.003 * wall_x**2 if curved else np.zeros_like(wall_x)
                wall_y = wall_center + side * 5.8
                walls.append(
                    np.column_stack((wall_x.ravel(), wall_y.ravel(), wall_height.ravel()))
                )
            points = np.vstack((points, *walls))

        points = points[rng.random(len(points)) > 0.12]
        points[:, 2] += rng.normal(0.0, 0.012, len(points))
        centerline_fn = (lambda x: 0.003 * x**2) if curved else (lambda x: 0.0)
        return points.astype(np.float32), centerline_fn

    @staticmethod
    def make_wall_only(length: float = 34.0) -> np.ndarray:
        x_values = np.arange(0.7, length, 0.15, dtype=np.float32)
        y_values = np.arange(-3.0, 3.0, 0.15, dtype=np.float32)
        xx, yy = np.meshgrid(x_values, y_values, indexing="ij")
        road = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
        walls = []
        wall_z = np.arange(0.3, 3.2, 0.12, dtype=np.float32)
        wall_x, wall_height = np.meshgrid(x_values, wall_z)
        for side in (-1.0, 1.0):
            wall_y = np.full(wall_x.size, side * 3.25, dtype=np.float32)
            walls.append(np.column_stack((wall_x.ravel(), wall_y, wall_height.ravel())))
        return np.vstack((road, *walls)).astype(np.float32)

    @staticmethod
    def make_sparse_32_channel_scan(curvature: float = 0.003) -> np.ndarray:
        """Generate a sparse spinning scan with 32 downward LiDAR channels."""
        origin_z = 2.2
        azimuth = np.deg2rad(np.arange(1800, dtype=np.float32) * 360.0 / 1800.0)
        elevation = np.deg2rad(np.linspace(-25.0, 15.0, 32, dtype=np.float32))
        azimuth_grid, elevation_grid = np.meshgrid(azimuth, elevation, indexing="ij")
        dx = np.cos(elevation_grid) * np.cos(azimuth_grid)
        dy = np.cos(elevation_grid) * np.sin(azimuth_grid)
        dz = np.sin(elevation_grid)
        downward = (dz < -1e-4) & (dx > 0.0)
        dx, dy, dz = dx[downward], dy[downward], dz[downward]

        road_distance = -origin_z / dz
        shoulder_distance = -(origin_z - 0.16) / dz
        road_x = road_distance * dx
        road_y = road_distance * dy
        road_center = curvature * road_x**2
        on_road = np.abs(road_y - road_center) <= 3.0
        distance = np.where(on_road, road_distance, shoulder_distance)
        x = distance * dx
        y = distance * dy
        z = origin_z + distance * dz
        valid = (x >= 0.5) & (x <= 38.0) & (np.abs(y) <= 12.0)
        return np.column_stack((x[valid], y[valid], z[valid])).astype(np.float32)

    def test_straight_road_edges_are_detected_on_both_sides(self) -> None:
        cloud, _ = self.make_road(curved=False)
        result = self.analyzer.analyze(cloud)

        for key, side in (("left_edge_points", -1.0), ("right_edge_points", 1.0)):
            points = result[key]
            points = points[np.isfinite(points[:, 0])]
            self.assertGreater(len(points), 20)
            self.assertLess(float(np.median(np.abs(points[:, 1] - side * 3.0))), 0.35)

    def test_curved_road_edges_follow_the_changing_centerline(self) -> None:
        cloud, centerline = self.make_road(curved=True)
        result = self.analyzer.analyze(cloud)

        for key, side in (("left_edge_points", -1.0), ("right_edge_points", 1.0)):
            points = result[key]
            points = points[np.isfinite(points[:, 0])]
            self.assertGreater(len(points), 20)
            expected = np.asarray([centerline(x) + side * 3.0 for x in points[:, 0]])
            self.assertLess(float(np.median(np.abs(points[:, 1] - expected))), 0.45)
            self.assertGreater(float(points[-1, 1] - points[0, 1]), 1.5)

    def test_s_curve_edges_follow_both_changes_in_curvature(self) -> None:
        rng = np.random.default_rng(8)
        x_values = np.arange(0.7, 34.0, 0.12, dtype=np.float32)
        y_values = np.arange(-10.0, 10.0, 0.12, dtype=np.float32)
        xx, yy = np.meshgrid(x_values, y_values, indexing="ij")
        centerline = 2.4 * np.sin(xx / 8.0)
        lateral_offset = yy - centerline
        z = np.where(np.abs(lateral_offset) <= 3.0, 0.0, 0.16)
        points = np.column_stack((xx.ravel(), yy.ravel(), z.ravel()))
        points = points[rng.random(len(points)) > 0.10]
        points[:, 2] += rng.normal(0.0, 0.008, len(points))
        self.analyzer.update_settings({
            "LIDAR_BOUNDARY_FORWARD_RANGE_M": 35.0,
            "LIDAR_BOUNDARY_SIDE_RANGE_M": 10.0,
        })
        result = self.analyzer.analyze(points.astype(np.float32))

        for key, side in (("left_edge_points", -1.0), ("right_edge_points", 1.0)):
            edge = result[key]
            edge = edge[np.isfinite(edge[:, 0])]
            expected = 2.4 * np.sin(edge[:, 0] / 8.0) + side * 3.0
            self.assertGreater(len(edge), 25)
            self.assertLess(float(np.median(np.abs(edge[:, 1] - expected))), 0.30)
            self.assertGreater(float(np.ptp(edge[:, 1])), 1.5)

    def test_sparse_32_channel_scan_tracks_curved_edges(self) -> None:
        cloud = self.make_sparse_32_channel_scan()
        self.analyzer.update_settings({
            "LIDAR_BOUNDARY_FORWARD_RANGE_M": 38.0,
            "LIDAR_BOUNDARY_SIDE_RANGE_M": 12.0,
        })
        result = self.analyzer.analyze(cloud)

        for key, side in (("left_edge_points", -1.0), ("right_edge_points", 1.0)):
            points = result[key]
            points = points[np.isfinite(points[:, 0])]
            self.assertGreater(len(points), 20)
            self.assertGreater(float(points[-1, 0] - points[0, 0]), 20.0)
            expected = 0.003 * points[:, 0] ** 2 + side * 3.0
            self.assertLess(float(np.median(np.abs(points[:, 1] - expected))), 0.45)

    def test_parallel_vertical_roadside_faces_are_detected(self) -> None:
        # The wall is flush with the road edge and runs parallel to the vehicle.
        # Its large within-cell height span must count as edge evidence.
        result = self.analyzer.analyze(self.make_wall_only())

        for key, side in (("left_edge_points", -1.0), ("right_edge_points", 1.0)):
            points = result[key]
            points = points[np.isfinite(points[:, 0])]
            self.assertGreater(len(points), 20)
            self.assertLess(
                float(np.median(np.abs(points[:, 1] - side * 3.25))),
                0.35,
            )

    def test_short_occlusion_gap_is_curve_interpolated(self) -> None:
        anchors = [(1.0, 0.0), (1.6, 0.2), (4.6, 1.8), (5.2, 2.0)]
        points = self.analyzer._pack_track(anchors)
        self.assertGreater(len(points), len(anchors))
        self.assertTrue(np.all(np.diff(points[:, 0]) > 0.0))
        self.assertTrue(np.all(np.isfinite(points)))

    def test_long_occlusion_keeps_separate_segments(self) -> None:
        anchors = [(1.0, 0.0), (1.6, 0.1), (15.0, 4.0), (15.6, 4.2)]
        points = self.analyzer._pack_track(anchors)
        self.assertTrue(np.isnan(points[:, 0]).any())

    def test_disabled_detector_returns_empty_polylines(self) -> None:
        self.analyzer.update_settings({"LIDAR_BOUNDARY_ENABLED": False})
        result = self.analyzer.analyze(np.zeros((200, 3), dtype=np.float32))
        self.assertTrue(all(len(points) == 0 for points in result.values()))


if __name__ == "__main__":
    unittest.main()
