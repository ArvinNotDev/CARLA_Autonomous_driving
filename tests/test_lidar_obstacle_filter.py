import numpy as np

from sensors.lidar_obstacle_filter import inspect_lidar_obstacle_on_side


DEFAULTS = {
    "min_height_m": 0.85,
    "max_height_m": 8.0,
    "min_side_distance_m": 1.25,
    "max_side_distance_m": 8.0,
    "rear_extent_m": 12.0,
    "max_forward_distance_m": 25.0,
    "min_points": 3,
}


def test_positive_drivable_error_checks_right_side():
    points = np.array([[5.0, 2.0, 1.2], [6.0, 2.5, 1.0], [7.0, 3.0, 2.0]])
    result = inspect_lidar_obstacle_on_side(points, 35.0, **DEFAULTS)
    assert result["blocked"] is True
    assert result["side"] == "right"
    assert result["point_count"] == 3


def test_negative_drivable_error_checks_left_side():
    points = np.array([[5.0, -2.0, 1.2], [6.0, -2.5, 1.0], [7.0, -3.0, 2.0]])
    result = inspect_lidar_obstacle_on_side(points, -35.0, **DEFAULTS)
    assert result["blocked"] is True
    assert result["side"] == "left"


def test_object_behind_vehicle_on_side_is_detected():
    points = np.array([[-8.0, 2.0, 1.2], [-7.0, 2.5, 1.0], [-6.0, 3.0, 2.0]])
    result = inspect_lidar_obstacle_on_side(points, 35.0, **DEFAULTS)
    assert result["blocked"] is True
    assert result["point_count"] == 3


def test_object_ahead_and_behind_are_checked_in_same_side_corridor():
    points = np.array([
        [-10.0, 2.0, 1.2],
        [-8.0, 2.0, 1.2],
        [4.0, 2.0, 1.2],
    ])
    result = inspect_lidar_obstacle_on_side(points, 35.0, **DEFAULTS)
    assert result["blocked"] is True
    assert result["point_count"] == 3


def test_object_beyond_rear_extent_does_not_count():
    points = np.array([[-14.0, 2.0, 1.2], [-13.0, 2.5, 1.0], [-12.5, 3.0, 2.0]])
    result = inspect_lidar_obstacle_on_side(points, 35.0, **DEFAULTS)
    assert result["blocked"] is False


def test_object_on_opposite_side_does_not_suppress_recovery():
    points = np.array([[5.0, -2.0, 1.2], [6.0, -2.5, 1.0], [7.0, -3.0, 2.0]])
    result = inspect_lidar_obstacle_on_side(points, 35.0, **DEFAULTS)
    assert result["blocked"] is False


def test_low_road_returns_do_not_count_as_objects():
    points = np.array([[5.0, 2.0, 0.1], [6.0, 2.5, 0.4], [7.0, 3.0, 0.7]])
    result = inspect_lidar_obstacle_on_side(points, 35.0, **DEFAULTS)
    assert result["blocked"] is False


def test_too_few_points_do_not_suppress_recovery():
    points = np.array([[5.0, 2.0, 1.2], [6.0, 2.5, 1.0]])
    result = inspect_lidar_obstacle_on_side(points, 35.0, **DEFAULTS)
    assert result["blocked"] is False


def test_points_outside_side_or_forward_window_do_not_count():
    points = np.array([
        [30.0, 2.0, 1.2],
        [5.0, 10.0, 1.2],
        [-2.0, 2.0, 0.4],
    ])
    result = inspect_lidar_obstacle_on_side(points, 35.0, **DEFAULTS)
    assert result["blocked"] is False
