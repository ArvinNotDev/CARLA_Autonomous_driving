from __future__ import annotations

from typing import Any

import numpy as np


def inspect_lidar_obstacle_on_side(
    points: np.ndarray | None,
    drivable_error: float,
    *,
    min_height_m: float,
    max_height_m: float,
    min_side_distance_m: float,
    max_side_distance_m: float,
    min_forward_distance_m: float,
    max_forward_distance_m: float,
    min_points: int,
) -> dict[str, Any]:
    """Detect a sufficiently tall LiDAR return on the side implied by the mask error.

    LiDAR points use vehicle-local CARLA coordinates: x forward, y right,
    and z as height above the estimated road surface.
    A positive DrivableAreaDebugger error means the visible drivable region is
    shifted left, so the potentially obstructed side is the vehicle's right.
    """
    side = "right" if float(drivable_error) >= 0.0 else "left"
    empty = {
        "blocked": False,
        "side": side,
        "point_count": 0,
        "max_height_m": None,
        "reason": "no_valid_scan",
    }
    if points is None:
        return empty

    cloud = np.asarray(points, dtype=np.float32)
    if cloud.ndim != 2 or cloud.shape[1] < 3 or cloud.size == 0:
        return empty

    finite = np.isfinite(cloud[:, :3]).all(axis=1)
    if not np.any(finite):
        return empty
    cloud = cloud[finite, :3]

    side_sign = 1.0 if side == "right" else -1.0
    lateral = cloud[:, 1] * side_sign
    height = cloud[:, 2]

    min_side = max(0.0, float(min_side_distance_m))
    max_side = max(min_side, float(max_side_distance_m))
    min_forward = float(min_forward_distance_m)
    max_forward = max(min_forward, float(max_forward_distance_m))
    min_height = max(0.0, float(min_height_m))
    max_height = max(min_height, float(max_height_m))

    candidate_mask = (
        (cloud[:, 0] >= min_forward)
        & (cloud[:, 0] <= max_forward)
        & (lateral >= min_side)
        & (lateral <= max_side)
        & (height >= min_height)
        & (height <= max_height)
    )
    candidate_heights = height[candidate_mask]
    point_count = int(candidate_heights.size)
    required_points = max(1, int(min_points))
    blocked = point_count >= required_points

    return {
        "blocked": blocked,
        "side": side,
        "point_count": point_count,
        "max_height_m": (
            float(candidate_heights.max()) if point_count else None
        ),
        "reason": "side_object_detected" if blocked else "no_side_object",
    }
