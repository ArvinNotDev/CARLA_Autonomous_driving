from __future__ import annotations

from pathlib import Path

import numpy as np


def clamp(value: float, min_value: float, max_value: float) -> float:
    return float(max(min_value, min(max_value, value)))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def vehicle_speed_kmh(vehicle) -> float:
    """
    CARLA vehicle speed in km/h.
    """
    vel = vehicle.get_velocity()
    speed_ms = float(np.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z))
    return speed_ms * 3.6