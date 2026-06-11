from __future__ import annotations

from typing import Any

import carla
import numpy as np

import config_city as conf


def cfg(name: str, default: Any) -> Any:
    return getattr(conf, name, default)


def make_stop_control() -> carla.VehicleControl:
    return carla.VehicleControl(
        throttle=0.0,
        steer=0.0,
        brake=1.0,
        reverse=False,
        hand_brake=False,
        manual_gear_shift=False,
    )


def blank_frame() -> np.ndarray:
    return np.zeros(
        (
            cfg("CAMERA_IMAGE_HEIGHT", 720),
            cfg("CAMERA_IMAGE_WIDTH", 1280),
            3,
        ),
        dtype=np.uint8,
    )