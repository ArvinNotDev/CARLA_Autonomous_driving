from __future__ import annotations

from typing import Any, Tuple, Optional

import cv2
import numpy as np

import config_city as conf
from utils.vehicle_utils import blank_frame


def cfg(name: str, default: Any) -> Any:
    return getattr(conf, name, default)


class Renderer:
    def __init__(self, window_name: str = "CARLA") -> None:
        self.window_name = window_name

    @staticmethod
    def overlay_text(
        img: np.ndarray,
        text: str,
        org: Tuple[int, int],
        scale: float = 0.7,
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
    ) -> None:
        cv2.putText(
            img,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def show_frame(self, screen: Optional[np.ndarray], overlay: Optional[dict] = None) -> None:
        if screen is None:
            screen = blank_frame()

        if overlay:
            y = 28
            for text in overlay.get("lines", []):
                self.overlay_text(screen, str(text), (12, y), 0.55, (255, 255, 255), 1)
                y += 20
        conf.debug_frame_buffer = screen

        if cfg("SHOW_OPENCV_WINDOW", False):
            cv2.imshow(self.window_name, cv2.resize(screen, (1280, 720)))
            cv2.waitKey(1)

    def render_mode_overlay(self, screen: np.ndarray, mode: str, error: float) -> None:
        self.overlay_text(screen, f"MODE: {mode}", (20, 30), 0.9, (255, 255, 0), 2)
        self.overlay_text(screen, f"error: {error:.2f}", (20, 60), 0.7, (255, 255, 255), 2)
