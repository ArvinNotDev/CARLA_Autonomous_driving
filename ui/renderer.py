from __future__ import annotations

from typing import Any, Tuple, Optional

import cv2
import numpy as np

import config_city as conf
from utils.vehicle_utils import blank_frame


def cfg(name: str, default: Any) -> Any:
    return getattr(conf, name, default)


class Renderer:
    """
    CPU-only debug compositor.

    The PySide6 panel owns presentation. OpenCV HighGUI is intentionally not
    driven from the CARLA/control thread, avoiding two competing GUI event loops.
    """

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

    @staticmethod
    def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError(f"unsupported debug frame shape: {arr.shape}")
        return np.ascontiguousarray(arr[:, :, :3])

    def compose(self, screen: Optional[np.ndarray], overlay: Optional[dict] = None) -> np.ndarray:
        if screen is None:
            out = blank_frame()
        else:
            out = self._ensure_bgr(screen).copy()

        h, w = out.shape[:2]
        if overlay:
            y = 28
            for text in overlay.get("lines", []):
                self.overlay_text(out, str(text), (12, min(y, h - 8)), 0.55, (255, 255, 255), 1)
                y += 20
        return np.ascontiguousarray(out)

    def show_frame(self, screen: Optional[np.ndarray], overlay: Optional[dict] = None) -> np.ndarray:
        """Compatibility wrapper; presentation is performed by ControlPanel."""
        return self.compose(screen, overlay)

    def render_mode_overlay(self, screen: np.ndarray, mode: str, error: float) -> None:
        self.overlay_text(screen, f"MODE: {mode}", (20, 30), 0.9, (255, 255, 0), 2)
        self.overlay_text(screen, f"error: {error:.2f}", (20, 60), 0.7, (255, 255, 255), 2)
