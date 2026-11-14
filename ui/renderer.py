from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from utils.vehicle_utils import blank_frame


class Renderer:
    """Prepare clean camera frames for the PySide6 runtime panel."""

    def __init__(self, window_name: str = "CARLA") -> None:
        self.window_name = window_name

    @staticmethod
    def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError(f"unsupported debug frame shape: {arr.shape}")
        return np.ascontiguousarray(arr[:, :, :3])

    def compose(self, screen: Optional[np.ndarray], overlay: Optional[dict] = None) -> np.ndarray:
        """Return an unannotated image; telemetry is drawn once in the Qt sidebar."""
        if screen is None:
            return blank_frame()
        return self._ensure_bgr(screen).copy()

    def show_frame(self, screen: Optional[np.ndarray], overlay: Optional[dict] = None) -> np.ndarray:
        """Compatibility wrapper; presentation is performed by ControlPanel."""
        return self.compose(screen, overlay)

    def render_mode_overlay(self, screen: np.ndarray, mode: str, error: float) -> None:
        """Compatibility no-op; mode and error are shown in the Qt telemetry panel."""
        return
