from __future__ import annotations

import math
import threading
from collections import deque
from typing import Any


class AvisAdaptiveROIController:
    """AVIS-only horizontal ROI recovery driven by the last seven detections.

    The configured rectangles remain the permanent calibration. Runtime
    expansion is temporary and is applied only to the active vision backend.
    """

    HISTORY_FRAMES = 7
    SIDES = {
        "ll": ("ll_left_roi", "ll_right_roi"),
        "rl": ("rl_left_roi", "rl_right_roi"),
    }

    def __init__(self, settings: dict[str, Any]) -> None:
        self._lock = threading.RLock()
        self._settings: dict[str, Any] = {}
        self._history = {
            side: deque(maxlen=self.HISTORY_FRAMES) for side in self.SIDES
        }
        self._expansion = {
            side: {"left": 0.0, "right": 0.0} for side in self.SIDES
        }
        self._center_frames = {side: 0 for side in self.SIDES}
        self.configure(settings)

    @staticmethod
    def _number(settings: dict[str, Any], key: str, default: float) -> float:
        try:
            value = float(settings.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default

    def configure(self, settings: dict[str, Any]) -> None:
        """Update permanent ROI coordinates and adaptive tuning safely."""
        with self._lock:
            updated = dict(self._settings)
            updated.update(settings)
            roi_changed = any(
                self._settings.get(key) != updated.get(key)
                for keys in self.SIDES.values()
                for key in keys
            )
            enabled_changed = (
                self._settings.get("adaptive_roi_enabled", True)
                != updated.get("adaptive_roi_enabled", True)
            )
            self._settings = updated
            if roi_changed or enabled_changed:
                for side in self.SIDES:
                    self._history[side].clear()
                    self._expansion[side] = {"left": 0.0, "right": 0.0}
                    self._center_frames[side] = 0
            else:
                maximum = min(
                    0.30,
                    max(0.0, self._number(
                        updated, "adaptive_roi_max_expansion_ratio", 0.12
                    )),
                )
                for side in self.SIDES:
                    self._expansion[side]["left"] = min(
                        self._expansion[side]["left"], maximum
                    )
                    self._expansion[side]["right"] = min(
                        self._expansion[side]["right"], maximum
                    )

    def _active_settings_locked(self) -> dict[str, Any]:
        active = dict(self._settings)
        if not bool(active.get("adaptive_roi_enabled", True)):
            return active

        for side, (left_key, right_key) in self.SIDES.items():
            expansion = self._expansion[side]
            active[left_key] = max(
                0.0,
                self._number(active, left_key, 0.0) - expansion["left"],
            )
            active[right_key] = min(
                1.0,
                self._number(active, right_key, 1.0) + expansion["right"],
            )
        return active

    def active_settings(self) -> dict[str, Any]:
        with self._lock:
            return self._active_settings_locked()

    def _observe_side_locked(
        self,
        side: str,
        x: int | float | None,
        frame_width: int,
    ) -> None:
        left_key, right_key = self.SIDES[side]
        left = self._number(self._settings, left_key, 0.0)
        right = self._number(self._settings, right_key, 1.0)
        width = max(1.0, (right - left) * float(frame_width))

        ratio: float | None = None
        if x is not None:
            try:
                pixel_x = float(x)
            except (TypeError, ValueError):
                pixel_x = math.nan
            if math.isfinite(pixel_x):
                ratio = (pixel_x - left * frame_width) / width
                ratio = min(1.0, max(0.0, ratio))

        history = self._history[side]
        history.append(ratio)
        if ratio is None:
            return

        recent = [sample for sample in history if sample is not None]
        edge = min(0.45, max(0.05, self._number(
            self._settings, "adaptive_roi_edge_trigger_ratio", 0.22
        )))
        near_left = sum(sample <= edge for sample in recent)
        near_right = sum(sample >= 1.0 - edge for sample in recent)
        trend = recent[-1] - recent[-3] if len(recent) >= 3 else 0.0

        expand_step = min(0.05, max(0.001, self._number(
            self._settings, "adaptive_roi_expand_step_ratio", 0.02
        )))
        maximum = min(0.30, max(0.0, self._number(
            self._settings, "adaptive_roi_max_expansion_ratio", 0.12
        )))
        restore_step = min(0.05, max(0.001, self._number(
            self._settings, "adaptive_roi_restore_step_ratio", 0.005
        )))
        recovery_frames = max(1, int(self._number(
            self._settings, "adaptive_roi_recovery_frames", 7
        )))

        escaping_left = (
            ratio <= edge
            and near_left >= 2
            and (trend <= -0.015 or ratio <= edge * 0.5)
        )
        escaping_right = (
            ratio >= 1.0 - edge
            and near_right >= 2
            and (trend >= 0.015 or ratio >= 1.0 - edge * 0.5)
        )

        expansion = self._expansion[side]
        if bool(self._settings.get("adaptive_roi_enabled", True)) and maximum > 0.0:
            if escaping_left:
                expansion["left"] = min(maximum, expansion["left"] + expand_step)
                self._center_frames[side] = 0
                return
            if escaping_right:
                expansion["right"] = min(maximum, expansion["right"] + expand_step)
                self._center_frames[side] = 0
                return

        if not any(expansion.values()):
            self._center_frames[side] = 0
            return

        # Recovery is allowed only after the marking is back in the central
        # part of its originally configured ROI for several inference frames.
        if 0.35 <= ratio <= 0.65:
            self._center_frames[side] += 1
        else:
            self._center_frames[side] = 0

        if self._center_frames[side] >= recovery_frames:
            expansion["left"] = max(0.0, expansion["left"] - restore_step)
            expansion["right"] = max(0.0, expansion["right"] - restore_step)
            if expansion["left"] == 0.0 and expansion["right"] == 0.0:
                self._center_frames[side] = 0

    def observe(
        self,
        left_x: int | float | None,
        right_x: int | float | None,
        frame_width: int,
    ) -> None:
        """Consume one distinct inference result for each lane ROI."""
        if frame_width <= 0:
            return
        with self._lock:
            self._observe_side_locked("ll", left_x, frame_width)
            self._observe_side_locked("rl", right_x, frame_width)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            active = self._active_settings_locked()
            result: dict[str, dict[str, Any]] = {}
            for side, (left_key, right_key) in self.SIDES.items():
                result[side] = {
                    "left": float(active[left_key]),
                    "right": float(active[right_key]),
                    "expanded_left": float(self._expansion[side]["left"]),
                    "expanded_right": float(self._expansion[side]["right"]),
                    "history_frames": len(self._history[side]),
                    "center_frames": self._center_frames[side],
                    "expanded": any(self._expansion[side].values()),
                }
            return result
