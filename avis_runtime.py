from __future__ import annotations

import argparse
from collections import deque
import importlib.util
import json
import math
import numpy as np
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Optional


class AvisFpsMonitor:
    """Rolling-window FPS meter for AVIS frame, inference and control rates."""

    def __init__(self, window_seconds: float = 2.0) -> None:
        self.window_seconds = max(0.25, float(window_seconds))
        self._timestamps: deque[float] = deque()

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def record(self, timestamp: float | None = None) -> None:
        sample = time.perf_counter() if timestamp is None else float(timestamp)
        if not math.isfinite(sample):
            return
        if self._timestamps and sample <= self._timestamps[-1]:
            return
        self._timestamps.append(sample)
        self._prune(sample)

    def fps(self, now: float | None = None) -> float:
        current = time.perf_counter() if now is None else float(now)
        if not math.isfinite(current):
            return 0.0
        self._prune(current)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0.0:
            return 0.0
        return float((len(self._timestamps) - 1) / elapsed)


PROJECT_ROOT = Path(__file__).resolve().parent
AVIS_PROFILE_NAME = "AVIS_StableV2"

# AVIS copies its StableV2-compatible camera and lane-following settings.
# Legacy drivable-area threshold/debug keys remain in the profile contract for
# compatibility, but the AVIS model adapter does not extract or use that mask.
# CARLA sensors, map navigation, junction, and LiDAR settings are excluded.
AVIS_PROFILE_FIELDS = (
    "AUTO_MODE_DEFAULT",
    "KP",
    "KI",
    "KD",
    "STEER_LIMIT",
    "MAX_STEER_STEP",
    "CONTROL_LOOP_HZ",
    "VISION_INFERENCE_INTERVAL_SECONDS",
    "LANE_PROB_THRESHOLD",
    "LANE_CENTER_SMOOTH_ALPHA",
    "DRIVABLE_PROB_THRESHOLD",
    "RL_TOP_ROI",
    "RL_BOTTOM_ROI",
    "RL_LEFT_ROI",
    "RL_RIGHT_ROI",
    "LL_TOP_ROI",
    "LL_BOTTOM_ROI",
    "LL_LEFT_ROI",
    "LL_RIGHT_ROI",
    "VISION_DEBUG",
    "DEBUG_SHOW_ROIS",
    "DEBUG_SHOW_LANE_MASK",
    "DEBUG_SHOW_DRIVABLE_AREA",
    "DEBUG_SHOW_TEXT",
    "DEBUG_PANEL_HZ",
    "MODEL_PATH",
    "MODEL_IMAGE_SIZE",
    "ACCELERATION_MPS2",
    "MIN_SPEED_KMH",
    "MAX_SPEED_KMH",
)


def load_avis_profile(path: Path | str | None = None) -> dict[str, Any]:
    profile_path = Path(path) if path is not None else PROJECT_ROOT / "runtime_profiles.json"
    if not profile_path.is_absolute():
        profile_path = PROJECT_ROOT / profile_path

    try:
        profiles = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read runtime profiles: {profile_path}") from exc

    profile = profiles.get(AVIS_PROFILE_NAME) if isinstance(profiles, dict) else None
    if not isinstance(profile, dict):
        raise RuntimeError(
            f"Runtime profile '{AVIS_PROFILE_NAME}' is missing from {profile_path}."
        )

    selected = {key: profile[key] for key in AVIS_PROFILE_FIELDS if key in profile}
    missing = [key for key in AVIS_PROFILE_FIELDS if key not in selected]
    if missing:
        raise RuntimeError(
            f"Profile '{AVIS_PROFILE_NAME}' is incomplete. Missing: {', '.join(missing)}"
        )
    return selected


def apply_avis_profile(profile: dict[str, Any], config_module: ModuleType) -> None:
    for key in AVIS_PROFILE_FIELDS:
        if key in profile:
            setattr(config_module, key, profile[key])


def normalize_avis_controls(throttle: float, steering: float) -> tuple[int, int]:
    """Convert autonomous throttle [0, 1] and steering [-1, 1] to AVIS commands."""
    throttle_value = float(throttle)
    steering_value = float(steering)
    if not math.isfinite(throttle_value) or not math.isfinite(steering_value):
        raise ValueError("AVIS controls must be finite numbers.")
    speed_command = int(round(min(1.0, max(0.0, throttle_value)) * 100.0))
    steering_command = int(round(min(1.0, max(-1.0, steering_value)) * 100.0))
    return speed_command, steering_command


def normalize_avis_manual_controls(
    throttle: float,
    steering: float,
) -> tuple[int, int]:
    """Convert bidirectional manual throttle and steering to AVIS API commands."""
    throttle_value = float(throttle)
    steering_value = float(steering)
    if not math.isfinite(throttle_value) or not math.isfinite(steering_value):
        raise ValueError("AVIS manual controls must be finite numbers.")
    return (
        int(round(min(1.0, max(-1.0, throttle_value)) * 100.0)),
        int(round(min(1.0, max(-1.0, steering_value)) * 100.0)),
    )


def select_avis_target_speed(
    steering: float,
    minimum_kmh: float,
    maximum_kmh: float,
) -> float:
    """Keep StableV2's steering-demand speed reduction for AVIS lane following."""
    minimum = float(minimum_kmh)
    maximum = float(maximum_kmh)
    steering_value = float(steering)
    if not all(math.isfinite(value) for value in (minimum, maximum, steering_value)):
        raise ValueError("Steering and speed settings must be finite.")
    if minimum <= 0.0 or maximum <= minimum:
        raise ValueError("AVIS speed limits must satisfy 0 < minimum < maximum.")

    steering_demand = min(1.0, max(0.0, abs(steering_value)))
    turn_factor = min(1.0, max(0.0, (steering_demand - 0.15) / 0.55))
    return maximum - ((maximum - minimum) * turn_factor)



class AvisSteeringController:
    """StableV2 steering PID/rate-limit logic without CARLA-specific types."""

    def __init__(
        self,
        kp: float = 0.015,
        ki: float = 0.0,
        kd: float = 0.002,
        steer_limit: float = 0.75,
        max_steer_step: float = 0.10,
        integral_limit: float = 3000.0,
    ) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.steer_limit = min(1.0, max(0.0, float(steer_limit)))
        self.max_steer_step = min(1.0, max(0.0, float(max_steer_step)))
        self.integral_limit = abs(float(integral_limit))
        self.reset()

    def configure(
        self,
        *,
        kp: float,
        ki: float,
        kd: float,
        steer_limit: float,
        max_steer_step: float,
    ) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.steer_limit = min(1.0, max(0.0, float(steer_limit)))
        self.max_steer_step = min(1.0, max(0.0, float(max_steer_step)))
        self.reset()

    def reset(self) -> None:
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_steer = 0.0

    def update(self, error: float) -> float:
        error_value = float(error)
        if not math.isfinite(error_value):
            raise ValueError("Lane-centre error must be finite.")

        self.integral += error_value
        self.integral = min(
            self.integral_limit,
            max(-self.integral_limit, self.integral),
        )
        derivative = error_value - self.prev_error
        self.prev_error = error_value

        steer = (
            self.kp * error_value
            + self.ki * self.integral
            + self.kd * derivative
        )
        steer = min(self.steer_limit, max(-self.steer_limit, steer))

        delta = steer - self.last_steer
        if delta > self.max_steer_step:
            steer = self.last_steer + self.max_steer_step
        elif delta < -self.max_steer_step:
            steer = self.last_steer - self.max_steer_step

        self.last_steer = steer
        return float(steer)


class AvisObstacleLaneChangeController:
    """AVIS-only obstacle lane-change state machine without GPS dependencies."""

    RIGHT_LANE = "RIGHT_LANE"
    STEER_LEFT = "STEER_LEFT"
    ACQUIRE_LEFT_LINE = "ACQUIRE_LEFT_LINE"
    LEFT_LANE = "LEFT_LANE"
    STEER_RIGHT = "STEER_RIGHT"
    ACQUIRE_RIGHT_LINE = "ACQUIRE_RIGHT_LINE"
    FAILED = "FAILED"

    FOLLOWUP_STEER_MAGNITUDE = 0.20
    FOLLOWUP_PULSE_DISTANCE_M = 0.30
    LINE_CHECK_INTERVAL_M = 0.15

    def __init__(self) -> None:
        self.phase = self.RIGHT_LANE
        self.phase_distance_m = 0.0
        self.left_lane_distance_m = 0.0
        self.last_obstacle_distance_m: float | None = None
        self.left_lane_target_distance_m: float | None = None
        self.return_reason = "DISTANCE"
        self.obstacle_confirm_frames = 0
        self.sensor_clear_frames = 0
        self.obstacle_latched = False
        self.last_obstacle_sensor: str | None = None
        self.line_confirm_frames = 0
        self.last_line_observation_id = None
        self.line_history: dict[str, deque[float | None]] = {
            "left": deque(maxlen=7),
            "right": deque(maxlen=7),
        }
        self.last_history_observation_id = None
        self.line_target_side: str | None = None
        self.line_baseline_x: float | None = None
        self.line_old_track_x: float | None = None
        self.line_old_missing_frames = 0
        self.line_old_cleared = False
        self.line_candidate_x: float | None = None
        self.steering_pulse_index = 0
        self.line_search_distance_m = 0.0

    @staticmethod
    def front_sensor_distance_cm(sensors: Any) -> float | None:
        """AVIS sensor order is left, middle/front, right; all values are cm."""
        try:
            value = float(sensors[1])
        except (TypeError, ValueError, IndexError, KeyError):
            return None
        if not math.isfinite(value) or value < 0.0:
            return None
        return value

    @staticmethod
    def sensor_distances_cm(sensors: Any) -> tuple[float | None, float | None, float | None]:
        """Return AVIS side/front/side distances in centimetres."""
        values: list[float | None] = []
        for index in range(3):
            try:
                value = float(sensors[index])
            except (TypeError, ValueError, IndexError, KeyError):
                value = float("nan")
            values.append(value if math.isfinite(value) and value >= 0.0 else None)
        return values[0], values[1], values[2]

    @staticmethod
    def _positive_setting(settings: dict[str, Any], key: str, default: float) -> float:
        try:
            value = float(settings.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) and value > 0.0 else default

    def _target_lane(self) -> str:
        if self.phase in {self.STEER_LEFT, self.ACQUIRE_LEFT_LINE, self.LEFT_LANE}:
            return "left"
        return "right"

    def snapshot(self, sensors: Any = None) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "lane_side": "left" if self.phase in {
                self.LEFT_LANE, self.STEER_RIGHT, self.ACQUIRE_RIGHT_LINE
            } else "right",
            "target_lane": self._target_lane(),
            "left_lane_distance_m": float(self.left_lane_distance_m),
            "left_lane_target_distance_m": self.left_lane_target_distance_m,
            "last_obstacle_distance_m": self.last_obstacle_distance_m,
            "return_reason": self.return_reason,
            "phase_distance_m": float(self.phase_distance_m),
            "line_search_distance_m": float(self.line_search_distance_m),
            "steering_pulse_number": (
                self.steering_pulse_index + 1
                if self.phase in {
                    self.STEER_LEFT, self.ACQUIRE_LEFT_LINE,
                    self.STEER_RIGHT, self.ACQUIRE_RIGHT_LINE,
                }
                else 0
            ),
            "front_sensor_cm": self.front_sensor_distance_cm(sensors),
            "sensor_distances_cm": list(self.sensor_distances_cm(sensors)),
            "obstacle_trigger_sensor": self.last_obstacle_sensor,
            "target_line_side": self.line_target_side,
            "target_line_baseline_x": self.line_baseline_x,
            "target_line_old_track_x": self.line_old_track_x,
            "target_line_old_cleared": self.line_old_cleared,
            "target_line_candidate_x": self.line_candidate_x,
            "line_confirm_frames": self.line_confirm_frames,
            "active": self.phase in {
                self.STEER_LEFT, self.ACQUIRE_LEFT_LINE,
                self.STEER_RIGHT, self.ACQUIRE_RIGHT_LINE,
            },
            "failed": self.phase == self.FAILED,
            "steering_override": None,
            "safe_stop": self.phase == self.FAILED,
            "pid_reset": False,
        }

    @staticmethod
    def _line_x(value: Any) -> float | None:
        try:
            coordinate = float(value)
        except (TypeError, ValueError):
            return None
        return coordinate if math.isfinite(coordinate) else None

    @staticmethod
    def _median(values: list[float]) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[middle])
        return float((ordered[middle - 1] + ordered[middle]) / 2.0)

    def _record_lane_history(
        self,
        left_x: int | float | None,
        right_x: int | float | None,
        observation_id: Any,
    ) -> None:
        if observation_id is None or observation_id == self.last_history_observation_id:
            return
        if self.phase not in {self.RIGHT_LANE, self.LEFT_LANE}:
            return
        self.last_history_observation_id = observation_id
        self.line_history["left"].append(self._line_x(left_x))
        self.line_history["right"].append(self._line_x(right_x))

    @staticmethod
    def _target_roi_width_px(
        side: str,
        settings: dict[str, Any],
        frame_width: int | float | None,
    ) -> float:
        try:
            image_width = float(frame_width) if frame_width is not None else 320.0
        except (TypeError, ValueError):
            image_width = 320.0
        if not math.isfinite(image_width) or image_width <= 0.0:
            image_width = 320.0
        prefix = "ll" if side == "left" else "rl"
        try:
            roi_left = float(settings.get(f"{prefix}_left_roi", 0.24 if side == "left" else 0.56))
            roi_right = float(settings.get(f"{prefix}_right_roi", 0.45 if side == "left" else 0.78))
        except (TypeError, ValueError):
            roi_left, roi_right = (0.24, 0.45) if side == "left" else (0.56, 0.78)
        if not all(math.isfinite(value) for value in (roi_left, roi_right)) or roi_right <= roi_left:
            roi_left, roi_right = (0.24, 0.45) if side == "left" else (0.56, 0.78)
        return max(1.0, (roi_right - roi_left) * image_width)

    def _start_target_line_tracking(
        self,
        side: str,
        observation_id: Any = None,
    ) -> None:
        self.line_target_side = side
        previous = [
            value for value in self.line_history[side]
            if value is not None and math.isfinite(value)
        ]
        # The median of recent, distinct inference frames is more resistant to
        # a single noisy/misclassified ROI sample than using the latest x alone.
        self.line_baseline_x = self._median(previous[-5:]) if len(previous) >= 2 else None
        self.line_old_track_x = self.line_baseline_x
        self.line_old_missing_frames = 0
        self.line_old_cleared = False
        self.line_candidate_x = None
        self.line_confirm_frames = 0
        # The frame which triggered the manoeuvre belongs to the old-lane
        # history; do not count it again as a post-trigger observation.
        self.last_line_observation_id = observation_id
        self.steering_pulse_index = 0
        self.line_search_distance_m = 0.0

    def _reset_target_line_tracking(self) -> None:
        self.line_target_side = None
        self.line_baseline_x = None
        self.line_old_track_x = None
        self.line_old_missing_frames = 0
        self.line_old_cleared = False
        self.line_candidate_x = None
        self.line_confirm_frames = 0
        self.last_line_observation_id = None
        self.line_history["left"].clear()
        self.line_history["right"].clear()
        self.steering_pulse_index = 0
        self.line_search_distance_m = 0.0

    def _confirm_new_line_candidate(
        self,
        coordinate: float,
        tolerance_px: float,
        frames_required: int,
    ) -> None:
        if (
            self.line_candidate_x is not None
            and abs(coordinate - self.line_candidate_x) <= tolerance_px
        ):
            self.line_candidate_x += (
                coordinate - self.line_candidate_x
            ) / (self.line_confirm_frames + 1)
            self.line_confirm_frames += 1
        else:
            self.line_candidate_x = coordinate
            self.line_confirm_frames = 1
        if self.line_confirm_frames >= frames_required:
            self.line_old_cleared = True

    def _advance_target_line_tracking(
        self,
        value: int | float | None,
        *,
        roi_width_px: float,
        frames_required: int,
    ) -> None:
        coordinate = self._line_x(value)
        old_line_tolerance = max(5.0, roi_width_px * 0.10)
        candidate_tolerance = max(3.0, roi_width_px * 0.06)

        if coordinate is None:
            self.line_old_missing_frames += 1
            self.line_candidate_x = None
            self.line_confirm_frames = 0
            if self.line_old_missing_frames >= 2:
                self.line_old_cleared = True
            return

        self.line_old_missing_frames = 0
        if self.line_old_track_x is None and not self.line_old_cleared:
            # No reliable pre-manoeuvre sample was available. Treat the first
            # visible target-ROI line as the old marking, never as the new lane.
            self.line_old_track_x = coordinate
            self.line_candidate_x = None
            self.line_confirm_frames = 0
            return

        if (
            not self.line_old_cleared
            and self.line_old_track_x is not None
            and abs(coordinate - self.line_old_track_x) <= old_line_tolerance
        ):
            # Follow the same physical marking while it moves across the ROI.
            self.line_old_track_x = coordinate
            self.line_candidate_x = None
            self.line_confirm_frames = 0
            return

        if (
            self.line_old_cleared
            and self.line_old_track_x is not None
            and abs(coordinate - self.line_old_track_x) <= old_line_tolerance
        ):
            # The old marking re-entered the ROI; it must not be mistaken for
            # the destination-lane marking after a brief detection dropout.
            self.line_old_cleared = False
            self.line_old_track_x = coordinate
            self.line_candidate_x = None
            self.line_confirm_frames = 0
            return

        self._confirm_new_line_candidate(
            coordinate,
            candidate_tolerance,
            frames_required,
        )

    def _begin_right_return(
        self,
        reason: str,
        observation_id: Any = None,
    ) -> None:
        self._start_target_line_tracking("right", observation_id)
        self.phase = self.STEER_RIGHT
        self.phase_distance_m = 0.0
        self.return_reason = reason

    def update(
        self,
        *,
        sensors: Any,
        speed_kmh: float,
        dt: float,
        left_x: int | float | None,
        right_x: int | float | None,
        settings: dict[str, Any],
        allow_new_maneuver: bool = True,
        observation_id: Any = None,
        frame_width: int | float | None = None,
    ) -> dict[str, Any]:
        sensor_distances = self.sensor_distances_cm(sensors)
        try:
            speed = float(speed_kmh)
            elapsed = float(dt)
        except (TypeError, ValueError):
            speed, elapsed = 0.0, 0.0
        if not math.isfinite(speed):
            speed = 0.0
        if not math.isfinite(elapsed):
            elapsed = 0.0
        travelled_m = max(0.0, speed) / 3.6 * min(0.25, max(0.0, elapsed))
        self._record_lane_history(left_x, right_x, observation_id)

        if self.phase in {
            self.STEER_LEFT, self.ACQUIRE_LEFT_LINE,
            self.STEER_RIGHT, self.ACQUIRE_RIGHT_LINE,
        }:
            self.phase_distance_m += travelled_m
            self.line_search_distance_m += travelled_m
        elif self.phase == self.LEFT_LANE:
            self.left_lane_distance_m += travelled_m

        trigger_cm = self._positive_setting(
            settings, "obstacle_front_threshold_cm", 950.0
        )
        configured_clear_cm = self._positive_setting(
            settings, "obstacle_sensor_clear_threshold_cm", 900.0
        )
        # Keep sensor hysteresis valid automatically; users should not have to
        # tune two coupled fields just to change the obstacle trigger distance.
        clear_cm = max(configured_clear_cm, trigger_cm + 1.0)
        trigger_frames = max(
            1, int(self._positive_setting(settings, "obstacle_trigger_confirm_frames", 3))
        )
        clear_frames_required = max(
            1, int(self._positive_setting(
                settings, "obstacle_sensor_clear_confirm_frames", 5
            ))
        )

        # Clear the latch only after every distance sensor reports a clear
        # path; one side sensor still seeing an obstacle must keep it latched.
        all_sensors_clear = all(
            distance is not None and distance >= clear_cm
            for distance in sensor_distances
        )
        if all_sensors_clear:
            self.sensor_clear_frames += 1
            if self.sensor_clear_frames >= clear_frames_required:
                self.obstacle_latched = False
                self.obstacle_confirm_frames = 0
        else:
            self.sensor_clear_frames = 0

        pid_reset = False
        detection_enabled = bool(settings.get("obstacle_lane_change_enabled", True))
        # Any of the left, front, or right sensors can request the lane change.
        # Distance sensing stays active even when lane perception is missing.
        can_check_obstacle = self.phase in {self.RIGHT_LANE, self.LEFT_LANE}
        triggered_sensors = [
            (index, distance)
            for index, distance in enumerate(sensor_distances)
            if distance is not None and distance <= trigger_cm
        ]
        if detection_enabled and can_check_obstacle and not self.obstacle_latched:
            if triggered_sensors:
                self.obstacle_confirm_frames += 1
            else:
                self.obstacle_confirm_frames = 0

            if self.obstacle_confirm_frames >= trigger_frames:
                self.obstacle_latched = True
                self.obstacle_confirm_frames = 0
                sensor_names = ("LEFT", "FRONT", "RIGHT")
                sensor_index, obstacle_distance = min(
                    triggered_sensors, key=lambda item: item[1]
                )
                self.last_obstacle_sensor = sensor_names[sensor_index]
                self.last_obstacle_distance_m = obstacle_distance / 100.0
                clearance_m = self._positive_setting(
                    settings, "obstacle_clearance_after_pass_m", 5.0
                )
                self.left_lane_target_distance_m = (
                    self.last_obstacle_distance_m + clearance_m
                )

                if self.phase == self.RIGHT_LANE:
                    self._start_target_line_tracking("left", observation_id)
                    self.phase = self.STEER_LEFT
                    self.phase_distance_m = 0.0
                    self.return_reason = "OBSTACLE"
                    pid_reset = True
                elif self.phase == self.LEFT_LANE:
                    # Any new obstacle sensor trigger while in the left lane
                    # overrides the previous pass-distance target and returns
                    # the vehicle to the right lane.
                    self._begin_right_return("OBSTACLE_IN_LEFT_LANE", observation_id)
                    pid_reset = True

        steer_distance = self._positive_setting(
            settings, "obstacle_lane_change_steer_distance_m", 3.0
        )
        left_steer = min(
            1.0,
            self._positive_setting(settings, "obstacle_lane_change_left_steer", 0.20),
        )
        right_steer = min(
            1.0,
            self._positive_setting(settings, "obstacle_lane_change_right_steer", 0.20),
        )
        line_frames_required = max(
            1, int(self._positive_setting(
                settings, "obstacle_lane_line_confirm_frames", 3
            ))
        )
        line_search_max_m = self._positive_setting(
            settings, "obstacle_lane_line_search_max_distance_m", 40.0
        )
        if self.phase == self.LEFT_LANE:
            target_distance = self.left_lane_target_distance_m
            if target_distance is not None and self.left_lane_distance_m >= target_distance:
                self._begin_right_return("DISTANCE", observation_id)
                pid_reset = True

        # Track the pre-manoeuvre marking through both steering and acquisition
        # phases. Only distinct inference results advance this state machine.
        if self.phase in {self.STEER_LEFT, self.ACQUIRE_LEFT_LINE}:
            target_side = "left"
            target_x = left_x
        elif self.phase in {self.STEER_RIGHT, self.ACQUIRE_RIGHT_LINE}:
            target_side = "right"
            target_x = right_x
        else:
            target_side = None
            target_x = None

        new_line_observation = False
        if (
            target_side is not None
            and observation_id is not None
            and observation_id != self.last_line_observation_id
        ):
            new_line_observation = True
            self.last_line_observation_id = observation_id
            self._advance_target_line_tracking(
                target_x,
                roi_width_px=self._target_roi_width_px(
                    target_side,
                    settings,
                    frame_width,
                ),
                frames_required=line_frames_required,
            )

        steering_override: float | None = None
        safe_stop = False
        target_line_confirmed = (
            self.line_old_cleared
            and self.line_confirm_frames >= line_frames_required
        )

        if self.phase == self.STEER_LEFT:
            pulse_distance = (
                steer_distance
                if self.steering_pulse_index == 0
                else self.FOLLOWUP_PULSE_DISTANCE_M
            )
            # Losing the old line is not proof that the destination line has
            # been found. Keep applying correction until a new line is stable.
            if target_line_confirmed or self.phase_distance_m >= pulse_distance:
                self.phase = self.ACQUIRE_LEFT_LINE
                self.phase_distance_m = 0.0
                steering_override = 0.0
            else:
                pulse_steer = (
                    left_steer
                    if self.steering_pulse_index == 0
                    else min(self.FOLLOWUP_STEER_MAGNITUDE, left_steer)
                )
                steering_override = -pulse_steer

        elif self.phase == self.ACQUIRE_LEFT_LINE:
            steering_override = 0.0
            if target_line_confirmed:
                self.phase = self.LEFT_LANE
                self.phase_distance_m = 0.0
                self.left_lane_distance_m = 0.0
                self._reset_target_line_tracking()
                pid_reset = True
            elif (
                self.phase_distance_m >= self.LINE_CHECK_INTERVAL_M
                and new_line_observation
            ):
                # Check the newest ROI result first; if no confirmed destination
                # marking exists, start the next 0.20 correction immediately.
                self.phase = self.STEER_LEFT
                self.phase_distance_m = 0.0
                self.steering_pulse_index += 1
                steering_override = -min(
                    self.FOLLOWUP_STEER_MAGNITUDE, left_steer
                )

        elif self.phase == self.STEER_RIGHT:
            pulse_distance = (
                steer_distance
                if self.steering_pulse_index == 0
                else self.FOLLOWUP_PULSE_DISTANCE_M
            )
            if target_line_confirmed or self.phase_distance_m >= pulse_distance:
                self.phase = self.ACQUIRE_RIGHT_LINE
                self.phase_distance_m = 0.0
                steering_override = 0.0
            else:
                pulse_steer = (
                    right_steer
                    if self.steering_pulse_index == 0
                    else min(self.FOLLOWUP_STEER_MAGNITUDE, right_steer)
                )
                steering_override = pulse_steer

        elif self.phase == self.ACQUIRE_RIGHT_LINE:
            steering_override = 0.0
            if target_line_confirmed:
                self.phase = self.RIGHT_LANE
                self.phase_distance_m = 0.0
                self.left_lane_distance_m = 0.0
                self.left_lane_target_distance_m = None
                self.last_obstacle_distance_m = None
                self._reset_target_line_tracking()
                self.return_reason = "DISTANCE"
                pid_reset = True
            elif (
                self.phase_distance_m >= self.LINE_CHECK_INTERVAL_M
                and new_line_observation
            ):
                self.phase = self.STEER_RIGHT
                self.phase_distance_m = 0.0
                self.steering_pulse_index += 1
                steering_override = min(
                    self.FOLLOWUP_STEER_MAGNITUDE, right_steer
                )
        
        elif self.phase == self.FAILED:
            safe_stop = True
            steering_override = 0.0

        # Evaluate after phase transitions so telemetry reflects the actual
        # active line-acquisition phase for this control update.
        search_limit_reached = (
            self.phase in {
                self.STEER_LEFT, self.ACQUIRE_LEFT_LINE,
                self.STEER_RIGHT, self.ACQUIRE_RIGHT_LINE,
            }
            and self.line_search_distance_m >= line_search_max_m
        )
        result = self.snapshot(sensors)
        result.update({
            "steering_override": steering_override,
            "safe_stop": safe_stop or self.phase == self.FAILED,
            "pid_reset": pid_reset,
            "obstacle_latched": self.obstacle_latched,
            "line_confirm_frames": self.line_confirm_frames,
            "target_line_confirmed": target_line_confirmed,
            "search_limit_reached": search_limit_reached,
        })
        return result



class AvisObstacleLaneClassifier:
    """ONNX Runtime adapter for RoadForge's AVIS two-head image classifier."""

    DEFAULT_PATH = (
        PROJECT_ROOT
        / "models_and_datasets"
        / "Avis_models"
        / "obstacle_and_lane_Avis_model.onnx"
    )

    def __init__(self, model_path: Path | str | None = None) -> None:
        import onnxruntime as ort

        self.model_path = Path(model_path or self.DEFAULT_PATH).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(
                "AVIS obstacle classifier was not found. Expected the RoadForge "
                f"ONNX model at: {self.model_path}"
            )

        available = set(ort.get_available_providers())
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers,
        )
        metadata = self.session.get_modelmeta().custom_metadata_map
        if metadata.get("task") != "classification":
            raise RuntimeError(
                "The AVIS ONNX model must declare task=classification in its metadata."
            )
        if metadata.get("roadforge_format") != "avis_multitask_classification_v1":
            raise RuntimeError(
                "The selected ONNX graph is not a supported RoadForge AVIS "
                "multi-task classifier."
            )

        try:
            self.mean = np.asarray(
                json.loads(metadata["normalization_mean"]), dtype=np.float32
            )
            self.std = np.asarray(
                json.loads(metadata["normalization_std"]), dtype=np.float32
            )
            self.tasks = json.loads(metadata["tasks"])
            self.input_size = int(metadata["input_size"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "The AVIS ONNX graph is missing valid preprocessing/task metadata."
            ) from exc

        if (
            self.mean.shape != (3,)
            or self.std.shape != (3,)
            or not np.all(np.isfinite(self.mean))
            or not np.all(np.isfinite(self.std))
            or np.any(self.std <= 0)
        ):
            raise RuntimeError("AVIS classifier RGB normalization metadata is invalid.")
        if self.input_size < 32 or self.input_size > 4096:
            raise RuntimeError(f"Invalid AVIS classifier input size: {self.input_size}.")

        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(float)":
            raise RuntimeError("AVIS classifier must expose one float32 image input.")
        input_shape = inputs[0].shape
        if len(input_shape) != 4 or input_shape[1] not in (3, "3"):
            raise RuntimeError(
                f"AVIS classifier input must be NCHW RGB; received {input_shape!r}."
            )
        for axis in (2, 3):
            expected = self.input_size
            if isinstance(input_shape[axis], int) and input_shape[axis] != expected:
                raise RuntimeError(
                    f"AVIS classifier input resolution is {input_shape[axis]}, "
                    f"but metadata declares {expected}."
                )

        if not isinstance(self.tasks, list):
            raise RuntimeError("AVIS classifier task metadata must be a list.")
        self.task_specs: dict[str, dict[str, Any]] = {}
        for task in self.tasks:
            if not isinstance(task, dict):
                raise RuntimeError("AVIS classifier contains an invalid task entry.")
            key = str(task.get("key", "")).strip()
            classes = task.get("classes")
            output_name = str(task.get("output_name", "")).strip()
            if not key or not isinstance(classes, list) or len(classes) < 2 or not output_name:
                raise RuntimeError("AVIS classifier task metadata is incomplete.")
            if key in self.task_specs:
                raise RuntimeError(f"Duplicate AVIS classifier task key: {key}.")
            self.task_specs[key] = task

        expected = {
            "obstacle_ahead": ["no", "yes"],
            "lane": ["right", "left"],
        }
        for key, classes in expected.items():
            task = self.task_specs.get(key)
            if task is None or task.get("classes") != classes:
                raise RuntimeError(
                    f"AVIS classifier must contain task {key!r} with classes {classes!r}."
                )

        output_names = [item["output_name"] for item in self.tasks]
        graph_outputs = [item.name for item in self.session.get_outputs()]
        if graph_outputs != output_names:
            raise RuntimeError(
                "AVIS classifier ONNX outputs do not match the task metadata order: "
                f"graph={graph_outputs!r}, metadata={output_names!r}."
            )

        self.input_name = inputs[0].name
        self.output_names = output_names
        self.providers = list(self.session.get_providers())
        print(
            "[AVIS] RoadForge multi-task classifier loaded: "
            f"{self.model_path} · input={self.input_size} · providers={self.providers}"
        )

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        logits = np.asarray(values, dtype=np.float32).reshape(-1)
        if logits.size < 2 or not np.all(np.isfinite(logits)):
            raise RuntimeError("AVIS classifier produced invalid task logits.")
        shifted = logits - float(np.max(logits))
        exponentials = np.exp(shifted)
        total = float(np.sum(exponentials))
        if not math.isfinite(total) or total <= 0.0:
            raise RuntimeError("AVIS classifier softmax could not be normalized.")
        return exponentials / total

    def predict(self, camera_frame: Any) -> dict[str, Any]:
        from PIL import Image

        frame = np.asarray(camera_frame)
        if frame.ndim != 3 or frame.shape[2] < 3 or frame.size == 0:
            raise ValueError(
                f"AVIS classifier expects a BGR camera frame; received {frame.shape!r}."
            )
        if frame.dtype != np.uint8:
            raise ValueError(
                f"AVIS classifier expects uint8 camera pixels; received {frame.dtype}."
            )

        # AVIS Engine supplies BGR. RoadForge trained on RGB PIL images using
        # Resize((224, 224)), ToTensor(), then ImageNet normalization.
        rgb = np.ascontiguousarray(frame[:, :, :3][:, :, ::-1])
        image = Image.fromarray(rgb, mode="RGB").resize(
            (self.input_size, self.input_size),
            resample=Image.Resampling.BILINEAR,
        )
        tensor = np.asarray(image, dtype=np.float32) / 255.0
        tensor = (tensor - self.mean.reshape(1, 1, 3)) / self.std.reshape(1, 1, 3)
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)

        raw_outputs = self.session.run(
            self.output_names,
            {self.input_name: tensor},
        )
        if len(raw_outputs) != len(self.tasks):
            raise RuntimeError(
                f"AVIS classifier returned {len(raw_outputs)} outputs; "
                f"expected {len(self.tasks)}."
            )

        predictions: dict[str, Any] = {}
        for task, raw in zip(self.tasks, raw_outputs):
            key = task["key"]
            classes = list(task["classes"])
            values = np.asarray(raw)
            expected_shape = (1, len(classes))
            if values.shape != expected_shape:
                raise RuntimeError(
                    f"AVIS classifier output {key!r} has shape {values.shape!r}; "
                    f"expected {expected_shape!r}."
                )
            probabilities = self._softmax(values[0])
            selected = int(np.argmax(probabilities))
            predictions[key] = {
                "label": classes[selected],
                "confidence": float(probabilities[selected]),
                "probabilities": {
                    label: float(probabilities[index])
                    for index, label in enumerate(classes)
                },
            }

        return {
            "valid": True,
            "backbone": str(self.session.get_modelmeta().custom_metadata_map.get("backbone", "unknown")),
            "input_size": self.input_size,
            "providers": list(self.providers),
            **predictions,
        }


class AvisCombinedVisionProcessor:
    """Keep lane-line geometry and AVIS classification in one worker request."""

    def __init__(self, lane_processor, obstacle_classifier: AvisObstacleLaneClassifier):
        object.__setattr__(self, "lane_processor", lane_processor)
        object.__setattr__(self, "obstacle_classifier", obstacle_classifier)

    def __getattr__(self, name):
        return getattr(self.lane_processor, name)

    def __setattr__(self, name, value):
        lane_processor = self.__dict__.get("lane_processor")
        if lane_processor is not None and hasattr(lane_processor, name):
            setattr(lane_processor, name, value)
            return
        object.__setattr__(self, name, value)

    def detect(self, semantic_frame, rgb_frame=None):
        result = self.lane_processor.detect(semantic_frame, rgb_frame=rgb_frame)
        if not isinstance(result, dict):
            raise TypeError("AVIS lane-line processor returned a non-dict result.")
        combined = dict(result)
        camera_frame = rgb_frame if rgb_frame is not None else semantic_frame
        combined["obstacle_classifier"] = self.obstacle_classifier.predict(camera_frame)
        debug = combined.get("debug")
        if isinstance(debug, dict):
            debug["obstacle_classifier"] = combined["obstacle_classifier"]
        return combined


class AvisModelObstacleLaneChangeController(AvisObstacleLaneChangeController):
    """Model-triggered right-to-left pass with right-side sensor-confirmed return."""

    WAIT_OBSTACLE_PASS = "WAIT_OBSTACLE_PASS"
    RETURN_DELAY = "RETURN_DELAY"

    def __init__(self) -> None:
        super().__init__()
        self.model_confirm_frames = 0
        self.model_rearm_frames = 0
        self.last_model_observation_id = None
        self.model_trigger_latched = False
        self.model_obstacle_probability: float | None = None
        self.model_lane_probability: float | None = None
        self.model_lane_label: str | None = None
        self.pass_sensor_stage = "WAITING_FOR_OBSTACLE"
        self.pass_sensor_seen_frames = 0
        self.pass_sensor_clear_frames = 0
        self.pass_sensor_started_at: float | None = None
        self.pass_sensor_cleared_at: float | None = None
        self.pass_sensor_timeout = False

    @staticmethod
    def _nonnegative_setting(
        settings: dict[str, Any], key: str, default: float
    ) -> float:
        try:
            value = float(settings.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) and value >= 0.0 else default

    def update(
        self,
        *,
        sensors: Any,
        speed_kmh: float,
        dt: float,
        left_x: int | float | None,
        right_x: int | float | None,
        settings: dict[str, Any],
        classifier_output: dict[str, Any] | None,
        allow_new_maneuver: bool = True,
        observation_id: Any = None,
        frame_width: int | float | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        current_time = time.monotonic() if now is None else float(now)
        if not math.isfinite(current_time):
            current_time = time.monotonic()

        detection_enabled = bool(settings.get("obstacle_lane_change_enabled", True))
        observation_is_new = (
            observation_id is not None
            and observation_id != self.last_model_observation_id
        )
        if observation_is_new:
            self.last_model_observation_id = observation_id
            obstacle = (classifier_output or {}).get("obstacle_ahead", {})
            lane = (classifier_output or {}).get("lane", {})
            obstacle_probs = obstacle.get("probabilities", {}) if isinstance(obstacle, dict) else {}
            lane_probs = lane.get("probabilities", {}) if isinstance(lane, dict) else {}
            try:
                obstacle_probability = float(obstacle_probs.get("yes", 0.0))
                no_probability = float(obstacle_probs.get("no", 0.0))
                lane_probability = float(lane_probs.get("left", 0.0))
                right_probability = float(lane_probs.get("right", 0.0))
            except (TypeError, ValueError):
                obstacle_probability = no_probability = lane_probability = right_probability = 0.0
            valid_output = bool((classifier_output or {}).get("valid")) and all(
                math.isfinite(value)
                for value in (
                    obstacle_probability, no_probability,
                    lane_probability, right_probability,
                )
            )
            obstacle_threshold = self._nonnegative_setting(
                settings, "obstacle_model_confidence_threshold", 0.65
            )
            lane_threshold = self._nonnegative_setting(
                settings, "obstacle_lane_confidence_threshold", 0.65
            )
            self.model_obstacle_probability = obstacle_probability if valid_output else None
            self.model_lane_probability = lane_probability if valid_output else None
            self.model_lane_label = (
                str(lane.get("label", "")).lower()
                if valid_output and isinstance(lane, dict) else None
            )
            obstacle_confirm_frames = max(
                1, int(self._nonnegative_setting(
                    settings, "obstacle_model_confirm_frames", 3
                ))
            )
            rearm_frames_required = max(
                1, int(self._nonnegative_setting(
                    settings, "obstacle_model_rearm_frames", 3
                ))
            )
            model_requests_left_pass = (
                valid_output
                and obstacle_probability >= obstacle_threshold
                and no_probability < obstacle_probability
                and lane_probability >= lane_threshold
                and lane_probability > right_probability
            )
            model_reports_clear = (
                valid_output
                and no_probability >= obstacle_probability
            )

            if (
                self.phase == self.RIGHT_LANE
                and self.model_trigger_latched
                and model_reports_clear
            ):
                self.model_rearm_frames += 1
                if self.model_rearm_frames >= rearm_frames_required:
                    self.model_trigger_latched = False
                    self.model_rearm_frames = 0
            elif self.phase == self.RIGHT_LANE and not model_reports_clear:
                self.model_rearm_frames = 0

            if (
                detection_enabled
                and allow_new_maneuver
                and self.phase == self.RIGHT_LANE
                and not self.model_trigger_latched
            ):
                if model_requests_left_pass:
                    self.model_confirm_frames += 1
                else:
                    self.model_confirm_frames = 0
                if self.model_confirm_frames >= obstacle_confirm_frames:
                    self.model_trigger_latched = True
                    self.model_confirm_frames = 0
                    self._start_target_line_tracking("left", observation_id)
                    self.phase = self.STEER_LEFT
                    self.phase_distance_m = 0.0
                    self.left_lane_distance_m = 0.0
                    self.left_lane_target_distance_m = None
                    self.last_obstacle_sensor = "MODEL"
                    self.return_reason = "MODEL_OBSTACLE_AHEAD"
            elif not model_requests_left_pass:
                self.model_confirm_frames = 0

        # Disable the legacy three-sensor trigger and distance-based return.
        # The inherited controller still supplies its pulsed steering and
        # stable target-line acquisition for both lane changes.
        base_settings = dict(settings)
        base_settings["obstacle_lane_change_enabled"] = False
        base_settings["obstacle_front_threshold_cm"] = 0.0
        base_settings["obstacle_sensor_clear_threshold_cm"] = 0.0
        base_settings["obstacle_clearance_after_pass_m"] = 0.0
        original_phase = self.phase
        pass_wait_active = original_phase in {
            self.WAIT_OBSTACLE_PASS, self.RETURN_DELAY
        }
        if pass_wait_active:
            self.phase = self.LEFT_LANE
        result = super().update(
            sensors=sensors,
            speed_kmh=speed_kmh,
            dt=dt,
            left_x=left_x,
            right_x=right_x,
            settings=base_settings,
            allow_new_maneuver=False,
            observation_id=observation_id,
            frame_width=frame_width,
        )
        if pass_wait_active and self.phase == self.LEFT_LANE:
            self.phase = original_phase

        pid_reset = bool(result.get("pid_reset"))
        if original_phase in {self.STEER_LEFT, self.ACQUIRE_LEFT_LINE} and self.phase == self.LEFT_LANE:
            # The left lane is confirmed by the same temporal ROI tracker used
            # by the CARLA lane-change manager's completion checks.
            self.phase = self.WAIT_OBSTACLE_PASS
            self.pass_sensor_stage = "WAITING_FOR_OBSTACLE"
            self.pass_sensor_seen_frames = 0
            self.pass_sensor_clear_frames = 0
            self.pass_sensor_started_at = current_time
            self.pass_sensor_cleared_at = None
            self.pass_sensor_timeout = False
            self.left_lane_target_distance_m = None

        right_sensor_cm = self.sensor_distances_cm(sensors)[2]
        presence_threshold = self._nonnegative_setting(
            settings, "obstacle_pass_sensor_presence_threshold_cm", 250.0
        )
        clear_threshold = self._nonnegative_setting(
            settings, "obstacle_pass_sensor_clear_threshold_cm", 550.0
        )
        presence_frames_required = max(
            1, int(self._nonnegative_setting(
                settings, "obstacle_pass_sensor_presence_frames", 2
            ))
        )
        clear_frames_required = max(
            1, int(self._nonnegative_setting(
                settings, "obstacle_pass_sensor_clear_frames", 3
            ))
        )
        return_delay = self._nonnegative_setting(
            settings, "obstacle_return_delay_seconds", 2.0
        )
        pass_timeout = self._nonnegative_setting(
            settings, "obstacle_pass_sensor_timeout_seconds", 30.0
        )

        if self.phase == self.WAIT_OBSTACLE_PASS:
            if right_sensor_cm is not None and right_sensor_cm <= presence_threshold:
                self.pass_sensor_seen_frames += 1
                if self.pass_sensor_seen_frames >= presence_frames_required:
                    self.pass_sensor_stage = "OBSTACLE_SEEN_ON_RIGHT"
            else:
                self.pass_sensor_seen_frames = 0

            if self.pass_sensor_stage == "OBSTACLE_SEEN_ON_RIGHT":
                if right_sensor_cm is not None and right_sensor_cm >= clear_threshold:
                    self.pass_sensor_clear_frames += 1
                else:
                    self.pass_sensor_clear_frames = 0
                if self.pass_sensor_clear_frames >= clear_frames_required:
                    self.pass_sensor_cleared_at = current_time
                    self.pass_sensor_stage = "OBSTACLE_CLEARED"
                    self.phase = self.RETURN_DELAY

            if (
                self.phase == self.WAIT_OBSTACLE_PASS
                and self.pass_sensor_started_at is not None
                and current_time - self.pass_sensor_started_at >= pass_timeout
            ):
                self.phase = self.FAILED
                self.pass_sensor_timeout = True
                self.return_reason = "RIGHT_SENSOR_PASS_CONFIRMATION_TIMEOUT"

        if (
            self.phase == self.RETURN_DELAY
            and self.pass_sensor_cleared_at is not None
            and current_time - self.pass_sensor_cleared_at >= return_delay
        ):
            self._begin_right_return("RIGHT_SENSOR_CONFIRMED_PASS", observation_id)
            # The inherited CARLA-style helper resets its default reason to
            # DISTANCE; preserve the sensor-confirmed reason for AVIS telemetry.
            self.return_reason = "RIGHT_SENSOR_CONFIRMED_PASS"
            pid_reset = True

        # The side-angle is selected only after the car has acquired the left
        # lane; restore the normal sensor spread before steering back right.
        sensor_angle_override = (
            int(settings.get("obstacle_pass_sensor_angle", settings.get("sensor_angle", 40)))
            if self.phase in {self.WAIT_OBSTACLE_PASS, self.RETURN_DELAY}
            else int(settings.get("sensor_angle", 40))
        )
        search_limit = self._nonnegative_setting(
            settings, "obstacle_lane_line_search_max_distance_m", 40.0
        )
        search_limit_reached = bool(result.get("search_limit_reached"))
        if (
            self.phase in {
                self.STEER_LEFT, self.ACQUIRE_LEFT_LINE,
                self.STEER_RIGHT, self.ACQUIRE_RIGHT_LINE,
            }
            and search_limit > 0
            and self.line_search_distance_m >= search_limit
        ):
            self.phase = self.FAILED
            self.return_reason = "TARGET_LANE_LINE_SEARCH_TIMEOUT"

        safe_stop = self.phase == self.FAILED
        if safe_stop:
            result["steering_override"] = 0.0
        result.update({
            "phase": self.phase,
            "lane_side": (
                "left"
                if self.phase in {
                    self.STEER_RIGHT, self.ACQUIRE_RIGHT_LINE,
                    self.LEFT_LANE, self.WAIT_OBSTACLE_PASS, self.RETURN_DELAY,
                }
                else "right"
            ),
            "target_lane": "right" if self.phase in {
                self.STEER_RIGHT, self.ACQUIRE_RIGHT_LINE,
            } else "left" if self.phase in {
                self.STEER_LEFT, self.ACQUIRE_LEFT_LINE,
                self.LEFT_LANE, self.WAIT_OBSTACLE_PASS, self.RETURN_DELAY,
            } else "right",
            "pid_reset": pid_reset,
            "safe_stop": safe_stop,
            "obstacle_latched": self.model_trigger_latched,
            "obstacle_trigger_sensor": "MODEL" if self.model_trigger_latched else self.last_obstacle_sensor,
            "return_reason": self.return_reason,
            "obstacle_model_confirm_frames": self.model_confirm_frames,
            "obstacle_model_rearm_frames": self.model_rearm_frames,
            "obstacle_model_probability": self.model_obstacle_probability,
            "obstacle_lane_probability": self.model_lane_probability,
            "obstacle_lane_prediction": self.model_lane_label,
            "pass_sensor_stage": self.pass_sensor_stage,
            "pass_sensor_right_cm": right_sensor_cm,
            "pass_sensor_seen_frames": self.pass_sensor_seen_frames,
            "pass_sensor_clear_frames": self.pass_sensor_clear_frames,
            "pass_sensor_cleared_at": self.pass_sensor_cleared_at,
            "return_delay_remaining_s": (
                max(0.0, return_delay - (current_time - self.pass_sensor_cleared_at))
                if self.phase == self.RETURN_DELAY and self.pass_sensor_cleared_at is not None
                else 0.0
            ),
            "pass_sensor_timeout": self.pass_sensor_timeout,
            "sensor_angle_override": sensor_angle_override,
            "search_limit_reached": search_limit_reached,
        })
        return result

def _candidate_api_directories(api_path: str | None) -> list[Path]:
    candidates: list[Path] = []
    if api_path:
        supplied = Path(api_path).expanduser()
        candidates.append(supplied.parent if supplied.is_file() else supplied)

    environment_path = os.environ.get("AVIS_ENGINE_API_PATH")
    if environment_path:
        supplied = Path(environment_path).expanduser()
        candidates.append(supplied.parent if supplied.is_file() else supplied)

    candidates.extend(
        (
            PROJECT_ROOT / "AVIS-Engine-Python-API",
            PROJECT_ROOT / "avis-engine-python-api",
            PROJECT_ROOT.parent / "AVIS-Engine-Python-API",
        )
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def load_avis_api(api_path: str | None = None) -> ModuleType:
    """Load the vendor API while isolating its top-level utility module."""
    engine_file: Optional[Path] = None
    for directory in _candidate_api_directories(api_path):
        candidate = directory / "avisengine.py"
        if candidate.is_file():
            engine_file = candidate.resolve()
            break

    if engine_file is None:
        try:
            discovered = importlib.util.find_spec("avisengine")
        except (ImportError, ValueError):
            discovered = None
        if discovered is not None and discovered.origin:
            engine_file = Path(discovered.origin).resolve()

    if engine_file is None:
        raise RuntimeError(
            "The AVIS Python API was not found. Clone the 1.2.4-compatible "
            "FIRA-ACL branch into the project folder, or pass --avis-api-path:\n"
            "git clone --branch FIRA-ACL --single-branch "
            "https://github.com/AVISEngine/AVIS-Engine-Python-API.git"
        )

    api_utils_file = engine_file.with_name("utils.py")
    previous_utils = sys.modules.get("utils")
    previous_engine = sys.modules.get("avisengine")
    try:
        if api_utils_file.is_file():
            utils_spec = importlib.util.spec_from_file_location("utils", api_utils_file)
            if utils_spec is None or utils_spec.loader is None:
                raise ImportError(f"Could not load AVIS utility module: {api_utils_file}")
            utils_module = importlib.util.module_from_spec(utils_spec)
            sys.modules["utils"] = utils_module
            utils_spec.loader.exec_module(utils_module)

        engine_spec = importlib.util.spec_from_file_location("avisengine", engine_file)
        if engine_spec is None or engine_spec.loader is None:
            raise ImportError(f"Could not load AVIS API module: {engine_file}")
        engine_module = importlib.util.module_from_spec(engine_spec)
        sys.modules["avisengine"] = engine_module
        engine_spec.loader.exec_module(engine_module)
    finally:
        if previous_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = previous_utils
        if previous_engine is None:
            sys.modules.pop("avisengine", None)
        else:
            sys.modules["avisengine"] = previous_engine

    if not callable(getattr(engine_module, "Car", None)):
        raise RuntimeError(f"The AVIS API at {engine_file} does not expose Car().")
    return engine_module


class AvisEngineClient:
    """Defensive adapter around the vendor's synchronous TCP client."""

    def __init__(self, api_module: ModuleType, host: str, port: int) -> None:
        self.car = api_module.Car()
        self._connected = False
        self._closed = False

        # The vendor destructor calls stop() even after explicit shutdown.
        # This adapter owns shutdown and makes that destructor path harmless.
        self.car.stop = lambda: None
        try:
            self._connected = bool(self.car.connect(host, int(port)))
        except Exception:
            self._close_socket()
            self._closed = True
            raise

        if not self._connected:
            self._close_socket()
            self._closed = True
            raise ConnectionError(
                f"Could not connect to AVIS Engine at {host}:{int(port)}. "
                "Check that the simulator server is running and its IP/port match."
            )

    def set_sensor_angle(self, angle: int) -> None:
        if self._closed:
            raise RuntimeError("The AVIS connection is already closed.")
        self.car.setSensorAngle(int(angle))

    def read_data(self) -> tuple[Any, float, Any]:
        if self._closed or not self._connected:
            raise RuntimeError("The AVIS connection is not active.")
        self.car.getData()
        image = self.car.getImage()
        speed = self.car.getSpeed()
        sensors = self.car.getSensors()
        try:
            speed_kmh = float(speed)
        except (TypeError, ValueError):
            speed_kmh = 0.0
        if not math.isfinite(speed_kmh):
            speed_kmh = 0.0
        return image, max(0.0, speed_kmh), sensors

    def send_control(self, throttle: float, steering: float) -> None:
        if self._closed or not self._connected:
            raise RuntimeError("The AVIS connection is not active.")
        speed_command, steering_command = normalize_avis_controls(throttle, steering)
        self.car.setSpeed(speed_command)
        self.car.setSteering(steering_command)

    def send_manual_control(self, throttle: float, steering: float) -> None:
        if self._closed or not self._connected:
            raise RuntimeError("The AVIS connection is not active.")
        speed_command, steering_command = normalize_avis_manual_controls(
            throttle, steering
        )
        self.car.setSpeed(speed_command)
        self.car.setSteering(steering_command)

    def _close_socket(self) -> None:
        sock = getattr(self.car, "sock", None)
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._connected:
            try:
                self.car.setSpeed(0)
            except Exception:
                pass
            try:
                self.car.setSteering(0)
            except Exception:
                pass
            sock = getattr(self.car, "sock", None)
            if sock is not None:
                try:
                    sock.sendall(b"stop")
                except OSError:
                    pass
        self._close_socket()

    def __enter__(self) -> "AvisEngineClient":
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the existing StableV2 vision/PID driving pipeline on AVIS Engine."
    )
    parser.add_argument("--avis", action="store_true", help="Select the AVIS Engine runtime.")
    parser.add_argument(
        "--avis-host",
        default=os.environ.get("AVIS_HOST", "127.0.0.1"),
        help="AVIS server IP address (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--avis-port",
        type=int,
        default=int(os.environ.get("AVIS_PORT", "25001")),
        help="AVIS server TCP port (default: 25001).",
    )
    parser.add_argument(
        "--avis-api-path",
        default=None,
        help="Path to the AVIS Python API repository or avisengine.py file.",
    )
    parser.add_argument(
        "--avis-sensor-angle",
        type=int,
        default=40,
        help="Angle between the three AVIS distance sensors (default: 40 degrees).",
    )
    parser.add_argument(
        "--avis-startup-delay",
        type=float,
        default=3.0,
        help="Seconds to wait after connecting before reading camera data.",
    )
    parser.add_argument(
        "--avis-headless",
        action="store_true",
        help="Disable the local OpenCV preview window; Flask remains enabled.",
    )
    parser.add_argument(
        "--avis-web-host",
        default=os.environ.get("AVIS_WEB_HOST", "127.0.0.1"),
        help="AVIS dashboard bind address (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--avis-web-port",
        type=int,
        default=int(os.environ.get("AVIS_WEB_PORT", "5001")),
        help="AVIS dashboard port (default: 5001).",
    )
    parser.add_argument(
        "--avis-no-web",
        action="store_true",
        help="Disable the AVIS-only Flask dashboard.",
    )
    return parser


AVIS_ROI_CONFIG_KEYS = {
    "ll_top_roi": "LL_TOP_ROI",
    "ll_bottom_roi": "LL_BOTTOM_ROI",
    "ll_left_roi": "LL_LEFT_ROI",
    "ll_right_roi": "LL_RIGHT_ROI",
    "rl_top_roi": "RL_TOP_ROI",
    "rl_bottom_roi": "RL_BOTTOM_ROI",
    "rl_left_roi": "RL_LEFT_ROI",
    "rl_right_roi": "RL_RIGHT_ROI",
}

AVIS_SHARED_VISION_CONFIG_KEYS = {
    "min_side_pixels": "MIN_SIDE_PIXELS",
    "lane_offset_ratio": "FALLBACK_LANE_OFFSET_RATIO",
    "center_smoothing": "LANE_CENTER_SMOOTH_ALPHA",
}


def _apply_avis_roi_settings(conf: ModuleType, settings: dict[str, Any]) -> None:
    """Apply AVIS-only ROI and lane geometry values to this process's config."""
    for mapping in (AVIS_ROI_CONFIG_KEYS, AVIS_SHARED_VISION_CONFIG_KEYS):
        for setting_key, config_key in mapping.items():
            if setting_key in settings:
                setattr(conf, config_key, float(settings[setting_key]))


def _build_avis_settings(conf: ModuleType, sensor_angle: int) -> dict[str, Any]:
    return {
        "vision_backend": "ultralytics",
        "control_loop_hz": float(conf.CONTROL_LOOP_HZ),
        "inference_interval_ms": max(
            10.0, float(conf.VISION_INFERENCE_INTERVAL_SECONDS) * 1000.0
        ),
        "sensor_angle": int(sensor_angle),
        "min_speed_kmh": float(conf.MIN_SPEED_KMH),
        "max_speed_kmh": float(conf.MAX_SPEED_KMH),
        "acceleration_mps2": float(conf.ACCELERATION_MPS2),
        "kp": float(conf.KP),
        "ki": float(conf.KI),
        "kd": float(conf.KD),
        "steer_limit": float(conf.STEER_LIMIT),
        "max_steer_step": float(conf.MAX_STEER_STEP),
        "min_side_pixels": int(conf.MIN_SIDE_PIXELS),
        "lane_offset_ratio": float(conf.FALLBACK_LANE_OFFSET_RATIO),
        "center_smoothing": float(conf.LANE_CENTER_SMOOTH_ALPHA),
        "ll_top_roi": float(conf.LL_TOP_ROI),
        "ll_bottom_roi": float(conf.LL_BOTTOM_ROI),
        "ll_left_roi": float(conf.LL_LEFT_ROI),
        "ll_right_roi": float(conf.LL_RIGHT_ROI),
        "rl_top_roi": float(conf.RL_TOP_ROI),
        "rl_bottom_roi": float(conf.RL_BOTTOM_ROI),
        "rl_left_roi": float(conf.RL_LEFT_ROI),
        "rl_right_roi": float(conf.RL_RIGHT_ROI),
    }


def _create_avis_vision_processor(
    backend: str,
    settings: dict[str, Any],
    adaptive_roi_controller=None,
):
    if backend == "classical":
        from avis_classical_vision import AvisClassicalVisionProcessor

        return AvisClassicalVisionProcessor(
            settings,
            adaptive_roi_controller=adaptive_roi_controller,
        )

    from vision.city_vision_processing import VisionProcessor
    from avis_lane_line_selection import TemporalLaneLineSelector

    class AvisVisionProcessor(VisionProcessor):
        """AVIS model adapter that extracts and visualizes lane lines only."""

        def detect(self, semantic_frame, rgb_frame=None):
            # Apply adaptive AVIS ROI coordinates before resolving crop bounds.
            if self.adaptive_roi_controller is not None:
                import config_city as avis_conf

                _apply_avis_roi_settings(
                    avis_conf,
                    self.adaptive_roi_controller.active_settings(),
                )

            frame = rgb_frame if rgb_frame is not None else semantic_frame
            self._avis_roi_bounds = {}
            self._avis_line_selection_stats = {}
            if frame is not None and getattr(frame, "ndim", 0) >= 2:
                import config_city as avis_conf

                height, width = frame.shape[:2]
                self._avis_roi_bounds = {
                    "ll": (
                        int(float(avis_conf.LL_TOP_ROI) * height),
                        int(float(avis_conf.LL_BOTTOM_ROI) * height),
                        int(float(avis_conf.LL_LEFT_ROI) * width),
                        int(float(avis_conf.LL_RIGHT_ROI) * width),
                    ),
                    "rl": (
                        int(float(avis_conf.RL_TOP_ROI) * height),
                        int(float(avis_conf.RL_BOTTOM_ROI) * height),
                        int(float(avis_conf.RL_LEFT_ROI) * width),
                        int(float(avis_conf.RL_RIGHT_ROI) * width),
                    ),
                }
                signature = (
                    (height, width),
                    tuple(
                        (key, settings.get(key))
                        for key in (
                            "hough_threshold",
                            "hough_min_line_length_ratio",
                            "hough_max_line_gap",
                            "min_lane_angle_deg",
                            "max_lane_angle_deg",
                        )
                    ),
                )
                if signature != self._avis_selection_signature:
                    for selector in self._avis_line_selectors.values():
                        selector.reset()
                    self._avis_selection_signature = signature

            result = super().detect(semantic_frame, rgb_frame=rgb_frame)
            if isinstance(result, dict):
                debug = result.get("debug")
                if isinstance(debug, dict):
                    for side, prefix in (("ll", "left"), ("rl", "right")):
                        stats = self._avis_line_selection_stats.get(side, {})
                        debug[f"{prefix}_hough_segments"] = int(stats.get("segments", 0))
                        debug[f"{prefix}_hough_candidates"] = int(stats.get("candidates", 0))
                        debug[f"{prefix}_selected_line_x"] = stats.get("selected_x")
                        debug[f"{prefix}_selected_line_quality"] = stats.get("quality")
                        debug[f"{prefix}_selected_line_stable_frames"] = int(
                            stats.get("stable_frames", 0)
                        )
                        debug[f"{prefix}_selected_line_missed_frames"] = int(
                            stats.get("missed_frames", 0)
                        )
                        debug[f"{prefix}_pending_switch_frames"] = int(
                            stats.get("pending_switch_frames", 0)
                        )
                        debug[f"{prefix}_selected_line_switched"] = bool(
                            stats.get("switched", False)
                        )
                        debug[f"{prefix}_selected_line_predicted"] = bool(
                            stats.get("predicted", False)
                        )

            if (
                self.adaptive_roi_controller is not None
                and frame is not None
                and getattr(frame, "ndim", 0) >= 2
            ):
                debug = result.get("debug")
                if isinstance(debug, dict):
                    self.adaptive_roi_controller.observe(
                        debug.get("left_x"),
                        debug.get("right_x"),
                        int(frame.shape[1]),
                    )
                    debug["adaptive_roi"] = self.adaptive_roi_controller.snapshot()
            return result

        def _boundary_from_roi(
            self,
            full_mask,
            top,
            bottom,
            left,
            right,
        ):
            bounds = (int(top), int(bottom), int(left), int(right))
            side = next(
                (
                    name
                    for name, expected in self._avis_roi_bounds.items()
                    if expected == bounds
                ),
                None,
            )
            if side is not None and bottom > top and right > left:
                selection = self._avis_line_selectors[side].select(
                    full_mask[top:bottom, left:right],
                    settings,
                    x_offset=float(left),
                )
                self._avis_line_selection_stats[side] = {
                    "segments": selection.segment_count,
                    "candidates": selection.candidate_count,
                    "selected_x": selection.selected_x,
                    "quality": selection.quality,
                    "stable_frames": selection.stable_frames,
                    "missed_frames": selection.missed_frames,
                    "pending_switch_frames": selection.pending_switch_frames,
                    "switched": selection.switched,
                    "predicted": selection.predicted,
                }
                # Always replace raw pixels; never reintroduce all candidate
                # lines when Hough selection has a brief dropout.
                full_mask[top:bottom, left:right] = selection.mask

            return super()._boundary_from_roi(
                full_mask, top, bottom, left, right
            )

        @staticmethod
        def _class_ids_from_names(
            class_names: dict[int, str],
        ) -> tuple[int, None, None]:
            """Require only lane_lines; all other semantic classes are ignored."""
            lane_ids = [
                class_id
                for class_id, name in class_names.items()
                if name == "lane_lines"
            ]
            if len(lane_ids) != 1:
                raise RuntimeError(
                    "The AVIS Ultralytics checkpoint must contain exactly one "
                    "'lane_lines' class; drivable-area and vehicle classes are "
                    "not required. "
                    f"Found names {class_names!r}"
                )
            return lane_ids[0], None, None

        def _extract_masks_from_ultralytics(self, camera_frame):
            """Extract only lane-line pixels; never construct other class masks."""
            import numpy as np

            predictions = self._predict_ultralytics(camera_frame)
            if not predictions:
                raise RuntimeError("Ultralytics returned no prediction result.")
            semantic_output = getattr(predictions[0], "semantic_mask", None)
            if semantic_output is None or getattr(semantic_output, "data", None) is None:
                raise RuntimeError(
                    "The configured checkpoint did not return semantic_mask. "
                    "AVIS requires an Ultralytics semantic-segmentation checkpoint "
                    "with a lane_lines class."
                )

            semantic_data = semantic_output.data
            if hasattr(semantic_data, "detach"):
                semantic_data = semantic_data.detach()
            if hasattr(semantic_data, "cpu"):
                semantic_data = semantic_data.cpu()
            if hasattr(semantic_data, "numpy"):
                semantic_data = semantic_data.numpy()

            class_map = np.asarray(semantic_data)
            # Remove only leading batch dimensions. A full squeeze would turn
            # valid one-row or one-column masks into one-dimensional arrays.
            while class_map.ndim > 2 and class_map.shape[0] == 1:
                class_map = class_map[0]
            if class_map.ndim != 2:
                raise RuntimeError(
                    "AVIS semantic output must be a two-dimensional class-ID map; "
                    f"received shape {class_map.shape!r}"
                )

            frame_height, frame_width = np.asarray(camera_frame).shape[:2]
            if class_map.shape != (frame_height, frame_width):
                import cv2

                class_map = cv2.resize(
                    class_map.astype(np.float32),
                    (frame_width, frame_height),
                    interpolation=cv2.INTER_NEAREST,
                )

            lane_mask = (class_map == self.lane_class_id).astype(np.uint8) * 255
            return None, lane_mask, None

        def __init__(self, mode: str):
            super().__init__(mode)
            self.adaptive_roi_controller = adaptive_roi_controller
            self._avis_line_selectors = {
                "ll": TemporalLaneLineSelector("ll"),
                "rl": TemporalLaneLineSelector("rl"),
            }
            self._avis_selection_signature = None
            self.last_lane_half_width: float | None = None
            print(
                "[AVIS] Model inference is used for lane_lines only; "
                "drivable-area and vehicle masks are disabled."
            )

        def _pick_sticky_lane(
            self,
            left_x,
            right_x,
            frame_center,
            lane_offset_px,
        ):
            """Use AVIS midpoint centring for model and classical lane masks."""
            if left_x is not None and right_x is not None and right_x > left_x:
                self.last_lane_half_width = (right_x - left_x) / 2.0
                self.active_side = None
                return "both", left_x + self.last_lane_half_width

            # If one lane marking disappears temporarily, infer the target
            # from the most recently measured lane half-width. Only use the
            # configured fixed ratio before any complete lane width is known.
            offset_px = (
                self.last_lane_half_width
                if self.last_lane_half_width is not None
                else lane_offset_px
            )

            if self.active_side == "right":
                if right_x is not None:
                    return "only_right", right_x - offset_px
                if left_x is not None:
                    self.active_side = "left"
                    return "only_left", left_x + offset_px
                return (
                    "none",
                    self.last_lane_center
                    if self.last_lane_center is not None
                    else frame_center,
                )

            if self.active_side == "left":
                if left_x is not None:
                    return "only_left", left_x + offset_px
                if right_x is not None:
                    self.active_side = "right"
                    return "only_right", right_x - offset_px
                return (
                    "none",
                    self.last_lane_center
                    if self.last_lane_center is not None
                    else frame_center,
                )

            if right_x is not None:
                self.active_side = "right"
                return "only_right", right_x - offset_px
            if left_x is not None:
                self.active_side = "left"
                return "only_left", left_x + offset_px
            return (
                "none",
                self.last_lane_center
                if self.last_lane_center is not None
                else frame_center,
            )

        def _draw_debug(
            self,
            frame,
            line_mask,
            drivable_mask,
            left_x,
            right_x,
            lane_center,
            lane_type,
            vehicles_mask=None,
        ):
            import cv2
            import numpy as np

            # Keep the shared frame-centre, ROI and boundary annotations, but
            # explicitly disable every non-lane segmentation overlay.
            visual = super()._draw_debug(
                frame=frame,
                line_mask=None,
                drivable_mask=None,
                left_x=left_x,
                right_x=right_x,
                lane_center=lane_center,
                lane_type=lane_type,
                vehicles_mask=None,
            )
            if line_mask is None:
                return visual

            lane = np.asarray(line_mask)
            if lane.ndim == 3:
                lane_pixels = np.any(lane > 0, axis=2)
            elif lane.ndim == 2:
                lane_pixels = lane > 0
            else:
                return visual

            height, width = visual.shape[:2]
            if lane_pixels.shape != (height, width):
                lane_pixels = cv2.resize(
                    lane_pixels.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0

            # OpenCV uses BGR: (0, 0, 255) is pure red.
            visual[lane_pixels] = (0, 0, 255)
            return visual

    return AvisVisionProcessor("ultralytics")


def _encode_avis_preview(cv2_module, image) -> bytes | None:
    if image is None:
        return None
    try:
        ok, encoded = cv2_module.imencode(
            ".jpg",
            image,
            [int(cv2_module.IMWRITE_JPEG_QUALITY), 78],
        )
    except Exception:
        return None
    return encoded.tobytes() if ok else None


def _colorize_avis_lane_mask(cv2_module, np_module, line_mask, frame_shape):
    """Create a BGR debug preview where every selected lane pixel is pure red."""
    height, width = int(frame_shape[0]), int(frame_shape[1])
    preview = np_module.zeros((height, width, 3), dtype=np_module.uint8)
    if line_mask is None:
        return preview

    mask = np_module.asarray(line_mask)
    if mask.ndim == 3:
        selected = np_module.any(mask > 0, axis=2)
    elif mask.ndim == 2:
        selected = mask > 0
    else:
        return preview

    if selected.shape != (height, width):
        selected = cv2_module.resize(
            selected.astype(np_module.uint8),
            (width, height),
            interpolation=cv2_module.INTER_NEAREST,
        ) > 0

    preview[selected] = (0, 0, 255)
    return preview


def run_avis(args: argparse.Namespace) -> int:
    client: Optional[AvisEngineClient] = None
    inference_worker = None
    dashboard_server = None
    cv2_module = None
    dashboard_state = None

    try:
        import config_city as conf

        from avis_dashboard import AvisDashboardServer, AvisDashboardState

        profile = load_avis_profile()
        apply_avis_profile(profile, conf)

        # Load the vendor API before project modules that may use a local
        # package with the same top-level utility-module name.
        avis_api = load_avis_api(args.avis_api_path)

        import cv2
        import numpy as np
        from controllers.speed_controller import SpeedPIDController
        from trajectory.worker import InferenceWorker

        cv2_module = cv2
        print(f"[AVIS] Loading AVIS-specific profile: {AVIS_PROFILE_NAME}")

        dashboard_state = AvisDashboardState(
            initial_settings=_build_avis_settings(conf, args.avis_sensor_angle),
            settings_path=PROJECT_ROOT / "avis_settings.json",
        )
        settings, settings_revision = dashboard_state.settings_snapshot()
        _apply_avis_roi_settings(conf, settings)
        if settings["vision_backend"] == "ultralytics":
            print(f"[AVIS] Lane-line checkpoint: {conf.MODEL_PATH}")

        obstacle_classifier = AvisObstacleLaneClassifier()
        from avis_adaptive_roi import AvisAdaptiveROIController

        adaptive_roi_controller = AvisAdaptiveROIController(settings)
        try:
            lane_vision_processor = _create_avis_vision_processor(
                settings["vision_backend"],
                settings,
                adaptive_roi_controller=adaptive_roi_controller,
            )
        except (FileNotFoundError, ImportError) as exc:
            if settings["vision_backend"] != "ultralytics":
                raise
            print(
                "[AVIS] Ultralytics model is unavailable; switching this AVIS "
                f"session to classical lane detection: {exc}"
            )
            settings = dashboard_state.update_settings(
                {"vision_backend": "classical"}
            )
            settings_revision = dashboard_state.settings_snapshot()[1]
            adaptive_roi_controller.configure(settings)
            lane_vision_processor = _create_avis_vision_processor(
                "classical",
                settings,
                adaptive_roi_controller=adaptive_roi_controller,
            )

        vision_processor = AvisCombinedVisionProcessor(
            lane_vision_processor, obstacle_classifier
        )
        print("[AVIS] Obstacle detection and lane-change direction use the RoadForge multi-task ONNX classifier.")
        inference_worker = InferenceWorker(vision_processor=vision_processor)
        inference_worker.start()

        client = AvisEngineClient(avis_api, args.avis_host, args.avis_port)
        client.send_control(0.0, 0.0)
        client.set_sensor_angle(int(settings["sensor_angle"]))

        steering_controller = AvisSteeringController(
            kp=float(settings["kp"]),
            ki=float(settings["ki"]),
            kd=float(settings["kd"]),
            steer_limit=float(settings["steer_limit"]),
            max_steer_step=float(settings["max_steer_step"]),
        )
        speed_controller = SpeedPIDController(
            acceleration_mps2=float(settings["acceleration_mps2"]),
            min_speed_kmh=float(settings["min_speed_kmh"]),
            max_speed_kmh=float(settings["max_speed_kmh"]),
        )
        obstacle_lane_change = AvisModelObstacleLaneChangeController()

        if not bool(conf.AUTO_MODE_DEFAULT):
            dashboard_state.set_mode("stopped")

        if not args.avis_no_web:
            dashboard_server = AvisDashboardServer(
                dashboard_state,
                host=args.avis_web_host,
                port=args.avis_web_port,
            )
            dashboard_server.start()
            dashboard_address = (
                "127.0.0.1"
                if args.avis_web_host in {"0.0.0.0", "::"}
                else args.avis_web_host
            )
            print(
                f"[AVIS WEB] Dashboard: http://{dashboard_address}:{args.avis_web_port}"
            )
            if args.avis_web_host not in {"127.0.0.1", "localhost", "::1"}:
                print(
                    "[AVIS WEB] WARNING: the dashboard has no authentication. "
                    "Do not expose it to untrusted networks."
                )

        print(f"[AVIS] Connected to {args.avis_host}:{args.avis_port}.")
        print(
            f"[AVIS] Waiting {max(0.0, args.avis_startup_delay):.1f}s "
            "for the simulator camera stream..."
        )
        time.sleep(max(0.0, float(args.avis_startup_delay)))

        control_period = 1.0 / float(settings["control_loop_hz"])
        inference_interval = float(settings["inference_interval_ms"]) / 1000.0
        active_backend = str(settings["vision_backend"])
        active_sensor_angle = int(settings["sensor_angle"])
        last_inference_submit = 0.0
        last_vision_result = None
        last_dashboard_publish = 0.0
        frame_id = 0
        last_error_message = None
        consecutive_connection_errors = 0
        last_control_time = time.perf_counter()
        recovery_steering = 0.0
        next_tick = last_control_time
        frame_read_fps = AvisFpsMonitor(window_seconds=2.0)
        inference_fps = AvisFpsMonitor(window_seconds=2.0)
        control_loop_fps = AvisFpsMonitor(window_seconds=2.0)
        last_inference_fps_timestamp: float | None = None

        print("[AVIS] AVIS runtime is ready.")
        print("[AVIS] Use the web dashboard for auto/manual mode and live tuning.")
        print("[AVIS] Press Q or Esc in the local preview window to stop.")

        while True:
            control_loop_fps.record()
            try:
                frame, current_speed, sensors = client.read_data()
                if isinstance(frame, np.ndarray) and frame.size > 0:
                    frame_read_fps.record()
                consecutive_connection_errors = 0
            except Exception as exc:
                consecutive_connection_errors += 1
                client.send_control(0.0, 0.0)
                signature = f"{type(exc).__name__}: {exc}"
                if signature != last_error_message:
                    print(f"[AVIS] Data receive failed: {signature}")
                    last_error_message = signature
                if consecutive_connection_errors >= 3:
                    raise ConnectionError(
                        "AVIS data reception failed three consecutive times."
                    ) from exc
                continue

            current_settings, current_revision = dashboard_state.settings_snapshot()
            if current_revision != settings_revision:
                try:
                    adaptive_roi_controller.configure(current_settings)
                    _apply_avis_roi_settings(conf, current_settings)
                    if int(current_settings["sensor_angle"]) != active_sensor_angle:
                        client.set_sensor_angle(int(current_settings["sensor_angle"]))
                        active_sensor_angle = int(current_settings["sensor_angle"])

                    steering_controller.configure(
                        kp=float(current_settings["kp"]),
                        ki=float(current_settings["ki"]),
                        kd=float(current_settings["kd"]),
                        steer_limit=float(current_settings["steer_limit"]),
                        max_steer_step=float(current_settings["max_steer_step"]),
                    )
                    speed_controller.configure(
                        acceleration_mps2=float(current_settings["acceleration_mps2"]),
                        min_speed_kmh=float(current_settings["min_speed_kmh"]),
                        max_speed_kmh=float(current_settings["max_speed_kmh"]),
                    )

                    selected_backend = str(current_settings["vision_backend"])
                    if selected_backend != active_backend:
                        lane_replacement = _create_avis_vision_processor(
                            selected_backend,
                            current_settings,
                            adaptive_roi_controller=adaptive_roi_controller,
                        )
                        replacement = AvisCombinedVisionProcessor(
                            lane_replacement, obstacle_classifier
                        )
                        inference_worker.set_vision_processor(replacement)
                        vision_processor = replacement
                        active_backend = selected_backend
                        print(f"[AVIS] Vision backend changed to {active_backend}.")
                    elif active_backend == "classical":
                        vision_processor.update_settings(current_settings)
                    else:
                        vision_processor.min_side_pixels = int(
                            current_settings["min_side_pixels"]
                        )
                        vision_processor.fallback_lane_offset_ratio = float(
                            current_settings["lane_offset_ratio"]
                        )
                        vision_processor.lane_center_alpha = float(
                            current_settings["center_smoothing"]
                        )

                    if any(
                        settings.get(key) != current_settings.get(key)
                        for key in (
                            *AVIS_ROI_CONFIG_KEYS,
                            *AVIS_SHARED_VISION_CONFIG_KEYS,
                        )
                    ) and active_backend == "ultralytics":
                        # Reset only the AVIS-side sticky lane state after
                        # editing shared ROI/centre geometry settings.
                        vision_processor.last_lane_center = None
                        vision_processor.active_side = None

                    control_period = 1.0 / float(current_settings["control_loop_hz"])
                    inference_interval = (
                        float(current_settings["inference_interval_ms"]) / 1000.0
                    )
                    last_vision_result = None
                    last_inference_submit = 0.0
                    settings_revision = current_revision
                    settings = current_settings
                except Exception as exc:
                    error_message = (
                        f"Could not apply AVIS settings: {type(exc).__name__}: {exc}"
                    )
                    print(f"[AVIS] {error_message}")
                    previous_backend = active_backend
                    dashboard_state.update_settings(
                        {"vision_backend": previous_backend}, persist=True
                    )
                    settings, settings_revision = dashboard_state.settings_snapshot()

            valid_frame = (
                isinstance(frame, np.ndarray)
                and frame.ndim == 3
                and frame.shape[2] >= 3
                and frame.size > 0
            )
            vision_error = None
            vision_data: dict[str, Any] = {}
            fps_metrics = {
                "frame_read_fps": frame_read_fps.fps(),
                "inference_fps": inference_fps.fps(),
                "control_loop_fps": control_loop_fps.fps(),
            }
            if valid_frame:
                frame_id += 1
                frame_bgr = np.ascontiguousarray(frame[:, :, :3])
                now = time.perf_counter()

                if now - last_inference_submit >= inference_interval:
                    inference_worker.submit_vision(
                        source_frame_id=frame_id,
                        semantic_frame=None,
                        rgb_frame=frame_bgr,
                    )
                    last_inference_submit = now

                vision_snapshot, vision_busy, vision_error, _ = (
                    inference_worker.vision_snapshot()
                )
                if vision_snapshot is not None and vision_snapshot.valid:
                    last_vision_result = vision_snapshot
                    completed_at = float(vision_snapshot.completed_at)
                    if completed_at != last_inference_fps_timestamp:
                        inference_fps.record(completed_at)
                        last_inference_fps_timestamp = completed_at
                if last_vision_result is not None:
                    vision_data = last_vision_result.result or {}
                fps_metrics = {
                    "frame_read_fps": frame_read_fps.fps(),
                    "inference_fps": inference_fps.fps(),
                    "control_loop_fps": control_loop_fps.fps(),
                }

                drive_mode, manual_throttle, manual_steering, manual_active = (
                    dashboard_state.drive_state()
                )
                throttle = 0.0
                steering = 0.0
                target_speed = 0.0
                coasting = False
                lane_valid = False
                lane_error = 0.0
                status = "WAITING FOR FIRST PREDICTION"

                debug_data = vision_data.get("debug", {})
                if not isinstance(debug_data, dict):
                    debug_data = {}
                lane_change_state = obstacle_lane_change.snapshot(sensors)

                if drive_mode == "manual":
                    throttle = manual_throttle if manual_active else 0.0
                    steering = manual_steering if manual_active else 0.0
                    client.send_manual_control(throttle, steering)
                    status = (
                        "MANUAL"
                        if manual_active
                        else "MANUAL / INPUT TIMED OUT - ZERO THROTTLE"
                    )
                elif drive_mode == "stopped":
                    client.send_control(0.0, 0.0)
                    status = "STOPPED"
                else:
                    result_fresh = False
                    if last_vision_result is not None:
                        result_age = max(
                            0.0,
                            time.perf_counter() - last_vision_result.completed_at,
                        )
                        result_fresh = result_age <= max(
                            0.75, inference_interval * 8.0
                        )

                    lane_valid = bool(
                        result_fresh
                        and vision_data.get(
                            "lane_valid",
                            vision_data.get("lane_type", "none") != "none",
                        )
                    )
                    lane_error = (
                        float(vision_data.get("error", 0.0) or 0.0)
                        if result_fresh else 0.0
                    )
                    control_dt = min(
                        0.1,
                        max(0.005, time.perf_counter() - last_control_time),
                    )
                    lane_change_state = obstacle_lane_change.update(
                        sensors=sensors,
                        speed_kmh=current_speed,
                        dt=control_dt,
                        left_x=debug_data.get("left_x") if result_fresh else None,
                        right_x=debug_data.get("right_x") if result_fresh else None,
                        settings=current_settings,
                        classifier_output=(
                            vision_data.get("obstacle_classifier") if result_fresh else None
                        ),
                        allow_new_maneuver=True,
                        observation_id=(
                            last_vision_result.completed_at if result_fresh else None
                        ),
                        frame_width=int(frame_bgr.shape[1]),
                    )
                    requested_sensor_angle = int(
                        lane_change_state.get(
                            "sensor_angle_override", current_settings["sensor_angle"]
                        )
                    )
                    if requested_sensor_angle != active_sensor_angle:
                        client.set_sensor_angle(requested_sensor_angle)
                        active_sensor_angle = requested_sensor_angle
                    if lane_change_state.get("pid_reset"):
                        steering_controller.reset()

                    minimum_speed = float(current_settings["min_speed_kmh"])
                    steering_override = lane_change_state.get("steering_override")
                    if lane_change_state.get("safe_stop"):
                        # Reserved for an explicit controller failure; ordinary
                        # lane loss and line-search timeout never enter this path.
                        throttle = 0.0
                        steering = 0.0
                        status = "LANE CHANGE CONTROLLER FAILURE"
                    elif steering_override is not None:
                        # Steering and target-line acquisition phases all run
                        # at the configured minimum moving speed.
                        steering = float(steering_override)
                        recovery_steering = steering
                        target_speed = minimum_speed
                        throttle, requested_brake = speed_controller.update(
                            current_speed, target_speed, dt=control_dt
                        )
                        coasting = requested_brake > 0.0
                        if coasting:
                            throttle = 0.0
                        if lane_change_state["phase"] == "STEER_LEFT":
                            status = (
                                "OBSTACLE AVOIDANCE / LEFT STEER PULSE "
                                f"{lane_change_state.get('steering_pulse_number', 1)}"
                            )
                        elif lane_change_state["phase"] == "ACQUIRE_LEFT_LINE":
                            status = (
                                "OBSTACLE AVOIDANCE / LOW-SPEED LEFT ROI SEARCH"
                                if lane_change_state.get("search_limit_reached")
                                else "OBSTACLE AVOIDANCE / CHECKING LEFT ROI"
                            )
                        elif lane_change_state["phase"] == "STEER_RIGHT":
                            status = (
                                "OBSTACLE AVOIDANCE / RIGHT STEER PULSE "
                                f"{lane_change_state.get('steering_pulse_number', 1)}"
                            )
                        elif lane_change_state["phase"] == "ACQUIRE_RIGHT_LINE":
                            status = (
                                "OBSTACLE AVOIDANCE / LOW-SPEED RIGHT ROI SEARCH"
                                if lane_change_state.get("search_limit_reached")
                                else "OBSTACLE AVOIDANCE / CHECKING RIGHT ROI"
                            )
                        else:
                            status = "OBSTACLE AVOIDANCE"
                    elif not result_fresh:
                        recovery_steering = max(
                            -0.25, min(0.25, recovery_steering * 0.85)
                        )
                        steering = recovery_steering
                        target_speed = minimum_speed
                        throttle, requested_brake = speed_controller.update(
                            current_speed, target_speed, dt=control_dt
                        )
                        coasting = requested_brake > 0.0
                        if coasting:
                            throttle = 0.0
                        status = "VISION RESULT STALE / LOW-SPEED FORWARD"
                    elif not lane_valid:
                        # Keep a gentle forward crawl while lane perception recovers.
                        # Decay the last steering command instead of holding a sharp turn.
                        recovery_steering = max(
                            -0.25, min(0.25, recovery_steering * 0.85)
                        )
                        steering = recovery_steering
                        target_speed = minimum_speed
                        throttle, requested_brake = speed_controller.update(
                            current_speed, target_speed, dt=control_dt
                        )
                        coasting = requested_brake > 0.0
                        if coasting:
                            throttle = 0.0
                        status = "LANE NOT DETECTED / LOW-SPEED FORWARD"
                    else:
                        steering = steering_controller.update(lane_error)
                        recovery_steering = steering
                        base_target_speed = select_avis_target_speed(
                            steering,
                            minimum_speed,
                            float(current_settings["max_speed_kmh"]),
                        )
                        target_speed = base_target_speed
                        throttle, requested_brake = speed_controller.update(
                            current_speed, target_speed, dt=control_dt
                        )
                        coasting = requested_brake > 0.0
                        if coasting:
                            throttle = 0.0
                        if lane_change_state["phase"] == "WAIT_OBSTACLE_PASS":
                            status = "OBSTACLE AVOIDANCE / WAITING FOR RIGHT SENSOR PASS CONFIRMATION"
                        elif lane_change_state["phase"] == "RETURN_DELAY":
                            status = (
                                "OBSTACLE AVOIDANCE / RIGHT SENSOR CLEAR · RETURN IN "
                                f"{lane_change_state.get('return_delay_remaining_s', 0.0):.1f}s"
                            )
                        elif lane_change_state["phase"] == "LEFT_LANE":
                            status = "OBSTACLE AVOIDANCE / LEFT LANE"
                        else:
                            status = "AUTO / COAST" if coasting else "AUTO"
                    client.send_control(throttle, steering)

                
                if not isinstance(debug_data, dict):
                    debug_data = {}
                combined = debug_data.get("combined")
                line_mask = debug_data.get("lane_mask")

                display = (
                    np.ascontiguousarray(combined).copy()
                    if isinstance(combined, np.ndarray) and combined.size
                    else frame_bgr.copy()
                )
                cv2.putText(
                    display,
                    f"AVIS | {status}",
                    (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 230, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    (
                        f"Speed: {current_speed:.1f} km/h | "
                        f"Target: {target_speed:.1f} km/h | "
                        f"Throttle: {throttle * 100:.0f}%"
                    ),
                    (12, 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    f"Steering: {steering:+.3f} ({steering * 100:+.0f})",
                    (12, 76),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                if isinstance(sensors, (list, tuple)) and len(sensors) >= 3:
                    cv2.putText(
                        display,
                        f"Distance sensors: L {sensors[0]} | C {sensors[1]} | R {sensors[2]} cm",
                        (12, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (220, 220, 220),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.putText(
                    display,
                    (
                        f"FPS | Read: {fps_metrics['frame_read_fps']:.1f} | "
                        f"Inference: {fps_metrics['inference_fps']:.1f} | "
                        f"Control: {fps_metrics['control_loop_fps']:.1f}"
                    ),
                    (12, 124),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (100, 235, 255),
                    1,
                    cv2.LINE_AA,
                )

                now = time.perf_counter()
                if now - last_dashboard_publish >= 0.10:
                    dashboard_state.publish_images(
                        {
                            "camera": _encode_avis_preview(cv2, frame_bgr),
                            "combined": _encode_avis_preview(cv2, display),
                            "line-mask": _encode_avis_preview(
                                cv2,
                                _colorize_avis_lane_mask(
                                    cv2, np, line_mask, frame_bgr.shape[:2]
                                ),
                            ),
                        }
                    )
                    last_dashboard_publish = now

                dashboard_state.publish_telemetry(
                    {
                        "connected": True,
                        "status": status,
                        "speed_kmh": round(float(current_speed), 2),
                        "target_speed_kmh": round(float(target_speed), 2),
                        "throttle": round(float(throttle), 3),
                        "steering": round(float(steering), 3),
                        "lane_error_px": round(float(lane_error), 2),
                        "lane_valid": lane_valid,
                        "sensors_cm": (
                            list(sensors[:3])
                            if isinstance(sensors, (list, tuple)) and len(sensors) >= 3
                            else None
                        ),
                        "front_sensor_cm": lane_change_state.get("front_sensor_cm"),
                        "sensor_distances_cm": lane_change_state.get("sensor_distances_cm"),
                        "obstacle_trigger_sensor": lane_change_state.get("obstacle_trigger_sensor"),
                        "lane_change_phase": lane_change_state.get("phase", "RIGHT_LANE"),
                        "lane_change_side": lane_change_state.get("lane_side", "right"),
                        "lane_change_target": lane_change_state.get("target_lane", "right"),
                        "lane_change_distance_m": round(
                            float(lane_change_state.get("left_lane_distance_m", 0.0)), 2
                        ),
                        "lane_change_target_distance_m": lane_change_state.get(
                            "left_lane_target_distance_m"
                        ),
                        "lane_change_obstacle_distance_m": lane_change_state.get(
                            "last_obstacle_distance_m"
                        ),
                        "lane_change_return_reason": lane_change_state.get(
                            "return_reason", "DISTANCE"
                        ),
                        "lane_change_target_line_side": lane_change_state.get("target_line_side"),
                        "lane_change_target_line_baseline_x": lane_change_state.get("target_line_baseline_x"),
                        "lane_change_target_line_old_track_x": lane_change_state.get("target_line_old_track_x"),
                        "lane_change_target_line_old_cleared": lane_change_state.get("target_line_old_cleared", False),
                        "lane_change_target_line_candidate_x": lane_change_state.get("target_line_candidate_x"),
                        "lane_change_line_confirm_frames": lane_change_state.get("line_confirm_frames", 0),
                        "lane_change_segment_distance_m": round(
                            float(lane_change_state.get("phase_distance_m", 0.0)), 2
                        ),
                        "lane_change_line_search_distance_m": round(
                            float(lane_change_state.get("line_search_distance_m", 0.0)), 2
                        ),
                        "lane_change_steering_pulse_number": lane_change_state.get(
                            "steering_pulse_number", 0
                        ),
                        "obstacle_classifier": vision_data.get("obstacle_classifier"),
                        "obstacle_model_confirm_frames": lane_change_state.get("obstacle_model_confirm_frames", 0),
                        "obstacle_model_probability": lane_change_state.get("obstacle_model_probability"),
                        "obstacle_lane_probability": lane_change_state.get("obstacle_lane_probability"),
                        "obstacle_lane_prediction": lane_change_state.get("obstacle_lane_prediction"),
                        "obstacle_pass_sensor_stage": lane_change_state.get("pass_sensor_stage"),
                        "obstacle_pass_right_sensor_cm": lane_change_state.get("pass_sensor_right_cm"),
                        "obstacle_pass_clear_frames": lane_change_state.get("pass_sensor_clear_frames", 0),
                        "obstacle_return_delay_remaining_s": lane_change_state.get("return_delay_remaining_s", 0.0),
                        "obstacle_pass_sensor_timeout": lane_change_state.get("pass_sensor_timeout", False),
                        "active_sensor_angle": active_sensor_angle,
                        "adaptive_roi": vision_data.get("debug", {}).get(
                            "adaptive_roi", adaptive_roi_controller.snapshot()
                        ),
                        **fps_metrics,
                        "vision_error": (
                            str(vision_error) if vision_error else None
                        ),
                    }
                )
            else:
                client.send_control(0.0, 0.0)
                display = np.zeros((360, 720, 3), dtype=np.uint8)
                cv2.putText(
                    display,
                    "WAITING FOR AVIS CAMERA...",
                    (28, 185),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
                dashboard_state.publish_telemetry(
                    {
                        "connected": True,
                        "status": "Waiting for camera",
                        "speed_kmh": float(current_speed),
                        "target_speed_kmh": 0.0,
                        "throttle": 0.0,
                        "steering": 0.0,
                        "lane_error_px": 0.0,
                        "lane_valid": False,
                        "sensors_cm": (
                            list(sensors[:3])
                            if isinstance(sensors, (list, tuple)) and len(sensors) >= 3
                            else None
                        ),
                        **fps_metrics,
                        "vision_error": "No valid AVIS camera frame.",
                    }
                )

            last_control_time = time.perf_counter()

            if not args.avis_headless:
                try:
                    cv2.imshow("AVIS Engine | AVIS Runtime", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                except cv2.error as exc:
                    print(
                        f"[AVIS] OpenCV preview unavailable; continuing headless: {exc}"
                    )
                    args.avis_headless = True

            next_tick += control_period
            now = time.perf_counter()
            if next_tick < now:
                next_tick = now
            time.sleep(max(0.0, next_tick - now))

        return 0

    except KeyboardInterrupt:
        print("\n[AVIS] Stop requested.")
        return 0
    except Exception as exc:
        print(f"[AVIS] Runtime failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        if dashboard_state is not None:
            dashboard_state.set_mode("stopped")
        if client is not None:
            client.close()
        if dashboard_server is not None:
            dashboard_server.stop()
        if inference_worker is not None:
            try:
                inference_worker.stop()
            except Exception as exc:
                print(f"[AVIS] Inference worker shutdown warning: {exc}")
        if cv2_module is not None:
            try:
                cv2_module.destroyAllWindows()
            except Exception:
                pass


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.avis:
        print("AVIS mode requires --avis.")
        return 2
    if not 1 <= args.avis_port <= 65535:
        print("[AVIS] Port must be between 1 and 65535.")
        return 2
    if not 10 <= args.avis_sensor_angle <= 90:
        print("[AVIS] Sensor angle must be between 10 and 90 degrees.")
        return 2
    if not 1 <= args.avis_web_port <= 65535:
        print("[AVIS] Web dashboard port must be between 1 and 65535.")
        return 2
    return run_avis(args)


if __name__ == "__main__":
    raise SystemExit(main())
