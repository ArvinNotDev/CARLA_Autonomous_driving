from __future__ import annotations

import math
import threading
import time
from typing import Optional, Dict

import config_city as conf


class LaneChangeManager:
    """
    Nonblocking lane-change state machine.

    Steering remains owned by this manager through VERIFY, CHANGE and SETTLE.
    The ordinary lane-follow controller is released only after the camera has
    observed a lane-change angle transition followed by stable lane markings
    in the destination lane over distinct vision frames.
    """

    def __init__(
        self,
        vehicle,
        planner,
        lane_change_debounce_seconds: float = 1.5,
    ) -> None:
        self.vehicle = vehicle
        self.planner = planner

        self.lane_change_debounce_seconds = float(lane_change_debounce_seconds)
        self.lane_change_debounce_until = 0.0
        self.lane_change_attempts = 0
        self.line_angles: Dict[str, Optional[float]] = {"left": None, "right": None}
        self.angle_cooldown_until = 0.0

        self.phase = "IDLE"  # IDLE / VERIFY / CHANGE / SETTLE
        self.direction: Optional[str] = None
        self._segment_start_location = None
        self._state_lock = threading.Lock()

        self._planner_query_active = False
        self._last_planner_query_at = 0.0
        self._request_id = 0
        self._latest_planner_result: Optional[tuple[int, str]] = None

        self._last_confirmation_frame_id: Optional[int] = None
        self._baseline_line_angles: Dict[str, float] = {}
        self._angle_transition_seen = False
        self._stable_angle_frames = 0
        self._steering_distance_reached = False
        self._settle_started_at: Optional[float] = None
        self._settle_timed_out = False

    def _movement_active(self) -> bool:
        with self._state_lock:
            return self.phase in {"VERIFY", "CHANGE", "SETTLE"}

    def movement_active(self) -> bool:
        return self._movement_active()

    def phase_name(self) -> str:
        with self._state_lock:
            return self.phase

    def confirmation_timed_out(self) -> bool:
        with self._state_lock:
            return self.phase == "SETTLE" and self._settle_timed_out

    def _angle_guard_triggered(self) -> bool:
        threshold = float(getattr(conf, "LANE_CHANGE_LINE_ANGLE_THRESHOLD_DEG", 20.0))
        return any(
            angle is not None and abs(float(angle)) > threshold
            for angle in self.line_angles.values()
        )

    def _reset_completion_evidence(self) -> None:
        self._last_confirmation_frame_id = None
        self._baseline_line_angles = {}
        self._angle_transition_seen = False
        self._stable_angle_frames = 0
        self._steering_distance_reached = False
        self._settle_started_at = None
        self._settle_timed_out = False

    def _apply_planner_result(self, current_location) -> None:
        if current_location is None:
            return
        with self._state_lock:
            result = self._latest_planner_result
            if result is None or self.phase != "IDLE":
                return
            self._latest_planner_result = None
            _, direction = result
            self.direction = direction
            self.phase = "VERIFY"
            self._segment_start_location = current_location
            self._reset_completion_evidence()

    def _capture_baseline_angles(self) -> None:
        baseline: Dict[str, float] = {}
        for side, angle in self.line_angles.items():
            if angle is None:
                continue
            try:
                value = float(angle)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                baseline[side] = value
        self._baseline_line_angles = baseline

    def _record_angle_evidence(
        self,
        line_angles: Dict[str, Optional[float]],
        lane_type: Optional[str],
        source_frame_id: Optional[int],
    ) -> None:
        # Vision results are held between inference completions. Only count
        # distinct source frames, otherwise a single result could satisfy the
        # multi-frame confirmation merely because the control loop runs faster.
        if source_frame_id is None:
            return
        try:
            frame_id = int(source_frame_id)
        except (TypeError, ValueError):
            return
        if frame_id == self._last_confirmation_frame_id:
            return
        self._last_confirmation_frame_id = frame_id

        current_angles: Dict[str, float] = {}
        for side, angle in line_angles.items():
            if angle is None:
                continue
            try:
                value = float(angle)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                current_angles[side] = value

        line_available = bool(current_angles) and lane_type not in {None, "", "none"}
        if not line_available:
            self._stable_angle_frames = 0
            return

        enter_angle = max(
            1.0, float(getattr(conf, "LANE_CHANGE_ANGLE_ENTER_DEG", 10.0))
        )
        settle_angle = max(
            0.5, float(getattr(conf, "LANE_CHANGE_ANGLE_SETTLE_DEG", 6.0))
        )

        # Compare each visible lane boundary to its pre-maneuver angle. This
        # works on curved roads too; requiring angles near zero would only work
        # for a perfectly straight road and could leave the override stuck.
        common_sides = self._baseline_line_angles.keys() & current_angles.keys()
        if not common_sides:
            self._stable_angle_frames = 0
            return

        deltas = {
            side: current_angles[side] - self._baseline_line_angles[side]
            for side in common_sides
        }
        if not self._angle_transition_seen:
            if any(abs(delta) >= enter_angle for delta in deltas.values()):
                self._angle_transition_seen = True
            self._stable_angle_frames = 0
            return

        if all(abs(delta) <= settle_angle for delta in deltas.values()):
            self._stable_angle_frames += 1
        else:
            self._stable_angle_frames = 0

    def _completion_confirmed(self) -> bool:
        if not self._steering_distance_reached or not self._angle_transition_seen:
            return False
        required_frames = max(
            2, int(getattr(conf, "LANE_CHANGE_CONFIRM_FRAMES", 3))
        )
        return self._stable_angle_frames >= required_frames

    def update(
        self,
        current_location,
        goal_location,
        line_angles: Optional[Dict[str, Optional[float]]] = None,
        lane_type: Optional[str] = None,
        source_frame_id: Optional[int] = None,
    ) -> None:
        if self.vehicle is None:
            return

        if line_angles is not None:
            self.line_angles = {
                "left": line_angles.get("left"),
                "right": line_angles.get("right"),
            }

        self._apply_planner_result(current_location)

        with self._state_lock:
            phase = self.phase
            segment_start = self._segment_start_location

        now = time.monotonic()
        if phase == "VERIFY":
            if segment_start is not None and current_location is not None:
                try:
                    traveled = float(current_location.distance(segment_start))
                except Exception:
                    traveled = 0.0
                verify_distance = max(
                    0.1, float(getattr(conf, "LANE_CHANGE_VERIFY_DISTANCE_M", 1.0))
                )
                if traveled >= verify_distance:
                    with self._state_lock:
                        if self.phase == "VERIFY":
                            self.phase = "CHANGE"
                            self._segment_start_location = current_location
                            self._reset_completion_evidence()
                            self._capture_baseline_angles()
            return

        if phase in {"CHANGE", "SETTLE"}:
            self._record_angle_evidence(
                self.line_angles, lane_type, source_frame_id
            )

            if phase == "CHANGE" and segment_start is not None and current_location is not None:
                try:
                    traveled = float(current_location.distance(segment_start))
                except Exception:
                    traveled = 0.0
                steering_distance = max(
                    0.25, float(getattr(conf, "LANE_CHANGE_STEER_DISTANCE_M", 2.0))
                )
                if traveled >= steering_distance:
                    with self._state_lock:
                        if self.phase == "CHANGE":
                            self._steering_distance_reached = True
                            self.phase = "SETTLE"
                            self._settle_started_at = now
                    self.lane_change_attempts += 1
                    if (
                        self.lane_change_attempts
                        % max(1, int(getattr(conf, "LANE_CHANGE_ANGLE_CHECK_EVERY", 2)))
                        == 0
                        and self._angle_guard_triggered()
                    ):
                        self.angle_cooldown_until = now + float(
                            getattr(conf, "LANE_CHANGE_ANGLE_COOLDOWN_SECONDS", 5.0)
                        )

            if self._completion_confirmed():
                self._finish()
                return

            if (
                self.phase == "SETTLE"
                and self._settle_started_at is not None
                and now - self._settle_started_at
                >= max(1.0, float(getattr(conf, "LANE_CHANGE_CONFIRM_TIMEOUT_SECONDS", 8.0)))
            ):
                self._settle_timed_out = True
            return

        if phase != "IDLE":
            return

        if now < self.angle_cooldown_until or time.time() < self.lane_change_debounce_until:
            return

        query_interval = float(
            getattr(conf, "LANE_CHANGE_PLANNER_CHECK_INTERVAL_SECONDS", 0.50)
        )
        with self._state_lock:
            if self._planner_query_active or (
                now - self._last_planner_query_at < query_interval
            ):
                return
            self._planner_query_active = True
            self._last_planner_query_at = now
            self._request_id += 1
            request_id = self._request_id

        def query_worker() -> None:
            direction = ""
            try:
                text = self.planner.get_next_maneuver_text(current_location, goal_location)
                if text == "CHANGE_LANE_LEFT":
                    direction = "left"
                elif text == "CHANGE_LANE_RIGHT":
                    direction = "right"
                if direction:
                    with self._state_lock:
                        if request_id == self._request_id and self.phase == "IDLE":
                            self._latest_planner_result = (request_id, direction)
                            self.lane_change_debounce_until = (
                                time.time() + self.lane_change_debounce_seconds
                            )
            except Exception:
                pass
            finally:
                with self._state_lock:
                    self._planner_query_active = False

        threading.Thread(
            target=query_worker,
            name="carla-lane-planner-query",
            daemon=True,
        ).start()

    def control_override(self):
        """Return a lane-change command until destination-lane stability is confirmed."""
        import carla

        with self._state_lock:
            phase = self.phase
            direction = self.direction

        if phase == "VERIFY":
            throttle = 0.08
            steer = 0.0
        elif phase == "CHANGE":
            throttle = 0.10
            steer = -0.2 if direction == "left" else 0.2
        elif phase == "SETTLE":
            if self._settle_timed_out:
                return carla.VehicleControl(
                    throttle=0.0,
                    steer=0.0,
                    brake=1.0,
                    reverse=False,
                    hand_brake=False,
                    manual_gear_shift=False,
                )
            throttle = 0.10
            settle_steer = max(
                0.0, min(0.2, float(getattr(conf, "LANE_CHANGE_SETTLE_STEER", 0.08)))
            )
            steer = -settle_steer if direction == "left" else settle_steer
        else:
            return None

        return carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=0.0,
            reverse=False,
            hand_brake=False,
            manual_gear_shift=False,
        )

    def _finish(self) -> None:
        with self._state_lock:
            self.phase = "IDLE"
            self.direction = None
            self._segment_start_location = None
            self._reset_completion_evidence()
