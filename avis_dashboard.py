from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from typing import Any

from flask import Flask, Response, jsonify, request

from avis_profiles import AvisGitHubProfileSync
from werkzeug.serving import BaseWSGIServer, make_server


DEFAULT_AVIS_SETTINGS: dict[str, Any] = {
    "vision_backend": "ultralytics",
    "auto_mode_default": True,
    "lane_model_path": "models_and_datasets/Avis_models/lane_and_drivable-area_model_Avis.pt",
    "obstacle_model_path": "models_and_datasets/Avis_models/obstacle_and_lane_Avis_model.onnx",
    "model_image_size": 512,
    "lane_prob_threshold": 0.50,
    "drivable_prob_threshold": 0.50,
    "vision_debug": True,
    "debug_show_rois": True,
    "debug_show_lane_mask": True,
    "debug_show_drivable_area": True,
    "debug_show_text": True,
    "debug_panel_hz": 40.0,
    "control_loop_hz": 30.0,
    "inference_interval_ms": 25.0,
    "sensor_angle": 40,
    "min_speed_kmh": 8.0,
    "max_speed_kmh": 12.0,
    "acceleration_mps2": 1.5,
    "kp": 0.015,
    "ki": 0.0,
    "kd": 0.002,
    "steer_limit": 0.75,
    "max_steer_step": 0.10,
    "blur_kernel": 5,
    "canny_low": 55,
    "canny_high": 150,
    "hough_threshold": 12,
    "hough_min_line_length_ratio": 0.035,
    "hough_max_line_gap": 24,
    "ll_top_roi": 0.57,
    "ll_bottom_roi": 0.69,
    "ll_left_roi": 0.24,
    "ll_right_roi": 0.45,
    "rl_top_roi": 0.57,
    "rl_bottom_roi": 0.69,
    "rl_left_roi": 0.56,
    "rl_right_roi": 0.78,
    "white_l_min": 165,
    "white_s_max": 115,
    "yellow_enabled": True,
    "yellow_h_min": 12,
    "yellow_h_max": 39,
    "yellow_s_min": 65,
    "yellow_l_min": 75,
    "morphology_kernel": 3,
    "min_lane_angle_deg": 0.0,
    "max_lane_angle_deg": 80.0,
    "min_side_pixels": 12,
    "lane_offset_ratio": 0.16,
    "center_smoothing": 0.30,
    "obstacle_detection_enabled": True,
    "obstacle_lane_change_enabled": True,
    "city_crosswalk_enabled": False,
    "city_crosswalk_stop_seconds": 5.0,
    "city_crosswalk_rearm_clear_seconds": 1.0,
    "city_crosswalk_min_line_length_ratio": 0.35,
    "city_crosswalk_max_line_angle_deg": 12.0,
    "city_crosswalk_confirm_frames": 2,
    "cw_top_roi": 0.55,
    "cw_bottom_roi": 0.88,
    "cw_left_roi": 0.08,
    "cw_right_roi": 0.92,
    "obstacle_model_confidence_threshold": 0.65,
    "obstacle_lane_confidence_threshold": 0.65,
    "obstacle_model_confirm_frames": 3,
    "obstacle_model_rearm_frames": 3,
    "obstacle_pass_sensor_angle": 70,
    "obstacle_pass_sensor_presence_threshold_cm": 250,
    "obstacle_pass_sensor_clear_threshold_cm": 550,
    "obstacle_pass_sensor_presence_frames": 2,
    "obstacle_pass_sensor_clear_frames": 3,
    "obstacle_pass_clearance_distance_m": 10.0,
    "obstacle_return_delay_seconds": 2.0,
    "obstacle_pass_sensor_timeout_seconds": 30.0,
    "obstacle_front_threshold_cm": 950,
    "obstacle_trigger_confirm_frames": 3,
    "obstacle_sensor_clear_threshold_cm": 900,
    "obstacle_sensor_clear_confirm_frames": 5,
    "obstacle_lane_change_speed_kmh": 4.5,
    "obstacle_lane_change_steer_distance_m": 1.0,
    "obstacle_lane_change_left_steer": 0.30,
    "obstacle_lane_change_right_steer": 0.30,
    "obstacle_lane_change_retry_steer": 0.20,
    "obstacle_lane_change_retry_pulse_distance_m": 0.15,
    "obstacle_lane_line_confirm_frames": 3,
    "obstacle_lane_line_search_max_distance_m": 40.0,
    "obstacle_clearance_after_pass_m": 5.0,
    "adaptive_roi_enabled": True,
    "drivable_recovery_enabled": True,
    "adaptive_roi_max_expansion_ratio": 0.12,
    "adaptive_roi_edge_trigger_ratio": 0.22,
    "adaptive_roi_expand_step_ratio": 0.02,
    "adaptive_roi_recovery_frames": 7,
    "adaptive_roi_restore_step_ratio": 0.005,
}


_NUMERIC_LIMITS: dict[str, tuple[float, float, bool]] = {
    "control_loop_hz": (10.0, 60.0, False),
    "model_image_size": (160.0, 2048.0, True),
    "lane_prob_threshold": (0.0, 1.0, False),
    "drivable_prob_threshold": (0.0, 1.0, False),
    "debug_panel_hz": (1.0, 120.0, False),
    "inference_interval_ms": (10.0, 500.0, False),
    "sensor_angle": (10.0, 90.0, True),
    "min_speed_kmh": (1.0, 60.0, False),
    "max_speed_kmh": (1.5, 100.0, False),
    "acceleration_mps2": (0.2, 8.0, False),
    "kp": (0.0, 0.2, False),
    "ki": (0.0, 0.1, False),
    "kd": (0.0, 0.1, False),
    "steer_limit": (0.05, 1.0, False),
    "max_steer_step": (0.01, 1.0, False),
    "blur_kernel": (1.0, 31.0, True),
    "canny_low": (0.0, 254.0, True),
    "canny_high": (1.0, 255.0, True),
    "hough_threshold": (1.0, 200.0, True),
    "hough_min_line_length_ratio": (0.005, 0.5, False),
    "hough_max_line_gap": (0.0, 250.0, True),
    "ll_top_roi": (0.0, 0.98, False),
    "ll_bottom_roi": (0.02, 1.0, False),
    "ll_left_roi": (0.0, 0.98, False),
    "ll_right_roi": (0.02, 1.0, False),
    "rl_top_roi": (0.0, 0.98, False),
    "rl_bottom_roi": (0.02, 1.0, False),
    "rl_left_roi": (0.0, 0.98, False),
    "rl_right_roi": (0.02, 1.0, False),
    "white_l_min": (0.0, 255.0, True),
    "white_s_max": (0.0, 255.0, True),
    "yellow_h_min": (0.0, 179.0, True),
    "yellow_h_max": (0.0, 179.0, True),
    "yellow_s_min": (0.0, 255.0, True),
    "yellow_l_min": (0.0, 255.0, True),
    "morphology_kernel": (1.0, 9.0, True),
    "min_lane_angle_deg": (0.0, 60.0, False),
    "max_lane_angle_deg": (1.0, 89.0, False),
    "min_side_pixels": (1.0, 500.0, True),
    "lane_offset_ratio": (0.03, 0.45, False),
    "center_smoothing": (0.01, 1.0, False),
    "cw_top_roi": (0.0, 0.98, False),
    "cw_bottom_roi": (0.02, 1.0, False),
    "cw_left_roi": (0.0, 0.98, False),
    "cw_right_roi": (0.02, 1.0, False),
    "city_crosswalk_stop_seconds": (0.5, 120.0, False),
    "city_crosswalk_rearm_clear_seconds": (0.2, 30.0, False),
    "city_crosswalk_min_line_length_ratio": (0.10, 0.95, False),
    "city_crosswalk_max_line_angle_deg": (1.0, 30.0, False),
    "city_crosswalk_confirm_frames": (1.0, 3.0, True),
    "obstacle_front_threshold_cm": (50.0, 1499.0, True),
    "obstacle_trigger_confirm_frames": (1.0, 30.0, True),
    "obstacle_sensor_clear_threshold_cm": (51.0, 1500.0, True),
    "obstacle_sensor_clear_confirm_frames": (1.0, 30.0, True),
    "obstacle_lane_change_speed_kmh": (1.0, 20.0, False),
    "obstacle_lane_change_steer_distance_m": (0.1, 100.0, False),
    "obstacle_lane_change_left_steer": (0.001, 1.0, False),
    "obstacle_lane_change_right_steer": (0.001, 1.0, False),
    "obstacle_lane_change_retry_steer": (0.001, 1.0, False),
    "obstacle_lane_change_retry_pulse_distance_m": (0.02, 5.0, False),
    "obstacle_lane_line_confirm_frames": (1.0, 60.0, True),
    "obstacle_lane_line_search_max_distance_m": (0.1, 1000.0, False),
    "obstacle_clearance_after_pass_m": (0.0, 1000.0, False),
    "obstacle_model_confidence_threshold": (0.50, 0.99, False),
    "obstacle_lane_confidence_threshold": (0.50, 0.99, False),
    "obstacle_model_confirm_frames": (1.0, 30.0, True),
    "obstacle_model_rearm_frames": (1.0, 30.0, True),
    "obstacle_pass_sensor_angle": (10.0, 90.0, True),
    "obstacle_pass_sensor_presence_threshold_cm": (20.0, 1499.0, True),
    "obstacle_pass_sensor_clear_threshold_cm": (21.0, 1500.0, True),
    "obstacle_pass_sensor_presence_frames": (1.0, 30.0, True),
    "obstacle_pass_sensor_clear_frames": (1.0, 30.0, True),
    "obstacle_pass_clearance_distance_m": (0.1, 100.0, False),
    "obstacle_return_delay_seconds": (0.0, 60.0, False),
    "obstacle_pass_sensor_timeout_seconds": (1.0, 120.0, False),
    "adaptive_roi_max_expansion_ratio": (0.0, 0.30, False),
    "adaptive_roi_edge_trigger_ratio": (0.05, 0.45, False),
    "adaptive_roi_expand_step_ratio": (0.001, 0.05, False),
    "adaptive_roi_recovery_frames": (1.0, 60.0, True),
    "adaptive_roi_restore_step_ratio": (0.001, 0.05, False),
}



def _migrate_legacy_lane_change_tuning(settings: dict[str, Any]) -> dict[str, Any]:
    """Move saved default lane-change tuning to the more precise AVIS defaults."""
    migrated = dict(settings)
    for key, old_default, new_default in (
        ("obstacle_lane_change_steer_distance_m", 3.0, 1.0),
        ("obstacle_lane_change_retry_pulse_distance_m", 0.30, 0.15),
    ):
        if key not in migrated:
            continue
        try:
            value = float(migrated[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and math.isclose(value, old_default, abs_tol=1e-9):
            migrated[key] = new_default
    return migrated


def validate_avis_settings(
    current: dict[str, Any],
    updates: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(updates, dict):
        raise ValueError("Settings payload must be a JSON object.")

    result = dict(current)
    allowed = set(DEFAULT_AVIS_SETTINGS)
    unknown = sorted(set(updates) - allowed)
    if unknown:
        raise ValueError(f"Unsupported AVIS setting(s): {', '.join(unknown)}")

    for key, value in updates.items():
        if key == "vision_backend":
            if value not in {"ultralytics", "classical"}:
                raise ValueError("vision_backend must be 'ultralytics' or 'classical'.")
            result[key] = value
            continue
        if key in {"lane_model_path", "obstacle_model_path"}:
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a model path string.")
            model_path = value.strip().replace("\\", "/")
            expected_suffix = ".pt" if key == "lane_model_path" else ".onnx"
            if not model_path or chr(0) in model_path or not model_path.lower().endswith(expected_suffix):
                raise ValueError(f"{key} must point to a {expected_suffix} model file.")
            result[key] = model_path
            continue
        if key in {
            "auto_mode_default", "vision_debug", "debug_show_rois",
            "debug_show_lane_mask", "debug_show_drivable_area", "debug_show_text",
            "yellow_enabled", "obstacle_detection_enabled",
            "obstacle_lane_change_enabled", "city_crosswalk_enabled",
            "adaptive_roi_enabled", "drivable_recovery_enabled",
        }:
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be a boolean.")
            result[key] = value
            continue

        minimum, maximum, integer = _NUMERIC_LIMITS[key]
        if isinstance(value, bool):
            raise ValueError(f"{key} must be numeric.")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be numeric.") from exc
        if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
            raise ValueError(f"{key} must be between {minimum:g} and {maximum:g}.")
        result[key] = int(round(numeric)) if integer else numeric

    if result["min_speed_kmh"] >= result["max_speed_kmh"]:
        raise ValueError("Maximum speed must be greater than minimum speed.")
    if result["canny_low"] >= result["canny_high"]:
        raise ValueError("Canny high threshold must be greater than low threshold.")
    if result["yellow_h_min"] >= result["yellow_h_max"]:
        raise ValueError("Yellow hue maximum must be greater than its minimum.")
    for side in ("ll", "rl"):
        if result[f"{side}_bottom_roi"] <= result[f"{side}_top_roi"]:
            raise ValueError(f"{side.upper()} ROI bottom must be below its top.")
        if result[f"{side}_right_roi"] <= result[f"{side}_left_roi"]:
            raise ValueError(f"{side.upper()} ROI right edge must be after its left edge.")
    if result["max_lane_angle_deg"] <= result["min_lane_angle_deg"]:
        raise ValueError("Maximum lane angle must exceed minimum lane angle.")
    return result


class AvisDashboardState:
    """Thread-safe state shared only by the AVIS runtime and its Flask UI."""

    def __init__(
        self,
        initial_settings: dict[str, Any] | None = None,
        settings_path: Path | str = "avis_settings.json",
        manual_timeout_seconds: float = 0.45,
        profile_path: Path | str | None = None,
        github_sync: AvisGitHubProfileSync | None = None,
        capture_directory: Path | str | None = None,
        default_profile_name: str = "AVIS_StableV2",
    ) -> None:
        self._lock = threading.RLock()
        self.settings_path = Path(settings_path)
        self.profile_path = (
            Path(profile_path)
            if profile_path is not None
            else self.settings_path.with_name("avis_profiles.json")
        )
        self.github_sync = github_sync or AvisGitHubProfileSync()
        self.capture_directory = (
            Path(capture_directory).expanduser()
            if capture_directory is not None
            else Path.home() / "Documents" / "AVIS_RoadForge_Captures"
        )
        self.capture_session_id = uuid4().hex
        self.manual_timeout_seconds = max(0.15, float(manual_timeout_seconds))
        self._settings = dict(DEFAULT_AVIS_SETTINGS)
        if initial_settings:
            self._settings = validate_avis_settings(self._settings, initial_settings)
        self._load_saved_settings()
        self._profiles: dict[str, dict[str, Any]] = {}
        self._default_profile_name = self._normalize_profile_name(default_profile_name)
        self._active_profile_name = self._default_profile_name
        self._load_saved_profiles()
        self._settings_revision = 0
        self._mode = "auto" if self._settings.get("auto_mode_default", True) else "stopped"
        self._manual_throttle = 0.0
        self._manual_steering = 0.0
        self._manual_updated_at = 0.0
        self._telemetry: dict[str, Any] = {
            "connected": False,
            "status": "Starting",
            "speed_kmh": 0.0,
            "target_speed_kmh": 0.0,
            "throttle": 0.0,
            "steering": 0.0,
            "lane_error_px": 0.0,
            "lane_valid": False,
            "sensors_cm": None,
            "front_sensor_cm": None,
            "sensor_distances_cm": None,
            "obstacle_trigger_sensor": None,
            "obstacle_classifier": None,
            "obstacle_model_confirm_frames": 0,
            "obstacle_model_probability": None,
            "obstacle_model_accepted": False,
            "obstacle_model_confidence_threshold": self._settings["obstacle_model_confidence_threshold"],
            "obstacle_lane_probability": None,
            "obstacle_lane_prediction": None,
            "obstacle_lane_confidence": None,
            "model_lane_confirm_frames": 0,
            "lane_change_attempt_number": 0,
            "lane_change_retry_steer": None,
            "lane_change_retry_pulse_distance_m": None,
            "return_distance_stage": "IDLE",
            "return_after_distance_m": 10.0,
            "return_distance_travelled_m": 0.0,
            "return_distance_remaining_m": 0.0,
            "active_sensor_angle": None,
            "lane_change_phase": "RIGHT_LANE",
            "lane_change_side": "right",
            "lane_change_target": "right",
            "lane_change_distance_m": 0.0,
            "lane_change_target_distance_m": None,
            "lane_change_obstacle_distance_m": None,
            "lane_change_return_reason": "DISTANCE",
            "lane_change_segment_distance_m": 0.0,
            "lane_change_target_line_side": None,
            "lane_change_target_line_baseline_x": None,
            "lane_change_target_line_old_track_x": None,
            "lane_change_target_line_old_cleared": False,
            "lane_change_target_line_candidate_x": None,
            "lane_change_line_confirm_frames": 0,
            "lane_change_line_search_distance_m": 0.0,
            "lane_change_steering_pulse_number": 0,
            "frame_read_fps": 0.0,
            "inference_fps": 0.0,
            "control_loop_fps": 0.0,
            "vision_error": None,
        }
        self._images: dict[str, bytes] = {}
        self._image_versions: dict[str, int] = {}
        # Keep each completed classifier result so browser polling cannot skip
        # fast inference updates between /api/status requests.
        self._model_prediction_history: list[dict[str, Any]] = []

    def _load_saved_settings(self) -> None:
        if not self.settings_path.is_file():
            return
        try:
            saved = json.loads(self.settings_path.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                # Drop settings from the previous AVIS-only trapezoid ROI
                # implementation while preserving compatible user tuning.
                for legacy_key in (
                    "roi_top_ratio",
                    "roi_bottom_ratio",
                    "roi_top_half_width_ratio",
                    "roi_bottom_half_width_ratio",
                    "lane_width_ratio",
                    "obstacle_return_distance_m",
                    "obstacle_slowdown_start_cm",
                    "city_pedestrian_stop_enabled",
                    "city_pedestrian_stop_seconds",
                ):
                    saved.pop(legacy_key, None)
            saved = _migrate_legacy_lane_change_tuning(saved)
            self._settings = validate_avis_settings(self._settings, saved)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"[AVIS WEB] Ignoring invalid settings file: {exc}")

    def settings_snapshot(self) -> tuple[dict[str, Any], int]:
        with self._lock:
            return dict(self._settings), self._settings_revision

    def update_settings(self, updates: dict[str, Any], persist: bool = True) -> dict[str, Any]:
        with self._lock:
            updated = validate_avis_settings(self._settings, updates)
            if persist:
                self.settings_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path = self.settings_path.with_suffix(
                    self.settings_path.suffix + ".tmp"
                )
                temporary_path.write_text(
                    json.dumps(updated, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary_path, self.settings_path)
            self._settings = updated
            self._settings_revision += 1
            return dict(self._settings)

    @staticmethod
    def _normalize_profile_name(name: Any) -> str:
        value = str(name or "").strip()
        if not value or len(value) > 48 or any(
            not (character.isalnum() or character in " _.-")
            for character in value
        ):
            raise ValueError(
                "Profile name must be 1-48 characters and may contain letters, "
                "numbers, spaces, dots, underscores or hyphens."
            )
        return value

    @staticmethod
    def _profile_key(name: str) -> str:
        return name.strip().casefold()

    def _load_saved_profiles(self) -> None:
        loaded: dict[str, dict[str, Any]] = {}
        requested_active = "AVIS_StableV2"
        if self.profile_path.is_file():
            try:
                document = json.loads(self.profile_path.read_text(encoding="utf-8"))
                if isinstance(document, dict):
                    requested_active = str(document.get("active_profile") or requested_active)
                entries = document.get("profiles", []) if isinstance(document, dict) else []
                if isinstance(entries, list):
                    for entry in entries:
                        if not isinstance(entry, dict) or not isinstance(entry.get("settings"), dict):
                            continue
                        try:
                            name = self._normalize_profile_name(entry.get("name"))
                            saved_settings = dict(entry["settings"])
                            saved_settings.pop("obstacle_slowdown_start_cm", None)
                            saved_settings.pop("city_pedestrian_stop_enabled", None)
                            saved_settings.pop("city_pedestrian_stop_seconds", None)
                            saved_settings = _migrate_legacy_lane_change_tuning(saved_settings)
                            settings = validate_avis_settings(
                                dict(DEFAULT_AVIS_SETTINGS), saved_settings
                            )
                        except (TypeError, ValueError):
                            continue
                        loaded[self._profile_key(name)] = {
                            "schema_version": 1,
                            "name": name,
                            "settings": settings,
                            "saved_at": entry.get("saved_at"),
                        }
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                print(f"[AVIS WEB] Ignoring invalid profile file: {exc}")

        default_key = self._profile_key(self._default_profile_name)
        if default_key not in loaded:
            loaded[default_key] = {
                "schema_version": 1,
                "name": self._default_profile_name,
                "settings": dict(self._settings),
                "saved_at": None,
            }
        self._profiles = loaded
        active_key = self._profile_key(requested_active)
        if active_key not in loaded:
            active_key = default_key
        self._active_profile_name = str(loaded[active_key]["name"])

    def _persist_profiles_locked(self) -> None:
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "schema_version": 1,
            "active_profile": self._active_profile_name,
            "profiles": sorted(
                self._profiles.values(),
                key=lambda item: str(item["name"]).casefold(),
            ),
        }
        temporary_path = self.profile_path.with_suffix(
            self.profile_path.suffix + ".tmp"
        )
        temporary_path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, self.profile_path)

    def profiles_snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "name": item["name"],
                    "saved_at": item.get("saved_at"),
                }
                for item in sorted(
                    self._profiles.values(),
                    key=lambda profile: str(profile["name"]).casefold(),
                )
            ]

    def get_profile(self, name: Any) -> dict[str, Any]:
        normalized = self._normalize_profile_name(name)
        with self._lock:
            profile = self._profiles.get(self._profile_key(normalized))
            if profile is None:
                raise ValueError(f"AVIS profile '{normalized}' was not found.")
            return json.loads(json.dumps(profile, ensure_ascii=False))

    def save_profile(self, name: Any) -> dict[str, Any]:
        normalized = self._normalize_profile_name(name)
        with self._lock:
            profile = {
                "schema_version": 1,
                "name": normalized,
                "settings": dict(self._settings),
                "saved_at": time.time(),
            }
            self._profiles[self._profile_key(normalized)] = profile
            self._active_profile_name = normalized
            self._persist_profiles_locked()
            return json.loads(json.dumps(profile, ensure_ascii=False))

    def import_profile(self, document: Any) -> dict[str, Any]:
        if not isinstance(document, dict):
            raise ValueError("Profile import must be a JSON object.")
        if document.get("schema_version", 1) != 1:
            raise ValueError("Unsupported AVIS profile schema version.")
        normalized = self._normalize_profile_name(document.get("name"))
        settings = document.get("settings")
        if not isinstance(settings, dict):
            raise ValueError("The imported profile does not contain a settings object.")
        compatible_settings = dict(settings)
        compatible_settings.pop("obstacle_slowdown_start_cm", None)
        compatible_settings.pop("city_pedestrian_stop_enabled", None)
        compatible_settings.pop("city_pedestrian_stop_seconds", None)
        compatible_settings = _migrate_legacy_lane_change_tuning(compatible_settings)
        validated = validate_avis_settings(
            dict(DEFAULT_AVIS_SETTINGS), compatible_settings
        )
        profile = {
            "schema_version": 1,
            "name": normalized,
            "settings": validated,
            "saved_at": document.get("saved_at") or time.time(),
        }
        with self._lock:
            self._profiles[self._profile_key(normalized)] = profile
            self._persist_profiles_locked()
            return json.loads(json.dumps(profile, ensure_ascii=False))

    def export_profile(self, name: Any) -> dict[str, Any]:
        return self.get_profile(name)

    def active_profile_name(self) -> str:
        with self._lock:
            return self._active_profile_name

    def load_profile(self, name: Any) -> dict[str, Any]:
        normalized = self._normalize_profile_name(name)
        profile = self.get_profile(normalized)
        settings = self.update_settings(profile["settings"], persist=True)
        with self._lock:
            self._active_profile_name = normalized
            self._persist_profiles_locked()
        return settings

    def delete_profile(self, name: Any) -> None:
        normalized = self._normalize_profile_name(name)
        if self._profile_key(normalized) in {
            self._profile_key("AVIS_StableV2"),
            self._profile_key(self._default_profile_name),
        }:
            raise ValueError("The AVIS baseline profile for this runtime mode cannot be deleted.")
        with self._lock:
            key = self._profile_key(normalized)
            if key not in self._profiles:
                raise ValueError(f"AVIS profile '{normalized}' was not found.")
            del self._profiles[key]
            if self._profile_key(self._active_profile_name) == key:
                self._active_profile_name = "AVIS_StableV2"
            self._persist_profiles_locked()

    def set_mode(self, mode: str) -> str:
        if mode not in {"auto", "manual", "stopped"}:
            raise ValueError("Mode must be auto, manual, or stopped.")
        with self._lock:
            self._mode = mode
            self._manual_throttle = 0.0
            self._manual_steering = 0.0
            self._manual_updated_at = 0.0
            return self._mode

    def set_manual_controls(self, throttle: Any, steering: Any) -> None:
        try:
            throttle_value = float(throttle)
            steering_value = float(steering)
        except (TypeError, ValueError) as exc:
            raise ValueError("Manual controls must be numeric.") from exc
        if not math.isfinite(throttle_value) or not math.isfinite(steering_value):
            raise ValueError("Manual controls must be finite.")
        if not -1.0 <= throttle_value <= 1.0:
            raise ValueError("Manual throttle must be between -1 and 1.")
        if not -1.0 <= steering_value <= 1.0:
            raise ValueError("Manual steering must be between -1 and 1.")
        with self._lock:
            if self._mode != "manual":
                raise RuntimeError("Switch to manual mode before sending controls.")
            self._manual_throttle = throttle_value
            self._manual_steering = steering_value
            self._manual_updated_at = time.monotonic()

    def drive_state(self) -> tuple[str, float, float, bool]:
        with self._lock:
            mode = self._mode
            fresh = (
                mode == "manual"
                and self._manual_updated_at > 0.0
                and time.monotonic() - self._manual_updated_at
                <= self.manual_timeout_seconds
            )
            if mode == "manual" and fresh:
                return mode, self._manual_throttle, self._manual_steering, True
            return mode, 0.0, 0.0, False

    def set_connection_status(self, connected: bool, status: str) -> None:
        """Update connection state without discarding the latest dashboard telemetry."""
        with self._lock:
            self._telemetry["connected"] = bool(connected)
            self._telemetry["status"] = str(status)
            if not connected:
                self._telemetry.update(
                    {
                        "speed_kmh": 0.0,
                        "target_speed_kmh": 0.0,
                        "throttle": 0.0,
                        "steering": 0.0,
                        "lane_valid": False,
                        "sensors_cm": None,
                        "front_sensor_cm": None,
                        "sensor_distances_cm": None,
                    }
                )

    def publish_telemetry(self, telemetry: dict[str, Any]) -> None:
        with self._lock:
            self._telemetry = dict(telemetry)

    def publish_model_prediction(self, prediction: dict[str, Any]) -> None:
        """Append one completed model output to a bounded live history."""
        if not isinstance(prediction, dict):
            return
        try:
            request_id = int(prediction.get("request_id"))
            frame_id = int(prediction.get("frame_id"))
        except (TypeError, ValueError):
            return
        obstacle = prediction.get("obstacle")
        lane = prediction.get("lane")
        if not isinstance(obstacle, dict) or not isinstance(lane, dict):
            return
        event = dict(prediction)
        event["request_id"] = request_id
        event["frame_id"] = frame_id
        with self._lock:
            if (
                self._model_prediction_history
                and self._model_prediction_history[-1].get("request_id") == request_id
            ):
                return
            self._model_prediction_history.append(event)
            if len(self._model_prediction_history) > 30:
                del self._model_prediction_history[:-30]

    def publish_images(self, images: dict[str, bytes]) -> None:
        with self._lock:
            for name, payload in images.items():
                if name not in {"camera", "combined", "line-mask", "drivable-area"}:
                    continue
                if payload:
                    self._images[name] = payload
                    self._image_versions[name] = self._image_versions.get(name, 0) + 1

    def image_snapshot(self, name: str) -> tuple[bytes | None, int]:
        with self._lock:
            return self._images.get(name), self._image_versions.get(name, 0)

    def capture_roadforge_frame(self) -> dict[str, Any]:
        """Persist the latest front-camera JPEG and its AVIS session telemetry."""
        with self._lock:
            payload = self._images.get("camera")
            version = self._image_versions.get("camera", 0)
            telemetry = dict(self._telemetry)
            mode = self._mode
            session_id = self.capture_session_id
        if not payload:
            raise RuntimeError("No front-camera frame is available yet. Wait for the live camera stream.")

        now = datetime.now().astimezone()
        filename = f"avis_{now.strftime('%Y%m%d_%H%M%S_%f')}_{uuid4().hex[:8]}.jpg"
        images_dir = self.capture_directory / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        image_path = images_dir / filename
        image_path.write_bytes(payload)
        entry = {
            "image": f"images/{filename}",
            "filename": filename,
            "captured_at": now.isoformat(timespec="milliseconds"),
            "session_id": session_id,
            "camera_view": "front_rgb",
            "camera_version": int(version),
            "mode": mode,
            "telemetry": telemetry,
        }
        manifest_path = self.capture_directory / "manifest.jsonl"
        with self._lock:
            with manifest_path.open("a", encoding="utf-8") as manifest:
                manifest.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
                manifest.flush()
        return {
            "captured": True,
            "filename": filename,
            "image": entry["image"],
            "path": str(image_path.resolve()),
            "manifest": str(manifest_path.resolve()),
            "session_id": session_id,
            "camera_version": int(version),
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            mode = self._mode
            manual_fresh = (
                mode == "manual"
                and self._manual_updated_at > 0.0
                and time.monotonic() - self._manual_updated_at
                <= self.manual_timeout_seconds
            )
            return {
                "settings": dict(self._settings),
                "mode": mode,
                "active_profile": self._active_profile_name,
                "manual_active": manual_fresh,
                "telemetry": dict(self._telemetry),
                "model_prediction_history": [
                    dict(item) for item in self._model_prediction_history
                ],
            }


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AVIS Control Studio</title>
<style>
:root{color-scheme:dark;--bg:#080d17;--panel:#101a2a;--panel2:#142238;--line:#26364d;--text:#e9f1fb;--muted:#8fa2bb;--accent:#52c7f5;--cyan:#80e4ff;--danger:#ff647c;--ok:#68dfa7;--radius:16px;font-family:Inter,Segoe UI,Arial,sans-serif}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 20% -10%,#15324a 0,transparent 42%),var(--bg);color:var(--text);min-height:100vh}
header{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:20px clamp(16px,4vw,48px);border-bottom:1px solid #1c2b40;background:#080d17c9;position:sticky;top:0;z-index:5;backdrop-filter:blur(14px)}
.brand{display:flex;gap:13px;align-items:center}.logo{width:42px;height:42px;border-radius:13px;background:linear-gradient(145deg,#52c7f5,#1775bc);display:grid;place-items:center;color:#03101c;font-weight:900;letter-spacing:-1px;box-shadow:0 8px 28px #1b9bd544}.brand h1{font-size:17px;margin:0 0 3px}.brand p{font-size:11px;color:var(--muted);margin:0;letter-spacing:.1em;text-transform:uppercase}
.status{display:flex;align-items:center;gap:9px;color:var(--muted);font-size:12px}.dot{width:8px;height:8px;border-radius:50%;background:#f1b74a;box-shadow:0 0 12px currentColor}.dot.online{background:var(--ok)}
main{max-width:1500px;margin:0 auto;padding:24px clamp(14px,3vw,38px) 48px}.topline{display:flex;align-items:end;justify-content:space-between;gap:20px;margin-bottom:18px}.eyebrow{color:var(--cyan);font-size:11px;text-transform:uppercase;letter-spacing:.14em;font-weight:700}.topline h2{font-size:clamp(22px,3vw,30px);margin:7px 0 4px;letter-spacing:-.035em}.muted{color:var(--muted);font-size:13px;line-height:1.5}
.grid{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(310px,.8fr);gap:16px;align-items:start}.stack{display:grid;gap:16px}.panel{background:linear-gradient(145deg,#142238ee,#0e1828ee);border:1px solid var(--line);border-radius:var(--radius);padding:16px;box-shadow:0 18px 50px #0002}.panel-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:13px}.panel h3{font-size:14px;margin:0}.tag{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:#a7bdd2;background:#1e3048;padding:6px 8px;border-radius:8px}
.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:16px}.metric{background:#0c1523;border:1px solid #21334a;border-radius:12px;padding:12px}.metric span{display:block;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.08em}.metric strong{display:block;margin-top:7px;font-size:20px;font-variant-numeric:tabular-nums}
.views{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.view{min-width:0;background:#070c14;border:1px solid #22344c;border-radius:12px;overflow:hidden}.view-title{padding:9px 11px;color:#a9bfd5;font-size:11px;display:flex;justify-content:space-between;gap:8px}.view img{width:100%;aspect-ratio:16/9;object-fit:contain;display:block;background:#05080e}.view img.roi-source{height:auto;aspect-ratio:auto;object-fit:contain}.view small{color:#71869e}
.roi-editor-wrap{position:relative;width:100%;background:#05080e;overflow:hidden}.roi-editor-wrap img{width:100%;height:auto;display:block}.roi-editor-wrap canvas{position:absolute;inset:0;width:100%;height:100%;touch-action:none;cursor:crosshair}.roi-toolbar{display:flex;align-items:center;gap:7px;flex-wrap:wrap;padding:10px 0 2px}.roi-toolbar .btn{padding:8px 11px}.roi-toolbar .btn.active{background:#123b54;border-color:#42bde9;color:#b9f0ff}.roi-help{font-size:10px;color:#839ab2;line-height:1.5;margin-top:5px}
.mode-row{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.btn{border:1px solid #304761;border-radius:10px;background:#1a2b42;color:var(--text);font:600 12px inherit;padding:11px 12px;cursor:pointer;transition:background .15s,border-color .15s,transform .15s}.btn:hover{background:#23415c;border-color:#4a8ab6}.btn:active{transform:scale(.985)}.btn.active{background:#123b54;border-color:#42bde9;color:#b9f0ff}.btn.stop{background:#3b1c2b;border-color:#713348;color:#ffc1cb}.btn.stop:hover{background:#542034}
.manual-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:15px}.control label,.field label{display:flex;justify-content:space-between;gap:10px;color:#a9bdd2;font-size:11px;margin-bottom:8px}.control output{color:var(--cyan);font-variant-numeric:tabular-nums}input[type=range]{width:100%;accent-color:var(--accent)}.keyhelp{font-size:10px;color:#7289a2;line-height:1.6;margin-top:10px}
.fields{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.field input,.field select{width:100%;background:#0a1422;border:1px solid #2a3d55;border-radius:9px;padding:9px 10px;color:var(--text);font:13px inherit;outline:none}.field input:focus,.field select:focus{border-color:var(--accent)}.field.full{grid-column:1/-1}.field.checkbox-field label{justify-content:flex-start;align-items:center;gap:10px;margin:0;cursor:pointer}.field.checkbox-field input[type=checkbox]{width:18px;height:18px;flex:0 0 18px;margin:0;padding:0;accent-color:var(--accent)}.field.checkbox-field span{color:var(--text);font-size:12px}.section-label{color:var(--cyan);font-size:10px;text-transform:uppercase;letter-spacing:.13em;margin:18px 0 10px}.section-label:first-child{margin-top:2px}.hint{font-size:10px;color:#748ba5;line-height:1.5;margin-top:5px}.actions{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-top:15px}.save{background:linear-gradient(120deg,#1688bb,#2166a8);border:0;color:white}.message{font-size:11px;color:var(--ok);min-height:16px}.classical-box{border-top:1px solid #26374b;margin-top:16px;padding-top:4px}.hidden{display:none!important}
.app-header{padding:16px clamp(16px,3vw,38px);border-bottom:1px solid #22364d;background:#071321ef}
.header-main{min-width:0}.header-eyebrow{color:var(--cyan);font-size:10px;font-weight:800;letter-spacing:.15em;text-transform:uppercase}
.header-title{font-size:clamp(18px,2.1vw,24px);font-weight:850;letter-spacing:-.035em;margin:4px 0}
.header-subtitle{color:var(--muted);font-size:11px;line-height:1.5}
.header-status{display:flex;justify-content:flex-end;align-items:center;gap:8px;flex-wrap:wrap}
.status-pill{display:inline-flex;align-items:center;gap:8px;border:1px solid #2a4059;background:#102137;border-radius:999px;padding:7px 11px;color:#b5c9dc;font-size:10px}
.right-panel{min-width:0;position:sticky;top:90px;max-height:calc(100vh - 106px);overflow:auto;scrollbar-width:thin;scrollbar-color:#35516d transparent}
.panel-tabs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:5px;padding:6px;background:#0b1726;border:1px solid #263b52;border-radius:12px;margin-bottom:12px;position:sticky;top:0;z-index:3}
.panel-tab{min-width:0;border:1px solid transparent;border-radius:8px;background:transparent;color:#91a8bf;padding:10px 7px;font:650 11px inherit;cursor:pointer}
.panel-tab:hover{color:#e9f1fb;background:#172b41}
.panel-tab.active{color:#b9efff;background:#123b54;border-color:#42a9d4}
.panel-content{display:none;gap:12px}
.panel-content.active{display:grid}
.panel-content>.panel,.panel-content>form.panel{margin:0}
.metrics{margin:0 0 2px}
.views .view:first-child,.views .view:last-child{grid-column:1/-1}
.roi-editor-wrap{background:#030812}
.roi-editor-wrap canvas{outline:0}
@media(max-width:980px){.grid{grid-template-columns:1fr}.topline{align-items:start;flex-direction:column}.metrics{grid-template-columns:repeat(2,1fr)}.right-panel{position:static;max-height:none;overflow:visible}}
@media(max-width:560px){header{padding:13px 14px}.brand h1{font-size:14px}.status{font-size:10px}.header-status{justify-content:flex-start}.views{grid-template-columns:1fr}.manual-grid,.fields{grid-template-columns:1fr}.panel{padding:13px}.metric strong{font-size:17px}.panel-tabs{position:static}}
/* AVIS performance telemetry */
.fps-metrics{grid-template-columns:repeat(3,minmax(0,1fr));margin-top:-6px}
.fps-metrics .metric{padding:9px 11px}
.fps-metrics .metric strong{font-size:18px}
@media(max-width:560px){.fps-metrics{grid-template-columns:repeat(2,minmax(0,1fr))}}
.model-prediction-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}
.model-prediction-card{min-width:0;background:#0b1625;border:1px solid #293d55;border-radius:12px;padding:12px}
.model-prediction-card span{display:block;color:#8fa2bb;font-size:10px;text-transform:uppercase;letter-spacing:.08em}
.model-prediction-card strong{display:block;margin-top:8px;font-size:19px;line-height:1.25;overflow-wrap:anywhere}
.model-prediction-card small{display:block;margin-top:6px;color:#91a8bf;font-size:11px;font-variant-numeric:tabular-nums}
.model-state-yes{color:#ff7b8e}.model-state-no{color:#68dfa7}.model-state-left{color:#80e4ff}.model-state-right{color:#f2c879}
.model-prediction-meta{margin-top:10px;color:#91a8bf;font-size:11px;font-variant-numeric:tabular-nums}
.model-history-head{display:flex;justify-content:space-between;align-items:center;gap:10px;margin:15px 0 8px;color:#9ab0c7;font-size:10px;text-transform:uppercase;letter-spacing:.1em}
.model-history-list{display:grid;gap:5px;max-height:245px;overflow:auto;scrollbar-width:thin;scrollbar-color:#35516d transparent}
.model-history-row{display:grid;grid-template-columns:.65fr 1fr 1fr .8fr;gap:8px;align-items:center;padding:8px 9px;border:1px solid #22364c;border-radius:8px;background:#0a1421;font-size:11px;font-variant-numeric:tabular-nums}
.model-history-row span{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.model-history-row .history-obstacle-yes{color:#ff7b8e}.model-history-row .history-obstacle-no{color:#68dfa7}
.model-history-row .history-lane-left{color:#80e4ff}.model-history-row .history-lane-right{color:#f2c879}
.model-history-empty{padding:12px;border:1px dashed #30455e;border-radius:9px;color:#8198b2;font-size:12px;text-align:center}
@media(max-width:560px){.model-prediction-grid{grid-template-columns:1fr 1fr}.model-prediction-card:last-child{grid-column:1/-1}.model-history-row{grid-template-columns:.6fr 1fr 1fr}.model-history-row span:last-child{display:none}}
</style>
</head>
<body>
<header class="app-header">
  <div class="header-main"><div class="header-eyebrow">Autonomous Driving · AVIS Runtime Console</div><div class="header-title">AVIS Runtime Studio</div><div class="header-subtitle">Live perception, vehicle control and AVIS-only calibration in one browser workspace.</div></div>
  <div class="header-status"><span class="status-pill"><span id="connectionDot" class="dot"></span><span id="connectionText">Connecting to AVIS</span></span><span class="status-pill" id="backendTag">VISION —</span></div>
</header>
<main>
  <div class="topline"><div><div class="eyebrow">Live control / perception</div><h2>Vehicle perception workspace</h2><div class="muted">All AVIS settings, including model paths, model/line selection, confidence thresholds and ROI calibration, are saved with the active profile and synced to GitHub when settings are applied.</div></div><div class="tag">AVIS ONLY</div></div>
  <div class="grid">
    <section class="stack">
      <div class="panel">
        <div class="panel-head"><h3>Perception monitor</h3><span class="tag">Live camera / masks</span></div>
        <div class="views">
          <div class="view"><div class="view-title"><span>AVIS camera / ROI editor</span><small>Drag to move · handles to resize</small></div><div class="roi-editor-wrap"><img id="roiSource" class="roi-source" src="/stream/camera" alt="AVIS camera"><canvas id="roiCanvas"></canvas></div><div class="roi-toolbar"><button class="btn active" id="roiLeftButton" type="button">Left ROI</button><button class="btn" id="roiRightButton" type="button">Right ROI</button><button class="btn" id="roiCrosswalkButton" type="button">Crosswalk ROI</button><button class="btn" id="resetRoiButton" type="button">Reset selected ROI</button><label class="hint"><input type="checkbox" id="showRois" checked> Show all ROIs</label><span class="message" id="roiCanvasMessage"></span></div><div class="roi-help">LL / RL rectangles control the lane_lines mask. CW is the independent Crosswalk detector crop (city mode only). Drag it to move, or resize it with its corner handles. ROI edits are saved to the active mode profile.</div></div>
          <div class="view"><div class="view-title"><span>Lane overlay</span><small>Detected boundaries</small></div><img src="/stream/combined" alt="Lane overlay"></div>
          <div class="view"><div class="view-title"><span>Lane-line mask</span><small>Red pixels = lane-line pixels used for tracking</small></div><img src="/stream/line-mask" alt="Lane-line mask"></div>
          <div class="view" id="drivableAreaView"><div class="view-title"><span>Drivable Area mask</span><small>Green pixels = model road mask</small></div><img src="/stream/drivable-area" alt="Model drivable area mask"></div>
        </div>
        <div class="hint" id="backendDescription">The semantic model's lane_lines pixels drive normal lane tracking. The green drivable-area mask is also available for optional, temporally filtered lane-centre correction.</div>
      </div>
    </section>
    <aside class="right-panel">
      <nav class="panel-tabs" aria-label="AVIS runtime workspace">
        <button type="button" class="panel-tab active" data-panel="live">Live monitor</button>
        <button type="button" class="panel-tab" data-panel="settings">Settings</button>
        <button type="button" class="panel-tab" data-panel="roi">ROI tools</button>
      </nav>
      <section class="panel-content active" id="panel-live">
  <div class="metrics">
    <div class="metric"><span>Speed</span><strong id="speedValue">0.0 <small>km/h</small></strong></div>
    <div class="metric"><span>Target speed</span><strong id="targetValue">0.0 <small>km/h</small></strong></div>
    <div class="metric"><span>Steering</span><strong id="steerValue">0.00</strong></div>
    <div class="metric"><span>Lane error</span><strong id="errorValue">-- <small>px</small></strong></div>
  </div>
  <div class="metrics fps-metrics" aria-label="Runtime performance">
    <div class="metric"><span>Frame read</span><strong id="frameReadFpsValue">0.0 <small>FPS</small></strong></div>
    <div class="metric"><span>Vision inference</span><strong id="inferenceFpsValue">0.0 <small>FPS</small></strong></div>
    <div class="metric"><span>Control loop</span><strong id="controlLoopFpsValue">0.0 <small>FPS</small></strong></div>
  </div>
      <div class="panel">
        <div class="panel-head"><h3>Drive mode</h3><span class="tag" id="modeTag">AUTO</span></div>
        <div class="mode-row">
          <button class="btn active" id="autoButton" type="button">Autonomous</button>
          <button class="btn" id="manualButton" type="button">Manual</button>
          <button class="btn stop" id="stopButton" type="button">Emergency stop</button>
        </div>
        <div class="manual-grid">
          <div class="control"><label for="throttleInput">Throttle / reverse <output id="throttleOutput">0%</output></label><input id="throttleInput" type="range" min="-100" max="100" value="0" step="1"></div>
          <div class="control"><label for="steeringInput">Steering <output id="steeringOutput">0%</output></label><input id="steeringInput" type="range" min="-100" max="100" value="0" step="1"></div>
        </div>
        <div class="keyhelp">Manual mode: W = forward, S = reverse, A / D = steer, Space = zero throttle and steering. The dashboard sends a short heartbeat; loss of heartbeat releases throttle.</div>
        <div class="actions"><button class="btn save" id="roadforgeCaptureButton" type="button">Capture frame for RoadForge</button><span class="message" id="roadforgeCaptureMessage" aria-live="polite"></span></div>
        <div class="hint">Saves the latest front-camera frame and AVIS telemetry to Documents/AVIS_RoadForge_Captures. This does not pause or modify vehicle control.</div>
      </div>
      <div class="panel">
        <div class="panel-head"><h3>Model inference outputs</h3><span class="tag" id="modelPredictionStatus">WAITING FOR MODEL</span></div>
        <div class="model-prediction-grid">
          <div class="model-prediction-card"><span>Obstacle ahead</span><strong id="modelObstacleLabel">--</strong><small id="modelObstacleConfidence">Confidence: --</small><small id="modelObstacleDecision">Waiting for confidence gate</small></div>
          <div class="model-prediction-card"><span>Model lane prediction</span><strong id="modelLaneLabel">--</strong><small id="modelLaneConfidence">Confidence: --</small></div>
          <div class="model-prediction-card"><span>Vehicle lane state</span><strong id="modelCurrentLane">--</strong><small id="modelLanePhase">Waiting for lane state</small></div>
        </div>
        <div class="model-prediction-meta" id="modelPredictionMeta">Waiting for the first completed classifier inference.</div>
        <div class="model-history-head"><span>Recent predictions · each completed inference</span><span id="modelPredictionCount">0 records</span></div>
        <div class="model-history-row" aria-hidden="true"><span>Frame</span><span>Obstacle</span><span>Lane output</span><span>Time</span></div>
        <div class="model-history-list" id="modelPredictionHistory" aria-live="polite"><div class="model-history-empty">No model predictions yet.</div></div>
      </div>
      <div class="panel">
        <div class="panel-head"><h3>Runtime telemetry</h3><span class="tag" id="runtimeStatus">Starting</span></div>
        <div class="muted" id="runtimeDetails">Waiting for the first camera frame.</div>
      </div>

      </section>
      <section class="panel-content" id="panel-settings">
      <form class="panel" id="settingsForm" novalidate>
        <div class="panel-head"><h3>AVIS runtime settings</h3><span class="tag">Live apply</span></div>
        <div class="section-label">Profiles · save / load / transfer</div>
        <div class="fields">
          <div class="field full"><label for="profileName">Profile name</label><input id="profileName" type="text" maxlength="48" placeholder="e.g. AVIS Night Track"></div>
          <div class="field full"><label for="localProfileSelect">Saved profiles</label><select id="localProfileSelect"></select></div>
        </div>
        <div class="actions">
          <button class="btn save" id="saveProfileButton" type="button">Save profile</button>
          <button class="btn" id="loadProfileButton" type="button">Load</button>
          <button class="btn" id="exportProfileButton" type="button">Export JSON</button>
          <button class="btn" id="importProfileButton" type="button">Import JSON</button>
          <button class="btn" id="deleteProfileButton" type="button">Delete</button>
          <input id="profileImportFile" type="file" accept=".json,application/json" hidden>
        </div>
        <div class="message" id="profilesMessage"></div>
        <div class="section-label">GitHub profile sync</div>
        <div class="fields">
          <div class="field full"><label for="githubProfileSelect">Profiles stored in GitHub · main / avis_profiles</label><select id="githubProfileSelect"><option value="">Refresh to load remote profiles</option></select></div>
        </div>
        <div class="actions">
          <button class="btn" id="refreshGitHubProfilesButton" type="button">Refresh GitHub</button>
          <button class="btn" id="publishGitHubProfileButton" type="button">Publish selected</button>
          <button class="btn" id="loadGitHubProfileButton" type="button">Load from GitHub</button>
        </div>
        <div class="hint" id="githubProfilesMessage">Publishing uses GitHub CLI authentication on this PC (gh auth login), or GITHUB_TOKEN / GH_TOKEN. No token is stored in the dashboard or project files.</div>
        <div class="divider"></div>
        <div class="section-label">Lane-line perception backend</div>
        <div class="fields">
          <div class="field full"><label for="vision_backend">Lane detection mode · model or line</label><select id="vision_backend" data-setting="vision_backend"><option value="ultralytics">AI segmentation model · lane_lines mask</option><option value="classical">Classical line detection · HLS / Canny / Hough</option></select></div>
          <div class="field full checkbox-field"><label for="auto_mode_default"><input id="auto_mode_default" type="checkbox" data-setting="auto_mode_default"><span>Start in autonomous mode after application launch</span></label><div class="hint">This affects the next AVIS runtime launch; it does not override the current drive mode.</div></div>
          <div class="field full"><label for="lane_model_path">Lane segmentation model checkpoint (.pt)</label><input id="lane_model_path" data-setting="lane_model_path" type="text" spellcheck="false" placeholder="models_and_datasets/Avis_models/lane_and_drivable-area_model_Avis.pt"></div>
          <div class="field full"><label for="obstacle_model_path">Obstacle / lane classifier model (.onnx)</label><input id="obstacle_model_path" data-setting="obstacle_model_path" type="text" spellcheck="false" placeholder="models_and_datasets/Avis_models/obstacle_and_lane_Avis_model.onnx"></div>
          <div class="field"><label for="model_image_size">Lane model inference size</label><input id="model_image_size" data-setting="model_image_size" type="number" min="160" max="2048" step="32"></div>
          <div class="field"><label for="lane_prob_threshold">Lane probability threshold</label><input id="lane_prob_threshold" data-setting="lane_prob_threshold" type="number" min="0" max="1" step=".01"></div>
          <div class="field"><label for="drivable_prob_threshold">Drivable-area probability threshold</label><input id="drivable_prob_threshold" data-setting="drivable_prob_threshold" type="number" min="0" max="1" step=".01"></div>
          <div class="field"><label for="debug_panel_hz">Debug panel update rate (Hz)</label><input id="debug_panel_hz" data-setting="debug_panel_hz" type="number" min="1" max="120" step="1"></div>
          <div class="field full checkbox-field"><label for="vision_debug"><input id="vision_debug" type="checkbox" data-setting="vision_debug"><span>Enable vision debug outputs</span></label></div>
          <div class="field checkbox-field"><label for="debug_show_rois"><input id="debug_show_rois" type="checkbox" data-setting="debug_show_rois"><span>Show ROI overlays</span></label></div>
          <div class="field checkbox-field"><label for="debug_show_lane_mask"><input id="debug_show_lane_mask" type="checkbox" data-setting="debug_show_lane_mask"><span>Show lane mask</span></label></div>
          <div class="field checkbox-field"><label for="debug_show_drivable_area"><input id="debug_show_drivable_area" type="checkbox" data-setting="debug_show_drivable_area"><span>Show drivable-area mask</span></label></div>
          <div class="field checkbox-field"><label for="debug_show_text"><input id="debug_show_text" type="checkbox" data-setting="debug_show_text"><span>Show debug text</span></label></div>
          <div class="field"><label for="sensor_angle">Normal sensor angle (degrees)</label><input id="sensor_angle" data-setting="sensor_angle" type="number" min="10" max="90" step="1"></div>
          <div class="field"><label for="control_loop_hz">Control frequency (Hz)</label><input id="control_loop_hz" data-setting="control_loop_hz" type="number" min="10" max="60" step="1"></div>
          <div class="field"><label for="inference_interval_ms">Inference interval (ms)</label><input id="inference_interval_ms" data-setting="inference_interval_ms" type="number" min="10" max="500" step="1"></div>
        </div>
        <div class="section-label">Speed controller</div>
        <div class="fields">
          <div class="field"><label for="min_speed_kmh">Minimum speed (km/h)</label><input id="min_speed_kmh" data-setting="min_speed_kmh" type="number" min="1" max="60" step=".5"></div>
          <div class="field"><label for="max_speed_kmh">Maximum speed (km/h)</label><input id="max_speed_kmh" data-setting="max_speed_kmh" type="number" min="1.5" max="100" step=".5"></div>
          <div class="field full"><label for="acceleration_mps2">Acceleration limit (m/s²)</label><input id="acceleration_mps2" data-setting="acceleration_mps2" type="number" min=".2" max="8" step=".1"></div>
        </div>
        <div class="section-label">Obstacle avoidance / automatic lane change</div>
        <div class="fields">
          <div class="field full checkbox-field"><label for="obstacle_detection_enabled"><input id="obstacle_detection_enabled" type="checkbox" data-setting="obstacle_detection_enabled"><span>Load and run the obstacle / lane ONNX classifier</span></label><div class="hint">Controls the ONNX classifier itself. City mode can turn it off without affecting the independent lane-segmentation model.</div></div>
          <div class="field full checkbox-field"><label for="obstacle_lane_change_enabled"><input id="obstacle_lane_change_enabled" type="checkbox" data-setting="obstacle_lane_change_enabled"><span>Enable automatic obstacle avoidance / lane changes</span></label><div class="hint">Controls lane-change behaviour only. Disabling this does not unload the classifier; use the setting above for that.</div></div>
          <div class="field"><label for="obstacle_model_confidence_threshold">Obstacle "yes" confidence</label><input id="obstacle_model_confidence_threshold" data-setting="obstacle_model_confidence_threshold" type="number" min=".5" max=".99" step=".01"></div>
          <div class="field"><label for="obstacle_lane_confidence_threshold">Lane prediction confidence</label><input id="obstacle_lane_confidence_threshold" data-setting="obstacle_lane_confidence_threshold" type="number" min=".5" max=".99" step=".01"></div>
          <div class="field"><label for="obstacle_model_confirm_frames">Model trigger confirmation frames</label><input id="obstacle_model_confirm_frames" data-setting="obstacle_model_confirm_frames" type="number" min="1" max="30" step="1"></div>
          <div class="field"><label for="obstacle_model_rearm_frames">Clear frames before next trigger</label><input id="obstacle_model_rearm_frames" data-setting="obstacle_model_rearm_frames" type="number" min="1" max="30" step="1"></div>
          <div class="field"><label for="obstacle_lane_change_speed_kmh">Lane-change speed (km/h)</label><input id="obstacle_lane_change_speed_kmh" data-setting="obstacle_lane_change_speed_kmh" type="number" min="1" max="20" step=".5"></div>
          <div class="field"><label for="obstacle_lane_change_steer_distance_m">Initial steering distance (m)</label><input id="obstacle_lane_change_steer_distance_m" data-setting="obstacle_lane_change_steer_distance_m" type="number" min=".1" max="100" step="any"></div>
          <div class="field"><label for="obstacle_lane_line_confirm_frames">Model lane confirmation frames</label><input id="obstacle_lane_line_confirm_frames" data-setting="obstacle_lane_line_confirm_frames" type="number" min="1" max="20" step="1"></div>
          <div class="field"><label for="obstacle_lane_change_left_steer">First left steering attempt</label><input id="obstacle_lane_change_left_steer" data-setting="obstacle_lane_change_left_steer" type="number" min=".001" max="1" step="any"></div>
          <div class="field"><label for="obstacle_lane_change_right_steer">First right steering attempt</label><input id="obstacle_lane_change_right_steer" data-setting="obstacle_lane_change_right_steer" type="number" min=".001" max="1" step="any"></div>
          <div class="field"><label for="obstacle_lane_change_retry_steer">Gentle retry steering</label><input id="obstacle_lane_change_retry_steer" data-setting="obstacle_lane_change_retry_steer" type="number" min=".001" max="1" step="any"></div>
          <div class="field"><label for="obstacle_lane_change_retry_pulse_distance_m">Gentle retry distance (m)</label><input id="obstacle_lane_change_retry_pulse_distance_m" data-setting="obstacle_lane_change_retry_pulse_distance_m" type="number" min=".05" max="10" step=".05"></div>
          <div class="field full"><div class="hint">During both lane changes, AVIS uses the dedicated low lane-change speed and shorter steering pulses. It keeps retrying until the model confirms the destination lane; no distance sensors are used.</div></div>
        </div>
        <div class="section-label">City crosswalk detection and stop</div>
        <div class="fields">
          <div class="field full checkbox-field"><label for="city_crosswalk_enabled"><input id="city_crosswalk_enabled" type="checkbox" data-setting="city_crosswalk_enabled"><span>Enable CARLA-style Crosswalk ROI detection in City mode</span></label><div class="hint">Uses the same OpenCV grayscale, Gaussian blur, Canny and HoughLinesP detector with temporal confirmation as CARLA's non-model junction mode. It runs independently of the lane-segmentation model.</div></div>
          <div class="field"><label for="city_crosswalk_stop_seconds">Stop duration (seconds)</label><input id="city_crosswalk_stop_seconds" data-setting="city_crosswalk_stop_seconds" type="number" min="0.5" max="120" step="0.5"></div>
          <div class="field"><label for="city_crosswalk_rearm_clear_seconds">Clear time before re-arming (seconds)</label><input id="city_crosswalk_rearm_clear_seconds" data-setting="city_crosswalk_rearm_clear_seconds" type="number" min="0.2" max="30" step="0.1"></div>
          <div class="field"><label for="city_crosswalk_min_line_length_ratio">Minimum line length / ROI width</label><input id="city_crosswalk_min_line_length_ratio" data-setting="city_crosswalk_min_line_length_ratio" type="number" min="0.1" max="0.95" step="0.01"></div>
          <div class="field"><label for="city_crosswalk_max_line_angle_deg">Maximum line angle (degrees)</label><input id="city_crosswalk_max_line_angle_deg" data-setting="city_crosswalk_max_line_angle_deg" type="number" min="1" max="30" step="0.5"></div>
          <div class="field"><label for="city_crosswalk_confirm_frames">Confirmation frames (out of 3)</label><input id="city_crosswalk_confirm_frames" data-setting="city_crosswalk_confirm_frames" type="number" min="1" max="3" step="1"></div>
          <div class="field full"><div class="hint">Edit the Crosswalk ROI in ROI tools. One confirmed marking starts one stop interval; the detector re-arms after the marking disappears for the configured clear time, preventing repeated stops on the same crosswalk.</div></div>
        </div>
        <div class="section-label">Return to the right lane</div>
        <div class="hint">After the model confirms the vehicle has entered the left lane, AVIS travels exactly 4 metres, then starts the right-lane change. The return uses the same model-confirmed steering pulse, pause, and gentler retry sequence as the initial move to the left. Right-side sensor readings and sensor timeouts do not control this return.</div>
        <div class="section-label">Additional obstacle and lane-change tuning</div>
        <div class="fields">
          <div class="field"><label for="obstacle_front_threshold_cm">Front sensor obstacle threshold (cm)</label><input id="obstacle_front_threshold_cm" data-setting="obstacle_front_threshold_cm" type="number" min="50" max="1499" step="1"></div>
          <div class="field"><label for="obstacle_trigger_confirm_frames">Sensor trigger confirmation frames</label><input id="obstacle_trigger_confirm_frames" data-setting="obstacle_trigger_confirm_frames" type="number" min="1" max="30" step="1"></div>
          <div class="field"><label for="obstacle_sensor_clear_threshold_cm">Sensor clear threshold (cm)</label><input id="obstacle_sensor_clear_threshold_cm" data-setting="obstacle_sensor_clear_threshold_cm" type="number" min="51" max="1500" step="1"></div>
          <div class="field"><label for="obstacle_sensor_clear_confirm_frames">Sensor clear confirmation frames</label><input id="obstacle_sensor_clear_confirm_frames" data-setting="obstacle_sensor_clear_confirm_frames" type="number" min="1" max="30" step="1"></div>
          <div class="field"><label for="obstacle_pass_sensor_angle">Pass-check sensor angle</label><input id="obstacle_pass_sensor_angle" data-setting="obstacle_pass_sensor_angle" type="number" min="10" max="90" step="1"></div>
          <div class="field"><label for="obstacle_pass_sensor_presence_threshold_cm">Pass sensor presence threshold (cm)</label><input id="obstacle_pass_sensor_presence_threshold_cm" data-setting="obstacle_pass_sensor_presence_threshold_cm" type="number" min="20" max="1499" step="1"></div>
          <div class="field"><label for="obstacle_pass_sensor_clear_threshold_cm">Pass sensor clear threshold (cm)</label><input id="obstacle_pass_sensor_clear_threshold_cm" data-setting="obstacle_pass_sensor_clear_threshold_cm" type="number" min="21" max="1500" step="1"></div>
          <div class="field"><label for="obstacle_pass_sensor_presence_frames">Pass sensor presence frames</label><input id="obstacle_pass_sensor_presence_frames" data-setting="obstacle_pass_sensor_presence_frames" type="number" min="1" max="30" step="1"></div>
          <div class="field"><label for="obstacle_pass_sensor_clear_frames">Pass sensor clear frames</label><input id="obstacle_pass_sensor_clear_frames" data-setting="obstacle_pass_sensor_clear_frames" type="number" min="1" max="30" step="1"></div>
          <div class="field"><label for="obstacle_pass_clearance_distance_m">Pass clearance distance (m)</label><input id="obstacle_pass_clearance_distance_m" data-setting="obstacle_pass_clearance_distance_m" type="number" min=".1" max="100" step=".1"></div>
          <div class="field"><label for="obstacle_return_delay_seconds">Return delay (seconds)</label><input id="obstacle_return_delay_seconds" data-setting="obstacle_return_delay_seconds" type="number" min="0" max="60" step=".1"></div>
          <div class="field"><label for="obstacle_pass_sensor_timeout_seconds">Pass sensor timeout (seconds)</label><input id="obstacle_pass_sensor_timeout_seconds" data-setting="obstacle_pass_sensor_timeout_seconds" type="number" min="1" max="120" step="1"></div>
          <div class="field"><label for="obstacle_lane_line_search_max_distance_m">Lane search limit (m)</label><input id="obstacle_lane_line_search_max_distance_m" data-setting="obstacle_lane_line_search_max_distance_m" type="number" min=".1" max="1000" step=".1"></div>
          <div class="field"><label for="obstacle_clearance_after_pass_m">Clearance after pass (m)</label><input id="obstacle_clearance_after_pass_m" data-setting="obstacle_clearance_after_pass_m" type="number" min="0" max="1000" step=".1"></div>
        </div>
        <div class="hint">All values in this section are included in local and GitHub profiles for complete configuration portability. Sensor-based fields are retained for compatible controller profiles; the current model-confirmed lane-change controller does not use them for pass confirmation.</div>
        <div class="section-label">Drivable-area path correction</div>
        <div class="fields">
          <div class="field full"><label for="drivable_recovery_enabled">Use drivable-area lane correction</label><select id="drivable_recovery_enabled" data-setting="drivable_recovery_enabled"><option value="true">Enabled</option><option value="false">Disabled</option></select></div>
        </div>
        <div class="hint">Uses the lower central DW ROI (x=10–90%, y=60–72%) to estimate the road centre. Isolated mask speckles and short flicker are filtered using row-wise outlier rejection, a five-inference median and a 1.5-second persistence gate. The left correction lasts up to 10 seconds and is rate-limited by the steering controller. This option requires the Ultralytics backend.</div>
        <div class="section-label">Adaptive lane ROI recovery</div>
        <div class="fields">
          <div class="field full"><label for="adaptive_roi_enabled">Adaptive ROI expansion</label><select id="adaptive_roi_enabled" data-setting="adaptive_roi_enabled"><option value="true">Enabled</option><option value="false">Disabled</option></select></div>
          <div class="field"><label for="adaptive_roi_max_expansion_ratio">Maximum expansion (frame ratio)</label><input id="adaptive_roi_max_expansion_ratio" data-setting="adaptive_roi_max_expansion_ratio" type="number" min="0" max=".30" step=".01"></div>
          <div class="field"><label for="adaptive_roi_edge_trigger_ratio">Edge trigger ratio</label><input id="adaptive_roi_edge_trigger_ratio" data-setting="adaptive_roi_edge_trigger_ratio" type="number" min=".05" max=".45" step=".01"></div>
          <div class="field"><label for="adaptive_roi_expand_step_ratio">Expansion per frame</label><input id="adaptive_roi_expand_step_ratio" data-setting="adaptive_roi_expand_step_ratio" type="number" min=".001" max=".05" step=".001"></div>
          <div class="field"><label for="adaptive_roi_recovery_frames">Center confirmation frames</label><input id="adaptive_roi_recovery_frames" data-setting="adaptive_roi_recovery_frames" type="number" min="1" max="60" step="1"></div>
          <div class="field full"><label for="adaptive_roi_restore_step_ratio">Restore step per frame</label><input id="adaptive_roi_restore_step_ratio" data-setting="adaptive_roi_restore_step_ratio" type="number" min=".001" max=".05" step=".001"></div>
        </div>
        <div class="hint">The last 7 inference results are tracked independently for LL and RL. When a marking trends toward an ROI edge, that side expands temporarily; after the marking stays near the center of its original ROI for the configured number of frames, the rectangle gradually returns to its saved size.</div>
        <div class="section-label">Steering PID</div>
        <div class="fields">
          <div class="field"><label for="kp">Kp</label><input id="kp" data-setting="kp" type="number" min="0" max=".2" step=".001"></div>
          <div class="field"><label for="ki">Ki</label><input id="ki" data-setting="ki" type="number" min="0" max=".1" step=".001"></div>
          <div class="field"><label for="kd">Kd</label><input id="kd" data-setting="kd" type="number" min="0" max=".1" step=".001"></div>
          <div class="field"><label for="steer_limit">Steering limit</label><input id="steer_limit" data-setting="steer_limit" type="number" min=".05" max="1" step=".01"></div>
          <div class="field full"><label for="max_steer_step">Maximum steering change / tick</label><input id="max_steer_step" data-setting="max_steer_step" type="number" min=".01" max="1" step=".01"></div>
        </div>
        <div class="classical-box" id="classicalSettings">
          <div class="section-label">Classical lane detection</div>
          <div class="fields">
            <div class="field"><label for="blur_kernel">Gaussian blur kernel</label><input id="blur_kernel" data-setting="blur_kernel" type="number" min="1" max="31" step="2"></div>
            <div class="field"><label for="morphology_kernel">Morphology kernel</label><input id="morphology_kernel" data-setting="morphology_kernel" type="number" min="1" max="9" step="1"></div>
            <div class="field"><label for="canny_low">Canny low threshold</label><input id="canny_low" data-setting="canny_low" type="number" min="0" max="254" step="1"></div>
            <div class="field"><label for="canny_high">Canny high threshold</label><input id="canny_high" data-setting="canny_high" type="number" min="1" max="255" step="1"></div>
            <div class="field"><label for="hough_threshold">Hough threshold</label><input id="hough_threshold" data-setting="hough_threshold" type="number" min="1" max="200" step="1"></div>
            <div class="field"><label for="hough_max_line_gap">Hough max gap (px)</label><input id="hough_max_line_gap" data-setting="hough_max_line_gap" type="number" min="0" max="250" step="1"></div>
            <div class="field"><label for="hough_min_line_length_ratio">Min line length / width</label><input id="hough_min_line_length_ratio" data-setting="hough_min_line_length_ratio" type="number" min=".005" max=".5" step=".005"></div>
            <div class="field"><label for="min_side_pixels">Minimum mask pixels / side</label><input id="min_side_pixels" data-setting="min_side_pixels" type="number" min="1" max="500" step="1"></div>
            <div class="field"><label for="lane_offset_ratio">Lane offset / frame width</label><input id="lane_offset_ratio" data-setting="lane_offset_ratio" type="number" min=".03" max=".45" step=".01"></div>
            <div class="field"><label for="white_l_min">White lightness min</label><input id="white_l_min" data-setting="white_l_min" type="number" min="0" max="255" step="1"></div>
            <div class="field"><label for="white_s_max">White saturation max</label><input id="white_s_max" data-setting="white_s_max" type="number" min="0" max="255" step="1"></div>
            <div class="field"><label for="yellow_h_min">Yellow hue min</label><input id="yellow_h_min" data-setting="yellow_h_min" type="number" min="0" max="179" step="1"></div>
            <div class="field"><label for="yellow_h_max">Yellow hue max</label><input id="yellow_h_max" data-setting="yellow_h_max" type="number" min="0" max="179" step="1"></div>
            <div class="field"><label for="yellow_s_min">Yellow saturation min</label><input id="yellow_s_min" data-setting="yellow_s_min" type="number" min="0" max="255" step="1"></div>
            <div class="field"><label for="yellow_l_min">Yellow lightness min</label><input id="yellow_l_min" data-setting="yellow_l_min" type="number" min="0" max="255" step="1"></div>

            <div class="field"><label for="min_lane_angle_deg">Minimum lane angle (°)</label><input id="min_lane_angle_deg" data-setting="min_lane_angle_deg" type="number" min="0" max="60" step="1"></div>
            <div class="field"><label for="max_lane_angle_deg">Maximum lane angle (°)</label><input id="max_lane_angle_deg" data-setting="max_lane_angle_deg" type="number" min="10" max="89" step="1"></div>
            <div class="field"><label for="center_smoothing">Center smoothing (α)</label><input id="center_smoothing" data-setting="center_smoothing" type="number" min=".01" max="1" step=".01"></div>
            <div class="field"><label for="yellow_enabled">Detect yellow markings</label><select id="yellow_enabled" data-setting="yellow_enabled"><option value="true">Enabled</option><option value="false">Disabled</option></select></div>
          </div>
          <div class="hint">Only pixels inside each LL/RL crop are processed. Hough fragments are grouped and scored by line coverage and support. LL keeps the rightmost stable line and RL the leftmost stable line; a more extreme candidate must persist across three inference frames before switching. A short detection dropout holds the previous track for up to two frames instead of merging all lines.</div>
        </div>
        <div class="actions"><button class="btn save" type="submit">Apply and save settings</button><span class="message" id="settingsMessage"></span></div>
      </form>
      </section>
      <section class="panel-content" id="panel-roi">
        <div class="panel">
          <div class="panel-head"><h3>Lane ROI calibration</h3><span class="tag">LL / RL</span></div>
          <div class="hint">Edit normalized coordinates directly or drag the coloured rectangles on the live camera. This calibration is shared by both AVIS vision backends.</div>
        <div class="section-label">Normalized ROI coordinates</div>
        <div class="fields">
          <div class="field"><label for="ll_top_roi">Left ROI · top</label><input id="ll_top_roi" data-setting="ll_top_roi" type="number" min="0" max=".98" step=".01"></div>
          <div class="field"><label for="ll_bottom_roi">Left ROI · bottom</label><input id="ll_bottom_roi" data-setting="ll_bottom_roi" type="number" min=".02" max="1" step=".01"></div>
          <div class="field"><label for="ll_left_roi">Left ROI · left</label><input id="ll_left_roi" data-setting="ll_left_roi" type="number" min="0" max=".98" step=".01"></div>
          <div class="field"><label for="ll_right_roi">Left ROI · right</label><input id="ll_right_roi" data-setting="ll_right_roi" type="number" min=".02" max="1" step=".01"></div>
          <div class="field"><label for="rl_top_roi">Right ROI · top</label><input id="rl_top_roi" data-setting="rl_top_roi" type="number" min="0" max=".98" step=".01"></div>
          <div class="field"><label for="rl_bottom_roi">Right ROI · bottom</label><input id="rl_bottom_roi" data-setting="rl_bottom_roi" type="number" min=".02" max="1" step=".01"></div>
          <div class="field"><label for="rl_left_roi">Right ROI · left</label><input id="rl_left_roi" data-setting="rl_left_roi" type="number" min="0" max=".98" step=".01"></div>
          <div class="field"><label for="rl_right_roi">Right ROI · right</label><input id="rl_right_roi" data-setting="rl_right_roi" type="number" min=".02" max="1" step=".01"></div>
          <div class="field"><label for="cw_top_roi">Crosswalk ROI · top</label><input id="cw_top_roi" data-setting="cw_top_roi" type="number" min="0" max=".98" step=".01"></div>
          <div class="field"><label for="cw_bottom_roi">Crosswalk ROI · bottom</label><input id="cw_bottom_roi" data-setting="cw_bottom_roi" type="number" min=".02" max="1" step=".01"></div>
          <div class="field"><label for="cw_left_roi">Crosswalk ROI · left</label><input id="cw_left_roi" data-setting="cw_left_roi" type="number" min="0" max=".98" step=".01"></div>
          <div class="field"><label for="cw_right_roi">Crosswalk ROI · right</label><input id="cw_right_roi" data-setting="cw_right_roi" type="number" min=".02" max="1" step=".01"></div>
        </div>
        <div class="hint">Values are normalized from 0 to 1. Use the live camera overlay to move and resize the rectangles, or enter exact coordinates here.</div>

          <div class="actions"><button class="btn save" id="saveRoiButton" type="button">Save selected ROI</button><span class="message" id="roiMessage"></span></div>
          <div class="hint">Tip: keep each rectangle tightly around its corresponding lane marking. Avoid including road texture, vehicle edges or neighbouring lane markings.</div>
        </div>
      </section>
    </aside>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
let currentMode = "auto";
let manualThrottle = 0;
let manualSteering = 0;
const settingInputs = Array.from(document.querySelectorAll("[data-setting]"));
const roiCanvas = $("roiCanvas");
const roiSource = $("roiSource");
const roiContext = roiCanvas ? roiCanvas.getContext("2d") : null;
const ROI_GROUPS = {
  LL: {keys:["ll_top_roi","ll_bottom_roi","ll_left_roi","ll_right_roi"], color:"#52e0a0", label:"LEFT LANE ROI"},
  RL: {keys:["rl_top_roi","rl_bottom_roi","rl_left_roi","rl_right_roi"], color:"#58c9ff", label:"RIGHT LANE ROI"},
  CW: {keys:["cw_top_roi","cw_bottom_roi","cw_left_roi","cw_right_roi"], color:"#ffd27b", label:"CROSSWALK ROI"}
};
const ROI_PROFILE_DEFAULTS = {
  ll_top_roi:0.57, ll_bottom_roi:0.69, ll_left_roi:0.24, ll_right_roi:0.45,
  rl_top_roi:0.57, rl_bottom_roi:0.69, rl_left_roi:0.56, rl_right_roi:0.78,
  cw_top_roi:0.55, cw_bottom_roi:0.88, cw_left_roi:0.08, cw_right_roi:0.92
};
let selectedROI = "LL";
let roiDrag = null;
let roiCanvasWidth = 0;
let roiCanvasHeight = 0;
function roiRect(group) {
  const keys = ROI_GROUPS[group].keys;
  return {
    top:Number($(keys[0]).value), bottom:Number($(keys[1]).value),
    left:Number($(keys[2]).value), right:Number($(keys[3]).value)
  };
}
function setROIInput(key, value) {
  $(key).value = Math.max(0,Math.min(1,value)).toFixed(3);
}
function drawROIOverlays() {
  if (!roiCanvas || !roiSource || !roiContext || !roiSource.complete || !roiSource.naturalWidth) return;
  const rect = roiCanvas.getBoundingClientRect();
  if (rect.width < 2 || rect.height < 2) return;
  const dpr = Math.max(1,window.devicePixelRatio || 1);
  const pixelWidth = Math.round(rect.width*dpr);
  const pixelHeight = Math.round(rect.height*dpr);
  if (roiCanvas.width !== pixelWidth || roiCanvas.height !== pixelHeight) {
    roiCanvas.width = pixelWidth;
    roiCanvas.height = pixelHeight;
  }
  if (roiCanvasWidth !== rect.width || roiCanvasHeight !== rect.height) {
    roiCanvasWidth = rect.width;
    roiCanvasHeight = rect.height;
  }
  const ctx = roiContext;
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,rect.width,rect.height);
  if (!$("showRois").checked) return;

  for (const group of ["LL","RL","CW"]) {
    const values = roiRect(group);
    const x = values.left*rect.width;
    const y = values.top*rect.height;
    const width = (values.right-values.left)*rect.width;
    const height = (values.bottom-values.top)*rect.height;
    if (width <= 0 || height <= 0) continue;
    const selected = group === selectedROI;
    const color = ROI_GROUPS[group].color;
    ctx.save();
    ctx.fillStyle = color;
    ctx.globalAlpha = selected ? 0.20 : 0.10;
    ctx.fillRect(x,y,width,height);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = selected ? "#ffffff" : color;
    ctx.lineWidth = selected ? 2.5 : 1.5;
    ctx.setLineDash(selected ? [] : [5,4]);
    ctx.strokeRect(x,y,width,height);
    ctx.setLineDash([]);
    ctx.font = "600 10px Segoe UI,Arial";
    const label = ROI_GROUPS[group].label;
    const labelWidth = Math.min(width,ctx.measureText(label).width+12);
    ctx.fillStyle = color;
    ctx.fillRect(x+2,y+2,labelWidth,17);
    ctx.fillStyle = "#06111d";
    ctx.fillText(label,x+7,y+14);
    if (selected) {
      const handles = [[x,y],[x+width,y],[x,y+height],[x+width,y+height]];
      ctx.fillStyle = "#ffffff";
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      for (const [hx,hy] of handles) {
        ctx.beginPath();
        ctx.rect(hx-5,hy-5,10,10);
        ctx.fill();
        ctx.stroke();
      }
    }
    ctx.restore();
  }
}
function selectROI(group) {
  selectedROI = group;
  $("roiLeftButton").classList.toggle("active",group==="LL");
  $("roiRightButton").classList.toggle("active",group==="RL");
  $("roiCrosswalkButton").classList.toggle("active",group==="CW");
  drawROIOverlays();
}
function setActivePanel(panelName) {
  document.querySelectorAll(".panel-tab").forEach(button =>
    button.classList.toggle("active", button.dataset.panel === panelName)
  );
  document.querySelectorAll(".panel-content").forEach(panel =>
    panel.classList.toggle("active", panel.id === "panel-" + panelName)
  );
  requestAnimationFrame(drawROIOverlays);
}
document.querySelectorAll(".panel-tab").forEach(button =>
  button.addEventListener("click", () => setActivePanel(button.dataset.panel))
);
async function saveROISelection() {
  const keys = ROI_GROUPS[selectedROI].keys;
  const payload = {};
  keys.forEach(key => payload[key] = Number($(key).value));
  try {
    const data = await postJson("/api/settings",payload);
    keys.forEach(key => $(key).value = data.settings[key]);
    drawROIOverlays();
    $("settingsForm").dataset.dirty = "true";
    const syncSuffix = data.github_synced ? " · GitHub synced" : " · GitHub: " + (data.github_message || "not synced");
    showMessage(selectedROI + " ROI saved" + syncSuffix + ".", !data.github_synced);
    $("roiCanvasMessage").textContent = selectedROI + " ROI saved" + syncSuffix + ".";
    $("roiCanvasMessage").style.color = data.github_synced ? "#68dfa7" : "#ffb36b";
    return true;
  } catch (error) {
    showMessage(error.message,true);
    $("roiCanvasMessage").textContent = error.message;
    $("roiCanvasMessage").style.color = "#ff8a9d";
    return false;
  }
}
function getROIPointerPosition(event) {
  const bounds = roiCanvas.getBoundingClientRect();
  return {
    x:(event.clientX-bounds.left)/bounds.width,
    y:(event.clientY-bounds.top)/bounds.height,
    width:bounds.width,
    height:bounds.height
  };
}
function hitTestROI(position) {
  const values = roiRect(selectedROI);
  const corners = {
    nw:{x:values.left,y:values.top},
    ne:{x:values.right,y:values.top},
    sw:{x:values.left,y:values.bottom},
    se:{x:values.right,y:values.bottom}
  };
  const toleranceX = 12/position.width;
  const toleranceY = 12/position.height;
  for (const [name,point] of Object.entries(corners)) {
    if (Math.abs(position.x-point.x)<=toleranceX && Math.abs(position.y-point.y)<=toleranceY) {
      return {type:"resize",corner:name,values};
    }
  }
  if (position.x>=values.left && position.x<=values.right &&
      position.y>=values.top && position.y<=values.bottom) {
    return {type:"move",values};
  }
  return {type:"none",values};
}
if (roiCanvas) {
  roiCanvas.addEventListener("pointerdown",event=>{
    const position = getROIPointerPosition(event);
    const hit = hitTestROI(position);
    if (hit.type==="none") return;
    event.preventDefault();
    roiCanvas.setPointerCapture(event.pointerId);
    roiDrag = {
      pointerId:event.pointerId,
      type:hit.type,
      corner:hit.corner,
      start:position,
      original:{...hit.values}
    };
    roiCanvas.style.cursor = hit.type==="move" ? "grabbing" : "nwse-resize";
  });
  roiCanvas.addEventListener("pointermove",event=>{
    const position = getROIPointerPosition(event);
    if (!roiDrag || roiDrag.pointerId!==event.pointerId) {
      const hit = hitTestROI(position);
      roiCanvas.style.cursor = hit.type==="move" ? "move" : hit.type==="resize" ? "nwse-resize" : "crosshair";
      return;
    }
    const dx = position.x-roiDrag.start.x;
    const dy = position.y-roiDrag.start.y;
    const next = {...roiDrag.original};
    if (roiDrag.type==="move") {
      const width = next.right-next.left;
      const height = next.bottom-next.top;
      next.left = Math.max(0,Math.min(1-width,next.left+dx));
      next.right = next.left+width;
      next.top = Math.max(0,Math.min(1-height,next.top+dy));
      next.bottom = next.top+height;
    } else {
      const corner = roiDrag.corner;
      if (corner.includes("n")) next.top = Math.max(0,Math.min(next.bottom-0.02,roiDrag.original.top+dy));
      if (corner.includes("s")) next.bottom = Math.max(next.top+0.02,Math.min(1,roiDrag.original.bottom+dy));
      if (corner.includes("w")) next.left = Math.max(0,Math.min(next.right-0.02,roiDrag.original.left+dx));
      if (corner.includes("e")) next.right = Math.max(next.left+0.02,Math.min(1,roiDrag.original.right+dx));
    }
    const keys = ROI_GROUPS[selectedROI].keys;
    setROIInput(keys[0],next.top);
    setROIInput(keys[1],next.bottom);
    setROIInput(keys[2],next.left);
    setROIInput(keys[3],next.right);
    drawROIOverlays();
  });
  const finishROIDrag = async event=>{
    if (!roiDrag || roiDrag.pointerId!==event.pointerId) return;
    roiDrag = null;
    roiCanvas.style.cursor = "crosshair";
    await saveROISelection();
  };
  roiCanvas.addEventListener("pointerup",finishROIDrag);
  roiCanvas.addEventListener("pointercancel",finishROIDrag);
}
if ($("roiLeftButton")) $("roiLeftButton").addEventListener("click",()=>selectROI("LL"));
if ($("roiRightButton")) $("roiRightButton").addEventListener("click",()=>selectROI("RL"));
if ($("roiCrosswalkButton")) $("roiCrosswalkButton").addEventListener("click",()=>selectROI("CW"));
if ($("showRois")) $("showRois").addEventListener("change",drawROIOverlays);
if ($("saveRoiButton")) $("saveRoiButton").addEventListener("click",async()=>{
  const saved = await saveROISelection();
  $("roiMessage").textContent = saved ? selectedROI + " ROI saved." : "ROI could not be saved.";
  $("roiMessage").style.color = saved ? "#68dfa7" : "#ff8a9d";
});
if ($("resetRoiButton")) $("resetRoiButton").addEventListener("click",async()=>{
  const keys = ROI_GROUPS[selectedROI].keys;
  const payload = {};
  keys.forEach(key=>payload[key]=ROI_PROFILE_DEFAULTS[key]);
  try {
    const data=await postJson("/api/settings",payload);
    keys.forEach(key => $(key).value = data.settings[key]);
    drawROIOverlays();
    $("settingsForm").dataset.dirty = "true";
    showMessage(selectedROI+" ROI reset to AVIS_StableV2.",false);
  } catch(error) {
    showMessage(error.message,true);
  }
});
Object.values(ROI_GROUPS).flatMap(group=>group.keys).forEach(key=>{
  const input=$(key);
  if(input) input.addEventListener("input",drawROIOverlays);
});
window.addEventListener("resize",drawROIOverlays);
setInterval(drawROIOverlays,120);
selectROI("LL");
function showMessage(text, isError) {
  $("settingsMessage").textContent = text;
  $("settingsMessage").style.color = isError ? "#ff8a9d" : "#68dfa7";
}
function setModeButtons(mode) {
  currentMode = mode;
  $("modeTag").textContent = mode.toUpperCase();
  $("autoButton").classList.toggle("active", mode === "auto");
  $("manualButton").classList.toggle("active", mode === "manual");
}
function updateBackendVisibility(backend) {
  const classical = backend === "classical";
  $("classicalSettings").classList.toggle("hidden", !classical);
  $("backendDescription").textContent = classical
    ? "Classical CV continues to process its own LL/RL crops. Its backend does not provide a semantic drivable-area mask, so drivable-area path correction is unavailable."
    : "Model lane_lines pixels drive normal tracking; an ROI with insufficient model pixels uses the classical HLS/Canny/Hough fallback. The semantic drivable-area mask can optionally provide temporally filtered lane correction.";
}
function fillSettings(settings) {
  settingInputs.forEach(input => {
    const key = input.dataset.setting;
    if (!(key in settings)) return;
    if (input.type === "checkbox") {
      input.checked = Boolean(settings[key]);
    } else if (input.tagName === "SELECT" && (key === "yellow_enabled" || key === "adaptive_roi_enabled" || key === "drivable_recovery_enabled")) {
      input.value = settings[key] ? "true" : "false";
    } else {
      input.value = settings[key];
    }
  });
  updateBackendVisibility(settings.vision_backend);
  drawROIOverlays();
}
function collectSettings() {
  const result = {};
  settingInputs.forEach(input => {
    const key = input.dataset.setting;
    if (input.type === "checkbox") result[key] = input.checked;
    else if (input.tagName === "SELECT" && key === "vision_backend") result[key] = input.value;
    else if (input.tagName === "SELECT" && (key === "yellow_enabled" || key === "adaptive_roi_enabled" || key === "drivable_recovery_enabled")) result[key] = input.value === "true";
    else if (input.type === "text") {
      const raw = input.value.trim();
      if (raw !== "") result[key] = raw;
    } else {
      const raw = input.value.trim();
      if (raw === "") return;
      const numeric = Number(raw);
      if (Number.isFinite(numeric)) result[key] = numeric;
    }
  });
  return result;
}
async function postJson(url, payload) {
  const response = await fetch(url, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Request failed");
  return data;
}
$("vision_backend").addEventListener("change", () => {
  updateBackendVisibility($("vision_backend").value);
});
let settingsEditRevision = 0;
$("settingsForm").addEventListener("submit", async event => {
  event.preventDefault();
  const form = $("settingsForm");
  const submitButton = form.querySelector('button[type="submit"]');
  const submittedRevision = settingsEditRevision;
  submitButton.disabled = true;
  showMessage("Applying settings...", false);
  try {
    const data = await postJson("/api/settings", collectSettings());
    const syncMessage = data.github_synced
      ? " · GitHub profile " + data.profile_name + " updated"
      : " · GitHub not synced: " + (data.github_message || "authentication required");
    if (settingsEditRevision === submittedRevision) {
      fillSettings(data.settings);
      delete form.dataset.dirty;
      showMessage("Settings applied and saved locally" + syncMessage + ".", !data.github_synced);
    } else {
      form.dataset.dirty = "true";
      showMessage("Settings saved" + syncMessage + ". New edits are still pending; apply again.", !data.github_synced);
    }
  } catch (error) {
    form.dataset.dirty = "true";
    showMessage(error.message, true);
  } finally {
    submitButton.disabled = false;
  }
});
async function changeMode(mode) {
  try {
    const data = await postJson("/api/mode", {mode});
    setModeButtons(data.mode);
    manualThrottle = 0;
    manualSteering = 0;
    $("throttleInput").value = "0";
    $("steeringInput").value = "0";
    updateManualLabels();
  } catch (error) {
    alert(error.message);
  }
}
$("autoButton").addEventListener("click", () => changeMode("auto"));
$("manualButton").addEventListener("click", () => changeMode("manual"));
$("stopButton").addEventListener("click", async () => {
  try {
    await postJson("/api/stop", {});
    setModeButtons("stopped");
    manualThrottle = 0;
    manualSteering = 0;
    $("throttleInput").value = "0";
    $("steeringInput").value = "0";
    updateManualLabels();
  } catch (error) {
    alert(error.message);
  }
});
function updateManualLabels() {
  $("throttleOutput").textContent = (manualThrottle < 0 ? "REV " : "FWD ") + Math.round(Math.abs(manualThrottle) * 100) + "%";
  $("steeringOutput").textContent = Math.round(manualSteering * 100) + "%";
}
$("throttleInput").addEventListener("input", event => {
  manualThrottle = Number(event.target.value) / 100;
  updateManualLabels();
});
$("steeringInput").addEventListener("input", event => {
  manualSteering = Number(event.target.value) / 100;
  updateManualLabels();
});
const pressed = new Set();
document.addEventListener("keydown", event => {
  const key = event.key.toLowerCase();
  if (!["w","s","a","d"," "].includes(key) || currentMode !== "manual") return;
  event.preventDefault();
  pressed.add(key);
  manualThrottle = pressed.has(" ") ? 0 : pressed.has("w") ? 0.42 : pressed.has("s") ? -0.30 : 0;
  manualSteering = pressed.has(" ") ? 0 : (pressed.has("d") ? 0.45 : 0) - (pressed.has("a") ? 0.45 : 0);
  $("throttleInput").value = String(Math.round(manualThrottle * 100));
  $("steeringInput").value = String(Math.round(manualSteering * 100));
  updateManualLabels();
});
document.addEventListener("keyup", event => {
  const key = event.key.toLowerCase();
  pressed.delete(key);
  if (currentMode === "manual" && ["w","s","a","d"," "].includes(key)) {
    manualThrottle = pressed.has(" ") ? 0 : pressed.has("w") ? 0.42 : pressed.has("s") ? -0.30 : 0;
    manualSteering = pressed.has(" ") ? 0 : (pressed.has("d") ? 0.45 : 0) - (pressed.has("a") ? 0.45 : 0);
    $("throttleInput").value = String(Math.round(manualThrottle * 100));
    $("steeringInput").value = String(Math.round(manualSteering * 100));
    updateManualLabels();
  }
});
window.addEventListener("blur", () => {
  pressed.clear();
  manualThrottle = 0;
  manualSteering = 0;
  $("throttleInput").value = "0";
  $("steeringInput").value = "0";
  updateManualLabels();
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    manualThrottle = 0;
    manualSteering = 0;
    $("throttleInput").value = "0";
    $("steeringInput").value = "0";
    updateManualLabels();
  }
});
async function heartbeat() {
  if (currentMode !== "manual" || document.hidden) return;
  try {
    await postJson("/api/manual", {throttle:manualThrottle, steering:manualSteering});
  } catch (error) {
    setModeButtons("stopped");
  }
}
function modelConfidenceText(value) {
  const confidence = Number(value);
  return Number.isFinite(confidence) ? "Confidence: " + (confidence * 100).toFixed(1) + "%" : "Confidence: --";
}
function modelPredictionTime(value) {
  const timestamp = Number(value);
  return Number.isFinite(timestamp) && timestamp > 0
    ? new Date(timestamp * 1000).toLocaleTimeString()
    : "--";
}
function renderModelPredictions(history, telemetry) {
  const records = Array.isArray(history) ? history : [];
  const latest = records.length ? records[records.length - 1] : null;
  const obstacle = latest?.obstacle || telemetry.obstacle_classifier?.obstacle_ahead || null;
  const lane = latest?.lane || telemetry.obstacle_classifier?.lane || null;
  const obstacleLabel = String(obstacle?.label || "").toLowerCase();
  const laneLabel = String(lane?.label || "").toLowerCase();

  const obstacleValue = $("modelObstacleLabel");
  obstacleValue.textContent = obstacleLabel === "yes" ? "YES · OBSTACLE" : obstacleLabel === "no" ? "NO · CLEAR" : "WAITING";
  obstacleValue.className = obstacleLabel === "yes" ? "model-state-yes" : obstacleLabel === "no" ? "model-state-no" : "";
  $("modelObstacleConfidence").textContent = modelConfidenceText(obstacle?.confidence);
  const threshold = Number(telemetry.obstacle_model_confidence_threshold ?? 0.65);
  const thresholdText = Number.isFinite(threshold) ? (threshold * 100).toFixed(0) + "%" : "--";
  const accepted = Boolean(telemetry.obstacle_model_accepted);
  const avoidanceEnabled = telemetry.obstacle_avoidance_enabled !== false;
  $("modelObstacleDecision").textContent = !avoidanceEnabled
    ? "AVOIDANCE DISABLED"
    : obstacleLabel === "yes"
      ? (accepted ? "ACCEPTED FOR AVOIDANCE" : "REJECTED · BELOW THRESHOLD") + " · Gate " + thresholdText
      : obstacleLabel === "no" ? "CLEAR · Gate " + thresholdText : "Waiting for confidence gate";
  $("modelObstacleDecision").className = accepted ? "model-state-yes" : obstacleLabel === "yes" ? "model-state-no" : "";

  const laneValue = $("modelLaneLabel");
  laneValue.textContent = laneLabel ? laneLabel.toUpperCase() : "WAITING";
  laneValue.className = laneLabel === "left" ? "model-state-left" : laneLabel === "right" ? "model-state-right" : "";
  $("modelLaneConfidence").textContent = modelConfidenceText(lane?.confidence);

  const currentLane = String(telemetry.lane_change_side || "").toLowerCase();
  const currentLaneValue = $("modelCurrentLane");
  currentLaneValue.textContent = currentLane ? currentLane.toUpperCase() : "--";
  currentLaneValue.className = currentLane === "left" ? "model-state-left" : currentLane === "right" ? "model-state-right" : "";
  $("modelLanePhase").textContent = String(telemetry.lane_change_phase || "Waiting for lane state").replaceAll("_", " ");

  if (latest) {
    const ageMs = Math.max(0, Date.now() - Number(latest.recorded_at || 0) * 1000);
    $("modelPredictionStatus").textContent = ageMs < 1500 ? "LIVE · MODEL OUTPUT" : "LAST OUTPUT · STALE";
    $("modelPredictionMeta").textContent =
      String(latest.backbone || "Classifier") +
      " · Inference #" + String(latest.request_id) +
      " · Camera frame #" + String(latest.frame_id) +
      " · " + modelPredictionTime(latest.recorded_at) +
      " · " + (ageMs < 1000 ? ageMs.toFixed(0) + " ms ago" : (ageMs / 1000).toFixed(1) + " s ago");
  } else {
    $("modelPredictionStatus").textContent = "WAITING FOR MODEL";
    $("modelPredictionMeta").textContent = "No completed classifier inference has been received yet.";
  }

  const list = $("modelPredictionHistory");
  list.replaceChildren();
  const visible = records.slice(-15).reverse();
  $("modelPredictionCount").textContent = records.length + " stored · " + visible.length + " shown";
  if (!visible.length) {
    const empty = document.createElement("div");
    empty.className = "model-history-empty";
    empty.textContent = "No model predictions yet.";
    list.appendChild(empty);
    return;
  }
  for (const item of visible) {
    const row = document.createElement("div");
    row.className = "model-history-row";
    const obstacleName = String(item.obstacle?.label || "--").toLowerCase();
    const laneName = String(item.lane?.label || "--").toLowerCase();
    const cells = [
      "#" + String(item.frame_id ?? "--"),
      obstacleName === "yes" ? "YES · " + (Number(item.obstacle?.confidence || 0) * 100).toFixed(0) + "%" : obstacleName === "no" ? "NO · " + (Number(item.obstacle?.confidence || 0) * 100).toFixed(0) + "%" : "--",
      (laneName === "left" || laneName === "right" ? laneName.toUpperCase() : "--") + " · " + modelConfidenceText(item.lane?.confidence).replace("Confidence: ", ""),
      modelPredictionTime(item.recorded_at),
    ];
    cells.forEach((value, index) => {
      const cell = document.createElement("span");
      cell.textContent = value;
      if (index === 1) cell.className = obstacleName === "yes" ? "history-obstacle-yes" : obstacleName === "no" ? "history-obstacle-no" : "";
      if (index === 2) cell.className = laneName === "left" ? "history-lane-left" : laneName === "right" ? "history-lane-right" : "";
      row.appendChild(cell);
    });
    list.appendChild(row);
  }
}
async function refreshStatus() {
  try {
    const response = await fetch("/api/status", {cache:"no-store"});
    if (!response.ok) throw new Error("Runtime unavailable");
    const data = await response.json();
    const t = data.telemetry || {};
    renderModelPredictions(data.model_prediction_history || [], t);
    $("connectionDot").classList.toggle("online", !!t.connected);
    $("connectionText").textContent = t.connected ? "AVIS connected" : "Waiting for AVIS";
    $("backendTag").textContent = "VISION: " + String(data.settings.vision_backend).toUpperCase();
    $("speedValue").innerHTML = Number(t.speed_kmh || 0).toFixed(1) + " <small>km/h</small>";
    $("targetValue").innerHTML = Number(t.target_speed_kmh || 0).toFixed(1) + " <small>km/h</small>";
    $("steerValue").textContent = Number(t.steering || 0).toFixed(2);
    $("errorValue").innerHTML = (t.lane_valid ? Number(t.lane_error_px || 0).toFixed(1) : "--") + " <small>px</small>";
    $("frameReadFpsValue").innerHTML = Number(t.frame_read_fps || 0).toFixed(1) + " <small>FPS</small>";
    $("inferenceFpsValue").innerHTML = Number(t.inference_fps || 0).toFixed(1) + " <small>FPS</small>";
    $("controlLoopFpsValue").innerHTML = Number(t.control_loop_fps || 0).toFixed(1) + " <small>FPS</small>";
    $("runtimeStatus").textContent = t.status || "Running";
    $("runtimeDetails").textContent =
      "Drive mode: " + data.mode.toUpperCase() +
      " · Throttle: " + Math.round(Number(t.throttle || 0) * 100) + "%" +
      " · Lane: " + (t.lane_valid ? "tracking" : "not detected") +
      " · Sensors (L / C / R): " + (Array.isArray(t.sensors_cm) ? t.sensors_cm.join(" / ") + " cm" : "--") +
      " · Lane-change: " + String(t.lane_change_phase || "RIGHT_LANE").replaceAll("_"," ") +
      " · Side: " + String(t.lane_change_side || "right").toUpperCase() +
      " · Attempt: " + (Number(t.lane_change_attempt_number || 0) > 0 ? "#" + String(t.lane_change_attempt_number) : "--") +
      " · Model lane confirm: " + String(t.model_lane_confirm_frames || 0) + "/" + String(data.settings.obstacle_lane_line_confirm_frames || 3) +
      " · Return distance: " + Number(t.return_distance_travelled_m || 0).toFixed(1) + "/" + Number(t.return_after_distance_m || 10).toFixed(1) + " m" +
      " · Obstacle model: " + String(t.obstacle_classifier?.obstacle_ahead?.label || "--") +
      " (" + (t.obstacle_model_probability == null ? "--" : (Number(t.obstacle_model_probability) * 100).toFixed(0) + "%") + ")" +
       " · Avoidance gate: " + (t.obstacle_avoidance_enabled === false ? "DISABLED" : t.obstacle_model_accepted ? "ACCEPTED" : "REJECTED") +
       " · Confidence threshold: " + (Number(t.obstacle_model_confidence_threshold ?? data.settings.obstacle_model_confidence_threshold ?? 0.65) * 100).toFixed(0) + "%" +
      " · Lane choice: " + String(t.obstacle_lane_prediction || "--") +
      " · Sensor angle: " + (t.active_sensor_angle == null ? "--" : String(t.active_sensor_angle) + "°") +
      (t.vision_error ? " · Vision: " + t.vision_error : "");
    if (data.mode !== currentMode) setModeButtons(data.mode);
    if (!$("settingsForm").dataset.dirty) fillSettings(data.settings);
  } catch (error) {
    $("connectionDot").classList.remove("online");
    $("connectionText").textContent = "Dashboard disconnected";
  }
}
settingInputs.forEach(input => input.addEventListener("input", () => {
  settingsEditRevision += 1;
  $("settingsForm").dataset.dirty = "true";
  updateBackendVisibility($("vision_backend").value);
  if (input.dataset.setting.endsWith("_roi")) drawROIOverlays();
}));
$("settingsForm").addEventListener("change", () => {
  settingsEditRevision += 1;
  $("settingsForm").dataset.dirty = "true";
});

async function refreshLocalProfiles(selectedName) {
  const response = await fetch("/api/profiles", {cache:"no-store"});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Could not read local profiles.");
  const select = $("localProfileSelect");
  const previous = selectedName || select.value;
  select.replaceChildren();
  for (const profile of (data.profiles || [])) {
    const option = document.createElement("option");
    option.value = profile.name;
    option.textContent = profile.name;
    select.appendChild(option);
  }
  if (previous && Array.from(select.options).some(option => option.value === previous)) {
    select.value = previous;
  }
  if (select.value) $("profileName").value = select.value;
}
async function refreshGitHubProfiles() {
  const message = $("githubProfilesMessage");
  message.textContent = "Reading public AVIS profiles from GitHub...";
  try {
    const response = await fetch("/api/github/profiles", {cache:"no-store"});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not read GitHub profiles.");
    const select = $("githubProfileSelect");
    select.replaceChildren();
    for (const name of (data.profiles || [])) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      select.appendChild(option);
    }
    if (!select.options.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No profiles found";
      select.appendChild(option);
    }
    message.textContent = (data.profiles || []).length + " GitHub profile(s) available.";
  } catch (error) {
    message.textContent = error.message;
  }
}
$("localProfileSelect").addEventListener("change", () => {
  $("profileName").value = $("localProfileSelect").value;
});
$("saveProfileButton").addEventListener("click", async () => {
  const name = $("profileName").value.trim();
  if (!name) { $("profilesMessage").textContent = "Enter a profile name first."; return; }
  try {
    const saved = await postJson("/api/profiles/save", {name, settings: collectSettings()});
    fillSettings(saved.settings);
    delete $("settingsForm").dataset.dirty;
    await refreshLocalProfiles(name);
    $("profilesMessage").textContent = saved.github_synced
      ? "Profile saved locally and published to GitHub."
      : "Profile saved locally. GitHub: " + (saved.github_message || "not published.");
  } catch (error) {
    $("profilesMessage").textContent = error.message;
  }
});
$("loadProfileButton").addEventListener("click", async () => {
  const name = $("localProfileSelect").value;
  if (!name) return;
  try {
    const data = await postJson("/api/profiles/load", {name});
    fillSettings(data.settings);
    delete $("settingsForm").dataset.dirty;
    settingsEditRevision += 1;
    $("profileName").value = name;
    $("profilesMessage").textContent = "Loaded profile: " + name;
  } catch (error) { $("profilesMessage").textContent = error.message; }
});
$("deleteProfileButton").addEventListener("click", async () => {
  const name = $("localProfileSelect").value;
  if (!name || !confirm("Delete profile '" + name + "'?")) return;
  try {
    await postJson("/api/profiles/delete", {name});
    await refreshLocalProfiles();
    $("profilesMessage").textContent = "Deleted profile: " + name;
  } catch (error) { $("profilesMessage").textContent = error.message; }
});
$("exportProfileButton").addEventListener("click", async () => {
  const name = $("localProfileSelect").value;
  if (!name) return;
  try {
    const response = await fetch("/api/profiles/export?name=" + encodeURIComponent(name));
    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.error || "Could not export profile.");
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = name.replace(/[^A-Za-z0-9_.-]+/g, "_") + ".json";
    link.click();
    URL.revokeObjectURL(url);
    $("profilesMessage").textContent = "Exported " + name + ".json";
  } catch (error) { $("profilesMessage").textContent = error.message; }
});
$("importProfileButton").addEventListener("click", () => $("profileImportFile").click());
$("profileImportFile").addEventListener("change", async event => {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  try {
    const document = JSON.parse(await file.text());
    const data = await postJson("/api/profiles/import", document);
    await refreshLocalProfiles(data.profile.name);
    fillSettings(data.settings);
    delete $("settingsForm").dataset.dirty;
    settingsEditRevision += 1;
    $("profilesMessage").textContent = "Imported and loaded profile: " + data.profile.name;
  } catch (error) { $("profilesMessage").textContent = error.message; }
  finally { event.target.value = ""; }
});
$("refreshGitHubProfilesButton").addEventListener("click", refreshGitHubProfiles);
$("publishGitHubProfileButton").addEventListener("click", async () => {
  const name = $("localProfileSelect").value;
  if (!name) return;
  try {
    await postJson("/api/github/profiles/publish", {name});
    $("githubProfilesMessage").textContent = "Published " + name + " to GitHub.";
    await refreshGitHubProfiles();
  } catch (error) { $("githubProfilesMessage").textContent = error.message; }
});
$("loadGitHubProfileButton").addEventListener("click", async () => {
  const name = $("githubProfileSelect").value;
  if (!name) return;
  try {
    const data = await postJson("/api/github/profiles/load", {name});
    await refreshLocalProfiles(data.profile.name);
    fillSettings(data.settings);
    delete $("settingsForm").dataset.dirty;
    settingsEditRevision += 1;
    $("profilesMessage").textContent = "Loaded GitHub profile: " + name;
    $("githubProfilesMessage").textContent = "Downloaded and applied " + name + ".";
  } catch (error) { $("githubProfilesMessage").textContent = error.message; }
});

const roadforgeCaptureButton = $("roadforgeCaptureButton");
if (roadforgeCaptureButton) {
  roadforgeCaptureButton.addEventListener("click", async () => {
    const message = $("roadforgeCaptureMessage");
    roadforgeCaptureButton.disabled = true;
    message.textContent = "Saving latest camera frame…";
    message.style.color = "#68dfa7";
    try {
      const response = await fetch("/api/roadforge/capture", {method: "POST"});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Could not capture the AVIS frame.");
      message.textContent = "Captured " + data.filename + " · session " + data.session_id.slice(0, 8);
    } catch (error) {
      message.textContent = error.message;
      message.style.color = "#ff8a9d";
    } finally {
      roadforgeCaptureButton.disabled = false;
    }
  });
}

refreshLocalProfiles().catch(error => {
  $("profilesMessage").textContent = error.message;
});
updateManualLabels();
refreshStatus();
setInterval(refreshStatus, 500);
setInterval(heartbeat, 140);
</script>
</body>
</html>
"""


def create_avis_dashboard_app(state: AvisDashboardState) -> Flask:
    app = Flask("avis_dashboard")
    app.config["JSON_SORT_KEYS"] = False

    @app.get("/")
    def index():
        return Response(DASHBOARD_HTML, mimetype="text/html")

    @app.get("/api/status")
    def status():
        return jsonify(state.snapshot())

    @app.get("/api/profiles")
    def profiles():
        return jsonify({"profiles": state.profiles_snapshot()})

    @app.post("/api/profiles/save")
    def save_profile():
        payload = request.get_json(silent=True) or {}
        try:
            if isinstance(payload.get("settings"), dict):
                state.update_settings(payload["settings"])
            profile = state.save_profile(payload.get("name"))
        except (TypeError, ValueError, OSError) as exc:
            return jsonify({"error": str(exc)}), 400
        github_synced = False
        github_message = "Run 'gh auth login' on this PC, then publish the profile to GitHub."
        if request.remote_addr in {"127.0.0.1", "::1", "localhost"}:
            try:
                state.github_sync.publish_profile(profile)
                github_synced = True
                github_message = "Published to GitHub."
            except Exception as exc:
                github_message = str(exc)
        else:
            github_message = "GitHub publishing is restricted to a local dashboard connection."
        current_settings, _ = state.settings_snapshot()
        return jsonify({
            "profile": profile,
            "settings": current_settings,
            "profile_name": profile["name"],
            "github_synced": github_synced,
            "github_message": github_message,
        })

    @app.post("/api/profiles/load")
    def load_profile():
        payload = request.get_json(silent=True) or {}
        try:
            settings = state.load_profile(payload.get("name"))
        except (TypeError, ValueError, OSError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"settings": settings})

    @app.post("/api/profiles/delete")
    def delete_profile():
        payload = request.get_json(silent=True) or {}
        try:
            state.delete_profile(payload.get("name"))
        except (TypeError, ValueError, OSError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"deleted": True})

    @app.post("/api/profiles/import")
    def import_profile():
        payload = request.get_json(silent=True)
        try:
            profile = state.import_profile(payload)
            settings = state.load_profile(profile["name"])
        except (TypeError, ValueError, OSError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"profile": profile, "settings": settings})

    @app.get("/api/profiles/export")
    def export_profile():
        try:
            profile = state.export_profile(request.args.get("name"))
        except (TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", profile["name"]) + ".json"
        return Response(
            json.dumps(profile, indent=2, ensure_ascii=False) + "\n",
            mimetype="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/github/profiles")
    def github_profiles():
        try:
            return jsonify({"profiles": state.github_sync.list_profiles()})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 502

    @app.post("/api/github/profiles/publish")
    def publish_github_profile():
        if request.remote_addr not in {"127.0.0.1", "::1", "localhost"}:
            return jsonify({"error": "GitHub publishing is restricted to a local dashboard connection."}), 403
        payload = request.get_json(silent=True) or {}
        try:
            profile = state.get_profile(payload.get("name"))
            result = state.github_sync.publish_profile(profile)
        except (TypeError, ValueError, OSError, RuntimeError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"published": True, "result": result})

    @app.post("/api/github/profiles/load")
    def load_github_profile():
        payload = request.get_json(silent=True) or {}
        try:
            remote = state.github_sync.load_profile(str(payload.get("name") or ""))
            profile = state.import_profile(remote)
            settings = state.load_profile(profile["name"])
        except (TypeError, ValueError, OSError, RuntimeError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"profile": profile, "settings": settings})

    @app.post("/api/settings")
    def settings():
        payload = request.get_json(silent=True)
        try:
            updated = state.update_settings(payload)
            # Applying settings or saving an ROI also updates the active local
            # profile and publishes the complete settings snapshot to GitHub.
            profile = state.save_profile(state.active_profile_name())
        except (TypeError, ValueError, OSError) as exc:
            return jsonify({"error": str(exc)}), 400

        github_synced = False
        github_message = "Run 'gh auth login' on this PC, then apply settings again to sync GitHub."
        if request.remote_addr in {"127.0.0.1", "::1", "localhost"}:
            try:
                result = state.github_sync.publish_profile(profile)
                github_synced = True
                github_message = str(result.get("html_url") or "Published to GitHub.")
            except Exception as exc:
                github_message = str(exc)
        else:
            github_message = "Settings were saved locally; GitHub publishing requires a local dashboard connection."
        return jsonify({
            "settings": updated,
            "profile_name": profile["name"],
            "github_synced": github_synced,
            "github_message": github_message,
        })

    @app.post("/api/mode")
    def mode():
        payload = request.get_json(silent=True) or {}
        try:
            selected = state.set_mode(payload.get("mode"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"mode": selected})

    @app.post("/api/manual")
    def manual():
        payload = request.get_json(silent=True) or {}
        try:
            state.set_manual_controls(
                payload.get("throttle", 0.0),
                payload.get("steering", 0.0),
            )
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"accepted": True})

    @app.post("/api/stop")
    def stop():
        state.set_mode("stopped")
        return jsonify({"mode": "stopped"})

    @app.post("/api/roadforge/capture")
    def capture_for_roadforge():
        if request.remote_addr not in {"127.0.0.1", "::1", "localhost"}:
            return jsonify({"error": "AVIS frame capture is restricted to the local dashboard."}), 403
        try:
            result = state.capture_roadforge_frame()
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        except OSError as exc:
            return jsonify({"error": f"Could not write the capture: {exc}"}), 500
        return jsonify(result)

    @app.get("/stream/<view_name>")
    def stream(view_name: str):
        if view_name not in {"camera", "combined", "line-mask", "drivable-area"}:
            return jsonify({"error": "Unknown image view."}), 404

        def generate():
            last_version = -1
            while True:
                payload, version = state.image_snapshot(view_name)
                if payload is not None and version != last_version:
                    last_version = version
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Cache-Control: no-cache\r\n\r\n"
                        + payload
                        + b"\r\n"
                    )
                time.sleep(0.06)

        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    return app


class AvisDashboardServer:
    def __init__(
        self,
        state: AvisDashboardState,
        host: str = "127.0.0.1",
        port: int = 5001,
    ) -> None:
        self.state = state
        self.host = host
        self.port = int(port)
        self._server: BaseWSGIServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._server is not None:
            return
        app = create_avis_dashboard_app(self.state)
        try:
            self._server = make_server(
                self.host,
                self.port,
                app,
                threaded=True,
            )
        except OSError as exc:
            raise RuntimeError(
                f"Could not start AVIS dashboard on {self.host}:{self.port}: {exc}"
            ) from exc
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="avis-flask-dashboard",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        server = self._server
        if server is None:
            return
        self._server = None
        server.shutdown()
        server.server_close()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
