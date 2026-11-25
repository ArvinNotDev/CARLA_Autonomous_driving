from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request
from werkzeug.serving import BaseWSGIServer, make_server


DEFAULT_AVIS_SETTINGS: dict[str, Any] = {
    "vision_backend": "ultralytics",
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
}


_NUMERIC_LIMITS: dict[str, tuple[float, float, bool]] = {
    "control_loop_hz": (10.0, 60.0, False),
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
}


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
        if key == "yellow_enabled":
            if not isinstance(value, bool):
                raise ValueError("yellow_enabled must be a boolean.")
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
    ) -> None:
        self._lock = threading.RLock()
        self.settings_path = Path(settings_path)
        self.manual_timeout_seconds = max(0.15, float(manual_timeout_seconds))
        self._settings = dict(DEFAULT_AVIS_SETTINGS)
        if initial_settings:
            self._settings = validate_avis_settings(self._settings, initial_settings)
        self._load_saved_settings()
        self._settings_revision = 0
        self._mode = "auto"
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
            "vision_error": None,
        }
        self._images: dict[str, bytes] = {}
        self._image_versions: dict[str, int] = {}

    def _load_saved_settings(self) -> None:
        if not self.settings_path.is_file():
            return
        try:
            saved = json.loads(self.settings_path.read_text(encoding="utf-8"))
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

    def publish_telemetry(self, telemetry: dict[str, Any]) -> None:
        with self._lock:
            self._telemetry = dict(telemetry)

    def publish_images(self, images: dict[str, bytes]) -> None:
        with self._lock:
            for name, payload in images.items():
                if name not in {"camera", "combined", "line-mask", "drivable-mask"}:
                    continue
                if payload:
                    self._images[name] = payload
                    self._image_versions[name] = self._image_versions.get(name, 0) + 1

    def image_snapshot(self, name: str) -> tuple[bytes | None, int]:
        with self._lock:
            return self._images.get(name), self._image_versions.get(name, 0)

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
                "manual_active": manual_fresh,
                "telemetry": dict(self._telemetry),
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
.views{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.view{min-width:0;background:#070c14;border:1px solid #22344c;border-radius:12px;overflow:hidden}.view-title{padding:9px 11px;color:#a9bfd5;font-size:11px;display:flex;justify-content:space-between;gap:8px}.view img{width:100%;aspect-ratio:16/9;object-fit:contain;display:block;background:#05080e}.view small{color:#71869e}
.mode-row{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.btn{border:1px solid #304761;border-radius:10px;background:#1a2b42;color:var(--text);font:600 12px inherit;padding:11px 12px;cursor:pointer;transition:background .15s,border-color .15s,transform .15s}.btn:hover{background:#23415c;border-color:#4a8ab6}.btn:active{transform:scale(.985)}.btn.active{background:#123b54;border-color:#42bde9;color:#b9f0ff}.btn.stop{background:#3b1c2b;border-color:#713348;color:#ffc1cb}.btn.stop:hover{background:#542034}
.manual-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:15px}.control label,.field label{display:flex;justify-content:space-between;gap:10px;color:#a9bdd2;font-size:11px;margin-bottom:8px}.control output{color:var(--cyan);font-variant-numeric:tabular-nums}input[type=range]{width:100%;accent-color:var(--accent)}.keyhelp{font-size:10px;color:#7289a2;line-height:1.6;margin-top:10px}
.fields{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.field input,.field select{width:100%;background:#0a1422;border:1px solid #2a3d55;border-radius:9px;padding:9px 10px;color:var(--text);font:13px inherit;outline:none}.field input:focus,.field select:focus{border-color:var(--accent)}.field.full{grid-column:1/-1}.section-label{color:var(--cyan);font-size:10px;text-transform:uppercase;letter-spacing:.13em;margin:18px 0 10px}.section-label:first-child{margin-top:2px}.hint{font-size:10px;color:#748ba5;line-height:1.5;margin-top:5px}.actions{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-top:15px}.save{background:linear-gradient(120deg,#1688bb,#2166a8);border:0;color:white}.message{font-size:11px;color:var(--ok);min-height:16px}.classical-box{border-top:1px solid #26374b;margin-top:16px;padding-top:4px}.hidden{display:none!important}
@media(max-width:980px){.grid{grid-template-columns:1fr}.topline{align-items:start;flex-direction:column}.metrics{grid-template-columns:repeat(2,1fr)}}
@media(max-width:560px){header{padding:13px 14px}.brand h1{font-size:14px}.status{font-size:10px}.views{grid-template-columns:1fr}.manual-grid,.fields{grid-template-columns:1fr}.panel{padding:13px}.metric strong{font-size:17px}}
</style>
</head>
<body>
<header>
  <div class="brand"><div class="logo">AV</div><div><h1>AVIS Control Studio</h1><p>Independent vehicle runtime</p></div></div>
  <div class="status"><span id="connectionDot" class="dot"></span><span id="connectionText">Connecting to runtime</span></div>
</header>
<main>
  <div class="topline"><div><div class="eyebrow">Live control / perception</div><h2>AVIS vehicle dashboard</h2><div class="muted">Runtime settings are isolated in avis_settings.json and never write to CARLA profiles.</div></div><div class="tag" id="backendTag">VISION: --</div></div>
  <div class="metrics">
    <div class="metric"><span>Speed</span><strong id="speedValue">0.0 <small>km/h</small></strong></div>
    <div class="metric"><span>Target speed</span><strong id="targetValue">0.0 <small>km/h</small></strong></div>
    <div class="metric"><span>Steering</span><strong id="steerValue">0.00</strong></div>
    <div class="metric"><span>Lane error</span><strong id="errorValue">-- <small>px</small></strong></div>
  </div>
  <div class="grid">
    <section class="stack">
      <div class="panel">
        <div class="panel-head"><h3>Perception monitor</h3><span class="tag">Live camera / masks</span></div>
        <div class="views">
          <div class="view"><div class="view-title"><span>AVIS camera</span><small>Input · BGR</small></div><img src="/stream/camera" alt="AVIS camera"></div>
          <div class="view"><div class="view-title"><span>Lane overlay</span><small>Detected boundaries</small></div><img src="/stream/combined" alt="Lane overlay"></div>
          <div class="view"><div class="view-title"><span>Lane mask</span><small>Binary lane boundaries</small></div><img src="/stream/line-mask" alt="Lane mask"></div>
          <div class="view"><div class="view-title"><span>Drivable area</span><small>Semantic or estimated corridor</small></div><img src="/stream/drivable-mask" alt="Drivable-area mask"></div>
        </div>
        <div class="hint">Classical mode estimates a drivable corridor from lane boundaries. It is not a semantic road-surface model.</div>
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
      </div>
      <div class="panel">
        <div class="panel-head"><h3>Runtime telemetry</h3><span class="tag" id="runtimeStatus">Starting</span></div>
        <div class="muted" id="runtimeDetails">Waiting for the first camera frame.</div>
      </div>
    </section>
    <aside class="stack">
      <form class="panel" id="settingsForm">
        <div class="panel-head"><h3>AVIS runtime settings</h3><span class="tag">Live apply</span></div>
        <div class="section-label">Vision engine</div>
        <div class="fields">
          <div class="field full"><label for="vision_backend">Vision backend</label><select id="vision_backend" data-setting="vision_backend"><option value="ultralytics">Ultralytics semantic model</option><option value="classical">Classical CV · HLS / Canny / Hough</option></select></div>
          <div class="field"><label for="sensor_angle">Sensor angle (degrees)</label><input id="sensor_angle" data-setting="sensor_angle" type="number" min="10" max="90" step="1"></div>
          <div class="field"><label for="control_loop_hz">Control frequency (Hz)</label><input id="control_loop_hz" data-setting="control_loop_hz" type="number" min="10" max="60" step="1"></div>
          <div class="field"><label for="inference_interval_ms">Inference interval (ms)</label><input id="inference_interval_ms" data-setting="inference_interval_ms" type="number" min="10" max="500" step="1"></div>
        </div>
        <div class="section-label">Speed controller</div>
        <div class="fields">
          <div class="field"><label for="min_speed_kmh">Minimum speed (km/h)</label><input id="min_speed_kmh" data-setting="min_speed_kmh" type="number" min="1" max="60" step=".5"></div>
          <div class="field"><label for="max_speed_kmh">Maximum speed (km/h)</label><input id="max_speed_kmh" data-setting="max_speed_kmh" type="number" min="1.5" max="100" step=".5"></div>
          <div class="field full"><label for="acceleration_mps2">Acceleration limit (m/s²)</label><input id="acceleration_mps2" data-setting="acceleration_mps2" type="number" min=".2" max="8" step=".1"></div>
        </div>
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
            <div class="field"><label for="lane_width_ratio">Fallback lane width / image</label><input id="lane_width_ratio" data-setting="lane_width_ratio" type="number" min=".15" max=".9" step=".01"></div>
            <div class="field"><label for="white_l_min">White lightness min</label><input id="white_l_min" data-setting="white_l_min" type="number" min="0" max="255" step="1"></div>
            <div class="field"><label for="white_s_max">White saturation max</label><input id="white_s_max" data-setting="white_s_max" type="number" min="0" max="255" step="1"></div>
            <div class="field"><label for="yellow_h_min">Yellow hue min</label><input id="yellow_h_min" data-setting="yellow_h_min" type="number" min="0" max="179" step="1"></div>
            <div class="field"><label for="yellow_h_max">Yellow hue max</label><input id="yellow_h_max" data-setting="yellow_h_max" type="number" min="0" max="179" step="1"></div>
            <div class="field"><label for="yellow_s_min">Yellow saturation min</label><input id="yellow_s_min" data-setting="yellow_s_min" type="number" min="0" max="255" step="1"></div>
            <div class="field"><label for="yellow_l_min">Yellow lightness min</label><input id="yellow_l_min" data-setting="yellow_l_min" type="number" min="0" max="255" step="1"></div>
            <div class="field"><label for="roi_top_ratio">ROI top / frame height</label><input id="roi_top_ratio" data-setting="roi_top_ratio" type="number" min=".1" max=".9" step=".01"></div>
            <div class="field"><label for="roi_bottom_ratio">ROI bottom / frame height</label><input id="roi_bottom_ratio" data-setting="roi_bottom_ratio" type="number" min=".2" max="1" step=".01"></div>
            <div class="field"><label for="roi_top_half_width_ratio">ROI top half-width</label><input id="roi_top_half_width_ratio" data-setting="roi_top_half_width_ratio" type="number" min=".02" max=".48" step=".01"></div>
            <div class="field"><label for="roi_bottom_half_width_ratio">ROI bottom half-width</label><input id="roi_bottom_half_width_ratio" data-setting="roi_bottom_half_width_ratio" type="number" min=".15" max=".5" step=".01"></div>
            <div class="field"><label for="min_lane_angle_deg">Minimum lane angle (°)</label><input id="min_lane_angle_deg" data-setting="min_lane_angle_deg" type="number" min="0" max="60" step="1"></div>
            <div class="field"><label for="max_lane_angle_deg">Maximum lane angle (°)</label><input id="max_lane_angle_deg" data-setting="max_lane_angle_deg" type="number" min="10" max="89" step="1"></div>
            <div class="field"><label for="center_smoothing">Center smoothing (α)</label><input id="center_smoothing" data-setting="center_smoothing" type="number" min=".01" max="1" step=".01"></div>
            <div class="field"><label for="yellow_enabled">Detect yellow markings</label><select id="yellow_enabled" data-setting="yellow_enabled"><option value="true">Enabled</option><option value="false">Disabled</option></select></div>
          </div>
          <div class="hint">Tune these values against the AVIS track camera. The ROI is trapezoidal; Hough segments are grouped by left/right image side and fitted before lane-width estimation.</div>
        </div>
        <div class="actions"><button class="btn save" type="submit">Apply and save settings</button><span class="message" id="settingsMessage"></span></div>
      </form>
    </aside>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
let currentMode = "auto";
let manualThrottle = 0;
let manualSteering = 0;
const settingInputs = Array.from(document.querySelectorAll("[data-setting]"));
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
function fillSettings(settings) {
  settingInputs.forEach(input => {
    const key = input.dataset.setting;
    if (!(key in settings)) return;
    if (input.tagName === "SELECT" && (key === "yellow_enabled")) {
      input.value = settings[key] ? "true" : "false";
    } else {
      input.value = settings[key];
    }
  });
  $("classicalSettings").classList.toggle("hidden", settings.vision_backend !== "classical");
}
function collectSettings() {
  const result = {};
  settingInputs.forEach(input => {
    const key = input.dataset.setting;
    if (input.tagName === "SELECT" && key === "vision_backend") result[key] = input.value;
    else if (input.tagName === "SELECT" && key === "yellow_enabled") result[key] = input.value === "true";
    else result[key] = input.valueAsNumber;
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
  $("classicalSettings").classList.toggle("hidden", $("vision_backend").value !== "classical");
});
$("settingsForm").addEventListener("submit", async event => {
  event.preventDefault();
  try {
    const data = await postJson("/api/settings", collectSettings());
    fillSettings(data.settings);
    showMessage("Settings applied and saved.", false);
  } catch (error) {
    showMessage(error.message, true);
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
async function refreshStatus() {
  try {
    const response = await fetch("/api/status", {cache:"no-store"});
    if (!response.ok) throw new Error("Runtime unavailable");
    const data = await response.json();
    const t = data.telemetry || {};
    $("connectionDot").classList.toggle("online", !!t.connected);
    $("connectionText").textContent = t.connected ? "AVIS connected" : "Waiting for AVIS";
    $("backendTag").textContent = "VISION: " + String(data.settings.vision_backend).toUpperCase();
    $("speedValue").innerHTML = Number(t.speed_kmh || 0).toFixed(1) + " <small>km/h</small>";
    $("targetValue").innerHTML = Number(t.target_speed_kmh || 0).toFixed(1) + " <small>km/h</small>";
    $("steerValue").textContent = Number(t.steering || 0).toFixed(2);
    $("errorValue").innerHTML = (t.lane_valid ? Number(t.lane_error_px || 0).toFixed(1) : "--") + " <small>px</small>";
    $("runtimeStatus").textContent = t.status || "Running";
    $("runtimeDetails").textContent =
      "Drive mode: " + data.mode.toUpperCase() +
      " · Throttle: " + Math.round(Number(t.throttle || 0) * 100) + "%" +
      " · Lane: " + (t.lane_valid ? "tracking" : "not detected") +
      " · Sensors (L / C / R): " + (Array.isArray(t.sensors_cm) ? t.sensors_cm.join(" / ") + " cm" : "--") +
      (t.vision_error ? " · Vision: " + t.vision_error : "");
    if (data.mode !== currentMode) setModeButtons(data.mode);
    if (!$("settingsForm").dataset.dirty) fillSettings(data.settings);
  } catch (error) {
    $("connectionDot").classList.remove("online");
    $("connectionText").textContent = "Dashboard disconnected";
  }
}
settingInputs.forEach(input => input.addEventListener("input", () => {
  $("settingsForm").dataset.dirty = "true";
  $("classicalSettings").classList.toggle("hidden", $("vision_backend").value !== "classical");
}));
$("settingsForm").addEventListener("change", () => {$("settingsForm").dataset.dirty = "true";});
$("settingsForm").addEventListener("submit", () => {delete $("settingsForm").dataset.dirty;});
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

    @app.post("/api/settings")
    def settings():
        payload = request.get_json(silent=True)
        try:
            updated = state.update_settings(payload)
        except (TypeError, ValueError, OSError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"settings": updated})

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

    @app.get("/stream/<view_name>")
    def stream(view_name: str):
        if view_name not in {"camera", "combined", "line-mask", "drivable-mask"}:
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
