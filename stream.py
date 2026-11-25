import logging
import math
import threading
from pathlib import Path

import cv2
import json
import os
from flask import Flask, Response, request, render_template_string, jsonify
from werkzeug.serving import make_server

import config_city as conf
from runtime_settings import PROFILE_PATH, SETTING_SPECS, RuntimeSettings
logging.getLogger('werkzeug').disabled = True

logger = logging.getLogger(__name__)
stop_event = threading.Event()
app = Flask(__name__)
_server_thread = None
_server = None
_server_lock = threading.Lock()
_settings_callback = None
_restart_callback = None
_telemetry_callback = None
_lidar_provider = None
_vehicle_callback = None
_vehicle_options = []
_vehicle_selected = None
_runtime_status = "Waiting for CARLA runtime"
_profile_lock = threading.RLock()

# --- ROI variable names ---
VARIABLES = [
    "RL_TOP_ROI", "RL_BOTTOM_ROI", "RL_LEFT_ROI", "RL_RIGHT_ROI",
    "LL_TOP_ROI", "LL_BOTTOM_ROI", "LL_LEFT_ROI", "LL_RIGHT_ROI",
    "CW_TOP_ROI", "CW_BOTTOM_ROI", "CW_LEFT_ROI", "CW_RIGHT_ROI",
    "DW_TOP_ROI", "DW_BOTTOM_ROI", "DW_LEFT_ROI", "DW_RIGHT_ROI"
]

# --- Advanced variables (to confirm) ---
ADVANCED_VARS = {
    "LANE_THRESHOLD": int(getattr(conf, "LANE_THRESHOLD", 180)),
    "CROSSWALK_THRESHOLD": int(getattr(conf, "CROSSWALK_THRESHOLD", 180)),
    "CROSSWALK_SLEEP": float(getattr(conf, "CROSSWALK_SLEEP", 3.0)),
    "CROSSWALK_THRESH_SPEND": float(getattr(conf, "CROSSWALK_THRESH_SPEND", 8.0)),
    "RUN_LVL": getattr(conf, "RUN_LVL", "MOVE"),
}

# UI settings filename and functions
def ui_filename():
    return "roi_config_ui.json"

def load_ui_settings():
    defaults = {
        "colors": {
            "RL": "#ff7b7b",
            "LL": "#7bffb8",
            "CW": "#ffd27b",
            "DW": "#a78bfa",
        },
        "visible": {
            "RL": True,
            "LL": True,
            "CW": True,
            "DW": True,
        },
    }
    fname = ui_filename()
    if os.path.exists(fname):
        try:
            with open(fname, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for section in ("colors", "visible"):
                    stored = data.get(section, {})
                    if isinstance(stored, dict):
                        defaults[section].update({
                            key: value
                            for key, value in stored.items()
                            if key in defaults[section]
                        })
        except Exception:
            logger.exception("Failed load UI settings")
    return defaults

def save_ui_settings(settings):
    fname = ui_filename()
    try:
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception:
        logger.exception("Failed writing UI settings")

UI_SETTINGS = load_ui_settings()

# save ROI + advanced to JSON (primary config JSON)
def config_filename():
    return "roi_config.json"

def save_conf_to_json():
    filename = config_filename()

    data = {var: float(getattr(conf, var, 0.0)) for var in VARIABLES}
    data.update({
        "LANE_THRESHOLD": int(getattr(conf, "LANE_THRESHOLD", ADVANCED_VARS["LANE_THRESHOLD"])),
        "CROSSWALK_THRESHOLD": int(getattr(conf, "CROSSWALK_THRESHOLD", ADVANCED_VARS["CROSSWALK_THRESHOLD"])),
        "CROSSWALK_SLEEP": float(getattr(conf, "CROSSWALK_SLEEP", ADVANCED_VARS["CROSSWALK_SLEEP"])),
        "CROSSWALK_THRESH_SPEND": float(getattr(conf, "CROSSWALK_THRESH_SPEND", ADVANCED_VARS["CROSSWALK_THRESH_SPEND"])),
        "RUN_LVL": getattr(conf, "RUN_LVL", ADVANCED_VARS["RUN_LVL"]),
    })

    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        logger.exception("Failed writing config JSON")

def _current_values():
    return {var: float(getattr(conf, var, 0.0)) for var in VARIABLES}

def _current_advanced():
    return {
        "LANE_THRESHOLD": int(getattr(conf, "LANE_THRESHOLD", ADVANCED_VARS["LANE_THRESHOLD"])),
        "CROSSWALK_THRESHOLD": int(getattr(conf, "CROSSWALK_THRESHOLD", ADVANCED_VARS["CROSSWALK_THRESHOLD"])),
        "CROSSWALK_SLEEP": float(getattr(conf, "CROSSWALK_SLEEP", ADVANCED_VARS["CROSSWALK_SLEEP"])),
        "CROSSWALK_THRESH_SPEND": float(getattr(conf, "CROSSWALK_THRESH_SPEND", ADVANCED_VARS["CROSSWALK_THRESH_SPEND"])),
        "RUN_LVL": getattr(conf, "RUN_LVL", ADVANCED_VARS["RUN_LVL"]),
    }

def set_settings_callback(callback):
    global _settings_callback
    _settings_callback = callback

def set_restart_callback(callback):
    global _restart_callback
    _restart_callback = callback

def set_telemetry_callback(callback):
    global _telemetry_callback
    _telemetry_callback = callback

def set_lidar_provider(callback):
    global _lidar_provider
    _lidar_provider = callback

def set_runtime_status(status):
    global _runtime_status
    _runtime_status = str(status)

def set_vehicle_options(options, selected=None):
    global _vehicle_options, _vehicle_selected
    _vehicle_options = [
        {"id": str(item[0]), "label": str(item[1])}
        for item in (options or [])
        if isinstance(item, (tuple, list)) and len(item) >= 2
    ]
    _vehicle_selected = str(selected) if selected else None

def set_vehicle_callback(callback):
    global _vehicle_callback
    _vehicle_callback = callback

def _setting_values():
    values = {}
    for key, *_ in SETTING_SPECS:
        value = getattr(conf, key, None)
        if isinstance(value, tuple):
            value = list(value)
        values[key] = value
    return values

def _coerce_setting(spec, value):
    key, _label, kind, limits, _realtime = spec
    if kind == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
            return value.strip().lower() in {"1", "true", "yes", "on"}
        raise ValueError(f"{key} must be a boolean")
    if kind == "int":
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{key} must be finite")
        result = int(number)
    elif kind == "float":
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{key} must be finite")
    elif kind == "sequence":
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, dict):
            raise ValueError(f"{key} must be a JSON object")
        result = {}
        for direction in ("LEFT", "RIGHT", "STRAIGHT"):
            actions = value.get(direction, [])
            if not isinstance(actions, list) or len(actions) > 30:
                raise ValueError(f"{key}.{direction} must be a list of at most 30 actions")
            normalized = []
            for action in actions:
                if not isinstance(action, dict):
                    raise ValueError(f"{key}.{direction} contains an invalid action")
                distance = float(action.get("distance_m", 0))
                steering = float(action.get("steering_value", 0))
                if not math.isfinite(distance) or distance < 0:
                    raise ValueError(f"{key}.{direction} distance must be finite and non-negative")
                if not math.isfinite(steering) or not -1.0 <= steering <= 1.0:
                    raise ValueError(f"{key}.{direction} steering must be between -1 and 1")
                normalized.append({"distance_m": distance, "steering_value": steering})
            result[direction] = normalized
        unknown = set(value) - {"LEFT", "RIGHT", "STRAIGHT"}
        if unknown:
            raise ValueError(f"{key} contains unsupported maneuvers: {', '.join(sorted(unknown))}")
        return result
    elif kind == "choice":
        result = str(value)
        if limits and result not in limits:
            raise ValueError(f"unsupported choice for {key}: {result}")
        return result
    else:
        result = str(value).strip()
        if not result:
            raise ValueError(f"{key} cannot be empty")
    if limits:
        result = max(limits[0], min(limits[1], result))
    return result

def _local_request_only():
    return request.remote_addr in {"127.0.0.1", "::1", "localhost"}

def _notify_settings(updated):
    if not updated:
        return
    with _profile_lock:
        RuntimeSettings.persist_values(updated)
    if any(key in VARIABLES for key in updated):
        save_conf_to_json()
    if _settings_callback is not None:
        try:
            _settings_callback(updated)
        except Exception:
            logger.exception("Runtime settings callback failed")

# --- HTML template (full UI) ---
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CARLA Perception Studio | ROI Calibration</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{color-scheme:dark;--bg:#07111e;--surface:#101d2e;--surface-2:#0b1726;--surface-3:#16283d;--border:#29415b;--accent:#55c9e7;--accent-2:#259fbd;--text:#e8f1fb;--muted:#91a6bc;--good:#62d6a3;--warn:#f3bd69;--danger:#ef8080;--radius:15px}
*{box-sizing:border-box}html{min-width:320px;background:var(--bg);scroll-behavior:smooth}
body{margin:0;min-height:100vh;padding:20px;color:var(--text);font-family:Inter,"Segoe UI",Roboto,Arial,sans-serif;background:radial-gradient(ellipse at 10% 0%,rgba(25,126,157,.16),transparent 38rem),linear-gradient(145deg,#07111e 0%,#0b1728 58%,#081321 100%)}
.app-shell{width:min(100%,1740px);margin:0 auto}.header{display:flex;justify-content:space-between;align-items:center;gap:18px;margin:0 auto 16px;padding:4px 2px 16px;border-bottom:1px solid rgba(145,166,188,.16)}
.header-main{min-width:0}.eyebrow{font-size:10px;font-weight:800;letter-spacing:1.6px;text-transform:uppercase;color:var(--accent);margin-bottom:5px}
.h1{font-size:22px;font-weight:780;letter-spacing:-.5px;color:#f4f8fe}.subtitle{font-size:12px;color:var(--muted);margin-top:5px;line-height:1.45}
.header-status{display:flex;gap:9px;align-items:center;flex-wrap:wrap;justify-content:flex-end}.status-pill{padding:7px 11px;border-radius:999px;border:1px solid var(--border);background:#0d1d2e;color:#b8ccdf;font-size:11px;font-weight:750}
.status-dot{display:inline-block;width:7px;height:7px;margin-right:7px;border-radius:50%;background:var(--warn);box-shadow:0 0 10px currentColor}
.container{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(360px,.85fr);gap:16px;align-items:start}
.card{min-width:0;padding:16px;background:linear-gradient(155deg,rgba(18,34,53,.97),rgba(11,23,38,.98));border:1px solid var(--border);border-radius:var(--radius);box-shadow:0 14px 38px rgba(0,0,0,.19)}
.camera-card{position:sticky;top:14px}.card-heading{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:13px}
.section-title{color:#f0f6fd;font-size:14px;font-weight:780;letter-spacing:.1px}.section-subtitle{color:var(--muted);font-size:11px;line-height:1.45;margin-top:4px}
.small,.small-muted{color:var(--muted);font-size:11px;line-height:1.5}.controls{display:flex;gap:8px;align-items:center;margin-bottom:11px;flex-wrap:wrap}
select,input[type=number],input[type=text],input[type=color],textarea{min-height:36px;color:var(--text);background:#081523;border:1px solid #304a65;border-radius:8px;padding:7px 10px;outline:none;font:inherit;font-size:12px}
select:focus,input:focus,textarea:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(85,201,231,.12)}input[type=color]{width:43px;min-height:34px;padding:3px;cursor:pointer}
input[type=checkbox]{accent-color:var(--accent-2);width:16px;height:16px;vertical-align:middle}input[type=number]{-moz-appearance:textfield}
input[type=number]::-webkit-inner-spin-button,input[type=number]::-webkit-outer-spin-button{-webkit-appearance:none;margin:0}
button.btn,button.ghost{min-height:35px;padding:8px 12px;border-radius:8px;cursor:pointer;font:inherit;font-size:12px;font-weight:750;transition:background .16s ease,border-color .16s ease,transform .16s ease}
button.btn{color:#041521;background:linear-gradient(135deg,#72deef,#39b8d4);border:1px solid #63cde4;box-shadow:0 5px 14px rgba(27,166,198,.14)}button.btn:hover{background:#8ce6f4;transform:translateY(-1px)}
button.ghost{color:#c5d6e7;background:#13263b;border:1px solid #304760}button.ghost:hover{background:#1a344d;border-color:#527792}
button:disabled{opacity:.45;cursor:not-allowed;transform:none!important}button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
canvas{display:block;width:100%;height:auto;background:#030a12;border:1px solid #2d425a;border-radius:10px}#video_canvas{aspect-ratio:3/2;touch-action:none}
.camera-footer{display:flex;justify-content:space-between;gap:10px;align-items:center;flex-wrap:wrap;margin-top:10px}
.right-panel{display:flex;flex-direction:column;gap:12px;min-width:0}
.panel-tabs{position:sticky;top:8px;z-index:5;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:5px;padding:5px;background:#091625;border:1px solid var(--border);border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,.18)}
.panel-tab{min-height:36px;border:1px solid transparent;border-radius:8px;background:transparent;color:#9eb2c8;font:inherit;font-size:12px;font-weight:750;cursor:pointer}
.panel-tab:hover{color:#e4f4ff;background:#13263a}.panel-tab.active{color:#eafaff;background:#153449;border-color:#2e7990;box-shadow:inset 0 0 0 1px rgba(85,201,231,.08)}
.panel-content{display:none;flex-direction:column;gap:12px;min-width:0}.panel-content.active{display:flex}
.telemetry-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.metric{padding:10px 11px;background:#0a1828;border:1px solid #233a53;border-radius:10px;min-width:0}
.metric-label{display:block;color:#8fa7bf;font-size:10px;letter-spacing:.45px;text-transform:uppercase;margin-bottom:5px}.metric-value{display:block;color:#edf6ff;font-size:16px;font-weight:780;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.metric-value.compact{font-size:12px;line-height:1.5}.metric-note{font-size:10px;color:#7f95aa;margin-top:3px}
#lidar_canvas{width:min(100%,360px);aspect-ratio:1/1;margin:0 auto;background:#071321}.lidar-legend{display:flex;gap:10px;align-items:center;justify-content:center;flex-wrap:wrap;color:var(--muted);font-size:10px;margin-top:8px}
.legend-dot{width:8px;height:8px;display:inline-block;border-radius:50%;margin-right:4px}pre{overflow:auto;max-height:190px;padding:11px;background:#071321;border:1px solid #243950;border-radius:9px;color:#9db2c8;font-size:11px;line-height:1.45}
.settings-heading{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;margin-bottom:12px}.settings-toolbar{display:grid;grid-template-columns:minmax(0,1fr);gap:9px;padding:11px;background:#0b1929;border:1px solid #263d56;border-radius:11px;margin-bottom:10px}
#settings_search{width:100%;min-width:0}.setting-categories{display:flex;gap:6px;flex-wrap:wrap}
.category-chip{border:1px solid #2c455f;background:#102239;color:#a9bfd4;border-radius:999px;padding:6px 9px;font-size:10px;font-weight:750;cursor:pointer}.category-chip:hover,.category-chip.active{background:#16394c;border-color:#347d95;color:#e0f9ff}
.settings-scroll{max-height:calc(100vh - 355px);min-height:280px;overflow:auto;padding:1px 5px 2px 0;scrollbar-color:#36516c #091624;scrollbar-width:thin;overscroll-behavior:contain}
.setting-group{background:rgba(7,19,32,.5);border:1px solid #233b54;border-radius:11px;margin-bottom:8px;overflow:hidden}.setting-group summary{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:11px 12px;cursor:pointer;list-style:none;font-size:12px;font-weight:780;color:#dceafa;background:rgba(19,39,60,.75)}
.setting-group summary::-webkit-details-marker{display:none}.setting-group summary:after{content:"＋";color:var(--accent);font-size:15px}.setting-group[open] summary:after{content:"−"}
.setting-count{margin-left:auto;color:#7992a9;font-size:10px;font-weight:600}.setting-group-body{padding:5px 11px 10px}
.form-row{display:grid;grid-template-columns:minmax(0,1fr) minmax(130px,1.05fr);gap:12px;align-items:center;padding:7px 0;border-bottom:1px solid rgba(110,139,166,.11)}.form-row:last-child{border-bottom:0}
.form-row .label{color:#a9bdd1;font-size:11px;line-height:1.35;min-width:0}.form-row input:not([type=checkbox]),.form-row select,.form-row textarea{width:100%;min-width:0}
.form-row input[type=checkbox]{justify-self:start}.form-row textarea{min-height:140px;resize:vertical;font-family:Consolas,monospace;font-size:11px;line-height:1.5}
.sequence-row{grid-template-columns:minmax(0,1fr)}.sequence-row .label{font-weight:700;color:#c8d9e9}.form-help{grid-column:1/-1;color:#748ca4;font-size:10px;line-height:1.4;margin-top:-6px}
.profile-bar{display:grid;grid-template-columns:minmax(0,1fr) minmax(90px,.8fr) auto auto;gap:7px;align-items:center;margin-bottom:11px}.profile-bar select,.profile-bar input{width:100%;min-width:0}
.vehicle-bar{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:7px;align-items:center;margin-bottom:12px}.vehicle-bar select{width:100%;min-width:0}
.settings-actions{display:flex;gap:7px;align-items:center;flex-wrap:wrap;padding-top:11px;border-top:1px solid rgba(145,166,188,.15);margin-top:10px}
.message{font-size:11px;color:var(--muted);min-height:16px;line-height:1.45}.message.success{color:var(--good)}.message.warning{color:var(--warn)}.message.error{color:var(--danger)}
.input-wrap{display:grid;grid-template-columns:minmax(110px,1fr) 76px auto auto;align-items:center;gap:7px;margin-bottom:8px}.input-wrap .label{color:var(--muted);font-size:11px;line-height:1.3}
.counter{width:76px;min-width:0;text-align:center;font-variant-numeric:tabular-nums}.toggle-row{display:flex;gap:10px;align-items:center;margin-bottom:9px;flex-wrap:wrap}.toggle-row label{color:#c9d7e6;font-size:12px}
.toast{position:fixed;right:20px;bottom:20px;z-index:20;padding:11px 15px;color:var(--text);background:#14263b;border:1px solid #3c5975;border-radius:10px;box-shadow:0 12px 34px rgba(0,0,0,.35);font-size:12px}
@media(max-width:1120px){body{padding:14px}.container{grid-template-columns:minmax(0,1fr)}.camera-card{position:relative;top:auto}.right-panel{min-width:0}.settings-scroll{max-height:none}}
@media(max-width:650px){body{padding:8px}.header{align-items:flex-start;flex-direction:column}.header-status{justify-content:flex-start}.h1{font-size:19px}.card{padding:12px;border-radius:12px}.container{gap:10px}.profile-bar{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}.profile-bar button{width:100%}.form-row{grid-template-columns:minmax(0,1fr);gap:5px}.form-row input:not([type=checkbox]),.form-row select,.form-row textarea{width:100%}.input-wrap{grid-template-columns:minmax(0,1fr) 76px}.input-wrap .btn-inc{grid-row:2}.telemetry-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.panel-tabs{top:4px}}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{transition-duration:.01ms!important;scroll-behavior:auto!important}}
</style>
</head>
<body>
<div class="app-shell">
  <header class="header"><div class="header-main"><div class="eyebrow">Autonomous Driving · Runtime Console</div><div class="h1">CARLA Runtime Studio</div><div class="subtitle">Live perception, vehicle telemetry, LiDAR monitoring and complete runtime configuration in one browser workspace.</div></div>
    <div class="header-status"><span class="status-pill"><span class="status-dot" id="runtime_dot"></span><span id="runtime_status">Connecting to runtime…</span></span><span class="status-pill" id="frame_dimensions">FRAME —</span></div>
  </header>
  <div class="container">
    <section class="card camera-card">
      <div class="card-heading"><div><div class="section-title">Live perception camera</div><div class="section-subtitle">The active CARLA camera stream with editable ROI overlays</div></div><span class="status-pill" id="camera_status">WAITING FOR FRAME</span></div>
      <div class="controls"><label class="small" for="variable_select">Selected ROI variable</label><select id="variable_select"><option value="NONE">— Select —</option>{% for var in variables %}<option value="{{ var }}">{{ var }}</option>{% endfor %}</select>
        <button id="clear_selection" class="ghost">Clear selection</button><button id="center_reset" class="ghost">Center selected</button><div style="flex:1"></div><label class="small">Show ROIs <input id="show_rects" type="checkbox" checked></label></div>
      <canvas id="video_canvas" width="900" height="600"></canvas>
      <div class="camera-footer"><div class="small">Select a region, drag inside to move it, or drag a corner to resize</div><div style="display:flex;gap:7px;flex-wrap:wrap"><button id="refresh_vals" class="ghost">Refresh ROI</button><button id="download_json" class="ghost">Export ROI JSON</button></div></div>
    </section>
    <aside class="right-panel">
      <nav class="panel-tabs" aria-label="Runtime workspace"><button class="panel-tab active" data-panel="live">Live monitor</button><button class="panel-tab" data-panel="settings">Settings</button><button class="panel-tab" data-panel="roi">ROI tools</button></nav>
      <section class="panel-content active" id="panel-live">
        <div class="card"><div class="card-heading"><div><div class="section-title">Runtime telemetry</div><div class="section-subtitle">Live control, perception and navigation status</div></div><span class="small" id="telemetry_updated">—</span></div>
          <div class="telemetry-grid" id="telemetry_grid"><div class="metric"><span class="metric-label">Control FPS</span><span class="metric-value">—</span></div><div class="metric"><span class="metric-label">Driving mode</span><span class="metric-value">—</span></div><div class="metric"><span class="metric-label">Vision FPS</span><span class="metric-value">—</span></div><div class="metric"><span class="metric-label">Trajectory FPS</span><span class="metric-value">—</span></div></div></div>
        <div class="card"><div class="card-heading"><div><div class="section-title">LiDAR radar</div><div class="section-subtitle">Top-down point cloud, height-coloured · live scan</div></div><span class="status-pill" id="lidar_status">DISABLED</span></div>
          <canvas id="lidar_canvas" width="360" height="360"></canvas><div class="lidar-legend"><span><i class="legend-dot" style="background:#2db7d0"></i>Ground</span><span><i class="legend-dot" style="background:#5bd77b"></i>Low</span><span><i class="legend-dot" style="background:#f1c64e"></i>Medium</span><span><i class="legend-dot" style="background:#ec6754"></i>High</span></div></div>
        <div class="card"><div class="card-heading"><div><div class="section-title">Live configuration snapshot</div><div class="section-subtitle">Current ROI values for quick verification</div></div></div><pre id="values_json" style="white-space:pre-wrap;margin:0">{{ values|tojson(indent=2) }}</pre></div>
      </section>
      <section class="panel-content" id="panel-settings">
        <div class="card"><div class="settings-heading"><div><div class="section-title">Runtime configuration</div><div class="section-subtitle">All controls previously exposed in the desktop panel, grouped by function</div></div><span class="status-pill" id="settings_count">— SETTINGS</span></div>
          <div class="profile-bar"><select id="profile_select" aria-label="Saved profile"></select><input id="profile_name" type="text" value="default" placeholder="Profile name" aria-label="Profile name"><button id="load_profile" class="ghost">Load</button><button id="save_profile" class="btn">Save</button></div>
          <div class="vehicle-bar"><select id="vehicle_select" aria-label="Vehicle blueprint"></select><button id="apply_vehicle" class="ghost">Use vehicle</button></div>
          <div class="settings-toolbar"><input id="settings_search" type="text" placeholder="Search by setting name or config key…" autocomplete="off"><div class="setting-categories" id="setting_categories"></div></div>
          <div class="settings-scroll" id="settings_scroll"><div id="runtime_settings"></div></div>
          <div class="settings-actions"><button id="apply_runtime" class="btn">Apply settings</button><button id="refresh_runtime" class="ghost">Refresh</button><button id="restart_runtime" class="ghost" style="display:none">Restart CARLA to apply</button></div>
          <div class="message" id="runtime_msg"></div><div class="message" id="profile_msg"></div><div class="message" id="vehicle_msg"></div></div>
      </section>
      <section class="panel-content" id="panel-roi">
        <div class="card"><div class="section-title">Manual ROI calibration</div><div class="section-subtitle" style="margin-bottom:12px">Edit normalized coordinates directly or use the camera overlay. Changes are saved to the active runtime profile.</div>
          <div class="inputs" id="inputs_container">{% for var in variables %}<div class="input-wrap"><div class="label">{{ var }}</div><input name="{{ var }}" type="number" step="0.01" min="0" max="1" class="counter" value="{{ '%.2f' % values[var] }}"><button type="button" class="ghost btn-inc" data-var="{{ var }}" data-delta="0.1">+0.1</button><button type="button" class="ghost btn-inc" data-var="{{ var }}" data-delta="-0.1">−0.1</button></div>{% endfor %}</div>
          <div class="settings-actions"><button id="apply_all" class="btn">Apply ROI values</button><button id="refresh_roi" class="ghost">Reload values</button></div></div>
        <div class="card"><div class="section-title">ROI visibility & colours</div>
          <div class="toggle-row"><label><input type="checkbox" id="vis_rl"> Show right lane</label><input type="color" id="color_rl" value="{{ ui.colors.RL }}"></div>
          <div class="toggle-row"><label><input type="checkbox" id="vis_ll"> Show left lane</label><input type="color" id="color_ll" value="{{ ui.colors.LL }}"></div>
          <div class="toggle-row"><label><input type="checkbox" id="vis_cw"> Show crosswalk / junction ROI</label><input type="color" id="color_cw" value="{{ ui.colors.CW }}"></div>
          <div class="small-muted" style="margin:-5px 0 9px 25px">Area used by the horizontal crosswalk-line junction detector</div>
          <div class="toggle-row"><label><input type="checkbox" id="vis_dw"> Show drivable-width ROI</label><input type="color" id="color_dw" value="{{ ui.colors.DW }}"></div>
          <div class="small-muted" style="margin:-5px 0 9px 25px">Sampling region used for road-width measurements</div></div>
      </section>
    </aside>
  </div>
</div>
<div id="toast" class="toast" style="display:none"></div>
<script>
const canvas = document.getElementById('video_canvas');
const ctx = canvas.getContext('2d');
const select = document.getElementById('variable_select');
const valuesPre = document.getElementById('values_json');
const inputsContainer = document.getElementById('inputs_container');
const showRectsCheckbox = document.getElementById('show_rects');

let values = {{ values|tojson }};
let ui = {{ ui|tojson }};
let advanced = {{ advanced|tojson }};
let selectedVar = 'NONE';
let markerHighlight = null;
let fetchFrameTimer = null;
let dragState = null;
let debounceTimers = {};
let runtimeValues = {{ runtime_values|tojson }};
const runtimeSpecs = {{ runtime_specs|tojson }};

const SETTING_GROUPS = [
 {name:'Driving & PID',match:key=>['AUTO_MODE_DEFAULT','FIXED_THROTTLE','KP','KI','KD','STEER_LIMIT','MAX_STEER_STEP','TARGET_SPEED_KMH','CONTROL_LOOP_HZ'].includes(key)},
 {name:'Trajectory model',match:key=>key.startsWith('TRAJECTORY_')},
 {name:'Junctions & crosswalks',match:key=>key.startsWith('JUNCTION_')||key.startsWith('INTERSECTION_')||key.startsWith('CROSSWALK_')||key==='RUN_LVL'},
 {name:'Lane changes',match:key=>key.startsWith('LANE_CHANGE_')||key.startsWith('POST_JUNCTION_')},
 {name:'Vision & perception',match:key=>key.startsWith('VISION_')||key.startsWith('LANE_')||key.startsWith('DRIVABLE_PROB_')||key==='MODEL_PATH'},
 {name:'Drivable recovery & LiDAR',match:key=>key.startsWith('DRIVABLE_RECOVERY_')||key.startsWith('LIDAR_')},
 {name:'ROI calibration',match:key=>key.endsWith('_ROI')},
 {name:'Debug display',match:key=>key.startsWith('DEBUG_')},
 {name:'Camera & sensors',match:key=>key.startsWith('CAMERA_')||key.startsWith('ENABLE_')},
 {name:'Other',match:key=>true}
];
function categoryFor(key){return SETTING_GROUPS.find(group=>group.match(key))?.name||'Other';}
function attachWheelGuard(root){root.querySelectorAll('input[type="number"]').forEach(input=>input.addEventListener('wheel',event=>{if(document.activeElement===input){event.preventDefault();input.blur();}},{passive:false}));}
function renderRuntimeSettings(){
 const root=document.getElementById('runtime_settings'),categoryRoot=document.getElementById('setting_categories');root.innerHTML='';categoryRoot.innerHTML='';
 const grouped=new Map();for(const spec of runtimeSpecs){const name=categoryFor(spec[0]);if(!grouped.has(name))grouped.set(name,[]);grouped.get(name).push(spec);}
 let index=0;
 for(const [category,specs] of grouped.entries()){
  const details=document.createElement('details');details.className='setting-group';details.dataset.category=category;details.open=index===0;
  const summary=document.createElement('summary');summary.textContent=category;const count=document.createElement('span');count.className='setting-count';count.textContent=specs.length+' SETTINGS';summary.appendChild(count);details.appendChild(summary);
  const body=document.createElement('div');body.className='setting-group-body';
  for(const spec of specs){
   const [key,label,kind,limits,realtime]=spec,row=document.createElement('div');row.className='form-row setting-row';row.dataset.key=key;
   const title=document.createElement('label');title.className='label';title.textContent=label+(realtime?'':' · restart required');title.title=key;row.appendChild(title);
   let input;
   if(kind==='bool'){input=document.createElement('input');input.type='checkbox';input.checked=!!runtimeValues[key];}
   else if(kind==='choice'){input=document.createElement('select');for(const value of (limits||[])){const option=document.createElement('option');option.value=value;option.textContent=value==='model'?'Junction model (ResNet18)':value==='crosswalk_roi'?'Crosswalk ROI — horizontal line':value;input.appendChild(option);}input.value=runtimeValues[key]??'';}
   else if(kind==='sequence'){row.classList.add('sequence-row');input=document.createElement('textarea');input.rows=8;input.spellcheck=false;input.value=JSON.stringify(runtimeValues[key]||{},null,2);}
   else{input=document.createElement('input');input.type=kind==='str'?'text':'number';if(kind!=='str'&&limits){input.min=limits[0];input.max=limits[1];input.step=limits[2];}input.value=runtimeValues[key]??'';}
   input.dataset.key=key;input.setAttribute('aria-label',label);row.appendChild(input);
   if(kind==='sequence'){const help=document.createElement('div');help.className='form-help';help.textContent='JSON format: LEFT, RIGHT and STRAIGHT arrays containing distance_m and steering_value for each action. Steering must be between −1 and 1.';row.appendChild(help);}
   body.appendChild(row);
  }
  details.appendChild(body);root.appendChild(details);
  const chip=document.createElement('button');chip.type='button';chip.className='category-chip'+(index===0?' active':'');chip.textContent=category;chip.dataset.category=category;
  chip.addEventListener('click',()=>{root.querySelectorAll('.setting-group').forEach(group=>group.open=group.dataset.category===category);categoryRoot.querySelectorAll('.category-chip').forEach(button=>button.classList.toggle('active',button.dataset.category===category));details.scrollIntoView({behavior:'smooth',block:'nearest'});});
  categoryRoot.appendChild(chip);index++;
 }
 document.getElementById('settings_count').textContent=runtimeSpecs.length+' SETTINGS';attachWheelGuard(root);filterRuntimeSettings(document.getElementById('settings_search').value);
}
function filterRuntimeSettings(query){
 const term=(query||'').trim().toLowerCase();
 document.querySelectorAll('#runtime_settings .setting-group').forEach(group=>{let visible=0;group.querySelectorAll('.setting-row').forEach(row=>{const key=row.dataset.key||'',label=row.querySelector('.label')?.textContent||'',show=!term||(key+' '+label).toLowerCase().includes(term);row.style.display=show?'':'none';if(show)visible++;});group.style.display=visible?'':'none';if(term&&visible)group.open=true;});
 document.querySelectorAll('.category-chip').forEach(chip=>{const group=[...document.querySelectorAll('#runtime_settings .setting-group')].find(item=>item.dataset.category===chip.dataset.category);chip.style.display=group&&group.style.display!=='none'?'':'none';});
}
renderRuntimeSettings();
document.getElementById('settings_search').addEventListener('input',event=>filterRuntimeSettings(event.target.value));
document.getElementById('refresh_runtime').addEventListener('click',()=>fetch('/get_settings').then(r=>r.json()).then(data=>{runtimeValues=data.values||{};renderRuntimeSettings();startFrameLoop();showToast('Runtime settings refreshed');}).catch(()=>showToast('Refresh failed')));
document.getElementById('apply_runtime').addEventListener('click',()=>{
 const payload={};
 try{document.querySelectorAll('#runtime_settings [data-key]').forEach(input=>{const key=input.dataset.key,spec=runtimeSpecs.find(item=>item[0]===key);if(spec[2]==='bool')payload[key]=input.checked;else if(spec[2]==='sequence')payload[key]=JSON.parse(input.value);else if(spec[2]==='int'){if(!input.value.trim())throw new Error(key+' is required');payload[key]=parseInt(input.value,10);}else if(spec[2]==='float'){if(!input.value.trim())throw new Error(key+' is required');payload[key]=parseFloat(input.value);}else payload[key]=input.value;});}
 catch(error){const msg=document.getElementById('runtime_msg');msg.textContent=error.message.includes('JSON')?'Invalid junction movement JSON.':error.message;msg.className='message error';return;}
 fetch('/set_settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)})
 .then(response=>response.json().then(data=>({ok:response.ok,data}))).then(({ok,data})=>{
  if(!ok||!data.success)throw new Error(data.error||'Could not apply settings');runtimeValues=data.values||runtimeValues;renderRuntimeSettings();startFrameLoop();
  document.getElementById('restart_runtime').style.display=data.reset_required?'':'none';
  const msg=document.getElementById('runtime_msg');msg.textContent=data.changed?.length?data.changed.length+' setting(s) updated.'+(data.reset_required?' Restart CARLA to activate hardware/model changes.':' Live changes are active.'):'No settings changed.';
  msg.className='message '+(data.reset_required?'warning':'success');showToast(data.reset_required?'Settings saved · restart required':'Settings applied live');loadProfiles();
 }).catch(error=>{const msg=document.getElementById('runtime_msg');msg.textContent=error.message;msg.className='message error';});
});
document.getElementById('restart_runtime').addEventListener('click',()=>fetch('/restart_runtime',{method:'POST'}).then(r=>r.json().then(data=>({ok:r.ok,data}))).then(({ok,data})=>{if(!ok||!data.success)throw new Error(data.error||'Restart could not be scheduled');const msg=document.getElementById('runtime_msg');msg.textContent='Restart scheduled. The current vehicle will respawn at its selected spawn point.';msg.className='message warning';showToast('CARLA runtime restart scheduled');}).catch(error=>{const msg=document.getElementById('runtime_msg');msg.textContent=error.message;msg.className='message error';}));
function loadProfiles(){fetch('/get_profiles').then(r=>r.json()).then(data=>{const select=document.getElementById('profile_select'),names=data.profiles||[];select.innerHTML='';names.forEach(name=>{const option=document.createElement('option');option.value=name;option.textContent=name;select.appendChild(option);});if(data.active&&names.includes(data.active))select.value=data.active;document.getElementById('profile_name').value=data.active||select.value||'default';}).catch(()=>{});}
document.getElementById('save_profile').addEventListener('click',()=>{const name=document.getElementById('profile_name').value.trim();fetch('/save_profile',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})}).then(r=>r.json().then(data=>({ok:r.ok,data}))).then(({ok,data})=>{if(!ok||!data.success)throw new Error(data.error||'Could not save profile');const msg=document.getElementById('profile_msg');msg.textContent='Saved profile “'+name+'”';msg.className='message success';loadProfiles();showToast('Profile saved');}).catch(error=>{const msg=document.getElementById('profile_msg');msg.textContent=error.message;msg.className='message error';});});
document.getElementById('load_profile').addEventListener('click',()=>{const name=document.getElementById('profile_select').value;fetch('/load_profile',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})}).then(r=>r.json().then(data=>({ok:r.ok,data}))).then(({ok,data})=>{if(!ok||!data.success)throw new Error(data.error||'Could not load profile');runtimeValues=data.values||{};renderRuntimeSettings();startFrameLoop();document.getElementById('profile_name').value=name;document.getElementById('restart_runtime').style.display=data.reset_required?'':'none';const msg=document.getElementById('profile_msg');msg.textContent=data.reset_required?'Profile loaded. Restart CARLA to apply hardware/model changes.':'Profile loaded and applied.';msg.className='message '+(data.reset_required?'warning':'success');showToast('Profile loaded');}).catch(error=>{const msg=document.getElementById('profile_msg');msg.textContent=error.message;msg.className='message error';});});
function loadVehicles(){fetch('/get_vehicles').then(r=>r.json()).then(data=>{const select=document.getElementById('vehicle_select');select.innerHTML='';(data.vehicles||[]).forEach(vehicle=>{const option=document.createElement('option');option.value=vehicle.id;option.textContent=vehicle.label;select.appendChild(option);});select.value=data.selected||'';}).catch(()=>{});}
document.getElementById('apply_vehicle').addEventListener('click',()=>{const blueprint=document.getElementById('vehicle_select').value;fetch('/set_vehicle',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({blueprint})}).then(r=>r.json().then(data=>({ok:r.ok,data}))).then(({ok,data})=>{if(!ok||!data.success)throw new Error(data.error||'Could not change vehicle');const msg=document.getElementById('vehicle_msg');msg.textContent='Vehicle change queued. CARLA will respawn it at the current spawn point.';msg.className='message warning';showToast('Vehicle change scheduled');}).catch(error=>{const msg=document.getElementById('vehicle_msg');msg.textContent=error.message;msg.className='message error';});});
document.querySelectorAll('.panel-tab').forEach(button=>button.addEventListener('click',()=>{const panel=button.dataset.panel;document.querySelectorAll('.panel-tab').forEach(item=>item.classList.toggle('active',item===button));document.querySelectorAll('.panel-content').forEach(item=>item.classList.toggle('active',item.id==='panel-'+panel));}));
function metricCard(label,value,compact=false){const card=document.createElement('div');card.className='metric';const title=document.createElement('span');title.className='metric-label';title.textContent=label;const content=document.createElement('span');content.className='metric-value'+(compact?' compact':'');content.textContent=value??'—';card.append(title,content);return card;}
function refreshTelemetry(){fetch('/get_telemetry').then(r=>r.json()).then(data=>{const root=document.getElementById('telemetry_grid');root.innerHTML='';const items=[['Control FPS',data.control_fps==null?'—':Number(data.control_fps).toFixed(1)],['Driving mode',data.driving_mode||'—'],['Vision FPS',data.vision_fps==null?'—':Number(data.vision_fps).toFixed(1)],['Trajectory FPS',data.trajectory_fps==null?'—':Number(data.trajectory_fps).toFixed(1)],['Vehicle speed',data.speed_kmh==null?'—':Number(data.speed_kmh).toFixed(1)+' km/h'],['Junction phase',data.junction_phase||'—'],['Lane change',data.lane_change_phase||'—'],['Lane error',data.lane_error_px==null?'—':Number(data.lane_error_px).toFixed(1)+' px'],['Road width',data.road_width_index_px==null?'—':Number(data.road_width_index_px).toFixed(0)+' px'],['GPS',Array.isArray(data.gps)?data.gps.map(v=>Number(v).toFixed(1)).join(', '):'—'],['Vehicle',data.vehicle_blueprint||'—'],['Map',data.map_name||'—']];items.forEach(item=>root.appendChild(metricCard(item[0],item[1],['GPS','Vehicle','Map'].includes(item[0]))));document.getElementById('telemetry_updated').textContent='Updated '+new Date().toLocaleTimeString();document.getElementById('runtime_status').textContent=data.runtime_status||'Runtime connected';document.getElementById('runtime_dot').style.background=data.running===false?'#ef8080':'#62d6a3';document.getElementById('camera_status').textContent=data.frame_available?'STREAMING':'WAITING FOR FRAME';document.getElementById('frame_dimensions').textContent=data.frame_dimensions||'FRAME —';}).catch(()=>{document.getElementById('runtime_status').textContent='Telemetry unavailable';document.getElementById('runtime_dot').style.background='#ef8080';});}
function lidarHeightColor(z){if(z<0)return '#3b83d0';if(z<1)return '#2db7d0';if(z<2)return '#5bd77b';if(z<4)return '#f1c64e';return '#ec6754';}
function drawLidar(data){const canvas=document.getElementById('lidar_canvas'),ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height,cx=w/2,cy=h/2,radius=Math.min(w,h)*.46;ctx.clearRect(0,0,w,h);ctx.fillStyle='#071321';ctx.fillRect(0,0,w,h);const maxRange=Math.max(5,Number(data.display_range_m)||80),step=maxRange<=30?5:maxRange<=60?10:maxRange<=125?25:50;ctx.strokeStyle='#29445e';ctx.lineWidth=1;for(let d=step;d<maxRange;d+=step){ctx.beginPath();ctx.arc(cx,cy,radius*d/maxRange,0,Math.PI*2);ctx.stroke();}ctx.beginPath();ctx.arc(cx,cy,radius,0,Math.PI*2);ctx.stroke();ctx.beginPath();ctx.moveTo(cx-radius,cy);ctx.lineTo(cx+radius,cy);ctx.moveTo(cx,cy-radius);ctx.lineTo(cx,cy+radius);ctx.stroke();const points=data.points||[];if(data.enabled){const stride=Math.max(1,Math.ceil(points.length/7000));for(let i=0;i<points.length;i+=stride){const point=points[i],x=Number(point[0]),y=Number(point[1]),z=Number(point[2]);if(!Number.isFinite(x)||!Number.isFinite(y)||Math.hypot(x,y)>maxRange)continue;const px=cx+y*radius/maxRange,py=cy-x*radius/maxRange;ctx.fillStyle=lidarHeightColor(z);ctx.fillRect(px,py,2,2);}}ctx.fillStyle='#f3f7fc';ctx.strokeStyle='#0b1928';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(cx,cy-13);ctx.lineTo(cx+8,cy+10);ctx.lineTo(cx,cy+5);ctx.lineTo(cx-8,cy+10);ctx.closePath();ctx.fill();ctx.stroke();document.getElementById('lidar_status').textContent=!data.enabled?'DISABLED':data.complete?'COMPLETE SCAN':'SCANNING';}
function refreshLidar(){fetch('/get_lidar_scan').then(r=>r.json()).then(drawLidar).catch(()=>{document.getElementById('lidar_status').textContent='UNAVAILABLE';});}
loadProfiles();loadVehicles();refreshTelemetry();refreshLidar();setInterval(refreshTelemetry,700);setInterval(refreshLidar,1100);
document.addEventListener('wheel',event=>{const target=event.target;if(target instanceof HTMLInputElement&&target.type==='number'&&document.activeElement===target){event.preventDefault();target.blur();}},{capture:true,passive:false});
renderRuntimeSettings();

document.getElementById('refresh_runtime').addEventListener('click', ()=>{
  fetch('/get_settings').then(r=>r.json()).then(data=>{
    runtimeValues = data.values || {};
    renderRuntimeSettings();
    showToast('Runtime settings refreshed');
  }).catch(()=>showToast('Refresh failed'));
});

document.getElementById('apply_runtime').addEventListener('click', ()=>{
  const payload = {};
  try {
    document.querySelectorAll('#runtime_settings [data-key]').forEach(input=>{
      const key = input.dataset.key;
      const spec = runtimeSpecs.find(item=>item[0] === key);
      if(spec[2] === 'bool') payload[key] = input.checked;
      else if(spec[2] === 'sequence') payload[key] = JSON.parse(input.value);
      else if(spec[2] === 'int') payload[key] = parseInt(input.value);
      else if(spec[2] === 'float') payload[key] = parseFloat(input.value);
      else payload[key] = input.value;
    });
  } catch(error) {
    document.getElementById('runtime_msg').textContent = 'Invalid JSON sequence.';
    return;
  }
  fetch('/set_settings',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(payload)
  }).then(r=>r.json()).then(data=>{
    if(!data.success) throw new Error(data.error || 'failed');
    runtimeValues = data.values || runtimeValues;
    renderRuntimeSettings();
    document.getElementById('runtime_msg').textContent = data.reset_required
      ? 'Applied. Restart required for reset-marked settings.'
      : 'Applied live.';
    showToast(data.reset_required ? 'Applied; restart required' : 'Applied live');
  }).catch(error=>{
    document.getElementById('runtime_msg').textContent = error.message;
  });
});

function clamp01(v){ return Math.max(0, Math.min(1, v)); }
function round01(v){ return Math.round(v*100)/100; }

function showToast(text, timeout=2000){
    const t = document.getElementById('toast');
    t.textContent = text;
    t.style.display = 'block';
    setTimeout(()=> t.style.display='none', timeout);
}

function varColorByKey(key){
  if(key === 'RL') return ui.colors.RL;
  if(key === 'LL') return ui.colors.LL;
  if(key === 'CW') return ui.colors.CW;
  if(key === 'DW') return ui.colors.DW;
  return '#9ad0ff';
}

function rectFromVars(topVar, bottomVar, leftVar, rightVar, width, height){
    const t = parseFloat(values[topVar]);
    const b = parseFloat(values[bottomVar]);
    const l = parseFloat(values[leftVar]);
    const r = parseFloat(values[rightVar]);
    if (isNaN(t) || isNaN(b) || isNaN(l) || isNaN(r)) return null;
    const x = l * width;
    const y = t * height;
    const w = Math.max(2, (r - l) * width);
    const h = Math.max(2, (b - t) * height);
    return {x, y, w, h, t, b, l, r};
}

function computeRects(){
    const w = canvas.width, h = canvas.height;
    return {
        RL: rectFromVars('RL_TOP_ROI','RL_BOTTOM_ROI','RL_LEFT_ROI','RL_RIGHT_ROI', w, h),
        LL: rectFromVars('LL_TOP_ROI','LL_BOTTOM_ROI','LL_LEFT_ROI','LL_RIGHT_ROI', w, h),
        CW: rectFromVars('CW_TOP_ROI','CW_BOTTOM_ROI','CW_LEFT_ROI','CW_RIGHT_ROI', w, h),
        DW: rectFromVars('DW_TOP_ROI','DW_BOTTOM_ROI','DW_LEFT_ROI','DW_RIGHT_ROI', w, h)
    };
}

function startFrameLoop(){
    if(fetchFrameTimer) clearInterval(fetchFrameTimer);
    const requestedHz = Number(runtimeValues.DEBUG_PANEL_HZ) || 20;
    const intervalMs = Math.max(16, Math.min(1000, 1000 / requestedHz));
    fetchFrameTimer = setInterval(fetchFrameOnce, intervalMs);
}
startFrameLoop();

function fetchFrameOnce(){
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = '/video_feed_frame?t=' + Date.now();
    img.onload = () => {
        const rect = canvas.getBoundingClientRect();
        const dispW = Math.floor(rect.width);
        const dispH = Math.floor(rect.height);
        if (canvas.width !== dispW || canvas.height !== dispH) {
            canvas.width = dispW; canvas.height = dispH;
        }
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        drawOverlays();
    };
    img.onerror = () => {
        const rect = canvas.getBoundingClientRect();
        const dispW = Math.floor(rect.width);
        const dispH = Math.floor(rect.height);
        if (canvas.width !== dispW || canvas.height !== dispH) {
            canvas.width = dispW; canvas.height = dispH;
        }
        ctx.clearRect(0,0,canvas.width,canvas.height);
        drawOverlays();
    };
}

function drawHandle(cx, cy, size, fill, stroke){
    ctx.beginPath();
    ctx.fillStyle = fill;
    ctx.strokeStyle = stroke || '#000';
    ctx.lineWidth = 1;
    ctx.rect(cx - size/2, cy - size/2, size, size);
    ctx.fill();
    ctx.stroke();
}

function drawOverlays(){
    const showRects = showRectsCheckbox.checked;
    const rects = computeRects();
    ctx.save();
    if(showRects){
        for(const key of ['RL','LL','CW','DW']){
            if(!ui.visible[key]) continue;
            const r = rects[key];
            if(!r) continue;
            const color = varColorByKey(key);
            ctx.globalAlpha = 0.28;
            ctx.fillStyle = color;
            ctx.fillRect(r.x, r.y, r.w, r.h);
            ctx.globalAlpha = 1;
            ctx.strokeStyle = (markerHighlight === key)? '#ffffff' : color;
            ctx.lineWidth = (markerHighlight === key)? 3 : 2;
            ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1);

            const roiLabels = {
                RL: 'RIGHT LANE ROI',
                LL: 'LEFT LANE ROI',
                CW: 'CROSSWALK ROI',
                DW: 'DRIVABLE WIDTH ROI'
            };
            ctx.font = '12px Arial';
            const label = roiLabels[key] || key;
            const labelWidth = ctx.measureText(label).width + 14;
            const labelX = key === 'DW'
                ? r.x + Math.max(6, r.w - labelWidth - 6)
                : r.x + 6;
            ctx.fillStyle = color;
            ctx.fillRect(labelX, r.y + 6, labelWidth, 18);
            ctx.fillStyle = '#012';
            ctx.fillText(label, labelX + 7, r.y + 19);

            ctx.fillStyle = '#fff';
            ctx.font = '11px Arial';
            const nums = `${(r.t).toFixed(2)} , ${(r.l).toFixed(2)} , ${((r.w/canvas.width)).toFixed(2)} , ${((r.h/canvas.height)).toFixed(2)}`;
            ctx.fillText(nums, r.x + 6, r.y + r.h - 6);

            const size = 10;
            drawHandle(r.x, r.y, size, '#fff', color);
            drawHandle(r.x + r.w, r.y, size, '#fff', color);
            drawHandle(r.x, r.y + r.h, size, '#fff', color);
            drawHandle(r.x + r.w, r.y + r.h, size, '#fff', color);
        }
    }
    ctx.restore();
}

function hitTest(px, py){
    const rects = computeRects();
    for(const key of ['DW','RL','LL','CW']){
        if(!ui.visible[key]) continue;
        const r = rects[key];
        if(!r) continue;
        const size = 12;
        const corners = {
            nw: {x: r.x, y: r.y},
            ne: {x: r.x + r.w, y: r.y},
            sw: {x: r.x, y: r.y + r.h},
            se: {x: r.x + r.w, y: r.y + r.h}
        };
        for(const c of Object.keys(corners)){
            const cx = corners[c].x, cy = corners[c].y;
            if(px >= cx - size && px <= cx + size && py >= cy - size && py <= cy + size){
                return {type:'handle', group:key, corner:c, rect:r};
            }
        }
        if(px >= r.x && px <= r.x + r.w && py >= r.y && py <= r.y + r.h){
            return {type:'inside', group:key, rect:r};
        }
    }
    return {type:'none'};
}

canvas.addEventListener('mousedown', (e)=>{
    const rectB = canvas.getBoundingClientRect();
    const px = (e.clientX - rectB.left);
    const py = (e.clientY - rectB.top);
    const hit = hitTest(px, py);
    if(hit.type === 'handle'){
        dragState = {
            type: 'resize',
            group: hit.group,
            corner: hit.corner,
            origRect: hit.rect,
            start: {x: px, y: py}
        };
        markerHighlight = hit.group;
        selectedVar = mapGroupToDefaultVar(hit.group);
        document.getElementById('variable_select').value = selectedVar;
    } else if(hit.type === 'inside'){
        dragState = {
            type: 'move',
            group: hit.group,
            origRect: hit.rect,
            start: {x: px, y: py}
        };
        markerHighlight = hit.group;
        selectedVar = mapGroupToDefaultVar(hit.group);
        document.getElementById('variable_select').value = selectedVar;
    } else {
        dragState = null;
    }
});

canvas.addEventListener('mousemove', (e)=>{
    const rectB = canvas.getBoundingClientRect();
    const px = (e.clientX - rectB.left);
    const py = (e.clientY - rectB.top);
    const hover = hitTest(px, py);
    if(dragState) {
        canvas.style.cursor = (dragState.type === 'move') ? 'grabbing' : 'nwse-resize';
    } else if(hover.type === 'handle'){
        canvas.style.cursor = 'nwse-resize';
    } else if(hover.type === 'inside'){
        canvas.style.cursor = 'move';
    } else {
        canvas.style.cursor = 'default';
    }

    if(!dragState) return;
    const dx = px - dragState.start.x;
    const dy = py - dragState.start.y;
    const w = canvas.width, h = canvas.height;
    const group = dragState.group;
    let topVar, bottomVar, leftVar, rightVar;
    if(group === 'RL'){
        topVar='RL_TOP_ROI'; bottomVar='RL_BOTTOM_ROI'; leftVar='RL_LEFT_ROI'; rightVar='RL_RIGHT_ROI';
    } else if(group === 'LL'){
        topVar='LL_TOP_ROI'; bottomVar='LL_BOTTOM_ROI'; leftVar='LL_LEFT_ROI'; rightVar='LL_RIGHT_ROI';
    } else if(group === 'CW'){
        topVar='CW_TOP_ROI'; bottomVar='CW_BOTTOM_ROI'; leftVar='CW_LEFT_ROI'; rightVar='CW_RIGHT_ROI';
    } else {
        topVar='DW_TOP_ROI'; bottomVar='DW_BOTTOM_ROI'; leftVar='DW_LEFT_ROI'; rightVar='DW_RIGHT_ROI';
    }
    const orig = dragState.origRect;
    let nx_t = orig.t, nx_b = orig.b, nx_l = orig.l, nx_r = orig.r;
    if(dragState.type === 'move'){
        const dnx = dx / w; const dny = dy / h;
        nx_t = clamp01(orig.t + dny);
        nx_b = clamp01(orig.b + dny);
        nx_l = clamp01(orig.l + dnx);
        nx_r = clamp01(orig.r + dnx);
        if(nx_r - nx_l < 0.02){
            const mid = (nx_l + nx_r)/2; nx_l = Math.max(0, mid - 0.01); nx_r = Math.min(1, mid + 0.01);
        }
        if(nx_b - nx_t < 0.02){
            const midv = (nx_t + nx_b)/2; nx_t = Math.max(0, midv - 0.01); nx_b = Math.min(1, midv + 0.01);
        }
    } else if(dragState.type === 'resize'){
        const dnx = dx / w; const dny = dy / h;
        if(dragState.corner === 'nw'){ nx_t = clamp01(orig.t + dny); nx_l = clamp01(orig.l + dnx); }
        if(dragState.corner === 'ne'){ nx_t = clamp01(orig.t + dny); nx_r = clamp01(orig.r + dnx); }
        if(dragState.corner === 'sw'){ nx_b = clamp01(orig.b + dny); nx_l = clamp01(orig.l + dnx); }
        if(dragState.corner === 'se'){ nx_b = clamp01(orig.b + dny); nx_r = clamp01(orig.r + dnx); }
        if(nx_b <= nx_t) nx_b = clamp01(nx_t + 0.01);
        if(nx_r <= nx_l) nx_r = clamp01(nx_l + 0.01);
    }

    values[topVar] = round01(nx_t);
    values[bottomVar] = round01(nx_b);
    values[leftVar] = round01(nx_l);
    values[rightVar] = round01(nx_r);
    updateInputsFromValues(); updateValuesPre(); drawOverlays();

    if(debounceTimers[group]) clearTimeout(debounceTimers[group]);
    debounceTimers[group] = setTimeout(()=>{
        const payload = {};
        payload[topVar]=values[topVar]; payload[bottomVar]=values[bottomVar];
        payload[leftVar]=values[leftVar]; payload[rightVar]=values[rightVar];
        sendUpdate(payload);
        debounceTimers[group] = null;
    }, 220);
});

window.addEventListener('mouseup', ()=>{
    if(!dragState) return;
    const group = dragState.group;
    let topVar, bottomVar, leftVar, rightVar;
    if(group === 'RL'){
        topVar='RL_TOP_ROI'; bottomVar='RL_BOTTOM_ROI'; leftVar='RL_LEFT_ROI'; rightVar='RL_RIGHT_ROI';
    } else if(group === 'LL'){
        topVar='LL_TOP_ROI'; bottomVar='LL_BOTTOM_ROI'; leftVar='LL_LEFT_ROI'; rightVar='LL_RIGHT_ROI';
    } else if(group === 'CW'){
        topVar='CW_TOP_ROI'; bottomVar='CW_BOTTOM_ROI'; leftVar='CW_LEFT_ROI'; rightVar='CW_RIGHT_ROI';
    } else {
        topVar='DW_TOP_ROI'; bottomVar='DW_BOTTOM_ROI'; leftVar='DW_LEFT_ROI'; rightVar='DW_RIGHT_ROI';
    }
    const payload = {};
    payload[topVar]=values[topVar]; payload[bottomVar]=values[bottomVar];
    payload[leftVar]=values[leftVar]; payload[rightVar]=values[rightVar];
    sendUpdate(payload);
    dragState = null;
    canvas.style.cursor = 'default';
});

canvas.addEventListener('click', (e)=>{
    if(dragState) return;
    const rect = canvas.getBoundingClientRect();
    const clickX = (e.clientX - rect.left) / rect.width;
    const clickY = (e.clientY - rect.top) / rect.height;
    const px = clickX * canvas.width;
    const py = clickY * canvas.height;
    const hit = hitTest(px, py);
    if(hit.type === 'inside'){
        selectedVar = mapGroupToDefaultVar(hit.group);
        document.getElementById('variable_select').value = selectedVar;
        markerHighlight = hit.group; drawOverlays();
        return;
    }

    const sel = document.getElementById('variable_select').value;
    if(sel === 'NONE'){ markerHighlight=null; drawOverlays(); return; }

    const group = groupFromVar(sel);
    if(!ui.visible[group]){
        const ok = window.confirm(`${group} is hidden. Show it and select?`);
        if(ok){
            ui.visible[group] = true;
            saveUiSettingsLocal();
            drawOverlays();
        } else {
            document.getElementById('variable_select').value = 'NONE';
            selectedVar = 'NONE'; markerHighlight = null; drawOverlays();
            return;
        }
    }

    let toSend = clickY;
    if(sel.includes('LEFT') || sel.includes('RIGHT')) toSend = clickX;
    toSend = clamp01(round01(toSend));

    fetch('/set_variable',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({variable:sel, value: toSend})
    }).then(r=>r.json()).then(j=>{
        if(j && j.success){
            values[sel]=toSend;
            updateInputsFromValues();
            updateValuesPre();
            drawOverlays();
        }
    }).catch(console.warn);
});

function updateInputsFromValues(){
    for(const input of inputsContainer.querySelectorAll('input')){
        const name = input.name;
        if(values[name] !== undefined) input.value = (round01(values[name])).toFixed(2);
    }
}

document.querySelectorAll('.btn-inc').forEach(btn=>{
    btn.addEventListener('click', ()=>{
        const v = btn.dataset.var;
        const d = parseFloat(btn.dataset.delta);
        const input = inputsContainer.querySelector(`input[name="${v}"]`);
        let val = parseFloat(input.value) || 0;
        val = clamp01(round01(val + d));
        input.value = val.toFixed(2);
        debounceSendSingle(v, val);
    });
});

function debounceSendSingle(varName, val){
    if(debounceTimers[varName]) clearTimeout(debounceTimers[varName]);
    debounceTimers[varName] = setTimeout(()=>{
        sendUpdate({[varName]: val});
        debounceTimers[varName] = null;
    }, 250);
}

for(const input of inputsContainer.querySelectorAll('input')){
    input.addEventListener('input', (e)=>{
        const name = e.target.name;
        let v = parseFloat(e.target.value);
        if(isNaN(v)) return;
        v = clamp01(round01(v));
        values[name] = v;
        updateValuesPre();
        debounceSendSingle(name, v);
    });
}

document.getElementById('apply_all').addEventListener('click', ()=>{
    const data = {};
    for(const input of inputsContainer.querySelectorAll('input')){
        const name = input.name;
        let val = parseFloat(input.value);
        if(isNaN(val)) continue;
        val = clamp01(round01(val));
        data[name]=val;
    }
    sendUpdate(data);
});

function sendUpdate(data){
    fetch('/update_conf',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify(data)
    })
    .then(r=>r.json())
    .then(j=>{
        if(j && j.values){
            values = j.values;
            updateInputsFromValues();
            updateValuesPre();
            drawOverlays();
            showToast('Saved');
        }
    })
    .catch(console.warn);
}

document.getElementById('refresh_vals').addEventListener('click', ()=>{ refreshValues(); });
document.getElementById('refresh_roi').addEventListener('click', ()=>{ refreshValues(); showToast('ROI values refreshed'); });

function refreshValues(){
    fetch('/get_values')
    .then(r=>r.json())
    .then(j=>{
        if(j && j.values){
            values = j.values;
            updateInputsFromValues();
            updateValuesPre();
            drawOverlays();
        }
    })
    .catch(console.warn);
}

document.getElementById('clear_selection').addEventListener('click', ()=>{
    select.value='NONE';
    selectedVar='NONE';
    markerHighlight=null;
    drawOverlays();
});

document.getElementById('center_reset').addEventListener('click', ()=>{
    const v = select.value;
    if(v==='NONE') return;
    fetch('/set_variable',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({variable:v, value:0.5})
    }).then(r=>r.json()).then(j=>{
        if(j.success){
            values[v]=0.5;
            updateInputsFromValues();
            updateValuesPre();
            drawOverlays();
            showToast('Center applied');
        }
    });
});

document.getElementById('download_json').addEventListener('click', ()=>{
    fetch('/get_values').then(r=>r.json()).then(j=>{
        const blob = new Blob([JSON.stringify(j.values, null, 2)], {type:'application/json'});
        const url = URL.createObjectURL(blob);
        const a=document.createElement('a');
        a.href=url;
        a.download='roi_config.json';
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    });
});

function saveUiSettingsLocal(){
    fetch('/set_ui', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify(ui)
    })
    .then(r=>r.json())
    .then(j=>{
        if(!j.success) console.warn('ui save failed', j);
        else showToast('UI saved');
    });
}

document.getElementById('vis_rl').addEventListener('change', (e)=>{
    ui.visible.RL = e.target.checked;
    if(!ui.visible.RL && markerHighlight === 'RL'){
        markerHighlight = null;
        select.value='NONE';
        selectedVar='NONE';
    }
    saveUiSettingsLocal();
    drawOverlays();
});
document.getElementById('vis_ll').addEventListener('change', (e)=>{
    ui.visible.LL = e.target.checked;
    if(!ui.visible.LL && markerHighlight === 'LL'){
        markerHighlight = null;
        select.value='NONE';
        selectedVar='NONE';
    }
    saveUiSettingsLocal();
    drawOverlays();
});
document.getElementById('vis_cw').addEventListener('change', (e)=>{
    ui.visible.CW = e.target.checked;
    if(!ui.visible.CW && markerHighlight === 'CW'){
        markerHighlight = null;
        select.value='NONE';
        selectedVar='NONE';
    }
    saveUiSettingsLocal();
    drawOverlays();
});
document.getElementById('vis_dw').addEventListener('change', (e)=>{
    ui.visible.DW = e.target.checked;
    if(!ui.visible.DW && markerHighlight === 'DW'){
        markerHighlight = null;
        select.value='NONE';
        selectedVar='NONE';
    }
    saveUiSettingsLocal();
    drawOverlays();
});
document.getElementById('color_rl').addEventListener('input', (e)=>{ ui.colors.RL = e.target.value; saveUiSettingsLocal(); drawOverlays(); });
document.getElementById('color_ll').addEventListener('input', (e)=>{ ui.colors.LL = e.target.value; saveUiSettingsLocal(); drawOverlays(); });
document.getElementById('color_cw').addEventListener('input', (e)=>{ ui.colors.CW = e.target.value; saveUiSettingsLocal(); drawOverlays(); });
document.getElementById('color_dw').addEventListener('input', (e)=>{ ui.colors.DW = e.target.value; saveUiSettingsLocal(); drawOverlays(); });

function refreshInit(){
    updateInputsFromValues();
    updateValuesPre();
    drawOverlays();
    document.getElementById('vis_rl').checked = ui.visible.RL;
    document.getElementById('vis_ll').checked = ui.visible.LL;
    document.getElementById('vis_cw').checked = ui.visible.CW;
    document.getElementById('vis_dw').checked = ui.visible.DW;
    document.getElementById('color_rl').value = ui.colors.RL;
    document.getElementById('color_ll').value = ui.colors.LL;
    document.getElementById('color_cw').value = ui.colors.CW;
    document.getElementById('color_dw').value = ui.colors.DW;
}
refreshInit();
setInterval(refreshValues, 3000);

function updateValuesPre(){
    valuesPre.textContent = JSON.stringify(values, null, 2);
}
function groupFromVar(varname){
    if(varname.startsWith('RL_')) return 'RL';
    if(varname.startsWith('LL_')) return 'LL';
    if(varname.startsWith('CW_')) return 'CW';
    return 'DW';
}
function mapGroupToDefaultVar(group){
    if(group === 'RL') return 'RL_TOP_ROI';
    if(group === 'LL') return 'LL_TOP_ROI';
    if(group === 'CW') return 'CW_TOP_ROI';
    return 'DW_TOP_ROI';
}
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(
        HTML_TEMPLATE,
        variables=VARIABLES,
        values=_current_values(),
        ui=UI_SETTINGS,
        advanced=_current_advanced(),
        mode="ROI configuration",
        runtime_values=_setting_values(),
        runtime_specs=[
            [key, label, kind, limits, realtime]
            for key, label, kind, limits, realtime in SETTING_SPECS
        ],
    )

@app.route('/update_conf', methods=['POST'])
def update_conf():
    if not _local_request_only():
        return jsonify(success=False, error="configuration changes are local-only"), 403
    data = {}
    if request.is_json:
        data = request.get_json() or {}
    else:
        data = request.form.to_dict()

    updated = {}
    for var in VARIABLES:
        if var in data:
            try:
                val = float(data[var])
                val = max(0.0, min(1.0, val))
                setattr(conf, var, val)
                updated[var] = val
            except (ValueError, TypeError):
                pass

    _notify_settings(updated)

    return jsonify(success=True, values=_current_values())

@app.route('/set_variable', methods=['POST'])
def set_variable():
    if not _local_request_only():
        return jsonify(success=False, error="configuration changes are local-only"), 403
    data = request.get_json() or {}
    var = data.get('variable')
    try:
        val = float(data.get('value', 0))
    except (ValueError, TypeError):
        return jsonify(success=False, error="invalid value"), 400

    val = max(0.0, min(1.0, val))
    if var in VARIABLES:
        setattr(conf, var, val)
        _notify_settings({var: val})
        return jsonify(success=True, variable=var, value=val)
    return jsonify(success=False, error="unknown variable"), 400

@app.route('/get_values')
def get_values():
    return jsonify(values=_current_values())

@app.route('/get_advanced')
def get_advanced():
    return jsonify(advanced=_current_advanced())

@app.route('/get_settings')
def get_settings():
    return jsonify(values=_setting_values())

@app.route('/set_settings', methods=['POST'])
def set_settings():
    if not _local_request_only():
        return jsonify(success=False, error="settings changes are local-only"), 403
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(success=False, error="JSON object required"), 400
    specs = {spec[0]: spec for spec in SETTING_SPECS}
    unknown = sorted(set(data) - set(specs))
    if unknown:
        return jsonify(success=False, error=f"unknown setting(s): {', '.join(unknown)}"), 400
    try:
        updated = {key: _coerce_setting(specs[key], value) for key, value in data.items()}
    except (TypeError, ValueError, json.JSONDecodeError, OverflowError) as exc:
        return jsonify(success=False, error=f"invalid setting: {exc}"), 400
    current = _setting_values()
    changed = {key: value for key, value in updated.items() if current.get(key) != value}
    reset_required = any(not specs[key][4] and not (key == "TRAJECTORY_MODEL_ENABLED" and value is False) for key, value in changed.items())
    for key, value in changed.items():
        setattr(conf, key, value)
    _notify_settings(changed)
    return jsonify(success=True, values=_setting_values(), reset_required=reset_required, changed=list(changed))


def _read_profiles():
    with _profile_lock:
        try:
            data = json.loads(Path(PROFILE_PATH).read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

def _write_profiles(profiles):
    path = Path(PROFILE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with _profile_lock:
        temporary.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
        temporary.replace(path)

@app.route('/get_profiles')
def get_profiles():
    profiles = _read_profiles()
    names = sorted(name for name, value in profiles.items() if not name.startswith("_") and isinstance(value, dict))
    active = profiles.get("_active_profile")
    if active not in names:
        active = "default" if "default" in names else (names[0] if names else None)
    return jsonify(profiles=names, active=active)

@app.route('/save_profile', methods=['POST'])
def save_profile():
    if not _local_request_only():
        return jsonify(success=False, error="profile changes are local-only"), 403
    data = request.get_json(silent=True)
    name = str((data or {}).get("name", "")).strip() if isinstance(data, dict) else ""
    if not name or name.startswith("_") or len(name) > 64 or any(char in name for char in "/\\\\"):
        return jsonify(success=False, error="Enter a valid profile name (1–64 characters)."), 400
    profiles = _read_profiles()
    profiles[name] = _setting_values()
    profiles["_active_profile"] = name
    try:
        _write_profiles(profiles)
    except OSError as exc:
        logger.exception("Could not save runtime profile")
        return jsonify(success=False, error=str(exc)), 500
    return jsonify(success=True, profiles=sorted(key for key, value in profiles.items() if not key.startswith("_") and isinstance(value, dict)), active=name)

@app.route('/load_profile', methods=['POST'])
def load_profile():
    if not _local_request_only():
        return jsonify(success=False, error="profile changes are local-only"), 403
    data = request.get_json(silent=True)
    name = str((data or {}).get("name", "")).strip() if isinstance(data, dict) else ""
    profiles = _read_profiles()
    profile = profiles.get(name)
    if not name or name.startswith("_") or not isinstance(profile, dict):
        return jsonify(success=False, error="profile not found"), 404
    specs = {spec[0]: spec for spec in SETTING_SPECS}
    try:
        values = {key: _coerce_setting(specs[key], value) for key, value in profile.items() if key in specs}
    except (TypeError, ValueError, json.JSONDecodeError, OverflowError) as exc:
        return jsonify(success=False, error=f"profile contains invalid settings: {exc}"), 400
    current = _setting_values()
    changed = {key: value for key, value in values.items() if current.get(key) != value}
    reset_required = any(not specs[key][4] and not (key == "TRAJECTORY_MODEL_ENABLED" and value is False) for key, value in changed.items())
    for key, value in changed.items():
        setattr(conf, key, value)
    profiles["_active_profile"] = name
    try:
        _write_profiles(profiles)
    except OSError as exc:
        logger.exception("Could not set active runtime profile")
        return jsonify(success=False, error=str(exc)), 500
    _notify_settings(changed)
    return jsonify(success=True, values=_setting_values(), reset_required=reset_required, changed=list(changed))

@app.route('/restart_runtime', methods=['POST'])
def restart_runtime():
    if not _local_request_only():
        return jsonify(success=False, error="runtime controls are local-only"), 403
    if not callable(_restart_callback):
        return jsonify(success=False, error="CARLA runtime is not ready to restart"), 409
    try:
        scheduled = _restart_callback()
    except Exception as exc:
        logger.exception("Runtime restart request failed")
        return jsonify(success=False, error=str(exc)), 500
    if scheduled is False:
        return jsonify(success=False, error="CARLA runtime is not ready to restart"), 409
    return jsonify(success=True)

@app.route('/get_vehicles')
def get_vehicles():
    return jsonify(vehicles=_vehicle_options, selected=_vehicle_selected)

@app.route('/set_vehicle', methods=['POST'])
def set_vehicle():
    if not _local_request_only():
        return jsonify(success=False, error="vehicle changes are local-only"), 403
    data = request.get_json(silent=True)
    blueprint = str((data or {}).get("blueprint", "")).strip() if isinstance(data, dict) else ""
    if not blueprint or blueprint not in {item["id"] for item in _vehicle_options}:
        return jsonify(success=False, error="unknown vehicle blueprint"), 400
    if not callable(_vehicle_callback):
        return jsonify(success=False, error="vehicle controls are not ready"), 409
    try:
        accepted = _vehicle_callback(blueprint)
    except Exception as exc:
        logger.exception("Vehicle change request failed")
        return jsonify(success=False, error=str(exc)), 500
    if accepted is False:
        return jsonify(success=False, error="vehicle change was not accepted"), 409
    global _vehicle_selected
    _vehicle_selected = blueprint
    return jsonify(success=True, selected=blueprint)

@app.route('/get_telemetry')
def get_telemetry():
    data = {}
    if callable(_telemetry_callback):
        try:
            snapshot = _telemetry_callback()
            if isinstance(snapshot, dict):
                data.update(snapshot)
        except Exception:
            logger.exception("Could not collect runtime telemetry")
    data["runtime_status"] = _runtime_status
    data["running"] = bool(data.get("running", False))
    return jsonify(data)

@app.route('/get_lidar_scan')
def get_lidar_scan():
    payload = {"enabled": False, "points": [], "complete": False, "display_range_m": float(getattr(conf, "LIDAR_DISPLAY_RANGE_M", 80.0))}
    if callable(_lidar_provider):
        try:
            data = _lidar_provider()
            if isinstance(data, dict):
                payload.update(data)
        except Exception:
            logger.exception("Could not collect LiDAR scan")
    points = payload.get("points")
    try:
        if points is not None:
            stride = max(1, math.ceil(len(points) / 5000))
            payload["points"] = points[::stride].tolist() if hasattr(points, "tolist") else list(points[::stride])
        else:
            payload["points"] = []
    except (TypeError, ValueError):
        payload["points"] = []
    return jsonify(payload)

@app.route('/set_advanced', methods=['POST'])
def set_advanced():
    if not _local_request_only():
        return jsonify(success=False, error="configuration changes are local-only"), 403
    data = request.get_json() or {}
    updated = {}
    try:
        if "LANE_THRESHOLD" in data:
            val = int(data["LANE_THRESHOLD"])
            val = max(0, min(255, val))
            setattr(conf, "LANE_THRESHOLD", val)
            updated["LANE_THRESHOLD"] = val

        if "CROSSWALK_THRESHOLD" in data:
            val = int(data["CROSSWALK_THRESHOLD"])
            val = max(0, min(255, val))
            setattr(conf, "CROSSWALK_THRESHOLD", val)
            updated["CROSSWALK_THRESHOLD"] = val

        if "CROSSWALK_SLEEP" in data:
            val = float(data["CROSSWALK_SLEEP"])
            setattr(conf, "CROSSWALK_SLEEP", val)
            updated["CROSSWALK_SLEEP"] = val

        if "CROSSWALK_THRESH_SPEND" in data:
            val = float(data["CROSSWALK_THRESH_SPEND"])
            setattr(conf, "CROSSWALK_THRESH_SPEND", val)
            updated["CROSSWALK_THRESH_SPEND"] = val

        if "RUN_LVL" in data:
            val = data["RUN_LVL"] if data["RUN_LVL"] in ("MOVE", "STOP") else "MOVE"
            setattr(conf, "RUN_LVL", val)
            updated["RUN_LVL"] = val
    except Exception:
        logger.exception("Invalid advanced payload")
        return jsonify(success=False, error="invalid payload"), 400

    _notify_settings(updated)

    return jsonify(success=True, advanced=_current_advanced())

@app.route('/get_ui')
def get_ui():
    return jsonify(ui=UI_SETTINGS)

@app.route('/set_ui', methods=['POST'])
def set_ui():
    if not _local_request_only():
        return jsonify(success=False, error="configuration changes are local-only"), 403
    data = request.get_json() or {}
    colors = UI_SETTINGS.get("colors", {})
    visible = UI_SETTINGS.get("visible", {})

    if "colors" in data:
        for k, v in data["colors"].items():
            if k in colors and isinstance(v, str):
                colors[k] = v

    if "visible" in data:
        for k, v in data["visible"].items():
            if k in visible:
                visible[k] = bool(v)

    UI_SETTINGS["colors"] = colors
    UI_SETTINGS["visible"] = visible
    save_ui_settings(UI_SETTINGS)
    return jsonify(success=True, ui=UI_SETTINGS)

@app.route('/video_feed_frame')
def video_feed_frame():
    frame = getattr(conf, "debug_frame_buffer", None)
    if frame is None:
        return Response('', status=204)

    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        return Response('', status=204)

    return Response(buffer.tobytes(), mimetype='image/jpeg')

def start_stream(host=None, port=None):
    if host is None:
        host = getattr(conf, "STREAM_HOST", "0.0.0.0")
    if port is None:
        port = getattr(conf, "STREAM_PORT", 5000)

    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)

def start_stream_background(host=None, port=None):
    global _server, _server_thread
    with _server_lock:
        if _server is not None:
            return False

        if host is None:
            host = getattr(conf, "STREAM_HOST", "127.0.0.1")
        if port is None:
            port = getattr(conf, "STREAM_PORT", 5000)

        # Bind synchronously so port conflicts are reported to the GUI immediately.
        server = make_server(host, int(port), app, threaded=True)
        thread = threading.Thread(
            target=server.serve_forever,
            name="carla-debug-stream",
            daemon=True,
        )
        _server = server
        _server_thread = thread
        try:
            thread.start()
        except Exception:
            _server = None
            _server_thread = None
            server.server_close()
            raise
        return True

def stop_stream(host=None, port=None):
    global _server, _server_thread
    with _server_lock:
        server = _server
        thread = _server_thread
        if server is None:
            return False
        _server = None
        _server_thread = None

    try:
        server.shutdown()
        server.server_close()
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        return True
    except Exception:
        logger.exception("Failed to stop Flask editor")
        return False

if __name__ == '__main__':
    start_stream()
