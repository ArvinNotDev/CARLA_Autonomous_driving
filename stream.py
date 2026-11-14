import logging
import cv2
import json
import os
from flask import Flask, Response, request, render_template_string, jsonify
import threading
from werkzeug.serving import make_server

import config_city as conf
from runtime_settings import SETTING_SPECS, RuntimeSettings
logging.getLogger('werkzeug').disabled = True

logger = logging.getLogger(__name__)
stop_event = threading.Event()
app = Flask(__name__)
_server_thread = None
_server = None
_server_lock = threading.Lock()
_settings_callback = None

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

def _setting_values():
    values = {}
    for key, *_ in SETTING_SPECS:
        value = getattr(conf, key, None)
        if isinstance(value, tuple):
            value = list(value)
        values[key] = value
    return values

def _coerce_setting(spec, value):
    _key, _label, kind, limits, _realtime = spec
    if kind == "bool":
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if kind == "int":
        result = int(float(value))
    elif kind == "float":
        result = float(value)
    elif kind == "sequence":
        if isinstance(value, str):
            value = json.loads(value)
        return value
    elif kind == "choice":
        result = str(value)
        if limits and result not in limits:
            raise ValueError(f"unsupported choice for {_key}: {result}")
        return result
    else:
        return str(value)
    if limits:
        result = max(limits[0], min(limits[1], result))
    return result

def _local_request_only():
    return request.remote_addr in {"127.0.0.1", "::1", "localhost"}

def _notify_settings(updated):
    if not updated:
        return
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
:root {
  color-scheme: dark;
  --bg: #08111f;
  --surface: #101d2f;
  --surface-2: #0c1828;
  --surface-3: #15263d;
  --border: #263b55;
  --accent: #55c9e7;
  --accent-strong: #1496b8;
  --text: #e8f0f9;
  --muted: #91a5bb;
  --radius: 14px;
}
* { box-sizing: border-box; }
html { min-width: 320px; background: var(--bg); }
body {
  margin: 0;
  min-height: 100vh;
  padding: 22px;
  color: var(--text);
  font-family: Inter, "Segoe UI", Roboto, Arial, sans-serif;
  background:
    radial-gradient(ellipse at 12% 0%, rgba(22, 120, 155, .17), transparent 36rem),
    linear-gradient(145deg, #08111f 0%, #0b1728 55%, #091321 100%);
}
.container {
  width: min(100%, 1660px);
  margin: 0 auto;
  display: grid;
  grid-template-columns: minmax(0, 1.45fr) minmax(360px, .75fr);
  gap: 18px;
  align-items: start;
}
.card {
  min-width: 0;
  padding: 18px;
  background: linear-gradient(160deg, rgba(19, 35, 55, .96), rgba(12, 24, 40, .97));
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: 0 16px 42px rgba(0, 0, 0, .22);
}
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  padding-bottom: 14px;
  margin-bottom: 14px;
  border-bottom: 1px solid rgba(145, 165, 187, .15);
}
.h1 { color: #f2f8ff; font-size: 20px; font-weight: 750; letter-spacing: -.3px; }
.controls { display: flex; gap: 9px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }
select, input[type=number], input[type=color], input[type=text], textarea {
  min-height: 36px;
  color: var(--text);
  background: #091524;
  border: 1px solid #304760;
  border-radius: 8px;
  padding: 7px 10px;
  outline: none;
}
select:focus, input:focus, textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(85, 201, 231, .12); }
input[type=color] { width: 44px; min-height: 34px; padding: 3px; cursor: pointer; }
input[type=checkbox] { accent-color: var(--accent-strong); width: 16px; height: 16px; vertical-align: middle; }
button.btn, button.ghost {
  min-height: 36px;
  padding: 8px 13px;
  border-radius: 8px;
  cursor: pointer;
  font: inherit;
  font-size: 13px;
  font-weight: 700;
  transition: background .16s ease, border-color .16s ease, transform .16s ease;
}
button.btn {
  color: #041521;
  background: linear-gradient(135deg, #6cdbef, #36b7d5);
  border: 1px solid #63cde4;
  box-shadow: 0 5px 14px rgba(27, 166, 198, .15);
}
button.btn:hover { background: #8ce6f4; transform: translateY(-1px); }
button.ghost { color: #c3d4e5; background: #14253a; border: 1px solid #304760; }
button.ghost:hover { background: #1b344e; border-color: #4e7692; }
button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
canvas {
  display: block;
  width: 100%;
  height: auto;
  aspect-ratio: 3 / 2;
  background: #030a12;
  border: 1px solid #2d425a;
  border-radius: 10px;
  touch-action: none;
}
.label { flex: 0 0 148px; color: var(--muted); font-size: 12px; line-height: 1.35; }
.small, .small-muted { color: var(--muted); font-size: 12px; line-height: 1.5; }
.right-panel { display: flex; flex-direction: column; gap: 14px; min-width: 0; }
.inputs { max-height: 470px; overflow: auto; padding: 2px 7px 2px 0; scrollbar-color: #39536e #0b1726; scrollbar-width: thin; }
.input-wrap {
  display: grid;
  grid-template-columns: minmax(115px, 1fr) 76px auto auto;
  align-items: center;
  gap: 7px;
  margin-bottom: 8px;
}
.counter { width: 76px; min-width: 0; text-align: center; font-variant-numeric: tabular-nums; }
.section-title { margin-bottom: 12px; color: #eef7ff; font-size: 14px; font-weight: 750; letter-spacing: .1px; }
.toggle-row { display: flex; gap: 10px; align-items: center; margin-bottom: 10px; flex-wrap: wrap; }
.toggle-row label { color: #c9d7e6; font-size: 13px; }
.form-row { display: flex; gap: 10px; align-items: center; margin: 8px 0; }
.form-row .label { flex-basis: 170px; }
.form-row input:not([type=checkbox]), .form-row select, .form-row textarea { flex: 1; min-width: 0; }
.toast {
  position: fixed;
  right: 22px;
  bottom: 22px;
  z-index: 10;
  padding: 12px 16px;
  color: var(--text);
  background: #13243a;
  border: 1px solid #3c5975;
  border-radius: 10px;
  box-shadow: 0 12px 34px rgba(0, 0, 0, .35);
}
pre {
  overflow: auto;
  padding: 12px;
  background: #081524;
  border: 1px solid #263b55;
  border-radius: 9px;
}
@media (max-width: 1080px) {
  body { padding: 14px; }
  .container { grid-template-columns: minmax(0, 1fr); }
  .right-panel { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .right-panel > .card:last-child { grid-column: 1 / -1; }
}
@media (max-width: 680px) {
  body { padding: 9px; }
  .card { padding: 13px; border-radius: 11px; }
  .header { align-items: flex-start; flex-direction: column; }
  .h1 { font-size: 18px; }
  .right-panel { display: flex; }
  .input-wrap { grid-template-columns: minmax(0, 1fr) 72px; }
  .input-wrap .btn-inc { grid-row: 2; }
  .label { flex-basis: 120px; }
  .form-row { align-items: flex-start; flex-direction: column; gap: 5px; }
  .form-row .label { flex-basis: auto; }
  .form-row input:not([type=checkbox]), .form-row select, .form-row textarea { width: 100%; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition-duration: .01ms !important; animation-duration: .01ms !important; }
}
</style>
</head>
<body>
<div class="container">
  <div class="card">
    <div class="header">
      <div class="h1">CARLA Perception Studio</div>
      <div class="small">ROI calibration workspace · drag, resize, inspect and tune regions</div>
    </div>

    <div class="controls">
      <label class="small" for="variable_select">Select variable</label>
      <select id="variable_select">
        <option value="NONE">-- NONE --</option>
        {% for var in variables %}
          <option value="{{ var }}">{{ var }}</option>
        {% endfor %}
      </select>

      <button id="clear_selection" class="ghost">Clear</button>
      <button id="center_reset" class="ghost">Center Selected</button>
      <div style="flex:1"></div>
      <label class="small">Show ROIs <input id="show_rects" type="checkbox" checked></label>
    </div>

    <canvas id="video_canvas" width="900" height="600"></canvas>

    <div style="margin-top:8px;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
      <button id="apply_all" class="btn">Apply manual</button>
      <button id="refresh_vals" class="ghost">Refresh</button>
      <button id="download_json" class="ghost">Download JSON</button>
      <div style="flex:1"></div>
      <div class="small">Tip: Click in rectangle to select; drag corners to resize, drag inside to move.</div>
    </div>
  </div>

  <div class="right-panel">
    <div class="card">
      <div class="section-title">Manual ROI Values</div>
      <div class="inputs" id="inputs_container">
        {% for var in variables %}
        <div class="input-wrap">
          <div class="label">{{ var }}</div>
          <input name="{{ var }}" type="number" step="0.01" min="0" max="1" class="counter" value="{{ '%.2f' % values[var] }}">
          <button type="button" class="ghost btn-inc" data-var="{{ var }}" data-delta="0.1">+0.1</button>
          <button type="button" class="ghost btn-inc" data-var="{{ var }}" data-delta="-0.1">-0.1</button>
        </div>
        {% endfor %}
      </div>
    </div>

    <div class="card">
      <div class="section-title">ROI Visibility & Colors</div>
      <div class="toggle-row">
        <label><input type="checkbox" id="vis_rl"> Show Right Lane</label>
        <input type="color" id="color_rl" value="{{ ui.colors.RL }}">
      </div>
      <div class="toggle-row">
        <label><input type="checkbox" id="vis_ll"> Show Left Lane</label>
        <input type="color" id="color_ll" value="{{ ui.colors.LL }}">
      </div>
      <div class="toggle-row">
        <label><input type="checkbox" id="vis_cw"> Show Crosswalk Detection ROI</label>
        <input type="color" id="color_cw" value="{{ ui.colors.CW }}">
      </div>
      <div class="small-muted" style="margin:-4px 0 8px 24px">
        Area used by the current crosswalk / junction line detector.
      </div>
      <div class="toggle-row">
        <label><input type="checkbox" id="vis_dw"> Show Drivable-area Width ROI</label>
        <input type="color" id="color_dw" value="{{ ui.colors.DW }}">
      </div>
      <div class="small-muted" style="margin:-4px 0 8px 24px">
        Sampling area reserved for measuring drivable-road width.
      </div>
    </div>

    <div class="card">
      <div class="section-title">Advanced Settings (Confirm to apply)</div>
      <div class="form-row">
        <label class="label">LANE_THRESHOLD</label>
        <input id="lane_threshold" type="number" min="0" max="255" step="1" value="{{ advanced.LANE_THRESHOLD }}">
      </div>
      <div class="form-row">
        <label class="label">CROSSWALK_THRESHOLD</label>
        <input id="cross_thresh" type="number" min="0" max="255" step="1" value="{{ advanced.CROSSWALK_THRESHOLD }}">
      </div>
      <div class="form-row">
        <label class="label">CROSSWALK_SLEEP (sec)</label>
        <input id="cross_sleep" type="number" step="0.1" min="0" value="{{ advanced.CROSSWALK_SLEEP }}">
      </div>
      <div class="form-row">
        <label class="label">CROSSWALK_THRESH_SPEND (sec)</label>
        <input id="cross_spend" type="number" step="0.1" min="0" value="{{ advanced.CROSSWALK_THRESH_SPEND }}">
      </div>
      <div class="form-row">
        <label class="label">RUN_LVL</label>
        <select id="run_lvl">
          <option value="MOVE" {% if advanced.RUN_LVL == 'MOVE' %}selected{% endif %}>MOVE</option>
          <option value="STOP" {% if advanced.RUN_LVL == 'STOP' %}selected{% endif %}>STOP</option>
        </select>
      </div>

      <div style="display:flex;gap:8px;margin-top:8px">
        <button id="confirm_advanced" class="btn">Confirm</button>
        <button id="cancel_advanced" class="ghost">Cancel</button>
        <div style="flex:1"></div>
        <div id="advanced_msg" class="small-muted" style="align-self:center"></div>
      </div>
    </div>

    <div class="card">
      <div class="section-title">Runtime Control Panel Settings</div>
      <div class="small-muted">Settings marked [reset] need a CARLA restart. The others apply live.</div>
      <div id="runtime_settings" class="inputs"></div>
      <div style="display:flex;gap:8px;margin-top:8px">
        <button id="apply_runtime" class="btn">Apply runtime settings</button>
        <button id="refresh_runtime" class="ghost">Refresh settings</button>
      </div>
      <div id="runtime_msg" class="small-muted" style="margin-top:6px"></div>
    </div>

    <div class="card">
      <div class="section-title">Live values</div>
      <pre id="values_json" style="white-space:pre-wrap;color:var(--muted);font-size:13px;margin:0">{{ values|tojson }}</pre>
    </div>
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

function renderRuntimeSettings(){
  const root = document.getElementById('runtime_settings');
  root.innerHTML = '';
  for(const spec of runtimeSpecs){
    const [key, label, kind, limits, realtime] = spec;
    const row = document.createElement('div');
    row.className = 'form-row';
    const title = document.createElement('label');
    title.className = 'label';
    title.textContent = label + (realtime ? '' : ' [reset]');
    row.appendChild(title);
    let input;
    if(kind === 'bool'){
      input = document.createElement('input');
      input.type = 'checkbox';
      input.checked = !!runtimeValues[key];
    } else if(kind === 'choice'){
      input = document.createElement('select');
      for(const value of (limits || [])){
        const option = document.createElement('option');
        option.value = value;
        option.textContent = value === 'model'
          ? 'Junction model (ResNet18)'
          : value === 'crosswalk_roi'
            ? 'Crosswalk ROI — horizontal line'
            : value;
        input.appendChild(option);
      }
      input.value = runtimeValues[key] ?? '';
    } else if(kind === 'sequence'){
      input = document.createElement('textarea');
      input.rows = 3;
      input.value = JSON.stringify(runtimeValues[key] || {}, null, 2);
    } else {
      input = document.createElement('input');
      input.type = kind === 'str' ? 'text' : 'number';
      if(kind !== 'str' && limits){
        input.min = limits[0]; input.max = limits[1]; input.step = limits[2];
      }
      input.value = runtimeValues[key] ?? '';
    }
    input.dataset.key = key;
    input.style.flex = '1';
    row.appendChild(input);
    root.appendChild(row);
  }
}
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
    fetchFrameTimer = setInterval(fetchFrameOnce, 120);
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

const advLane = document.getElementById('lane_threshold');
const advCrossThresh = document.getElementById('cross_thresh');
const advCrossSleep = document.getElementById('cross_sleep');
const advCrossSpend = document.getElementById('cross_spend');
const advRunLvl = document.getElementById('run_lvl');
const advMsg = document.getElementById('advanced_msg');

function loadAdvanced(){
    fetch('/get_advanced').then(r=>r.json()).then(j=>{
        if(j && j.advanced){
            advanced = j.advanced;
            advLane.value = advanced.LANE_THRESHOLD;
            advCrossThresh.value = advanced.CROSSWALK_THRESHOLD;
            advCrossSleep.value = advanced.CROSSWALK_SLEEP;
            advCrossSpend.value = advanced.CROSSWALK_THRESH_SPEND;
            advRunLvl.value = advanced.RUN_LVL;
            advMsg.textContent = '';
        }
    }).catch(console.warn);
}
loadAdvanced();

document.getElementById('confirm_advanced').addEventListener('click', ()=>{
    const payload = {
        LANE_THRESHOLD: parseInt(advLane.value,10) || 0,
        CROSSWALK_THRESHOLD: parseInt(advCrossThresh.value,10) || 0,
        CROSSWALK_SLEEP: parseFloat(advCrossSleep.value) || 0,
        CROSSWALK_THRESH_SPEND: parseFloat(advCrossSpend.value) || 0,
        RUN_LVL: advRunLvl.value === 'STOP' ? 'STOP' : 'MOVE'
    };
    fetch('/set_advanced', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify(payload)
    })
    .then(r=>r.json())
    .then(j=>{
        if(j && j.success){
            advanced = j.advanced;
            advMsg.textContent = 'Applied';
            showToast('Advanced settings applied');
            refreshValues();
            setTimeout(()=> advMsg.textContent = '', 1800);
        } else {
            advMsg.textContent = 'Failed';
        }
    })
    .catch(()=>{
        advMsg.textContent = 'Error';
    });
});

document.getElementById('cancel_advanced').addEventListener('click', ()=>{
    loadAdvanced();
    showToast('Canceled');
});

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
    data = request.get_json() or {}
    specs = {spec[0]: spec for spec in SETTING_SPECS}
    updated = {}
    reset_required = False
    try:
        for key, value in data.items():
            spec = specs.get(key)
            if spec is None:
                continue
            coerced = _coerce_setting(spec, value)
            setattr(conf, key, coerced)
            updated[key] = coerced
            reset_required = reset_required or not spec[4]
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return jsonify(success=False, error=f"invalid setting: {exc}"), 400

    _notify_settings(updated)
    return jsonify(success=True, values=_setting_values(), reset_required=reset_required)

@app.route('/set_advanced', methods=['POST'])
def set_advanced():
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
