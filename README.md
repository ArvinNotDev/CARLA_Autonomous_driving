# CARLA Autonomous Driving

[![GUI smoke tests](https://github.com/ArvinNotDev/CARLA_Autonomous_driving/actions/workflows/gui-smoke-tests.yml/badge.svg?branch=main)](https://github.com/ArvinNotDev/CARLA_Autonomous_driving/actions/workflows/gui-smoke-tests.yml)
![Python](https://img.shields.io/badge/Python-3.12%20recommended-3776AB?logo=python&logoColor=white)
![CARLA](https://img.shields.io/badge/CARLA-1.0-222222)
![Interface](https://img.shields.io/badge/Interface-Flask%20Web%20Dashboard-0F9D8A)

An experimental autonomous-driving research project built around **CARLA 1.0**. The stack combines camera-based lane and drivable-area perception, route planning, junction handling, lane-change control, optional trajectory steering, and a live Flask browser dashboard.

> **Current development target:** CARLA 1.0 (UE5) on the `main` branch. The legacy CARLA 0.9.16 baseline is preserved separately on `main_9.16`.

---

## Version tracks

The repository keeps the two incompatible CARLA generations on separate branches. Keep the CARLA server and Python API from the same installation; a 0.9.16 client must not be used with a 1.0 server, or vice versa.

| Branch | Target | Purpose |
| --- | --- | --- |
| `main` | **CARLA 1.0** | Primary development branch for the current v1 codebase |
| `main_9.16` | **CARLA 0.9.16** | Preserved legacy baseline for the previous CARLA generation |

The `main_9.16` branch is a snapshot of the former `main` at the point before the CARLA 1.0 integration was promoted. New v1 development should happen directly on `main`; use `main_9.16` only when maintaining or reproducing the legacy 0.9.16 setup.

### Switch versions

**CARLA 1.0 (current):**

```bash
git fetch origin
git switch main
git pull --ff-only origin main
```

**CARLA 0.9.16 (legacy):**

```bash
git fetch origin
git switch main_9.16
git pull --ff-only origin main_9.16
```

---

## Capabilities

- **Lane perception:** YOLO26s-sem ONNX inference for lane and drivable-area understanding, with configurable thresholds and regions of interest.
- **Drivable-area recovery:** a configurable recovery gate that reacts to sustained perception errors and can use LiDAR obstacle checks to suppress unsafe corrections.
- **Route planning:** CARLA Global Route Planner integration using the navigation agents bundled with the selected CARLA 1.0 installation.
- **Junction handling:** route-aware junction behavior with crosswalk-ROI detection or an optional ResNet18 junction classifier.
- **Lane-change control:** explicit steering ownership during lane changes, distance-based steering attempts, asynchronous route rechecks, and post-junction protection.
- **Optional trajectory steering:** waypoint prediction and trajectory-based steering can be enabled from a runtime profile when required.
- **Manual override:** keyboard driving alongside autonomous control.
- **Live diagnostics:** camera/debug view, FPS metrics, GPS, lane and drivable-area measurements, junction/lane-change state, and optional LiDAR radar.
- **Runtime configuration:** persistent named profiles, a Flask-based ROI editor, live settings, and automatic in-place simulation restart for sensor/model changes.

## Runtime architecture

The application is organized around three execution paths:

1. **Web dashboard:** Flask serves the live browser console for perception frames, telemetry, LiDAR, ROI calibration, profiles, vehicle selection, and every runtime setting. PySide6 is no longer used for runtime settings or debugging; it remains for the initial CARLA installation, map, and spawn/goal selectors.
2. **Vehicle control:** one dedicated control thread owns vehicle commands and runs at the configured `CONTROL_LOOP_HZ`.
3. **Inference:** a serialized inference worker handles vision, optional trajectory prediction, and model-based junction inference. It keeps the newest pending work rather than building an unbounded queue of stale camera frames.

A latest-frame buffer connects the control and inference paths to Flask. The browser refreshes camera frames at `DEBUG_PANEL_HZ` and telemetry independently, so web rendering and polling do not define the vehicle-control cadence.


### Ultralytics semantic model

The active lane/drivable-area/vehicle perception model uses an Ultralytics YOLO26-sem checkpoint rather than ONNX Runtime. Place your trained checkpoint at:

`models_and_datasets/lane_and_drivable-area_model.pt`

The checkpoint must expose semantic class IDs named `lane_lines`, `ego_drivable_area`, and `vehicles`. The input size defaults to 512 and can be adjusted in the web dashboard under **Settings → Vision & perception**. PyTorch CUDA is selected automatically when available; otherwise inference uses CPU. If CUDA inference raises an error at runtime, the processor retries on CPU.

### Runtime settings and restart behavior

Settings are divided into two groups:

- **Live settings** update the active controller, thresholds, ROI values, or display behavior without rebuilding the simulation.
- **Sensor/model settings** trigger an automatic in-place runtime restart. The control loop is stopped, the active sensors and ego vehicle are cleaned up, and the vehicle is spawned again at the previously selected spawn transform. The web dashboard remains available; the map and goal selection are retained.

Vehicle selection schedules a debounced restart. Explicit settings restarts avoid respawning while you are still editing.

---

## Requirements

### Software

- Windows 10/11 (the packaged CARLA 1.0 distribution is the target environment).
- **CARLA 1.0** server.
- **Python 3.12, 64-bit recommended.** The application selects the CARLA wheel matching the active interpreter and platform from the installed CARLA package.
- Git.
- Python dependencies listed in `requirements.txt`.

The project uses NumPy, OpenCV, Pillow, PyTorch, ONNX Runtime, Pygame, PySide6, and NetworkX. CUDA is optional; the available inference providers depend on the installed runtime and GPU environment.

### CARLA Python API

Do **not** install an unrelated CARLA client from PyPI for the v1 branch. The application loads the matching Python wheel shipped with the selected CARLA 1.0 installation:

```text
< CARLA 1.0 installation >/
├── CarlaUnreal.exe
├── Engine/
└── PythonAPI/
    └── carla/
        └── dist/
            └── carla-<version>-cp<python>-...whl
```

At startup, the application asks whether to use the previously saved installation or select another folder. You can choose the CARLA installation root, its `PythonAPI` folder, or its `PythonAPI/carla` subfolder. The wheel is installed into an application-local cache and loaded before other CARLA packages. The global Python environment is not used as a fallback.

The application also loads CARLA's bundled navigation agents and verifies the client/server versions after connecting. If no compatible wheel is available, startup stops with a diagnostic instead of silently loading a different CARLA generation.

---

## Installation

Clone the repository and create a virtual environment:

```powershell
git clone https://github.com/ArvinNotDev/CARLA_Autonomous_driving.git
cd CARLA_Autonomous_driving

py -3.12 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The CARLA Python API is deliberately provided by the selected CARLA installation and is not installed by the requirements file.

The runtime expects the following model assets to be present:

- `models_and_datasets/lane_and_drivable-area_model.onnx`
- `models_and_datasets/models/junction_model_resnet18.pt` when model-based junction detection is selected
- `models_and_datasets/models/trajectory_modelV2.pt` when trajectory steering is enabled

---

## Run

1. Start the **CARLA 1.0** server.
2. Launch the application from the repository root:

```bash
python main.py
```

For a lower-rendering-cost windowed launch on Windows, you can start the server with:

```bat
CarlaUnreal.exe -quality-level=Low -ResX=1280 -ResY=720 -windowed
```

The simulator window resolution and the ego camera sensor resolution are separate settings. The bundled `StableV2` profile retains its own camera configuration.

On every application launch, choose **Use saved installation** or **Choose another folder** in the CARLA installation dialog. After connecting, select the map, spawn point, and goal. The Flask runtime dashboard starts automatically. Open `http://127.0.0.1:5000` in your browser (or use the configured `STREAM_PORT`).

#
### AVIS Engine 1.2.4

The AVIS runtime is an independent backend. It does not initialize or import the
CARLA Python API, and it does not alter the normal `python main.py` startup path.
It has its own Flask dashboard, AVIS-only settings file, manual controls, and
vision pipeline. The Ultralytics backend reuses the StableV2 camera perception
model and AVIS PID/speed defaults. The Classical CV backend does not run a
segmentation model to obtain lane pixels: it independently crops the configured
left-lane (LL) and right-lane (RL) rectangles, then runs HLS white/yellow
thresholding, Gaussian blur, Canny, morphology, and HoughLinesP strictly inside
each crop. Its resulting binary mask follows the same downstream procedure as
the model-backed lane mask: median x-coordinate per ROI, line-angle estimation,
sticky-side selection, lane-offset calculation, and centre smoothing. The
classical drivable-area view is an estimated corridor, not a semantic
road-surface prediction.

Use the Python API revision compatible with AVIS 1.2.4 (the vendor's
`FIRA-ACL` branch). Clone it beside the project:

```powershell
git clone --branch FIRA-ACL --single-branch https://github.com/AVISEngine/AVIS-Engine-Python-API.git
```

Alternatively, set `AVIS_ENGINE_API_PATH` to the API repository folder, or pass
`--avis-api-path` when launching. Start AVIS Engine, select a track, open the
Info Panel, and start its TCP server (default IP `127.0.0.1`, port `25001`).
Then run:

```powershell
python main.py --avis
```

The AVIS Flask dashboard starts at `http://127.0.0.1:5001`. It provides
live camera/overlay views, the lane and drivable-area masks, Auto / Manual /
Emergency Stop modes, forward/reverse manual controls, and live controller
and classical-vision settings. The live ROI editor overlays the camera image:
select LL or RL, drag the rectangle to move it, or drag a corner to resize it.
Exact normalized coordinates can also be entered numerically. These two ROIs
are shared by Ultralytics and Classical CV. Settings are saved separately in
`avis_settings.json`; this file is local and git-ignored. They do not modify
`runtime_profiles.json`, CARLA configuration, or CARLA controllers.

Select **Classical CV** in the Vision backend field to use the adjustable HLS,
blur, Canny, morphology, and HoughLinesP pipeline. Both backends use the same
LL/RL ROI rectangles and the same downstream lane-centre calculation. Select
**Ultralytics** to use the trained semantic checkpoint. When the configured Ultralytics
checkpoint is missing, AVIS starts in classical mode and reports the fallback.

For manual driving, switch to **Manual** and use `W` / `S` for forward /
reverse and `A` / `D` for steering. The dashboard applies a short command
watchdog: if browser heartbeats stop, AVIS releases throttle and centers
steering. Use **Emergency stop** to disarm both automatic and manual commands.

Optional dashboard overrides:

```powershell
python main.py --avis --avis-web-host 127.0.0.1 --avis-web-port 5001
```

To access the dashboard from another device on your LAN, bind to
`0.0.0.0` with `--avis-web-host 0.0.0.0` and open the host PC's LAN address
on port `5001`. The dashboard has no authentication; use this only on a
trusted private network. Add `--avis-no-web` to disable Flask, or
`--avis-headless` to disable only the local OpenCV preview.

Optional connection overrides:

```powershell
python main.py --avis --avis-host 127.0.0.1 --avis-port 25001
```

The AVIS settings profile is `AVIS_StableV2`. It copies the StableV2 lane
perception, PID, inference cadence, and speed settings while omitting CARLA-only
camera placement, map navigation, junction, trajectory, and LiDAR settings.
The AVIS API supplies only RGB frames, speed feedback, and three distance
readings; the distance readings are displayed for diagnostics and are not used
to introduce a different driving strategy.

AVIS accepts engine power and steering commands but does not expose a separate
brake control in this API revision. When the speed controller requests braking,
the AVIS runtime releases engine power and allows the vehicle to coast. Use the
simulator's own safety controls and supervise initial runs.

Press `Q` or `Esc` in the OpenCV preview to stop the AVIS runtime. Add
`--avis-headless` to disable the preview window.

## Controls

| Input | Action |
| --- | --- |
| `E` | Toggle autonomous/manual driving |
| `W` / `A` / `S` / `D` | Manual throttle, steering, and reverse |
| `Space` | Brake in manual mode |
| `Esc` | Exit the control window/application |

The ROI editor can also be started separately when no CARLA session is running:

```bash
python stream.py
```

---

## Default runtime profile: StableV2

The v1 branch starts with the existing **`StableV2`** profile selected. Its stored profile values are preserved as-is; the profile is not replaced by a separate low-GPU preset.

Key camera and runtime values in that profile:

| Setting | StableV2 |
| --- | ---: |
| RGB camera resolution | 512 × 320 |
| RGB camera sensor interval | 0.025 s |
| Camera field of view | 120° |
| Control loop | 40 Hz |
| Dashboard refresh | 40 Hz |
| Semantic camera | Enabled |
| Alternate camera | Enabled |
| LiDAR | Disabled |
| Trajectory model | Disabled by default |
| Junction detection | Crosswalk ROI |

Use the **Settings** tab in the browser dashboard to search settings, load or save named profiles, choose a vehicle, and apply or restart runtime changes. Profile values are stored in `runtime_profiles.json`.

**Hardware note:** CARLA 1.0 is a demanding UE5-based simulator. Lowering the server's render quality and display resolution may reduce GPU load, but does not guarantee compatibility with GPUs below CARLA's recommended hardware level. The project's sensor resolution is controlled separately by the runtime profile.

---

## Perception and driving details

### Junction detection

The **Junction detection method** setting selects either the trained ResNet18 classifier or **Crosswalk ROI — horizontal line**. ROI mode searches the configured `CW_*` rectangle for a sufficiently long, near-horizontal marking, confirms it across consecutive frames, and then enters the same route-planning and crosswalk-stop pipeline. The detector exposes line-length, angle, confirmation, and re-arm settings.

### Lane changes

During a lane change, the lane-change manager retains steering ownership while route-planner requests are rechecked asynchronously. The normal lane-follow controller is released only when the active maneuver is complete. Configurable approach distance, attempt distance, steering magnitude, planner interval, and debounce settings control the process.

### Drivable-area recovery

The recovery gate monitors sustained drivable-area errors before requesting corrective steering. Threshold, time window, steering, and throttle can be configured. When enabled, the optional LiDAR gate can suppress corrections toward side obstacles; LiDAR is not used as a road-edge or shoulder detector.

### Road-span index

The independent `DW_*` ROI reports the median horizontal span of positive drivable-mask pixels across valid image rows. It is displayed in pixels and as a frame-width ratio. This is a relative image-space index, **not a measurement of physical road width in metres**.

---

## Repository layout

```text
main.py                         Application setup, control loop, runtime restart
carla_manager.py                CARLA client, map, vehicle spawn, cleanup
config_city.py                  Runtime defaults and pipeline configuration
runtime_settings.py             Profile loading, applying, and persistence
runtime_profiles.json           Named runtime profiles (StableV2 is active)
controllers/                    PID and manual input
driving/                        Autonomous driving and recovery gates
navigation/                     Route, junction, and lane-change logic
sensors/                        RGB/semantic cameras and optional LiDAR
trajectory/                     Waypoint model, steering agent, visualization
vision/                         ONNX/segmentation perception and debug views
ui/                             Legacy desktop panel, renderers, spawn/goal startup dialogs
models_and_datasets/             Model checkpoints and ONNX assets
tests/                           Regression and GUI smoke tests
```

## Troubleshooting

**CARLA client/server version mismatch**

- Confirm that the running server is CARLA 1.0.
- Re-run `python main.py` and select the PythonAPI folder from that exact CARLA 1.0 installation.
- Do not use the 0.9.16 Python API with a 1.0 server.

**No compatible CARLA wheel found**

- Check that `PythonAPI/carla/dist` contains a wheel matching the active Python version and Windows architecture.
- Use Python 3.12 64-bit if you are unsure which interpreter to choose.
- Re-open the installation selector and choose the CARLA root or its `PythonAPI` folder.

**The vehicle does not move after spawning**

- Confirm that the server is running and that the selected spawn point is on a drivable lane.
- Open the web dashboard and confirm that Auto mode is enabled in **Settings → Driving & PID**.
- Allow the camera stream to initialize after a restart.

**No ONNX inference**

- Verify the Ultralytics `.pt` model path in the runtime settings and confirm the checkpoint contains the expected semantic class names.
- Check the startup log for `Ultralytics device: CUDA:0` or the CPU fallback message. Confirm that the installed PyTorch build supports the NVIDIA GPU.
- Check that the ONNX model file exists.

**The dashboard does not appear**

- Install the dependencies from `requirements.txt`.
- Run `python main.py` from the repository root, then open `http://127.0.0.1:5000`. Check the terminal output for Flask bind or startup errors.

---

## Tests

Run the local regression tests with:

```bash
python -m pytest -q
```

The GitHub Actions smoke-test workflow compiles the application modules and runs the CARLA installation, legacy component, perception-debugger, and StableV2-profile checks. The Flask dashboard tests cover settings validation and browser routes. The CI checks do not replace an end-to-end run against a locally installed CARLA 1.0 server. The CI checks do not replace an end-to-end run against a locally installed CARLA 1.0 server.

## Third-party notice

Perception code includes adaptations from **YOLOPv2-ONNX-Sample** by Kazuhito00. The corresponding MIT license notice is included at `LICENSES/YOLOPv2-ONNX-Sample-MIT.txt`.
