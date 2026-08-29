# CARLA Autonomous Driving

An experimental autonomous-driving stack for [CARLA](https://carla.org/) with lane perception, drivable-area reasoning, route planning, junction handling, trajectory prediction, live debugging, and manual override.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![CARLA](https://img.shields.io/badge/CARLA-0.9.x-222222)
![PySide6](https://img.shields.io/badge/UI-PySide6-41CD52)
![License](https://img.shields.io/badge/License-MIT-blue)

## What it does

- Lane following from YOLOPv2 ONNX perception and optional semantic segmentation.
- Learned waypoint/trajectory steering for junctions, preceded by a short configurable static meter-based lead-in.
- Route planning, junction classification, lane-change protection, spawn/goal selection, and manual WASD control.
- A PySide6 runtime panel for live tuning of throttle, PID gains, target speed, trajectory parameters, ROIs, model thresholds, and debug overlays.
- Persistent named profiles in `runtime_profiles.json`.
- Live FPS/GPS overlays and trajectory, ROI, lane-mask, and vision debugging.

## Runtime architecture

The PySide6 GUI runs its normal Qt event loop on the main thread. Vehicle control runs at a fixed cadence (`CONTROL_LOOP_HZ`) on one dedicated control thread. A single inference worker serializes vision, trajectory, and junction-model inference and keeps only the newest pending frame for each task type.

The debug panel uses a latest-frame buffer and its own Qt refresh timer (`DEBUG_PANEL_HZ`). It converts and scales frames only when the panel actually refreshes. Display FPS is measured from newly painted debug frames over a rolling time window, not from control-loop iterations or inference completions.

Trajectory output is an immutable, timestamped result. Busy/failed inference never replaces the last valid prediction with `None`; result age and worker state are exposed in the diagnostics overlay.

## Junction behavior

Junction handling is intentionally one pipeline:

1. Detect a junction and determine the next route maneuver.
2. Run a short configurable static meter-based lead-in.
3. Return control to the trajectory model for the configurable takeover duration with the route command (`LEFT`, `RIGHT`, or `STRAIGHT`).

The planner may run asynchronously, but no junction worker calls `vehicle.apply_control()`. The control thread is the sole owner of vehicle commands during normal driving, lane changes, static junction lead-ins, and trajectory takeover. There is no blocking render-loop sleep and no legacy full-left recovery command.

## Requirements

- CARLA 0.9.x server running on `localhost:2000`.
- Python 3.10+ (3.12 recommended).
- A working CARLA Python API installation.
- NVIDIA CUDA is optional; ONNX Runtime and PyTorch will use CPU when CUDA is unavailable.

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

The runtime expects the YOLOPv2 ONNX model and trajectory/junction checkpoints under `models_and_datasets/models/`.

## Run

Start the CARLA server, then:

```bash
python main.py
```

Choose a spawn point and goal. The runtime control panel opens automatically. Press `E` in the CARLA control window to toggle autonomous/manual mode; `WASD` controls the car in manual mode; `Esc` exits.

The existing ROI/advanced editor is also available through:

```bash
python stream.py
```

## Runtime profiles

Use the profile controls at the bottom of the PySide6 panel to save or load a named configuration. The last saved values are written to `runtime_profiles.json`, which is intentionally local and ignored by Git for machine-specific tuning.

Live settings update the active controller or perception pipeline immediately. Settings that change CARLA sensor resources or model loading are clearly marked **Hardware (reset)** and show a reset prompt.

## Project layout

```text
main.py                    Application loop and live control-panel integration
carla_manager.py           CARLA connection, spawn, and cleanup
controllers/               PID and keyboard control
driving/                   Autonomous driving orchestration
navigation/                Route, junction, lane-side, and lane-change logic
sensors/                   RGB, semantic, and trajectory cameras
trajectory/                Waypoint model, steering agent, visualization
vision/                    ONNX/segmentation perception and debug views
ui/                        Spawn/goal picker, renderer, runtime control panel
config_city.py             Defaults and JSON overrides
roi_config.json            ROI/advanced settings
```

## Troubleshooting

- **The car does not move after spawning:** confirm CARLA is running, the selected spawn point is on a drivable lane, and that the panel is in Auto mode. The app applies a short startup throttle grace period while camera frames become available.
- **No ONNX inference:** verify `MODEL_PATH`, install `onnxruntime` (or `onnxruntime-gpu`), and check that the model file exists.
- **Panel does not appear:** install `PySide6` and run from the repository root.
- **CARLA API import errors:** add your CARLA `PythonAPI` directory to `PYTHONPATH` or update the paths used by `navigation/global_planner.py`.

## Third-party notice

Perception code includes adaptations from **YOLOPv2-ONNX-Sample** by Kazuhito00. The original MIT license is included in `LICENSES/YOLOPv2-ONNX-Sample-MIT.txt`.
