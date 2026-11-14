# CARLA Autonomous Driving

An experimental autonomous-driving stack for [CARLA](https://carla.org/) with lane perception, drivable-area reasoning, route planning, junction handling, trajectory prediction, live debugging, and manual override.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![CARLA](https://img.shields.io/badge/CARLA-0.9.x-222222)
![PySide6](https://img.shields.io/badge/UI-PySide6-41CD52)
![License](https://img.shields.io/badge/License-MIT-blue)

## What it does

- Lane following from YOLO26s-sem ONNX perception and optional semantic segmentation.
- Learned waypoint/trajectory steering for junctions, preceded by a short configurable static meter-based lead-in.
- Route planning, junction classification, lane-change protection, spawn/goal selection, and manual WASD control.
- A PySide6 runtime panel for live tuning of throttle, PID gains, target speed, trajectory parameters, ROIs, model thresholds, and debug overlays.
- Persistent named profiles in `runtime_profiles.json`.
- Live FPS/GPS overlays and trajectory, ROI, lane-mask, and vision debugging.

## Runtime architecture

The PySide6 GUI runs its normal Qt event loop on the main thread. Vehicle control runs at a fixed cadence (`CONTROL_LOOP_HZ`) on one dedicated control thread. A single inference worker serializes vision, trajectory, and junction-model inference and keeps only the newest pending frame for each task type.

The debug panel uses a latest-frame buffer and its own Qt refresh timer (`DEBUG_PANEL_HZ`). It converts and scales frames only when the panel actually refreshes. Display FPS is measured from newly painted debug frames over a rolling time window, not from control-loop iterations or inference completions.

Trajectory output is an immutable, timestamped result. Busy/failed inference never replaces the last valid prediction with `None`; result age and worker state are exposed in the diagnostics overlay.

## Optional trajectory model

The trajectory steering model is controlled by the `TRAJECTORY_MODEL_ENABLED` profile setting and the runtime-panel checkbox. When disabled, the application does not import the trajectory PyTorch agent or load `trajectory_modelV2.pt`, and no trajectory inference requests are submitted. The fixed-distance junction lead-in remains active; after its final segment, control returns directly to the lane-line controller instead of entering trajectory takeover. Disabling the model during a run takes effect immediately: new requests stop, queued work is cleared, and the model is released after any already-running inference completes. Enabling it again requires a runtime restart so the checkpoint can be loaded outside the control loop. The `StableV2` profile defaults this option to off.

## Junction behavior

Junction handling is intentionally one pipeline:

1. Detect a junction and determine the next route maneuver.
2. Run a short configurable static meter-based lead-in.
3. Return control to the trajectory model for the configurable takeover duration with the route command (`LEFT`, `RIGHT`, or `STRAIGHT`).

The planner may run asynchronously, but no junction worker calls `vehicle.apply_control()`. The control thread is the sole owner of vehicle commands during normal driving, lane changes, static junction lead-ins, and trajectory takeover. There is no blocking render-loop sleep and no legacy full-left recovery command.

## Lane-change steering handoff

A lane change keeps steering ownership away from the normal lane-follow PID while the CARLA route planner continues to return `CHANGE_LANE_LEFT` or `CHANGE_LANE_RIGHT`. Camera lane-line angles do not decide when the maneuver is complete and cannot change steering during this phase. After the configured approach distance, the manager performs two attempts using the configured direction-specific steering magnitude, then two attempts with zero steering. Each attempt covers the configured attempt distance. If GPS still requests the same lane change after both straight attempts, the two-steer/two-straight cycle repeats. GPS is queried asynchronously after each attempt; while that query is pending, the vehicle continues straight under lane-change-manager control.

Only when GPS no longer requests the active lane change is the manager released and ordinary lane-follow PID steering allowed again. If GPS immediately requests the opposite lane change, control ownership stays with the lane-change manager and that maneuver starts without an intervening PID handoff.

After a junction crosswalk stop and its configured movement/trajectory sequence are complete, fresh lane-change planner queries are held for six seconds by default. The system also suppresses new lane-change requests while trajectory takeover is active, including when the lane controller is used as a fallback because a fresh trajectory prediction is not available. When the delay expires, GPS is queried again and a lane change starts only if the route planner still requests one. The delay is configurable through **Wait after junction before lane change (s)**.

The **Lane change** runtime settings expose the approach distance, distance per attempt, steering magnitude, planner query interval, debounce, and the post-junction delay.

## Junction detection modes

The runtime setting **Junction detection method** selects either the trained ResNet18 junction classifier or **Crosswalk ROI — horizontal line**. The ROI mode uses the normalized `CW_TOP_ROI`, `CW_BOTTOM_ROI`, `CW_LEFT_ROI`, and `CW_RIGHT_ROI` values edited in the Flask ROI editor. It searches the camera image inside that rectangle for a sufficiently long, near-horizontal edge using OpenCV, then confirms it across consecutive vision frames. A confirmed line enters the same junction-planning and full-brake `CROSSWALK` state used by the existing controller. The detected segment is highlighted in the debug image.

The detector exposes minimum line-length ratio, angle tolerance, confirmation frames, and a clear-time re-arm delay in the **Junction detection** settings tab. The re-arm delay prevents adjacent zebra stripes from repeatedly triggering stops. Changing between the two detection modes requires a runtime restart; in ROI mode the PyTorch junction classifier is not imported or loaded. The included `StableV2` profile uses ROI detection, while the other bundled profiles retain the model-based method.

## Optional LiDAR radar

LiDAR is disabled by default in the application and every included runtime profile. Use the **LiDAR: OFF/ON** button beside the radar in the runtime panel to start or completely stop the CARLA sensor while the simulation is running. When enabled, the radar displays the point cloud; the optional recovery safety check can also use LiDAR to avoid steering corrections toward detected side obstacles. LiDAR is not used to detect road edges or road shoulders. The radar zoom is available under **LiDAR display**, while **LiDAR sensor range** is under **Hardware (reset)** and requires a run restart.

## Requirements

- CARLA 0.9.x server running on `localhost:2000`.
- Python 3.10+ (3.12 recommended).
- A working CARLA Python API installation.
- NVIDIA CUDA is optional; ONNX Runtime and PyTorch will use CPU when CUDA is unavailable.

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

The runtime expects `models_and_datasets/lane_and_drivable-area_model.onnx` and trajectory/junction checkpoints under `models_and_datasets/models/`.

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

Use the profile controls at the bottom of the PySide6 panel to save or load a named configuration. The repository includes the `stableV1` profile as the default startup profile; the last saved values are written to `runtime_profiles.json`.

Live settings update the active controller or perception pipeline immediately. Settings that change CARLA sensor resources or model loading are clearly marked **Hardware (reset)** and show a reset prompt.

## Project layout

```text
main.py                    Application loop and live control-panel integration
carla_manager.py           CARLA connection, spawn, and cleanup
controllers/               PID and keyboard control
driving/                   Autonomous driving orchestration
navigation/                Route, junction, lane-side, and lane-change logic
sensors/                   RGB/semantic cameras and optional LiDAR
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
