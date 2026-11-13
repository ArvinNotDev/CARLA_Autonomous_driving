from __future__ import annotations

import socket
from typing import Any, Callable, Optional

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QPaintEvent
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QPushButton, QScrollArea, QSpinBox,
    QTabWidget, QVBoxLayout, QWidget,
)

import config_city as conf

from runtime_metrics import DisplayMetrics
from runtime_settings import SETTING_SPECS, RuntimeSettings
from ui.frame_store import DebugFrameStore
from ui.lidar_radar import LidarRadarWidget


class DebugFrameLabel(QLabel):
    """Counts each newly submitted pixmap once, when it is actually painted."""

    def __init__(self, display_metrics: DisplayMetrics, parent=None) -> None:
        super().__init__(parent)
        self._display_metrics = display_metrics
        self._frame_token = 0
        self._last_painted_token = 0

    def setPixmapForFrame(self, pixmap: QPixmap) -> None:
        self._frame_token += 1
        self.setPixmap(pixmap)

    def setPixmapResized(self, pixmap: QPixmap) -> None:
        self.setPixmap(pixmap)

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        if self._last_painted_token != self._frame_token:
            self._last_painted_token = self._frame_token
            self._display_metrics.record_presented()


class JunctionSequenceEditor(QWidget):
    """Edit ordered distance/steering actions for each junction maneuver."""

    _maneuvers = (("LEFT", "Left Turn"), ("RIGHT", "Right Turn"), ("STRAIGHT", "Straight"))
    changed = Signal(dict)

    def __init__(self, sequences: Optional[dict[str, Any]] = None, parent=None) -> None:
        super().__init__(parent)
        self._sequences = self._normalize(sequences or {})
        self._layout = QVBoxLayout(self)
        self._rebuild()

    @classmethod
    def _normalize(cls, sequences: Any) -> dict[str, list[dict[str, float]]]:
        result = {}
        for maneuver, _title in cls._maneuvers:
            actions = []
            raw_actions = sequences.get(maneuver, []) if isinstance(sequences, dict) else []
            for action in raw_actions if isinstance(raw_actions, list) else []:
                if not isinstance(action, dict):
                    continue
                try:
                    distance = max(0.0, float(action.get("distance_m", 0.0)))
                    steering = max(-1.0, min(1.0, float(action.get("steering_value", 0.0))))
                except (TypeError, ValueError):
                    continue
                if distance > 0.0:
                    actions.append({"distance_m": distance, "steering_value": steering})
            result[maneuver] = actions
        return result

    def sequences(self) -> dict[str, list[dict[str, float]]]:
        return {key: [dict(item) for item in value] for key, value in self._sequences.items()}

    def set_sequences(self, sequences: Any) -> None:
        self._sequences = self._normalize(sequences)
        self._rebuild()

    def _rebuild(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._layout.addWidget(QLabel(
            "هر action به ترتیب اجرا می‌شود؛ steering value بین -1 (چپ) و +1 (راست) است."
        ))
        for maneuver, title in self._maneuvers:
            box = QGroupBox(title)
            box_layout = QVBoxLayout(box)
            grid = QGridLayout()
            for col, text in enumerate(("#", "Distance (m)", "Steering value", "Order", "Action")):
                grid.addWidget(QLabel(text), 0, col)
            actions = self._sequences[maneuver]
            for index, action in enumerate(actions):
                distance = QDoubleSpinBox()
                distance.setRange(0.01, 1000.0)
                distance.setSingleStep(0.5)
                distance.setDecimals(2)
                distance.setValue(action["distance_m"])
                steering = QDoubleSpinBox()
                steering.setRange(-1.0, 1.0)
                steering.setSingleStep(0.05)
                steering.setDecimals(3)
                steering.setValue(action["steering_value"])
                distance.valueChanged.connect(
                    lambda value, m=maneuver, i=index: self._update(m, i, distance_m=value)
                )
                steering.valueChanged.connect(
                    lambda value, m=maneuver, i=index: self._update(m, i, steering_value=value)
                )
                grid.addWidget(QLabel(str(index + 1)), index + 1, 0)
                grid.addWidget(distance, index + 1, 1)
                grid.addWidget(steering, index + 1, 2)
                order = QWidget()
                order_layout = QHBoxLayout(order)
                order_layout.setContentsMargins(0, 0, 0, 0)
                up = QPushButton("↑")
                down = QPushButton("↓")
                up.setEnabled(index > 0)
                down.setEnabled(index < len(actions) - 1)
                up.clicked.connect(lambda _=False, m=maneuver, i=index: self._move(m, i, -1))
                down.clicked.connect(lambda _=False, m=maneuver, i=index: self._move(m, i, 1))
                order_layout.addWidget(up)
                order_layout.addWidget(down)
                grid.addWidget(order, index + 1, 3)
                remove = QPushButton("Remove")
                remove.clicked.connect(lambda _=False, m=maneuver, i=index: self._remove(m, i))
                grid.addWidget(remove, index + 1, 4)
            box_layout.addLayout(grid)
            add = QPushButton("Add steering action")
            add.clicked.connect(lambda _=False, m=maneuver: self._add(m))
            box_layout.addWidget(add)
            self._layout.addWidget(box)
        self._layout.addStretch(1)

    def _notify(self) -> None:
        self.changed.emit(self.sequences())

    def _update(self, maneuver: str, index: int, **values: float) -> None:
        if index < len(self._sequences[maneuver]):
            self._sequences[maneuver][index].update(values)
            self._notify()

    def _add(self, maneuver: str) -> None:
        self._sequences[maneuver].append({"distance_m": 1.0, "steering_value": 0.0})
        self._rebuild()
        self._notify()

    def _remove(self, maneuver: str, index: int) -> None:
        if index < len(self._sequences[maneuver]):
            self._sequences[maneuver].pop(index)
            self._rebuild()
            self._notify()

    def _move(self, maneuver: str, index: int, offset: int) -> None:
        actions = self._sequences[maneuver]
        target = index + offset
        if 0 <= target < len(actions):
            actions[index], actions[target] = actions[target], actions[index]
            self._rebuild()
            self._notify()


class ControlPanel(QMainWindow):
    settings_changed = Signal(dict)
    external_settings_changed = Signal(dict)
    reset_requested = Signal()
    stream_toggle_requested = Signal(bool)

    def __init__(
        self,
        settings: RuntimeSettings,
        *,
        display_metrics: Optional[DisplayMetrics] = None,
        frame_store: Optional[DebugFrameStore] = None,
        lidar_manager=None,
        metrics_provider: Optional[Callable[[], dict[str, Any]]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.settings = settings
        self.external_settings_changed.connect(self._apply_external_settings)
        self.display_metrics = display_metrics or DisplayMetrics()
        self.frame_store = frame_store
        self.lidar_radar = LidarRadarWidget(lidar_manager)
        self.metrics_provider = metrics_provider
        self.widgets: dict[str, QWidget] = {}
        self.lidar_toggle = QPushButton()
        self.lidar_toggle.setCheckable(True)
        self.lidar_toggle.setToolTip("Start or completely stop the CARLA roof LiDAR sensor.")
        self.lidar_toggle.toggled.connect(self._update_lidar_toggle_text)
        self.lidar_toggle.clicked.connect(lambda _checked=False: self._emit("LIDAR_ENABLED"))
        self.widgets["LIDAR_ENABLED"] = self.lidar_toggle
        self._last_frame_sequence = 0
        self._last_scaled_size = None
        self._last_qimage: Optional[QImage] = None
        self.setWindowTitle("CARLA Runtime Control Panel")
        self.resize(1120, 900)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        root = QWidget()
        root_layout = QVBoxLayout(root)

        self.debug_frame = DebugFrameLabel(self.display_metrics)
        self.debug_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.debug_frame.setMinimumHeight(360)
        self.debug_frame.setStyleSheet("background:#101418; color:#cdd6f4;")
        live_view_row = QHBoxLayout()
        live_view_row.addWidget(self.debug_frame, 2)
        lidar_panel = QWidget()
        lidar_layout = QVBoxLayout(lidar_panel)
        lidar_layout.setContentsMargins(0, 0, 0, 0)
        lidar_controls = QHBoxLayout()
        lidar_controls.addWidget(self.lidar_toggle)
        zoom_in_button = QPushButton("Zoom +")
        zoom_in_button.setToolTip("Zoom in; show a shorter LiDAR distance range")
        zoom_in_button.clicked.connect(self.lidar_radar.zoom_in)
        zoom_out_button = QPushButton("Zoom −")
        zoom_out_button.setToolTip("Zoom out; show a longer LiDAR distance range")
        zoom_out_button.clicked.connect(self.lidar_radar.zoom_out)
        lidar_controls.addWidget(zoom_in_button)
        lidar_controls.addWidget(zoom_out_button)
        lidar_controls.addStretch(1)
        lidar_layout.addLayout(lidar_controls)
        lidar_layout.addWidget(self.lidar_radar, 1)
        live_view_row.addWidget(lidar_panel, 1)
        root_layout.addLayout(live_view_row, 1)

        self.metrics_label = QLabel("display FPS: 0.0 | control FPS: 0.0 | trajectory FPS: 0.0 | vision FPS: 0.0")
        self.metrics_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root_layout.addWidget(self.metrics_label)

        stream_row = QHBoxLayout()
        self.stream_toggle = QPushButton("Flask editor: ON")
        self.stream_toggle.setCheckable(True)
        self.stream_toggle.setChecked(True)
        self.stream_toggle.clicked.connect(self._emit_stream_toggle)
        stream_row.addWidget(self.stream_toggle)
        stream_row.addWidget(QLabel("Use the Flask editor for live settings and ROI editing."))
        stream_row.addStretch(1)
        root_layout.addLayout(stream_row)

        flask_url = self._flask_urls()
        flask_label = QLabel(f"Flask editor URL: {flask_url}")
        flask_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        root_layout.addWidget(flask_label)

        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs)
        self.status = QLabel("Live settings apply immediately.")
        root_layout.addWidget(self.status)
        self.setCentralWidget(root)

        groups = {
            "Driving": {
                "AUTO_MODE_DEFAULT", "FIXED_THROTTLE", "KP", "KI", "KD",
                "STEER_LIMIT", "MAX_STEER_STEP", "TARGET_SPEED_KMH", "CONTROL_LOOP_HZ",
            },
            "Trajectory + junction lead-in": {
                "TRAJECTORY_MODEL_ENABLED", "TRAJECTORY_INFERENCE_INTERVAL_SECONDS", "TRAJECTORY_STEER_GAIN",
                "TRAJECTORY_MAX_STEER", "TRAJECTORY_DEBUG_SCALE",
                "JUNCTION_STATIC_THROTTLE", "JUNCTION_STATIC_TIMEOUT_SECONDS",
                "JUNCTION_ENTRY_DISTANCE_M", "JUNCTION_TRAJECTORY_WINDOW_SECONDS",
            },
            "Junction detection": {
                "JUNCTION_DETECTION_MODE", "INTERSECTION_CHECK_INTERVAL_SECONDS",
                "CROSSWALK_SLEEP", "JUNCTION_ROI_MIN_LINE_LENGTH_RATIO",
                "JUNCTION_ROI_MAX_LINE_ANGLE_DEG", "JUNCTION_ROI_CONFIRM_FRAMES",
                "JUNCTION_ROI_REARM_CLEAR_SECONDS",
            },
            "Vision / debug": {
                "VISION_INFERENCE_INTERVAL_SECONDS", "DEBUG_PANEL_HZ",
                "LANE_PROB_THRESHOLD", "LANE_CENTER_SMOOTH_ALPHA", "DRIVABLE_PROB_THRESHOLD", "LANE_THRESHOLD",
                "RL_TOP_ROI", "RL_BOTTOM_ROI", "RL_LEFT_ROI", "RL_RIGHT_ROI",
                "LL_TOP_ROI", "LL_BOTTOM_ROI", "LL_LEFT_ROI", "LL_RIGHT_ROI",
                "CW_TOP_ROI", "CW_BOTTOM_ROI", "CW_LEFT_ROI", "CW_RIGHT_ROI",
                "CROSSWALK_THRESHOLD", "LANE_CHANGE_DEBOUNCE_SECONDS",
                "LANE_CHANGE_LINE_ANGLE_THRESHOLD_DEG",
                "LANE_CHANGE_PLANNER_CHECK_INTERVAL_SECONDS",
                "VISION_DEBUG", "DEBUG_SHOW_ROIS", "DEBUG_SHOW_LANE_MASK", "DEBUG_SHOW_DRIVABLE_AREA",
                "DEBUG_SHOW_TRAJECTORY", "DEBUG_SHOW_FPS", "DEBUG_SHOW_GPS",
                "DEBUG_SHOW_TEXT",
            },
            "Recovery + obstacle safety": {
                "DRIVABLE_RECOVERY_ENABLED", "DRIVABLE_RECOVERY_ERROR_THRESHOLD",
                "DRIVABLE_RECOVERY_WINDOW_SECONDS", "DRIVABLE_RECOVERY_STEER",
                "DRIVABLE_RECOVERY_THROTTLE", "DRIVABLE_RECOVERY_LIDAR_GATE_ENABLED",
                "DRIVABLE_RECOVERY_LIDAR_MIN_HEIGHT_M", "DRIVABLE_RECOVERY_LIDAR_MAX_HEIGHT_M",
                "DRIVABLE_RECOVERY_LIDAR_MIN_SIDE_DISTANCE_M", "DRIVABLE_RECOVERY_LIDAR_MAX_SIDE_DISTANCE_M",
                "DRIVABLE_RECOVERY_LIDAR_REAR_EXTENT_M", "DRIVABLE_RECOVERY_LIDAR_MAX_FORWARD_DISTANCE_M",
                "DRIVABLE_RECOVERY_LIDAR_MIN_POINTS", "DRIVABLE_RECOVERY_LIDAR_MAX_SCAN_AGE_SECONDS",
            },
            "LiDAR display": {
                "LIDAR_DISPLAY_RANGE_M",
            },
            "Camera": {
                "CAMERA_X", "CAMERA_Y", "CAMERA_Z", "CAMERA_FOV",
            },
            "Hardware (reset)": {
                "CAMERA_IMAGE_WIDTH", "CAMERA_IMAGE_HEIGHT", "CAMERA_SENSOR_TICK", "LIDAR_RANGE_M", "MODEL_PATH",
                "ENABLE_SEMANTIC_CAMERA", "ENABLE_ALT_CAMERA",
            },
        }
        for title, keys in groups.items():
            page = QWidget()
            form = QFormLayout(page)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            for spec in SETTING_SPECS:
                if spec[0] in keys and spec[2] != "sequence":
                    self._add_setting(form, spec)
            if title == "Hardware (reset)":
                form.addRow("ONNX model input", QLabel("512 × 512 (fixed by model)"))
            self.tabs.addTab(self._scroll(page), title)

        self.movement_editor = JunctionSequenceEditor(
            getattr(conf, "JUNCTION_MOVEMENT_SEQUENCES", {})
        )
        self.widgets["JUNCTION_MOVEMENT_SEQUENCES"] = self.movement_editor
        self.movement_editor.changed.connect(self._emit_movement_sequences)
        self.tabs.addTab(self._scroll(self.movement_editor), "Junction movements")

        self._build_profiles(root_layout)
        self._load_values(self.settings.snapshot())

        self._refresh_timer = QTimer(self)
        self._set_refresh_rate(float(getattr(conf, "DEBUG_PANEL_HZ", 15.0)))
        self._refresh_timer.timeout.connect(self._refresh_live_view)
        self._refresh_timer.start()

        self._metrics_timer = QTimer(self)
        self._metrics_timer.setInterval(250)
        self._metrics_timer.timeout.connect(self._refresh_metrics)
        self._metrics_timer.start()

    @staticmethod
    def _flask_urls() -> str:
        host = str(getattr(conf, "STREAM_HOST", "127.0.0.1"))
        port = int(getattr(conf, "STREAM_PORT", 5000))
        if host not in {"0.0.0.0", "::", ""}:
            return f"http://{host}:{port}"

        urls = [f"http://127.0.0.1:{port}"]
        try:
            local_ip = socket.gethostbyname(socket.gethostname())
            if local_ip not in {"127.0.0.1", "0.0.0.0"}:
                urls.append(f"http://{local_ip}:{port}")
        except OSError:
            pass
        return " / ".join(urls)

    def _set_refresh_rate(self, refresh_hz: float) -> None:
        hz = max(5.0, min(60.0, float(refresh_hz)))
        self._refresh_timer.setInterval(max(16, int(round(1000.0 / hz))))

    def set_frame_store(self, store: DebugFrameStore) -> None:
        self.frame_store = store

    def set_metrics_provider(self, provider: Callable[[], dict[str, Any]]) -> None:
        self.metrics_provider = provider

    def update_debug_frame(self, frame, now: float | None = None) -> None:
        """Compatibility API. New code should publish into frame_store."""
        if frame is None:
            return
        if self.frame_store is None:
            self._set_frame(frame)

    def _refresh_live_view(self) -> None:
        if self.frame_store is not None:
            packet = self.frame_store.latest()
            if packet is not None and packet.sequence != self._last_frame_sequence:
                self._last_frame_sequence = packet.sequence
                self._set_frame(packet.image)
        self.lidar_radar.refresh_scan()

    def _set_frame(self, frame) -> None:
        try:
            # CameraManager/renderer use BGR for OpenCV; Qt expects RGB.
            # Make the result contiguous: ``frame[:, :, ::-1]`` is a
            # negative-stride view that QImage cannot reliably consume.
            import numpy as np

            array = np.asarray(frame)
            if array.ndim == 2:
                rgb = np.ascontiguousarray(array)
                h, w = rgb.shape
                image = QImage(
                    rgb.data,
                    w,
                    h,
                    rgb.strides[0],
                    QImage.Format.Format_Grayscale8,
                ).copy()
            else:
                if array.shape[2] == 4:
                    rgb = np.ascontiguousarray(array[:, :, [2, 1, 0]])
                elif array.shape[2] >= 3:
                    rgb = np.ascontiguousarray(array[:, :, :3][:, :, ::-1])
                else:
                    raise ValueError(f"unsupported frame shape: {array.shape}")
                h, w = rgb.shape[:2]
                image = QImage(
                    rgb.data,
                    w,
                    h,
                    rgb.strides[0],
                    QImage.Format.Format_RGB888,
                ).copy()
            self._last_qimage = image
            self._apply_scaled_pixmap(new_frame=True)
        except Exception as exc:
            self.status.setText(f"Debug frame update failed: {exc}")

    def _apply_scaled_pixmap(self, *, new_frame: bool) -> None:
        if self._last_qimage is None:
            return
        size = self.debug_frame.size()
        size_key = (size.width(), size.height())
        if size_key[0] <= 0 or size_key[1] <= 0:
            return
        self._last_scaled_size = size_key
        pixmap = QPixmap.fromImage(self._last_qimage).scaled(
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        if new_frame:
            self.debug_frame.setPixmapForFrame(pixmap)
        else:
            self.debug_frame.setPixmapResized(pixmap)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._last_qimage is not None:
            self._apply_scaled_pixmap(new_frame=False)

    def _refresh_metrics(self) -> None:
        display_fps = self.display_metrics.fps()
        metrics = self.metrics_provider() if self.metrics_provider is not None else {}
        control_fps = float(metrics.get("control_fps", 0.0))
        traj_fps = float(metrics.get("trajectory_fps", 0.0))
        vision_fps = float(metrics.get("vision_fps", 0.0))
        traj_age = metrics.get("trajectory_age_s")
        phase = str(metrics.get("junction_phase", "IDLE"))
        busy_t = bool(metrics.get("trajectory_busy", False))
        busy_v = bool(metrics.get("vision_busy", False))
        trajectory_enabled = bool(metrics.get("trajectory_enabled", True))
        age_text = "—" if traj_age is None else f"{float(traj_age):.2f}s"
        trajectory_text = (
            f"trajectory FPS: {traj_fps:.1f} | trajectory age: {age_text}"
            if trajectory_enabled
            else "trajectory: OFF"
        )
        trajectory_worker_state = (
            "busy" if busy_t else "idle" if trajectory_enabled else "off"
        )
        self.metrics_label.setText(
            f"display FPS: {display_fps:.1f} | control FPS: {control_fps:.1f} | "
            f"{trajectory_text} | vision FPS: {vision_fps:.1f} | junction: {phase} | "
            f"workers: T={trajectory_worker_state} V={'busy' if busy_v else 'idle'}"
        )

    def _scroll(self, widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        return scroll

    def _add_setting(self, form: QFormLayout, spec) -> None:
        key, label, kind, limits, realtime = spec
        if key == "TRAJECTORY_MODEL_ENABLED":
            label = f"{label} [restart required to enable]"
        elif not realtime:
            label = f"{label} [restart required]"
        if kind == "bool":
            widget = QCheckBox()
            widget.stateChanged.connect(lambda _state, k=key: self._emit(k))
        elif kind == "choice":
            widget = QComboBox()
            if key == "JUNCTION_DETECTION_MODE":
                widget.addItem("Junction model (ResNet18)", "model")
                widget.addItem("Crosswalk ROI — horizontal line", "crosswalk_roi")
            else:
                widget.addItems(list(limits))
            widget.currentTextChanged.connect(lambda _text, k=key: self._emit(k))
        elif kind == "int":
            widget = QSpinBox()
            widget.setRange(int(limits[0]), int(limits[1]))
            widget.setSingleStep(int(limits[2]))
            self.widgets[key] = widget
        elif kind == "float":
            widget = QDoubleSpinBox()
            widget.setRange(float(limits[0]), float(limits[1]))
            widget.setSingleStep(float(limits[2]))
            widget.setDecimals(4)
        else:
            widget = QLineEdit()
            widget.editingFinished.connect(lambda k=key: self._emit(k))

        if key not in self.widgets:
            self.widgets[key] = widget
        if kind == "int":
            widget.valueChanged.connect(lambda _value, k=key: self._emit(k))
        elif kind == "float":
            widget.valueChanged.connect(lambda _value, k=key: self._emit(k))
        form.addRow(label, widget)

    def _build_profiles(self, root_layout: QVBoxLayout) -> None:
        box = QGroupBox("Profiles (persist between sessions)")
        layout = QHBoxLayout(box)
        self.profile_combo = QComboBox()
        names = sorted(name for name in self.settings.profiles if not name.startswith("_"))
        self.profile_combo.addItems(names)
        active = self.settings.profiles.get("_active_profile")
        if isinstance(active, str) and active in names:
            self.profile_combo.setCurrentText(active)
        self.profile_name = QLineEdit("default")
        load = QPushButton("Load")
        save = QPushButton("Save")
        load.clicked.connect(self._load_profile)
        save.clicked.connect(self._save_profile)
        layout.addWidget(self.profile_combo, 1)
        layout.addWidget(self.profile_name, 1)
        layout.addWidget(load)
        layout.addWidget(save)
        root_layout.addWidget(box)

    def _value(self, key: str) -> Any:
        widget = self.widgets[key]
        if isinstance(widget, JunctionSequenceEditor):
            return widget.sequences()
        if isinstance(widget, (QCheckBox, QPushButton)):
            return widget.isChecked()
        if isinstance(widget, QComboBox):
            data = widget.currentData()
            return data if data is not None else widget.currentText()
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.value()
        return widget.text()

    def _load_values(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            widget = self.widgets.get(key)
            if widget is None or value is None:
                continue
            if isinstance(widget, JunctionSequenceEditor):
                widget.set_sequences(value)
                continue
            widget.blockSignals(True)
            if isinstance(widget, (QCheckBox, QPushButton)):
                widget.setChecked(bool(value))
                if key == "LIDAR_ENABLED":
                    self._update_lidar_toggle_text(bool(value))
            elif isinstance(widget, QComboBox):
                if widget.currentData() is not None:
                    index = widget.findData(value)
                    if index >= 0:
                        widget.setCurrentIndex(index)
                else:
                    widget.setCurrentText(str(value))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            else:
                widget.setText(str(value))
            widget.blockSignals(False)

    def _emit(self, key: str) -> None:
        value = self._value(key)
        self.settings.apply({key: value})
        self.settings.persist_values({key: value})
        spec = next(s for s in SETTING_SPECS if s[0] == key)
        requires_restart = not spec[4] and not (
            key == "TRAJECTORY_MODEL_ENABLED" and not bool(value)
        )
        if requires_restart:
            self.status.setText(f"{spec[1]} changed — restart the run to apply it.")
            self.reset_requested.emit()
        elif key == "TRAJECTORY_MODEL_ENABLED":
            self.status.setText("Trajectory model disabled live; releasing its resources.")
        else:
            self.status.setText(f"{spec[1]} updated live.")
        if key == "DEBUG_PANEL_HZ":
            self._set_refresh_rate(float(value))
        self.settings_changed.emit({key: value})

    def _update_lidar_toggle_text(self, enabled: bool) -> None:
        self.lidar_toggle.setText("LiDAR: ON" if enabled else "LiDAR: OFF")

    def set_lidar_enabled(self, enabled: bool, status: str | None = None) -> None:
        """Synchronize the live LiDAR button without emitting another setting change."""
        self.lidar_toggle.blockSignals(True)
        self.lidar_toggle.setChecked(bool(enabled))
        self.lidar_toggle.blockSignals(False)
        self._update_lidar_toggle_text(bool(enabled))
        if status:
            self.status.setText(status)

    def _emit_stream_toggle(self, enabled: bool) -> None:
        self.stream_toggle.setText("Flask editor: ON" if enabled else "Flask editor: OFF")
        self.stream_toggle_requested.emit(bool(enabled))

    def _emit_movement_sequences(self, value: dict[str, Any]) -> None:
        self.settings.apply({"JUNCTION_MOVEMENT_SEQUENCES": value})
        self.settings.persist_values({"JUNCTION_MOVEMENT_SEQUENCES": value})
        self.status.setText("Junction movement sequence updated live.")
        self.settings_changed.emit({"JUNCTION_MOVEMENT_SEQUENCES": value})

    def sync_external_settings(self, values: dict[str, Any]) -> None:
        """Thread-safe entry point for settings changed by the Flask editor."""
        self.external_settings_changed.emit(dict(values))

    def _apply_external_settings(self, values: dict[str, Any]) -> None:
        self.settings.apply(values)
        self._load_values(values)
        if "DEBUG_PANEL_HZ" in values:
            self._set_refresh_rate(float(values["DEBUG_PANEL_HZ"]))
        changed = ", ".join(values.keys())
        self.status.setText(f"Flask settings synchronized: {changed}")

    def _load_profile(self) -> None:
        values = self.settings.profiles.get(self.profile_combo.currentText())
        if values:
            requires_restart = any(
                key in values and getattr(conf, key, None) != values[key]
                for key, _label, _kind, _limits, realtime in SETTING_SPECS
                if not realtime
                and not (
                    key == "TRAJECTORY_MODEL_ENABLED"
                    and values.get(key) is False
                )
            )
            self.settings.apply(values)
            self._load_values(values)
            self.settings_changed.emit(values)
            if requires_restart:
                self.status.setText(
                    f"Loaded profile: {self.profile_combo.currentText()} — restart required."
                )
                self.reset_requested.emit()
            else:
                self.status.setText(f"Loaded profile: {self.profile_combo.currentText()}")

    def _save_profile(self) -> None:
        name = self.profile_name.text().strip() or "default"
        values = {key: self._value(key) for key in self.widgets}
        self.settings.save_profile(name, values)
        self.settings.profiles["_active_profile"] = name
        self.settings.save()
        if self.profile_combo.findText(name) < 0:
            self.profile_combo.addItem(name)
        self.profile_combo.setCurrentText(name)
        self.status.setText(f"Saved profile: {name}")
