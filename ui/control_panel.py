from __future__ import annotations

import socket
from typing import Any, Callable, Optional

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QPaintEvent
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QFrame, QGroupBox, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QPushButton, QScrollArea, QSpinBox, QSplitter,
    QTabWidget, QVBoxLayout, QWidget,
)

import config_city as conf

from runtime_metrics import DisplayMetrics
from runtime_settings import SETTING_SPECS, RuntimeSettings
from ui.frame_store import DebugFrameStore
from ui.lidar_radar import LidarRadarWidget


class NoWheelSpinBox(QSpinBox):
    """A numeric field that never changes value when the mouse wheel scrolls."""

    def wheelEvent(self, event) -> None:
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """A floating-point field that never changes value on mouse-wheel scroll."""

    def wheelEvent(self, event) -> None:
        event.ignore()


class NoWheelComboBox(QComboBox):
    """A dropdown that does not change selection while the user scrolls."""

    def wheelEvent(self, event) -> None:
        event.ignore()


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
                distance = NoWheelDoubleSpinBox()
                distance.setRange(0.01, 1000.0)
                distance.setSingleStep(0.5)
                distance.setDecimals(2)
                distance.setValue(action["distance_m"])
                steering = NoWheelDoubleSpinBox()
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
        self.resize(1500, 1000)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        root = QWidget()
        root.setObjectName("runtimeRoot")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(18, 16, 18, 14)
        root_layout.setSpacing(12)

        header = QFrame()
        header.setObjectName("topBar")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 14, 18, 14)
        header_layout.setSpacing(14)

        brand_column = QVBoxLayout()
        brand_column.setSpacing(3)
        eyebrow = QLabel("CARLA AUTONOMOUS SYSTEMS")
        eyebrow.setObjectName("eyebrowLabel")
        title = QLabel("Runtime Command Center")
        title.setObjectName("pageTitle")
        subtitle = QLabel("Live perception, vehicle telemetry and runtime configuration")
        subtitle.setObjectName("mutedLabel")
        brand_column.addWidget(eyebrow)
        brand_column.addWidget(title)
        brand_column.addWidget(subtitle)
        header_layout.addLayout(brand_column, 1)

        header_actions = QVBoxLayout()
        header_actions.setSpacing(5)
        self.stream_toggle = QPushButton("ROI Studio · OFF")
        self.stream_toggle.setObjectName("primaryButton")
        self.stream_toggle.setCheckable(True)
        self.stream_toggle.setChecked(False)
        self.stream_toggle.setToolTip("Start or stop the optional Flask ROI and advanced-settings editor.")
        self.stream_toggle.clicked.connect(self._emit_stream_toggle)
        header_actions.addWidget(self.stream_toggle)
        flask_label = QLabel(f"Local editor  ·  {self._flask_urls()}")
        flask_label.setObjectName("mutedLabel")
        flask_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        flask_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header_actions.addWidget(flask_label)
        header_layout.addLayout(header_actions)
        root_layout.addWidget(header)

        self.workspace_splitter = QSplitter(Qt.Orientation.Vertical)
        self.workspace_splitter.setObjectName("workspaceSplitter")
        self.workspace_splitter.setChildrenCollapsible(False)
        self.workspace_splitter.setHandleWidth(7)

        monitor = QFrame()
        monitor.setObjectName("monitorSurface")
        monitor_layout = QVBoxLayout(monitor)
        monitor_layout.setContentsMargins(14, 14, 14, 12)
        monitor_layout.setSpacing(10)

        live_view_row = QSplitter(Qt.Orientation.Horizontal)
        live_view_row.setObjectName("liveViewSplitter")
        live_view_row.setChildrenCollapsible(False)
        live_view_row.setHandleWidth(7)

        camera_card = QFrame()
        camera_card.setObjectName("surfaceCard")
        camera_card.setMinimumWidth(470)
        camera_layout = QVBoxLayout(camera_card)
        camera_layout.setContentsMargins(12, 10, 12, 12)
        camera_layout.setSpacing(8)
        camera_header = QHBoxLayout()
        camera_title_column = QVBoxLayout()
        camera_title_column.setSpacing(2)
        camera_title = QLabel("Perception camera")
        camera_title.setObjectName("sectionTitle")
        camera_subtitle = QLabel("Live image · visual overlays only")
        camera_subtitle.setObjectName("mutedLabel")
        camera_title_column.addWidget(camera_title)
        camera_title_column.addWidget(camera_subtitle)
        camera_header.addLayout(camera_title_column, 1)
        self.camera_dimensions_label = QLabel("WAITING FOR FRAME")
        self.camera_dimensions_label.setObjectName("statusPill")
        camera_header.addWidget(self.camera_dimensions_label)
        camera_layout.addLayout(camera_header)

        self.debug_frame = DebugFrameLabel(self.display_metrics)
        self.debug_frame.setObjectName("cameraViewport")
        self.debug_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.debug_frame.setMinimumSize(360, 220)
        self.debug_frame.setText("Waiting for camera frames…")
        camera_layout.addWidget(self.debug_frame, 1)
        live_view_row.addWidget(camera_card)

        lidar_card = QFrame()
        lidar_card.setObjectName("surfaceCard")
        lidar_card.setMinimumWidth(245)
        lidar_layout = QVBoxLayout(lidar_card)
        lidar_layout.setContentsMargins(12, 10, 12, 12)
        lidar_layout.setSpacing(8)
        lidar_header = QHBoxLayout()
        lidar_title_column = QVBoxLayout()
        lidar_title_column.setSpacing(2)
        lidar_title = QLabel("LiDAR radar")
        lidar_title.setObjectName("sectionTitle")
        lidar_subtitle = QLabel("Top-down obstacle point cloud")
        lidar_subtitle.setObjectName("mutedLabel")
        lidar_title_column.addWidget(lidar_title)
        lidar_title_column.addWidget(lidar_subtitle)
        lidar_header.addLayout(lidar_title_column, 1)
        self.lidar_toggle.setObjectName("primaryButton")
        lidar_header.addWidget(self.lidar_toggle)
        lidar_layout.addLayout(lidar_header)

        lidar_controls = QHBoxLayout()
        lidar_controls.setSpacing(6)
        zoom_in_button = QPushButton("＋  Zoom in")
        zoom_in_button.setObjectName("quietButton")
        zoom_in_button.setToolTip("Show a shorter LiDAR distance range")
        zoom_in_button.clicked.connect(self.lidar_radar.zoom_in)
        zoom_out_button = QPushButton("－  Zoom out")
        zoom_out_button.setObjectName("quietButton")
        zoom_out_button.setToolTip("Show a longer LiDAR distance range")
        zoom_out_button.clicked.connect(self.lidar_radar.zoom_out)
        lidar_controls.addWidget(zoom_in_button)
        lidar_controls.addWidget(zoom_out_button)
        lidar_controls.addStretch(1)
        lidar_layout.addLayout(lidar_controls)
        lidar_layout.addWidget(self.lidar_radar, 1)
        live_view_row.addWidget(lidar_card)

        live_info_card = QFrame()
        live_info_card.setObjectName("surfaceCard")
        live_info_card.setMinimumWidth(285)
        live_info_card.setMaximumWidth(390)
        live_info_layout = QVBoxLayout(live_info_card)
        live_info_layout.setContentsMargins(12, 10, 12, 10)
        live_info_layout.setSpacing(7)
        live_info_title = QLabel("Live intelligence")
        live_info_title.setObjectName("sectionTitle")
        live_info_subtitle = QLabel("Continuously updated · no image text")
        live_info_subtitle.setObjectName("mutedLabel")
        live_info_layout.addWidget(live_info_title)
        live_info_layout.addWidget(live_info_subtitle)

        info_scroll = QScrollArea()
        info_scroll.setWidgetResizable(True)
        info_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        info_content = QWidget()
        info_content_layout = QVBoxLayout(info_content)
        info_content_layout.setContentsMargins(2, 4, 2, 4)
        info_content_layout.setSpacing(8)
        self.live_values: dict[str, QLabel] = {}

        road_group = QGroupBox("Road span · DW ROI")
        road_group_layout = QVBoxLayout(road_group)
        road_group_layout.setContentsMargins(10, 10, 10, 9)
        road_value = QLabel("— px")
        road_value.setObjectName("roadSpanValue")
        road_value.setToolTip("Median horizontal drivable-mask span across valid rows in the DW ROI. This is a comparison index, not metres.")
        road_meta = QLabel("Waiting for a valid drivable-area mask")
        road_meta.setObjectName("telemetryHint")
        road_meta.setWordWrap(True)
        road_group_layout.addWidget(road_value)
        road_group_layout.addWidget(road_meta)
        self.live_values["road_width"] = road_value
        self.live_values["road_width_meta"] = road_meta

        def add_live_group(title: str, fields: tuple[tuple[str, str, str], ...]) -> None:
            group = QGroupBox(title)
            form = QFormLayout(group)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            form.setHorizontalSpacing(10)
            form.setVerticalSpacing(6)
            for key, label_text, initial in fields:
                label = QLabel(label_text)
                label.setObjectName("telemetryLabel")
                value = QLabel(initial)
                value.setObjectName("telemetryValue")
                value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                value.setWordWrap(True)
                self.live_values[key] = value
                form.addRow(label, value)
            info_content_layout.addWidget(group)

        info_content_layout.addWidget(road_group)
        add_live_group("ROI measurement", (
            ("road_ratio", "Frame ratio", "—"),
            ("road_rows", "Valid scan rows", "0"),
            ("road_roi", "ROI bounds (px)", "—"),
        ))
        add_live_group("Perception", (
            ("lane_type", "Detected lane", "—"),
            ("lane_error", "Lateral error", "—"),
            ("lane_center", "Lane centre", "—"),
            ("lane_bounds", "Lane boundaries", "—"),
        ))
        add_live_group("Vehicle / navigation", (
            ("mode", "Driving mode", "—"),
            ("junction", "Junction phase", "IDLE"),
            ("lane_change", "Lane change", "IDLE"),
            ("gps", "GPS · x / y / z", "—"),
            ("trajectory", "Trajectory worker", "OFF"),
        ))
        info_content_layout.addStretch(1)
        info_scroll.setWidget(info_content)
        live_info_layout.addWidget(info_scroll, 1)
        live_view_row.addWidget(live_info_card)

        live_view_row.setStretchFactor(0, 6)
        live_view_row.setStretchFactor(1, 3)
        live_view_row.setStretchFactor(2, 3)
        live_view_row.setSizes([780, 330, 350])
        monitor_layout.addWidget(live_view_row, 1)

        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(8)
        self.metric_values: dict[str, QLabel] = {}
        metric_specs = (
            ("DISPLAY FPS", "display", "0.0"),
            ("CONTROL FPS", "control", "0.0"),
            ("VISION FPS", "vision", "0.0"),
            ("TRAJECTORY FPS", "trajectory", "0.0"),
        )
        for caption, key, initial in metric_specs:
            card = QFrame()
            card.setObjectName("metricCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(11, 7, 11, 7)
            card_layout.setSpacing(2)
            caption_label = QLabel(caption)
            caption_label.setObjectName("metricCaption")
            value_label = QLabel(initial)
            value_label.setObjectName("metricValue")
            value_label.setMinimumWidth(0)
            card_layout.addWidget(caption_label)
            card_layout.addWidget(value_label)
            metrics_row.addWidget(card, 1)
            self.metric_values[key] = value_label
        monitor_layout.addLayout(metrics_row)

        settings_surface = QFrame()
        settings_surface.setObjectName("settingsSurface")
        settings_layout = QVBoxLayout(settings_surface)
        settings_layout.setContentsMargins(14, 12, 14, 12)
        settings_layout.setSpacing(9)

        settings_header = QHBoxLayout()
        settings_heading = QVBoxLayout()
        settings_heading.setSpacing(2)
        settings_title = QLabel("Runtime configuration")
        settings_title.setObjectName("sectionTitle")
        settings_subtitle = QLabel("Changes are applied live unless a setting is marked as restart-required.")
        settings_subtitle.setObjectName("mutedLabel")
        settings_heading.addWidget(settings_title)
        settings_heading.addWidget(settings_subtitle)
        settings_header.addLayout(settings_heading, 1)
        self.settings_search = QLineEdit()
        self.settings_search.setObjectName("settingsSearch")
        self.settings_search.setPlaceholderText("Filter settings by name or key…")
        self.settings_search.setClearButtonEnabled(True)
        self.settings_search.setMaximumWidth(310)
        self.settings_search.setToolTip("Search setting labels and configuration keys across all categories.")
        settings_header.addWidget(self.settings_search)
        settings_layout.addLayout(settings_header)

        profile_surface = QFrame()
        profile_surface.setObjectName("profileSurface")
        profile_layout = QVBoxLayout(profile_surface)
        profile_layout.setContentsMargins(9, 7, 9, 7)
        profile_layout.setSpacing(0)
        self._build_profiles(profile_layout)
        settings_layout.addWidget(profile_surface)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("settingsTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(False)
        self.settings_search.textChanged.connect(self._filter_settings)
        settings_layout.addWidget(self.tabs, 1)

        self.workspace_splitter.addWidget(monitor)
        self.workspace_splitter.addWidget(settings_surface)
        self.workspace_splitter.setStretchFactor(0, 4)
        self.workspace_splitter.setStretchFactor(1, 6)
        self.workspace_splitter.setSizes([420, 650])
        root_layout.addWidget(self.workspace_splitter, 1)

        self.status = QLabel("READY  ·  Live settings apply immediately.")
        self.status.setObjectName("footerStatus")
        self.status.setMinimumHeight(28)
        root_layout.addWidget(self.status)
        self.setCentralWidget(root)

        self.setWindowTitle("CARLA Runtime Command Center")
        self.setMinimumSize(1180, 860)
        self.resize(1660, 1120)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setStyleSheet("""
            QMainWindow, QWidget#runtimeRoot {
                background: #0a1220;
                color: #e7eef8;
                font-family: "Segoe UI", "Inter", sans-serif;
                font-size: 10pt;
            }
            QFrame#topBar, QFrame#monitorSurface, QFrame#settingsSurface {
                background: #101b2c;
                border: 1px solid #23354c;
                border-radius: 14px;
            }
            QFrame#surfaceCard {
                background: #0c1727;
                border: 1px solid #263a53;
                border-radius: 12px;
            }
            QFrame#profileSurface {
                background: #0c1727;
                border: 1px solid #263a53;
                border-radius: 10px;
            }
            QLabel#eyebrowLabel {
                color: #65c9e8;
                font-size: 8pt;
                font-weight: 700;
                letter-spacing: 1px;
            }
            QLabel#pageTitle {
                color: #f1f6fd;
                font-size: 19pt;
                font-weight: 700;
            }
            QLabel#sectionTitle {
                color: #edf4fc;
                font-size: 11pt;
                font-weight: 700;
            }
            QLabel#mutedLabel, QLabel#metricCaption {
                color: #8fa3ba;
                font-size: 8pt;
            }
            QLabel#statusPill {
                color: #8fe2f3;
                background: #123246;
                border: 1px solid #23536a;
                border-radius: 8px;
                padding: 5px 9px;
                font-size: 8pt;
                font-weight: 700;
            }
            QLabel#metricCaption {
                font-weight: 700;
                letter-spacing: 0.4px;
            }
            QLabel#metricValue {
                color: #e9f6ff;
                font-size: 14pt;
                font-weight: 700;
            }
            QFrame#metricCard {
                background: #0b1727;
                border: 1px solid #23384f;
                border-radius: 10px;
            }
            QLabel#diagnosticsLine {
                color: #7f94aa;
                font-size: 8pt;
                padding: 2px 4px;
            }
            QLabel#cameraViewport {
                background: #050b13;
                color: #6f879f;
                border: 1px solid #22354c;
                border-radius: 9px;
                font-size: 10pt;
            }
            QLabel#footerStatus {
                color: #9db2c8;
                background: #101b2c;
                border: 1px solid #23354c;
                border-radius: 8px;
                padding: 5px 10px;
                font-size: 8pt;
            }
            QLineEdit#settingsSearch {
                min-height: 28px;
                max-width: 310px;
                background: #091524;
                border: 1px solid #30465e;
                border-radius: 8px;
                padding: 5px 10px;
                color: #e5edf7;
            }
            QTabWidget#settingsTabs::pane {
                border: 1px solid #263a53;
                border-radius: 9px;
                background: #0c1727;
                top: -1px;
            }
            QTabWidget#settingsTabs QTabBar::tab {
                background: #142239;
                color: #9db2c8;
                border: 1px solid #263a53;
                padding: 10px 15px;
                margin-right: 5px;
                font-weight: 650;
                border-top-left-radius: 7px;
                border-top-right-radius: 7px;
                min-height: 22px;
            }
            QTabWidget#settingsTabs QTabBar::tab:selected {
                background: #1a3651;
                color: #f0f8ff;
                border-bottom-color: #65c9e8;
            }
            QTabWidget#settingsTabs QTabBar::tab:hover:!selected {
                background: #1b2c43;
            }
            QGroupBox {
                color: #d8e7f7;
                font-weight: 700;
                border: 1px solid #2a4058;
                border-radius: 9px;
                margin-top: 10px;
                padding: 10px 8px 8px 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #83cfe5;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background: #091321;
                color: #e5edf7;
                border: 1px solid #30455e;
                border-radius: 7px;
                padding: 7px 10px;
                min-height: 32px;
                min-width: 150px;
                selection-background-color: #1f6680;
            }
            QLabel#telemetryLabel {
                color: #8fa3ba;
                font-size: 8pt;
            }
            QLabel#telemetryValue {
                color: #e5eef8;
                font-size: 9pt;
                font-weight: 650;
            }
            QLabel#telemetryHint {
                color: #8fa3ba;
                font-size: 8pt;
            }
            QLabel#roadSpanValue {
                color: #79d9ef;
                font-size: 25pt;
                font-weight: 800;
                padding: 2px 0;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #55b8d4;
            }
            QComboBox QAbstractItemView {
                background: #102036;
                color: #e5edf7;
                selection-background-color: #1a526b;
                border: 1px solid #30455e;
            }
            QCheckBox {
                spacing: 8px;
                color: #dce8f5;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1px solid #536d88;
                background: #0a1524;
            }
            QCheckBox::indicator:checked {
                background: #24a6c4;
                border-color: #54d4e8;
            }
            QPushButton {
                color: #dceaf6;
                background: #17283d;
                border: 1px solid #30465f;
                border-radius: 7px;
                padding: 7px 11px;
                min-height: 25px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #203b56;
                border-color: #47728d;
            }
            QPushButton:pressed {
                background: #102238;
            }
            QPushButton#primaryButton {
                color: #eafaff;
                background: #14627b;
                border: 1px solid #268ba7;
            }
            QPushButton#primaryButton:hover {
                background: #197b98;
            }
            QPushButton#quietButton {
                background: #122238;
                border-color: #2a4058;
                color: #adc1d4;
            }
            QScrollArea, QScrollArea > QWidget > QWidget {
                background: transparent;
                border: none;
            }
            QSplitter::handle {
                background: #263950;
                border-radius: 3px;
            }
            QScrollBar:vertical {
                background: #0b1726;
                width: 10px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background: #344b64;
                border-radius: 4px;
                min-height: 28px;
            }
        """)

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
            "Lane change": {
                "LANE_CHANGE_DEBOUNCE_SECONDS",
                "LANE_CHANGE_PLANNER_CHECK_INTERVAL_SECONDS",
                "LANE_CHANGE_VERIFY_DISTANCE_M", "LANE_CHANGE_STEER_DISTANCE_M",
                "LANE_CHANGE_STEER_VALUE",
                "POST_JUNCTION_LANE_CHANGE_DELAY_SECONDS",
            },
            "Vision / debug": {
                "VISION_INFERENCE_INTERVAL_SECONDS", "DEBUG_PANEL_HZ",
                "LANE_PROB_THRESHOLD", "LANE_CENTER_SMOOTH_ALPHA", "DRIVABLE_PROB_THRESHOLD", "LANE_THRESHOLD",
                "RL_TOP_ROI", "RL_BOTTOM_ROI", "RL_LEFT_ROI", "RL_RIGHT_ROI",
                "LL_TOP_ROI", "LL_BOTTOM_ROI", "LL_LEFT_ROI", "LL_RIGHT_ROI",
                "CW_TOP_ROI", "CW_BOTTOM_ROI", "CW_LEFT_ROI", "CW_RIGHT_ROI",
                "DW_TOP_ROI", "DW_BOTTOM_ROI", "DW_LEFT_ROI", "DW_RIGHT_ROI",
                "CROSSWALK_THRESHOLD",
                "VISION_DEBUG", "DEBUG_SHOW_ROIS", "DEBUG_SHOW_LANE_MASK", "DEBUG_SHOW_DRIVABLE_AREA",
                "DEBUG_SHOW_TRAJECTORY",
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
            if title == "Lane change":
                handoff_help = QLabel(
                    "Lane-follow steering stays locked during the maneuver. "
                    "GPS decides when the requested lane change is complete. "
                    "The controller keeps steering ownership until GPS no longer "
                    "requests a lane change."
                )
                handoff_help.setWordWrap(True)
                form.addRow("Steering ownership", handoff_help)
            if title == "Hardware (reset)":
                form.addRow("ONNX model input", QLabel("512 × 512 (fixed by model)"))
            self.tabs.addTab(self._scroll(page), title)

        self.movement_editor = JunctionSequenceEditor(
            getattr(conf, "JUNCTION_MOVEMENT_SEQUENCES", {})
        )
        self.widgets["JUNCTION_MOVEMENT_SEQUENCES"] = self.movement_editor
        self.movement_editor.changed.connect(self._emit_movement_sequences)
        self.tabs.addTab(self._scroll(self.movement_editor), "Junction movements")

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
            if hasattr(self, "camera_dimensions_label"):
                self.camera_dimensions_label.setText(f"{w} × {h}")
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
        trajectory_enabled = bool(metrics.get("trajectory_enabled", True))

        if hasattr(self, "metric_values"):
            self.metric_values["display"].setText(f"{display_fps:.1f}")
            self.metric_values["control"].setText(f"{control_fps:.1f}")
            self.metric_values["vision"].setText(f"{vision_fps:.1f}")
            self.metric_values["trajectory"].setText(
                f"{traj_fps:.1f}" if trajectory_enabled else "OFF"
            )
        self._update_live_telemetry(metrics)

    def _update_live_telemetry(self, metrics: dict[str, Any]) -> None:
        """Refresh the persistent side panel from the latest control-loop snapshot."""
        values = self.live_values

        width_px = metrics.get("road_width_index_px")
        ratio = metrics.get("road_width_ratio")
        valid_rows = int(metrics.get("road_valid_rows") or 0)
        if width_px is None:
            values["road_width"].setText("— px")
            values["road_width_meta"].setText("No valid drivable pixels inside the DW ROI")
        else:
            values["road_width"].setText(f"{float(width_px):.0f} px")
            ratio_text = "—" if ratio is None else f"{float(ratio) * 100.0:.1f}% of frame width"
            values["road_width_meta"].setText(
                f"Median row span · {ratio_text} · {valid_rows} valid rows · comparison index, not metres"
            )

        roi = metrics.get("road_roi")
        if isinstance(roi, (tuple, list)) and len(roi) == 4:
            values["road_roi"].setText(
                f"x {int(roi[0])}–{int(roi[2])}  ·  y {int(roi[1])}–{int(roi[3])}"
            )
        else:
            values["road_roi"].setText("—")
        values["road_ratio"].setText(
            "—" if ratio is None else f"{float(ratio) * 100.0:.1f}%"
        )
        values["road_rows"].setText(str(valid_rows))

        lane_type = str(metrics.get("lane_type") or "—")
        values["lane_type"].setText(lane_type.replace("_", " ").upper())
        lane_error = metrics.get("lane_error_px")
        values["lane_error"].setText(
            "—" if lane_error is None else f"{float(lane_error):+.1f} px"
        )
        lane_center = metrics.get("lane_center_x")
        values["lane_center"].setText(
            "—" if lane_center is None else f"{float(lane_center):.1f} px"
        )
        left_x = metrics.get("lane_left_x")
        right_x = metrics.get("lane_right_x")
        values["lane_bounds"].setText(
            f"L {left_x if left_x is not None else '—'}  ·  R {right_x if right_x is not None else '—'}"
        )

        values["mode"].setText(str(metrics.get("driving_mode", "—")).upper())
        values["junction"].setText(
            str(metrics.get("junction_phase", "IDLE")).replace("_", " ")
        )
        lane_phase = str(metrics.get("lane_change_phase", "IDLE")).replace("_", " ")
        direction = metrics.get("lane_change_direction")
        if lane_phase != "IDLE" and direction:
            lane_phase = f"{lane_phase} · {str(direction).upper()}"
        values["lane_change"].setText(lane_phase)

        gps = metrics.get("gps")
        if isinstance(gps, (tuple, list)) and len(gps) >= 3:
            values["gps"].setText(
                f"{float(gps[0]):.1f}, {float(gps[1]):.1f}, {float(gps[2]):.1f}"
            )
        else:
            values["gps"].setText("Unavailable")

        if not bool(metrics.get("trajectory_enabled", True)):
            trajectory_text = "OFF"
        else:
            age = metrics.get("trajectory_age_s")
            trajectory_text = "BUSY" if metrics.get("trajectory_busy") else "IDLE"
            if age is not None:
                trajectory_text += f" · {float(age):.2f}s old"
        values["trajectory"].setText(trajectory_text)

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
            widget = NoWheelComboBox()
            if key == "JUNCTION_DETECTION_MODE":
                widget.addItem("Junction model (ResNet18)", "model")
                widget.addItem("Crosswalk ROI — horizontal line", "crosswalk_roi")
            else:
                widget.addItems(list(limits))
            widget.currentTextChanged.connect(lambda _text, k=key: self._emit(k))
        elif kind == "int":
            widget = NoWheelSpinBox()
            widget.setRange(int(limits[0]), int(limits[1]))
            widget.setSingleStep(int(limits[2]))
            self.widgets[key] = widget
        elif kind == "float":
            widget = NoWheelDoubleSpinBox()
            widget.setRange(float(limits[0]), float(limits[1]))
            widget.setSingleStep(float(limits[2]))
            widget.setDecimals(4)
        else:
            widget = QLineEdit()
            widget.editingFinished.connect(lambda k=key: self._emit(k))

        if key not in self.widgets:
            self.widgets[key] = widget
        widget.setProperty("settingKey", key)
        label_widget = QLabel(label)
        label_widget.setProperty("settingKey", key)
        if kind == "int":
            widget.valueChanged.connect(lambda _value, k=key: self._emit(k))
        elif kind == "float":
            widget.valueChanged.connect(lambda _value, k=key: self._emit(k))
        form.addRow(label_widget, widget)

    def _filter_settings(self, text: str) -> None:
        """Filter setting rows by their visible label or configuration key."""
        query = text.strip().casefold()
        for index in range(self.tabs.count()):
            scroll = self.tabs.widget(index)
            page = scroll.widget() if isinstance(scroll, QScrollArea) else scroll
            form = page.layout()
            if not isinstance(form, QFormLayout):
                continue
            for row in range(form.rowCount()):
                label_item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
                field_item = form.itemAt(row, QFormLayout.ItemRole.FieldRole)
                field_widget = field_item.widget() if field_item is not None else None
                if field_widget is None or not field_widget.property("settingKey"):
                    continue
                label_widget = label_item.widget() if label_item is not None else None
                label_text = label_widget.text() if isinstance(label_widget, QLabel) else ""
                key = str(field_widget.property("settingKey"))
                visible = not query or query in f"{key} {label_text}".casefold()
                if label_widget is not None:
                    label_widget.setVisible(visible)
                field_widget.setVisible(visible)

    def _build_profiles(self, root_layout: QVBoxLayout) -> None:
        box = QGroupBox("Profiles (persist between sessions)")
        layout = QHBoxLayout(box)
        self.profile_combo = NoWheelComboBox()
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
        self.stream_toggle.setText("ROI Studio · ON" if enabled else "ROI Studio · OFF")
        self.stream_toggle_requested.emit(bool(enabled))

    def set_flask_stream_state(self, enabled: bool, status: str | None = None) -> None:
        """Synchronize the Flask server toggle after a start/stop request."""
        self.stream_toggle.blockSignals(True)
        self.stream_toggle.setChecked(bool(enabled))
        self.stream_toggle.setText(
            "ROI Studio · ON" if enabled else "ROI Studio · OFF"
        )
        self.stream_toggle.blockSignals(False)
        if status:
            self.status.setText(status)

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
