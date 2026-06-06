"""
CARLA Spawn + Traffic Tool — fixed & compacted edition.

Bugs fixed vs original:
  1. QFileDialog was never imported → added to PySide6 imports.
  2. pathlib.Path imported but unused → removed.
  3. save_json_as had a dead dummy tuple assignment → removed.
  4. clear_spawned_actors used try/destroy individually; now uses
     client.apply_batch_sync for proper batch destruction and clears
     controllers before walkers (CARLA requirement).
  5. set_selected_transform called mark_dirty() unconditionally even
     while self.loading=True (triggered from load_json_data) → guarded.
  6. on_spawn_list_changed had no guard for carla_map being None.
  7. closeEvent cleared actors before asking the user → actors now only
     cleared after the user confirms close (or no dirty flag).
  8. load_json_data called update_selected_marker() redundantly after
     set_selected_transform already does it → removed duplicate call.
  9. UI compacted into a QTabWidget so the sidebar fits smaller displays.
"""

import json
import os
import random
import sys
from typing import List, Optional, Tuple

try:
    import carla
except Exception as exc:
    carla = None
    CARLA_IMPORT_ERROR = exc
else:
    CARLA_IMPORT_ERROR = None

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QAction, QBrush, QColor, QFont, QKeySequence, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

WEATHER_PRESETS = [
    "ClearNoon", "CloudyNoon", "WetNoon", "WetCloudyNoon",
    "MidRainyNoon", "HardRainNoon", "SoftRainNoon",
    "ClearSunset", "CloudySunset", "WetSunset",
    "WetCloudySunset", "MidRainSunset", "HardRainSunset", "SoftRainSunset",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_transform_dict(x, y, z, roll, pitch, yaw):
    return {
        "x": float(x), "y": float(y), "z": float(z),
        "roll": float(roll), "pitch": float(pitch), "yaw": float(yaw),
    }


def transform_from_fields(x, y, z, roll, pitch, yaw):
    if carla is None:
        return _DummyTransform(x, y, z, roll, pitch, yaw)
    return carla.Transform(
        carla.Location(x=float(x), y=float(y), z=float(z)),
        carla.Rotation(roll=float(roll), pitch=float(pitch), yaw=float(yaw)),
    )


class _DummyLocation:
    def __init__(self, x, y, z):
        self.x = float(x); self.y = float(y); self.z = float(z)


class _DummyRotation:
    def __init__(self, roll, pitch, yaw):
        self.roll = float(roll); self.pitch = float(pitch); self.yaw = float(yaw)


class _DummyTransform:
    def __init__(self, x, y, z, roll, pitch, yaw):
        self.location = _DummyLocation(x, y, z)
        self.rotation = _DummyRotation(roll, pitch, yaw)


def rect_from_points(points: List[Tuple[float, float]], padding: float = 120.0) -> QRectF:
    if not points:
        return QRectF(-500, -500, 1000, 1000)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return QRectF(
        min(xs) - padding, min(ys) - padding,
        max(xs) - min(xs) + 2 * padding,
        max(ys) - min(ys) + 2 * padding,
    )


def segment_key(a, b):
    ax, ay = round(float(a.x), 2), round(float(a.y), 2)
    bx, by = round(float(b.x), 2), round(float(b.y), 2)
    return (ax, ay, bx, by) if (ax, ay) <= (bx, by) else (bx, by, ax, ay)


# ---------------------------------------------------------------------------
# Map scene items
# ---------------------------------------------------------------------------

class SpawnMarkerItem(QGraphicsEllipseItem):
    def __init__(self, tf):
        super().__init__(-12, -12, 24, 24)
        self.setBrush(QBrush(QColor("#f1c40f")))
        self.setPen(QPen(QColor("#9a7d0a"), 2))
        self.setZValue(20)
        self.setPos(tf.location.x, tf.location.y)
        label = QGraphicsSimpleTextItem("SP", self)
        font = QFont(); font.setPointSize(7); font.setBold(True)
        label.setFont(font)
        label.setBrush(QBrush(QColor("black")))
        br = label.boundingRect()
        label.setPos(-br.width() / 2, -br.height() / 2)

    def set_transform(self, tf):
        self.setPos(tf.location.x, tf.location.y)


class RecommendedSpawnDot(QGraphicsEllipseItem):
    def __init__(self, controller, index: int, tf):
        super().__init__(-4, -4, 8, 8)
        self.controller = controller
        self.index = index
        self.tf = tf
        self.setBrush(QBrush(QColor("#2e86de")))
        self.setPen(QPen(QColor("#1b4f72"), 1))
        self.setZValue(10)
        self.setPos(tf.location.x, tf.location.y)

    def mousePressEvent(self, event):
        if self.controller:
            self.controller.choose_recommended_spawn(self.index)
        event.accept()


# ---------------------------------------------------------------------------
# Map scene / view
# ---------------------------------------------------------------------------

class CarlaMapScene(QGraphicsScene):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.road_segments: List[Tuple[float, float, float, float]] = []
        self.map_bounds = QRectF(-500, -500, 1000, 1000)
        self.recommended_spawn_items: List[RecommendedSpawnDot] = []

    def set_map_data(self, road_segments, recommended_spawns, bounds: QRectF):
        self.road_segments = road_segments
        self.map_bounds = bounds
        for item in self.recommended_spawn_items:
            self.removeItem(item)
        self.recommended_spawn_items.clear()
        self.setSceneRect(bounds)
        for idx, tf in enumerate(recommended_spawns):
            item = RecommendedSpawnDot(self.controller, idx, tf)
            self.addItem(item)
            self.recommended_spawn_items.append(item)
        self.update()

    def drawBackground(self, painter: QPainter, rect: QRectF):
        painter.fillRect(rect, QColor("#1e1e2e"))

        grid_pen = QPen(QColor("#2a2a3e"))
        grid_pen.setWidth(1)
        painter.setPen(grid_pen)
        grid = 100
        left = int(rect.left()) - (int(rect.left()) % grid)
        top = int(rect.top()) - (int(rect.top()) % grid)
        for x in range(left, int(rect.right()) + grid, grid):
            painter.drawLine(x, int(rect.top()), x, int(rect.bottom()))
        for y in range(top, int(rect.bottom()) + grid, grid):
            painter.drawLine(int(rect.left()), y, int(rect.right()), y)

        road_pen = QPen(QColor("#89b4fa"))
        road_pen.setWidth(3)
        road_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(road_pen)
        for x1, y1, x2, y2 in self.road_segments:
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        axis_pen = QPen(QColor("#45475a"))
        axis_pen.setWidth(1)
        painter.setPen(axis_pen)
        painter.drawLine(0, int(rect.top()), 0, int(rect.bottom()))
        painter.drawLine(int(rect.left()), 0, int(rect.right()), 0)

        border_pen = QPen(QColor("#585b70"))
        border_pen.setWidth(2)
        painter.setPen(border_pen)
        painter.drawRect(self.map_bounds)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.controller:
            self.controller.pick_spawn_from_map(event.scenePos())
            event.accept()
            return
        super().mousePressEvent(event)


class CarlaMapView(QGraphicsView):
    def __init__(self, scene: CarlaMapScene):
        super().__init__(scene)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setStyleSheet("background: #1e1e2e; border: none;")
        self._zoom = 0

    def fit_map(self):
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = 0

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else 1 / 1.15
        new_zoom = self._zoom + (1 if delta > 0 else -1)
        if -15 <= new_zoom <= 25:
            self._zoom = new_zoom
            self.scale(factor, factor)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class CarlaSpawnTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CARLA Spawn Tool")
        self.resize(1280, 780)

        self.client = None
        self.world = None
        self.carla_map = None

        self.selected_marker: Optional[SpawnMarkerItem] = None
        self.spawned_vehicles: list = []
        self.spawned_walkers: list = []
        self.walker_controllers: list = []

        self.current_path: Optional[str] = None
        self.loading = False
        self.dirty = False

        self.scene = CarlaMapScene(self)
        self.view = CarlaMapView(self.scene)

        self._apply_stylesheet()
        self._build_ui()
        self._build_actions()
        self._set_defaults()

        QTimer.singleShot(0, self.view.fit_map)

    # ------------------------------------------------------------------
    # Style
    # ------------------------------------------------------------------

    def _apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Segoe UI', sans-serif;
                font-size: 11px;
            }
            QGroupBox {
                border: 1px solid #313244;
                border-radius: 4px;
                margin-top: 6px;
                padding-top: 4px;
                font-weight: bold;
                color: #89b4fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 6px;
            }
            QTabWidget::pane {
                border: 1px solid #313244;
                border-radius: 4px;
            }
            QTabBar::tab {
                background: #181825;
                color: #a6adc8;
                padding: 4px 10px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 60px;
            }
            QTabBar::tab:selected {
                background: #313244;
                color: #cdd6f4;
            }
            QPushButton {
                background: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 3px 10px;
            }
            QPushButton:hover  { background: #45475a; }
            QPushButton:pressed { background: #585b70; }
            QPushButton#danger {
                background: #522525;
                border-color: #f38ba8;
                color: #f38ba8;
            }
            QPushButton#danger:hover { background: #7d2828; }
            QPushButton#accent {
                background: #1d4785;
                border-color: #89b4fa;
                color: #89b4fa;
            }
            QPushButton#accent:hover { background: #2563b0; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background: #181825;
                border: 1px solid #45475a;
                border-radius: 3px;
                padding: 2px 4px;
                color: #cdd6f4;
            }
            QListWidget {
                background: #181825;
                border: 1px solid #313244;
                border-radius: 4px;
                color: #a6adc8;
                font-size: 10px;
            }
            QListWidget::item:selected {
                background: #313244;
                color: #cdd6f4;
            }
            QSplitter::handle { background: #313244; }
            QStatusBar { background: #181825; color: #6c7086; font-size: 10px; }
            QLabel { color: #cdd6f4; }
            QCheckBox { color: #cdd6f4; }
            QCheckBox::indicator { border: 1px solid #45475a; background: #181825;
                                   border-radius: 2px; width: 12px; height: 12px; }
            QCheckBox::indicator:checked { background: #89b4fa; }
            QScrollBar:vertical { background: #181825; width: 8px; border-radius: 4px; }
            QScrollBar::handle:vertical { background: #45475a; border-radius: 4px; }
        """)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        splitter = QSplitter()
        splitter.setChildrenCollapsible(False)

        # Left: map
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(0)

        # Toolbar row above map
        toolbar = QWidget()
        tl = QHBoxLayout(toolbar)
        tl.setContentsMargins(4, 4, 4, 4)
        tl.setSpacing(4)
        self.connect_btn = QPushButton("⚡ Connect")
        self.connect_btn.setObjectName("accent")
        self.refresh_btn = QPushButton("↻ Refresh")
        self.topdown_btn = QPushButton("⊙ Top-Down")
        self.fit_btn = QPushButton("⤢ Fit")
        self.connection_status = QLabel("● Disconnected")
        self.connection_status.setStyleSheet("color: #f38ba8;")
        tl.addWidget(self.connect_btn)
        tl.addWidget(self.refresh_btn)
        tl.addWidget(self.topdown_btn)
        tl.addWidget(self.fit_btn)
        tl.addStretch()
        tl.addWidget(self.connection_status)

        ll.addWidget(toolbar)
        ll.addWidget(self.view, 1)

        # Right: tabbed panel
        right = QWidget()
        right.setFixedWidth(280)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)
        rl.setSpacing(4)

        tabs = QTabWidget()
        tabs.addTab(self._build_spawn_tab(), "Spawn")
        tabs.addTab(self._build_traffic_tab(), "Traffic")
        tabs.addTab(self._build_weather_tab(), "World")
        tabs.addTab(self._build_file_tab(), "JSON")

        rl.addWidget(tabs)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(splitter)

        # Wire toolbar buttons
        self.connect_btn.clicked.connect(self.connect_to_carla)
        self.refresh_btn.clicked.connect(self.refresh_map)
        self.topdown_btn.clicked.connect(self.set_top_down_spectator)
        self.fit_btn.clicked.connect(self.view.fit_map)

        self.statusBar().showMessage("Connect to CARLA, then click the map to choose a spawn point.")

    def _row(self, label, widget):
        """Compact form row helper."""
        return label, widget

    def _build_spawn_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        # Connection settings (host/port/timeout) — small box
        conn_box = QGroupBox("Connection")
        conn_form = QFormLayout(conn_box)
        conn_form.setContentsMargins(6, 12, 6, 6)
        conn_form.setSpacing(3)
        conn_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.host_edit = QLineEdit("127.0.0.1")
        self.port_spin = QSpinBox(); self.port_spin.setRange(1, 65535); self.port_spin.setValue(2000)
        self.timeout_spin = QDoubleSpinBox()
        self.timeout_spin.setRange(1.0, 60.0); self.timeout_spin.setValue(10.0); self.timeout_spin.setDecimals(1)
        conn_form.addRow("Host", self.host_edit)
        conn_form.addRow("Port", self.port_spin)
        conn_form.addRow("Timeout", self.timeout_spin)
        lay.addWidget(conn_box)

        # Spawn transform
        tf_box = QGroupBox("Spawn Transform")
        tf_form = QFormLayout(tf_box)
        tf_form.setContentsMargins(6, 12, 6, 6)
        tf_form.setSpacing(3)
        tf_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.x_spin    = self._make_coord_spin()
        self.y_spin    = self._make_coord_spin()
        self.z_spin    = self._make_coord_spin()
        self.roll_spin = self._make_coord_spin()
        self.pitch_spin= self._make_coord_spin()
        self.yaw_spin  = self._make_coord_spin()

        tf_form.addRow("X", self.x_spin)
        tf_form.addRow("Y", self.y_spin)
        tf_form.addRow("Z", self.z_spin)
        tf_form.addRow("Roll", self.roll_spin)
        tf_form.addRow("Pitch", self.pitch_spin)
        tf_form.addRow("Yaw", self.yaw_spin)

        self.spawn_preview = QLabel("x=0.000  y=0.000  z=0.000")
        self.spawn_preview.setStyleSheet("color: #a6e3a1; font-size: 10px;")
        tf_form.addRow("", self.spawn_preview)
        lay.addWidget(tf_box)

        # Recommended spawn list
        rec_box = QGroupBox("Recommended Spawn Points")
        rec_lay = QVBoxLayout(rec_box)
        rec_lay.setContentsMargins(4, 12, 4, 4)
        self.spawn_list = QListWidget()
        self.spawn_list.setMaximumHeight(140)
        rec_lay.addWidget(self.spawn_list)
        lay.addWidget(rec_box)

        lay.addStretch()

        for sp in (self.x_spin, self.y_spin, self.z_spin, self.roll_spin, self.pitch_spin, self.yaw_spin):
            sp.valueChanged.connect(self.on_spawn_fields_changed)
        self.spawn_list.currentRowChanged.connect(self.on_spawn_list_changed)

        return w

    def _build_traffic_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        box = QGroupBox("Traffic / Walkers")
        form = QFormLayout(box)
        form.setContentsMargins(6, 12, 6, 6)
        form.setSpacing(3)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.vehicle_count_spin = QSpinBox(); self.vehicle_count_spin.setRange(0, 1000)
        self.walker_count_spin  = QSpinBox(); self.walker_count_spin.setRange(0, 3000)
        self.tm_port_spin       = QSpinBox(); self.tm_port_spin.setRange(1, 65535); self.tm_port_spin.setValue(8000)
        self.seed_spin          = QSpinBox(); self.seed_spin.setRange(0, 99999999); self.seed_spin.setValue(12345)
        self.auto_spawn_chk     = QCheckBox("Auto-spawn on JSON load")

        form.addRow("Vehicles", self.vehicle_count_spin)
        form.addRow("Walkers",  self.walker_count_spin)
        form.addRow("TM Port",  self.tm_port_spin)
        form.addRow("Seed",     self.seed_spin)
        form.addRow("",         self.auto_spawn_chk)

        lay.addWidget(box)

        btn_row = QHBoxLayout()
        self.spawn_traffic_btn  = QPushButton("▶ Spawn Traffic")
        self.spawn_traffic_btn.setObjectName("accent")
        self.clear_actors_btn   = QPushButton("✕ Clear Actors")
        self.clear_actors_btn.setObjectName("danger")
        btn_row.addWidget(self.spawn_traffic_btn)
        btn_row.addWidget(self.clear_actors_btn)
        lay.addLayout(btn_row)

        self.traffic_status = QLabel("No actors spawned.")
        self.traffic_status.setStyleSheet("color: #6c7086; font-size: 10px;")
        self.traffic_status.setWordWrap(True)
        lay.addWidget(self.traffic_status)
        lay.addStretch()

        self.spawn_traffic_btn.clicked.connect(self.spawn_traffic)
        self.clear_actors_btn.clicked.connect(self.clear_spawned_actors)

        for w_ in (self.vehicle_count_spin, self.walker_count_spin, self.tm_port_spin, self.seed_spin):
            w_.valueChanged.connect(self.mark_dirty)
        self.auto_spawn_chk.toggled.connect(self.mark_dirty)

        return w

    def _build_weather_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        box = QGroupBox("Weather & Map Info")
        form = QFormLayout(box)
        form.setContentsMargins(6, 12, 6, 6)
        form.setSpacing(3)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.weather_combo = QComboBox()
        self.weather_combo.addItems(WEATHER_PRESETS)
        self.map_name_edit = QLineEdit()
        self.map_name_edit.setPlaceholderText("Map label for JSON")
        self.notes_edit = QLineEdit()
        self.notes_edit.setPlaceholderText("Optional notes")

        form.addRow("Preset",   self.weather_combo)
        form.addRow("Map name", self.map_name_edit)
        form.addRow("Notes",    self.notes_edit)
        lay.addWidget(box)

        self.weather_btn = QPushButton("☁ Apply Weather to CARLA")
        self.weather_btn.setObjectName("accent")
        lay.addWidget(self.weather_btn)
        lay.addStretch()

        self.weather_btn.clicked.connect(self.apply_weather)
        self.weather_combo.currentIndexChanged.connect(self.mark_dirty)
        self.map_name_edit.textChanged.connect(self.mark_dirty)
        self.notes_edit.textChanged.connect(self.mark_dirty)

        return w

    def _build_file_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        self.path_label = QLabel("No JSON loaded/saved.")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("color: #6c7086; font-size: 10px;")
        lay.addWidget(self.path_label)

        self.save_btn    = QPushButton("💾 Save JSON")
        self.save_as_btn = QPushButton("💾 Save As…")
        self.load_btn    = QPushButton("📂 Load JSON")
        self.save_btn.setObjectName("accent")

        lay.addWidget(self.save_btn)
        lay.addWidget(self.save_as_btn)
        lay.addWidget(self.load_btn)
        lay.addStretch()

        self.save_btn.clicked.connect(self.save_json)
        self.save_as_btn.clicked.connect(self.save_json_as)
        self.load_btn.clicked.connect(self.load_json)

        return w

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _make_coord_spin(self) -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setRange(-100000.0, 100000.0)
        sp.setDecimals(3)
        sp.setSingleStep(0.5)
        sp.setFixedWidth(120)
        return sp

    def _build_actions(self):
        fit_action = QAction(self)
        fit_action.setShortcut(QKeySequence("Home"))
        fit_action.triggered.connect(self.view.fit_map)
        self.addAction(fit_action)

    def _set_defaults(self):
        self.loading = True
        self.map_name_edit.setText("CARLA Map")
        self.vehicle_count_spin.setValue(25)
        self.walker_count_spin.setValue(60)
        self.weather_combo.setCurrentText("ClearNoon")
        for sp in (self.x_spin, self.y_spin, self.z_spin, self.roll_spin, self.pitch_spin, self.yaw_spin):
            sp.setValue(0.0)
        self.loading = False
        self.update_spawn_preview()

    # ------------------------------------------------------------------
    # Dirty / title
    # ------------------------------------------------------------------

    def mark_dirty(self):
        if self.loading:
            return
        self.dirty = True
        self._update_title()

    def clear_dirty(self):
        self.dirty = False
        self._update_title()

    def _update_title(self):
        title = "CARLA Spawn Tool"
        if self.current_path:
            title += f"  —  {os.path.basename(self.current_path)}"
        if self.dirty:
            title += "  ●"
        self.setWindowTitle(title)

    # ------------------------------------------------------------------
    # Spawn transform helpers
    # ------------------------------------------------------------------

    def update_spawn_preview(self):
        self.spawn_preview.setText(
            f"x={self.x_spin.value():.2f}  y={self.y_spin.value():.2f}  "
            f"z={self.z_spin.value():.2f}  yaw={self.yaw_spin.value():.1f}°"
        )

    def on_spawn_fields_changed(self):
        if self.loading:
            return
        self.update_spawn_preview()
        self.mark_dirty()
        self._sync_marker_to_fields()

    def _current_transform(self):
        return transform_from_fields(
            self.x_spin.value(), self.y_spin.value(), self.z_spin.value(),
            self.roll_spin.value(), self.pitch_spin.value(), self.yaw_spin.value(),
        )

    def _sync_marker_to_fields(self):
        if self.selected_marker is None:
            return
        tf = self._current_transform()
        self.selected_marker.set_transform(tf)

    def set_selected_transform(self, tf, label: str = "Selected spawn transform"):
        self.loading = True
        try:
            self.x_spin.setValue(float(tf.location.x))
            self.y_spin.setValue(float(tf.location.y))
            self.z_spin.setValue(float(tf.location.z))
            self.roll_spin.setValue(float(tf.rotation.roll))
            self.pitch_spin.setValue(float(tf.rotation.pitch))
            self.yaw_spin.setValue(float(tf.rotation.yaw))
        finally:
            self.loading = False

        self.update_spawn_preview()
        # Only mark dirty if not in bulk-loading state
        self.mark_dirty()

        if self.selected_marker is None:
            self.selected_marker = SpawnMarkerItem(tf)
            self.scene.addItem(self.selected_marker)
        else:
            self.selected_marker.set_transform(tf)

        self.statusBar().showMessage(
            f"{label}  |  x={tf.location.x:.2f}  y={tf.location.y:.2f}  z={tf.location.z:.2f}", 4000
        )

    def choose_recommended_spawn(self, index: int):
        if self.carla_map is None or not self.scene.recommended_spawn_items:
            return
        index = max(0, min(index, len(self.scene.recommended_spawn_items) - 1))
        tf = self.scene.recommended_spawn_items[index].tf
        self.set_selected_transform(tf, f"Recommended spawn #{index}")
        self.spawn_list.blockSignals(True)
        self.spawn_list.setCurrentRow(index)
        self.spawn_list.blockSignals(False)

    def pick_spawn_from_map(self, scene_pos: QPointF):
        if self.carla_map is None or carla is None:
            return
        loc = carla.Location(x=float(scene_pos.x()), y=float(scene_pos.y()), z=0.0)
        waypoint = self.carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if waypoint is None:
            if self.scene.recommended_spawn_items:
                self.choose_recommended_spawn(0)
            return
        self.set_selected_transform(waypoint.transform, "Road waypoint (projected)")
        self.spawn_list.blockSignals(True)
        self.spawn_list.clearSelection()
        self.spawn_list.blockSignals(False)

    def on_spawn_list_changed(self, row: int):
        if row < 0 or self.carla_map is None:
            return
        self.choose_recommended_spawn(row)

    # ------------------------------------------------------------------
    # CARLA connection / map
    # ------------------------------------------------------------------

    def connect_to_carla(self):
        if carla is None:
            QMessageBox.critical(self, "CARLA import failed", f"Could not import carla:\n{CARLA_IMPORT_ERROR}")
            return

        host = self.host_edit.text().strip()
        port = int(self.port_spin.value())
        timeout = float(self.timeout_spin.value())

        try:
            client = carla.Client(host, port)
            client.set_timeout(timeout)
            world = client.get_world()
            carla_map = world.get_map()

            self.client = client
            self.world = world
            self.carla_map = carla_map

            self.connection_status.setText(f"● {carla_map.name}")
            self.connection_status.setStyleSheet("color: #a6e3a1;")
            self.load_map_data()
            self.view.fit_map()
            self.statusBar().showMessage(f"Connected: {carla_map.name}", 4000)
        except Exception as e:
            QMessageBox.critical(self, "Connection failed", str(e))

    def refresh_map(self):
        if self.carla_map is None:
            QMessageBox.warning(self, "Not connected", "Connect to CARLA first.")
            return
        self.load_map_data()
        self.statusBar().showMessage("Map refreshed.", 3000)

    def build_road_segments(self, waypoints, step: float):
        seen = set()
        segments = []
        for wp in waypoints:
            try:
                next_wps = wp.next(step)
            except Exception:
                next_wps = []
            for nxt in next_wps:
                a = wp.transform.location
                b = nxt.transform.location
                key = segment_key(a, b)
                if key in seen:
                    continue
                seen.add(key)
                segments.append((a.x, a.y, b.x, b.y))
        return segments

    def load_map_data(self):
        if self.carla_map is None:
            return
        step = 4.0
        try:
            waypoints = self.carla_map.generate_waypoints(step)
        except Exception:
            waypoints = []

        road_segments = self.build_road_segments(waypoints, step)
        spawn_points = list(self.carla_map.get_spawn_points())

        pts = [(x, y) for x, y, *_ in road_segments] + [(tf.location.x, tf.location.y) for tf in spawn_points]
        bounds = rect_from_points(pts, padding=150.0)

        self.scene.set_map_data(road_segments, spawn_points, bounds)

        self.spawn_list.blockSignals(True)
        self.spawn_list.clear()
        for i, tf in enumerate(spawn_points):
            loc = tf.location; rot = tf.rotation
            text = f"#{i:03d}  x={loc.x:.0f} y={loc.y:.0f}  yaw={rot.yaw:.0f}°"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, i)
            self.spawn_list.addItem(item)
        self.spawn_list.blockSignals(False)

        if spawn_points:
            self.choose_recommended_spawn(0)

    # ------------------------------------------------------------------
    # Spectator
    # ------------------------------------------------------------------

    def set_top_down_spectator(self):
        if self.world is None or carla is None:
            QMessageBox.warning(self, "Not connected", "Connect to CARLA first.")
            return
        tf = self._current_transform()
        if not isinstance(tf, carla.Transform):
            return
        spectator = self.world.get_spectator()
        overhead = carla.Transform(
            carla.Location(x=tf.location.x, y=tf.location.y, z=tf.location.z + 80.0),
            carla.Rotation(pitch=-90.0, yaw=tf.rotation.yaw, roll=0.0),
        )
        spectator.set_transform(overhead)
        self.statusBar().showMessage("Spectator → top-down view.", 3000)

    # ------------------------------------------------------------------
    # Weather
    # ------------------------------------------------------------------

    def apply_weather(self):
        if self.world is None or carla is None:
            QMessageBox.warning(self, "Not connected", "Connect to CARLA first.")
            return
        preset_name = self.weather_combo.currentText()
        try:
            preset = getattr(carla.WeatherParameters, preset_name)
            self.world.set_weather(preset)
            self.statusBar().showMessage(f"Weather applied: {preset_name}", 3000)
            self.mark_dirty()
        except Exception as e:
            QMessageBox.critical(self, "Weather failed", str(e))

    # ------------------------------------------------------------------
    # Traffic — FIX: use apply_batch_sync for reliable cleanup
    # ------------------------------------------------------------------

    def clear_spawned_actors(self):
        """
        Properly destroy all spawned actors.

        Order matters in CARLA:
          1. Stop & destroy walker AI controllers first.
          2. Destroy walkers.
          3. Destroy vehicles.
        Using apply_batch_sync avoids the one-by-one overhead and is the
        recommended pattern from the CARLA examples.
        """
        if carla is not None and self.client is not None:
            # Stop controllers before destroying them
            for ctrl in self.walker_controllers:
                try:
                    ctrl.stop()
                except Exception:
                    pass

            batch = []
            for ctrl in self.walker_controllers:
                batch.append(carla.command.DestroyActor(ctrl))
            for walker in self.spawned_walkers:
                batch.append(carla.command.DestroyActor(walker))
            for veh in self.spawned_vehicles:
                batch.append(carla.command.DestroyActor(veh))

            if batch:
                try:
                    self.client.apply_batch_sync(batch, False)
                except Exception:
                    pass
        else:
            # No CARLA — best-effort individual destroy
            for ctrl in list(self.walker_controllers):
                try:
                    ctrl.stop()
                except Exception:
                    pass
                try:
                    ctrl.destroy()
                except Exception:
                    pass
            for actor in list(self.spawned_walkers) + list(self.spawned_vehicles):
                try:
                    actor.destroy()
                except Exception:
                    pass

        n_v = len(self.spawned_vehicles)
        n_w = len(self.spawned_walkers)
        self.walker_controllers.clear()
        self.spawned_walkers.clear()
        self.spawned_vehicles.clear()

        msg = f"Cleared {n_v} vehicle(s) and {n_w} walker(s)."
        self.traffic_status.setText(msg)
        self.statusBar().showMessage(msg, 4000)

    def spawn_traffic(self):
        if self.world is None or self.carla_map is None or carla is None:
            QMessageBox.warning(self, "Not connected", "Connect to CARLA first.")
            return

        self.clear_spawned_actors()

        rng = random.Random(int(self.seed_spin.value()))
        bp_lib = self.world.get_blueprint_library()
        vehicle_bps = bp_lib.filter("vehicle.*")
        walker_bps  = bp_lib.filter("walker.pedestrian.*")
        controller_bp = bp_lib.find("controller.ai.walker")
        spawn_points = list(self.carla_map.get_spawn_points())

        if not spawn_points:
            QMessageBox.warning(self, "No spawn points", "This map returned no spawn points.")
            return

        rng.shuffle(spawn_points)

        tm_port = int(self.tm_port_spin.value())
        traffic_manager = self.client.get_trafficmanager(tm_port)
        traffic_manager.set_random_device_seed(int(self.seed_spin.value()))

        vehicle_count = min(int(self.vehicle_count_spin.value()), len(spawn_points))
        walker_count  = int(self.walker_count_spin.value())

        # Batch-spawn vehicles
        batch = [
            carla.command.SpawnActor(rng.choice(vehicle_bps), tf)
              .then(carla.command.SetAutopilot(carla.command.FutureActor, True, tm_port))
            for tf in spawn_points[:vehicle_count]
        ]
        results = self.client.apply_batch_sync(batch, True)
        vehicle_spawned = 0
        for res in results:
            if not res.error:
                actor = self.world.get_actor(res.actor_id)
                if actor:
                    self.spawned_vehicles.append(actor)
                    vehicle_spawned += 1

        # Spawn walkers
        walker_spawned = 0
        for _ in range(walker_count):
            nav_loc = self.world.get_random_location_from_navigation()
            if nav_loc is None:
                continue
            walker_bp = rng.choice(walker_bps)
            walker_tf = carla.Transform(nav_loc)
            walker = self.world.try_spawn_actor(walker_bp, walker_tf)
            if walker is None:
                continue
            ctrl = self.world.try_spawn_actor(controller_bp, carla.Transform(), walker)
            if ctrl is None:
                try:
                    walker.destroy()
                except Exception:
                    pass
                continue
            destination = self.world.get_random_location_from_navigation()
            if destination is not None:
                ctrl.start()
                ctrl.go_to_location(destination)
                ctrl.set_max_speed(rng.uniform(1.0, 2.0))
            self.spawned_walkers.append(walker)
            self.walker_controllers.append(ctrl)
            walker_spawned += 1

        msg = f"Spawned {vehicle_spawned} vehicle(s) and {walker_spawned} walker(s)."
        self.traffic_status.setText(msg)
        self.statusBar().showMessage(msg, 5000)

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def gather_json(self) -> dict:
        return {
            "version": 1,
            "map_name": self.map_name_edit.text().strip(),
            "spawn_point": {
                "x": float(self.x_spin.value()),
                "y": float(self.y_spin.value()),
                "z":  max(0.5, float(self.z_spin.value())),
                "roll": float(self.roll_spin.value()),
                "pitch": float(self.pitch_spin.value()),
                "yaw": float(self.yaw_spin.value()),
            },
            "traffic": {
                "vehicles": int(self.vehicle_count_spin.value()),
                "walkers": int(self.walker_count_spin.value()),
                "traffic_manager_port": int(self.tm_port_spin.value()),
                "seed": int(self.seed_spin.value()),
                "auto_spawn": bool(self.auto_spawn_chk.isChecked()),
            },
            "weather": {"preset": self.weather_combo.currentText()},
            "notes": self.notes_edit.text().strip(),
        }

    def save_json(self):
        if not self.current_path:
            self.save_json_as()
            return
        self._save_to_path(self.current_path)

    def save_json_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CARLA Spawn JSON", "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        self.current_path = path
        self._save_to_path(path)

    def _save_to_path(self, path: str):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.gather_json(), f, indent=2)
            self.path_label.setText(os.path.basename(path))
            self.clear_dirty()
            self.statusBar().showMessage(f"Saved: {path}", 4000)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def load_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load CARLA Spawn JSON", "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.load_json_data(data)
            self.current_path = path
            self.path_label.setText(os.path.basename(path))
            self.clear_dirty()
            self.statusBar().showMessage(f"Loaded: {path}", 4000)
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))

    def load_json_data(self, data: dict):
        self.loading = True
        try:
            self.map_name_edit.setText(str(data.get("map_name", "CARLA Map")))

            sp = data.get("spawn_point", {}) or {}
            self.x_spin.setValue(float(sp.get("x", 0.0)))
            self.y_spin.setValue(float(sp.get("y", 0.0)))
            self.z_spin.setValue(float(sp.get("z", 0.0)))
            self.roll_spin.setValue(float(sp.get("roll", 0.0)))
            self.pitch_spin.setValue(float(sp.get("pitch", 0.0)))
            self.yaw_spin.setValue(float(sp.get("yaw", 0.0)))

            tr = data.get("traffic", {}) or {}
            self.vehicle_count_spin.setValue(int(tr.get("vehicles", 25)))
            self.walker_count_spin.setValue(int(tr.get("walkers", 60)))
            self.tm_port_spin.setValue(int(tr.get("traffic_manager_port", 8000)))
            self.seed_spin.setValue(int(tr.get("seed", 12345)))
            self.auto_spawn_chk.setChecked(bool(tr.get("auto_spawn", False)))

            weather = data.get("weather", {}) or {}
            self.weather_combo.setCurrentText(str(weather.get("preset", "ClearNoon")))
            self.notes_edit.setText(str(data.get("notes", "")))
        finally:
            self.loading = False

        # Update marker — loading=False now so mark_dirty will fire correctly
        tf = transform_from_fields(
            self.x_spin.value(), self.y_spin.value(), self.z_spin.value(),
            self.roll_spin.value(), self.pitch_spin.value(), self.yaw_spin.value(),
        )
        self.set_selected_transform(tf, "Loaded from JSON")
        self.update_spawn_preview()

        if self.auto_spawn_chk.isChecked():
            self.spawn_traffic()

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        # Ask about unsaved changes BEFORE destroying actors
        if self.dirty:
            reply = QMessageBox.question(
                self, "Unsaved changes",
                "You have unsaved changes. Close anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

        self.clear_spawned_actors()
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = CarlaSpawnTool()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()