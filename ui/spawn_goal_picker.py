from __future__ import annotations

import sys
from typing import List, Tuple

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
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


def rect_from_points(points: List[Tuple[float, float]], padding: float = 120.0) -> QRectF:
    if not points:
        return QRectF(-500, -500, 1000, 1000)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return QRectF(
        min(xs) - padding,
        min(ys) - padding,
        max(xs) - min(xs) + 2 * padding,
        max(ys) - min(ys) + 2 * padding,
    )


def segment_key(a, b):
    ax, ay = round(float(a.x), 2), round(float(a.y), 2)
    bx, by = round(float(b.x), 2), round(float(b.y), 2)
    return (ax, ay, bx, by) if (ax, ay) <= (bx, by) else (bx, by, ax, ay)


class _Location:
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class _Rotation:
    def __init__(self, roll, pitch, yaw):
        self.roll = float(roll)
        self.pitch = float(pitch)
        self.yaw = float(yaw)


class _Transform:
    def __init__(self, x, y, z, roll, pitch, yaw):
        self.location = _Location(x, y, z)
        self.rotation = _Rotation(roll, pitch, yaw)


def make_transform(x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
    if carla is None:
        return _Transform(x, y, z, roll, pitch, yaw)
    return carla.Transform(
        carla.Location(x=float(x), y=float(y), z=float(z)),
        carla.Rotation(roll=float(roll), pitch=float(pitch), yaw=float(yaw)),
    )


class SpawnGoalMarker(QGraphicsEllipseItem):
    def __init__(self, title: str, color: str, outline: str, tf=None, is_goal: bool = False):
        super().__init__(-13, -13, 26, 26)
        self.is_goal = is_goal
        self.setBrush(QBrush(QColor(color)))
        self.setPen(QPen(QColor(outline), 2))
        self.setZValue(20 if not is_goal else 21)

        label = QGraphicsSimpleTextItem(title, self)
        font = QFont()
        font.setPointSize(8)
        font.setBold(True)
        label.setFont(font)
        label.setBrush(QBrush(QColor("black")))
        br = label.boundingRect()
        label.setPos(-br.width() / 2, -br.height() / 2)

        if tf is not None:
            self.set_transform(tf)

    def set_transform(self, tf):
        self.setPos(float(tf.location.x), float(tf.location.y))


class RoadDotItem(QGraphicsEllipseItem):
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
            self.controller.set_spawn_from_recommended(self.index)
        event.accept()


class CarlaMapScene(QGraphicsScene):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.road_segments: List[Tuple[float, float, float, float]] = []
        self.map_bounds = QRectF(-500, -500, 1000, 1000)
        self.recommended_spawn_items: List[RoadDotItem] = []

    def set_map_data(self, road_segments, recommended_spawns, bounds: QRectF):
        self.road_segments = road_segments
        self.map_bounds = bounds

        for item in self.recommended_spawn_items:
            self.removeItem(item)
        self.recommended_spawn_items.clear()

        self.setSceneRect(bounds)
        for idx, tf in enumerate(recommended_spawns):
            item = RoadDotItem(self.controller, idx, tf)
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
            self.controller.map_clicked(event.scenePos())
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


class SpawnGoalPicker(QDialog):
    """
    Pick both:
      - spawn transform (carla.Transform)
      - goal location (carla.Location)
    """

    def __init__(self, world, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CARLA Spawn + Goal Picker")
        self.resize(1380, 820)

        self.world = world
        self.carla_map = world.get_map() if world is not None else None

        self.loading = False
        self.mode = "spawn"

        self.scene = CarlaMapScene(self)
        self.view = CarlaMapView(self.scene)

        self.spawn_tf = make_transform(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if carla is not None:
            self.goal_loc = carla.Location(x=0.0, y=0.0, z=0.0)
        else:
            self.goal_loc = _Location(0.0, 0.0, 0.0)

        self.spawn_marker = None
        self.goal_marker = None

        self._build_ui()
        self._apply_style()
        self._build_actions()
        self._set_defaults()

        if self.carla_map is not None:
            self.load_map_data()

        QTimer.singleShot(0, self.view.fit_map)

    def _apply_style(self):
        self.setStyleSheet("""
            QDialog, QWidget {
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
            QPushButton {
                background: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 4px 10px;
            }
            QPushButton:hover  { background: #45475a; }
            QPushButton:pressed { background: #585b70; }
            QPushButton#accent {
                background: #1d4785;
                border-color: #89b4fa;
                color: #89b4fa;
            }
            QPushButton#accent:hover { background: #2563b0; }
            QPushButton#danger {
                background: #522525;
                border-color: #f38ba8;
                color: #f38ba8;
            }
            QPushButton#danger:hover { background: #7d2828; }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                background: #181825;
                border: 1px solid #45475a;
                border-radius: 3px;
                padding: 2px 4px;
                color: #cdd6f4;
            }
            QRadioButton, QCheckBox { color: #cdd6f4; }
            QLabel { color: #cdd6f4; }
            QSplitter::handle { background: #313244; }
        """)

    def _build_actions(self):
        fit_action = QAction(self)
        fit_action.setShortcut(QKeySequence("Home"))
        fit_action.triggered.connect(self.view.fit_map)
        self.addAction(fit_action)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter()
        splitter.setChildrenCollapsible(False)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        toolbar = QWidget()
        tl = QHBoxLayout(toolbar)
        tl.setContentsMargins(4, 4, 4, 4)
        tl.setSpacing(4)

        self.refresh_btn = QPushButton("↻ Refresh")
        self.fit_btn = QPushButton("⤢ Fit")
        self.status_label = QLabel("Left click the map to place the active marker.")
        self.status_label.setStyleSheet("color: #a6adc8; font-size: 10px;")

        tl.addWidget(self.refresh_btn)
        tl.addWidget(self.fit_btn)
        tl.addStretch()
        tl.addWidget(self.status_label)

        left_layout.addWidget(toolbar)
        left_layout.addWidget(self.view, 1)

        right = QWidget()
        right.setFixedWidth(330)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(6)

        mode_box = QGroupBox("Active Marker")
        mode_layout = QVBoxLayout(mode_box)
        mode_layout.setContentsMargins(6, 12, 6, 6)
        mode_layout.setSpacing(4)

        self.spawn_radio = QRadioButton("Place Spawn")
        self.goal_radio = QRadioButton("Place Goal")
        self.spawn_radio.setChecked(True)

        group = QButtonGroup(self)
        group.addButton(self.spawn_radio)
        group.addButton(self.goal_radio)

        self.snap_chk = QCheckBox("Snap to road when possible")
        self.snap_chk.setChecked(True)

        mode_layout.addWidget(self.spawn_radio)
        mode_layout.addWidget(self.goal_radio)
        mode_layout.addWidget(self.snap_chk)
        right_layout.addWidget(mode_box)

        spawn_box = QGroupBox("Spawn Transform")
        spawn_form = QFormLayout(spawn_box)
        spawn_form.setContentsMargins(6, 12, 6, 6)
        spawn_form.setSpacing(4)
        spawn_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.sx = self._spin()
        self.sy = self._spin()
        self.sz = self._spin()
        self.sroll = self._spin()
        self.spitch = self._spin()
        self.syaw = self._spin()

        spawn_form.addRow("X", self.sx)
        spawn_form.addRow("Y", self.sy)
        spawn_form.addRow("Z", self.sz)
        spawn_form.addRow("Roll", self.sroll)
        spawn_form.addRow("Pitch", self.spitch)
        spawn_form.addRow("Yaw", self.syaw)

        right_layout.addWidget(spawn_box)

        goal_box = QGroupBox("Goal Location")
        goal_form = QFormLayout(goal_box)
        goal_form.setContentsMargins(6, 12, 6, 6)
        goal_form.setSpacing(4)
        goal_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.gx = self._spin()
        self.gy = self._spin()
        self.gz = self._spin()

        goal_form.addRow("X", self.gx)
        goal_form.addRow("Y", self.gy)
        goal_form.addRow("Z", self.gz)

        right_layout.addWidget(goal_box)

        self.map_info = QLabel("Map not loaded.")
        self.map_info.setWordWrap(True)
        self.map_info.setStyleSheet("color: #6c7086; font-size: 10px;")
        right_layout.addWidget(self.map_info)

        btn_row = QHBoxLayout()
        self.accept_btn = QPushButton("Use Selected Points")
        self.accept_btn.setObjectName("accent")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("danger")
        btn_row.addWidget(self.accept_btn)
        btn_row.addWidget(self.cancel_btn)

        right_layout.addLayout(btn_row)
        right_layout.addStretch()

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        root.addWidget(splitter)

        self.refresh_btn.clicked.connect(self.load_map_data)
        self.fit_btn.clicked.connect(self.view.fit_map)
        self.accept_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

        self.spawn_radio.toggled.connect(self._mode_changed)
        self.goal_radio.toggled.connect(self._mode_changed)
        self.snap_chk.toggled.connect(self._mode_changed)

        for sp in (self.sx, self.sy, self.sz, self.sroll, self.spitch, self.syaw):
            sp.valueChanged.connect(self._fields_changed)
        for sp in (self.gx, self.gy, self.gz):
            sp.valueChanged.connect(self._fields_changed)

    def _spin(self) -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setRange(-100000.0, 100000.0)
        sp.setDecimals(3)
        sp.setSingleStep(0.5)
        sp.setFixedWidth(130)
        return sp

    def _set_defaults(self):
        self.loading = True
        try:
            self.sx.setValue(0.0)
            self.sy.setValue(0.0)
            self.sz.setValue(0.0)
            self.sroll.setValue(0.0)
            self.spitch.setValue(0.0)
            self.syaw.setValue(0.0)

            self.gx.setValue(0.0)
            self.gy.setValue(0.0)
            self.gz.setValue(0.0)
        finally:
            self.loading = False

        self._sync_spawn_marker()
        self._sync_goal_marker()

    def _mode_changed(self):
        if self.spawn_radio.isChecked():
            self.mode = "spawn"
            self.status_label.setText("Spawn mode: click the map to set the spawn point.")
        elif self.goal_radio.isChecked():
            self.mode = "goal"
            self.status_label.setText("Goal mode: click the map to set the goal point.")

    def _fields_changed(self):
        if self.loading:
            return
        self._sync_spawn_marker()
        self._sync_goal_marker()

    def _sync_spawn_marker(self):
        if self.spawn_marker is None:
            self.spawn_marker = SpawnGoalMarker("S", "#f1c40f", "#9a7d0a", is_goal=False)
            self.scene.addItem(self.spawn_marker)

        self.spawn_tf = make_transform(
            self.sx.value(), self.sy.value(), self.sz.value(),
            self.sroll.value(), self.spitch.value(), self.syaw.value(),
        )
        self.spawn_marker.set_transform(self.spawn_tf)

    def _sync_goal_marker(self):
        if self.goal_marker is None:
            self.goal_marker = SpawnGoalMarker("G", "#2ecc71", "#1e8449", is_goal=True)
            self.scene.addItem(self.goal_marker)

        if carla is not None:
            self.goal_loc = carla.Location(
                x=float(self.gx.value()),
                y=float(self.gy.value()),
                z=float(self.gz.value()),
            )
        else:
            self.goal_loc = _Location(self.gx.value(), self.gy.value(), self.gz.value())

        self.goal_marker.setPos(float(self.goal_loc.x), float(self.goal_loc.y))

    def _project_scene_pos(self, scene_pos: QPointF):
        if self.carla_map is None or carla is None or not self.snap_chk.isChecked():
            if carla is not None:
                return carla.Location(x=float(scene_pos.x()), y=float(scene_pos.y()), z=0.0), 0.0
            return _Location(scene_pos.x(), scene_pos.y(), 0.0), 0.0

        loc = carla.Location(x=float(scene_pos.x()), y=float(scene_pos.y()), z=0.0)
        try:
            wp = self.carla_map.get_waypoint(
                loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except Exception:
            wp = None

        if wp is None:
            return loc, 0.0

        tf = wp.transform
        return tf.location, float(tf.rotation.yaw)

    def map_clicked(self, scene_pos: QPointF):
        if self.mode == "spawn":
            loc, yaw = self._project_scene_pos(scene_pos)
            self.loading = True
            try:
                self.sx.setValue(float(loc.x))
                self.sy.setValue(float(loc.y))
                self.sz.setValue(float(getattr(loc, "z", 0.0)))
                self.syaw.setValue(float(yaw))
            finally:
                self.loading = False

            self._sync_spawn_marker()
            self.status_label.setText(f"Spawn set at x={loc.x:.2f}, y={loc.y:.2f}")
        else:
            loc, _ = self._project_scene_pos(scene_pos)
            self.loading = True
            try:
                self.gx.setValue(float(loc.x))
                self.gy.setValue(float(loc.y))
                self.gz.setValue(float(getattr(loc, "z", 0.0)))
            finally:
                self.loading = False

            self._sync_goal_marker()
            self.status_label.setText(f"Goal set at x={loc.x:.2f}, y={loc.y:.2f}")

    def set_spawn_from_recommended(self, index: int):
        if self.carla_map is None or not self.scene.recommended_spawn_items:
            return

        index = max(0, min(index, len(self.scene.recommended_spawn_items) - 1))
        tf = self.scene.recommended_spawn_items[index].tf

        self.spawn_radio.setChecked(True)
        self.loading = True
        try:
            self.sx.setValue(float(tf.location.x))
            self.sy.setValue(float(tf.location.y))
            self.sz.setValue(float(tf.location.z))
            self.sroll.setValue(float(tf.rotation.roll))
            self.spitch.setValue(float(tf.rotation.pitch))
            self.syaw.setValue(float(tf.rotation.yaw))
        finally:
            self.loading = False

        self._sync_spawn_marker()
        self.status_label.setText(f"Spawn set from recommended point #{index}")

    def _build_road_segments(self, waypoints, step: float):
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
            self.map_info.setText("CARLA map not available.")
            return

        step = 4.0
        try:
            waypoints = self.carla_map.generate_waypoints(step)
        except Exception:
            waypoints = []

        road_segments = self._build_road_segments(waypoints, step)
        spawn_points = list(self.carla_map.get_spawn_points())

        pts = [(x, y) for x, y, *_ in road_segments] + [(tf.location.x, tf.location.y) for tf in spawn_points]
        bounds = rect_from_points(pts, padding=150.0)
        self.scene.set_map_data(road_segments, spawn_points, bounds)

        self.map_info.setText(f"Map: {self.carla_map.name}\nSpawn points: {len(spawn_points)}")

        if spawn_points and self.spawn_marker is None:
            self.set_spawn_from_recommended(0)

    def get_spawn_transform(self):
        return self.spawn_tf

    def get_goal_location(self):
        return self.goal_loc


def main():
    app = QApplication(sys.argv)
    if carla is None:
        QMessageBox.critical(None, "CARLA import failed", f"Could not import carla:\n{CARLA_IMPORT_ERROR}")
        return 1

    from carla import Client

    client = Client("127.0.0.1", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    dlg = SpawnGoalPicker(world)
    if dlg.exec() == QDialog.DialogCode.Accepted:
        print("Spawn:", dlg.get_spawn_transform())
        print("Goal:", dlg.get_goal_location())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())