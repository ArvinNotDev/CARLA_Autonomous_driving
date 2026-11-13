from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import QWidget

import config_city as conf


class LidarRadarWidget(QWidget):
    """Lightweight top-down display for the latest accumulated LiDAR scan."""

    _height_stops = (
        (-2.0, (45, 90, 240)),
        (0.0, (30, 200, 210)),
        (1.0, (55, 220, 120)),
        (2.0, (240, 220, 60)),
        (4.0, (250, 125, 35)),
        (6.0, (235, 50, 55)),
    )

    def __init__(self, lidar_manager=None, parent=None) -> None:
        super().__init__(parent)
        self.lidar_manager = lidar_manager
        self._points: np.ndarray | None = None
        self._heading_deg = 0.0
        self._complete = False
        self._road_geometry: dict = {}
        sensor_range = min(
            float(conf.LIDAR_MAX_RANGE_M), max(1.0, float(conf.LIDAR_RANGE_M))
        )
        self.display_range_m = min(
            sensor_range,
            max(5.0, float(getattr(conf, "LIDAR_DISPLAY_RANGE_M", sensor_range))),
        )
        self.setMinimumSize(320, 320)

    def zoom_in(self) -> None:
        self.display_range_m = max(5.0, self.display_range_m - 5.0)
        self.update()

    def zoom_out(self) -> None:
        self.display_range_m = min(self._sensor_range(), self.display_range_m + 5.0)
        self.update()

    def refresh_scan(self) -> None:
        if self.lidar_manager is None:
            return
        points, heading_deg, complete = self.lidar_manager.snapshot()
        geometry = self.lidar_manager.road_geometry_snapshot()
        if (
            points is self._points
            and heading_deg == self._heading_deg
            and geometry is self._road_geometry
        ):
            return
        self._points = points
        self._heading_deg = heading_deg
        self._complete = complete
        self._road_geometry = geometry
        self.update()

    @classmethod
    def _height_colors(cls, height: np.ndarray) -> np.ndarray:
        limits = [stop[0] for stop in cls._height_stops]
        clipped = np.clip(height, limits[0], limits[-1])
        colors = np.empty((len(height), 3), dtype=np.uint8)
        for channel in range(3):
            colors[:, channel] = np.interp(
                clipped,
                limits,
                [stop[1][channel] for stop in cls._height_stops],
            ).astype(np.uint8)
        return colors

    def _point_image(self, diameter: int, radius: float) -> QImage:
        canvas = np.zeros((diameter, diameter, 3), dtype=np.uint8)
        points = self._points
        if points is None or not len(points):
            return QImage(
                canvas.data,
                diameter,
                diameter,
                canvas.strides[0],
                QImage.Format.Format_RGB888,
            ).copy()

        max_range = self._display_range()
        px = (diameter / 2.0 + points[:, 1] * radius / max_range).astype(np.int32)
        py = (diameter / 2.0 - points[:, 0] * radius / max_range).astype(np.int32)
        inside = (
            (px >= 0)
            & (px < diameter)
            & (py >= 0)
            & (py < diameter)
            & (np.hypot(points[:, 0], points[:, 1]) <= max_range)
        )
        if np.any(inside):
            colors = self._height_colors(points[inside, 2])
            canvas[py[inside], px[inside]] = colors

        cells = np.asarray(
            self._road_geometry.get("surface_cells", np.empty((0, 3))),
            dtype=np.float32,
        )
        if cells.ndim == 2 and cells.shape[1] >= 3 and len(cells):
            cell_size = max(
                1,
                int(round(
                    float(self._road_geometry.get("surface_resolution_m", 0.25))
                    * diameter / (2.0 * max_range)
                )),
            )
            cell_px = (
                diameter / 2.0 + cells[:, 1] * radius / max_range
            ).astype(np.int32)
            cell_py = (
                diameter / 2.0 - cells[:, 0] * radius / max_range
            ).astype(np.int32)
            cell_labels = cells[:, 2].astype(np.uint8)
            for label, rgb in (
                (1, np.asarray((35, 115, 215), dtype=np.float32)),
                (2, np.asarray((238, 145, 42), dtype=np.float32)),
            ):
                selected = cell_labels == label
                if not np.any(selected):
                    continue
                cx = cell_px[selected]
                cy = cell_py[selected]
                for offset_y in range(cell_size):
                    for offset_x in range(cell_size):
                        draw_x = cx + offset_x
                        draw_y = cy + offset_y
                        valid_cells = (
                            (draw_x >= 0)
                            & (draw_x < diameter)
                            & (draw_y >= 0)
                            & (draw_y < diameter)
                        )
                        if not np.any(valid_cells):
                            continue
                        target = canvas[
                            draw_y[valid_cells], draw_x[valid_cells]
                        ].astype(np.float32)
                        canvas[
                            draw_y[valid_cells], draw_x[valid_cells]
                        ] = (target * 0.28 + rgb * 0.72).astype(np.uint8)

        return QImage(
            canvas.data,
            diameter,
            diameter,
            canvas.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#101418"))

        width = self.width()
        height = self.height()
        max_range = self._display_range()
        diameter = max(80, int(min(width - 160, height - 100)))
        diameter = min(diameter, int(height - 88))
        radius = diameter / 2.0
        center_x = max(radius + 16, (width - 96) / 2.0)
        center_y = height / 2.0 + 24.0
        image = self._point_image(diameter, radius)
        target = QRectF(center_x - radius, center_y - radius, diameter, diameter)
        painter.drawImage(target, image)
        painter.save()
        geometry_clip = QPainterPath()
        geometry_clip.addEllipse(QPointF(center_x, center_y), radius, radius)
        painter.setClipPath(geometry_clip)
        self._draw_geometry_boundaries(
            painter, center_x, center_y, radius, max_range
        )
        painter.restore()

        painter.setPen(QPen(QColor("#52616b"), 1))
        for distance in self._ring_distances(max_range):
            ring_radius = radius * distance / max_range
            painter.drawEllipse(QPointF(center_x, center_y), ring_radius, ring_radius)
            painter.setPen(QColor("#a9b7c0"))
            painter.drawText(
                QPointF(center_x + 4, center_y - ring_radius - 3),
                f"{distance:g} m",
            )
            painter.setPen(QPen(QColor("#52616b"), 1))

        painter.drawLine(QPointF(center_x - radius, center_y), QPointF(center_x + radius, center_y))
        painter.drawLine(QPointF(center_x, center_y - radius), QPointF(center_x, center_y + radius))
        self._draw_vehicle(painter, center_x, center_y)
        self._draw_height_legend(painter, center_x + radius + 22, center_y, height)

        state = "360° scan" if self._complete else "Accumulating 360° scan"
        if self.lidar_manager is None:
            state = "LiDAR unavailable"
        painter.setPen(QColor("#e6edf3"))
        painter.drawText(
            QRectF(12, 8, width - 24, 24),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"CARLA LiDAR • {state} • heading {self._heading_deg:.1f}° • range {max_range:g} m",
        )
        geometry = self._road_geometry
        road_width = geometry.get("road_width_m")
        left_width = geometry.get("shoulder_width_left_m")
        right_width = geometry.get("shoulder_width_right_m")
        left_height = geometry.get("shoulder_height_left_m")
        right_height = geometry.get("shoulder_height_right_m")
        shoulder_length_left = geometry.get("shoulder_length_left_m")
        shoulder_length_right = geometry.get("shoulder_length_right_m")

        def metric(value, suffix="m", signed=False):
            if value is None:
                return "—"
            number = float(value)
            rendered = f"{number:+.2f}" if signed else f"{number:.2f}"
            return f"{rendered} {suffix}"

        road_text = (
            f"Road width: {metric(road_width)}"
            if road_width is not None
            else "Road width: boundary not reliably detected"
        )
        shoulder_text = (
            f"Shoulder L/R: {metric(left_width)} / {metric(right_width)}"
        )
        height_text = (
            f"Shoulder height L/R: {metric(left_height, signed=True)} / {metric(right_height, signed=True)}"
        )
        length_text = (
            "visible shoulder run L/R: "
            f"{metric(shoulder_length_left)} / {metric(shoulder_length_right)}"
        )
        coverage = float(geometry.get("road_width_confidence", 0.0))
        status = str(geometry.get("status", "waiting_for_scan"))
        painter.setPen(QColor("#dce7ee"))
        painter.drawText(
            QRectF(12, 32, width - 24, 20),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"{road_text}   |   {shoulder_text}",
        )
        painter.drawText(
            QRectF(12, 51, width - 24, 20),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"{height_text}   |   {length_text}",
        )
        painter.drawText(
            QRectF(12, 70, width - 24, 18),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"Edge coverage: {coverage:.0%} • {status} • Blue: road | Orange: shoulder",
        )
        painter.end()

    def _draw_geometry_boundaries(
        self,
        painter: QPainter,
        center_x: float,
        center_y: float,
        radius: float,
        max_range: float,
    ) -> None:
        geometry = self._road_geometry
        lines = (
            ("left_edge_points", QColor("#5bd7ff"), 2),
            ("right_edge_points", QColor("#5bd7ff"), 2),
            ("left_shoulder_outer_points", QColor("#ffad42"), 2),
            ("right_shoulder_outer_points", QColor("#ffad42"), 2),
        )
        for key, color, line_width in lines:
            points = np.asarray(
                geometry.get(key, np.empty((0, 2))), dtype=np.float32
            )
            if points.ndim != 2 or points.shape[0] < 2:
                continue
            painter.setPen(QPen(color, line_width))
            mapped = [
                QPointF(
                    center_x + float(point[1]) * radius / max_range,
                    center_y - float(point[0]) * radius / max_range,
                )
                for point in points
            ]
            for first, second in zip(mapped, mapped[1:]):
                painter.drawLine(first, second)

    def _display_range(self) -> float:
        self.display_range_m = min(self._sensor_range(), max(5.0, self.display_range_m))
        return self.display_range_m

    @staticmethod
    def _sensor_range() -> float:
        return min(
            float(conf.LIDAR_MAX_RANGE_M), max(1.0, float(conf.LIDAR_RANGE_M))
        )

    @staticmethod
    def _ring_distances(max_range: float) -> list[float]:
        step = 5.0 if max_range <= 30.0 else 10.0 if max_range <= 60.0 else 25.0 if max_range <= 125.0 else 50.0
        distances = []
        distance = step
        while distance < max_range:
            distances.append(distance)
            distance += step
        distances.append(max_range)
        return distances

    @staticmethod
    def _draw_vehicle(painter: QPainter, center_x: float, center_y: float) -> None:
        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.setBrush(QColor("#f4f7fb"))
        painter.drawPolygon(
            QPolygonF(
                [
                    QPointF(center_x, center_y - 15),
                    QPointF(center_x + 9, center_y + 11),
                    QPointF(center_x, center_y + 6),
                    QPointF(center_x - 9, center_y + 11),
                ]
            )
        )

    @classmethod
    def _draw_height_legend(
        cls, painter: QPainter, x: float, center_y: float, canvas_height: int
    ) -> None:
        legend_height = min(240.0, max(120.0, canvas_height - 100.0))
        top = center_y - legend_height / 2.0
        rect = QRectF(x, top, 18, legend_height)
        gradient = QLinearGradient(QPointF(x, top), QPointF(x, top + legend_height))
        for height_m, rgb in reversed(cls._height_stops):
            position = 1.0 - (height_m + 2.0) / 8.0
            gradient.setColorAt(position, QColor(*rgb))
        painter.fillRect(rect, gradient)
        painter.setPen(QColor("#e6edf3"))
        painter.drawText(QPointF(x - 3, top - 8), "Height above road (m)")
        for height_m in (-2, 0, 2, 4, 6):
            y = top + (6.0 - height_m) / 8.0 * legend_height
            painter.drawText(QPointF(x + 24, y + 4), f"{height_m:+d}")
