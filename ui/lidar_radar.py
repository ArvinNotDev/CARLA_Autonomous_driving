from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import (
    QColor,
    QImage,
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
        geometry = self.lidar_manager.road_edge_snapshot()
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

        max_range = self._display_range()
        diameter = max(80, int(min(self.width() - 24, self.height() - 24)))
        radius = diameter / 2.0
        center_x = self.width() / 2.0
        center_y = self.height() / 2.0

        image = self._point_image(diameter, radius)
        target = QRectF(center_x - radius, center_y - radius, diameter, diameter)
        painter.drawImage(target, image)

        painter.save()
        clip = QPainterPath()
        clip.addEllipse(QPointF(center_x, center_y), radius, radius)
        painter.setClipPath(clip)
        self._draw_geometry_boundaries(
            painter, center_x, center_y, radius, max_range
        )
        painter.restore()

        painter.setPen(QPen(QColor("#52616b"), 1))
        for distance in self._ring_distances(max_range):
            ring_radius = radius * distance / max_range
            painter.drawEllipse(
                QPointF(center_x, center_y), ring_radius, ring_radius
            )

        painter.drawLine(
            QPointF(center_x - radius, center_y),
            QPointF(center_x + radius, center_y),
        )
        painter.drawLine(
            QPointF(center_x, center_y - radius),
            QPointF(center_x, center_y + radius),
        )
        self._draw_vehicle(painter, center_x, center_y)
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
        edge_color = QColor("#35d9e8")
        for key in ("left_edge_points", "right_edge_points"):
            points = np.asarray(
                geometry.get(key, np.empty((0, 2))), dtype=np.float32
            )
            if points.ndim != 2 or points.shape[0] < 2:
                continue

            painter.setPen(QPen(edge_color, 2.5))
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

