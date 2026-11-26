from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def select_lane_line_mask(
    binary_roi: np.ndarray,
    roi_side: str,
    settings: dict[str, Any],
) -> tuple[np.ndarray, int, int, float | None]:
    """Return only the lane-marking candidate nearest the inner ROI edge.

    Hough fragments are grouped by their projected x-position near the bottom
    of the ROI and by orientation. LL keeps the rightmost group; RL keeps the
    leftmost group. This avoids merging multiple markings into one median.
    """
    roi = np.asarray(binary_roi)
    if roi.ndim != 2:
        raise ValueError("Lane ROI mask must be a single-channel image.")
    height, width = roi.shape
    selected_mask = np.zeros((height, width), dtype=np.uint8)
    if height < 3 or width < 3 or roi_side not in {"ll", "rl"}:
        return selected_mask, 0, 0, None

    def number(key: str, default: float) -> float:
        try:
            value = float(settings.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default

    min_line_ratio = min(0.5, max(0.005, number("hough_min_line_length_ratio", 0.035)))
    min_line_length = max(5, int(round(width * min_line_ratio)))
    threshold = max(1, int(round(number("hough_threshold", 12))))
    max_line_gap = max(0, int(round(number("hough_max_line_gap", 24))))
    min_angle = min(89.0, max(0.0, number("min_lane_angle_deg", 0.0)))
    max_angle = min(89.0, max(min_angle, number("max_lane_angle_deg", 80.0)))

    segments = cv2.HoughLinesP(
        np.where(roi > 0, 255, 0).astype(np.uint8),
        rho=1,
        theta=np.pi / 180.0,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if segments is None:
        return selected_mask, 0, 0, None

    reference_y = float(height - 1)
    records: list[dict[str, Any]] = []
    margin = max(4.0, width * 0.08)
    for raw_segment in segments:
        x1, y1, x2, y2 = (
            int(value) for value in np.asarray(raw_segment).reshape(-1)[:4]
        )
        if y2 < y1:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        dy = y2 - y1
        dx = x2 - x1
        if dy < 2:
            continue
        length = math.hypot(float(dx), float(dy))
        if length < min_line_length:
            continue
        angle = abs(math.degrees(math.atan2(float(abs(dx)), float(dy))))
        if not min_angle <= angle <= max_angle:
            continue

        slope = float(dx) / float(dy)
        intercept = float(x1) - slope * float(y1)
        projected_x = slope * reference_y + intercept
        if projected_x < -margin or projected_x > width - 1 + margin:
            continue
        records.append({
            "segment": (x1, y1, x2, y2),
            "length": length,
            "angle": angle,
            "x": projected_x,
        })

    if not records:
        return selected_mask, 0, 0, None

    # Hough often splits one painted marking into several adjacent segments.
    # Group only fragments that agree in both projected position and angle.
    cluster_gap = max(4.0, width * 0.075)
    cluster_angle = 18.0
    clusters: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: item["x"]):
        best_cluster = None
        best_distance = math.inf
        for cluster in clusters:
            distance = abs(record["x"] - cluster["x"])
            angle_distance = abs(record["angle"] - cluster["angle"])
            if distance <= cluster_gap and angle_distance <= cluster_angle:
                if distance < best_distance:
                    best_cluster = cluster
                    best_distance = distance
        if best_cluster is None:
            clusters.append({
                "records": [record],
                "weight": record["length"],
                "x": record["x"],
                "angle": record["angle"],
            })
        else:
            best_cluster["records"].append(record)
            total = best_cluster["weight"] + record["length"]
            best_cluster["x"] = (
                best_cluster["x"] * best_cluster["weight"]
                + record["x"] * record["length"]
            ) / total
            best_cluster["angle"] = (
                best_cluster["angle"] * best_cluster["weight"]
                + record["angle"] * record["length"]
            ) / total
            best_cluster["weight"] = total

    min_vertical_span = max(3.0, height * 0.08)
    candidates = []
    for cluster in clusters:
        group = cluster["records"]
        y_values = [
            y
            for record in group
            for y in (record["segment"][1], record["segment"][3])
        ]
        if max(y_values) - min(y_values) < min_vertical_span:
            continue
        candidates.append(cluster)

    if not candidates:
        return selected_mask, len(records), 0, None

    selected = (
        max(candidates, key=lambda item: item["x"])
        if roi_side == "ll"
        else min(candidates, key=lambda item: item["x"])
    )
    for record in selected["records"]:
        segment = record["segment"]
        cv2.line(
            selected_mask,
            (segment[0], segment[1]),
            (segment[2], segment[3]),
            255,
            thickness=3,
            lineType=cv2.LINE_AA,
        )

    return selected_mask, len(records), len(candidates), float(selected["x"])
