from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class _LaneLineCandidate:
    """One geometrically coherent lane-marking candidate in an ROI."""

    x_bottom: float
    slope: float
    y_top: float
    y_bottom: float
    quality: float
    segment_count: int
    support_length: float

    @property
    def angle_deg(self) -> float:
        return float(math.degrees(math.atan(self.slope)))


@dataclass(frozen=True)
class LaneLineSelection:
    """Selected line and diagnostics for one inference frame."""

    mask: np.ndarray
    segment_count: int
    candidate_count: int
    selected_x: float | None
    quality: float | None
    stable_frames: int
    missed_frames: int
    pending_switch_frames: int
    switched: bool
    predicted: bool


def _number(settings: dict[str, Any], key: str, default: float) -> float:
    try:
        value = float(settings.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _line_candidates(
    binary_roi: np.ndarray,
    roi_side: str,
    settings: dict[str, Any],
    x_offset: float,
) -> tuple[list[_LaneLineCandidate], int]:
    roi = np.asarray(binary_roi)
    if roi.ndim != 2:
        raise ValueError("Lane ROI mask must be a single-channel image.")
    height, width = roi.shape
    if height < 3 or width < 3 or roi_side not in {"ll", "rl"}:
        return [], 0

    min_line_ratio = min(
        0.5, max(0.005, _number(settings, "hough_min_line_length_ratio", 0.035))
    )
    min_line_length = max(5, int(round(width * min_line_ratio)))
    threshold = max(1, int(round(_number(settings, "hough_threshold", 12))))
    max_line_gap = max(0, int(round(_number(settings, "hough_max_line_gap", 24))))
    min_angle = min(89.0, max(0.0, _number(settings, "min_lane_angle_deg", 0.0)))
    max_angle = min(
        89.0,
        max(min_angle, _number(settings, "max_lane_angle_deg", 80.0)),
    )

    lines = cv2.HoughLinesP(
        np.where(roi > 0, 255, 0).astype(np.uint8),
        rho=1,
        theta=np.pi / 180.0,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return [], 0

    reference_y = float(height - 1)
    margin = max(4.0, width * 0.08)
    records: list[dict[str, Any]] = []
    for raw_line in lines:
        values = np.asarray(raw_line).reshape(-1)
        if values.size < 4:
            continue
        x1, y1, x2, y2 = (int(value) for value in values[:4])
        if y2 < y1:
            x1, x2 = x2, x1
            y1, y2 = y2, y1

        dy = y2 - y1
        if dy < 3:
            continue
        dx = x2 - x1
        length = math.hypot(float(dx), float(dy))
        if length < min_line_length:
            continue

        slope = float(dx) / float(dy)
        angle = float(math.degrees(math.atan(slope)))
        if abs(angle) < min_angle or abs(angle) > max_angle:
            continue
        x_bottom_local = float(x1) + slope * (reference_y - float(y1))
        if x_bottom_local < -margin or x_bottom_local > width - 1 + margin:
            continue

        records.append({
            "segment": (x1, y1, x2, y2),
            "length": length,
            "slope": slope,
            "angle": angle,
            "x_bottom": x_bottom_local,
        })

    if not records:
        return [], 0

    # Group split Hough fragments by projected position and signed slope.
    # The bounded spatial gate keeps adjacent markings as separate candidates.
    cluster_gap = max(3.5, min(8.0, width * 0.085))
    cluster_angle = 12.0
    clusters: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: item["x_bottom"]):
        best_cluster = None
        best_cost = math.inf
        for cluster in clusters:
            dx = abs(record["x_bottom"] - cluster["x_bottom"])
            da = abs(record["angle"] - cluster["angle"])
            if dx <= cluster_gap and da <= cluster_angle:
                cost = dx / cluster_gap + da / cluster_angle
                if cost < best_cost:
                    best_cluster = cluster
                    best_cost = cost

        if best_cluster is None:
            clusters.append({
                "records": [record],
                "weight": record["length"],
                "x_bottom": record["x_bottom"],
                "slope": record["slope"],
                "angle": record["angle"],
            })
            continue

        best_cluster["records"].append(record)
        total_weight = best_cluster["weight"] + record["length"]
        best_cluster["x_bottom"] = (
            best_cluster["x_bottom"] * best_cluster["weight"]
            + record["x_bottom"] * record["length"]
        ) / total_weight
        best_cluster["slope"] = (
            best_cluster["slope"] * best_cluster["weight"]
            + record["slope"] * record["length"]
        ) / total_weight
        best_cluster["angle"] = float(
            math.degrees(math.atan(best_cluster["slope"]))
        )
        best_cluster["weight"] = total_weight

    candidates: list[_LaneLineCandidate] = []
    min_vertical_span = max(5.0, height * 0.14)
    for cluster in clusters:
        group = cluster["records"]
        intervals = sorted(
            (
                min(record["segment"][1], record["segment"][3]),
                max(record["segment"][1], record["segment"][3]),
            )
            for record in group
        )
        y_top = float(intervals[0][0])
        y_bottom = float(intervals[-1][1])
        if y_bottom - y_top < min_vertical_span:
            continue

        # Use union coverage; overlapping fragments must not inflate support.
        merge_gap = max(2.0, height * 0.025)
        covered = 0.0
        interval_start, interval_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= interval_end + merge_gap:
                interval_end = max(interval_end, end)
            else:
                covered += float(interval_end - interval_start)
                interval_start, interval_end = start, end
        covered += float(interval_end - interval_start)

        total_length = sum(float(record["length"]) for record in group)
        coverage_score = min(1.0, covered / max(1.0, float(height)))
        length_score = min(1.0, total_length / max(1.0, height * 0.90))
        fragment_score = min(1.0, len(group) / 3.0)
        quality = (
            0.50 * coverage_score
            + 0.35 * length_score
            + 0.15 * fragment_score
        )
        if quality < 0.16:
            continue

        # Refine the line with length-weighted least squares over all endpoints.
        points_y: list[float] = []
        points_x: list[float] = []
        point_weights: list[float] = []
        for record in group:
            x1, y1, x2, y2 = record["segment"]
            weight = max(1.0, float(record["length"]) * 0.5)
            points_y.extend((float(y1), float(y2)))
            points_x.extend((float(x1), float(x2)))
            point_weights.extend((weight, weight))

        weights = np.asarray(point_weights, dtype=np.float64)
        ys = np.asarray(points_y, dtype=np.float64)
        xs = np.asarray(points_x, dtype=np.float64)
        weight_sum = float(np.sum(weights))
        mean_y = float(np.sum(ys * weights) / weight_sum)
        mean_x = float(np.sum(xs * weights) / weight_sum)
        variance_y = float(np.sum(weights * (ys - mean_y) ** 2))
        if variance_y > 1e-6:
            slope = float(
                np.sum(weights * (ys - mean_y) * (xs - mean_x))
                / variance_y
            )
        else:
            slope = float(cluster["slope"])
        x_bottom_local = mean_x + slope * (reference_y - mean_y)

        candidates.append(_LaneLineCandidate(
            x_bottom=float(x_bottom_local + x_offset),
            slope=slope,
            y_top=y_top,
            y_bottom=y_bottom,
            quality=float(quality),
            segment_count=len(group),
            support_length=float(total_length),
        ))

    candidates.sort(key=lambda candidate: candidate.x_bottom)
    return candidates, len(records)


def _blend_candidate(
    previous: _LaneLineCandidate,
    current: _LaneLineCandidate,
    alpha: float,
) -> _LaneLineCandidate:
    alpha = min(1.0, max(0.0, float(alpha)))
    keep = 1.0 - alpha
    return _LaneLineCandidate(
        x_bottom=keep * previous.x_bottom + alpha * current.x_bottom,
        slope=keep * previous.slope + alpha * current.slope,
        y_top=keep * previous.y_top + alpha * current.y_top,
        y_bottom=keep * previous.y_bottom + alpha * current.y_bottom,
        quality=keep * previous.quality + alpha * current.quality,
        segment_count=current.segment_count,
        support_length=current.support_length,
    )


class TemporalLaneLineSelector:
    """Stable LL/RL selector with edge ordering and temporal hysteresis."""

    SWITCH_CONFIRM_FRAMES = 3
    REACQUIRE_CONFIRM_FRAMES = 2
    MAX_MISSED_FRAMES = 2
    TRACK_SMOOTHING_ALPHA = 0.45

    def __init__(self, roi_side: str) -> None:
        if roi_side not in {"ll", "rl"}:
            raise ValueError("roi_side must be 'll' or 'rl'.")
        self.roi_side = roi_side
        self.reset()

    def reset(self) -> None:
        self._track: _LaneLineCandidate | None = None
        self._pending: _LaneLineCandidate | None = None
        self._pending_frames = 0
        self._missed_frames = 0
        self._stable_frames = 0
        self._ever_locked = False

    def _is_more_inward(
        self,
        candidate: _LaneLineCandidate,
        reference: _LaneLineCandidate,
        roi_width: int,
    ) -> bool:
        margin = max(2.0, float(roi_width) * 0.025)
        if self.roi_side == "ll":
            return candidate.x_bottom > reference.x_bottom + margin
        return candidate.x_bottom < reference.x_bottom - margin

    def _candidate_match_cost(
        self,
        candidate: _LaneLineCandidate,
        reference: _LaneLineCandidate,
        roi_width: int,
    ) -> float | None:
        x_tolerance = max(6.0, float(roi_width) * 0.12)
        angle_tolerance = 18.0
        x_delta = abs(candidate.x_bottom - reference.x_bottom)
        angle_delta = abs(candidate.angle_deg - reference.angle_deg)
        if x_delta > x_tolerance or angle_delta > angle_tolerance:
            return None
        return x_delta / x_tolerance + 0.55 * angle_delta / angle_tolerance

    def _update_pending(
        self,
        candidate: _LaneLineCandidate,
        roi_width: int,
    ) -> None:
        if self._pending is not None:
            x_tolerance = max(3.0, float(roi_width) * 0.055)
            angle_tolerance = 12.0
            same_candidate = (
                abs(candidate.x_bottom - self._pending.x_bottom) <= x_tolerance
                and abs(candidate.angle_deg - self._pending.angle_deg) <= angle_tolerance
            )
            if same_candidate:
                self._pending = _blend_candidate(
                    self._pending, candidate, alpha=0.55
                )
                self._pending_frames += 1
                return

        self._pending = candidate
        self._pending_frames = 1

    def _clear_pending(self) -> None:
        self._pending = None
        self._pending_frames = 0

    def _edge_candidate(
        self,
        candidates: list[_LaneLineCandidate],
    ) -> _LaneLineCandidate | None:
        if not candidates:
            return None
        if self.roi_side == "ll":
            return max(candidates, key=lambda item: (item.x_bottom, item.quality))
        return min(candidates, key=lambda item: (item.x_bottom, -item.quality))

    @staticmethod
    def _render_track(
        height: int,
        width: int,
        track: _LaneLineCandidate,
        x_offset: float,
    ) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        y0 = max(0, min(height - 1, int(round(track.y_top))))
        y1 = max(0, min(height - 1, int(round(track.y_bottom))))
        if y1 <= y0:
            y0, y1 = 0, height - 1
        reference_y = float(height - 1)
        local_x_bottom = track.x_bottom - float(x_offset)
        x0 = int(round(local_x_bottom + track.slope * (float(y0) - reference_y)))
        x1 = int(round(local_x_bottom + track.slope * (float(y1) - reference_y)))
        cv2.line(
            mask,
            (x0, y0),
            (x1, y1),
            255,
            thickness=3,
            lineType=cv2.LINE_AA,
        )
        return mask

    def select(
        self,
        binary_roi: np.ndarray,
        settings: dict[str, Any],
        *,
        x_offset: float = 0.0,
    ) -> LaneLineSelection:
        roi = np.asarray(binary_roi)
        if roi.ndim != 2:
            raise ValueError("Lane ROI mask must be a single-channel image.")
        height, width = roi.shape
        candidates, segment_count = _line_candidates(
            roi, self.roi_side, settings, float(x_offset)
        )
        edge = self._edge_candidate(candidates)
        switched = False

        if self._track is None:
            if edge is None:
                self._clear_pending()
            else:
                if self._ever_locked:
                    self._update_pending(edge, width)
                    required = self.REACQUIRE_CONFIRM_FRAMES
                else:
                    required = 1
                if self._pending_frames >= required:
                    self._track = self._pending
                    self._ever_locked = True
                    self._stable_frames = 1
                    self._missed_frames = 0
                    self._clear_pending()
                    switched = True
        else:
            matches = [
                (self._candidate_match_cost(candidate, self._track, width), candidate)
                for candidate in candidates
            ]
            matches = [(cost, candidate) for cost, candidate in matches if cost is not None]
            matched = min(matches, key=lambda item: item[0])[1] if matches else None

            if matched is not None:
                self._missed_frames = 0
                self._stable_frames += 1
                self._track = _blend_candidate(
                    self._track, matched, alpha=self.TRACK_SMOOTHING_ALPHA
                )
                if (
                    edge is not None
                    and edge is not matched
                    and self._is_more_inward(edge, matched, width)
                ):
                    self._update_pending(edge, width)
                    if self._pending_frames >= self.SWITCH_CONFIRM_FRAMES:
                        self._track = self._pending
                        self._stable_frames = 1
                        self._clear_pending()
                        switched = True
                else:
                    self._clear_pending()
            else:
                self._missed_frames += 1
                self._stable_frames = 0
                if edge is not None:
                    self._update_pending(edge, width)
                    if (
                        self._missed_frames >= self.MAX_MISSED_FRAMES
                        and self._pending_frames >= self.REACQUIRE_CONFIRM_FRAMES
                    ):
                        self._track = self._pending
                        self._stable_frames = 1
                        self._missed_frames = 0
                        self._clear_pending()
                        switched = True
                    elif self._missed_frames > self.MAX_MISSED_FRAMES:
                        self._track = None
                        self._clear_pending()
                else:
                    self._clear_pending()
                    if self._missed_frames > self.MAX_MISSED_FRAMES:
                        self._track = None

        selected_x = self._track.x_bottom if self._track is not None else None
        quality = self._track.quality if self._track is not None else None
        if self._track is not None:
            predicted = self._missed_frames > 0
            mask = self._render_track(height, width, self._track, x_offset)
        else:
            predicted = False
            mask = np.zeros((height, width), dtype=np.uint8)

        return LaneLineSelection(
            mask=mask,
            segment_count=segment_count,
            candidate_count=len(candidates),
            selected_x=float(selected_x) if selected_x is not None else None,
            quality=float(quality) if quality is not None else None,
            stable_frames=int(self._stable_frames),
            missed_frames=int(self._missed_frames),
            pending_switch_frames=int(self._pending_frames),
            switched=switched,
            predicted=predicted,
        )


def select_lane_line_mask(
    binary_roi: np.ndarray,
    roi_side: str,
    settings: dict[str, Any],
) -> tuple[np.ndarray, int, int, float | None]:
    """Stateless compatibility wrapper; production uses persistent selectors."""
    selection = TemporalLaneLineSelector(roi_side).select(binary_roi, settings)
    return (
        selection.mask,
        selection.segment_count,
        selection.candidate_count,
        selection.selected_x,
    )
