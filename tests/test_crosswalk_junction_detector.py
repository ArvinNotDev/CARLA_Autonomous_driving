import cv2
import numpy as np

import config_city as conf
from navigation.crosswalk_junction_detector import CrosswalkRoiJunctionDetector


def _set_roi(monkeypatch, *, left=0.10, right=0.90, top=0.10, bottom=0.90):
    monkeypatch.setattr(conf, "CW_LEFT_ROI", left)
    monkeypatch.setattr(conf, "CW_RIGHT_ROI", right)
    monkeypatch.setattr(conf, "CW_TOP_ROI", top)
    monkeypatch.setattr(conf, "CW_BOTTOM_ROI", bottom)
    monkeypatch.setattr(conf, "JUNCTION_ROI_MIN_LINE_LENGTH_RATIO", 0.35)
    monkeypatch.setattr(conf, "JUNCTION_ROI_MAX_LINE_ANGLE_DEG", 12.0)
    monkeypatch.setattr(conf, "JUNCTION_ROI_CONFIRM_FRAMES", 2)


def _frame_with_line():
    frame = np.zeros((240, 360, 3), dtype=np.uint8)
    cv2.line(frame, (42, 145), (318, 145), (255, 255, 255), 5)
    return frame


def test_horizontal_line_inside_crosswalk_roi_is_confirmed(monkeypatch):
    _set_roi(monkeypatch)
    detector = CrosswalkRoiJunctionDetector()
    frame = _frame_with_line()

    first = detector.detect(frame)
    second = detector.detect(frame)

    assert first["line_present"] is True
    assert first["detected"] is False
    assert second["detected"] is True
    assert second["line"] is not None
    assert second["length_ratio"] >= 0.35


def test_horizontal_line_outside_crosswalk_roi_is_ignored(monkeypatch):
    _set_roi(monkeypatch, left=0.35, right=0.70, top=0.20, bottom=0.80)
    frame = np.zeros((240, 360, 3), dtype=np.uint8)
    cv2.line(frame, (5, 145), (95, 145), (255, 255, 255), 5)

    result = CrosswalkRoiJunctionDetector().detect(frame)

    assert result["line_present"] is False
    assert result["detected"] is False


def test_vertical_line_does_not_trigger_junction(monkeypatch):
    _set_roi(monkeypatch)
    frame = np.zeros((240, 360, 3), dtype=np.uint8)
    cv2.line(frame, (180, 35), (180, 205), (255, 255, 255), 5)

    result = CrosswalkRoiJunctionDetector().detect(frame)

    assert result["line_present"] is False
    assert result["detected"] is False


def test_confirmation_vote_rejects_single_frame_noise(monkeypatch):
    _set_roi(monkeypatch)
    detector = CrosswalkRoiJunctionDetector()
    line_frame = _frame_with_line()
    blank_frame = np.zeros_like(line_frame)

    assert detector.detect(line_frame)["detected"] is False
    assert detector.detect(blank_frame)["detected"] is False
    assert detector.detect(blank_frame)["detected"] is False
