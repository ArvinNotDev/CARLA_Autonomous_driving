from __future__ import annotations

import cv2
import numpy as np


def make_bev_canvas(pred_ego: np.ndarray, width: int = 640, height: int = 360) -> np.ndarray:
    """
    Simple BEV canvas for displaying predicted waypoints.
    This is only for visualization.
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # grid
    for x in range(0, width, 40):
        cv2.line(canvas, (x, 0), (x, height - 1), (40, 40, 40), 1)
    for y in range(0, height, 40):
        cv2.line(canvas, (0, y), (width - 1, y), (40, 40, 40), 1)

    return canvas


def overlay_text(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img.copy()
    x = 12
    y = 24
    for line in lines:
        cv2.putText(
            out,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 22
    return out

def draw_waypoints(frame, pred):
    img = frame.copy()

    h, w = img.shape[:2]

    origin_x = w // 2
    origin_y = h - 50

    scale = 2.2 # pixels per meter

    prev_pt = None

    for i, (x, y) in enumerate(pred):
        px = int(origin_x + y * scale)
        py = int(origin_y - x * scale)

        cv2.circle(img, (px, py), 5, (0, 255, 0), -1)

        cv2.putText(
            img,
            str(i + 1),
            (px + 5, py - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        if prev_pt is not None:
            cv2.line(img, prev_pt, (px, py), (0, 255, 255), 2)

        prev_pt = (px, py)

    return img