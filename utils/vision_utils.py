import cv2
import numpy as np


def birdseye_view(drivable_mask):
    """
    Warp the drivable-area mask into bird's-eye view.
    """
    h, w = drivable_mask.shape[:2]

    src = np.float32([
        [w * 0.05, h * 0.98],  # bottom-left
        [w * 0.95, h * 0.98],  # bottom-right
        [w * 0.70, h * 0.65],  # top-right
        [w * 0.30, h * 0.65],  # top-left
    ])

    dst = np.float32([
        [w * 0.20, 0],
        [w * 0.80, 0],
        [w * 0.80, h - 1],
        [w * 0.20, h - 1],
    ])

    matrix = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(
        drivable_mask,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR
    )

    return warped


def find_drivable_center_x(drivable_mask: np.ndarray, row_ratio: float = 0.85):
    """
    Find the center x of the drivable area using a horizontal scanline
    near the bottom of the mask.

    Returns:
        center_x, debug_info
    """
    h, w = drivable_mask.shape[:2]

    if len(drivable_mask.shape) == 3:
        gray = cv2.cvtColor(drivable_mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = drivable_mask.copy()

    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    y = int(h * row_ratio)
    y = max(0, min(h - 1, y))

    row = binary[y]
    xs = np.where(row > 0)[0]

    debug = {
        "binary": binary,
        "scanline_y": y,
        "row": row,
        "xs": xs,
    }

    if len(xs) == 0:
        center_x = w // 2
        debug["left_x"] = None
        debug["right_x"] = None
        debug["center_x"] = center_x
        return center_x, debug

    left_x = int(xs[0])
    right_x = int(xs[-1])
    center_x = int((left_x + right_x) / 2)

    debug["left_x"] = left_x
    debug["right_x"] = right_x
    debug["center_x"] = center_x

    return center_x, debug


def show_drivable_debug(original_mask: np.ndarray):
    """
    Show debug windows for the original mask, bird-eye mask,
    and center detection.
    """
    bird = birdseye_view(original_mask)
    center_x, debug = find_drivable_center_x(bird, row_ratio=0.85)

    if len(bird.shape) == 2:
        bird_vis = cv2.cvtColor(bird, cv2.COLOR_GRAY2BGR)
    else:
        bird_vis = bird.copy()

    y = debug["scanline_y"]
    cv2.line(bird_vis, (0, y), (bird_vis.shape[1] - 1, y), (0, 255, 255), 2)

    if debug["left_x"] is not None and debug["right_x"] is not None:
        cv2.circle(bird_vis, (debug["left_x"], y), 6, (0, 255, 0), -1)
        cv2.circle(bird_vis, (debug["right_x"], y), 6, (0, 0, 255), -1)
        cv2.circle(bird_vis, (center_x, y), 6, (255, 0, 0), -1)

    info = f"center_x={center_x}"
    cv2.putText(
        bird_vis,
        info,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Original Mask", original_mask)
    cv2.imshow("Bird Eye Mask", bird_vis)
    cv2.imshow("Binary Mask", debug["binary"])
    cv2.waitKey(1)

    return center_x, bird, debug