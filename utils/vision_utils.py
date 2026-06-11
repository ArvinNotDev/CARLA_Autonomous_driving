import cv2
import numpy as np


def birdseye_view(drivable_mask):
    """
    Convert a front-facing drivable-area mask to a bird's-eye view.

    Args:
        drivable_mask: Binary or grayscale drivable area image.

    Returns:
        Warped bird's-eye-view image.
    """

    h, w = drivable_mask.shape[:2]

    # Source points (trapezoid in original image)
    src = np.float32([
        [w * 0.05, h * 0.98],  # bottom-left
        [w * 0.95, h * 0.98],  # bottom-right
        [w * 0.70, h * 0.65],  # top-right
        [w * 0.30, h * 0.65],  # top-left
        
    ])

    # Destination points (rectangle in bird-eye view)
    dst = np.float32([
        [w * 0.20, 0],
        [w * 0.80, 0],
        [w * 0.80, h],
        [w * 0.20, h],
    ])

    matrix = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(
        drivable_mask,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR
    )

    return warped