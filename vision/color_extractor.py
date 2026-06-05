import cv2
import numpy as np


class HSVColorThresholdExtractor:
    def __init__(self, morph_kernel_size: int = 1):
        self.kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)

        self.ranges = {
            "green": {
                "lower": np.array([35, 50, 50], dtype=np.uint8),
                "upper": np.array([85, 255, 255], dtype=np.uint8),
            },
            "dark_purple": {
                "lower": np.array([130, 50, 10], dtype=np.uint8),
                "upper": np.array([165, 255, 180], dtype=np.uint8),
            }
        }

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return mask

    def extract(self, frame_bgr: np.ndarray) -> dict:
        """
        Returns:
            {
                "green": np.ndarray mask,
                "dark_purple": np.ndarray mask
            }
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        result = {}
        for name, bounds in self.ranges.items():
            mask = cv2.inRange(hsv, bounds["lower"], bounds["upper"])
            mask = self._clean_mask(mask)
            result[name] = mask

        return result

    def extract_with_stats(self, frame_bgr: np.ndarray) -> dict:
        masks = self.extract(frame_bgr)

        return {
            "green": {
                "mask": masks["green"],
                "pixels": int(np.count_nonzero(masks["green"])),
            },
            "dark_purple": {
                "mask": masks["dark_purple"],
                "pixels": int(np.count_nonzero(masks["dark_purple"])),
            },
        }