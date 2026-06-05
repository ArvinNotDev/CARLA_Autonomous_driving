import cv2
import numpy as np

def lidar_to_bev(points, size=800, scale=10):
    img = np.zeros((size, size, 3), dtype=np.uint8)

    center = size // 2

    xyz = points[:, :3]

    for x, y, z in xyz:
        px = int(center + y * scale)
        py = int(size - x * scale)

        if 0 <= px < size and 0 <= py < size:
            img[py, px] = (255, 255, 255)

    return img