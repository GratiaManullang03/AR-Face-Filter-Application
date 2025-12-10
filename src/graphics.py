"""
Graphics module for image transformations and overlay operations.
Handles rotation, scaling, and alpha blending for AR filters.
"""

import cv2
import numpy as np
import math
from typing import Tuple


class GraphicsEngine:
    """Handles all graphics operations for AR filter rendering."""

    @staticmethod
    def calculate_angle(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate rotation angle between two points in degrees."""
        delta_x = point2[0] - point1[0]
        delta_y = point2[1] - point1[1]
        # Negate angle to fix inverted rotation (mirror effect from camera flip)
        return -math.degrees(math.atan2(delta_y, delta_x))

    @staticmethod
    def calculate_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points."""
        delta_x = point2[0] - point1[0]
        delta_y = point2[1] - point1[1]
        return math.sqrt(delta_x ** 2 + delta_y ** 2)

    @staticmethod
    def resize_image(image: np.ndarray, target_width: int) -> np.ndarray:
        """Resize image to target width, maintaining aspect ratio."""
        if target_width <= 0:
            return image
        height, width = image.shape[:2]
        target_height = int(target_width * height / width)
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image around its center point."""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new dimensions
        cos_val = abs(rotation_matrix[0, 0])
        sin_val = abs(rotation_matrix[0, 1])
        new_width = int(height * sin_val + width * cos_val)
        new_height = int(height * cos_val + width * sin_val)

        # Adjust for new dimensions
        rotation_matrix[0, 2] += (new_width - width) / 2
        rotation_matrix[1, 2] += (new_height - height) / 2

        return cv2.warpAffine(
            image, rotation_matrix, (new_width, new_height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

    @staticmethod
    def overlay_image(background: np.ndarray, overlay: np.ndarray,
                     position: Tuple[int, int]) -> np.ndarray:
        """Overlay image with alpha channel onto background at position."""
        if overlay.shape[2] != 4:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

        overlay_h, overlay_w = overlay.shape[:2]
        bg_h, bg_w = background.shape[:2]

        # Calculate position (top-left from center)
        x = position[0] - overlay_w // 2
        y = position[1] - overlay_h // 2

        # Calculate bounds
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(bg_w, x + overlay_w), min(bg_h, y + overlay_h)

        if x1 >= x2 or y1 >= y2:
            return background

        # Calculate overlay region
        ox1, oy1 = max(0, -x), max(0, -y)
        ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

        # Extract regions
        bg_region = background[y1:y2, x1:x2]
        overlay_region = overlay[oy1:oy2, ox1:ox2]

        if bg_region.shape[:2] != overlay_region.shape[:2]:
            return background

        # Alpha blending
        alpha = (overlay_region[:, :, 3] / 255.0)[:, :, np.newaxis]
        blended = (alpha * overlay_region[:, :, :3] +
                  (1 - alpha) * bg_region).astype(np.uint8)

        background[y1:y2, x1:x2] = blended
        return background
