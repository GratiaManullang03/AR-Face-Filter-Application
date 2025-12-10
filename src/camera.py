"""
Camera module for webcam input handling.
Provides a clean interface for camera operations.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from . import config


class Camera:
    """Handles webcam capture and frame management."""

    def __init__(
        self,
        camera_index: int = config.CAMERA_INDEX,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT
    ):
        """
        Initialize the camera.

        Args:
            camera_index: Index of the camera device
            width: Frame width
            height: Frame height
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.capture: Optional[cv2.VideoCapture] = None
        self.is_opened = False

    def open(self) -> bool:
        """
        Open the camera device.

        Returns:
            True if camera opened successfully, False otherwise
        """
        self.capture = cv2.VideoCapture(self.camera_index)

        if not self.capture.isOpened():
            return False

        # Set camera properties
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.is_opened = True
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.

        Returns:
            Tuple of (success, frame)
            - success: Boolean indicating if frame was read successfully
            - frame: NumPy array of the frame (BGR format) or None if failed
        """
        if not self.is_opened or self.capture is None:
            return False, None

        success, frame = self.capture.read()

        if not success:
            return False, None

        # Flip frame horizontally for mirror effect (more intuitive for users)
        frame = cv2.flip(frame, 1)

        return True, frame

    def release(self) -> None:
        """Release the camera resources."""
        if self.capture is not None:
            self.capture.release()
            self.is_opened = False

    def get_fps(self) -> float:
        """
        Get the current FPS of the camera.

        Returns:
            FPS value
        """
        if self.capture is None:
            return 0.0

        return self.capture.get(cv2.CAP_PROP_FPS)

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False
