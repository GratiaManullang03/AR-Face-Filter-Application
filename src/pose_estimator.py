"""
Face pose estimation for 3D object placement.

Estimates the 3D pose (rotation and translation) of a face from 2D landmarks
using solvePnP algorithm. This enables accurate 3D object placement that
responds to head movements.
"""

import logging
from typing import Optional, Tuple
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class FacePoseEstimator:
    """
    Estimates face pose (rotation and translation) from 2D landmarks.

    Uses a subset of facial landmarks and their known 3D positions to
    compute the rotation and translation of the face relative to the camera.
    """

    def __init__(self, image_width: int, image_height: int):
        """
        Initialize pose estimator.

        Args:
            image_width: Width of camera frame in pixels
            image_height: Height of camera frame in pixels
        """
        self.image_width = image_width
        self.image_height = image_height

        # Camera intrinsic parameters (estimated)
        # Focal length approximation
        focal_length = image_width
        center = (image_width / 2, image_height / 2)

        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        # Assume no lens distortion
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # 3D model points (generic face model in cm)
        # Based on average human face proportions
        self.model_points_3d = np.array([
            (0.0, 0.0, 0.0),           # Nose tip
            (0.0, -3.3, -0.5),         # Chin
            (-2.2, 1.5, -1.5),         # Left eye left corner
            (2.2, 1.5, -1.5),          # Right eye right corner
            (-1.5, -1.5, -1.0),        # Left mouth corner
            (1.5, -1.5, -1.0)          # Right mouth corner
        ], dtype=np.float64)

        # Corresponding landmark indices in MediaPipe 478-point model
        self.landmark_indices = [
            1,    # Nose tip
            152,  # Chin
            33,   # Left eye left corner
            263,  # Right eye right corner
            61,   # Left mouth corner
            291   # Right mouth corner
        ]

    def estimate_pose(
        self,
        landmarks: list[Tuple[float, float]]
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Estimate face pose from 2D landmarks.

        Args:
            landmarks: List of (x, y) facial landmark coordinates

        Returns:
            Tuple of (rotation_vector, translation_vector) or None if estimation failed
        """
        try:
            # Extract the subset of landmarks we need
            image_points = np.array([
                landmarks[idx] for idx in self.landmark_indices
            ], dtype=np.float64)

            # Solve PnP to get rotation and translation vectors
            success, rotation_vec, translation_vec = cv2.solvePnP(
                self.model_points_3d,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                return rotation_vec, translation_vec
            else:
                return None

        except Exception as e:
            logger.warning(f"Pose estimation failed: {e}")
            return None

    def get_rotation_matrix(self, rotation_vec: np.ndarray) -> np.ndarray:
        """
        Convert rotation vector to rotation matrix.

        Args:
            rotation_vec: Rotation vector from solvePnP

        Returns:
            3x3 rotation matrix
        """
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        return rotation_mat

    def get_euler_angles(self, rotation_vec: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert rotation vector to Euler angles (pitch, yaw, roll).

        Args:
            rotation_vec: Rotation vector from solvePnP

        Returns:
            Tuple of (pitch, yaw, roll) in degrees
        """
        rotation_mat = self.get_rotation_matrix(rotation_vec)

        # Extract Euler angles from rotation matrix
        # Using the convention: R = Rz(roll) * Ry(yaw) * Rx(pitch)
        sy = np.sqrt(rotation_mat[0, 0]**2 + rotation_mat[1, 0]**2)

        singular = sy < 1e-6

        if not singular:
            pitch = np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2])
            yaw = np.arctan2(-rotation_mat[2, 0], sy)
            roll = np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0])
        else:
            pitch = np.arctan2(-rotation_mat[1, 2], rotation_mat[1, 1])
            yaw = np.arctan2(-rotation_mat[2, 0], sy)
            roll = 0

        # Convert to degrees
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)
        roll = np.degrees(roll)

        return pitch, yaw, roll

    def project_points(
        self,
        points_3d: np.ndarray,
        rotation_vec: np.ndarray,
        translation_vec: np.ndarray
    ) -> np.ndarray:
        """
        Project 3D points to 2D image coordinates.

        Args:
            points_3d: Array of 3D points (N, 3)
            rotation_vec: Rotation vector
            translation_vec: Translation vector

        Returns:
            Array of 2D points (N, 2)
        """
        points_2d, _ = cv2.projectPoints(
            points_3d,
            rotation_vec,
            translation_vec,
            self.camera_matrix,
            self.dist_coeffs
        )

        # Reshape from (N, 1, 2) to (N, 2)
        return points_2d.reshape(-1, 2)
