"""
Face detection module using MediaPipe Face Mesh.
Handles facial landmark detection and processing.
"""

import mediapipe as mp
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from . import config


@dataclass
class FaceLandmarks:
    """Container for facial landmarks data."""
    landmarks: List[Tuple[int, int]]  # List of (x, y) coordinates
    raw_landmarks: any  # Raw MediaPipe landmarks object


class FaceDetector:
    """Detects faces and extracts facial landmarks using MediaPipe Face Mesh."""

    def __init__(
        self,
        max_faces: int = config.MAX_NUM_FACES,
        min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE
    ):
        """
        Initialize the face detector.

        Args:
            max_faces: Maximum number of faces to detect
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
        """
        self.max_faces = max_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=self.max_faces,
            refine_landmarks=True,  # Enable iris landmarks
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )

    def detect(self, frame: np.ndarray) -> List[FaceLandmarks]:
        """
        Detect faces and extract landmarks from a frame.

        Args:
            frame: Input image frame (BGR format from OpenCV)

        Returns:
            List of FaceLandmarks objects (empty if no faces detected)
        """
        # Convert BGR to RGB (MediaPipe uses RGB)
        rgb_frame = frame[:, :, ::-1]

        # Process the frame
        results = self.face_mesh.process(rgb_frame)

        # Check if any face was detected
        if not results.multi_face_landmarks:
            return []

        # Process all detected faces
        all_faces = []
        height, width = frame.shape[:2]

        for face_landmarks in results.multi_face_landmarks:
            # Convert normalized landmarks to pixel coordinates
            landmarks = self._normalize_landmarks(face_landmarks, width, height)
            all_faces.append(FaceLandmarks(
                landmarks=landmarks,
                raw_landmarks=face_landmarks
            ))

        return all_faces

    def _normalize_landmarks(
        self,
        face_landmarks,
        width: int,
        height: int
    ) -> List[Tuple[int, int]]:
        """
        Convert normalized landmarks to pixel coordinates.

        Args:
            face_landmarks: MediaPipe face landmarks object
            width: Frame width
            height: Frame height

        Returns:
            List of (x, y) pixel coordinates
        """
        landmarks = []

        for landmark in face_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            landmarks.append((x, y))

        return landmarks

    def get_landmark_point(
        self,
        face_landmarks: FaceLandmarks,
        index: int
    ) -> Optional[Tuple[int, int]]:
        """
        Get a specific landmark point by index.

        Args:
            face_landmarks: FaceLandmarks object
            index: Landmark index

        Returns:
            (x, y) coordinate or None if index is invalid
        """
        if index < 0 or index >= len(face_landmarks.landmarks):
            return None

        return face_landmarks.landmarks[index]

    def get_landmarks_center(
        self,
        face_landmarks: FaceLandmarks,
        indices: List[int]
    ) -> Tuple[int, int]:
        """
        Calculate the center point of multiple landmarks.

        Args:
            face_landmarks: FaceLandmarks object
            indices: List of landmark indices

        Returns:
            (x, y) center coordinate
        """
        points = [
            self.get_landmark_point(face_landmarks, idx)
            for idx in indices
        ]

        # Filter out None values
        valid_points = [p for p in points if p is not None]

        if not valid_points:
            return (0, 0)

        # Calculate average position
        avg_x = sum(p[0] for p in valid_points) // len(valid_points)
        avg_y = sum(p[1] for p in valid_points) // len(valid_points)

        return (avg_x, avg_y)

    def release(self) -> None:
        """Release MediaPipe resources."""
        try:
            if self.face_mesh:
                self.face_mesh.close()
        except ValueError:
            pass  # Already closed

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False
