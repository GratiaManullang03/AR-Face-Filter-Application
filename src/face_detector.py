"""
Face detection module using MediaPipe Face Mesh.
Handles facial landmark detection and processing.
"""

import mediapipe as mp
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from . import config
from .utils import LandmarkStabilizer


@dataclass
class FaceLandmarks:
    """Container for facial landmarks data."""
    landmarks: List[Tuple[int, int]]  # List of (x, y) coordinates
    landmarks_3d: List[Tuple[float, float, float]]  # List of (x, y, z) normalized coordinates
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

        # Initialize Stabilizer (One per face ideally, but for simplicity assuming 1 primary face)
        # Increased alpha for responsiveness (less lag)
        self.stabilizer = LandmarkStabilizer(alpha=0.7)

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
            self.stabilizer.reset() # Reset smoothing if face lost
            return []

        # Process all detected faces
        all_faces = []
        height, width = frame.shape[:2]

        # NOTE: Currently smoothing works best for single face tracking.
        # If tracking multiple faces, we would need a dict of stabilizers mapped to face IDs.
        # For this prototype, we just stabilize the first face found.
        
        for i, face_landmarks in enumerate(results.multi_face_landmarks):
            # Convert normalized landmarks to pixel coordinates
            # Get raw numpy arrays for stabilization
            raw_landmarks_np = self._normalize_landmarks_np(face_landmarks, width, height)
            raw_landmarks_3d_np = self._extract_3d_landmarks_np(face_landmarks)

            # Apply stabilization (only to first face for now to avoid ID swapping issues)
            if i == 0:
                smooth_landmarks, smooth_landmarks_3d = self.stabilizer.update(
                    raw_landmarks_np, raw_landmarks_3d_np
                )
            else:
                smooth_landmarks = raw_landmarks_np
                smooth_landmarks_3d = raw_landmarks_3d_np

            # Convert back to list of tuples for compatibility
            landmarks_list = [tuple(map(int, pt)) for pt in smooth_landmarks]
            landmarks_3d_list = [tuple(map(float, pt)) for pt in smooth_landmarks_3d]

            all_faces.append(FaceLandmarks(
                landmarks=landmarks_list,
                landmarks_3d=landmarks_3d_list,
                raw_landmarks=face_landmarks
            ))

        return all_faces

    def _normalize_landmarks_np(
        self,
        face_landmarks,
        width: int,
        height: int
    ) -> np.ndarray:
        """Convert normalized landmarks to pixel coordinates (NumPy)."""
        coords = np.array([(lm.x * width, lm.y * height) for lm in face_landmarks.landmark], dtype=np.float32)
        return coords

    def _extract_3d_landmarks_np(
        self,
        face_landmarks
    ) -> np.ndarray:
        """Extract 3D landmarks (NumPy)."""
        coords = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark], dtype=np.float32)
        return coords

    def get_landmark_point(
        self,
        face_landmarks: FaceLandmarks,
        index: int
    ) -> Optional[Tuple[int, int]]:
        """Get a specific landmark point by index."""
        if index < 0 or index >= len(face_landmarks.landmarks):
            return None
        return face_landmarks.landmarks[index]

    def get_landmarks_center(
        self,
        face_landmarks: FaceLandmarks,
        indices: List[int]
    ) -> Tuple[int, int]:
        """Calculate the center point of multiple landmarks."""
        points = [
            self.get_landmark_point(face_landmarks, idx)
            for idx in indices
        ]
        valid_points = [p for p in points if p is not None]

        if not valid_points:
            return (0, 0)

        avg_x = sum(p[0] for p in valid_points) // len(valid_points)
        avg_y = sum(p[1] for p in valid_points) // len(valid_points)

        return (avg_x, avg_y)

    def get_landmarks_as_array(
        self,
        face_landmarks: FaceLandmarks
    ) -> np.ndarray:
        """Get all landmarks as a NumPy array for mesh rendering."""
        return np.array(face_landmarks.landmarks, dtype=np.float32)

    def release(self) -> None:
        """Release MediaPipe resources."""
        try:
            if self.face_mesh:
                self.face_mesh.close()
        except ValueError:
            pass  # Already closed

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False