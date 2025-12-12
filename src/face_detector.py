"""
Face detection module using MediaPipe Face Landmarker (Latest API).
Handles facial landmark detection and processing with proper video mode support.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from . import config
from .utils import LandmarkStabilizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FaceLandmarks:
    """Container for facial landmarks data."""
    landmarks: List[Tuple[int, int]]  # List of (x, y) coordinates
    landmarks_3d: List[Tuple[float, float, float]]  # List of (x, y, z) normalized coordinates
    raw_landmarks: any  # Raw MediaPipe landmarks object


class FaceDetector:
    """
    Detects faces and extracts facial landmarks using MediaPipe Face Landmarker (Latest API).

    Uses VIDEO mode for optimal real-time webcam processing with timestamp tracking.
    Supports proper resource management via context manager pattern.
    """

    def __init__(
        self,
        max_faces: int = config.MAX_NUM_FACES,
        min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE,
        model_path: Optional[str] = None
    ):
        """
        Initialize the face detector with Face Landmarker API.

        Args:
            max_faces: Maximum number of faces to detect
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
            model_path: Path to face_landmarker.task model file
        """
        self.max_faces = max_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Determine model path
        if model_path is None:
            model_path = str(Path(__file__).parent.parent / "models" / "face_landmarker.task")

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Face Landmarker model not found at {model_path}. "
                f"Download it from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )

        logger.info(f"Loading Face Landmarker model from: {model_path}")

        # Initialize Face Landmarker with VIDEO mode for real-time webcam
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,  # VIDEO mode for webcam with timestamp
            num_faces=self.max_faces,
            min_face_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_face_blendshapes=False,  # Not needed for this app
            output_facial_transformation_matrixes=False  # Not needed for this app
        )

        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        # Timestamp tracking for VIDEO mode
        self.start_time = time.time()
        self.frame_count = 0

        # Initialize Stabilizer
        # NOTE: MediaPipe has built-in smoothing when num_faces=1
        # We keep custom stabilizer for consistency and additional control
        self.stabilizer = LandmarkStabilizer(alpha=0.7)

        logger.info(f"Face Landmarker initialized (VIDEO mode, max_faces={max_faces})")

    def detect(self, frame: np.ndarray) -> List[FaceLandmarks]:
        """
        Detect faces and extract landmarks from a frame using VIDEO mode with timestamp.

        Args:
            frame: Input image frame (BGR format from OpenCV)

        Returns:
            List of FaceLandmarks objects (empty if no faces detected)
        """
        try:
            # Convert BGR to RGB (MediaPipe uses RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Generate timestamp in milliseconds for VIDEO mode
            # MediaPipe requires monotonically increasing timestamps
            self.frame_count += 1
            timestamp_ms = int((time.time() - self.start_time) * 1000)

            # Convert numpy array to MediaPipe Image
            # Make sure the array is contiguous in memory
            rgb_frame_contiguous = np.ascontiguousarray(rgb_frame)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame_contiguous)

            # Process the frame with timestamp (VIDEO mode)
            results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            # Check if any face was detected
            if not results.face_landmarks:
                self.stabilizer.reset()  # Reset smoothing if face lost
                return []

            # Process all detected faces
            all_faces = []
            height, width = frame.shape[:2]

            # NOTE: Smoothing works best for single face tracking.
            # If tracking multiple faces, we would need a dict of stabilizers mapped to face IDs.
            # For this prototype, we stabilize only the first face found.

            for i, face_landmarks in enumerate(results.face_landmarks):
                # Convert normalized landmarks to pixel coordinates
                raw_landmarks_np = self._normalize_landmarks_np(face_landmarks, width, height)
                raw_landmarks_3d_np = self._extract_3d_landmarks_np(face_landmarks)

                # Apply stabilization (only to first face to avoid ID swapping issues)
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

        except Exception as e:
            logger.error(f"Error during face detection: {e}")
            return []

    def _normalize_landmarks_np(
        self,
        face_landmarks,
        width: int,
        height: int
    ) -> np.ndarray:
        """
        Convert normalized landmarks to pixel coordinates (NumPy).

        Args:
            face_landmarks: MediaPipe NormalizedLandmarkList
            width: Frame width in pixels
            height: Frame height in pixels

        Returns:
            Nx2 array of (x, y) pixel coordinates
        """
        # Face Landmarker API returns landmarks as NormalizedLandmark objects
        coords = np.array([(lm.x * width, lm.y * height) for lm in face_landmarks], dtype=np.float32)
        return coords

    def _extract_3d_landmarks_np(
        self,
        face_landmarks
    ) -> np.ndarray:
        """
        Extract 3D landmarks (NumPy).

        Args:
            face_landmarks: MediaPipe NormalizedLandmarkList

        Returns:
            Nx3 array of (x, y, z) normalized coordinates
        """
        # Face Landmarker API returns landmarks with z-depth
        coords = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks], dtype=np.float32)
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
            if hasattr(self, 'landmarker') and self.landmarker:
                self.landmarker.close()
                logger.info("Face Landmarker resources released")
        except Exception as e:
            logger.warning(f"Error releasing Face Landmarker: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic resource cleanup."""
        self.release()
        return False