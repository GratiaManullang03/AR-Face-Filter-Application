"""
Utility functions for AR Face Filter application.
Updated to use MediaPipe Face Landmarker API.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_texture_landmarks_from_image(
    texture_path: str,
    model_path: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Detect facial landmarks in a texture image using MediaPipe Face Landmarker API.

    Args:
        texture_path: Path to texture image
        model_path: Path to face_landmarker.task model file (optional)

    Returns:
        Nx2 array of landmark coordinates or None if detection fails
    """
    try:
        import mediapipe as mp

        # Determine model path
        if model_path is None:
            model_path = str(Path(__file__).parent.parent / "models" / "face_landmarker.task")

        if not Path(model_path).exists():
            logger.error(f"Face Landmarker model not found at {model_path}")
            return None

        # Load texture
        texture = cv2.imread(texture_path)
        if texture is None:
            logger.error(f"Could not load texture from {texture_path}")
            return None

        h, w = texture.shape[:2]

        # Initialize Face Landmarker with IMAGE mode
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Convert BGR to RGB
        rgb = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Detect landmarks
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            results = landmarker.detect(mp_image)

            if not results.face_landmarks:
                logger.warning("No face detected in texture")
                return None

            # Extract landmarks
            face_landmarks = results.face_landmarks[0]
            landmarks = np.zeros((len(face_landmarks), 2), dtype=np.float32)

            for i, landmark in enumerate(face_landmarks):
                landmarks[i, 0] = landmark.x * w
                landmarks[i, 1] = landmark.y * h

            return landmarks

    except Exception as e:
        logger.error(f"Error detecting landmarks in texture: {e}")
        return None


class LandmarkStabilizer:
    """
    Stabilizes facial landmarks using Exponential Moving Average (EMA) 
    to reduce jitter.
    """
    def __init__(self, alpha: float = 0.3):
        """
        Args:
            alpha: Smoothing factor (0.0 to 1.0).
                   Lower = smoother but more lag (ghosting).
                   Higher = more responsive but more jitter.
                   0.3 is a good balance for 30fps.
        """
        self.alpha = alpha
        self.prev_landmarks: Optional[np.ndarray] = None
        self.prev_landmarks_3d: Optional[np.ndarray] = None

    def update(self, 
               current_landmarks: np.ndarray, 
               current_landmarks_3d: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply smoothing to the current landmarks.
        """
        # Initialize if first frame
        if self.prev_landmarks is None:
            self.prev_landmarks = current_landmarks.copy()
            self.prev_landmarks_3d = current_landmarks_3d.copy()
            return current_landmarks, current_landmarks_3d

        # Apply EMA formula: 
        # Smoothed = Alpha * Current + (1 - Alpha) * Previous
        
        # 2D Landmarks
        smoothed = (self.alpha * current_landmarks + 
                   (1 - self.alpha) * self.prev_landmarks)
        
        # 3D Landmarks
        smoothed_3d = (self.alpha * current_landmarks_3d + 
                      (1 - self.alpha) * self.prev_landmarks_3d)

        # Update state
        self.prev_landmarks = smoothed
        self.prev_landmarks_3d = smoothed_3d

        return smoothed, smoothed_3d

    def reset(self):
        """Reset stabilizer state (e.g., when face is lost)."""
        self.prev_landmarks = None
        self.prev_landmarks_3d = None