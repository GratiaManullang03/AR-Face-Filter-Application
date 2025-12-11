"""
Utility functions for AR Face Filter application.
"""

import cv2
import numpy as np
from typing import Optional, Union, List, Tuple

def create_texture_landmarks_from_image(texture_path: str) -> Optional[np.ndarray]:
    """
    Detect facial landmarks in a texture image using MediaPipe.
    
    Args:
        texture_path: Path to texture image

    Returns:
        Nx2 array of landmark coordinates or None if detection fails
    """
    import mediapipe as mp

    # Load texture
    texture = cv2.imread(texture_path)
    if texture is None:
        print(f"Error: Could not load texture from {texture_path}")
        return None

    h, w = texture.shape[:2]

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        # Convert BGR to RGB
        rgb = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            print("Warning: No face detected in texture")
            return None

        # Extract landmarks
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = np.zeros((len(face_landmarks.landmark), 2), dtype=np.float32)

        for i, landmark in enumerate(face_landmarks.landmark):
            landmarks[i, 0] = landmark.x * w
            landmarks[i, 1] = landmark.y * h

        return landmarks


class LandmarkStabilizer:
    """
    Stabilizes facial landmarks using Exponential Moving Average (EMA) 
    to reduce jitter.
    """
    def __init__(self, alpha: float = 0.6):
        """
        Args:
            alpha: Smoothing factor (0.0 to 1.0).
                   Lower = smoother but more lag (ghosting).
                   Higher = more responsive but more jitter.
                   0.6 is a good balance for 30fps.
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