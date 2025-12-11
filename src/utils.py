"""
Utility functions for AR Face Filter application.
"""

import cv2
import numpy as np
from typing import Optional

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
