"""
Face Mesh Texture Rendering module - FIXED VERSION.
Implements proper UV mapping and triangle-based affine warping.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from . import config


class MeshRenderer:
    """
    Renders face textures using triangle-based affine transformations.
    Uses MediaPipe's face mesh topology for proper UV mapping.
    """

    def __init__(self, texture_image: np.ndarray, debug_mode: bool = False):
        """
        Initialize the mesh renderer with a texture image.

        Args:
            texture_image: Source texture image (BGR format)
            debug_mode: If True, render wireframe overlay for debugging
        """
        self.texture = texture_image.copy()
        self.debug_mode = debug_mode
        self.texture_h, self.texture_w = texture_image.shape[:2]

        # Source UV landmarks (normalized coordinates in texture space)
        self.texture_landmarks: Optional[np.ndarray] = None
        self.triangles: List[Tuple[int, int, int]] = config.FACE_MESH_TRIANGLES

        print(f"MeshRenderer initialized with {len(self.triangles)} triangles")

    def set_texture_landmarks_from_detection(
        self,
        texture_path: str
    ) -> bool:
        """
        Detect face landmarks in the texture image using MediaPipe.
        This gives us the UV coordinates for warping.

        Args:
            texture_path: Path to the texture image

        Returns:
            True if landmarks were detected successfully
        """
        import mediapipe as mp

        # Initialize MediaPipe Face Mesh for static images
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            # Load and process the texture image
            texture = cv2.imread(texture_path)
            if texture is None:
                print(f"Error: Could not load texture from {texture_path}")
                return False

            # Convert BGR to RGB
            rgb = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                print("Warning: No face detected in texture image")
                return False

            # Extract landmarks and convert to pixel coordinates
            face_landmarks = results.multi_face_landmarks[0]
            h, w = texture.shape[:2]

            self.texture_landmarks = np.zeros((len(face_landmarks.landmark), 2), dtype=np.float32)
            for i, landmark in enumerate(face_landmarks.landmark):
                self.texture_landmarks[i, 0] = landmark.x * w
                self.texture_landmarks[i, 1] = landmark.y * h

            print(f"Detected {len(self.texture_landmarks)} landmarks in texture")
            return True

    def set_texture_landmarks_manual(
        self,
        landmarks: np.ndarray
    ) -> None:
        """
        Manually set texture landmarks (for pre-computed UV maps).

        Args:
            landmarks: Nx2 array of pixel coordinates in texture image
        """
        self.texture_landmarks = landmarks.astype(np.float32)

    def render(
        self,
        frame: np.ndarray,
        face_landmarks: np.ndarray,
        opacity: float = 0.8
    ) -> np.ndarray:
        """
        Render the texture onto the frame using detected face landmarks.

        Args:
            frame: Target video frame (BGR)
            face_landmarks: Nx2 array of detected face landmarks
            opacity: Blending opacity (0.0 to 1.0)

        Returns:
            Frame with texture applied
        """
        if self.texture_landmarks is None:
            raise ValueError("Texture landmarks not set. Call set_texture_landmarks_from_detection() first.")

        # Ensure face_landmarks is float32
        dst_landmarks = face_landmarks.astype(np.float32)

        # Create output frame
        output = frame.copy()

        # Warp each triangle
        for triangle_indices in self.triangles:
            self._warp_triangle(
                output,
                self.texture,
                self.texture_landmarks,
                dst_landmarks,
                triangle_indices
            )

        # Blend with original frame
        if opacity < 1.0:
            output = cv2.addWeighted(frame, 1.0 - opacity, output, opacity, 0)

        # Draw wireframe if in debug mode
        if self.debug_mode:
            output = self._draw_wireframe(output, dst_landmarks)

        return output

    def _warp_triangle(
        self,
        output_frame: np.ndarray,
        source_texture: np.ndarray,
        src_landmarks: np.ndarray,
        dst_landmarks: np.ndarray,
        triangle_indices: Tuple[int, int, int]
    ) -> None:
        """
        Warp a single triangle from source texture to destination frame.
        This is the core affine transformation logic.

        Args:
            output_frame: Destination frame (modified in-place)
            source_texture: Source texture image
            src_landmarks: Source landmark coordinates
            dst_landmarks: Destination landmark coordinates
            triangle_indices: Indices of the three vertices
        """
        # Get triangle vertices
        src_tri = np.float32([
            src_landmarks[triangle_indices[0]],
            src_landmarks[triangle_indices[1]],
            src_landmarks[triangle_indices[2]]
        ])

        dst_tri = np.float32([
            dst_landmarks[triangle_indices[0]],
            dst_landmarks[triangle_indices[1]],
            dst_landmarks[triangle_indices[2]]
        ])

        # Get bounding rectangles
        src_rect = cv2.boundingRect(src_tri)
        dst_rect = cv2.boundingRect(dst_tri)

        # Offset triangles to bounding box coordinates
        src_tri_cropped = src_tri - np.array([[src_rect[0], src_rect[1]]], dtype=np.float32)
        dst_tri_cropped = dst_tri - np.array([[dst_rect[0], dst_rect[1]]], dtype=np.float32)

        # Extract source region
        src_cropped = source_texture[
            src_rect[1]:src_rect[1] + src_rect[3],
            src_rect[0]:src_rect[0] + src_rect[2]
        ]

        if src_cropped.size == 0 or dst_rect[2] <= 0 or dst_rect[3] <= 0:
            return  # Skip invalid triangles

        # Create mask for destination triangle
        mask = np.zeros((dst_rect[3], dst_rect[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(dst_tri_cropped), (1.0, 1.0, 1.0), cv2.LINE_AA)

        # Calculate affine transform
        try:
            warp_mat = cv2.getAffineTransform(src_tri_cropped, dst_tri_cropped)
        except cv2.error:
            return  # Skip degenerate triangles

        # Warp the cropped source region
        warped = cv2.warpAffine(
            src_cropped,
            warp_mat,
            (dst_rect[2], dst_rect[3]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        # Apply mask
        warped = warped.astype(np.float32) * mask

        # Extract destination region
        dst_y1 = max(0, dst_rect[1])
        dst_y2 = min(output_frame.shape[0], dst_rect[1] + dst_rect[3])
        dst_x1 = max(0, dst_rect[0])
        dst_x2 = min(output_frame.shape[1], dst_rect[0] + dst_rect[2])

        if dst_y1 >= dst_y2 or dst_x1 >= dst_x2:
            return  # Out of bounds

        # Calculate crop offsets if triangle is partially outside frame
        crop_y1 = dst_y1 - dst_rect[1]
        crop_y2 = crop_y1 + (dst_y2 - dst_y1)
        crop_x1 = dst_x1 - dst_rect[0]
        crop_x2 = crop_x1 + (dst_x2 - dst_x1)

        # Get regions
        warped_region = warped[crop_y1:crop_y2, crop_x1:crop_x2]
        mask_region = mask[crop_y1:crop_y2, crop_x1:crop_x2]
        dst_region = output_frame[dst_y1:dst_y2, dst_x1:dst_x2]

        # Ensure dimensions match
        if warped_region.shape[:2] != dst_region.shape[:2]:
            return

        # Blend using mask
        # Formula: output = mask * warped + (1 - mask) * original
        output_frame[dst_y1:dst_y2, dst_x1:dst_x2] = (
            mask_region * warped_region +
            (1.0 - mask_region) * dst_region.astype(np.float32)
        ).astype(np.uint8)

    def _draw_wireframe(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray
    ) -> np.ndarray:
        """
        Draw wireframe overlay for debugging.

        Args:
            frame: Frame to draw on
            landmarks: Face landmarks

        Returns:
            Frame with wireframe overlay
        """
        for triangle_indices in self.triangles:
            pt1 = tuple(landmarks[triangle_indices[0]].astype(int))
            pt2 = tuple(landmarks[triangle_indices[1]].astype(int))
            pt3 = tuple(landmarks[triangle_indices[2]].astype(int))

            cv2.line(frame, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.line(frame, pt2, pt3, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.line(frame, pt3, pt1, (0, 255, 0), 1, cv2.LINE_AA)

        return frame


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
