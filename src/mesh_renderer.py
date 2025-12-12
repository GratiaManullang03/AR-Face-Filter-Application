"""
Face Mesh Texture Rendering module - UPDATED VERSION.
Implements proper UV mapping and triangle-based affine warping.
Compatible with MediaPipe Face Landmarker API.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from . import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        self.corrected_triangles: Optional[List[Tuple[int, int, int]]] = None

        print(f"MeshRenderer initialized with {len(self.triangles)} triangles")

    def calibrate_winding_order(self, landmarks_3d: np.ndarray) -> bool:
        """
        Calibrate triangle winding order based on the first detected face.
        Ensures all triangles are wound Counter-Clockwise (CCW) relative to the camera.
        This enables accurate backface culling.
        
        Args:
            landmarks_3d: Nx3 array of 3D landmarks (x, y, z) from a frontal face
            
        Returns:
            True if calibration was successful (face was frontal enough), False otherwise.
        """
        if self.corrected_triangles is not None:
            return True

        # Check if face is frontal enough for calibration
        # Use Outer Eye landmarks to check for Yaw
        # Left Eye Outer: 33, Right Eye Outer: 263 (detected as 362 in config, checking standard MP)
        # Using indices from config would be safer but let's assume standard MP topology here or use passed indices.
        # Let's use indices 33 (Left) and 263 (Right) as they are standard "outer" approx.
        # Actually, let's look at the Z difference.
        
        # Standard MediaPipe indices:
        # 33: Left Eye Outer corner
        # 263: Right Eye Outer corner
        
        left_eye_z = landmarks_3d[33][2]
        right_eye_z = landmarks_3d[263][2]
        
        # Calculate Yaw-ish metric (difference in depth)
        yaw_diff = abs(left_eye_z - right_eye_z)
        
        # Threshold: 0.1 is usually a good heuristic for "mostly frontal"
        if yaw_diff > 0.1:
            # Face is rotated, do not calibrate yet
            return False

        print("Calibrating mesh winding order for backface culling...")
        corrected = []
        
        flipped_count = 0
        
        for tri in self.triangles:
            v1 = landmarks_3d[tri[0]]
            v2 = landmarks_3d[tri[1]]
            v3 = landmarks_3d[tri[2]]
            
            # Calculate normal
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = np.cross(edge1, edge2)
            
            # We assume a frontal face. Normals should point TOWARDS camera (Negative Z).
            # If Normal.z > 0, the triangle is facing away (or wound CW).
            # We flip it to make it face the camera.
            if normal[2] > 0:
                # Flip winding: (0, 1, 2) -> (0, 2, 1)
                new_tri = (tri[0], tri[2], tri[1])
                corrected.append(new_tri)
                flipped_count += 1
            else:
                corrected.append(tri)
        
        self.corrected_triangles = corrected
        print(f"Mesh calibration complete: Flipped {flipped_count}/{len(self.triangles)} triangles")
        return True

    def set_texture_landmarks_from_detection(
        self,
        texture_path: str,
        model_path: Optional[str] = None
    ) -> bool:
        """
        Detect face landmarks in the texture image using MediaPipe Face Landmarker API.
        This gives us the UV coordinates for warping.

        Args:
            texture_path: Path to the texture image
            model_path: Path to face_landmarker.task model file

        Returns:
            True if landmarks were detected successfully
        """
        try:
            # Determine model path
            if model_path is None:
                model_path = str(Path(__file__).parent.parent / "models" / "face_landmarker.task")

            if not Path(model_path).exists():
                logger.error(f"Face Landmarker model not found at {model_path}")
                return False

            # Initialize Face Landmarker with IMAGE mode for static texture
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,  # IMAGE mode for static texture
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            # Load and process the texture image
            texture = cv2.imread(texture_path)
            if texture is None:
                logger.error(f"Could not load texture from {texture_path}")
                return False

            # Convert BGR to RGB
            rgb = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
            h, w = texture.shape[:2]

            # Convert to MediaPipe Image
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Create landmarker and detect
            with vision.FaceLandmarker.create_from_options(options) as landmarker:
                results = landmarker.detect(mp_image)

                if not results.face_landmarks:
                    logger.warning("No face detected in texture image")
                    return False

                # Extract landmarks and convert to pixel coordinates
                face_landmarks = results.face_landmarks[0]

                self.texture_landmarks = np.zeros((len(face_landmarks), 2), dtype=np.float32)
                for i, landmark in enumerate(face_landmarks):
                    self.texture_landmarks[i, 0] = landmark.x * w
                    self.texture_landmarks[i, 1] = landmark.y * h

                logger.info(f"Detected {len(self.texture_landmarks)} landmarks in texture")
                return True

        except Exception as e:
            logger.error(f"Error detecting landmarks in texture: {e}")
            return False

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
        landmarks_3d: Optional[np.ndarray] = None,
        opacity: float = 0.8
    ) -> np.ndarray:
        """
        Render the texture onto the frame using detected face landmarks.
        """
        # PRIORITY 1: Use Canonical UVs if defined in config
        if hasattr(config, 'CANONICAL_FACE_MESH_UV') and config.CANONICAL_FACE_MESH_UV is not None:
            # If we haven't converted them to texture space yet, do it now
            if self.texture_landmarks is None:
                print("Using Canonical UVs from config...")
                uvs = np.array(config.CANONICAL_FACE_MESH_UV, dtype=np.float32)
                # Scale UVs (0.0-1.0) to Texture Size
                self.texture_landmarks = uvs * np.array([self.texture_w, self.texture_h], dtype=np.float32)

        # PRIORITY 2: Use detected landmarks (Fallback)
        if self.texture_landmarks is None:
             raise ValueError("Texture landmarks not set. Call set_texture_landmarks_from_detection() first or define CANONICAL_FACE_MESH_UV.")

        # Ensure face_landmarks is float32
        dst_landmarks = face_landmarks.astype(np.float32)

        # Store 3D landmarks for culling
        self.landmarks_3d = landmarks_3d

        # Create output frame
        output = frame.copy()

        # Use corrected topology if available, otherwise raw triangles
        triangles_to_render = self.corrected_triangles if self.corrected_triangles is not None else self.triangles

        # ============================================================
        # Z-SORTING (PAINTER'S ALGORITHM)
        # ============================================================
        # We must draw triangles from back to front to handle occlusion correctly.
        # Otherwise, a far triangle (e.g. cheek) drawn later will cover a near triangle (e.g. nose).
        
        if landmarks_3d is not None:
            # Calculate depth (average Z) for each triangle
            # MediaPipe Z: Smaller = Closer, Larger = Further.
            # We want to draw Further (Large Z) first, Closer (Small Z) last.
            # So we sort by Z descending.
            
            triangle_depths = []
            for i, tri in enumerate(triangles_to_render):
                v1_z = landmarks_3d[tri[0]][2]
                v2_z = landmarks_3d[tri[1]][2]
                v3_z = landmarks_3d[tri[2]][2]
                avg_z = (v1_z + v2_z + v3_z) / 3.0
                triangle_depths.append((avg_z, tri))
            
            # Sort by depth descending
            triangle_depths.sort(key=lambda x: x[0], reverse=True)
            
            # Extract sorted triangles
            sorted_triangles = [t[1] for t in triangle_depths]
            
            # Use sorted list
            final_triangles_list = sorted_triangles
        else:
            # Fallback if no 3D landmarks
            final_triangles_list = triangles_to_render

        # Warp each triangle
        for triangle_indices in final_triangles_list:
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
            output = self._draw_wireframe(output, dst_landmarks, landmarks_3d)

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
        This is the core affine transformation logic with backface culling.

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

        # ============================================================
        # BACKFACE CULLING & LIGHTING
        # ============================================================
        # Use 3D landmarks with Z-depth for accurate culling and lighting
        brightness = 1.0
        
        if hasattr(self, 'landmarks_3d') and self.landmarks_3d is not None:
            # Get 3D vertices for this triangle
            v1 = self.landmarks_3d[triangle_indices[0]]
            v2 = self.landmarks_3d[triangle_indices[1]]
            v3 = self.landmarks_3d[triangle_indices[2]]

            # Calculate surface normal using cross product
            # edge1 = v2 - v1
            # edge2 = v3 - v1
            # normal = edge1 × edge2
            edge1 = np.array([v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]])
            edge2 = np.array([v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]])

            # Cross product gives us the normal vector
            normal = np.cross(edge1, edge2)
            
            # Normalize normal vector for lighting
            norm_len = np.linalg.norm(normal)
            if norm_len > 0:
                normal_unit = normal / norm_len
            else:
                normal_unit = np.array([0, 0, -1]) # Fallback

            # Camera is looking down negative Z axis
            # After calibration, all visible triangles should have normal.z < 0
            # If normal.z > 0, it's a backface -> CULL
            if normal[2] > 0:
                return
            
            # --- LIGHTING CALCULATION ---
            # Simple directional light from camera (0, 0, -1)
            # Dot product of Normal and LightDir
            # LightDir is (0, 0, -1). Normal points to camera (negative Z).
            # So we want alignment between Normal and (0, 0, -1).
            # dot = nx*0 + ny*0 + nz*(-1) = -nz
            
            # Since normal.z is negative for visible triangles, -normal.z is positive.
            intensity = -normal_unit[2] 
            intensity = max(0.0, min(1.0, intensity))
            
            # Ambient + Diffuse
            # Make lighting very subtle to avoid "faceted" low-poly look
            # Ambient 0.85, Diffuse 0.15
            brightness = 0.85 + 0.15 * intensity
        
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
        
        # ============================================================
        # SEAM FIX: Overdraw edges to prevent gaps between triangles
        # ============================================================
        # Draw the triangle contour with a small thickness to cover anti-aliasing gaps
        cv2.polylines(mask, [np.int32(dst_tri_cropped)], True, (1.0, 1.0, 1.0), 1, cv2.LINE_AA)

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

        # Convert to float for processing
        warped = warped.astype(np.float32)

        # Apply Lighting
        if brightness != 1.0:
            warped *= brightness

        # Apply mask
        warped *= mask

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
        landmarks: np.ndarray,
        landmarks_3d: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Draw wireframe overlay for debugging.
        Green = visible triangles, Red = culled (backface) triangles.

        Args:
            frame: Frame to draw on
            landmarks: Face landmarks (2D)
            landmarks_3d: Face landmarks (3D) for accurate culling visualization

        Returns:
            Frame with wireframe overlay
        """
        # Use corrected topology if available
        triangles_to_render = self.corrected_triangles if self.corrected_triangles is not None else self.triangles

        for triangle_indices in triangles_to_render:
            pt1 = tuple(landmarks[triangle_indices[0]].astype(int))
            pt2 = tuple(landmarks[triangle_indices[1]].astype(int))
            pt3 = tuple(landmarks[triangle_indices[2]].astype(int))

            # Draw all triangles
            # Green = Frontface (visible)
            # Red = Backface (culled)
            
            color = (0, 255, 0) # Default Green
            
            if landmarks_3d is not None:
                v1 = landmarks_3d[triangle_indices[0]]
                v2 = landmarks_3d[triangle_indices[1]]
                v3 = landmarks_3d[triangle_indices[2]]
                
                edge1 = v2 - v1
                edge2 = v3 - v1
                normal = np.cross(edge1, edge2)
                
                # If normal.z > 0, it is backface (culled)
                if normal[2] > 0:
                    color = (0, 0, 255) # Red

            cv2.line(frame, pt1, pt2, color, 1, cv2.LINE_AA)
            cv2.line(frame, pt2, pt3, color, 1, cv2.LINE_AA)
            cv2.line(frame, pt3, pt1, color, 1, cv2.LINE_AA)

        return frame
