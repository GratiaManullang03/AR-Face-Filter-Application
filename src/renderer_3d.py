"""
3D Object Renderer with texture mapping and depth sorting.

Renders 3D OBJ models onto video frames with proper perspective projection,
texture mapping, backface culling, z-sorting, and view frustum culling.
"""

import logging
from typing import Optional, Tuple, List
import numpy as np
import cv2

from src.obj_loader import OBJModel, Material
from src.pose_estimator import FacePoseEstimator
from src.object_3d import Object3D

logger = logging.getLogger(__name__)


class Renderer3D:
    """
    Renders 3D objects onto 2D images with proper projection and texturing.

    Supports:
    - Perspective projection using face pose
    - Texture mapping with UV coordinates
    - Solid color rendering for untextured materials
    - Z-sorting for correct occlusion
    - Backface culling
    """

    def __init__(self, image_width: int, image_height: int):
        """
        Initialize 3D renderer.

        Args:
            image_width: Width of output image
            image_height: Height of output image
        """
        self.image_width = image_width
        self.image_height = image_height
        self.pose_estimator = FacePoseEstimator(image_width, image_height)

    def render_object(
        self,
        frame: np.ndarray,
        obj3d: Object3D,
        landmarks: list[Tuple[float, float]],
        landmarks_3d: Optional[list[Tuple[float, float, float]]] = None
    ) -> np.ndarray:
        """
        Render a 3D object onto the frame based on face landmarks.

        Args:
            frame: Input frame (will not be modified)
            obj3d: 3D object to render
            landmarks: 2D facial landmarks (x, y)
            landmarks_3d: Optional 3D landmarks for better pose estimation

        Returns:
            Frame with rendered 3D object
        """
        if not obj3d.is_loaded or not obj3d.model:
            return frame

        # Estimate face pose
        pose_result = self.pose_estimator.estimate_pose(landmarks)
        if pose_result is None:
            logger.warning("Failed to estimate pose")
            return frame

        rotation_vec, translation_vec = pose_result

        # Get anchor position for object placement
        anchor_pos = obj3d.get_anchor_position(landmarks)
        if anchor_pos is None:
            return frame

        # Calculate face width for scaling
        # Use distance between left and right eye outer corners
        try:
            left_eye = landmarks[33]  # Left eye outer
            right_eye = landmarks[263]  # Right eye outer
            face_width = np.linalg.norm(
                np.array(left_eye) - np.array(right_eye)
            )
        except (IndexError, TypeError):
            face_width = 100.0  # Default fallback

        scale = obj3d.calculate_scale(landmarks, face_width)

        # Render the model
        output = frame.copy()
        output = self._render_model(
            output,
            obj3d.model,
            rotation_vec,
            translation_vec,
            anchor_pos,
            scale,
            obj3d.config
        )

        return output

    def _render_model(
        self,
        frame: np.ndarray,
        model: OBJModel,
        rotation_vec: np.ndarray,
        translation_vec: np.ndarray,
        anchor_pos: Tuple[float, float],
        scale: float,
        config
    ) -> np.ndarray:
        """
        Render OBJ model with proper transformation and projection.

        Args:
            frame: Frame to render onto
            model: OBJ model to render
            rotation_vec: Face rotation vector
            translation_vec: Face translation vector
            anchor_pos: 2D anchor position for object
            scale: Scale factor
            config: Object3D configuration

        Returns:
            Frame with rendered model
        """
        # Get vertices as numpy array
        vertices = np.array(model.vertices, dtype=np.float32)

        if len(vertices) == 0:
            return frame

        # Apply object-space transformations
        # 1. Scale
        vertices = vertices * scale

        # 2. Apply config offsets
        vertices[:, 0] += config.offset_x
        vertices[:, 1] += config.offset_y
        vertices[:, 2] += config.offset_z

        # 3. Apply config rotation offsets
        if config.rotation_x != 0 or config.rotation_y != 0 or config.rotation_z != 0:
            vertices = self._apply_rotation_offsets(
                vertices,
                config.rotation_x,
                config.rotation_y,
                config.rotation_z
            )

        # Project vertices to 2D
        try:
            projected_2d = self.pose_estimator.project_points(
                vertices,
                rotation_vec,
                translation_vec
            )
        except Exception as e:
            logger.warning(f"Projection failed: {e}")
            return frame

        # Calculate Z-depth for each vertex (for z-sorting)
        rotation_mat = self.pose_estimator.get_rotation_matrix(rotation_vec)
        vertices_camera = np.dot(vertices, rotation_mat.T) + translation_vec.T
        z_depths = vertices_camera[:, 2]

        # Render faces with z-sorting and frustum culling
        frame = self._render_faces(
            frame,
            model,
            projected_2d,
            z_depths
        )

        return frame

    def _apply_rotation_offsets(
        self,
        vertices: np.ndarray,
        rx: float,
        ry: float,
        rz: float
    ) -> np.ndarray:
        """
        Apply rotation offsets to vertices.

        Args:
            vertices: Array of vertices (N, 3)
            rx: Rotation around X axis in degrees
            ry: Rotation around Y axis in degrees
            rz: Rotation around Z axis in degrees

        Returns:
            Rotated vertices
        """
        # Convert to radians
        rx_rad = np.radians(rx)
        ry_rad = np.radians(ry)
        rz_rad = np.radians(rz)

        # Rotation matrices
        # X-axis rotation
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx_rad), -np.sin(rx_rad)],
            [0, np.sin(rx_rad), np.cos(rx_rad)]
        ])

        # Y-axis rotation
        Ry = np.array([
            [np.cos(ry_rad), 0, np.sin(ry_rad)],
            [0, 1, 0],
            [-np.sin(ry_rad), 0, np.cos(ry_rad)]
        ])

        # Z-axis rotation
        Rz = np.array([
            [np.cos(rz_rad), -np.sin(rz_rad), 0],
            [np.sin(rz_rad), np.cos(rz_rad), 0],
            [0, 0, 1]
        ])

        # Combined rotation: Rz * Ry * Rx
        R = Rz @ Ry @ Rx

        return np.dot(vertices, R.T)

    def _is_face_in_frustum(
        self,
        face_vertices_2d: List[Tuple[float, float]],
        margin: float = 100.0
    ) -> bool:
        """
        Check if a face is within view frustum (with margin).

        Args:
            face_vertices_2d: List of 2D projected vertices for the face
            margin: Margin beyond screen bounds to consider (pixels)

        Returns:
            True if face could be visible, False if definitely outside
        """
        # Get bounding box of face
        xs = [v[0] for v in face_vertices_2d]
        ys = [v[1] for v in face_vertices_2d]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Check if completely outside screen bounds (with margin)
        if max_x < -margin or min_x > self.image_width + margin:
            return False
        if max_y < -margin or min_y > self.image_height + margin:
            return False

        return True

    def _render_faces(
        self,
        frame: np.ndarray,
        model: OBJModel,
        projected_2d: np.ndarray,
        z_depths: np.ndarray,
        subsample: int = 1
    ) -> np.ndarray:
        """
        Render all faces with z-sorting, backface culling, and frustum culling.

        Args:
            frame: Frame to render onto
            model: OBJ model
            projected_2d: Projected 2D vertices
            z_depths: Z-depth for each vertex
            subsample: Only render 1 out of N faces (DEPRECATED - use LOD instead)

        Returns:
            Frame with rendered faces
        """
        # Prepare face data with average z-depth for sorting
        face_data = []
        culled_count = 0

        for i, face in enumerate(model.faces):
            # Legacy subsample support (deprecated, use LOD instead)
            if subsample > 1 and i % subsample != 0:
                continue

            if len(face['vertices']) < 3:
                continue  # Skip degenerate faces

            # Get vertex indices
            v_indices = face['vertices']

            # View frustum culling - skip faces completely outside screen
            face_verts_2d = [
                (projected_2d[idx][0], projected_2d[idx][1])
                for idx in v_indices
            ]
            if not self._is_face_in_frustum(face_verts_2d):
                culled_count += 1
                continue

            # Calculate average Z-depth for this face
            avg_z = np.mean([z_depths[idx] for idx in v_indices])

            # Skip faces behind camera
            if avg_z <= 0:
                culled_count += 1
                continue

            face_data.append({
                'face': face,
                'avg_z': avg_z
            })

        # Log culling stats for debugging
        if culled_count > 0:
            logger.debug(
                f"Frustum culling: {culled_count}/{len(model.faces)} faces culled"
            )

        # Sort faces by Z-depth (far to near - painter's algorithm)
        face_data.sort(key=lambda x: x['avg_z'], reverse=True)

        # Render each face
        for data in face_data:
            face = data['face']
            frame = self._render_single_face(
                frame,
                model,
                face,
                projected_2d
            )

        return frame

    def _render_single_face(
        self,
        frame: np.ndarray,
        model: OBJModel,
        face: dict,
        projected_2d: np.ndarray
    ) -> np.ndarray:
        """
        Render a single face (triangle or polygon).

        Args:
            frame: Frame to render onto
            model: OBJ model
            face: Face data dictionary
            projected_2d: Projected 2D vertices

        Returns:
            Frame with rendered face
        """
        v_indices = face['vertices']

        if len(v_indices) < 3:
            return frame

        # Get 2D points for this face
        pts_2d = np.array([projected_2d[i] for i in v_indices], dtype=np.int32)

        # Check if face is visible (backface culling using winding order)
        if not self._is_face_visible(pts_2d):
            return frame

        # Get material
        material = None
        if face['material'] and face['material'] in model.materials:
            material = model.materials[face['material']]

        # Render textured or solid color
        if material and material.diffuse_texture is not None and len(face['tex_coords']) == len(v_indices):
            # Render with texture
            frame = self._render_textured_face(
                frame,
                pts_2d,
                face['tex_coords'],
                model.tex_coords,
                material
            )
        else:
            # Render solid color
            color = self._get_face_color(material)
            cv2.fillPoly(frame, [pts_2d], color)

        return frame

    def _is_face_visible(self, pts_2d: np.ndarray) -> bool:
        """
        Check if face is visible using winding order (backface culling).

        Args:
            pts_2d: 2D points of the face

        Returns:
            True if face is visible (front-facing)
        """
        if len(pts_2d) < 3:
            return False

        # Calculate signed area (cross product in 2D)
        # Positive = counter-clockwise = front-facing
        p0, p1, p2 = pts_2d[0], pts_2d[1], pts_2d[2]
        cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])

        return cross > 0

    def _get_face_color(self, material: Optional[Material]) -> Tuple[int, int, int]:
        """
        Get BGR color for a face from its material.

        Args:
            material: Material or None

        Returns:
            BGR color tuple
        """
        if material:
            # Convert RGB diffuse color to BGR for OpenCV
            r, g, b = material.diffuse_color
            return (int(b * 255), int(g * 255), int(r * 255))
        else:
            # Default gray color
            return (180, 180, 180)

    def _render_textured_face(
        self,
        frame: np.ndarray,
        pts_2d: np.ndarray,
        tex_coord_indices: list,
        tex_coords: list,
        material: Material
    ) -> np.ndarray:
        """
        Render a face with texture mapping.

        Args:
            frame: Frame to render onto
            pts_2d: 2D screen coordinates of face vertices
            tex_coord_indices: Texture coordinate indices
            tex_coords: Full list of texture coordinates
            material: Material with texture

        Returns:
            Frame with textured face
        """
        texture = material.diffuse_texture
        if texture is None:
            return frame

        tex_h, tex_w = texture.shape[:2]

        # For triangles, use affine transformation
        if len(pts_2d) == 3:
            # Get texture coordinates for this triangle
            uv = np.array([tex_coords[i] for i in tex_coord_indices], dtype=np.float32)

            # Convert UV (0-1) to texture pixel coordinates
            tex_pts = np.array([
                [uv[0][0] * tex_w, (1 - uv[0][1]) * tex_h],
                [uv[1][0] * tex_w, (1 - uv[1][1]) * tex_h],
                [uv[2][0] * tex_w, (1 - uv[2][1]) * tex_h]
            ], dtype=np.float32)

            # Calculate affine transformation from texture to screen
            transform = cv2.getAffineTransform(tex_pts, pts_2d.astype(np.float32))

            # Get bounding box
            x_min = max(0, int(np.min(pts_2d[:, 0])))
            y_min = max(0, int(np.min(pts_2d[:, 1])))
            x_max = min(frame.shape[1], int(np.max(pts_2d[:, 0])) + 1)
            y_max = min(frame.shape[0], int(np.max(pts_2d[:, 1])) + 1)

            if x_max <= x_min or y_max <= y_min:
                return frame

            # Create mask for the triangle
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            cv2.fillPoly(mask, [pts_2d], 255)

            # Warp texture
            warped = cv2.warpAffine(
                texture,
                transform,
                (frame.shape[1], frame.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_TRANSPARENT
            )

            # Apply mask and blend
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            frame = np.where(mask_3ch > 0, warped, frame)

        else:
            # For polygons with more than 3 vertices, render solid color
            color = self._get_face_color(material)
            cv2.fillPoly(frame, [pts_2d], color)

        return frame
