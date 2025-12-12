"""
3D Object data structures and transformations.

Defines data classes for 3D objects that can be placed and rendered
on detected faces in the AR face filter application.
"""

from typing import Optional, Dict
import numpy as np
import logging

from src.obj_loader import OBJModel, load_obj
from src.asset_loader import SmartAssetLoader
from src.mesh_optimizer import LODGenerator

logger = logging.getLogger(__name__)


class Object3D:
    """
    Represents a 3D object that can be rendered on a face.

    Handles loading, transformation, and rendering of 3D OBJ models
    with proper pose estimation, projection, and LOD management.
    """

    def __init__(self, config, use_smart_loader: bool = True):
        """
        Initialize 3D object.

        Args:
            config: Configuration for the 3D object
            use_smart_loader: Use SmartAssetLoader for auto-normalization and LOD
        """
        self.config = config
        self.model: Optional[OBJModel] = None
        self.lods: Dict[str, OBJModel] = {}
        self.current_lod: str = 'high'
        self.transform_info: Dict = {}
        self.is_loaded = False
        self.use_smart_loader = use_smart_loader

        # Load the OBJ model
        self.load()

    def load(self) -> bool:
        """
        Load the OBJ model from file with optimization.

        Returns:
            True if loading succeeded, False otherwise
        """
        try:
            if self.use_smart_loader:
                # Use smart loader for auto-normalization and LOD
                loader = SmartAssetLoader()
                self.lods, self.transform_info = loader.load_and_optimize(
                    self.config.obj_path,
                    auto_normalize=True,
                    generate_lods=True,
                    target_size=1.0
                )

                if self.lods:
                    self.model = self.lods.get('high')
                    self.is_loaded = True
                    logger.info(
                        f"Loaded {self.config.name} with {len(self.lods)} LOD levels"
                    )
                    return True
            else:
                # Fallback to simple loading
                self.model = load_obj(self.config.obj_path)
                if self.model and len(self.model.vertices) > 0:
                    self.is_loaded = True
                    self.lods['high'] = self.model
                    return True

            return False
        except Exception as e:
            logger.error(f"Error loading 3D object {self.config.name}: {e}")
            return False

    def get_vertices_array(self) -> np.ndarray:
        """
        Get vertices as numpy array for efficient processing.

        Returns:
            Array of shape (N, 3) containing vertex positions
        """
        if not self.is_loaded or not self.model:
            return np.array([])

        return np.array(self.model.vertices, dtype=np.float32)

    def get_anchor_position(self, landmarks: list) -> Optional[tuple[float, float]]:
        """
        Calculate anchor position from face landmarks.

        Args:
            landmarks: List of (x, y) facial landmark coordinates

        Returns:
            (x, y) anchor position or None if landmarks invalid
        """
        if not self.config.anchor_landmarks:
            return None

        try:
            # Calculate average position of anchor landmarks
            anchor_points = [landmarks[i] for i in self.config.anchor_landmarks]
            anchor_x = sum(p[0] for p in anchor_points) / len(anchor_points)
            anchor_y = sum(p[1] for p in anchor_points) / len(anchor_points)
            return (anchor_x, anchor_y)
        except (IndexError, TypeError):
            return None

    def calculate_scale(self, landmarks: list, face_width: float) -> float:
        """
        Calculate appropriate scale for the object based on face size.

        Args:
            landmarks: List of facial landmark coordinates
            face_width: Width of detected face

        Returns:
            Scale factor
        """
        # Base scale on face width and config multiplier
        base_scale = face_width * 0.001  # Convert to object space
        return base_scale * self.config.scale_multiplier

    def set_lod(self, lod_level: str):
        """
        Set current LOD level.

        Args:
            lod_level: LOD level ('high', 'medium', 'low', 'very_low')
        """
        if lod_level in self.lods:
            self.current_lod = lod_level
            self.model = self.lods[lod_level]
            logger.debug(f"{self.config.name}: Switched to LOD {lod_level}")
        else:
            logger.warning(f"{self.config.name}: LOD {lod_level} not available")

    def auto_select_lod(self, current_fps: float, target_fps: float = 30.0):
        """
        Automatically select appropriate LOD based on FPS.

        Args:
            current_fps: Current frames per second
            target_fps: Target FPS to maintain
        """
        recommended_lod = LODGenerator.select_lod_by_fps(current_fps, target_fps)
        self.set_lod(recommended_lod)

    def get_current_model(self) -> Optional[OBJModel]:
        """
        Get the current LOD model.

        Returns:
            Current OBJModel based on LOD setting
        """
        return self.model

    def __repr__(self) -> str:
        """String representation of the 3D object."""
        status = "loaded" if self.is_loaded else "not loaded"
        vert_count = len(self.model.vertices) if self.model else 0
        face_count = len(self.model.faces) if self.model else 0
        lod_info = f", LOD={self.current_lod}" if self.lods else ""
        return (
            f"Object3D(name='{self.config.name}', "
            f"status={status}, "
            f"vertices={vert_count}, "
            f"faces={face_count}{lod_info})"
        )
