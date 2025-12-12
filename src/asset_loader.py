"""
Smart asset loader with automatic normalization and optimization.

Handles loading 3D models with automatic coordinate normalization,
LOD generation, and caching for optimal performance.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

from .obj_loader import load_obj, OBJModel
from .mesh_optimizer import (
    MeshNormalizer,
    MeshSimplifier,
    LODGenerator,
    MeshCache
)

logger = logging.getLogger(__name__)


class SmartAssetLoader:
    """
    Loads and optimizes 3D assets with automatic normalization.

    Features:
    - Auto-detects if model needs normalization
    - Generates multiple LOD levels
    - Caches processed meshes for fast reload
    - Provides standardized coordinate space for all models
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize asset loader.

        Args:
            cache_dir: Directory for caching processed meshes
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / ".cache" / "meshes"

        self.cache = MeshCache(cache_dir)
        self.normalizer = MeshNormalizer()
        self.lod_generator = LODGenerator()

    def load_and_optimize(
        self,
        obj_path: Path,
        auto_normalize: bool = True,
        generate_lods: bool = True,
        target_size: float = 1.0
    ) -> Tuple[Dict[str, OBJModel], Dict]:
        """
        Load OBJ file with automatic optimization.

        Args:
            obj_path: Path to OBJ file
            auto_normalize: Automatically normalize to unit box
            generate_lods: Generate LOD levels
            target_size: Target size for normalization

        Returns:
            Tuple of (LOD dict, transform info dict)
            LOD dict maps 'high'/'medium'/'low'/'very_low' to OBJModel
        """
        # Try to load from cache first
        cache_params = {
            'auto_normalize': auto_normalize,
            'generate_lods': generate_lods,
            'target_size': target_size
        }

        cached = self.cache.get(obj_path, 'load_optimize', cache_params)
        if cached is not None:
            return cached

        # Load original model
        logger.info(f"Loading 3D model: {obj_path.name}")
        model = load_obj(obj_path)

        if model is None:
            logger.error(f"Failed to load model: {obj_path}")
            return {}, {}

        transform_info = {}

        # Auto-normalize if needed
        if auto_normalize:
            needs_normalization = self._needs_normalization(model, target_size)

            if needs_normalization:
                logger.info(f"Auto-normalizing {obj_path.name}")
                model, transform_info = self.normalizer.normalize_to_unit_box(
                    model,
                    target_size=target_size,
                    center_at_origin=True
                )
            else:
                logger.info(f"{obj_path.name} already normalized, skipping")

        # Generate LODs
        lods = {}
        if generate_lods:
            lods = self.lod_generator.generate_lods(model)
        else:
            lods['high'] = model

        # Cache the result
        result = (lods, transform_info)
        self.cache.set(obj_path, 'load_optimize', cache_params, result)

        return lods, transform_info

    def _needs_normalization(self, model: OBJModel, target_size: float) -> bool:
        """
        Check if model needs normalization.

        A model needs normalization if:
        - It's not centered near origin
        - Its size is significantly different from target
        """
        if len(model.vertices) == 0:
            return False

        vertices = np.array(model.vertices)
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)
        center = (min_coords + max_coords) / 2
        size = max_coords - min_coords
        max_dim = size.max()

        # Check if center is far from origin (threshold: 10% of size)
        center_distance = np.linalg.norm(center)
        if center_distance > max_dim * 0.1:
            logger.debug(f"Model off-center: {center}, distance: {center_distance:.3f}")
            return True

        # Check if size is significantly different from target
        # (threshold: 20% difference)
        size_ratio = max_dim / target_size if target_size > 0 else 0
        if size_ratio < 0.8 or size_ratio > 1.2:
            logger.debug(f"Model size mismatch: {max_dim:.3f} vs target {target_size}")
            return True

        return False

    def get_recommended_config(
        self,
        obj_path: Path,
        object_type: str = 'glasses'
    ) -> Dict:
        """
        Get recommended configuration for an object.

        Args:
            obj_path: Path to OBJ file
            object_type: Type of object ('glasses', 'hat', 'mask', etc.)

        Returns:
            Dictionary with recommended scale, offsets, and rotations
        """
        # Load model to analyze
        model = load_obj(obj_path)
        if model is None:
            return {}

        vertices = np.array(model.vertices)
        if len(vertices) == 0:
            return {}

        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)
        size = max_coords - min_coords

        # Base recommendations by type
        recommendations = {
            'glasses': {
                'scale_multiplier': 1.0,  # After normalization, usually 1.0 works
                'offset_x': 0.0,
                'offset_y': -0.5,  # Slightly below nose bridge
                'offset_z': -2.0,  # Forward from face
                'rotation_x': 0.0,
                'rotation_y': 180.0,  # Face forward
                'rotation_z': 0.0,
            },
            'hat': {
                'scale_multiplier': 1.2,  # Slightly larger
                'offset_x': 0.0,
                'offset_y': 3.5,  # Above head
                'offset_z': -1.5,  # Slightly forward
                'rotation_x': -10.0,  # Tilt slightly forward
                'rotation_y': 0.0,  # Face same direction
                'rotation_z': 0.0,
            },
            'mask': {
                'scale_multiplier': 1.0,
                'offset_x': 0.0,
                'offset_y': 0.0,
                'offset_z': -1.5,
                'rotation_x': 0.0,
                'rotation_y': 0.0,
                'rotation_z': 0.0,
            }
        }

        config = recommendations.get(object_type, recommendations['mask'])

        # Add metadata
        config['metadata'] = {
            'original_size': size.tolist(),
            'vertices': len(model.vertices),
            'faces': len(model.faces),
            'recommended_subsample': self._get_recommended_subsample(model)
        }

        return config

    def _get_recommended_subsample(self, model: OBJModel) -> int:
        """
        Get recommended subsample rate based on model complexity.

        Args:
            model: OBJ model

        Returns:
            Recommended subsample rate (1 = no subsampling)
        """
        face_count = len(model.faces)

        # Adaptive subsampling based on complexity
        if face_count < 2000:
            return 1  # No subsampling for light models
        elif face_count < 5000:
            return 2  # Light subsampling
        elif face_count < 10000:
            return 5  # Medium subsampling
        else:
            return 10  # Heavy subsampling for complex models

        return 1
