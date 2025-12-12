"""
Mesh optimization utilities for 3D models.

Provides mesh simplification, normalization, and LOD generation
to improve rendering performance for AR face filters.
"""

import logging
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
import hashlib
import pickle

from .obj_loader import OBJModel

logger = logging.getLogger(__name__)


class MeshSimplifier:
    """Simplifies 3D meshes by reducing triangle count."""

    @staticmethod
    def simplify_uniform(model: OBJModel, reduction_ratio: float) -> OBJModel:
        """
        Simplify mesh using uniform face decimation.

        Args:
            model: Original OBJ model
            reduction_ratio: Ratio of faces to keep (0.0-1.0)

        Returns:
            Simplified OBJ model
        """
        if reduction_ratio >= 1.0:
            return model

        # Calculate how many faces to keep
        target_faces = max(1, int(len(model.faces) * reduction_ratio))
        step = max(1, len(model.faces) // target_faces)

        # Create simplified model
        simplified = OBJModel()
        simplified.vertices = model.vertices.copy()
        simplified.tex_coords = model.tex_coords.copy()
        simplified.normals = model.normals.copy()
        simplified.materials = model.materials.copy()

        # Keep every Nth face for uniform distribution
        simplified.faces = model.faces[::step]

        logger.info(
            f"Simplified mesh: {len(model.faces)} → {len(simplified.faces)} faces "
            f"({reduction_ratio*100:.1f}% kept)"
        )

        return simplified


class MeshNormalizer:
    """Normalizes 3D mesh coordinates to standard space."""

    @staticmethod
    def normalize_to_unit_box(
        model: OBJModel,
        target_size: float = 1.0,
        center_at_origin: bool = True
    ) -> Tuple[OBJModel, Dict]:
        """
        Normalize mesh to fit in a unit box centered at origin.

        Args:
            model: Original OBJ model
            target_size: Target maximum dimension
            center_at_origin: Whether to center mesh at (0, 0, 0)

        Returns:
            Tuple of (normalized model, transform info dict)
        """
        if len(model.vertices) == 0:
            return model, {}

        vertices = np.array(model.vertices)

        # Calculate current bounding box
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)
        current_center = (min_coords + max_coords) / 2
        current_size = max_coords - min_coords
        max_dimension = current_size.max()

        # Calculate normalization transform
        scale_factor = target_size / max_dimension if max_dimension > 0 else 1.0

        # Create normalized model
        normalized = OBJModel()
        normalized.tex_coords = model.tex_coords.copy()
        normalized.normals = model.normals.copy()
        normalized.materials = model.materials.copy()
        normalized.faces = model.faces.copy()
        normalized.current_material = model.current_material

        # Apply normalization
        if center_at_origin:
            vertices = vertices - current_center  # Center at origin

        vertices = vertices * scale_factor  # Scale to target size

        normalized.vertices = [(float(v[0]), float(v[1]), float(v[2])) for v in vertices]

        # Store transform info for reference
        transform_info = {
            'original_center': current_center.tolist(),
            'original_size': current_size.tolist(),
            'original_max_dimension': float(max_dimension),
            'scale_factor': float(scale_factor),
            'target_size': target_size,
            'centered': center_at_origin
        }

        logger.info(
            f"Normalized mesh: center {current_center} → (0,0,0), "
            f"size {current_size.max():.3f} → {target_size:.3f} "
            f"(scale: {scale_factor:.3f})"
        )

        return normalized, transform_info


class MeshCache:
    """Caches processed meshes to avoid reprocessing."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, obj_path: Path, operation: str, params: dict) -> str:
        """Generate cache key from file path and parameters."""
        # Include file modification time in hash
        mtime = obj_path.stat().st_mtime
        key_data = f"{obj_path}_{mtime}_{operation}_{str(sorted(params.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, obj_path: Path, operation: str, params: dict) -> Optional[OBJModel]:
        """Retrieve cached processed mesh."""
        cache_key = self._get_cache_key(obj_path, operation, params)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    logger.debug(f"Cache hit for {obj_path.name}")
                    return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                cache_file.unlink(missing_ok=True)

        return None

    def set(self, obj_path: Path, operation: str, params: dict, model: OBJModel):
        """Store processed mesh in cache."""
        cache_key = self._get_cache_key(obj_path, operation, params)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug(f"Cached processed mesh for {obj_path.name}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")


class LODGenerator:
    """Generates Level of Detail (LOD) versions of meshes."""

    # LOD levels: percentage of faces to keep
    LOD_LEVELS = {
        'high': 1.0,      # 100% - original
        'medium': 0.5,    # 50% - half faces
        'low': 0.2,       # 20% - fifth of faces
        'very_low': 0.1   # 10% - tenth of faces
    }

    @staticmethod
    def generate_lods(model: OBJModel) -> Dict[str, OBJModel]:
        """
        Generate all LOD levels for a model.

        Args:
            model: Original OBJ model

        Returns:
            Dictionary mapping LOD level name to simplified model
        """
        lods = {}

        for lod_name, ratio in LODGenerator.LOD_LEVELS.items():
            if ratio == 1.0:
                lods[lod_name] = model
            else:
                lods[lod_name] = MeshSimplifier.simplify_uniform(model, ratio)

        return lods

    @staticmethod
    def select_lod_by_fps(current_fps: float, target_fps: float = 30.0) -> str:
        """
        Select appropriate LOD level based on current FPS.

        Args:
            current_fps: Current frames per second
            target_fps: Target FPS to maintain

        Returns:
            LOD level name ('high', 'medium', 'low', 'very_low')
        """
        if current_fps >= target_fps * 0.9:  # Within 90% of target
            return 'high'
        elif current_fps >= target_fps * 0.6:  # Within 60% of target
            return 'medium'
        elif current_fps >= target_fps * 0.3:  # Within 30% of target
            return 'low'
        else:
            return 'very_low'
