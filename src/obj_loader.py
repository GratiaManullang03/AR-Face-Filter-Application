"""
OBJ and MTL file loader for 3D models.

This module handles loading Wavefront OBJ files with their associated MTL
material files and textures. Supports vertices, texture coordinates, normals,
faces, and materials with diffuse textures.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Material:
    """Represents a material with color and texture properties."""

    def __init__(self, name: str):
        self.name = name
        self.ambient_color = (1.0, 1.0, 1.0)  # Ka
        self.diffuse_color = (1.0, 1.0, 1.0)  # Kd
        self.specular_color = (1.0, 1.0, 1.0)  # Ks
        self.specular_exponent = 100.0  # Ns
        self.transparency = 1.0  # d (1.0 = opaque)
        self.illumination_model = 2  # illum
        self.diffuse_texture: Optional[np.ndarray] = None
        self.texture_path: Optional[str] = None


class OBJModel:
    """Represents a loaded OBJ model with geometry and materials."""

    def __init__(self):
        self.vertices: List[Tuple[float, float, float]] = []
        self.tex_coords: List[Tuple[float, float]] = []
        self.normals: List[Tuple[float, float, float]] = []
        self.faces: List[Dict] = []  # Each face has: vertices, tex_coords, normals, material
        self.materials: Dict[str, Material] = {}
        self.current_material: Optional[str] = None


def load_mtl(mtl_path: Path, base_dir: Path) -> Dict[str, Material]:
    """
    Load MTL material file.

    Args:
        mtl_path: Path to MTL file
        base_dir: Base directory for resolving texture paths

    Returns:
        Dictionary mapping material name to Material object
    """
    materials = {}
    current_material = None

    try:
        with open(mtl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if not parts:
                    continue

                cmd = parts[0]

                if cmd == 'newmtl':
                    # New material definition
                    material_name = ' '.join(parts[1:])
                    current_material = Material(material_name)
                    materials[material_name] = current_material

                elif current_material:
                    if cmd == 'Ka':
                        # Ambient color
                        current_material.ambient_color = (
                            float(parts[1]), float(parts[2]), float(parts[3])
                        )
                    elif cmd == 'Kd':
                        # Diffuse color
                        current_material.diffuse_color = (
                            float(parts[1]), float(parts[2]), float(parts[3])
                        )
                    elif cmd == 'Ks':
                        # Specular color
                        current_material.specular_color = (
                            float(parts[1]), float(parts[2]), float(parts[3])
                        )
                    elif cmd == 'Ns':
                        # Specular exponent
                        current_material.specular_exponent = float(parts[1])
                    elif cmd == 'd':
                        # Transparency
                        current_material.transparency = float(parts[1])
                    elif cmd == 'illum':
                        # Illumination model
                        current_material.illumination_model = int(parts[1])
                    elif cmd == 'map_Kd':
                        # Diffuse texture map
                        texture_path = ' '.join(parts[1:])
                        # Remove -bm flag if present
                        if '-bm' in texture_path:
                            texture_path = texture_path.split('-bm')[0].strip()

                        current_material.texture_path = texture_path

                        # Try to load texture
                        full_texture_path = base_dir / texture_path
                        if full_texture_path.exists():
                            texture = cv2.imread(str(full_texture_path))
                            if texture is not None:
                                # Convert BGR to RGB
                                current_material.diffuse_texture = cv2.cvtColor(
                                    texture, cv2.COLOR_BGR2RGB
                                )
                                logger.info(
                                    f"Loaded texture for {current_material.name}: "
                                    f"{texture_path} ({texture.shape})"
                                )
                            else:
                                logger.warning(
                                    f"Failed to load texture: {full_texture_path}"
                                )
                        else:
                            logger.warning(
                                f"Texture file not found: {full_texture_path}"
                            )

        logger.info(f"Loaded {len(materials)} materials from {mtl_path.name}")
        return materials

    except Exception as e:
        logger.error(f"Error loading MTL file {mtl_path}: {e}")
        return {}


def load_obj(obj_path: Path) -> Optional[OBJModel]:
    """
    Load OBJ file with materials and textures.

    Args:
        obj_path: Path to OBJ file

    Returns:
        OBJModel object or None if loading failed
    """
    model = OBJModel()
    base_dir = obj_path.parent

    try:
        with open(obj_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if not parts:
                    continue

                cmd = parts[0]

                try:
                    if cmd == 'v':
                        # Vertex position
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        model.vertices.append((x, y, z))

                    elif cmd == 'vt':
                        # Texture coordinate
                        u, v = float(parts[1]), float(parts[2])
                        model.tex_coords.append((u, v))

                    elif cmd == 'vn':
                        # Vertex normal
                        nx, ny, nz = float(parts[1]), float(parts[2]), float(parts[3])
                        model.normals.append((nx, ny, nz))

                    elif cmd == 'f':
                        # Face (can be v, v/vt, v/vt/vn, or v//vn)
                        face = {
                            'vertices': [],
                            'tex_coords': [],
                            'normals': [],
                            'material': model.current_material
                        }

                        for i in range(1, len(parts)):
                            indices = parts[i].split('/')

                            # Vertex index (always present)
                            v_idx = int(indices[0])
                            face['vertices'].append(v_idx - 1)  # OBJ uses 1-based indexing

                            # Texture coordinate index
                            if len(indices) > 1 and indices[1]:
                                vt_idx = int(indices[1])
                                face['tex_coords'].append(vt_idx - 1)

                            # Normal index
                            if len(indices) > 2 and indices[2]:
                                vn_idx = int(indices[2])
                                face['normals'].append(vn_idx - 1)

                        model.faces.append(face)

                    elif cmd == 'mtllib':
                        # Material library
                        mtl_filename = ' '.join(parts[1:])
                        mtl_path = base_dir / mtl_filename
                        if mtl_path.exists():
                            model.materials = load_mtl(mtl_path, base_dir)
                        else:
                            logger.warning(f"MTL file not found: {mtl_path}")

                    elif cmd == 'usemtl':
                        # Use material
                        material_name = ' '.join(parts[1:])
                        model.current_material = material_name

                except (ValueError, IndexError) as e:
                    logger.warning(
                        f"Error parsing line {line_num} in {obj_path.name}: {line} ({e})"
                    )
                    continue

        logger.info(
            f"Loaded OBJ: {obj_path.name} - "
            f"{len(model.vertices)} vertices, {len(model.faces)} faces, "
            f"{len(model.materials)} materials"
        )
        return model

    except Exception as e:
        logger.error(f"Error loading OBJ file {obj_path}: {e}")
        return None
