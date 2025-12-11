"""
Data models for AR Filter application.
Contains classes for Filters and Texture Masks.
"""

import cv2
import numpy as np
from typing import Optional
from . import config
from .mesh_renderer import MeshRenderer

class ARFilter:
    """Single AR filter with loaded image and configuration."""

    def __init__(self, cfg: config.FilterConfig):
        self.config = cfg
        self.image = None
        self.load_image()

    def load_image(self) -> bool:
        """Load filter image from disk."""
        if not self.config.asset_path.exists():
            print(f"Warning: Not found: {self.config.asset_path}")
            return False
        self.image = cv2.imread(str(self.config.asset_path), cv2.IMREAD_UNCHANGED)
        if self.image is None:
            print(f"Error: Failed: {self.config.asset_path}")
        return self.image is not None

    def is_loaded(self) -> bool:
        return self.image is not None


class TextureMask:
    """Single texture mask with loaded image and renderer."""

    def __init__(self, cfg: config.TextureMaskConfig):
        self.config = cfg
        self.renderer: Optional[MeshRenderer] = None
        self.load_texture()

    def load_texture(self) -> bool:
        """Load texture and initialize renderer."""
        if not self.config.asset_path.exists():
            print(f"Warning: Texture not found: {self.config.asset_path}")
            return False

        texture_img = cv2.imread(str(self.config.asset_path))
        if texture_img is None:
            print(f"Error: Failed to load texture: {self.config.asset_path}")
            return False

        # Initialize mesh renderer
        self.renderer = MeshRenderer(
            texture_img,
            debug_mode=self.config.debug_wireframe
        )

        # Extract landmarks from the texture image using MediaPipe
        # OR use Canonical UVs if configured
        print(f"Initializing texture mask: {self.config.asset_path.name}...")
        
        # Priority 1: Use Global Canonical UVs if available
        if config.CANONICAL_FACE_MESH_UV is not None:
            # Pass directly to renderer (it handles scaling to texture size)
            pass
            
        # Priority 2: Try to generate Global Canonical UVs from the Reference Wireframe (faceMesh.png)
        # This is much more stable than detecting on the individual skin-texture images.
        else:
            print("Canonical UVs not set. Attempting to generate from Reference Wireframe...")
            from .utils import create_texture_landmarks_from_image
            
            # Try to find the debug/wireframe image
            ref_path = config.ASSETS_DIR / "faceMesh.png"
            
            if ref_path.exists():
                print(f"Loading reference UVs from: {ref_path.name}")
                ref_landmarks = create_texture_landmarks_from_image(str(ref_path))
                
                if ref_landmarks is not None:
                    # Normalize landmarks (0.0 - 1.0) to create Canonical UVs
                    ref_h, ref_w = cv2.imread(str(ref_path)).shape[:2]
                    config.CANONICAL_FACE_MESH_UV = ref_landmarks / np.array([ref_w, ref_h], dtype=np.float32)
                    print("✓ Global Canonical UVs generated successfully!")
                else:
                    print("Warning: Failed to detect face in reference image.")
            else:
                print("Warning: Reference image 'faceMesh.png' not found.")

        # Priority 3: Fallback - Detect face in this specific texture
        # (Only happens if Priority 1 & 2 failed)
        if config.CANONICAL_FACE_MESH_UV is None:
            print(f"Fallback: Detecting face in texture {self.config.asset_path.name}...")
            success = self.renderer.set_texture_landmarks_from_detection(
                str(self.config.asset_path)
            )
            if not success:
                print(f"ERROR: Could not detect face in texture image!")
                return False

        print(f"✓ Loaded texture mask: {self.config.asset_path.name}")
        return True

    def is_loaded(self) -> bool:
        return self.renderer is not None
