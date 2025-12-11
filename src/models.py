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
        
        # Check if we have Canonical UVs available (Priority 1)
        if config.CANONICAL_FACE_MESH_UV is not None:
            # Renderer will handle the mapping in its render method or initialization
            pass
        else:
            # Fallback: Detect face in texture (Priority 2)
            print(f"Detecting face in texture for UV mapping...")
            success = self.renderer.set_texture_landmarks_from_detection(
                str(self.config.asset_path)
            )

            if not success:
                print(f"ERROR: Could not detect face in texture image!")
                print(f"Make sure {self.config.asset_path.name} contains a clear frontal face.")
                return False

        print(f"✓ Loaded texture mask: {self.config.asset_path.name}")
        return True

    def is_loaded(self) -> bool:
        return self.renderer is not None
