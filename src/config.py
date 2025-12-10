"""
Configuration module for AR Face Filter application.
Contains constants, paths, and facial landmark mappings.
"""

from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# MediaPipe Face Mesh settings
MAX_NUM_FACES = 5  # Support up to 5 faces
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Face Mesh landmark indices (MediaPipe Face Mesh has 468 landmarks)
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png


@dataclass
class LandmarkGroup:
    """Represents a group of facial landmarks for a specific region."""
    indices: List[int]
    name: str


# Key landmark groups for AR filters
class FacialLandmarks:
    """Facial landmark indices for MediaPipe Face Mesh."""

    # Eyes
    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    RIGHT_EYE_OUTER = 362
    RIGHT_EYE_INNER = 263
    LEFT_EYE_CENTER = 468  # Iris center (if using refine_landmarks)
    RIGHT_EYE_CENTER = 473  # Iris center (if using refine_landmarks)

    # Nose
    NOSE_TIP = 1
    NOSE_BRIDGE = 6
    NOSE_BOTTOM = 2

    # Mouth
    UPPER_LIP_TOP = 13
    UPPER_LIP_BOTTOM = 14
    LOWER_LIP_TOP = 312
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291

    # Chin
    CHIN_BOTTOM = 152
    CHIN_LEFT = 176
    CHIN_RIGHT = 400

    # Forehead
    FOREHEAD_CENTER = 10
    FOREHEAD_LEFT = 109
    FOREHEAD_RIGHT = 338

    # Temple (for rotation calculation)
    LEFT_TEMPLE = 234
    RIGHT_TEMPLE = 454

    # Jawline
    JAW_LEFT = 58
    JAW_RIGHT = 288


@dataclass
class FilterConfig:
    """Configuration for a specific AR filter."""
    asset_path: Path
    anchor_landmarks: List[int]  # Landmarks to calculate center position
    scale_landmarks: Tuple[int, int]  # Two landmarks to calculate scale
    rotation_landmarks: Tuple[int, int]  # Two landmarks to calculate rotation
    scale_multiplier: float  # Fine-tune the size
    y_offset: int  # Vertical offset in pixels
    x_offset: int  # Horizontal offset in pixels


# Filter configurations
FILTERS: Dict[str, FilterConfig] = {
    "glasses": FilterConfig(
        asset_path=ASSETS_DIR / "glasses.png",
        anchor_landmarks=[
            FacialLandmarks.LEFT_EYE_INNER,
            FacialLandmarks.RIGHT_EYE_INNER,
            FacialLandmarks.NOSE_BRIDGE
        ],
        scale_landmarks=(
            FacialLandmarks.LEFT_EYE_OUTER,
            FacialLandmarks.RIGHT_EYE_OUTER
        ),
        rotation_landmarks=(
            FacialLandmarks.LEFT_EYE_OUTER,
            FacialLandmarks.RIGHT_EYE_OUTER
        ),
        scale_multiplier=2.5,
        y_offset=8,
        x_offset=-20
    ),

    "mustache": FilterConfig(
        asset_path=ASSETS_DIR / "mustache.png",
        anchor_landmarks=[
            FacialLandmarks.UPPER_LIP_TOP,
            FacialLandmarks.NOSE_BOTTOM
        ],
        scale_landmarks=(
            FacialLandmarks.MOUTH_LEFT,
            FacialLandmarks.MOUTH_RIGHT
        ),
        rotation_landmarks=(
            FacialLandmarks.LEFT_TEMPLE,
            FacialLandmarks.RIGHT_TEMPLE
        ),
        scale_multiplier=1.3,
        y_offset=-5,
        x_offset=0
    ),

    "beard": FilterConfig(
        asset_path=ASSETS_DIR / "beard.png",
        anchor_landmarks=[
            FacialLandmarks.CHIN_BOTTOM,
            FacialLandmarks.LOWER_LIP_TOP
        ],
        scale_landmarks=(
            FacialLandmarks.JAW_LEFT,
            FacialLandmarks.JAW_RIGHT
        ),
        rotation_landmarks=(
            FacialLandmarks.LEFT_TEMPLE,
            FacialLandmarks.RIGHT_TEMPLE
        ),
        scale_multiplier=1.8,
        y_offset=10,
        x_offset=0
    ),

    "headband": FilterConfig(
        asset_path=ASSETS_DIR / "headband.png",
        anchor_landmarks=[
            FacialLandmarks.FOREHEAD_CENTER,
            FacialLandmarks.FOREHEAD_LEFT,
            FacialLandmarks.FOREHEAD_RIGHT
        ],
        scale_landmarks=(
            FacialLandmarks.LEFT_TEMPLE,
            FacialLandmarks.RIGHT_TEMPLE
        ),
        rotation_landmarks=(
            FacialLandmarks.LEFT_TEMPLE,
            FacialLandmarks.RIGHT_TEMPLE
        ),
        scale_multiplier=2.2,
        y_offset=-30,
        x_offset=0
    )
}

# Display settings
WINDOW_NAME = "AR Face Filter"
FPS_DISPLAY = True
SHOW_LANDMARKS = False  # Debug mode: show facial landmarks
SHOW_INSTRUCTIONS = True  # Show keyboard instructions

# Default active filters (all enabled by default)
DEFAULT_ACTIVE_FILTERS = {
    "glasses": False,
    "mustache": False,
    "beard": False,
    "headband": False
}
