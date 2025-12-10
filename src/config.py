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
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374

    # Nose
    NOSE_TIP = 1
    NOSE_BRIDGE = 6
    NOSE_BOTTOM = 2

    # Mouth - Extended landmarks for gesture detection
    UPPER_LIP_TOP = 13
    UPPER_LIP_BOTTOM = 14
    LOWER_LIP_TOP = 312
    LOWER_LIP_BOTTOM = 17  # Bottom of lower lip
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    # Additional mouth landmarks for better MAR calculation
    UPPER_LIP_CENTER = 0  # Center of upper lip
    LOWER_LIP_CENTER = 17  # Center of lower lip

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

    # Eyebrows - for brow raise detection
    LEFT_EYEBROW_INNER = 107
    LEFT_EYEBROW_CENTER = 66
    LEFT_EYEBROW_OUTER = 105
    RIGHT_EYEBROW_INNER = 336
    RIGHT_EYEBROW_CENTER = 296
    RIGHT_EYEBROW_OUTER = 334

    # Additional landmarks for precise gesture detection
    LEFT_EYEBROW_TOP = 70   # Top of left eyebrow arch
    RIGHT_EYEBROW_TOP = 300  # Top of right eyebrow arch


# ============================================================================
# GESTURE DETECTION CONFIGURATION
# ============================================================================

@dataclass
class GestureConfig:
    """Configuration for a specific gesture."""
    threshold: float
    cooldown_frames: int  # Frames to wait after triggering
    required_frames: int  # Consecutive frames needed to trigger
    name: str


# Gesture thresholds and settings
# Mouth Aspect Ratio (MAR) = vertical_distance / horizontal_distance
# When mouth is open, MAR increases
MOUTH_OPEN_THRESHOLD = 0.35  # Ratio threshold for mouth open detection
MOUTH_OPEN_COOLDOWN_FRAMES = 30  # ~1 second at 30fps
MOUTH_OPEN_REQUIRED_FRAMES = 5  # Hold for 5 frames to trigger

# Brow Raise Ratio = (brow_to_eye_distance) / (face_height)
# When brows are raised, this ratio increases
BROW_RAISE_THRESHOLD = 0.11  # Ratio threshold for brow raise
BROW_RAISE_COOLDOWN_FRAMES = 30  # ~1 second at 30fps
BROW_RAISE_REQUIRED_FRAMES = 5  # Hold for 5 frames to trigger

# Eye Aspect Ratio (EAR) for potential eye blink detection (future)
EYE_BLINK_THRESHOLD = 0.2
EYE_BLINK_COOLDOWN_FRAMES = 15
EYE_BLINK_REQUIRED_FRAMES = 3

# Gesture configurations dictionary
GESTURE_CONFIGS: Dict[str, GestureConfig] = {
    "mouth_open": GestureConfig(
        threshold=MOUTH_OPEN_THRESHOLD,
        cooldown_frames=MOUTH_OPEN_COOLDOWN_FRAMES,
        required_frames=MOUTH_OPEN_REQUIRED_FRAMES,
        name="Mouth Open"
    ),
    "brow_raise": GestureConfig(
        threshold=BROW_RAISE_THRESHOLD,
        cooldown_frames=BROW_RAISE_COOLDOWN_FRAMES,
        required_frames=BROW_RAISE_REQUIRED_FRAMES,
        name="Brow Raise"
    ),
}

# Landmark indices for gesture calculations
class GestureLandmarks:
    """Landmark indices specifically for gesture detection."""

    # Mouth landmarks for MAR calculation (vertical opening)
    MOUTH_VERTICAL = {
        "upper": [13, 312],  # Upper lip landmarks
        "lower": [14, 17],   # Lower lip landmarks
    }

    # Mouth corners for horizontal reference
    MOUTH_HORIZONTAL = {
        "left": 61,
        "right": 291,
    }

    # Eyebrow landmarks (for averaging)
    LEFT_EYEBROW = [107, 66, 105, 70]
    RIGHT_EYEBROW = [336, 296, 334, 300]

    # Eye landmarks for reference distance
    LEFT_EYE_TOP = 159
    RIGHT_EYE_TOP = 386

    # Face height reference points
    FACE_TOP = 10  # Forehead
    FACE_BOTTOM = 152  # Chin


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

# Texture mask configurations
TEXTURE_MASKS: Dict[str, "TextureMaskConfig"] = {}  # Will be populated below


@dataclass
class TextureMaskConfig:
    """Configuration for face mesh texture masks."""
    asset_path: Path
    opacity: float  # 0.0 to 1.0
    debug_wireframe: bool  # Show triangle wireframe
    subsample: int  # Render every Nth triangle (1 = all)


# MediaPipe Face Mesh Triangulation
# Converted from mediapipe.solutions.face_mesh.FACEMESH_TESSELATION
# This creates triangles from the edge connections
def _build_face_mesh_triangles():
    """
    Build triangle list from MediaPipe FACEMESH_TESSELATION connections.
    FACEMESH_TESSELATION provides edges, we need to convert to triangles.
    """
    import mediapipe as mp

    # Get the tesselation (edge connections)
    tesselation = mp.solutions.face_mesh.FACEMESH_TESSELATION

    # Build adjacency map
    adjacency = {}
    for edge in tesselation:
        a, b = edge
        if a not in adjacency:
            adjacency[a] = set()
        if b not in adjacency:
            adjacency[b] = set()
        adjacency[a].add(b)
        adjacency[b].add(a)

    # Find triangles by looking for cycles of length 3
    triangles = set()
    for a in adjacency:
        for b in adjacency[a]:
            if b > a:  # Avoid duplicates
                # Find common neighbors
                common = adjacency[a] & adjacency[b]
                for c in common:
                    if c > b:  # Ensure unique ordering
                        triangle = tuple(sorted([a, b, c]))
                        triangles.add(triangle)

    return list(triangles)

# Generate triangles on module import
FACE_MESH_TRIANGLES = _build_face_mesh_triangles()

# Initialize texture masks
TEXTURE_MASKS["masculine"] = TextureMaskConfig(
    asset_path=ASSETS_DIR / "faceMasculine.jpg",
    opacity=0.7,
    debug_wireframe=False,
    subsample=1
)

TEXTURE_MASKS["feminine"] = TextureMaskConfig(
    asset_path=ASSETS_DIR / "faceFeminine.jpg",
    opacity=0.7,
    debug_wireframe=False,
    subsample=1
)

TEXTURE_MASKS["debug"] = TextureMaskConfig(
    asset_path=ASSETS_DIR / "faceMesh.png",
    opacity=0.5,
    debug_wireframe=True,
    subsample=1
)

# Custom texture mask (001.png)
TEXTURE_MASKS["custom"] = TextureMaskConfig(
    asset_path=ASSETS_DIR / "001.png",
    opacity=0.7,
    debug_wireframe=False,
    subsample=1
)

# Display settings
WINDOW_NAME = "AR Face Filter"
FPS_DISPLAY = True
SHOW_LANDMARKS = False  # Debug mode: show facial landmarks
SHOW_INSTRUCTIONS = True  # Show keyboard instructions
SHOW_GESTURE_STATUS = True  # Show gesture detection status

# Default active filters (all enabled by default)
DEFAULT_ACTIVE_FILTERS = {
    "glasses": False,
    "mustache": False,
    "beard": False,
    "headband": False
}

# Default active texture mask (None by default)
DEFAULT_ACTIVE_TEXTURE_MASK = None  # Can be "masculine", "feminine", or "debug"

# Gesture control settings
GESTURE_CONTROL_ENABLED = True  # Master toggle for gesture controls