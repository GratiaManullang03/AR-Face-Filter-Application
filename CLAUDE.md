# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A real-time AR face filter application using Python, OpenCV, and **MediaPipe Face Landmarker API (Latest)**. The application detects facial landmarks via webcam and applies two types of effects:
1. **2D Filters**: PNG overlays (glasses, mustache, beard, headband) that scale and rotate with head movement
2. **3D Texture Masks**: Face mesh textures mapped using triangle-based affine warping with proper UV mapping, backface culling, and z-sorting

**Important**: This application uses the **latest MediaPipe Face Landmarker API** (not the legacy `solutions.face_mesh`), which provides:
- VIDEO mode with timestamp tracking for optimal real-time performance
- Built-in smoothing when `num_faces=1`
- Better accuracy and performance
- Proper resource management via context managers

## Running the Application

```bash
# Activate virtual environment (if using one)
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download MediaPipe Face Landmarker model (REQUIRED - first time only)
./download_models.sh
# Or manually:
# wget -O models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# Run the application
python main.py
```

**Requirements:**
- Webcam
- PNG assets in `assets/` directory
- Face Landmarker model file (~3.6MB) in `models/` directory

## Architecture

The codebase follows Clean Architecture with files intentionally kept under 200 lines. The architecture has distinct layers:

### Core Pipeline Flow

1. **Camera Capture** ([camera.py](src/camera.py)) → Raw video frames
2. **Face Detection** ([face_detector.py](src/face_detector.py)) → 478 facial landmarks (2D pixel + 3D normalized)
3. **Landmark Stabilization** ([utils.py](src/utils.py)) → EMA smoothing to reduce jitter
4. **Gesture Detection** ([gesture_detector.py](src/gesture_detector.py)) → Analyzes landmark ratios to detect mouth open/brow raise
5. **Rendering** → Two parallel systems:
   - **2D Filters** ([graphics.py](src/graphics.py) + [filter_app.py](src/filter_app.py)) → Affine transformations for PNG overlays
   - **3D Texture Masks** ([mesh_renderer.py](src/mesh_renderer.py)) → Triangle-by-triangle UV warping
6. **UI Overlay** ([ui.py](src/ui.py)) → FPS, instructions, gesture status

### Key Technical Components

#### Face Detection & Stabilization (UPDATED - Latest API)
- [face_detector.py](src/face_detector.py) uses **MediaPipe Face Landmarker API** (not legacy Face Mesh)
- **VIDEO Mode**: Uses `RunningMode.VIDEO` with timestamp tracking for optimal real-time webcam processing
- **Timestamp Tracking**: Each frame gets monotonically increasing timestamp for proper tracking
- Returns `FaceLandmarks` dataclass containing:
  - `landmarks`: List of (x, y) pixel coordinates (478 landmarks)
  - `landmarks_3d`: List of (x, y, z) normalized coordinates for depth
  - `raw_landmarks`: Original MediaPipe NormalizedLandmarkList
- **Stabilization** ([utils.py](src/utils.py)): `LandmarkStabilizer` applies Exponential Moving Average (EMA) with `alpha=0.7` for responsiveness
  - Only applied to first detected face to avoid ID-swapping jitter
  - Resets when face is lost
  - Note: MediaPipe has built-in smoothing when `num_faces=1`, custom stabilizer provides additional control
- **Error Handling**: Comprehensive try-catch with logging for detection failures
- **Resource Management**: Proper `__enter__`/`__exit__` context manager support

#### Gesture Detection System
- [gesture_detector.py](src/gesture_detector.py) implements **ratio-based detection** for camera-distance invariance:
  - **Mouth Aspect Ratio (MAR)** = `vertical_distance / horizontal_distance`
  - **Brow Raise Ratio** = `avg_brow_to_eye_distance / face_height`
- **Debouncing logic**:
  - `required_frames`: Must hold gesture for N consecutive frames to trigger
  - `cooldown_frames`: Prevents rapid re-triggering after activation
- Callbacks registered in [filter_app.py](src/filter_app.py):
  - Mouth open → Cycles texture masks
  - Brow raise → Toggles glasses filter

#### 3D Mesh Rendering Pipeline
[mesh_renderer.py](src/mesh_renderer.py) implements sophisticated triangle-based warping:

1. **Triangle Topology**: Uses `config.FACE_MESH_TRIANGLES` (extracted from MediaPipe's `FACEMESH_TESSELATION`)
2. **Winding Order Calibration** (`calibrate_winding_order`):
   - Runs once on first frontal face detection (yaw threshold < 0.1)
   - Corrects triangle winding to ensure all visible triangles are Counter-Clockwise (CCW)
   - Enables accurate backface culling
3. **Z-Sorting (Painter's Algorithm)**:
   - Triangles sorted by average Z-depth (descending: far to near)
   - Prevents occlusion artifacts (e.g., cheek covering nose)
4. **Backface Culling**:
   - Calculates surface normal via cross product: `normal = edge1 × edge2`
   - Culls if `normal.z > 0` (facing away from camera)
5. **Lighting**:
   - Simple directional light from camera: `intensity = -normal_unit[2]`
   - Blends ambient (85%) + diffuse (15%) to avoid faceted look
6. **Seam Prevention**: Uses `cv2.polylines` with thickness=1 to overdraw triangle edges
7. **UV Mapping**:
   - **Priority 1**: Use `config.CANONICAL_FACE_MESH_UV` if defined (standard UV coordinates)
   - **Priority 2**: Detect landmarks in texture image using MediaPipe (`set_texture_landmarks_from_detection`)

#### 2D Filter System
- [graphics.py](src/graphics.py) handles image transformations:
  - Rotation angle: `math.atan2(delta_y, delta_x)`
  - Scale: Euclidean distance between landmarks × `scale_multiplier`
  - Alpha blending with proper edge handling
- [config.py](src/config.py) defines `FilterConfig` for each filter:
  - `anchor_landmarks`: Position reference points
  - `scale_landmarks`: Two points for dynamic sizing
  - `rotation_landmarks`: Two points for angle calculation
  - `x_offset`, `y_offset`: Fine-tune positioning

### Configuration System

[config.py](src/config.py) is the central configuration hub:

- **Facial Landmarks**: `FacialLandmarks` class maps MediaPipe's 468 landmarks to semantic names (e.g., `LEFT_EYE_OUTER = 33`)
- **Filter Definitions**: `FILTERS` dict maps filter names to `FilterConfig` objects
- **Texture Masks**: `TEXTURE_MASKS` dict for 3D face textures with opacity and debug settings
- **Gesture Thresholds**: `GESTURE_CONFIGS` dict with threshold/cooldown/required_frames per gesture
- **Display Settings**: `FPS_DISPLAY`, `SHOW_LANDMARKS`, `SHOW_GESTURE_STATUS`, etc.
- **Face Mesh Triangles**: Auto-generated from MediaPipe's tesselation on import

### Models and Data Flow

[models.py](src/models.py) defines data containers:
- `ARFilter`: Wraps `FilterConfig` + loaded PNG image
- `TextureMask`: Wraps `TextureMaskConfig` + loaded texture + `MeshRenderer` instance

Each `TextureMask` has its own `MeshRenderer` with calibrated winding order.

## Key Technical Details

### MediaPipe Integration
- Face Mesh provides 468 landmarks with 3D coordinates (x, y, z)
- Z-coordinate is normalized depth (smaller = closer to camera, larger = further)
- `refine_landmarks=True` adds iris landmarks (468-473)
- Supports up to 5 faces (`MAX_NUM_FACES = 5`)

### Coordinate Systems
- **MediaPipe Output**: Normalized (0.0-1.0) coordinates
- **Landmarks 2D**: Pixel coordinates (x, y) in frame
- **Landmarks 3D**: Normalized (x, y, z) with Z as depth
- **Texture UV**: Pixel coordinates in texture image OR normalized (0.0-1.0) if using `CANONICAL_FACE_MESH_UV`

### Performance Considerations
- MediaPipe Face Mesh runs at ~30 FPS on modern hardware
- Stabilization only applied to first face to avoid multi-face ID swapping
- Triangle rendering optimized with z-sorting and culling
- Set `MAX_NUM_FACES = 1` in config for better performance

### Screenshot System
- Press 'S' to save current frame
- Saves to `captures/` directory with timestamp: `capture_YYYYMMDD_HHMMSS.png`
- Screenshots include all active filters and UI overlays

## Keyboard Controls Reference

**2D Filters:**
- `1-4`: Toggle individual filters (glasses, mustache, beard, headband)
- `A`: Enable all 2D filters
- `D`: Disable all 2D filters

**3D Texture Masks:**
- `M`: Toggle masculine texture
- `F`: Toggle feminine texture
- `C`: Toggle custom texture
- `W`: Toggle wireframe debug mode
- `N`: Disable all texture masks

**Other:**
- `S`: Save screenshot
- `G`: Toggle gesture detection on/off
- `Q`: Quit application

## Adding New Features

### Adding a New 2D Filter

1. Add PNG asset to `assets/` directory (with alpha channel)
2. Define filter in [config.py](src/config.py):

```python
FILTERS["my_filter"] = FilterConfig(
    asset_path=ASSETS_DIR / "my_filter.png",
    anchor_landmarks=[1, 6],  # Use FacialLandmarks constants
    scale_landmarks=(61, 291),
    rotation_landmarks=(234, 454),
    scale_multiplier=1.5,
    y_offset=0,
    x_offset=0
)
```

3. Add keyboard binding in [filter_app.py](src/filter_app.py) `_handle_keyboard` method
4. Update `DEFAULT_ACTIVE_FILTERS` in config if needed

### Adding a New Texture Mask

1. Add texture image to `assets/` directory
2. Define in [config.py](src/config.py):

```python
TEXTURE_MASKS["new_mask"] = TextureMaskConfig(
    asset_path=ASSETS_DIR / "new_texture.jpg",
    opacity=0.7,
    debug_wireframe=False,
    subsample=1
)
```

3. The `MeshRenderer` will auto-detect face landmarks in the texture OR use `CANONICAL_FACE_MESH_UV` if defined

### Adding a New Gesture

1. Define gesture config in [config.py](src/config.py):

```python
GESTURE_CONFIGS["new_gesture"] = GestureConfig(
    threshold=0.5,
    cooldown_frames=30,
    required_frames=5,
    name="New Gesture"
)
```

2. Add `GestureType` enum value in [gesture_detector.py](src/gesture_detector.py)
3. Implement detection method (e.g., `_check_new_gesture`) using ratio-based calculations
4. Register callback in [filter_app.py](src/filter_app.py) `_setup_gesture_callbacks`

## Landmark Reference

MediaPipe Face Mesh provides 468 landmarks. See [MediaPipe Face Mesh visualization](https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png).

Key landmarks used in this project (defined in `FacialLandmarks` class):
- Eyes: 33, 133, 159, 145 (left), 362, 263, 386, 374 (right)
- Nose: 1 (tip), 6 (bridge), 2 (bottom)
- Mouth: 13 (upper lip top), 17 (lower lip bottom), 61/291 (corners)
- Chin: 152 (bottom)
- Forehead: 10 (center), 109/338 (left/right)
- Temples: 234/454 (used for rotation)
- Eyebrows: 107, 66, 105, 70 (left), 336, 296, 334, 300 (right)

## Common Issues

### Texture Mask Seams/Gaps
- Caused by anti-aliasing gaps between triangles
- Fixed in [mesh_renderer.py](src/mesh_renderer.py) using `cv2.polylines` overdraw

### Filter Jitter
- Adjust `alpha` in `LandmarkStabilizer` (lower = smoother, higher = responsive)
- Current setting: `alpha=0.7` for good responsiveness

### Texture Appearing Inside-Out
- Indicates incorrect triangle winding order
- Ensure face is frontal during first detection for calibration
- Check `calibrate_winding_order` was called successfully

### Occlusion Artifacts (Wrong Depth Order)
- Triangle rendering order matters
- Verify z-sorting is enabled (requires `landmarks_3d` parameter)

### Gesture False Triggers
- Increase `required_frames` for stricter detection
- Increase `threshold` value
- Adjust landmark indices in `GestureLandmarks` for better accuracy
