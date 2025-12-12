# AR Face Filter Application 🎭

A robust, modular, real-time Augmented Reality face filter application built with Python, OpenCV, and **MediaPipe's latest APIs** (Face Landmarker + Hand Landmarker). This application detects facial landmarks and hand gestures via webcam, applying dynamic 2D PNG overlays and 3D face mesh textures that respond to head movements and gestures.

## ✨ Features

### Face Tracking & Filters
- 🎯 **Real-time Face Detection**: MediaPipe Face Landmarker API with 478 facial landmarks
- 📹 **VIDEO Mode**: Optimized timestamp-based tracking for real-time webcam processing
- 👥 **Multiple Face Support**: Detects and applies filters to up to 5 faces simultaneously
- 🎨 **2D PNG Filters**: Glasses, mustache, beard, headband with dynamic scaling and rotation
- 🌀 **3D Face Mesh Textures**: Triangle-based UV mapping with backface culling and z-sorting
- 🎭 **Alpha Channel Support**: Proper transparency handling for PNG overlays
- ⚡ **Performance Optimized**: Built-in smoothing, EMA stabilization, ~30 FPS

### Gesture Controls
- 😮 **Face Gestures**:
  - Mouth open → Cycle through 3D texture masks
  - Eyebrows raised → Toggle glasses filter
- 👌 **Hand Gesture - OK Sign**:
  - Make "OK" gesture (thumb + index forming circle) → Take screenshot!
  - Built-in cooldown to prevent accidental triggers
- 🔄 **Toggle Gestures**: Enable/disable gesture detection on-the-fly

### Developer Features
- 🏗️ **Clean Architecture**: Modular design with separation of concerns
- 📝 **Comprehensive Logging**: Error handling and debugging information
- 🔒 **Resource Management**: Proper cleanup via context managers
- 📚 **Well Documented**: Inline comments and external documentation

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam
- PNG filter assets (with alpha channel)

### Step-by-Step Setup

#### 1️⃣ Clone or Download the Project

```bash
git clone https://github.com/GratiaManullang03/AR-Face-Filter-Application
cd AR-Face-Filter-Application
```

#### 2️⃣ Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### 3️⃣ Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies installed:**
- `opencv-python` (>=4.8.0) - Image processing and camera handling
- `mediapipe` (>=0.10.0) - Face and hand landmark detection
- `numpy` (>=1.24.0) - Numerical operations

#### 4️⃣ Download MediaPipe Model Files

The application requires two AI model files (~11MB total):

**Option A - Automatic Download (Recommended)**:
```bash
./download_models.sh
```

**Option B - Manual Download**:
```bash
mkdir -p models

# Face Landmarker model (~3.6MB)
wget -O models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# Hand Landmarker model (~7.6MB)
wget -O models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

Models will be saved to the `models/` directory.

#### 5️⃣ Add Filter Assets

Place your PNG files (with transparency/alpha channel) in the `assets/` directory:

**Required 2D Filters:**
- `assets/glasses.png`
- `assets/mustache.png`
- `assets/beard.png`
- `assets/headband.png`

**Optional 3D Texture Masks:**
- `assets/masculine_face.jpg`
- `assets/feminine_face.jpg`
- `assets/custom_texture_mask.jpg`

**Note**: Asset files are not included in the repository. You need to provide your own PNG/JPG files.

#### 6️⃣ Run the Application

```bash
python main.py
```

The webcam window will open showing the live feed with filters!

---

## 🐳 Docker Setup (Alternative)

For easy deployment without manual setup, use Docker:

```bash
# Allow X11 display access (Linux)
xhost +local:docker

# Build and run
docker-compose build
docker-compose up
```

Models are automatically downloaded during build. Screenshots are saved to `./captures/` on your host machine.

**For detailed Docker instructions** (Windows, macOS, troubleshooting), see [README.Docker.md](README.Docker.md)

---

## 🎮 Controls

### Keyboard Controls

#### 2D Filters (PNG Overlays)
| Key | Action |
|-----|--------|
| `1` | Toggle **Glasses** filter |
| `2` | Toggle **Mustache** filter |
| `3` | Toggle **Beard** filter |
| `4` | Toggle **Headband** filter |
| `A` | Enable **ALL** 2D filters |
| `D` | Disable **ALL** 2D filters |

#### 3D Texture Masks
| Key | Action |
|-----|--------|
| `M` | Toggle **Masculine** face texture |
| `F` | Toggle **Feminine** face texture |
| `C` | Toggle **Custom** texture mask |
| `W` | Toggle **Wireframe** debug mode |
| `N` | Disable all texture masks |

#### Other Controls
| Key | Action |
|-----|--------|
| `S` | Save **Screenshot** to `captures/` |
| `G` | Toggle **Gesture Detection** on/off |
| `H` | Toggle **Hand Landmarks** visualization |
| `Q` | **Quit** application |

### Gesture Controls

#### Face Gestures (Toggle with `G`)
- 😮 **Open your mouth wide** → Cycles through 3D texture masks
- 🤨 **Raise your eyebrows** → Toggles glasses filter on/off

#### Hand Gesture
- 👌 **OK Sign** (thumb + index forming circle, other fingers up) → **Captures screenshot!**
  - Works continuously in background
  - 2-second cooldown between captures
  - Visual feedback on screen

**Tip**: Press `H` to see hand landmarks and verify gesture detection!

---

## 📂 Project Structure

```
ar-face-filter/
├── assets/                      # PNG/JPG filter assets (user-provided)
│   ├── glasses.png
│   ├── mustache.png
│   ├── beard.png
│   ├── headband.png
│   ├── masculine_face.jpg
│   ├── feminine_face.jpg
│   └── custom_texture_mask.jpg
│
├── models/                      # MediaPipe AI models (downloaded via script)
│   ├── face_landmarker.task    (~3.6MB)
│   ├── hand_landmarker.task    (~7.6MB)
│   └── .gitkeep
│
├── src/                         # Source code (Clean Architecture)
│   ├── __init__.py
│   ├── camera.py               # Webcam capture
│   ├── config.py               # Configuration & constants
│   ├── face_detector.py        # Face Landmarker API (VIDEO mode)
│   ├── hand_detector.py        # Hand Landmarker API & OK gesture
│   ├── gesture_detector.py     # Face gesture detection (mouth, brow)
│   ├── graphics.py             # 2D image transformations
│   ├── mesh_renderer.py        # 3D mesh rendering with UV mapping
│   ├── models.py               # Data models
│   ├── ui.py                   # UI overlay & FPS display
│   ├── utils.py                # Utilities & stabilization
│   └── filter_app.py           # Main application orchestrator
│
├── captures/                    # Screenshots saved here (auto-created)
├── main.py                      # Application entry point
├── requirements.txt             # Python dependencies
├── download_models.sh           # Model download script
├── CLAUDE.md                    # Developer documentation
├── MEDIAPIPE_BEST_PRACTICES.md  # MediaPipe implementation guide
├── UPGRADE_SUMMARY.md           # API migration notes
└── README.md                    # This file
```

**All source files follow Clean Architecture principles with files under 200 lines.**

---

## 🔧 Configuration

Edit [src/config.py](src/config.py) to customize:

### Camera Settings
```python
CAMERA_INDEX = 0           # Camera device (0 = default)
FRAME_WIDTH = 1280         # Resolution width
FRAME_HEIGHT = 720         # Resolution height
```

### Display Settings
```python
FPS_DISPLAY = True         # Show FPS counter
SHOW_LANDMARKS = False     # Debug: Show face landmarks
SHOW_GESTURE_STATUS = True # Show gesture detection status
```

### Detection Settings
```python
MAX_NUM_FACES = 5                  # Max faces to detect
MIN_DETECTION_CONFIDENCE = 0.5     # Face detection threshold
MIN_TRACKING_CONFIDENCE = 0.5      # Tracking threshold
```

### Filter Defaults
```python
DEFAULT_ACTIVE_FILTERS = {
    "glasses": False,
    "mustache": False,
    "beard": False,
    "headband": False
}
```

---

## 🎨 Customization

### Adding a New 2D Filter

1. **Add PNG asset** with alpha channel to `assets/my_filter.png`

2. **Edit [src/config.py](src/config.py)**, add to `FILTERS` dictionary:

```python
"my_filter": FilterConfig(
    asset_path=ASSETS_DIR / "my_filter.png",
    anchor_landmarks=[1, 6],           # Position: nose bridge
    scale_landmarks=(61, 291),         # Scale: mouth corners
    rotation_landmarks=(234, 454),     # Rotation: temples
    scale_multiplier=1.5,              # Size adjustment
    y_offset=0,                        # Vertical offset (pixels)
    x_offset=0                         # Horizontal offset (pixels)
)
```

3. **Add keyboard binding** in [src/filter_app.py](src/filter_app.py) `_handle_keyboard()` method

### Adding a New 3D Texture Mask

1. **Add texture image** (JPG/PNG with visible face) to `assets/`

2. **Edit [src/config.py](src/config.py)**, add to `TEXTURE_MASKS` dictionary:

```python
"my_texture": TextureMaskConfig(
    asset_path=ASSETS_DIR / "my_texture.jpg",
    opacity=0.8,              # 0.0 = transparent, 1.0 = opaque
    debug_wireframe=False,    # Show triangle wireframe
    subsample=1               # Subsample factor (1 = full resolution)
)
```

3. **Add keyboard binding** in [src/filter_app.py](src/filter_app.py) `_handle_keyboard()` method

### MediaPipe Landmark Reference

MediaPipe Face Mesh provides 478 landmarks. Key landmarks:

```python
# Eyes
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_OUTER = 362
RIGHT_EYE_INNER = 263

# Nose
NOSE_TIP = 1
NOSE_BRIDGE = 6
NOSE_BOTTOM = 2

# Mouth
UPPER_LIP_TOP = 13
LOWER_LIP_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

# Face outline
CHIN_BOTTOM = 152
FOREHEAD_CENTER = 10
LEFT_TEMPLE = 234
RIGHT_TEMPLE = 454
```

**Full visualization**: [MediaPipe Face Mesh Landmarks](https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png)

---

## 🏗️ Architecture

### Core Pipeline Flow

1. **Camera Capture** → Raw video frames (BGR format)
2. **Face Detection** → 478 facial landmarks (2D + 3D coordinates)
3. **Hand Detection** → 21 hand landmarks + gesture recognition
4. **Landmark Stabilization** → EMA smoothing to reduce jitter
5. **Face Gesture Detection** → Mouth open, brow raise detection
6. **Rendering** → Apply 2D filters + 3D mesh textures
7. **UI Overlay** → FPS, instructions, gesture status
8. **Display** → Show final frame with all effects

### Key Technical Components

#### 1. Face Detection ([src/face_detector.py](src/face_detector.py))
- Uses **MediaPipe Face Landmarker API** (Latest, not legacy)
- **VIDEO mode** with monotonically increasing timestamps
- Returns 478 landmarks with 3D depth information
- Custom EMA stabilization (α=0.7) for first detected face
- Comprehensive error handling with logging

#### 2. Hand Detection ([src/hand_detector.py](src/hand_detector.py))
- Uses **MediaPipe Hand Landmarker API**
- Detects up to 2 hands simultaneously
- **OK Gesture Recognition**: Thumb + index forming circle
- Ratio-based detection for distance invariance
- 2-second cooldown to prevent double-triggers

#### 3. 3D Mesh Rendering ([src/mesh_renderer.py](src/mesh_renderer.py))
- Triangle-based affine transformations
- **Backface Culling**: Hides invisible triangles
- **Z-Sorting**: Painter's algorithm for correct occlusion
- **Lighting**: Simple directional lighting (ambient + diffuse)
- **UV Mapping**: Proper texture coordinate mapping
- **Seam Prevention**: Overdraw triangle edges

#### 4. Gesture Detection ([src/gesture_detector.py](src/gesture_detector.py))
- **Ratio-based** detection (distance invariant)
- **Debouncing**: Required consecutive frames to trigger
- **Cooldown**: Prevents rapid re-triggering
- Callbacks for mouth open, brow raise

---

## 📊 Performance

- **Frame Rate**: ~30 FPS on modern hardware (i5/Ryzen 5 or better)
- **Latency**: ~33ms per frame (VIDEO mode optimization)
- **Memory**: ~200-300MB RAM usage
- **CPU**: Single-threaded (MediaPipe uses GPU when available)

### Optimization Tips

1. **Reduce Resolution**:
   ```python
   # In config.py
   FRAME_WIDTH = 640
   FRAME_HEIGHT = 480
   ```

2. **Single Face Mode** (faster):
   ```python
   MAX_NUM_FACES = 1  # Enables built-in MediaPipe smoothing
   ```

3. **Disable Hand Detection** (if not needed):
   ```python
   # Comment out in filter_app.py
   # hands = self.hand_detector.detect(frame)
   ```

---

## 🐛 Troubleshooting

### Camera Not Opening
**Problem**: "Error: Could not open camera"

**Solutions**:
- Check camera permissions (especially on macOS/Linux)
- Verify camera is not in use by another application
- Try different camera index:
  ```python
  # In config.py
  CAMERA_INDEX = 1  # Try 0, 1, 2, etc.
  ```
- On Linux, check `/dev/video*` permissions

### Model Files Not Found
**Problem**: "FileNotFoundError: Face Landmarker model not found"

**Solutions**:
```bash
# Re-run model download script
./download_models.sh

# Or manually download
mkdir -p models
wget -O models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
wget -O models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

### Filters Not Showing
**Problem**: Filters invisible or misaligned

**Solutions**:
- Ensure PNG files have **alpha channel** (RGBA, not RGB)
- Check file names match exactly (case-sensitive)
- Verify assets exist: `ls -la assets/`
- Adjust positioning:
  ```python
  # In config.py, modify FilterConfig
  scale_multiplier=2.0,  # Increase size
  y_offset=-20,          # Move up
  x_offset=10            # Move right
  ```

### Poor Performance / Low FPS
**Problem**: Choppy video, FPS < 20

**Solutions**:
1. Reduce resolution (see Optimization Tips above)
2. Disable gesture detection: Press `G`
3. Close other applications using camera/CPU
4. Ensure good lighting (helps detection speed)
5. Update graphics drivers

### Hand Gesture Not Working
**Problem**: OK gesture not triggering screenshot

**Solutions**:
1. Press `H` to visualize hand landmarks
2. Ensure good lighting and clear hand visibility
3. Form clear "OK" gesture: thumb + index circle, other fingers extended
4. Check cooldown (wait 2 seconds between gestures)
5. Verify model downloaded:
   ```bash
   ls -lh models/hand_landmarker.task
   ```

### Texture Mask Inside-Out
**Problem**: 3D mesh appears inverted

**Solutions**:
- Ensure face is **frontal** during first detection (for calibration)
- Check winding order calibration succeeded (check logs)
- Press `W` to toggle wireframe debug mode
- Green wireframe = correct, Red = backface (culled)

---

## 📚 Documentation

- **[CLAUDE.md](CLAUDE.md)** - Developer guide for working with this codebase
- **[MEDIAPIPE_BEST_PRACTICES.md](MEDIAPIPE_BEST_PRACTICES.md)** - MediaPipe API best practices

---

## 🔬 Technical Details

### Geometry Mathematics

**Rotation Angle**:
```python
angle = math.atan2(y2 - y1, x2 - x1)  # Radians
angle_degrees = math.degrees(angle)
```

**Scale Calculation**:
```python
distance = math.hypot(x2 - x1, y2 - y1)  # Euclidean distance
target_width = distance × scale_multiplier
```

**Alpha Blending**:
```python
alpha = overlay_alpha / 255.0
result = alpha × foreground + (1 - alpha) × background
```

### MediaPipe Integration

**Face Landmarker API** (VIDEO mode):
```python
from mediapipe.tasks.python import vision

options = vision.FaceLandmarkerOptions(
    running_mode=vision.RunningMode.VIDEO,
    num_faces=5,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = vision.FaceLandmarker.create_from_options(options)
results = landmarker.detect_for_video(mp_image, timestamp_ms)
```

**Hand Landmarker API** (VIDEO mode):
```python
options = vision.HandLandmarkerOptions(
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = vision.HandLandmarker.create_from_options(options)
results = landmarker.detect_for_video(mp_image, timestamp_ms)
```

---

## 📄 License

This project is provided as-is for educational and demonstration purposes.

---

## 🙏 Acknowledgments

- **[MediaPipe](https://developers.google.com/mediapipe)** - Google's ML solutions for face and hand detection
- **[OpenCV](https://opencv.org/)** - Computer vision library
- **[Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)** - Robert C. Martin's architectural principles

---

## 🆘 Support

For issues or questions:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Review documentation in [CLAUDE.md](CLAUDE.md)
3. Verify you're using the latest MediaPipe APIs (not legacy `mp.solutions`)
4. Check logs for error messages (logged to console)

---

**Happy filtering! 🎉**
