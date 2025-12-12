# AR Face Filter Application

A robust, modular, real-time Augmented Reality face filter application built with Python, OpenCV, and **MediaPipe Face Landmarker API (Latest)**. This application detects facial landmarks via webcam and overlays PNG assets that dynamically scale and rotate with head movements.

## Features

- **Real-time Face Detection**: Uses MediaPipe Face Landmarker API with 478 facial landmarks
- **VIDEO Mode**: Optimized timestamp-based tracking for real-time webcam processing
- **Multiple Face Support**: Detects and applies filters to up to 5 faces simultaneously
- **Interactive Filter Selection**: Toggle individual filters on/off with keyboard controls
- **Dynamic Transformations**: Filters automatically scale and rotate with head movement
- **Alpha Channel Support**: Proper transparency handling for PNG overlays
- **Clean Architecture**: Modular design with clear separation of concerns
- **Performance Optimized**: FPS display, efficient processing, and built-in smoothing
- **Proper Error Handling**: Comprehensive logging and error recovery

## Project Structure

```
ar-face-filter/
├── assets/               # PNG filter assets (user-provided)
│   ├── glasses.png
│   ├── mustache.png
│   ├── beard.png
│   └── headband.png
├── src/
│   ├── __init__.py
│   ├── config.py         # Configuration and landmark mappings (183 lines)
│   ├── camera.py         # Webcam handling (102 lines)
│   ├── face_detector.py  # MediaPipe Face Mesh logic (174 lines)
│   ├── graphics.py       # Image transformations (100 lines)
│   └── filter_app.py     # Main application logic (194 lines)
├── main.py               # Entry point (52 lines)
├── requirements.txt
└── README.md
```

All source files are **under 200 lines** as per Clean Architecture principles.

## Technical Implementation

### Architecture

The application follows **Clean Architecture** with distinct layers:

1. **Input Layer** ([camera.py](src/camera.py)): Webcam capture and frame management
2. **Processing Layer** ([face_detector.py](src/face_detector.py)): Face detection using MediaPipe
3. **Rendering Layer** ([graphics.py](src/graphics.py)): Image transformations and overlay
4. **Configuration** ([config.py](src/config.py)): Constants and landmark mappings
5. **Application** ([filter_app.py](src/filter_app.py)): Orchestrates all components

### Key Components

#### 1. Face Detection ([face_detector.py](src/face_detector.py))
- Uses **MediaPipe Face Landmarker API** (Latest) with 478 landmarks
- **VIDEO mode** with timestamp tracking for optimal real-time performance
- Converts normalized coordinates to pixel positions
- **Built-in smoothing** when `num_faces=1` + custom EMA stabilizer
- Proper error handling and logging
- Calculates center points from multiple landmarks

#### 2. Graphics Engine ([graphics.py](src/graphics.py))
- **Angle Calculation**: `math.atan2()` for rotation between landmarks
- **Distance Calculation**: Euclidean distance for dynamic scaling
- **Image Rotation**: OpenCV affine transformations with proper bounds
- **Alpha Blending**: Proper transparency overlay with edge case handling

#### 3. Filter Mapping ([config.py](src/config.py))

Each filter is configured with:
- **Anchor Landmarks**: Position reference points
- **Scale Landmarks**: Points to calculate size
- **Rotation Landmarks**: Points to calculate angle
- **Offsets**: Fine-tune positioning

```python
glasses → Eyes/nose bridge
mustache → Upper lip/philtrum
beard → Chin/jawline
headband → Forehead/hairline
```

## Installation

### Prerequisites
- Python 3.8+
- Webcam
- PNG filter assets (with alpha channel)

### Setup

1. **Clone/Download the project**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download MediaPipe Face Landmarker model**:

The application uses MediaPipe's Face Landmarker API which requires a model file.

**Option A - Automatic (Recommended)**:
```bash
./download_models.sh
```

**Option B - Manual**:
```bash
mkdir -p models
wget -O models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

The model file (`face_landmarker.task`, ~3.6MB) will be downloaded to the `models/` directory.

4. **Add filter assets**:
Place your PNG files (with transparency) in the `assets/` directory:
- `glasses.png`
- `mustache.png`
- `beard.png`
- `headband.png`

### Docker Setup (Alternative)

If you prefer running the application in Docker:

1. **Prerequisites**:
   - Docker and Docker Compose installed
   - Webcam access
   - X11 server for GUI display

2. **Allow X11 connections** (Linux):
```bash
xhost +local:docker
```

3. **Build and run**:
```bash
docker-compose build
docker-compose up
```

4. **Stop the application**:
```bash
docker-compose down
```

**Note**: Docker setup requires privileged mode for camera access and X11 forwarding for display. This works best on Linux systems.

## Usage

### Running the Application

```bash
python main.py
```

### Keyboard Controls

- **1**: Toggle glasses filter on/off
- **2**: Toggle mustache filter on/off
- **3**: Toggle beard filter on/off
- **4**: Toggle headband filter on/off
- **A**: Enable all filters
- **D**: Disable all filters
- **Q**: Quit the application

By default, all filters start disabled. Press the number keys to enable individual filters, or press 'A' to enable all at once.

### Configuration

Edit [src/config.py](src/config.py) to:
- Adjust camera resolution
- Modify filter positions and sizes
- Enable/disable FPS display
- Enable debug mode (show landmarks)

```python
# Example configuration
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS_DISPLAY = True
SHOW_LANDMARKS = False  # Debug mode
```

## Customization

### Adding New Filters

1. Add your PNG asset to `assets/` directory
2. Edit [src/config.py](src/config.py) and add to `FILTERS` dictionary:

```python
"my_filter": FilterConfig(
    asset_path=ASSETS_DIR / "my_filter.png",
    anchor_landmarks=[...],      # Positioning landmarks
    scale_landmarks=(..., ...),  # Sizing landmarks
    rotation_landmarks=(..., ...), # Rotation landmarks
    scale_multiplier=2.0,        # Size adjustment
    y_offset=0,                  # Vertical offset
    x_offset=0                   # Horizontal offset
)
```

### Landmark Reference

MediaPipe Face Mesh provides 468 landmarks. Key landmarks used in this project:

```python
Eyes: 33, 133, 362, 263
Nose: 1, 6, 2
Mouth: 13, 14, 61, 291
Chin: 152
Forehead: 10, 109, 338
Temples: 234, 454
```

Full landmark map: [MediaPipe Face Mesh](https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png)

## Technical Details

### Geometry Mathematics

**Rotation Angle Calculation**:
```python
angle = math.atan2(delta_y, delta_x)  # Radians
angle_degrees = math.degrees(angle)
```

**Scale Calculation**:
```python
distance = sqrt((x2-x1)² + (y2-y1)²)
target_width = distance × scale_multiplier
```

**Alpha Blending**:
```python
alpha = overlay_alpha_channel / 255.0
result = alpha × overlay + (1 - alpha) × background
```

### Performance Considerations

- MediaPipe Face Mesh runs at ~30 FPS on modern hardware
- Image transformations use OpenCV's optimized functions
- Single face detection (`MAX_NUM_FACES = 1`) for optimal performance

## Dependencies

- **opencv-python** (>=4.8.0): Image processing and camera handling
- **mediapipe** (>=0.10.0): Face landmark detection
- **numpy** (>=1.24.0): Numerical operations

## Troubleshooting

### Camera Not Opening
- Check camera permissions
- Verify camera index (default: 0) in [config.py](src/config.py)
- Try changing `CAMERA_INDEX` if multiple cameras

### Filters Not Loading
- Ensure PNG files exist in `assets/` directory
- Check file names match configuration
- Verify PNG files have alpha channel (RGBA)

### Poor Performance
- Reduce `FRAME_WIDTH` and `FRAME_HEIGHT`
- Ensure good lighting for face detection
- Close other camera applications

### Filters Misaligned
- Adjust `scale_multiplier` for size
- Modify `x_offset` and `y_offset` for position
- Check landmark indices for your use case

## License

This project is provided as-is for educational and demonstration purposes.

## Acknowledgments

- **MediaPipe**: Google's ML solution for face detection
- **OpenCV**: Computer vision library
- **Clean Architecture**: Robert C. Martin's architectural principles
