# MediaPipe Best Practices Implementation

This document outlines how this project implements MediaPipe best practices based on the [official MediaPipe documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker).

## ✅ Best Practices Implemented

### 1. Using Face Landmarker API (Not Legacy Face Mesh)

**Official Recommendation**: Use the latest `mediapipe.tasks.python.vision.FaceLandmarker` API instead of the deprecated `mediapipe.solutions.face_mesh.FaceMesh`.

**Implementation** ([face_detector.py](src/face_detector.py:71-83)):
```python
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,  # VIDEO mode for webcam
    num_faces=self.max_faces,
    min_face_detection_confidence=self.min_detection_confidence,
    min_tracking_confidence=self.min_tracking_confidence
)

self.landmarker = vision.FaceLandmarker.create_from_options(options)
```

### 2. Proper Running Mode Selection

**Official Recommendation**: Choose the appropriate running mode based on your input type:
- `IMAGE` for single images
- `VIDEO` for pre-recorded video files
- `LIVE_STREAM` for real-time camera with async callbacks

**Implementation**: We use **VIDEO mode** for real-time webcam processing with synchronous processing ([face_detector.py](src/face_detector.py:75)):
```python
running_mode=vision.RunningMode.VIDEO,  # VIDEO mode for webcam with timestamp
```

For static texture images, we use **IMAGE mode** ([mesh_renderer.py](src/mesh_renderer.py:142), [utils.py](src/utils.py:56)):
```python
running_mode=vision.RunningMode.IMAGE,  # IMAGE mode for static texture
```

### 3. Timestamp Tracking for VIDEO Mode

**Official Recommendation**: VIDEO mode requires monotonically increasing timestamps in milliseconds for proper tracking.

**Implementation** ([face_detector.py](src/face_detector.py:86-87, 112-113)):
```python
# Initialize timestamp tracking
self.start_time = time.time()
self.frame_count = 0

# In detect() method:
self.frame_count += 1
timestamp_ms = int((time.time() - self.start_time) * 1000)
results = self.landmarker.detect_for_video(mp_image, timestamp_ms)
```

### 4. Proper Resource Management

**Official Recommendation**: Use context managers or explicitly call `close()` to release MediaPipe resources.

**Implementation** ([face_detector.py](src/face_detector.py:239-255)):
```python
def release(self) -> None:
    """Release MediaPipe resources."""
    try:
        if hasattr(self, 'landmarker') and self.landmarker:
            self.landmarker.close()
            logger.info("Face Landmarker resources released")
    except Exception as e:
        logger.warning(f"Error releasing Face Landmarker: {e}")

def __enter__(self):
    """Context manager entry."""
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    """Context manager exit with automatic resource cleanup."""
    self.release()
    return False
```

Usage:
```python
with FaceDetector() as detector:
    # detector will be automatically cleaned up
    pass
```

### 5. Confidence Threshold Configuration

**Official Recommendation**: Configure three confidence parameters based on your use case:
- `min_face_detection_confidence`: Initial detection threshold (default 0.5)
- `min_face_presence_confidence`: Landmark quality threshold (default 0.5)
- `min_tracking_confidence`: Tracking reliability threshold (default 0.5)

**Implementation** ([config.py](src/config.py:20-22)):
```python
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
```

### 6. Error Handling

**Official Recommendation**: Implement proper error handling for detection failures.

**Implementation** ([face_detector.py](src/face_detector.py:106-162)):
```python
try:
    # Convert BGR to RGB
    rgb_frame = frame[:, :, ::-1]

    # Generate timestamp
    timestamp_ms = int((time.time() - self.start_time) * 1000)

    # Convert to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Process frame
    results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

    # Handle results...

except Exception as e:
    logger.error(f"Error during face detection: {e}")
    return []
```

### 7. Logging

**Implementation**: Comprehensive logging throughout the codebase:
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Loading Face Landmarker model from: {model_path}")
logger.error(f"Error during face detection: {e}")
logger.warning("No face detected in texture")
```

### 8. Model File Management

**Official Recommendation**: Download and reference the model file correctly.

**Implementation**:
- Model stored in `models/face_landmarker.task`
- Automatic download script: `download_models.sh`
- Model path auto-detection with fallback ([face_detector.py](src/face_detector.py:59-67)):
```python
if model_path is None:
    model_path = str(Path(__file__).parent.parent / "models" / "face_landmarker.task")

if not Path(model_path).exists():
    raise FileNotFoundError(
        f"Face Landmarker model not found at {model_path}. "
        f"Download it from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    )
```

### 9. Built-in Smoothing

**Official Recommendation**: When `num_faces=1`, MediaPipe provides built-in landmark smoothing for better results.

**Implementation**: We set `num_faces` in config and additionally apply custom EMA smoothing for extra control ([face_detector.py](src/face_detector.py:89-92)):
```python
# NOTE: MediaPipe has built-in smoothing when num_faces=1
# We keep custom stabilizer for consistency and additional control
self.stabilizer = LandmarkStabilizer(alpha=0.7)
```

### 10. Image Format Conversion

**Official Recommendation**: MediaPipe expects RGB format, convert OpenCV's BGR appropriately.

**Implementation** ([face_detector.py](src/face_detector.py:107-116)):
```python
# Convert BGR to RGB (MediaPipe uses RGB)
rgb_frame = frame[:, :, ::-1]

# Convert to MediaPipe Image
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
```

## Performance Optimizations

### 1. VIDEO Mode for Webcam
Using VIDEO mode instead of IMAGE mode enables:
- Frame-skipping via tracking (reduces latency)
- Temporal smoothing across frames
- Better performance than processing each frame independently

### 2. Efficient Landmark Extraction
Direct conversion to NumPy arrays for processing ([face_detector.py](src/face_detector.py:164-200)):
```python
def _normalize_landmarks_np(self, face_landmarks, width: int, height: int) -> np.ndarray:
    """Convert normalized landmarks to pixel coordinates (NumPy)."""
    coords = np.array([(lm.x * width, lm.y * height) for lm in face_landmarks], dtype=np.float32)
    return coords
```

## References

- [Face Landmarker Python Guide](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)
- [Face Landmarker Overview](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
- [MediaPipe GitHub Repository](https://github.com/google-ai-edge/mediapipe)

## Migration Notes

This project was migrated from the legacy `mp.solutions.face_mesh.FaceMesh` API to the modern `vision.FaceLandmarker` API. Key changes:

1. **API**: `FaceMesh` → `FaceLandmarker`
2. **Mode**: No running mode → `RunningMode.VIDEO`
3. **Method**: `process(frame)` → `detect_for_video(mp_image, timestamp_ms)`
4. **Results**: `multi_face_landmarks` → `face_landmarks`
5. **Landmarks**: `.landmark` attribute → direct iteration over landmarks
6. **Count**: 468 landmarks → 478 landmarks (with refined iris)

## Verification

To verify the implementation follows best practices, check:

```bash
# Test Face Landmarker initialization
source venv/bin/activate
python -c "from src.face_detector import FaceDetector; d = FaceDetector(); print('✓ OK'); d.release()"

# Check model file exists
ls -lh models/face_landmarker.task

# Run full application
python main.py
```

Expected output should show:
- `INFO:src.face_detector:Loading Face Landmarker model from: ...`
- `INFO:src.face_detector:Face Landmarker initialized (VIDEO mode, max_faces=...)`
- No errors during initialization
