"""
Main AR Filter application logic.
Orchestrates camera, face detection, and filter rendering.
"""

import cv2
import numpy as np
import time
from typing import Dict

from .camera import Camera
from .face_detector import FaceDetector, FaceLandmarks
from .graphics import GraphicsEngine
from . import config


class ARFilter:
    """Single AR filter with loaded image and configuration."""

    def __init__(self, cfg: config.FilterConfig):
        self.config, self.image = cfg, None
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


class FilterApplication:
    """Main application orchestrating the AR face filter system."""

    def __init__(self):
        """Initialize the filter application."""
        self.camera = Camera()
        self.face_detector = FaceDetector()
        self.graphics = GraphicsEngine()
        self.filters: Dict[str, ARFilter] = {}
        self.active_filters: Dict[str, bool] = {}
        self.running = False

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0

        # Load all filters
        self._load_filters()
        # Initialize active filters state
        self.active_filters = config.DEFAULT_ACTIVE_FILTERS.copy()

    def _load_filters(self) -> None:
        """Load all AR filters from configuration."""
        for name, cfg in config.FILTERS.items():
            flt = ARFilter(cfg)
            if flt.is_loaded():
                self.filters[name] = flt
                print(f"Loaded: {name}")
            else:
                print(f"Skipped: {name}")

    def _apply_filter(self, frame: np.ndarray, face: FaceLandmarks,
                     flt: ARFilter) -> np.ndarray:
        """Apply a single AR filter to the frame."""
        if not flt.is_loaded():
            return frame

        # Get rotation and scale points
        get = lambda i: self.face_detector.get_landmark_point(face, i)
        rot_p1, rot_p2 = get(flt.config.rotation_landmarks[0]), get(flt.config.rotation_landmarks[1])
        if not (rot_p1 and rot_p2):
            return frame

        scale_p1, scale_p2 = get(flt.config.scale_landmarks[0]), get(flt.config.scale_landmarks[1])
        if not (scale_p1 and scale_p2):
            return frame

        # Calculate transformations
        angle = self.graphics.calculate_angle(rot_p1, rot_p2)
        width = int(self.graphics.calculate_distance(scale_p1, scale_p2) * flt.config.scale_multiplier)
        if width <= 0:
            return frame

        # Apply transformations
        resized = self.graphics.resize_image(flt.image, width)
        rotated = self.graphics.rotate_image(resized, angle)
        anchor = self.face_detector.get_landmarks_center(face, flt.config.anchor_landmarks)
        pos = (anchor[0] + flt.config.x_offset, anchor[1] + flt.config.y_offset)

        return self.graphics.overlay_image(frame, rotated, pos)

    def _draw_landmarks(self, frame: np.ndarray, face: FaceLandmarks) -> np.ndarray:
        """Draw facial landmarks (debug mode)."""
        for x, y in face.landmarks:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        return frame

    def _update_fps(self) -> None:
        """Update FPS calculation."""
        self.frame_count += 1
        if (t := time.time() - self.start_time) > 1.0:
            self.fps = self.frame_count / t
            self.frame_count, self.start_time = 0, time.time()

    def _draw_fps(self, frame: np.ndarray) -> np.ndarray:
        """Draw FPS counter."""
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame

    def _draw_instructions(self, frame: np.ndarray) -> np.ndarray:
        """Draw keyboard instructions and filter status."""
        y, h = 60, 25
        keys = {"1": "glasses", "2": "mustache", "3": "beard", "4": "headband"}

        for key, name in keys.items():
            if name in self.active_filters:
                status = "ON" if self.active_filters[name] else "OFF"
                color = (0, 255, 0) if self.active_filters[name] else (0, 0, 255)
                cv2.putText(frame, f"[{key}] {name.capitalize()}: {status}",
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y += h

        cv2.putText(frame, "[A] All ON  [D] All OFF  [Q] Quit",
                   (10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def _handle_keyboard(self, key: int) -> bool:
        """Handle keyboard input for filter toggling."""
        if key == ord('q'):
            return False

        keys_map = {ord('1'): "glasses", ord('2'): "mustache",
                   ord('3'): "beard", ord('4'): "headband"}

        if key in keys_map:
            name = keys_map[key]
            self.active_filters[name] = not self.active_filters.get(name, False)
        elif key in (ord('a'), ord('A')):
            for name in self.active_filters:
                self.active_filters[name] = True
        elif key in (ord('d'), ord('D')):
            for name in self.active_filters:
                self.active_filters[name] = False
        return True

    def run(self) -> None:
        """Main application loop."""
        if not self.camera.open():
            print("Error: Could not open camera")
            return

        print(f"AR Face Filter ({len(self.filters)} filters) | 1-4=Toggle A=All D=None Q=Quit")
        self.running = True

        try:
            while self.running:
                success, frame = self.camera.read()
                if not success:
                    break

                for face in self.face_detector.detect(frame):
                    for name, flt in self.filters.items():
                        if self.active_filters.get(name, False):
                            frame = self._apply_filter(frame, face, flt)
                    if config.SHOW_LANDMARKS:
                        frame = self._draw_landmarks(frame, face)

                self._update_fps()
                if config.FPS_DISPLAY:
                    frame = self._draw_fps(frame)
                if config.SHOW_INSTRUCTIONS:
                    frame = self._draw_instructions(frame)

                cv2.imshow(config.WINDOW_NAME, frame)
                if (k := cv2.waitKey(1) & 0xFF) != 255 and not self._handle_keyboard(k):
                    break
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources."""
        self.running = False
        self.camera.release()
        self.face_detector.release()
        cv2.destroyAllWindows()
