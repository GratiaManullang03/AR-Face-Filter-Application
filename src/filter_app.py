"""
Main AR Filter application logic.
Orchestrates camera, face detection, gesture detection, and filter rendering.
"""

import cv2
import numpy as np
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .camera import Camera
from .face_detector import FaceDetector, FaceLandmarks
from .graphics import GraphicsEngine
from .gesture_detector import GestureDetector, GestureType, GestureEvent
from . import config
from .models import ARFilter, TextureMask
from .ui import UIManager


class FilterApplication:
    """Main application orchestrating the AR face filter system."""

    def __init__(self):
        """Initialize the filter application."""
        self.camera = Camera()
        self.face_detector = FaceDetector()
        self.graphics = GraphicsEngine()
        self.gesture_detector = GestureDetector(
            enabled=config.GESTURE_CONTROL_ENABLED
        )
        self.ui = UIManager()
        
        self.filters: Dict[str, ARFilter] = {}
        self.texture_masks: Dict[str, TextureMask] = {}
        self.active_filters: Dict[str, bool] = {}
        self.active_texture_mask: Optional[str] = None
        self.running = False

        # Filter/mask lists for cycling
        self._filter_names: List[str] = []
        self._mask_names: List[str] = []

        # Load all filters
        self._load_filters()
        self._load_texture_masks()
        
        # Initialize active filters state
        self.active_filters = config.DEFAULT_ACTIVE_FILTERS.copy()
        self.active_texture_mask = config.DEFAULT_ACTIVE_TEXTURE_MASK

        # Setup gesture callbacks
        self._setup_gesture_callbacks()

    def _load_filters(self) -> None:
        """Load all AR filters from configuration."""
        for name, cfg in config.FILTERS.items():
            flt = ARFilter(cfg)
            if flt.is_loaded():
                self.filters[name] = flt
                self._filter_names.append(name)
                print(f"Loaded filter: {name}")
            else:
                print(f"Skipped filter: {name}")

    def _load_texture_masks(self) -> None:
        """Load all texture masks from configuration."""
        for name, cfg in config.TEXTURE_MASKS.items():
            mask = TextureMask(cfg)
            if mask.is_loaded():
                self.texture_masks[name] = mask
                self._mask_names.append(name)
            else:
                print(f"Skipped texture mask: {name}")

    def _setup_gesture_callbacks(self) -> None:
        """Register callbacks for gesture events."""
        # Mouth open -> Cycle texture masks
        self.gesture_detector.register_callback(
            GestureType.MOUTH_OPEN,
            self._on_mouth_open
        )
        
        # Brow raise -> Toggle glasses filter
        self.gesture_detector.register_callback(
            GestureType.BROW_RAISE,
            self._on_brow_raise
        )

    def _on_mouth_open(self, event: GestureEvent) -> None:
        """Handle mouth open gesture."""
        if not self._mask_names:
            return

        if self.active_texture_mask is None:
            self.active_texture_mask = self._mask_names[0]
        else:
            try:
                current_idx = self._mask_names.index(self.active_texture_mask)
                next_idx = (current_idx + 1) % (len(self._mask_names) + 1)
                
                if next_idx >= len(self._mask_names):
                    self.active_texture_mask = None
                else:
                    self.active_texture_mask = self._mask_names[next_idx]
            except ValueError:
                self.active_texture_mask = self._mask_names[0]

        mask_name = self.active_texture_mask or "None"
        self.ui.add_gesture_message(
            f"MOUTH OPEN → Texture: {mask_name.upper()}",
            (0, 255, 255)
        )

    def _on_brow_raise(self, event: GestureEvent) -> None:
        """Handle brow raise gesture."""
        if "glasses" in self.active_filters:
            self.active_filters["glasses"] = not self.active_filters["glasses"]
            status = "ON" if self.active_filters["glasses"] else "OFF"
            self.ui.add_gesture_message(
                f"BROWS RAISED → Glasses: {status}",
                (0, 255, 0)
            )

    def _save_screenshot(self, frame: np.ndarray) -> None:
        """Save the current frame as a screenshot."""
        try:
            # Create directory if not exists
            save_dir = Path("captures")
            save_dir.mkdir(exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = save_dir / f"capture_{timestamp}.png"

            # Save image
            cv2.imwrite(str(filename), frame)
            
            # Show visual feedback
            self.ui.add_gesture_message(f"Saved: {filename.name}", (255, 255, 255))
            print(f"Screenshot saved to {filename}")
            
        except Exception as e:
            print(f"Error saving screenshot: {e}")
            self.ui.add_gesture_message("Error saving screenshot!", (0, 0, 255))

    def _apply_filter(self, frame: np.ndarray, face: FaceLandmarks,
                     flt: ARFilter) -> np.ndarray:
        """Apply a single AR filter to the frame."""
        if not flt.is_loaded():
            return frame

        get = lambda i: self.face_detector.get_landmark_point(face, i)
        rot_p1 = get(flt.config.rotation_landmarks[0])
        rot_p2 = get(flt.config.rotation_landmarks[1])
        if not (rot_p1 and rot_p2):
            return frame

        scale_p1 = get(flt.config.scale_landmarks[0])
        scale_p2 = get(flt.config.scale_landmarks[1])
        if not (scale_p1 and scale_p2):
            return frame

        angle = self.graphics.calculate_angle(rot_p1, rot_p2)
        width = int(
            self.graphics.calculate_distance(scale_p1, scale_p2) 
            * flt.config.scale_multiplier
        )
        if width <= 0:
            return frame

        resized = self.graphics.resize_image(flt.image, width)
        rotated = self.graphics.rotate_image(resized, angle)
        anchor = self.face_detector.get_landmarks_center(
            face, flt.config.anchor_landmarks
        )
        pos = (anchor[0] + flt.config.x_offset, anchor[1] + flt.config.y_offset)

        return self.graphics.overlay_image(frame, rotated, pos)

    def _apply_texture_mask(
        self,
        frame: np.ndarray,
        face: FaceLandmarks,
        mask: TextureMask
    ) -> np.ndarray:
        """Apply a texture mask to the frame using face mesh warping."""
        if not mask.is_loaded():
            return frame

        face_landmarks_array = self.face_detector.get_landmarks_as_array(face)
        landmarks_3d_array = np.array(face.landmarks_3d, dtype=np.float32)

        if mask.config.debug_wireframe and self.active_texture_mask == "debug":
            return self._draw_wireframe(frame, face_landmarks_array)

        try:
            if mask.renderer.corrected_triangles is None:
                mask.renderer.calibrate_winding_order(landmarks_3d_array)

            frame = mask.renderer.render(
                frame,
                face_landmarks_array,
                landmarks_3d=landmarks_3d_array,
                opacity=mask.config.opacity
            )
        except Exception:
            pass

        return frame

    def _draw_wireframe(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray
    ) -> np.ndarray:
        """Draw wireframe overlay for debugging."""
        triangles = config.FACE_MESH_TRIANGLES

        for tri_indices in triangles:
            pt1 = tuple(landmarks[tri_indices[0]].astype(int))
            pt2 = tuple(landmarks[tri_indices[1]].astype(int))
            pt3 = tuple(landmarks[tri_indices[2]].astype(int))

            cv2.line(frame, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.line(frame, pt2, pt3, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.line(frame, pt3, pt1, (0, 255, 0), 1, cv2.LINE_AA)

        return frame

    def _handle_keyboard(self, key: int, frame: np.ndarray) -> bool:
        """Handle keyboard input."""
        if key == ord('q') or key == ord('Q'):
            return False

        # Screenshot
        if key == ord('s') or key == ord('S'):
            self._save_screenshot(frame)
            return True

        # 2D Filter controls
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

        # Texture Mask controls
        elif key in (ord('m'), ord('M')):
            self.active_texture_mask = (
                "masculine" if self.active_texture_mask != "masculine" else None
            )
        elif key in (ord('f'), ord('F')):
            self.active_texture_mask = (
                "feminine" if self.active_texture_mask != "feminine" else None
            )
        elif key in (ord('w'), ord('W')):
            self.active_texture_mask = (
                "debug" if self.active_texture_mask != "debug" else None
            )
        elif key in (ord('c'), ord('C')):
            self.active_texture_mask = (
                "custom" if self.active_texture_mask != "custom" else None
            )
        elif key in (ord('n'), ord('N')):
            self.active_texture_mask = None

        # Gesture control toggle
        elif key in (ord('g'), ord('G')):
            self.gesture_detector.set_enabled(not self.gesture_detector.enabled)
            status = "enabled" if self.gesture_detector.enabled else "disabled"
            print(f"Gesture detection {status}")

        return True

    def run(self) -> None:
        """Main application loop."""
        if not self.camera.open():
            print("Error: Could not open camera")
            return

        print(f"\nAR Face Filter Initialized")
        print(f"  ... (Initialization logs)")
        print(f"  [S] Save Screenshot")
        
        self.running = True

        try:
            while self.running:
                success, frame = self.camera.read()
                if not success:
                    break
                
                # Keep a clean copy for processing if needed, 
                # but usually we want to screenshot the AR result.
                display_frame = frame.copy()

                faces = self.face_detector.detect(frame) # Detect on raw frame
                
                for face in faces:
                    if face == faces[0]:
                        self.gesture_detector.update(face)

                    if (
                        self.active_texture_mask
                        and self.active_texture_mask in self.texture_masks
                    ):
                        mask = self.texture_masks[self.active_texture_mask]
                        display_frame = self._apply_texture_mask(display_frame, face, mask)

                    for name, flt in self.filters.items():
                        if self.active_filters.get(name, False):
                            display_frame = self._apply_filter(display_frame, face, flt)

                    if config.SHOW_LANDMARKS:
                        display_frame = self.ui.draw_landmarks(display_frame, face)

                # Draw UI
                self.ui.update_fps()
                display_frame = self.ui.draw_fps(display_frame)
                display_frame = self.ui.draw_instructions(
                    display_frame, 
                    self.active_filters, 
                    self.active_texture_mask,
                    self.gesture_detector.enabled
                )
                display_frame = self.ui.draw_gesture_status(display_frame, self.gesture_detector)
                display_frame = self.ui.draw_gesture_messages(display_frame)

                cv2.imshow(config.WINDOW_NAME, display_frame)
                
                # Pass display_frame to handle_keyboard for screenshot
                if (k := cv2.waitKey(1) & 0xFF) != 255:
                    if not self._handle_keyboard(k, display_frame):
                        break
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources."""
        self.running = False
        self.camera.release()
        self.face_detector.release()
        cv2.destroyAllWindows()