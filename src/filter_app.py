"""
Main AR Filter application logic.
Orchestrates camera, face detection, gesture detection, and filter rendering.
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional

from .camera import Camera
from .face_detector import FaceDetector, FaceLandmarks
from .graphics import GraphicsEngine
from .mesh_renderer import MeshRenderer
from .gesture_detector import GestureDetector, GestureType, GestureEvent
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
        print(f"Detecting face in texture: {self.config.asset_path.name}...")
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
        
        self.filters: Dict[str, ARFilter] = {}
        self.texture_masks: Dict[str, TextureMask] = {}
        self.active_filters: Dict[str, bool] = {}
        self.active_texture_mask: Optional[str] = None
        self.running = False

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0

        # Gesture event display
        self._gesture_display_messages: List[Dict] = []
        self._message_duration = 1.5  # seconds

        # Filter/mask lists for cycling
        self._filter_names: List[str] = []
        self._mask_names: List[str] = []
        self._current_filter_index = -1  # -1 means no filter active

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
        """
        Handle mouth open gesture.
        Cycles through texture masks (3D face paint).
        """
        # Cycle through texture masks
        if not self._mask_names:
            return

        if self.active_texture_mask is None:
            # Activate first mask
            self.active_texture_mask = self._mask_names[0]
        else:
            # Find current index and cycle to next
            try:
                current_idx = self._mask_names.index(self.active_texture_mask)
                next_idx = (current_idx + 1) % (len(self._mask_names) + 1)
                
                if next_idx >= len(self._mask_names):
                    self.active_texture_mask = None  # Cycle back to none
                else:
                    self.active_texture_mask = self._mask_names[next_idx]
            except ValueError:
                self.active_texture_mask = self._mask_names[0]

        # Add display message
        mask_name = self.active_texture_mask or "None"
        self._add_gesture_message(
            f"MOUTH OPEN → Texture: {mask_name.upper()}",
            (0, 255, 255)  # Yellow
        )

    def _on_brow_raise(self, event: GestureEvent) -> None:
        """
        Handle brow raise gesture.
        Toggles the glasses filter on/off.
        """
        if "glasses" in self.active_filters:
            self.active_filters["glasses"] = not self.active_filters["glasses"]
            status = "ON" if self.active_filters["glasses"] else "OFF"
            self._add_gesture_message(
                f"BROWS RAISED → Glasses: {status}",
                (0, 255, 0)  # Green
            )

    def _add_gesture_message(self, message: str, color: tuple) -> None:
        """Add a message to display on screen."""
        self._gesture_display_messages.append({
            "text": message,
            "color": color,
            "timestamp": time.time()
        })

    def _clean_old_messages(self) -> None:
        """Remove expired gesture messages."""
        current_time = time.time()
        self._gesture_display_messages = [
            msg for msg in self._gesture_display_messages
            if current_time - msg["timestamp"] < self._message_duration
        ]

    def _apply_filter(self, frame: np.ndarray, face: FaceLandmarks,
                     flt: ARFilter) -> np.ndarray:
        """Apply a single AR filter to the frame."""
        if not flt.is_loaded():
            return frame

        # Get rotation and scale points
        get = lambda i: self.face_detector.get_landmark_point(face, i)
        rot_p1 = get(flt.config.rotation_landmarks[0])
        rot_p2 = get(flt.config.rotation_landmarks[1])
        if not (rot_p1 and rot_p2):
            return frame

        scale_p1 = get(flt.config.scale_landmarks[0])
        scale_p2 = get(flt.config.scale_landmarks[1])
        if not (scale_p1 and scale_p2):
            return frame

        # Calculate transformations
        angle = self.graphics.calculate_angle(rot_p1, rot_p2)
        width = int(
            self.graphics.calculate_distance(scale_p1, scale_p2) 
            * flt.config.scale_multiplier
        )
        if width <= 0:
            return frame

        # Apply transformations
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

        # Get landmarks as numpy array
        face_landmarks_array = self.face_detector.get_landmarks_as_array(face)

        # Special case: debug wireframe mode
        if mask.config.debug_wireframe and self.active_texture_mask == "debug":
            return self._draw_wireframe(frame, face_landmarks_array)

        # Render the texture mask
        try:
            frame = mask.renderer.render(
                frame,
                face_landmarks_array,
                opacity=mask.config.opacity
            )
        except Exception:
            pass  # Silently handle rendering errors

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

    def _draw_gesture_status(self, frame: np.ndarray) -> np.ndarray:
        """Draw gesture detection status and debug info."""
        if not config.SHOW_GESTURE_STATUS:
            return frame

        h, w = frame.shape[:2]
        debug_info = self.gesture_detector.get_debug_info()
        
        # Position on right side of screen
        x_pos = w - 280
        y_start = 60
        line_height = 22

        # Header
        cv2.putText(
            frame, "=== Gesture Detection ===",
            (x_pos, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )
        y = y_start + line_height

        # Mouth Open status
        mouth = debug_info["mouth_open"]
        mouth_color = (0, 255, 0) if mouth["is_active"] else (100, 100, 100)
        progress = min(1.0, mouth["frames"] / config.MOUTH_OPEN_REQUIRED_FRAMES)
        bar_width = int(50 * progress)
        
        cv2.putText(
            frame, f"Mouth: {mouth['ratio']:.3f}/{mouth['threshold']:.2f}",
            (x_pos, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, mouth_color, 1
        )
        # Progress bar
        cv2.rectangle(frame, (x_pos + 170, y - 10), (x_pos + 220, y), (50, 50, 50), -1)
        if bar_width > 0:
            cv2.rectangle(
                frame, (x_pos + 170, y - 10), (x_pos + 170 + bar_width, y),
                (0, 200, 200), -1
            )
        y += line_height

        # Brow Raise status
        brow = debug_info["brow_raise"]
        brow_color = (0, 255, 0) if brow["is_active"] else (100, 100, 100)
        progress = min(1.0, brow["frames"] / config.BROW_RAISE_REQUIRED_FRAMES)
        bar_width = int(50 * progress)
        
        cv2.putText(
            frame, f"Brows: {brow['ratio']:.3f}/{brow['threshold']:.3f}",
            (x_pos, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, brow_color, 1
        )
        # Progress bar
        cv2.rectangle(frame, (x_pos + 170, y - 10), (x_pos + 220, y), (50, 50, 50), -1)
        if bar_width > 0:
            cv2.rectangle(
                frame, (x_pos + 170, y - 10), (x_pos + 170 + bar_width, y),
                (0, 200, 200), -1
            )
        y += line_height

        # Cooldown indicators
        if mouth["cooldown"] > 0:
            cv2.putText(
                frame, f"Mouth cooldown: {mouth['cooldown']}",
                (x_pos, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1
            )
            y += line_height
        if brow["cooldown"] > 0:
            cv2.putText(
                frame, f"Brow cooldown: {brow['cooldown']}",
                (x_pos, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1
            )
            y += line_height

        return frame

    def _draw_gesture_messages(self, frame: np.ndarray) -> np.ndarray:
        """Draw gesture trigger notification messages."""
        self._clean_old_messages()
        
        if not self._gesture_display_messages:
            return frame

        h, w = frame.shape[:2]
        
        # Draw messages at top center
        for i, msg in enumerate(self._gesture_display_messages[-3:]):  # Show last 3
            text = msg["text"]
            color = msg["color"]
            
            # Calculate text size for centering
            (text_w, text_h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )
            x = (w - text_w) // 2
            y = 80 + i * 40
            
            # Draw background
            cv2.rectangle(
                frame, (x - 10, y - 25), (x + text_w + 10, y + 10),
                (0, 0, 0), -1
            )
            cv2.rectangle(
                frame, (x - 10, y - 25), (x + text_w + 10, y + 10),
                color, 2
            )
            
            # Draw text
            cv2.putText(
                frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )

        return frame

    def _draw_instructions(self, frame: np.ndarray) -> np.ndarray:
        """Draw keyboard instructions and filter status."""
        y, h = 60, 25

        # 2D Filters section
        cv2.putText(frame, "=== 2D Stickers ===",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += h

        keys = {"1": "glasses", "2": "mustache", "3": "beard", "4": "headband"}
        for key, name in keys.items():
            if name in self.active_filters:
                status = "ON" if self.active_filters[name] else "OFF"
                color = (0, 255, 0) if self.active_filters[name] else (0, 0, 255)
                cv2.putText(frame, f"[{key}] {name.capitalize()}: {status}",
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y += h

        # Texture Masks section
        y += 5
        cv2.putText(frame, "=== 3D Texture Masks ===",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += h

        mask_keys = {"M": "masculine", "F": "feminine", "W": "debug"}
        for key, name in mask_keys.items():
            status = "ACTIVE" if self.active_texture_mask == name else "off"
            color = (0, 255, 255) if self.active_texture_mask == name else (100, 100, 100)
            cv2.putText(frame, f"[{key}] {name.capitalize()}: {status}",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += h

        # Gesture controls section
        y += 5
        cv2.putText(frame, "=== Gesture Controls ===",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += h
        
        gesture_status = "ON" if self.gesture_detector.enabled else "OFF"
        gesture_color = (0, 255, 255) if self.gesture_detector.enabled else (100, 100, 100)
        cv2.putText(frame, f"[G] Gestures: {gesture_status}",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gesture_color, 1)
        y += h
        
        cv2.putText(frame, "Open Mouth: Cycle masks",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        y += h - 5
        cv2.putText(frame, "Raise Brows: Toggle glasses",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        y += h

        # General controls
        y += 5
        cv2.putText(frame, "[A] All 2D ON  [D] All OFF  [N] No Mask  [Q] Quit",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def _handle_keyboard(self, key: int) -> bool:
        """Handle keyboard input for filter and texture mask toggling."""
        if key == ord('q') or key == ord('Q'):
            return False

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
        print(f"  2D Filters: {len(self.filters)}")
        print(f"  3D Texture Masks: {len(self.texture_masks)}")
        print(f"  Gesture Detection: {'Enabled' if self.gesture_detector.enabled else 'Disabled'}")
        print(f"\nControls:")
        print(f"  1-4: Toggle 2D stickers")
        print(f"  M/F/W: Switch texture masks (Masculine/Feminine/Wireframe)")
        print(f"  N: Disable texture masks")
        print(f"  G: Toggle gesture detection")
        print(f"  A/D: All 2D filters on/off")
        print(f"  Q: Quit")
        print(f"\nGesture Controls:")
        print(f"  Open Mouth: Cycle through texture masks")
        print(f"  Raise Eyebrows: Toggle glasses filter\n")

        self.running = True

        try:
            while self.running:
                success, frame = self.camera.read()
                if not success:
                    break

                faces = self.face_detector.detect(frame)
                
                for face in faces:
                    # Process gestures for first face only
                    if face == faces[0]:
                        self.gesture_detector.update(face)

                    # Apply texture mask FIRST (base layer)
                    if (self.active_texture_mask and 
                        self.active_texture_mask in self.texture_masks):
                        mask = self.texture_masks[self.active_texture_mask]
                        frame = self._apply_texture_mask(frame, face, mask)

                    # Apply 2D filters on top of texture mask
                    for name, flt in self.filters.items():
                        if self.active_filters.get(name, False):
                            frame = self._apply_filter(frame, face, flt)

                    if config.SHOW_LANDMARKS:
                        frame = self._draw_landmarks(frame, face)

                # Draw UI elements
                self._update_fps()
                if config.FPS_DISPLAY:
                    frame = self._draw_fps(frame)
                if config.SHOW_INSTRUCTIONS:
                    frame = self._draw_instructions(frame)
                
                # Draw gesture status (right side)
                frame = self._draw_gesture_status(frame)
                
                # Draw gesture trigger messages (top center)
                frame = self._draw_gesture_messages(frame)

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