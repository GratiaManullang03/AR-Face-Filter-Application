"""
User Interface (UI) module for AR Face Filter.
Handles drawing of HUD, status messages, FPS, and instructions.
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Optional
from . import config
from .gesture_detector import GestureDetector

class UIManager:
    """Manages on-screen display elements."""

    def __init__(self):
        self.gesture_display_messages: List[Dict] = []
        self.message_duration = 1.5  # seconds
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0

    def update_fps(self) -> None:
        """Update FPS calculation."""
        self.frame_count += 1
        if (t := time.time() - self.start_time) > 1.0:
            self.fps = self.frame_count / t
            self.frame_count, self.start_time = 0, time.time()

    def add_gesture_message(self, message: str, color: tuple) -> None:
        """Add a message to display on screen."""
        self.gesture_display_messages.append({
            "text": message,
            "color": color,
            "timestamp": time.time()
        })

    def clean_old_messages(self) -> None:
        """Remove expired gesture messages."""
        current_time = time.time()
        self.gesture_display_messages = [
            msg for msg in self.gesture_display_messages
            if current_time - msg["timestamp"] < self.message_duration
        ]

    def draw_fps(self, frame: np.ndarray) -> np.ndarray:
        """Draw FPS counter."""
        if not config.FPS_DISPLAY:
            return frame
            
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame

    def draw_gesture_status(self, frame: np.ndarray, gesture_detector: GestureDetector) -> np.ndarray:
        """Draw gesture detection status and debug info."""
        if not config.SHOW_GESTURE_STATUS:
            return frame

        h, w = frame.shape[:2]
        debug_info = gesture_detector.get_debug_info()
        
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

    def draw_gesture_messages(self, frame: np.ndarray) -> np.ndarray:
        """Draw gesture trigger notification messages."""
        self.clean_old_messages()
        
        if not self.gesture_display_messages:
            return frame

        h, w = frame.shape[:2]
        
        # Draw messages at top center
        for i, msg in enumerate(self.gesture_display_messages[-3:]):  # Show last 3
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

    def draw_instructions(self, frame: np.ndarray, active_filters: Dict[str, bool], active_texture_mask: Optional[str], gesture_enabled: bool) -> np.ndarray:
        """Draw keyboard instructions and filter status."""
        if not config.SHOW_INSTRUCTIONS:
            return frame
            
        y, h = 60, 25

        # 2D Filters section
        cv2.putText(frame, "=== 2D Stickers ===",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += h

        keys = {"1": "glasses", "2": "mustache", "3": "beard", "4": "headband"}
        for key, name in keys.items():
            if name in active_filters:
                status = "ON" if active_filters[name] else "OFF"
                color = (0, 255, 0) if active_filters[name] else (0, 0, 255)
                cv2.putText(frame, f"[{key}] {name.capitalize()}: {status}",
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y += h

        # Texture Masks section
        y += 5
        cv2.putText(frame, "=== 3D Texture Masks ===",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += h

        mask_keys = {"M": "masculine", "F": "feminine", "W": "debug", "C": "custom"}
        for key, name in mask_keys.items():
            status = "ACTIVE" if active_texture_mask == name else "off"
            color = (0, 255, 255) if active_texture_mask == name else (100, 100, 100)
            cv2.putText(frame, f"[{key}] {name.capitalize()}: {status}",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += h

        # Gesture controls section
        y += 5
        cv2.putText(frame, "=== Gesture Controls ===",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += h
        
        gesture_status = "ON" if gesture_enabled else "OFF"
        gesture_color = (0, 255, 255) if gesture_enabled else (100, 100, 100)
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
        cv2.putText(frame, "[S] Screenshot  [Q] Quit",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += h
        cv2.putText(frame, "[A] All 2D ON  [D] All OFF  [N] No Mask",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def draw_landmarks(self, frame: np.ndarray, face) -> np.ndarray:
        """Draw facial landmarks (debug mode)."""
        for x, y in face.landmarks:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        return frame
