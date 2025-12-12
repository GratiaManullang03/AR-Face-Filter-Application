"""
Gesture Detection module for facial gesture recognition.
Analyzes facial landmarks to detect specific gestures like mouth open and brow raise.
Uses ratio-based calculations for distance-invariant detection.
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum, auto

from . import config
from .face_detector import FaceLandmarks


class GestureType(Enum):
    """Enumeration of supported gesture types."""
    MOUTH_OPEN = auto()
    BROW_RAISE = auto()
    # Future gestures can be added here
    # EYE_BLINK = auto()
    # HEAD_TILT = auto()


@dataclass
class GestureState:
    """Tracks the state of a single gesture for debouncing."""
    consecutive_frames: int = 0
    cooldown_remaining: int = 0
    is_active: bool = False
    last_triggered_time: float = 0.0
    current_ratio: float = 0.0


@dataclass
class GestureEvent:
    """Represents a detected gesture event."""
    gesture_type: GestureType
    confidence: float  # 0.0 to 1.0, how far above threshold
    timestamp: float


class GestureDetector:
    """
    Detects facial gestures by analyzing landmark positions.
    
    Uses ratio-based calculations for robustness against camera distance.
    Implements debouncing to prevent rapid/flickering triggers.
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize the gesture detector.
        
        Args:
            enabled: Whether gesture detection is active
        """
        self.enabled = enabled
        
        # State tracking for each gesture type
        self._states: Dict[GestureType, GestureState] = {
            GestureType.MOUTH_OPEN: GestureState(),
            GestureType.BROW_RAISE: GestureState(),
        }
        
        # Event callbacks
        self._callbacks: Dict[GestureType, List[Callable[[GestureEvent], None]]] = {
            gesture: [] for gesture in GestureType
        }
        
        # Recent events for UI display
        self._recent_events: List[GestureEvent] = []
        self._event_display_duration = 1.0  # seconds

    def register_callback(
        self,
        gesture_type: GestureType,
        callback: Callable[[GestureEvent], None]
    ) -> None:
        """
        Register a callback function for a specific gesture.
        
        Args:
            gesture_type: The gesture to listen for
            callback: Function to call when gesture is triggered
        """
        self._callbacks[gesture_type].append(callback)

    def unregister_callback(
        self,
        gesture_type: GestureType,
        callback: Callable[[GestureEvent], None]
    ) -> None:
        """Remove a previously registered callback."""
        if callback in self._callbacks[gesture_type]:
            self._callbacks[gesture_type].remove(callback)

    def update(self, face_landmarks: FaceLandmarks) -> List[GestureEvent]:
        """
        Process a frame and detect gestures.
        
        Args:
            face_landmarks: Detected facial landmarks
            
        Returns:
            List of triggered gesture events
        """
        if not self.enabled:
            return []

        triggered_events: List[GestureEvent] = []
        landmarks = face_landmarks.landmarks
        
        # Update cooldowns
        self._update_cooldowns()
        
        # Check each gesture
        mouth_event = self._check_mouth_open(landmarks)
        if mouth_event:
            triggered_events.append(mouth_event)
            
        brow_event = self._check_brow_raise(landmarks)
        if brow_event:
            triggered_events.append(brow_event)
        
        # Fire callbacks and store recent events
        for event in triggered_events:
            self._recent_events.append(event)
            for callback in self._callbacks[event.gesture_type]:
                callback(event)
        
        # Clean old events
        self._clean_old_events()
        
        return triggered_events

    def _update_cooldowns(self) -> None:
        """Decrement cooldown counters for all gestures."""
        for state in self._states.values():
            if state.cooldown_remaining > 0:
                state.cooldown_remaining -= 1

    def _clean_old_events(self) -> None:
        """Remove events older than display duration."""
        current_time = time.time()
        self._recent_events = [
            e for e in self._recent_events
            if current_time - e.timestamp < self._event_display_duration
        ]

    def _calculate_distance(
        self,
        p1: Tuple[int, int],
        p2: Tuple[int, int]
    ) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            p1: First point (x, y)
            p2: Second point (x, y)
            
        Returns:
            Distance between points
        """
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def _get_landmark(
        self,
        landmarks: List[Tuple[int, int]],
        index: int
    ) -> Optional[Tuple[int, int]]:
        """Safely get a landmark by index."""
        if 0 <= index < len(landmarks):
            return landmarks[index]
        return None

    def _calculate_mouth_aspect_ratio(
        self,
        landmarks: List[Tuple[int, int]]
    ) -> float:
        """
        Calculate Mouth Aspect Ratio (MAR).
        
        MAR = vertical_distance / horizontal_distance
        Higher MAR indicates more open mouth.
        
        Args:
            landmarks: List of facial landmarks
            
        Returns:
            Mouth aspect ratio (0.0 if calculation fails)
        """
        # Get mouth corner landmarks for horizontal distance
        left_corner = self._get_landmark(
            landmarks, 
            config.GestureLandmarks.MOUTH_HORIZONTAL["left"]
        )
        right_corner = self._get_landmark(
            landmarks,
            config.GestureLandmarks.MOUTH_HORIZONTAL["right"]
        )
        
        if not left_corner or not right_corner:
            return 0.0
        
        # Calculate horizontal distance
        horizontal_dist = self._calculate_distance(left_corner, right_corner)
        
        if horizontal_dist < 1.0:  # Avoid division by zero
            return 0.0
        
        # Get vertical landmarks - upper lip top (13) and lower lip bottom (17)
        upper_lip = self._get_landmark(landmarks, 13)
        lower_lip = self._get_landmark(landmarks, 17)
        
        if not upper_lip or not lower_lip:
            return 0.0
        
        # Calculate vertical distance
        vertical_dist = self._calculate_distance(upper_lip, lower_lip)
        
        # Calculate MAR
        mar = vertical_dist / horizontal_dist
        
        return mar

    def _calculate_brow_raise_ratio(
        self,
        landmarks: List[Tuple[int, int]]
    ) -> float:
        """
        Calculate Brow Raise Ratio.
        
        Ratio = avg(brow_to_eye_distance) / face_height
        Higher ratio indicates raised eyebrows.
        
        Args:
            landmarks: List of facial landmarks
            
        Returns:
            Brow raise ratio (0.0 if calculation fails)
        """
        # Get face height reference
        face_top = self._get_landmark(
            landmarks,
            config.GestureLandmarks.FACE_TOP
        )
        face_bottom = self._get_landmark(
            landmarks,
            config.GestureLandmarks.FACE_BOTTOM
        )
        
        if not face_top or not face_bottom:
            return 0.0
        
        face_height = self._calculate_distance(face_top, face_bottom)
        
        if face_height < 1.0:  # Avoid division by zero
            return 0.0
        
        # Calculate left brow to eye distance
        left_brow_points = [
            self._get_landmark(landmarks, idx)
            for idx in config.GestureLandmarks.LEFT_EYEBROW
        ]
        left_eye_top = self._get_landmark(
            landmarks,
            config.GestureLandmarks.LEFT_EYE_TOP
        )
        
        # Calculate right brow to eye distance
        right_brow_points = [
            self._get_landmark(landmarks, idx)
            for idx in config.GestureLandmarks.RIGHT_EYEBROW
        ]
        right_eye_top = self._get_landmark(
            landmarks,
            config.GestureLandmarks.RIGHT_EYE_TOP
        )
        
        # Filter out None values
        left_brow_valid = [p for p in left_brow_points if p is not None]
        right_brow_valid = [p for p in right_brow_points if p is not None]
        
        if not left_brow_valid or not right_brow_valid:
            return 0.0
        if not left_eye_top or not right_eye_top:
            return 0.0
        
        # Average brow positions
        left_brow_avg_y = sum(p[1] for p in left_brow_valid) / len(left_brow_valid)
        right_brow_avg_y = sum(p[1] for p in right_brow_valid) / len(right_brow_valid)
        
        # Calculate distances (in Y direction, since brow raise is vertical)
        left_dist = abs(left_eye_top[1] - left_brow_avg_y)
        right_dist = abs(right_eye_top[1] - right_brow_avg_y)
        
        # Average distance
        avg_dist = (left_dist + right_dist) / 2.0
        
        # Calculate ratio
        ratio = avg_dist / face_height
        
        return ratio

    def _check_gesture(
        self,
        gesture_type: GestureType,
        current_ratio: float,
        threshold: float,
        required_frames: int,
        cooldown_frames: int
    ) -> Optional[GestureEvent]:
        """
        Generic gesture checking with debounce logic.
        
        Args:
            gesture_type: Type of gesture being checked
            current_ratio: Current calculated ratio
            threshold: Threshold for activation
            required_frames: Frames needed to trigger
            cooldown_frames: Frames to wait after trigger
            
        Returns:
            GestureEvent if triggered, None otherwise
        """
        state = self._states[gesture_type]
        state.current_ratio = current_ratio
        
        # Check if on cooldown
        if state.cooldown_remaining > 0:
            state.consecutive_frames = 0
            state.is_active = False
            return None
        
        # Check if gesture is detected
        is_detected = current_ratio > threshold
        
        if is_detected:
            state.consecutive_frames += 1
            state.is_active = True
            
            # Check if held long enough to trigger
            if state.consecutive_frames >= required_frames:
                # Calculate confidence (how far above threshold)
                confidence = min(1.0, (current_ratio - threshold) / threshold)
                
                # Reset and start cooldown
                state.consecutive_frames = 0
                state.cooldown_remaining = cooldown_frames
                state.last_triggered_time = time.time()
                
                return GestureEvent(
                    gesture_type=gesture_type,
                    confidence=confidence,
                    timestamp=time.time()
                )
        else:
            state.consecutive_frames = 0
            state.is_active = False
        
        return None

    def _check_mouth_open(
        self,
        landmarks: List[Tuple[int, int]]
    ) -> Optional[GestureEvent]:
        """Check for mouth open gesture."""
        mar = self._calculate_mouth_aspect_ratio(landmarks)
        
        return self._check_gesture(
            gesture_type=GestureType.MOUTH_OPEN,
            current_ratio=mar,
            threshold=config.MOUTH_OPEN_THRESHOLD,
            required_frames=config.MOUTH_OPEN_REQUIRED_FRAMES,
            cooldown_frames=config.MOUTH_OPEN_COOLDOWN_FRAMES
        )

    def _check_brow_raise(
        self,
        landmarks: List[Tuple[int, int]]
    ) -> Optional[GestureEvent]:
        """Check for brow raise gesture."""
        ratio = self._calculate_brow_raise_ratio(landmarks)
        
        return self._check_gesture(
            gesture_type=GestureType.BROW_RAISE,
            current_ratio=ratio,
            threshold=config.BROW_RAISE_THRESHOLD,
            required_frames=config.BROW_RAISE_REQUIRED_FRAMES,
            cooldown_frames=config.BROW_RAISE_COOLDOWN_FRAMES
        )

    def get_state(self, gesture_type: GestureType) -> GestureState:
        """Get the current state of a gesture."""
        return self._states[gesture_type]

    def get_recent_events(self) -> List[GestureEvent]:
        """Get list of recently triggered events for display."""
        return self._recent_events.copy()

    def get_debug_info(self) -> Dict[str, any]:
        """
        Get debug information about current gesture states.
        
        Returns:
            Dictionary with debug data for each gesture
        """
        return {
            "mouth_open": {
                "ratio": self._states[GestureType.MOUTH_OPEN].current_ratio,
                "threshold": config.MOUTH_OPEN_THRESHOLD,
                "is_active": self._states[GestureType.MOUTH_OPEN].is_active,
                "frames": self._states[GestureType.MOUTH_OPEN].consecutive_frames,
                "cooldown": self._states[GestureType.MOUTH_OPEN].cooldown_remaining,
            },
            "brow_raise": {
                "ratio": self._states[GestureType.BROW_RAISE].current_ratio,
                "threshold": config.BROW_RAISE_THRESHOLD,
                "is_active": self._states[GestureType.BROW_RAISE].is_active,
                "frames": self._states[GestureType.BROW_RAISE].consecutive_frames,
                "cooldown": self._states[GestureType.BROW_RAISE].cooldown_remaining,
            },
        }

    def reset(self) -> None:
        """Reset all gesture states."""
        for state in self._states.values():
            state.consecutive_frames = 0
            state.cooldown_remaining = 0
            state.is_active = False
            state.current_ratio = 0.0

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable gesture detection."""
        self.enabled = enabled
        if not enabled:
            self.reset()