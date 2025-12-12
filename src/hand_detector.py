"""
Hand detection and gesture recognition using MediaPipe Hand Landmarker API.
Detects "OK" gesture (thumb + index finger forming circle) for photo capture.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HandLandmarks:
    """Container for hand landmarks data."""
    landmarks: List[Tuple[int, int]]  # 21 hand landmarks
    handedness: str  # "Left" or "Right"
    confidence: float


class HandGestureDetector:
    """
    Detects hand gestures using MediaPipe Hand Landmarker API.

    Supported gestures:
    - OK gesture (👌): Thumb + index finger forming circle
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize hand gesture detector.

        Args:
            model_path: Path to hand_landmarker.task model file
        """
        # Determine model path
        if model_path is None:
            model_path = str(Path(__file__).parent.parent / "models" / "hand_landmarker.task")

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Hand Landmarker model not found at {model_path}. "
                f"Download it from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )

        logger.info(f"Loading Hand Landmarker model from: {model_path}")

        # Initialize Hand Landmarker with VIDEO mode
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,  # Detect up to 2 hands
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.landmarker = vision.HandLandmarker.create_from_options(options)

        # Timestamp tracking
        self.start_time = time.time()
        self.frame_count = 0

        # Gesture detection state
        self.last_ok_gesture_time = 0
        self.ok_gesture_cooldown = 2.0  # seconds

        # Callback for OK gesture
        self.ok_gesture_callback: Optional[Callable[[], None]] = None

        logger.info("Hand Landmarker initialized (VIDEO mode)")

    def detect(self, frame: np.ndarray) -> List[HandLandmarks]:
        """
        Detect hands in frame.

        Args:
            frame: Input image frame (BGR format from OpenCV)

        Returns:
            List of HandLandmarks objects
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Generate timestamp
            self.frame_count += 1
            timestamp_ms = int((time.time() - self.start_time) * 1000)

            # Convert to MediaPipe Image
            rgb_frame_contiguous = np.ascontiguousarray(rgb_frame)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame_contiguous)

            # Detect hands
            results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            if not results.hand_landmarks:
                return []

            # Process hands
            hands = []
            height, width = frame.shape[:2]

            for i, hand_landmarks in enumerate(results.hand_landmarks):
                # Convert to pixel coordinates
                landmarks = [
                    (int(lm.x * width), int(lm.y * height))
                    for lm in hand_landmarks
                ]

                # Get handedness (left or right)
                handedness = results.handedness[i][0].category_name if i < len(results.handedness) else "Unknown"
                confidence = results.handedness[i][0].score if i < len(results.handedness) else 0.0

                hands.append(HandLandmarks(
                    landmarks=landmarks,
                    handedness=handedness,
                    confidence=confidence
                ))

            # Check for OK gesture
            self._check_ok_gesture(hands)

            return hands

        except Exception as e:
            logger.error(f"Error during hand detection: {e}")
            return []

    def _check_ok_gesture(self, hands: List[HandLandmarks]) -> None:
        """
        Check if any hand is making OK gesture (👌).

        OK gesture: Thumb tip touches index finger tip, other fingers extended.
        """
        current_time = time.time()

        # Cooldown check
        if current_time - self.last_ok_gesture_time < self.ok_gesture_cooldown:
            return

        for hand in hands:
            if self._is_ok_gesture(hand.landmarks):
                logger.info(f"OK gesture detected! ({hand.handedness} hand)")
                self.last_ok_gesture_time = current_time

                # Trigger callback
                if self.ok_gesture_callback:
                    self.ok_gesture_callback()

                break

    def _is_ok_gesture(self, landmarks: List[Tuple[int, int]]) -> bool:
        """
        Detect OK gesture (thumb + index forming circle).

        Hand landmarks indices:
        0: Wrist
        4: Thumb tip
        8: Index finger tip
        12: Middle finger tip
        16: Ring finger tip
        20: Pinky tip
        """
        if len(landmarks) < 21:
            return False

        # Get key points
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        wrist = landmarks[0]
        index_mcp = landmarks[5]  # Index metacarpophalangeal joint

        # 1. Check if thumb and index tips are close (forming circle)
        thumb_index_dist = math.hypot(
            thumb_tip[0] - index_tip[0],
            thumb_tip[1] - index_tip[1]
        )

        # Reference distance: wrist to index MCP
        reference_dist = math.hypot(
            wrist[0] - index_mcp[0],
            wrist[1] - index_mcp[1]
        )

        # Thumb and index should be close relative to hand size
        if reference_dist > 0:
            circle_ratio = thumb_index_dist / reference_dist
            if circle_ratio > 0.3:  # Too far apart
                return False
        else:
            return False

        # 2. Check if other fingers are extended (away from wrist)
        # Middle, ring, pinky should be further from wrist than their respective MCPs
        middle_mcp = landmarks[9]
        ring_mcp = landmarks[13]
        pinky_mcp = landmarks[17]

        def distance_from_wrist(point):
            return math.hypot(point[0] - wrist[0], point[1] - wrist[1])

        middle_extended = distance_from_wrist(middle_tip) > distance_from_wrist(middle_mcp) * 1.2
        ring_extended = distance_from_wrist(ring_tip) > distance_from_wrist(ring_mcp) * 1.2
        pinky_extended = distance_from_wrist(pinky_tip) > distance_from_wrist(pinky_mcp) * 1.2

        # At least 2 out of 3 fingers should be extended
        extended_count = sum([middle_extended, ring_extended, pinky_extended])

        return extended_count >= 2

    def register_ok_gesture_callback(self, callback: Callable[[], None]) -> None:
        """Register callback for OK gesture detection."""
        self.ok_gesture_callback = callback

    def draw_hands(self, frame: np.ndarray, hands: List[HandLandmarks]) -> np.ndarray:
        """
        Draw hand landmarks on frame for visualization.

        Args:
            frame: Frame to draw on
            hands: List of detected hands

        Returns:
            Frame with hand landmarks drawn
        """
        output = frame.copy()

        for hand in hands:
            # Draw landmarks
            for landmark in hand.landmarks:
                cv2.circle(output, landmark, 5, (0, 255, 0), -1)

            # Draw connections (simplified)
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                (5, 9), (9, 13), (13, 17)  # Palm
            ]

            for start_idx, end_idx in connections:
                if start_idx < len(hand.landmarks) and end_idx < len(hand.landmarks):
                    start = hand.landmarks[start_idx]
                    end = hand.landmarks[end_idx]
                    cv2.line(output, start, end, (255, 0, 0), 2)

            # Draw handedness label
            if hand.landmarks:
                wrist = hand.landmarks[0]
                cv2.putText(
                    output,
                    f"{hand.handedness} ({hand.confidence:.2f})",
                    (wrist[0], wrist[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2
                )

        return output

    def release(self) -> None:
        """Release resources."""
        try:
            if hasattr(self, 'landmarker') and self.landmarker:
                self.landmarker.close()
                logger.info("Hand Landmarker resources released")
        except Exception as e:
            logger.warning(f"Error releasing Hand Landmarker: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
