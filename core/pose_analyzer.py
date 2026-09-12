"""
IBVAP — Posture & Behaviour Analysis (Skeletal Detection)
Uses MediaPipe Pose for lightweight skeleton detection.

Detects:
- Crouching: Low hip position relative to shoulders, knee angle < 110°
- Climbing: Wrists above shoulders + upward body motion
- Erratic movement: Handled in behavior.py (centroid-based)

LIMITATION: Accuracy depends on camera angle, lighting, and occlusion.
MediaPipe works best on frontal/side views. Low-confidence detections are flagged.
"""

import cv2
import math
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class PoseResult:
    """Result of pose analysis for a single person."""
    posture: str = "standing"       # "standing", "crouching", "climbing", "unknown"
    confidence: float = 0.0         # Pose estimation confidence
    knee_angle: float = 180.0       # Average knee angle (degrees)
    is_arms_raised: bool = False    # Wrists above shoulders
    landmarks: Optional[Dict] = None  # Raw landmark data
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class PoseAnalyzer:
    """
    MediaPipe Pose-based posture classification.
    Runs on cropped person bounding boxes for performance.
    """

    def __init__(self, confidence_threshold: float = 0.6):
        """
        Args:
            confidence_threshold: Only flag postures above this confidence
        """
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # Fastest model
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.confidence_threshold = confidence_threshold

        # MediaPipe landmark indices
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.LEFT_KNEE = 25
        self.RIGHT_KNEE = 26
        self.LEFT_ANKLE = 27
        self.RIGHT_ANKLE = 28
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        self.LEFT_ELBOW = 13
        self.RIGHT_ELBOW = 14

    def analyze(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> PoseResult:
        """
        Analyze posture of a person within the given bounding box.

        Args:
            frame: Full BGR frame
            bbox: (x1, y1, x2, y2) person bounding box

        Returns:
            PoseResult with posture classification
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Clamp bbox to frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 - x1 < 30 or y2 - y1 < 50:
            return PoseResult(posture="unknown", confidence=0.0)

        # Crop person region
        person_crop = frame[y1:y2, x1:x2]

        # Convert BGR → RGB for MediaPipe
        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

        # Run pose estimation
        results = self.pose.process(rgb_crop)

        if not results.pose_landmarks:
            return PoseResult(posture="unknown", confidence=0.0)

        landmarks = results.pose_landmarks.landmark

        # Calculate average visibility as confidence proxy
        avg_confidence = self._avg_visibility(landmarks)
        if avg_confidence < self.confidence_threshold:
            return PoseResult(posture="unknown", confidence=avg_confidence)

        # Extract key joint positions (normalized to crop)
        joints = self._extract_joints(landmarks, person_crop.shape)

        # Convert to absolute coordinates for drawing (ensure int cast for OpenCV)
        absolute_keypoints = {}
        for name, (jx, jy, vis) in joints.items():
            if vis > 0.5:  # Only store visible keypoints
                absolute_keypoints[name] = (int(jx + x1), int(jy + y1))

        # Classify posture
        posture, tags = self._classify_posture(joints)

        # Calculate knee angle
        knee_angle = self._calculate_knee_angle(joints)

        # Check arms raised
        is_arms_raised = self._check_arms_raised(joints)

        return PoseResult(
            posture=posture,
            confidence=avg_confidence,
            knee_angle=knee_angle,
            is_arms_raised=is_arms_raised,
            tags=tags,
            landmarks=absolute_keypoints
        )

    def _avg_visibility(self, landmarks) -> float:
        """Calculate average visibility of key landmarks."""
        key_indices = [
            self.LEFT_SHOULDER, self.RIGHT_SHOULDER,
            self.LEFT_HIP, self.RIGHT_HIP,
            self.LEFT_KNEE, self.RIGHT_KNEE,
        ]
        visibilities = [landmarks[i].visibility for i in key_indices]
        return sum(visibilities) / len(visibilities)

    def _extract_joints(self, landmarks, crop_shape: Tuple) -> Dict:
        """Extract key joint positions as pixel coordinates."""
        h, w = crop_shape[:2]
        joints = {}
        indices = {
            "left_shoulder": self.LEFT_SHOULDER,
            "right_shoulder": self.RIGHT_SHOULDER,
            "left_hip": self.LEFT_HIP,
            "right_hip": self.RIGHT_HIP,
            "left_knee": self.LEFT_KNEE,
            "right_knee": self.RIGHT_KNEE,
            "left_ankle": self.LEFT_ANKLE,
            "right_ankle": self.RIGHT_ANKLE,
            "left_wrist": self.LEFT_WRIST,
            "right_wrist": self.RIGHT_WRIST,
            "left_elbow": self.LEFT_ELBOW,
            "right_elbow": self.RIGHT_ELBOW,
        }
        for name, idx in indices.items():
            lm = landmarks[idx]
            joints[name] = (int(lm.x * w), int(lm.y * h), lm.visibility)
        return joints

    def _classify_posture(self, joints: Dict) -> Tuple[str, List[str]]:
        """
        Classify posture based on joint geometry.

        Returns:
            Tuple of (posture_name, list_of_behavior_tags)
        """
        tags = []

        # --- Crouching Detection ---
        # Hip is low relative to shoulder height range
        shoulder_y = (joints["left_shoulder"][1] + joints["right_shoulder"][1]) / 2
        hip_y = (joints["left_hip"][1] + joints["right_hip"][1]) / 2
        knee_y = (joints["left_knee"][1] + joints["right_knee"][1]) / 2
        shoulder_x = (joints["left_shoulder"][0] + joints["right_shoulder"][0]) / 2
        hip_x = (joints["left_hip"][0] + joints["right_hip"][0]) / 2

        # --- Fallen / Lying Down Detection ---
        # If the horizontal distance between hip and shoulder is greater than vertical
        vertical_dist = abs(hip_y - shoulder_y)
        horizontal_dist = abs(hip_x - shoulder_x)
        if horizontal_dist > vertical_dist * 1.5 and vertical_dist < 50:
            tags.append("fallen")
            return "fallen", tags

        # In image coordinates, Y increases downward
        shoulder_to_hip = hip_y - shoulder_y
        hip_to_knee = knee_y - hip_y

        # If hip is very close to knee level → crouching
        knee_angle = self._calculate_knee_angle(joints)
        if knee_angle < 110 and hip_to_knee < shoulder_to_hip * 0.6:
            tags.append("crouching")
            return "crouching", tags

        # --- Surrendering / Hands Up Detection ---
        # Wrists are significantly above shoulders
        left_raised = joints["left_wrist"][1] < (joints["left_shoulder"][1] - 30)
        right_raised = joints["right_wrist"][1] < (joints["right_shoulder"][1] - 30)
        if left_raised and right_raised:
            tags.append("surrendering")
            return "surrendering", tags

        # --- Climbing Detection ---
        # Wrists above shoulders + vertical body orientation
        is_arms_raised = self._check_arms_raised(joints)
        if is_arms_raised:
            # Check if body is in a climbing posture (arms high, one leg raised)
            ankle_y_avg = (joints["left_ankle"][1] + joints["right_ankle"][1]) / 2
            ankle_diff = abs(joints["left_ankle"][1] - joints["right_ankle"][1])

            # One foot significantly higher than the other → climbing
            body_height = ankle_y_avg - shoulder_y
            if ankle_diff > body_height * 0.3 and body_height > 0:
                tags.append("climbing")
                return "climbing", tags

            # Even without asymmetric feet, arms raised is noteworthy
            tags.append("arms_raised")

        return "standing", tags

    def _calculate_knee_angle(self, joints: Dict) -> float:
        """
        Calculate average knee angle (hip-knee-ankle).
        Straight leg ≈ 180°, deep squat ≈ 60-90°.
        """
        angles = []
        for side in ["left", "right"]:
            hip = joints[f"{side}_hip"][:2]
            knee = joints[f"{side}_knee"][:2]
            ankle = joints[f"{side}_ankle"][:2]
            angle = self._angle_between(hip, knee, ankle)
            angles.append(angle)
        return sum(angles) / len(angles)

    def _check_arms_raised(self, joints: Dict) -> bool:
        """Check if either wrist is above its corresponding shoulder."""
        left_raised = joints["left_wrist"][1] < joints["left_shoulder"][1]
        right_raised = joints["right_wrist"][1] < joints["right_shoulder"][1]
        return left_raised or right_raised

    @staticmethod
    def _angle_between(
        p1: Tuple[int, int],
        vertex: Tuple[int, int],
        p2: Tuple[int, int],
    ) -> float:
        """
        Calculate angle at vertex between p1-vertex-p2.

        Returns:
            Angle in degrees (0-180)
        """
        v1 = (p1[0] - vertex[0], p1[1] - vertex[1])
        v2 = (p2[0] - vertex[0], p2[1] - vertex[1])

        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        if mag1 == 0 or mag2 == 0:
            return 180.0

        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_angle))

    def draw_skeleton(self, frame: np.ndarray, keypoints: Dict[str, Tuple[int, int]]) -> np.ndarray:
        """Draws the skeletal posture onto the frame."""
        if not keypoints:
            return frame
            
        connections = [
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
            ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
            ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
            ("right_hip", "right_knee"), ("right_knee", "right_ankle")
        ]
        
        # Draw connections
        for p1_name, p2_name in connections:
            if p1_name in keypoints and p2_name in keypoints:
                p1 = keypoints[p1_name]
                p2 = keypoints[p2_name]
                cv2.line(frame, p1, p2, (0, 255, 255), 2)  # Yellow lines
                
        # Draw joints
        for name, pt in keypoints.items():
            cv2.circle(frame, pt, 4, (0, 0, 255), -1)  # Red joints
            
        return frame
