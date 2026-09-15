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

    def __init__(self, confidence_threshold: float = 0.4):
        """
        Args:
            confidence_threshold: Only flag postures above this confidence
        """
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        # Use static_image_mode=True so each person crop is processed independently
        # without tracking state corruption or cross-entity jitter/lag.
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,  # Maximum accuracy model for precise limb tracking
            enable_segmentation=False,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self.confidence_threshold = confidence_threshold

        # MediaPipe landmark indices
        self.NOSE = 0
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

    def analyze(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], prev_landmarks: Optional[Dict] = None) -> PoseResult:
        """
        Analyze posture of a person within the given bounding box using an anatomically
        padded crop to ensure limbs and head are not cut off.

        Args:
            frame: Full BGR frame
            bbox: (x1, y1, x2, y2) person bounding box
            prev_landmarks: Optional previous frame keypoints for EMA smoothing

        Returns:
            PoseResult with posture classification and smoothed keypoints
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        bw = max(10, x2 - x1)
        bh = max(20, y2 - y1)

        if bw < 25 or bh < 45:
            return PoseResult(posture="unknown", confidence=0.0)

        # Intelligent anatomical padding (avoids cutting off head/arms/legs)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        target_w = max(int(bw * 1.8), int(bh * 0.7))
        target_h = int(bh * 1.4)

        px1 = max(0, cx - target_w // 2)
        px2 = min(w, cx + target_w // 2)
        py1 = max(0, cy - target_h // 2)
        py2 = min(h, cy + target_h // 2)

        person_crop = frame[py1:py2, px1:px2]
        ch, cw = person_crop.shape[:2]
        if ch < 40 or cw < 25:
            return PoseResult(posture="unknown", confidence=0.0)

        # Convert BGR -> RGB
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

        # Extract key joint positions (normalized to crop -> mapped to full frame)
        joints = self._extract_joints(landmarks, person_crop.shape)

        # Convert to absolute coordinates for drawing
        absolute_keypoints = {}
        for name, (jx, jy, vis) in joints.items():
            if vis > 0.35:
                abs_x = int(jx + px1)
                abs_y = int(jy + py1)
                # Exponential Moving Average (EMA) smoothing if previous landmarks exist
                if prev_landmarks and name in prev_landmarks:
                    old_x, old_y = prev_landmarks[name]
                    abs_x = int(0.7 * abs_x + 0.3 * old_x)
                    abs_y = int(0.7 * abs_y + 0.3 * old_y)
                absolute_keypoints[name] = (abs_x, abs_y)

        # Compute mid-points for anatomical integrity
        if "left_shoulder" in absolute_keypoints and "right_shoulder" in absolute_keypoints:
            ls = absolute_keypoints["left_shoulder"]
            rs = absolute_keypoints["right_shoulder"]
            absolute_keypoints["neck"] = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)

        if "left_hip" in absolute_keypoints and "right_hip" in absolute_keypoints:
            lh = absolute_keypoints["left_hip"]
            rh = absolute_keypoints["right_hip"]
            absolute_keypoints["mid_hip"] = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)

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
            "nose": self.NOSE,
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

        shoulder_y = (joints["left_shoulder"][1] + joints["right_shoulder"][1]) / 2
        hip_y = (joints["left_hip"][1] + joints["right_hip"][1]) / 2
        knee_y = (joints["left_knee"][1] + joints["right_knee"][1]) / 2
        shoulder_x = (joints["left_shoulder"][0] + joints["right_shoulder"][0]) / 2
        hip_x = (joints["left_hip"][0] + joints["right_hip"][0]) / 2

        # --- Fallen / Lying Down Detection ---
        vertical_dist = abs(hip_y - shoulder_y)
        horizontal_dist = abs(hip_x - shoulder_x)
        if horizontal_dist > vertical_dist * 1.5 and vertical_dist < 50:
            tags.append("fallen")
            return "fallen", tags

        # In image coordinates, Y increases downward
        shoulder_to_hip = hip_y - shoulder_y
        hip_to_knee = knee_y - hip_y

        # If hip is very close to knee level -> crouching
        knee_angle = self._calculate_knee_angle(joints)
        if knee_angle < 110 and hip_to_knee < shoulder_to_hip * 0.6:
            tags.append("crouching")
            return "crouching", tags

        # --- Surrendering / Hands Up Detection ---
        left_raised = joints["left_wrist"][1] < (joints["left_shoulder"][1] - 30)
        right_raised = joints["right_wrist"][1] < (joints["right_shoulder"][1] - 30)
        if left_raised and right_raised:
            tags.append("surrendering")
            return "surrendering", tags

        # --- Climbing Detection ---
        is_arms_raised = self._check_arms_raised(joints)
        if is_arms_raised:
            ankle_y_avg = (joints["left_ankle"][1] + joints["right_ankle"][1]) / 2
            ankle_diff = abs(joints["left_ankle"][1] - joints["right_ankle"][1])

            body_height = ankle_y_avg - shoulder_y
            if ankle_diff > body_height * 0.3 and body_height > 0:
                tags.append("climbing")
                return "climbing", tags

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
        """Calculate angle at vertex between p1-vertex-p2."""
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
        """
        Draws an anti-aliased tactical skeletal posture with anatomical integrity:
        - Head & Spine in Neon Tactical Cyan
        - Arms in Electric Sky Blue
        - Legs in Tactical Emerald Green
        - Dual-ring crisp anti-aliased joint nodes
        """
        if not keypoints:
            return frame

        # Color palette (BGR)
        C_TORSO = (255, 230, 0)     # Neon Tactical Cyan
        C_ARMS  = (245, 180, 50)     # Electric Sky Blue
        C_LEGS  = (80, 245, 100)     # Tactical Emerald Green

        # Limb segments grouped by anatomical region
        segments = [
            # Head & Torso / Spine
            ("nose", "neck", C_TORSO, 2),
            ("neck", "mid_hip", C_TORSO, 2),
            ("left_shoulder", "neck", C_TORSO, 2),
            ("right_shoulder", "neck", C_TORSO, 2),
            ("left_hip", "mid_hip", C_TORSO, 2),
            ("right_hip", "mid_hip", C_TORSO, 2),
            # Direct shoulder-hip connections if neck/mid_hip not available
            ("left_shoulder", "right_shoulder", C_TORSO, 2),
            ("left_shoulder", "left_hip", C_TORSO, 1),
            ("right_shoulder", "right_hip", C_TORSO, 1),
            ("left_hip", "right_hip", C_TORSO, 2),

            # Left Arm
            ("left_shoulder", "left_elbow", C_ARMS, 2),
            ("left_elbow", "left_wrist", C_ARMS, 2),

            # Right Arm
            ("right_shoulder", "right_elbow", C_ARMS, 2),
            ("right_elbow", "right_wrist", C_ARMS, 2),

            # Left Leg
            ("left_hip", "left_knee", C_LEGS, 2),
            ("left_knee", "left_ankle", C_LEGS, 2),

            # Right Leg
            ("right_hip", "right_knee", C_LEGS, 2),
            ("right_knee", "right_ankle", C_LEGS, 2),
        ]

        drawn_connections = set()
        for p1_name, p2_name, color, thickness in segments:
            pair_key = tuple(sorted([p1_name, p2_name]))
            if pair_key in drawn_connections:
                continue
            if p1_name in keypoints and p2_name in keypoints:
                p1 = keypoints[p1_name]
                p2 = keypoints[p2_name]
                cv2.line(frame, p1, p2, color, thickness, lineType=cv2.LINE_AA)
                drawn_connections.add(pair_key)

        # Draw anti-aliased dual-ring joint nodes
        for name, pt in keypoints.items():
            # Outer neon dot
            cv2.circle(frame, pt, 3, (0, 240, 255), -1, lineType=cv2.LINE_AA)
            # Inner white center core
            cv2.circle(frame, pt, 1, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        return frame
