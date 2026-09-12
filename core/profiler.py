"""
IBVAP — Entity Profiler Module
Builds profile cards for tracked entities with physical descriptors,
behavior tags, and threat score breakdown.

No identity storage — entity IDs are temporary (UNKNOWN-XX).
No face recognition or watchlist matching.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field


@dataclass
class EntityProfile:
    """Complete profile card for a tracked entity."""
    entity_id: str
    class_name: str                            # "person", "car", etc.
    threat_score: int = 0
    threat_level: str = "low"
    threat_color: str = "#4CAF50"
    threat_breakdown: List[tuple] = field(default_factory=list)

    # Behavior tags
    behavior_tags: List[str] = field(default_factory=list)
    posture: str = "standing"
    loitering_duration: float = 0.0

    # Physical descriptors
    estimated_build: str = "unknown"           # "small", "medium", "large"
    estimated_height_relative: str = "unknown" # "short", "medium", "tall"
    dominant_color_bgr: Optional[Tuple[int, int, int]] = None
    dominant_color_name: str = "unknown"
    carried_objects: List[str] = field(default_factory=list)

    # Tracking info
    first_seen: str = ""                       # Formatted timestamp
    duration_in_zone: float = 0.0
    last_position: Optional[Tuple[int, int]] = None
    zone_name: str = ""

    # Vehicle info (if vehicle)
    vehicle_plate: str = ""


class EntityProfiler:
    """
    Builds entity profiles from tracked data and frame analysis.
    """

    # Build estimation from bounding box area (relative to frame)
    BUILD_THRESHOLDS = {
        "small": 0.02,    # < 2% of frame area
        "medium": 0.06,   # 2-6% of frame area
        "large": 1.0,     # > 6% of frame area
    }

    def build_profile(
        self,
        entity,  # TrackedEntity from tracker
        frame: np.ndarray,
        threat_assessment=None,  # ThreatAssessment
        group_report=None,  # GroupReport
    ) -> EntityProfile:
        """
        Build a complete profile card for an entity.

        Args:
            entity: TrackedEntity object
            frame: Current BGR frame
            threat_assessment: Threat scoring result
            group_report: Group analysis result

        Returns:
            EntityProfile with all descriptors
        """
        profile = EntityProfile(
            entity_id=entity.entity_id,
            class_name=entity.class_name,
        )

        # Threat info
        if threat_assessment:
            profile.threat_score = threat_assessment.score
            profile.threat_level = threat_assessment.level
            profile.threat_color = threat_assessment.color
            profile.threat_breakdown = threat_assessment.breakdown

        # Behavior tags
        profile.behavior_tags = list(entity.behavior_tags)
        profile.posture = entity.posture or "standing"
        profile.loitering_duration = entity.duration_in_zone

        # Physical descriptors (persons only)
        if entity.class_name == "person":
            profile.estimated_build = self._estimate_build(entity.bbox, frame.shape)
            profile.estimated_height_relative = self._estimate_height(entity.bbox, frame.shape)
            profile.dominant_color_bgr = self._get_dominant_color(frame, entity.bbox)
            profile.dominant_color_name = self._color_name(profile.dominant_color_bgr)
            profile.carried_objects = list(entity.carried_objects)

        # Tracking info
        from datetime import datetime
        profile.first_seen = datetime.fromtimestamp(entity.first_seen).strftime("%H:%M:%S")
        profile.duration_in_zone = entity.duration_in_zone
        profile.last_position = entity.centroid

        # Vehicle info
        if entity.vehicle_plate:
            profile.vehicle_plate = entity.vehicle_plate

        return profile

    def _estimate_build(
        self,
        bbox: Tuple[int, int, int, int],
        frame_shape: Tuple,
    ) -> str:
        """Estimate body build from bounding box area relative to frame."""
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        frame_area = frame_shape[0] * frame_shape[1]
        ratio = bbox_area / max(1, frame_area)

        if ratio < self.BUILD_THRESHOLDS["small"]:
            return "small"
        elif ratio < self.BUILD_THRESHOLDS["medium"]:
            return "medium"
        else:
            return "large"

    def _estimate_height(
        self,
        bbox: Tuple[int, int, int, int],
        frame_shape: Tuple,
    ) -> str:
        """Estimate relative height from bounding box height vs frame height."""
        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1
        frame_height = frame_shape[0]
        ratio = bbox_height / max(1, frame_height)

        if ratio < 0.25:
            return "short"
        elif ratio < 0.5:
            return "medium"
        else:
            return "tall"

    def _get_dominant_color(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[Tuple[int, int, int]]:
        """
        Get dominant clothing color from the middle section of the person bbox.
        Uses the torso region (middle 40-70% of height) to avoid head/feet.
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Clamp to frame
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        bh = y2 - y1
        if bh < 10:
            return None

        # Torso region: 30-70% of bbox height
        torso_y1 = y1 + int(bh * 0.3)
        torso_y2 = y1 + int(bh * 0.7)

        torso_crop = frame[torso_y1:torso_y2, x1:x2]
        if torso_crop.size == 0:
            return None

        # Mean color of torso region
        mean_color = cv2.mean(torso_crop)[:3]
        return (int(mean_color[0]), int(mean_color[1]), int(mean_color[2]))

    @staticmethod
    def _color_name(bgr: Optional[Tuple[int, int, int]]) -> str:
        """Convert BGR color to approximate color name."""
        if bgr is None:
            return "unknown"

        b, g, r = bgr

        # Convert to HSV for better color classification
        pixel = np.uint8([[[b, g, r]]])
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

        # Low saturation = grayscale
        if s < 40:
            if v < 60:
                return "black"
            elif v < 180:
                return "gray"
            else:
                return "white"

        # Map hue to color name
        if h < 10 or h > 170:
            return "red"
        elif h < 25:
            return "orange"
        elif h < 35:
            return "yellow"
        elif h < 85:
            return "green"
        elif h < 130:
            return "blue"
        elif h < 170:
            return "purple"

        return "unknown"
