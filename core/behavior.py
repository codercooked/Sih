"""
IBVAP — Behavior Analysis Module
Handles loitering detection, group formation/crowd detection,
and approach speed/direction analysis.
"""

import time
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class BehaviorReport:
    """Analysis results for a single entity in a single frame."""
    entity_id: str
    is_loitering: bool = False
    loitering_duration: float = 0.0           # seconds
    loitering_level: str = "none"             # "none", "suspicious", "high_risk"
    speed: float = 0.0                        # pixels per frame
    speed_category: str = "stationary"        # "stationary", "walking", "running"
    is_moving_toward_zone: bool = False
    direction_angle: float = 0.0              # degrees
    tags: List[str] = field(default_factory=list)


@dataclass
class GroupReport:
    """Group/crowd analysis for a zone in a single frame."""
    zone_name: str
    person_count: int = 0
    is_group: bool = False                    # 3+ people
    is_crowd: bool = False                    # 5+ people
    group_level: str = "none"                 # "none", "group", "crowd"
    tags: List[str] = field(default_factory=list)


class BehaviorAnalyzer:
    """
    Analyzes entity behaviors: loitering, speed, direction, and group formation.
    """

    def __init__(
        self,
        loitering_suspicious_sec: float = 30.0,
        loitering_high_risk_sec: float = 90.0,
        group_threshold: int = 3,
        crowd_threshold: int = 5,
        speed_running_threshold: float = 15.0,
    ):
        """
        Args:
            loitering_suspicious_sec: Seconds in zone before "suspicious" tag
            loitering_high_risk_sec: Seconds in zone before "high_risk" tag
            group_threshold: People in zone for "group formation"
            crowd_threshold: People in zone for "crowd intrusion"
            speed_running_threshold: Pixel displacement per frame for "running"
        """
        self.loitering_suspicious_sec = loitering_suspicious_sec
        self.loitering_high_risk_sec = loitering_high_risk_sec
        self.group_threshold = group_threshold
        self.crowd_threshold = crowd_threshold
        self.speed_running_threshold = speed_running_threshold

    def analyze_entity(
        self,
        entity_id: str,
        position_history: List[Tuple[int, int]],
        is_in_zone: bool,
        zone_entry_time: Optional[float],
        zone_center: Optional[Tuple[int, int]] = None,
        fps: float = 30.0,
    ) -> BehaviorReport:
        """
        Analyze behavior for a single tracked entity.

        Args:
            entity_id: Entity identifier
            position_history: Recent centroid positions
            is_in_zone: Whether entity is currently in restricted zone
            zone_entry_time: When entity entered the zone (None if not in zone)
            zone_center: Center of the zone polygon (for direction analysis)

        Returns:
            BehaviorReport with loitering, speed, and direction info
        """
        report = BehaviorReport(entity_id=entity_id)

        # --- Loitering Detection ---
        if is_in_zone and zone_entry_time is not None:
            report.loitering_duration = time.time() - zone_entry_time
            if report.loitering_duration >= self.loitering_high_risk_sec:
                report.is_loitering = True
                report.loitering_level = "high_risk"
                report.tags.append("loitering_high_risk")
            elif report.loitering_duration >= self.loitering_suspicious_sec:
                report.is_loitering = True
                report.loitering_level = "suspicious"
                report.tags.append("loitering_suspicious")

        # --- Speed & Direction ---
        if len(position_history) >= 2:
            # Speed: displacement between last two positions
            p1 = position_history[-2]
            p2 = position_history[-1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            report.speed = math.sqrt(dx * dx + dy * dy)

            # Smooth speed over last 5 frames for stability
            if len(position_history) >= 5:
                speeds = []
                for i in range(-5, -1):
                    pa = position_history[i]
                    pb = position_history[i + 1]
                    d = math.sqrt((pb[0] - pa[0]) ** 2 + (pb[1] - pa[1]) ** 2)
                    speeds.append(d)
                # Multiply by FPS to get pixels per second instead of pixels per frame
                report.speed = np.mean(speeds) * fps

            # Categorize speed based on pixels per second
            if report.speed < 30.0:
                report.speed_category = "stationary"
            elif report.speed < self.speed_running_threshold:
                report.speed_category = "walking"
            else:
                report.speed_category = "running"
                report.tags.append("running")

            # Direction: angle of movement
            report.direction_angle = math.degrees(math.atan2(dy, dx))

            # Check if moving toward zone center
            if zone_center is not None and report.speed > 2.0:
                # Vector from entity to zone center
                to_zone_x = zone_center[0] - p2[0]
                to_zone_y = zone_center[1] - p2[1]

                # Dot product of movement vector and toward-zone vector
                dot = dx * to_zone_x + dy * to_zone_y
                report.is_moving_toward_zone = dot > 0

                if report.is_moving_toward_zone:
                    report.tags.append("approaching_zone")

        return report

    def analyze_group(
        self,
        zone_name: str,
        persons_in_zone: int,
    ) -> GroupReport:
        """
        Analyze group formation / crowd detection for a zone.

        Args:
            zone_name: Name of the restricted zone
            persons_in_zone: Number of detected persons currently in the zone

        Returns:
            GroupReport with group/crowd status
        """
        report = GroupReport(
            zone_name=zone_name,
            person_count=persons_in_zone,
        )

        if persons_in_zone >= self.crowd_threshold:
            report.is_group = True
            report.is_crowd = True
            report.group_level = "crowd"
            report.tags.append("crowd_intrusion")
        elif persons_in_zone >= self.group_threshold:
            report.is_group = True
            report.group_level = "group"
            report.tags.append("group_formation")

        return report

    def detect_erratic_movement(
        self,
        position_history: List[Tuple[int, int]],
        window: int = 15,
    ) -> bool:
        """
        Detect erratic/unstable movement by checking variance in position changes.

        High variance in direction and speed → erratic movement.

        Args:
            position_history: Recent centroid positions
            window: Number of frames to analyze

        Returns:
            True if erratic movement detected
        """
        if len(position_history) < window:
            return False

        recent = position_history[-window:]
        directions = []
        speeds = []

        for i in range(1, len(recent)):
            dx = recent[i][0] - recent[i - 1][0]
            dy = recent[i][1] - recent[i - 1][1]
            speed = math.sqrt(dx * dx + dy * dy)
            direction = math.atan2(dy, dx)
            directions.append(direction)
            speeds.append(speed)

        if len(directions) < 3:
            return False

        # High variance in direction + reasonable speed = erratic
        dir_variance = np.var(directions)
        mean_speed = np.mean(speeds)

        # Threshold: significant direction changes while moving
        return dir_variance > 1.0 and mean_speed > 3.0
