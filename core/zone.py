"""
IBVAP — Virtual Fence / Zone Intrusion Detection Module
Defines restricted zones as polygons and performs point-in-polygon checks.
Implements tripwire logic (entering vs. exiting detection).
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ZoneEvent:
    """Result of a zone check for an entity."""
    is_inside: bool
    zone_name: str
    direction: Optional[str] = None  # "ENTERING", "EXITING", or None
    distance_to_boundary: float = 0.0


class RestrictedZone:
    """
    A restricted zone defined by a polygon.
    Performs point-in-polygon checks and tripwire direction detection.
    """

    def __init__(self, name: str, polygon: List[List[int]]):
        """
        Args:
            name: Human-readable zone name
            polygon: List of [x, y] coordinate pairs defining the zone boundary
        """
        self.name = name
        self.polygon = np.array(polygon, dtype=np.int32)
        self._polygon_points = [(p[0], p[1]) for p in polygon]

    def point_in_polygon(self, point: Tuple[int, int]) -> bool:
        """
        Ray-casting algorithm for point-in-polygon test.
        No external dependencies (no Shapely needed).

        Args:
            point: (x, y) coordinate to test

        Returns:
            True if point is inside the polygon
        """
        x, y = point
        n = len(self._polygon_points)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = self._polygon_points[i]
            xj, yj = self._polygon_points[j]

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i

        return inside

    def contains_point(self, point: Tuple[int, int]) -> bool:
        """Alias for point_in_polygon."""
        return self.point_in_polygon(point)

    def check_entity(
        self,
        current_pos: Tuple[int, int],
        previous_pos: Optional[Tuple[int, int]] = None,
    ) -> ZoneEvent:
        """
        Check if an entity is inside the zone and detect crossing direction.

        Tripwire logic:
        - If was outside last frame and inside this frame → ENTERING
        - If was inside last frame and outside this frame → EXITING

        Args:
            current_pos: Current (x, y) centroid position
            previous_pos: Previous frame's (x, y) position (None if first detection)

        Returns:
            ZoneEvent with intrusion status and direction
        """
        is_inside = self.point_in_polygon(current_pos)
        direction = None

        if previous_pos is not None:
            was_inside = self.point_in_polygon(previous_pos)
            if not was_inside and is_inside:
                direction = "ENTERING"
            elif was_inside and not is_inside:
                direction = "EXITING"

        # Calculate approximate distance to nearest boundary edge
        distance = self._distance_to_boundary(current_pos)

        return ZoneEvent(
            is_inside=is_inside,
            zone_name=self.name,
            direction=direction,
            distance_to_boundary=distance,
        )

    def _distance_to_boundary(self, point: Tuple[int, int]) -> float:
        """
        Approximate minimum distance from point to polygon boundary.

        Args:
            point: (x, y) coordinate

        Returns:
            Minimum distance in pixels to any polygon edge
        """
        px, py = float(point[0]), float(point[1])
        min_dist = float("inf")
        n = len(self._polygon_points)

        for i in range(n):
            x1, y1 = float(self._polygon_points[i][0]), float(self._polygon_points[i][1])
            x2, y2 = float(self._polygon_points[(i + 1) % n][0]), float(self._polygon_points[(i + 1) % n][1])

            # Distance from point to line segment
            dx, dy = x2 - x1, y2 - y1
            length_sq = dx * dx + dy * dy

            if length_sq == 0:
                dist = np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
            else:
                t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy
                dist = np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

            min_dist = min(min_dist, dist)

        return min_dist

    def get_polygon_for_drawing(self) -> np.ndarray:
        """Return polygon points in format suitable for cv2.polylines / cv2.fillPoly."""
        return self.polygon.reshape((-1, 1, 2))

    def get_center(self) -> Tuple[int, int]:
        """Return the centroid of the zone polygon."""
        cx = int(np.mean(self.polygon[:, 0]))
        cy = int(np.mean(self.polygon[:, 1]))
        return (cx, cy)


class ZoneManager:
    """Manages multiple restricted zones."""

    def __init__(self):
        self.zones: List[RestrictedZone] = []

    def add_zone(self, name: str, polygon: List[List[int]]):
        """Add a restricted zone."""
        zone = RestrictedZone(name, polygon)
        self.zones.append(zone)

    def load_from_config(self, config: dict):
        """Load zone(s) from config dictionary."""
        # Single zone from config
        if "zone_polygon" in config and "zone_name" in config:
            self.add_zone(config["zone_name"], config["zone_polygon"])

        # Multiple zones (future)
        if "zones" in config:
            for zone_def in config["zones"]:
                self.add_zone(zone_def["name"], zone_def["polygon"])

    def check_all_zones(
        self,
        current_pos: Tuple[int, int],
        previous_pos: Optional[Tuple[int, int]] = None,
    ) -> List[ZoneEvent]:
        """Check entity against all zones."""
        events = []
        for zone in self.zones:
            event = zone.check_entity(current_pos, previous_pos)
            events.append(event)
        return events

    def is_inside_any_zone(self, point: Tuple[int, int]) -> bool:
        """Quick check if point is inside any restricted zone."""
        return any(zone.point_in_polygon(point) for zone in self.zones)
