"""
IBVAP — Centroid-Based Multi-Object Tracker
Assigns temporary IDs (UNKNOWN-01, UNKNOWN-02, etc.) to detected entities.
Uses simple Euclidean distance matching across frames.

LIMITATION: This is centroid-based tracking, not re-ID.
Works well for entities entering/leaving zones.
Not robust to long-term occlusion or cross-camera tracking.
For production, upgrade to ByteTrack or DeepSort.
"""

import time
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class TrackedEntity:
    """State for a single tracked entity."""
    entity_id: str                          # e.g., "UNKNOWN-01"
    class_name: str                         # "person", "car", etc.
    centroid: Tuple[int, int]               # Current position
    bbox: Tuple[int, int, int, int]         # Current bounding box
    confidence: float                       # Detection confidence
    first_seen: float                       # Timestamp (time.time())
    last_seen: float                        # Timestamp
    external_track_id: Optional[int] = None # ID from ByteTrack/BoT-SORT, when enabled
    position_history: List[Tuple[int, int]] = field(default_factory=list)
    disappeared_frames: int = 0             # Frames since last seen
    zone_entry_time: Optional[float] = None # When entity entered a zone
    is_in_zone: bool = False
    behavior_tags: List[str] = field(default_factory=list)
    threat_score: int = 0
    threat_breakdown: Dict[str, int] = field(default_factory=dict)
    vehicle_plate: Optional[str] = None
    carried_objects: List[str] = field(default_factory=list)
    posture: Optional[str] = None           # "standing", "crouching", "climbing"
    skeleton: Optional[Dict[str, Tuple[int, int]]] = None  # Keypoints for drawing
    dominant_color: Optional[Tuple[int, int, int]] = None  # BGR
    estimated_build: Optional[str] = None   # "small", "medium", "large"

    @property
    def duration_in_zone(self) -> float:
        """Seconds the entity has been in the zone."""
        if self.zone_entry_time is None:
            return 0.0
        return time.time() - self.zone_entry_time

    @property
    def total_duration(self) -> float:
        """Seconds since first detection."""
        return self.last_seen - self.first_seen

    def update_position(self, centroid: Tuple[int, int], bbox: Tuple[int, int, int, int]):
        """Update entity position and record history."""
        self.centroid = centroid
        self.bbox = bbox
        self.last_seen = time.time()
        self.disappeared_frames = 0
        self.position_history.append(centroid)
        # Keep last 60 positions (about 2-4 seconds at 15-30 FPS)
        if len(self.position_history) > 60:
            self.position_history = self.position_history[-60:]


class CentroidTracker:
    """
    Simple centroid-based multi-object tracker.
    
    Algorithm:
    1. For each new frame, compute centroids of all detections
    2. Match to existing tracks by minimum Euclidean distance
    3. Unmatched detections become new tracks
    4. Unmatched tracks increment disappeared counter
    5. Tracks exceeding max_disappeared are removed
    """

    def __init__(self, max_disappeared: int = 30, max_distance: float = 80.0):
        """
        Args:
            max_disappeared: Frames before dropping a track
            max_distance: Max pixel distance for centroid matching
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_id = 1
        self.entities: OrderedDict[str, TrackedEntity] = OrderedDict()
        self._total_tracked = 0

    @property
    def total_tracked(self) -> int:
        """Total entities ever tracked (for stats)."""
        return self._total_tracked

    @property
    def active_count(self) -> int:
        """Number of currently active tracks."""
        return len(self.entities)

    def _generate_id(self) -> str:
        """Generate next entity ID (UNKNOWN-01, UNKNOWN-02, ...)."""
        entity_id = f"UNKNOWN-{self.next_id:02d}"
        self.next_id += 1
        return entity_id

    def update(self, detections: list) -> Dict[str, TrackedEntity]:
        """
        Update tracker with new detections.

        Args:
            detections: List of Detection objects from the detector

        Returns:
            Dictionary of entity_id → TrackedEntity for all active entities
        """
        now = time.time()

        # If no detections, increment disappeared for all
        if len(detections) == 0:
            for entity_id in list(self.entities.keys()):
                self.entities[entity_id].disappeared_frames += 1
                if self.entities[entity_id].disappeared_frames > self.max_disappeared:
                    del self.entities[entity_id]
            return self.entities

        # Extract centroids and metadata from new detections
        new_centroids = np.array([d.centroid for d in detections])

        # If no existing tracks, register all as new
        if len(self.entities) == 0:
            for detection in detections:
                self._register(detection, now)
            return self.entities

        # Match existing tracks to new detections
        existing_ids = list(self.entities.keys())
        existing_centroids = np.array([
            self.entities[eid].centroid for eid in existing_ids
        ])

        # Compute distance matrix
        dist_matrix = np.linalg.norm(
            existing_centroids[:, np.newaxis] - new_centroids[np.newaxis, :],
            axis=2,
        )

        # Hungarian-style greedy matching (simple version)
        matched_existing = set()
        matched_new = set()

        # Prefer the detector's persistent ID when available. This preserves
        # identities through crossings/occlusions better than centroid distance.
        existing_by_external_id = {
            entity.external_track_id: row
            for row, entity_id in enumerate(existing_ids)
            for entity in [self.entities[entity_id]]
            if entity.external_track_id is not None
        }
        for col, detection in enumerate(detections):
            external_id = getattr(detection, "track_id", None)
            row = existing_by_external_id.get(external_id)
            if row is None or row in matched_existing:
                continue
            entity = self.entities[existing_ids[row]]
            if entity.class_name != detection.class_name:
                continue
            entity.update_position(detection.centroid, detection.bbox)
            entity.confidence = detection.confidence
            matched_existing.add(row)
            matched_new.add(col)

        # Sort by distance and greedily match
        rows, cols = np.unravel_index(
            np.argsort(dist_matrix, axis=None), dist_matrix.shape
        )

        for row, col in zip(rows, cols):
            if row in matched_existing or col in matched_new:
                continue
            # Never let a track change semantic type because a nearby
            # detection belongs to another class.
            if self.entities[existing_ids[row]].class_name != detections[col].class_name:
                continue
            if dist_matrix[row, col] > self.max_distance:
                continue

            entity_id = existing_ids[row]
            detection = detections[col]

            # Update existing track
            self.entities[entity_id].update_position(
                detection.centroid, detection.bbox
            )
            self.entities[entity_id].confidence = detection.confidence
            self.entities[entity_id].external_track_id = getattr(detection, "track_id", None)

            matched_existing.add(row)
            matched_new.add(col)

        # Handle unmatched existing tracks (disappeared)
        for row in range(len(existing_ids)):
            if row not in matched_existing:
                entity_id = existing_ids[row]
                self.entities[entity_id].disappeared_frames += 1
                if self.entities[entity_id].disappeared_frames > self.max_disappeared:
                    del self.entities[entity_id]

        # Register detections that could not be matched to an existing track.
        # Otherwise new entities are silently dropped once tracking has begun.
        for col, detection in enumerate(detections):
            if col not in matched_new:
                self._register(detection, now)

        # Deduplicate active tracks for vehicles
        self._deduplicate_tracks()

        return self.entities

    def _deduplicate_tracks(self):
        """
        Merges or removes duplicate active vehicle tracks that are locked onto
        the same physical vehicle (e.g. centroids close or bboxes overlapping).
        """
        vehicle_classes = {"car", "motorcycle", "bus", "truck"}
        active_ids = list(self.entities.keys())
        to_delete = set()

        for i in range(len(active_ids)):
            idA = active_ids[i]
            if idA in to_delete or idA not in self.entities:
                continue
            eA = self.entities[idA]
            if eA.class_name not in vehicle_classes:
                continue

            for j in range(i + 1, len(active_ids)):
                idB = active_ids[j]
                if idB in to_delete or idB not in self.entities:
                    continue
                eB = self.entities[idB]
                if eB.class_name not in vehicle_classes:
                    continue

                # Check centroid distance
                dist = np.linalg.norm(np.array(eA.centroid) - np.array(eB.centroid))

                # Check bbox IoU
                boxA = eA.bbox
                boxB = eB.bbox
                xA = max(boxA[0], boxB[0])
                yA = max(boxA[1], boxB[1])
                xB = min(boxA[2], boxB[2])
                yB = min(boxA[3], boxB[3])
                inter_w = max(0, xB - xA)
                inter_h = max(0, yB - yA)
                inter_area = inter_w * inter_h

                areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
                min_area = min(areaA, areaB) + 1e-6
                overlap_ratio = inter_area / float(min_area)
                iou = inter_area / float(areaA + areaB - inter_area + 1e-6)

                # If centroids are very close (< 45 px) or boxes heavily overlap
                if dist < 45.0 or iou > 0.30 or overlap_ratio > 0.50:
                    primary, secondary = (eA, eB) if (eA.first_seen <= eB.first_seen) else (eB, eA)
                    sec_id = secondary.entity_id

                    # Transfer valuable attributes if primary lacks them
                    if not primary.vehicle_plate and secondary.vehicle_plate:
                        primary.vehicle_plate = secondary.vehicle_plate
                    primary.threat_score = max(primary.threat_score, secondary.threat_score)
                    if secondary.is_in_zone:
                        primary.is_in_zone = True

                    to_delete.add(sec_id)

        for eid in to_delete:
            if eid in self.entities:
                del self.entities[eid]


    def _register(self, detection, timestamp: float):
        """Register a new tracked entity."""
        entity_id = self._generate_id()
        entity = TrackedEntity(
            entity_id=entity_id,
            class_name=detection.class_name,
            centroid=detection.centroid,
            bbox=detection.bbox,
            confidence=detection.confidence,
            external_track_id=getattr(detection, "track_id", None),
            first_seen=timestamp,
            last_seen=timestamp,
            position_history=[detection.centroid],
        )
        self.entities[entity_id] = entity
        self._total_tracked += 1

    def get_persons(self) -> Dict[str, TrackedEntity]:
        """Get all tracked persons."""
        return {
            eid: e for eid, e in self.entities.items()
            if e.class_name == "person"
        }

    def get_vehicles(self) -> Dict[str, TrackedEntity]:
        """Get all tracked vehicles."""
        vehicle_classes = {"car", "motorcycle", "bus", "truck"}
        return {
            eid: e for eid, e in self.entities.items()
            if e.class_name in vehicle_classes
        }

    def get_baggage(self) -> Dict[str, TrackedEntity]:
        """Get all tracked baggage."""
        baggage_classes = {"backpack", "suitcase", "handbag"}
        return {
            eid: e for eid, e in self.entities.items()
            if e.class_name in baggage_classes
        }

    def get_entity(self, entity_id: str) -> Optional[TrackedEntity]:
        """Get a specific entity by ID."""
        return self.entities.get(entity_id)
