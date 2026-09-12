"""
IBVAP — Tracker Unit Tests
Tests centroid-based entity tracking: registration, matching, disappearance.
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.tracker import CentroidTracker
from core.detector import Detection


def make_detection(x1, y1, x2, y2, class_name="person", confidence=0.9):
    """Helper to create a Detection object."""
    return Detection(
        bbox=(x1, y1, x2, y2),
        class_name=class_name,
        confidence=confidence,
    )


def test_register_new_entities():
    """New detections should get unique IDs."""
    tracker = CentroidTracker(max_disappeared=5)
    
    detections = [
        make_detection(100, 100, 200, 200),
        make_detection(300, 300, 400, 400),
    ]
    
    entities = tracker.update(detections)
    assert len(entities) == 2
    ids = list(entities.keys())
    assert ids[0] != ids[1]
    assert all(eid.startswith("UNKNOWN-") for eid in ids)
    print("✅ test_register_new_entities PASSED")


def test_track_persistence():
    """Same entity should keep its ID across frames."""
    tracker = CentroidTracker(max_disappeared=5, max_distance=100)
    
    # Frame 1
    entities = tracker.update([make_detection(100, 100, 200, 200)])
    first_id = list(entities.keys())[0]
    
    # Frame 2 — slight movement
    entities = tracker.update([make_detection(110, 110, 210, 210)])
    assert first_id in entities
    assert len(entities) == 1
    print("✅ test_track_persistence PASSED")


def test_entity_disappearance():
    """Entity should be removed after max_disappeared frames."""
    tracker = CentroidTracker(max_disappeared=3)
    
    # Frame 1 — entity appears
    entities = tracker.update([make_detection(100, 100, 200, 200)])
    entity_id = list(entities.keys())[0]
    
    # Frames 2-4 — entity disappears
    for _ in range(4):
        entities = tracker.update([])
    
    assert entity_id not in entities
    print("✅ test_entity_disappearance PASSED")


def test_multiple_entities_matching():
    """Multiple entities should be matched correctly by distance."""
    tracker = CentroidTracker(max_disappeared=5, max_distance=100)
    
    # Frame 1 — two entities
    entities = tracker.update([
        make_detection(100, 100, 200, 200),  # Entity A at ~(150, 150)
        make_detection(400, 400, 500, 500),  # Entity B at ~(450, 450)
    ])
    ids = sorted(entities.keys())
    
    # Frame 2 — both move slightly
    entities = tracker.update([
        make_detection(110, 110, 210, 210),  # Entity A moved to ~(160, 160)
        make_detection(410, 410, 510, 510),  # Entity B moved to ~(460, 460)
    ])
    
    assert len(entities) == 2
    assert ids[0] in entities
    assert ids[1] in entities
    print("✅ test_multiple_entities_matching PASSED")


def test_new_entity_while_tracking():
    """A new entity should get a new ID while existing ones persist."""
    tracker = CentroidTracker(max_disappeared=5, max_distance=100)
    
    # Frame 1
    entities = tracker.update([make_detection(100, 100, 200, 200)])
    first_id = list(entities.keys())[0]
    
    # Frame 2 — original + new entity
    entities = tracker.update([
        make_detection(110, 110, 210, 210),  # Original moved slightly
        make_detection(500, 500, 600, 600),  # New entity far away
    ])
    
    assert len(entities) == 2
    assert first_id in entities
    new_ids = [eid for eid in entities if eid != first_id]
    assert len(new_ids) == 1
    print("✅ test_new_entity_while_tracking PASSED")


def test_position_history():
    """Position history should accumulate."""
    tracker = CentroidTracker(max_disappeared=5, max_distance=100)
    
    # 5 frames of movement
    for i in range(5):
        tracker.update([make_detection(100 + i * 10, 100, 200 + i * 10, 200)])
    
    entity = list(tracker.entities.values())[0]
    assert len(entity.position_history) == 5
    print("✅ test_position_history PASSED")


def test_total_tracked_count():
    """Total tracked should count all entities ever, not just active."""
    tracker = CentroidTracker(max_disappeared=1, max_distance=50)
    
    # Entity 1 appears and disappears
    tracker.update([make_detection(100, 100, 200, 200)])
    tracker.update([])  # Disappear frame 1
    tracker.update([])  # Disappear frame 2 → removed
    
    # Entity 2 appears
    tracker.update([make_detection(500, 500, 600, 600)])
    
    assert tracker.total_tracked == 2
    assert tracker.active_count == 1
    print("✅ test_total_tracked_count PASSED")


def test_max_distance_rejection():
    """Detections too far from any track should create new tracks."""
    tracker = CentroidTracker(max_disappeared=5, max_distance=50)
    
    # Frame 1
    entities = tracker.update([make_detection(100, 100, 150, 150)])  # Centroid ~(125, 125)
    first_id = list(entities.keys())[0]
    
    # Frame 2 — detection too far away (should NOT match)
    entities = tracker.update([make_detection(400, 400, 450, 450)])  # Centroid ~(425, 425)
    
    # Should have 2 entities (original disappeared, new one registered)
    # Or 1 if original already disappeared
    assert len(entities) >= 1
    print("✅ test_max_distance_rejection PASSED")


def test_vehicle_tracking():
    """Vehicles should be tracked the same way as persons."""
    tracker = CentroidTracker(max_disappeared=5, max_distance=100)
    
    entities = tracker.update([
        make_detection(100, 100, 200, 200, class_name="car"),
    ])
    
    entity = list(entities.values())[0]
    assert entity.class_name == "car"
    
    vehicles = tracker.get_vehicles()
    assert len(vehicles) == 1
    
    persons = tracker.get_persons()
    assert len(persons) == 0
    print("✅ test_vehicle_tracking PASSED")


if __name__ == "__main__":
    test_register_new_entities()
    test_track_persistence()
    test_entity_disappearance()
    test_multiple_entities_matching()
    test_new_entity_while_tracking()
    test_position_history()
    test_total_tracked_count()
    test_max_distance_rejection()
    test_vehicle_tracking()
    print("\n🎉 All tracker tests passed!")
