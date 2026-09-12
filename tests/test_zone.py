"""
IBVAP — Zone Geometry Unit Tests
Tests point-in-polygon, tripwire direction, and distance-to-boundary.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.zone import RestrictedZone, ZoneManager


def test_point_inside_rectangle():
    """Points clearly inside a rectangular zone."""
    zone = RestrictedZone("Test", [[100, 100], [500, 100], [500, 400], [100, 400]])
    
    # Center
    assert zone.point_in_polygon((300, 250)) is True
    # Near top-left
    assert zone.point_in_polygon((150, 150)) is True
    # Near bottom-right
    assert zone.point_in_polygon((450, 350)) is True
    print("✅ test_point_inside_rectangle PASSED")


def test_point_outside_rectangle():
    """Points clearly outside a rectangular zone."""
    zone = RestrictedZone("Test", [[100, 100], [500, 100], [500, 400], [100, 400]])
    
    # Above zone
    assert zone.point_in_polygon((300, 50)) is False
    # Below zone
    assert zone.point_in_polygon((300, 450)) is False
    # Left of zone
    assert zone.point_in_polygon((50, 250)) is False
    # Right of zone
    assert zone.point_in_polygon((550, 250)) is False
    # Far away
    assert zone.point_in_polygon((0, 0)) is False
    assert zone.point_in_polygon((1000, 1000)) is False
    print("✅ test_point_outside_rectangle PASSED")


def test_point_on_boundary():
    """Points on or very near the boundary (edge cases)."""
    zone = RestrictedZone("Test", [[100, 100], [500, 100], [500, 400], [100, 400]])
    
    # Just inside boundary
    assert zone.point_in_polygon((101, 101)) is True
    assert zone.point_in_polygon((499, 399)) is True
    print("✅ test_point_on_boundary PASSED")


def test_tripwire_entering():
    """Entity moving from outside to inside = ENTERING."""
    zone = RestrictedZone("Test", [[100, 100], [500, 100], [500, 400], [100, 400]])
    
    # From outside to inside
    event = zone.check_entity(
        current_pos=(300, 250),   # Inside
        previous_pos=(50, 250),   # Outside
    )
    assert event.is_inside is True
    assert event.direction == "ENTERING"
    print("✅ test_tripwire_entering PASSED")


def test_tripwire_exiting():
    """Entity moving from inside to outside = EXITING."""
    zone = RestrictedZone("Test", [[100, 100], [500, 100], [500, 400], [100, 400]])
    
    event = zone.check_entity(
        current_pos=(50, 250),    # Outside
        previous_pos=(300, 250),  # Inside
    )
    assert event.is_inside is False
    assert event.direction == "EXITING"
    print("✅ test_tripwire_exiting PASSED")


def test_tripwire_staying_inside():
    """Entity staying inside = no direction change."""
    zone = RestrictedZone("Test", [[100, 100], [500, 100], [500, 400], [100, 400]])
    
    event = zone.check_entity(
        current_pos=(300, 250),
        previous_pos=(250, 200),
    )
    assert event.is_inside is True
    assert event.direction is None
    print("✅ test_tripwire_staying_inside PASSED")


def test_tripwire_staying_outside():
    """Entity staying outside = no direction change."""
    zone = RestrictedZone("Test", [[100, 100], [500, 100], [500, 400], [100, 400]])
    
    event = zone.check_entity(
        current_pos=(50, 50),
        previous_pos=(30, 30),
    )
    assert event.is_inside is False
    assert event.direction is None
    print("✅ test_tripwire_staying_outside PASSED")


def test_triangle_zone():
    """Test with a triangular zone."""
    zone = RestrictedZone("Triangle", [[200, 100], [400, 100], [300, 300]])
    
    # Center of triangle
    assert zone.point_in_polygon((300, 180)) is True
    # Outside triangle
    assert zone.point_in_polygon((100, 200)) is False
    print("✅ test_triangle_zone PASSED")


def test_distance_to_boundary():
    """Test distance calculation to nearest boundary."""
    zone = RestrictedZone("Test", [[0, 0], [100, 0], [100, 100], [0, 100]])
    
    # Center should be ~50 pixels from nearest edge
    dist = zone._distance_to_boundary((50, 50))
    assert 49.0 <= dist <= 51.0, f"Expected ~50, got {dist}"
    
    # Near edge should be small distance
    dist = zone._distance_to_boundary((5, 50))
    assert dist < 10, f"Expected <10, got {dist}"
    print("✅ test_distance_to_boundary PASSED")


def test_zone_manager():
    """Test ZoneManager with multiple zones."""
    manager = ZoneManager()
    manager.add_zone("Zone A", [[0, 0], [100, 0], [100, 100], [0, 100]])
    manager.add_zone("Zone B", [[200, 200], [300, 200], [300, 300], [200, 300]])
    
    assert manager.is_inside_any_zone((50, 50)) is True    # In Zone A
    assert manager.is_inside_any_zone((250, 250)) is True   # In Zone B
    assert manager.is_inside_any_zone((150, 150)) is False   # In neither
    print("✅ test_zone_manager PASSED")


if __name__ == "__main__":
    test_point_inside_rectangle()
    test_point_outside_rectangle()
    test_point_on_boundary()
    test_tripwire_entering()
    test_tripwire_exiting()
    test_tripwire_staying_inside()
    test_tripwire_staying_outside()
    test_triangle_zone()
    test_distance_to_boundary()
    test_zone_manager()
    print("\n🎉 All zone tests passed!")
