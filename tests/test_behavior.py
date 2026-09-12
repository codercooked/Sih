"""
IBVAP — Behavior Analysis Unit Tests
Tests loitering, group formation, speed categorization, and erratic movement.
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.behavior import BehaviorAnalyzer


def test_no_loitering():
    """Entity in zone for less than threshold = no loitering."""
    analyzer = BehaviorAnalyzer(loitering_suspicious_sec=30)
    
    result = analyzer.analyze_entity(
        entity_id="UNKNOWN-01",
        position_history=[(100, 100), (110, 110)],
        is_in_zone=True,
        zone_entry_time=time.time() - 10,  # 10 seconds ago
    )
    
    assert result.is_loitering is False
    assert result.loitering_level == "none"
    print("✅ test_no_loitering PASSED")


def test_suspicious_loitering():
    """Entity in zone for 30+ seconds = suspicious."""
    analyzer = BehaviorAnalyzer(loitering_suspicious_sec=30, loitering_high_risk_sec=90)
    
    result = analyzer.analyze_entity(
        entity_id="UNKNOWN-01",
        position_history=[(100, 100), (101, 101)],
        is_in_zone=True,
        zone_entry_time=time.time() - 45,  # 45 seconds ago
    )
    
    assert result.is_loitering is True
    assert result.loitering_level == "suspicious"
    assert "loitering_suspicious" in result.tags
    print("✅ test_suspicious_loitering PASSED")


def test_high_risk_loitering():
    """Entity in zone for 90+ seconds = high risk."""
    analyzer = BehaviorAnalyzer(loitering_suspicious_sec=30, loitering_high_risk_sec=90)
    
    result = analyzer.analyze_entity(
        entity_id="UNKNOWN-01",
        position_history=[(100, 100), (101, 101)],
        is_in_zone=True,
        zone_entry_time=time.time() - 120,  # 120 seconds ago
    )
    
    assert result.is_loitering is True
    assert result.loitering_level == "high_risk"
    assert "loitering_high_risk" in result.tags
    print("✅ test_high_risk_loitering PASSED")


def test_not_in_zone_no_loitering():
    """Entity NOT in zone should not trigger loitering."""
    analyzer = BehaviorAnalyzer()
    
    result = analyzer.analyze_entity(
        entity_id="UNKNOWN-01",
        position_history=[(100, 100), (110, 110)],
        is_in_zone=False,
        zone_entry_time=None,
    )
    
    assert result.is_loitering is False
    assert result.loitering_duration == 0.0
    print("✅ test_not_in_zone_no_loitering PASSED")


def test_speed_stationary():
    """Very little movement = stationary."""
    analyzer = BehaviorAnalyzer(speed_running_threshold=15.0)
    
    result = analyzer.analyze_entity(
        entity_id="UNKNOWN-01",
        position_history=[(100, 100), (101, 100)],  # 1 pixel
        is_in_zone=False,
        zone_entry_time=None,
    )
    
    assert result.speed_category == "stationary"
    print("✅ test_speed_stationary PASSED")


def test_speed_walking():
    """Moderate movement = walking."""
    analyzer = BehaviorAnalyzer(speed_running_threshold=15.0)
    
    result = analyzer.analyze_entity(
        entity_id="UNKNOWN-01",
        position_history=[(100, 100), (108, 100)],  # 8 pixels
        is_in_zone=False,
        zone_entry_time=None,
    )
    
    assert result.speed_category == "walking"
    print("✅ test_speed_walking PASSED")


def test_speed_running():
    """Fast movement = running."""
    analyzer = BehaviorAnalyzer(speed_running_threshold=15.0)
    
    result = analyzer.analyze_entity(
        entity_id="UNKNOWN-01",
        position_history=[(100, 100), (130, 100)],  # 30 pixels
        is_in_zone=False,
        zone_entry_time=None,
    )
    
    assert result.speed_category == "running"
    assert "running" in result.tags
    print("✅ test_speed_running PASSED")


def test_group_formation():
    """3+ persons in zone = group formation."""
    analyzer = BehaviorAnalyzer(group_threshold=3, crowd_threshold=5)
    
    result = analyzer.analyze_group(zone_name="Test Zone", persons_in_zone=3)
    
    assert result.is_group is True
    assert result.is_crowd is False
    assert result.group_level == "group"
    assert "group_formation" in result.tags
    print("✅ test_group_formation PASSED")


def test_crowd_intrusion():
    """5+ persons in zone = crowd intrusion."""
    analyzer = BehaviorAnalyzer(group_threshold=3, crowd_threshold=5)
    
    result = analyzer.analyze_group(zone_name="Test Zone", persons_in_zone=7)
    
    assert result.is_group is True
    assert result.is_crowd is True
    assert result.group_level == "crowd"
    assert "crowd_intrusion" in result.tags
    print("✅ test_crowd_intrusion PASSED")


def test_no_group():
    """Less than 3 persons = no group."""
    analyzer = BehaviorAnalyzer(group_threshold=3)
    
    result = analyzer.analyze_group(zone_name="Test Zone", persons_in_zone=2)
    
    assert result.is_group is False
    assert result.group_level == "none"
    print("✅ test_no_group PASSED")


def test_erratic_movement():
    """High direction variance + reasonable speed = erratic."""
    analyzer = BehaviorAnalyzer()
    
    # Extreme zig-zag pattern (large direction changes + fast speed)
    history = [
        (100, 100), (200, 110), (100, 120), (200, 130),
        (100, 140), (200, 150), (100, 160), (200, 170),
        (100, 180), (200, 190), (100, 200), (200, 210),
        (100, 220), (200, 230), (100, 240),
    ]
    
    result = analyzer.detect_erratic_movement(history, window=15)
    # Zig-zag has high direction variance
    assert result == True
    print("✅ test_erratic_movement PASSED")


def test_straight_movement_not_erratic():
    """Straight movement = NOT erratic."""
    analyzer = BehaviorAnalyzer()
    
    # Straight line
    history = [(100 + i * 10, 100) for i in range(15)]
    
    result = analyzer.detect_erratic_movement(history, window=15)
    assert result == False
    print("✅ test_straight_movement_not_erratic PASSED")


def test_approach_direction():
    """Entity moving toward zone center = approaching."""
    analyzer = BehaviorAnalyzer()
    
    result = analyzer.analyze_entity(
        entity_id="UNKNOWN-01",
        position_history=[(100, 100), (120, 100)],  # Moving right
        is_in_zone=False,
        zone_entry_time=None,
        zone_center=(200, 100),  # Zone is to the right
    )
    
    assert result.is_moving_toward_zone is True
    assert "approaching_zone" in result.tags
    print("✅ test_approach_direction PASSED")


if __name__ == "__main__":
    test_no_loitering()
    test_suspicious_loitering()
    test_high_risk_loitering()
    test_not_in_zone_no_loitering()
    test_speed_stationary()
    test_speed_walking()
    test_speed_running()
    test_group_formation()
    test_crowd_intrusion()
    test_no_group()
    test_erratic_movement()
    test_straight_movement_not_erratic()
    test_approach_direction()
    print("\n🎉 All behavior tests passed!")
