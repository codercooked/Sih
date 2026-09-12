"""
IBVAP — Threat Scoring Unit Tests
Tests rule-based threat score calculation, factor breakdown, and capping.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.threat_scorer import ThreatScorer, ThreatAssessment


def test_no_factors():
    """No threat factors = score 0."""
    scorer = ThreatScorer()
    result = scorer.calculate()
    assert result.score == 0
    assert result.level == "low"
    assert len(result.breakdown) == 0
    print("✅ test_no_factors PASSED")


def test_zone_entry_only():
    """Zone entry alone = 25 points."""
    scorer = ThreatScorer()
    result = scorer.calculate(is_in_zone=True)
    assert result.score == 25
    assert result.level == "low"
    assert len(result.breakdown) == 1
    assert result.breakdown[0][1] == 25
    print("✅ test_zone_entry_only PASSED")


def test_loitering_60s():
    """Loitering >60s in zone."""
    scorer = ThreatScorer()
    result = scorer.calculate(is_in_zone=True, loitering_duration=65)
    # zone_entry (25) + loitering_60s (20) = 45
    assert result.score == 45
    assert result.level == "medium"
    print("✅ test_loitering_60s PASSED")


def test_loitering_90s():
    """Loitering >90s = high-risk (replaces 60s score, doesn't stack)."""
    scorer = ThreatScorer()
    result = scorer.calculate(is_in_zone=True, loitering_duration=95)
    # zone_entry (25) + loitering_90s (35) = 60
    assert result.score == 60
    assert result.level == "high"
    print("✅ test_loitering_90s PASSED")


def test_group_formation():
    """3+ persons in zone."""
    scorer = ThreatScorer()
    result = scorer.calculate(is_in_zone=True, persons_in_zone=3)
    # zone_entry (25) + group_3plus (15) = 40
    assert result.score == 40
    print("✅ test_group_formation PASSED")


def test_crowd_intrusion():
    """5+ persons in zone (replaces group score)."""
    scorer = ThreatScorer()
    result = scorer.calculate(is_in_zone=True, persons_in_zone=6)
    # zone_entry (25) + group_5plus (25) = 50
    assert result.score == 50
    print("✅ test_crowd_intrusion PASSED")


def test_running_speed():
    """Running speed detected."""
    scorer = ThreatScorer()
    result = scorer.calculate(is_in_zone=True, speed_category="running")
    # zone_entry (25) + running (15) = 40
    assert result.score == 40
    print("✅ test_running_speed PASSED")


def test_crouching():
    """Crouching posture detected."""
    scorer = ThreatScorer()
    result = scorer.calculate(is_in_zone=True, posture="crouching")
    # zone_entry (25) + crouching (20) = 45
    assert result.score == 45
    print("✅ test_crouching PASSED")


def test_climbing():
    """Climbing posture detected."""
    scorer = ThreatScorer()
    result = scorer.calculate(is_in_zone=True, posture="climbing")
    # zone_entry (25) + climbing (30) = 55
    assert result.score == 55
    print("✅ test_climbing PASSED")


def test_night_context():
    """Night/low-light context."""
    scorer = ThreatScorer()
    result = scorer.calculate(is_in_zone=True, is_night_mode=True)
    # zone_entry (25) + night (10) = 35
    assert result.score == 35
    print("✅ test_night_context PASSED")


def test_maximum_threat():
    """All factors active = should be capped at 100."""
    scorer = ThreatScorer()
    result = scorer.calculate(
        is_in_zone=True,
        loitering_duration=100,
        persons_in_zone=6,
        speed_category="running",
        is_moving_toward_zone=True,
        is_night_mode=True,
        posture="climbing",
        is_erratic=True,
    )
    # Sum: 25 + 35 + 25 + 15 + 10 + 10 + 30 + 15 = 165 → capped at 100
    assert result.score == 100
    assert result.level == "critical"
    print("✅ test_maximum_threat PASSED")


def test_score_capping():
    """Score should never exceed 100."""
    scorer = ThreatScorer()
    result = scorer.calculate(
        is_in_zone=True,
        loitering_duration=100,
        persons_in_zone=6,
        speed_category="running",
        is_moving_toward_zone=True,
        is_night_mode=True,
        posture="climbing",
        is_erratic=True,
    )
    assert result.score <= 100
    print("✅ test_score_capping PASSED")


def test_breakdown_transparency():
    """Breakdown should list every contributing factor."""
    scorer = ThreatScorer()
    result = scorer.calculate(
        is_in_zone=True,
        loitering_duration=65,
        speed_category="running",
        is_night_mode=True,
    )
    factor_names = [name for name, pts in result.breakdown]
    assert any("zone" in name.lower() for name in factor_names)
    assert any("loitering" in name.lower() for name in factor_names)
    assert any("running" in name.lower() or "speed" in name.lower() for name in factor_names)
    assert any("night" in name.lower() for name in factor_names)
    print("✅ test_breakdown_transparency PASSED")


def test_threat_levels():
    """Test all threat level thresholds."""
    scorer = ThreatScorer()
    
    # Low: 0-29
    r = scorer.calculate()
    assert r.level == "low"
    
    # Medium: 30-59
    r = scorer.calculate(is_in_zone=True, is_night_mode=True)  # 35
    assert r.level == "medium"
    
    # High: 60-79
    r = scorer.calculate(is_in_zone=True, loitering_duration=95)  # 60
    assert r.level == "high"
    
    # Critical: 80-100
    r = scorer.calculate(
        is_in_zone=True, loitering_duration=95,
        posture="crouching", is_night_mode=True,
    )  # 25+35+20+10 = 90
    assert r.level == "critical"
    print("✅ test_threat_levels PASSED")


def test_custom_weights():
    """Test with custom weight overrides."""
    custom = {"zone_entry": 50, "crouching": 40}
    scorer = ThreatScorer(weights=custom)
    result = scorer.calculate(is_in_zone=True, posture="crouching")
    # 50 + 40 = 90
    assert result.score == 90
    print("✅ test_custom_weights PASSED")


if __name__ == "__main__":
    test_no_factors()
    test_zone_entry_only()
    test_loitering_60s()
    test_loitering_90s()
    test_group_formation()
    test_crowd_intrusion()
    test_running_speed()
    test_crouching()
    test_climbing()
    test_night_context()
    test_maximum_threat()
    test_score_capping()
    test_breakdown_transparency()
    test_threat_levels()
    test_custom_weights()
    print("\n🎉 All threat score tests passed!")
