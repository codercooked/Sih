"""
IBVAP — FRS & Vehicle Security Test Suite
Fulfills SIH Problem Statement 26187:
Validates pure-software facial recognition (FRS) and tactical vehicle security database.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.frs import FacialRecognitionSystem, FRSResult
from core.anpr import ANPREngine, PlateResult, BLACKLIST_REGISTRY, WHITELIST_REGISTRY


def test_frs_default_watchlist():
    """Verify standard defense watchlist is pre-populated."""
    frs = FacialRecognitionSystem()
    assert len(frs.profiles) >= 4, f"Expected at least 4 default profiles, got {len(frs.profiles)}"
    
    # Check for suspect and authorized personnel
    categories = [p["category"] for p in frs.profiles.values()]
    assert "SUSPECT" in categories, "Watchlist must contain SUSPECT entries"
    assert "AUTHORIZED_BSF" in categories, "Watchlist must contain AUTHORIZED_BSF entries"
    print(f"✅ FRS default watchlist loaded with {len(frs.profiles)} profiles")


def test_frs_identification_flow():
    """Verify FRS identification returns valid FRSResult."""
    frs = FacialRecognitionSystem()
    test_face = np.random.randint(60, 200, (120, 120, 3), dtype=np.uint8)
    result = frs.identify(test_face)

    assert isinstance(result, FRSResult)
    assert result.category in ["SUSPECT", "AUTHORIZED_BSF", "CIVILIAN", "UNKNOWN"]
    assert 0.0 <= result.confidence_score <= 100.0
    print(f"✅ FRS identification flow passed (Matched: {result.name}, Category: {result.category})")


def test_frs_enrollment():
    """Verify live enrollment of a new border person of interest."""
    frs = FacialRecognitionSystem()
    initial_count = len(frs.profiles)
    sample_face = np.random.randint(50, 180, (100, 100, 3), dtype=np.uint8)

    new_id = frs.enroll_face(
        face_images=[sample_face],
        name="Test Infiltrator X",
        category="SUSPECT",
        role="Perimeter Breach Suspect",
        notes="Test enrollment"
    )

    assert new_id in frs.profiles
    assert len(frs.profiles) == initial_count + 1
    assert frs.profiles[new_id]["name"] == "Test Infiltrator X"
    print(f"✅ FRS enrollment passed (New ID: {new_id})")


def test_anpr_security_registry():
    """Verify vehicle security classification and blacklisting."""
    # Test Blacklist Plate
    for plate in BLACKLIST_REGISTRY.keys():
        assert plate in BLACKLIST_REGISTRY
        assert BLACKLIST_REGISTRY[plate]["reason"]

    # Test Whitelist Plate
    for plate in WHITELIST_REGISTRY.keys():
        assert plate in WHITELIST_REGISTRY

    # Test classification helper
    assert ANPREngine.classify_vehicle("truck") == "Heavy Cargo / Transport"
    assert ANPREngine.classify_vehicle("motorcycle") == "Two-Wheeler / Recon Axis"
    assert ANPREngine.classify_vehicle("car") == "Light Motor Vehicle"

    records = ANPREngine.get_security_database_records()
    assert len(records) == len(BLACKLIST_REGISTRY) + len(WHITELIST_REGISTRY)
    print(f"✅ ANPR Vehicle Security Registry passed ({len(records)} records validated)")


def run_all():
    print("=" * 65)
    print("  IBVAP SIH 26187 FRS & VEHICLE SECURITY TEST SUITE")
    print("=" * 65)
    test_frs_default_watchlist()
    test_frs_identification_flow()
    test_frs_enrollment()
    test_anpr_security_registry()
    print("=" * 65)
    print("  ALL FRS & VEHICLE SECURITY CHECKS PASSED SUCCESSFULLY ✅")
    print("=" * 65)


if __name__ == "__main__":
    run_all()
