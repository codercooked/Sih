"""
IBVAP — Verification Script for Clean Tactical HUD & Biometric Face Intercept Card
Tests:
1. Modern HUD overlays (corner reticles, non-blocking zone labels, threat badges)
2. FaceScanner biometric detection & candidate face crop extraction
3. FRS identification and database ID card matching
4. UI Face ID match card HTML generation & integrity
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.face_scanner import FaceScanner
from core.frs import FacialRecognitionSystem, FRSResult
from ui.video_overlay import (
    draw_detections,
    draw_zone,
    draw_threat_badge,
    draw_plate_text,
    draw_fps,
    _draw_tactical_reticle,
    _draw_pill_badge,
)
from ui.face_id_card import _bgr_to_base64


def test_tactical_hud_rendering():
    """Verify clean tactical reticles and badges render on frame without errors."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. Reticle and pill badge
    _draw_tactical_reticle(frame, 100, 100, 200, 300, (0, 230, 118), corner_len=14)
    _draw_pill_badge(frame, "ID-01 • Person [45%]", 100, 95, border_color=(0, 230, 118))

    # 2. Non-obstructive Zone polygon
    polygon = np.array([[50, 50], [400, 50], [400, 350], [50, 350]])
    frame = draw_zone(frame, polygon, is_intruded=True, zone_name="Restricted BOP Sector")

    # 3. Threat badge
    frame = draw_threat_badge(frame, score=85, level="critical")

    # 4. Plate text
    frame = draw_plate_text(frame, "DL-01-AB-1234", (150, 250), confidence=0.92, security_status="AUTHORIZED")

    # 5. FPS pill
    frame = draw_fps(frame, 28.5)

    assert frame.shape == (480, 640, 3)
    assert np.sum(frame) > 0, "Annotated frame must have drawn pixels"
    print("✅ Tactical HUD overlay rendering verified (reticles, zone pill, threat badge, FPS)")


def test_face_scanner_and_crop_extraction():
    """Verify FaceScanner detects or extracts face crops and identifies against watchlist."""
    scanner = FaceScanner()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Place a synthetic person bbox
    bbox = [100, 80, 220, 320]
    face_data = scanner.scan_for_faces(frame, bbox)

    assert isinstance(face_data, list)
    assert len(face_data) >= 1, "Expected at least 1 candidate face extraction from person bbox"
    
    candidate = face_data[0]
    assert "face_crop" in candidate
    assert candidate["face_crop"] is not None
    assert "frs_result" in candidate
    assert "box" in candidate
    print(f"✅ FaceScanner candidate extraction verified: box={candidate['box']}, FRS result={candidate['frs_result'].name}")


def test_face_id_card_html_pipeline():
    """Verify base64 encoding and metadata formatting for Face & ID card."""
    sample_img = np.zeros((100, 100, 3), dtype=np.uint8)
    b64 = _bgr_to_base64(sample_img)
    assert b64.startswith("data:image/jpeg;base64,")
    assert len(b64) > 30

    frs = FacialRecognitionSystem()
    assert 101 in frs.profiles
    profile_101 = frs.profiles[101]
    assert profile_101["id_card_number"] == "POI-IND-10492"
    assert profile_101["clearance"] == "CRITICAL RED NOTICE"
    print(f"✅ Biometric ID Record pipeline verified: Profile 101 ID={profile_101['id_card_number']}, Name={profile_101['name']}")


def run_all():
    print("=" * 65)
    print("  IBVAP CLEAN HUD & BIOMETRIC FACE-TO-ID VERIFICATION SUITE")
    print("=" * 65)
    test_tactical_hud_rendering()
    test_face_scanner_and_crop_extraction()
    test_face_id_card_html_pipeline()
    print("=" * 65)
    print("  ALL HUD & BIOMETRIC CHECKS PASSED SUCCESSFULLY ✅")
    print("=" * 65)


if __name__ == "__main__":
    run_all()
