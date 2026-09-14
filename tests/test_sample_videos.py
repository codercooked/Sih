"""
IBVAP — Sample Testing Videos Verification Test Suite
Tests availability, video metadata validity, and end-to-end pipeline ingestion
for all bundled sample testing files in sample_videos/.
"""

import os
import sys
import time
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.detector import ObjectDetector
from core.tracker import CentroidTracker
from core.behavior import BehaviorAnalyzer
from core.threat_scorer import ThreatScorer
from core.zone import ZoneManager, RestrictedZone

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_videos")

VIDEO_FILES = [
    "01_perimeter_people_surveillance.mp4",
    "02_border_checkpoint_multiclass.mp4",
    "03_vehicle_perimeter_tracking.mp4",
    "04_single_intruder_incursion.mp4",
    "05_night_vision_incursion.mp4",
    "06_traffic_checkpoint_overview.mp4",
    "07_biometric_face_intercept_sentry.mp4",
    "08_checkpoint_face_surveillance.mp4",
]


def test_sample_videos_directory_exists_and_populated():
    """Verify sample_videos/ exists and contains required test videos."""
    assert os.path.exists(SAMPLE_DIR), "sample_videos/ directory must exist"
    video_files = [f for f in os.listdir(SAMPLE_DIR) if f.endswith(('.mp4', '.avi'))]
    assert len(video_files) >= 8, f"Expected at least 8 sample videos, found {len(video_files)}: {video_files}"
    for required in VIDEO_FILES:
        assert required in video_files, f"Missing required sample video: {required}"
    print(f"✅ Found {len(video_files)} sample test videos in sample_videos/")


def test_sample_video_readable(filename):
    """Verify each video file can be opened by OpenCV and has valid video properties."""
    video_path = os.path.join(SAMPLE_DIR, filename)
    assert os.path.exists(video_path), f"Video file {filename} does not exist"

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video {filename} with OpenCV"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    assert width > 0, f"Invalid width {width} for {filename}"
    assert height > 0, f"Invalid height {height} for {filename}"
    assert fps > 0, f"Invalid fps {fps} for {filename}"
    assert total_frames > 0, f"Invalid frame count {total_frames} for {filename}"

    ret, frame = cap.read()
    assert ret is True, f"Failed to read initial frame from {filename}"
    assert frame.shape[0] == height and frame.shape[1] == width
    cap.release()
    print(f"✅ {filename}: {width}x{height} @ {fps:.1f}fps, {total_frames} frames verified")


def test_sample_video_pipeline_e2e():
    """Verify end-to-end detection, tracking, behavior analysis, and threat scoring."""
    video_path = os.path.join(SAMPLE_DIR, "04_single_intruder_incursion.mp4")
    assert os.path.exists(video_path)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 90)
    detector = ObjectDetector(model_name="yolov8n.pt", confidence_threshold=0.35)
    tracker = CentroidTracker(max_disappeared=30, max_distance=100)
    behavior = BehaviorAnalyzer(loitering_suspicious_sec=5, loitering_high_risk_sec=15)
    threat = ThreatScorer()

    # Define test perimeter zone
    zone = RestrictedZone("TestPerimeter", [[50, 50], [600, 50], [600, 400], [50, 400]])

    processed = 0
    total_detections = 0
    total_assessments = 0

    while cap.isOpened() and processed < 15:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Detect
        detections = detector.detect(frame)
        total_detections += len(detections)

        # 2. Track
        entities = tracker.update(detections)

        # 3. Behavior & Threat Analysis
        for entity in entities.values():
            in_zone = zone.point_in_polygon(entity.centroid)
            entry_t = time.time() - 10 if in_zone else None
            b_report = behavior.analyze_entity(
                entity_id=entity.entity_id,
                position_history=entity.position_history,
                is_in_zone=in_zone,
                zone_entry_time=entry_t,
                zone_center=(325, 225)
            )

            t_report = threat.calculate(
                is_in_zone=in_zone,
                loitering_duration=b_report.loitering_duration,
                approach_speed=b_report.speed,
                is_night=False
            )
            total_assessments += 1
            assert 0 <= t_report.score <= 100
            assert t_report.level in ["low", "medium", "high", "critical"]

        processed += 1

    cap.release()
    assert processed == 15, f"Expected 15 processed frames, got {processed}"
    assert total_detections > 0, f"Expected positive detections, got {total_detections}"
    assert total_assessments > 0, f"Expected positive threat assessments, got {total_assessments}"
    print(f"✅ Pipeline E2E test passed: processed {processed} frames, {total_detections} detections, {total_assessments} threat assessments")


def run_all():
    print("=" * 65)
    print("  IBVAP SAMPLE TESTING FILES & PIPELINE VERIFICATION SUITE")
    print("=" * 65)
    test_sample_videos_directory_exists_and_populated()
    print("-" * 65)
    for vid in VIDEO_FILES:
        test_sample_video_readable(vid)
    print("-" * 65)
    test_sample_video_pipeline_e2e()
    print("=" * 65)
    print("  ALL SAMPLE TESTING FILE CHECKS PASSED SUCCESSFULLY ✅")
    print("=" * 65)


if __name__ == "__main__":
    run_all()
