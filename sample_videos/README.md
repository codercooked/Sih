# IBVAP — Sample Testing Videos Repository

This directory contains benchmark test surveillance video clips tailored for verifying and demonstrating IBVAP's real-time computer vision, behavior analytics, and threat scoring pipeline out of the box.

---

## 📹 Available Sample Testing Videos

| Video File | Resolution & FPS | Frame Count | Primary Detection & Analytics Scenario |
| :--- | :--- | :--- | :--- |
| **`01_perimeter_people_surveillance.mp4`** | 768x432 @ 12.0 FPS | 596 frames | **Perimeter Loitering & Multi-Person Activity**: Pedestrians entering surveillance zone, testing dwell time and group formation. |
| **`02_border_checkpoint_multiclass.mp4`** | 768x432 @ 12.0 FPS | 647 frames | **Multi-Class Checkpoint Analysis**: Mixed traffic containing pedestrians, bicycles, and vehicles approaching access barriers. |
| **`03_vehicle_perimeter_tracking.mp4`** | 768x432 @ 12.5 FPS | 377 frames | **Vehicle Tracking & Speed Monitoring**: Cars moving along roadway boundaries; testing velocity estimation and license plate OCR. |
| **`04_single_intruder_incursion.mp4`** | 768x432 @ 10.0 FPS | 1394 frames | **Restricted Boundary Intrusion**: Individual pedestrian crossing perimeter line directly into high-risk zone polygon. |
| **`05_night_vision_incursion.mp4`** | 768x432 @ 10.0 FPS | 250 frames | **Night & Low-Light Infiltration**: Night surveillance video with low ambient illumination. Recommended for testing **Night Mode (CLAHE)** and **Drone Thermal HUD**. |
| **`06_traffic_checkpoint_overview.mp4`** | 1280x720 @ 30.0 FPS | 300 frames | **Wide-Area Border Checkpoint**: Full HD overview of active multi-lane vehicle checkpoint with sustained tracking. |

---

## 🚀 How to Use in the Dashboard

1. Launch or open the IBVAP dashboard at `http://localhost:8501`.
2. In the sidebar under **Video Source**, select **`Sample Video`**.
3. Use the **Select Sample** dropdown to choose any of the 6 test files above.
4. Click **Start Surveillance** to initiate real-time AI inference, tracking, behavior scoring, and threat logging.

---

## 🧪 Automated Verification

You can verify and benchmark all sample testing files headlessly at any time by running:
```bash
python3 tests/test_sample_videos.py
```
This script checks:
- Video file integrity, dimensions, FPS, and decoder compatibility.
- End-to-end integration across YOLOv8 detector, CentroidTracker, BehaviorAnalyzer, and ThreatScorer.
