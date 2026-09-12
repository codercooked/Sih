# 🛡️ IBVAP — Intelligent Border Video Analytics Platform

A **pure-software** AI video analytics system for border surveillance using existing CCTV infrastructure. No dedicated hardware required — runs on standard IP cameras and consumer-grade computers.

## Features

| Feature | Technology | Status |
|---|---|---|
| Human & Vehicle Detection | YOLOv8 (Ultralytics) | ✅ |
| Face Detection | OpenCV Haar Cascade | ✅ |
| Virtual Fence / Zone Intrusion | Ray-casting point-in-polygon | ✅ |
| Tripwire Direction Detection | Frame-to-frame centroid analysis | ✅ |
| Loitering Detection | Per-entity timer in zone | ✅ |
| Group / Crowd Detection | Person count in zone | ✅ |
| Speed & Direction Analysis | Centroid velocity vector | ✅ |
| Posture Detection (Crouch/Climb) | MediaPipe Pose | ✅ |
| Night-Time Enhancement | OpenCV CLAHE | ✅ |
| License Plate Reading (ANPR) | OpenCV + EasyOCR | ✅ |
| Threat Scoring (0-100) | Rule-based weighted factors | ✅ |
| Event Logging | SQLite database | ✅ |
| Live Dashboard | Streamlit | ✅ |
| Entity Profile Cards | Physical descriptors + breakdown | ✅ |

## Quick Start

```bash
# 1. Clone/download the project
cd ibvap

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

## Usage

1. **Select video source** from the sidebar (webcam, video file, or RTSP stream)
2. **Click "▶️ Start Monitoring"** to begin analysis
3. **Watch the live feed** with bounding boxes, zone overlay, and threat scores
4. **Check the alert panel** on the right for real-time events
5. **Click entities** to see full profile cards with threat breakdown
6. **Export event log** to CSV using the sidebar button

## Architecture

```
Video Source (webcam / file / RTSP)
        ↓
CLAHE Enhancement (if low-light detected)
        ↓
YOLOv8 Detection (person, vehicle, face)
        ↓
Centroid-Based Tracking (UNKNOWN-XX IDs)
        ↓
Zone Geometry Check (point-in-polygon + tripwire)
        ↓
Behaviour Analysis Engine
  ├── Loitering timer (30s suspicious, 90s high-risk)
  ├── Group/crowd count (3+ group, 5+ crowd)
  ├── Speed/direction (stationary/walking/running + toward zone)
  └── Erratic movement (direction variance)
        ↓
MediaPipe Pose Analysis (every 3rd frame)
  ├── Crouching detection (knee angle < 110°)
  └── Climbing detection (wrists above shoulders)
        ↓
ANPR Layer (vehicles in zone only)
  ├── Bottom-35% crop
  ├── Bilateral filter + adaptive threshold
  └── EasyOCR character recognition
        ↓
Threat Scoring Engine (rule-based, 0-100)
  └── Transparent factor breakdown
        ↓
Alert & Logging
  ├── SQLite event insert
  ├── JPEG snapshot save
  └── Streamlit dashboard update
```

## Threat Scoring

The threat score is a **transparent, rule-based** weighted sum:

| Factor | Points |
|---|---|
| Restricted zone entry | +25 |
| Loitering >60s | +20 |
| Loitering >90s | +35 |
| Group of 3+ | +15 |
| Crowd of 5+ | +25 |
| Running speed | +15 |
| Moving toward boundary | +10 |
| Night/low-light context | +10 |
| Crouching detected | +20 |
| Climbing detected | +30 |
| Erratic movement | +15 |

**Total capped at 100.** Every alert shows the full breakdown.

## Configuration

Edit `config.json` to customize:
- Zone polygon coordinates
- Loitering/group thresholds
- Threat score weights
- Feature toggles (face/pose/ANPR)
- Night mode brightness threshold

## Testing

```bash
# Run all tests
python tests/test_zone.py
python tests/test_threat_score.py
python tests/test_tracker.py
python tests/test_behavior.py
```

## Custom Model Training and Evaluation

The detector supports Ultralytics ByteTrack by default. Set
`tracking_method` to `botsort` in `config.json` when stronger occlusion
handling or ReID is needed. If the Ultralytics tracker is unavailable, the
application falls back to the local tracker.

Prepare a real, camera-diverse YOLO dataset with `train`, `val`, and `test`
splits, then train from pretrained weights:

```bash
python tools/train_detector.py --data datasets/ibvap/data.yaml --epochs 50
```

Do not use the synthetic threat generator as a production ground truth. Label
real event windows and evaluate threshold quality with:

```bash
python tools/evaluate_threats.py labelled_events.csv --threshold 60
```

The CSV must contain `threat_score` and `is_threat` columns. Track precision,
recall, F1, and false-positive rate before changing alert thresholds.

## Demo Script (for Judges)

1. **Start dashboard** → "🟢 SYSTEM ACTIVE" visible
2. **Walk into zone** (normal pace) → person detected, box appears, score ~25-30
3. **Stop and crouch** → posture detection, score jumps to ~45-50
4. **Stand for 90s** → loitering accumulates, score > 60
5. **Second person enters** → group detected if 3+, score rises
6. **Both approach boundary** → direction detection, score 85+
7. **Check event log** → all events timestamped in SQLite
8. **View profile card** → full factor breakdown, clothing color, build estimate
9. **Export CSV** → verify data integrity

**Total demo time:** 5-7 minutes.

## Known Limitations (Honest Disclosure)

> **Transparency about limitations is a strength, not a weakness.**

1. **Threat scoring is rule-based, not ML-trained.** Weights are tunable but not learned from data. Future: train a classifier on real incident data.

2. **Night detection is software CLAHE enhancement, not thermal/IR imaging.** Works on contrast improvement only. Not as effective as thermal cameras.

3. **ANPR accuracy is 70-85% on clean, well-lit plates.** Drops significantly with poor lighting, extreme angles, or damaged plates. All attempts are logged regardless of confidence.

4. **Posture detection depends on camera angle.** MediaPipe works best on frontal/side views. Extreme angles or heavy occlusion reduce accuracy. Low-confidence detections are flagged.

5. **Tracking is centroid-based, not re-identification.** Works for single-camera zone monitoring. Cannot re-identify entities after long occlusion or across multiple cameras. For production: upgrade to ByteTrack or DeepSort.

6. **No multi-camera handoff.** Each camera runs independently.

7. **No face recognition.** Face detection only — deliberate design choice for privacy.

8. **No central command integration.** Dashboard is local. Future: REST API for command center.

## Project Structure

```
ibvap/
├── app.py                     # Streamlit dashboard
├── config.json                # Configuration
├── requirements.txt           # Dependencies
├── core/
│   ├── detector.py            # YOLOv8 detection
│   ├── tracker.py             # Centroid-based tracking
│   ├── zone.py                # Zone geometry + tripwire
│   ├── behavior.py            # Loitering, group, speed
│   ├── pose_analyzer.py       # MediaPipe pose
│   ├── anpr.py                # Plate recognition
│   ├── night_enhance.py       # CLAHE enhancement
│   ├── threat_scorer.py       # Rule-based scoring
│   └── profiler.py            # Entity profile cards
├── database/
│   └── event_store.py         # SQLite logging
├── ui/
│   ├── video_overlay.py       # Frame drawing
│   ├── alert_panel.py         # Alert log
│   ├── profile_card.py        # Profile cards
│   └── status_bar.py          # System status
├── alerts/                    # Snapshot storage
└── tests/                     # Unit tests
```

## Dependencies

- Python 3.9+
- ultralytics (YOLOv8)
- opencv-python
- streamlit
- mediapipe
- easyocr
- numpy, pandas, Pillow

## License

Educational project — built for college evaluation.
