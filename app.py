"""
IBVAP — Intelligent Border Video Analytics Platform
Main Streamlit Dashboard Application

Integrates all core modules: detection, tracking, zone analysis,
behavior analysis, pose estimation, ANPR, threat scoring, and event logging.

Run with: streamlit run app.py
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import streamlit as st
from datetime import datetime
from html import escape

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_project_path(path: str) -> str:
    """Resolve configured relative paths against the project, not cwd."""
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.detector import ObjectDetector
from core.tracker import CentroidTracker
from core.zone import ZoneManager
from core.behavior import BehaviorAnalyzer
from core.night_enhance import NightEnhancer
from core.threat_scorer import ThreatScorer
from core.profiler import EntityProfiler
from core.heatmap import HeatmapAccumulator
from database.event_store import EventStore
from ui.video_overlay import (
    draw_detections, draw_zone, draw_threat_badge,
    draw_plate_text, draw_fps, draw_night_mode_indicator,
    draw_trajectory, draw_face_blur, draw_weapon_alert,
)
from ui.alert_panel import render_alert_panel, render_alert_summary, render_incident_workflow
from ui.profile_card import render_profile_card, render_mini_profile
from ui.status_bar import render_status_bar
from ui.analytics_dashboard import render_analytics_dashboard
from ui.geospatial_map import render_geospatial_map
from ui.threat_analysis_studio import render_threat_analysis_studio
from core.trajectory_predictor import predict_trajectory, compute_direction_toward_point
from core.narrative_engine import generate_narrative
from core.report_generator import generate_html_report

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IBVAP — Border Video Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* IBVAP operator-console theme */
    .stApp {
        background-color: #0b1020;
        background-image:
            linear-gradient(to right, rgba(148, 163, 184, 0.055) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(148, 163, 184, 0.055) 1px, transparent 1px),
            radial-gradient(circle at 50% 60%, rgba(236, 72, 153, 0.10) 0%, rgba(168, 85, 247, 0.045) 34%, transparent 68%);
        background-size: 40px 40px, 40px 40px, 100% 100%;
        background-attachment: fixed;
    }
    [data-testid="stSidebar"] > div:first-child { background: #111827; border-right: 1px solid #263247; }
    [data-testid="stSidebar"] { min-width: 230px; max-width: 230px; }
    .main .block-container {
        padding-top: 1.25rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }
    .ibvap-brand { display: flex; align-items: baseline; gap: 12px; margin: 0 0 1.25rem 0; }
    .ibvap-brand .brand-name { color: #67e8f9; font-size: 1.8rem; font-weight: 800; letter-spacing: .06em; }
    .ibvap-brand .brand-section { color: #f8fafc; font-size: 1.05rem; font-weight: 650; letter-spacing: .12em; }
    .ibvap-brand .brand-subtitle { color: #94a3b8; font-size: .78rem; letter-spacing: .08em; margin-left: auto; }
    .standby-feed { min-height: 320px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; border: 1px dashed #355072; border-radius: 14px; background: radial-gradient(circle at center, rgba(14, 165, 233, .12), rgba(15, 23, 42, .78) 56%); }
    .standby-feed .camera-icon { font-size: 2.6rem; margin-bottom: .5rem; }
    .standby-feed .eyebrow { color: #38bdf8; letter-spacing: .14em; font-size: .72rem; font-weight: 700; }
    .standby-feed h3 { color: #f8fafc; margin: .5rem 0; }
    .standby-feed p { color: #94a3b8; max-width: 400px; margin: 0; }
    h1 { color: #67e8f9 !important; font-size: 1.8rem !important; }
    h3 {
        color: #cbd5e1 !important;
        font-size: 1.1rem !important;
    }
    [data-testid="stMetric"] { background: rgba(17, 24, 39, .82); border: 1px solid #263247; border-radius: 12px; padding: 12px 14px; }
    [data-testid="stMetricValue"] { font-size: 1.35rem !important; color: #f8fafc !important;
    }
    [data-testid="stMetricLabel"] { font-size: 0.72rem !important; color: #94a3b8 !important; }
    .stButton > button { border-radius: 9px; border: 1px solid #334155; background: #172033; color: #e2e8f0; font-weight: 650; }
    .stButton > button:hover { border-color: #67e8f9; color: #67e8f9; }
    [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid #263247; }
    [data-baseweb="tab"] { color: #94a3b8; padding: 10px 14px; }
    [aria-selected="true"][data-baseweb="tab"] { color: #67e8f9; border-bottom-color: #67e8f9; }
    hr { border-color: #263247; }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Load Config ────────────────────────────────────────────────────────────
@st.cache_data
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


# ─── Initialize Components (cached) ────────────────────────────────────────
@st.cache_resource
def init_detector(conf_threshold, tracking_method):
    return ObjectDetector(
        confidence_threshold=conf_threshold,
        tracking_method=tracking_method,
    )


@st.cache_resource
def init_pose_analyzer(confidence_threshold):
    try:
        from core.pose_analyzer import PoseAnalyzer
        return PoseAnalyzer(confidence_threshold=confidence_threshold)
    except Exception as exc:
        # MediaPipe has changed APIs across releases; optional pose analysis
        # must not prevent the dashboard from starting.
        print(f"Pose analysis unavailable: {exc}")
        return None


@st.cache_resource
def init_anpr(confidence_threshold):
    try:
        from core.anpr import ANPREngine
        return ANPREngine(confidence_threshold=confidence_threshold)
    except Exception as exc:
        print(f"ANPR unavailable: {exc}")
        return None


@st.cache_resource
def init_event_store(db_path, output_dir):
    return EventStore(db_path=db_path, output_dir=output_dir)


@st.cache_resource
def init_face_scanner():
    try:
        from core.face_scanner import FaceScanner
        return FaceScanner()
    except Exception as exc:
        print(f"Face scanning unavailable: {exc}")
        return None


@st.cache_resource
def init_ai_insights():
    from core.ai_insights import AIInsightsAnalyzer
    return AIInsightsAnalyzer()


def init_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        "is_running": False,
        "tracker": None,
        "frame_count": 0,
        "fps": 0.0,
        "last_fps_time": time.time(),
        "fps_frame_count": 0,
        "selected_entity": None,
        "last_log_time": {},  # entity_id → last log timestamp
        "max_threat_score": 0,
        "ai_insights_cache": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ─── Main Application ──────────────────────────────────────────────────────
def main():
    init_session_state()
    config = load_config()

    # Title
    st.markdown(
        '<div class="ibvap-brand">'
        '<span class="brand-name">🛡️ IBVAP</span>'
        '<span class="brand-section">Perimeter operations</span>'
        '<span class="brand-subtitle">Border video analytics</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ─── Sidebar Controls ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Controls")

        # Video source
        source_type = st.selectbox(
            "Video Source",
            ["Sample Video", "Webcam (0)", "Video File", "RTSP Stream"],
            key="source_type",
        )

        video_source = 0
        if source_type == "Sample Video":
            sample_dir = os.path.join(os.path.dirname(__file__), "sample_videos")
            if os.path.exists(sample_dir):
                samples = [f for f in os.listdir(sample_dir) if f.endswith(('.mp4', '.avi'))]
                if samples:
                    selected_sample = st.selectbox("Select Sample", samples)
                    video_source = os.path.join(sample_dir, selected_sample)
                else:
                    st.warning("No sample videos found.")
                    video_source = None
            else:
                st.warning("Sample videos directory not found.")
                video_source = None
        elif source_type == "Video File":
            uploaded = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])
            if uploaded:
                # Save uploaded file temporarily
                temp_path = os.path.join(BASE_DIR, "temp_video.mp4")
                with open(temp_path, "wb") as f:
                    f.write(uploaded.read())
                video_source = temp_path
            else:
                st.info("Upload a video file to begin.")
                video_source = None
        elif source_type == "RTSP Stream":
            rtsp_url = st.text_input("RTSP URL", placeholder="rtsp://...")
            video_source = rtsp_url if rtsp_url else None

        st.divider()

        # Feature toggles
        st.markdown("### Detection & tracking")
        face_enabled = st.checkbox("Face Detection", value=config.get("face_detection_enabled", True))
        pose_enabled = st.checkbox("Pose Estimation", value=config.get("pose_estimation_enabled", True))
        anpr_enabled = st.checkbox("ANPR (Plate Reading)", value=config.get("anpr_enabled", True))
        heatmap_enabled = st.checkbox("Human movement heatmap", value=True)
        baggage_enabled = st.checkbox("Abandoned Baggage Detection", value=True)
        trajectory_enabled = st.checkbox("Trajectory prediction", value=True)
        weapon_detection_enabled = st.checkbox("Weapon detection", value=True)
        predictive_breach_enabled = st.checkbox("Predictive breach alert", value=True)
        narrative_enabled = st.checkbox("Threat narrative", value=True)

        # Hardware Integration (Mock for SIH)
        st.markdown("### Integrations")
        sms_alerts_enabled = st.checkbox("SMS alerts", value=True)
        siren_enabled = st.checkbox("Local siren", value=True)
        audio_alerts_enabled = st.checkbox("Audio alerts", value=True)

        st.divider()

        # AI API Configuration
        st.markdown("### Intelligence settings")
        gemini_api_key = st.text_input(
            "Gemini API Key",
            value=os.environ.get("GEMINI_API_KEY", ""),
            type="password",
            help="Enter Google Gemini API key for live LLM intelligence briefings (optional, autonomous fallback active)",
        )
        gemini_model = st.selectbox(
            "Gemini Model",
            ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
            index=0,
        )

        st.divider()

        # Zone configuration
        st.markdown("### Zone configuration")
        zone_polygon_str = st.text_area(
            "Zone Polygon (JSON)",
            value=json.dumps(config.get("zone_polygon", [[100, 100], [500, 100], [500, 400], [100, 400]])),
            height=80,
        )

        st.divider()

        # Actions
        st.markdown("### Actions")
        if st.button("🗑️ Clear Event Log"):
            event_store = init_event_store(
                resolve_project_path(config.get("database_path", "./ibvap_events.db")),
                resolve_project_path(config.get("output_dir", "./alerts/")),
            )
            event_store.clear_all()
            st.success("Event log cleared!")

        if st.button("📥 Export CSV"):
            event_store = init_event_store(
                resolve_project_path(config.get("database_path", "./ibvap_events.db")),
                resolve_project_path(config.get("output_dir", "./alerts/")),
            )
            csv_path = event_store.export_csv()
            st.success(f"Exported to {csv_path}")

        if st.button("📄 Generate PDF Report"):
            report_path = generate_html_report(
                db_path=resolve_project_path(config.get("database_path", "./ibvap_events.db")),
                output_path=os.path.join(BASE_DIR, "ibvap_incident_report.html"),
                zone_name=config.get("zone_name", "Restricted Area"),
            )
            with open(report_path, 'r') as f:
                html_content = f.read()
            st.download_button(
                label="⬇️ Download Report",
                data=html_content,
                file_name="IBVAP_Incident_Report.html",
                mime="text/html",
                use_container_width=True,
            )
            st.success("Report generated!")

        st.divider()

        # Limitations disclaimer
        st.markdown(
            '<div style="padding: 8px; background: #111827; border: 1px solid #263247; border-radius: 8px; '
            'font-size: 11px; color: #888;">'
            '<strong style="color:#cbd5e1;">Included capabilities</strong><br>'
            '• Deep Learning Neural Network (4-layer MLP, 100K records)<br>'
            '• Trajectory Prediction & Direction Analysis<br>'
            '• AI-Powered Threat Narratives<br>'
            '• Weapon Detection (Knife, Bat, Scissors)<br>'
            '• Privacy Face Blur (DPDPA 2023 Compliant)<br>'
            '• Heatmap, Skeleton, Abandoned Baggage, ANPR<br>'
            '• One-Click PDF Incident Reports<br>'
            '• SMS/Siren/Audio Alert Simulation'
            '</div>',
            unsafe_allow_html=True,
        )

    # ─── Initialize Components ──────────────────────────────────────────
    detector = init_detector(
        config.get("detection_confidence_threshold", 0.5),
        config.get("tracking_method", "bytetrack"),
    )
    detector.face_detection_enabled = face_enabled and not detector.face_cascade.empty()

    zone_manager = ZoneManager()
    try:
        polygon = json.loads(zone_polygon_str)
        zone_manager.add_zone(config.get("zone_name", "Restricted Area"), polygon)
    except json.JSONDecodeError:
        st.error("Invalid zone polygon JSON!")
        return

    behavior_analyzer = BehaviorAnalyzer(
        loitering_suspicious_sec=config.get("loitering_threshold_suspicious_sec", 30),
        loitering_high_risk_sec=config.get("loitering_threshold_high_risk_sec", 90),
        group_threshold=config.get("group_threshold_count", 3),
        crowd_threshold=config.get("crowd_threshold_count", 5),
        speed_running_threshold=config.get("speed_threshold_running", 15.0),
    )

    night_enhancer = NightEnhancer(
        brightness_threshold=config.get("night_mode_brightness_threshold", 80),
        clip_limit=config.get("clahe_clip_limit", 2.0),
    )

    threat_scorer = ThreatScorer(weights=config.get("threat_score_weights"))
    profiler = EntityProfiler()
    event_store = init_event_store(
        resolve_project_path(config.get("database_path", "./ibvap_events.db")),
        resolve_project_path(config.get("output_dir", "./alerts/")),
    )
    face_scanner = init_face_scanner()

    # Initialize tracker in session state
    if st.session_state.tracker is None:
        st.session_state.tracker = CentroidTracker(
            max_disappeared=config.get("tracker_max_disappeared", 30),
            max_distance=config.get("tracker_max_distance", 80),
        )
    tracker = st.session_state.tracker

    # Lazy-load heavy modules
    pose_analyzer = None
    if pose_enabled:
        pose_analyzer = init_pose_analyzer(config.get("pose_confidence_threshold", 0.6))

    anpr_engine = None
    if anpr_enabled:
        anpr_engine = init_anpr(config.get("anpr_confidence_threshold", 0.4))

    # ─── Main Layout ────────────────────────────────────────────────────
    tab_live, tab_threat_lab, tab_analytics, tab_map, tab_ai = st.tabs([
        "🔴 Live Surveillance", 
        "🎯 Threat Analysis & Forensics",
        "📊 Analytics Dashboard", 
        "🗺️ Geo-Spatial Command",
        "🧠 Strategic AI Intelligence"
    ])

    with tab_threat_lab:
        render_threat_analysis_studio(
            threat_scorer=threat_scorer,
            db_path=resolve_project_path(config.get("database_path", "./ibvap_events.db"))
        )

    with tab_analytics:
        render_analytics_dashboard(resolve_project_path(config.get("database_path", "./ibvap_events.db")))
        
    with tab_map:
        recent_events = event_store.get_recent_events(limit=10)
        render_geospatial_map(recent_events)

    with tab_ai:
        st.markdown("## Strategic intelligence")
        st.caption("Multi-sensor telemetry synthesis & LLM tactical reasoning (Powered by Google Gemini)")

        all_events = event_store.get_recent_events(limit=50)
        current_entities = st.session_state.tracker.entities if st.session_state.tracker else {}
        stats_payload = {
            "max_threat_score": st.session_state.max_threat_score,
            "total_frames": st.session_state.frame_count,
        }

        col_ai_btn, col_ai_info = st.columns([2, 3])
        with col_ai_btn:
            run_ai = st.button("⚡ Generate AI Strategic Intelligence Briefing", type="primary", use_container_width=True)
        with col_ai_info:
            if gemini_api_key and len(gemini_api_key.strip()) > 8:
                st.success(f"🟢 Connected to Google Gemini ({gemini_model})")
            else:
                st.info("ℹ️ Running in Autonomous Tactical Synthesis Mode (add Gemini Key in sidebar for Cloud LLM)")

        if run_ai or st.session_state.get("ai_insights_cache") is None:
            if all_events or current_entities or run_ai:
                with st.spinner("Synthesizing surveillance data with AI engine..."):
                    insights_engine = init_ai_insights()
                    result = insights_engine.analyze(
                        events=all_events,
                        active_entities=current_entities,
                        stats=stats_payload,
                        api_key=gemini_api_key,
                        model_name=gemini_model,
                    )
                    st.session_state["ai_insights_cache"] = result

        insights_data = st.session_state.get("ai_insights_cache")
        if insights_data:
            defcon = insights_data.get("defcon_level", "DEFCON 3")
            color = insights_data.get("alert_color", "#FFCC00")
            provider = insights_data.get("provider", "VIGIL-AI")

            st.markdown(
                f"""<div style="
                    padding: 18px; 
                    border-radius: 12px; 
                    border: 2px solid {color}; 
                    background: linear-gradient(135deg, rgba(30,30,30,0.9), rgba(15,15,15,0.95)); 
                    margin-bottom: 20px;
                    box-shadow: 0 0 15px {color}33;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 11px; text-transform: uppercase; color: #a1a1aa; letter-spacing: 1px;">Current Tactical Threat Posture</span>
                            <h2 style="margin: 0; color: {color}; font-size: 24px; font-weight: 800;">{defcon}</h2>
                        </div>
                        <div style="text-align: right;">
                            <span style="background: rgba(255,255,255,0.1); padding: 5px 12px; border-radius: 20px; font-size: 11px; color: #e4e4e7; font-weight: 600;">Engine: {provider}</span>
                            <div style="font-size: 11px; color: #71717a; margin-top: 6px;">Synchronized: {insights_data.get('timestamp', '')}</div>
                        </div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

            key_insights = insights_data.get("key_insights", [])
            if key_insights:
                st.markdown("##### ⚡ Key Tactical Flash Points")
                cols = st.columns(len(key_insights)) if len(key_insights) <= 3 else st.columns(3)
                for idx, insight in enumerate(key_insights[:3]):
                    with cols[idx % len(cols)]:
                        st.info(insight)

            st.markdown(insights_data.get("markdown_report", ""))

            st.download_button(
                label="📥 Download Tactical Intelligence Report (.md)",
                data=insights_data.get("markdown_report", ""),
                file_name=f"VIGIL_AI_Intelligence_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            )
        else:
            st.info("Start video monitoring to record telemetry, then click 'Generate AI Strategic Intelligence Briefing'.")

    with tab_live:
        # Health belongs at the top of the workspace, not in the narrow
        # incident rail. This keeps operational metrics legible.
        status_placeholder = st.empty()
        st.markdown('<div style="height: 10px"></div>', unsafe_allow_html=True)
        col_video, col_panel = st.columns([3, 2], gap="large")

    with col_panel:

        # Profile card area
        profile_placeholder = st.empty()

        # Alert log
        alert_placeholder = st.empty()

    with col_video:
        # Start/Stop button
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            start_btn = st.button(
                "▶️ Start Monitoring" if not st.session_state.is_running else "⏸️ Pause",
                use_container_width=True,
                type="primary",
            )
        with col_btn2:
            stop_btn = st.button("⏹️ Stop & Reset", use_container_width=True)

        if start_btn:
            st.session_state.is_running = not st.session_state.is_running
            if st.session_state.is_running:
                # Do not carry heat from a previous run into a new camera
                # session, especially when switching from people to vehicles.
                st.session_state.pop("heatmap_accumulator", None)
                st.session_state.tracker = CentroidTracker(
                    max_disappeared=config.get("tracker_max_disappeared", 30),
                    max_distance=config.get("tracker_max_distance", 80),
                )
                tracker = st.session_state.tracker

        if stop_btn:
            st.session_state.is_running = False
            st.session_state.tracker = None
            st.session_state.pop("heatmap_accumulator", None)
            st.session_state.frame_count = 0
            st.session_state.max_threat_score = 0
            st.rerun()

        # Video display placeholder
        video_placeholder = st.empty()

    # ─── Video Processing Loop ──────────────────────────────────────────
    if st.session_state.is_running and video_source is not None:
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            st.error(f"Cannot open video source: {video_source}")
            st.session_state.is_running = False
            return

        # Reset FPS counter
        st.session_state.last_fps_time = time.time()
        st.session_state.fps_frame_count = 0

        if heatmap_enabled and 'heatmap_accumulator' not in st.session_state:
            # Initialize with our target high-FPS resolution (640x360)
            # Match the resized processing frame to avoid OpenCV blend
            # errors when the heatmap is composited over the live feed.
            st.session_state.heatmap_accumulator = HeatmapAccumulator(512, 288)

        # Streamlit widgets must be created once per script run. The video
        # loop can refresh visual placeholders repeatedly, but rendering the
        # incident selectbox on every frame creates duplicate widget keys.
        incident_workflow_rendered = False
        last_detections = []

        while st.session_state.is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # Loop video for demo
                if isinstance(video_source, str) and os.path.isfile(video_source):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    st.warning("Video stream ended.")
                    st.session_state.is_running = False
                    break
            
            # Keep the local CPU pipeline responsive. The original sample
            # videos are larger than necessary for the operator preview.
            frame = cv2.resize(frame, (512, 288), interpolation=cv2.INTER_AREA)
            
            st.session_state.frame_count += 1
            
            # Render the lightweight preview more often than the analytics
            # pipeline so the operator feed stays responsive.
            render_this_frame = (st.session_state.frame_count % 2 == 0)
            inference_this_frame = st.session_state.frame_count % 4 == 0

            # Fast display path: between analytics frames, do not run the
            # tracker, zone logic, pose model, heatmap, or event pipeline.
            # This is the key latency fix for CPU-only local playback.
            if not inference_this_frame:
                if render_this_frame:
                    video_placeholder.image(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True,
                    )
                continue

            # ── Step 1: Night Enhancement ──
            frame, is_night_mode = night_enhancer.enhance(frame)

            # ── Step 2: Object Detection ──
            # YOLO inference is the most expensive operation in the local
            # CPU pipeline. Reuse the latest detections on alternate frames
            # so the preview remains fluid while tracking continues.
            if inference_this_frame or not last_detections:
                last_detections = detector.detect(frame)
            detections = last_detections
            person_detections = detector.get_persons(detections)
            vehicle_detections = detector.get_vehicles(detections)

            # ── Step 3: Update Tracker (persons + vehicles) ──
            trackable = person_detections + vehicle_detections
            entities = tracker.update(trackable)

            # ── Step 4: Zone Analysis ──
            zone = zone_manager.zones[0] if zone_manager.zones else None
            persons_in_zone = 0
            is_zone_intruded = False

            for entity_id, entity in entities.items():
                if zone:
                    prev_pos = entity.position_history[-2] if len(entity.position_history) >= 2 else None
                    zone_event = zone.check_entity(entity.centroid, prev_pos)
                    entity.is_in_zone = zone_event.is_inside

                    if zone_event.is_inside:
                        is_zone_intruded = True
                        if entity.class_name == "person":
                            persons_in_zone += 1
                        if entity.zone_entry_time is None:
                            entity.zone_entry_time = time.time()

                        # Add zone entry tag
                        if "zone_entry" not in entity.behavior_tags:
                            entity.behavior_tags.append("zone_entry")

                        # Check for entering event
                        if zone_event.direction == "ENTERING":
                            if "entering" not in entity.behavior_tags:
                                entity.behavior_tags.append("entering")
                    else:
                        entity.zone_entry_time = None
                        entity.is_in_zone = False

            # ── Step 5: Behavior Analysis ──
            weapon_dets = detector.get_weapons(detections) if weapon_detection_enabled else []

            for entity_id, entity in entities.items():
                is_person = entity.class_name == "person"

                zone_center = zone.get_center() if zone else None
                behavior = behavior_analyzer.analyze_entity(
                    entity_id=entity_id,
                    position_history=entity.position_history,
                    is_in_zone=entity.is_in_zone,
                    zone_entry_time=entity.zone_entry_time,
                    zone_center=zone_center,
                )

                # Update entity with behavior tags
                for tag in behavior.tags:
                    if tag not in entity.behavior_tags:
                        entity.behavior_tags.append(tag)

                # Heatmap only human movement; vehicle tracks stay available
                # for detection and alerts without adding overlay workload.
                if inference_this_frame and heatmap_enabled and 'heatmap_accumulator' in st.session_state and is_person:
                    cx, cy = entity.centroid
                    st.session_state.heatmap_accumulator.add_point(cx, cy)

                # Check erratic movement
                is_erratic = behavior_analyzer.detect_erratic_movement(entity.position_history)
                if is_erratic and "erratic_movement" not in entity.behavior_tags:
                    entity.behavior_tags.append("erratic_movement")

                # ── Step 6: Pose Analysis (every 3rd frame for performance) ──
                if pose_analyzer and is_person and inference_this_frame:
                    try:
                        pose_result = pose_analyzer.analyze(frame, entity.bbox)
                        if pose_result.confidence >= config.get("pose_confidence_threshold", 0.6):
                            entity.posture = pose_result.posture
                            entity.skeleton = pose_result.landmarks
                            for tag in pose_result.tags:
                                if tag not in entity.behavior_tags:
                                    entity.behavior_tags.append(tag)
                    except Exception:
                        pass  # Pose estimation can fail on edge cases

                # ── Step 6b: Weapon Detection ──
                has_weapon = False
                weapon_type = ""
                if weapon_dets:
                    ex1, ey1, ex2, ey2 = entity.bbox
                    for weapon in weapon_dets:
                        wx1, wy1, wx2, wy2 = weapon.bbox
                        wcx = (wx1 + wx2) // 2
                        wcy = (wy1 + wy2) // 2
                        if ex1 <= wcx <= ex2 and ey1 <= wcy <= ey2:
                            has_weapon = True
                            weapon_type = weapon.class_name
                            break
                    if has_weapon and "weapon_detected" not in entity.behavior_tags:
                        entity.behavior_tags.append("weapon_detected")

                # ── Step 6c: Direction Toward Zone ──
                direction_toward = 0.0
                if zone_center and len(entity.position_history) >= 2:
                    direction_toward = compute_direction_toward_point(
                        entity.position_history, zone_center
                    )

                # ── Step 7: Threat Scoring ──
                group_report = behavior_analyzer.analyze_group(
                    zone_name=config.get("zone_name", "Restricted Area"),
                    persons_in_zone=persons_in_zone,
                )

                threat = threat_scorer.calculate(
                    entity_type=1 if entity.class_name in ["car", "motorcycle", "truck", "bus"] else 0,
                    is_in_zone=entity.is_in_zone,
                    loitering_duration=entity.duration_in_zone,
                    persons_in_zone=persons_in_zone,
                    speed_category=behavior.speed_category,
                    is_moving_toward_zone=behavior.is_moving_toward_zone,
                    is_night_mode=is_night_mode,
                    posture=entity.posture or "standing",
                    is_erratic=is_erratic,
                    hour_of_day=datetime.now().hour,
                    direction_toward_zone=direction_toward,
                    crowd_density_gradient=0.0,
                    has_weapon=has_weapon,
                    time_since_last=time.time() - entity.last_seen if hasattr(entity, 'last_seen') else 0.0,
                    has_readable_plate=1 if (entity.vehicle_plate and entity.vehicle_plate != "UNREADABLE") else 0,
                    is_unauthorized_plate=1 if getattr(entity, 'is_unauthorized_plate', False) else 0,
                )

                entity.threat_score = threat.score
                entity.threat_breakdown = {name: pts for name, pts in threat.breakdown}

                # ── Step 7.5: Biometric Face Scanning ──
                entity.face_data = []
                if face_enabled and face_scanner is not None and entity.class_name == "person":
                    face_data = face_scanner.scan_for_faces(frame, entity.bbox)
                    if face_data:
                        entity.face_data = face_data
                        for data in face_data:
                            if data["watchlist_match"]:
                                threat.score = max(threat.score, 100)
                                entity.threat_score = threat.score
                                if "WATCHLIST MATCH" not in entity.behavior_tags:
                                    entity.behavior_tags.append("WATCHLIST MATCH")
                                entity.threat_breakdown["Biometric Watchlist"] = 100
                                break

                # Track max threat
                if threat.score > st.session_state.max_threat_score:
                    st.session_state.max_threat_score = threat.score

                # ── Step 8: Event Logging (Persons) ──
                should_log = False
                last_log = st.session_state.last_log_time.get(entity_id, 0)
                time_since_log = time.time() - last_log

                if entity.is_in_zone and (
                    time_since_log > 30  # Every 30s of loitering
                    or last_log == 0     # First detection in zone
                    or threat.score >= 60 and time_since_log > 10  # High threat, more frequent
                ):
                    should_log = True

                if should_log:
                    snapshot_path = event_store.save_snapshot(frame, entity_id)
                    event_store.log_event(
                        entity_id=entity_id,
                        zone_name=config.get("zone_name", ""),
                        threat_score=threat.score,
                        threat_level=threat.level,
                        behaviour_tags=entity.behavior_tags,
                        snapshot_path=snapshot_path,
                        vehicle_plate=entity.vehicle_plate or "",
                        num_persons_in_zone=persons_in_zone,
                        speed_category=behavior.speed_category,
                        posture=entity.posture or "standing",
                        loitering_duration_sec=entity.duration_in_zone,
                    )
                    st.session_state.last_log_time[entity_id] = time.time()
                    
                    # Hardware Integration Mocks (SIH Feature)
                    if threat.score >= 80:
                        if sms_alerts_enabled:
                            st.toast(f"📱 SMS Sent to Commander: CRITICAL threat from {entity_id} at {config.get('zone_name', 'Sector')}!", icon="📱")
                        if siren_enabled:
                            st.toast(f"🚨 Local Siren Triggered for {entity_id}!", icon="🚨")

            # ── Decay Heatmap ──
            if heatmap_enabled and 'heatmap_accumulator' in st.session_state:
                st.session_state.heatmap_accumulator.update()

            # ── Step 7b: Abandoned Baggage ──
            if baggage_enabled:
                baggage_items = tracker.get_baggage()
                persons = tracker.get_persons()
                for bag_id, bag in baggage_items.items():
                    if bag.is_in_zone and bag.duration_in_zone > 10:
                        min_dist = float('inf')
                        for person_id, person in persons.items():
                            dist = np.linalg.norm(np.array(bag.centroid) - np.array(person.centroid))
                            min_dist = min(min_dist, dist)
                        
                        if min_dist > 150: # Pixels away
                            if "abandoned_baggage" not in bag.behavior_tags:
                                bag.behavior_tags.append("abandoned_baggage")
                            bag.threat_score = 85
                            bag.threat_breakdown = {"Abandoned Baggage Detected": 85}
                            if bag.threat_score > st.session_state.max_threat_score:
                                st.session_state.max_threat_score = bag.threat_score
                            
                            # Log Baggage Event
                            last_log = st.session_state.last_log_time.get(bag_id, 0)
                            if time.time() - last_log > 30 or last_log == 0:
                                snapshot_path = event_store.save_snapshot(frame, bag_id)
                                event_store.log_event(
                                    entity_id=bag_id,
                                    zone_name=config.get("zone_name", ""),
                                    threat_score=bag.threat_score,
                                    threat_level="critical",
                                    behaviour_tags=bag.behavior_tags,
                                    snapshot_path=snapshot_path,
                                    vehicle_plate="",
                                    num_persons_in_zone=persons_in_zone,
                                    speed_category="stationary",
                                    posture="unknown",
                                    loitering_duration_sec=bag.duration_in_zone,
                                )
                                st.session_state.last_log_time[bag_id] = time.time()
                                
                                # Hardware Integration Mocks (SIH Feature)
                                if sms_alerts_enabled:
                                    st.toast(f"📱 SMS Sent to Commander: Abandoned Baggage detected at {config.get('zone_name', 'Sector')}!", icon="📱")
                                if siren_enabled:
                                    st.toast(f"🚨 Local Siren Triggered for Baggage Alert!", icon="🚨")



            # ── Step 9: ANPR (for vehicles in zone) ──
            if anpr_engine:
                for entity_id, entity in entities.items():
                    if entity.class_name in ("car", "truck", "bus", "motorcycle"):
                        if entity.vehicle_plate is None and entity.is_in_zone:
                            try:
                                plate_result = anpr_engine.read_plate(frame, entity.bbox)
                                if plate_result.is_readable:
                                    entity.vehicle_plate = plate_result.text
                                    entity.is_unauthorized_plate = plate_result.is_unauthorized
                            except Exception:
                                pass

            # ── Step 10: Draw Overlays ──
            display_frame = frame.copy()

            has_current_person = bool(person_detections)
            if heatmap_enabled and has_current_person and 'heatmap_accumulator' in st.session_state:
                display_frame = st.session_state.heatmap_accumulator.blend(display_frame)

            # Draw zone
            if zone:
                display_frame = draw_zone(
                    display_frame,
                    zone.polygon,
                    is_intruded=is_zone_intruded,
                    zone_name=config.get("zone_name", "Restricted Area"),
                )

            # Draw detections and tracked entities
            display_frame = draw_detections(display_frame, detections, entities)

            # Draw threat badge (highest score)
            if entities:
                max_entity = max(entities.values(), key=lambda e: e.threat_score)
                if max_entity.threat_score > 0:
                    level = "low"
                    if max_entity.threat_score >= 80:
                        level = "critical"
                    elif max_entity.threat_score >= 60:
                        level = "high"
                    elif max_entity.threat_score >= 30:
                        level = "medium"
                    display_frame = draw_threat_badge(
                        display_frame, max_entity.threat_score, level
                    )

            # Draw vehicle plates
            for entity_id, entity in entities.items():
                if entity.vehicle_plate:
                    display_frame = draw_plate_text(
                        display_frame,
                        entity.vehicle_plate,
                        (entity.bbox[0], entity.bbox[1] - 30),
                    )
                # Draw Biometric Scans
                if hasattr(entity, 'face_data') and entity.face_data:
                    display_frame = face_scanner.draw_biometric_scan(display_frame, entity.face_data)
                # Draw Skeleton / Pose Landmarks
                if hasattr(entity, 'skeleton') and entity.skeleton:
                    if pose_analyzer:
                        display_frame = pose_analyzer.draw_skeleton(display_frame, entity.skeleton)

            # ── Step 10b: Trajectory Prediction Lines ──
            if trajectory_enabled:
                for entity_id, entity in entities.items():
                    if entity.class_name == "person" and len(entity.position_history) >= 5:
                        predicted = predict_trajectory(entity.position_history)
                        if predicted:
                            display_frame = draw_trajectory(display_frame, predicted)

            # ── Step 10c: Predictive Zone Breach Alert ──
            if predictive_breach_enabled and zone:
                for entity_id, entity in entities.items():
                    if entity.class_name == "person" and not entity.is_in_zone and len(entity.position_history) >= 5:
                        predicted = predict_trajectory(entity.position_history, num_future_points=20)
                        if predicted:
                            # Check if any predicted point falls inside the zone
                            for pt in predicted:
                                if zone.point_in_polygon(pt):
                                    # PREDICTIVE ALERT — person will breach in the future
                                    h_f, w_f = display_frame.shape[:2]
                                    banner = f"PREDICTIVE ALERT: {entity_id} approaching zone!"
                                    (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                                    bx = (w_f - tw) // 2
                                    by = h_f - 40
                                    cv2.rectangle(display_frame, (bx - 10, by - th - 10), (bx + tw + 10, by + 10), (0, 140, 255), -1)
                                    cv2.putText(display_frame, banner, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                                    
                                    if 'last_predictive_time' not in st.session_state:
                                        st.session_state.last_predictive_time = 0
                                    if time.time() - st.session_state.last_predictive_time > 8:
                                        st.toast(f"⚡ PREDICTIVE: {entity_id} will breach zone in ~5 seconds!", icon="⚡")
                                        st.session_state.last_predictive_time = time.time()
                                    break

            # ── Step 10d: Weapon Alert Overlay ──
            if weapon_detection_enabled:
                weapon_dets = detector.get_weapons(detections)
                if weapon_dets:
                    display_frame = draw_weapon_alert(display_frame, weapon_dets)

            # Night mode indicator
            if is_night_mode:
                display_frame = draw_night_mode_indicator(display_frame)

            # FPS calculation
            st.session_state.fps_frame_count += 1
            elapsed = time.time() - st.session_state.last_fps_time
            if elapsed >= 1.0:
                st.session_state.fps = st.session_state.fps_frame_count / elapsed
                st.session_state.fps_frame_count = 0
                st.session_state.last_fps_time = time.time()

            display_frame = draw_fps(display_frame, st.session_state.fps)

            # ── Step 10e: Audio Alert (browser beep) ──
            if audio_alerts_enabled and entities:
                max_e = max(entities.values(), key=lambda e: e.threat_score)
                if max_e.threat_score >= 80:
                    if 'last_audio_time' not in st.session_state:
                        st.session_state.last_audio_time = 0
                    if time.time() - st.session_state.last_audio_time > 10:
                        st.toast("🔊 AUDIO ALERT: Critical threat detected!", icon="🔊")
                        st.session_state.last_audio_time = time.time()

            # ── Step 11: Update Dashboard (Smooth 30 FPS, Zero Blinking) ──
            if render_this_frame:
                # Check for critical threat to display Drone Intercept HUD as seamless PiP overlay
                if entities:
                    max_e = max(entities.values(), key=lambda e: e.threat_score)
                    if max_e.threat_score >= 80: # Critical threat
                        from ui.drone_hud import generate_drone_hud_frame
                        hud_pip = generate_drone_hud_frame(frame, max_e.bbox, target_w=150, target_h=120)
                        if hud_pip is not None:
                            h_pip, w_pip = hud_pip.shape[:2]
                            display_frame[10:10+h_pip, -10-w_pip:-10] = hud_pip

                # Single unified image feed — never swaps containers or causes DOM flickering
                display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(display_rgb, channels="RGB", use_container_width=True)

                now_t = time.time()

                # Status bar (throttled to 1.5s to prevent metrics tearing / strobe effect)
                if 'last_status_update' not in st.session_state:
                    st.session_state.last_status_update = 0.0

                if now_t - st.session_state.last_status_update >= 1.5:
                    st.session_state.last_status_update = now_t
                    with status_placeholder.container():
                        render_status_bar(
                            is_active=True,
                            fps=st.session_state.fps,
                            total_entities=tracker.total_tracked,
                            active_entities=tracker.active_count,
                            total_alerts=event_store.get_event_count(),
                            is_night_mode=is_night_mode,
                            video_source=str(video_source),
                        )

                # Profile card and Alert log update (Throttled to 2.5s for calm, readable UI)
                if 'last_dashboard_update' not in st.session_state:
                    st.session_state.last_dashboard_update = 0.0

                if now_t - st.session_state.last_dashboard_update >= 2.5:
                    st.session_state.last_dashboard_update = now_t
                    
                    # Profile card for highest-threat entity
                    with profile_placeholder.container():
                        if entities:
                            max_entity = max(entities.values(), key=lambda e: e.threat_score)
                            if max_entity.threat_score > 0:
                                threat_for_profile = threat_scorer.calculate(
                                    entity_type=1 if max_entity.class_name in ["car", "motorcycle", "truck", "bus"] else 0,
                                    is_in_zone=max_entity.is_in_zone,
                                    loitering_duration=max_entity.duration_in_zone,
                                    persons_in_zone=persons_in_zone,
                                    posture=max_entity.posture or "standing",
                                    is_night_mode=is_night_mode,
                                    has_readable_plate=1 if (max_entity.vehicle_plate and max_entity.vehicle_plate != "UNREADABLE") else 0,
                                    is_unauthorized_plate=1 if getattr(max_entity, 'is_unauthorized_plate', False) else 0,
                                )
                                profile = profiler.build_profile(
                                    max_entity, frame, threat_for_profile
                                )
                                profile.zone_name = config.get("zone_name", "")
                                render_profile_card(profile)
    
                    # Alert log + AI Narrative
                    with alert_placeholder.container():
                        # AI Narrative for highest threat entity
                        if narrative_enabled and entities:
                            max_entity = max(entities.values(), key=lambda e: e.threat_score)
                            if max_entity.threat_score > 10:
                                narrative = generate_narrative(
                                    entity_id=max_entity.entity_id,
                                    entity_type=max_entity.class_name,
                                    zone_name=config.get("zone_name", "Restricted Area"),
                                    threat_score=max_entity.threat_score,
                                    threat_level="critical" if max_entity.threat_score >= 80 else "high" if max_entity.threat_score >= 60 else "medium" if max_entity.threat_score >= 30 else "low",
                                    behavior_tags=max_entity.behavior_tags,
                                    posture=max_entity.posture or "standing",
                                    loitering_duration=max_entity.duration_in_zone,
                                    is_night_mode=is_night_mode,
                                    vehicle_plate=max_entity.vehicle_plate or "",
                                    persons_in_zone=persons_in_zone,
                                    has_weapon="weapon_detected" in max_entity.behavior_tags,
                                )
                                st.markdown(
                                    f'<div style="padding: 10px 14px; background: linear-gradient(135deg, #1a1a2e, #16213e); '
                                    f'border-radius: 8px; border-left: 4px solid #69F0AE; margin-bottom: 10px; font-size: 13px;">'
                                    f'<strong style="color: #69F0AE;">🤖 AI Threat Narrative</strong><br>'
                                    f'<span style="color: #c9d1d9;">{narrative}</span>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
    
                        recent_events = event_store.get_recent_events(limit=15)
                        render_alert_summary(recent_events)
                        if not incident_workflow_rendered:
                            render_incident_workflow(recent_events, event_store)
                            incident_workflow_rendered = True
                        render_alert_panel(recent_events, max_display=10)

                # Smooth frame rate pacing (avoids websocket browser overload)
                time.sleep(0.03)

        cap.release()

    elif not st.session_state.is_running:
        # Show idle state
        with status_placeholder.container():
            render_status_bar(
                is_active=False,
                fps=0.0,
                total_entities=0,
                active_entities=0,
                total_alerts=event_store.get_event_count(),
            )

        with alert_placeholder.container():
            recent_events = event_store.get_recent_events(limit=15)
            render_alert_summary(recent_events)
            render_incident_workflow(recent_events, event_store)
            render_alert_panel(recent_events, max_display=10)

        source_name = escape(os.path.basename(str(video_source)) if video_source else "No source selected")
        video_placeholder.markdown(
            f'''<div class="standby-feed">
                <div class="camera-icon">◉</div>
                <div class="eyebrow">SURVEILLANCE FEED · STANDBY</div>
                <h3>Monitoring is ready to deploy</h3>
                <p>Selected source: <strong>{source_name}</strong><br>
                Start monitoring to activate detection, tracking, and incident automation.</p>
            </div>''',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
