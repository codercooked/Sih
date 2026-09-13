"""
IBVAP — Multi-BOP Command & Control Grid
Fulfills SIH Problem Statement 26187:
"Border security forces deploy CCTV cameras at Border Out Posts(BOPs), check posts,
 border roads, and other strategic locations... Improve situational awareness and
 response time for border security forces."

Provides synchronized 4-sector surveillance grid monitoring strategic border outposts.
"""

import streamlit as st
import os
import cv2


BOP_SECTORS = [
    {
        "id": "BOP-01",
        "name": "North Gate Checkpoint",
        "cam": "CAM-01 [1080p Optical]",
        "file": "02_border_checkpoint_multiclass.mp4",
        "threat_level": "MEDIUM",
        "score": 48,
        "badge_color": "#FFC107",
        "activity": "Mixed vehicle & pedestrian transit",
        "active_entities": 4
    },
    {
        "id": "BOP-02",
        "name": "East Perimeter Fence Line",
        "cam": "CAM-02 [1080p Optical]",
        "file": "01_perimeter_people_surveillance.mp4",
        "threat_level": "CRITICAL",
        "score": 88,
        "badge_color": "#F44336",
        "activity": "Restricted zone loitering / incursion",
        "active_entities": 6
    },
    {
        "id": "BOP-03",
        "name": "Watchtower 04 Elevation",
        "cam": "CAM-03 [FLIR / Low-Light]",
        "file": "05_night_vision_incursion.mp4",
        "threat_level": "HIGH",
        "score": 72,
        "badge_color": "#FF9800",
        "activity": "Night-time low illumination incursion",
        "active_entities": 2
    },
    {
        "id": "BOP-04",
        "name": "Border Road Axis (South)",
        "cam": "CAM-04 [High-Speed ANPR]",
        "file": "03_vehicle_perimeter_tracking.mp4",
        "threat_level": "LOW",
        "score": 18,
        "badge_color": "#4CAF50",
        "activity": "Highway vehicle perimeter tracking",
        "active_entities": 3
    },
]


def render_bop_command_grid():
    """
    Renders 4-sector synchronized Multi-BOP Command Center Grid.
    """
    st.markdown("""
    <div style="background: linear-gradient(90deg, #0f1b29 0%, #162a45 100%); padding: 18px 24px; border-radius: 12px; border-left: 5px solid #00d4ff; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 style="margin: 0; color: #ffffff; font-family: monospace; letter-spacing: 1px;">
                    📹 MULTI-BOP COMMAND & CONTROL SURVEILLANCE GRID
                </h3>
                <p style="margin: 4px 0 0 0; color: #8892b0; font-size: 0.88rem;">
                    Central Command Monitoring • 4 Border Out Posts • Standard IP CCTV Feeds Synced
                </p>
            </div>
            <div>
                <span style="background: rgba(0, 212, 255, 0.15); color: #00d4ff; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-family: monospace; border: 1px solid #00d4ff;">
                    QUAD-SECTOR LIVE
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    sample_base = os.path.join(os.path.dirname(__file__), "..", "sample_videos")

    # Render 2x2 Grid
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    cols = [row1_col1, row1_col2, row2_col1, row2_col2]

    for idx, bop in enumerate(BOP_SECTORS):
        with cols[idx]:
            # Sector Header
            st.markdown(f"""
            <div style="background: #112240; padding: 10px 14px; border-radius: 8px 8px 0 0; border: 1px solid #233554; border-bottom: none; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-weight: bold; color: #00d4ff; font-family: monospace;">{bop['id']} — {bop['name']}</span>
                    <br><span style="font-size: 0.75rem; color: #8892b0;">{bop['cam']}</span>
                </div>
                <div>
                    <span style="background: {bop['badge_color']}; color: #ffffff; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: bold;">
                        THREAT: {bop['score']}/100 ({bop['threat_level']})
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Video Preview Frame
            vid_path = os.path.join(sample_base, bop["file"])
            if os.path.exists(vid_path):
                cap = cv2.VideoCapture(vid_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 45) # Sample representative frame
                ret, frame = cap.read()
                cap.release()

                if ret:
                    # Annotate HUD frame
                    h, w = frame.shape[:2]
                    cv2.rectangle(frame, (10, 10), (w - 10, h - 10), (0, 212, 255), 1)
                    cv2.putText(frame, f"REC ● {bop['id']} LIVE", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(rgb_frame, use_container_width=True)
                else:
                    st.warning(f"Unable to read feed for {bop['id']}")
            else:
                st.info(f"Connecting to RTSP feed for {bop['id']}...")

            # Sector Telemetry Footer
            st.markdown(f"""
            <div style="background: #0a192f; padding: 8px 12px; border-radius: 0 0 8px 8px; border: 1px solid #233554; border-top: none; font-size: 0.8rem; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; color: #ccd6f6;">
                    <span><b>Activity:</b> {bop['activity']}</span>
                    <span style="color: #64ffda;"><b>Entities:</b> {bop['active_entities']} Active</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
