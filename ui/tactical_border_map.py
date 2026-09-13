"""
IBVAP — Tactical Geospatial Border Map & Radar View
Fulfills SIH Problem Statement 26187:
"Improve situational awareness and response time for border security forces.
 Support integration with existing command and control systems."

Renders an interactive 2D Tactical Border Map representing Border Out Posts (BOPs),
watchtower arcs, restricted buffer zones, and live telemetry of detected entities
relative to the International Border (IB) Zero-Line.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time


# Simulated Border Geography Specifications
SECTORS_DATA = [
    {
        "bop_id": "BOP-01 (Alpha)",
        "sector": "North Gate Checkpost",
        "cam_id": "CAM-01",
        "x": 180,
        "y": 140,
        "status": "ONLINE",
        "fov_angle": 60,
        "threat_level": "LOW",
        "description": "Primary vehicle and personnel transit axis"
    },
    {
        "bop_id": "BOP-02 (Bravo)",
        "sector": "East Perimeter Fence",
        "cam_id": "CAM-02",
        "x": 420,
        "y": 280,
        "status": "ONLINE",
        "fov_angle": 90,
        "threat_level": "CRITICAL",
        "description": "Dense scrubland border perimeter fence"
    },
    {
        "bop_id": "BOP-03 (Charlie)",
        "sector": "Watchtower 04 Elevation",
        "cam_id": "CAM-03",
        "x": 680,
        "y": 210,
        "status": "ONLINE",
        "fov_angle": 120,
        "threat_level": "HIGH",
        "description": "High-altitude FLIR / Low-light sector view"
    },
    {
        "bop_id": "BOP-04 (Delta)",
        "sector": "Border Road Axis",
        "cam_id": "CAM-04",
        "x": 860,
        "y": 380,
        "status": "ONLINE",
        "fov_angle": 75,
        "threat_level": "MEDIUM",
        "description": "Tactical supply road and perimeter patrol route"
    },
]


def render_tactical_border_map(live_entities: list = None, threat_level: str = "HIGH"):
    """
    Render full tactical border defense GIS radar interface.
    """
    st.markdown("""
    <div style="background: linear-gradient(90deg, #0b1426 0%, #112240 100%); padding: 18px 24px; border-radius: 12px; border-left: 5px solid #00d4ff; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 style="margin: 0; color: #ffffff; font-family: monospace; letter-spacing: 1px;">
                    🗺️ TACTICAL BORDER GIS RADAR & SITUATIONAL COMMAND
                </h3>
                <p style="margin: 4px 0 0 0; color: #8892b0; font-size: 0.88rem;">
                    Sector: Western Border Outposts (BOP-01 to BOP-04) &nbsp;|&nbsp; Grid Reference: 31°38'N, 74°52'E &nbsp;|&nbsp; Zero-Line Fence Telemetry
                </p>
            </div>
            <div>
                <span style="background: rgba(0, 230, 118, 0.15); color: #00e676; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-family: monospace; border: 1px solid #00e676;">
                    ● SATELLITE TELEMETRY SYNCED
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Top KPI Metric Row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Monitored Sector Length", "8.4 km", "+0.6 km covered")
    with c2:
        st.metric("Active Border Posts", "4 / 4 BOPs", "100% Online")
    with c3:
        st.metric("Perimeter Breaches", "1 Confirmed", "Sector Bravo", delta_color="inverse")
    with c4:
        st.metric("Min. Distance to Zero Line", "18.4 m", "-4.2 m closing", delta_color="inverse")
    with c5:
        st.metric("QRT Response Readiness", "Ready", "3 Units on Standby")

    # 2. Build Interactive Plotly 2D Tactical Defense Map
    fig = go.Figure()

    # (A) Border Zero Line (International Border Boundary)
    ib_x = np.linspace(0, 1000, 100)
    ib_y = 50 + 25 * np.sin(ib_x / 150)
    fig.add_trace(go.Scatter(
        x=ib_x, y=ib_y,
        mode='lines',
        name='Border Zero-Line (IB)',
        line=dict(color='#ff3344', width=3, dash='dashdot'),
        hoverinfo='name'
    ))

    # (B) Restricted Buffer Zone (0m to 150m from border)
    buffer_y = ib_y + 120
    fig.add_trace(go.Scatter(
        x=np.concatenate([ib_x, ib_x[::-1]]),
        y=np.concatenate([ib_y, buffer_y[::-1]]),
        fill='toself',
        fillcolor='rgba(255, 51, 68, 0.08)',
        line=dict(color='rgba(255, 51, 68, 0.2)', width=1),
        name='High-Risk Buffer Zone (150m)',
        hoverinfo='name'
    ))

    # (C) Secondary Perimeter Fence
    fig.add_trace(go.Scatter(
        x=ib_x, y=buffer_y,
        mode='lines',
        name='Perimeter Smart Fence (CCTV Axis)',
        line=dict(color='#ffaa00', width=2, dash='dot'),
        hoverinfo='name'
    ))

    # (D) Border Out Posts (BOPs) & Watchtowers
    bop_xs = [s["x"] for s in SECTORS_DATA]
    bop_ys = [s["y"] for s in SECTORS_DATA]
    bop_texts = [f"{s['bop_id']}<br>{s['sector']}<br>Cam: {s['cam_id']}" for s in SECTORS_DATA]

    fig.add_trace(go.Scatter(
        x=bop_xs, y=bop_ys,
        mode='markers+text',
        marker=dict(symbol='square', size=16, color='#00d4ff', line=dict(color='#ffffff', width=2)),
        text=[s["bop_id"] for s in SECTORS_DATA],
        textposition="bottom center",
        name='Border Out Posts (BOPs)',
        hovertext=bop_texts,
        hoverinfo='text'
    ))

    # (E) Vision Cones / Surveillance Coverage Arcs
    for s in SECTORS_DATA:
        cx, cy = s["x"], s["y"]
        arc_x = [cx, cx - 70, cx + 70, cx]
        arc_y = [cy, cy - 90, cy - 90, cy]
        fig.add_trace(go.Scatter(
            x=arc_x, y=arc_y,
            fill='toself',
            fillcolor='rgba(0, 212, 255, 0.10)',
            line=dict(color='rgba(0, 212, 255, 0.4)', width=1),
            name=f"Coverage Arc {s['cam_id']}",
            showlegend=False,
            hoverinfo='skip'
        ))

    # (F) Active Intruder & Entity Targets (Dynamic Telemetry)
    # Default simulated tactical targets if live feed not currently streaming
    targets = [
        {"id": "TARGET-01 (UNKNOWN)", "x": 440, "y": 195, "type": "Pedestrian Intruder", "threat": 88, "dist": 18.4, "status": "INTRUSION"},
        {"id": "TARGET-02 (VEHICLE)", "x": 190, "y": 95, "type": "Suspicious Pickup", "threat": 65, "dist": 34.0, "status": "APPROACHING"},
        {"id": "TARGET-03 (UNKNOWN)", "x": 710, "y": 240, "type": "Loitering Entity", "threat": 45, "dist": 72.1, "status": "LOITERING"},
        {"id": "BSF-PATROL-04", "x": 830, "y": 360, "type": "Authorized Sentry", "threat": 0, "dist": 180.0, "status": "PATROL"},
    ]

    target_xs = [t["x"] for t in targets]
    target_ys = [t["y"] for t in targets]
    target_colors = ["#F44336" if t["threat"] >= 75 else ("#FF9800" if t["threat"] >= 35 else "#4CAF50") for t in targets]
    target_labels = [f"<b>{t['id']}</b><br>Dist: {t['dist']}m to IB<br>Threat: {t['threat']}/100" for t in targets]

    fig.add_trace(go.Scatter(
        x=target_xs, y=target_ys,
        mode='markers+text',
        marker=dict(symbol='cross', size=18, color=target_colors, line=dict(color='#ffffff', width=2)),
        text=[t["id"].split()[0] for t in targets],
        textposition="top right",
        name='Detected Tactical Targets',
        hovertext=target_labels,
        hoverinfo='text'
    ))

    fig.update_layout(
        title=dict(
            text="TACTICAL SITUATIONAL MAP — 2D BORDER SECTOR RADAR",
            font=dict(color="#ffffff", size=14, family="monospace")
        ),
        paper_bgcolor="#0b1426",
        plot_bgcolor="#0e1b30",
        margin=dict(l=20, r=20, t=40, b=20),
        height=480,
        xaxis=dict(
            title="Border Distance Axis (Meters Easting)",
            color="#8892b0",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            range=[0, 1000]
        ),
        yaxis=dict(
            title="Depth from Forward Posts (Meters)",
            color="#8892b0",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            range=[0, 450]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#ffffff", size=10)
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # 3. Live Tactical Sector Telemetry Table
    st.markdown("#### 📡 Real-Time Border Sector Status & Intrusion Telemetry")
    telemetry_data = []
    for t in targets:
        rec = "🚨 IMMEDIATE QRT DISPATCH" if t["threat"] >= 75 else ("⚠️ SENTRY INTERCEPT" if t["threat"] >= 40 else "✅ CLEAR / AUTHORIZED")
        telemetry_data.append({
            "Target ID": t["id"],
            "Classification": t["type"],
            "Sector Node": "BOP-02 East Fence" if t["threat"] >= 75 else ("BOP-01 North Gate" if "VEHICLE" in t["id"] else "BOP-03 Tower"),
            "Zero-Line Offset": f"{t['dist']} m",
            "Threat Score": f"{t['threat']} / 100",
            "Target Status": t["status"],
            "Recommended Tactical Action": rec
        })

    df_telemetry = pd.DataFrame(telemetry_data)
    st.dataframe(df_telemetry, use_container_width=True, hide_index=True)
