"""
IBVAP — Geo-Spatial Command Map
Simulates a border deployment map displaying camera locations and active threat pins.
"""

import streamlit as st
import pandas as pd
import numpy as np

def render_geospatial_map(recent_events: list):
    st.markdown("## 🗺️ Geo-Spatial Command")
    st.markdown("Live tactical map of camera sectors and active threats.")

    # Base coordinates (simulated Indian border region)
    base_lat = 32.2432
    base_lon = 75.5412

    # Simulate camera sectors
    sectors = [
        {"name": "Sector Alpha — Camera 01", "lat": base_lat + 0.012, "lon": base_lon - 0.018, "status": "SECURE"},
        {"name": "Sector Bravo — Camera 02", "lat": base_lat - 0.008, "lon": base_lon + 0.015, "status": "SECURE"},
        {"name": "Sector Charlie — Camera 03", "lat": base_lat + 0.022, "lon": base_lon + 0.005, "status": "SECURE"},
        {"name": "Command HQ", "lat": base_lat - 0.015, "lon": base_lon - 0.010, "status": "OPERATIONAL"},
    ]

    # Check recent events for threats to assign to sectors
    has_critical = any(e.get("threat_level") == "critical" for e in recent_events[:5])
    has_high = any(e.get("threat_level") == "high" for e in recent_events[:5])

    if has_critical:
        sectors[1]["status"] = "CRITICAL"
    elif has_high:
        sectors[1]["status"] = "HIGH ALERT"

    # Status display
    st.markdown("### 📡 Sector Status")
    cols = st.columns(4)
    for i, s in enumerate(sectors):
        status = s["status"]
        if status == "CRITICAL":
            icon = "🔴"
            color = "#F44336"
        elif status == "HIGH ALERT":
            icon = "🟠"
            color = "#FF9800"
        elif status == "OPERATIONAL":
            icon = "🔵"
            color = "#2196F3"
        else:
            icon = "🟢"
            color = "#4CAF50"
        
        cols[i].markdown(
            f'<div style="padding: 12px; background: #161b22; border-radius: 8px; '
            f'border-left: 4px solid {color}; text-align: center;">'
            f'<div style="font-size: 20px;">{icon}</div>'
            f'<div style="font-weight: 600; color: {color};">{s["name"].split("—")[0].strip()}</div>'
            f'<div style="font-size: 11px; color: #8b949e;">{status}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Use st.map for the map display (simple and reliable)
    map_df = pd.DataFrame({
        "lat": [s["lat"] for s in sectors],
        "lon": [s["lon"] for s in sectors],
    })
    
    st.map(map_df, zoom=12, use_container_width=True)
    
    st.divider()
    
    # Sector Details Table
    st.markdown("### 📋 Deployment Details")
    detail_data = []
    for s in sectors:
        detail_data.append({
            "Sector": s["name"],
            "Latitude": f"{s['lat']:.4f}",
            "Longitude": f"{s['lon']:.4f}",
            "Status": s["status"],
        })
    st.dataframe(pd.DataFrame(detail_data), use_container_width=True, hide_index=True)
