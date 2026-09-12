"""
IBVAP — System Status Bar Component
Displays system status, FPS, entity count, and alert count in Streamlit.
"""

import streamlit as st


def render_status_bar(
    is_active: bool = True,
    fps: float = 0.0,
    total_entities: int = 0,
    active_entities: int = 0,
    total_alerts: int = 0,
    is_night_mode: bool = False,
    video_source: str = "Unknown",
):
    """
    Render the system status bar at the top of the dashboard.
    """
    # System status indicator
    if is_active:
        st.markdown(
            '<div style="padding: 8px 16px; background: #1B5E20; border-radius: 8px; '
            'text-align: center; margin-bottom: 12px;">'
            '<span style="color: #69F0AE; font-weight: bold; font-size: 16px;">'
            '🟢 SYSTEM ACTIVE</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="padding: 8px 16px; background: #B71C1C; border-radius: 8px; '
            'text-align: center; margin-bottom: 12px;">'
            '<span style="color: #FF8A80; font-weight: bold; font-size: 16px;">'
            '🔴 SYSTEM PAUSED</span></div>',
            unsafe_allow_html=True,
        )

    # The status strip is full-width in the live console, so the most useful
    # health metrics can stay visible in one scan-friendly row.
    cols = st.columns(4)
    with cols[0]:
        st.metric("FPS", f"{fps:.1f}")
    with cols[1]:
        st.metric("Active Entities", active_entities)
    with cols[2]:
        st.metric("Total Tracked", total_entities)
    with cols[3]:
        st.metric("Alerts", total_alerts)

    # Night mode indicator
    if is_night_mode:
        st.markdown(
            '<span style="background: #FF6F00; color: white; padding: 4px 10px; '
            'border-radius: 4px; font-size: 12px;">🌙 NIGHT MODE (CLAHE Enhanced)</span>',
            unsafe_allow_html=True,
        )
