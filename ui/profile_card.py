"""
IBVAP — Entity Profile Card Component
Renders detailed entity profile with threat breakdown in Streamlit.
"""

import streamlit as st
from html import escape
from typing import Optional


def render_profile_card(profile, show_snapshot: bool = True):
    """
    Render a full entity profile card.
    
    Args:
        profile: EntityProfile object
        show_snapshot: Whether to show snapshot image
    """
    if profile is None:
        return

    # Header with entity ID and threat badge
    threat_emoji = {
        "critical": "🔴",
        "high": "🟠",
        "medium": "🟡",
        "low": "🟢",
    }
    emoji = threat_emoji.get(profile.threat_level, "⚪")

    st.markdown(f"### {emoji} {escape(str(profile.entity_id))}")

    # Threat score with color bar
    score_color = profile.threat_color
    st.markdown(
        f"""<div style="
            background: linear-gradient(90deg, {score_color} {profile.threat_score}%, #333 {profile.threat_score}%);
            height: 8px;
            border-radius: 4px;
            margin-bottom: 12px;
        "></div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"**Threat Score:** `{profile.threat_score}/100` ({profile.threat_level.upper()})"
    )

    # --- Explainable AI (XAI) Factor Attribution Breakdown ---
    if profile.threat_breakdown:
        st.markdown("##### 🧠 Explainable AI (XAI) Attribution")
        for factor_name, points in profile.threat_breakdown:
            if points > 0:
                pct = min(100, int((points / max(1, profile.threat_score)) * 100))
                st.markdown(
                    f"""<div style="margin-bottom: 6px;">
                        <div style="display: flex; justify-content: space-between; font-size: 12px; color: #e4e4e7;">
                            <span>{factor_name}</span>
                            <span style="color: #38BDF8; font-family: monospace; font-weight: bold;">+{points} pts</span>
                        </div>
                        <div style="background: rgba(255,255,255,0.08); height: 5px; border-radius: 3px; overflow: hidden; margin-top: 3px;">
                            <div style="background: linear-gradient(90deg, #0284C7, #38BDF8); width: {pct}%; height: 100%;"></div>
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )

    st.divider()

    # --- Behavior Tags ---
    if profile.behavior_tags:
        tags_html = " ".join(
            f'<span style="background: #FF5722; color: white; padding: 2px 8px; '
            f'border-radius: 12px; font-size: 12px; margin: 2px;">{escape(str(tag))}</span>'
            for tag in profile.behavior_tags
        )
        st.markdown(f"**Behaviors:** {tags_html}", unsafe_allow_html=True)

    if profile.posture and profile.posture != "standing":
        st.warning(f"⚠ Posture: **{profile.posture.upper()}**")

    if profile.loitering_duration > 0:
        st.markdown(f"**Loitering Duration:** {profile.loitering_duration:.0f}s")

    st.divider()

    # --- Physical Descriptors (persons only) ---
    if profile.class_name == "person":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Build:** {profile.estimated_build}")
            st.markdown(f"**Height:** {profile.estimated_height_relative}")
        with col2:
            st.markdown(f"**Clothing:** {profile.dominant_color_name}")
            if profile.carried_objects:
                st.markdown(f"**Objects:** {', '.join(profile.carried_objects)}")

    # --- Vehicle Info ---
    if profile.vehicle_plate:
        st.markdown(f"**🚗 License Plate:** `{profile.vehicle_plate}`")

    # --- Tracking Info ---
    st.divider()
    st.markdown(f"**First Seen:** {profile.first_seen}")
    if profile.last_position:
        st.markdown(f"**Last Position:** ({profile.last_position[0]}, {profile.last_position[1]})")
    if profile.zone_name:
        st.markdown(f"**Zone:** {profile.zone_name}")


def render_mini_profile(entity, threat_score: int = 0):
    """
    Render a compact mini-profile for the sidebar.
    
    Args:
        entity: TrackedEntity object
        threat_score: Current threat score
    """
    if threat_score >= 80:
        emoji = "🔴"
    elif threat_score >= 60:
        emoji = "🟠"
    elif threat_score >= 30:
        emoji = "🟡"
    else:
        emoji = "🟢"

    tags = ", ".join(entity.behavior_tags[:3]) if entity.behavior_tags else "monitoring"

    st.markdown(
        f"**{emoji} {entity.entity_id}** ({entity.class_name}) — "
        f"Score: {threat_score} — {tags}"
    )
