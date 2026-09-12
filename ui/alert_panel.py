"""
IBVAP — Alert Panel Component
Renders real-time scrollable alert log in Streamlit.
"""

import streamlit as st
import json
from typing import List, Dict
from datetime import datetime


import base64
import os
from html import escape

INCIDENT_STATUSES = ("new", "acknowledged", "investigating", "confirmed", "resolved", "dismissed")

def get_base64_image(filepath):
    if not filepath or not os.path.exists(filepath):
        return ""
    try:
        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

def render_alert_panel(events: List[Dict], max_display: int = 15):
    """
    Render the real-time alert log panel.
    
    Args:
        events: List of event dictionaries from EventStore
        max_display: Maximum events to display
    """
    st.markdown("### 🚨 Live Incident Log")

    if not events:
        st.info("No active incidents. Monitoring...")
        return

    table_html = [
        '<table style="width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; background: rgba(30, 30, 30, 0.7); border-radius: 8px; overflow: hidden;">',
        '<thead><tr style="background: rgba(40, 40, 40, 0.9); color: #B0BEC5; border-bottom: 2px solid #444;">',
        '<th style="padding: 10px;">Time</th>',
        '<th style="padding: 10px;">Entity</th>',
        '<th style="padding: 10px;">Threat Score</th>',
        '<th style="padding: 10px;">Details</th>',
        '<th style="padding: 10px;">Snapshot</th>',
        '</tr></thead><tbody>'
    ]

    for event in events[:max_display]:
        threat_score = event.get("threat_score", 0)
        threat_level = event.get("threat_level", "low")
        entity_id = event.get("entity_id", "UNKNOWN")
        zone_name = event.get("zone_name", "")
        last_updated = event.get("last_updated", "")
        snapshot_path = event.get("snapshot_path", "")

        # Color-coded indicator
        if threat_level == "critical":
            icon = "🔴"
            color = "#FF3366"
        elif threat_level == "high":
            icon = "🟠"
            color = "#FF9933"
        elif threat_level == "medium":
            icon = "🟡"
            color = "#FFCC00"
        else:
            icon = "🟢"
            color = "#00E676"

        try:
            ts = datetime.fromisoformat(last_updated).strftime("%H:%M:%S")
        except (ValueError, TypeError):
            ts = str(last_updated)[:8]

        tags = event.get("behaviour_tags", [])
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except (json.JSONDecodeError, TypeError):
                tags = [tags] if tags else []

        tags_str = ", ".join([escape(str(t).replace("_", " ").title()) for t in tags]) if tags else "Zone Entry"

        posture = escape(str(event.get("posture", "")))
        plate = escape(str(event.get("vehicle_plate", "")))
        entity_id = escape(str(entity_id))
        zone_name = escape(str(zone_name))
        status = escape(str(event.get("status") or "new").upper())
        details = f"<strong>Incident: {status}</strong><br>{tags_str}"
        if plate and plate != "UNREADABLE":
            details += f"<br><span style='color: #8b949e;'>Plate: {plate}</span>"
        if posture and posture != "unknown":
            details += f"<br><span style='color: #8b949e;'>Posture: {posture.title()}</span>"

        img_html = ""
        if snapshot_path:
            b64_img = get_base64_image(snapshot_path)
            if b64_img:
                img_src = f"data:image/jpeg;base64,{b64_img}"
                img_html = f'<a href="{img_src}" target="_blank"><img src="{img_src}" style="width: 50px; height: 50px; object-fit: cover; border-radius: 4px; border: 1px solid {color}; cursor: pointer;" title="Click to view full image"></a>'

        row = (
            f'<tr style="border-bottom: 1px solid #333;">'
            f'<td style="padding: 10px; font-family: monospace; color: #a1a1aa;">{ts}</td>'
            f'<td style="padding: 10px; font-weight: bold; color: #f4f4f5;">{icon} {entity_id}</td>'
            f'<td style="padding: 10px; font-weight: bold; color: {color};">{threat_score}</td>'
            f'<td style="padding: 10px; color: #d4d4d8;">{details}</td>'
            f'<td style="padding: 10px;">{img_html}</td>'
            f'</tr>'
        )
        table_html.append(row)
        
    table_html.append('</tbody></table>')
    st.markdown("".join(table_html), unsafe_allow_html=True)


def render_alert_summary(events: List[Dict]):
    """Render a summary of alert statistics."""
    if not events:
        return

    total = len(events)
    critical = sum(1 for e in events if e.get("threat_level") == "critical")
    high = sum(1 for e in events if e.get("threat_level") == "high")

    cols = st.columns(3)
    with cols[0]:
        st.metric("Total", total)
    with cols[1]:
        st.metric("Critical", critical)
    with cols[2]:
        st.metric("High", high)


def render_incident_workflow(events: List[Dict], event_store):
    """Render operator controls for acknowledging and resolving incidents."""
    if not events:
        return

    st.markdown("### 🧭 Incident Workflow")
    counts = {status: 0 for status in INCIDENT_STATUSES}
    for event in events:
        status = event.get("status") or "new"
        counts[status] = counts.get(status, 0) + 1

    st.caption(
        f"Open: {counts['new'] + counts['acknowledged'] + counts['investigating'] + counts['confirmed']}  ·  "
        f"Resolved: {counts['resolved']}  ·  Dismissed: {counts['dismissed']}"
    )

    options = {
        f"#{event.get('id', '?')} · {event.get('entity_id', 'UNKNOWN')} · "
        f"score {event.get('threat_score', 0)} · {(event.get('status') or 'new').upper()}": event
        for event in events
    }
    selected_label = st.selectbox("Select incident", list(options), key="incident_selector")
    selected = options[selected_label]
    event_id = selected.get("id")
    current_status = selected.get("status") or "new"
    current_index = INCIDENT_STATUSES.index(current_status) if current_status in INCIDENT_STATUSES else 0

    col1, col2 = st.columns([1, 2])
    with col1:
        status = st.selectbox(
            "Status",
            INCIDENT_STATUSES,
            index=current_index,
            key=f"incident_status_{event_id}",
        )
    with col2:
        notes = st.text_input(
            "Operator note",
            value=selected.get("operator_notes") or "",
            key=f"incident_notes_{event_id}",
        )

    if st.button("Save incident update", key=f"incident_save_{event_id}", type="primary"):
        if event_store.update_incident(event_id, status, notes):
            st.success(f"Incident #{event_id} updated to {status}.")
            st.rerun()
