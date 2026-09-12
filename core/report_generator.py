"""
IBVAP — PDF Incident Report Generator
Generates professional PDF reports with incident summaries, charts, and snapshots.
Uses only the Python standard library + reportlab or falls back to HTML.
"""

import os
import time
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
from html import escape


def generate_html_report(
    db_path: str,
    output_path: str = "./ibvap_incident_report.html",
    zone_name: str = "Restricted Area",
) -> str:
    """
    Generate a professional HTML incident report from the event database.
    
    Args:
        db_path: Path to SQLite database
        output_path: Path for the output HTML report
        zone_name: Name of the restricted zone
    
    Returns:
        Path to the generated report
    """
    # Query all events
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM events ORDER BY last_updated DESC")
    events = [dict(row) for row in cursor.fetchall()]
    
    # Get stats
    cursor.execute("SELECT COUNT(*) FROM events")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM events WHERE threat_level = 'critical'")
    critical = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM events WHERE threat_level = 'high'")
    high = cursor.fetchone()[0]
    cursor.execute("SELECT AVG(threat_score) FROM events")
    avg_score = cursor.fetchone()[0] or 0
    cursor.execute("SELECT COUNT(DISTINCT vehicle_plate) FROM events WHERE vehicle_plate != ''")
    unique_vehicles = cursor.fetchone()[0]
    conn.close()
    
    now = datetime.now().strftime("%d %B %Y, %H:%M IST")
    safe_zone_name = escape(str(zone_name))
    
    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IBVAP Incident Report — {escape(now)}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            padding: 40px;
        }}
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 30px 40px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #69F0AE;
            font-size: 28px;
            margin-bottom: 8px;
        }}
        .header .subtitle {{
            color: #8b949e;
            font-size: 14px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }}
        .metric-card .value {{
            font-size: 36px;
            font-weight: 700;
            color: #69F0AE;
        }}
        .metric-card.critical .value {{ color: #F44336; }}
        .metric-card.high .value {{ color: #FF9800; }}
        .metric-card .label {{
            font-size: 12px;
            color: #8b949e;
            text-transform: uppercase;
            margin-top: 5px;
        }}
        .section-title {{
            font-size: 20px;
            color: #69F0AE;
            margin: 30px 0 15px;
            padding-bottom: 8px;
            border-bottom: 1px solid #30363d;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        th {{
            background: #161b22;
            color: #69F0AE;
            padding: 12px 15px;
            text-align: left;
            font-size: 12px;
            text-transform: uppercase;
            border-bottom: 2px solid #30363d;
        }}
        td {{
            padding: 10px 15px;
            border-bottom: 1px solid #21262d;
            font-size: 13px;
        }}
        tr:hover {{ background: #161b22; }}
        .badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }}
        .badge-critical {{ background: #F44336; color: white; }}
        .badge-high {{ background: #FF9800; color: white; }}
        .badge-medium {{ background: #FFC107; color: #000; }}
        .badge-low {{ background: #4CAF50; color: white; }}
        .footer {{
            margin-top: 40px;
            padding: 20px;
            text-align: center;
            color: #484f58;
            font-size: 11px;
            border-top: 1px solid #21262d;
        }}
        .confidential {{
            color: #F44336;
            font-weight: 700;
            font-size: 13px;
            text-align: center;
            margin: 20px 0;
        }}
        @media print {{
            body {{ background: white; color: #333; padding: 20px; }}
            .header {{ background: #f0f0f0; }}
            .metric-card {{ background: #f5f5f5; border-color: #ddd; }}
            th {{ background: #eee; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ IBVAP — Incident Report</h1>
        <div class="subtitle">Intelligent Border Video Analytics Platform • Generated: {now}</div>
        <div class="subtitle">Zone: {safe_zone_name} • Classification: CONFIDENTIAL</div>
    </div>
    
    <div class="confidential">⚠️ CLASSIFIED — AUTHORIZED PERSONNEL ONLY ⚠️</div>
    
    <div class="metrics">
        <div class="metric-card">
            <div class="value">{total}</div>
            <div class="label">Total Incidents</div>
        </div>
        <div class="metric-card critical">
            <div class="value">{critical}</div>
            <div class="label">Critical Threats</div>
        </div>
        <div class="metric-card high">
            <div class="value">{high}</div>
            <div class="label">High Threats</div>
        </div>
        <div class="metric-card">
            <div class="value">{avg_score:.0f}</div>
            <div class="label">Avg. Threat Score</div>
        </div>
    </div>
    
    <div class="section-title">📋 Incident Log</div>
    <table>
        <thead>
            <tr>
                <th>Time</th>
                <th>Entity</th>
                <th>Score</th>
                <th>Level</th>
                <th>Zone</th>
                <th>Speed</th>
                <th>Posture</th>
                <th>Plate</th>
            </tr>
        </thead>
        <tbody>
"""
    
    for event in events[:50]:  # Limit to 50 for readability
        ts = event.get('last_updated', '')
        try:
            ts = datetime.fromisoformat(ts).strftime("%H:%M:%S")
        except (ValueError, TypeError):
            ts = str(ts)[:8]
        
        level = str(event.get('threat_level', 'low'))
        badge_class = f"badge-{level}"
        safe_entity = escape(str(event.get('entity_id', 'N/A')))
        safe_zone = escape(str(event.get('zone_name', '')))
        safe_speed = escape(str(event.get('speed_category', '')))
        safe_posture = escape(str(event.get('posture', '')))
        safe_plate = escape(str(event.get('vehicle_plate', '') or '—'))
        
        html += f"""            <tr>
                <td>{ts}</td>
                <td>{safe_entity}</td>
                <td><strong>{event.get('threat_score', 0)}</strong></td>
                <td><span class="badge {escape(badge_class)}">{escape(level.upper())}</span></td>
                <td>{safe_zone}</td>
                <td>{safe_speed}</td>
                <td>{safe_posture}</td>
                <td>{safe_plate}</td>
            </tr>
"""
    
    html += f"""        </tbody>
    </table>
    
    <div class="section-title">📊 Summary Statistics</div>
    <div class="metrics">
        <div class="metric-card">
            <div class="value">{unique_vehicles}</div>
            <div class="label">Unique Vehicles</div>
        </div>
        <div class="metric-card">
            <div class="value">{len([e for e in events if e.get('posture') == 'crouching'])}</div>
            <div class="label">Crouching Detected</div>
        </div>
        <div class="metric-card">
            <div class="value">{len([e for e in events if e.get('speed_category') == 'running'])}</div>
            <div class="label">Running Detected</div>
        </div>
        <div class="metric-card">
            <div class="value">{len(set(e.get('entity_id', '') for e in events))}</div>
            <div class="label">Unique Subjects</div>
        </div>
    </div>
    
    <div class="footer">
        <p>Report generated by IBVAP — Intelligent Border Video Analytics Platform</p>
        <p>Powered by Deep Learning (MLP Neural Network) • YOLOv8 • EasyOCR • Streamlit</p>
        <p>© 2026 IBVAP Team — Smart India Hackathon</p>
    </div>
</body>
</html>"""
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    return output_path
