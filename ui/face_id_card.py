"""
IBVAP — Live Biometric Face & ID Match Card
Fulfills SIH Problem Statement 26187:
"Support facial recognition, vehicle identification, and behavioral analytics through software...
 without requiring dedicated FRS, ANPR, or smart-camera hardware."

Provides side-by-side Live CCTV Face Intercept vs Database ID Card Verification.
"""

import streamlit as st
import cv2
import os
import time
import base64
from typing import Optional, Dict, Any


def _bgr_to_base64(img_bgr) -> str:
    """Convert a BGR numpy image to base64 data URI for clean HTML embedding."""
    if img_bgr is None or img_bgr.size == 0:
        return ""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ret, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ret:
        return ""
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def render_face_id_match_card(placeholder, intercept_data: Optional[Dict[str, Any]] = None):
    """
    Renders the Live Biometric Face Intercept & Government/Defense ID Match Card.
    """
    if not intercept_data:
        placeholder.markdown("""
        <div style="background: rgba(15, 23, 42, 0.75); border: 1px dashed #334155; border-radius: 10px; padding: 14px; text-align: center; margin-bottom: 15px;">
            <div style="font-size: 1.2rem; margin-bottom: 4px;">🎯</div>
            <div style="font-family: monospace; font-size: 0.85rem; font-weight: 700; color: #38bdf8; letter-spacing: 0.08em;">
                LIVE FRS SENTRY ACTIVE
            </div>
            <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 2px;">
                Scanning surveillance video for faces • Real-time database matching enabled
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    frs_res = intercept_data.get("frs_result")
    face_crop = intercept_data.get("face_crop")
    timestamp = intercept_data.get("timestamp", time.strftime("%H:%M:%S IST"))
    camera_name = intercept_data.get("camera", "CAM-01 [Border Sector]")
    entity_id = intercept_data.get("entity_id", "UNKNOWN-01")

    # Encode captured face crop
    captured_b64 = _bgr_to_base64(face_crop)

    # Encode registered ID reference photo
    ref_b64 = ""
    if frs_res and getattr(frs_res, "photo_path", None) and os.path.exists(frs_res.photo_path):
        ref_img = cv2.imread(frs_res.photo_path)
        ref_b64 = _bgr_to_base64(ref_img)
    elif frs_res and frs_res.is_identified:
        # Fallback to database/faces/{label_id}.jpg
        fallback_path = os.path.join(os.path.dirname(__file__), "..", "database", "faces", f"{frs_res.label_id}.jpg")
        if os.path.exists(fallback_path):
            ref_img = cv2.imread(fallback_path)
            ref_b64 = _bgr_to_base64(ref_img)

    # Styling based on category
    category = getattr(frs_res, "category", "UNKNOWN") if frs_res else "UNKNOWN"
    name = getattr(frs_res, "name", "Unidentified Person") if frs_res else "Unidentified Person"
    id_num = getattr(frs_res, "id_card_number", "UNREGISTERED-CIVILIAN") if frs_res else "UNREGISTERED-CIVILIAN"
    role = getattr(frs_res, "role", "Unregistered Individual") if frs_res else "Unregistered Individual"
    clearance = getattr(frs_res, "clearance", "UNVERIFIED") if frs_res else "UNVERIFIED"
    conf = getattr(frs_res, "confidence_score", 0.0) if frs_res else 0.0

    if category == "SUSPECT":
        theme_border = "#ef4444"
        theme_bg = "linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(15, 23, 42, 0.95) 100%)"
        status_badge = "🚨 CRITICAL WATCHLIST MATCH"
        status_color = "#ef4444"
        directive = "⚠️ DIRECTIVE: IMMEDIATE SENTRY INTERCEPT • SECTOR ALERT BROADCAST"
    elif category == "AUTHORIZED_BSF":
        theme_border = "#10b981"
        theme_bg = "linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(15, 23, 42, 0.95) 100%)"
        status_badge = "✅ VERIFIED DEFENSE SENTRY"
        status_color = "#10b981"
        directive = "🛡️ DIRECTIVE: AUTHORIZED BORDER PERSONNEL • NORMAL PATROL"
    else:
        theme_border = "#f59e0b"
        theme_bg = "linear-gradient(135deg, rgba(245, 158, 11, 0.10) 0%, rgba(15, 23, 42, 0.95) 100%)"
        status_badge = "⚠️ UNREGISTERED INDIVIDUAL"
        status_color = "#f59e0b"
        directive = "🔍 DIRECTIVE: SENTRY VERIFICATION REQUIRED • CHECKPOINT INSPECTION"

    # Determine target and record captions
    if frs_res and getattr(frs_res, "is_identified", False):
        target_display = f"Target: {name}"
        record_display = f"{id_num}"
    else:
        target_display = f"Target: {entity_id}"
        record_display = "UNREGISTERED-CIVILIAN"

    # Image tags with balanced 125px portrait framing
    if captured_b64:
        img_captured_html = f'<img src="{captured_b64}" style="width: 100%; height: 125px; object-fit: cover; object-position: center; border-radius: 6px; border: 2px solid {theme_border};">'
    else:
        img_captured_html = '<div style="width: 100%; height: 125px; background: #1e293b; border-radius: 6px; display: flex; align-items: center; justify-content: center; color: #64748b; font-size: 0.75rem;">NO LIVE CROP</div>'

    if ref_b64:
        img_ref_html = f'<img src="{ref_b64}" style="width: 100%; height: 125px; object-fit: cover; object-position: center; border-radius: 6px; border: 2px solid {theme_border};">'
    else:
        img_ref_html = f'<div style="width: 100%; height: 125px; background: #1e293b; border-radius: 6px; display: flex; flex-direction: column; align-items: center; justify-content: center; color: {status_color}; font-size: 0.75rem; border: 1px dashed {theme_border};"><span>🪪</span><span>NO ID ON FILE</span></div>'

    card_html = f"""
    <div style="background: {theme_bg}; border: 1.5px solid {theme_border}; border-radius: 12px; padding: 14px; margin-bottom: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.4);">
        <!-- Top Status Bar -->
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 8px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.1rem;">🎯</span>
                <div>
                    <span style="font-family: monospace; font-size: 0.85rem; font-weight: 800; color: #f8fafc; letter-spacing: 0.05em;">
                        BIOMETRIC FACE INTERCEPT
                    </span>
                    <span style="font-size: 0.72rem; color: #94a3b8; margin-left: 6px;">
                        [{camera_name}]
                    </span>
                </div>
            </div>
            <span style="background: rgba(0,0,0,0.4); color: {status_color}; border: 1px solid {status_color}; padding: 3px 10px; border-radius: 14px; font-size: 0.72rem; font-weight: 800; font-family: monospace;">
                {status_badge}
            </span>
        </div>

        <!-- 2-Column Comparison Layout -->
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px;">
            <!-- Column 1: Captured Face -->
            <div style="background: rgba(15, 23, 42, 0.7); padding: 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
                <div style="font-size: 0.7rem; font-weight: 700; color: #38bdf8; font-family: monospace; margin-bottom: 6px; display: flex; justify-content: space-between;">
                    <span>📸 CCTV CAPTURE</span>
                    <span style="color: #64748b;">{timestamp}</span>
                </div>
                {img_captured_html}
                <div style="font-size: 0.68rem; color: #94a3b8; margin-top: 5px; text-align: center; font-family: monospace; font-weight: 600;">
                    {target_display}
                </div>
            </div>

            <!-- Column 2: Database Match File -->
            <div style="background: rgba(15, 23, 42, 0.7); padding: 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
                <div style="font-size: 0.7rem; font-weight: 700; color: #a78bfa; font-family: monospace; margin-bottom: 6px; display: flex; justify-content: space-between;">
                    <span>🪪 ID RECORD</span>
                    <span style="color: {status_color}; font-weight: 800;">{conf}% MATCH</span>
                </div>
                {img_ref_html}
                <div style="font-size: 0.68rem; color: #cbd5e1; margin-top: 5px; text-align: center; font-family: monospace; font-weight: 700;">
                    {record_display}
                </div>
            </div>
        </div>

        <!-- Identity Details & Verification Bar -->
        <div style="background: rgba(15, 23, 42, 0.85); border-radius: 8px; padding: 10px 12px; border: 1px solid rgba(255,255,255,0.06); font-size: 0.8rem;">
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px;">
                <span style="font-size: 0.95rem; font-weight: 800; color: #ffffff;">{name}</span>
                <span style="font-size: 0.72rem; color: {status_color}; font-weight: 700; font-family: monospace;">{clearance}</span>
            </div>
            <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 6px;">
                <b>Designation / Role:</b> {role}
            </div>
            <div style="background: rgba(0,0,0,0.3); border-radius: 4px; height: 6px; width: 100%; overflow: hidden; margin-bottom: 8px;">
                <div style="background: {status_color}; height: 100%; width: {min(100.0, max(5.0, conf))}%;"></div>
            </div>
            <div style="font-size: 0.72rem; font-weight: 700; color: {status_color}; font-family: monospace; letter-spacing: 0.02em;">
                {directive}
            </div>
        </div>
    </div>
    """

    placeholder.markdown(card_html, unsafe_allow_html=True)
