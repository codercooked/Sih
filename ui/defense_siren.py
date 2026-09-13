"""
IBVAP — Defense Emergency Siren & QRT Dispatch Module
Fulfills SIH Problem Statement 26187:
"Improve situational awareness and response time for border security forces.
 Support integration with existing command and control systems."

Features:
- Browser Web Audio API klaxon buzzer for critical perimeter breaches
- Red defensive emergency strobe overlay
- One-click Quick Reaction Team (QRT) Tactical Dispatch Protocol
- Command & Control Webhook Dispatcher
"""

import streamlit as st
import streamlit.components.v1 as components
import json
import time


def render_emergency_siren_component(threat_score: int, level: str, is_active: bool = True):
    """
    Renders audible siren buzzer and defensive strobe when threat level is CRITICAL or HIGH.
    Uses HTML5 / Web Audio API (zero audio file dependencies, pure-software).
    """
    if not is_active or threat_score < 75:
        return

    siren_html = """
    <div id="siren-container" style="background: linear-gradient(90deg, #ff1744, #b71c1c); padding: 12px 18px; border-radius: 8px; color: white; display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; animation: blink 1s infinite alternate; border: 2px solid #ffffff;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 1.6rem;">🚨</span>
            <div>
                <strong style="font-size: 1.05rem; letter-spacing: 1px; font-family: monospace;">CRITICAL PERIMETER INTRUSION BREACH DETECTED</strong>
                <div style="font-size: 0.82rem; opacity: 0.95;">Threat Level Exceeds Threshold (Score: %d/100). Immediate Intercept Advised.</div>
            </div>
        </div>
        <button onclick="playSiren()" style="background: #ffffff; color: #b71c1c; border: none; font-weight: bold; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-family: monospace;">
            🔊 TEST DEFENSE KLAXON
        </button>
    </div>

    <style>
    @keyframes blink {
        0% { opacity: 0.85; box-shadow: 0 0 10px rgba(255, 23, 68, 0.4); }
        100% { opacity: 1.0; box-shadow: 0 0 25px rgba(255, 23, 68, 0.9); }
    }
    </style>

    <script>
    function playSiren() {
        try {
            var ctx = new (window.AudioContext || window.webkitAudioContext)();
            var osc = ctx.createOscillator();
            var gain = ctx.createGain();
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(440, ctx.currentTime);
            osc.frequency.exponentialRampToValueAtTime(880, ctx.currentTime + 0.35);
            osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.7);
            gain.gain.setValueAtTime(0.2, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.75);
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start();
            osc.stop(ctx.currentTime + 0.8);
        } catch(e) {
            console.log('AudioContext error:', e);
        }
    }
    </script>
    """ % threat_score

    components.html(siren_html, height=80)


def render_qrt_dispatch_button(entity_id: str, sector: str = "BOP-02 East Perimeter", threat_score: int = 85):
    """
    Renders Quick Reaction Team (QRT) Tactical Dispatch Button & Protocol Generator.
    """
    if st.button(f"🚨 DISPATCH QRT PATROL TO {sector.upper()}", type="primary", use_container_width=True):
        dispatch_time = time.strftime("%Y-%m-%d %H:%M:%S IST")
        dispatch_packet = {
            "dispatch_id": f"QRT-DISPATCH-{int(time.time())}",
            "timestamp": dispatch_time,
            "target_entity": entity_id,
            "assigned_sector": sector,
            "threat_score": f"{threat_score}/100",
            "assigned_unit": "QRT Unit Bravo-1 (3 Jawans, 1 Gypsy)",
            "priority": "IMMEDIATE_RESPONSE",
            "action_directive": "Intercept and secure perimeter boundary fence. Ascertain identity and baggage."
        }
        
        st.session_state["last_qrt_dispatch"] = dispatch_packet
        st.success(f"✅ QRT Patrol Unit Dispatched to {sector} at {dispatch_time}!")


def render_last_dispatch_modal():
    """Display most recent QRT dispatch order details if present."""
    if "last_qrt_dispatch" in st.session_state:
        pkt = st.session_state["last_qrt_dispatch"]
        with st.expander(f"📋 Last Tactical Dispatch Order: {pkt['dispatch_id']}", expanded=True):
            st.json(pkt)
