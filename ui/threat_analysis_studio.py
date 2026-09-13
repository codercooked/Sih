"""
IBVAP — Tactical Threat Analysis & Forensics Studio
Provides interactive simulation, counterfactual 'what-if' testing,
multi-vector radar risk profiling, Explainable AI (XAI) waterfall decomposition,
dynamic 2D threat sensitivity surfaces, and historical incident forensics.
"""

import math
import os
import sqlite3
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from core.threat_scorer import ThreatScorer, ThreatAssessment

# Tactical Preset Scenarios for Rapid Evaluation
PRESET_SCENARIOS = {
    "🥷 Covert Night Infiltration": {
        "description": "Crouched stealth crawl under cover of darkness (02:30 AM) inside restricted sector.",
        "entity_type": 0,
        "is_in_zone": True,
        "loitering_duration": 140.0,
        "persons_in_zone": 1,
        "speed_category": "walking",
        "is_moving_toward_zone": False,
        "is_night_mode": True,
        "posture": "crouching",
        "is_erratic": False,
        "hour_of_day": 2,
        "direction_toward_zone": 0.2,
        "crowd_density_gradient": 0.0,
        "has_weapon": False,
        "distance_to_boundary": 0.0,
        "acceleration_magnitude": 0.4,
    },
    "🧗 Perimeter Fence Scaling": {
        "description": "High-urgency breach: subject scaling outer boundary mesh with rapid vertical acceleration.",
        "entity_type": 0,
        "is_in_zone": False,
        "loitering_duration": 35.0,
        "persons_in_zone": 0,
        "speed_category": "running",
        "is_moving_toward_zone": True,
        "is_night_mode": True,
        "posture": "climbing",
        "is_erratic": True,
        "hour_of_day": 23,
        "direction_toward_zone": 0.95,
        "crowd_density_gradient": 0.0,
        "has_weapon": False,
        "distance_to_boundary": 0.05,
        "acceleration_magnitude": 3.2,
    },
    "🚗 High-Speed Vehicle Ramming": {
        "description": "Unregistered vehicle accelerating rapidly toward border checkpoint barrier at night.",
        "entity_type": 1,
        "is_in_zone": False,
        "loitering_duration": 10.0,
        "persons_in_zone": 0,
        "speed_category": "running",
        "is_moving_toward_zone": True,
        "is_night_mode": True,
        "posture": "unknown",
        "is_erratic": True,
        "hour_of_day": 1,
        "direction_toward_zone": 1.0,
        "crowd_density_gradient": 0.0,
        "has_weapon": False,
        "has_readable_plate": 0,
        "is_unauthorized_plate": 1,
        "distance_to_boundary": 0.15,
        "acceleration_magnitude": 4.1,
    },
    "⚠️ Armed Perimeter Incursion": {
        "description": "Subject carrying visible weapon moving deliberately toward critical tactical zone.",
        "entity_type": 0,
        "is_in_zone": True,
        "loitering_duration": 65.0,
        "persons_in_zone": 1,
        "speed_category": "walking",
        "is_moving_toward_zone": False,
        "is_night_mode": False,
        "posture": "standing",
        "is_erratic": False,
        "hour_of_day": 14,
        "direction_toward_zone": 0.7,
        "crowd_density_gradient": 0.0,
        "has_weapon": True,
        "distance_to_boundary": 0.0,
        "acceleration_magnitude": 0.8,
    },
    "👥 Coordinated Crowd Decoy Infiltration": {
        "description": "High-density crowd disturbance near perimeter fence with multiple individuals in sector.",
        "entity_type": 0,
        "is_in_zone": True,
        "loitering_duration": 85.0,
        "persons_in_zone": 8,
        "speed_category": "walking",
        "is_moving_toward_zone": True,
        "is_night_mode": False,
        "posture": "standing",
        "is_erratic": True,
        "hour_of_day": 17,
        "direction_toward_zone": 0.8,
        "crowd_density_gradient": 4.2,
        "has_weapon": False,
        "distance_to_boundary": 0.0,
        "acceleration_magnitude": 1.6,
    },
    "🏳️ Surrendering Defector / Infiltrator": {
        "description": "Subject hands raised in surrender posture, stationary inside boundary buffer zone.",
        "entity_type": 0,
        "is_in_zone": True,
        "loitering_duration": 45.0,
        "persons_in_zone": 1,
        "speed_category": "stationary",
        "is_moving_toward_zone": False,
        "is_night_mode": False,
        "posture": "surrendering",
        "is_erratic": False,
        "hour_of_day": 10,
        "direction_toward_zone": 0.0,
        "crowd_density_gradient": 0.0,
        "has_weapon": False,
        "distance_to_boundary": 0.0,
        "acceleration_magnitude": 0.0,
    },
    "🌾 Benign Border Civilian / Farmer": {
        "description": "Local farmer walking parallel to outer boundary fence during daylight, fully benign.",
        "entity_type": 0,
        "is_in_zone": False,
        "loitering_duration": 15.0,
        "persons_in_zone": 0,
        "speed_category": "walking",
        "is_moving_toward_zone": False,
        "is_night_mode": False,
        "posture": "standing",
        "is_erratic": False,
        "hour_of_day": 11,
        "direction_toward_zone": 0.05,
        "crowd_density_gradient": 0.0,
        "has_weapon": False,
        "distance_to_boundary": 0.85,
        "acceleration_magnitude": 0.2,
    },
}


def render_threat_radar_chart(vector_dict: dict, title: str = "Tactical Threat Radar Profile"):
    """Renders an aesthetic military radar (spider) chart on dark canvas."""
    categories = list(vector_dict.keys())
    values = list(vector_dict.values())
    num_vars = len(categories)

    # Compute angle for each category
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # Close polygon
    values_closed = values + [values[0]]
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True), facecolor='#0e1117')
    ax.set_facecolor('#131722')

    # Color scaling based on max value
    max_val = max(values)
    accent_color = '#ef4444' if max_val >= 75 else ('#f59e0b' if max_val >= 45 else '#10b981')

    # Draw polygon and fill
    ax.plot(angles_closed, values_closed, color=accent_color, linewidth=2.5, linestyle='solid')
    ax.fill(angles_closed, values_closed, color=accent_color, alpha=0.35)

    # Fix axis to 0-100 scale
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25', '50', '75', '100'], color='#64748b', size=8)
    ax.grid(color='#334155', linestyle='--', linewidth=0.7)

    # Set labels
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, color='#e2e8f0', size=9, weight='bold')
    ax.spines['polar'].set_color('#475569')

    plt.title(title, color='#f8fafc', size=11, weight='bold', pad=15)
    plt.tight_layout()
    return fig


def calculate_tactical_vectors(params: dict, score: int) -> dict:
    """Derives 6 core tactical threat dimensions on a 0-100 normalized scale."""
    # 1. Penetration Depth
    if params["is_in_zone"]:
        pen = 85 + min(15, (params["loitering_duration"] / 60.0) * 8)
    else:
        dist = params.get("distance_to_boundary", 1.0)
        pen = max(0, (1.0 - dist) * 70)

    # 2. Kinematic Hostility
    posture = params.get("posture", "standing")
    posture_map = {"climbing": 95, "crouching": 75, "fallen": 40, "standing": 15, "surrendering": 5, "unknown": 30}
    kinematic = posture_map.get(posture, 20)
    if params.get("is_erratic", False):
        kinematic = min(100, kinematic + 15)

    # 3. Kinetic Energy / Velocity
    speed = params.get("speed_category", "stationary")
    speed_map = {"running": 75, "walking": 35, "stationary": 10}
    accel = params.get("acceleration_magnitude", 0.0)
    velocity = min(100, speed_map.get(speed, 10) + (accel * 7.5))

    # 4. Temporal / Diurnal Risk
    hour = params.get("hour_of_day", 12)
    diurnal_curve = 0.5 * (1.0 + math.sin((hour - 3) * math.pi / 12 - math.pi / 2))
    temporal = diurnal_curve * 70 + (30 if params.get("is_night_mode", False) else 0)

    # 5. Ballistic / Lethal Threat
    ballistic = 95 if params.get("has_weapon", False) else 10

    # 6. Concealment & Evasion
    concealment = 10
    if params.get("entity_type", 0) == 1:
        if params.get("is_unauthorized_plate", 0) == 1:
            concealment = 98
        elif params.get("has_readable_plate", 1) == 0:
            concealment = 70
    else:
        if params.get("is_night_mode", False) and posture in ("crouching", "climbing"):
            concealment = 85
        elif params.get("is_erratic", False):
            concealment = 60

    return {
        "Penetration": round(min(100, pen), 1),
        "Kinematics": round(min(100, kinematic), 1),
        "Velocity": round(min(100, velocity), 1),
        "Temporal Risk": round(min(100, temporal), 1),
        "Ballistic": round(min(100, ballistic), 1),
        "Concealment": round(min(100, concealment), 1),
    }


def render_sop_recommendation(score: int, params: dict):
    """Generates military Rules of Engagement (ROE) and Standard Operating Procedure (SOP)."""
    if score >= 80:
        badge = "🔴 DEFCON 1 — CRITICAL LETHAL BREACH"
        border = "#ef4444"
        sop_steps = [
            "🚨 Immediate QRT (Quick Reaction Team) Section dispatch to sector coordinates.",
            "💡 Activate high-intensity perimeter spotlights and directional acoustic disruptor.",
            "🛰️ Dispatch autonomous tactical drone for aerial target lock and continuous optical tracking.",
            "🔒 Initiate perimeter gate automatic electromagnetic lockdown.",
            "📞 Transmit real-time encrypted telemetry packet to Command Operations Center (COC).",
        ]
    elif score >= 60:
        badge = "🟠 DEFCON 2 — HIGH TACTICAL THREAT"
        border = "#f97316"
        sop_steps = [
            "📢 Sound localized automated perimeter siren & audible verbal hail: 'Halt, Restricted Zone'.",
            "📹 Lock PTZ thermal/optical surveillance cameras on subject tracking bounding box.",
            "🚓 Dispatch perimeter mobile patrol vehicle to outer fence intercept point.",
            "⚠️ Elevate video stream telemetry priority to 60 FPS recording with biometrics.",
        ]
    elif score >= 30:
        badge = "🟡 DEFCON 3 — ELEVATED SUSPICION"
        border = "#eab308"
        sop_steps = [
            "👁️ Maintain active multi-object bounding box lock and loitering duration clock.",
            "📐 Track predictive trajectory extrapolation vector to identify target penetration point.",
            "📝 Log entity footprint and kinematics into local SQLite incident database.",
        ]
    else:
        badge = "🟢 DEFCON 4 — ROUTINE / LOW THREAT"
        border = "#22c55e"
        sop_steps = [
            "✅ Standard background observation mode.",
            "📊 Normal automated telemetry logging without operator alert trigger.",
        ]

    st.markdown(
        f"""
        <div style="
            border: 1px solid {border};
            background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(30,41,59,0.7));
            border-radius: 10px;
            padding: 16px;
            margin-top: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        ">
            <div style="font-size: 14px; font-weight: 800; color: {border}; margin-bottom: 8px;">
                {badge}
            </div>
            <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8; margin-bottom: 8px;">
                Mandated Rules of Engagement (ROE) & Response Directives:
            </div>
            <ul style="margin: 0; padding-left: 20px; color: #cbd5e1; font-size: 13px; line-height: 1.6;">
                {''.join(f'<li>{s}</li>' for s in sop_steps)}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_threat_analysis_studio(threat_scorer: ThreatScorer, db_path: str = "./ibvap_events.db"):
    """Primary entry point for the Threat Analysis & Forensics Studio tab."""
    st.markdown("## 🎯 Tactical Threat Analysis & Forensics Lab")
    st.caption("Military-Grade Threat Modeling, Counterfactual Simulation, Real-Time XAI Decomposition & Spatial Sensitivity")

    tab_sim, tab_compare, tab_surface, tab_history = st.tabs([
        "🎮 Threat Simulator (What-If)",
        "⚖️ Dual Scenario Comparison",
        "🗺️ 2D Sensitivity Surface",
        "🗄️ Historical Threat Forensics",
    ])

    # ──────────────────────────────────────────────────────────────────────────
    # SUB-TAB 1: Interactive Threat Simulator & What-If Sandbox
    # ──────────────────────────────────────────────────────────────────────────
    with tab_sim:
        st.markdown("### 🔬 Interactive Threat Simulation & Counterfactual Sandbox")
        st.markdown(
            "Test hypothetical scenarios against the **3-Way Stacking Championship Ensemble** "
            "(`HistGradientBoosting` + `RandomForest` + `Deep MLP`). Adjust any tactical variable to view instantaneous model consensus and XAI feature impact."
        )

        # Preset Loader
        st.markdown("##### ⚡ Quick-Load Tactical Threat Archetypes:")
        preset_cols = st.columns(len(PRESET_SCENARIOS))
        selected_preset = None
        for i, (name, cfg) in enumerate(PRESET_SCENARIOS.items()):
            short_name = name.split(" ")[1] if len(name.split(" ")) > 1 else name
            if preset_cols[i].button(name.split(" ")[0] + " " + short_name, key=f"btn_pre_{i}", help=cfg["description"], use_container_width=True):
                st.session_state["sim_loaded_preset"] = cfg
                st.rerun()

        loaded_cfg = st.session_state.get("sim_loaded_preset", PRESET_SCENARIOS["🥷 Covert Night Infiltration"])

        st.markdown("---")
        c_left, c_right = st.columns([1, 1])

        with c_left:
            st.markdown("#### 👤 1. Entity & Biometric Kinematics")
            entity_type_str = st.radio(
                "Entity Classification",
                ["Person (Infantry/Intruder)", "Vehicle (Transport/Automobile)"],
                index=0 if loaded_cfg.get("entity_type", 0) == 0 else 1,
                horizontal=True,
            )
            entity_type = 0 if "Person" in entity_type_str else 1

            if entity_type == 0:
                posture_opts = ["standing", "crouching", "climbing", "surrendering", "fallen"]
                posture_idx = posture_opts.index(loaded_cfg.get("posture", "standing")) if loaded_cfg.get("posture") in posture_opts else 0
                posture = st.selectbox("Biometric Posture (MediaPipe 3D)", posture_opts, index=posture_idx)
                has_weapon = st.toggle("🔫 Weapon / Lethal Threat Object Detected", value=bool(loaded_cfg.get("has_weapon", False)))
                has_readable_plate = 0
                is_unauthorized_plate = 0
            else:
                posture = "unknown"
                has_weapon = False
                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    has_readable_plate = 1 if st.toggle("Plate Readable (ANPR)", value=bool(loaded_cfg.get("has_readable_plate", 1))) else 0
                with col_v2:
                    is_unauthorized_plate = 1 if st.toggle("🚨 Stolen / Watchlist Match", value=bool(loaded_cfg.get("is_unauthorized_plate", 0))) else 0

            speed_opts = ["stationary", "walking", "running"]
            speed_idx = speed_opts.index(loaded_cfg.get("speed_category", "stationary")) if loaded_cfg.get("speed_category") in speed_opts else 0
            speed_cat = st.select_slider("Locomotion Velocity Category", options=speed_opts, value=speed_opts[speed_idx])
            accel_mag = st.slider("Kinetic Acceleration Magnitude (m/s²)", min_value=0.0, max_value=5.0, value=float(loaded_cfg.get("acceleration_magnitude", 0.5)), step=0.1)
            is_erratic = st.toggle("🔀 Erratic / Zig-Zag Evasive Maneuver", value=bool(loaded_cfg.get("is_erratic", False)))

        with c_right:
            st.markdown("#### 🌐 2. Spatial Perimeter & Environment")
            is_in_zone = st.toggle("🚨 Inside Restricted Security Zone", value=bool(loaded_cfg.get("is_in_zone", True)))
            
            c_s1, c_s2 = st.columns(2)
            with c_s1:
                loitering_dur = st.slider("Loitering Dwell Time (seconds)", min_value=0, max_value=300, value=int(loaded_cfg.get("loitering_duration", 60)), step=5)
            with c_s2:
                dist_boundary = st.slider("Distance to Outer Perimeter (0 = At Fence)", min_value=0.0, max_value=1.0, value=float(loaded_cfg.get("distance_to_boundary", 0.0)), step=0.05, help="0.0 represents zero boundary buffer (in contact with fence)")

            c_s3, c_s4 = st.columns(2)
            with c_s3:
                hour_of_day = st.slider("Hour of Day (24-Hour Diurnal Clock)", min_value=0, max_value=23, value=int(loaded_cfg.get("hour_of_day", 2)), format="%02d:00")
            with c_s4:
                is_night_mode = st.toggle("🌙 Cover of Darkness / Night Mode", value=bool(loaded_cfg.get("is_night_mode", True)))

            c_s5, c_s6 = st.columns(2)
            with c_s5:
                persons_in_zone = st.slider("Persons in Sector (Crowd/Group)", min_value=0, max_value=15, value=int(loaded_cfg.get("persons_in_zone", 1)))
            with c_s6:
                crowd_density = st.slider("Crowd Density Gradient", min_value=-2.0, max_value=5.0, value=float(loaded_cfg.get("crowd_density_gradient", 0.0)), step=0.5)

            is_moving_toward = not is_in_zone and (dist_boundary < 0.6 or speed_cat != "stationary")
            dir_toward = 0.8 if is_moving_toward else 0.1

        # Package parameters
        sim_params = {
            "entity_type": entity_type,
            "is_in_zone": is_in_zone,
            "loitering_duration": float(loitering_dur),
            "persons_in_zone": int(persons_in_zone),
            "speed_category": speed_cat,
            "is_moving_toward_zone": is_moving_toward,
            "is_night_mode": is_night_mode,
            "posture": posture,
            "is_erratic": is_erratic,
            "hour_of_day": int(hour_of_day),
            "direction_toward_zone": dir_toward,
            "crowd_density_gradient": float(crowd_density),
            "has_weapon": has_weapon,
            "time_since_last": 10.0,
            "has_readable_plate": has_readable_plate,
            "is_unauthorized_plate": is_unauthorized_plate,
            "distance_to_boundary": float(dist_boundary),
            "acceleration_magnitude": float(accel_mag),
        }

        # Run model prediction
        assessment: ThreatAssessment = threat_scorer.calculate(**sim_params)

        st.markdown("---")
        st.markdown("### 📊 Real-Time Threat Score & Multi-Model Consensus")

        # Top Metric Banner
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Threat Index", f"{assessment.score}/100", delta=assessment.level.upper())
        m2.metric("Threat Classification", assessment.level.upper())
        m3.metric("Ensemble Confidence", f"{assessment.confidence * 100:.1f}%")

        # Calculate sub-model predictions
        hgb_val, rf_val, mlp_val = 0, 0, 0
        if hasattr(threat_scorer, "ml_model") and threat_scorer.ml_model:
            SPEED_MAP = {"stationary": 0, "walking": 1, "running": 2}
            POSTURE_MAP = {"standing": 0, "crouching": 1, "climbing": 2, "surrendering": 3, "fallen": 4, "unknown": 0}
            raw_v = [
                sim_params["entity_type"], int(sim_params["is_in_zone"]), sim_params["loitering_duration"],
                sim_params["persons_in_zone"], SPEED_MAP.get(sim_params["speed_category"], 0),
                int(sim_params["is_moving_toward_zone"]), int(sim_params["is_night_mode"]),
                POSTURE_MAP.get(sim_params["posture"], 0), int(sim_params["is_erratic"]),
                sim_params["hour_of_day"], sim_params["direction_toward_zone"], sim_params["crowd_density_gradient"],
                int(sim_params["has_weapon"]), sim_params["time_since_last"], sim_params["has_readable_plate"],
                sim_params["is_unauthorized_plate"], sim_params["distance_to_boundary"], sim_params["acceleration_magnitude"]
            ]
            try:
                hgb_val = round(float(threat_scorer.ml_model.hgb.predict([raw_v])[0]), 1)
                rf_val = round(float(threat_scorer.ml_model.rf.predict([raw_v])[0]), 1)
                if threat_scorer.ml_model.scaler:
                    scaled_v = threat_scorer.ml_model.scaler.transform([raw_v])
                    mlp_val = round(float(threat_scorer.ml_model.mlp.predict(scaled_v)[0]), 1)
            except Exception:
                pass

        m4.metric("Model Architecture", "Stacking Meta-Ensemble", "3 Deep Estimators")

        # Sub-model consensus progress indicators
        st.markdown("###### 🤖 Sub-Model Voting Consensus Breakdown:")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.caption(f"⚡ **HistGradientBoosting (Weight: 45%)**: `{max(0, min(100, hgb_val))}` pts")
            st.progress(max(0.0, min(1.0, hgb_val / 100.0)))
        with col_m2:
            st.caption(f"🌲 **Random Forest 150 Trees (Weight: 35%)**: `{max(0, min(100, rf_val))}` pts")
            st.progress(max(0.0, min(1.0, rf_val / 100.0)))
        with col_m3:
            st.caption(f"🧠 **Deep Neural Net MLP (Weight: 20%)**: `{max(0, min(100, mlp_val))}` pts")
            st.progress(max(0.0, min(1.0, mlp_val / 100.0)))

        # Radar Profile & XAI Decomposition
        col_radar, col_xai = st.columns([1, 1])

        with col_radar:
            st.markdown("#### 🕸️ Multi-Vector Threat Radar Profile")
            vectors = calculate_tactical_vectors(sim_params, assessment.score)
            radar_fig = render_threat_radar_chart(vectors, title="6-Axis Tactical Defense Vector Assessment")
            st.pyplot(radar_fig, use_container_width=True)

        with col_xai:
            st.markdown("#### 🔍 Explainable AI (XAI) Point Attribution")
            st.caption("Decomposition of points awarded/deducted by the trained ensemble")

            if assessment.xai_factors:
                xai_df = pd.DataFrame(assessment.xai_factors)
                # Plot horizontal bar chart
                fig_bar, ax_bar = plt.subplots(figsize=(5, 3.8), facecolor='#0e1117')
                ax_bar.set_facecolor('#131722')

                y_pos = np.arange(len(xai_df))
                bars = ax_bar.barh(y_pos, xai_df["points"], color='#38bdf8', alpha=0.85, edgecolor='#0284c7', height=0.55)
                ax_bar.set_yticks(y_pos)
                ax_bar.set_yticklabels(xai_df["label"], color='#e2e8f0', fontsize=8)
                ax_bar.invert_yaxis()  # top-down
                ax_bar.set_xlabel("Points Added to Threat Index", color='#94a3b8', fontsize=8)
                ax_bar.tick_params(axis='x', colors='#94a3b8')
                ax_bar.grid(axis='x', color='#334155', linestyle='--', alpha=0.5)
                for bar in bars:
                    w = bar.get_width()
                    ax_bar.text(w + 0.8, bar.get_y() + bar.get_height()/2, f"+{int(w)} pts", va='center', color='#f8fafc', fontsize=8, weight='bold')

                plt.tight_layout()
                st.pyplot(fig_bar, use_container_width=True)
            else:
                st.info("ℹ️ Baseline normal behavior. No anomalous risk points added by Explainable AI.")

        # ROE and SOP Directives
        render_sop_recommendation(assessment.score, sim_params)

    # ──────────────────────────────────────────────────────────────────────────
    # SUB-TAB 2: Dual Scenario Comparison (Scenario A vs Scenario B)
    # ──────────────────────────────────────────────────────────────────────────
    with tab_compare:
        st.markdown("### ⚖️ Side-by-Side Tactical Scenario Differential Audit")
        st.caption("Compare two security situations side-by-side to understand why one escalates into a lethal alert while the other remains benign.")

        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("#### 🔵 Scenario A (Baseline)")
            pre_a = st.selectbox("Select Archetype A", list(PRESET_SCENARIOS.keys()), index=0, key="sel_arch_a")
            cfg_a = dict(PRESET_SCENARIOS[pre_a])
            weapon_a = st.toggle("Armed Threat (A)", value=cfg_a["has_weapon"], key="tog_w_a")
            cfg_a["has_weapon"] = weapon_a
            night_a = st.toggle("Night Context (A)", value=cfg_a["is_night_mode"], key="tog_n_a")
            cfg_a["is_night_mode"] = night_a
            loiter_a = st.slider("Loitering Time A (sec)", 0, 300, int(cfg_a["loitering_duration"]), key="sl_l_a")
            cfg_a["loitering_duration"] = loiter_a

            res_a = threat_scorer.calculate(**cfg_a)
            st.metric("Scenario A Threat Score", f"{res_a.score}/100", delta=res_a.level.upper())

        with sc2:
            st.markdown("#### 🔴 Scenario B (Comparison / Counterfactual)")
            pre_b = st.selectbox("Select Archetype B", list(PRESET_SCENARIOS.keys()), index=1, key="sel_arch_b")
            cfg_b = dict(PRESET_SCENARIOS[pre_b])
            weapon_b = st.toggle("Armed Threat (B)", value=cfg_b["has_weapon"], key="tog_w_b")
            cfg_b["has_weapon"] = weapon_b
            night_b = st.toggle("Night Context (B)", value=cfg_b["is_night_mode"], key="tog_n_b")
            cfg_b["is_night_mode"] = night_b
            loiter_b = st.slider("Loitering Time B (sec)", 0, 300, int(cfg_b["loitering_duration"]), key="sl_l_b")
            cfg_b["loitering_duration"] = loiter_b

            res_b = threat_scorer.calculate(**cfg_b)
            st.metric("Scenario B Threat Score", f"{res_b.score}/100", delta=res_b.level.upper())

        # Differential Delta Analysis
        delta_score = res_b.score - res_a.score
        st.markdown("---")
        st.markdown("#### 📈 Threat Differential Analysis (Scenario B minus Scenario A)")
        d_col1, d_col2 = st.columns([1, 2])
        with d_col1:
            st.metric("Net Threat Differential", f"{delta_score:+d} pts", help="Positive value indicates Scenario B is more dangerous")
            if delta_score > 25:
                st.error("⚠️ Scenario B represents a high-risk escalation requiring immediate military escalation.")
            elif delta_score < -25:
                st.success("✅ Scenario B significantly de-escalates perimeter risk.")
            else:
                st.info("ℹ️ Scenarios have comparable tactical risk severity.")

        with d_col2:
            vec_a = calculate_tactical_vectors(cfg_a, res_a.score)
            vec_b = calculate_tactical_vectors(cfg_b, res_b.score)
            comp_data = []
            for k in vec_a.keys():
                comp_data.append({
                    "Tactical Dimension": k,
                    "Scenario A": f"{vec_a[k]:.1f}",
                    "Scenario B": f"{vec_b[k]:.1f}",
                    "Variance": f"{vec_b[k] - vec_a[k]:+.1f}",
                })
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    # ──────────────────────────────────────────────────────────────────────────
    # SUB-TAB 3: 2D Threat Surface Sensitivity Matrix
    # ──────────────────────────────────────────────────────────────────────────
    with tab_surface:
        st.markdown("### 🗺️ Dynamic 2D Threat Surface Sensitivity Matrix")
        st.caption("Visualizes non-linear threat score transition boundaries across critical continuous parameters.")

        m_type = st.radio(
            "Select Multi-Dimensional Surface Plane:",
            [
                "Distance to Boundary vs. Loitering Dwell Time",
                "Kinetic Velocity (Speed) vs. Time of Day (Diurnal Curve)",
                "Crowd Density Gradient vs. Loitering Dwell Time",
            ],
            horizontal=True,
        )

        with st.spinner("Generating 2D Sensitivity Surface with Ensemble Predictor..."):
            grid_size = 12
            fig_surf, ax_surf = plt.subplots(figsize=(7, 4.5), facecolor='#0e1117')
            ax_surf.set_facecolor('#131722')

            if "Distance to Boundary" in m_type:
                x_vals = np.linspace(0.0, 1.0, grid_size) # distance 0 to 1
                y_vals = np.linspace(0, 180, grid_size)   # loitering seconds
                Z = np.zeros((grid_size, grid_size))
                for i, y in enumerate(y_vals):
                    for j, x in enumerate(x_vals):
                        eval_p = dict(PRESET_SCENARIOS["🥷 Covert Night Infiltration"])
                        eval_p["distance_to_boundary"] = float(x)
                        eval_p["is_in_zone"] = True if x < 0.2 else False
                        eval_p["loitering_duration"] = float(y)
                        Z[i, j] = threat_scorer.calculate(**eval_p).score

                im = ax_surf.imshow(Z, origin='lower', extent=[0.0, 1.0, 0, 180], cmap='turbo', aspect='auto')
                ax_surf.set_xlabel("Distance to Perimeter (0.0 = Zero Buffer / Direct Contact)", color='#94a3b8')
                ax_surf.set_ylabel("Loitering Dwell Time (Seconds)", color='#94a3b8')
                ax_surf.set_title("Threat Response Matrix: Distance vs Loiter Time", color='#f8fafc', weight='bold')

            elif "Kinetic Velocity" in m_type:
                x_vals = np.linspace(0, 23, grid_size) # hours
                y_vals = np.linspace(0.0, 4.5, grid_size) # acceleration m/s2
                Z = np.zeros((grid_size, grid_size))
                for i, y in enumerate(y_vals):
                    for j, x in enumerate(x_vals):
                        eval_p = dict(PRESET_SCENARIOS["🧗 Perimeter Fence Scaling"])
                        eval_p["hour_of_day"] = int(x)
                        eval_p["is_night_mode"] = True if (x < 6 or x > 20) else False
                        eval_p["acceleration_magnitude"] = float(y)
                        Z[i, j] = threat_scorer.calculate(**eval_p).score

                im = ax_surf.imshow(Z, origin='lower', extent=[0, 23, 0.0, 4.5], cmap='turbo', aspect='auto')
                ax_surf.set_xlabel("Hour of Day (24-Hour Diurnal Cycle)", color='#94a3b8')
                ax_surf.set_ylabel("Acceleration Magnitude (m/s²)", color='#94a3b8')
                ax_surf.set_title("Threat Matrix: Diurnal Hour vs Kinetic Acceleration", color='#f8fafc', weight='bold')

            else:
                x_vals = np.linspace(-2.0, 5.0, grid_size) # crowd gradient
                y_vals = np.linspace(0, 180, grid_size)    # loitering seconds
                Z = np.zeros((grid_size, grid_size))
                for i, y in enumerate(y_vals):
                    for j, x in enumerate(x_vals):
                        eval_p = dict(PRESET_SCENARIOS["👥 Coordinated Crowd Decoy Infiltration"])
                        eval_p["crowd_density_gradient"] = float(x)
                        eval_p["loitering_duration"] = float(y)
                        Z[i, j] = threat_scorer.calculate(**eval_p).score

                im = ax_surf.imshow(Z, origin='lower', extent=[-2.0, 5.0, 0, 180], cmap='turbo', aspect='auto')
                ax_surf.set_xlabel("Crowd Density Gradient", color='#94a3b8')
                ax_surf.set_ylabel("Loitering Dwell Time (Seconds)", color='#94a3b8')
                ax_surf.set_title("Threat Matrix: Crowd Density vs Loiter Dwell", color='#f8fafc', weight='bold')

            cbar = plt.colorbar(im, ax=ax_surf)
            cbar.set_label('Calculated Threat Index (0–100)', color='#94a3b8')
            cbar.ax.yaxis.set_tick_params(color='#94a3b8')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#cbd5e1')
            ax_surf.tick_params(colors='#94a3b8')
            plt.tight_layout()
            st.pyplot(fig_surf, use_container_width=True)

        st.info("💡 **Operational Insight:** The gradient transition boundary from yellow (30–60) to red (80+) reveals the exact parameter tipping point where automated lethal defense protocols engage.")

    # ──────────────────────────────────────────────────────────────────────────
    # SUB-TAB 4: Historical Threat Forensics
    # ──────────────────────────────────────────────────────────────────────────
    with tab_history:
        st.markdown("### 🗄️ Historical Incident Threat Forensics")
        st.caption("Reconstruct, audit, and re-score previous incident telemetry records from SQLite event store.")

        try:
            conn = sqlite3.connect(db_path)
            events_df = pd.read_sql_query("SELECT * FROM events ORDER BY last_updated DESC LIMIT 100", conn)
            conn.close()
        except Exception as e:
            events_df = pd.DataFrame()

        if events_df.empty:
            st.info("📡 No recorded events found in database. Run video detection on live feeds to populate historical events.")
        else:
            col_sel, col_det = st.columns([1, 2])
            with col_sel:
                selected_event_id = st.selectbox(
                    "Select Historical Incident to Audit:",
                    events_df["entity_id"].unique(),
                    index=0,
                )
                chosen_row = events_df[events_df["entity_id"] == selected_event_id].iloc[0]

                st.markdown(f"**Timestamp:** `{chosen_row.get('last_updated', 'N/A')}`")
                st.markdown(f"**Threat Score:** `{chosen_row.get('threat_score', 0)}/100`")
                st.markdown(f"**Threat Level:** `{chosen_row.get('threat_level', 'N/A').upper()}`")
                st.markdown(f"**Zone:** `{chosen_row.get('zone_name', 'N/A')}`")
                st.markdown(f"**Posture:** `{chosen_row.get('posture', 'standing')}`")

                snap_path = chosen_row.get("snapshot_path", "")
                if snap_path and os.path.exists(snap_path):
                    st.image(snap_path, caption=f"Event Snapshot ({selected_event_id})", use_column_width=True)

            with col_det:
                st.markdown("#### 🔬 Forensic Vector Deconstruction for Selected Event")
                # Re-score event
                h_posture = str(chosen_row.get("posture", "standing"))
                h_loiter = float(chosen_row.get("loitering_duration_sec", 0.0))
                h_speed = str(chosen_row.get("speed_category", "stationary"))
                h_entity_type = 1 if chosen_row.get("vehicle_plate", "") != "" else 0

                h_params = {
                    "entity_type": h_entity_type,
                    "is_in_zone": True if chosen_row.get("zone_name", "") != "" else False,
                    "loitering_duration": h_loiter,
                    "persons_in_zone": int(chosen_row.get("num_persons_in_zone", 1)),
                    "speed_category": h_speed,
                    "is_moving_toward_zone": False,
                    "is_night_mode": False,
                    "posture": h_posture,
                    "is_erratic": False,
                    "hour_of_day": 12,
                    "direction_toward_zone": 0.2,
                    "crowd_density_gradient": 0.0,
                    "has_weapon": False,
                    "distance_to_boundary": 0.0,
                    "acceleration_magnitude": 0.5,
                }
                h_assessment = threat_scorer.calculate(**h_params)
                h_vectors = calculate_tactical_vectors(h_params, h_assessment.score)
                h_radar = render_threat_radar_chart(h_vectors, title=f"Tactical Profile: {selected_event_id}")
                st.pyplot(h_radar, use_container_width=True)
                render_sop_recommendation(h_assessment.score, h_params)
