"""
IBVAP — AI Narrative Engine
Generates natural language threat reports from detection data.
Uses template-based generation for reliability and speed,
with optional Hugging Face transformer integration.
"""

import time
import random
from datetime import datetime
from typing import Optional


# Pre-built narrative templates for different scenarios
TEMPLATES = {
    "zone_entry": [
        "At {time}, {entity_type} ({entity_id}) was detected entering the restricted zone '{zone}'. Initial threat assessment: {level}.",
        "Perimeter breach detected at {time}. {entity_type} ({entity_id}) crossed into '{zone}'. Monitoring initiated. Threat level: {level}.",
        "ALERT: {entity_type} ({entity_id}) entered restricted area '{zone}' at {time}. Automated threat assessment classified as {level}.",
    ],
    "loitering": [
        "Subject {entity_id} has been loitering in '{zone}' for {loiter_time}. Behavioral pattern suggests {posture_desc}. Threat level escalated to {level}.",
        "Prolonged presence detected: {entity_id} has remained inside '{zone}' for {loiter_time}. Posture analysis indicates {posture_desc}. Current threat: {level}.",
        "Extended loitering alert for {entity_id} in '{zone}' ({loiter_time}). Subject exhibiting {posture_desc} behavior. Threat assessment: {level}.",
    ],
    "high_threat": [
        "CRITICAL: {entity_type} ({entity_id}) poses immediate threat at '{zone}'. Score: {score}/100. Factors: {factors}. Recommended action: Deploy rapid response unit.",
        "HIGH PRIORITY ALERT: Subject {entity_id} flagged with threat score {score}/100 at '{zone}'. Contributing factors include {factors}. Immediate intervention recommended.",
        "EMERGENCY: Threat level {level} for {entity_id} at '{zone}'. AI analysis: {factors}. Score: {score}/100. Sector commander has been notified.",
    ],
    "weapon": [
        "⚠️ WEAPON ALERT: Suspicious object ({weapon_type}) detected near {entity_id} in '{zone}'. Threat score automatically escalated to {score}/100. Armed response protocol activated.",
        "CRITICAL ALERT: {weapon_type} detected in proximity to {entity_id} at '{zone}'. AI system has escalated threat to {score}/100. All units notified.",
    ],
    "crowd": [
        "Crowd formation detected in '{zone}': {crowd_count} individuals identified. Group behavior analysis initiated. Current threat assessment: {level}.",
        "Multiple subjects ({crowd_count}) detected in restricted area '{zone}'. Crowd density exceeds safety threshold. Threat level: {level}.",
    ],
    "night": [
        "Night-time intrusion detected at {time}. {entity_type} ({entity_id}) operating under low-light conditions in '{zone}'. CLAHE night vision active. Threat: {level}.",
    ],
    "vehicle": [
        "Vehicle ({entity_id}) detected in restricted zone '{zone}' at {time}. License plate: {plate}. Automated ANPR cross-reference initiated.",
        "ANPR Alert: Vehicle {entity_id} with plate '{plate}' entered '{zone}'. Running against watchlist database. Threat assessment: {level}.",
    ],
    "erratic": [
        "Anomalous movement detected: {entity_id} exhibiting erratic trajectory in '{zone}'. Pattern analysis suggests potential evasion tactics. Threat: {level}.",
        "Subject {entity_id} displaying irregular movement patterns at '{zone}'. AI behavioral model flags possible evasive behavior. Score: {score}/100.",
    ],
    "summary": [
        "INCIDENT SUMMARY — {time}\nLocation: {zone}\nSubject: {entity_id} ({entity_type})\nThreat Level: {level} ({score}/100)\nBehavior: {factors}\nPosture: {posture_desc}\nDuration in Zone: {loiter_time}\nRecommended Action: {action}",
    ],
}

POSTURE_DESCRIPTIONS = {
    "standing": "normal standing posture",
    "crouching": "evasive crouching — possible concealment attempt",
    "climbing": "active fence/barrier climbing — probable breach in progress",
    "surrendering": "hands raised in surrender posture — possible compliance",
    "fallen": "subject is on the ground — possible medical emergency or takedown",
    "unknown": "posture undetermined",
}

RECOMMENDED_ACTIONS = {
    "low": "Continue passive monitoring.",
    "medium": "Dispatch patrol for visual confirmation.",
    "high": "Deploy rapid response team. Alert sector commander.",
    "critical": "IMMEDIATE INTERVENTION. Armed response authorized. Lock down sector.",
}


def generate_narrative(
    entity_id: str = "UNKNOWN-01",
    entity_type: str = "person",
    zone_name: str = "Restricted Area",
    threat_score: int = 0,
    threat_level: str = "low",
    behavior_tags: list = None,
    posture: str = "standing",
    loitering_duration: float = 0.0,
    is_night_mode: bool = False,
    vehicle_plate: str = "",
    persons_in_zone: int = 0,
    has_weapon: bool = False,
    weapon_type: str = "",
) -> str:
    """
    Generate a natural language threat narrative from detection data.
    
    Returns:
        Human-readable threat report string
    """
    behavior_tags = behavior_tags or []
    current_time = datetime.now().strftime("%H:%M IST")
    posture_desc = POSTURE_DESCRIPTIONS.get(posture, "unknown posture")
    action = RECOMMENDED_ACTIONS.get(threat_level, "Monitor.")
    
    # Format loitering time
    if loitering_duration > 60:
        loiter_str = f"{loitering_duration / 60:.1f} min"
    elif loitering_duration > 0:
        loiter_str = f"{int(loitering_duration)} sec"
    else:
        loiter_str = "briefly"
    
    # Build factors string from behavior tags
    factor_parts = []
    if "zone_entry" in behavior_tags:
        factor_parts.append("unauthorized entry")
    if "loitering_suspicious" in behavior_tags or "loitering_high_risk" in behavior_tags:
        factor_parts.append(f"prolonged loitering")
    if "erratic_movement" in behavior_tags:
        factor_parts.append("erratic movement")
    if posture in ["crouching", "climbing", "fallen", "surrendering"]:
        factor_parts.append(posture_desc)
    if has_weapon:
        factor_parts.append(f"weapon ({weapon_type})")
    if is_night_mode:
        factor_parts.append("night op")
    if persons_in_zone >= 3:
        factor_parts.append(f"crowd ({persons_in_zone})")
    
    factors_str = ", ".join(factor_parts) if factor_parts else "unspecified anomalies"
    
    # Template variables
    vars = {
        "time": current_time,
        "entity_id": entity_id,
        "entity_type": entity_type.upper(),
        "zone": zone_name,
        "level": threat_level.upper(),
        "score": threat_score,
        "factors": factors_str,
        "posture_desc": posture_desc,
        "loiter_time": loiter_str,
        "action": action,
        "plate": vehicle_plate or "UNREADABLE",
        "crowd_count": persons_in_zone,
        "weapon_type": weapon_type or "unknown",
    }
    
    # Select the best template category
    if has_weapon:
        category = "weapon"
    elif threat_score >= 80:
        category = "high_threat"
    elif vehicle_plate:
        category = "vehicle"
    elif posture in ["crouching", "climbing"]:
        category = "erratic"
    elif "erratic_movement" in behavior_tags:
        category = "erratic"
    elif loitering_duration > 30:
        category = "loitering"
    elif persons_in_zone >= 3:
        category = "crowd"
    else:
        category = "zone_entry"
    
    templates = TEMPLATES.get(category, TEMPLATES["zone_entry"])
    template = random.choice(templates)
    
    try:
        narrative = template.format(**vars)
    except KeyError:
        narrative = f"Intelligence Alert: {entity_type.upper()} ({entity_id}) in {zone_name}. Threat: {threat_level.upper()} ({threat_score}/100). Flags: {factors_str}."
    
    return narrative


def generate_incident_summary(
    entity_id: str = "UNKNOWN-01",
    entity_type: str = "person",
    zone_name: str = "Restricted Area",
    threat_score: int = 0,
    threat_level: str = "low",
    behavior_tags: list = None,
    posture: str = "standing",
    loitering_duration: float = 0.0,
    is_night_mode: bool = False,
    vehicle_plate: str = "",
    persons_in_zone: int = 0,
    has_weapon: bool = False,
    weapon_type: str = "",
) -> str:
    """Generate a structured incident summary report."""
    behavior_tags = behavior_tags or []
    current_time = datetime.now().strftime("%H:%M IST on %d-%b-%Y")
    posture_desc = POSTURE_DESCRIPTIONS.get(posture, "unknown")
    action = RECOMMENDED_ACTIONS.get(threat_level, "Monitor.")
    
    if loitering_duration >= 60:
        loiter_str = f"{loitering_duration / 60:.1f} minutes"
    else:
        loiter_str = f"{int(loitering_duration)} seconds"
    
    factor_parts = []
    if "zone_entry" in behavior_tags:
        factor_parts.append("zone intrusion")
    if "erratic_movement" in behavior_tags:
        factor_parts.append("erratic movement")
    if posture != "standing":
        factor_parts.append(posture_desc)
    if has_weapon:
        factor_parts.append(f"weapon ({weapon_type})")
    if is_night_mode:
        factor_parts.append("night operation")
    
    factors_str = ", ".join(factor_parts) if factor_parts else "standard monitoring"
    
    vars = {
        "time": current_time,
        "entity_id": entity_id,
        "entity_type": entity_type,
        "zone": zone_name,
        "level": threat_level.upper(),
        "score": threat_score,
        "factors": factors_str,
        "posture_desc": posture_desc,
        "loiter_time": loiter_str,
        "action": action,
    }
    
    template = TEMPLATES["summary"][0]
    try:
        return template.format(**vars)
    except KeyError:
        return f"Incident Report: {entity_id} at {zone_name}. Score: {threat_score}/100."
