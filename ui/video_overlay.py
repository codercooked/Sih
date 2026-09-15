"""
IBVAP — Video Overlay Renderer
Draws bounding boxes, zone polygons, threat badges, skeleton overlays,
and ANPR text on video frames for the Streamlit dashboard.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict


# Color palette (BGR format)
COLORS = {
    "person": (0, 255, 128),       # Green
    "car": (255, 178, 50),         # Blue-ish
    "truck": (255, 178, 50),
    "motorcycle": (255, 178, 50),
    "bus": (255, 178, 50),
    "face": (255, 100, 255),       # Pink
    "backpack": (100, 200, 255),   # Light blue
    "handbag": (100, 200, 255),
    "suitcase": (100, 200, 255),
    "zone_safe": (0, 200, 0),      # Green
    "zone_intrusion": (0, 0, 255), # Red
    "skeleton": (0, 255, 255),     # Yellow
    "threat_low": (0, 200, 0),     # Green
    "threat_medium": (0, 200, 255),# Yellow
    "threat_high": (0, 128, 255),  # Orange
    "threat_critical": (0, 0, 255),# Red
}

THREAT_COLORS = {
    "low": COLORS["threat_low"],
    "medium": COLORS["threat_medium"],
    "high": COLORS["threat_high"],
    "critical": COLORS["threat_critical"],
}


def draw_skeleton(frame: np.ndarray, keypoints: Dict[str, Tuple[int, int]]) -> np.ndarray:
    """Draw skeleton lines connecting keypoints."""
    annotated = frame.copy()
    connections = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle")
    ]
    # Draw lines
    for p1_name, p2_name in connections:
        if p1_name in keypoints and p2_name in keypoints:
            cv2.line(annotated, keypoints[p1_name], keypoints[p2_name], COLORS["skeleton"], 2)
    
    # Draw points
    for name, pt in keypoints.items():
        cv2.circle(annotated, pt, 4, (0, 0, 255), -1)
        
    return annotated


def _draw_tactical_reticle(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int],
    corner_len: int = 12,
    subtle_box: bool = True
):
    """Draws sleek military C2 corner reticles (L-brackets) with optional subtle 1px border."""
    w = x2 - x1
    h = y2 - y1
    c_len = max(6, min(corner_len, int(min(w, h) * 0.25)))

    if subtle_box:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    # Top-Left Bracket
    cv2.line(img, (x1, y1), (x1 + c_len, y1), color, 2, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + c_len), color, 2, cv2.LINE_AA)
    # Top-Right Bracket
    cv2.line(img, (x2, y1), (x2 - c_len, y1), color, 2, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + c_len), color, 2, cv2.LINE_AA)
    # Bottom-Left Bracket
    cv2.line(img, (x1, y2), (x1 + c_len, y2), color, 2, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - c_len), color, 2, cv2.LINE_AA)
    # Bottom-Right Bracket
    cv2.line(img, (x2, y2), (x2 - c_len, y2), color, 2, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - c_len), color, 2, cv2.LINE_AA)


def _draw_pill_badge(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    border_color: Tuple[int, int, int] = (56, 189, 248),
    bg_color: Tuple[int, int, int] = (15, 23, 42),
    text_color: Tuple[int, int, int] = (248, 250, 252),
    font_scale: float = 0.40,
    padding: int = 4
):
    """Draws a compact, dark translucent pill badge with crisp antialiased text."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    bx1 = max(2, x)
    by1 = max(th + padding * 2 + 2, y)
    
    # Semi-dark background
    cv2.rectangle(img, (bx1, by1 - th - padding * 2), (bx1 + tw + padding * 2, by1), bg_color, -1)
    cv2.rectangle(img, (bx1, by1 - th - padding * 2), (bx1 + tw + padding * 2, by1), border_color, 1, cv2.LINE_AA)
    cv2.putText(
        img, text, (bx1 + padding, by1 - padding - 1),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA
    )


def draw_detections(frame: np.ndarray, detections: list, tracked_entities: dict = None) -> np.ndarray:
    """
    Draw clean tactical targeting boxes/reticles without floating text or names, keeping video uncluttered.
    """
    annotated = frame.copy()

    # Draw tracked entities with sleek boxes
    if tracked_entities:
        for entity_id, entity in tracked_entities.items():
            x1, y1, x2, y2 = map(int, entity.bbox)
            color = COLORS.get(entity.class_name, (200, 200, 200))

            # Threat and Zone styling
            if entity.is_in_zone or entity.threat_score >= 80:
                color = (0, 0, 240)  # Crimson Red
            elif entity.threat_score >= 50:
                color = (0, 140, 255)  # Tactical Amber
            elif entity.class_name == "person":
                color = (0, 230, 118)  # Tactical Emerald
            elif entity.class_name in ("car", "truck", "motorcycle", "bus"):
                color = (255, 178, 50)  # Cyan-Blue

            # Sleek corner reticle & subtle box (no names or text cluttering the video)
            _draw_tactical_reticle(annotated, x1, y1, x2, y2, color, corner_len=14, subtle_box=True)

            # Subtle skeleton overlay
            if entity.skeleton:
                annotated = draw_skeleton(annotated, entity.skeleton)
    else:
        # Raw detections fallback: sleek boxes only
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            color = COLORS.get(det.class_name, (200, 200, 200))
            _draw_tactical_reticle(annotated, x1, y1, x2, y2, color, corner_len=10, subtle_box=True)

    return annotated


def draw_zone(
    frame: np.ndarray,
    polygon: np.ndarray,
    is_intruded: bool = False,
    zone_name: str = "",
) -> np.ndarray:
    """
    Draw restricted zone polygon boundary on frame.
    Clean perimeter line only — zero text overlays for professional surveillance output.
    """
    annotated = frame.copy()
    pts = polygon.reshape((-1, 1, 2))

    if is_intruded:
        # Subtle semi-transparent red tint (10% opacity — doesn't obscure subjects)
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [polygon], (0, 0, 180))
        cv2.addWeighted(overlay, 0.10, annotated, 0.90, 0, annotated)
        cv2.polylines(annotated, [pts], True, (0, 0, 240), 2, cv2.LINE_AA)
    else:
        # Crisp emerald perimeter line
        cv2.polylines(annotated, [pts], True, (0, 200, 100), 1, cv2.LINE_AA)

    return annotated


def draw_threat_badge(
    frame: np.ndarray,
    score: int,
    level: str = "low",
    position: Tuple[int, int] = (15, 20),
) -> np.ndarray:
    """
    No-op: threat information is displayed in the side panel, not on the video feed.
    Kept for API compatibility.
    """
    return frame


def draw_plate_text(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    confidence: float = 0.0,
    security_status: str = "UNKNOWN",
) -> np.ndarray:
    """Draw ANPR plate text with sleek tactical badge above vehicle."""
    annotated = frame.copy()
    if not text or text == "UNREADABLE":
        return annotated

    if security_status == "BLACKLISTED":
        label = f"🚨 BLACKLIST: {text} ({confidence:.0%})"
        border_color = (0, 0, 240)
    elif security_status == "AUTHORIZED":
        label = f"✅ AUTH DEFENSE: {text} ({confidence:.0%})"
        border_color = (0, 200, 100)
    else:
        label = f"🚗 PLATE: {text} ({confidence:.0%})"
        border_color = (0, 180, 255)

    x, y = position
    _draw_pill_badge(annotated, label, max(2, x), max(18, y), border_color=border_color, font_scale=0.42)
    return annotated


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """No-op: FPS is displayed in the side panel status bar, not on the video feed."""
    return frame


def draw_night_mode_indicator(frame: np.ndarray) -> np.ndarray:
    """No-op: night mode status is displayed in the side panel, not on the video feed."""
    return frame


def draw_trajectory(
    frame: np.ndarray,
    predicted_points: list,
    color: Tuple[int, int, int] = (255, 255, 0),  # Cyan
) -> np.ndarray:
    """
    Draw predicted trajectory as a fading dashed line with arrow.
    
    Args:
        frame: BGR image
        predicted_points: List of (x, y) predicted future positions
        color: BGR color for the line
    """
    if not predicted_points or len(predicted_points) < 2:
        return frame
    
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    n = len(predicted_points)
    
    for i in range(n - 1):
        x1, y1 = predicted_points[i]
        x2, y2 = predicted_points[i + 1]
        
        # Clamp to frame bounds
        if not (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
            break
        
        # Fade opacity: bright at start, fading towards end
        alpha = 1.0 - (i / n) * 0.7
        faded_color = tuple(int(c * alpha) for c in color)
        
        # Draw dashed line segment
        thickness = max(1, 3 - i // 5)
        cv2.line(annotated, (x1, y1), (x2, y2), faded_color, thickness, cv2.LINE_AA)
    
    # Draw arrowhead at the last point
    if len(predicted_points) >= 2:
        last = predicted_points[-1]
        prev = predicted_points[-2]
        if 0 <= last[0] < w and 0 <= last[1] < h:
            cv2.arrowedLine(
                annotated, prev, last, color, 2, cv2.LINE_AA, tipLength=0.4
            )
    
    # Draw dots at each predicted point
    for i, (x, y) in enumerate(predicted_points):
        if 0 <= x < w and 0 <= y < h:
            radius = max(1, 4 - i // 4)
            alpha = 1.0 - (i / n) * 0.6
            dot_color = tuple(int(c * alpha) for c in color)
            cv2.circle(annotated, (x, y), radius, dot_color, -1)
    
    return annotated


def draw_face_blur(
    frame: np.ndarray,
    face_detections: list,
    blur_strength: int = 51,
) -> np.ndarray:
    """
    Apply Gaussian blur to detected face regions for privacy compliance.
    
    Args:
        frame: BGR image
        face_detections: List of Detection objects with class_name == "face"
        blur_strength: Gaussian kernel size (must be odd)
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    
    for det in face_detections:
        x1, y1, x2, y2 = det.bbox
        # Clamp
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue
        
        # Apply strong Gaussian blur
        face_region = annotated[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(face_region, (blur_strength, blur_strength), 30)
        annotated[y1:y2, x1:x2] = blurred
        
        # Draw privacy indicator
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (200, 200, 200), 1)
        cv2.putText(
            annotated, "PRIVACY", (x1 + 2, y1 + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA,
        )
    
    return annotated


def draw_weapon_alert(
    frame: np.ndarray,
    weapon_detections: list,
) -> np.ndarray:
    """
    Draw clean weapon detection reticles without text labels.
    Alert details are surfaced in the side panel and event log.
    """
    annotated = frame.copy()

    for det in weapon_detections:
        x1, y1, x2, y2 = det.bbox
        # Bright red tactical reticle around detected weapon
        _draw_tactical_reticle(annotated, x1, y1, x2, y2, (0, 0, 255), corner_len=10, subtle_box=True)

    return annotated

