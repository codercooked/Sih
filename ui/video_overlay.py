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


def draw_detections(frame: np.ndarray, detections: list, tracked_entities: dict = None) -> np.ndarray:
    """
    Draw bounding boxes with class labels and confidence on frame.
    
    Args:
        frame: BGR image
        detections: List of Detection objects
        tracked_entities: Dict of entity_id → TrackedEntity (for drawing IDs)
    
    Returns:
        Annotated frame
    """
    annotated = frame.copy()

    # Draw tracked entities with IDs
    if tracked_entities:
        for entity_id, entity in tracked_entities.items():
            x1, y1, x2, y2 = entity.bbox
            color = COLORS.get(entity.class_name, (200, 200, 200))

            # Red box if in zone
            if entity.is_in_zone:
                color = (0, 0, 255)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label with entity ID + class
            label = f"{entity.entity_id} ({entity.class_name})"
            if entity.threat_score > 0:
                label += f" [{entity.threat_score}]"

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                annotated, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

            # Draw posture tag if present
            if entity.posture and entity.posture != "standing":
                posture_label = f"⚠ {entity.posture.upper()}"
                cv2.putText(
                    annotated, posture_label, (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA,
                )

            # Draw skeleton if present
            if entity.skeleton:
                annotated = draw_skeleton(annotated, entity.skeleton)
    else:
        # Draw raw detections without IDs
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = COLORS.get(det.class_name, (200, 200, 200))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{det.class_name} {det.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                annotated, label, (x1 + 2, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
            )

    return annotated


def draw_zone(
    frame: np.ndarray,
    polygon: np.ndarray,
    is_intruded: bool = False,
    zone_name: str = "",
) -> np.ndarray:
    """
    Draw restricted zone polygon on frame.
    Green when safe, red with fill when intrusion detected.
    """
    annotated = frame.copy()
    pts = polygon.reshape((-1, 1, 2))

    if is_intruded:
        # Semi-transparent red fill
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [polygon], (0, 0, 180))
        cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)
        cv2.polylines(annotated, [pts], True, COLORS["zone_intrusion"], 3)
    else:
        # Green outline
        cv2.polylines(annotated, [pts], True, COLORS["zone_safe"], 2)

    # Zone label
    if zone_name:
        cx = int(np.mean(polygon[:, 0]))
        cy = int(np.mean(polygon[:, 1]))
        status = "⚠ INTRUSION" if is_intruded else "MONITORING"
        label = f"{zone_name}: {status}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(
            annotated, label, (cx - tw // 2, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            COLORS["zone_intrusion"] if is_intruded else COLORS["zone_safe"],
            2, cv2.LINE_AA,
        )

    return annotated


def draw_threat_badge(
    frame: np.ndarray,
    score: int,
    level: str = "low",
    position: Tuple[int, int] = (10, 10),
) -> np.ndarray:
    """
    Draw threat score badge in corner of frame.
    Color-coded by threat level.
    """
    annotated = frame.copy()
    color = THREAT_COLORS.get(level, COLORS["threat_low"])

    # Badge background
    badge_text = f"THREAT: {score}/100"
    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    x, y = position
    cv2.rectangle(
        annotated, (x, y), (x + tw + 20, y + th + 20),
        color, -1,
    )
    cv2.putText(
        annotated, badge_text, (x + 10, y + th + 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )

    return annotated


def draw_plate_text(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    confidence: float = 0.0,
) -> np.ndarray:
    """Draw ANPR plate text near the vehicle."""
    annotated = frame.copy()
    if not text or text == "UNREADABLE":
        return annotated

    label = f"PLATE: {text} ({confidence:.0%})"
    x, y = position
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

    # Yellow background
    cv2.rectangle(annotated, (x, y - th - 8), (x + tw + 10, y + 4), (0, 200, 255), -1)
    cv2.putText(
        annotated, label, (x + 5, y - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
    )

    return annotated


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draw FPS counter on frame."""
    annotated = frame.copy()
    h = annotated.shape[0]
    label = f"FPS: {fps:.1f}"
    cv2.putText(
        annotated, label, (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
    )
    return annotated


def draw_night_mode_indicator(frame: np.ndarray) -> np.ndarray:
    """Draw night mode indicator on frame."""
    annotated = frame.copy()
    w = annotated.shape[1]
    label = "NIGHT MODE (CLAHE Enhanced)"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(
        annotated, label, (w - tw - 10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA,
    )
    return annotated


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
    Draw flashing weapon alert badges on detected weapons.
    
    Args:
        frame: BGR image
        weapon_detections: List of Detection objects for weapons
    """
    annotated = frame.copy()
    
    for det in weapon_detections:
        x1, y1, x2, y2 = det.bbox
        
        # Bright red box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        # Warning label
        label = f"WEAPON: {det.class_name.upper()}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 12), (x1 + tw + 10, y1), (0, 0, 200), -1)
        cv2.putText(
            annotated, label, (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )
    
    # Large warning banner if any weapons detected
    if weapon_detections:
        h, w = annotated.shape[:2]
        banner = "⚠ WEAPON DETECTED — ARMED RESPONSE ACTIVATED ⚠"
        (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        bx = (w - tw) // 2
        by = 50
        cv2.rectangle(annotated, (bx - 10, by - th - 10), (bx + tw + 10, by + 10), (0, 0, 200), -1)
        cv2.putText(
            annotated, banner, (bx, by),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
        )
    
    return annotated

