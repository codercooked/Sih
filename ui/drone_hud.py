import cv2
import numpy as np
import time
import base64

def generate_drone_hud(frame, bbox):
    """
    Takes the main frame and the bounding box of a critical threat.
    Returns a base64 encoded image of the Drone Intercept HUD.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    
    # Add some padding to the crop
    pad_x = int((x2 - x1) * 0.5)
    pad_y = int((y2 - y1) * 0.5)
    
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)
    
    if cx2 <= cx1 or cy2 <= cy1:
        return None
        
    roi = frame[cy1:cy2, cx1:cx2].copy()
    if roi.size == 0:
        return None
        
    # Resize for a uniform drone feed size
    target_h, target_w = 400, 400
    roi = cv2.resize(roi, (target_w, target_h))
    
    # Apply "Thermal Vision" effect (COLORMAP_JET or INFERNO)
    thermal = cv2.applyColorMap(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_INFERNO)
    
    # Overlay Crosshairs
    center_x, center_y = target_w // 2, target_h // 2
    color = (0, 255, 0) # Tactical Green
    
    # Outer circle
    cv2.circle(thermal, (center_x, center_y), 100, color, 1)
    cv2.circle(thermal, (center_x, center_y), 102, color, 1)
    
    # Crosshairs
    cv2.line(thermal, (center_x - 120, center_y), (center_x - 20, center_y), color, 2)
    cv2.line(thermal, (center_x + 20, center_y), (center_x + 120, center_y), color, 2)
    cv2.line(thermal, (center_x, center_y - 120), (center_x, center_y - 20), color, 2)
    cv2.line(thermal, (center_x, center_y + 20), (center_x, center_y + 120), color, 2)
    
    # Center dot
    cv2.circle(thermal, (center_x, center_y), 2, (0, 0, 255), -1)
    
    # Telemetry data
    cv2.putText(thermal, "DRONE INTERCEPT MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(thermal, f"T-MINUS: {45 - (int(time.time()) % 45)}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.putText(thermal, "RNG: 1420m", (target_w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(thermal, "ALT: 850m", (target_w - 120, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(thermal, "SPD: 42kt", (target_w - 120, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Flashing REC
    if int(time.time() * 2) % 2 == 0:
        cv2.circle(thermal, (20, target_h - 20), 8, (0, 0, 255), -1)
        cv2.putText(thermal, "REC", (35, target_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
    # Encode to base64
    _, buffer = cv2.imencode('.jpg', thermal)
    b64_img = base64.b64encode(buffer).decode('utf-8')
    return b64_img


def generate_drone_hud_frame(frame, bbox, target_w=150, target_h=120):
    """
    Takes the main frame and the bounding box of a critical threat.
    Returns the OpenCV BGR image of the Drone Intercept HUD ready for PiP overlay.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]

    # Add padding
    pad_x = int((x2 - x1) * 0.4)
    pad_y = int((y2 - y1) * 0.4)

    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)

    if cx2 <= cx1 or cy2 <= cy1:
        return None

    roi = frame[cy1:cy2, cx1:cx2].copy()
    if roi.size == 0:
        return None

    roi = cv2.resize(roi, (target_w, target_h))
    thermal = cv2.applyColorMap(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_INFERNO)

    center_x, center_y = target_w // 2, target_h // 2
    color = (0, 255, 0)

    # Reticle and crosshairs
    cv2.circle(thermal, (center_x, center_y), min(target_w, target_h) // 4, color, 1)
    cv2.line(thermal, (center_x - 30, center_y), (center_x - 6, center_y), color, 1)
    cv2.line(thermal, (center_x + 6, center_y), (center_x + 30, center_y), color, 1)
    cv2.line(thermal, (center_x, center_y - 30), (center_x, center_y - 6), color, 1)
    cv2.line(thermal, (center_x, center_y + 6), (center_x, center_y + 30), color, 1)

    # Telemetry HUD
    cv2.putText(thermal, "UAV INTERCEPT", (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)
    cv2.putText(thermal, f"LOCK {45 - (int(time.time()) % 45)}s", (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 0), 1)

    # Tactical border
    cv2.rectangle(thermal, (0, 0), (target_w - 1, target_h - 1), (0, 0, 255), 2)
    return thermal

