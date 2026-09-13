"""
IBVAP — Biometric Face Scanner & Facial Recognition System (FRS)
Fulfills SIH Problem Statement 26187:
Performs software-based face detection and recognition against a defense
watchlist without requiring dedicated FRS smart-camera hardware.
"""

import cv2
import numpy as np
import time
from core.frs import FacialRecognitionSystem, FRSResult


class FaceScanner:
    """
    Combines Haar Cascade face localization with OpenCV LBPH Facial Recognition.
    """

    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.frs = FacialRecognitionSystem()

    def scan_for_faces(self, frame, bbox):
        """
        Scans a specific bounding box in the frame for faces and performs FRS identification.
        Returns a list of detected face data: {"box": (x, y, w, h), "watchlist_match": dict or None, "frs_result": FRSResult}
        """
        x1, y1, x2, y2 = map(int, bbox)

        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return []
        if self.face_cascade.empty():
            return []

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return []

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray_roi,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(25, 25)
        )

        results = []
        for (fx, fy, fw, fh) in faces:
            global_fx = x1 + fx
            global_fy = y1 + fy
            face_crop = roi[fy:fy + fh, fx:fx + fw]

            frs_res = self.frs.identify(face_crop)

            match = None
            if frs_res.is_identified:
                status_label = f"🚨 {frs_res.name}" if frs_res.category == "SUSPECT" else f"✅ {frs_res.name}"
                match = {
                    "name": frs_res.name,
                    "category": frs_res.category,
                    "role": frs_res.role,
                    "match_confidence": f"{frs_res.confidence_score}%",
                    "status": status_label,
                    "color_hex": frs_res.color_hex
                }

            results.append({
                "box": (global_fx, global_fy, fw, fh),
                "watchlist_match": match,
                "frs_result": frs_res
            })

        return results

    def draw_biometric_scan(self, frame, face_data):
        """
        Draws tactical biometric targeting HUD and identity badge over detected faces.
        """
        for data in face_data:
            x, y, w, h = data["box"]
            match = data["watchlist_match"]

            # Color logic
            if match:
                if match.get("category") == "SUSPECT":
                    color = (0, 0, 255) # Red for Suspect
                else:
                    color = (0, 230, 118) # Green for Authorized BSF
            else:
                color = (255, 215, 0) # Cyan/Gold for Unidentified

            thickness = 2
            length = int(w * 0.25)

            # High-tech corner reticle
            cv2.line(frame, (x, y), (x + length, y), color, thickness)
            cv2.line(frame, (x, y), (x, y + length), color, thickness)
            cv2.line(frame, (x + w, y), (x + w - length, y), color, thickness)
            cv2.line(frame, (x + w, y), (x + w, y + length), color, thickness)
            cv2.line(frame, (x, y + h), (x + length, y + h), color, thickness)
            cv2.line(frame, (x, y + h), (x, y + h - length), color, thickness)
            cv2.line(frame, (x + w, y + h), (x + w - length, y + h), color, thickness)
            cv2.line(frame, (x + w, y + h), (x + w, y + h - length), color, thickness)

            # Scanning sweep line
            scan_y = int(y + (time.time() * 120 % max(1, h)))
            cv2.line(frame, (x, scan_y), (x + w, scan_y), color, 1)

            # Identification Badge Overlay
            if match:
                badge_w = max(w, 180)
                badge_bg = (0, 0, 180) if match.get("category") == "SUSPECT" else (0, 140, 60)
                cv2.rectangle(frame, (x, y - 36), (x + badge_w, y), badge_bg, -1)
                cv2.rectangle(frame, (x, y - 36), (x + badge_w, y), color, 1)

                cv2.putText(frame, match["status"], (x + 4, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
                subtext = f"FRS MATCH: {match['match_confidence']} ({match['role']})"
                cv2.putText(frame, subtext[:32], (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)
            else:
                cv2.putText(frame, "FRS: SCANNING...", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

        return frame
