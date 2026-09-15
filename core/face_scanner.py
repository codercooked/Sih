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

        # Robust CCTV fallback: if Haar misses due to camera angle/distance,
        # extract candidate biometric facial head region from upper anatomy
        if len(faces) == 0 and (y2 - y1) >= 45 and (x2 - x1) >= 20:
            box_w = x2 - x1
            box_h = y2 - y1
            fw = max(20, int(box_w * 0.55))
            fh = max(20, int(box_h * 0.25))
            fx = max(0, int(box_w * 0.22))
            fy = max(0, int(box_h * 0.02))
            faces = [(fx, fy, min(fw, box_w - fx), min(fh, box_h - fy))]

        results = []
        for (fx, fy, fw, fh) in faces:
            global_fx = x1 + fx
            global_fy = y1 + fy

            face_crop_raw = roi[fy:fy + fh, fx:fx + fw]
            if face_crop_raw.size == 0:
                continue

            # Natural, balanced portrait crop for FRS and UI display (includes full head context)
            cx, cy = fx + fw // 2, fy + fh // 2
            dim = int(max(fw, fh) * 1.55)  # Expanded for full head
            rx1 = max(0, cx - dim // 2)
            rx2 = min(roi.shape[1], cx + dim // 2)
            ry1 = max(0, cy - int(dim * 0.60))
            ry2 = min(roi.shape[0], cy + int(dim * 0.60))

            full_head_crop = roi[ry1:ry2, rx1:rx2]
            if full_head_crop.size == 0:
                full_head_crop = face_crop_raw

            # Run FRS on the full head crop for much higher accuracy
            frs_res = self.frs.identify(full_head_crop)

            # Upscale if low resolution using Lanczos interpolation so CCTV capture appears crisp
            display_crop = full_head_crop
            if display_crop.shape[0] < 140 or display_crop.shape[1] < 140:
                display_crop = cv2.resize(display_crop, (150, 150), interpolation=cv2.INTER_LANCZOS4)

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
                "frs_result": frs_res,
                "face_crop": display_crop
            })

        return results

    def draw_biometric_scan(self, frame, face_data):
        """
        Draws clean, defense-grade biometric targeting reticles without visual clutter.
        """
        for data in face_data:
            x, y, w, h = data["box"]
            match = data["watchlist_match"]

            # Color logic
            if match:
                if match.get("category") == "SUSPECT":
                    color = (0, 0, 240) # Crimson Red
                else:
                    color = (0, 230, 118) # Emerald Green
            else:
                color = (0, 212, 255) # Cyan

            # Subtle bounding frame around tracked face (leaves face clear, zero text)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)

            # Sleek corner brackets
            corner_len = max(5, int(min(w, h) * 0.25))
            # Top-left
            cv2.line(frame, (x, y), (x + corner_len, y), color, 2, cv2.LINE_AA)
            cv2.line(frame, (x, y), (x, y + corner_len), color, 2, cv2.LINE_AA)
            # Top-right
            cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, 2, cv2.LINE_AA)
            cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, 2, cv2.LINE_AA)
            # Bottom-left
            cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, 2, cv2.LINE_AA)
            cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, 2, cv2.LINE_AA)
            # Bottom-right
            cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, 2, cv2.LINE_AA)
            cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, 2, cv2.LINE_AA)

        return frame
