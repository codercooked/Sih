import cv2
import numpy as np

class FaceScanner:
    def __init__(self):
        # Load the pre-trained Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        
    def scan_for_faces(self, frame, bbox):
        """
        Scans a specific bounding box in the frame for faces.
        Returns a list of detected face data: {"box": (x, y, w, h), "watchlist_match": dict or None}
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure bbox is within frame bounds
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
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray_roi,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(20, 20)
        )
        
        results = []
        for (fx, fy, fw, fh) in faces:
            # Adjust face coordinates to global frame
            global_fx = x1 + fx
            global_fy = y1 + fy
            
            # Detection is not identification.  Do not fabricate watchlist
            # matches from image coordinates; that creates dangerous false
            # positives and misrepresents Haar detection as biometrics.
            match = None
                
            results.append({
                "box": (global_fx, global_fy, fw, fh),
                "watchlist_match": match
            })
            
        return results

    def draw_biometric_scan(self, frame, face_data):
        """
        Draws a high-tech biometric scanning grid over detected faces.
        """
        for data in face_data:
            x, y, w, h = data["box"]
            match = data["watchlist_match"]
            
            # Draw targeting corners
            color = (0, 0, 255) if match else (255, 255, 0) # Red if match, Cyan if normal scan
            thickness = 2
            length = int(w * 0.2)
            
            # Top-left
            cv2.line(frame, (x, y), (x + length, y), color, thickness)
            cv2.line(frame, (x, y), (x, y + length), color, thickness)
            # Top-right
            cv2.line(frame, (x + w, y), (x + w - length, y), color, thickness)
            cv2.line(frame, (x + w, y), (x + w, y + length), color, thickness)
            # Bottom-left
            cv2.line(frame, (x, y + h), (x + length, y + h), color, thickness)
            cv2.line(frame, (x, y + h), (x, y + h - length), color, thickness)
            # Bottom-right
            cv2.line(frame, (x + w, y + h), (x + w - length, y + h), color, thickness)
            cv2.line(frame, (x + w, y + h), (x + w, y + h - length), color, thickness)
            
            # Draw scanning line (simulated)
            import time
            scan_y = int(y + (time.time() * 100 % h))
            cv2.line(frame, (x, scan_y), (x + w, scan_y), (0, 255, 0), 1)
            
            if match:
                # Flashy Watchlist Text
                label1 = f"MATCH: {match['match_confidence']}"
                label2 = f"{match['status']}"
                
                cv2.rectangle(frame, (x, y - 35), (x + max(w, 120), y), (0, 0, 255), -1)
                cv2.putText(frame, label1, (x + 2, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, label2, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
        return frame
