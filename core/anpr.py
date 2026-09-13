"""
IBVAP — Automatic Number Plate Recognition (ANPR) & Vehicle Security Module
Fulfills SIH Problem Statement 26187:
Enables vehicle detection, classification, and license plate recognition directly
from standard CCTV streams without requiring proprietary ANPR smart cameras.

Maintains a tactical Border Checkpoint Vehicle Database:
  - BLACKLIST: Flagged stolen, smuggling, or unauthorized cross-border vehicles
  - WHITELIST: Authorized BSF, Defense, and Government convoys
  - UNKNOWN: Unregistered civilian vehicles requiring checkpoint inspection
"""

import cv2
import re
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field


@dataclass
class PlateResult:
    """ANPR & Vehicle Security result."""
    text: str                               # Detected plate text, or "UNREADABLE"
    confidence: float                       # OCR confidence (0-1)
    is_readable: bool                       # Whether confidence exceeds threshold
    bbox_in_vehicle: Optional[Tuple[int, int, int, int]] = None
    raw_text: str = ""
    is_unauthorized: bool = False           # True if on blacklist
    security_status: str = "UNKNOWN"        # "BLACKLISTED", "AUTHORIZED", "UNKNOWN"
    vehicle_category: str = "Light Vehicle" # "Light Vehicle", "Heavy Cargo", "Defense Convoy", "Two-Wheeler"
    vehicle_details: str = ""               # Additional database notes
    alert_color: str = "#FFC107"            # Hex color for UI display


# Tactical Border Checkpoint Vehicle Security Database
BLACKLIST_REGISTRY: Dict[str, Dict[str, str]] = {
    "HR26DK8337": {"owner": "Unknown / Shell Entity", "reason": "Suspected Contraband Transport", "type": "Dark SUV"},
    "DL8CAF5030": {"owner": "Flagged Syndicate", "reason": "Inter-State Smuggling Watchlist", "type": "Sedan"},
    "MH02CB1234": {"owner": "Stolen Vehicle Database", "reason": "Reported Armed Robbery Escort", "type": "White Pickup"},
    "JK02BA7711": {"owner": "Border Watchlist", "reason": "Unauthorized Night Transit Attempt", "type": "Mini Truck"},
    "PB02X9999":  {"owner": "Narcotics Control Bureau", "reason": "High-Priority Border Interdiction", "type": "Bolero Camper"},
    "GJ01XY5678": {"owner": "Suspect Courier", "reason": "Unregistered Night Reconnaissance", "type": "Cargo Van"},
}

WHITELIST_REGISTRY: Dict[str, Dict[str, str]] = {
    "BSF-01":     {"owner": "Border Security Force", "reason": "BOP Patrol Gypsy", "type": "Defense Transport"},
    "ARMY-108":   {"owner": "Indian Army", "reason": "Border Road Logistics Convoy", "type": "Military Truck"},
    "POLICE-100": {"owner": "State Border Police", "reason": "Highway Interceptor Vehicle", "type": "Police Cruiser"},
    "GOV-IN-01":  {"owner": "Border Area Development", "reason": "Civil Administration Official", "type": "Govt Vehicle"},
}


class ANPREngine:
    """
    License plate recognition and tactical vehicle security pipeline.
    """

    def __init__(self, confidence_threshold: float = 0.4, languages: list = None):
        """
        Args:
            confidence_threshold: Minimum OCR confidence to consider readable
            languages: OCR languages (default: English)
        """
        self.confidence_threshold = confidence_threshold
        self.languages = languages or ["en"]
        self._reader = None

    @property
    def reader(self):
        """Lazy-load EasyOCR reader."""
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(self.languages, gpu=False)
        return self._reader

    def read_plate(
        self,
        frame: np.ndarray,
        vehicle_bbox: Tuple[int, int, int, int],
        class_name: str = "car",
    ) -> PlateResult:
        """
        Attempt to read license plate from a vehicle detection using EasyOCR.
        """
        x1, y1, x2, y2 = vehicle_bbox
        h, w = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        v_cat = self.classify_vehicle(class_name)

        if y2 - y1 < 20 or x2 - x1 < 30:
            return PlateResult(
                text="UNREADABLE",
                confidence=0.0,
                is_readable=False,
                vehicle_category=v_cat,
                security_status="UNKNOWN",
                alert_color="#9E9E9E"
            )

        vehicle_crop = frame[y1:y2, x1:x2]

        try:
            results = self.reader.readtext(vehicle_crop, detail=1)
        except Exception:
            return PlateResult(
                text="UNREADABLE",
                confidence=0.0,
                is_readable=False,
                vehicle_category=v_cat,
                security_status="UNKNOWN",
                alert_color="#9E9E9E"
            )

        if not results:
            return PlateResult(
                text="UNREADABLE",
                confidence=0.0,
                is_readable=False,
                vehicle_category=v_cat,
                security_status="UNKNOWN",
                alert_color="#9E9E9E"
            )

        best_text = ""
        best_confidence = 0.0

        for (_, text, confidence) in results:
            if confidence > best_confidence:
                cleaned = self._clean_plate_text(text)
                if 4 <= len(cleaned) <= 12 and any(c.isdigit() for c in cleaned):
                    best_text = text
                    best_confidence = confidence

        if not best_text:
            return PlateResult(
                text="UNREADABLE",
                confidence=0.0,
                is_readable=False,
                vehicle_category=v_cat,
                security_status="UNKNOWN",
                alert_color="#9E9E9E"
            )

        cleaned_text = self._clean_plate_text(best_text)
        is_readable = best_confidence >= self.confidence_threshold

        # Cross-reference security database
        security_status = "UNKNOWN"
        vehicle_details = "Civilian vehicle — standard checkpoint registration."
        alert_color = "#FFC107" # Yellow
        is_unauthorized = False

        if cleaned_text in BLACKLIST_REGISTRY:
            security_status = "BLACKLISTED"
            is_unauthorized = True
            info = BLACKLIST_REGISTRY[cleaned_text]
            vehicle_details = f"⚠️ FLAG: {info['reason']} ({info['type']})"
            alert_color = "#F44336" # Red
        elif cleaned_text in WHITELIST_REGISTRY:
            security_status = "AUTHORIZED"
            info = WHITELIST_REGISTRY[cleaned_text]
            vehicle_details = f"✅ AUTH: {info['owner']} - {info['reason']}"
            alert_color = "#4CAF50" # Green

        return PlateResult(
            text=cleaned_text if is_readable else "UNREADABLE",
            confidence=best_confidence,
            is_readable=is_readable,
            bbox_in_vehicle=vehicle_bbox,
            raw_text=best_text,
            is_unauthorized=is_unauthorized,
            security_status=security_status,
            vehicle_category=v_cat,
            vehicle_details=vehicle_details,
            alert_color=alert_color
        )

    @staticmethod
    def classify_vehicle(class_name: str) -> str:
        """Categorize YOLO vehicle class into border defense categories."""
        c = class_name.lower()
        if c in ["truck", "bus"]:
            return "Heavy Cargo / Transport"
        elif c in ["motorcycle", "bicycle"]:
            return "Two-Wheeler / Recon Axis"
        else:
            return "Light Motor Vehicle"

    @staticmethod
    def _clean_plate_text(text: str) -> str:
        """Clean OCR output."""
        cleaned = re.sub(r"[^A-Za-z0-9\-\s]", "", text)
        cleaned = cleaned.strip().upper()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    @classmethod
    def get_security_database_records(cls) -> List[Dict]:
        """Return combined database for UI registry inspection."""
        records = []
        for plate, info in BLACKLIST_REGISTRY.items():
            records.append({
                "Plate Number": plate,
                "Status": "🚨 BLACKLISTED",
                "Entity / Agency": info["owner"],
                "Alert Trigger / Reason": info["reason"],
                "Vehicle Type": info["type"]
            })
        for plate, info in WHITELIST_REGISTRY.items():
            records.append({
                "Plate Number": plate,
                "Status": "✅ AUTHORIZED",
                "Entity / Agency": info["owner"],
                "Alert Trigger / Reason": info["reason"],
                "Vehicle Type": info["type"]
            })
        return records
