"""
IBVAP — Automatic Number Plate Recognition (ANPR) Module
Uses OpenCV for plate region crop + EasyOCR for character recognition.

LIMITATION: Accuracy depends heavily on lighting, angle, and plate condition.
Estimated 70-85% on clean, well-lit plates. Lower in poor conditions.
All attempts are logged regardless of confidence.
"""

import cv2
import re
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class PlateResult:
    """ANPR result for a single vehicle."""
    text: str                    # Detected plate text, or "UNREADABLE"
    confidence: float            # OCR confidence (0-1)
    is_readable: bool            # Whether confidence exceeds threshold
    bbox_in_vehicle: Optional[Tuple[int, int, int, int]] = None
    raw_text: str = ""
    is_unauthorized: bool = False # True if on watchlist

# Simulated watchlist of stolen or unauthorized plates
WATCHLIST = {
    "HR26DK8337", "DL8CAF5030", "MH02CB1234", "KA05AB9876", "GJ01XY5678"
}


class ANPREngine:
    """
    License plate recognition pipeline:
    1. Crop bottom portion of vehicle bounding box
    2. Preprocess (grayscale, contrast enhancement, bilateral filter)
    3. OCR via EasyOCR
    4. Clean and validate text
    """

    def __init__(self, confidence_threshold: float = 0.4, languages: list = None):
        """
        Args:
            confidence_threshold: Minimum OCR confidence to consider readable
            languages: OCR languages (default: English)
        """
        self.confidence_threshold = confidence_threshold
        self.languages = languages or ["en"]
        self._reader = None  # Lazy initialization (EasyOCR is slow to load)

    @property
    def reader(self):
        """Lazy-load EasyOCR reader (downloads models on first use)."""
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(self.languages, gpu=False)
        return self._reader

    def read_plate(
        self,
        frame: np.ndarray,
        vehicle_bbox: Tuple[int, int, int, int],
    ) -> PlateResult:
        """
        Attempt to read license plate from a vehicle detection using EasyOCR's ML detector.
        """
        x1, y1, x2, y2 = vehicle_bbox
        h, w = frame.shape[:2]
        
        # Clamp to frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if y2 - y1 < 20 or x2 - x1 < 30:
            return PlateResult(text="UNREADABLE", confidence=0.0, is_readable=False)

        # Use the entire vehicle crop. EasyOCR's CRAFT detector is much better
        # at finding text than a rigid 35% crop heuristic.
        vehicle_crop = frame[y1:y2, x1:x2]
        
        # We don't apply harsh thresholding anymore because it destroys details
        # for the ML OCR model. Just pass the raw RGB/BGR image.
        try:
            results = self.reader.readtext(vehicle_crop, detail=1)
        except Exception:
            return PlateResult(text="UNREADABLE", confidence=0.0, is_readable=False)

        if not results:
            return PlateResult(text="UNREADABLE", confidence=0.0, is_readable=False)

        best_text = ""
        best_confidence = 0.0

        for (bbox_pts, text, confidence) in results:
            if confidence > best_confidence:
                # Basic sanity check to avoid reading "TOYOTA" as a plate
                cleaned = self._clean_plate_text(text)
                # A plate typically has 4 to 10 characters and contains at least 1 number
                if 4 <= len(cleaned) <= 10 and any(c.isdigit() for c in cleaned):
                    best_text = text
                    best_confidence = confidence

        if not best_text:
            return PlateResult(text="UNREADABLE", confidence=0.0, is_readable=False)

        cleaned_text = self._clean_plate_text(best_text)
        is_readable = best_confidence >= self.confidence_threshold
        is_unauthorized = cleaned_text in WATCHLIST

        return PlateResult(
            text=cleaned_text if is_readable else "UNREADABLE",
            confidence=best_confidence,
            is_readable=is_readable,
            bbox_in_vehicle=vehicle_bbox,
            raw_text=best_text,
            is_unauthorized=is_unauthorized,
        )

    @staticmethod
    def _clean_plate_text(text: str) -> str:
        """
        Clean OCR output:
        - Remove special characters except hyphens and spaces
        - Convert to uppercase
        - Strip whitespace
        """
        # Keep only alphanumeric, hyphens, spaces
        cleaned = re.sub(r"[^A-Za-z0-9\-\s]", "", text)
        cleaned = cleaned.strip().upper()
        # Collapse multiple spaces
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned
