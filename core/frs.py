"""
IBVAP — Pure-Software Facial Recognition System (FRS) Module
Fulfills SIH Problem Statement 26187:
Enables facial identification directly from standard CCTV streams without
requiring expensive, dedicated FRS smart-camera hardware.

Uses OpenCV Local Binary Pattern Histograms (LBPH) for real-time, low-power,
illumination-robust face classification entirely on CPU.
"""

import os
import cv2
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class FRSResult:
    """Result of facial recognition for a detected face crop."""
    is_identified: bool
    label_id: int
    name: str
    category: str              # "SUSPECT", "AUTHORIZED_BSF", "CIVILIAN", "UNKNOWN"
    role: str                  # e.g., "Infiltration Watchlist", "BSF Patrol Jawan"
    confidence_score: float    # 0.0 to 100.0%
    distance: float            # LBPH raw distance
    color_hex: str             # Visual alert color (Red for Suspect, Green for Authorized, Yellow for Civilian)
    id_card_number: str = "N/A"
    clearance: str = "UNVERIFIED"
    photo_path: Optional[str] = None
    notes: str = ""


class FacialRecognitionSystem:
    """
    Pure-software edge Facial Recognition System for Border Surveillance.
    Maintains a local Watchlist & Personnel Registry:
      - High-Priority Infiltration Suspects (Red Alert)
      - Authorized Border Security Personnel / BSF Patrol (Green / Whitelist)
      - Unregistered / Civilian (Amber / Investigation Required)
    """

    DEFAULT_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "database")
    MODEL_PATH = os.path.join(DEFAULT_DB_DIR, "frs_model.xml")
    PROFILES_PATH = os.path.join(DEFAULT_DB_DIR, "frs_profiles.json")

    def __init__(self, confidence_threshold: float = 135.0):
        """
        Args:
            confidence_threshold: Distance threshold below which match is accepted.
                                  (In LBPH, lower distance = closer match).
        """
        self.confidence_threshold = confidence_threshold
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
        self.profiles: Dict[int, Dict] = {}
        self.is_trained = False

        os.makedirs(self.DEFAULT_DB_DIR, exist_ok=True)
        self._load_or_initialize()

    def _load_or_initialize(self):
        """Load enrolled profiles and trained model, or initialize standard defense watchlist."""
        if os.path.exists(self.PROFILES_PATH) and os.path.exists(self.MODEL_PATH):
            try:
                with open(self.PROFILES_PATH, "r") as f:
                    raw = json.load(f)
                    self.profiles = {int(k): v for k, v in raw.items()}
                for k, p in self.profiles.items():
                    if "id_card_number" not in p:
                        p["id_card_number"] = f"POI-IND-{k}" if p.get("category") == "SUSPECT" else f"BSF-SNT-{k}"
                    if "clearance" not in p:
                        p["clearance"] = "CRITICAL RED NOTICE" if p.get("category") == "SUSPECT" else "LEVEL-2 BORDER SENTRY"
                self.recognizer.read(self.MODEL_PATH)
                self.is_trained = True
                return
            except Exception:
                pass

        # Seed standard border defense demo profiles
        self._seed_default_watchlist()

    def _seed_default_watchlist(self):
        """Initialize representative border security watchlist and synthesize training samples."""
        self.profiles = {
            101: {
                "name": "Tariq Mahmood",
                "category": "SUSPECT",
                "role": "High-Risk Border Infiltrator",
                "id_card_number": "POI-IND-10492",
                "clearance": "CRITICAL RED NOTICE",
                "notes": "Flagged on IB Western Sector alert list.",
                "color_hex": "#F44336"
            },
            102: {
                "name": "Vikram Rawat",
                "category": "SUSPECT",
                "role": "Cross-Border Contraband Courier",
                "id_card_number": "POI-IND-20831",
                "clearance": "WATCHLIST CONTRABAND",
                "notes": "History of unauthorized perimeter breaches.",
                "color_hex": "#F44336"
            },
            201: {
                "name": "Ct. Rajesh Sharma",
                "category": "AUTHORIZED_BSF",
                "role": "BSF Sentry Patrol Alpha",
                "id_card_number": "BSF-SNT-4108",
                "clearance": "LEVEL-2 BORDER SENTRY",
                "notes": "Assigned to BOP-01 North Gate perimeter.",
                "color_hex": "#4CAF50"
            },
            202: {
                "name": "Insp. Amarjit Singh",
                "category": "AUTHORIZED_BSF",
                "role": "BOP Commander / Duty Officer",
                "id_card_number": "BSF-OFF-0922",
                "clearance": "LEVEL-3 DUTY COMMANDER",
                "notes": "Sector Command Officer.",
                "color_hex": "#4CAF50"
            },
        }

        # Synthesize baseline biometric patterns so model starts pre-trained for testing
        faces = []
        labels = []
        for label_id in self.profiles.keys():
            # Generate deterministic synthetic facial feature prototypes
            np.random.seed(label_id)
            base_pattern = np.random.randint(60, 200, (120, 120), dtype=np.uint8)
            for variation in range(5):
                noise = np.random.randint(-15, 15, (120, 120), dtype=np.int16)
                face_var = np.clip(base_pattern.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                faces.append(face_var)
                labels.append(label_id)

        self.recognizer.train(faces, np.array(labels, dtype=np.int32))
        self.is_trained = True
        self.save_database()

    def enroll_face(
        self,
        face_images: List[np.ndarray],
        name: str,
        category: str,
        role: str,
        id_card_number: str = "",
        clearance: str = "",
        notes: str = ""
    ) -> int:
        """
        Enroll a new person of interest or authorized jawan into the FRS database.
        """
        new_id = (max(self.profiles.keys()) + 1) if self.profiles else 101
        color_hex = "#F44336" if category == "SUSPECT" else ("#4CAF50" if category == "AUTHORIZED_BSF" else "#FFC107")
        if not id_card_number:
            id_card_number = f"POI-IND-{new_id}" if category == "SUSPECT" else f"BSF-SNT-{new_id}"
        if not clearance:
            clearance = "CRITICAL RED NOTICE" if category == "SUSPECT" else "LEVEL-2 BORDER SENTRY"

        self.profiles[new_id] = {
            "name": name,
            "category": category,
            "role": role,
            "id_card_number": id_card_number,
            "clearance": clearance,
            "notes": notes,
            "color_hex": color_hex
        }

        # Prepare normalized 120x120 grayscale samples
        processed = []
        for img in face_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            resized = cv2.resize(gray, (120, 120))
            processed.append(resized)
            # Add slight augmentations for robustness
            processed.append(cv2.flip(resized, 1))

        if not self.is_trained:
            self.recognizer.train(processed, np.array([new_id] * len(processed), dtype=np.int32))
            self.is_trained = True
        else:
            self.recognizer.update(processed, np.array([new_id] * len(processed), dtype=np.int32))

        self.save_database()
        return new_id

    def identify(self, face_crop: np.ndarray) -> FRSResult:
        """
        Identify a detected face crop against the border security watchlist.

        Args:
            face_crop: BGR or grayscale cropped face image.

        Returns:
            FRSResult with match identity, category, and match confidence.
        """
        if not self.is_trained or face_crop is None or face_crop.size == 0:
            return FRSResult(
                is_identified=False,
                label_id=-1,
                name="Unidentified Face",
                category="UNKNOWN",
                role="Unregistered",
                confidence_score=0.0,
                distance=999.0,
                color_hex="#9E9E9E",
                notes="FRS model not trained or invalid crop"
            )

        try:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
            resized = cv2.resize(gray, (120, 120))
            resized = cv2.equalizeHist(resized)  # Robust to illumination

            label, distance = self.recognizer.predict(resized)

            if distance <= self.confidence_threshold and label in self.profiles:
                # Calibrated confidence for LBPH in surveillance video [0 .. threshold] -> [99% .. 60%]
                conf_pct = round(max(55.0, 99.0 - (distance / self.confidence_threshold) * 42.0), 1)
                profile = self.profiles[label]
                photo_file = os.path.join(self.DEFAULT_DB_DIR, "faces", f"{label}.jpg")
                p_path = photo_file if os.path.exists(photo_file) else None
                return FRSResult(
                    is_identified=True,
                    label_id=label,
                    name=profile["name"],
                    category=profile["category"],
                    role=profile["role"],
                    confidence_score=conf_pct,
                    distance=round(distance, 1),
                    color_hex=profile.get("color_hex", "#4CAF50"),
                    id_card_number=profile.get("id_card_number", f"DEF-ID-{label}"),
                    clearance=profile.get("clearance", "VERIFIED CLEARANCE"),
                    photo_path=p_path,
                    notes=profile.get("notes", "")
                )
            else:
                conf_pct = round(max(5.0, min(45.0, (1.0 - (distance / 200.0)) * 45.0)), 1)
                return FRSResult(
                    is_identified=False,
                    label_id=-1,
                    name="Unknown Individual",
                    category="UNKNOWN",
                    role="Unregistered Person",
                    confidence_score=conf_pct,
                    distance=round(distance, 1),
                    color_hex="#FFC107",
                    id_card_number="UNREGISTERED-CIVILIAN",
                    clearance="NO CLEARANCE ON RECORD",
                    photo_path=None,
                    notes="Face detected on CCTV; no registered defense or watchlist record found."
                )

        except Exception as e:
            return FRSResult(
                is_identified=False,
                label_id=-1,
                name="Processing Error",
                category="UNKNOWN",
                role="Error",
                confidence_score=0.0,
                distance=999.0,
                color_hex="#9E9E9E",
                notes=str(e)
            )

    def save_database(self):
        """Save profiles and model to disk."""
        try:
            with open(self.PROFILES_PATH, "w") as f:
                json.dump(self.profiles, f, indent=2)
            self.recognizer.write(self.MODEL_PATH)
        except Exception as e:
            print(f"Warning: Failed to persist FRS database: {e}")

    def list_watchlist(self) -> List[Dict]:
        """Return all enrolled profiles as a clean list for UI table display."""
        records = []
        for lid, prof in self.profiles.items():
            records.append({
                "ID": f"FRS-{lid}",
                "Name": prof["name"],
                "Category": prof["category"],
                "Designation / Role": prof["role"],
                "Notes": prof.get("notes", "")
            })
        return records
