"""
IBVAP — Object Detection Module
Uses YOLOv8 (ultralytics) for person, vehicle, and face detection.
Pretrained model — no custom training required.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Detection:
    """A single detection result."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    class_name: str
    confidence: float
    track_id: Optional[int] = None
    centroid: Tuple[int, int] = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height


# COCO class IDs we care about
PERSON_CLASS_ID = 0
VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CARRIED_OBJECT_IDS = {24: "backpack", 26: "handbag", 28: "suitcase"}
WEAPON_CLASS_IDS = {43: "knife", 76: "scissors", 34: "baseball bat"}
ALL_TARGET_IDS = {PERSON_CLASS_ID: "person", **VEHICLE_CLASS_IDS, **CARRIED_OBJECT_IDS, **WEAPON_CLASS_IDS}


class ObjectDetector:
    """YOLOv8-based detector for persons, vehicles, and faces."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        tracking_method: Optional[str] = None,
    ):
        """
        Initialize the detector.

        Args:
            model_name: YOLOv8 model variant (yolov8n.pt for speed)
            confidence_threshold: Minimum confidence to keep a detection
        """
        # Lazy import to avoid slow startup if not used
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.tracking_method = tracking_method if tracking_method in {"bytetrack", "botsort"} else None
        self._tracking_failed = False

        # Face detector — OpenCV Haar Cascade (lighter than YOLO-face)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.face_detection_enabled = not self.face_cascade.empty()

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single frame.

        Args:
            frame: BGR image (numpy array)

        Returns:
            List of Detection objects for persons, vehicles, and faces
        """
        detections = []

        # --- YOLOv8 Detection (person + vehicle + carried objects) ---
        predict_kwargs = {
            "conf": self.confidence_threshold,
            "iou": 0.40,
            "agnostic_nms": True,
            "verbose": False,
            # Keep CPU inference responsive for the local dashboard. The
            # source frame is already downscaled before it reaches YOLO.
            "imgsz": 512,
            "classes": list(ALL_TARGET_IDS.keys()),
        }
        if self.tracking_method and not self._tracking_failed:
            try:
                results = self.model.track(
                    frame,
                    persist=True,
                    tracker=f"{self.tracking_method}.yaml",
                    **predict_kwargs,
                )
            except Exception as exc:
                # Keep the application usable if a tracker config is missing
                # or the installed Ultralytics version lacks track support.
                self._tracking_failed = True
                print(f"Ultralytics tracking unavailable; using detection fallback: {exc}")
                results = self.model.predict(frame, **predict_kwargs)
        else:
            results = self.model.predict(frame, **predict_kwargs)

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    track_id = None
                    if self.tracking_method and getattr(box, "id", None) is not None:
                        track_id = int(box.id[0])

                    class_name = ALL_TARGET_IDS.get(class_id, f"class_{class_id}")
                    detections.append(Detection(
                        bbox=(x1, y1, x2, y2),
                        class_name=class_name,
                        confidence=confidence,
                        track_id=track_id,
                    ))

        # --- Deduplicate overlapping vehicle/person detections ---
        detections = self._deduplicate_detections(detections)

        # --- Face Detection (OpenCV Haar Cascade) ---
        if self.face_detection_enabled:
            face_detections = self._detect_faces(frame)
            detections.extend(face_detections)

        return detections

    def _deduplicate_detections(self, detections: List[Detection]) -> List[Detection]:
        """
        Suppresses duplicate and overlapping detections of the same physical vehicle or person.
        Ensures a single vehicle is never detected twice (e.g. as both car & truck or split box).
        """
        if len(detections) <= 1:
            return detections

        vehicle_classes = set(VEHICLE_CLASS_IDS.values())

        # Separate vehicles and others
        vehicles = [d for d in detections if d.class_name in vehicle_classes]
        others = [d for d in detections if d.class_name not in vehicle_classes]

        # Sort vehicles by confidence descending
        vehicles.sort(key=lambda d: d.confidence, reverse=True)
        kept_vehicles: List[Detection] = []

        for v in vehicles:
            is_duplicate = False
            boxA = v.bbox
            areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])

            for kept in kept_vehicles:
                boxB = kept.bbox
                areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

                # Calculate intersection
                xA = max(boxA[0], boxB[0])
                yA = max(boxA[1], boxB[1])
                xB = min(boxA[2], boxB[2])
                yB = min(boxA[3], boxB[3])

                inter_w = max(0, xB - xA)
                inter_h = max(0, yB - yA)
                inter_area = inter_w * inter_h

                if inter_area > 0:
                    union_area = areaA + areaB - inter_area
                    iou = inter_area / float(union_area + 1e-6)
                    containment = inter_area / float(min(areaA, areaB) + 1e-6)

                    # Suppress if moderate IoU or high containment
                    if iou > 0.35 or containment > 0.55:
                        is_duplicate = True
                        break

            if not is_duplicate:
                kept_vehicles.append(v)

        return kept_vehicles + others

    def _detect_faces(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect faces using OpenCV Haar Cascade.
        Detection only — no identity matching.

        Args:
            frame: BGR image

        Returns:
            List of face Detection objects
        """
        if self.face_cascade.empty():
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        detections = []
        for (x, y, w, h) in faces:
            detections.append(Detection(
                bbox=(x, y, x + w, y + h),
                class_name="face",
                confidence=0.8,  # Haar doesn't give confidence; use fixed value
            ))

        return detections

    def get_persons(self, detections: List[Detection]) -> List[Detection]:
        """Filter detections to persons only."""
        return [d for d in detections if d.class_name == "person"]

    def get_vehicles(self, detections: List[Detection]) -> List[Detection]:
        """Filter detections to vehicles only."""
        vehicle_classes = set(VEHICLE_CLASS_IDS.values())
        return [d for d in detections if d.class_name in vehicle_classes]

    def get_faces(self, detections: List[Detection]) -> List[Detection]:
        """Filter detections to faces only."""
        return [d for d in detections if d.class_name == "face"]

    def get_carried_objects(self, detections: List[Detection]) -> List[Detection]:
        """Filter detections to carried objects (backpack, handbag, suitcase)."""
        carried_classes = set(CARRIED_OBJECT_IDS.values())
        return [d for d in detections if d.class_name in carried_classes]

    def get_weapons(self, detections: List[Detection]) -> List[Detection]:
        """Filter detections to weapon/suspicious objects (knife, scissors, bat)."""
        weapon_classes = set(WEAPON_CLASS_IDS.values())
        return [d for d in detections if d.class_name in weapon_classes]
