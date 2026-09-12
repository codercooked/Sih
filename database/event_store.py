"""
IBVAP — Event Store (SQLite Database + Snapshot Storage)
Local event logging with no identity storage.

Schema stores: entity_id, timestamps, zone, threat score, behavior tags,
snapshot path, vehicle plate, and person count. Never stores faces, names,
or identity embeddings.
"""

import os
import json
import csv
import sqlite3
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple


class EventStore:
    """
    SQLite-based event logging for IBVAP alerts and detections.
    Includes snapshot image storage to disk.
    """

    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entity_id TEXT NOT NULL,
        first_detected TIMESTAMP NOT NULL,
        last_updated TIMESTAMP NOT NULL,
        zone_name TEXT,
        threat_score INTEGER DEFAULT 0,
        threat_level TEXT DEFAULT 'low',
        behaviour_tags TEXT,
        snapshot_path TEXT,
        vehicle_plate TEXT,
        num_persons_in_zone INTEGER DEFAULT 0,
        speed_category TEXT,
        posture TEXT,
        loitering_duration_sec REAL DEFAULT 0,
        notes TEXT,
        status TEXT DEFAULT 'new',
        acknowledged_by TEXT,
        acknowledged_at TIMESTAMP,
        resolved_at TIMESTAMP,
        operator_notes TEXT
    )
    """

    CREATE_INDEX_SQL = [
        "CREATE INDEX IF NOT EXISTS idx_entity_id ON events(entity_id)",
        "CREATE INDEX IF NOT EXISTS idx_threat_score ON events(threat_score DESC)",
        "CREATE INDEX IF NOT EXISTS idx_last_updated ON events(last_updated DESC)",
    ]
    INCIDENT_STATUSES = ("new", "acknowledged", "investigating", "confirmed", "resolved", "dismissed")

    def __init__(self, db_path: str = "./ibvap_events.db", output_dir: str = "./alerts/"):
        """
        Args:
            db_path: Path to SQLite database file
            output_dir: Directory for snapshot images
        """
        self.db_path = db_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create database and tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(self.CREATE_TABLE_SQL)
        for idx_sql in self.CREATE_INDEX_SQL:
            cursor.execute(idx_sql)
        existing_columns = {row[1] for row in cursor.execute("PRAGMA table_info(events)").fetchall()}
        migrations = {
            "status": "ALTER TABLE events ADD COLUMN status TEXT DEFAULT 'new'",
            "acknowledged_by": "ALTER TABLE events ADD COLUMN acknowledged_by TEXT",
            "acknowledged_at": "ALTER TABLE events ADD COLUMN acknowledged_at TIMESTAMP",
            "resolved_at": "ALTER TABLE events ADD COLUMN resolved_at TIMESTAMP",
            "operator_notes": "ALTER TABLE events ADD COLUMN operator_notes TEXT",
        }
        for column, sql in migrations.items():
            if column not in existing_columns:
                cursor.execute(sql)
        cursor.execute("UPDATE events SET status = 'new' WHERE status IS NULL OR status = ''")
        conn.commit()
        conn.close()

    def log_event(
        self,
        entity_id: str,
        zone_name: str = "",
        threat_score: int = 0,
        threat_level: str = "low",
        behaviour_tags: List[str] = None,
        snapshot_path: str = "",
        vehicle_plate: str = "",
        num_persons_in_zone: int = 0,
        speed_category: str = "",
        posture: str = "",
        loitering_duration_sec: float = 0.0,
        notes: str = "",
        first_detected: Optional[str] = None,
    ) -> int:
        """
        Log an event to the database.

        Args:
            entity_id: Entity identifier (UNKNOWN-XX)
            zone_name: Restricted zone name
            threat_score: 0-100
            threat_level: "low", "medium", "high", "critical"
            behaviour_tags: List of behavior tags
            snapshot_path: Path to snapshot image
            vehicle_plate: ANPR plate text
            num_persons_in_zone: Current person count in zone
            speed_category: "stationary", "walking", "running"
            posture: "standing", "crouching", "climbing"
            loitering_duration_sec: Seconds entity has been loitering
            notes: Free-form notes
            first_detected: Override first_detected timestamp

        Returns:
            Row ID of inserted event
        """
        now = datetime.now().isoformat()
        tags_str = json.dumps(behaviour_tags) if behaviour_tags else "[]"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO events (
                entity_id, first_detected, last_updated, zone_name,
                threat_score, threat_level, behaviour_tags, snapshot_path,
                vehicle_plate, num_persons_in_zone, speed_category,
                posture, loitering_duration_sec, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entity_id,
                first_detected or now,
                now,
                zone_name,
                threat_score,
                threat_level,
                tags_str,
                snapshot_path,
                vehicle_plate,
                num_persons_in_zone,
                speed_category,
                posture,
                loitering_duration_sec,
                notes,
            ),
        )
        row_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return row_id

    def save_snapshot(
        self,
        frame: np.ndarray,
        entity_id: str,
        quality: int = 85,
    ) -> str:
        """
        Save a frame snapshot for an alert.

        Args:
            frame: BGR image
            entity_id: Entity identifier for filename
            quality: JPEG quality (0-100)

        Returns:
            Path to saved snapshot
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{entity_id}_{timestamp}.jpg"
        filepath = os.path.join(self.output_dir, filename)

        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return filepath

    def get_recent_events(self, limit: int = 50) -> List[Dict]:
        """
        Get most recent events, newest first.

        Args:
            limit: Maximum number of events to return

        Returns:
            List of event dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM events ORDER BY last_updated DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()
        conn.close()

        events = []
        for row in rows:
            event = dict(row)
            # Parse behaviour_tags JSON
            try:
                event["behaviour_tags"] = json.loads(event.get("behaviour_tags", "[]"))
            except (json.JSONDecodeError, TypeError):
                event["behaviour_tags"] = []
            events.append(event)

        return events

    def get_events_by_entity(self, entity_id: str) -> List[Dict]:
        """Get all events for a specific entity."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM events WHERE entity_id = ? ORDER BY last_updated DESC",
            (entity_id,),
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_event_count(self) -> int:
        """Get total number of events."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM events")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def update_incident(
        self,
        event_id: int,
        status: str,
        operator_notes: str = "",
        acknowledged_by: str = "operator",
    ) -> bool:
        """Update the operator workflow state for an incident."""
        if status not in self.INCIDENT_STATUSES:
            raise ValueError(f"Unsupported incident status: {status}")
        now = datetime.now().isoformat()
        acknowledged_at = now if status in {"acknowledged", "investigating", "confirmed", "resolved"} else None
        resolved_at = now if status in {"resolved", "dismissed"} else None
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """UPDATE events
               SET status = ?, operator_notes = ?, acknowledged_by = ?,
                   acknowledged_at = COALESCE(acknowledged_at, ?),
                   resolved_at = ?, last_updated = ?
               WHERE id = ?""",
            (status, operator_notes, acknowledged_by if acknowledged_at else None,
             acknowledged_at, resolved_at, now, event_id),
        )
        changed = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return changed

    def get_incident_counts(self) -> Dict[str, int]:
        """Return counts grouped by incident workflow status."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT status, COUNT(*) FROM events GROUP BY status")
        counts = {status: 0 for status in self.INCIDENT_STATUSES}
        counts.update({status or "new": count for status, count in cursor.fetchall()})
        conn.close()
        return counts

    def get_high_threat_events(self, min_score: int = 60) -> List[Dict]:
        """Get events with threat score above threshold."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM events WHERE threat_score >= ? ORDER BY threat_score DESC",
            (min_score,),
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def export_csv(self, filepath: str = "./ibvap_events_export.csv") -> str:
        """
        Export all events to CSV file.

        Args:
            filepath: Output CSV path

        Returns:
            Path to exported CSV
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM events ORDER BY last_updated DESC")
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(rows)

        return filepath

    def clear_all(self):
        """Clear all events and their snapshots from the configured output directory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT snapshot_path FROM events WHERE snapshot_path IS NOT NULL AND snapshot_path != ''")
        snapshot_paths = [row[0] for row in cursor.fetchall()]
        cursor.execute("DELETE FROM events")
        conn.commit()
        conn.close()

        # Only remove files that resolve inside this store's snapshot folder.
        # This prevents a malformed database path from deleting arbitrary files.
        output_root = os.path.realpath(self.output_dir)
        for snapshot_path in snapshot_paths:
            try:
                resolved = os.path.realpath(snapshot_path)
                if os.path.commonpath([output_root, resolved]) == output_root and os.path.isfile(resolved):
                    os.remove(resolved)
            except (OSError, ValueError):
                continue
