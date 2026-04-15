"""
database.py
SQLite storage for evaluation results.

Schema:
    evaluations (
        id          INTEGER PRIMARY KEY,
        method      TEXT,
        camera_id   INTEGER,
        precision   REAL,
        recall      REAL,
        f1          REAL,
        accuracy    REAL,
        fps         REAL,
        avg_latency REAL,
        n_frames    INTEGER,
        timestamp   TEXT
    )
"""

import sqlite3
import os
from datetime import datetime
from typing import Dict, Any, List

from config import settings


class Database:
    def __init__(self):
        self._path = settings.DB_PATH
        self._init_db()

    # ──────────────────────────────────────────
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    method      TEXT    NOT NULL,
                    camera_id   INTEGER DEFAULT 0,
                    precision   REAL,
                    recall      REAL,
                    f1          REAL,
                    accuracy    REAL,
                    fps         REAL,
                    avg_latency REAL,
                    n_frames    INTEGER,
                    timestamp   TEXT
                )
            """)
            conn.commit()

    # ──────────────────────────────────────────
    def save_evaluation(
        self, method: str, metrics: Dict[str, Any], camera_id: int = 0
    ) -> int:
        """Insert an evaluation result row. Returns the new row id."""
        if "error" in metrics:
            return -1

        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO evaluations
                    (method, camera_id, precision, recall, f1, accuracy,
                     fps, avg_latency, n_frames, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    method,
                    camera_id,
                    metrics.get("precision"),
                    metrics.get("recall"),
                    metrics.get("f1"),
                    metrics.get("accuracy"),
                    metrics.get("fps"),
                    metrics.get("avg_latency_ms"),
                    metrics.get("n_frames"),
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
            return cur.lastrowid

    # ──────────────────────────────────────────
    def get_all_evaluations(self) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM evaluations ORDER BY timestamp DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_latest_per_method(self) -> List[Dict]:
        """Return the most recent evaluation for each method."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM evaluations
                WHERE id IN (
                    SELECT MAX(id) FROM evaluations GROUP BY method
                )
                ORDER BY method
            """).fetchall()
        return [dict(r) for r in rows]
