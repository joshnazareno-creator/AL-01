"""AL-01 v1.2 — lightweight SQLite persistence layer.

Tables
------
- ``interactions`` — structured interaction log (survives restarts)
- ``growth_snapshots`` — periodic growth metric snapshots
- ``metadata`` — key/value store for identity, version, first-run timestamp
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("al01.database")

_SCHEMA_VERSION = 1

_CREATE_INTERACTIONS = """
CREATE TABLE IF NOT EXISTS interactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    user_input  TEXT    NOT NULL,
    response    TEXT    NOT NULL,
    mood        TEXT,
    organism_state TEXT,
    extra       TEXT
);
"""

_CREATE_GROWTH_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS growth_snapshots (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT    NOT NULL,
    interaction_count INTEGER NOT NULL DEFAULT 0,
    memory_size       INTEGER NOT NULL DEFAULT 0,
    awareness         REAL    NOT NULL DEFAULT 0.0,
    evolution_count   INTEGER NOT NULL DEFAULT 0,
    version           TEXT
);
"""

_CREATE_METADATA = """
CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class Database:
    """Thread-safe SQLite wrapper for AL-01 persistent storage."""

    def __init__(self, db_path: str = "al01.db") -> None:
        self._db_path = os.path.abspath(db_path)
        self._lock = threading.RLock()
        self._ensure_schema()
        logger.info("[DB] SQLite database ready at %s", self._db_path)

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _ensure_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(_CREATE_INTERACTIONS)
                conn.execute(_CREATE_GROWTH_SNAPSHOTS)
                conn.execute(_CREATE_METADATA)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_interactions_ts ON interactions(timestamp);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_growth_ts ON growth_snapshots(timestamp);"
                )
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Interactions
    # ------------------------------------------------------------------

    def write_interaction(
        self,
        *,
        user_input: str,
        response: str,
        mood: Optional[str] = None,
        organism_state: Optional[str] = None,
        timestamp: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Insert an interaction row and return its ``id``."""
        ts = timestamp or _utc_now()
        extra_json = json.dumps(extra, default=str) if extra else None
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """INSERT INTO interactions
                       (timestamp, user_input, response, mood, organism_state, extra)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (ts, user_input, response, mood, organism_state, extra_json),
                )
                conn.commit()
                row_id = cur.lastrowid
            finally:
                conn.close()
        return row_id  # type: ignore[return-value]

    def recent_interactions(self, n: int = 50) -> List[Dict[str, Any]]:
        """Return the most recent *n* interactions (oldest-first)."""
        safe_n = max(1, int(n))
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM interactions ORDER BY id DESC LIMIT ?", (safe_n,)
                ).fetchall()
            finally:
                conn.close()
        # Return oldest-first
        return [_row_to_dict(r) for r in reversed(rows)]

    def search_interactions(
        self,
        keyword: str,
        limit: int = 20,
        since_timestamp: Optional[str] = None,
        contains_all: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Full-text keyword search across user_input, response, mood.

        Parameters
        ----------
        keyword : str
            Primary keyword — matches if it appears in user_input, response, or mood.
        limit : int
            Maximum results.
        since_timestamp : str, optional
            ISO-8601 lower bound on timestamp (inclusive).
        contains_all : list[str], optional
            Additional keywords that must *all* appear (across the three fields combined).
        """
        safe_limit = max(1, int(limit))
        kw = f"%{keyword}%"

        conditions = [
            "(user_input LIKE ? COLLATE NOCASE OR response LIKE ? COLLATE NOCASE OR mood LIKE ? COLLATE NOCASE)"
        ]
        params: list[Any] = [kw, kw, kw]

        if since_timestamp:
            conditions.append("timestamp >= ?")
            params.append(since_timestamp)

        where = " AND ".join(conditions)
        query = f"SELECT * FROM interactions WHERE {where} ORDER BY id DESC LIMIT ?"
        params.append(safe_limit)

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(query, params).fetchall()
            finally:
                conn.close()

        results = [_row_to_dict(r) for r in reversed(rows)]

        # Client-side filter for contains_all (small result set, acceptable overhead)
        if contains_all:
            filtered: List[Dict[str, Any]] = []
            for row in results:
                combined = " ".join(
                    str(row.get(f, "")) for f in ("user_input", "response", "mood")
                ).lower()
                if all(term.lower() in combined for term in contains_all):
                    filtered.append(row)
            results = filtered

        return results

    def interaction_count(self) -> int:
        """Total number of recorded interactions."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT COUNT(*) AS cnt FROM interactions").fetchone()
            finally:
                conn.close()
        return row["cnt"] if row else 0

    def first_interaction_timestamp(self) -> Optional[str]:
        """Timestamp of the very first recorded interaction, or ``None``."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT timestamp FROM interactions ORDER BY id ASC LIMIT 1"
                ).fetchone()
            finally:
                conn.close()
        return row["timestamp"] if row else None

    # ------------------------------------------------------------------
    # Growth snapshots
    # ------------------------------------------------------------------

    def write_growth_snapshot(
        self,
        *,
        interaction_count: int,
        memory_size: int,
        awareness: float,
        evolution_count: int,
        version: str,
        timestamp: Optional[str] = None,
    ) -> int:
        ts = timestamp or _utc_now()
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """INSERT INTO growth_snapshots
                       (timestamp, interaction_count, memory_size, awareness, evolution_count, version)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (ts, interaction_count, memory_size, awareness, evolution_count, version),
                )
                conn.commit()
                row_id = cur.lastrowid
            finally:
                conn.close()
        return row_id  # type: ignore[return-value]

    def latest_growth_snapshot(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM growth_snapshots ORDER BY id DESC LIMIT 1"
                ).fetchone()
            finally:
                conn.close()
        return _row_to_dict(row) if row else None

    def growth_summary(self) -> Dict[str, Any]:
        """Aggregate growth data since the first recorded snapshot."""
        with self._lock:
            conn = self._connect()
            try:
                total_interactions = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM interactions"
                ).fetchone()["cnt"]
                first_ts = conn.execute(
                    "SELECT MIN(timestamp) AS ts FROM growth_snapshots"
                ).fetchone()["ts"]
                latest = conn.execute(
                    "SELECT * FROM growth_snapshots ORDER BY id DESC LIMIT 1"
                ).fetchone()
                snapshot_count = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM growth_snapshots"
                ).fetchone()["cnt"]
            finally:
                conn.close()
        return {
            "total_interactions": total_interactions,
            "first_snapshot_timestamp": first_ts,
            "latest_snapshot": _row_to_dict(latest) if latest else None,
            "snapshot_count": snapshot_count,
        }

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def set_metadata(self, key: str, value: str) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                    (key, value),
                )
                conn.commit()
            finally:
                conn.close()

    def get_metadata(self, key: str) -> Optional[str]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT value FROM metadata WHERE key = ?", (key,)
                ).fetchone()
            finally:
                conn.close()
        return row["value"] if row else None

    def close(self) -> None:
        """No persistent connection to close (connections are per-call)."""
        pass


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    # Parse extra JSON if present
    if "extra" in d and d["extra"] is not None:
        try:
            d["extra"] = json.loads(d["extra"])
        except (json.JSONDecodeError, TypeError):
            pass
    return d


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
