from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from copy import deepcopy
from datetime import date, datetime, timezone
from types import MappingProxyType
from typing import Any, Dict, List, Optional

from al01.database import Database


class MemoryManager:
    """Local-first persistence manager with throttled Firestore replication + SQLite DB.

    All writes go to local JSON **first** (fast, never blocks).  Firestore
    replication is **coalesced**: memory events are batched and state is synced
    at most once per ``_FIRESTORE_SYNC_INTERVAL`` seconds so we never exceed
    Firestore Spark free-tier quotas (20 K writes / day).

    v2.0: Daily state backups, exponential retry, file logging.
    """

    _FIRESTORE_SYNC_INTERVAL: float = 30.0  # seconds between Firestore syncs
    _FIRESTORE_MAX_RETRIES: int = 3
    _BACKUP_INTERVAL: float = 86400.0  # 24 hours

    # Event types that are local-only (never sent to Firestore)
    _LOCAL_ONLY_EVENTS: frozenset[str] = frozenset({
        "pulse", "loop_cycle",
    })

    def __init__(
        self,
        data_dir: str = ".",
        credential_path: Optional[str] = "ServiceAccountKey.json",
        database: Optional[Database] = None,
    ) -> None:
        self._data_dir = data_dir
        self._state_fallback_path = os.path.join(data_dir, "state.json")
        self._memory_fallback_path = os.path.join(data_dir, "memory.json")
        self._lock = threading.RLock()
        self._logger = logging.getLogger("al01.memory")

        # SQLite database (lightweight local persistence)
        self._database = database or Database(db_path=os.path.join(data_dir, "al01.db"))

        self._db: Any = None
        self._firestore: Any = None
        self._use_firestore = False

        # Firestore throttle state
        self._fs_pending_events: List[Dict[str, Any]] = []
        self._fs_pending_state: Optional[Dict[str, Any]] = None
        self._fs_last_sync: float = 0.0
        self._fs_lock = threading.Lock()

        # Daily write counter (resets at midnight UTC)
        self._fs_writes_today: int = 0
        self._fs_writes_date: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Daily backup tracking
        self._backup_dir = os.path.join(data_dir, "backups")
        os.makedirs(self._backup_dir, exist_ok=True)
        self._last_backup: float = 0.0

        # File logger (al01.log)
        self._file_logger = logging.getLogger("al01.file")
        log_path = os.path.join(data_dir, "al01.log")
        if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == os.path.abspath(log_path)
                   for h in self._file_logger.handlers):
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s"))
            self._file_logger.addHandler(fh)
            self._file_logger.setLevel(logging.INFO)

        self._init_firestore(credential_path)

    def firestore_enabled(self) -> bool:
        return self._use_firestore

    # ------------------------------------------------------------------
    # Throttled Firestore replication
    # ------------------------------------------------------------------

    def _fs_enqueue_event(self, event: Dict[str, Any]) -> None:
        """Queue a memory event for the next Firestore sync."""
        with self._fs_lock:
            self._fs_pending_events.append(event)
        self._fs_maybe_flush()

    def _fs_enqueue_state(self, state: Dict[str, Any]) -> None:
        """Mark state as dirty for the next Firestore sync (last-write-wins)."""
        with self._fs_lock:
            self._fs_pending_state = state
        self._fs_maybe_flush()

    def _fs_maybe_flush(self) -> None:
        """Flush to Firestore if enough time has elapsed since the last sync."""
        now = time.monotonic()
        with self._fs_lock:
            if now - self._fs_last_sync < self._FIRESTORE_SYNC_INTERVAL:
                return
            events = self._fs_pending_events
            self._fs_pending_events = []
            state = self._fs_pending_state
            self._fs_pending_state = None
            self._fs_last_sync = now

        if events or state:
            t = threading.Thread(
                target=self._fs_flush_batch, args=(events, state), daemon=True,
            )
            t.start()

    def _fs_flush_batch(
        self, events: List[Dict[str, Any]], state: Optional[Dict[str, Any]],
    ) -> None:
        """Background batch write with exponential backoff retry."""
        write_count = 0
        for attempt in range(1, self._FIRESTORE_MAX_RETRIES + 1):
            try:
                batch = self._db.batch()

                if state:
                    safe = self._firestore_safe(state)
                    batch.set(self._state_ref(), safe)
                    write_count += 1

                if events:
                    ref = self._memory_entries_ref()
                    for evt in events:
                        safe = self._firestore_safe(evt)
                        doc_ref = ref.document()
                        batch.set(doc_ref, safe)
                        write_count += 1

                if write_count:
                    batch.commit()

                self._fs_record_writes(write_count)
                self._file_logger.info("[FIRESTORE] Batch sync ok: %d writes", write_count)
                return  # success
            except Exception as exc:
                wait = 2 ** (attempt - 1)  # 1s, 2s, 4s
                self._logger.warning(
                    "[FIRESTORE] Attempt %d/%d failed (%d ops): %s — retrying in %ds",
                    attempt, self._FIRESTORE_MAX_RETRIES, write_count, exc, wait,
                )
                self._file_logger.warning(
                    "[FIRESTORE] Retry %d/%d: %s", attempt, self._FIRESTORE_MAX_RETRIES, exc,
                )
                if attempt < self._FIRESTORE_MAX_RETRIES:
                    time.sleep(wait)
                    write_count = 0  # reset for retry
                else:
                    self._logger.error("[FIRESTORE] All retries exhausted — batch dropped")
                    self._file_logger.error("[FIRESTORE] All retries exhausted — batch dropped")

    def _fs_record_writes(self, count: int) -> None:
        """Track daily Firestore writes and log running total."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._fs_lock:
            if today != self._fs_writes_date:
                self._logger.info(
                    "[FIRESTORE] Daily write counter reset (yesterday=%s total=%d)",
                    self._fs_writes_date, self._fs_writes_today,
                )
                self._fs_writes_today = 0
                self._fs_writes_date = today
            self._fs_writes_today += count
            daily_total = self._fs_writes_today
        self._logger.info(
            "[FIRESTORE] Sync complete: +%d writes | today=%d / 20000 (%.1f%%)",
            count, daily_total, daily_total / 200,  # 200 = 20000/100
        )

    @property
    def firestore_writes_today(self) -> int:
        """Current daily Firestore write count (resets at midnight UTC)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._fs_lock:
            if today != self._fs_writes_date:
                return 0
            return self._fs_writes_today

    def flush_firestore(self) -> None:
        """Force an immediate Firestore sync (called on shutdown)."""
        if not self._use_firestore:
            return
        with self._fs_lock:
            events = self._fs_pending_events
            self._fs_pending_events = []
            state = self._fs_pending_state
            self._fs_pending_state = None
            self._fs_last_sync = time.monotonic()
        if events or state:
            self._fs_flush_batch(events, state)

    def maybe_daily_backup(self) -> Optional[str]:
        """Export full state to a timestamped backup file if 24h have passed.

        Returns the backup path if written, else None.
        """
        now = time.monotonic()
        if now - self._last_backup < self._BACKUP_INTERVAL:
            return None

        self._last_backup = now
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = os.path.join(self._backup_dir, f"state_backup_{timestamp}.json")

        try:
            state = self._load_local_state()
            memory = self._load_local_memory_entries()
            backup = {
                "timestamp": timestamp,
                "state": state,
                "memory_entries_count": len(memory),
                "memory_entries": memory[-100:],  # last 100 entries to keep size down
            }
            with open(backup_path, "w", encoding="utf-8") as fh:
                json.dump(backup, fh, indent=2, default=str)
            self._logger.info("[BACKUP] Daily backup written: %s", backup_path)
            self._file_logger.info("[BACKUP] Daily backup: %s", backup_path)
            return backup_path
        except Exception as exc:
            self._logger.error("[BACKUP] Failed to write daily backup: %s", exc)
            self._file_logger.error("[BACKUP] Failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Write / Read
    # ------------------------------------------------------------------

    def write_memory(self, event: Dict[str, Any]) -> None:
        # Always write to local JSON first (instant)
        with self._lock:
            local_payload = self._normalize_event(event, source="local")
            local_entries = self._load_local_memory_entries()
            local_entries.append(local_payload)
            self._save_local_memory_entries(local_entries)

        # Only replicate significant events to Firestore (skip noise)
        if self._use_firestore:
            event_type = str(event.get("event_type") or event.get("type") or "event")
            if event_type not in self._LOCAL_ONLY_EVENTS:
                cloud_payload = self._normalize_event(event, source="cloud")
                self._fs_enqueue_event(cloud_payload)

    def read_memory(self, limit: int) -> List[Dict[str, Any]]:
        safe_limit = max(0, int(limit))
        if safe_limit == 0:
            return []

        with self._lock:

            entries = self._load_local_memory_entries()
            return entries[-safe_limit:]

    def search_memory(
        self,
        keyword: str,
        limit: int = 20,
        since_timestamp: Optional[str] = None,
        contains_all: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search across user_input, response, and mood fields.

        Parameters
        ----------
        keyword : str
            Primary search term.
        limit : int
            Maximum results to return.
        since_timestamp : str, optional
            ISO-8601 lower bound (inclusive).
        contains_all : list[str], optional
            Additional terms that must *all* appear.

        Returns structured results:
        ``{id, timestamp, user_input, response, mood, state}``
        """
        # Prefer SQLite structured search when database is available
        results = self._database.search_interactions(
            keyword=keyword,
            limit=limit,
            since_timestamp=since_timestamp,
            contains_all=contains_all,
        )
        if results:
            return [
                {
                    "id": r.get("id"),
                    "timestamp": r.get("timestamp"),
                    "user_input": r.get("user_input"),
                    "response": r.get("response"),
                    "mood": r.get("mood"),
                    "state": r.get("organism_state"),
                }
                for r in results
            ]

        # Fallback: search local memory entries
        keyword_lower = keyword.lower()
        safe_limit = max(1, int(limit))

        with self._lock:
            candidates = self._load_local_memory_entries()

        matches: List[Dict[str, Any]] = []
        for entry in candidates:
            serialized = json.dumps(entry, default=str).lower()
            if keyword_lower in serialized:
                matches.append(entry)
                if len(matches) >= safe_limit:
                    break
        return matches

    def memory_size(self) -> int:
        """Return the total number of memory entries."""
        with self._lock:
            return len(self._load_local_memory_entries())

    @property
    def database(self) -> Database:
        """Expose the underlying SQLite database."""
        return self._database

    def load_state(self) -> Dict[str, Any]:
        """Load state from local JSON (fast path). Falls back to Firestore only
        when no local state file exists yet (initial clone / fresh machine)."""
        with self._lock:
            raw = self._load_local_state()

            # If no local state, try Firestore as a one-time recovery
            if not raw and self._use_firestore:
                try:
                    doc = self._state_ref().get()
                    if doc.exists:
                        raw = self._sanitize_dict(doc.to_dict() or {})
                        # Persist locally so future loads are instant
                        if raw:
                            self._save_local_state(raw)
                except Exception as exc:
                    self._log_fallback_error(exc)

            # Verify checksum if present
            stored_checksum = raw.pop("checksum", None)
            if stored_checksum is not None:
                body = json.dumps(
                    {k: v for k, v in sorted(raw.items()) if k != "checksum"},
                    sort_keys=True,
                )
                expected = hashlib.sha256(body.encode()).hexdigest()
                if stored_checksum != expected:
                    self._logger.warning(
                        "State checksum mismatch (stored=%s expected=%s). "
                        "Returning raw state anyway.",
                        stored_checksum,
                        expected,
                    )
            return raw

    def save_state(self, state: Dict[str, Any]) -> None:
        with self._lock:
            clean_state = self._sanitize_dict(state)
            # Inject integrity metadata
            clean_state["state_version"] = clean_state.get("state_version", 0)
            body = json.dumps(
                {k: v for k, v in sorted(clean_state.items()) if k != "checksum"},
                sort_keys=True,
            )
            clean_state["checksum"] = hashlib.sha256(body.encode()).hexdigest()

            # Always write to local JSON first (instant, never blocks)
            self._save_local_state(clean_state)

        # Queue for throttled Firestore replication
        if self._use_firestore:
            self._fs_enqueue_state(clean_state)

    def _init_firestore(self, credential_path: Optional[str]) -> None:
        if credential_path is None:
            self._use_firestore = False
            return
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore, initialize_app

            cert_path = credential_path

            if not firebase_admin._apps:
                cred = credentials.Certificate(cert_path)
                initialize_app(cred)

            self._db = firestore.client()
            self._firestore = firestore
            self._use_firestore = True
        except Exception as exc:
            self._use_firestore = False
            self._log_fallback_error(exc)

    def _memory_entries_ref(self) -> Any:
        return self._db.collection("al01_memory").document("entries").collection("entries")

    def _state_ref(self) -> Any:
        return self._db.collection("al01_state").document("core")

    def _normalize_event(self, event: Dict[str, Any], source: str) -> Dict[str, Any]:
        clean_event = self._sanitize_dict(event)
        event_type = str(clean_event.get("event_type") or clean_event.get("type") or "event")
        payload = clean_event.get("payload")

        if payload is None:
            payload = {
                key: value
                for key, value in clean_event.items()
                if key not in {"timestamp", "event_type", "type", "payload", "source"}
            }

        clean_payload = self._sanitize_value(payload)

        return {
            "timestamp": clean_event.get("timestamp") or self._utc_now(),
            "event_type": event_type,
            "payload": clean_payload,
            "source": source,
        }

    def _load_local_state(self) -> Dict[str, Any]:
        if not os.path.exists(self._state_fallback_path):
            return {}
        try:
            with open(self._state_fallback_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, dict):
                return self._sanitize_dict(raw)
        except Exception as exc:
            self._log_fallback_error(exc)
        return {}

    def _save_local_state(self, state: Dict[str, Any]) -> None:
        self._ensure_parent_dir(self._state_fallback_path)
        data = json.dumps(self._sanitize_dict(state), indent=2)
        # Atomic write: tmp file + rename
        dir_part = os.path.dirname(self._state_fallback_path) or "."
        fd, tmp_path = tempfile.mkstemp(dir=dir_part, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(data)
            os.replace(tmp_path, self._state_fallback_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def _load_local_memory_entries(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self._memory_fallback_path):
            return []
        try:
            with open(self._memory_fallback_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)

            if isinstance(raw, dict):
                entries = raw.get("entries", [])
            elif isinstance(raw, list):
                entries = raw
            else:
                entries = []

            if not isinstance(entries, list):
                return []
            return [self._sanitize_dict(item) for item in entries if isinstance(item, dict)]
        except Exception as exc:
            self._log_fallback_error(exc)
            return []

    def _save_local_memory_entries(self, entries: List[Dict[str, Any]]) -> None:
        self._ensure_parent_dir(self._memory_fallback_path)
        clean_entries = [self._sanitize_dict(item) for item in entries if isinstance(item, dict)]
        with open(self._memory_fallback_path, "w", encoding="utf-8") as fh:
            json.dump({"entries": clean_entries}, fh, indent=2)

    def _firestore_safe(self, value: Dict[str, Any]) -> Dict[str, Any]:
        """JSON round-trip gate: guarantees only Firestore-compatible primitives."""
        return json.loads(json.dumps(value, default=str))

    def _sanitize_dict(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, MappingProxyType):
            value = dict(value)
        if not isinstance(value, dict):
            return {}
        return {
            str(key): self._sanitize_value(item)
            for key, item in value.items()
        }

    def _sanitize_value(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return value
        if isinstance(value, str):
            return value
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, enum.Enum):
            return value.value
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, MappingProxyType):
            return {str(k): self._sanitize_value(v) for k, v in value.items()}
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {str(k): self._sanitize_value(v) for k, v in dataclasses.asdict(value).items()}
        if isinstance(value, dict):
            return {str(k): self._sanitize_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [self._sanitize_value(item) for item in value]
        return str(value)

    def _ensure_parent_dir(self, path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _log_fallback_error(self, exc: Exception) -> None:
        self._logger.warning("Firestore unavailable; using local JSON fallback: %s", exc)
