from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import logging
import logging.handlers
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
    v3.21: Bounded memory — cap at 1000 entries, SQLite long-term archive,
           atomic writes, write-throttling, auto-trim at 10 MB.
    """

    _FIRESTORE_SYNC_INTERVAL: float = 30.0  # seconds between Firestore syncs
    _FIRESTORE_MAX_RETRIES: int = 3
    _BACKUP_INTERVAL: float = 86400.0  # 24 hours
    _MEMORY_MAX_ENTRIES: int = 1000  # v3.21: rolling window (was 5000)

    # Event types that are local-only (never sent to Firestore)
    _LOCAL_ONLY_EVENTS: frozenset[str] = frozenset({
        "pulse", "loop_cycle",
    })

    # v3.21: Write throttling — batch this many memory writes in-memory
    # before flushing to disk.  Reduces I/O from "every write" to periodic.
    _MEMORY_FLUSH_INTERVAL: int = 10  # flush to disk every N writes
    _MEMORY_FLUSH_SECONDS: float = 60.0  # …or every 60 s, whichever comes first

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
        self._backup_dir = os.path.join(os.path.abspath(data_dir), "backups")
        os.makedirs(self._backup_dir, exist_ok=True)
        self._last_backup: float = 0.0

        # File logger (al01.log) — v3.22: absolute path + rotation
        self._file_logger = logging.getLogger("al01.file")
        log_path = os.path.join(os.path.abspath(data_dir), "al01.log")
        if not any(isinstance(h, logging.handlers.RotatingFileHandler) and getattr(h, 'baseFilename', '') == os.path.abspath(log_path)
                   for h in self._file_logger.handlers):
            fh = logging.handlers.RotatingFileHandler(
                log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8",
            )
            fh.setFormatter(logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s"))
            self._file_logger.addHandler(fh)
            self._file_logger.setLevel(logging.INFO)

        self._init_firestore(credential_path)

        # In-memory entry cache — avoids repeated multi-GB json.load calls.
        # Populated lazily on first access; invalidated only by
        # _save_local_memory_entries (the sole write path).
        self._entries_cache: Optional[List[Dict[str, Any]]] = None
        self._entry_count: int = 0

        # v3.21: Write-throttle bookkeeping
        self._pending_writes: int = 0  # writes since last disk flush
        self._last_flush_time: float = time.monotonic()

        # Startup trim: if memory.json exceeds a sane threshold, truncate
        # to _MEMORY_MAX_ENTRIES to break the "too-big-to-load" deadlock.
        self._startup_trim_if_needed()

    def firestore_enabled(self) -> bool:
        return self._use_firestore

    def flush_memory(self) -> None:
        """Flush any pending in-memory entries to disk.

        Call this on shutdown to ensure no buffered writes are lost.
        """
        with self._lock:
            if self._pending_writes > 0 and self._entries_cache is not None:
                self._save_local_memory_entries(self._entries_cache)
                self._pending_writes = 0
                self._last_flush_time = time.monotonic()

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
            with self._lock:
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
            # Trim to cap — keep newest entries only (v3.21: 1000 rolling window)
            if len(local_entries) > self._MEMORY_MAX_ENTRIES:
                local_entries = local_entries[-self._MEMORY_MAX_ENTRIES:]
            # Update in-memory cache immediately
            self._entries_cache = local_entries
            self._entry_count = len(local_entries)
            self._pending_writes += 1

            # v3.21: Flush to disk only when batch threshold or time threshold is reached
            now = time.monotonic()
            should_flush = (
                self._pending_writes >= self._MEMORY_FLUSH_INTERVAL
                or (now - self._last_flush_time) >= self._MEMORY_FLUSH_SECONDS
            )
            if should_flush:
                self._save_local_memory_entries(local_entries)
                self._pending_writes = 0
                self._last_flush_time = now

        # v3.21: Archive to SQLite for long-term storage
        try:
            self._database.write_memory_event(
                timestamp=local_payload.get("timestamp", self._utc_now()),
                event_type=local_payload.get("event_type", "event"),
                payload=json.dumps(local_payload.get("payload"), default=str)
                if local_payload.get("payload") is not None
                else None,
                source="local",
            )
        except Exception as exc:
            self._logger.warning("[MEMORY] SQLite archive failed: %s", exc)

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
        """Return the total number of memory entries (cached)."""
        return self._entry_count

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

    _MEMORY_FILE_SIZE_THRESHOLD: int = 10 * 1024 * 1024  # v3.21: 10 MB (was 50 MB)
    _MEMORY_NUCLEAR_THRESHOLD: int = 500 * 1024 * 1024  # 500 MB — too large to parse

    def _startup_trim_if_needed(self) -> None:
        """If memory.json exceeds 50 MB, trim to cap.

        This breaks the self-reinforcing deadlock where the file is too
        large for ``_load_local_memory_entries`` to complete, so the
        5000-entry cap inside ``write_memory`` can never execute.

        Files above 500 MB are replaced outright (parsing them would
        OOM or freeze the process for minutes/hours).
        """
        try:
            if not os.path.exists(self._memory_fallback_path):
                return
            size = os.path.getsize(self._memory_fallback_path)
            if size <= self._MEMORY_FILE_SIZE_THRESHOLD:
                # Small enough — lazy load later via cache.
                return

            self._logger.warning(
                "[MEMORY] memory.json is %.1f MB — trimming to %d entries",
                size / (1024 * 1024), self._MEMORY_MAX_ENTRIES,
            )

            # Files above the nuclear threshold cannot be parsed in
            # reasonable time/memory — reset to empty and move on.
            if size > self._MEMORY_NUCLEAR_THRESHOLD:
                self._logger.warning(
                    "[MEMORY] memory.json is %.1f GB — too large to parse, "
                    "resetting to empty",
                    size / (1024 * 1024 * 1024),
                )
                self._save_local_memory_entries([])
                return

            # Load the oversized file (one-time cost, < 500 MB)
            with open(self._memory_fallback_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)

            if isinstance(raw, dict):
                entries = raw.get("entries", [])
            elif isinstance(raw, list):
                entries = raw
            else:
                entries = []

            if not isinstance(entries, list):
                entries = []

            trimmed = entries[-self._MEMORY_MAX_ENTRIES:]

            # v3.22: Also truncate oversized payloads to prevent
            # writing back entries that individually weigh megabytes
            # (caused by the reflection-doubling bug fixed in v3.22).
            for entry in trimmed:
                if isinstance(entry, dict) and "payload" in entry:
                    p = json.dumps(entry["payload"], default=str)
                    if len(p) > self._MAX_PAYLOAD_BYTES:
                        entry["payload"] = {"_truncated": True, "summary": p[:self._MAX_PAYLOAD_BYTES]}

            self._save_local_memory_entries(trimmed)
            self._logger.info(
                "[MEMORY] Trimmed memory.json from %d to %d entries",
                len(entries), len(trimmed),
            )
        except Exception as exc:
            self._logger.error("[MEMORY] Startup trim failed: %s", exc)

    def _memory_entries_ref(self) -> Any:
        return self._db.collection("al01_memory").document("entries").collection("entries")

    def _state_ref(self) -> Any:
        return self._db.collection("al01_state").document("core")

    # v3.22: Hard cap on per-event payload size (bytes of JSON representation).
    # Prevents exponential blow-up from recursive reflection summaries.
    _MAX_PAYLOAD_BYTES: int = 10_000

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

        # v3.22: Truncate oversized payloads before they hit disk
        serialized = json.dumps(clean_payload, default=str)
        if len(serialized) > self._MAX_PAYLOAD_BYTES:
            clean_payload = {"_truncated": True, "summary": serialized[:self._MAX_PAYLOAD_BYTES]}

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
        if self._entries_cache is not None:
            return self._entries_cache
        if not os.path.exists(self._memory_fallback_path):
            self._entries_cache = []
            self._entry_count = 0
            return self._entries_cache
        try:
            # Safety: never attempt to parse files above the nuclear threshold
            fsize = os.path.getsize(self._memory_fallback_path)
            if fsize > self._MEMORY_NUCLEAR_THRESHOLD:
                self._logger.warning(
                    "[MEMORY] memory.json is %.1f GB — too large to load, "
                    "returning empty",
                    fsize / (1024 * 1024 * 1024),
                )
                self._entries_cache = []
                self._entry_count = 0
                return self._entries_cache

            with open(self._memory_fallback_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)

            if isinstance(raw, dict):
                entries = raw.get("entries", [])
            elif isinstance(raw, list):
                entries = raw
            else:
                entries = []

            if not isinstance(entries, list):
                self._entries_cache = []
                self._entry_count = 0
                return self._entries_cache
            self._entries_cache = [self._sanitize_dict(item) for item in entries if isinstance(item, dict)]
            self._entry_count = len(self._entries_cache)
            return self._entries_cache
        except Exception as exc:
            self._log_fallback_error(exc)
            self._entries_cache = []
            self._entry_count = 0
            return self._entries_cache

    def _save_local_memory_entries(self, entries: List[Dict[str, Any]]) -> None:
        self._ensure_parent_dir(self._memory_fallback_path)
        clean_entries = [self._sanitize_dict(item) for item in entries if isinstance(item, dict)]
        # Safety-net cap enforcement (the last gate before disk)
        if len(clean_entries) > self._MEMORY_MAX_ENTRIES:
            clean_entries = clean_entries[-self._MEMORY_MAX_ENTRIES:]
        # v3.21: Atomic write via tmpfile + rename (crash-safe)
        data = json.dumps({"entries": clean_entries}, indent=2)
        dir_part = os.path.dirname(self._memory_fallback_path) or "."
        fd, tmp_path = tempfile.mkstemp(dir=dir_part, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(data)
            os.replace(tmp_path, self._memory_fallback_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
        # Update cache so subsequent reads avoid disk
        self._entries_cache = clean_entries
        self._entry_count = len(clean_entries)

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
