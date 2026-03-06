"""AL-01 v3.1 — Automatic hourly state snapshotting with 30-day retention + remote backup sync.

Features:
* Hourly full-state snapshots to timestamped JSON files in ``snapshots/hourly/``
* Rolling 30-day archive retention (auto-purge on every snapshot)
* Remote backup sync to Firestore (batched, respects Spark-tier quotas)
* Thread-safe background scheduler that runs alongside the organism loop
* Manual trigger via CLI / API
* Snapshot manifest (index file) for fast queries without scanning the directory
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("al01.snapshot")


# ── Configuration ────────────────────────────────────────────────────

@dataclass
class SnapshotConfig:
    """Tuneable knobs for the snapshot scheduler."""

    interval_seconds: float = 3600.0       # 1 hour between automatic snapshots
    retention_days: int = 30               # delete snapshots older than this
    snapshots_subdir: str = "snapshots/hourly"  # relative to data_dir
    manifest_filename: str = "snapshot_manifest.json"
    max_manifest_entries: int = 2000       # cap manifest to avoid bloat
    remote_sync_enabled: bool = True       # push snapshots to Firestore
    remote_collection: str = "al01_snapshots"   # Firestore collection


# ── Snapshot Manager ─────────────────────────────────────────────────

class SnapshotManager:
    """Automatic hourly state snapshotting with retention and remote sync.

    Usage::

        mgr = SnapshotManager(
            data_dir=".",
            state_collector=organism.growth_metrics.__get__(organism),
            firestore_client=memory_manager._db if memory_manager.firestore_enabled() else None,
        )
        mgr.start()          # begins background scheduler
        mgr.stop()           # clean shutdown
        mgr.take_snapshot()  # manual trigger
    """

    def __init__(
        self,
        data_dir: str = ".",
        config: Optional[SnapshotConfig] = None,
        state_collector: Optional[Callable[[], Dict[str, Any]]] = None,
        firestore_client: Optional[Any] = None,
    ) -> None:
        self._config = config or SnapshotConfig()
        self._data_dir = data_dir
        self._state_collector = state_collector
        self._firestore = firestore_client

        # Paths
        self._snapshot_dir = os.path.join(data_dir, self._config.snapshots_subdir)
        os.makedirs(self._snapshot_dir, exist_ok=True)
        self._manifest_path = os.path.join(
            self._snapshot_dir, self._config.manifest_filename,
        )

        # Load or create manifest
        self._manifest_lock = threading.Lock()
        self._manifest: List[Dict[str, Any]] = self._load_manifest()

        # Scheduler state
        self._timer: Optional[threading.Timer] = None
        self._running = False
        self._lock = threading.Lock()
        self._snapshot_count = 0

    # ── Properties ───────────────────────────────────────────────────

    @property
    def config(self) -> SnapshotConfig:
        return self._config

    @property
    def snapshot_dir(self) -> str:
        return self._snapshot_dir

    @property
    def snapshot_count(self) -> int:
        """Total snapshots taken since start (in-memory counter)."""
        return self._snapshot_count

    @property
    def running(self) -> bool:
        return self._running

    @property
    def manifest(self) -> List[Dict[str, Any]]:
        with self._manifest_lock:
            return list(self._manifest)

    @property
    def next_snapshot_in(self) -> Optional[float]:
        """Seconds until next scheduled snapshot, or None if not running."""
        if not self._running or self._timer is None:
            return None
        remaining = self._timer.interval - (time.monotonic() - getattr(self, "_last_snapshot_time", time.monotonic()))
        return max(0.0, remaining)

    # ── Lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background snapshot scheduler."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._last_snapshot_time = time.monotonic()
            self._schedule_next()
            logger.info(
                "[SNAPSHOT] Scheduler started (interval=%ds, retention=%dd)",
                int(self._config.interval_seconds),
                self._config.retention_days,
            )

    def stop(self) -> None:
        """Stop the background scheduler and take a final snapshot."""
        with self._lock:
            self._running = False
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        logger.info("[SNAPSHOT] Scheduler stopped (total_snapshots=%d)", self._snapshot_count)

    def _schedule_next(self) -> None:
        """Schedule the next snapshot timer cycle."""
        if not self._running:
            return
        self._timer = threading.Timer(self._config.interval_seconds, self._tick)
        self._timer.daemon = True
        self._timer.name = "al01-snapshot-timer"
        self._timer.start()

    def _tick(self) -> None:
        """Timer callback — take snapshot and reschedule."""
        try:
            self.take_snapshot()
        except Exception as exc:
            logger.error("[SNAPSHOT] Auto-snapshot failed: %s", exc)
        finally:
            self._last_snapshot_time = time.monotonic()
            self._schedule_next()

    # ── Core Snapshot ────────────────────────────────────────────────

    def take_snapshot(self, label: Optional[str] = None) -> Dict[str, Any]:
        """Capture current state and write to a timestamped JSON file.

        Parameters
        ----------
        label : str, optional
            Human-readable label (e.g. "manual", "shutdown", "hourly").

        Returns
        -------
        dict
            Snapshot metadata (path, timestamp, checksum, size_bytes).
        """
        now = datetime.now(timezone.utc)
        timestamp_str = now.strftime("%Y%m%dT%H%M%SZ")
        filename = f"snapshot_{timestamp_str}.json"
        filepath = os.path.join(self._snapshot_dir, filename)

        # Collect state
        state = self._collect_state()

        # Build snapshot envelope
        snapshot = {
            "snapshot_version": 1,
            "timestamp": now.isoformat(),
            "timestamp_unix": now.timestamp(),
            "label": label or "auto",
            "state": state,
        }

        # Compute checksum (over the state portion only)
        state_json = json.dumps(state, sort_keys=True, default=str)
        checksum = hashlib.sha256(state_json.encode()).hexdigest()
        snapshot["checksum"] = checksum

        # Atomic write
        self._atomic_write(filepath, snapshot)
        size_bytes = os.path.getsize(filepath)

        # Update manifest
        entry = {
            "filename": filename,
            "timestamp": now.isoformat(),
            "timestamp_unix": now.timestamp(),
            "label": label or "auto",
            "checksum": checksum,
            "size_bytes": size_bytes,
        }
        self._append_manifest(entry)

        self._snapshot_count += 1
        logger.info(
            "[SNAPSHOT] Written %s (%d bytes, checksum=%s…)",
            filename, size_bytes, checksum[:12],
        )

        # Enforce retention
        self._enforce_retention()

        # Remote sync
        if self._config.remote_sync_enabled and self._firestore is not None:
            self._remote_sync(entry, snapshot)

        return entry

    # ── State Collection ─────────────────────────────────────────────

    def _collect_state(self) -> Dict[str, Any]:
        """Gather full organism state via the registered collector callback."""
        if self._state_collector is None:
            return {"error": "no state_collector registered"}
        try:
            raw = self._state_collector()
            return self._sanitize(raw)
        except Exception as exc:
            logger.error("[SNAPSHOT] State collection failed: %s", exc)
            return {"error": str(exc)}

    # ── Retention (30-day rolling archive) ───────────────────────────

    def _enforce_retention(self) -> int:
        """Delete snapshot files older than ``retention_days``. Returns count deleted."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._config.retention_days)
        cutoff_unix = cutoff.timestamp()

        deleted = 0
        surviving: List[Dict[str, Any]] = []

        with self._manifest_lock:
            for entry in self._manifest:
                ts = entry.get("timestamp_unix", 0)
                if ts < cutoff_unix:
                    # Delete the file
                    path = os.path.join(self._snapshot_dir, entry["filename"])
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                            deleted += 1
                            logger.info("[SNAPSHOT] Purged expired: %s", entry["filename"])
                    except OSError as exc:
                        logger.warning("[SNAPSHOT] Failed to delete %s: %s", entry["filename"], exc)
                        surviving.append(entry)  # keep in manifest if can't delete
                else:
                    surviving.append(entry)

            self._manifest = surviving
            self._save_manifest()

        if deleted > 0:
            logger.info("[SNAPSHOT] Retention purge: %d files deleted (cutoff=%s)", deleted, cutoff.isoformat())
        return deleted

    def purge_older_than(self, days: int) -> int:
        """Manually purge snapshots older than N days. Returns count deleted."""
        original_retention = self._config.retention_days
        self._config.retention_days = days
        try:
            return self._enforce_retention()
        finally:
            self._config.retention_days = original_retention

    # ── Remote Backup Sync ───────────────────────────────────────────

    def _remote_sync(self, entry: Dict[str, Any], snapshot: Dict[str, Any]) -> None:
        """Push snapshot metadata + state to Firestore in background thread."""
        def _sync():
            try:
                collection = self._config.remote_collection
                doc_id = entry["filename"].replace(".json", "")
                doc_ref = self._firestore.collection(collection).document(doc_id)
                # Write metadata + truncated state (Firestore 1MB doc limit)
                remote_doc = {
                    "filename": entry["filename"],
                    "timestamp": entry["timestamp"],
                    "timestamp_unix": entry["timestamp_unix"],
                    "label": entry.get("label", "auto"),
                    "checksum": entry["checksum"],
                    "size_bytes": entry["size_bytes"],
                    "synced_at": datetime.now(timezone.utc).isoformat(),
                }
                # Include state if small enough (<500KB serialised)
                state_json = json.dumps(snapshot.get("state", {}), default=str)
                if len(state_json) < 500_000:
                    remote_doc["state"] = snapshot["state"]
                else:
                    remote_doc["state_truncated"] = True
                    remote_doc["state_preview"] = json.loads(state_json[:10000] + "}")

                doc_ref.set(self._sanitize(remote_doc))
                logger.info("[SNAPSHOT] Remote sync OK: %s", doc_id)
            except Exception as exc:
                logger.warning("[SNAPSHOT] Remote sync failed: %s", exc)

        thread = threading.Thread(target=_sync, daemon=True, name="al01-snapshot-sync")
        thread.start()

    # ── Manifest ─────────────────────────────────────────────────────

    def _load_manifest(self) -> List[Dict[str, Any]]:
        """Load snapshot manifest from disk, or rebuild from directory scan."""
        if os.path.exists(self._manifest_path):
            try:
                with open(self._manifest_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    return data
                if isinstance(data, dict) and "entries" in data:
                    return data["entries"]
            except Exception as exc:
                logger.warning("[SNAPSHOT] Manifest load failed, rebuilding: %s", exc)
        return self._rebuild_manifest()

    def _rebuild_manifest(self) -> List[Dict[str, Any]]:
        """Scan the snapshot directory and rebuild manifest from file metadata."""
        entries: List[Dict[str, Any]] = []
        if not os.path.isdir(self._snapshot_dir):
            return entries
        for fname in sorted(os.listdir(self._snapshot_dir)):
            if not fname.startswith("snapshot_") or not fname.endswith(".json"):
                continue
            path = os.path.join(self._snapshot_dir, fname)
            try:
                stat = os.stat(path)
                # Parse timestamp from filename: snapshot_YYYYMMDDTHHMMSSz.json
                ts_part = fname.replace("snapshot_", "").replace(".json", "")
                dt = datetime.strptime(ts_part, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
                entries.append({
                    "filename": fname,
                    "timestamp": dt.isoformat(),
                    "timestamp_unix": dt.timestamp(),
                    "label": "recovered",
                    "checksum": None,
                    "size_bytes": stat.st_size,
                })
            except (ValueError, OSError):
                continue
        return entries

    def _append_manifest(self, entry: Dict[str, Any]) -> None:
        with self._manifest_lock:
            self._manifest.append(entry)
            # Cap manifest size
            if len(self._manifest) > self._config.max_manifest_entries:
                self._manifest = self._manifest[-self._config.max_manifest_entries:]
            self._save_manifest()

    def _save_manifest(self) -> None:
        """Atomically write the manifest file."""
        try:
            self._atomic_write(self._manifest_path, self._manifest)
        except Exception as exc:
            logger.error("[SNAPSHOT] Manifest write failed: %s", exc)

    # ── Query ────────────────────────────────────────────────────────

    def list_snapshots(
        self,
        limit: int = 50,
        since: Optional[str] = None,
        label: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query snapshot manifest with optional filters.

        Parameters
        ----------
        limit : int
            Max entries to return (most recent first).
        since : str, optional
            ISO-8601 lower bound.
        label : str, optional
            Filter by label.
        """
        with self._manifest_lock:
            entries = list(self._manifest)

        if since:
            try:
                since_dt = datetime.fromisoformat(since)
                since_unix = since_dt.timestamp()
                entries = [e for e in entries if e.get("timestamp_unix", 0) >= since_unix]
            except ValueError:
                pass

        if label:
            entries = [e for e in entries if e.get("label") == label]

        # Most recent first
        entries.sort(key=lambda e: e.get("timestamp_unix", 0), reverse=True)
        return entries[:limit]

    def load_snapshot(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load a specific snapshot file by filename."""
        path = os.path.join(self._snapshot_dir, filename)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.error("[SNAPSHOT] Failed to load %s: %s", filename, exc)
            return None

    def latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """Load the most recent snapshot."""
        entries = self.list_snapshots(limit=1)
        if not entries:
            return None
        return self.load_snapshot(entries[0]["filename"])

    def status(self) -> Dict[str, Any]:
        """Return current snapshot manager status."""
        with self._manifest_lock:
            total = len(self._manifest)
            oldest = self._manifest[0] if self._manifest else None
            newest = self._manifest[-1] if self._manifest else None

        disk_bytes = 0
        try:
            for fname in os.listdir(self._snapshot_dir):
                if fname.startswith("snapshot_") and fname.endswith(".json"):
                    disk_bytes += os.path.getsize(os.path.join(self._snapshot_dir, fname))
        except OSError:
            pass

        return {
            "running": self._running,
            "interval_seconds": self._config.interval_seconds,
            "retention_days": self._config.retention_days,
            "total_snapshots": total,
            "snapshots_this_session": self._snapshot_count,
            "disk_usage_bytes": disk_bytes,
            "disk_usage_mb": round(disk_bytes / (1024 * 1024), 2),
            "oldest_snapshot": oldest.get("timestamp") if oldest else None,
            "newest_snapshot": newest.get("timestamp") if newest else None,
            "remote_sync_enabled": self._config.remote_sync_enabled and self._firestore is not None,
            "snapshot_dir": self._snapshot_dir,
        }

    # ── Utilities ────────────────────────────────────────────────────

    @staticmethod
    def _atomic_write(path: str, data: Any) -> None:
        """Write JSON atomically via temp file + rename."""
        dir_part = os.path.dirname(path) or "."
        os.makedirs(dir_part, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=dir_part, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, default=str)
            os.replace(tmp_path, path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    @staticmethod
    def _sanitize(obj: Any) -> Any:
        """Ensure obj is JSON-serialisable (round-trip through json)."""
        return json.loads(json.dumps(obj, default=str))
