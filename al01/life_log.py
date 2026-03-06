"""AL-01 VITAL — append-only life log with hash-chain integrity.

Files managed:
  data/life_log.jsonl   – append-only event log (one JSON object per line)
  data/head.json        – current chain head (hash, seq, organism_id)
  data/snapshots/       – periodic state checkpoints linked to the chain
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("al01.life_log")

_GENESIS_HASH = "0" * 64  # SHA-256 zero hash for the first event


class LifeLog:
    """Append-only event log secured by a SHA-256 hash chain."""

    def __init__(
        self,
        data_dir: str = "data",
        organism_id: str = "AL-01",
        snapshot_interval: int = 50,
        verify_depth: int = 200,
    ) -> None:
        self._data_dir = data_dir
        self._organism_id = organism_id
        self._snapshot_interval = snapshot_interval
        self._verify_depth = verify_depth
        self._lock = threading.RLock()

        self._log_path = os.path.join(data_dir, "life_log.jsonl")
        self._head_path = os.path.join(data_dir, "head.json")
        self._snapshot_dir = os.path.join(data_dir, "snapshots")

        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(self._snapshot_dir, exist_ok=True)

        # Load or initialize head
        self._head: Dict[str, Any] = self._load_head()
        self._integrity_status: str = "UNKNOWN"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def integrity_status(self) -> str:
        return self._integrity_status

    @property
    def head(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._head)

    @property
    def head_seq(self) -> int:
        return int(self._head.get("head_seq", 0))

    def startup_verify(self) -> bool:
        """Verify the last N entries on startup. Sets integrity_status.

        v3.4: If chain is broken (usually from force-killed process), re-anchor
        head to the last event so future appends produce a valid chain again.
        """
        ok = self.verify(last_n=self._verify_depth)
        self._integrity_status = "OK" if ok else "BROKEN"
        if not ok:
            logger.error("[VITAL] Hash-chain verification FAILED — integrity_status=BROKEN")
            # Re-anchor head to the last event so new appends are valid
            self._repair_head()
            self.append_event(
                event_type="integrity_alert",
                payload={"message": "Hash-chain verification failed on startup — head re-anchored"},
            )
        else:
            logger.info("[VITAL] Hash-chain verified (last %d entries) — integrity_status=OK", self._verify_depth)
        return ok

    def _repair_head(self) -> None:
        """Re-anchor head to the last event in the log file.

        This does NOT fix the broken link, but ensures that all future
        appends start from a valid prev_hash so the chain is healthy
        going forward.
        """
        events = self._read_tail(1)
        if events:
            last = events[-1]
            self._head = {
                "head_hash": last.get("hash", _GENESIS_HASH),
                "head_seq": last.get("seq", 0),
                "created_at": self._head.get("created_at", _utc_now()),
                "organism_id": self._organism_id,
                "updated_at": _utc_now(),
            }
            self._save_head()
            logger.warning(
                "[VITAL] Head re-anchored to seq=%d hash=%s",
                self._head["head_seq"],
                self._head["head_hash"][:16],
            )
        else:
            logger.warning("[VITAL] No events found — cannot repair head")

    def append_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Append a single event to the life log and return the event dict."""
        with self._lock:
            now = _utc_now()
            prev_hash = self._head.get("head_hash", _GENESIS_HASH)
            seq = int(self._head.get("head_seq", 0)) + 1

            event = {
                "event_id": str(uuid.uuid4()),
                "t": now,
                "seq": seq,
                "organism_id": self._organism_id,
                "prev_hash": prev_hash,
                "event_type": event_type,
                "payload": _sanitize(payload),
            }
            if extra_fields:
                event.update({k: _sanitize(v) for k, v in extra_fields.items()
                              if k not in event})

            event["hash"] = _compute_hash(prev_hash, event["payload"], now, seq)

            # Append to JSONL (never rewrite)
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, separators=(",", ":"), default=str) + "\n")

            # Update head
            self._head = {
                "head_hash": event["hash"],
                "head_seq": seq,
                "created_at": self._head.get("created_at", now),
                "organism_id": self._organism_id,
                "updated_at": now,
            }
            self._save_head()

            # Snapshot checkpoint?
            if seq > 0 and seq % self._snapshot_interval == 0:
                logger.info("[VITAL] Snapshot checkpoint at seq=%d", seq)

            return event

    def should_snapshot(self) -> bool:
        """Return True if the current seq is on a snapshot boundary."""
        seq = int(self._head.get("head_seq", 0))
        return seq > 0 and seq % self._snapshot_interval == 0

    def write_snapshot(self, state: Dict[str, Any]) -> str:
        """Write a snapshot checkpoint linked to the current head hash."""
        with self._lock:
            seq = int(self._head.get("head_seq", 0))
            snap = {
                "seq": seq,
                "snap_hash": self._head.get("head_hash", _GENESIS_HASH),
                "organism_id": self._organism_id,
                "timestamp": _utc_now(),
                "state": _sanitize(state),
            }
            path = os.path.join(self._snapshot_dir, f"snap_{seq}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(snap, fh, indent=2, default=str)
            logger.info("[VITAL] Snapshot written: %s", path)
            return path

    def load_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """Load the most recent snapshot file, or None."""
        snaps = sorted(
            [f for f in os.listdir(self._snapshot_dir) if f.startswith("snap_") and f.endswith(".json")],
            key=lambda f: int(f.replace("snap_", "").replace(".json", "")),
        ) if os.path.isdir(self._snapshot_dir) else []
        if not snaps:
            return None
        path = os.path.join(self._snapshot_dir, snaps[-1])
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def verify(self, last_n: int = 500) -> bool:
        """Verify the hash chain for the last *last_n* events. Returns True if valid."""
        events = self._read_tail(last_n)
        if not events:
            return True  # Empty log is trivially valid

        for i, ev in enumerate(events):
            expected_hash = _compute_hash(
                ev["prev_hash"],
                ev["payload"],
                ev["t"],
                ev["seq"],
            )
            if ev.get("hash") != expected_hash:
                logger.error(
                    "[VITAL] Chain break at seq=%d: expected=%s got=%s",
                    ev.get("seq", "?"),
                    expected_hash,
                    ev.get("hash", "?"),
                )
                return False

            # Verify linkage between consecutive events
            if i > 0:
                if ev["prev_hash"] != events[i - 1].get("hash"):
                    logger.error(
                        "[VITAL] Link break at seq=%d: prev_hash doesn't match prior event hash",
                        ev.get("seq", "?"),
                    )
                    return False

        # Verify head matches last event
        if events:
            last_hash = events[-1].get("hash")
            head_hash = self._head.get("head_hash")
            if head_hash and last_hash != head_hash:
                logger.error("[VITAL] Head hash mismatch: head=%s last_event=%s", head_hash, last_hash)
                return False

        return True

    def verify_full_report(self, last_n: int = 500) -> Dict[str, Any]:
        """Return a structured verification report with PASS/FAIL + first broken seq."""
        events = self._read_tail(last_n)
        if not events:
            return {"status": "PASS", "checked": 0, "first_broken_seq": None}

        for i, ev in enumerate(events):
            expected_hash = _compute_hash(
                ev["prev_hash"],
                ev["payload"],
                ev["t"],
                ev["seq"],
            )
            if ev.get("hash") != expected_hash:
                return {
                    "status": "FAIL",
                    "checked": i + 1,
                    "first_broken_seq": ev.get("seq"),
                    "reason": "hash mismatch",
                }
            if i > 0 and ev["prev_hash"] != events[i - 1].get("hash"):
                return {
                    "status": "FAIL",
                    "checked": i + 1,
                    "first_broken_seq": ev.get("seq"),
                    "reason": "link break",
                }

        # Check head
        if events:
            last_hash = events[-1].get("hash")
            head_hash = self._head.get("head_hash")
            if head_hash and last_hash != head_hash:
                return {
                    "status": "FAIL",
                    "checked": len(events),
                    "first_broken_seq": events[-1].get("seq"),
                    "reason": "head hash mismatch",
                }

        return {"status": "PASS", "checked": len(events), "first_broken_seq": None}

    def event_count(self) -> int:
        """Total events in the log file."""
        if not os.path.exists(self._log_path):
            return 0
        with open(self._log_path, "r", encoding="utf-8") as fh:
            return sum(1 for _ in fh)

    def repair_chain(self) -> Dict[str, Any]:
        """Scan the full chain, truncate at the first broken link, and re-anchor head.

        Steps:
            1. Read every event from life_log.jsonl.
            2. Walk forward verifying hashes and linkage.
            3. Record the last valid sequence number.
            4. Back up the original file to ``life_log_backup.jsonl``.
            5. Rewrite the log with only the valid prefix.
            6. Re-anchor head.json to the last valid event.

        Returns a report dict with keys:
            total_events, first_broken_seq, last_valid_seq,
            events_dropped, backup_path, status.
        """
        with self._lock:
            backup_path = os.path.join(self._data_dir, "life_log_backup.jsonl")

            # -- 1. Read all events --
            all_events: List[Dict[str, Any]] = []
            if os.path.exists(self._log_path):
                with open(self._log_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            all_events.append(json.loads(line))

            total = len(all_events)
            if total == 0:
                return {
                    "total_events": 0,
                    "first_broken_seq": None,
                    "last_valid_seq": 0,
                    "events_dropped": 0,
                    "backup_path": None,
                    "status": "EMPTY",
                }

            # -- 2. Walk forward to find first break --
            valid_events: List[Dict[str, Any]] = []
            first_broken_seq: Optional[int] = None

            for i, ev in enumerate(all_events):
                # Verify own hash
                expected_hash = _compute_hash(
                    ev["prev_hash"], ev["payload"], ev["t"], ev["seq"],
                )
                if ev.get("hash") != expected_hash:
                    first_broken_seq = ev.get("seq")
                    logger.warning(
                        "[REPAIR] Hash mismatch at seq=%d (expected=%s got=%s)",
                        ev.get("seq", "?"), expected_hash[:16],
                        str(ev.get("hash", "?"))[:16],
                    )
                    break

                # Verify linkage to previous
                if i > 0:
                    if ev["prev_hash"] != all_events[i - 1].get("hash"):
                        first_broken_seq = ev.get("seq")
                        logger.warning(
                            "[REPAIR] Link break at seq=%d", ev.get("seq", "?"),
                        )
                        break

                valid_events.append(ev)

            # If no break found, chain is already clean
            if first_broken_seq is None:
                # Still re-anchor head in case it drifted
                last_ev = valid_events[-1] if valid_events else None
                if last_ev:
                    self._head = {
                        "head_hash": last_ev["hash"],
                        "head_seq": last_ev["seq"],
                        "created_at": self._head.get("created_at", _utc_now()),
                        "organism_id": self._organism_id,
                        "updated_at": _utc_now(),
                    }
                    self._save_head()
                return {
                    "total_events": total,
                    "first_broken_seq": None,
                    "last_valid_seq": last_ev["seq"] if last_ev else 0,
                    "events_dropped": 0,
                    "backup_path": None,
                    "status": "CLEAN",
                }

            last_valid_seq = valid_events[-1]["seq"] if valid_events else 0
            events_dropped = total - len(valid_events)

            # -- 4. Back up original --
            import shutil
            shutil.copy2(self._log_path, backup_path)
            logger.info("[REPAIR] Backup created: %s (%d events)", backup_path, total)

            # -- 5. Rewrite with valid prefix only --
            with open(self._log_path, "w", encoding="utf-8") as fh:
                for ev in valid_events:
                    fh.write(
                        json.dumps(ev, separators=(",", ":"), default=str) + "\n"
                    )
            logger.info(
                "[REPAIR] Rewrote life_log.jsonl: %d valid events (dropped %d)",
                len(valid_events), events_dropped,
            )

            # -- 6. Re-anchor head --
            if valid_events:
                last_ev = valid_events[-1]
                self._head = {
                    "head_hash": last_ev["hash"],
                    "head_seq": last_ev["seq"],
                    "created_at": self._head.get("created_at", _utc_now()),
                    "organism_id": self._organism_id,
                    "updated_at": _utc_now(),
                }
            else:
                self._head = {
                    "head_hash": _GENESIS_HASH,
                    "head_seq": 0,
                    "created_at": _utc_now(),
                    "organism_id": self._organism_id,
                    "updated_at": _utc_now(),
                }
            self._save_head()

            # -- 7. Log the repair event into the now-clean chain --
            self.append_event(
                event_type="chain_repair",
                payload={
                    "message": "Hash chain repaired — truncated at first broken link",
                    "first_broken_seq": first_broken_seq,
                    "last_valid_seq": last_valid_seq,
                    "events_dropped": events_dropped,
                    "backup_path": backup_path,
                },
            )

            self._integrity_status = "OK"
            logger.warning(
                "[REPAIR] Chain repaired: valid=%d dropped=%d first_broken=%d",
                len(valid_events), events_dropped, first_broken_seq,
            )

            return {
                "total_events": total,
                "first_broken_seq": first_broken_seq,
                "last_valid_seq": last_valid_seq,
                "events_dropped": events_dropped,
                "backup_path": backup_path,
                "status": "REPAIRED",
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_head(self) -> Dict[str, Any]:
        if os.path.exists(self._head_path):
            try:
                with open(self._head_path, "r", encoding="utf-8") as fh:
                    head = json.load(fh)
                if isinstance(head, dict):
                    return head
            except Exception as exc:
                logger.warning("[VITAL] Failed to load head.json: %s", exc)
        return {
            "head_hash": _GENESIS_HASH,
            "head_seq": 0,
            "created_at": _utc_now(),
            "organism_id": self._organism_id,
        }

    def _save_head(self) -> None:
        with open(self._head_path, "w", encoding="utf-8") as fh:
            json.dump(self._head, fh, indent=2, default=str)

    def _read_tail(self, n: int) -> List[Dict[str, Any]]:
        """Read the last *n* events from the JSONL file."""
        if not os.path.exists(self._log_path):
            return []
        events: List[Dict[str, Any]] = []
        try:
            with open(self._log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        except Exception as exc:
            logger.error("[VITAL] Failed to read life_log.jsonl: %s", exc)
            return []
        return events[-n:] if n < len(events) else events


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _compute_hash(prev_hash: str, payload: Any, timestamp: str, seq: int) -> str:
    """SHA-256( prev_hash + canonical_json(payload) + timestamp + seq )."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    raw = f"{prev_hash}{canonical}{timestamp}{seq}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sanitize(value: Any) -> Any:
    """Ensure value is JSON-serializable."""
    try:
        json.dumps(value, default=str)
        return value
    except (TypeError, ValueError):
        return str(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
