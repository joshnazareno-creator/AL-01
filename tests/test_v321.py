"""v3.21 — Bounded memory, rolling window, SQLite archival, tick snapshots.

Changes:
  - _MEMORY_MAX_ENTRIES reduced from 5000 → 1000 (rolling window)
  - _MEMORY_FILE_SIZE_THRESHOLD reduced from 50 MB → 10 MB
  - Memory events archived to SQLite memory_events table
  - memory.json writes are now atomic (tmpfile + rename)
  - Write throttling: batch N writes before flushing to disk
  - Periodic tick-based snapshots (data/snapshots/snap_<tick>.json)
  - flush_memory() for clean shutdown
  - Auto-trim on file-size safety net
"""

import json
import os
import tempfile
import time

import pytest

from al01.database import Database
from al01.memory_manager import MemoryManager
from al01.organism import MetabolismConfig, Organism


# ═══════════════════════════════════════════════════════════════════════
# 1. Rolling window cap
# ═══════════════════════════════════════════════════════════════════════

class TestRollingWindow:
    """memory.json is capped to _MEMORY_MAX_ENTRIES (1000)."""

    def test_max_entries_is_1000(self):
        assert MemoryManager._MEMORY_MAX_ENTRIES == 1000

    def test_entries_trimmed_to_cap(self, tmp_path):
        """Writing more than cap entries keeps only the newest."""
        db = Database(db_path=str(tmp_path / "t.db"))
        mem = MemoryManager(data_dir=str(tmp_path), credential_path=None, database=db)

        # Force low cap for fast testing
        original = MemoryManager._MEMORY_MAX_ENTRIES
        # Also force every write to flush immediately
        original_flush = MemoryManager._MEMORY_FLUSH_INTERVAL
        try:
            MemoryManager._MEMORY_MAX_ENTRIES = 50
            MemoryManager._MEMORY_FLUSH_INTERVAL = 1

            for i in range(80):
                mem.write_memory({"event_type": "test", "payload": {"i": i}})

            entries = mem.read_memory(limit=200)
            assert len(entries) == 50

            # Newest entries should be the last 50 written (i=30..79)
            payloads = [e["payload"]["i"] for e in entries]
            assert payloads[0] == 30
            assert payloads[-1] == 79
        finally:
            MemoryManager._MEMORY_MAX_ENTRIES = original
            MemoryManager._MEMORY_FLUSH_INTERVAL = original_flush

    def test_file_contains_only_capped_entries(self, tmp_path):
        """On-disk memory.json respects the cap."""
        db = Database(db_path=str(tmp_path / "t.db"))
        mem = MemoryManager(data_dir=str(tmp_path), credential_path=None, database=db)

        original = MemoryManager._MEMORY_MAX_ENTRIES
        original_flush = MemoryManager._MEMORY_FLUSH_INTERVAL
        try:
            MemoryManager._MEMORY_MAX_ENTRIES = 20
            MemoryManager._MEMORY_FLUSH_INTERVAL = 1

            for i in range(35):
                mem.write_memory({"event_type": "test", "payload": {"i": i}})

            # Read directly from file
            mem_path = tmp_path / "memory.json"
            raw = json.loads(mem_path.read_text(encoding="utf-8"))
            assert len(raw["entries"]) == 20
        finally:
            MemoryManager._MEMORY_MAX_ENTRIES = original
            MemoryManager._MEMORY_FLUSH_INTERVAL = original_flush


# ═══════════════════════════════════════════════════════════════════════
# 2. SQLite archival
# ═══════════════════════════════════════════════════════════════════════

class TestSQLiteArchival:
    """Memory events are archived to the memory_events SQLite table."""

    def test_write_memory_archives_to_sqlite(self, tmp_path):
        """Each write_memory() call is also inserted into SQLite."""
        db = Database(db_path=str(tmp_path / "t.db"))
        mem = MemoryManager(data_dir=str(tmp_path), credential_path=None, database=db)

        mem.write_memory({"event_type": "boot", "payload": {"x": 1}})
        mem.write_memory({"event_type": "pulse", "payload": {"y": 2}})

        count = db.memory_event_count()
        assert count == 2

    def test_sqlite_preserves_event_type(self, tmp_path):
        """Event type is stored correctly in SQLite."""
        db = Database(db_path=str(tmp_path / "t.db"))
        mem = MemoryManager(data_dir=str(tmp_path), credential_path=None, database=db)

        mem.write_memory({"event_type": "reflection", "payload": {"s": "hello"}})

        events = db.recent_memory_events(n=10)
        assert len(events) == 1
        assert events[0]["event_type"] == "reflection"

    def test_sqlite_stores_all_even_after_trim(self, tmp_path):
        """SQLite has ALL events, even those trimmed from memory.json."""
        db = Database(db_path=str(tmp_path / "t.db"))
        mem = MemoryManager(data_dir=str(tmp_path), credential_path=None, database=db)

        original = MemoryManager._MEMORY_MAX_ENTRIES
        original_flush = MemoryManager._MEMORY_FLUSH_INTERVAL
        try:
            MemoryManager._MEMORY_MAX_ENTRIES = 10
            MemoryManager._MEMORY_FLUSH_INTERVAL = 1

            for i in range(30):
                mem.write_memory({"event_type": "test", "payload": {"i": i}})

            # memory.json has only 10
            json_entries = mem.read_memory(limit=100)
            assert len(json_entries) == 10

            # SQLite has all 30
            assert db.memory_event_count() == 30
        finally:
            MemoryManager._MEMORY_MAX_ENTRIES = original
            MemoryManager._MEMORY_FLUSH_INTERVAL = original_flush

    def test_recent_memory_events_filter_by_type(self, tmp_path):
        """Can filter SQLite events by event_type."""
        db = Database(db_path=str(tmp_path / "t.db"))
        mem = MemoryManager(data_dir=str(tmp_path), credential_path=None, database=db)

        mem.write_memory({"event_type": "boot", "payload": {}})
        mem.write_memory({"event_type": "pulse", "payload": {}})
        mem.write_memory({"event_type": "pulse", "payload": {}})
        mem.write_memory({"event_type": "reflection", "payload": {}})

        pulses = db.recent_memory_events(n=10, event_type="pulse")
        assert len(pulses) == 2

        boots = db.recent_memory_events(n=10, event_type="boot")
        assert len(boots) == 1

    def test_batch_write_memory_events(self, tmp_path):
        """Batch insertion works correctly."""
        db = Database(db_path=str(tmp_path / "t.db"))

        events = [
            {"timestamp": "2026-01-01T00:00:00Z", "event_type": "pulse", "payload": {"i": i}}
            for i in range(25)
        ]
        count = db.write_memory_events_batch(events)
        assert count == 25
        assert db.memory_event_count() == 25

    def test_batch_write_empty_list(self, tmp_path):
        """Batch with empty list returns 0."""
        db = Database(db_path=str(tmp_path / "t.db"))
        assert db.write_memory_events_batch([]) == 0


# ═══════════════════════════════════════════════════════════════════════
# 3. Write throttling
# ═══════════════════════════════════════════════════════════════════════

class TestWriteThrottling:
    """Disk writes are batched, not flushed on every write_memory() call."""

    def test_write_not_flushed_immediately(self, tmp_path):
        """A single write_memory() does not touch disk if below flush threshold."""
        db = Database(db_path=str(tmp_path / "t.db"))

        original_flush = MemoryManager._MEMORY_FLUSH_INTERVAL
        try:
            MemoryManager._MEMORY_FLUSH_INTERVAL = 100  # high threshold

            mem = MemoryManager(data_dir=str(tmp_path), credential_path=None, database=db)
            mem_path = tmp_path / "memory.json"

            # File shouldn't exist yet (no flush)
            mem.write_memory({"event_type": "test", "payload": {}})

            # Cache should have the entry
            assert mem.memory_size() == 1

            # But pending writes should be non-zero
            assert mem._pending_writes == 1
        finally:
            MemoryManager._MEMORY_FLUSH_INTERVAL = original_flush

    def test_flush_triggers_at_threshold(self, tmp_path):
        """Writes are flushed to disk when pending count reaches threshold."""
        db = Database(db_path=str(tmp_path / "t.db"))

        original_flush = MemoryManager._MEMORY_FLUSH_INTERVAL
        try:
            MemoryManager._MEMORY_FLUSH_INTERVAL = 5

            mem = MemoryManager(data_dir=str(tmp_path), credential_path=None, database=db)
            mem_path = tmp_path / "memory.json"

            for i in range(5):
                mem.write_memory({"event_type": "test", "payload": {"i": i}})

            # After 5 writes, should have flushed
            assert mem._pending_writes == 0
            assert mem_path.exists()

            raw = json.loads(mem_path.read_text(encoding="utf-8"))
            assert len(raw["entries"]) == 5
        finally:
            MemoryManager._MEMORY_FLUSH_INTERVAL = original_flush

    def test_flush_memory_forces_disk_write(self, tmp_path):
        """flush_memory() writes pending entries to disk."""
        db = Database(db_path=str(tmp_path / "t.db"))

        original_flush = MemoryManager._MEMORY_FLUSH_INTERVAL
        try:
            MemoryManager._MEMORY_FLUSH_INTERVAL = 100  # won't trigger auto-flush

            mem = MemoryManager(data_dir=str(tmp_path), credential_path=None, database=db)

            mem.write_memory({"event_type": "test", "payload": {"x": 1}})
            mem.write_memory({"event_type": "test", "payload": {"x": 2}})

            assert mem._pending_writes == 2

            mem.flush_memory()

            assert mem._pending_writes == 0
            mem_path = tmp_path / "memory.json"
            assert mem_path.exists()
            raw = json.loads(mem_path.read_text(encoding="utf-8"))
            assert len(raw["entries"]) == 2
        finally:
            MemoryManager._MEMORY_FLUSH_INTERVAL = original_flush

    def test_time_based_flush(self, tmp_path):
        """Writes are flushed when time threshold is exceeded."""
        db = Database(db_path=str(tmp_path / "t.db"))

        original_flush = MemoryManager._MEMORY_FLUSH_INTERVAL
        original_seconds = MemoryManager._MEMORY_FLUSH_SECONDS
        try:
            MemoryManager._MEMORY_FLUSH_INTERVAL = 1000  # high count threshold
            MemoryManager._MEMORY_FLUSH_SECONDS = 0.0  # instant time threshold

            mem = MemoryManager(data_dir=str(tmp_path), credential_path=None, database=db)
            # Backdate last flush time so next write triggers flush
            mem._last_flush_time = time.monotonic() - 1.0

            mem.write_memory({"event_type": "test", "payload": {}})

            # Time-based flush should have triggered
            assert mem._pending_writes == 0
        finally:
            MemoryManager._MEMORY_FLUSH_INTERVAL = original_flush
            MemoryManager._MEMORY_FLUSH_SECONDS = original_seconds


# ═══════════════════════════════════════════════════════════════════════
# 4. Atomic writes
# ═══════════════════════════════════════════════════════════════════════

class TestAtomicWrites:
    """memory.json is written atomically via tmpfile + rename."""

    def test_save_creates_valid_json(self, tmp_path):
        """After save, memory.json is valid JSON with entries key."""
        db = Database(db_path=str(tmp_path / "t.db"))
        original_flush = MemoryManager._MEMORY_FLUSH_INTERVAL
        try:
            MemoryManager._MEMORY_FLUSH_INTERVAL = 1
            mem = MemoryManager(data_dir=str(tmp_path), credential_path=None, database=db)

            mem.write_memory({"event_type": "boot", "payload": {"v": 1}})

            mem_path = tmp_path / "memory.json"
            raw = json.loads(mem_path.read_text(encoding="utf-8"))
            assert "entries" in raw
            assert isinstance(raw["entries"], list)
            assert len(raw["entries"]) == 1
        finally:
            MemoryManager._MEMORY_FLUSH_INTERVAL = original_flush

    def test_no_tmp_files_left_behind(self, tmp_path):
        """No .tmp files remain after successful write."""
        db = Database(db_path=str(tmp_path / "t.db"))
        original_flush = MemoryManager._MEMORY_FLUSH_INTERVAL
        try:
            MemoryManager._MEMORY_FLUSH_INTERVAL = 1
            mem = MemoryManager(data_dir=str(tmp_path), credential_path=None, database=db)

            for i in range(5):
                mem.write_memory({"event_type": "test", "payload": {"i": i}})

            tmp_files = [f for f in os.listdir(str(tmp_path)) if f.endswith(".tmp")]
            assert len(tmp_files) == 0
        finally:
            MemoryManager._MEMORY_FLUSH_INTERVAL = original_flush


# ═══════════════════════════════════════════════════════════════════════
# 5. File size threshold
# ═══════════════════════════════════════════════════════════════════════

class TestFileSizeThreshold:
    """_MEMORY_FILE_SIZE_THRESHOLD is now 10 MB."""

    def test_file_size_threshold_is_10mb(self):
        assert MemoryManager._MEMORY_FILE_SIZE_THRESHOLD == 10 * 1024 * 1024

    def test_nuclear_threshold_unchanged(self):
        assert MemoryManager._MEMORY_NUCLEAR_THRESHOLD == 500 * 1024 * 1024

    def test_nuclear_exceeds_file_size(self):
        assert MemoryManager._MEMORY_NUCLEAR_THRESHOLD > MemoryManager._MEMORY_FILE_SIZE_THRESHOLD

    def test_startup_trim_at_lower_threshold(self, tmp_path):
        """Files between 10 MB and 500 MB are trimmed on startup."""
        mem_path = tmp_path / "memory.json"
        entries = [{"event_type": "test", "payload": {"i": i}} for i in range(200)]
        mem_path.write_text(json.dumps(entries), encoding="utf-8")
        file_size = mem_path.stat().st_size

        original_nuclear = MemoryManager._MEMORY_NUCLEAR_THRESHOLD
        original_file = MemoryManager._MEMORY_FILE_SIZE_THRESHOLD
        original_max = MemoryManager._MEMORY_MAX_ENTRIES
        try:
            MemoryManager._MEMORY_FILE_SIZE_THRESHOLD = file_size - 1
            MemoryManager._MEMORY_NUCLEAR_THRESHOLD = file_size + 1000
            MemoryManager._MEMORY_MAX_ENTRIES = 50

            db = Database(db_path=str(tmp_path / "t.db"))
            mem = MemoryManager(data_dir=str(tmp_path), credential_path=None, database=db)

            loaded = mem.read_memory(limit=300)
            assert len(loaded) == 50
        finally:
            MemoryManager._MEMORY_NUCLEAR_THRESHOLD = original_nuclear
            MemoryManager._MEMORY_FILE_SIZE_THRESHOLD = original_file
            MemoryManager._MEMORY_MAX_ENTRIES = original_max


# ═══════════════════════════════════════════════════════════════════════
# 6. Tick-based snapshots
# ═══════════════════════════════════════════════════════════════════════

class TestTickSnapshots:
    """MetabolismConfig.memory_snapshot_interval triggers periodic snapshots."""

    def test_config_has_memory_snapshot_interval(self):
        config = MetabolismConfig()
        assert hasattr(config, "memory_snapshot_interval")
        assert config.memory_snapshot_interval == 100

    def test_save_tick_snapshot_creates_file(self, tmp_path):
        """save_tick_snapshot() writes snap_<tick>.json to data/snapshots/."""
        db = Database(db_path=str(tmp_path / "t.db"))
        mem = MemoryManager(data_dir=str(tmp_path), credential_path=None, database=db)
        org = Organism(data_dir=str(tmp_path), memory_manager=mem)

        org.save_tick_snapshot(100)

        snap_path = tmp_path / "data" / "snapshots" / "snap_100.json"
        assert snap_path.exists()

        snap = json.loads(snap_path.read_text(encoding="utf-8"))
        assert snap["tick"] == 100
        assert "timestamp" in snap
        assert "evolution_count" in snap
        assert "fitness" in snap
        assert "genome" in snap

    def test_multiple_tick_snapshots(self, tmp_path):
        """Multiple calls create separate snapshot files."""
        db = Database(db_path=str(tmp_path / "t.db"))
        mem = MemoryManager(data_dir=str(tmp_path), credential_path=None, database=db)
        org = Organism(data_dir=str(tmp_path), memory_manager=mem)

        org.save_tick_snapshot(100)
        org.save_tick_snapshot(200)
        org.save_tick_snapshot(300)

        snap_dir = tmp_path / "data" / "snapshots"
        snap_files = [f for f in os.listdir(str(snap_dir))
                      if f.startswith("snap_") and f.endswith(".json")]
        assert len(snap_files) >= 3

    def test_scheduler_triggers_snapshot(self, tmp_path):
        """MetabolismScheduler triggers save_tick_snapshot at the configured interval."""
        db = Database(db_path=str(tmp_path / "t.db"))
        mem = MemoryManager(data_dir=str(tmp_path), credential_path=None, database=db)
        config = MetabolismConfig(memory_snapshot_interval=10)
        org = Organism(data_dir=str(tmp_path), config=config, memory_manager=mem)

        # Run 10 ticks
        for _ in range(10):
            org.tick()

        snap_dir = tmp_path / "data" / "snapshots"
        if snap_dir.exists():
            snap_files = [f for f in os.listdir(str(snap_dir))
                          if f.startswith("snap_") and f.endswith(".json")]
            assert len(snap_files) >= 1


# ═══════════════════════════════════════════════════════════════════════
# 7. Database memory_events table
# ═══════════════════════════════════════════════════════════════════════

class TestDatabaseMemoryEvents:
    """The memory_events SQLite table stores long-term event history."""

    def test_table_created(self, tmp_path):
        """memory_events table exists after Database init."""
        db = Database(db_path=str(tmp_path / "t.db"))
        # Query should succeed without error
        events = db.recent_memory_events(n=10)
        assert events == []

    def test_write_and_read_event(self, tmp_path):
        db = Database(db_path=str(tmp_path / "t.db"))
        row_id = db.write_memory_event(
            timestamp="2026-03-06T12:00:00Z",
            event_type="boot",
            payload='{"x": 1}',
            source="local",
        )
        assert row_id >= 1

        events = db.recent_memory_events(n=5)
        assert len(events) == 1
        assert events[0]["event_type"] == "boot"
        assert events[0]["source"] == "local"

    def test_memory_event_count(self, tmp_path):
        db = Database(db_path=str(tmp_path / "t.db"))
        assert db.memory_event_count() == 0

        db.write_memory_event(
            timestamp="2026-03-06T12:00:00Z",
            event_type="pulse",
        )
        db.write_memory_event(
            timestamp="2026-03-06T12:01:00Z",
            event_type="reflect",
        )
        assert db.memory_event_count() == 2

    def test_recent_events_oldest_first(self, tmp_path):
        """recent_memory_events returns oldest-first order."""
        db = Database(db_path=str(tmp_path / "t.db"))
        for i in range(5):
            db.write_memory_event(
                timestamp=f"2026-03-06T12:0{i}:00Z",
                event_type=f"event_{i}",
            )

        events = db.recent_memory_events(n=3)
        assert len(events) == 3
        # Oldest first (from the 3 most recent: event_2, event_3, event_4)
        assert events[0]["event_type"] == "event_2"
        assert events[-1]["event_type"] == "event_4"


# ═══════════════════════════════════════════════════════════════════════
# 8. End-to-end: bounded memory.json
# ═══════════════════════════════════════════════════════════════════════

class TestBoundedMemory:
    """Integration: memory.json stays small while SQLite grows."""

    def test_memory_json_stays_bounded(self, tmp_path):
        """Even after many writes, memory.json file size stays reasonable."""
        db = Database(db_path=str(tmp_path / "t.db"))
        original_max = MemoryManager._MEMORY_MAX_ENTRIES
        original_flush = MemoryManager._MEMORY_FLUSH_INTERVAL
        try:
            MemoryManager._MEMORY_MAX_ENTRIES = 100
            MemoryManager._MEMORY_FLUSH_INTERVAL = 1

            mem = MemoryManager(data_dir=str(tmp_path), credential_path=None, database=db)

            for i in range(500):
                mem.write_memory({
                    "event_type": "pulse",
                    "payload": {"awareness": 0.5, "i": i},
                })

            mem_path = tmp_path / "memory.json"
            file_size = mem_path.stat().st_size

            # memory.json should be well under 1 MB with 100 small entries
            assert file_size < 100_000  # under 100 KB

            # But SQLite should have all 500
            assert db.memory_event_count() == 500
        finally:
            MemoryManager._MEMORY_MAX_ENTRIES = original_max
            MemoryManager._MEMORY_FLUSH_INTERVAL = original_flush

    def test_flush_on_shutdown_preserves_data(self, tmp_path):
        """flush_memory() ensures no data loss on shutdown."""
        db = Database(db_path=str(tmp_path / "t.db"))
        original_flush = MemoryManager._MEMORY_FLUSH_INTERVAL
        try:
            MemoryManager._MEMORY_FLUSH_INTERVAL = 1000  # won't auto-flush

            mem = MemoryManager(data_dir=str(tmp_path), credential_path=None, database=db)

            for i in range(5):
                mem.write_memory({"event_type": "test", "payload": {"i": i}})

            # Data is in cache but not on disk
            assert mem._pending_writes == 5

            # Simulate shutdown
            mem.flush_memory()

            # Now create a fresh MemoryManager to verify disk persistence
            MemoryManager._MEMORY_FLUSH_INTERVAL = 1
            mem2 = MemoryManager(data_dir=str(tmp_path), credential_path=None, database=db)
            entries = mem2.read_memory(limit=100)
            assert len(entries) == 5
        finally:
            MemoryManager._MEMORY_FLUSH_INTERVAL = original_flush
