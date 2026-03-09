"""v3.20 — Fix memory.json freeze on startup.

Bug:
  _startup_trim_if_needed() called json.load() on a 32 GB memory.json,
  which either OOM'd or froze the process indefinitely.

Fix:
  Added _MEMORY_NUCLEAR_THRESHOLD (500 MB). Files above this size are
  replaced with an empty entries list instead of parsed. Also added a
  safety check in _load_local_memory_entries() to never attempt parsing
  files above the nuclear threshold.
"""

import json
import os
import tempfile

import pytest

from al01.database import Database
from al01.memory_manager import MemoryManager


# ═══════════════════════════════════════════════════════════════════════
# 1. Nuclear threshold resets oversized files
# ═══════════════════════════════════════════════════════════════════════

class TestNuclearThreshold:
    """Files above _MEMORY_NUCLEAR_THRESHOLD are replaced, not parsed."""

    def test_huge_file_is_reset_on_init(self, tmp_path):
        """memory.json above nuclear threshold is replaced with empty entries."""
        mem_path = tmp_path / "memory.json"
        # Write a file that's artificially "huge" — we'll lower the thresholds
        # so we don't need a real multi-GB file.
        entries = [{"event_type": "pulse", "payload": {"i": i}} for i in range(100)]
        mem_path.write_text(json.dumps(entries), encoding="utf-8")
        file_size = mem_path.stat().st_size

        # Monkeypatch thresholds so our small file exceeds them
        original_nuclear = MemoryManager._MEMORY_NUCLEAR_THRESHOLD
        original_file = MemoryManager._MEMORY_FILE_SIZE_THRESHOLD
        try:
            MemoryManager._MEMORY_NUCLEAR_THRESHOLD = file_size - 1
            MemoryManager._MEMORY_FILE_SIZE_THRESHOLD = file_size - 2

            db = Database(db_path=str(tmp_path / "t.db"))
            mem = MemoryManager(data_dir=str(tmp_path), credential_path=None, database=db)

            # File should have been reset to empty
            reloaded = json.loads(mem_path.read_text(encoding="utf-8"))
            if isinstance(reloaded, dict):
                assert len(reloaded.get("entries", [])) == 0
            else:
                assert len(reloaded) == 0
        finally:
            MemoryManager._MEMORY_NUCLEAR_THRESHOLD = original_nuclear
            MemoryManager._MEMORY_FILE_SIZE_THRESHOLD = original_file

    def test_normal_trim_still_works(self, tmp_path):
        """Files between FILE_SIZE_THRESHOLD and NUCLEAR_THRESHOLD are trimmed normally."""
        mem_path = tmp_path / "memory.json"
        entries = [{"event_type": "test", "payload": {"i": i}} for i in range(200)]
        mem_path.write_text(json.dumps(entries), encoding="utf-8")
        file_size = mem_path.stat().st_size

        original_nuclear = MemoryManager._MEMORY_NUCLEAR_THRESHOLD
        original_file = MemoryManager._MEMORY_FILE_SIZE_THRESHOLD
        original_max = MemoryManager._MEMORY_MAX_ENTRIES
        try:
            # File exceeds FILE_SIZE_THRESHOLD but not NUCLEAR_THRESHOLD
            MemoryManager._MEMORY_FILE_SIZE_THRESHOLD = file_size - 1
            MemoryManager._MEMORY_NUCLEAR_THRESHOLD = file_size + 1000
            MemoryManager._MEMORY_MAX_ENTRIES = 50

            db = Database(db_path=str(tmp_path / "t.db"))
            mem = MemoryManager(data_dir=str(tmp_path), credential_path=None, database=db)

            # Should be trimmed to 50 entries (not reset to 0)
            loaded = mem.read_memory(limit=100)
            assert len(loaded) == 50
        finally:
            MemoryManager._MEMORY_NUCLEAR_THRESHOLD = original_nuclear
            MemoryManager._MEMORY_FILE_SIZE_THRESHOLD = original_file
            MemoryManager._MEMORY_MAX_ENTRIES = original_max

    def test_file_below_threshold_untouched(self, tmp_path):
        """Files below FILE_SIZE_THRESHOLD are loaded normally."""
        mem_path = tmp_path / "memory.json"
        entries = [{"event_type": "test", "payload": {"i": i}} for i in range(10)]
        mem_path.write_text(json.dumps(entries), encoding="utf-8")

        db = Database(db_path=str(tmp_path / "t.db"))
        mem = MemoryManager(data_dir=str(tmp_path), credential_path=None, database=db)

        loaded = mem.read_memory(limit=100)
        assert len(loaded) == 10


# ═══════════════════════════════════════════════════════════════════════
# 2. _load_local_memory_entries guards
# ═══════════════════════════════════════════════════════════════════════

class TestLoadGuard:
    """_load_local_memory_entries refuses to parse files above nuclear threshold."""

    def test_load_refuses_oversized_file(self, tmp_path):
        """If trim somehow fails, load still returns empty for huge files."""
        mem_path = tmp_path / "memory.json"
        entries = [{"event_type": "test", "payload": {"i": i}} for i in range(50)]
        mem_path.write_text(json.dumps(entries), encoding="utf-8")
        file_size = mem_path.stat().st_size

        db = Database(db_path=str(tmp_path / "t.db"))
        # Create with normal thresholds (file is small, so trim won't fire)
        mem = MemoryManager(data_dir=str(tmp_path), credential_path=None, database=db)

        # Now lower the nuclear threshold and clear the cache
        original = MemoryManager._MEMORY_NUCLEAR_THRESHOLD
        try:
            MemoryManager._MEMORY_NUCLEAR_THRESHOLD = file_size - 1
            mem._entries_cache = None  # Force reload

            loaded = mem.read_memory(limit=100)
            assert len(loaded) == 0  # Refused to parse
        finally:
            MemoryManager._MEMORY_NUCLEAR_THRESHOLD = original

    def test_missing_file_returns_empty(self, tmp_path):
        """No memory.json at all returns empty list."""
        db = Database(db_path=str(tmp_path / "t.db"))
        mem = MemoryManager(data_dir=str(tmp_path), credential_path=None, database=db)
        loaded = mem.read_memory(limit=100)
        assert loaded == []


# ═══════════════════════════════════════════════════════════════════════
# 3. Constants
# ═══════════════════════════════════════════════════════════════════════

class TestThresholdConstants:
    """Thresholds have expected values."""

    def test_file_size_threshold(self):
        assert MemoryManager._MEMORY_FILE_SIZE_THRESHOLD == 10 * 1024 * 1024

    def test_nuclear_threshold(self):
        assert MemoryManager._MEMORY_NUCLEAR_THRESHOLD == 500 * 1024 * 1024

    def test_nuclear_exceeds_file_size(self):
        assert MemoryManager._MEMORY_NUCLEAR_THRESHOLD > MemoryManager._MEMORY_FILE_SIZE_THRESHOLD

    def test_max_entries(self):
        assert MemoryManager._MEMORY_MAX_ENTRIES == 1000
