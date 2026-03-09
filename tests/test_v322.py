"""Tests for AL-01 v3.22 — Absolute-path storage, JSONL rotation, snapshot retention."""

from __future__ import annotations

import json
import os
import tempfile
import time

import pytest

from al01 import storage


# ═══════════════════════════════════════════════════════════════════════
# 1. storage module — path resolution
# ═══════════════════════════════════════════════════════════════════════

class TestStoragePaths:
    """Verify that all storage paths are absolute and rooted correctly."""

    def test_base_dir_is_absolute(self):
        assert os.path.isabs(storage.BASE_DIR)

    def test_data_dir_is_absolute(self):
        assert os.path.isabs(storage.DATA_DIR)

    def test_data_dir_under_base(self):
        assert storage.DATA_DIR.startswith(storage.BASE_DIR)

    def test_db_path_is_absolute(self):
        assert os.path.isabs(storage.db_path())

    def test_log_path_is_absolute(self):
        assert os.path.isabs(storage.log_path())

    def test_env_path_is_absolute(self):
        assert os.path.isabs(storage.env_path())

    def test_db_path_ends_with_al01_db(self):
        assert storage.db_path().endswith("al01.db")

    def test_log_path_ends_with_al01_log(self):
        assert storage.log_path().endswith("al01.log")

    def test_env_path_ends_with_dotenv(self):
        assert storage.env_path().endswith(".env")

    def test_base_dir_function_matches_constant(self):
        assert storage.base_dir() == storage.BASE_DIR

    def test_data_dir_function_matches_constant(self):
        assert storage.data_dir() == storage.DATA_DIR

    def test_snapshot_dir_is_absolute(self):
        assert os.path.isabs(storage.SNAPSHOT_DIR)

    def test_backup_dir_is_absolute(self):
        assert os.path.isabs(storage.BACKUP_DIR)


class TestStorageEnvOverride:
    """Test AL01_BASE_DIR environment variable override."""

    def test_env_override(self, monkeypatch, tmp_path):
        custom = str(tmp_path / "custom_base")
        monkeypatch.setenv("AL01_BASE_DIR", custom)
        # Re-import to pick up env change
        import importlib
        importlib.reload(storage)
        try:
            assert storage.BASE_DIR == custom
            assert storage.DATA_DIR == os.path.join(custom, "data")
            assert storage.db_path() == os.path.join(custom, "db", "al01.db")
        finally:
            monkeypatch.delenv("AL01_BASE_DIR", raising=False)
            importlib.reload(storage)


# ═══════════════════════════════════════════════════════════════════════
# 2. JSONL rotation
# ═══════════════════════════════════════════════════════════════════════

class TestJsonlRotation:
    """Test rotate_jsonl() size-based rotation."""

    def _write_lines(self, path: str, n: int) -> None:
        """Write *n* JSON lines to *path*."""
        with open(path, "w", encoding="utf-8") as fh:
            for i in range(n):
                fh.write(json.dumps({"i": i}) + "\n")

    def test_no_rotation_under_limit(self, tmp_path):
        p = str(tmp_path / "test.jsonl")
        self._write_lines(p, 10)
        storage.rotate_jsonl(p, max_bytes=1024 * 1024)
        assert os.path.exists(p)
        assert not os.path.exists(f"{p}.1")

    def test_rotation_over_limit(self, tmp_path):
        p = str(tmp_path / "test.jsonl")
        self._write_lines(p, 5000)
        size = os.path.getsize(p)
        storage.rotate_jsonl(p, max_bytes=size - 1)
        # Original renamed to .1, original path no longer exists
        assert os.path.exists(f"{p}.1")
        assert not os.path.exists(p)

    def test_rotation_chain(self, tmp_path):
        """Rotating twice should produce .1 and .2."""
        p = str(tmp_path / "test.jsonl")

        # First rotation
        self._write_lines(p, 100)
        storage.rotate_jsonl(p, max_bytes=1)
        assert os.path.exists(f"{p}.1")

        # Write more and rotate again
        self._write_lines(p, 100)
        storage.rotate_jsonl(p, max_bytes=1)
        assert os.path.exists(f"{p}.2")  # old .1 shifted to .2
        assert os.path.exists(f"{p}.1")  # current file shifted to .1

    def test_rotation_respects_backup_count(self, tmp_path):
        """Only keep backup_count backups."""
        p = str(tmp_path / "test.jsonl")
        for _ in range(5):
            self._write_lines(p, 100)
            storage.rotate_jsonl(p, max_bytes=1, backup_count=2)
        # Should only have .1 and .2, not .3+
        assert os.path.exists(f"{p}.1")
        assert os.path.exists(f"{p}.2")
        assert not os.path.exists(f"{p}.3")

    def test_rotation_missing_file(self, tmp_path):
        """rotate_jsonl on a missing file should not raise."""
        p = str(tmp_path / "nonexistent.jsonl")
        storage.rotate_jsonl(p)  # Should not raise

    def test_rotation_empty_file(self, tmp_path):
        """An empty file should not be rotated."""
        p = str(tmp_path / "empty.jsonl")
        with open(p, "w") as fh:
            pass
        storage.rotate_jsonl(p, max_bytes=1)
        assert os.path.exists(p)
        assert not os.path.exists(f"{p}.1")


# ═══════════════════════════════════════════════════════════════════════
# 3. Tick snapshot cleanup
# ═══════════════════════════════════════════════════════════════════════

class TestSnapshotCleanup:
    """Test cleanup_tick_snapshots() retention logic."""

    def _create_snapshots(self, snap_dir: str, count: int):
        os.makedirs(snap_dir, exist_ok=True)
        for i in range(count):
            path = os.path.join(snap_dir, f"snap_{i}.json")
            with open(path, "w") as fh:
                json.dump({"tick": i}, fh)
            # Ensure distinct mtimes
            os.utime(path, (time.time() + i, time.time() + i))

    def test_no_cleanup_within_limit(self, tmp_path):
        snap_dir = str(tmp_path / "snapshots")
        self._create_snapshots(snap_dir, 5)
        removed = storage.cleanup_tick_snapshots(snap_dir, keep=10)
        assert removed == 0
        assert len(os.listdir(snap_dir)) == 5

    def test_cleanup_beyond_limit(self, tmp_path):
        snap_dir = str(tmp_path / "snapshots")
        self._create_snapshots(snap_dir, 10)
        removed = storage.cleanup_tick_snapshots(snap_dir, keep=3)
        assert removed == 7
        assert len(os.listdir(snap_dir)) == 3

    def test_cleanup_keeps_newest(self, tmp_path):
        snap_dir = str(tmp_path / "snapshots")
        self._create_snapshots(snap_dir, 10)
        storage.cleanup_tick_snapshots(snap_dir, keep=3)
        remaining = sorted(os.listdir(snap_dir))
        # The 3 with highest mtime should survive (i=7,8,9)
        assert "snap_7.json" in remaining
        assert "snap_8.json" in remaining
        assert "snap_9.json" in remaining

    def test_cleanup_nonexistent_dir(self, tmp_path):
        removed = storage.cleanup_tick_snapshots(str(tmp_path / "nope"), keep=5)
        assert removed == 0

    def test_cleanup_empty_dir(self, tmp_path):
        snap_dir = str(tmp_path / "empty_snaps")
        os.makedirs(snap_dir)
        removed = storage.cleanup_tick_snapshots(snap_dir, keep=5)
        assert removed == 0


# ═══════════════════════════════════════════════════════════════════════
# 4. MemoryManager — absolute paths & RotatingFileHandler
# ═══════════════════════════════════════════════════════════════════════

class TestMemoryManagerPaths:
    """Verify MemoryManager uses absolute paths and rotating handler."""

    def test_backup_dir_is_absolute(self, tmp_path):
        from al01.memory_manager import MemoryManager
        mm = MemoryManager(data_dir=str(tmp_path))
        assert os.path.isabs(mm._backup_dir)

    def test_file_logger_uses_rotating_handler(self, tmp_path):
        import logging.handlers
        from al01.memory_manager import MemoryManager
        mm = MemoryManager(data_dir=str(tmp_path))
        handlers = mm._file_logger.handlers
        rotating = [h for h in handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
        assert len(rotating) >= 1
        # Verify the handler's path is absolute
        assert os.path.isabs(rotating[0].baseFilename)


# ═══════════════════════════════════════════════════════════════════════
# 5. __main__ path verification
# ═══════════════════════════════════════════════════════════════════════

class TestMainPaths:
    """Verify __main__.py no longer contains relative path literals."""

    def test_no_bare_dot_data_dir(self):
        """The main() function should not use data_dir='.' or data_dir=\".\"."""
        import inspect
        from al01 import __main__
        source = inspect.getsource(__main__.main)
        # Should not contain data_dir="." or data_dir='.'
        assert 'data_dir="."' not in source
        assert "data_dir='.'" not in source

    def test_no_bare_al01_db(self):
        """Database should not be instantiated with db_path='al01.db'."""
        import inspect
        from al01 import __main__
        source = inspect.getsource(__main__.main)
        assert 'db_path="al01.db"' not in source
        assert "db_path='al01.db'" not in source

    def test_storage_import_present(self):
        """__main__ should import the storage module."""
        from al01 import __main__
        assert hasattr(__main__, 'storage') or 'storage' in dir(__main__)


# ═══════════════════════════════════════════════════════════════════════
# 6. Integration: JSONL writers import rotate_jsonl
# ═══════════════════════════════════════════════════════════════════════

class TestRotationIntegration:
    """Verify that JSONL-writing modules import rotate_jsonl."""

    def test_life_log_imports_rotate(self):
        from al01 import life_log
        assert hasattr(life_log, 'rotate_jsonl')

    def test_evolution_tracker_imports_rotate(self):
        from al01 import evolution_tracker
        assert hasattr(evolution_tracker, 'rotate_jsonl')

    def test_autonomy_imports_rotate(self):
        from al01 import autonomy
        assert hasattr(autonomy, 'rotate_jsonl')

    def test_organism_imports_rotate(self):
        from al01 import organism
        assert hasattr(organism, 'rotate_jsonl')

    def test_experiment_imports_rotate(self):
        from al01 import experiment
        assert hasattr(experiment, 'rotate_jsonl')


# ═══════════════════════════════════════════════════════════════════════
# 7. Constants sanity
# ═══════════════════════════════════════════════════════════════════════

class TestConstants:
    """Verify storage module constants are reasonable."""

    def test_jsonl_max_bytes(self):
        assert storage.JSONL_MAX_BYTES == 50 * 1024 * 1024

    def test_jsonl_backup_count(self):
        assert storage.JSONL_BACKUP_COUNT == 3

    def test_snapshot_max_count(self):
        assert storage.SNAPSHOT_MAX_COUNT == 200

    def test_disk_warn_bytes(self):
        assert storage.DISK_WARN_BYTES == 10 * 1024 * 1024 * 1024


# ═══════════════════════════════════════════════════════════════════════
# 8. DB directory & path
# ═══════════════════════════════════════════════════════════════════════

class TestDatabasePath:
    """Verify SQLite database goes to BASE_DIR/db/."""

    def test_db_path_in_db_subdir(self):
        parts = storage.db_path().replace("\\", "/").split("/")
        # The path should end with db/al01.db
        assert parts[-2] == "db"
        assert parts[-1] == "al01.db"

    def test_db_dir_is_absolute(self):
        assert os.path.isabs(storage.DB_DIR)

    def test_database_creates_parent_dir(self, tmp_path):
        from al01.database import Database
        db_file = str(tmp_path / "subdir" / "test.db")
        db = Database(db_path=db_file)
        assert os.path.exists(db_file)

    def test_database_path_is_absolute(self, tmp_path):
        from al01.database import Database
        db = Database(db_path=str(tmp_path / "test.db"))
        assert os.path.isabs(db._db_path)


# ═══════════════════════════════════════════════════════════════════════
# 9. Temp directory redirect
# ═══════════════════════════════════════════════════════════════════════

class TestTempDir:
    """Verify local temp directory under BASE_DIR."""

    def test_tmp_dir_is_absolute(self):
        assert os.path.isabs(storage.TMP_DIR)

    def test_tmp_dir_under_base(self):
        assert storage.TMP_DIR.startswith(storage.BASE_DIR)

    def test_tmp_dir_function_creates_dir(self, tmp_path, monkeypatch):
        import importlib
        monkeypatch.setenv("AL01_BASE_DIR", str(tmp_path))
        importlib.reload(storage)
        try:
            d = storage.tmp_dir()
            assert os.path.isdir(d)
            assert d == os.path.join(str(tmp_path), "tmp")
        finally:
            monkeypatch.delenv("AL01_BASE_DIR", raising=False)
            importlib.reload(storage)

    def test_ensure_dirs_creates_all(self, tmp_path, monkeypatch):
        import importlib
        monkeypatch.setenv("AL01_BASE_DIR", str(tmp_path))
        importlib.reload(storage)
        try:
            storage.ensure_dirs()
            assert os.path.isdir(storage.DATA_DIR)
            assert os.path.isdir(storage.DB_DIR)
            assert os.path.isdir(storage.BACKUP_DIR)
            assert os.path.isdir(storage.TMP_DIR)
        finally:
            monkeypatch.delenv("AL01_BASE_DIR", raising=False)
            importlib.reload(storage)


# ═══════════════════════════════════════════════════════════════════════
# 10. Disk usage monitoring
# ═══════════════════════════════════════════════════════════════════════

class TestDiskUsage:
    """Verify disk usage monitoring functions."""

    def test_dir_size_bytes_on_tmp(self, tmp_path):
        # Create a file with known size
        f = tmp_path / "test.bin"
        f.write_bytes(b"x" * 4096)
        assert storage.dir_size_bytes(str(tmp_path)) == 4096

    def test_dir_size_bytes_nonexistent(self, tmp_path):
        assert storage.dir_size_bytes(str(tmp_path / "nope")) == 0

    def test_check_disk_usage_returns_dict(self):
        result = storage.check_disk_usage()
        assert isinstance(result, dict)
        assert "used_bytes" in result
        assert "used_mb" in result
        assert "used_gb" in result
        assert "warning" in result
        assert "message" in result
        assert "base_dir" in result

    def test_check_disk_usage_warning_flag(self, tmp_path):
        # With a tiny threshold, should trigger warning
        import importlib
        original_base = storage.BASE_DIR
        # Temporarily point to tmp_path for this test
        storage.BASE_DIR = str(tmp_path)
        f = tmp_path / "test.bin"
        f.write_bytes(b"x" * 100)
        try:
            result = storage.check_disk_usage(warn_bytes=50)
            assert result["warning"] is True
            assert "exceeds" in result["message"]
        finally:
            storage.BASE_DIR = original_base

    def test_check_disk_usage_no_warning(self, tmp_path):
        storage_backup = storage.BASE_DIR
        storage.BASE_DIR = str(tmp_path)
        try:
            result = storage.check_disk_usage(warn_bytes=10 * 1024 * 1024 * 1024)
            assert result["warning"] is False
        finally:
            storage.BASE_DIR = storage_backup


# ═══════════════════════════════════════════════════════════════════════
# 11. Reflection doubling fix
# ═══════════════════════════════════════════════════════════════════════

class TestReflectionDoublingFix:
    """Verify reflection summaries don't grow exponentially."""

    def test_reflect_caps_fragment_size(self, tmp_path):
        from al01.organism import Organism
        o = Organism(data_dir=str(tmp_path))
        # Write a huge reflection entry
        huge = "x" * 50000
        o._memory_manager.write_memory({
            "event_type": "reflection",
            "payload": {"summary": huge},
        })
        # Now reflect — should not blow up
        o.reflect()
        entries = o._memory_manager.read_memory(limit=2)
        latest = entries[-1]
        # The summary should be capped, not 50000+
        summary = latest.get("payload", {}).get("summary", "")
        assert len(summary) < 5000

    def test_reflect_fragment_max_constant(self):
        from al01.organism import Organism
        assert Organism._REFLECT_FRAGMENT_MAX == 200


# ═══════════════════════════════════════════════════════════════════════
# 12. Payload size cap in MemoryManager
# ═══════════════════════════════════════════════════════════════════════

class TestPayloadSizeCap:
    """Verify oversized payloads are truncated before hitting disk."""

    def test_max_payload_constant(self):
        from al01.memory_manager import MemoryManager
        assert MemoryManager._MAX_PAYLOAD_BYTES == 10_000

    def test_oversized_payload_truncated(self, tmp_path):
        from al01.memory_manager import MemoryManager
        mm = MemoryManager(data_dir=str(tmp_path))
        huge_payload = {"data": "x" * 50000}
        mm.write_memory({
            "event_type": "test",
            "payload": huge_payload,
        })
        entries = mm.read_memory(limit=1)
        latest = entries[-1]
        payload = latest.get("payload", {})
        # Should be truncated
        assert payload.get("_truncated") is True
        serialized = json.dumps(payload)
        assert len(serialized) < 15000  # well under the 50KB original

    def test_small_payload_not_truncated(self, tmp_path):
        from al01.memory_manager import MemoryManager
        mm = MemoryManager(data_dir=str(tmp_path))
        mm.write_memory({
            "event_type": "test",
            "payload": {"value": 42},
        })
        entries = mm.read_memory(limit=1)
        latest = entries[-1]
        assert latest.get("payload", {}).get("_truncated") is not True


# ═══════════════════════════════════════════════════════════════════════
# 13. CLI uses absolute defaults
# ═══════════════════════════════════════════════════════════════════════

class TestCliDefaults:
    """Verify CLI defaults to absolute paths."""

    def test_cli_imports_storage(self):
        from al01 import cli
        assert hasattr(cli, 'storage')

    def test_experiment_protocol_uses_abspath(self, tmp_path):
        from al01.experiment import ExperimentProtocol
        ep = ExperimentProtocol(data_dir=".")
        assert os.path.isabs(ep._data_dir)
