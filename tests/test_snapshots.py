"""AL-01 v3.1 — Tests for automatic hourly snapshotting, retention, and remote sync.

Covers:
1. SnapshotConfig defaults
2. SnapshotManager lifecycle (start/stop)
3. take_snapshot (file creation, checksum, manifest)
4. 30-day rolling retention (purge old files)
5. Manifest persistence + rebuild from directory scan
6. list_snapshots filtering (limit, since, label)
7. load_snapshot / latest_snapshot
8. status() reporting
9. Deterministic state collection
10. Remote sync queueing (mock Firestore)
11. CLI commands (snapshot, snapshot-list, snapshot-status, snapshot-purge)
12. Organism integration (snapshot_manager property, growth_metrics field)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from al01.snapshot_manager import SnapshotConfig, SnapshotManager


# ==================================================================
# 1. SnapshotConfig
# ==================================================================

class TestSnapshotConfig(unittest.TestCase):
    """Default configuration is sane."""

    def test_defaults(self):
        cfg = SnapshotConfig()
        self.assertEqual(cfg.interval_seconds, 3600.0)
        self.assertEqual(cfg.retention_days, 30)
        self.assertTrue(cfg.remote_sync_enabled)
        self.assertEqual(cfg.snapshots_subdir, "snapshots/hourly")

    def test_custom_values(self):
        cfg = SnapshotConfig(interval_seconds=60, retention_days=7)
        self.assertEqual(cfg.interval_seconds, 60)
        self.assertEqual(cfg.retention_days, 7)


# ==================================================================
# 2. SnapshotManager lifecycle
# ==================================================================

class TestSnapshotManagerLifecycle(unittest.TestCase):
    """Start/stop scheduler."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.mgr = SnapshotManager(
            data_dir=self._tmpdir,
            config=SnapshotConfig(interval_seconds=9999),  # won't fire during test
            state_collector=lambda: {"test": "data"},
        )

    def test_not_running_initially(self):
        self.assertFalse(self.mgr.running)

    def test_start_sets_running(self):
        self.mgr.start()
        self.assertTrue(self.mgr.running)
        self.mgr.stop()

    def test_stop_clears_running(self):
        self.mgr.start()
        self.mgr.stop()
        self.assertFalse(self.mgr.running)

    def test_double_start_is_safe(self):
        self.mgr.start()
        self.mgr.start()  # no error
        self.assertTrue(self.mgr.running)
        self.mgr.stop()

    def test_double_stop_is_safe(self):
        self.mgr.start()
        self.mgr.stop()
        self.mgr.stop()  # no error


# ==================================================================
# 3. take_snapshot
# ==================================================================

class TestTakeSnapshot(unittest.TestCase):
    """Snapshot creation, checksum, and file output."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.state = {"awareness": 0.5, "evolution_count": 10}
        self.mgr = SnapshotManager(
            data_dir=self._tmpdir,
            config=SnapshotConfig(remote_sync_enabled=False),
            state_collector=lambda: self.state,
        )

    def test_creates_file(self):
        entry = self.mgr.take_snapshot(label="test")
        path = os.path.join(self.mgr.snapshot_dir, entry["filename"])
        self.assertTrue(os.path.exists(path))

    def test_file_is_valid_json(self):
        entry = self.mgr.take_snapshot()
        path = os.path.join(self.mgr.snapshot_dir, entry["filename"])
        with open(path, "r") as f:
            data = json.load(f)
        self.assertEqual(data["snapshot_version"], 1)
        self.assertEqual(data["state"]["awareness"], 0.5)

    def test_checksum_present(self):
        entry = self.mgr.take_snapshot()
        self.assertIn("checksum", entry)
        self.assertEqual(len(entry["checksum"]), 64)  # SHA-256

    def test_size_bytes_positive(self):
        entry = self.mgr.take_snapshot()
        self.assertGreater(entry["size_bytes"], 0)

    def test_label_stored(self):
        entry = self.mgr.take_snapshot(label="manual")
        self.assertEqual(entry["label"], "manual")

    def test_default_label_is_auto(self):
        entry = self.mgr.take_snapshot()
        self.assertEqual(entry["label"], "auto")

    def test_snapshot_count_increments(self):
        self.assertEqual(self.mgr.snapshot_count, 0)
        self.mgr.take_snapshot()
        self.assertEqual(self.mgr.snapshot_count, 1)
        self.mgr.take_snapshot()
        self.assertEqual(self.mgr.snapshot_count, 2)


# ==================================================================
# 4. Manifest
# ==================================================================

class TestManifest(unittest.TestCase):
    """Manifest tracks snapshots and persists to disk."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.mgr = SnapshotManager(
            data_dir=self._tmpdir,
            config=SnapshotConfig(remote_sync_enabled=False),
            state_collector=lambda: {"x": 1},
        )

    def test_manifest_empty_initially(self):
        self.assertEqual(len(self.mgr.manifest), 0)

    def test_manifest_grows_with_snapshots(self):
        self.mgr.take_snapshot()
        self.mgr.take_snapshot()
        self.assertEqual(len(self.mgr.manifest), 2)

    def test_manifest_persisted_to_disk(self):
        self.mgr.take_snapshot()
        manifest_path = os.path.join(
            self.mgr.snapshot_dir, self.mgr.config.manifest_filename,
        )
        self.assertTrue(os.path.exists(manifest_path))
        with open(manifest_path) as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)

    def test_manifest_rebuild_from_directory(self):
        self.mgr.take_snapshot(label="a")
        time.sleep(1.1)  # ensure different timestamp in filename
        self.mgr.take_snapshot(label="b")

        # Delete the manifest
        manifest_path = os.path.join(
            self.mgr.snapshot_dir, self.mgr.config.manifest_filename,
        )
        os.remove(manifest_path)

        # New manager rebuilds from directory scan
        mgr2 = SnapshotManager(
            data_dir=self._tmpdir,
            config=SnapshotConfig(remote_sync_enabled=False),
        )
        self.assertEqual(len(mgr2.manifest), 2)

    def test_manifest_capped(self):
        cfg = SnapshotConfig(max_manifest_entries=3, remote_sync_enabled=False)
        mgr = SnapshotManager(
            data_dir=self._tmpdir,
            config=cfg,
            state_collector=lambda: {"x": 1},
        )
        for _ in range(5):
            mgr.take_snapshot()
            time.sleep(0.01)  # ensure unique timestamps
        self.assertLessEqual(len(mgr.manifest), 3)


# ==================================================================
# 5. Rolling 30-day retention
# ==================================================================

class TestRetention(unittest.TestCase):
    """Old snapshots are purged."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def test_old_files_purged(self):
        mgr = SnapshotManager(
            data_dir=self._tmpdir,
            config=SnapshotConfig(retention_days=30, remote_sync_enabled=False),
            state_collector=lambda: {"x": 1},
        )

        # Create a fake old snapshot (31 days ago)
        old_ts = datetime.now(timezone.utc) - timedelta(days=31)
        old_fname = f"snapshot_{old_ts.strftime('%Y%m%dT%H%M%SZ')}.json"
        old_path = os.path.join(mgr.snapshot_dir, old_fname)
        with open(old_path, "w") as f:
            json.dump({"old": True}, f)

        # Add it to manifest manually
        mgr._manifest.append({
            "filename": old_fname,
            "timestamp": old_ts.isoformat(),
            "timestamp_unix": old_ts.timestamp(),
            "label": "old",
            "checksum": None,
            "size_bytes": 10,
        })

        # Take a new snapshot (triggers retention enforcement)
        mgr.take_snapshot()

        # Old file should be gone
        self.assertFalse(os.path.exists(old_path))
        # Manifest should only have the new entry
        filenames = [e["filename"] for e in mgr.manifest]
        self.assertNotIn(old_fname, filenames)

    def test_recent_files_kept(self):
        mgr = SnapshotManager(
            data_dir=self._tmpdir,
            config=SnapshotConfig(retention_days=30, remote_sync_enabled=False),
            state_collector=lambda: {"x": 1},
        )
        entry = mgr.take_snapshot()
        # Immediately enforce retention — fresh snapshot survives
        deleted = mgr._enforce_retention()
        self.assertEqual(deleted, 0)
        self.assertTrue(os.path.exists(
            os.path.join(mgr.snapshot_dir, entry["filename"])
        ))

    def test_purge_older_than(self):
        mgr = SnapshotManager(
            data_dir=self._tmpdir,
            config=SnapshotConfig(retention_days=365, remote_sync_enabled=False),
            state_collector=lambda: {"x": 1},
        )
        mgr.take_snapshot()
        # Purge everything older than 0 days (i.e. nothing since just created)
        # Actually, purge_older_than(0) would purge everything
        deleted = mgr.purge_older_than(0)
        self.assertEqual(deleted, 1)


# ==================================================================
# 6. list_snapshots filtering
# ==================================================================

class TestListSnapshots(unittest.TestCase):
    """Query snapshots with filters."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.mgr = SnapshotManager(
            data_dir=self._tmpdir,
            config=SnapshotConfig(remote_sync_enabled=False),
            state_collector=lambda: {"x": 1},
        )
        self.mgr.take_snapshot(label="auto")
        time.sleep(0.05)
        self.mgr.take_snapshot(label="manual")

    def test_limit(self):
        entries = self.mgr.list_snapshots(limit=1)
        self.assertEqual(len(entries), 1)

    def test_label_filter(self):
        entries = self.mgr.list_snapshots(label="manual")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["label"], "manual")

    def test_most_recent_first(self):
        entries = self.mgr.list_snapshots()
        self.assertGreaterEqual(
            entries[0]["timestamp_unix"],
            entries[1]["timestamp_unix"],
        )


# ==================================================================
# 7. load_snapshot / latest_snapshot
# ==================================================================

class TestLoadSnapshot(unittest.TestCase):
    """Load individual snapshots."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.mgr = SnapshotManager(
            data_dir=self._tmpdir,
            config=SnapshotConfig(remote_sync_enabled=False),
            state_collector=lambda: {"val": 42},
        )

    def test_load_snapshot(self):
        entry = self.mgr.take_snapshot()
        snap = self.mgr.load_snapshot(entry["filename"])
        self.assertIsNotNone(snap)
        self.assertEqual(snap["state"]["val"], 42)

    def test_load_nonexistent(self):
        snap = self.mgr.load_snapshot("does_not_exist.json")
        self.assertIsNone(snap)

    def test_latest_snapshot(self):
        self.mgr.take_snapshot(label="first")
        time.sleep(0.05)
        self.mgr.take_snapshot(label="second")
        snap = self.mgr.latest_snapshot()
        self.assertIsNotNone(snap)
        self.assertEqual(snap["label"], "second")

    def test_latest_when_empty(self):
        snap = self.mgr.latest_snapshot()
        self.assertIsNone(snap)


# ==================================================================
# 8. status()
# ==================================================================

class TestStatus(unittest.TestCase):
    """Status reporting."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.mgr = SnapshotManager(
            data_dir=self._tmpdir,
            config=SnapshotConfig(remote_sync_enabled=False),
            state_collector=lambda: {"x": 1},
        )

    def test_status_keys(self):
        status = self.mgr.status()
        expected_keys = {
            "running", "interval_seconds", "retention_days",
            "total_snapshots", "snapshots_this_session", "disk_usage_bytes",
            "disk_usage_mb", "oldest_snapshot", "newest_snapshot",
            "remote_sync_enabled", "snapshot_dir",
        }
        self.assertEqual(set(status.keys()), expected_keys)

    def test_status_after_snapshot(self):
        self.mgr.take_snapshot()
        status = self.mgr.status()
        self.assertEqual(status["total_snapshots"], 1)
        self.assertEqual(status["snapshots_this_session"], 1)
        self.assertGreater(status["disk_usage_bytes"], 0)
        self.assertIsNotNone(status["newest_snapshot"])


# ==================================================================
# 9. State collection
# ==================================================================

class TestStateCollection(unittest.TestCase):
    """State collector callback is called correctly."""

    def test_collects_state(self):
        tmpdir = tempfile.mkdtemp()
        called = {"count": 0}

        def collector():
            called["count"] += 1
            return {"awareness": 0.9, "iteration": called["count"]}

        mgr = SnapshotManager(
            data_dir=tmpdir,
            config=SnapshotConfig(remote_sync_enabled=False),
            state_collector=collector,
        )
        entry = mgr.take_snapshot()
        self.assertEqual(called["count"], 1)

        snap = mgr.load_snapshot(entry["filename"])
        self.assertEqual(snap["state"]["awareness"], 0.9)

    def test_no_collector_returns_error(self):
        tmpdir = tempfile.mkdtemp()
        mgr = SnapshotManager(
            data_dir=tmpdir,
            config=SnapshotConfig(remote_sync_enabled=False),
        )
        entry = mgr.take_snapshot()
        snap = mgr.load_snapshot(entry["filename"])
        self.assertIn("error", snap["state"])

    def test_collector_exception_caught(self):
        tmpdir = tempfile.mkdtemp()

        def bad_collector():
            raise RuntimeError("boom")

        mgr = SnapshotManager(
            data_dir=tmpdir,
            config=SnapshotConfig(remote_sync_enabled=False),
            state_collector=bad_collector,
        )
        entry = mgr.take_snapshot()
        snap = mgr.load_snapshot(entry["filename"])
        self.assertIn("error", snap["state"])
        self.assertIn("boom", snap["state"]["error"])


# ==================================================================
# 10. Remote sync
# ==================================================================

class TestRemoteSync(unittest.TestCase):
    """Remote sync pushes to Firestore mock."""

    def test_sync_called_when_enabled(self):
        tmpdir = tempfile.mkdtemp()
        mock_fs = MagicMock()
        mock_collection = MagicMock()
        mock_doc = MagicMock()
        mock_fs.collection.return_value = mock_collection
        mock_collection.document.return_value = mock_doc

        mgr = SnapshotManager(
            data_dir=tmpdir,
            config=SnapshotConfig(remote_sync_enabled=True),
            state_collector=lambda: {"x": 1},
            firestore_client=mock_fs,
        )
        mgr.take_snapshot()

        # Give the background thread time to run
        time.sleep(0.5)

        mock_fs.collection.assert_called_once_with("al01_snapshots")
        mock_doc.set.assert_called_once()

    def test_no_sync_when_disabled(self):
        tmpdir = tempfile.mkdtemp()
        mock_fs = MagicMock()

        mgr = SnapshotManager(
            data_dir=tmpdir,
            config=SnapshotConfig(remote_sync_enabled=False),
            state_collector=lambda: {"x": 1},
            firestore_client=mock_fs,
        )
        mgr.take_snapshot()
        time.sleep(0.2)

        mock_fs.collection.assert_not_called()

    def test_no_sync_without_client(self):
        tmpdir = tempfile.mkdtemp()
        mgr = SnapshotManager(
            data_dir=tmpdir,
            config=SnapshotConfig(remote_sync_enabled=True),
            state_collector=lambda: {"x": 1},
            firestore_client=None,
        )
        # Should not raise
        mgr.take_snapshot()


# ==================================================================
# 11. Automatic timer
# ==================================================================

class TestAutoTimer(unittest.TestCase):
    """Background timer fires snapshots automatically."""

    def test_auto_snapshot_fires(self):
        tmpdir = tempfile.mkdtemp()
        mgr = SnapshotManager(
            data_dir=tmpdir,
            config=SnapshotConfig(
                interval_seconds=0.3,  # 300ms for testing
                remote_sync_enabled=False,
            ),
            state_collector=lambda: {"tick": True},
        )
        mgr.start()
        time.sleep(0.8)  # wait for at least 1-2 timer fires
        mgr.stop()
        self.assertGreaterEqual(mgr.snapshot_count, 1)


# ==================================================================
# 12. Organism integration
# ==================================================================

class TestOrganismIntegration(unittest.TestCase):
    """Organism exposes snapshot_manager property and growth_metrics field."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def test_snapshot_manager_default_none(self):
        from al01.organism import Organism
        org = Organism(data_dir=self._tmpdir)
        self.assertIsNone(org.snapshot_manager)

    def test_snapshot_manager_wired(self):
        from al01.organism import Organism
        org = Organism(data_dir=self._tmpdir)
        mgr = SnapshotManager(
            data_dir=self._tmpdir,
            config=SnapshotConfig(remote_sync_enabled=False),
            state_collector=lambda: org.growth_metrics,
        )
        org._snapshot_manager = mgr
        self.assertIsNotNone(org.snapshot_manager)

    def test_growth_metrics_includes_snapshot_field(self):
        from al01.organism import Organism
        org = Organism(data_dir=self._tmpdir)
        mgr = SnapshotManager(
            data_dir=self._tmpdir,
            config=SnapshotConfig(remote_sync_enabled=False),
            state_collector=lambda: {},
        )
        org._snapshot_manager = mgr
        metrics = org.growth_metrics
        self.assertIn("snapshot_manager", metrics)
        self.assertIsNotNone(metrics["snapshot_manager"])

    def test_growth_metrics_snapshot_none_when_no_manager(self):
        from al01.organism import Organism
        org = Organism(data_dir=self._tmpdir)
        metrics = org.growth_metrics
        self.assertIn("snapshot_manager", metrics)
        self.assertIsNone(metrics["snapshot_manager"])


# ==================================================================
# 13. Version bump
# ==================================================================

class TestVersionBump(unittest.TestCase):
    """Version is 3.2."""

    def test_version(self):
        from al01.organism import VERSION
        self.assertEqual(VERSION, "3.9")


# ==================================================================
# 14. Atomic write safety
# ==================================================================

class TestAtomicWrite(unittest.TestCase):
    """Atomic write creates valid file and cleans up on failure."""

    def test_creates_valid_file(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "test.json")
        SnapshotManager._atomic_write(path, {"key": "value"})
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data["key"], "value")

    def test_creates_intermediate_dirs(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "deep", "nested", "test.json")
        SnapshotManager._atomic_write(path, {"nested": True})
        self.assertTrue(os.path.exists(path))


# ==================================================================
# 15. CLI commands (unit-test the functions directly)
# ==================================================================

class TestCLI(unittest.TestCase):
    """CLI snapshot commands work end-to-end."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        # Create a fake state.json
        state_path = os.path.join(self._tmpdir, "state.json")
        with open(state_path, "w") as f:
            json.dump({"awareness": 0.5, "evolution_count": 5}, f)

    def test_snapshot_now(self):
        from al01.cli import cmd_snapshot_now
        import argparse
        args = argparse.Namespace(
            data_dir=self._tmpdir,
            label="test",
            retention_days=30,
        )
        # Should not raise
        cmd_snapshot_now(args)
        # Check file exists
        snap_dir = os.path.join(self._tmpdir, "snapshots", "hourly")
        files = [f for f in os.listdir(snap_dir) if f.startswith("snapshot_")]
        self.assertGreater(len(files), 0)

    def test_snapshot_list(self):
        from al01.cli import cmd_snapshot_now, cmd_snapshot_list
        import argparse
        # Take a snapshot first
        args = argparse.Namespace(
            data_dir=self._tmpdir,
            label="test",
            retention_days=30,
        )
        cmd_snapshot_now(args)
        # List
        args2 = argparse.Namespace(
            data_dir=self._tmpdir,
            limit=10,
            label=None,
            retention_days=30,
        )
        # Should not raise
        cmd_snapshot_list(args2)

    def test_snapshot_status(self):
        from al01.cli import cmd_snapshot_status
        import argparse
        args = argparse.Namespace(
            data_dir=self._tmpdir,
            retention_days=30,
        )
        # Should not raise
        cmd_snapshot_status(args)

    def test_snapshot_purge(self):
        from al01.cli import cmd_snapshot_now, cmd_snapshot_purge
        import argparse
        # Take a snapshot
        args_take = argparse.Namespace(
            data_dir=self._tmpdir,
            label="test",
            retention_days=30,
        )
        cmd_snapshot_now(args_take)
        # Purge all (0 days keeps nothing)
        args = argparse.Namespace(
            data_dir=self._tmpdir,
            days=0,
            retention_days=30,
        )
        cmd_snapshot_purge(args)


if __name__ == "__main__":
    unittest.main()
