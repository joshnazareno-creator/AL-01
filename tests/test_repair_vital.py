"""Tests for the VITAL hash-chain repair tool (repair_chain + CLI command).

Covers:
 1. repair_chain on a clean chain (no-op)
 2. repair_chain on an empty log
 3. repair_chain detects and truncates at hash mismatch
 4. repair_chain detects and truncates at link break
 5. Backup file is created with all original events
 6. Repaired file contains only valid prefix
 7. Head is re-anchored to last valid event
 8. Post-repair appends produce a valid chain
 9. chain_repair event is logged after repair
10. CLI repair-vital command output
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from al01.life_log import LifeLog, _compute_hash, _GENESIS_HASH


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_log(tmp, n_events=20):
    """Create a LifeLog with *n_events* valid events and return it."""
    log = LifeLog(data_dir=tmp, organism_id="AL-01", snapshot_interval=9999)
    for i in range(n_events):
        log.append_event(event_type="test", payload={"cycle": i})
    return log


def _read_all_events(log_path):
    """Read every event from a JSONL file."""
    events = []
    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _corrupt_link_at(log_path, break_index):
    """Corrupt linkage at *break_index* by replacing prev_hash with garbage.

    This simulates the exact failure mode seen in production: a process
    crash between writing an event and flushing the prev_hash linkage.
    """
    events = _read_all_events(log_path)
    events[break_index]["prev_hash"] = "deadbeef" * 8  # 64-char garbage
    # Recompute the hash of the corrupted event so `hash` itself is valid
    # for the (now-broken) prev_hash — mirrors real corruption where the
    # event was written with a wrong prev_hash but a consistent own-hash.
    ev = events[break_index]
    ev["hash"] = _compute_hash(ev["prev_hash"], ev["payload"], ev["t"], ev["seq"])
    with open(log_path, "w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev, separators=(",", ":"), default=str) + "\n")


def _corrupt_hash_at(log_path, break_index):
    """Corrupt the own-hash of the event at *break_index*."""
    events = _read_all_events(log_path)
    events[break_index]["hash"] = "badc0ffee" + "0" * 55  # 64 chars, wrong
    with open(log_path, "w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev, separators=(",", ":"), default=str) + "\n")


# ═══════════════════════════════════════════════════════════════════════
# 1. Clean chain — no repair needed
# ═══════════════════════════════════════════════════════════════════════

class TestRepairCleanChain:
    def test_clean_chain_returns_clean_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _make_log(tmp, 20)
            report = log.repair_chain()
            assert report["status"] == "CLEAN"
            assert report["first_broken_seq"] is None
            assert report["events_dropped"] == 0
            assert report["backup_path"] is None
            assert report["last_valid_seq"] == 20

    def test_clean_chain_preserves_all_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _make_log(tmp, 15)
            log.repair_chain()
            assert log.event_count() == 15

    def test_clean_chain_head_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _make_log(tmp, 10)
            log.repair_chain()
            assert log.verify(last_n=10)


# ═══════════════════════════════════════════════════════════════════════
# 2. Empty log
# ═══════════════════════════════════════════════════════════════════════

class TestRepairEmptyLog:
    def test_empty_log_returns_empty_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = LifeLog(data_dir=tmp, organism_id="AL-01")
            report = log.repair_chain()
            assert report["status"] == "EMPTY"
            assert report["total_events"] == 0


# ═══════════════════════════════════════════════════════════════════════
# 3. Link break — truncate at broken link
# ═══════════════════════════════════════════════════════════════════════

class TestRepairLinkBreak:
    def test_detects_link_break(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _make_log(tmp, 30)
            log_path = os.path.join(tmp, "life_log.jsonl")
            # Break linkage at event index 15 (seq=16)
            _corrupt_link_at(log_path, 15)

            # Reload to pick up corrupted file
            log2 = LifeLog(data_dir=tmp, organism_id="AL-01")
            report = log2.repair_chain()

            assert report["status"] == "REPAIRED"
            assert report["first_broken_seq"] == 16
            assert report["last_valid_seq"] == 15
            assert report["events_dropped"] == 15  # events 16-30 dropped

    def test_backup_contains_all_original_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _make_log(tmp, 25)
            log_path = os.path.join(tmp, "life_log.jsonl")
            _corrupt_link_at(log_path, 10)

            log2 = LifeLog(data_dir=tmp, organism_id="AL-01")
            report = log2.repair_chain()

            backup_path = report["backup_path"]
            assert os.path.exists(backup_path)
            backup_events = _read_all_events(backup_path)
            assert len(backup_events) == 25  # all original events preserved

    def test_repaired_file_has_valid_prefix_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _make_log(tmp, 20)
            log_path = os.path.join(tmp, "life_log.jsonl")
            _corrupt_link_at(log_path, 12)

            log2 = LifeLog(data_dir=tmp, organism_id="AL-01")
            report = log2.repair_chain()

            # Repaired file = 12 valid events + 1 chain_repair event
            events = _read_all_events(log_path)
            assert len(events) == 13
            assert events[-1]["event_type"] == "chain_repair"

    def test_head_reanchored_after_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _make_log(tmp, 20)
            log_path = os.path.join(tmp, "life_log.jsonl")
            _corrupt_link_at(log_path, 10)

            log2 = LifeLog(data_dir=tmp, organism_id="AL-01")
            log2.repair_chain()

            # Head should now point to the chain_repair event (last appended)
            head = log2.head
            # head_seq = 10 valid + 1 repair event = 11
            assert head["head_seq"] == 11


# ═══════════════════════════════════════════════════════════════════════
# 4. Hash mismatch — truncate at corrupted hash
# ═══════════════════════════════════════════════════════════════════════

class TestRepairHashMismatch:
    def test_detects_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _make_log(tmp, 20)
            log_path = os.path.join(tmp, "life_log.jsonl")
            # Corrupt own-hash at event index 8 (seq=9)
            _corrupt_hash_at(log_path, 8)

            log2 = LifeLog(data_dir=tmp, organism_id="AL-01")
            report = log2.repair_chain()

            assert report["status"] == "REPAIRED"
            assert report["first_broken_seq"] == 9
            assert report["last_valid_seq"] == 8
            assert report["events_dropped"] == 12  # events 9-20

    def test_repaired_chain_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _make_log(tmp, 20)
            log_path = os.path.join(tmp, "life_log.jsonl")
            _corrupt_hash_at(log_path, 8)

            log2 = LifeLog(data_dir=tmp, organism_id="AL-01")
            log2.repair_chain()
            assert log2.verify(last_n=100)


# ═══════════════════════════════════════════════════════════════════════
# 5. Post-repair chain continuity
# ═══════════════════════════════════════════════════════════════════════

class TestPostRepairAppend:
    def test_appends_after_repair_are_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _make_log(tmp, 20)
            log_path = os.path.join(tmp, "life_log.jsonl")
            _corrupt_link_at(log_path, 10)

            log2 = LifeLog(data_dir=tmp, organism_id="AL-01")
            log2.repair_chain()

            # Append new events
            for i in range(5):
                log2.append_event(event_type="post_repair", payload={"i": i})

            # Full chain should verify
            assert log2.verify(last_n=100)
            # Total: 10 valid + 1 repair + 5 new = 16
            assert log2.event_count() == 16

    def test_verify_full_report_passes_after_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _make_log(tmp, 30)
            log_path = os.path.join(tmp, "life_log.jsonl")
            _corrupt_link_at(log_path, 20)

            log2 = LifeLog(data_dir=tmp, organism_id="AL-01")
            log2.repair_chain()

            report = log2.verify_full_report(last_n=100)
            assert report["status"] == "PASS"


# ═══════════════════════════════════════════════════════════════════════
# 6. Edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestRepairEdgeCases:
    def test_break_at_first_event(self):
        """If the very first event is corrupt, chain resets to genesis."""
        with tempfile.TemporaryDirectory() as tmp:
            log = _make_log(tmp, 10)
            log_path = os.path.join(tmp, "life_log.jsonl")
            _corrupt_hash_at(log_path, 0)

            log2 = LifeLog(data_dir=tmp, organism_id="AL-01")
            report = log2.repair_chain()

            assert report["status"] == "REPAIRED"
            assert report["first_broken_seq"] == 1
            assert report["last_valid_seq"] == 0
            assert report["events_dropped"] == 10

    def test_break_at_second_event(self):
        """Link break at second event preserves only the first."""
        with tempfile.TemporaryDirectory() as tmp:
            log = _make_log(tmp, 10)
            log_path = os.path.join(tmp, "life_log.jsonl")
            _corrupt_link_at(log_path, 1)

            log2 = LifeLog(data_dir=tmp, organism_id="AL-01")
            report = log2.repair_chain()

            assert report["status"] == "REPAIRED"
            assert report["last_valid_seq"] == 1
            assert report["events_dropped"] == 9

    def test_break_at_last_event(self):
        """Link break at the final event drops only one."""
        with tempfile.TemporaryDirectory() as tmp:
            log = _make_log(tmp, 10)
            log_path = os.path.join(tmp, "life_log.jsonl")
            _corrupt_link_at(log_path, 9)

            log2 = LifeLog(data_dir=tmp, organism_id="AL-01")
            report = log2.repair_chain()

            assert report["status"] == "REPAIRED"
            assert report["last_valid_seq"] == 9
            assert report["events_dropped"] == 1

    def test_chain_repair_event_logged(self):
        """A 'chain_repair' event is appended after successful repair."""
        with tempfile.TemporaryDirectory() as tmp:
            log = _make_log(tmp, 15)
            log_path = os.path.join(tmp, "life_log.jsonl")
            _corrupt_link_at(log_path, 8)

            log2 = LifeLog(data_dir=tmp, organism_id="AL-01")
            log2.repair_chain()

            events = _read_all_events(log_path)
            repair_events = [e for e in events if e["event_type"] == "chain_repair"]
            assert len(repair_events) == 1
            payload = repair_events[0]["payload"]
            assert payload["first_broken_seq"] == 9
            assert payload["events_dropped"] == 7

    def test_integrity_status_set_to_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _make_log(tmp, 10)
            log_path = os.path.join(tmp, "life_log.jsonl")
            _corrupt_link_at(log_path, 5)

            log2 = LifeLog(data_dir=tmp, organism_id="AL-01")
            log2.repair_chain()
            assert log2.integrity_status == "OK"


# ═══════════════════════════════════════════════════════════════════════
# 7. CLI command
# ═══════════════════════════════════════════════════════════════════════

class TestCLIRepairVital:
    def test_cli_repair_clean_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_log(tmp, 10)
            result = subprocess.run(
                [sys.executable, "-m", "al01.cli", "repair-vital", "--data-dir", tmp],
                capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), ".."),
            )
            assert result.returncode == 0
            assert "no repair needed" in result.stdout.lower() or "already valid" in result.stdout.lower()

    def test_cli_repair_broken_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _make_log(tmp, 20)
            _corrupt_link_at(os.path.join(tmp, "life_log.jsonl"), 10)

            result = subprocess.run(
                [sys.executable, "-m", "al01.cli", "repair-vital", "--data-dir", tmp],
                capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), ".."),
            )
            assert result.returncode == 0
            assert "11" in result.stdout  # first broken seq
            assert "10" in result.stdout  # last valid seq
            assert "PASS" in result.stdout or "restored" in result.stdout.lower()
