"""AL-01 CLI — ``python -m al01.cli verify --last 500``."""

from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from al01 import storage
from al01.genesis_vault import GenesisVault
from al01.life_log import LifeLog
from al01.snapshot_manager import SnapshotConfig, SnapshotManager


def cmd_verify(args: argparse.Namespace) -> None:
    """Verify the hash-chain integrity of the life log."""
    data_dir = args.data_dir
    last_n = args.last

    if not os.path.exists(os.path.join(data_dir, "life_log.jsonl")):
        print(f"No life log found at {data_dir}/life_log.jsonl")
        sys.exit(1)

    identity_path = os.path.join(os.path.dirname(__file__), "identity.json")
    organism_id = "AL-01"
    if os.path.exists(identity_path):
        with open(identity_path, "r", encoding="utf-8") as fh:
            identity = json.load(fh)
            organism_id = identity.get("name", organism_id)

    log = LifeLog(data_dir=data_dir, organism_id=organism_id)
    report = log.verify_full_report(last_n=last_n)

    total_events = log.event_count()
    print(f"AL-01 Life Log Verification")
    print(f"===========================")
    print(f"Log file:        {data_dir}/life_log.jsonl")
    print(f"Total events:    {total_events}")
    print(f"Checked (last):  {report['checked']}")
    print(f"Head seq:        {log.head.get('head_seq', 0)}")
    print(f"Head hash:       {log.head.get('head_hash', '?')[:16]}...")
    print(f"")

    if report["status"] == "PASS":
        print(f"Result: ✓ PASS")
        sys.exit(0)
    else:
        print(f"Result: ✗ FAIL")
        print(f"First broken seq: {report.get('first_broken_seq', '?')}")
        print(f"Reason: {report.get('reason', 'unknown')}")
        sys.exit(1)


# ── Repair command ───────────────────────────────────────────────────

def cmd_repair_vital(args: argparse.Namespace) -> None:
    """Repair the VITAL hash chain by truncating at the first broken link."""
    data_dir = args.data_dir

    if not os.path.exists(os.path.join(data_dir, "life_log.jsonl")):
        print(f"No life log found at {data_dir}/life_log.jsonl")
        sys.exit(1)

    identity_path = os.path.join(os.path.dirname(__file__), "identity.json")
    organism_id = "AL-01"
    if os.path.exists(identity_path):
        with open(identity_path, "r", encoding="utf-8") as fh:
            identity = json.load(fh)
            organism_id = identity.get("name", organism_id)

    log = LifeLog(data_dir=data_dir, organism_id=organism_id)

    total_events = log.event_count()
    print(f"AL-01 VITAL Chain Repair")
    print(f"========================")
    print(f"Log file:        {data_dir}/life_log.jsonl")
    print(f"Total events:    {total_events}")
    print()

    report = log.repair_chain()

    if report["status"] == "EMPTY":
        print("Result: Log is empty — nothing to repair.")
        sys.exit(0)

    if report["status"] == "CLEAN":
        print("Result: [OK] Chain is already valid — no repair needed.")
        print(f"Last valid seq:  {report['last_valid_seq']}")
        sys.exit(0)

    # REPAIRED
    print(f"First broken seq:   {report['first_broken_seq']}")
    print(f"Last valid seq:     {report['last_valid_seq']}")
    print(f"Events dropped:     {report['events_dropped']}")
    print(f"Backup saved to:    {report['backup_path']}")
    print()

    # Verify the repaired chain
    verify = log.verify_full_report(last_n=report['last_valid_seq'] + 10)
    if verify["status"] == "PASS":
        print(f"Post-repair verify: [OK] PASS")
        print(f"New head seq:       {log.head.get('head_seq', 0)}")
        print(f"New head hash:      {log.head.get('head_hash', '?')[:16]}...")
        print()
        print("Result: [OK] VITAL integrity restored.")
    else:
        print(f"Post-repair verify: [FAIL] — {verify.get('reason', 'unknown')}")
        print("Manual intervention may be required.")
        sys.exit(1)


# ── Snapshot commands ────────────────────────────────────────────────

def _make_snapshot_manager(args: argparse.Namespace) -> SnapshotManager:
    """Create a SnapshotManager pointing at the workspace directory."""
    data_dir = getattr(args, "data_dir", ".")
    config = SnapshotConfig(
        retention_days=getattr(args, "retention_days", 30),
    )
    return SnapshotManager(data_dir=data_dir, config=config)


def cmd_snapshot_now(args: argparse.Namespace) -> None:
    """Take an immediate snapshot of the current state.json."""
    mgr = _make_snapshot_manager(args)

    # Use a simple collector that reads state.json from disk
    state_path = os.path.join(args.data_dir, "state.json")
    if not os.path.exists(state_path):
        print(f"No state.json found at {args.data_dir}/state.json")
        sys.exit(1)

    with open(state_path, "r", encoding="utf-8") as fh:
        state = json.load(fh)

    mgr._state_collector = lambda: state  # type: ignore[assignment]
    entry = mgr.take_snapshot(label=args.label or "manual")
    print(f"Snapshot written: {entry['filename']}")
    print(f"  Timestamp: {entry['timestamp']}")
    print(f"  Checksum:  {entry['checksum'][:16]}...")
    print(f"  Size:      {entry['size_bytes']} bytes")


def cmd_snapshot_list(args: argparse.Namespace) -> None:
    """List recent snapshots."""
    mgr = _make_snapshot_manager(args)
    entries = mgr.list_snapshots(limit=args.limit, label=args.label)

    if not entries:
        print("No snapshots found.")
        return

    print(f"{'Filename':<40} {'Timestamp':<28} {'Label':<12} {'Size':>10}")
    print("-" * 94)
    for e in entries:
        size_str = f"{e.get('size_bytes', 0):,}"
        print(f"{e['filename']:<40} {e['timestamp']:<28} {e.get('label', '?'):<12} {size_str:>10}")
    print(f"\nTotal: {len(entries)} snapshot(s)")


def cmd_snapshot_status(args: argparse.Namespace) -> None:
    """Show snapshot manager status."""
    mgr = _make_snapshot_manager(args)
    status = mgr.status()

    print(f"AL-01 Snapshot Status")
    print(f"=====================")
    print(f"Snapshot dir:     {status['snapshot_dir']}")
    print(f"Total snapshots:  {status['total_snapshots']}")
    print(f"Disk usage:       {status['disk_usage_mb']:.2f} MB")
    print(f"Retention:        {status['retention_days']} days")
    print(f"Oldest:           {status['oldest_snapshot'] or 'N/A'}")
    print(f"Newest:           {status['newest_snapshot'] or 'N/A'}")
    print(f"Remote sync:      {'enabled' if status['remote_sync_enabled'] else 'disabled'}")


def cmd_snapshot_purge(args: argparse.Namespace) -> None:
    """Purge snapshots older than N days."""
    mgr = _make_snapshot_manager(args)
    deleted = mgr.purge_older_than(args.days)
    print(f"Purged {deleted} snapshot(s) older than {args.days} days.")


# ── Genesis Vault commands ───────────────────────────────────────────

def cmd_vault_status(args: argparse.Namespace) -> None:
    """Show Genesis Vault status."""
    vault = GenesisVault(data_dir=args.data_dir)
    status = vault.status()

    print(f"AL-01 Genesis Vault")
    print(f"===================")
    print(f"Seed name:      {status['seed_name']}")
    print(f"Created:        {status['created_at'] or 'N/A'}")
    print(f"Reseed count:   {status['reseed_count']}")
    print(f"Seed traits:    {json.dumps(status['seed_traits'], indent=2)}")
    print(f"Mutation rate:  {status['mutation_rate']}")
    print(f"Mutation delta: {status['mutation_delta']}")
    if status['last_reseed']:
        last = status['last_reseed']
        print(f"\nLast reseed:")
        print(f"  Reseed #:     {last['reseed_number']}")
        print(f"  Organism:     {last['organism_id']}")
        print(f"  Cycle:        {last['cycle']}")
        print(f"  Timestamp:    {last['timestamp']}")


def cmd_vault_history(args: argparse.Namespace) -> None:
    """Show reseed history."""
    vault = GenesisVault(data_dir=args.data_dir)
    history = vault.reseed_history

    if not history:
        print("No reseed events recorded.")
        return

    print(f"{'#':<4} {'Organism':<25} {'Cycle':<8} {'Timestamp':<28} {'Fitness':>8}")
    print("-" * 77)
    for h in history:
        print(
            f"{h['reseed_number']:<4} {h['organism_id']:<25} "
            f"{h['cycle']:<8} {h['timestamp']:<28} "
            f"{h.get('child_fitness', 0):>8.4f}"
        )
    print(f"\nTotal: {len(history)} reseed event(s)")


def main() -> None:
    parser = argparse.ArgumentParser(prog="al01.cli", description="AL-01 command-line tools")
    sub = parser.add_subparsers(dest="command")

    # v3.22: All CLI commands default to absolute paths via storage module
    _data = storage.DATA_DIR
    _base = storage.BASE_DIR

    # verify
    verify_p = sub.add_parser("verify", help="Verify life-log hash-chain integrity")
    verify_p.add_argument("--last", type=int, default=500, help="Number of recent entries to verify (default: 500)")
    verify_p.add_argument("--data-dir", type=str, default=_data, help=f"Data directory (default: {_data})")

    # snapshot now
    snap_now = sub.add_parser("snapshot", help="Take an immediate state snapshot")
    snap_now.add_argument("--data-dir", type=str, default=_base, help=f"Base directory (default: {_base})")
    snap_now.add_argument("--label", type=str, default="manual", help="Snapshot label")
    snap_now.add_argument("--retention-days", type=int, default=30, help="Retention period in days")

    # snapshot-list
    snap_list = sub.add_parser("snapshot-list", help="List recent snapshots")
    snap_list.add_argument("--data-dir", type=str, default=_base, help=f"Base directory (default: {_base})")
    snap_list.add_argument("--limit", type=int, default=20, help="Max entries to show")
    snap_list.add_argument("--label", type=str, default=None, help="Filter by label")
    snap_list.add_argument("--retention-days", type=int, default=30, help="Retention period in days")

    # snapshot-status
    snap_status = sub.add_parser("snapshot-status", help="Show snapshot manager status")
    snap_status.add_argument("--data-dir", type=str, default=_base, help=f"Base directory (default: {_base})")
    snap_status.add_argument("--retention-days", type=int, default=30, help="Retention period in days")

    # snapshot-purge
    snap_purge = sub.add_parser("snapshot-purge", help="Purge old snapshots")
    snap_purge.add_argument("--days", type=int, default=30, help="Delete snapshots older than N days")
    snap_purge.add_argument("--data-dir", type=str, default=_base, help=f"Base directory (default: {_base})")
    snap_purge.add_argument("--retention-days", type=int, default=30, help="Retention period in days")

    # vault
    vault_p = sub.add_parser("vault", help="Show Genesis Vault status")
    vault_p.add_argument("--data-dir", type=str, default=_data, help=f"Data directory (default: {_data})")

    # vault-history
    vault_hist = sub.add_parser("vault-history", help="Show reseed event history")
    vault_hist.add_argument("--data-dir", type=str, default=_data, help=f"Data directory (default: {_data})")

    # repair-vital
    repair_p = sub.add_parser("repair-vital", help="Repair broken VITAL hash chain")
    repair_p.add_argument("--data-dir", type=str, default=_data, help=f"Data directory (default: {_data})")

    args = parser.parse_args()
    # Resolve to absolute in case user passes a relative --data-dir
    if hasattr(args, 'data_dir'):
        args.data_dir = os.path.abspath(args.data_dir)

    if args.command == "verify":
        cmd_verify(args)
    elif args.command == "snapshot":
        cmd_snapshot_now(args)
    elif args.command == "snapshot-list":
        cmd_snapshot_list(args)
    elif args.command == "snapshot-status":
        cmd_snapshot_status(args)
    elif args.command == "snapshot-purge":
        cmd_snapshot_purge(args)
    elif args.command == "vault":
        cmd_vault_status(args)
    elif args.command == "vault-history":
        cmd_vault_history(args)
    elif args.command == "repair-vital":
        cmd_repair_vital(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
