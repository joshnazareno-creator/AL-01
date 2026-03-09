"""AL-01 v3.22 — Global storage configuration.

Ensures ALL file writes go to a single base directory on the correct drive,
regardless of the working directory at launch time.

Override via the ``AL01_BASE_DIR`` environment variable:
    set AL01_BASE_DIR=D:\\AL-01

If unset, defaults to the directory containing the al01 package (i.e. the
repository root), which is ``D:\\AL-01`` in the standard installation.

All path helpers return **absolute** paths so no module ever writes relative
to the working directory.
"""

from __future__ import annotations

import glob
import os

# ---------------------------------------------------------------------------
# Base directory resolution
# ---------------------------------------------------------------------------
# Priority:
#   1. AL01_BASE_DIR env var (explicit override)
#   2. Parent of the al01/ package directory (repo root)

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))       # .../al01/
_REPO_ROOT = os.path.dirname(_PACKAGE_DIR)                      # .../AL-01/

BASE_DIR: str = os.environ.get("AL01_BASE_DIR") or _REPO_ROOT

# ---------------------------------------------------------------------------
# Derived directory paths (all absolute)
# ---------------------------------------------------------------------------

DATA_DIR: str = os.path.join(BASE_DIR, "data")
LOG_DIR: str = BASE_DIR                     # al01.log lives at repo root
DB_DIR: str = os.path.join(BASE_DIR, "db")  # v3.22: dedicated db directory
BACKUP_DIR: str = os.path.join(BASE_DIR, "backups")
SNAPSHOT_DIR: str = os.path.join(BASE_DIR, "snapshots")
TMP_DIR: str = os.path.join(BASE_DIR, "tmp")  # v3.22: local temp dir (avoids C:\)

# ---------------------------------------------------------------------------
# Size limits
# ---------------------------------------------------------------------------

JSONL_MAX_BYTES: int = 50 * 1024 * 1024     # 50 MB per JSONL log file
JSONL_BACKUP_COUNT: int = 3                  # keep .1, .2, .3 backups
SNAPSHOT_MAX_COUNT: int = 200                # keep the newest 200 tick snapshots
DISK_WARN_BYTES: int = 10 * 1024 * 1024 * 1024  # warn if BASE_DIR exceeds 10 GB


def base_dir() -> str:
    """Absolute path to the AL-01 storage root."""
    return BASE_DIR


def data_dir() -> str:
    """Absolute path to ``data/`` (VITAL logs, genesis vault, etc.)."""
    return DATA_DIR


def db_path() -> str:
    """Absolute path to the SQLite database file."""
    return os.path.join(DB_DIR, "al01.db")


def log_path() -> str:
    """Absolute path to the main log file."""
    return os.path.join(LOG_DIR, "al01.log")


def env_path() -> str:
    """Absolute path to the ``.env`` file."""
    return os.path.join(BASE_DIR, ".env")


def tmp_dir() -> str:
    """Absolute path to the local temp directory (avoids system temp on C:)."""
    os.makedirs(TMP_DIR, exist_ok=True)
    return TMP_DIR


def ensure_dirs() -> None:
    """Create all storage directories if they don't exist."""
    for d in (DATA_DIR, DB_DIR, BACKUP_DIR, TMP_DIR):
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# JSONL rotation
# ---------------------------------------------------------------------------

def rotate_jsonl(path: str,
                 max_bytes: int = JSONL_MAX_BYTES,
                 backup_count: int = JSONL_BACKUP_COUNT) -> None:
    """Rotate *path* when it exceeds *max_bytes*.

    Uses numbered backups: ``path.1`` (newest) … ``path.{backup_count}``
    (oldest).  The oldest backup beyond *backup_count* is deleted.
    """
    try:
        if os.path.getsize(path) < max_bytes:
            return
    except OSError:
        return

    # Shift existing backups:  .3 → deleted, .2 → .3, .1 → .2
    for i in range(backup_count, 0, -1):
        src = f"{path}.{i}" if i > 1 else path  # .1 is the current file
        # Actually: shift .N → .N+1
    # Re-do properly: shift highest first
    for i in range(backup_count, 1, -1):
        src = f"{path}.{i - 1}"
        dst = f"{path}.{i}"
        if os.path.exists(src):
            try:
                if os.path.exists(dst):
                    os.remove(dst)
                os.rename(src, dst)
            except OSError:
                pass

    # Rename current file to .1
    try:
        dst_1 = f"{path}.1"
        if os.path.exists(dst_1):
            os.remove(dst_1)
        os.rename(path, dst_1)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Tick-snapshot retention
# ---------------------------------------------------------------------------

def cleanup_tick_snapshots(snapshot_dir: str | None = None,
                           keep: int = SNAPSHOT_MAX_COUNT) -> int:
    """Delete oldest ``snap_*.json`` files beyond *keep* newest.

    Returns the number of files deleted.
    """
    snap_dir = snapshot_dir or os.path.join(DATA_DIR, "snapshots")
    if not os.path.isdir(snap_dir):
        return 0

    files = sorted(
        glob.glob(os.path.join(snap_dir, "snap_*.json")),
        key=os.path.getmtime,
    )
    to_remove = files[: max(0, len(files) - keep)]
    removed = 0
    for fp in to_remove:
        try:
            os.remove(fp)
            removed += 1
        except OSError:
            pass
    return removed


# ---------------------------------------------------------------------------
# Disk usage monitoring
# ---------------------------------------------------------------------------

def dir_size_bytes(path: str | None = None) -> int:
    """Return total size in bytes of all files under *path* (recursive).

    Defaults to BASE_DIR.  Returns 0 on error.
    """
    root = path or BASE_DIR
    total = 0
    try:
        for dirpath, _dirnames, filenames in os.walk(root):
            for f in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, f))
                except OSError:
                    pass
    except OSError:
        pass
    return total


def check_disk_usage(warn_bytes: int = DISK_WARN_BYTES) -> dict:
    """Check BASE_DIR disk usage and return a status dict.

    Returns::

        {
            "base_dir": str,
            "used_bytes": int,
            "used_mb": float,
            "used_gb": float,
            "warn_threshold_gb": float,
            "warning": bool,
            "message": str,
        }
    """
    used = dir_size_bytes()
    used_mb = used / (1024 * 1024)
    used_gb = used / (1024 * 1024 * 1024)
    warn_gb = warn_bytes / (1024 * 1024 * 1024)
    warning = used >= warn_bytes
    if warning:
        msg = f"AL-01 storage exceeds {warn_gb:.0f} GB — currently {used_gb:.2f} GB"
    else:
        msg = f"AL-01 storage: {used_mb:.1f} MB ({used_gb:.2f} GB) — within {warn_gb:.0f} GB limit"
    return {
        "base_dir": BASE_DIR,
        "used_bytes": used,
        "used_mb": round(used_mb, 2),
        "used_gb": round(used_gb, 4),
        "warn_threshold_gb": round(warn_gb, 1),
        "warning": warning,
        "message": msg,
    }
