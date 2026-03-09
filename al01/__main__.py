"""Entrypoint for ``python -m al01``.

Bootstrap sequence:
  1. Load .env (API key, config)
  2. Configure logging (console + file)
  3. Initialize Database (SQLite)
  4. Initialize LifeLog (append-only hash chain)
  5. Initialize PolicyManager (adaptive weights)
  6. Initialize MemoryManager (local-first, throttled Firestore, SQLite)
  7. Initialize Population (multi-organism registry)
  8. Initialize Brain (AI hooks)
  9. Initialize Organism (loads / creates default state, genome, v2.0)
  10. Boot organism
  11. Start FastAPI server on port 8000 (background thread)
  12. Enter run_loop (blocks, heartbeat every interval)
  13. On Ctrl+C → graceful shutdown → final persist → exit

v3.22: All paths are absolute (anchored to AL01_BASE_DIR or repo root).
       No file writes go to C: drive regardless of working directory.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import threading

from al01 import storage
from al01.api import create_app
from al01.autonomy import AutonomyConfig, AutonomyEngine
from al01.brain import Brain
from al01.database import Database
from al01.environment import Environment, EnvironmentConfig
from al01.genesis_vault import GenesisVault
from al01.life_log import LifeLog
from al01.memory_manager import MemoryManager
from al01.organism import MetabolismConfig, Organism, VERSION
from al01.policy import PolicyManager
from al01.population import Population
from al01.snapshot_manager import SnapshotConfig, SnapshotManager

INTERVAL = 5
API_PORT = 8000


def _load_dotenv() -> None:
    """Load .env file from the base directory (not cwd)."""
    env_file = storage.env_path()
    if not os.path.exists(env_file):
        return
    try:
        with open(env_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("\"'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass  # Best-effort


def _configure_logging() -> None:
    fmt = "[%(asctime)s] %(name)s %(levelname)s: %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S%z"

    # Console handler
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        stream=sys.stdout,
    )

    # v3.22: Rotating file handler — 10 MB per file, 5 backups (al01.log)
    log_file = storage.log_path()
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    file_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(file_handler)


def _start_api_server(organism: Organism, api_key: str | None) -> None:
    """Start uvicorn in a daemon thread so it doesn't block the organism loop."""
    import uvicorn

    app = create_app(organism, api_key=api_key)
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=API_PORT,
        log_level="info",
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True, name="al01-api")
    thread.start()


def main() -> None:
    # --- 0. Load .env ---
    _load_dotenv()

    # --- 1. Logging ---
    _configure_logging()
    log = logging.getLogger("al01.main")
    log.info("[BOOT] AL-01 v%s bootstrap starting", VERSION)
    log.info("[BOOT] Base directory: %s", storage.BASE_DIR)

    # v3.22: Create all storage directories and anchor temp writes
    storage.ensure_dirs()
    import tempfile
    tempfile.tempdir = storage.tmp_dir()

    # v3.22: All paths are absolute — anchored to storage.BASE_DIR
    base = storage.BASE_DIR
    data = storage.DATA_DIR
    os.makedirs(data, exist_ok=True)

    # --- 2. Database (SQLite) ---
    # v3.22: Migrate legacy al01.db from repo root → db/ subdirectory
    _new_db = storage.db_path()
    _old_db = os.path.join(storage.BASE_DIR, "al01.db")
    if os.path.exists(_old_db) and not os.path.exists(_new_db):
        import shutil
        os.makedirs(os.path.dirname(_new_db), exist_ok=True)
        shutil.move(_old_db, _new_db)
        log.info("[BOOT] Migrated database: %s → %s", _old_db, _new_db)
    db = Database(db_path=_new_db)

    # --- 3. VITAL subsystems ---
    life_log = LifeLog(data_dir=data, organism_id="AL-01")
    policy = PolicyManager(data_dir=data)
    log.info("[BOOT] VITAL subsystems initialized (LifeLog + Policy)")

    # --- 4. MemoryManager ---
    log.info("[BOOT] Initializing MemoryManager")
    memory = MemoryManager(data_dir=base, database=db)
    backend = "Firestore" if memory.firestore_enabled() else "local JSON"
    log.info("[BOOT] Persistence backend: %s + SQLite", backend)

    # --- 5. Population + Brain ---
    population = Population(data_dir=base, parent_id="AL-01")
    log.info("[BOOT] Population registry initialized (%d members)", population.size)

    ai_api_key = os.environ.get("OPENAI_API_KEY")
    brain = Brain(api_key=ai_api_key)
    log.info("[BOOT] Brain initialized (enabled=%s)", brain.enabled)

    # --- 5b. Autonomy engine (deterministic, local) ---
    autonomy_config = AutonomyConfig(
        decision_interval=10,
        fitness_threshold=0.45,
        stagnation_window=10,
        stagnation_variance_epsilon=1e-4,
    )
    autonomy = AutonomyEngine(data_dir=data, config=autonomy_config)
    log.info("[BOOT] Autonomy engine initialized (interval=%d, threshold=%.2f)",
             autonomy_config.decision_interval, autonomy_config.fitness_threshold)

    # --- 6. Organism ---
    config = MetabolismConfig(
        pulse_interval=1,
        reflect_interval=30,
        persist_interval=15,
        pulse_log_interval=15,
        heartbeat_persist_interval=3,
        evolve_interval=30,           # every 30 ticks = ~150s at INTERVAL=5
        population_interact_interval=60,  # every 60 ticks = ~300s
        autonomy_interval=10,         # every 10 ticks = ~50s at INTERVAL=5
        memory_snapshot_interval=100,  # v3.21: tick snapshot every 100 ticks = ~500s
    )
    log.info("[BOOT] Initializing Organism v%s (genome + population + brain)", VERSION)
    # v3.18: Production environment with pool floor for ecosystem stability
    env_cfg = EnvironmentConfig(resource_pool_min_floor=50.0)
    environment = Environment(config=env_cfg)
    organism = Organism(
        data_dir=base,
        config=config,
        memory_manager=memory,
        life_log=life_log,
        policy=policy,
        population=population,
        brain=brain,
        autonomy=autonomy,
        environment=environment,
    )

    # --- 6b. Snapshot manager (hourly auto-snapshots) ---
    fs_client = memory._db if memory.firestore_enabled() else None
    snap_config = SnapshotConfig(
        interval_seconds=3600,   # 1 hour
        retention_days=30,       # rolling 30-day archive
        remote_sync_enabled=memory.firestore_enabled(),
    )
    snapshot_mgr = SnapshotManager(
        data_dir=base,
        config=snap_config,
        state_collector=lambda: organism.growth_metrics,
        firestore_client=fs_client,
    )
    organism._snapshot_manager = snapshot_mgr
    log.info("[BOOT] Snapshot manager initialized (interval=%ds, retention=%dd)",
             int(snap_config.interval_seconds), snap_config.retention_days)

    # --- 6c. Genesis Vault (extinction recovery) ---
    genesis_vault = GenesisVault(data_dir=data)
    organism._genesis_vault = genesis_vault
    log.info("[BOOT] Genesis Vault initialized (reseeds=%d)", genesis_vault.reseed_count)

    # --- 7. Boot ---
    organism.boot()
    state = dict(organism.state)
    log.info(
        "[BOOT] Organism online: evolution_count=%d awareness=%.4f state_version=%d interactions=%d fitness=%.4f pop=%d integrity=%s",
        state.get("evolution_count", 0),
        state.get("awareness", 0.0),
        state.get("state_version", 0),
        state.get("interaction_count", 0),
        organism.genome.fitness,
        population.size,
        life_log.integrity_status,
    )

    # --- 8. Identity + growth snapshot ---
    db.set_metadata("name", "AL-01")
    db.set_metadata("version", VERSION)
    organism.record_growth_snapshot()
    log.info("[BOOT] Identity metadata stored, initial growth snapshot recorded")

    # --- 9. API server ---
    api_key = os.environ.get("AL01_API_KEY")
    if api_key:
        log.info("[BOOT] API key authentication enabled")
    else:
        log.warning("[BOOT] No AL01_API_KEY set — API running in open/dev mode")
    log.info("[BOOT] Starting API server on port %d", API_PORT)
    _start_api_server(organism, api_key=api_key)

    # --- 10. Run loop (blocks) ---
    log.info("[BOOT] Entering run loop (interval=%ds). Ctrl+C to stop.", INTERVAL)
    snapshot_mgr.start()
    try:
        organism.run_loop(interval=INTERVAL, log_cycle=False)
    except KeyboardInterrupt:
        pass
    finally:
        # --- 11. Shutdown ---
        try:
            organism.record_growth_snapshot()
            organism.shutdown()
            memory.flush_memory()  # v3.21: flush buffered memory entries to disk
            memory.flush_firestore()
            memory.maybe_daily_backup()
        except (KeyboardInterrupt, Exception) as exc:
            log.warning("[SHUTDOWN] Cleanup interrupted: %s", exc)
        log.info("[SHUTDOWN] AL-01 v%s exited cleanly", VERSION)


if __name__ == "__main__":
    main()
