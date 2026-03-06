"""AL-01 v3.0 — Experimental Protocol System.

Provides reproducible experiment management with:
* Unique experiment ID
* Seeded randomness for full reproducibility
* Environment seed configuration
* 30-day autonomous run mode
* Start/stop/status tracking
* Experiment metadata and result snapshotting
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger("al01.experiment")


@dataclass
class ExperimentConfig:
    """Configuration for a reproducible experiment run."""

    experiment_id: str = ""
    """Unique experiment identifier.  Auto-generated if empty."""

    global_seed: int = 42
    """Master RNG seed — deterministic child seeds derived from this."""

    environment_seed: Optional[int] = None
    """Specific seed for the environment model.  Derived from global_seed if None."""

    genome_seed: Optional[int] = None
    """Specific seed for genome mutations.  Derived from global_seed if None."""

    duration_days: int = 30
    """How many days the autonomous run should last."""

    tick_interval_seconds: float = 5.0
    """Seconds between heartbeat ticks in autonomous mode."""

    # Population constraints
    max_population: int = 60
    """Maximum population size before pruning weakest."""

    min_population: int = 2
    """Minimum population (always keep at least this many alive)."""

    # Selection pressure
    survival_fitness_threshold: float = 0.2
    """Organisms below this fitness for too long die."""

    survival_grace_cycles: int = 20
    """How many cycles below threshold before death."""

    reproduction_fitness_threshold: float = 0.5
    """Minimum fitness to be allowed to reproduce."""

    reproduction_fitness_cycles: int = 5
    """Must maintain reproduction fitness for this many consecutive cycles."""

    reproduction_enabled: bool = True
    """Master toggle — if False, no reproduction occurs (for controlled experiments)."""

    # Death
    energy_death_threshold: float = 0.0
    """Energy at or below this → organism dies."""

    # Snapshot interval
    snapshot_interval_cycles: int = 100
    """Save a full experiment snapshot every N cycles."""

    # Metadata
    description: str = ""
    """Human-readable description of the experiment."""

    tags: List[str] = field(default_factory=list)


def generate_experiment_id(seed: int = 0) -> str:
    """Generate a unique experiment ID from seed + timestamp."""
    now = datetime.now(timezone.utc).isoformat()
    raw = f"{seed}-{now}-{random.random()}"
    return "EXP-" + hashlib.sha256(raw.encode()).hexdigest()[:12].upper()


class ExperimentProtocol:
    """Manages reproducible experiment lifecycle.

    Usage::

        proto = ExperimentProtocol(config)
        proto.start()
        # ... organism runs autonomously ...
        proto.snapshot(cycle, state)
        # ...
        proto.stop(reason="completed")
    """

    def __init__(
        self,
        config: Optional[ExperimentConfig] = None,
        data_dir: str = ".",
    ) -> None:
        self._config = config or ExperimentConfig()
        self._data_dir = data_dir

        # Auto-generate experiment ID if not set
        if not self._config.experiment_id:
            self._config.experiment_id = generate_experiment_id(self._config.global_seed)

        # Derive sub-seeds from global seed
        master_rng = random.Random(self._config.global_seed)
        if self._config.environment_seed is None:
            self._config.environment_seed = master_rng.randint(0, 2**31)
        if self._config.genome_seed is None:
            self._config.genome_seed = master_rng.randint(0, 2**31)

        # State
        self._active: bool = False
        self._start_time: Optional[str] = None
        self._end_time: Optional[str] = None
        self._stop_reason: Optional[str] = None
        self._total_cycles: int = 0
        self._snapshots: List[Dict[str, Any]] = []

        # Persistence
        self._experiment_dir = os.path.join(
            data_dir, "experiments", self._config.experiment_id
        )
        os.makedirs(self._experiment_dir, exist_ok=True)
        self._meta_path = os.path.join(self._experiment_dir, "metadata.json")
        self._log_path = os.path.join(self._experiment_dir, "experiment_log.jsonl")

        # Load existing state if resuming
        self._load_metadata()

    # ── Properties ───────────────────────────────────────────────────

    @property
    def config(self) -> ExperimentConfig:
        return self._config

    @property
    def experiment_id(self) -> str:
        return self._config.experiment_id

    @property
    def active(self) -> bool:
        return self._active

    @property
    def total_cycles(self) -> int:
        return self._total_cycles

    @property
    def environment_seed(self) -> int:
        return self._config.environment_seed  # type: ignore

    @property
    def genome_seed(self) -> int:
        return self._config.genome_seed  # type: ignore

    @property
    def global_seed(self) -> int:
        return self._config.global_seed

    @property
    def experiment_dir(self) -> str:
        return self._experiment_dir

    def should_die(self, energy: float) -> bool:
        """Check if an organism should die based on energy."""
        return energy <= self._config.energy_death_threshold

    def can_reproduce(self, fitness: float, consecutive_above: int) -> bool:
        """Check if an organism meets reproduction requirements."""
        return (
            fitness >= self._config.reproduction_fitness_threshold
            and consecutive_above >= self._config.reproduction_fitness_cycles
        )

    def should_prune(self, population_size: int) -> bool:
        """Check if population needs pruning."""
        return population_size > self._config.max_population

    def should_snapshot(self, cycle: int) -> bool:
        """Check if it's time for an experiment snapshot."""
        return cycle > 0 and cycle % self._config.snapshot_interval_cycles == 0

    def time_remaining(self) -> Optional[float]:
        """Seconds remaining in the experiment, or None if not active."""
        if not self._active or not self._start_time:
            return None
        try:
            started = datetime.fromisoformat(self._start_time)
            end = started + timedelta(days=self._config.duration_days)
            remaining = (end - datetime.now(timezone.utc)).total_seconds()
            return max(0.0, remaining)
        except Exception:
            return None

    def is_expired(self) -> bool:
        """Check if the experiment has exceeded its duration."""
        remaining = self.time_remaining()
        return remaining is not None and remaining <= 0.0

    # ── Lifecycle ────────────────────────────────────────────────────

    def start(self) -> Dict[str, Any]:
        """Start the experiment."""
        self._active = True
        self._start_time = _utc_now()
        self._save_metadata()
        self._append_log({
            "event": "experiment_start",
            "experiment_id": self._config.experiment_id,
            "global_seed": self._config.global_seed,
            "environment_seed": self._config.environment_seed,
            "genome_seed": self._config.genome_seed,
            "duration_days": self._config.duration_days,
            "timestamp": self._start_time,
        })
        logger.info(
            "[EXPERIMENT] Started %s (seed=%d, duration=%d days)",
            self._config.experiment_id,
            self._config.global_seed,
            self._config.duration_days,
        )
        return self.status()

    def stop(self, reason: str = "manual") -> Dict[str, Any]:
        """Stop the experiment."""
        self._active = False
        self._end_time = _utc_now()
        self._stop_reason = reason
        self._save_metadata()
        self._append_log({
            "event": "experiment_stop",
            "reason": reason,
            "total_cycles": self._total_cycles,
            "timestamp": self._end_time,
        })
        logger.info(
            "[EXPERIMENT] Stopped %s reason=%s cycles=%d",
            self._config.experiment_id, reason, self._total_cycles,
        )
        return self.status()

    def record_cycle(self, cycle: int) -> None:
        """Record that a cycle completed."""
        self._total_cycles = cycle
        # Auto-stop if expired
        if self.is_expired():
            self.stop(reason="duration_expired")

    def snapshot(self, cycle: int, state: Dict[str, Any]) -> None:
        """Save a full experiment snapshot."""
        snap = {
            "cycle": cycle,
            "timestamp": _utc_now(),
            "state": state,
        }
        self._snapshots.append(snap)

        # Write to file
        snap_path = os.path.join(
            self._experiment_dir,
            f"snapshot_{cycle:06d}.json",
        )
        try:
            with open(snap_path, "w", encoding="utf-8") as fh:
                json.dump(snap, fh, indent=2, default=str)
        except Exception as exc:
            logger.error("[EXPERIMENT] Snapshot write failed: %s", exc)

        self._append_log({
            "event": "snapshot",
            "cycle": cycle,
            "timestamp": _utc_now(),
        })

    def status(self) -> Dict[str, Any]:
        """Current experiment status."""
        return {
            "experiment_id": self._config.experiment_id,
            "active": self._active,
            "global_seed": self._config.global_seed,
            "environment_seed": self._config.environment_seed,
            "genome_seed": self._config.genome_seed,
            "duration_days": self._config.duration_days,
            "start_time": self._start_time,
            "end_time": self._end_time,
            "stop_reason": self._stop_reason,
            "total_cycles": self._total_cycles,
            "snapshots_taken": len(self._snapshots),
            "time_remaining_seconds": self.time_remaining(),
            "max_population": self._config.max_population,
            "survival_fitness_threshold": self._config.survival_fitness_threshold,
            "reproduction_fitness_threshold": self._config.reproduction_fitness_threshold,
            "energy_death_threshold": self._config.energy_death_threshold,
            "description": self._config.description,
            "tags": self._config.tags,
        }

    # ── Persistence ──────────────────────────────────────────────────

    def _save_metadata(self) -> None:
        meta = {
            "experiment_id": self._config.experiment_id,
            "global_seed": self._config.global_seed,
            "environment_seed": self._config.environment_seed,
            "genome_seed": self._config.genome_seed,
            "duration_days": self._config.duration_days,
            "max_population": self._config.max_population,
            "description": self._config.description,
            "tags": self._config.tags,
            "active": self._active,
            "start_time": self._start_time,
            "end_time": self._end_time,
            "stop_reason": self._stop_reason,
            "total_cycles": self._total_cycles,
        }
        try:
            with open(self._meta_path, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2)
        except Exception as exc:
            logger.error("[EXPERIMENT] Metadata write failed: %s", exc)

    def _load_metadata(self) -> None:
        if not os.path.exists(self._meta_path):
            return
        try:
            with open(self._meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            self._active = meta.get("active", False)
            self._start_time = meta.get("start_time")
            self._end_time = meta.get("end_time")
            self._stop_reason = meta.get("stop_reason")
            self._total_cycles = meta.get("total_cycles", 0)
            logger.info(
                "[EXPERIMENT] Resumed experiment %s (cycles=%d)",
                self._config.experiment_id, self._total_cycles,
            )
        except Exception as exc:
            logger.error("[EXPERIMENT] Metadata load failed: %s", exc)

    def _append_log(self, record: Dict[str, Any]) -> None:
        try:
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        except Exception as exc:
            logger.error("[EXPERIMENT] Log write failed: %s", exc)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
