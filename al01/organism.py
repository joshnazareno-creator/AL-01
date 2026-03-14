"""AL-01 v3.0 organism: bounded artificial ecosystem with lineage divergence.

v3.0 adds:
* True Environment Model (resource pool, scarcity, env variables)
* Evolution Tracker (generation ID, genome hash, mutation log, CSV export)
* Behavior Detection (strategy classification, convergence/divergence)
* Experiment Protocol (seeded reproducibility, 30-day autonomous mode)
* Selection Pressure (death at energy ≤ 0, population pruning, reproduction gates)
* Fully Autonomous Operation (no manual stimulus required)
"""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import math
import os
import random
import tempfile
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Dict, List, Optional

from al01.autonomy import (
    AutonomyConfig,
    AutonomyEngine,
    DECISION_ADAPT,
    DECISION_BLEND,
    DECISION_MUTATE,
)
from al01.behavior import PopulationBehaviorAnalyzer
from al01.brain import Brain
from al01.environment import Environment, EnvironmentConfig
from al01.evolution_tracker import EvolutionTracker, genome_hash
from al01.experiment import ExperimentConfig, ExperimentProtocol
from al01.genome import Genome
from al01.life_log import LifeLog
from al01.memory_manager import MemoryManager
from al01.policy import PolicyManager
from al01.population import LifecycleState, Population
from al01.genesis_vault import GenesisVault
from al01.gpt_bridge import GPTBridge, GPTBridgeConfig
from al01.snapshot_manager import SnapshotConfig, SnapshotManager
from al01.storage import rotate_jsonl, cleanup_tick_snapshots, check_disk_usage

VERSION = "3.9"

# ── Founder protection constants ──────────────────────────────────────
FOUNDER_MUTATION_CAP: float = 0.08      # AL-01 mutation rate never exceeds this
FOUNDER_ENERGY_FLOOR: float = 0.25      # AL-01 energy never drops below this
FOUNDER_FITNESS_FLOOR: float = 0.15     # AL-01 won't die unless below this

# ── v3.25: Founder revival survival fix ───────────────────────────────
FOUNDER_REVIVAL_ENERGY: float = 0.60           # energy restored on revival (was 0.5)
FOUNDER_REVIVAL_FITNESS_FLOOR: float = 0.20    # minimum fitness set on revival
FOUNDER_REVIVAL_GRACE_CYCLES: int = 30         # grace window: no forced mutation
FOUNDER_MAX_CONSECUTIVE_FORCED_MUTATE: int = 3 # cap forced mutations before cooldown
FOUNDER_MUTATE_COOLDOWN_CYCLES: int = 5        # stabilize cycles after hitting cap
FOUNDER_RECOVERY_SCARCITY_DAMPENING: float = 0.4  # scarcity multiplier softened to 40%

# ── v3.23: Stabilization constants ───────────────────────────────────
CONSERVATION_ENERGY_THRESHOLD: float = 0.10   # 10 % of max — enter conservation
ADAPTABILITY_RECOVERY_THRESHOLD: float = 0.20 # nudge adaptability when below this
ADAPTABILITY_RECOVERY_NUDGE: float = 0.02     # per-cycle upward nudge
ENERGY_EFFICIENCY_METABOLIC_SCALE: float = 0.30  # max 30 % cost reduction at trait=1.0

# ── v3.10: Rare reproduction constants ────────────────────────────────
RARE_REPRO_CYCLE_INTERVAL: int = 500     # evaluate every N cycles
RARE_REPRO_CHANCE: float = 0.05          # 5 % random gate
RARE_REPRO_POP_CAP: int = 50            # hard population cap for this path
RARE_REPRO_ENERGY_MIN: float = 0.65     # parent must have >= this energy
RARE_REPRO_FITNESS_MIN: float = 0.50    # parent must have >= this fitness
RARE_REPRO_STAGNATION_MAX: float = 0.90 # parent stagnation must be < this
RARE_REPRO_ENERGY_COST: float = 0.30    # deducted from parent on birth
RARE_REPRO_CHILD_ENERGY: float = 0.50   # child starts with this energy
RARE_REPRO_MUTATION_RATE: float = 0.10  # mutation variance for child genome
RARE_REPRO_COOLDOWN_CYCLES: int = 2000  # min cycles between births per parent

# ── v3.29: Restart recovery constants ─────────────────────────────────
RESTART_RECOVERY_CYCLES: int = 50       # cycles after boot before full selection

# ── v3.31: Extinction wave prevention constants ───────────────────────
MAX_FITNESS_DEATHS_PER_CYCLE: int = 3   # cap fitness-floor kills per child cycle
TRAIT_COLLAPSE_EMERGENCY_CYCLES: int = 20  # cycles of reduced death pressure
TRAIT_COLLAPSE_DEATH_CAP: int = 1       # max deaths per cycle during emergency
TRAIT_COLLAPSE_MUTATION_BOOST: float = 0.10  # extra mutation rate during emergency
TRAIT_COLLAPSE_VARIANCE_FLOOR: float = 0.001  # threshold for declaring collapse

# ── v3.32: Monoculture recovery constants ─────────────────────────────
STRATEGY_CONVERGENCE_THRESHOLD: float = 0.95  # same-strategy fraction → emergency
MONOCULTURE_PENALTY_FLOOR_CRITICAL: float = 0.50  # penalty floor when pop critical / emergency

# ── v3.12: Novelty & stagnation detection constants ─────────────────
NOVELTY_HISTORY_MAX: int = 500          # rolling window of novelty scores
NOVELTY_STAGNATION_WINDOW: int = 100    # last N births to average
NOVELTY_STAGNATION_THRESHOLD: float = 0.05  # avg novelty below this = stagnating
NOVELTY_STAGNATION_COOLDOWN: int = 1000    # cycles between stagnation interventions


# ── v3.9: Internal Communication Signal ───────────────────────────────
@dataclass
class InternalSignal:
    """Communication stub — a broadcast-ready signal representing the
    organism's internal state.  Other organisms (or future modules) can
    read this to coordinate behaviour.

    Fields:
        energy_state:   0–1 normalised energy (0 = depleted, 1 = full)
        stress_level:   0–1 composite stress (high = under pressure)
        novelty_drive:  0–1 desire for exploration (high = seeking novelty)
    """
    energy_state: float = 1.0
    stress_level: float = 0.0
    novelty_drive: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "energy_state": round(self.energy_state, 6),
            "stress_level": round(self.stress_level, 6),
            "novelty_drive": round(self.novelty_drive, 6),
        }


@dataclass
class CycleStats:
    """Per-cycle instrumentation counters (rolling window)."""

    window_size: int = 20

    # Rolling energy deltas for volatility (stddev)
    _energy_deltas: List[float] = field(default_factory=list)
    # Rolling floor-hit count (energy hit energy_min)
    _floor_hits: List[bool] = field(default_factory=list)
    # Per-organism alive streaks (consecutive cycles alive)
    _alive_streaks: Dict[str, int] = field(default_factory=dict)
    # Per-cycle energy spent vs value gained
    _energy_spent: List[float] = field(default_factory=list)
    _fitness_gained: List[float] = field(default_factory=list)

    def record_energy_delta(self, delta: float, hit_floor: bool) -> None:
        self._energy_deltas.append(delta)
        self._floor_hits.append(hit_floor)
        if len(self._energy_deltas) > self.window_size:
            self._energy_deltas.pop(0)
            self._floor_hits.pop(0)

    def record_efficiency(self, energy_cost: float, fitness_delta: float) -> None:
        self._energy_spent.append(energy_cost)
        self._fitness_gained.append(fitness_delta)
        if len(self._energy_spent) > self.window_size:
            self._energy_spent.pop(0)
            self._fitness_gained.pop(0)

    def tick_alive(self, organism_id: str) -> None:
        self._alive_streaks[organism_id] = self._alive_streaks.get(organism_id, 0) + 1

    def record_death(self, organism_id: str) -> None:
        self._alive_streaks.pop(organism_id, None)

    @property
    def energy_volatility(self) -> float:
        """Stddev of recent energy deltas."""
        if len(self._energy_deltas) < 2:
            return 0.0
        mean = sum(self._energy_deltas) / len(self._energy_deltas)
        variance = sum((d - mean) ** 2 for d in self._energy_deltas) / len(self._energy_deltas)
        return variance ** 0.5

    @property
    def floor_hits_rolling(self) -> int:
        return sum(1 for h in self._floor_hits if h)

    @property
    def efficiency_ratio(self) -> float:
        """Fitness gained per unit energy spent (0 if no spend)."""
        total_spent = sum(self._energy_spent)
        if total_spent <= 0:
            return 0.0
        return sum(self._fitness_gained) / total_spent

    def longest_alive_streak(self) -> int:
        if not self._alive_streaks:
            return 0
        return max(self._alive_streaks.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "energy_volatility": round(self.energy_volatility, 6),
            "floor_hits_rolling": self.floor_hits_rolling,
            "efficiency_ratio": round(self.efficiency_ratio, 6),
            "longest_alive_streak": self.longest_alive_streak(),
            "window_size": self.window_size,
        }


# ── v3.6: Structured Per-Cycle Logging ──────────────────────────────────

@dataclass
class CycleLogEntry:
    """One structured log entry per autonomy cycle (v3.6)."""
    cycle: int
    decision: str
    energy_before: float
    energy_after: float
    energy_delta: float
    fitness_before: float
    fitness_after: float
    fitness_delta: float
    mutation_applied: Optional[Dict[str, Any]] = None
    recovery_mode: bool = False
    stagnation_count: int = 0
    alerts: List[str] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "cycle": self.cycle,
            "decision": self.decision,
            "energy_before": round(self.energy_before, 6),
            "energy_after": round(self.energy_after, 6),
            "energy_delta": round(self.energy_delta, 6),
            "fitness_before": round(self.fitness_before, 6),
            "fitness_after": round(self.fitness_after, 6),
            "fitness_delta": round(self.fitness_delta, 6),
            "recovery_mode": self.recovery_mode,
            "stagnation_count": self.stagnation_count,
        }
        if self.mutation_applied:
            d["mutation_applied"] = self.mutation_applied
        if self.alerts:
            d["alerts"] = self.alerts
        if self.timestamp:
            d["timestamp"] = self.timestamp
        return d


class AlertGuardrails:
    """v3.6: Monitors for sustained bad states and emits alerts.

    Alerts:
    - ENERGY_CRITICAL: energy < 15% for > N cycles
    - FITNESS_STALLED: zero fitness improvement for > N cycles
    - TRAIT_COLLAPSED: trait variance near zero (all traits same)
    """

    def __init__(
        self,
        energy_critical_threshold: float = 0.15,
        energy_critical_cycles: int = 15,
        fitness_stall_cycles: int = 30,
        fitness_stall_epsilon: float = 0.001,
        trait_variance_floor: float = 0.001,
    ) -> None:
        self._energy_critical_threshold = energy_critical_threshold
        self._energy_critical_cycles = energy_critical_cycles
        self._fitness_stall_cycles = fitness_stall_cycles
        self._fitness_stall_epsilon = fitness_stall_epsilon
        self._trait_variance_floor = trait_variance_floor

        self._consecutive_low_energy: int = 0
        self._fitness_history: List[float] = []

    def check(
        self,
        energy: float,
        fitness: float,
        traits: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """Return list of alert strings (empty if healthy)."""
        alerts: List[str] = []

        # Energy critical
        if energy < self._energy_critical_threshold:
            self._consecutive_low_energy += 1
        else:
            self._consecutive_low_energy = 0
        if self._consecutive_low_energy > self._energy_critical_cycles:
            alerts.append(
                f"ENERGY_CRITICAL: energy {energy:.4f} < "
                f"{self._energy_critical_threshold} for "
                f"{self._consecutive_low_energy} cycles"
            )

        # Fitness stalled
        self._fitness_history.append(fitness)
        if len(self._fitness_history) > self._fitness_stall_cycles:
            self._fitness_history = self._fitness_history[
                -self._fitness_stall_cycles:
            ]
        if len(self._fitness_history) >= self._fitness_stall_cycles:
            mn = min(self._fitness_history)
            mx = max(self._fitness_history)
            if (mx - mn) < self._fitness_stall_epsilon:
                alerts.append(
                    f"FITNESS_STALLED: range {mx - mn:.6f} < "
                    f"{self._fitness_stall_epsilon} over "
                    f"{self._fitness_stall_cycles} cycles"
                )

        # Trait variance collapsed
        if traits:
            vals = list(traits.values())
            if len(vals) >= 2:
                mean_v = sum(vals) / len(vals)
                var_v = sum((v - mean_v) ** 2 for v in vals) / len(vals)
                if var_v < self._trait_variance_floor:
                    alerts.append(
                        f"TRAIT_COLLAPSED: variance {var_v:.6f} < "
                        f"{self._trait_variance_floor}"
                    )

        return alerts


class CycleLogger:
    """v3.6: Writes structured per-cycle JSON log entries."""

    def __init__(self, log_path: str, max_memory: int = 100) -> None:
        self._log_path = log_path
        self._max_memory = max_memory
        self._entries: List[CycleLogEntry] = []
        self._logger = logging.getLogger("al01.cycle_log")

    def record(self, entry: CycleLogEntry) -> None:
        self._entries.append(entry)
        if len(self._entries) > self._max_memory:
            self._entries.pop(0)
        # Write to disk (rotate when oversized)
        try:
            rotate_jsonl(self._log_path)
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")
        except Exception as exc:
            self._logger.error("[CYCLE_LOG] Failed to write: %s", exc)
        # Log alerts at WARNING level
        for alert in entry.alerts:
            self._logger.warning("[ALERT] %s", alert)

    @property
    def recent(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._entries[-20:]]

    @property
    def entries(self) -> List[CycleLogEntry]:
        return list(self._entries)


class OrganismState(enum.Enum):
    """Lifecycle states for the organism."""
    IDLE = "idle"
    LEARNING = "learning"
    REFLECTING = "reflecting"
    RESPONDING = "responding"


@dataclass
class MetabolismConfig:
    pulse_interval: int = 1
    reflect_interval: int = 10
    persist_interval: int = 5
    pulse_log_interval: int = 5
    heartbeat_persist_interval: int = 1  # persist every N heartbeat cycles
    evolve_interval: int = 30  # check stimuli / evolve every N ticks
    population_interact_interval: int = 60  # inter-organism interaction every N ticks
    autonomy_interval: int = 10  # autonomous decision cycle every N ticks
    environment_interval: int = 5  # environment tick every N ticks
    behavior_analysis_interval: int = 20  # behavior snapshot every N ticks
    auto_reproduce_interval: int = 15  # check autonomous reproduction every N ticks
    child_autonomy_interval: int = 10  # child evolution cycle every N ticks
    rare_reproduce_interval: int = 50  # rare reproduction gate every N ticks
    memory_snapshot_interval: int = 100  # v3.21: save memory snapshot every N ticks
    disk_check_interval: int = 500  # v3.22: check disk usage every N ticks


class MetabolismScheduler:
    def __init__(self, organism: "Organism", config: MetabolismConfig) -> None:
        self._organism = organism
        self._config = config
        self._tick_count: int = 0
        self._pulse_count: int = 0

    def tick(self) -> None:
        self._tick_count += 1

        if self._tick_count % self._config.pulse_interval == 0:
            self._pulse_count += 1
            log_this_pulse = (self._pulse_count % self._config.pulse_log_interval == 0)
            self._organism.pulse(log_observation=log_this_pulse)

        if self._tick_count % self._config.reflect_interval == 0:
            self._organism.reflect()

        if self._tick_count % self._config.persist_interval == 0:
            self._organism.persist()

        # Environment tick — advance env variables, resource pool, scarcity
        if self._tick_count % self._config.environment_interval == 0:
            self._organism.environment_tick()

        # Evolution cycle — process stimuli and mutate genome
        if self._tick_count % self._config.evolve_interval == 0:
            self._organism.evolve_cycle()

        # Population interactions
        if self._tick_count % self._config.population_interact_interval == 0:
            self._organism.population_interact()

        # Autonomous decision cycle
        if self._tick_count % self._config.autonomy_interval == 0:
            self._organism.autonomy_cycle()

        # ── v3.17: Death resolution BEFORE reproduction ──────────────
        # Child autonomy — evolve all children independently
        # (includes energy depletion, fitness floor death checks)
        if self._tick_count % self._config.child_autonomy_interval == 0:
            self._organism.child_autonomy_cycle()

        # Sync tick counter into population for birth-lock / idempotency
        self._organism._population.set_tick(self._tick_count)

        # ── Reproduction passes (only after death resolution) ────────
        # Autonomous reproduction check
        if self._tick_count % self._config.auto_reproduce_interval == 0:
            self._organism.auto_reproduce_cycle()

        # v3.16: Wake dormant organisms when conditions improve
        if self._tick_count % self._config.auto_reproduce_interval == 0:
            self._organism.wake_dormant_cycle()

        # v3.16: Stability reproduction — spawn when resources are abundant
        if self._tick_count % self._config.auto_reproduce_interval == 0:
            self._organism.stability_reproduction_cycle()

        # v3.16: Lone survivor reproduction — prevent total extinction
        if self._tick_count % self._config.auto_reproduce_interval == 0:
            self._organism.lone_survivor_reproduction()

        # Behavior analysis snapshot
        if self._tick_count % self._config.behavior_analysis_interval == 0:
            self._organism.behavior_analysis_cycle()

        # v3.10: Rare reproduction — heavily gated, checked infrequently
        if self._tick_count % self._config.rare_reproduce_interval == 0:
            self._organism.rare_reproduction_cycle()

        # v3.12: Novelty stagnation detection — checked with rare reproduction
        if self._tick_count % self._config.rare_reproduce_interval == 0:
            self._organism.novelty_stagnation_check()

        # v3.21: Periodic memory snapshot — saves lightweight tick snapshot
        if self._tick_count % self._config.memory_snapshot_interval == 0:
            self._organism.save_tick_snapshot(self._tick_count)

        # v3.22: Periodic disk usage check
        if self._tick_count % self._config.disk_check_interval == 0:
            self._organism.check_disk_usage()


class Organism:
    def __init__(
        self,
        data_dir: str = ".",
        config: Optional[MetabolismConfig] = None,
        memory_manager: Optional[MemoryManager] = None,
        life_log: Optional[LifeLog] = None,
        policy: Optional[PolicyManager] = None,
        population: Optional[Population] = None,
        brain: Optional[Brain] = None,
        autonomy: Optional[AutonomyEngine] = None,
        environment: Optional[Environment] = None,
        evolution_tracker: Optional[EvolutionTracker] = None,
        behavior_analyzer: Optional[PopulationBehaviorAnalyzer] = None,
        experiment: Optional[ExperimentProtocol] = None,
        snapshot_manager: Optional[SnapshotManager] = None,
        genesis_vault: Optional[GenesisVault] = None,
    ) -> None:
        self._data_dir = data_dir
        self.config: MetabolismConfig = config or MetabolismConfig()
        self._memory_manager = memory_manager or MemoryManager(data_dir=data_dir)
        self._logger = logging.getLogger("al01.organism")
        self._state_lock = threading.RLock()
        self._tick_lock = threading.Lock()  # v3.17: Concurrency guard — only one tick at a time
        self._loop_stop_event = threading.Event()
        self._loop_thread: Optional[threading.Thread] = None
        self._organism_state: OrganismState = OrganismState.IDLE
        self._boot_time: float = time.monotonic()
        self._boot_utc: str = _utc_now()

        # Dirty-tracking for persist — skip writes when nothing meaningful changed
        self._last_persisted_fingerprint: str = ""

        # VITAL subsystems
        vital_dir = os.path.join(data_dir, "data")
        self._life_log = life_log or LifeLog(data_dir=vital_dir, organism_id="AL-01")
        self._policy = policy or PolicyManager(data_dir=vital_dir)

        # v2.0: Genome, Population, Brain
        self._population = population or Population(data_dir=data_dir, parent_id="AL-01")
        self._brain = brain or Brain()
        self._autonomy = autonomy or AutonomyEngine(data_dir=os.path.join(data_dir, "data"))

        # v3.0: Environment, Evolution Tracker, Behavior Analyzer, Experiment
        self._environment = environment or Environment()
        self._evolution_tracker = evolution_tracker or EvolutionTracker(
            data_dir=os.path.join(data_dir, "data")
        )
        self._behavior_analyzer = behavior_analyzer or PopulationBehaviorAnalyzer()
        self._experiment = experiment  # None = no active experiment

        # v3.1: Snapshot manager (hourly auto-snapshots + retention + remote sync)
        self._snapshot_manager = snapshot_manager  # wired externally in __main__

        # v3.2: Genesis Vault (extinction recovery from frozen seed)
        self._genesis_vault = genesis_vault or GenesisVault(
            data_dir=os.path.join(data_dir, "data")
        )

        # Global tick counter for cycle tracking
        self._global_cycle: int = 0

        # v3.3: Per-organism fitness floor tracking for survival_grace_cycles
        self._below_fitness_cycles: Dict[str, int] = {}

        # v3.10: Per-organism cooldown for rare reproduction
        self._last_birth_cycle: Dict[str, int] = {}

        # v3.12: Novelty tracking — rolling window of per-birth novelty scores
        self._novelty_history: List[float] = []
        self._last_novelty_intervention_cycle: int = 0

        # v3.23: Conservation mode tracking — organisms at minimum energy
        self._conservation_mode: Dict[str, bool] = {}

        # v3.25: Founder revival survival tracking
        self._founder_revival_count: int = 0
        self._founder_revival_grace_remaining: int = 0
        self._founder_forced_mutate_streak: int = 0
        self._founder_mutate_cooldown: int = 0
        self._founder_recovery_mode: bool = False
        self._founder_cycles_since_revival: int = 0

        # v3.29: Restart recovery — protect organisms from artificial extinction on boot
        self._restart_recovery_remaining: int = 0

        # v3.31: Trait collapse emergency — reduce death pressure during convergence
        self._trait_collapse_emergency_remaining: int = 0

        # v3.4: Default population cap (overridden by ExperimentConfig when active)
        self._default_max_population: int = 60

        # v3.5: Per-cycle instrumentation
        self._cycle_stats = CycleStats()

        # v3.6: Structured per-cycle logging & alert guardrails
        cycle_log_path = os.path.join(data_dir, "data", "cycle_log.jsonl")
        self._cycle_logger = CycleLogger(cycle_log_path)
        self._alert_guardrails = AlertGuardrails()

        # Push cap into population so spawn_child can enforce it too
        self._population.max_population = self._default_max_population

        # v3.9: Memory drift interval (cycles) and last-applied tracker
        self._memory_drift_interval: int = 100
        self._last_memory_drift_cycle: int = 0

        loaded_state = self._memory_manager.load_state()
        had_prior_state = bool(loaded_state)
        self._state: Dict[str, Any] = self._normalize_state(loaded_state)

        # Restore or initialize genome
        genome_data = self._state.get("genome")
        if genome_data and isinstance(genome_data, dict):
            self._genome = Genome.from_dict(genome_data)
        else:
            self._genome = Genome()
        self._state["genome"] = self._genome.to_dict()

        # Register parent in evolution tracker if not already tracked
        if not self._evolution_tracker.get_lineage("AL-01"):
            self._evolution_tracker.register_organism(
                "AL-01", parent_id=None, traits=self._genome.traits, cycle=0,
            )

        # Stimuli queue (list of pending stimulus strings)
        if not isinstance(self._state.get("stimuli"), list):
            self._state["stimuli"] = []

        # Track last reproduction evolution_count to avoid double-spawning
        self._last_reproduce_at: int = int(self._state.get("last_reproduce_at", 0))

        # Temporal awareness — birth_time persists across restarts
        if not self._state.get("birth_time"):
            self._state["birth_time"] = self._boot_utc
        self._state["uptime_sessions"] = int(self._state.get("uptime_sessions", 0)) + 1
        self._state["last_tick_time"] = None

        # Recover interaction count from SQLite if it exceeds state value
        db_count = self._memory_manager.database.interaction_count()
        if db_count > int(self._state.get("interaction_count", 0)):
            self._state["interaction_count"] = db_count
            self._logger.info("[BOOT] Recovered %d interactions from database", db_count)

        if had_prior_state:
            self._logger.info(
                "[BOOT] Resumed from persisted state: evolution_count=%d awareness=%.4f state_version=%d interactions=%d",
                self._state["evolution_count"],
                self._state["awareness"],
                self._state["state_version"],
                self._state["interaction_count"],
            )
        else:
            self._logger.info("[BOOT] No prior state found — initializing defaults")

        self._memory_manager.save_state(self._serialize_state(self._state))

        self.scheduler = MetabolismScheduler(self, self.config)

    def tick(self) -> None:
        # v3.17: Concurrency guard — prevent overlapping tick execution
        if not self._tick_lock.acquire(blocking=False):
            self._logger.debug("[TICK] Skipped — another tick is already running")
            return
        try:
            self.scheduler.tick()
        finally:
            self._tick_lock.release()

    @property
    def state(self) -> Dict[str, Any]:
        with self._state_lock:
            return dict(deepcopy(self._state))

    @property
    def organism_state(self) -> str:
        """Current lifecycle state (idle / learning / reflecting / responding)."""
        return self._organism_state.value

    @property
    def statre(self) -> Dict[str, Any]:
        return self.state

    @property
    def life_log(self) -> LifeLog:
        return self._life_log

    @property
    def policy(self) -> PolicyManager:
        return self._policy

    @property
    def age_seconds(self) -> float:
        """Seconds since birth (first-ever boot)."""
        birth = self._state.get("birth_time")
        if not birth:
            return self.uptime_seconds
        try:
            born = datetime.fromisoformat(str(birth))
            now = datetime.now(timezone.utc)
            return (now - born).total_seconds()
        except Exception:
            return self.uptime_seconds

    def update_policy(self, changes: Dict[str, float], reason: str = "") -> Dict[str, Any]:
        """Update policy weights and log the change as a life-log event."""
        record = self._policy.update(changes, reason=reason)
        self._life_log.append_event(
            event_type="policy_change",
            payload=record,
        )
        return record

    # ------------------------------------------------------------------
    # v2.0: Genome, Stimuli, Evolution, Population, Brain
    # ------------------------------------------------------------------

    @property
    def genome(self) -> Genome:
        return self._genome

    @property
    def population(self) -> Population:
        return self._population

    @property
    def brain(self) -> Brain:
        return self._brain

    @property
    def autonomy(self) -> AutonomyEngine:
        return self._autonomy

    @property
    def environment(self) -> Environment:
        return self._environment

    @property
    def evolution_tracker(self) -> EvolutionTracker:
        return self._evolution_tracker

    @property
    def behavior_analyzer(self) -> PopulationBehaviorAnalyzer:
        return self._behavior_analyzer

    @property
    def experiment(self) -> Optional[ExperimentProtocol]:
        return self._experiment

    @property
    def snapshot_manager(self) -> Optional[SnapshotManager]:
        return self._snapshot_manager

    @property
    def genesis_vault(self) -> GenesisVault:
        return self._genesis_vault

    @property
    def gpt_bridge(self) -> GPTBridge:
        """Lazy-initialized GPT bridge (no overhead if unused)."""
        if not hasattr(self, "_gpt_bridge"):
            self._gpt_bridge = GPTBridge(self)
        return self._gpt_bridge

    @property
    def global_cycle(self) -> int:
        return self._global_cycle

    @property
    def cycle_stats(self) -> CycleStats:
        """Per-cycle rolling instrumentation counters."""
        return self._cycle_stats

    @property
    def cycle_logger(self) -> CycleLogger:
        """v3.6: Structured per-cycle log accessor."""
        return self._cycle_logger

    @property
    def alert_guardrails(self) -> AlertGuardrails:
        """v3.6: Alert guardrails accessor."""
        return self._alert_guardrails

    @property
    def is_restart_recovery(self) -> bool:
        """v3.29: True while in post-restart recovery window."""
        return self._restart_recovery_remaining > 0

    @property
    def restart_recovery_remaining(self) -> int:
        """v3.29: Cycles remaining in restart recovery window."""
        return self._restart_recovery_remaining

    @property
    def internal_signal(self) -> InternalSignal:
        """v3.9: Compute the current internal communication signal.

        * energy_state — directly from autonomy energy (0–1)
        * stress_level — composite of low energy, high stagnation, recovery mode
        * novelty_drive — based on stagnation depth and exploration mode
        """
        energy = self._autonomy.energy
        stag = self._autonomy.stagnation_count
        in_recovery = getattr(self._autonomy, '_recovery_mode', False)
        exploration = getattr(self._autonomy, '_exploration_mode', False)

        # Stress: higher when energy low, stagnation high, or in recovery
        energy_stress = max(0.0, 1.0 - energy * 2.0)         # 0 at energy ≥ 0.5
        stag_stress = min(1.0, stag / 200.0)                  # 1.0 at 200+ stagnation
        recovery_stress = 0.3 if in_recovery else 0.0
        stress = min(1.0, energy_stress + stag_stress + recovery_stress)

        # Novelty drive: increases with stagnation, maxes in exploration mode
        novelty = min(1.0, stag / 100.0)
        if exploration:
            novelty = min(1.0, novelty + 0.4)

        return InternalSignal(
            energy_state=max(0.0, min(1.0, energy)),
            stress_level=round(stress, 6),
            novelty_drive=round(novelty, 6),
        )

    # ------------------------------------------------------------------
    # Ecosystem Health Metric (v3.11)
    # ------------------------------------------------------------------

    def ecosystem_health(self) -> Dict[str, Any]:
        """Compute a single composite ecosystem health score.

        Formula::

            score = avg_fitness * 0.4
                  + population_diversity * 0.4
                  + resource_pool_health * 0.2

        Returns detailed breakdown alongside the composite score.
        """
        # Average fitness across living organisms
        fitness_map = self._population.population_fitness()
        if fitness_map:
            avg_fitness = sum(fitness_map.values()) / len(fitness_map)
        else:
            avg_fitness = 0.0

        # Population diversity (normalised genome entropy)
        diversity = self._population.diversity_metrics()
        genome_entropy = diversity.get("genome_entropy", 0.0)
        unique_hashes = diversity.get("unique_genome_hashes", 0)
        pop_size = self._population.size
        # Normalise entropy: max possible = log2(pop_size) when all unique
        max_entropy = math.log2(pop_size) if pop_size >= 2 else 1.0
        diversity_score = min(1.0, genome_entropy / max_entropy) if max_entropy > 0 else 0.0

        # Resource pool health
        pool_health = self._environment.pool_fraction

        # Composite score
        score = (avg_fitness * 0.4
                 + diversity_score * 0.4
                 + pool_health * 0.2)

        return {
            "score": round(score, 4),
            "avg_fitness": round(avg_fitness, 4),
            "diversity_score": round(diversity_score, 4),
            "pool_health": round(pool_health, 4),
            "genome_entropy": round(genome_entropy, 4),
            "unique_genomes": unique_hashes,
            "population_size": pop_size,
            "cycle": self._global_cycle,
        }

    # ------------------------------------------------------------------
    # Birth Event Data (v3.11) — for visual layer
    # ------------------------------------------------------------------

    def last_birth_events(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the last *n* reproduction events from the life log.

        This feeds the visual birth-event animation system with data about
        recent births: parent, child, cycle, and type of reproduction.
        """
        log_path = os.path.join(self._data_dir, "data", "life_log.jsonl")
        if not os.path.exists(log_path):
            return []
        repro_types = {
            "auto_reproduce", "stability_reproduction",
            "lone_survivor_reproduction", "rare_reproduction",
        }
        events: List[Dict[str, Any]] = []
        try:
            with open(log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get("event_type") in repro_types:
                        payload = entry.get("payload", {})
                        events.append({
                            "event_type": entry["event_type"],
                            "parent_id": payload.get("parent_id"),
                            "child_id": payload.get("child_id"),
                            "cycle": payload.get("cycle", 0),
                            "timestamp": entry.get("timestamp"),
                        })
        except Exception:
            pass
        return events[-n:]

    # ------------------------------------------------------------------
    # Novelty Metric (v3.12) — genome distance at birth
    # ------------------------------------------------------------------

    def _record_novelty(self, parent_genome: Genome, child_genome: Genome) -> float:
        """Compute and record the novelty score for a birth event.

        Novelty = Euclidean genome distance between parent and child.
        Appended to ``_novelty_history`` (capped at NOVELTY_HISTORY_MAX).
        Returns the novelty score.
        """
        from al01.genome import genome_distance
        novelty = genome_distance(parent_genome, child_genome)
        self._novelty_history.append(round(novelty, 6))
        if len(self._novelty_history) > NOVELTY_HISTORY_MAX:
            self._novelty_history = self._novelty_history[-NOVELTY_HISTORY_MAX:]
        return round(novelty, 6)

    @property
    def novelty_history(self) -> List[float]:
        """Rolling window of per-birth novelty scores."""
        return list(self._novelty_history)

    @property
    def avg_novelty(self) -> float:
        """Average novelty over the last NOVELTY_STAGNATION_WINDOW births."""
        window = self._novelty_history[-NOVELTY_STAGNATION_WINDOW:]
        if not window:
            return 0.0
        return round(sum(window) / len(window), 6)

    @property
    def novelty_rate(self) -> float:
        """Alias for avg_novelty — the current novelty rate."""
        return self.avg_novelty

    # ------------------------------------------------------------------
    # Diversity Score (v3.12)
    # ------------------------------------------------------------------

    def population_diversity_index(self) -> float:
        """Average pairwise genome distance across all living organisms.

        High value = healthy diversity.  Low value = monoculture risk.
        """
        return self._population.population_diversity()

    # ------------------------------------------------------------------
    # Evolution Stagnation Detection (v3.12)
    # ------------------------------------------------------------------

    def novelty_stagnation_check(self) -> Optional[Dict[str, Any]]:
        """Detect if evolution has stagnated based on novelty scores.

        If avg novelty over the last NOVELTY_STAGNATION_WINDOW births is
        below NOVELTY_STAGNATION_THRESHOLD, trigger countermeasures:
        1. Mutation storm via environment named event
        2. Log a stagnation alert
        3. Enter exploration mode for the founder

        Returns a record of the intervention, or None if not stagnating.
        """
        if len(self._novelty_history) < 10:
            return None  # not enough data yet

        avg = self.avg_novelty
        if avg >= NOVELTY_STAGNATION_THRESHOLD:
            return None  # healthy innovation

        # Cooldown — don't intervene too frequently
        if (self._global_cycle - self._last_novelty_intervention_cycle
                < NOVELTY_STAGNATION_COOLDOWN):
            return None

        self._last_novelty_intervention_cycle = self._global_cycle

        actions_taken: List[str] = []

        # 1. Trigger a mutation storm in the environment
        from al01.environment import NamedEvent, NAMED_EVENT_CATALOGUE
        storm_def = next(
            (e for e in NAMED_EVENT_CATALOGUE if e["name"] == "mutation_storm"),
            None,
        )
        if storm_def:
            event = NamedEvent(
                name=storm_def["name"],
                description="Stagnation response: forced mutation storm",
                effects=dict(storm_def["effects"]),
                remaining_cycles=20,
                started_at=_utc_now(),
            )
            self._environment._named_events.append(event)
            actions_taken.append("mutation_storm")

        # 2. Boost mutation rates for all organisms temporarily
        for oid in self._population.member_ids:
            m = self._population.get(oid)
            if m:
                current_offset = m.get("mutation_rate_offset", 0.0)
                self._population.update_member(oid, {
                    "mutation_rate_offset": round(current_offset + 0.05, 6),
                })
        actions_taken.append("mutation_rate_boost")

        # 3. Enter exploration mode for the founder
        self._autonomy.set_exploration_mode(True)
        actions_taken.append("exploration_mode")

        record = {
            "event": "novelty_stagnation_intervention",
            "cycle": self._global_cycle,
            "avg_novelty": avg,
            "threshold": NOVELTY_STAGNATION_THRESHOLD,
            "window_size": min(len(self._novelty_history), NOVELTY_STAGNATION_WINDOW),
            "actions": actions_taken,
            "timestamp": _utc_now(),
        }

        self._life_log.append_event(
            event_type="novelty_stagnation",
            payload=record,
        )
        self._logger.warning(
            "[STAGNATION] Novelty stagnation detected: avg=%.4f < %.4f — actions: %s",
            avg, NOVELTY_STAGNATION_THRESHOLD, ", ".join(actions_taken),
        )
        return record

    # ------------------------------------------------------------------
    # Evolution Dashboard Data (v3.12)
    # ------------------------------------------------------------------

    def evolution_dashboard(self) -> Dict[str, Any]:
        """Return a concise evolution dashboard snapshot.

        Includes: population, species count, novelty rate, diversity index,
        ecosystem health score, and stagnation status.
        """
        census = self._population.species_census()
        health = self.ecosystem_health()
        diversity = self._population.population_diversity()

        return {
            "population": self._population.size,
            "species": len(census),
            "species_breakdown": {k: len(v) for k, v in census.items()},
            "novelty_rate": self.novelty_rate,
            "diversity_index": round(diversity, 4),
            "ecosystem_health": health["score"],
            "avg_fitness": health["avg_fitness"],
            "stagnating": (self.avg_novelty < NOVELTY_STAGNATION_THRESHOLD
                           if len(self._novelty_history) >= 10 else False),
            "total_births_tracked": len(self._novelty_history),
            "cycle": self._global_cycle,
            "timestamp": _utc_now(),
        }

    @property
    def stimuli(self) -> List[str]:
        with self._state_lock:
            return list(self._state.get("stimuli", []))

    def add_stimulus(self, stimulus: str) -> None:
        """Append a stimulus to the queue.  If it contains 'query:', route to brain."""
        with self._state_lock:
            if not isinstance(self._state.get("stimuli"), list):
                self._state["stimuli"] = []
            self._state["stimuli"].append(stimulus)

        self._life_log.append_event(
            event_type="stimulus_received",
            payload={"stimulus": stimulus},
        )
        self._logger.info("[STIMULUS] Queued: %s", stimulus[:80])
        # Record for awareness model — hash for novelty tracking
        stim_hash = hashlib.md5(stimulus.encode()).hexdigest()
        self._autonomy.record_stimulus(stim_hash)

        # Brain hook: if stimulus starts with "query:" route to brain
        if stimulus.lower().startswith("query:"):
            query_text = stimulus[6:].strip()
            if query_text:
                env_snap = self._environment.state_snapshot()
                traits = self._genome.to_dict().get("traits", {})
                energy = float(self._state.get("energy", 1.0))
                result = self._brain.process_query(
                    query_text,
                    env_state=env_snap,
                    organism_traits=traits,
                    organism_energy=energy,
                    organism_fitness=self._genome.fitness,
                )
                self._apply_brain_nudges(result)

    def _apply_brain_nudges(self, brain_result: Dict[str, Any]) -> None:
        """Apply AI-recommended trait nudges to the genome."""
        nudges = brain_result.get("trait_nudges", {})
        for trait, delta in nudges.items():
            old = self._genome.get_trait(trait)
            self._genome.set_trait(trait, old + delta)
        if nudges:
            with self._state_lock:
                self._state["genome"] = self._genome.to_dict()
            self._life_log.append_event(
                event_type="brain_nudge",
                payload={
                    "source": brain_result.get("source", "unknown"),
                    "sentiment": brain_result.get("sentiment", 0),
                    "nudges": nudges,
                    "response": str(brain_result.get("response", ""))[:200],
                },
            )
            self._logger.info("[BRAIN] Applied %d trait nudge(s)", len(nudges))

    def evolve_cycle(self) -> Dict[str, Any]:
        """Process pending stimuli, mutate genome, update awareness from fitness.

        Called automatically every ``evolve_interval`` ticks, or manually via
        ``GET /evolve``.
        """
        self._set_organism_state(OrganismState.LEARNING)
        result: Dict[str, Any] = {"mutations": {}, "stimuli_processed": 0, "reproduced": False}

        with self._state_lock:
            pending = list(self._state.get("stimuli", []))
            self._state["stimuli"] = []

        if pending:
            result["stimuli_processed"] = len(pending)
            self._logger.info("[EVOLVE] Processing %d stimuli", len(pending))
            # Stimuli trigger mutation
            mutation_record = self._genome.mutate()
            result["mutations"] = mutation_record

            with self._state_lock:
                self._state["evolution_count"] = int(self._state.get("evolution_count", 0)) + 1
                self._state["genome"] = self._genome.to_dict()

            self._life_log.append_event(
                event_type="evolution",
                payload={
                    "stimuli_count": len(pending),
                    "stimuli": [s[:80] for s in pending[:10]],
                    "mutations": mutation_record,
                    "evolution_count": self._state.get("evolution_count", 0),
                    "awareness": self._genome.fitness,
                },
            )

            # Check for reproduction (v3.4: spawn_child enforces cap)
            # v3.5: Parent energy reserve — don't reproduce below 25%
            evo = int(self._state.get("evolution_count", 0))
            parent_energy = self._autonomy.energy
            energy_reserve = 0.25
            can_reproduce = (
                self._population.should_reproduce(evo)
                and evo > self._last_reproduce_at
                and parent_energy >= energy_reserve
            )
            if can_reproduce:
                child = self._population.spawn_child(self._genome, evo)
                if child:
                    self._last_reproduce_at = evo
                    with self._state_lock:
                        self._state["last_reproduce_at"] = evo
                    result["reproduced"] = True
                    result["child_id"] = child["id"]
                    # v3.11: species divergence check
                    self._population.check_speciation(child["id"], "AL-01")
                    # v3.12: novelty metric
                    child_genome = Genome.from_dict(child["genome"])
                    novelty = self._record_novelty(self._genome, child_genome)
                    self._life_log.append_event(
                        event_type="reproduction",
                        payload={
                            "child_id": child["id"],
                            "parent_evolution": evo,
                            "child_fitness": child["genome"]["fitness"],
                            "novelty": novelty,
                        },
                    )

        # Update parent genome in population registry
        self._population.update_member("AL-01", {
            "genome": self._genome.to_dict(),
            "evolution_count": self._state.get("evolution_count", 0),
            "interaction_count": self._state.get("interaction_count", 0),
            "state": self._organism_state.value,
        })

        self.persist(force=bool(pending))
        self._set_organism_state(OrganismState.IDLE)
        return result

    def force_evolve(self) -> Dict[str, Any]:
        """Force an immediate evolution cycle (for testing via API)."""
        # Add a synthetic stimulus if none pending
        with self._state_lock:
            if not self._state.get("stimuli"):
                self._state["stimuli"] = ["forced_evolution"]
        return self.evolve_cycle()

    def population_interact(self) -> List[Dict[str, Any]]:
        """Simulate interactions between population members."""
        interactions = self._population.simulate_interactions()
        if interactions:
            self._life_log.append_event(
                event_type="population_interaction",
                payload={"interactions": interactions},
            )
        return interactions

    # ------------------------------------------------------------------
    # v2.1: Autonomous Decision Cycle (fully local, no external API)
    # ------------------------------------------------------------------

    def autonomy_cycle(self) -> Dict[str, Any]:
        """Run one autonomous decision cycle — deterministic, fully local.

        v3.0: passes environment modifiers, handles death signalling,
        records fitness in tracker, triggers autonomous mutation.
        v3.4: revives AL-01 if dead — prevents permanent death loop.
        """
        self._global_cycle += 1

        # v3.29: Tick restart recovery countdown
        if self._restart_recovery_remaining > 0:
            self._restart_recovery_remaining -= 1
            if self._restart_recovery_remaining == 0:
                self._logger.info("[RECOVERY] Restart recovery window ended — full selection resumed")

        # v3.25: Tick founder recovery counters
        if self._founder_revival_grace_remaining > 0:
            self._founder_revival_grace_remaining -= 1
        if self._founder_mutate_cooldown > 0:
            self._founder_mutate_cooldown -= 1
        if self._founder_recovery_mode:
            self._founder_cycles_since_revival += 1

        with self._state_lock:
            # Use environment-weighted fitness instead of simple mean
            env_weights = self._autonomy.env_trait_weights
            fitness = self._genome.weighted_fitness(env_weights)
            awareness = float(self._state.get("awareness", 0.0))
            mutation_rate = self._genome.mutation_rate
            # v3.8: Founder protection — cap parent mutation rate
            mutation_rate = min(mutation_rate, FOUNDER_MUTATION_CAP)
            pending = len(self._state.get("stimuli", []))
            current_traits = self._genome.traits  # raw traits for identity tracking

        # Build environment modifiers
        env_modifiers = {
            "mutation_cost_multiplier": self._environment.mutation_cost_multiplier(),
            "energy_regen_rate": self._environment.energy_regen_rate(),
            "trait_decay_multiplier": self._environment.trait_decay_multiplier(),
            "fitness_noise_penalty": self._environment.fitness_noise_penalty(),
            "survival_modifier": self._environment.survival_modifier(),
        }

        # v3.13: Draw metabolic energy from the global resource pool (parent)
        pop_size = self._population.size
        metabolic_cost = self._environment.effective_metabolic_cost()
        granted = self._environment.request_energy(metabolic_cost, pop_size)
        grant_ratio = granted / metabolic_cost if metabolic_cost > 0 else 1.0
        env_modifiers["pool_grant_ratio"] = grant_ratio

        # v3.25: Pass founder recovery state to autonomy decide
        founder_in_grace = self._founder_revival_grace_remaining > 0
        founder_mutate_blocked = (
            founder_in_grace
            or self._founder_mutate_cooldown > 0
        )
        env_modifiers["founder_recovery_mode"] = self._founder_recovery_mode
        env_modifiers["founder_mutate_blocked"] = founder_mutate_blocked
        # v3.31: Pass trait collapse emergency to autonomy
        env_modifiers["trait_collapse_emergency"] = self._trait_collapse_emergency_remaining > 0

        energy_before = self._autonomy.energy

        record = self._autonomy.decide(
            fitness, awareness, mutation_rate, pending,
            current_traits=current_traits,
            env_modifiers=env_modifiers,
        )
        decision = record["decision"]
        computed_awareness = record["awareness"]
        energy = record.get("energy", 1.0)

        # v3.25: Track forced mutation streak + cooldown
        if decision == "mutate":
            self._founder_forced_mutate_streak += 1
            if self._founder_forced_mutate_streak >= FOUNDER_MAX_CONSECUTIVE_FORCED_MUTATE:
                self._founder_mutate_cooldown = FOUNDER_MUTATE_COOLDOWN_CYCLES
                self._logger.info(
                    "[FOUNDER-RECOVERY] Forced mutation cap reached (%d) — "
                    "entering %d-cycle cooldown",
                    self._founder_forced_mutate_streak,
                    FOUNDER_MUTATE_COOLDOWN_CYCLES,
                )
                self._founder_forced_mutate_streak = 0
        else:
            self._founder_forced_mutate_streak = 0

        # v3.25: Exit founder recovery mode when fitness is healthy
        if self._founder_recovery_mode and fitness >= FOUNDER_REVIVAL_FITNESS_FLOOR * 1.5:
            self._logger.info(
                "[FOUNDER-RECOVERY] Exiting recovery mode — fitness %.4f "
                "above threshold %.4f (survived %d cycles after revival)",
                fitness, FOUNDER_REVIVAL_FITNESS_FLOOR * 1.5,
                self._founder_cycles_since_revival,
            )
            self._founder_recovery_mode = False

        # v3.23: Adaptability recovery — nudge adaptability upward when dangerously low
        adaptability_val = self._genome.effective_traits.get("adaptability", 0.5)
        if adaptability_val < ADAPTABILITY_RECOVERY_THRESHOLD:
            old_a = self._genome.get_trait("adaptability")
            self._genome.set_trait("adaptability", old_a + ADAPTABILITY_RECOVERY_NUDGE)
            record["adaptability_recovery"] = {
                "old": round(old_a, 6),
                "new": round(self._genome.get_trait("adaptability"), 6),
            }

        # v3.5: Instrumentation — energy delta & floor hits
        energy_delta = energy - energy_before
        self._cycle_stats.record_energy_delta(
            energy_delta,
            hit_floor=(energy <= self._autonomy.config.energy_min + 0.001),
        )
        self._cycle_stats.tick_alive("AL-01")

        # Record fitness in tracker
        self._evolution_tracker.record_fitness(
            "AL-01", self._global_cycle, fitness, current_traits,
        )

        # Record in population
        self._population.record_fitness("AL-01", fitness)
        self._population.update_energy("AL-01", energy)

        # Track consecutive cycles above reproduction threshold
        repro_threshold = 0.5
        if self._experiment:
            repro_threshold = self._experiment.config.reproduction_fitness_threshold
        self._population.update_consecutive_repro("AL-01", fitness, repro_threshold)

        # Record in behavior analyzer
        self._behavior_analyzer.record_decision(
            "AL-01", decision, energy, fitness, current_traits,
        )

        # v3.9: Multi-objective fitness — combine trait fitness with external metrics
        survival_norm = min(1.0, self._global_cycle / 1000.0)  # normalise survival to 0–1
        eff_ratio = self._cycle_stats.efficiency_ratio
        eff_norm = min(1.0, max(0.0, eff_ratio))
        stability = self._autonomy._identity_persistence_score
        adapt_attempts = self._autonomy._adaptation_attempt_count
        adapt_success = self._autonomy._adaptation_success_count
        adapt_rate = (adapt_success / adapt_attempts) if adapt_attempts > 0 else 0.5

        # v3.10: Inject components into genome so genome.fitness uses
        # the multi-objective formula from this point forward.
        self._genome.set_fitness_components(
            survival=survival_norm,
            efficiency=eff_norm,
            stability=stability,
            adaptation=adapt_rate,
        )

        mo_result = self._genome.multi_objective_fitness(
            survival_time=survival_norm,
            energy_efficiency_ratio=eff_norm,
            stability_score=stability,
            adaptation_success_rate=adapt_rate,
        )
        record["multi_objective_fitness"] = mo_result
        record["fitness_components"] = self._genome.fitness_components
        # v3.3: Survival grace cycles — track consecutive cycles below fitness floor
        # v3.8: Use FOUNDER_FITNESS_FLOOR for the parent (much lower than children)
        # v3.25: Extended grace during founder recovery mode
        # v3.29: Skip founder death during restart recovery window
        if self._restart_recovery_remaining > 0:
            record["restart_recovery"] = True
        else:
            survival_threshold = 0.2
            founder_threshold = FOUNDER_FITNESS_FLOOR
            survival_grace = 20
            if self._experiment:
                survival_threshold = self._experiment.config.survival_fitness_threshold
                survival_grace = self._experiment.config.survival_grace_cycles
            # v3.13: Scarcity pressure reduces survival grace
            survival_grace = self._environment.effective_survival_grace(survival_grace)
            effective_threshold = founder_threshold
            if fitness < effective_threshold:
                self._below_fitness_cycles["AL-01"] = self._below_fitness_cycles.get("AL-01", 0) + 1
                if self._below_fitness_cycles["AL-01"] >= survival_grace:
                    # v3.25: Log detailed death context for founder spiral analysis
                    self._logger.warning(
                        "[FOUNDER-DEATH] AL-01 died: cause=fitness_floor "
                        "fitness=%.4f energy=%.4f forced_mutate_streak=%d "
                        "scarcity_mult=%.2f cycles_since_revival=%d "
                        "revival_count=%d recovery_mode=%s grace_remaining=%d",
                        fitness, energy,
                        self._founder_forced_mutate_streak,
                        env_modifiers.get("mutation_cost_multiplier", 1.0),
                        self._founder_cycles_since_revival,
                        self._founder_revival_count,
                        self._founder_recovery_mode,
                        self._founder_revival_grace_remaining,
                    )
                    self._handle_death("AL-01", "fitness_floor")
                    record["organism_died"] = True
                    record["death_cause"] = "fitness_floor"
                    return record
            else:
                self._below_fitness_cycles["AL-01"] = 0

        # Check for death (energy depletion)
        # v3.30: Skip founder energy death during restart recovery window
        if record.get("organism_died") and self._restart_recovery_remaining <= 0:
            # v3.25: Log detailed death context
            self._logger.warning(
                "[FOUNDER-DEATH] AL-01 died: cause=energy_depleted "
                "fitness=%.4f energy=%.4f forced_mutate_streak=%d "
                "scarcity_mult=%.2f cycles_since_revival=%d "
                "revival_count=%d recovery_mode=%s",
                fitness, energy,
                self._founder_forced_mutate_streak,
                env_modifiers.get("mutation_cost_multiplier", 1.0),
                self._founder_cycles_since_revival,
                self._founder_revival_count,
                self._founder_recovery_mode,
            )
            self._handle_death("AL-01", "energy_depleted")
            return record
        elif record.get("organism_died"):
            # During recovery, inject emergency energy instead of death cascade
            record["organism_died"] = False
            record["restart_recovery"] = True

        # Entropy decay — if engine signals idle, decay traits
        # v3.0: scale by environment entropy pressure
        # v3.6: Suppress entropy during recovery mode — give traits time to recover
        entropy_record: Optional[Dict] = None
        in_recovery = record.get("recovery_mode", False)
        if record.get("should_entropy_decay") and not in_recovery:
            effective_entropy_rate = (
                self._autonomy.config.entropy_rate
                * env_modifiers.get("trait_decay_multiplier", 1.0)
            )
            entropy_record = self._genome.decay_traits(effective_entropy_rate)
            if entropy_record:
                record["entropy_decay"] = entropy_record

        # v3.7: Variance kick — break zero-variance stagnation lock
        kick_result = self._autonomy.apply_variance_kick(self._genome)
        if kick_result:
            record["variance_kick"] = kick_result
            with self._state_lock:
                self._state["genome"] = self._genome.to_dict()
            self._life_log.append_event(
                event_type="variance_kick",
                payload=kick_result,
            )

        # v3.7: Escalating stagnation breaker — prevents stagnation > 800
        breaker_result = self._autonomy.break_stagnation(self._genome)
        if breaker_result:
            record["stagnation_breaker"] = breaker_result
            with self._state_lock:
                self._state["genome"] = self._genome.to_dict()
            self._life_log.append_event(
                event_type="stagnation_breaker",
                payload=breaker_result,
            )
            self._logger.info(
                "[STAGNATION] Breaker tier=%d action=%s stag_count=%d",
                breaker_result.get("tier", 0),
                breaker_result.get("action", "unknown"),
                breaker_result.get("stagnation_count", 0),
            )

        if decision == DECISION_MUTATE:
            # Use effective mutation rate from engine (drift + exploration)
            eff_mr = record.get("effective_mutation_rate", mutation_rate)
            # v3.8: Founder protection — cap effective mutation rate
            eff_mr = min(eff_mr, FOUNDER_MUTATION_CAP)
            eff_delta = record.get("effective_mutation_delta", self._genome._mutation_delta)
            saved_mr = self._genome._mutation_rate
            fitness_before = self._genome.fitness
            self._genome._mutation_rate = eff_mr
            # v3.6: Directed mutation bias — upward push when below threshold
            upward_bias = self._autonomy.config.directed_mutation_bias
            mutations = self._genome.mutate(delta_override=eff_delta,
                                            upward_bias=upward_bias)
            self._genome._mutation_rate = saved_mr
            fitness_after = self._genome.fitness
            with self._state_lock:
                self._state["awareness"] = computed_awareness
                self._state["energy"] = energy
                self._state["genome"] = self._genome.to_dict()
            record["mutations"] = mutations
            # Track mutation in evolution tracker
            self._evolution_tracker.record_mutation(
                "AL-01", self._global_cycle, mutations,
                fitness_before, fitness_after, self._genome.traits,
            )
            self._life_log.append_event(
                event_type="autonomy_mutate",
                payload=record,
            )

        elif decision == DECISION_ADAPT:
            traits = self._genome.to_dict().get("traits", {})
            if traits:
                weakest = min(traits, key=traits.get)
                old_val = self._genome.get_trait(weakest)
                nudge = self._autonomy.config.adapt_trait_nudge
                # v3.7: Scale adapt nudge with stagnation depth
                stag = self._autonomy.stagnation_count
                if stag >= self._autonomy.config.stagnation_tier2_threshold:
                    nudge *= 3.0  # triple nudge at tier 2+
                elif stag >= self._autonomy.config.stagnation_tier1_threshold:
                    nudge *= 2.0  # double nudge at tier 1+
                self._genome.set_trait(weakest, old_val + nudge)
                record["adapted_trait"] = weakest
                record["old_value"] = round(old_val, 6)
                record["new_value"] = round(self._genome.get_trait(weakest), 6)
            with self._state_lock:
                self._state["awareness"] = computed_awareness
                self._state["energy"] = energy
                self._state["genome"] = self._genome.to_dict()
            self._life_log.append_event(
                event_type="autonomy_adapt",
                payload=record,
            )

        elif decision == DECISION_BLEND:
            # Cooperative genome blending — both members survive
            blend_result = self._population.cooperative_blend(
                mutation_delta=self._genome._mutation_delta,
            )
            if blend_result:
                record["blend_result"] = blend_result
                # Refresh parent genome from population registry
                parent_data = self._population.get("AL-01")
                if parent_data and parent_data.get("genome"):
                    self._genome = Genome.from_dict(parent_data["genome"])
            with self._state_lock:
                self._state["awareness"] = computed_awareness
                self._state["energy"] = energy
                self._state["evolution_count"] = (
                    int(self._state.get("evolution_count", 0)) + 1
                )
                self._state["genome"] = self._genome.to_dict()
            self._life_log.append_event(
                event_type="autonomy_blend",
                payload=record,
            )

        else:
            # Stabilize — nothing to change, just log
            with self._state_lock:
                self._state["awareness"] = computed_awareness
                self._state["energy"] = energy
            self._life_log.append_event(
                event_type="autonomy_stabilize",
                payload=record,
            )

        # v3.9: Memory drift — periodic global variance adjustment
        mdrift = self.memory_drift()
        if mdrift:
            record["memory_drift"] = mdrift

        # v3.6: Structured per-cycle logging & alert guardrails
        fitness_after_cycle = self._genome.weighted_fitness(
            self._autonomy.env_trait_weights
        )
        alerts = self._alert_guardrails.check(
            energy=energy,
            fitness=fitness_after_cycle,
            traits=self._genome.traits,
        )
        mutation_info = record.get("mutations")
        log_entry = CycleLogEntry(
            cycle=self._global_cycle,
            decision=decision,
            energy_before=round(energy_before, 6),
            energy_after=round(energy, 6),
            energy_delta=round(energy - energy_before, 6),
            fitness_before=round(fitness, 6),
            fitness_after=round(fitness_after_cycle, 6),
            fitness_delta=round(fitness_after_cycle - fitness, 6),
            mutation_applied=mutation_info,
            recovery_mode=record.get("recovery_mode", False),
            stagnation_count=record.get("stagnation_count", 0),
            alerts=alerts,
            timestamp=record.get("timestamp", ""),
        )
        self._cycle_logger.record(log_entry)
        record["cycle_log"] = log_entry.to_dict()
        record["alerts"] = alerts

        # v3.5: Record efficiency (energy cost vs fitness gained)
        self._cycle_stats.record_efficiency(
            max(0, energy_before - energy),
            fitness_after_cycle - fitness,
        )

        # v3.9: Attach internal communication signal
        record["internal_signal"] = self.internal_signal.to_dict()

        return record

    # ------------------------------------------------------------------
    # v3.9: Memory Drift — global variance adjustment based on diversity
    # ------------------------------------------------------------------

    def memory_drift(self) -> Optional[Dict[str, Any]]:
        """Apply global trait nudges to ALL living organisms based on diversity.

        Every ``_memory_drift_interval`` cycles, compute the population
        diversity index and apply random variance proportional to it:
        * Low diversity → LARGER drift to push towards exploration.
        * High diversity → SMALLER drift to avoid destabilising.

        Returns a dict describing the drift, or None if not triggered.
        """
        if self._global_cycle - self._last_memory_drift_cycle < self._memory_drift_interval:
            return None

        self._last_memory_drift_cycle = self._global_cycle
        diversity = self._population.diversity_metrics()

        # Use genome entropy as the diversity index (0 = identical, higher = diverse)
        genome_entropy = diversity.get("genome_entropy", 0.0)

        # Drift magnitude inversely proportional to diversity
        # Low entropy → large drift (0.04), high entropy → small drift (0.005)
        max_drift = 0.04
        min_drift = 0.005
        # Normalise entropy: assume max realistic entropy ≈ 4.0 bits
        diversity_norm = min(1.0, genome_entropy / 4.0)
        drift_magnitude = max_drift - (max_drift - min_drift) * diversity_norm

        rng = random.Random(self._global_cycle)
        drift_record: Dict[str, Dict[str, float]] = {}

        for oid in self._population.member_ids:
            member = self._population.get(oid)
            if not member or member.get("lifecycle_state") == LifecycleState.DEAD:
                continue
            genome_data = member.get("genome", {})
            if not genome_data:
                continue

            child_genome = Genome.from_dict(genome_data)
            changes: Dict[str, float] = {}
            for trait in list(child_genome.traits.keys()):
                delta = rng.uniform(-drift_magnitude, drift_magnitude)
                old_val = child_genome.get_trait(trait)
                child_genome.set_trait(trait, old_val + delta)
                changes[trait] = round(delta, 6)

            self._population.update_member(oid, {"genome": child_genome.to_dict()})
            drift_record[oid] = changes

            # Also update the parent's live genome reference
            if oid == "AL-01":
                # Preserve fitness_components from the current genome
                old_fc = self._genome._fitness_components
                self._genome = child_genome
                if old_fc is not None:
                    self._genome._fitness_components = dict(old_fc)
                with self._state_lock:
                    self._state["genome"] = self._genome.to_dict()

        result = {
            "type": "memory_drift",
            "cycle": self._global_cycle,
            "genome_entropy": round(genome_entropy, 4),
            "drift_magnitude": round(drift_magnitude, 6),
            "organisms_affected": len(drift_record),
            "drift_record": drift_record,
        }
        self._life_log.append_event(event_type="memory_drift", payload=result)
        self._logger.info(
            "[MEMORY-DRIFT] cycle=%d entropy=%.3f drift_mag=%.4f affected=%d",
            self._global_cycle, genome_entropy, drift_magnitude, len(drift_record),
        )
        return result

    # ------------------------------------------------------------------
    # v3.0: Environment, Auto-reproduction, Behavior, Death
    # ------------------------------------------------------------------

    def environment_tick(self) -> Dict[str, Any]:
        """Advance the environment model by one cycle.

        Drifts variables, regenerates resources, triggers scarcity events.
        v3.13: Uses smart_regenerate with population efficiency data.
        v3.23: Population-scaled regen + extinction prevention guard.
        """
        record = self._environment.tick()

        # v3.13: Smart regeneration — factor in population efficiency
        # v3.23: Pass population_size for per-organism regen bonus
        pop_size = self._population.size
        avg_eff = self._cycle_stats.efficiency_ratio
        avg_eff = max(0.0, min(1.0, avg_eff)) if avg_eff else 0.5
        self._environment.smart_regenerate(
            avg_efficiency=avg_eff, population_size=pop_size,
        )

        # v3.14: Emergency regeneration when pool drops below floor
        emergency = self._environment.emergency_regenerate(pop_size)
        if emergency > 0:
            record["emergency_regen"] = round(emergency, 4)
            self._logger.info(
                "[ENV-TICK] Emergency regen activated: +%.1f energy (pop=%d)",
                emergency, pop_size,
            )

        # v3.23: Extinction prevention — boost resources when pop = 1
        extinction_regen = self._environment.extinction_prevention_regenerate(pop_size)
        if extinction_regen > 0:
            record["extinction_prevention_regen"] = round(extinction_regen, 4)
            self._logger.info(
                "[ENV-TICK] Extinction prevention activated: +%.1f energy (pop=%d)",
                extinction_regen, pop_size,
            )

        if self._experiment:
            self._experiment.record_cycle(self._global_cycle)
        return record

    def auto_reproduce_cycle(self) -> List[Dict[str, Any]]:
        """Check all living organisms for autonomous reproduction eligibility.

        v3.0: reproduction is triggered automatically when fitness stays
        above threshold for N consecutive cycles — no manual intervention.
        v3.4: reproduction_enabled toggle from experiment config.
        """
        # v3.17: Sync tick so birth-lock / idempotency guards work
        self._population.set_tick(self._global_cycle)
        reproductions: List[Dict[str, Any]] = []
        repro_threshold = 0.5
        repro_cycles = 5
        max_pop = self._default_max_population
        reproduction_enabled = True
        if self._experiment:
            repro_threshold = self._experiment.config.reproduction_fitness_threshold
            repro_cycles = self._experiment.config.reproduction_fitness_cycles
            max_pop = self._experiment.config.max_population
            reproduction_enabled = self._experiment.config.reproduction_enabled

        # v3.16: Resource-based carrying capacity — dynamic cap from pool
        carrying_cap = self._environment.resource_carrying_capacity()
        max_pop = min(max_pop, carrying_cap)

        # Sync cap into population so spawn_child enforces it everywhere
        self._population.max_population = max_pop

        # v3.13: Scarcity pressure raises reproduction threshold
        repro_threshold = self._environment.effective_reproduction_threshold(repro_threshold)

        # v3.4: Prune if already over cap (e.g. from pre-existing population)
        # v3.29: Skip pruning during restart recovery window
        if self._population.size > max_pop and self._restart_recovery_remaining <= 0:
            min_keep = 2
            if self._experiment:
                min_keep = self._experiment.config.min_population
            deaths = self._population.prune_weakest(max_pop, min_keep)
            for d in deaths:
                self._evolution_tracker.record_death(
                    d["organism_id"], self._global_cycle, d["cause"],
                )
                self._behavior_analyzer.remove_organism(d["organism_id"])
            if deaths:
                self._logger.info(
                    "[POP-CAP] Pruned %d over-cap organisms", len(deaths),
                )
                self.check_extinction_reseed()

        if not reproduction_enabled:
            return reproductions

        # Check population cap first
        if self._population.size >= max_pop:
            return reproductions

        for oid in self._population.member_ids:
            child = self._population.auto_reproduce(
                oid,
                fitness_threshold=repro_threshold,
                required_cycles=repro_cycles,
                energy_min=self._environment.config.repro_energy_min_auto,
                energy_cost=self._environment.config.repro_energy_cost,
            )
            if child:
                child_id = child["id"]
                child_traits = child.get("genome", {}).get("traits", {})
                # Register child in evolution tracker
                self._evolution_tracker.register_organism(
                    child_id, parent_id=oid, traits=child_traits,
                    cycle=self._global_cycle,
                )
                self._evolution_tracker.record_reproduction(
                    oid, child_id, self._global_cycle, child_traits,
                )
                reproductions.append(child)
                # v3.12: novelty metric
                parent_member = self._population.get(oid)
                parent_genome = Genome.from_dict(parent_member["genome"]) if parent_member else Genome()
                child_genome = Genome.from_dict(child.get("genome", {}))
                novelty = self._record_novelty(parent_genome, child_genome)
                self._life_log.append_event(
                    event_type="auto_reproduction",
                    payload={
                        "parent_id": oid,
                        "child_id": child_id,
                        "cycle": self._global_cycle,
                        "child_fitness": child.get("genome", {}).get("fitness", 0),
                        "novelty": novelty,
                    },
                )
                self._logger.info(
                    "[AUTO-REPRO] %s → %s at cycle %d",
                    oid, child_id, self._global_cycle,
                )

                # Check population cap after each spawn
                if self._population.size >= max_pop:
                    break

        # Prune if over cap
        # v3.29: Skip post-reproduction pruning during restart recovery window
        if self._population.size > max_pop and self._restart_recovery_remaining <= 0:
            min_keep = 2
            if self._experiment:
                min_keep = self._experiment.config.min_population
            deaths = self._population.prune_weakest(max_pop, min_keep)
            for d in deaths:
                self._evolution_tracker.record_death(
                    d["organism_id"], self._global_cycle, d["cause"],
                )
                self._behavior_analyzer.remove_organism(d["organism_id"])

            # v3.2: Check for extinction after pruning
            if deaths:
                self.check_extinction_reseed()

        return reproductions

    def child_autonomy_cycle(self) -> List[Dict[str, Any]]:
        """Run one autonomous evolution cycle for every living child organism.

        v3.4: Children evolve independently — same logic as the parent's
        autonomy_cycle but applied per-child using the shared environment
        and autonomy engine for decision parameters.

        Each child:
        1. Computes weighted fitness using current environment weights.
        2. Makes a decision (mutate if below threshold, adapt or stabilize).
        3. Applies mutations / adaptation to its own genome copy.
        4. Increments its own evolution_count.
        5. Records fitness in the evolution tracker.
        6. Applies survival grace cycles (fitness floor death).
        """
        results: List[Dict[str, Any]] = []
        env_weights = self._autonomy.env_trait_weights
        eff_threshold = self._autonomy._effective_fitness_threshold
        env_modifiers = {
            "mutation_cost_multiplier": self._environment.mutation_cost_multiplier(),
            "trait_decay_multiplier": self._environment.trait_decay_multiplier(),
        }

        # Survival grace params
        survival_threshold = 0.2
        survival_grace = 20
        if self._experiment:
            survival_threshold = self._experiment.config.survival_fitness_threshold
            survival_grace = self._experiment.config.survival_grace_cycles

        # v3.31: Track fitness-floor deaths this cycle for per-cycle cap
        fitness_deaths_this_cycle: int = 0

        # v3.31: Trait collapse emergency detection — check population-wide
        # trait variance BEFORE processing individual deaths
        if self._trait_collapse_emergency_remaining > 0:
            self._trait_collapse_emergency_remaining -= 1
            if self._trait_collapse_emergency_remaining == 0:
                self._logger.info(
                    "[ECOLOGY] Trait collapse emergency ended — "
                    "normal death pressure restored"
                )
        else:
            # Sample trait variance across living population
            all_trait_vecs: List[List[float]] = []
            for mid in self._population.member_ids:
                if mid == "AL-01":
                    continue
                m = self._population.get(mid)
                if m and m.get("genome"):
                    tv = list(Genome.from_dict(m["genome"]).traits.values())
                    if tv:
                        all_trait_vecs.append(tv)
            if len(all_trait_vecs) >= 3:
                # Compute mean INTER-organism variance per trait dimension
                n_traits = len(all_trait_vecs[0])
                per_trait_vars: List[float] = []
                for ti in range(n_traits):
                    vals = [tv[ti] for tv in all_trait_vecs if ti < len(tv)]
                    if len(vals) >= 2:
                        mean_v = sum(vals) / len(vals)
                        var_v = sum((v - mean_v) ** 2 for v in vals) / len(vals)
                        per_trait_vars.append(var_v)
                pop_mean_variance = (
                    sum(per_trait_vars) / len(per_trait_vars)
                    if per_trait_vars else 1.0
                )
                if pop_mean_variance < TRAIT_COLLAPSE_VARIANCE_FLOOR:
                    self._trait_collapse_emergency_remaining = (
                        TRAIT_COLLAPSE_EMERGENCY_CYCLES
                    )
                    self._logger.warning(
                        "[ECOLOGY] TRAIT COLLAPSE EMERGENCY: "
                        "pop_mean_variance=%.6f < %.4f — "
                        "activating %d-cycle emergency (death_cap=%d, "
                        "mutation_boost=+%.2f)",
                        pop_mean_variance,
                        TRAIT_COLLAPSE_VARIANCE_FLOOR,
                        TRAIT_COLLAPSE_EMERGENCY_CYCLES,
                        TRAIT_COLLAPSE_DEATH_CAP,
                        TRAIT_COLLAPSE_MUTATION_BOOST,
                    )

        # v3.32: Compute strategy distribution early — needed for convergence check
        strategy_dist = self._behavior_analyzer.population_strategy_distribution()

        # v3.32: Strategy convergence detection — even if traits are diverse,
        # if ≥95% of organisms share the same strategy, that's also a collapse
        if not self._trait_collapse_emergency_remaining:
            total_strats = sum(strategy_dist.values())
            if total_strats >= 3:
                max_strat_count = max(strategy_dist.values()) if strategy_dist else 0
                if max_strat_count / total_strats >= STRATEGY_CONVERGENCE_THRESHOLD:
                    dominant_strat = max(strategy_dist, key=lambda k: strategy_dist[k])
                    self._trait_collapse_emergency_remaining = (
                        TRAIT_COLLAPSE_EMERGENCY_CYCLES
                    )
                    self._logger.warning(
                        "[ECOLOGY] STRATEGY CONVERGENCE EMERGENCY: "
                        "strategy=%s fraction=%.2f (%d/%d) — "
                        "activating %d-cycle emergency (death_cap=%d, "
                        "mutation_boost=+%.2f, penalty_floor=%.2f)",
                        dominant_strat, max_strat_count / total_strats,
                        max_strat_count, total_strats,
                        TRAIT_COLLAPSE_EMERGENCY_CYCLES,
                        TRAIT_COLLAPSE_DEATH_CAP,
                        TRAIT_COLLAPSE_MUTATION_BOOST,
                        MONOCULTURE_PENALTY_FLOOR_CRITICAL,
                    )

        # v3.31: Compute effective death cap for this cycle
        if self._trait_collapse_emergency_remaining > 0:
            death_cap = TRAIT_COLLAPSE_DEATH_CAP
        else:
            death_cap = MAX_FITNESS_DEATHS_PER_CYCLE

        # v3.9: Elite protection — top organisms are shielded from mutation
        elite_set = set(self._population.elite_ids())

        # v3.32: Determine if population is in critical zone (at/near floor)
        _pop_critical = (
            self._population.size
            <= self._population.min_population_floor + 2
        )

        # v3.11: Check if shock is active for resilience bonus
        shock_active = self._environment.is_shock_active
        shock_resilience_bonus = self._environment.shock_resilience_bonus

        for oid in self._population.member_ids:
            if oid == "AL-01":
                continue  # parent is handled by autonomy_cycle()

            member = self._population.get(oid)
            if not member or member.get("lifecycle_state") == LifecycleState.DEAD:
                continue

            # v3.16: Skip dormant organisms — they consume 0 metabolic cost
            if member.get("lifecycle_state") == LifecycleState.DORMANT:
                continue

            genome_data = member.get("genome", {})
            if not genome_data:
                continue

            child_genome = Genome.from_dict(genome_data)

            # v3.10: Set multi-objective fitness components for children.
            # Use available proxies since children lack their own autonomy engine.
            child_evo_count = member.get("evolution_count", 0)
            child_energy = member.get("energy", 0.8)
            child_genome.set_fitness_components(
                survival=min(1.0, child_evo_count / 50.0),
                efficiency=child_energy,
                stability=child_genome.trait_fitness,
                adaptation=min(1.0, child_evo_count / 100.0),
            )

            fitness = child_genome.weighted_fitness(env_weights)
            raw_fitness = fitness  # v3.31: preserve pre-modifier value for logging
            child_traits = child_genome.traits

            # ── v3.11: Ecosystem pressure modifiers ──────────────────
            # 1. Anti-monoculture: diminishing returns for dominant strategy
            profile = self._behavior_analyzer.get_or_create_profile(oid)
            classification = profile.classify_strategy()
            organism_strategy = classification.get("strategy", "neutral")

            monoculture_penalty = self._population.strategy_dominance_penalty(
                strategy_dist, organism_strategy,
            )
            # v3.32: Soften monoculture penalty during emergency or
            # when population is at/near the floor — prevents zombie state
            # where penalty makes survival mathematically impossible.
            if (self._trait_collapse_emergency_remaining > 0 or _pop_critical):
                monoculture_penalty = max(
                    monoculture_penalty, MONOCULTURE_PENALTY_FLOOR_CRITICAL,
                )
            fitness *= monoculture_penalty

            # 2. Explorer novelty reward: boost when explorers are rare
            if organism_strategy == "explorer":
                novelty_mult = self._population.explorer_novelty_multiplier(
                    strategy_dist,
                )
                fitness *= novelty_mult

            # 3. Shock resilience bonus: reward resilient organisms
            if shock_active and organism_strategy == "resilient":
                fitness += shock_resilience_bonus
            # ─────────────────────────────────────────────────────────

            # Record fitness in tracker
            self._evolution_tracker.record_fitness(
                oid, self._global_cycle, fitness, child_traits,
            )
            # Record in population
            self._population.record_fitness(oid, fitness)

            # Track consecutive above repro threshold
            repro_threshold = 0.5
            if self._experiment:
                repro_threshold = self._experiment.config.reproduction_fitness_threshold
            self._population.update_consecutive_repro(oid, fitness, repro_threshold)

            # Record in behavior analyzer
            energy = member.get("energy", 0.8)
            decision = "stabilize"

            # v3.5: Energy decay per cycle for children (mirrors parent)
            cfg = self._autonomy.config
            # v3.23: energy_efficiency trait reduces metabolic cost
            eff_trait = child_genome.effective_traits.get("energy_efficiency", 0.5)
            eff_scale = 1.0 - eff_trait * ENERGY_EFFICIENCY_METABOLIC_SCALE
            eff_scale = max(0.3, min(1.0, eff_scale))
            energy -= cfg.energy_decay_per_cycle * eff_scale
            cost_mult = env_modifiers.get("mutation_cost_multiplier", 1.0)

            # v3.26: Sleeping state (was conservation mode) — reduced metabolism
            in_conservation = member.get("lifecycle_state") == LifecycleState.SLEEPING
            if in_conservation:
                conservation_frac = self._environment.config.conservation_metabolic_fraction
                metabolic_cost = self._environment.effective_metabolic_cost() * conservation_frac
            else:
                metabolic_cost = self._environment.effective_metabolic_cost()

            # v3.13: Draw metabolic energy from the global resource pool
            pop_size = self._population.size
            granted = self._environment.request_energy(metabolic_cost, pop_size)
            # Scale internal energy recovery by how much was actually granted
            grant_ratio = granted / metabolic_cost if metabolic_cost > 0 else 1.0
            # Partial grant → less energy recovery, simulating resource scarcity
            energy += 0.01 * grant_ratio  # small pool-fed recovery

            # v3.23: Adaptability recovery — nudge adaptability when dangerously low
            child_adaptability = child_genome.effective_traits.get("adaptability", 0.5)
            if child_adaptability < ADAPTABILITY_RECOVERY_THRESHOLD:
                old_a = child_genome.get_trait("adaptability")
                child_genome.set_trait("adaptability", old_a + ADAPTABILITY_RECOVERY_NUDGE)

            # v3.3: Survival grace cycles — fitness floor death for children
            # v3.13: Scarcity pressure reduces survival grace
            base_survival_grace = survival_grace
            effective_grace = self._environment.effective_survival_grace(base_survival_grace)
            # v3.31: During trait collapse emergency, extend grace period
            if self._trait_collapse_emergency_remaining > 0:
                effective_grace = max(effective_grace, TRAIT_COLLAPSE_EMERGENCY_CYCLES)
            # v3.29: Skip child death during restart recovery window
            if self._restart_recovery_remaining <= 0:
                if fitness < survival_threshold:
                    self._below_fitness_cycles[oid] = self._below_fitness_cycles.get(oid, 0) + 1
                    streak = self._below_fitness_cycles[oid]
                    if streak >= effective_grace:
                        # v3.31: Enforce per-cycle death cap
                        if fitness_deaths_this_cycle >= death_cap:
                            # Cap reached — log probation instead of killing
                            self._logger.info(
                                "[SURVIVAL] %s death deferred (cap %d/%d): "
                                "raw_fitness=%.4f effective_fitness=%.4f "
                                "threshold=%.4f streak=%d/%d "
                                "monoculture_pen=%.3f emergency=%s",
                                oid, fitness_deaths_this_cycle, death_cap,
                                raw_fitness, fitness,
                                survival_threshold, streak,
                                effective_grace, monoculture_penalty,
                                self._trait_collapse_emergency_remaining > 0,
                            )
                            results.append({
                                "organism_id": oid, "decision": "probation",
                                "cause": "death_cap_reached",
                                "raw_fitness": raw_fitness,
                                "effective_fitness": fitness,
                                "threshold": survival_threshold,
                                "streak": streak,
                            })
                            # Don't reset streak — they're still below threshold
                        else:
                            # v3.31: Detailed death diagnostics
                            self._logger.warning(
                                "[SURVIVAL] %s DEATH: raw_fitness=%.4f "
                                "effective_fitness=%.4f threshold=%.4f "
                                "streak=%d/%d monoculture_pen=%.3f "
                                "strategy=%s energy=%.4f "
                                "emergency=%s cycle=%d",
                                oid, raw_fitness, fitness,
                                survival_threshold, streak,
                                effective_grace, monoculture_penalty,
                                organism_strategy, energy,
                                self._trait_collapse_emergency_remaining > 0,
                                self._global_cycle,
                            )
                            self._handle_death(oid, "fitness_floor")
                            fitness_deaths_this_cycle += 1
                            results.append({
                                "organism_id": oid, "decision": "death",
                                "cause": "fitness_floor",
                                "raw_fitness": raw_fitness,
                                "effective_fitness": fitness,
                                "threshold": survival_threshold,
                                "streak": streak,
                                "monoculture_penalty": monoculture_penalty,
                            })
                            continue
                else:
                    self._below_fitness_cycles[oid] = 0

            # Decision logic — simplified version of parent's autonomy
            evo_count = member.get("evolution_count", 0)
            mutation_rate = genome_data.get("mutation_rate", 0.05)
            mutation_delta = genome_data.get("mutation_delta", 0.05)

            # Apply exploration mode boost if active
            if self._autonomy.exploration_mode:
                mutation_rate = min(1.0, mutation_rate + self._autonomy.config.stagnation_mutation_boost)

            # v3.31: Trait collapse emergency — boost mutation to break monoculture
            if self._trait_collapse_emergency_remaining > 0:
                mutation_rate = min(1.0, mutation_rate + TRAIT_COLLAPSE_MUTATION_BOOST)
                mutation_delta = min(0.3, mutation_delta + TRAIT_COLLAPSE_MUTATION_BOOST)

            # v3.23: Stress feedback — boost mutation rate under high stress
            child_energy_stress = max(0.0, 1.0 - energy * 2.0)
            child_stress = min(1.0, child_energy_stress)
            if child_stress > cfg.stress_exploration_threshold:
                mutation_rate = min(1.0, mutation_rate + cfg.stress_mutation_boost)

            # v3.13: Scarcity pressure — raise effective mutation cost
            if self._environment.is_scarcity_pressure:
                scarcity_sev = self._environment.scarcity_severity
                cost_mult *= (1.0 + scarcity_sev * 0.5)  # up to +50% cost

            # v3.9: Elite Protection — elites skip mutation, get crossover instead
            is_elite = oid in elite_set

            # v3.23: Conservation mode — force stabilize to conserve energy
            if in_conservation:
                decision = "conservation"
                energy += cfg.energy_stabilize_bonus * 0.5  # half-rate recovery
                evo_count += 0  # no evolution during conservation
            elif is_elite and fitness < eff_threshold:
                # Elite with low fitness → targeted crossover blend with a
                # random non-elite member (avoids disturbing other genomes
                # during iteration).
                decision = "elite_crossover"
                energy -= cfg.energy_adapt_cost * cost_mult
                other_ids = [
                    mid for mid in self._population.member_ids
                    if mid != oid and mid != "AL-01" and mid not in elite_set
                ]
                if other_ids:
                    partner_id = random.choice(other_ids)
                    partner = self._population.get(partner_id)
                    if partner and partner.get("genome"):
                        partner_genome = Genome.from_dict(partner["genome"])
                        child_genome.blend_with(partner_genome, noise=mutation_delta)
                evo_count += 1
            elif fitness < eff_threshold:
                # Low fitness → mutate
                decision = "mutate"
                energy -= cfg.energy_mutate_cost * cost_mult
                fitness_before = child_genome.fitness
                saved_mr = child_genome._mutation_rate
                child_genome._mutation_rate = mutation_rate
                mutations = child_genome.mutate(delta_override=mutation_delta)
                child_genome._mutation_rate = saved_mr
                fitness_after = child_genome.fitness
                evo_count += 1

                # Track mutation in evolution tracker
                self._evolution_tracker.record_mutation(
                    oid, self._global_cycle, mutations,
                    fitness_before, fitness_after, child_genome.traits,
                )
            elif fitness < eff_threshold * 1.2:
                # Marginally above threshold → adapt weakest trait
                decision = "adapt"
                energy -= cfg.energy_adapt_cost * cost_mult
                traits = child_genome.traits
                if traits:
                    weakest = min(traits, key=lambda k: traits[k])
                    old_val = child_genome.get_trait(weakest)
                    nudge = self._autonomy.config.adapt_trait_nudge
                    child_genome.set_trait(weakest, old_val + nudge)
                evo_count += 1
            else:
                # Healthy → stabilize (entropy decay for idle children)
                decision = "stabilize"
                energy += cfg.energy_stabilize_bonus
                effective_entropy_rate = (
                    self._autonomy.config.entropy_rate
                    * env_modifiers.get("trait_decay_multiplier", 1.0)
                )
                child_genome.decay_traits(effective_entropy_rate)

            # v3.5: Clamp child energy and check for energy death
            # v3.23: Conservation mode — organisms at minimum energy enter
            # conservation instead of dying immediately
            energy = max(cfg.energy_min, min(1.0, energy))
            if energy <= CONSERVATION_ENERGY_THRESHOLD and not in_conservation:
                # Enter sleeping state — reduced metabolism, no reproduction
                self._population.enter_sleeping(oid, cause="low_energy")
                self._logger.info(
                    "[SLEEPING] %s entering sleeping state (energy=%.4f)",
                    oid, energy,
                )
            elif energy > CONSERVATION_ENERGY_THRESHOLD * 2 and in_conservation:
                # Exit sleeping state once energy recovers above 2 × threshold
                self._population.wake_sleeping(oid)
                self._logger.info(
                    "[SLEEPING] %s waking from sleep (energy=%.4f)",
                    oid, energy,
                )
            # v3.30: Skip child energy death during restart recovery window
            if energy <= 0.0 and self._restart_recovery_remaining <= 0:
                self._logger.warning(
                    "[CHILD-ENERGY] %s energy depleted — death", oid,
                )
                self._handle_death(oid, "energy_depleted")
                results.append({
                    "organism_id": oid, "decision": "death",
                    "cause": "energy_depleted", "fitness": fitness,
                })
                continue
            elif energy <= 0.0:
                # During recovery, clamp energy to minimum instead of killing
                energy = cfg.energy_min

            # Write back updated genome, evolution_count, AND energy
            self._population.update_member(oid, {
                "genome": child_genome.to_dict(),
                "evolution_count": evo_count,
                "energy": round(energy, 6),
            })

            # v3.5: Child instrumentation
            energy_before_child = member.get("energy", 0.8)
            child_energy_delta = energy - energy_before_child
            self._cycle_stats.record_energy_delta(
                child_energy_delta,
                hit_floor=(energy <= cfg.energy_min + 0.001),
            )
            self._cycle_stats.tick_alive(oid)
            fitness_after_child = child_genome.fitness
            self._cycle_stats.record_efficiency(
                max(0, -child_energy_delta),
                fitness_after_child - fitness,
            )

            # Record in behavior analyzer
            self._behavior_analyzer.record_decision(
                oid, decision, energy, fitness, child_genome.traits,
            )

            results.append({
                "organism_id": oid,
                "decision": decision,
                "fitness": round(fitness, 6),
                "evolution_count": evo_count,
            })

        if results:
            self._logger.info(
                "[CHILD-EVOLVE] Evolved %d children at cycle %d",
                len(results), self._global_cycle,
            )
        return results

    def behavior_analysis_cycle(self) -> Dict[str, Any]:
        """Run population-level behavior analysis.

        Records trait/fitness snapshots and detects convergence/divergence.
        """
        pop_traits = self._population.population_traits()
        pop_fitness = self._population.population_fitness()
        analysis = self._behavior_analyzer.record_population_snapshot(
            pop_traits, pop_fitness,
        )
        return analysis

    def _handle_death(self, organism_id: str, cause: str) -> None:
        """Process organism death: children die permanently, AL-01 gets founder protection.

        v3.28: Children skip dormancy entirely and go straight to graveyard.
        Death is terminal and permanent for all children — no revival path.
        AL-01 (founder) receives emergency energy injection instead of dying,
        preserving the original lineage.
        """
        member = self._population.get(organism_id)

        # ── v3.28: AL-01 founder protection ─────────────────────────
        if organism_id == "AL-01" and member is not None:
            # Inject emergency energy instead of killing AL-01
            self._founder_revival_count += 1
            self._founder_recovery_mode = True
            self._founder_revival_grace_remaining = FOUNDER_REVIVAL_GRACE_CYCLES
            self._founder_cycles_since_revival = 0
            self._founder_forced_mutate_streak = 0

            new_energy = FOUNDER_REVIVAL_ENERGY
            self._population.update_energy("AL-01", new_energy)
            self._autonomy._energy = new_energy

            self._logger.warning(
                "[FOUNDER-RESCUE] AL-01 rescued from death: cause=%s "
                "revival_count=%d energy=%.4f grace_cycles=%d cycle=%d",
                cause, self._founder_revival_count, new_energy,
                FOUNDER_REVIVAL_GRACE_CYCLES, self._global_cycle,
            )
            self._life_log.append_event(
                event_type="founder_rescue",
                payload={
                    "organism_id": "AL-01",
                    "cause": cause,
                    "revival_count": self._founder_revival_count,
                    "energy_after": new_energy,
                    "cycle": self._global_cycle,
                },
            )
            # Check extinction recovery (population may still be critical)
            self.check_extinction_reseed()
            return

        # ── v3.28: Children — permanent death, no dormancy ──────────
        # Children skip dormancy entirely and go straight to graveyard.
        death_info = self._population.remove_member(
            organism_id, cause, death_cycle=self._global_cycle)
        if death_info:
            self._evolution_tracker.record_death(
                organism_id, self._global_cycle, cause,
            )
            self._behavior_analyzer.remove_organism(organism_id)
            self._cycle_stats.record_death(organism_id)
            self._life_log.append_event(
                event_type="organism_death",
                payload={
                    "organism_id": organism_id,
                    "cause": cause,
                    "cycle": self._global_cycle,
                    **death_info,
                },
            )
            self._logger.warning(
                "[DEATH] %s marked permanently dead → graveyard: cause=%s cycle=%d",
                organism_id, cause, self._global_cycle,
            )

            # v3.2: Check for extinction — reseed from vault if needed
            self.check_extinction_reseed()

    def check_extinction_reseed(self) -> Optional[Dict[str, Any]]:
        """v3.15 Extinction Recovery Protocol.

        If population < 5, auto-seed up to 10 new children from the
        top-fitness lineage.  Falls back to Genesis Vault if no fit
        organisms are available.

        Returns the reseed/recovery record if action was taken, or None.
        """
        pop_size = self._population.size
        EXTINCTION_THRESHOLD = 5
        RECOVERY_SPAWN_COUNT = 10

        # Original genesis vault reseed — population fully extinct
        if pop_size == 0:
            reseed = self._genesis_vault.check_and_reseed(
                population=self._population,
                evolution_tracker=self._evolution_tracker,
                life_log=self._life_log,
                behavior_analyzer=self._behavior_analyzer,
                global_cycle=self._global_cycle,
            )
            if reseed:
                self._logger.warning(
                    "[GENESIS VAULT] Reseed #%d: spawned %s from genesis seed at cycle %d",
                    reseed["reseed_number"],
                    reseed["organism_id"],
                    self._global_cycle,
                )
            return reseed

        # v3.15: Extinction recovery — pop < 5 but not zero
        if pop_size >= EXTINCTION_THRESHOLD:
            return None

        self._logger.warning(
            "[EXTINCTION RECOVERY] Population critically low (%d < %d) — "
            "initiating recovery protocol at cycle %d",
            pop_size, EXTINCTION_THRESHOLD, self._global_cycle,
        )

        # Find top-fitness organisms (living, dormant, or recently dead)
        top_members = self._population.top_fitness_members(n=5)
        if not top_members:
            # Fallback: use genesis vault seed
            self._logger.warning(
                "[EXTINCTION RECOVERY] No fit organisms found — falling back to genesis vault",
            )
            reseed = self._genesis_vault.check_and_reseed(
                population=self._population,
                evolution_tracker=self._evolution_tracker,
                life_log=self._life_log,
                behavior_analyzer=self._behavior_analyzer,
                global_cycle=self._global_cycle,
            )
            return reseed

        # Spawn children from the top-fitness lineage
        spawned = []
        seeds_used = []
        to_spawn = RECOVERY_SPAWN_COUNT
        for seed_member in top_members:
            if to_spawn <= 0:
                break
            seed_genome_data = seed_member.get("genome", {})
            if not seed_genome_data:
                continue
            seed_genome = Genome.from_dict(seed_genome_data)
            seed_id = seed_member.get("id", "unknown")
            seed_evo = seed_member.get("evolution_count", 0)
            seeds_used.append(seed_id)

            # Spawn multiple children per seed (distribute evenly)
            per_seed = max(1, to_spawn // max(1, len(top_members)))
            for _ in range(per_seed):
                if to_spawn <= 0:
                    break
                child = self._population.spawn_child(
                    seed_genome, parent_evolution=seed_evo, parent_id=seed_id,
                )
                if child:
                    child_id = child["id"]
                    child_traits = child.get("genome", {}).get("traits", {})
                    self._evolution_tracker.register_organism(
                        child_id, parent_id=seed_id, traits=child_traits,
                        cycle=self._global_cycle,
                    )
                    # v3.11: species divergence check
                    self._population.check_speciation(child_id, seed_id)
                    # v3.12: novelty metric
                    child_genome_obj = Genome.from_dict(child.get("genome", {}))
                    novelty = self._record_novelty(seed_genome, child_genome_obj)
                    spawned.append(child_id)
                    to_spawn -= 1

        # v3.28: Do NOT wake dormant organisms during extinction recovery.
        # Children with permanent death should never be revived.
        woke = []

        recovery_record = {
            "event": "extinction_recovery",
            "cycle": self._global_cycle,
            "population_before": pop_size,
            "population_after": self._population.size,
            "spawned": spawned,
            "seeds_used": seeds_used,
            "dormant_woken": woke,
            "timestamp": _utc_now(),
        }

        self._life_log.append_event(
            event_type="extinction_recovery",
            payload=recovery_record,
        )
        self._logger.warning(
            "[EXTINCTION RECOVERY] Spawned %d children from %d seeds, woke %d dormant "
            "(pop: %d → %d) at cycle %d",
            len(spawned), len(seeds_used), len(woke),
            pop_size, self._population.size, self._global_cycle,
        )
        return recovery_record

    # ------------------------------------------------------------------
    # v3.15: Dormant Lifecycle Management
    # ------------------------------------------------------------------

    def wake_dormant_cycle(self) -> List[Dict[str, Any]]:
        """v3.15: Check dormant organisms and wake those whose conditions improved.

        v3.28: Only AL-01 can legitimately enter dormancy. Children go
        straight to graveyard on death, so dormant_ids should only contain
        AL-01 (if at all). This method will NOT wake child organisms.

        Dormant organisms are woken when:
        - Resource pool is above scarcity threshold (conditions have improved)
        - OR population is critically low (< 5, extinction recovery)

        Returns list of wake records.
        """
        woke: List[Dict[str, Any]] = []
        dormant_ids = self._population.dormant_ids
        if not dormant_ids:
            return woke

        # v3.16: Configurable pool threshold for waking dormant organisms
        wake_threshold = self._environment.config.dormant_wake_pool_fraction
        pool_healthy = self._environment.pool_fraction >= wake_threshold
        pop_critical = self._population.size < 5

        if not pool_healthy and not pop_critical:
            return woke

        for did in dormant_ids:
            # v3.28: Only wake AL-01 (founder). Children with permanent
            # death should never be revived through dormancy waking.
            if did != "AL-01":
                self._logger.error(
                    "[ERROR] Dead child %s found in dormant_ids — "
                    "children should not enter dormancy. Skipping.",
                    did,
                )
                continue
            wake_record = self._population.wake_dormant(did, energy_boost=0.3)
            if wake_record:
                self._life_log.append_event(
                    event_type="dormant_wake",
                    payload={
                        "organism_id": did,
                        "cycle": self._global_cycle,
                        **wake_record,
                    },
                )
                self._logger.info(
                    "[DORMANT-WAKE] %s woke from dormancy at cycle %d (pool_healthy=%s, pop_critical=%s)",
                    did, self._global_cycle, pool_healthy, pop_critical,
                )
                woke.append(wake_record)
        return woke

    # ------------------------------------------------------------------
    # v3.16: Stability Reproduction
    # ------------------------------------------------------------------

    def stability_reproduction_cycle(self) -> List[Dict[str, Any]]:
        """v3.16: Spawn offspring when resources are abundant and fitness is high.

        Conditions (per organism):
        - Pool fraction ≥ ``stability_repro_pool_fraction`` (default 80%)
        - Organism fitness ≥ adaptive fitness threshold
        - Random probability per cycle (default 5%)
        - Population under carrying capacity

        Returns list of spawn records.
        """
        # v3.17: Sync tick so birth-lock / idempotency guards work
        self._population.set_tick(self._global_cycle)
        spawned: List[Dict[str, Any]] = []
        cfg = self._environment.config
        pool_frac = self._environment.pool_fraction

        if pool_frac < cfg.stability_repro_pool_fraction:
            return spawned

        # Dynamic carrying cap from resource pool
        carrying_cap = self._environment.resource_carrying_capacity()
        if self._population.size >= carrying_cap:
            return spawned

        # Adaptive fitness baseline
        eff_threshold = self._autonomy._effective_fitness_threshold

        for oid in self._population.member_ids:
            if self._population.size >= carrying_cap:
                break

            member = self._population.get(oid)
            if not member:
                continue
            if not LifecycleState.can_reproduce(
                    member.get("lifecycle_state", LifecycleState.ACTIVE)):
                continue

            fitness = member.get("genome", {}).get("fitness", 0.0)
            if fitness < eff_threshold:
                continue

            # v3.18: Energy gate for stability reproduction
            parent_energy = member.get("energy", 0.0)
            if parent_energy < cfg.repro_energy_min_stability:
                continue

            # Probability gate — slow rate, v3.18: scales with resource pool
            effective_prob = cfg.stability_repro_probability * pool_frac
            if random.random() > effective_prob:
                continue

            genome_data = member.get("genome", {})
            parent_genome = Genome.from_dict(genome_data)
            evo = member.get("evolution_count", 0)

            child = self._population.spawn_child(
                parent_genome, parent_evolution=evo, parent_id=oid,
            )
            if child:
                # v3.18: Deduct reproduction energy cost from parent
                self._population.deduct_energy(oid, cfg.repro_energy_cost)
                child_id = child["id"]
                child_traits = child.get("genome", {}).get("traits", {})
                self._evolution_tracker.register_organism(
                    child_id, parent_id=oid, traits=child_traits,
                    cycle=self._global_cycle,
                )
                self._evolution_tracker.record_reproduction(
                    oid, child_id, self._global_cycle, child_traits,
                )
                # v3.11: species divergence check
                self._population.check_speciation(child_id, oid)
                # v3.12: novelty metric
                child_genome_obj = Genome.from_dict(child.get("genome", {}))
                novelty = self._record_novelty(parent_genome, child_genome_obj)
                spawned.append(child)
                self._life_log.append_event(
                    event_type="stability_reproduction",
                    payload={
                        "parent_id": oid,
                        "child_id": child_id,
                        "cycle": self._global_cycle,
                        "pool_fraction": round(pool_frac, 4),
                        "parent_fitness": round(fitness, 4),
                        "novelty": novelty,
                    },
                )
                self._logger.info(
                    "[STABILITY-REPRO] %s → %s at cycle %d (pool=%.1f%%, fitness=%.4f)",
                    oid, child_id, self._global_cycle, pool_frac * 100, fitness,
                )

        return spawned

    # ------------------------------------------------------------------
    # v3.16: Lone Survivor Reproduction
    # ------------------------------------------------------------------

    def lone_survivor_reproduction(self) -> Optional[Dict[str, Any]]:
        """v3.16: Prevent total extinction — if pop == 1 and pool is healthy,
        auto-trigger reproduction with a small probability per cycle.

        Conditions:
        - Population size == 1
        - Pool fraction ≥ ``lone_survivor_pool_fraction`` (default 70%)
        - Random probability per cycle (default 10%)

        Returns spawn record or None.
        """
        # v3.17: Sync tick so birth-lock / idempotency guards work
        self._population.set_tick(self._global_cycle)
        if self._population.size != 1:
            return None

        cfg = self._environment.config
        if self._environment.pool_fraction < cfg.lone_survivor_pool_fraction:
            return None

        # v3.18: Probability scales with resource pool level
        effective_prob = cfg.lone_survivor_repro_probability * self._environment.pool_fraction
        if random.random() > effective_prob:
            return None

        # The lone survivor reproduces
        survivor_id = self._population.member_ids[0]
        member = self._population.get(survivor_id)
        if not member:
            return None

        # v3.18: Energy gate for lone-survivor reproduction
        survivor_energy = member.get("energy", 0.0)
        if survivor_energy < cfg.repro_energy_min_lone_survivor:
            return None

        genome_data = member.get("genome", {})
        parent_genome = Genome.from_dict(genome_data)
        evo = member.get("evolution_count", 0)

        child = self._population.spawn_child(
            parent_genome, parent_evolution=evo, parent_id=survivor_id,
        )
        if not child:
            return None

        # v3.18: Deduct reproduction energy cost from parent
        self._population.deduct_energy(survivor_id, cfg.repro_energy_cost)

        child_id = child["id"]
        child_traits = child.get("genome", {}).get("traits", {})
        self._evolution_tracker.register_organism(
            child_id, parent_id=survivor_id, traits=child_traits,
            cycle=self._global_cycle,
        )
        self._evolution_tracker.record_reproduction(
            survivor_id, child_id, self._global_cycle, child_traits,
        )
        # v3.11: species divergence check
        self._population.check_speciation(child_id, survivor_id)
        # v3.12: novelty metric
        child_genome_obj = Genome.from_dict(child.get("genome", {}))
        novelty = self._record_novelty(parent_genome, child_genome_obj)
        self._life_log.append_event(
            event_type="lone_survivor_reproduction",
            payload={
                "parent_id": survivor_id,
                "child_id": child_id,
                "cycle": self._global_cycle,
                "pool_fraction": round(self._environment.pool_fraction, 4),
                "novelty": novelty,
            },
        )
        self._logger.warning(
            "[LONE-SURVIVOR] %s → %s at cycle %d (pop=1, pool=%.1f%%)",
            survivor_id, child_id, self._global_cycle,
            self._environment.pool_fraction * 100,
        )
        return child

    # ------------------------------------------------------------------
    # Rare reproduction (v3.10)
    # ------------------------------------------------------------------

    def rare_reproduction_cycle(self) -> Optional[Dict[str, Any]]:
        """Rare, heavily-gated reproduction path.

        Rules
        -----
        * Evaluated every RARE_REPRO_CYCLE_INTERVAL cycles (500).
        * 5 % random chance gate.
        * Hard population cap of RARE_REPRO_POP_CAP (50).
        * Parent eligibility: energy >= 0.65, fitness >= 0.50,
          stagnation < 0.90 (estimated from fitness_history).
        * Cost: parent.energy -= 0.30; child.energy = 0.50.
        * Cooldown: 2000 cycles between births per parent.
        * Only ONE child per invocation (no loops).
        """
        # v3.17: Sync tick so birth-lock / idempotency guards work
        self._population.set_tick(self._global_cycle)
        # ── Gate 1: cycle interval ────────────────────────────────────
        if self._global_cycle % RARE_REPRO_CYCLE_INTERVAL != 0:
            return None

        # ── Gate 2: random chance, v3.18: scales with resource pool ────
        effective_chance = RARE_REPRO_CHANCE * self._environment.pool_fraction
        if random.random() >= effective_chance:
            return None

        # ── Gate 3: hard population cap ───────────────────────────────
        if self._population.size >= RARE_REPRO_POP_CAP:
            self._logger.debug(
                "[RARE-REPRO] Pop %d >= cap %d — skipped",
                self._population.size, RARE_REPRO_POP_CAP,
            )
            return None

        # ── Find one eligible parent ──────────────────────────────────
        candidates = list(self._population.member_ids)
        random.shuffle(candidates)

        for cid in candidates:
            member = self._population.get(cid)
            if member is None or not LifecycleState.can_reproduce(
                    member.get("lifecycle_state", LifecycleState.ACTIVE)):
                continue

            energy = member.get("energy", 0.0)
            if energy < RARE_REPRO_ENERGY_MIN:
                continue

            # Fitness: use latest entry in fitness_history
            fh = member.get("fitness_history", [])
            fitness = fh[-1] if fh else 0.0
            if fitness < RARE_REPRO_FITNESS_MIN:
                continue

            # Stagnation estimate from fitness_history
            stagnation = self._estimate_stagnation(fh)
            if stagnation >= RARE_REPRO_STAGNATION_MAX:
                continue

            # Cooldown check
            last_birth = self._last_birth_cycle.get(cid, -RARE_REPRO_COOLDOWN_CYCLES)
            if self._global_cycle - last_birth < RARE_REPRO_COOLDOWN_CYCLES:
                continue

            # ── Re-check population cap before committing ─────────────
            if self._population.size >= RARE_REPRO_POP_CAP:
                return None

            # ── Reproduce ─────────────────────────────────────────────
            parent_genome_data = member.get("genome", {})
            parent_genome = Genome.from_dict(parent_genome_data)

            parent_gen = member.get("generation_id", 0)
            child = self._population.spawn_child(
                parent_genome=parent_genome,
                parent_evolution=member.get("evolution_count", 0),
                parent_id=cid,
                mutation_variance=RARE_REPRO_MUTATION_RATE,
            )
            if child is None:
                return None  # cap enforced inside spawn_child

            child_id = child["id"]

            # v3.11: species divergence check
            self._population.check_speciation(child_id, cid)

            # v3.12: novelty metric
            child_genome_obj = Genome.from_dict(child.get("genome", {}))
            novelty = self._record_novelty(parent_genome, child_genome_obj)

            # Override child energy to spec
            self._population.update_energy(child_id, RARE_REPRO_CHILD_ENERGY)

            # Deduct parent energy cost
            new_parent_energy = round(energy - RARE_REPRO_ENERGY_COST, 6)
            self._population.update_energy(cid, max(new_parent_energy, 0.0))

            # Record cooldown
            self._last_birth_cycle[cid] = self._global_cycle

            # Evolution tracker entries
            child_traits = child.get("genome", {}).get("traits", {})
            self._evolution_tracker.register_organism(
                child_id, parent_id=cid, traits=child_traits,
                cycle=self._global_cycle,
            )
            self._evolution_tracker.record_reproduction(
                cid, child_id, self._global_cycle, child_traits,
            )

            # VITAL life-log entry
            self._life_log.append_event(
                event_type="rare_reproduction",
                payload={
                    "parent_id": cid,
                    "child_id": child_id,
                    "cycle": self._global_cycle,
                    "parent_energy_after": round(max(new_parent_energy, 0.0), 4),
                    "child_energy": RARE_REPRO_CHILD_ENERGY,
                    "parent_fitness": round(fitness, 4),
                    "stagnation_estimate": round(stagnation, 4),
                    "population_size": self._population.size,
                    "novelty": novelty,
                },
            )

            self._logger.info(
                "[RARE-REPRO] %s → %s at cycle %d "
                "(fit=%.3f, stag=%.3f, pop=%d)",
                cid, child_id, self._global_cycle,
                fitness, stagnation, self._population.size,
            )
            return child  # ONE child only — exit immediately

        return None  # no eligible parent found

    @staticmethod
    def _estimate_stagnation(fitness_history: list) -> float:
        """Return 0–1 stagnation estimate from a fitness history.

        Uses the last 20 entries.  If the range (max − min) is tiny the
        organism is stagnating (result → 1.0).  With fewer than 2 data
        points we assume zero stagnation (benefit of the doubt).
        """
        if len(fitness_history) < 2:
            return 0.0
        window = fitness_history[-20:]
        hi, lo = max(window), min(window)
        spread = hi - lo
        # Map spread → stagnation: spread=0 → 1.0, spread>=0.10 → 0.0
        return max(0.0, min(1.0, 1.0 - spread / 0.10))

    # ------------------------------------------------------------------
    # Stimulation (lightweight external poke)
    # ------------------------------------------------------------------

    def stimulate(self, stimulus: Optional[str] = None) -> None:
        """External stimulus: bump interaction_count, persist.

        If *stimulus* is provided, it is also added to the stimuli queue for
        the next evolution cycle (which handles record_stimulus internally).
        """
        self._set_organism_state(OrganismState.RESPONDING)
        with self._state_lock:
            self._state["interaction_count"] = int(self._state.get("interaction_count", 0)) + 1
        self._life_log.append_event(
            event_type="stimulate",
            payload={
                "interaction_count": self._state["interaction_count"],
                "awareness": self._state["awareness"],
                "stimulus": stimulus,
            },
        )
        # Queue stimulus for evolution processing
        if stimulus:
            self.add_stimulus(stimulus)  # calls record_stimulus internally
        else:
            self._autonomy.record_stimulus()  # bare stimulate
        self.persist(force=True)
        self._set_organism_state(OrganismState.IDLE)

    # ------------------------------------------------------------------
    # Interaction recording
    # ------------------------------------------------------------------

    def record_interaction(
        self,
        user_input: str,
        response: str,
        mood: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record a structured interaction into memory + SQLite and return the entry."""
        self._set_organism_state(OrganismState.LEARNING)

        now = _utc_now()
        with self._state_lock:
            self._state["interaction_count"] = int(self._state.get("interaction_count", 0)) + 1
            current_mood = mood or self._organism_state.value

        # Write to SQLite (primary structured store)
        row_id = self._memory_manager.database.write_interaction(
            user_input=user_input,
            response=response,
            mood=current_mood,
            organism_state=self._organism_state.value,
            timestamp=now,
        )

        # Also write to Firestore/JSON memory for backward compatibility
        entry = {
            "event_type": "interaction",
            "payload": {
                "user_input": user_input,
                "response": response,
                "mood": current_mood,
                "timestamp": now,
                "db_id": row_id,
            },
        }
        self._memory_manager.write_memory(entry)
        self._set_organism_state(OrganismState.IDLE)
        return entry

    def recent_interactions(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return the last *n* interactions from SQLite."""
        return self._memory_manager.database.recent_interactions(n=n)

    def search_memory(
        self,
        keyword: str,
        limit: int = 20,
        since_timestamp: Optional[str] = None,
        contains_all: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search memory entries by keyword across user_input, response, mood."""
        return self._memory_manager.search_memory(
            keyword,
            limit=limit,
            since_timestamp=since_timestamp,
            contains_all=contains_all,
        )

    @property
    def uptime_seconds(self) -> float:
        """Seconds since this process started."""
        return time.monotonic() - self._boot_time

    @property
    def growth_metrics(self) -> Dict[str, Any]:
        """Return current growth metrics + totals since first run."""
        with self._state_lock:
            interaction_count = int(self._state.get("interaction_count", 0))
            awareness = float(self._state.get("awareness", 0.0))
            evolution_count = int(self._state.get("evolution_count", 0))
        memory_size = self._memory_manager.memory_size()
        db_summary = self._memory_manager.database.growth_summary()
        return {
            "interaction_count": interaction_count,
            "memory_size": memory_size,
            "awareness": awareness,
            "evolution_count": evolution_count,
            "version": VERSION,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "age_seconds": round(self.age_seconds, 2),
            "birth_time": self._state.get("birth_time"),
            "uptime_sessions": int(self._state.get("uptime_sessions", 0)),
            "integrity_status": self._life_log.integrity_status,
            "life_log_events": self._life_log.head_seq,
            "policy_weights": self._policy.weights,
            "total_interactions_all_time": db_summary.get("total_interactions", interaction_count),
            "first_snapshot_timestamp": db_summary.get("first_snapshot_timestamp"),
            "snapshot_count": db_summary.get("snapshot_count", 0),
            "firestore_writes_today": self._memory_manager.firestore_writes_today,
            # v2.0 additions
            "genome": self._genome.to_dict(),
            "fitness": round(self._genome.fitness, 6),
            "fitness_components": self._genome.fitness_components,
            "population_size": self._population.size,
            "population_ids": self._population.member_ids,
            "pending_stimuli": len(self._state.get("stimuli", [])),
            "brain_enabled": self._brain.enabled,
            # v2.1 autonomy
            "stagnation_count": self._autonomy.stagnation_count,
            "autonomy": self._autonomy.summary(),
            # v2.3 vital score
            "vital_score": self._autonomy.compute_vital_score(),
            # v3.0 additions
            "global_cycle": self._global_cycle,
            "environment": self._environment.state_snapshot(),
            "trait_variance": self._population.trait_variance(),
            "behavior_summary": self._behavior_analyzer.summary(),
            "evolution_tracker": {
                "generation_counter": self._evolution_tracker.generation_counter,
                "mutation_events": self._evolution_tracker.mutation_event_count(),
                "lineages": len(self._evolution_tracker.get_all_lineages()),
            },
            "experiment": self._experiment.status() if self._experiment else None,
            # v3.1
            "snapshot_manager": self._snapshot_manager.status() if self._snapshot_manager else None,
        }

    def record_growth_snapshot(self) -> None:
        """Write a point-in-time growth snapshot to SQLite."""
        with self._state_lock:
            interaction_count = int(self._state.get("interaction_count", 0))
            awareness = float(self._state.get("awareness", 0.0))
            evolution_count = int(self._state.get("evolution_count", 0))
        memory_size = self._memory_manager.memory_size()
        self._memory_manager.database.write_growth_snapshot(
            interaction_count=interaction_count,
            memory_size=memory_size,
            awareness=awareness,
            evolution_count=evolution_count,
            version=VERSION,
        )

    def _set_organism_state(self, new_state: OrganismState) -> None:
        old = self._organism_state
        self._organism_state = new_state
        if old != new_state:
            self._logger.debug("[STATE] %s -> %s", old.value, new_state.value)

    def boot(self) -> None:
        now = _utc_now()
        with self._state_lock:
            evolution = int(self._state.get("evolution_count", 0)) + 1
            self._state["evolution_count"] = evolution
            self._state["last_boot_utc"] = now

        # v3.26: Restore _global_cycle from persisted state
        with self._state_lock:
            self._global_cycle = int(self._state.get("global_cycle", 0))
            # Restore founder state from persisted state
            self._founder_revival_count = int(self._state.get("founder_revival_count", 0))
            self._founder_recovery_mode = bool(self._state.get("founder_recovery_mode", False))
            self._founder_revival_grace_remaining = int(self._state.get("founder_revival_grace_remaining", 0))
            self._founder_forced_mutate_streak = int(self._state.get("founder_forced_mutate_streak", 0))
            self._founder_mutate_cooldown = int(self._state.get("founder_mutate_cooldown", 0))
            self._founder_cycles_since_revival = int(self._state.get("founder_cycles_since_revival", 0))

            # v3.29: Restore survival-critical counters so restart doesn't reset grace
            self._below_fitness_cycles = {
                k: int(v) for k, v in self._state.get("below_fitness_cycles", {}).items()
            }
            self._last_birth_cycle = {
                k: int(v) for k, v in self._state.get("last_birth_cycle", {}).items()
            }
            self._conservation_mode = {
                k: bool(v) for k, v in self._state.get("conservation_mode", {}).items()
            }
            # v3.31: Restore trait collapse emergency countdown
            self._trait_collapse_emergency_remaining = int(
                self._state.get("trait_collapse_emergency_remaining", 0)
            )

        # v3.29: Restore environment from persisted state (prevents trait-weight drift)
        env_state = self._state.get("environment_state")
        if env_state and isinstance(env_state, dict):
            from al01.environment import EnvironmentConfig
            restored_env = Environment.from_dict(env_state, config=self._environment._config)
            # Transfer restored internal state into the live environment object
            self._environment._cycle = restored_env._cycle
            self._environment._resource_pool = restored_env._resource_pool
            self._environment._temperature = restored_env._temperature
            self._environment._entropy_pressure = restored_env._entropy_pressure
            self._environment._resource_abundance = restored_env._resource_abundance
            self._environment._noise_level = restored_env._noise_level
            self._environment._scarcity_events = restored_env._scarcity_events
            self._environment._shift_log = restored_env._shift_log
            self._environment._scarcity_log = restored_env._scarcity_log
            self._environment._next_shift_cycle = restored_env._next_shift_cycle
            self._environment._shock_events = restored_env._shock_events
            self._environment._shock_log = restored_env._shock_log
            self._environment._named_events = restored_env._named_events
            self._environment._named_event_log = restored_env._named_event_log
            self._logger.info("[BOOT] Environment restored from persisted state (cycle=%d)", restored_env._cycle)

        # v3.29: Activate restart recovery window — suppress selection for N cycles
        self._restart_recovery_remaining = RESTART_RECOVERY_CYCLES
        self._logger.info(
            "[BOOT] Restart recovery window active for %d cycles",
            RESTART_RECOVERY_CYCLES,
        )

        # v3.28: Rescue AL-01 from graveyard on startup (founder protection)
        rescued = self._population.rescue_from_graveyard("AL-01")
        if rescued:
            self._founder_revival_count += 1
            self._founder_recovery_mode = True
            self._founder_revival_grace_remaining = FOUNDER_REVIVAL_GRACE_CYCLES
            self._logger.warning(
                "[FOUNDER-RESCUE] AL-01 rescued from graveyard on boot "
                "(revival_count=%d)",
                self._founder_revival_count,
            )

        # v3.28: Startup validation — log errors for dead children in _members
        for mid in list(self._population._members.keys()):
            if mid == "AL-01":
                continue
            m = self._population._members[mid]
            if m.get("lifecycle_state") == LifecycleState.DEAD:
                self._logger.error(
                    "[ERROR] Dead child %s found in _members on startup — "
                    "should be in graveyard. Moving to graveyard.",
                    mid,
                )
                self._population.remove_member(mid, cause="startup_cleanup")

        # Verify life-log integrity on startup
        self._life_log.startup_verify()

        self._logger.info("[BOOT] Boot #%d at %s", evolution, now)

        # Log boot event to life log
        self._life_log.append_event(
            event_type="boot",
            payload={
                "evolution_count": evolution,
                "last_boot_utc": now,
                "uptime_sessions": self._state.get("uptime_sessions", 1),
                "birth_time": self._state.get("birth_time"),
                "integrity_status": self._life_log.integrity_status,
            },
        )

        self._memory_manager.write_memory(
            {
                "event_type": "boot",
                "payload": {
                    "evolution_count": evolution,
                    "last_boot_utc": now,
                },
            }
        )
        self.persist(force=True)
        self._logger.info("[BOOT] Boot persisted successfully")

    def pulse(self, log_observation: bool = True) -> None:
        with self._state_lock:
            awareness = float(self._state.get("awareness", 0.0))

        if log_observation:
            self._logger.debug("Pulse: awareness=%.4f", awareness)
            self._memory_manager.write_memory(
                {
                    "event_type": "pulse",
                    "payload": {"awareness": awareness},
                }
            )

    # v3.22: Maximum size for a single reflection fragment to prevent
    # exponential payload doubling (reflection embeds previous reflections).
    _REFLECT_FRAGMENT_MAX: int = 200

    def reflect(self) -> None:
        self._set_organism_state(OrganismState.REFLECTING)
        recent = self._memory_manager.read_memory(limit=5)
        if not recent:
            return

        fragments = []
        for entry in recent:
            event_type = entry.get("event_type", "event")
            payload = entry.get("payload", {})
            # v3.22: For reflection entries, use a short label instead of
            # re-serialising the nested summary (which doubles in size
            # every cycle due to repeated JSON escaping).
            if event_type == "reflection":
                raw = str(payload.get("summary", ""))
                frag = f"reflection:[{raw[:self._REFLECT_FRAGMENT_MAX]}]"
            else:
                raw = json.dumps(payload, sort_keys=True)
                frag = f"{event_type}:{raw[:self._REFLECT_FRAGMENT_MAX]}"
            fragments.append(frag)
        summary = "; ".join(fragments)

        self._memory_manager.write_memory(
            {
                "event_type": "reflection",
                "payload": {"summary": summary},
            }
        )

        # Log reflection to life log
        self._life_log.append_event(
            event_type="reflect",
            payload={"summary": summary[:500]},
        )

        self._set_organism_state(OrganismState.IDLE)

    def persist(self, force: bool = False) -> None:
        with self._state_lock:
            # Build a fingerprint of the fields that matter
            fitness = self._genome.fitness if hasattr(self, '_genome') else 0.0
            fingerprint = (
                f"{self._state.get('awareness', 0.0):.4f}|"
                f"{self._state.get('evolution_count', 0)}|"
                f"{self._state.get('interaction_count', 0)}|"
                f"{fitness:.4f}"
            )
            if not force and fingerprint == self._last_persisted_fingerprint:
                return  # nothing meaningful changed — skip write
            self._last_persisted_fingerprint = fingerprint
            self._state["state_version"] = int(self._state.get("state_version", 0)) + 1
            self._state["age_seconds"] = round(self.age_seconds, 2)
            self._state["last_tick_time"] = _utc_now()
            # v3.26: Sync founder state to state dict before serialization
            self._state["global_cycle"] = self._global_cycle
            self._state["founder_revival_count"] = self._founder_revival_count
            self._state["founder_recovery_mode"] = self._founder_recovery_mode
            self._state["founder_revival_grace_remaining"] = self._founder_revival_grace_remaining
            self._state["founder_forced_mutate_streak"] = self._founder_forced_mutate_streak
            self._state["founder_mutate_cooldown"] = self._founder_mutate_cooldown
            self._state["founder_cycles_since_revival"] = self._founder_cycles_since_revival
            # v3.29: Sync survival counters & environment to state dict before serialization
            self._state["below_fitness_cycles"] = {k: int(v) for k, v in self._below_fitness_cycles.items()}
            self._state["last_birth_cycle"] = {k: int(v) for k, v in self._last_birth_cycle.items()}
            self._state["conservation_mode"] = {k: bool(v) for k, v in self._conservation_mode.items()}
            # v3.31: Persist trait collapse emergency countdown
            self._state["trait_collapse_emergency_remaining"] = self._trait_collapse_emergency_remaining
            self._state["environment_state"] = self._environment.to_dict()
            state_snapshot = self._serialize_state(self._state)
        self._memory_manager.save_state(state_snapshot)
        self._logger.info(
            "[PERSIST] state_version=%d evolution_count=%d awareness=%.4f",
            state_snapshot["state_version"],
            state_snapshot["evolution_count"],
            state_snapshot["awareness"],
        )

        # Life-log event for the Observe→Update→Persist→Reflect cycle
        self._life_log.append_event(
            event_type="persist",
            payload={
                "state_version": state_snapshot["state_version"],
                "evolution_count": state_snapshot["evolution_count"],
                "awareness": state_snapshot["awareness"],
                "interaction_count": state_snapshot.get("interaction_count", 0),
                "age_seconds": state_snapshot.get("age_seconds", 0),
            },
        )

        # Snapshot checkpoint if on boundary
        if self._life_log.should_snapshot():
            self._life_log.write_snapshot(state_snapshot)

    def check_disk_usage(self) -> None:
        """v3.22: Periodic disk usage check — warns if storage exceeds threshold."""
        try:
            status = check_disk_usage()
            if status["warning"]:
                self._logger.warning("[DISK] %s", status["message"])
            else:
                self._logger.info("[DISK] %s", status["message"])
        except Exception as exc:
            self._logger.warning("[DISK] Usage check failed: %s", exc)

    def save_tick_snapshot(self, tick: int) -> None:
        """Save a lightweight tick-based snapshot to data/snapshots/.

        Called every ``memory_snapshot_interval`` ticks (v3.21).  These
        periodic snapshots allow historical analysis without bloating
        memory.json.
        """
        snapshot_dir = os.path.join(self._data_dir, "data", "snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)

        with self._state_lock:
            snap = {
                "tick": tick,
                "timestamp": _utc_now(),
                "evolution_count": self._state.get("evolution_count", 0),
                "awareness": float(self._state.get("awareness", 0.0)),
                "fitness": self._genome.fitness,
                "energy": self._genome.traits.get("energy_efficiency", 0.0),
                "population_size": self._population.size,
                "genome": self._genome.to_dict(),
            }

        path = os.path.join(snapshot_dir, f"snap_{tick}.json")
        try:
            # Atomic write
            fd, tmp_path = tempfile.mkstemp(
                dir=snapshot_dir, suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(snap, fh, indent=2, default=str)
                os.replace(tmp_path, path)
            except BaseException:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise
            self._logger.info("[SNAPSHOT] Tick snapshot saved: %s", path)
            # v3.22: Prune old tick snapshots to prevent unbounded growth
            removed = cleanup_tick_snapshots(snapshot_dir)
            if removed:
                self._logger.info("[SNAPSHOT] Cleaned up %d old tick snapshots", removed)
        except Exception as exc:
            self._logger.warning("[SNAPSHOT] Tick snapshot failed: %s", exc)

    def run_loop(self, interval: int = 5, log_cycle: bool = False) -> None:
        safe_interval = max(1, int(interval))

        if self._loop_thread and self._loop_thread.is_alive():
            return

        self._loop_stop_event.clear()
        self._loop_thread = threading.Thread(
            target=self._loop_worker,
            args=(safe_interval, log_cycle),
            daemon=True,
            name="al01-run-loop",
        )
        self._loop_thread.start()

        try:
            while self._loop_thread.is_alive():
                self._loop_thread.join(timeout=0.5)
        except KeyboardInterrupt:
            self.stop_loop(timeout=2.0)

    def stop_loop(self, timeout: Optional[float] = None) -> None:
        self._loop_stop_event.set()
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=timeout)

    @property
    def loop_running(self) -> bool:
        return bool(self._loop_thread and self._loop_thread.is_alive())

    def shutdown(self) -> None:
        """Graceful shutdown: stop loop, persist final state, log exit."""
        self._logger.info("[SHUTDOWN] Shutdown requested")
        self.stop_loop(timeout=3.0)

        # Stop snapshot scheduler and take final snapshot
        if self._snapshot_manager is not None:
            try:
                self._snapshot_manager.take_snapshot(label="shutdown")
            except Exception as exc:
                self._logger.warning("[SHUTDOWN] Final snapshot failed: %s", exc)
            self._snapshot_manager.stop()

        self.persist(force=True)

        # v3.21: Flush buffered memory entries to disk
        self._memory_manager.flush_memory()

        # Log shutdown to life log
        self._life_log.append_event(
            event_type="shutdown",
            payload={
                "uptime_seconds": round(self.uptime_seconds, 2),
                "age_seconds": round(self.age_seconds, 2),
                "evolution_count": self._state.get("evolution_count", 0),
                "state_version": self._state.get("state_version", 0),
            },
        )

        # Confirm write by reloading
        confirm = self._memory_manager.load_state()
        confirmed_version = confirm.get("state_version", "?")
        self._logger.info(
            "[SHUTDOWN] Final state persisted and verified (state_version=%s). Organism offline.",
            confirmed_version,
        )

    def _loop_worker(self, interval: int, log_cycle: bool) -> None:
        cycle_number = 0
        while not self._loop_stop_event.is_set():
            cycle_number += 1
            cycle_started = time.monotonic()
            try:
                should_persist = (cycle_number % self.config.heartbeat_persist_interval == 0)
                self._run_cycle(log_cycle=log_cycle, force_persist=should_persist)
                with self._state_lock:
                    awareness = self._state.get("awareness", 0.0)
                    evolution = self._state.get("evolution_count", 0)
                    version = self._state.get("state_version", 0)
                self._logger.info(
                    "[HEARTBEAT] cycle=%d evolution_count=%d awareness=%.4f state_version=%d",
                    cycle_number,
                    evolution,
                    awareness,
                    version,
                )
            except Exception as exc:
                self._logger.error("[HEARTBEAT] Cycle %d failed (organism still running): %s", cycle_number, exc)

            elapsed = time.monotonic() - cycle_started
            wait_time = max(0.0, float(interval) - elapsed)
            self._loop_stop_event.wait(wait_time)

    def _run_cycle(self, log_cycle: bool, force_persist: bool = True) -> None:
        latest_state = self._memory_manager.load_state()
        with self._state_lock:
            merged = self._normalize_state(latest_state or self._state)
            # Never allow evolution_count to regress
            merged["evolution_count"] = max(
                merged["evolution_count"],
                int(self._state.get("evolution_count", 0)),
            )
            # Preserve state_version continuity
            merged["state_version"] = max(
                merged.get("state_version", 0),
                int(self._state.get("state_version", 0)),
            )
            self._state = merged

        self.tick()

        if log_cycle:
            with self._state_lock:
                awareness = float(self._state.get("awareness", 0.0))
                evolution_count = int(self._state.get("evolution_count", 0))
            self._memory_manager.write_memory(
                {
                    "event_type": "loop_cycle",
                    "payload": {
                        "awareness": awareness,
                        "evolution_count": evolution_count,
                    },
                }
            )

        if force_persist:
            self.persist()

    def _normalize_state(self, state: Any) -> Dict[str, Any]:
        base = state if isinstance(state, dict) else {}
        return {
            "awareness": float(base.get("awareness", 0.0)),
            "evolution_count": int(base.get("evolution_count", 0)),
            "last_boot_utc": base.get("last_boot_utc", None),
            "state_version": int(base.get("state_version", 0)),
            "interaction_count": int(base.get("interaction_count", 0)),
            "birth_time": base.get("birth_time", None),
            "age_seconds": float(base.get("age_seconds", 0.0)),
            "last_tick_time": base.get("last_tick_time", None),
            "uptime_sessions": int(base.get("uptime_sessions", 0)),
            # v2.0
            "genome": base.get("genome", None),
            "stimuli": base.get("stimuli", []),
            "last_reproduce_at": int(base.get("last_reproduce_at", 0)),
            # v2.2 energy
            "energy": float(base.get("energy", 1.0)),
            # v3.0
            "global_cycle": int(base.get("global_cycle", 0)),
            # v3.26: Founder state persistence
            "founder_revival_count": int(base.get("founder_revival_count", 0)),
            "founder_recovery_mode": bool(base.get("founder_recovery_mode", False)),
            "founder_revival_grace_remaining": int(base.get("founder_revival_grace_remaining", 0)),
            "founder_forced_mutate_streak": int(base.get("founder_forced_mutate_streak", 0)),
            "founder_mutate_cooldown": int(base.get("founder_mutate_cooldown", 0)),
            "founder_cycles_since_revival": int(base.get("founder_cycles_since_revival", 0)),
            # v3.29: Per-organism survival counters & environment state
            "below_fitness_cycles": base.get("below_fitness_cycles", {}),
            "last_birth_cycle": base.get("last_birth_cycle", {}),
            "conservation_mode": base.get("conservation_mode", {}),
            "environment_state": base.get("environment_state", None),
            # v3.31: Trait collapse emergency countdown
            "trait_collapse_emergency_remaining": int(base.get("trait_collapse_emergency_remaining", 0)),
        }

    def _serialize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "awareness": float(state.get("awareness", 0.0)),
            "evolution_count": int(state.get("evolution_count", 0)),
            "last_boot_utc": state.get("last_boot_utc", None),
            "state_version": int(state.get("state_version", 0)),
            "interaction_count": int(state.get("interaction_count", 0)),
            "birth_time": state.get("birth_time", None),
            "age_seconds": float(state.get("age_seconds", 0.0)),
            "last_tick_time": state.get("last_tick_time", None),
            "uptime_sessions": int(state.get("uptime_sessions", 0)),
            # v2.0
            "genome": state.get("genome"),
            "stimuli": state.get("stimuli", []),
            "last_reproduce_at": int(state.get("last_reproduce_at", 0)),
            # v2.2 energy
            "energy": float(state.get("energy", 1.0)),
            # v3.0
            "global_cycle": int(state.get("global_cycle", 0)),
            # v3.26: Founder state persistence
            "founder_revival_count": int(state.get("founder_revival_count", 0)),
            "founder_recovery_mode": bool(state.get("founder_recovery_mode", False)),
            "founder_revival_grace_remaining": int(state.get("founder_revival_grace_remaining", 0)),
            "founder_forced_mutate_streak": int(state.get("founder_forced_mutate_streak", 0)),
            "founder_mutate_cooldown": int(state.get("founder_mutate_cooldown", 0)),
            "founder_cycles_since_revival": int(state.get("founder_cycles_since_revival", 0)),
            # v3.29: Persist per-organism survival counters & environment state
            "below_fitness_cycles": {k: int(v) for k, v in self._below_fitness_cycles.items()},
            "last_birth_cycle": {k: int(v) for k, v in self._last_birth_cycle.items()},
            "conservation_mode": {k: bool(v) for k, v in self._conservation_mode.items()},
            "environment_state": self._environment.to_dict(),
            # v3.31
            "trait_collapse_emergency_remaining": int(self._trait_collapse_emergency_remaining),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
