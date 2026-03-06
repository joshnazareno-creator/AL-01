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
import os
import random
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
from al01.population import Population
from al01.genesis_vault import GenesisVault
from al01.gpt_bridge import GPTBridge, GPTBridgeConfig
from al01.snapshot_manager import SnapshotConfig, SnapshotManager

VERSION = "3.9"

# ── Founder protection constants ──────────────────────────────────────
FOUNDER_MUTATION_CAP: float = 0.08      # AL-01 mutation rate never exceeds this
FOUNDER_ENERGY_FLOOR: float = 0.25      # AL-01 energy never drops below this
FOUNDER_FITNESS_FLOOR: float = 0.15     # AL-01 won't die unless below this


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
        # Write to disk
        try:
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

        # Autonomous reproduction check
        if self._tick_count % self._config.auto_reproduce_interval == 0:
            self._organism.auto_reproduce_cycle()

        # Child autonomy — evolve all children independently
        if self._tick_count % self._config.child_autonomy_interval == 0:
            self._organism.child_autonomy_cycle()

        # Behavior analysis snapshot
        if self._tick_count % self._config.behavior_analysis_interval == 0:
            self._organism.behavior_analysis_cycle()


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

        # v3.4: Default population cap (overridden by ExperimentConfig when active)
        self._default_max_population: int = 50

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
        self.scheduler.tick()

    @property
    def state(self) -> Dict[str, Any]:
        with self._state_lock:
            return MappingProxyType(deepcopy(self._state))

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
                    self._life_log.append_event(
                        event_type="reproduction",
                        payload={
                            "child_id": child["id"],
                            "parent_evolution": evo,
                            "child_fitness": child["genome"]["fitness"],
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

        # v3.4: If AL-01 is dead in population, revive it with survival energy
        al01_record = self._population.get("AL-01")
        if al01_record and not al01_record.get("alive", True):
            self._logger.warning(
                "[REVIVE] AL-01 was dead — reviving with survival energy"
            )
            self._population.update_member("AL-01", {
                "alive": True,
                "death_info": None,
                "state": "idle",
                "energy": 0.5,
            })
            # v3.6: Reset survival grace counter so we get a fresh grace period
            self._below_fitness_cycles["AL-01"] = 0
            # Also reset the autonomy engine energy so the next decide()
            # doesn't immediately kill it again
            if self._autonomy.energy <= 0.0:
                self._autonomy._energy = 0.5

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

        energy_before = self._autonomy.energy

        record = self._autonomy.decide(
            fitness, awareness, mutation_rate, pending,
            current_traits=current_traits,
            env_modifiers=env_modifiers,
        )
        decision = record["decision"]
        computed_awareness = record["awareness"]
        energy = record.get("energy", 1.0)

        # v3.8: Founder protection — enforce energy floor for AL-01
        if energy < FOUNDER_ENERGY_FLOOR:
            energy = FOUNDER_ENERGY_FLOOR
            self._autonomy._energy = FOUNDER_ENERGY_FLOOR

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
        survival_threshold = 0.2
        founder_threshold = FOUNDER_FITNESS_FLOOR
        survival_grace = 20
        if self._experiment:
            survival_threshold = self._experiment.config.survival_fitness_threshold
            survival_grace = self._experiment.config.survival_grace_cycles
        effective_threshold = founder_threshold  # AL-01 gets founder protection
        if fitness < effective_threshold:
            self._below_fitness_cycles["AL-01"] = self._below_fitness_cycles.get("AL-01", 0) + 1
            if self._below_fitness_cycles["AL-01"] >= survival_grace:
                self._logger.warning(
                    "[SURVIVAL] AL-01 below fitness floor %.4f for %d cycles — triggering death",
                    survival_threshold, self._below_fitness_cycles["AL-01"],
                )
                self._handle_death("AL-01", "fitness_floor")
                record["organism_died"] = True
                record["death_cause"] = "fitness_floor"
                return record
        else:
            self._below_fitness_cycles["AL-01"] = 0

        # Check for death
        if record.get("organism_died"):
            self._handle_death("AL-01", "energy_depleted")
            return record

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
            if not member or not member.get("alive", True):
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
                self._genome = child_genome
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
        """
        record = self._environment.tick()
        if self._experiment:
            self._experiment.record_cycle(self._global_cycle)
        return record

    def auto_reproduce_cycle(self) -> List[Dict[str, Any]]:
        """Check all living organisms for autonomous reproduction eligibility.

        v3.0: reproduction is triggered automatically when fitness stays
        above threshold for N consecutive cycles — no manual intervention.
        v3.4: reproduction_enabled toggle from experiment config.
        """
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

        # Sync cap into population so spawn_child enforces it everywhere
        self._population.max_population = max_pop

        # v3.4: Prune if already over cap (e.g. from pre-existing population)
        if self._population.size > max_pop:
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
                self._life_log.append_event(
                    event_type="auto_reproduction",
                    payload={
                        "parent_id": oid,
                        "child_id": child_id,
                        "cycle": self._global_cycle,
                        "child_fitness": child.get("genome", {}).get("fitness", 0),
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
        if self._population.size > max_pop:
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

        # v3.9: Elite protection — top organisms are shielded from mutation
        elite_set = set(self._population.elite_ids())

        for oid in self._population.member_ids:
            if oid == "AL-01":
                continue  # parent is handled by autonomy_cycle()

            member = self._population.get(oid)
            if not member or not member.get("alive", True):
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
            child_traits = child_genome.traits

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
            energy -= cfg.energy_decay_per_cycle
            cost_mult = env_modifiers.get("mutation_cost_multiplier", 1.0)

            # v3.3: Survival grace cycles — fitness floor death for children
            if fitness < survival_threshold:
                self._below_fitness_cycles[oid] = self._below_fitness_cycles.get(oid, 0) + 1
                if self._below_fitness_cycles[oid] >= survival_grace:
                    self._logger.warning(
                        "[SURVIVAL] %s below fitness floor %.4f for %d cycles — death",
                        oid, survival_threshold, self._below_fitness_cycles[oid],
                    )
                    self._handle_death(oid, "fitness_floor")
                    results.append({
                        "organism_id": oid, "decision": "death",
                        "cause": "fitness_floor", "fitness": fitness,
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

            # v3.9: Elite Protection — elites skip mutation, get crossover instead
            is_elite = oid in elite_set

            if is_elite and fitness < eff_threshold:
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
                    weakest = min(traits, key=traits.get)
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
            energy = max(cfg.energy_min, min(1.0, energy))
            if energy <= 0.0:
                self._logger.warning(
                    "[CHILD-ENERGY] %s energy depleted — death", oid,
                )
                self._handle_death(oid, "energy_depleted")
                results.append({
                    "organism_id": oid, "decision": "death",
                    "cause": "energy_depleted", "fitness": fitness,
                })
                continue

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
        """Process organism death: remove from population, log, track."""
        death_info = self._population.remove_member(organism_id, cause)
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
                "[DEATH] %s died: %s at cycle %d",
                organism_id, cause, self._global_cycle,
            )

            # v3.2: Check for extinction — reseed from vault if needed
            self.check_extinction_reseed()

    def check_extinction_reseed(self) -> Optional[Dict[str, Any]]:
        """If population is extinct, reseed from the Genesis Vault.

        Returns the reseed record if reseeding happened, or None.
        """
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

        # v3.4: Ensure AL-01 is alive and has energy on boot
        al01 = self._population.get("AL-01")
        if al01:
            revive_fields: Dict[str, Any] = {}
            if not al01.get("alive", True):
                self._logger.warning("[BOOT] AL-01 was dead — reviving on boot")
                revive_fields["alive"] = True
                revive_fields["death_info"] = None
                revive_fields["state"] = "idle"
            if al01.get("energy", 0.0) <= 0.0:
                self._logger.warning("[BOOT] AL-01 energy=0 — restoring to 0.5")
                revive_fields["energy"] = 0.5
            if revive_fields:
                self._population.update_member("AL-01", revive_fields)

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

    def reflect(self) -> None:
        self._set_organism_state(OrganismState.REFLECTING)
        recent = self._memory_manager.read_memory(limit=5)
        if not recent:
            return

        fragments = []
        for entry in recent:
            event_type = entry.get("event_type", "event")
            payload = entry.get("payload", {})
            fragments.append(f"{event_type}:{json.dumps(payload, sort_keys=True)}")
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
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
