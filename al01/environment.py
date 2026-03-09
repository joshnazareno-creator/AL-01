"""AL-01 v3.0 — True Environment Model.

A bounded resource ecosystem with fluctuating environmental variables,
periodic scarcity events, and resource pools that constrain all organisms.

Environmental Variables
-----------------------
* **temperature** — drifts cyclically + randomly; affects mutation cost
  and energy regeneration.
* **entropy_pressure** — ambient chaos; accelerates trait decay and
  increases mutation spread.
* **resource_abundance** — available energy in the environment; constrains
  how much energy organisms can harvest per cycle.
* **noise_level** — information noise; reduces perception effectiveness
  and lowers fitness accuracy.

Resource Pool
-------------
Finite energy pool shared by all organisms.  Each cycle, organisms draw
from the pool (capped by ``resource_abundance``).  The pool regenerates
slowly.  Scarcity events can slash the pool temporarily.

Scarcity Events
---------------
Random catastrophic drops in resource availability.  Configurable
probability per cycle, severity, and duration.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("al01.environment")


# ── Configuration ────────────────────────────────────────────────────────

@dataclass
class EnvironmentConfig:
    """Tuneable knobs for the environment model."""

    # Resource pool
    resource_pool_max: float = 1000.0
    """Maximum energy available in the environment."""
    resource_pool_initial: float = 1000.0
    """Starting resource pool."""
    resource_regen_rate: float = 5.0
    """How much energy regenerates per cycle (before abundance scaling)."""

    # v3.18: Resource pool minimum floor
    resource_pool_min_floor: float = 0.0
    """Pool never drops below this value — ensures baseline regeneration.
    Recommended production value: 50.0."""

    # v3.13: Global resource pool — fair distribution & smart regen
    metabolic_base_cost: float = 1.0
    """Base energy a single organism consumes per cycle from the global pool."""
    scarcity_threshold: float = 0.25
    """Pool fraction (current/max) below which scarcity pressure activates."""
    scarcity_metabolic_multiplier: float = 1.5
    """Metabolic cost multiplier during mild scarcity."""
    adaptive_metabolism_threshold: float = 0.6
    """Scarcity severity above which adaptive metabolism kicks in (reduces cost)."""
    adaptive_metabolism_floor: float = 0.4
    """Minimum metabolic cost multiplier during extreme scarcity (adaptive metabolism)."""
    scarcity_reproduction_penalty: float = 0.5
    """Reproduction fitness threshold is multiplied by (1 + this) during scarcity."""
    scarcity_mortality_multiplier: float = 1.5
    """Survival grace cycles are divided by this during scarcity."""
    regen_efficiency_weight: float = 0.5
    """How much average organism efficiency influences regeneration (0–1)."""
    regen_overuse_damping: float = 0.5
    """Regen rate is reduced by this factor × overuse_ratio when pool < 50%."""

    # v3.15: Emergency regeneration — auto-regen when pool drops below threshold
    emergency_regen_threshold: float = 0.25
    """Pool fraction below which emergency regeneration activates (0–1)."""
    emergency_regen_population_scale: float = 1.0
    """Per-surviving-organism energy added during emergency regen."""

    # v3.16: Dormant wake threshold
    dormant_wake_pool_fraction: float = 0.50
    """Pool fraction above which dormant organisms are eligible to wake."""

    # v3.16: Stability reproduction
    stability_repro_pool_fraction: float = 0.80
    """Pool fraction above which stability-based reproduction is allowed."""
    stability_repro_probability: float = 0.05
    """Per-cycle probability of stability reproduction per eligible organism."""

    # v3.16: Resource-based carrying capacity
    carrying_capacity_scaling: float = 0.5
    """Scaling factor: max_pop = pool / metabolic_cost * scaling."""
    carrying_capacity_min: int = 5
    """Minimum carrying capacity regardless of resource level."""

    # v3.16: Lone survivor reproduction
    lone_survivor_pool_fraction: float = 0.70
    """Pool fraction above which lone survivor auto-reproduction activates."""
    lone_survivor_repro_probability: float = 0.10
    """Per-cycle probability of lone survivor triggering reproduction."""

    # v3.18: Reproduction energy gates
    repro_energy_min_auto: float = 0.50
    """Minimum parent energy required for auto-reproduction."""
    repro_energy_min_stability: float = 0.60
    """Minimum parent energy required for stability reproduction."""
    repro_energy_min_lone_survivor: float = 0.40
    """Minimum parent energy required for lone-survivor reproduction."""
    repro_energy_cost: float = 0.20
    """Energy deducted from parent after successful reproduction (auto/stability/lone)."""

    # Environmental variables — starting values and drift
    temperature_initial: float = 0.5
    temperature_drift: float = 0.02
    temperature_cycle_period: int = 100
    """Sinusoidal temperature cycle length in ticks."""
    temperature_min: float = 0.0
    temperature_max: float = 1.0

    entropy_pressure_initial: float = 0.3
    entropy_pressure_drift: float = 0.015
    entropy_pressure_min: float = 0.0
    entropy_pressure_max: float = 1.0

    resource_abundance_initial: float = 0.8
    resource_abundance_drift: float = 0.02
    resource_abundance_min: float = 0.1
    resource_abundance_max: float = 1.0

    noise_level_initial: float = 0.2
    noise_level_drift: float = 0.01
    noise_level_min: float = 0.0
    noise_level_max: float = 1.0

    # Scarcity events
    scarcity_probability: float = 0.02
    """Probability of a scarcity event per cycle."""
    scarcity_severity_min: float = 0.3
    scarcity_severity_max: float = 0.7
    """Severity as fraction of resource pool destroyed."""
    scarcity_duration_min: int = 5
    scarcity_duration_max: int = 20
    """How many cycles the scarcity depresses abundance."""

    # Environment shift — periodic major shifts
    shift_interval: int = 200
    """Legacy fixed interval (used as fallback)."""
    shift_magnitude: float = 0.15
    """How much variables shift during a major shift."""

    # v3.9: Dynamic shift interval — randomised between min/max each time
    dynamic_shift_min: int = 20
    """Minimum cycles between major environment shifts."""
    dynamic_shift_max: int = 50
    """Maximum cycles between major environment shifts."""

    # Modulation effects
    temperature_mutation_cost_scale: float = 0.5
    """Higher temperature → higher mutation cost multiplier."""
    temperature_energy_regen_scale: float = 0.3
    """Moderate temperature → best energy regen; extremes reduce it."""
    entropy_pressure_decay_scale: float = 2.0
    """Higher entropy_pressure → multiplier on trait decay rate."""
    noise_fitness_penalty: float = 0.15
    """Maximum fitness penalty from noise."""

    # v3.11: Rare shock events — low-probability catastrophic events
    # that temporarily favour resilience.  Probability is per *100*
    # cycles (checked each tick: effective_per_tick = shock_probability / 100).
    shock_probability: float = 0.02
    """Probability of a shock event per 100 cycles (1–3% recommended)."""
    shock_resilience_bonus: float = 0.15
    """Temporary additive fitness bonus for resilient organisms during shock."""
    shock_duration_min: int = 5
    """Minimum shock duration in cycles."""
    shock_duration_max: int = 15
    """Maximum shock duration in cycles."""
    shock_entropy_spike: float = 0.25
    """Entropy pressure added during a shock event."""

    # v3.23: Population-scaled regeneration
    population_regen_bonus: float = 0.5
    """Extra regen per surviving organism — prevents extinction cascades."""

    # v3.23: Extinction prevention guard
    extinction_prevention_regen_multiplier: float = 3.0
    """Regen multiplier when population = 1 (last survivor)."""
    extinction_prevention_pool_boost: float = 100.0
    """Flat pool energy injection when population = 1."""

    # v3.23: Conservation mode
    conservation_metabolic_fraction: float = 0.3
    """Organisms in conservation mode use this fraction of normal metabolic cost."""

    # Seeded randomness
    rng_seed: Optional[int] = None

    # v3.11: Named environmental events — rare global events
    named_event_probability: float = 0.02
    """Probability of a named event per 100 cycles."""
    named_event_duration_min: int = 10
    """Minimum duration of named events in cycles."""
    named_event_duration_max: int = 30
    """Maximum duration of named events in cycles."""


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ── Scarcity Event ───────────────────────────────────────────────────────

@dataclass
class ScarcityEvent:
    """Active scarcity event."""
    severity: float  # 0–1, fraction of pool lost
    remaining_cycles: int
    started_at: str  # ISO timestamp
    abundance_reduction: float  # how much abundance is depressed

    def tick(self) -> bool:
        """Advance one cycle.  Returns True if event is still active."""
        self.remaining_cycles -= 1
        return self.remaining_cycles > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": round(self.severity, 4),
            "remaining_cycles": self.remaining_cycles,
            "started_at": self.started_at,
            "abundance_reduction": round(self.abundance_reduction, 4),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScarcityEvent":
        return cls(
            severity=d["severity"],
            remaining_cycles=d["remaining_cycles"],
            started_at=d["started_at"],
            abundance_reduction=d["abundance_reduction"],
        )


# ── Shock Event ──────────────────────────────────────────────────────────

@dataclass
class ShockEvent:
    """v3.11: Rare catastrophic shock that temporarily favours resilience.

    While active the environment's entropy_pressure is spiked and
    organisms classified as 'resilient' receive a fitness bonus.
    """
    remaining_cycles: int
    entropy_spike: float
    resilience_bonus: float
    started_at: str  # ISO timestamp

    def tick(self) -> bool:
        """Advance one cycle.  Returns True if still active."""
        self.remaining_cycles -= 1
        return self.remaining_cycles > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "remaining_cycles": self.remaining_cycles,
            "entropy_spike": round(self.entropy_spike, 4),
            "resilience_bonus": round(self.resilience_bonus, 4),
            "started_at": self.started_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ShockEvent":
        return cls(
            remaining_cycles=d["remaining_cycles"],
            entropy_spike=d.get("entropy_spike", 0.25),
            resilience_bonus=d.get("resilience_bonus", 0.15),
            started_at=d.get("started_at", ""),
        )


# ── Named Environmental Events (v3.11) ──────────────────────────────────

# Catalogue of possible named events and their effects
NAMED_EVENT_CATALOGUE: List[Dict[str, Any]] = [
    {
        "name": "heat_wave",
        "description": "Extreme temperatures favour resilient organisms",
        "effects": {"temperature_boost": 0.25, "resilience_bonus": 0.10},
    },
    {
        "name": "resource_boom",
        "description": "Abundant resources — reproduction becomes easier",
        "effects": {"pool_regen_multiplier": 2.0, "abundance_boost": 0.20},
    },
    {
        "name": "scarcity_drought",
        "description": "Severe resource scarcity drains energy faster",
        "effects": {"pool_drain_fraction": 0.30, "metabolic_multiplier": 1.5},
    },
    {
        "name": "mutation_storm",
        "description": "Cosmic radiation doubles mutation rate",
        "effects": {"mutation_rate_multiplier": 2.0, "entropy_boost": 0.15},
    },
]


@dataclass
class NamedEvent:
    """A named, globally-visible environmental event with specific effects."""
    name: str
    description: str
    effects: Dict[str, float]
    remaining_cycles: int
    started_at: str

    def tick(self) -> bool:
        """Advance one cycle.  Returns True if still active."""
        self.remaining_cycles -= 1
        return self.remaining_cycles > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "effects": {k: round(v, 4) for k, v in self.effects.items()},
            "remaining_cycles": self.remaining_cycles,
            "started_at": self.started_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NamedEvent":
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            effects=d.get("effects", {}),
            remaining_cycles=d.get("remaining_cycles", 0),
            started_at=d.get("started_at", ""),
        )


# ── Environment Model ────────────────────────────────────────────────────

class Environment:
    """True environmental model with resource pool and fluctuating variables.

    Call ``tick()`` every cycle to advance the environment state.
    """

    def __init__(
        self,
        config: Optional[EnvironmentConfig] = None,
        rng_seed: Optional[int] = None,
    ) -> None:
        self._config = config or EnvironmentConfig()
        seed = rng_seed if rng_seed is not None else self._config.rng_seed
        self._rng = random.Random(seed)
        self._cycle: int = 0

        # Resource pool
        self._resource_pool: float = self._config.resource_pool_initial

        # Environmental variables
        self._temperature: float = self._config.temperature_initial
        self._entropy_pressure: float = self._config.entropy_pressure_initial
        self._resource_abundance: float = self._config.resource_abundance_initial
        self._noise_level: float = self._config.noise_level_initial

        # Active scarcity events
        self._scarcity_events: List[ScarcityEvent] = []

        # v3.11: Active shock events
        self._shock_events: List[ShockEvent] = []
        self._shock_log: List[Dict[str, Any]] = []

        # v3.11: Named environmental events
        self._named_events: List[NamedEvent] = []
        self._named_event_log: List[Dict[str, Any]] = []

        # v3.9: Dynamic shift scheduling — next shift at a random cycle
        self._next_shift_cycle: int = self._rng.randint(
            self._config.dynamic_shift_min,
            self._config.dynamic_shift_max,
        )

        # History for tracking
        self._shift_log: List[Dict[str, Any]] = []
        self._scarcity_log: List[Dict[str, Any]] = []

    # ── Properties ───────────────────────────────────────────────────

    @property
    def config(self) -> EnvironmentConfig:
        return self._config

    @property
    def cycle(self) -> int:
        return self._cycle

    @property
    def resource_pool(self) -> float:
        return self._resource_pool

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def entropy_pressure(self) -> float:
        return self._entropy_pressure

    @property
    def resource_abundance(self) -> float:
        return self._resource_abundance

    @property
    def noise_level(self) -> float:
        return self._noise_level

    @property
    def active_scarcity_events(self) -> List[ScarcityEvent]:
        return list(self._scarcity_events)

    @property
    def is_scarcity_active(self) -> bool:
        return len(self._scarcity_events) > 0

    @property
    def shock_events(self) -> List[ShockEvent]:
        """v3.11: Active shock events."""
        return list(self._shock_events)

    @property
    def is_shock_active(self) -> bool:
        """v3.11: True if at least one shock is in progress."""
        return len(self._shock_events) > 0

    @property
    def shock_resilience_bonus(self) -> float:
        """v3.11: Total resilience fitness bonus from all active shocks."""
        return sum(e.resilience_bonus for e in self._shock_events)

    @property
    def named_events(self) -> List[NamedEvent]:
        """v3.11: Active named environmental events."""
        return list(self._named_events)

    @property
    def active_named_event_names(self) -> List[str]:
        """v3.11: Names of currently active named events."""
        return [e.name for e in self._named_events]

    @property
    def named_event_log(self) -> List[Dict[str, Any]]:
        """v3.11: Full history of named events."""
        return list(self._named_event_log)

    @property
    def mutation_rate_multiplier(self) -> float:
        """v3.11: Combined mutation rate multiplier from named events."""
        m = 1.0
        for e in self._named_events:
            m *= e.effects.get("mutation_rate_multiplier", 1.0)
        return m

    @property
    def metabolic_multiplier(self) -> float:
        """v3.11: Combined metabolic cost multiplier from named events."""
        m = 1.0
        for e in self._named_events:
            m *= e.effects.get("metabolic_multiplier", 1.0)
        return m

    # ── Core tick ────────────────────────────────────────────────────

    def tick(self) -> Dict[str, Any]:
        """Advance one cycle.  Drifts variables, regenerates resources,
        checks for scarcity events and periodic shifts.

        Returns a record of what happened this cycle.
        """
        self._cycle += 1
        record: Dict[str, Any] = {"cycle": self._cycle, "events": []}

        # 1. Drift environmental variables
        self._drift_variables()

        # 2. Dynamic environment shift (v3.9: randomised interval 20–50 cycles)
        if self._cycle >= self._next_shift_cycle:
            shift_record = self._apply_shift()
            record["events"].append(shift_record)
            # Schedule the NEXT shift at a random future cycle
            self._next_shift_cycle = self._cycle + self._rng.randint(
                self._config.dynamic_shift_min,
                self._config.dynamic_shift_max,
            )

        # 3. Check for new scarcity event
        if self._rng.random() < self._config.scarcity_probability:
            scarcity_record = self._trigger_scarcity()
            record["events"].append(scarcity_record)

        # 3b. v3.11: Check for rare shock event (per-tick probability)
        shock_per_tick = self._config.shock_probability / 100.0
        if shock_per_tick > 0 and self._rng.random() < shock_per_tick:
            shock_record = self._trigger_shock()
            record["events"].append(shock_record)

        # 4. Tick active scarcity events
        self._tick_scarcity_events()

        # 4b. v3.11: Tick active shock events
        self._tick_shock_events()

        # 4c. v3.11: Check for named environmental event
        named_per_tick = self._config.named_event_probability / 100.0
        if named_per_tick > 0 and self._rng.random() < named_per_tick:
            named_record = self._trigger_named_event()
            if named_record:
                record["events"].append(named_record)

        # 4d. v3.11: Tick active named events and apply ongoing effects
        self._tick_named_events()

        # 5. Compute effective abundance (reduced during scarcity)
        effective_abundance = self._effective_abundance()

        # 6. Regenerate resource pool (v3.13: uses smart_regenerate)
        # The basic regen here is a fallback; the organism calls
        # smart_regenerate() with population efficiency data for the
        # full v3.13 logic.  tick() still does a baseline regen so
        # standalone Environment usage remains functional.
        regen = self._config.resource_regen_rate * effective_abundance
        self._resource_pool = min(
            self._config.resource_pool_max,
            self._resource_pool + regen,
        )

        # v3.18: Enforce minimum pool floor
        self._resource_pool = max(self._config.resource_pool_min_floor, self._resource_pool)

        record.update(self.state_snapshot())
        return record

    def consume_resources(self, amount: float) -> float:
        """Attempt to consume *amount* from the resource pool.

        Returns the actual amount consumed (may be less if pool is low).
        v3.18: Pool cannot drop below ``resource_pool_min_floor``.
        """
        floor = self._config.resource_pool_min_floor
        available = max(0.0, self._resource_pool - floor)
        actual = min(amount, available)
        self._resource_pool = max(floor, self._resource_pool - actual)
        return actual

    # ── v3.13: Global Resource Pool — fair distribution ──────────────

    def request_energy(self, energy_requested: float, population_size: int) -> float:
        """Fair energy allocation from the global pool.

        Each organism may draw at most ``current_energy / population_size``
        to prevent any single organism from draining the pool.

        Returns the actual energy granted (≤ *energy_requested*).
        v3.18: Pool cannot drop below ``resource_pool_min_floor``.
        """
        floor = self._config.resource_pool_min_floor
        available = max(0.0, self._resource_pool - floor)
        pop = max(1, population_size)
        fair_share = available / pop
        granted = min(energy_requested, fair_share)
        granted = max(0.0, min(granted, available))
        self._resource_pool = max(floor, self._resource_pool - granted)
        return granted

    @property
    def pool_fraction(self) -> float:
        """Current pool as fraction of capacity (0–1)."""
        if self._config.resource_pool_max <= 0:
            return 0.0
        return self._resource_pool / self._config.resource_pool_max

    @property
    def is_scarcity_pressure(self) -> bool:
        """True when the pool is below the scarcity threshold."""
        return self.pool_fraction < self._config.scarcity_threshold

    @property
    def scarcity_severity(self) -> float:
        """0.0 when pool is at or above threshold, 1.0 when pool is empty.

        Linear interpolation between threshold and 0.
        """
        frac = self.pool_fraction
        thresh = self._config.scarcity_threshold
        if frac >= thresh:
            return 0.0
        if thresh <= 0:
            return 1.0
        return 1.0 - (frac / thresh)

    def effective_metabolic_cost(self) -> float:
        """Per-organism metabolic cost with adaptive metabolism.

        v3.15 Adaptive Metabolism:
        - Mild scarcity (severity < adaptive_metabolism_threshold): cost INCREASES
          (original v3.14 behaviour — organisms under pressure).
        - High scarcity (severity >= adaptive_metabolism_threshold): cost DECREASES
          toward ``adaptive_metabolism_floor`` (organisms enter conservation mode
          to survive extreme conditions).
        """
        base = self._config.metabolic_base_cost
        if self.is_scarcity_pressure:
            severity = self.scarcity_severity
            threshold = self._config.adaptive_metabolism_threshold
            if severity < threshold:
                # Mild scarcity — cost increases linearly
                mult = 1.0 + severity * (self._config.scarcity_metabolic_multiplier - 1.0)
                return base * mult
            else:
                # High scarcity — adaptive metabolism kicks in, cost DECREASES
                # At severity == threshold: mult == scarcity_metabolic_multiplier (peak)
                # At severity == 1.0: mult == adaptive_metabolism_floor
                peak = self._config.scarcity_metabolic_multiplier
                floor = self._config.adaptive_metabolism_floor
                t = (severity - threshold) / (1.0 - threshold) if threshold < 1.0 else 0.0
                mult = peak - t * (peak - floor)
                return base * mult
        return base

    def effective_reproduction_threshold(self, base_threshold: float) -> float:
        """Reproduction fitness threshold, raised during scarcity."""
        if self.is_scarcity_pressure:
            severity = self.scarcity_severity
            penalty = severity * self._config.scarcity_reproduction_penalty
            return min(1.0, base_threshold * (1.0 + penalty))
        return base_threshold

    def effective_survival_grace(self, base_grace: int) -> int:
        """Survival grace cycles, reduced during scarcity (faster mortality)."""
        if self.is_scarcity_pressure:
            severity = self.scarcity_severity
            divisor = 1.0 + severity * (self._config.scarcity_mortality_multiplier - 1.0)
            return max(1, int(base_grace / divisor))
        return base_grace

    def resource_carrying_capacity(self) -> int:
        """v3.16: Dynamic carrying capacity based on current resource pool.

        Formula: ``max_pop = pool / metabolic_cost * scaling_factor``
        Clamped to ``[carrying_capacity_min, ABSOLUTE_POPULATION_CAP]``.
        """
        cfg = self._config
        metabolic = max(0.1, cfg.metabolic_base_cost)
        raw = self._resource_pool / metabolic * cfg.carrying_capacity_scaling
        return max(cfg.carrying_capacity_min, min(60, int(raw)))

    def emergency_regenerate(self, population_size: int) -> float:
        """v3.14: Emergency resource regeneration when pool drops below floor.

        Triggered when ``pool_fraction < emergency_regen_threshold``.
        Regen rate = ``resource_regen_rate * (1 + scarcity_severity)``
        scaled by surviving population count.

        Returns the actual energy regenerated (0.0 if not triggered).
        """
        cfg = self._config
        if self.pool_fraction >= cfg.emergency_regen_threshold:
            return 0.0

        pop = max(1, population_size)
        severity = self.scarcity_severity
        regen_rate = cfg.resource_regen_rate * (1.0 + severity)
        regen = regen_rate * pop * cfg.emergency_regen_population_scale

        old_pool = self._resource_pool
        self._resource_pool = min(cfg.resource_pool_max, self._resource_pool + regen)
        actual = self._resource_pool - old_pool
        if actual > 0:
            logger.info(
                "[ENV] Emergency regen: +%.1f energy (pool_fraction=%.3f, severity=%.3f, pop=%d)",
                actual, self.pool_fraction, severity, pop,
            )
        return actual

    def extinction_prevention_regenerate(self, population_size: int) -> float:
        """v3.23: Boost resource regeneration when population reaches 1.

        When only one organism remains, temporarily boosts resource
        availability to prevent permanent extinction.  Applies both a
        regen multiplier and a flat pool injection.

        Returns extra energy regenerated (0.0 if not triggered).
        """
        if population_size > 1:
            return 0.0
        cfg = self._config
        base_regen = cfg.resource_regen_rate
        boost = base_regen * (cfg.extinction_prevention_regen_multiplier - 1.0)
        boost += cfg.extinction_prevention_pool_boost
        old_pool = self._resource_pool
        self._resource_pool = min(cfg.resource_pool_max, self._resource_pool + boost)
        actual = self._resource_pool - old_pool
        if actual > 0:
            logger.info(
                "[ENV] Extinction prevention: +%.1f energy (pop=%d, pool=%.1f)",
                actual, population_size, self._resource_pool,
            )
        return actual

    def smart_regenerate(self, avg_efficiency: float = 0.5,
                         population_size: int = 1) -> float:
        """v3.13: Regenerate pool with overuse damping & efficiency scaling.

        *avg_efficiency* should be the mean efficiency ratio across the
        population (0–1).  Higher efficiency → faster regeneration.

        v3.23: *population_size* adds per-organism bonus to prevent
        extinction cascades — larger populations regenerate slightly more.

        Returns the actual energy regenerated this tick.
        """
        cfg = self._config
        base_regen = cfg.resource_regen_rate * self._effective_abundance()

        # Efficiency bonus: regen scales partly with population efficiency
        eff_factor = 1.0 + cfg.regen_efficiency_weight * (avg_efficiency - 0.5)
        eff_factor = max(0.3, min(1.5, eff_factor))  # clamp

        # Overuse damping: if pool is below 50%, reduce regen proportionally
        overuse_ratio = 0.0
        if self.pool_fraction < 0.5:
            overuse_ratio = 1.0 - (self.pool_fraction / 0.5)
        damping = 1.0 - cfg.regen_overuse_damping * overuse_ratio
        damping = max(0.2, damping)

        regen = base_regen * eff_factor * damping

        # v3.23: Population-scaled bonus — prevents extinction cascades
        pop_bonus = max(1, population_size) * cfg.population_regen_bonus
        regen += pop_bonus

        old_pool = self._resource_pool
        self._resource_pool = min(cfg.resource_pool_max, self._resource_pool + regen)
        return self._resource_pool - old_pool

    # ── Modulation API ───────────────────────────────────────────────

    def mutation_cost_multiplier(self) -> float:
        """Higher temperature → mutations cost more energy.

        Returns a multiplier ≥ 1.0.
        """
        scale = self._config.temperature_mutation_cost_scale
        return 1.0 + self._temperature * scale

    def energy_regen_rate(self) -> float:
        """Moderate temperature gives best regen; extremes reduce it.

        Returns a multiplier 0–1 applied to base energy regeneration.
        Also scales by resource abundance.
        """
        # Bell curve: optimal at temperature = 0.5
        temp_factor = 1.0 - 2.0 * abs(self._temperature - 0.5)
        temp_factor = max(0.1, temp_factor)

        # Scale by effective abundance
        eff_abundance = self._effective_abundance()
        return temp_factor * eff_abundance

    def trait_decay_multiplier(self) -> float:
        """Higher entropy_pressure → faster trait decay.

        Returns a multiplier ≥ 1.0.
        """
        return 1.0 + self._entropy_pressure * self._config.entropy_pressure_decay_scale

    def fitness_noise_penalty(self) -> float:
        """Noise reduces fitness accuracy.

        Returns a penalty 0 – ``noise_fitness_penalty``.
        """
        return self._noise_level * self._config.noise_fitness_penalty

    def survival_modifier(self) -> float:
        """Overall survival difficulty.  Lower → harder to survive.

        Combines resource abundance, temperature extremes, and entropy.
        Returns 0–1.
        """
        abundance_factor = self._effective_abundance()
        temp_factor = 1.0 - abs(self._temperature - 0.5) * 0.5  # 0.75–1.0
        entropy_factor = 1.0 - self._entropy_pressure * 0.3     # 0.7–1.0
        return _clamp(abundance_factor * temp_factor * entropy_factor, 0.05, 1.0)

    def env_trait_weight_modifiers(self) -> Dict[str, float]:
        """Environment-derived trait weight adjustments.

        Returns additive modifiers to the base trait weights based on
        current environmental conditions.
        """
        mods: Dict[str, float] = {}
        # High temp → resilience more important
        mods["resilience"] = self._temperature * 0.3
        # High entropy → adaptability more important
        mods["adaptability"] = self._entropy_pressure * 0.4
        # High noise → perception more important
        mods["perception"] = self._noise_level * 0.5
        # Low abundance → energy_efficiency more important
        mods["energy_efficiency"] = (1.0 - self._effective_abundance()) * 0.4
        # Stable environment → creativity more important
        stability = 1.0 - (self._entropy_pressure + self._noise_level) / 2.0
        mods["creativity"] = stability * 0.2
        return mods

    # ── State ────────────────────────────────────────────────────────

    def state_snapshot(self) -> Dict[str, Any]:
        """Current environment state as a dict."""
        return {
            "cycle": self._cycle,
            "resource_pool": round(self._resource_pool, 4),
            "temperature": round(self._temperature, 6),
            "entropy_pressure": round(self._entropy_pressure, 6),
            "resource_abundance": round(self._resource_abundance, 6),
            "noise_level": round(self._noise_level, 6),
            "effective_abundance": round(self._effective_abundance(), 6),
            "mutation_cost_multiplier": round(self.mutation_cost_multiplier(), 4),
            "energy_regen_rate": round(self.energy_regen_rate(), 4),
            "trait_decay_multiplier": round(self.trait_decay_multiplier(), 4),
            "fitness_noise_penalty": round(self.fitness_noise_penalty(), 6),
            "survival_modifier": round(self.survival_modifier(), 4),
            "active_scarcity_count": len(self._scarcity_events),
            "scarcity_events": [e.to_dict() for e in self._scarcity_events],
            "next_shift_cycle": self._next_shift_cycle,
            "active_shock_count": len(self._shock_events),
            "shock_events": [e.to_dict() for e in self._shock_events],
            "shock_resilience_bonus": round(self.shock_resilience_bonus, 4),
            "pool_fraction": round(self.pool_fraction, 4),
            "is_scarcity_pressure": self.is_scarcity_pressure,
            "scarcity_severity": round(self.scarcity_severity, 4),
            "effective_metabolic_cost": round(self.effective_metabolic_cost(), 4),
            "emergency_regen_active": self.pool_fraction < self._config.emergency_regen_threshold,
            "emergency_regen_threshold": self._config.emergency_regen_threshold,
            "adaptive_metabolism_active": (
                self.is_scarcity_pressure
                and self.scarcity_severity >= self._config.adaptive_metabolism_threshold
            ),
            "resource_carrying_capacity": self.resource_carrying_capacity(),
            "active_named_events": [e.to_dict() for e in self._named_events],
            "named_event_names": self.active_named_event_names,
            "mutation_rate_multiplier": round(self.mutation_rate_multiplier, 4),
            "metabolic_multiplier": round(self.metabolic_multiplier, 4),
        }

    def state_hash(self) -> str:
        """SHA-256 of current environment state for reproducibility."""
        snap = self.state_snapshot()
        raw = json.dumps(snap, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Full serialization for persistence."""
        return {
            "cycle": self._cycle,
            "resource_pool": self._resource_pool,
            "temperature": self._temperature,
            "entropy_pressure": self._entropy_pressure,
            "resource_abundance": self._resource_abundance,
            "noise_level": self._noise_level,
            "scarcity_events": [e.to_dict() for e in self._scarcity_events],
            "shift_log": self._shift_log[-50:],  # keep last 50
            "scarcity_log": self._scarcity_log[-50:],
            "next_shift_cycle": self._next_shift_cycle,
            "shock_events": [e.to_dict() for e in self._shock_events],
            "shock_log": self._shock_log[-50:],
            "named_events": [e.to_dict() for e in self._named_events],
            "named_event_log": self._named_event_log[-50:],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], config: Optional[EnvironmentConfig] = None) -> "Environment":
        """Reconstruct from serialized dict."""
        env = cls(config=config)
        env._cycle = int(data.get("cycle", 0))
        env._resource_pool = float(data.get("resource_pool", env._config.resource_pool_initial))
        env._temperature = float(data.get("temperature", env._config.temperature_initial))
        env._entropy_pressure = float(data.get("entropy_pressure", env._config.entropy_pressure_initial))
        env._resource_abundance = float(data.get("resource_abundance", env._config.resource_abundance_initial))
        env._noise_level = float(data.get("noise_level", env._config.noise_level_initial))
        env._scarcity_events = [
            ScarcityEvent.from_dict(e) for e in data.get("scarcity_events", [])
        ]
        env._shift_log = data.get("shift_log", [])
        env._scarcity_log = data.get("scarcity_log", [])
        # v3.9: Restore dynamic shift schedule
        if "next_shift_cycle" in data:
            env._next_shift_cycle = int(data["next_shift_cycle"])
        # v3.11: Restore shock events
        env._shock_events = [
            ShockEvent.from_dict(e) for e in data.get("shock_events", [])
        ]
        env._shock_log = data.get("shock_log", [])
        # v3.11: Restore named events
        env._named_events = [
            NamedEvent.from_dict(e) for e in data.get("named_events", [])
        ]
        env._named_event_log = data.get("named_event_log", [])
        return env

    # ── Internal ─────────────────────────────────────────────────────

    def _drift_variables(self) -> None:
        """Apply cyclic + random drift to all environmental variables."""
        cfg = self._config

        # Temperature: sinusoidal base + random perturbation
        if cfg.temperature_cycle_period > 0:
            cyclic = 0.5 + 0.3 * math.sin(
                2.0 * math.pi * self._cycle / cfg.temperature_cycle_period
            )
            noise = self._rng.uniform(-cfg.temperature_drift, cfg.temperature_drift)
            self._temperature = _clamp(
                cyclic + noise,
                cfg.temperature_min,
                cfg.temperature_max,
            )
        else:
            self._temperature += self._rng.uniform(-cfg.temperature_drift, cfg.temperature_drift)
            self._temperature = _clamp(self._temperature, cfg.temperature_min, cfg.temperature_max)

        # Entropy pressure: random walk
        self._entropy_pressure += self._rng.uniform(
            -cfg.entropy_pressure_drift, cfg.entropy_pressure_drift
        )
        self._entropy_pressure = _clamp(
            self._entropy_pressure,
            cfg.entropy_pressure_min,
            cfg.entropy_pressure_max,
        )

        # Resource abundance: random walk
        self._resource_abundance += self._rng.uniform(
            -cfg.resource_abundance_drift, cfg.resource_abundance_drift
        )
        self._resource_abundance = _clamp(
            self._resource_abundance,
            cfg.resource_abundance_min,
            cfg.resource_abundance_max,
        )

        # Noise level: random walk
        self._noise_level += self._rng.uniform(
            -cfg.noise_level_drift, cfg.noise_level_drift
        )
        self._noise_level = _clamp(
            self._noise_level,
            cfg.noise_level_min,
            cfg.noise_level_max,
        )

    def _apply_shift(self) -> Dict[str, Any]:
        """Major environment shift — large random perturbation."""
        cfg = self._config
        mag = cfg.shift_magnitude

        old = self.state_snapshot()

        self._temperature += self._rng.uniform(-mag, mag)
        self._temperature = _clamp(self._temperature, cfg.temperature_min, cfg.temperature_max)

        self._entropy_pressure += self._rng.uniform(-mag, mag)
        self._entropy_pressure = _clamp(self._entropy_pressure, cfg.entropy_pressure_min, cfg.entropy_pressure_max)

        self._resource_abundance += self._rng.uniform(-mag, mag)
        self._resource_abundance = _clamp(self._resource_abundance, cfg.resource_abundance_min, cfg.resource_abundance_max)

        self._noise_level += self._rng.uniform(-mag, mag)
        self._noise_level = _clamp(self._noise_level, cfg.noise_level_min, cfg.noise_level_max)

        record = {
            "type": "environment_shift",
            "cycle": self._cycle,
            "magnitude": mag,
            "before": {k: old[k] for k in ("temperature", "entropy_pressure", "resource_abundance", "noise_level")},
            "after": {
                "temperature": round(self._temperature, 6),
                "entropy_pressure": round(self._entropy_pressure, 6),
                "resource_abundance": round(self._resource_abundance, 6),
                "noise_level": round(self._noise_level, 6),
            },
            "timestamp": _utc_now(),
        }
        self._shift_log.append(record)
        logger.info("[ENV] Major shift at cycle %d", self._cycle)
        return record

    def _trigger_scarcity(self) -> Dict[str, Any]:
        """Trigger a scarcity event."""
        cfg = self._config
        severity = self._rng.uniform(cfg.scarcity_severity_min, cfg.scarcity_severity_max)
        duration = self._rng.randint(cfg.scarcity_duration_min, cfg.scarcity_duration_max)

        # Reduce resource pool immediately
        loss = self._resource_pool * severity
        self._resource_pool = max(0.0, self._resource_pool - loss)

        # Abundance reduction during event
        abundance_reduction = severity * 0.5

        event = ScarcityEvent(
            severity=severity,
            remaining_cycles=duration,
            started_at=_utc_now(),
            abundance_reduction=abundance_reduction,
        )
        self._scarcity_events.append(event)

        record = {
            "type": "scarcity_event",
            "cycle": self._cycle,
            "severity": round(severity, 4),
            "duration": duration,
            "pool_loss": round(loss, 4),
            "abundance_reduction": round(abundance_reduction, 4),
            "timestamp": _utc_now(),
        }
        self._scarcity_log.append(record)
        logger.warning(
            "[ENV] Scarcity event! severity=%.2f duration=%d pool_loss=%.1f",
            severity, duration, loss,
        )
        return record

    def _tick_scarcity_events(self) -> None:
        """Advance active scarcity events, remove expired ones."""
        active = []
        for event in self._scarcity_events:
            if event.tick():
                active.append(event)
            else:
                logger.info("[ENV] Scarcity event ended (severity=%.2f)", event.severity)
        self._scarcity_events = active

    def _trigger_shock(self) -> Dict[str, Any]:
        """v3.11: Trigger a rare shock event that favours resilience.

        During the shock:
        * entropy_pressure is temporarily spiked by ``shock_entropy_spike``.
        * Organisms classified as 'resilient' receive a fitness bonus.
        """
        cfg = self._config
        duration = self._rng.randint(cfg.shock_duration_min, cfg.shock_duration_max)
        event = ShockEvent(
            remaining_cycles=duration,
            entropy_spike=cfg.shock_entropy_spike,
            resilience_bonus=cfg.shock_resilience_bonus,
            started_at=_utc_now(),
        )
        self._shock_events.append(event)

        # Apply immediate entropy spike
        self._entropy_pressure = _clamp(
            self._entropy_pressure + cfg.shock_entropy_spike,
            cfg.entropy_pressure_min,
            cfg.entropy_pressure_max,
        )

        record = {
            "type": "shock_event",
            "cycle": self._cycle,
            "duration": duration,
            "entropy_spike": round(cfg.shock_entropy_spike, 4),
            "resilience_bonus": round(cfg.shock_resilience_bonus, 4),
            "timestamp": _utc_now(),
        }
        self._shock_log.append(record)
        logger.warning(
            "[ENV] SHOCK EVENT! duration=%d entropy_spike=%.2f resilience_bonus=%.2f",
            duration, cfg.shock_entropy_spike, cfg.shock_resilience_bonus,
        )
        return record

    def _tick_shock_events(self) -> None:
        """v3.11: Advance active shock events, remove expired ones."""
        active = []
        for event in self._shock_events:
            if event.tick():
                active.append(event)
            else:
                # Relax entropy spike when shock ends
                self._entropy_pressure = _clamp(
                    self._entropy_pressure - event.entropy_spike,
                    self._config.entropy_pressure_min,
                    self._config.entropy_pressure_max,
                )
                logger.info("[ENV] Shock event ended (entropy_spike=%.2f)", event.entropy_spike)
        self._shock_events = active

    # ── Named environmental events ───────────────────────────────────

    def _trigger_named_event(self) -> Optional[Dict[str, Any]]:
        """Pick a random named event from the catalogue and activate it."""
        cfg = self._config
        event_def = self._rng.choice(NAMED_EVENT_CATALOGUE)
        duration = self._rng.randint(cfg.named_event_duration_min,
                                      cfg.named_event_duration_max)
        event = NamedEvent(
            name=event_def["name"],
            description=event_def["description"],
            effects=dict(event_def["effects"]),
            remaining_cycles=duration,
            started_at=_utc_now(),
        )
        self._named_events.append(event)

        # Apply immediate one-shot effects
        effects = event.effects
        if "temperature_boost" in effects:
            self._temperature = _clamp(
                self._temperature + effects["temperature_boost"],
                self._config.temperature_min,
                self._config.temperature_max,
            )
        if "pool_drain_fraction" in effects:
            drain = self._resource_pool * effects["pool_drain_fraction"]
            self._resource_pool = max(0.0, self._resource_pool - drain)
        if "abundance_boost" in effects:
            self._resource_abundance = _clamp(
                self._resource_abundance + effects["abundance_boost"],
                0.05, 1.0,
            )
        if "entropy_boost" in effects:
            self._entropy_pressure = _clamp(
                self._entropy_pressure + effects["entropy_boost"],
                self._config.entropy_pressure_min,
                self._config.entropy_pressure_max,
            )

        record = {
            "type": "named_event",
            "name": event.name,
            "description": event.description,
            "cycle": self._cycle,
            "duration": duration,
            "effects": {k: round(v, 4) for k, v in effects.items()},
            "timestamp": _utc_now(),
        }
        self._named_event_log.append(record)
        logger.warning(
            "[ENV] NAMED EVENT: %s — %s (duration=%d)",
            event.name, event.description, duration,
        )
        return record

    def _tick_named_events(self) -> None:
        """Advance active named events; apply per-tick effects; expire."""
        active: List[NamedEvent] = []
        for event in self._named_events:
            if event.tick():
                # Per-tick ongoing effects
                if "pool_regen_multiplier" in event.effects:
                    bonus = self._config.resource_regen_rate * (
                        event.effects["pool_regen_multiplier"] - 1.0
                    )
                    self._resource_pool = min(
                        self._config.resource_pool_max,
                        self._resource_pool + max(0.0, bonus),
                    )
                active.append(event)
            else:
                # Reverse one-shot effects on expiry
                effects = event.effects
                if "temperature_boost" in effects:
                    self._temperature = _clamp(
                        self._temperature - effects["temperature_boost"],
                        self._config.temperature_min,
                        self._config.temperature_max,
                    )
                if "abundance_boost" in effects:
                    self._resource_abundance = _clamp(
                        self._resource_abundance - effects["abundance_boost"],
                        0.05, 1.0,
                    )
                if "entropy_boost" in effects:
                    self._entropy_pressure = _clamp(
                        self._entropy_pressure - effects["entropy_boost"],
                        self._config.entropy_pressure_min,
                        self._config.entropy_pressure_max,
                    )
                logger.info("[ENV] Named event ended: %s", event.name)
        self._named_events = active

    def _effective_abundance(self) -> float:
        """Resource abundance reduced by active scarcity events."""
        base = self._resource_abundance
        for event in self._scarcity_events:
            base -= event.abundance_reduction
        return _clamp(base, 0.05, 1.0)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
