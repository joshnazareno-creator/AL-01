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
    resource_pool_max: float = 100.0
    """Maximum energy available in the environment."""
    resource_pool_initial: float = 100.0
    """Starting resource pool."""
    resource_regen_rate: float = 2.0
    """How much energy regenerates per cycle (before abundance scaling)."""

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

    # Seeded randomness
    rng_seed: Optional[int] = None


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

        # 4. Tick active scarcity events
        self._tick_scarcity_events()

        # 5. Compute effective abundance (reduced during scarcity)
        effective_abundance = self._effective_abundance()

        # 6. Regenerate resource pool
        regen = self._config.resource_regen_rate * effective_abundance
        self._resource_pool = min(
            self._config.resource_pool_max,
            self._resource_pool + regen,
        )

        record.update(self.state_snapshot())
        return record

    def consume_resources(self, amount: float) -> float:
        """Attempt to consume *amount* from the resource pool.

        Returns the actual amount consumed (may be less if pool is low).
        """
        actual = min(amount, self._resource_pool)
        self._resource_pool = max(0.0, self._resource_pool - actual)
        return actual

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

    def _effective_abundance(self) -> float:
        """Resource abundance reduced by active scarcity events."""
        base = self._resource_abundance
        for event in self._scarcity_events:
            base -= event.abundance_reduction
        return _clamp(base, 0.05, 1.0)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
