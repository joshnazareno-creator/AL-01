"""AL-01 Autonomous Decision Engine + Computational Awareness Model.

Fully local, deterministic core.  No external API calls.

v3.0 — True ecosystem with selection pressure
----------------------------------------------
Builds on v2.3 and adds:

* **Environment integration** — decision cycle reads modifiers from the
  environment model (mutation cost multiplier, energy regen, trait decay,
  fitness noise, survival modifier).
* **Death signalling** — energy ≤ 0 emits ``organism_died`` flag.
  No more energy floor — organisms can die.
* **Auto mutation rate adjustment** — mutation rate drifts based on
  environment pressure and stagnation without manual intervention.

Decision Loop
-------------
  * **stabilize** — healthy fitness, no stagnation → energy bonus.
  * **adapt**     — stagnation detected → nudge weakest trait.
  * **mutate**    — effective fitness < threshold → genome mutation.
  * **blend**     — deep stagnation → cooperative genome blending.

Awareness Model
---------------
awareness = clamp(
    stimuli_rate   × 0.4  +
    decision_rate  × 0.3  +
    fitness_var    × 0.2  +
    (1 − stagnation) × 0.1  +
    novelty_accumulator,
    0.0, 1.0
)
"""

from __future__ import annotations

import json
import logging
import os

from al01.storage import rotate_jsonl
import random
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("al01.autonomy")

# ── Configuration ────────────────────────────────────────────────────────

@dataclass
class AutonomyConfig:
    """Tuneable knobs for the autonomous decision cycle."""

    decision_interval: int = 10
    """Run a decision every *decision_interval* heartbeats (ticks)."""

    fitness_threshold: float = 0.45
    """Below this fitness value → mutate."""

    stagnation_window: int = 10
    """Number of fitness samples kept for variance analysis."""

    stagnation_variance_epsilon: float = 1e-4
    """If variance of the last ``stagnation_window`` fitness values is below
    this, awareness is considered stagnant."""

    adapt_trait_nudge: float = 0.03
    """How much to nudge the weakest trait on an *adapt* decision."""

    # Awareness model weights — must sum to 1.0  (novelty is additive on top)
    awareness_w_stimuli: float = 0.4
    awareness_w_decisions: float = 0.3
    awareness_w_fitness_var: float = 0.2
    awareness_w_stagnation: float = 0.1

    # Normalisation caps — values above these → 1.0 for that component
    stimuli_rate_cap: float = 5.0
    """Stimuli received per decision cycle above this → component = 1.0."""

    decision_rate_cap: float = 20.0
    """Total decisions above this → component = 1.0."""

    fitness_variance_cap: float = 0.01
    """Fitness variance above this → component = 1.0."""

    # ── Energy system (soft resource pressure) ────────────────────────
    energy_initial: float = 1.0
    energy_decay_per_cycle: float = 0.005
    energy_stabilize_bonus: float = 0.02
    energy_adapt_cost: float = 0.01
    energy_mutate_cost: float = 0.015
    energy_blend_cost: float = 0.01
    energy_min: float = 0.10
    """Energy floor — organisms can't drop below 10% (prevents death spiral)."""
    energy_fitness_floor: float = 0.3
    """Below this energy level, effective fitness is reduced proportionally."""

    # ── v3.6: Recovery mode (break death-spiral lock) ─────────────────
    recovery_energy_threshold: float = 0.20
    """Enter recovery mode when energy stays below this for recovery_trigger_cycles."""
    recovery_trigger_cycles: int = 10
    """Consecutive low-energy cycles before recovery mode activates."""
    recovery_stabilize_bonus: float = 0.03
    """Extra energy bonus during recovery-mode stabilize (on top of normal)."""
    recovery_fitness_penalty_cap: float = 0.50
    """Max fitness penalty ratio from energy floor during recovery.
    Normal: energy/floor (can be 0.33). Recovery: max(energy/floor, this)."""
    mutate_energy_rebate: float = 0.005
    """Small energy rebate after mutate to prevent net-negative-only decisions."""
    adapt_energy_rebate: float = 0.003
    """Small energy rebate after adapt."""
    stagnation_delta_cap: float = 1.0
    """Maximum stagnation-scaled mutation delta (prevents wild swings)."""
    directed_mutation_bias: float = 0.02
    """Upward bias added to mutation delta when fitness is below threshold.
    Breaks zero-mean symmetry to allow gradual improvement."""
    env_regen_multiplier: float = 0.05
    """Multiplier for environment energy regen (was 0.01, too low).""" 

    # ── Environment drift ─────────────────────────────────────────────
    drift_step: float = 0.01
    drift_fitness_threshold_min: float = 0.30
    drift_fitness_threshold_max: float = 0.60
    drift_mutation_rate_min: float = 0.05
    drift_mutation_rate_max: float = 0.20

    # ── Stagnation adaptive response ──────────────────────────────────
    stagnation_mutation_boost: float = 0.05
    stagnation_exploration_cycles: int = 5
    stagnation_blend_multiplier: int = 2
    """Blend when stagnation_count > stagnation_window × this multiplier."""

    # ── Novelty-based awareness ───────────────────────────────────────
    novelty_growth_per_event: float = 0.05
    novelty_decay_per_cycle: float = 0.02
    novelty_max: float = 0.3

    # ── Environment trait weights (fluctuating fitness formula) ────────
    env_weight_drift_step: float = 0.03
    """Max drift per cycle for each trait weight."""
    env_weight_min: float = 0.5
    env_weight_max: float = 2.0
    """Trait weights stay in [min, max] range."""

    # ── Stagnation-scaled mutation diversity ──────────────────────────
    stagnation_delta_scale: float = 0.5
    """mutation_delta *= (1 + stagnation_count * scale / stagnation_window).
    v3.6: reduced from 1.0 to 0.5 — slower escalation, capped by stagnation_delta_cap."""

    # ── Variance kick (break zero-variance stagnation lock) ─────────
    variance_kick_threshold: float = 1e-6
    """When fitness variance falls below this, inject a trait perturbation."""
    variance_kick_magnitude: float = 0.03
    """Size of the random trait nudge when variance kick triggers."""
    variance_kick_cooldown: int = 5
    """Minimum cycles between variance kicks."""

    # ── Stagnation breaker (escalating emergency interventions) ────────
    stagnation_tier1_threshold: int = 50
    """Re-trigger exploration mode every this many stagnation cycles."""
    stagnation_tier2_threshold: int = 200
    """Double adapt nudge + increase variance kick magnitude."""
    stagnation_tier3_threshold: int = 500
    """Shuffle all traits by ±shuffle_magnitude to break deep lock."""
    stagnation_tier3_shuffle_magnitude: float = 0.10
    """Per-trait random perturbation at tier 3."""
    stagnation_hard_limit: int = 800
    """Hard reset: randomise all traits around defaults + clear history."""
    stagnation_hard_reset_spread: float = 0.15
    """Spread around default values on hard reset (0.5 ± this)."""

    # ── Entropy (trait decay) ─────────────────────────────────────────
    entropy_rate: float = 0.005
    """Per-cycle proportional trait decay when no stimulus received."""
    entropy_idle_threshold: int = 3
    """Cycles with no external stimulus before entropy kicks in."""

    # ── Vital Score ───────────────────────────────────────────────────
    vital_identity_weight: float = 0.25
    vital_variance_weight: float = 0.25
    vital_adaptation_weight: float = 0.25
    vital_entropy_resistance_weight: float = 0.25

    # ── v3.23: Stabilization — stress feedback & adaptability recovery ─
    stress_exploration_threshold: float = 0.60
    """Stress level above which exploration/mutation probability is boosted."""
    stress_mutation_boost: float = 0.03
    """Extra mutation rate when stress exceeds threshold."""
    adaptability_recovery_threshold: float = 0.20
    """When adaptability drops below this, apply a recovery nudge."""
    adaptability_recovery_nudge: float = 0.02
    """Upward nudge applied to adaptability each cycle when below threshold."""
    energy_efficiency_metabolic_scale: float = 0.30
    """How much energy_efficiency trait reduces per-cycle energy decay (0–1).
    At trait=1.0, energy decay is reduced by this fraction."""


# ── Decision types ───────────────────────────────────────────────────────

DECISION_STABILIZE = "stabilize"
DECISION_ADAPT = "adapt"
DECISION_MUTATE = "mutate"
DECISION_BLEND = "blend"


# ── Helpers ──────────────────────────────────────────────────────────────

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ── Awareness Model ──────────────────────────────────────────────────────

class AwarenessModel:
    """Compute awareness deterministically from measurable internal state.

    awareness = clamp(
        stimuli_rate   × w_stimuli   +
        decision_rate  × w_decisions +
        fitness_var    × w_fitness   +
        (1 − stagnation_flag) × w_stagnation
        + novelty_accumulator,
        0.0, 1.0
    )

    Each raw signal is normalised to [0, 1] before weighting.  The novelty
    accumulator grows with novel events (new unique stimuli, environmental
    drift, fitness-variance increase) and decays each idle cycle.

    All internal counters persist via the autonomy log and are rebuilt on
    recovery.
    """

    def __init__(self, config: AutonomyConfig) -> None:
        self._config = config
        # Running counters (recovered from log on reboot)
        self._stimuli_received: int = 0       # stimuli received since last decision
        self._total_stimuli: int = 0          # lifetime stimuli count
        self._last_awareness: float = 0.0     # most recently computed value
        # Novelty tracking
        self._unique_stimulus_hashes: set = set()
        self._novelty_accumulator: float = 0.0
        self._novel_events_this_cycle: int = 0
        self._prev_fitness_variance: float = 0.0

    # ── Public ───────────────────────────────────────────────────────

    @property
    def last_awareness(self) -> float:
        return self._last_awareness

    @property
    def total_stimuli(self) -> int:
        return self._total_stimuli

    @property
    def novelty_accumulator(self) -> float:
        return self._novelty_accumulator

    def record_stimulus(self, stimulus_hash: str = "") -> None:
        """Call once for every stimulus received (outside the decision cycle).

        If *stimulus_hash* is provided and has not been seen before, a
        novelty event is registered.
        """
        self._stimuli_received += 1
        self._total_stimuli += 1
        if stimulus_hash and stimulus_hash not in self._unique_stimulus_hashes:
            self._unique_stimulus_hashes.add(stimulus_hash)
            self._novel_events_this_cycle += 1

    def compute(
        self,
        total_decisions: int,
        fitness_variance: float,
        is_stagnant: bool,
        env_drift_magnitude: float = 0.0,
    ) -> Dict[str, Any]:
        """Compute awareness and return a breakdown dict.

        Parameters
        ----------
        total_decisions : int
            Cumulative decision count from the AutonomyEngine.
        fitness_variance : float
            Variance of the fitness history window.
        is_stagnant : bool
            Whether stagnation is currently detected.
        env_drift_magnitude : float
            Magnitude of environment drift this cycle (for novelty).

        Returns
        -------
        dict
            Keys: ``awareness``, ``stimuli_rate``, ``decision_rate``,
            ``fitness_variance_norm``, ``stagnation_component``,
            ``stimuli_received_this_cycle``, ``novelty_accumulator``,
            ``novelty_signals``.
        """
        cfg = self._config

        # 1. Stimuli rate — how many stimuli arrived since last decision
        stimuli_rate_raw = float(self._stimuli_received)
        stimuli_rate = _clamp(stimuli_rate_raw / cfg.stimuli_rate_cap)

        # 2. Decision activity rate — total decisions normalised
        decision_rate = _clamp(float(total_decisions) / cfg.decision_rate_cap)

        # 3. Fitness variance — normalised
        fitness_var_norm = _clamp(fitness_variance / cfg.fitness_variance_cap)

        # 4. Stagnation flag (inverted: NOT stagnant → 1.0)
        stagnation_component = 0.0 if is_stagnant else 1.0

        # Base weighted sum (existing formula)
        base_awareness = _clamp(
            stimuli_rate       * cfg.awareness_w_stimuli +
            decision_rate      * cfg.awareness_w_decisions +
            fitness_var_norm   * cfg.awareness_w_fitness_var +
            stagnation_component * cfg.awareness_w_stagnation,
        )

        # 5. Novelty accumulator — grows with novel events, decays otherwise
        fv_delta = max(0.0, fitness_variance - self._prev_fitness_variance)
        self._prev_fitness_variance = fitness_variance

        novel_signals = float(self._novel_events_this_cycle)
        if fv_delta > cfg.stagnation_variance_epsilon:
            novel_signals += 1.0
        if env_drift_magnitude > 0.001:
            novel_signals += 1.0

        if novel_signals > 0:
            self._novelty_accumulator += novel_signals * cfg.novelty_growth_per_event
        self._novelty_accumulator -= cfg.novelty_decay_per_cycle
        self._novelty_accumulator = _clamp(
            self._novelty_accumulator, 0.0, cfg.novelty_max
        )
        self._novel_events_this_cycle = 0

        # Final awareness = base + novelty bonus
        awareness = _clamp(base_awareness + self._novelty_accumulator)

        self._last_awareness = round(awareness, 6)

        breakdown: Dict[str, Any] = {
            "awareness": self._last_awareness,
            "stimuli_rate": round(stimuli_rate, 6),
            "decision_rate": round(decision_rate, 6),
            "fitness_variance_norm": round(fitness_var_norm, 6),
            "stagnation_component": stagnation_component,
            "stimuli_received_this_cycle": self._stimuli_received,
            "total_stimuli": self._total_stimuli,
            "novelty_accumulator": round(self._novelty_accumulator, 6),
            "novelty_signals": novel_signals,
        }

        # Reset per-cycle counter after computation
        self._stimuli_received = 0

        return breakdown

    def recover(
        self,
        total_stimuli: int,
        last_awareness: float,
        novelty_accumulator: float = 0.0,
    ) -> None:
        """Restore counters from persisted log data."""
        self._total_stimuli = total_stimuli
        self._last_awareness = last_awareness
        self._novelty_accumulator = novelty_accumulator


# ── Engine ───────────────────────────────────────────────────────────────

class AutonomyEngine:
    """Deterministic, fully-local decision loop for AL-01.

    v2.2 additions: energy system, environment drift, stagnation adaptive
    response, cooperative blend decision.
    """

    def __init__(
        self,
        data_dir: str = ".",
        config: Optional[AutonomyConfig] = None,
        rng_seed: Optional[int] = None,
    ) -> None:
        self._config = config or AutonomyConfig()
        self._data_dir = data_dir

        # Seeded RNG for reproducibility — environment drift uses this
        self._rng = random.Random(rng_seed)

        # Fitness history ring buffer (most recent at the end)
        self._fitness_history: List[float] = []

        # Stagnation counter (how many consecutive cycles awareness was flat)
        self._stagnation_count: int = 0

        # Decision counter
        self._total_decisions: int = 0

        # Computational awareness model
        self._awareness_model = AwarenessModel(self._config)

        # Energy system — soft resource pressure
        self._energy: float = self._config.energy_initial

        # Environment drift — effective values that shift over time
        self._effective_fitness_threshold: float = self._config.fitness_threshold
        self._mutation_rate_offset: float = 0.0

        # Environment trait weights — fluctuating importance per trait
        self._env_trait_weights: Dict[str, float] = {
            "adaptability": 1.0,
            "energy_efficiency": 1.0,
            "resilience": 1.0,
            "perception": 1.0,
            "creativity": 1.0,
        }

        # Entropy idle counter — how many decide() cycles without stimulus
        self._idle_cycles: int = 0

        # Vital Score tracking
        self._identity_persistence_score: float = 1.0  # starts perfect
        self._last_trait_snapshot: Optional[Dict[str, float]] = None
        self._entropy_resistance_accumulator: float = 1.0
        self._adaptation_success_count: int = 0
        self._adaptation_attempt_count: int = 0

        # Stagnation adaptive response
        self._exploration_mode: bool = False
        self._exploration_cycles_remaining: int = 0

        # v3.6: Recovery mode — breaks energy-fitness death spiral
        self._recovery_mode: bool = False
        self._low_energy_consecutive: int = 0

        # v3.7: Variance kick — breaks zero-variance stagnation lock
        self._variance_kick_cooldown_remaining: int = 0

        # Log file — append-only JSONL
        os.makedirs(data_dir, exist_ok=True)
        self._log_path = os.path.join(data_dir, "autonomy_log.jsonl")

        # Recover state from existing log on disk
        self._recover_state()

    # ── Public API ───────────────────────────────────────────────────

    @property
    def config(self) -> AutonomyConfig:
        return self._config

    @property
    def awareness_model(self) -> AwarenessModel:
        return self._awareness_model

    @property
    def awareness(self) -> float:
        """Most recently computed awareness value."""
        return self._awareness_model.last_awareness

    @property
    def energy(self) -> float:
        return self._energy

    @property
    def effective_fitness_threshold(self) -> float:
        return self._effective_fitness_threshold

    @property
    def mutation_rate_offset(self) -> float:
        return self._mutation_rate_offset

    @property
    def exploration_mode(self) -> bool:
        return self._exploration_mode

    @property
    def recovery_mode(self) -> bool:
        """True when the organism is in energy recovery mode (v3.6)."""
        return self._recovery_mode

    def set_exploration_mode(self, enabled: bool, cycles: int = 0) -> Dict[str, Any]:
        """Manually toggle exploration mode.

        Args:
            enabled: True to activate, False to deactivate.
            cycles: How many cycles to run exploration (0 = use config default).

        Returns:
            Dict with the new exploration state.
        """
        self._exploration_mode = enabled
        if enabled:
            self._exploration_cycles_remaining = (
                cycles if cycles > 0 else self._config.stagnation_exploration_cycles
            )
            logger.info(
                "[AUTONOMY] Manual exploration mode ON for %d cycles",
                self._exploration_cycles_remaining,
            )
        else:
            self._exploration_cycles_remaining = 0
            logger.info("[AUTONOMY] Manual exploration mode OFF")

        return {
            "exploration_mode": self._exploration_mode,
            "cycles_remaining": self._exploration_cycles_remaining,
        }

    @property
    def env_trait_weights(self) -> Dict[str, float]:
        """Current environment trait weights (read-only copy)."""
        return dict(self._env_trait_weights)

    def should_variance_kick(self) -> bool:
        """True when fitness variance is critically low and cooldown has elapsed."""
        if self._variance_kick_cooldown_remaining > 0:
            return False
        if len(self._fitness_history) < self._config.stagnation_window:
            return False
        window = self._fitness_history[-self._config.stagnation_window:]
        variance = statistics.variance(window) if len(window) >= 2 else 0.0
        return variance < self._config.variance_kick_threshold

    def apply_variance_kick(self, genome: Any) -> Optional[Dict[str, Any]]:
        """Inject a small random trait perturbation to break zero-variance lock.

        Picks one random trait and nudges it upward by variance_kick_magnitude.
        Resets cooldown timer. Returns a record of the kick or None.
        """
        if not self.should_variance_kick():
            return None
        cfg = self._config
        trait_names = list(genome.traits.keys())
        if not trait_names:
            return None
        # Pick the weakest trait — most impactful to nudge
        target = min(trait_names, key=lambda t: genome.get_trait(t))
        old_val = genome.get_trait(target)
        # Tier 2: double magnitude when stagnation is deep
        magnitude = cfg.variance_kick_magnitude
        if self._stagnation_count >= cfg.stagnation_tier2_threshold:
            magnitude *= 2.0
        genome.set_trait(target, old_val + magnitude)
        self._variance_kick_cooldown_remaining = cfg.variance_kick_cooldown
        logger.info(
            "[AUTONOMY] Variance kick: %s %.4f → %.4f (magnitude=%.3f)",
            target, old_val, genome.get_trait(target), magnitude,
        )
        return {
            "trait": target,
            "old_value": round(old_val, 6),
            "new_value": round(genome.get_trait(target), 6),
            "nudge": magnitude,
        }

    def break_stagnation(self, genome: Any) -> Optional[Dict[str, Any]]:
        """Escalating stagnation breaker — progressively stronger interventions.

        Tier 1 (>50):  Re-trigger exploration mode periodically.
        Tier 2 (>200): Handled via doubled variance kick magnitude.
        Tier 3 (>500): Shuffle ALL traits by random perturbation.
        Hard  (>800):  Randomise traits around defaults + clear fitness history.

        Returns a record of what happened, or None if below all thresholds.
        """
        cfg = self._config
        sc = self._stagnation_count
        if sc < cfg.stagnation_tier1_threshold:
            return None

        result: Dict[str, Any] = {"stagnation_count": sc, "tier": 0}

        # ── Hard limit (>800): full trait reset ──────────────────────
        if sc >= cfg.stagnation_hard_limit:
            result["tier"] = 4
            old_traits = dict(genome.traits)
            default_vals = {"adaptability": 0.5, "energy_efficiency": 0.5,
                           "resilience": 0.5, "perception": 0.5, "creativity": 0.5}
            new_traits = {}
            for t in genome.traits:
                base = default_vals.get(t, 0.5)
                delta = self._rng.uniform(-cfg.stagnation_hard_reset_spread,
                                          cfg.stagnation_hard_reset_spread)
                genome.set_trait(t, base + delta)
                new_traits[t] = round(genome.get_trait(t), 6)
            # Clear fitness history to break the variance deadlock
            self._fitness_history.clear()
            self._stagnation_count = 0
            self._exploration_mode = False
            result["action"] = "hard_reset"
            result["old_traits"] = {k: round(v, 6) for k, v in old_traits.items()}
            result["new_traits"] = new_traits
            logger.warning(
                "[AUTONOMY] STAGNATION HARD RESET at count=%d — "
                "traits randomised around defaults, history cleared", sc,
            )
            return result

        # ── Tier 3 (>500): shuffle all traits ────────────────────────
        if sc >= cfg.stagnation_tier3_threshold:
            result["tier"] = 3
            old_traits = dict(genome.traits)
            new_traits = {}
            for t in genome.traits:
                old_val = genome.get_trait(t)
                delta = self._rng.uniform(-cfg.stagnation_tier3_shuffle_magnitude,
                                          cfg.stagnation_tier3_shuffle_magnitude)
                genome.set_trait(t, old_val + delta)
                new_traits[t] = round(genome.get_trait(t), 6)
            result["action"] = "trait_shuffle"
            result["old_traits"] = {k: round(v, 6) for k, v in old_traits.items()}
            result["new_traits"] = new_traits
            logger.info(
                "[AUTONOMY] Stagnation tier-3 shuffle at count=%d", sc,
            )
            return result

        # ── Tier 1 (>50): periodically re-trigger exploration ────────
        if sc >= cfg.stagnation_tier1_threshold:
            result["tier"] = 1
            # Re-trigger exploration every tier1_threshold cycles
            if sc % cfg.stagnation_tier1_threshold == 0 and not self._exploration_mode:
                self._exploration_mode = True
                self._exploration_cycles_remaining = (
                    cfg.stagnation_exploration_cycles * 2  # longer burst
                )
                result["action"] = "exploration_retrigger"
                logger.info(
                    "[AUTONOMY] Stagnation tier-1: re-triggering exploration "
                    "at count=%d for %d cycles",
                    sc, self._exploration_cycles_remaining,
                )
            else:
                result["action"] = "tier1_active"
            return result

        return None

    @property
    def idle_cycles(self) -> int:
        return self._idle_cycles

    @property
    def should_entropy_decay(self) -> bool:
        """True when enough idle cycles have passed to trigger entropy."""
        return self._idle_cycles >= self._config.entropy_idle_threshold

    def stagnation_scaled_delta(self, base_delta: float) -> float:
        """Compute mutation delta scaled by stagnation pressure.

        Higher stagnation → wider mutation variance to explore more.
        v3.6: capped by stagnation_delta_cap to prevent wild swings.
        """
        cfg = self._config
        if cfg.stagnation_window <= 0:
            return base_delta
        scale = 1.0 + (self._stagnation_count * cfg.stagnation_delta_scale
                        / cfg.stagnation_window)
        result = base_delta * scale
        # v3.6: Cap to prevent chaotic mutations at high stagnation counts
        return min(result, cfg.stagnation_delta_cap)

    def compute_vital_score(self) -> Dict[str, Any]:
        """Compute Vital Score (0–100) from four sub-scores.

        Sub-scores (each 0–25 by default, weights configurable):
        1. **Identity persistence** — how stable is the organism's core
           identity signature across cycles.
        2. **Trait variance** — diversity of trait values (not all at same
           level).
        3. **Adaptation success** — fraction of adapt/mutate decisions that
           improved fitness.
        4. **Entropy resistance** — how well traits are maintained against
           decay.
        """
        cfg = self._config

        # 1. Identity persistence (rolling EMA, 0–1)
        identity = _clamp(self._identity_persistence_score)

        # 2. Trait variance — want some spread, not all traits identical
        #    We normalise variance of fitness_history as a proxy
        variance_score = 0.0
        if len(self._fitness_history) >= 2:
            fv = statistics.variance(self._fitness_history)
            # Normalize: a variance of 0.01+ → full score
            variance_score = _clamp(fv / 0.01)

        # 3. Adaptation success rate
        adaptation_score = 0.0
        if self._adaptation_attempt_count > 0:
            adaptation_score = (self._adaptation_success_count
                                / self._adaptation_attempt_count)
        adaptation_score = _clamp(adaptation_score)

        # 4. Entropy resistance (rolling EMA, 0–1)
        entropy_resistance = _clamp(self._entropy_resistance_accumulator)

        # Weighted sum → 0–100
        raw = (
            identity * cfg.vital_identity_weight +
            variance_score * cfg.vital_variance_weight +
            adaptation_score * cfg.vital_adaptation_weight +
            entropy_resistance * cfg.vital_entropy_resistance_weight
        )
        # Weights sum to 1.0 → raw is 0–1 → scale to 0–100
        vital_index = _clamp(raw) * 100.0

        return {
            "vital_index": round(vital_index, 2),
            "identity_persistence": round(identity * 100, 2),
            "trait_variance": round(variance_score * 100, 2),
            "adaptation_success": round(adaptation_score * 100, 2),
            "entropy_resistance": round(entropy_resistance * 100, 2),
            "adaptation_attempts": self._adaptation_attempt_count,
            "adaptation_successes": self._adaptation_success_count,
        }

    def record_stimulus(self, stimulus_hash: str = "") -> None:
        """Notify the awareness model that a stimulus was received."""
        self._awareness_model.record_stimulus(stimulus_hash)

    @property
    def fitness_history(self) -> List[float]:
        """Return a *copy* of the fitness history window."""
        return list(self._fitness_history)

    @property
    def stagnation_count(self) -> int:
        return self._stagnation_count

    @property
    def total_decisions(self) -> int:
        return self._total_decisions

    @property
    def is_stagnant(self) -> bool:
        """True when the fitness history window shows stagnation."""
        if len(self._fitness_history) < self._config.stagnation_window:
            return False
        window = self._fitness_history[-self._config.stagnation_window:]
        variance = statistics.variance(window) if len(window) >= 2 else 0.0
        return variance < self._config.stagnation_variance_epsilon

    @property
    def log_path(self) -> str:
        return self._log_path

    # ── Decision cycle ───────────────────────────────────────────────

    def decide(self, fitness: float, awareness: float,
               mutation_rate: float, pending_stimuli: int,
               current_traits: Optional[Dict[str, float]] = None,
               env_modifiers: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Run one decision cycle and return the decision record.

        Parameters
        ----------
        env_modifiers : dict, optional
            Environment-derived modifiers:
            - ``mutation_cost_multiplier`` (≥1.0)
            - ``energy_regen_rate`` (0–1)
            - ``trait_decay_multiplier`` (≥1.0)
            - ``fitness_noise_penalty`` (0–0.15)
            - ``survival_modifier`` (0–1)
        """
        env = env_modifiers or {}
        mutation_cost_mult = env.get("mutation_cost_multiplier", 1.0)
        energy_regen = env.get("energy_regen_rate", 0.0)
        fitness_noise = env.get("fitness_noise_penalty", 0.0)
        # 0. Environment drift (returns magnitude for novelty detection)
        drift_magnitude = self._drift_environment()

        # Track idle cycles (no external stimulus since last decide)
        if pending_stimuli == 0:
            self._idle_cycles += 1
        else:
            self._idle_cycles = 0

        # v3.7: Tick variance-kick cooldown
        if self._variance_kick_cooldown_remaining > 0:
            self._variance_kick_cooldown_remaining -= 1

        # Identity persistence tracking (trait snapshot similarity)
        if current_traits and self._last_trait_snapshot:
            diffs = []
            for k in current_traits:
                old_v = self._last_trait_snapshot.get(k, current_traits[k])
                diffs.append(abs(current_traits[k] - old_v))
            if diffs:
                avg_drift = sum(diffs) / len(diffs)
                # EMA: identity decays if traits shift a lot
                self._identity_persistence_score = (
                    0.9 * self._identity_persistence_score +
                    0.1 * (1.0 - min(1.0, avg_drift * 10))
                )
        if current_traits:
            self._last_trait_snapshot = dict(current_traits)

        # 1. Energy decay + environment regen
        # v3.23: energy_efficiency trait reduces per-cycle decay
        eff_trait = 0.5
        if current_traits:
            eff_trait = current_traits.get("energy_efficiency", 0.5)
        eff_scale = 1.0 - eff_trait * self._config.energy_efficiency_metabolic_scale
        eff_scale = max(0.3, min(1.0, eff_scale))
        self._energy -= self._config.energy_decay_per_cycle * eff_scale
        # v3.6: Use configurable env regen multiplier (was hardcoded 0.01)
        self._energy += energy_regen * self._config.env_regen_multiplier
        # v3.13: Scale regen by pool grant ratio (resource scarcity throttle)
        pool_grant = env_modifiers.get("pool_grant_ratio", 1.0) if env_modifiers else 1.0
        if pool_grant < 1.0:
            # Reduce net regen when the pool couldn't grant full metabolic cost
            regen_applied = energy_regen * self._config.env_regen_multiplier
            reduction = regen_applied * (1.0 - pool_grant) * 0.5
            self._energy -= reduction

        # v3.6: Recovery mode — track consecutive low-energy cycles
        cfg = self._config
        if self._energy < cfg.recovery_energy_threshold:
            self._low_energy_consecutive += 1
        else:
            self._low_energy_consecutive = 0
            if self._recovery_mode:
                self._recovery_mode = False
                logger.info("[AUTONOMY] Recovery mode OFF — energy above %.2f",
                            cfg.recovery_energy_threshold)

        if (not self._recovery_mode
                and self._low_energy_consecutive >= cfg.recovery_trigger_cycles):
            self._recovery_mode = True
            logger.info(
                "[AUTONOMY] Recovery mode ON — energy below %.2f for %d cycles",
                cfg.recovery_energy_threshold, self._low_energy_consecutive,
            )

        # 2. Track fitness — penalise with noise + low energy
        effective_fitness = fitness - fitness_noise
        if self._energy < self._config.energy_fitness_floor:
            penalty_ratio = self._energy / self._config.energy_fitness_floor
            # v3.6: During recovery, cap the penalty so fitness isn't crushed
            if self._recovery_mode:
                penalty_ratio = max(penalty_ratio,
                                    cfg.recovery_fitness_penalty_cap)
            effective_fitness = fitness * penalty_ratio

        self._fitness_history.append(round(effective_fitness, 6))
        if len(self._fitness_history) > self._config.stagnation_window:
            self._fitness_history = self._fitness_history[
                -self._config.stagnation_window:
            ]

        # 3. Evaluate stagnation
        stagnant = self.is_stagnant
        if stagnant:
            self._stagnation_count += 1
        else:
            self._stagnation_count = 0

        # 4. Stagnation adaptive response — exploration mode
        adaptation_reason: Optional[str] = None
        if stagnant and not self._exploration_mode:
            self._exploration_mode = True
            self._exploration_cycles_remaining = (
                self._config.stagnation_exploration_cycles
            )
            adaptation_reason = (
                f"stagnation detected (count={self._stagnation_count}): "
                f"boosting mutation_rate by {self._config.stagnation_mutation_boost}, "
                f"entering exploration mode for "
                f"{self._config.stagnation_exploration_cycles} cycles"
            )
            logger.info("[AUTONOMY] %s", adaptation_reason)

        # v3.7: Periodically re-enter exploration during extended stagnation
        if (stagnant
                and not self._exploration_mode
                and self._stagnation_count >= self._config.stagnation_tier1_threshold
                and self._stagnation_count % self._config.stagnation_tier1_threshold == 0):
            self._exploration_mode = True
            self._exploration_cycles_remaining = (
                self._config.stagnation_exploration_cycles * 2
            )
            adaptation_reason = (
                f"stagnation tier-1 retrigger (count={self._stagnation_count}): "
                f"exploration for {self._exploration_cycles_remaining} cycles"
            )
            logger.info("[AUTONOMY] %s", adaptation_reason)

        if self._exploration_mode:
            self._exploration_cycles_remaining -= 1
            if self._exploration_cycles_remaining <= 0:
                self._exploration_mode = False

        # v3.23: Stress feedback loop — high stress triggers adaptive behaviour
        # Compute composite stress (same formula as InternalSignal)
        energy_stress = max(0.0, 1.0 - self._energy * 2.0)
        stag_stress = min(1.0, self._stagnation_count / 200.0)
        recovery_stress = 0.3 if self._recovery_mode else 0.0
        computed_stress = min(1.0, energy_stress + stag_stress + recovery_stress)
        stress_boosted = (
            computed_stress > cfg.stress_exploration_threshold
            and not self._recovery_mode
        )
        if stress_boosted and not self._exploration_mode:
            self._exploration_mode = True
            self._exploration_cycles_remaining = max(
                self._exploration_cycles_remaining,
                cfg.stagnation_exploration_cycles,
            )
            if not adaptation_reason:
                adaptation_reason = (
                    f"stress feedback (stress={computed_stress:.3f} > "
                    f"{cfg.stress_exploration_threshold}): entering exploration"
                )
            logger.info(
                "[AUTONOMY] Stress feedback: stress=%.3f — boosting exploration",
                computed_stress,
            )

        # 5. Deterministic decision rules (priority order)
        #    v3.6: Recovery mode overrides — force STABILIZE to rebuild energy
        #    v3.25: Founder recovery mode — block forced mutation, prefer safe recovery
        blend_threshold = (
            self._config.stagnation_window
            * self._config.stagnation_blend_multiplier
        )
        eff_threshold = self._effective_fitness_threshold
        if self._exploration_mode:
            eff_threshold *= 0.8  # lower bar during exploration

        # v3.25: Founder recovery flags from env_modifiers
        founder_recovery = env.get("founder_recovery_mode", False)
        founder_mutate_blocked = env.get("founder_mutate_blocked", False)

        if self._recovery_mode:
            # v3.6: Recovery takes highest priority — force stabilize to
            # break the mutate→energy-drain→worse-fitness death spiral
            decision = DECISION_STABILIZE
            reason = (
                f"RECOVERY MODE: energy {self._energy:.4f} below "
                f"{cfg.recovery_energy_threshold:.2f} for "
                f"{self._low_energy_consecutive} cycles — forcing stabilize"
            )
        elif founder_mutate_blocked and effective_fitness < eff_threshold:
            # v3.25: Founder grace/cooldown active — use safe recovery
            # instead of forced mutation. Alternate stabilize and adapt (low-risk).
            if self._energy < 0.5:
                decision = DECISION_STABILIZE
                reason = (
                    f"FOUNDER RECOVERY: mutation blocked (grace/cooldown), "
                    f"energy {self._energy:.4f} < 0.5 — stabilizing"
                )
            else:
                decision = DECISION_ADAPT
                reason = (
                    f"FOUNDER RECOVERY: mutation blocked (grace/cooldown), "
                    f"energy {self._energy:.4f} >= 0.5 — safe adapt"
                )
        elif founder_recovery and effective_fitness < eff_threshold:
            # v3.25: Founder in recovery but not blocked — allow mutation
            # only every other cycle (alternating with stabilize)
            if self._total_decisions % 2 == 0:
                decision = DECISION_STABILIZE
                reason = (
                    f"FOUNDER RECOVERY: alternating stabilize "
                    f"(fitness {effective_fitness:.4f} < {eff_threshold:.4f})"
                )
            else:
                decision = DECISION_MUTATE
                reason = (
                    f"FOUNDER RECOVERY: allowing mutation "
                    f"(fitness {effective_fitness:.4f} < {eff_threshold:.4f})"
                )
        elif effective_fitness < eff_threshold:
            decision = DECISION_MUTATE
            reason = (
                f"fitness {effective_fitness:.4f} < threshold "
                f"{eff_threshold:.4f} (energy={self._energy:.4f})"
            )
        elif (
            self._stagnation_count > blend_threshold
            and self._stagnation_count > 0
        ):
            decision = DECISION_BLEND
            reason = (
                f"deep stagnation ({self._stagnation_count} > "
                f"{blend_threshold}): cooperative genome blend"
            )
        elif stagnant:
            decision = DECISION_ADAPT
            reason = (
                f"stagnation for {self._stagnation_count} cycles "
                f"(variance < {self._config.stagnation_variance_epsilon})"
            )
        else:
            decision = DECISION_STABILIZE
            reason = "fitness healthy and no stagnation"

        self._total_decisions += 1

        # Track adaptation success (fitness improved after mutate/adapt?)
        prev_fitness = self._fitness_history[-2] if len(self._fitness_history) >= 2 else 0.0
        if decision in (DECISION_MUTATE, DECISION_ADAPT):
            self._adaptation_attempt_count += 1
            if effective_fitness > prev_fitness:
                self._adaptation_success_count += 1

        # 6. Energy adjustment based on decision quality (env cost multiplier)
        self._apply_energy_delta(decision, mutation_cost_mult)

        # 7. Compute effective mutation rate (with drift + exploration boost)
        effective_mr = mutation_rate + self._mutation_rate_offset
        if self._exploration_mode:
            effective_mr += self._config.stagnation_mutation_boost
        # v3.23: Stress feedback — additional mutation boost under high stress
        if stress_boosted:
            effective_mr += cfg.stress_mutation_boost
        effective_mr = _clamp(effective_mr, 0.01, 0.50)

        # 7b. Stagnation-scaled mutation delta
        effective_delta = self.stagnation_scaled_delta(0.10)  # base delta

        # 8. Compute awareness from the model
        fitness_variance = 0.0
        if len(self._fitness_history) >= 2:
            fitness_variance = statistics.variance(self._fitness_history)

        # Entropy resistance update
        if self.should_entropy_decay:
            self._entropy_resistance_accumulator = (
                0.95 * self._entropy_resistance_accumulator
            )
        else:
            self._entropy_resistance_accumulator = min(
                1.0,
                self._entropy_resistance_accumulator + 0.01,
            )

        awareness_breakdown = self._awareness_model.compute(
            total_decisions=self._total_decisions,
            fitness_variance=fitness_variance,
            is_stagnant=stagnant,
            env_drift_magnitude=drift_magnitude,
        )

        computed_awareness = awareness_breakdown["awareness"]

        record: Dict[str, Any] = {
            "decision": decision,
            "reason": reason,
            "fitness": round(fitness, 6),
            "effective_fitness": round(effective_fitness, 6),
            "awareness": computed_awareness,
            "awareness_breakdown": awareness_breakdown,
            "mutation_rate": round(mutation_rate, 6),
            "effective_mutation_rate": round(effective_mr, 6),
            "effective_mutation_delta": round(effective_delta, 6),
            "mutation_rate_offset": round(self._mutation_rate_offset, 6),
            "pending_stimuli": pending_stimuli,
            "stagnation_count": self._stagnation_count,
            "fitness_history_len": len(self._fitness_history),
            "total_decisions": self._total_decisions,
            "energy": round(self._energy, 6),
            "effective_fitness_threshold": round(
                self._effective_fitness_threshold, 6
            ),
            "exploration_mode": self._exploration_mode,
            "recovery_mode": self._recovery_mode,
            "low_energy_consecutive": self._low_energy_consecutive,
            "founder_recovery_mode": founder_recovery,
            "founder_mutate_blocked": founder_mutate_blocked,
            "drift_magnitude": round(drift_magnitude, 6),
            "env_trait_weights": {k: round(v, 4)
                                  for k, v in self._env_trait_weights.items()},
            "idle_cycles": self._idle_cycles,
            "should_entropy_decay": self.should_entropy_decay,
            "organism_died": self._energy <= 0.0,
            "stress_level": round(computed_stress, 6),
            "stress_boosted": stress_boosted,
            "env_modifiers": {k: round(v, 6) for k, v in env.items()} if env else {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if adaptation_reason:
            record["adaptation_reason"] = adaptation_reason

        # 9. Persist to autonomy log
        self._append_log(record)

        logger.info(
            "[AUTONOMY] decision=%s awareness=%.4f fitness=%.4f(eff=%.4f) "
            "energy=%.4f stagnation=%d reason=%s",
            decision, computed_awareness, fitness, effective_fitness,
            self._energy, self._stagnation_count, reason,
        )

        return record

    def summary(self) -> Dict[str, Any]:
        """Return a snapshot of the engine state for /status."""
        variance = 0.0
        if len(self._fitness_history) >= 2:
            variance = statistics.variance(self._fitness_history)
        vital = self.compute_vital_score()
        return {
            "stagnation_count": self._stagnation_count,
            "is_stagnant": self.is_stagnant,
            "total_decisions": self._total_decisions,
            "fitness_history_len": len(self._fitness_history),
            "fitness_variance": round(variance, 8),
            "decision_interval": self._config.decision_interval,
            "fitness_threshold": self._config.fitness_threshold,
            "effective_fitness_threshold": round(
                self._effective_fitness_threshold, 6
            ),
            "stagnation_window": self._config.stagnation_window,
            "awareness": self._awareness_model.last_awareness,
            "total_stimuli": self._awareness_model.total_stimuli,
            "energy": round(self._energy, 6),
            "exploration_mode": self._exploration_mode,
            "recovery_mode": self._recovery_mode,
            "low_energy_consecutive": self._low_energy_consecutive,
            "novelty_accumulator": round(
                self._awareness_model.novelty_accumulator, 6
            ),
            "mutation_rate_offset": round(self._mutation_rate_offset, 6),
            "env_trait_weights": {k: round(v, 4)
                                  for k, v in self._env_trait_weights.items()},
            "idle_cycles": self._idle_cycles,
            "entropy_active": self.should_entropy_decay,
            "vital_score": vital,
        }

    # ── Internal ─────────────────────────────────────────────────────

    def _drift_environment(self) -> float:
        """Apply small random drift to environmental parameters.

        Drifts fitness_threshold, mutation_rate_offset, and per-trait
        environment weights.  Returns the total magnitude of drift
        (for novelty detection).
        """
        cfg = self._config
        step = cfg.drift_step

        old_ft = self._effective_fitness_threshold
        self._effective_fitness_threshold += self._rng.uniform(-step, step)
        self._effective_fitness_threshold = _clamp(
            self._effective_fitness_threshold,
            cfg.drift_fitness_threshold_min,
            cfg.drift_fitness_threshold_max,
        )

        old_offset = self._mutation_rate_offset
        self._mutation_rate_offset += self._rng.uniform(-step * 0.5, step * 0.5)
        self._mutation_rate_offset = _clamp(
            self._mutation_rate_offset, -0.05, 0.05
        )

        # Drift per-trait environment weights
        weight_drift = 0.0
        for trait in list(self._env_trait_weights.keys()):
            old_w = self._env_trait_weights[trait]
            delta_w = self._rng.uniform(
                -cfg.env_weight_drift_step, cfg.env_weight_drift_step
            )
            new_w = _clamp(
                old_w + delta_w,
                cfg.env_weight_min,
                cfg.env_weight_max,
            )
            self._env_trait_weights[trait] = new_w
            weight_drift += abs(new_w - old_w)

        return (abs(self._effective_fitness_threshold - old_ft)
                + abs(self._mutation_rate_offset - old_offset)
                + weight_drift)

    def _apply_energy_delta(self, decision: str, cost_multiplier: float = 1.0) -> None:
        """Adjust energy based on decision type.

        v3.0: cost_multiplier from environment increases mutation/adapt costs.
        Energy can now reach 0.0 (death).
        v3.6: Rebates for mutate/adapt, enhanced stabilize during recovery.
        """
        cfg = self._config
        if decision == DECISION_STABILIZE:
            bonus = cfg.energy_stabilize_bonus
            # v3.6: Extra recovery bonus during recovery mode
            if self._recovery_mode:
                bonus += cfg.recovery_stabilize_bonus
            self._energy += bonus
        elif decision == DECISION_MUTATE:
            self._energy -= cfg.energy_mutate_cost * cost_multiplier
            # v3.6: Small rebate so mutate isn't purely negative
            self._energy += cfg.mutate_energy_rebate
        elif decision == DECISION_ADAPT:
            self._energy -= cfg.energy_adapt_cost * cost_multiplier
            # v3.6: Small rebate for adapt
            self._energy += cfg.adapt_energy_rebate
        elif decision == DECISION_BLEND:
            self._energy -= cfg.energy_blend_cost * cost_multiplier
        # Cap at 1.0, floor at energy_min (default 0.0 → death possible)
        self._energy = max(cfg.energy_min, min(1.0, self._energy))

    def _append_log(self, record: Dict[str, Any]) -> None:
        """Append a decision record to the JSONL log file."""
        try:
            rotate_jsonl(self._log_path)
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception as exc:
            logger.error("[AUTONOMY] Failed to write log: %s", exc)

    def _recover_state(self) -> None:
        """Recover fitness_history, counters, energy, drift,
        and awareness from the existing log file."""
        if not os.path.exists(self._log_path):
            return
        try:
            with open(self._log_path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
            if not lines:
                return

            count = 0
            total_stimuli = 0
            last_awareness = 0.0
            last_energy = self._config.energy_initial
            last_eft = self._config.fitness_threshold
            last_mr_offset = 0.0
            last_novelty = 0.0
            last_env_weights: Optional[Dict[str, float]] = None
            last_recovery_mode: bool = False
            last_low_energy_consecutive: int = 0

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    fitness_val = entry.get("fitness")
                    if fitness_val is not None:
                        self._fitness_history.append(float(fitness_val))
                    ab = entry.get("awareness_breakdown")
                    if ab:
                        ts = ab.get("total_stimuli")
                        if ts is not None:
                            total_stimuli = int(ts)
                        na = ab.get("novelty_accumulator")
                        if na is not None:
                            last_novelty = float(na)
                    aw = entry.get("awareness")
                    if aw is not None:
                        last_awareness = float(aw)
                    en = entry.get("energy")
                    if en is not None:
                        last_energy = float(en)
                    eft = entry.get("effective_fitness_threshold")
                    if eft is not None:
                        last_eft = float(eft)
                    mro = entry.get("mutation_rate_offset")
                    if mro is not None:
                        last_mr_offset = float(mro)
                    etw = entry.get("env_trait_weights")
                    if etw and isinstance(etw, dict):
                        last_env_weights = etw
                    # v3.6: recovery mode state
                    rm = entry.get("recovery_mode")
                    if rm is not None:
                        last_recovery_mode = bool(rm)
                    lec = entry.get("low_energy_consecutive")
                    if lec is not None:
                        last_low_energy_consecutive = int(lec)
                    count += 1
                except (json.JSONDecodeError, ValueError):
                    continue

            # Trim to window
            if len(self._fitness_history) > self._config.stagnation_window:
                self._fitness_history = self._fitness_history[
                    -self._config.stagnation_window:
                ]

            self._total_decisions = count

            # Rebuild stagnation count from tail of log
            stag = 0
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    sc = entry.get("stagnation_count", 0)
                    if sc > 0:
                        stag = sc
                    break
                except (json.JSONDecodeError, ValueError):
                    continue
            self._stagnation_count = stag

            # Restore subsystems
            self._awareness_model.recover(
                total_stimuli, last_awareness, last_novelty
            )
            # v3.4: If recovered energy is at or below 0, grant a survival
            # minimum so the organism doesn't die immediately on boot.
            if last_energy <= 0.0:
                logger.warning(
                    "[AUTONOMY] Recovered energy=%.4f — boosting to survival minimum 0.5",
                    last_energy,
                )
                last_energy = 0.5
            self._energy = last_energy
            self._effective_fitness_threshold = last_eft
            self._mutation_rate_offset = last_mr_offset
            if last_env_weights:
                for k, v in last_env_weights.items():
                    if k in self._env_trait_weights:
                        self._env_trait_weights[k] = float(v)

            # v3.6: Restore recovery mode state
            self._recovery_mode = last_recovery_mode
            self._low_energy_consecutive = last_low_energy_consecutive

            logger.info(
                "[AUTONOMY] Recovered %d decisions, fitness_history=%d, "
                "stagnation=%d, awareness=%.4f, stimuli=%d, energy=%.4f, "
                "eft=%.4f, novelty=%.4f",
                self._total_decisions,
                len(self._fitness_history),
                self._stagnation_count,
                last_awareness,
                total_stimuli,
                last_energy,
                last_eft,
                last_novelty,
            )
        except Exception as exc:
            logger.error("[AUTONOMY] Recovery failed: %s", exc)
