"""AL-01 v2.3 — Genome: open-ended trait evolution with soft ceilings.

Traits have NO hard cap at 1.0.  A soft ceiling applies diminishing returns
above 1.0 via a logarithmic squash: effective = 1 + ln(raw) * scale.
Fitness is environment-weighted, traits decay via entropy, and trade-offs
create optimisation tension between adaptability/efficiency and creativity/cost.
"""

from __future__ import annotations

import copy
import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("al01.genome")

# Default starting traits
DEFAULT_TRAITS: Dict[str, float] = {
    "adaptability": 0.5,
    "energy_efficiency": 0.5,
    "resilience": 0.5,
    "perception": 0.5,
    "creativity": 0.5,
}

# Mutation parameters
DEFAULT_MUTATION_RATE: float = 0.10  # 10% chance per trait
DEFAULT_MUTATION_DELTA: float = 0.10  # max +/- shift per mutation

# Soft ceiling — diminishing returns above 1.0
SOFT_CEILING_SCALE: float = 0.3  # how much log(raw) contributes above 1.0

# Trait trade-off coefficients
#   high adaptability   → reduces energy_efficiency
#   high creativity     → increases energy cost (reduces energy_efficiency)
TRADEOFF_RULES: List[Tuple[str, str, float]] = [
    # (source_trait, target_trait, penalty_per_unit_above_threshold)
    ("adaptability", "energy_efficiency", 0.08),
    ("creativity",   "energy_efficiency", 0.06),
]
TRADEOFF_THRESHOLD: float = 0.7  # trade-off kicks in above this value

# Trait floor — no trait can be pushed below this value
TRAIT_FLOOR: float = 0.02

# Entropy — trait decay per cycle with no stimulus
DEFAULT_ENTROPY_RATE: float = 0.005


class Genome:
    """Mutable trait map with mutation, fitness, and reproduction helpers.

    v2.3: soft ceiling, trade-offs, entropy, environment-weighted fitness,
    independent child RNG seeds.
    """

    # v3.10: Multi-objective fitness weights (4 external components)
    MO_WEIGHTS: Dict[str, float] = {
        "survival": 0.35,
        "efficiency": 0.30,
        "stability": 0.20,
        "adaptation": 0.15,
    }

    def __init__(
        self,
        traits: Optional[Dict[str, float]] = None,
        mutation_rate: float = DEFAULT_MUTATION_RATE,
        mutation_delta: float = DEFAULT_MUTATION_DELTA,
        rng_seed: Optional[int] = None,
    ) -> None:
        self._traits: Dict[str, float] = {}
        src = traits if traits else DEFAULT_TRAITS
        for k, v in src.items():
            self._traits[k] = _soft_floor(float(v))  # floor only, no ceiling
        self._mutation_rate = mutation_rate
        self._mutation_delta = mutation_delta
        # Independent RNG — children get their own seed
        self._rng = random.Random(rng_seed)
        # v3.10: Multi-objective fitness components (set externally)
        self._fitness_components: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def traits(self) -> Dict[str, float]:
        """Raw trait values (may exceed 1.0)."""
        return dict(self._traits)

    @property
    def effective_traits(self) -> Dict[str, float]:
        """Trait values after soft-ceiling diminishing returns."""
        return {k: _soft_cap(v) for k, v in self._traits.items()}

    @property
    def trait_fitness(self) -> float:
        """Trait-based fitness = average of soft-capped trait values.

        v3.10: This is the original fitness formula, preserved for
        backward compatibility and for use by multi_objective_fitness().
        """
        if not self._traits:
            return 0.0
        capped = [_soft_cap(v) for v in self._traits.values()]
        return sum(capped) / len(capped)

    @property
    def fitness(self) -> float:
        """Primary fitness score.

        v3.10: When fitness_components have been set (by the organism
        lifecycle), returns a multi-objective weighted score:
            0.35 * survival + 0.30 * efficiency + 0.20 * stability + 0.15 * adaptation
        When components have NOT been set (fresh genome, tests), falls
        back to the old trait-average formula for backward compatibility.
        """
        if self._fitness_components is not None:
            w = self.MO_WEIGHTS
            return sum(
                self._fitness_components.get(k, 0.0) * wt
                for k, wt in w.items()
            )
        return self.trait_fitness

    def set_fitness_components(
        self,
        survival: float,
        efficiency: float,
        stability: float,
        adaptation: float,
    ) -> None:
        """Inject external lifecycle metrics for multi-objective fitness.

        All values are expected in [0, 1].  Clamped for safety.
        """
        self._fitness_components = {
            "survival": max(0.0, min(1.0, survival)),
            "efficiency": max(0.0, min(1.0, efficiency)),
            "stability": max(0.0, min(1.0, stability)),
            "adaptation": max(0.0, min(1.0, adaptation)),
        }

    @property
    def fitness_components(self) -> Optional[Dict[str, float]]:
        """Return the current multi-objective fitness breakdown, or None."""
        if self._fitness_components is None:
            return None
        return dict(self._fitness_components)

    def weighted_fitness(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Environment-weighted fitness.

        *weights* maps trait name → weight (default 1.0 each).
        Fitness = weighted average of soft-capped values.
        """
        if not self._traits:
            return 0.0
        w = weights or {}
        total_w = 0.0
        total_v = 0.0
        for k, v in self._traits.items():
            tw = w.get(k, 1.0)
            total_w += tw
            total_v += _soft_cap(v) * tw
        return total_v / total_w if total_w > 0 else 0.0

    def multi_objective_fitness(
        self,
        survival_time: float = 0.0,
        energy_efficiency_ratio: float = 0.0,
        stability_score: float = 0.0,
        adaptation_success_rate: float = 0.0,
        objective_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Multi-objective fitness combining trait fitness with external metrics.

        v3.9: Real organisms don't optimise one thing.  This blends:
        * trait_fitness — average of soft-capped genome traits (0–1)
        * survival_time — normalised survival duration (0–1)
        * energy_efficiency_ratio — energy spent vs value gained (0–1)
        * stability_score — identity persistence over time (0–1)
        * adaptation_success_rate — fraction of successful adaptations (0–1)

        *objective_weights* maps component name → weight.
        Default weights if not supplied.

        Returns dict with per-component scores + combined multi_fitness.
        """
        defaults: Dict[str, float] = {
            "trait_fitness": 0.40,
            "survival_time": 0.20,
            "energy_efficiency": 0.15,
            "stability": 0.10,
            "adaptation": 0.15,
        }
        w = objective_weights if objective_weights else defaults

        components: Dict[str, float] = {
            "trait_fitness": self.trait_fitness,
            "survival_time": max(0.0, min(1.0, survival_time)),
            "energy_efficiency": max(0.0, min(1.0, energy_efficiency_ratio)),
            "stability": max(0.0, min(1.0, stability_score)),
            "adaptation": max(0.0, min(1.0, adaptation_success_rate)),
        }

        total_w = sum(w.get(k, 0.0) for k in components)
        if total_w <= 0:
            total_w = 1.0
        multi_fitness = sum(
            components[k] * w.get(k, 0.0) for k in components
        ) / total_w

        return {
            "multi_fitness": round(multi_fitness, 6),
            "components": {k: round(v, 6) for k, v in components.items()},
            "weights": {k: round(w.get(k, 0.0), 4) for k in components},
        }

    @property
    def mutation_rate(self) -> float:
        return self._mutation_rate

    def mutate(self, delta_override: Optional[float] = None,
               upward_bias: float = 0.0) -> Dict[str, Any]:
        """Apply random mutations.  Returns a record of what changed.

        *delta_override* allows the autonomy engine to inject a
        stagnation-scaled delta.
        *upward_bias* (v3.6) adds a positive offset to break zero-mean
        symmetry when the organism needs to recover fitness.
        """
        effective_delta = delta_override if delta_override is not None else self._mutation_delta
        changes: Dict[str, Dict[str, float]] = {}
        for trait in list(self._traits.keys()):
            if self._rng.random() < self._mutation_rate:
                old = self._traits[trait]
                delta = self._rng.uniform(-effective_delta, effective_delta)
                # v3.6: add upward bias to break zero-mean symmetry
                delta += upward_bias
                new = _soft_floor(old + delta)  # floor at 0, no ceiling
                self._traits[trait] = new
                changes[trait] = {"old": round(old, 6), "new": round(new, 6), "delta": round(delta, 6)}
        # Apply trade-offs after mutation
        tradeoff_effects = self._apply_tradeoffs()
        if changes:
            logger.info("[GENOME] Mutated %d trait(s): %s", len(changes), list(changes.keys()))
        return {
            "mutated_traits": changes,
            "fitness": round(self.fitness, 6),
            "tradeoff_effects": tradeoff_effects,
        }

    def set_trait(self, name: str, value: float) -> None:
        self._traits[name] = _soft_floor(float(value))

    def get_trait(self, name: str) -> float:
        return self._traits.get(name, 0.0)

    def decay_traits(self, rate: float = DEFAULT_ENTROPY_RATE) -> Dict[str, Dict[str, float]]:
        """Entropy: all traits decay slightly toward 0.  Returns changes."""
        changes: Dict[str, Dict[str, float]] = {}
        for trait in list(self._traits.keys()):
            old = self._traits[trait]
            if old > 0.0:
                loss = old * rate  # proportional decay
                new = _soft_floor(old - loss)
                self._traits[trait] = new
                if abs(old - new) > 1e-8:
                    changes[trait] = {"old": round(old, 6), "new": round(new, 6), "decay": round(loss, 6)}
        if changes:
            logger.debug("[GENOME] Entropy decay: %d trait(s) decayed", len(changes))
        return changes

    def spawn_child(self, variance: float = 0.05) -> "Genome":
        """Create a child genome with an *independent* RNG seed.

        The child gets its own mutation trajectory separate from the parent.
        """
        child_seed = self._rng.randint(0, 2**31)
        child_rng = random.Random(child_seed)
        child_traits: Dict[str, float] = {}
        for k, v in self._traits.items():
            delta = child_rng.uniform(-variance, variance)
            child_traits[k] = _soft_floor(v + delta)
        # Child gets slightly different mutation params
        child_mr = max(0.01, self._mutation_rate + child_rng.uniform(-0.02, 0.02))
        child_md = max(0.01, self._mutation_delta + child_rng.uniform(-0.02, 0.02))
        child = Genome(
            traits=child_traits,
            mutation_rate=child_mr,
            mutation_delta=child_md,
            rng_seed=child_seed,
        )
        logger.info("[GENOME] Spawned child genome (fitness=%.4f, seed=%d)", child.fitness, child_seed)
        return child

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "traits": {k: round(v, 6) for k, v in self._traits.items()},
            "effective_traits": {k: round(v, 6) for k, v in self.effective_traits.items()},
            "fitness": round(self.fitness, 6),
            "trait_fitness": round(self.trait_fitness, 6),
            "mutation_rate": self._mutation_rate,
            "mutation_delta": self._mutation_delta,
        }
        if self._fitness_components is not None:
            d["fitness_components"] = {
                k: round(v, 6) for k, v in self._fitness_components.items()
            }
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Genome":
        """Reconstruct a Genome from a serialized dict."""
        traits = data.get("traits", DEFAULT_TRAITS)
        g = cls(
            traits=traits,
            mutation_rate=data.get("mutation_rate", DEFAULT_MUTATION_RATE),
            mutation_delta=data.get("mutation_delta", DEFAULT_MUTATION_DELTA),
        )
        # Restore fitness components if present
        fc = data.get("fitness_components")
        if fc and isinstance(fc, dict):
            g.set_fitness_components(
                survival=fc.get("survival", 0.0),
                efficiency=fc.get("efficiency", 0.0),
                stability=fc.get("stability", 0.0),
                adaptation=fc.get("adaptation", 0.0),
            )
        return g

    def transfer_energy(self, other: "Genome", amount: float = 0.02) -> Dict[str, Any]:
        """Transfer energy_efficiency between two genomes (inter-organism interaction)."""
        my_energy = self._traits.get("energy_efficiency", 0.5)
        their_energy = other._traits.get("energy_efficiency", 0.5)

        # Move a small amount from the higher to the lower
        if my_energy > their_energy:
            self._traits["energy_efficiency"] = _soft_floor(my_energy - amount)
            other._traits["energy_efficiency"] = _soft_floor(their_energy + amount)
            direction = "gave"
        else:
            self._traits["energy_efficiency"] = _soft_floor(my_energy + amount)
            other._traits["energy_efficiency"] = _soft_floor(their_energy - amount)
            direction = "received"

        return {
            "direction": direction,
            "amount": amount,
            "self_energy": round(self._traits["energy_efficiency"], 6),
            "other_energy": round(other._traits["energy_efficiency"], 6),
        }

    def blend_with(
        self,
        other: "Genome",
        blend_factor: float = 0.5,
        noise: float = 0.02,
    ) -> Dict[str, Any]:
        """Cooperatively blend traits toward *other*'s values.

        Non-lethal evolution — both genomes survive.  ``blend_factor``
        controls how much of *self* to keep (1.0 = no change, 0.5 = even
        average).
        """
        changes: Dict[str, Dict[str, float]] = {}
        for trait in list(self._traits.keys()):
            my_val = self._traits[trait]
            their_val = other._traits.get(trait, my_val)
            blended = my_val * blend_factor + their_val * (1 - blend_factor)
            delta = self._rng.uniform(-noise, noise)
            new_val = _soft_floor(blended + delta)
            changes[trait] = {"old": round(my_val, 6), "new": round(new_val, 6)}
            self._traits[trait] = new_val
        # Apply trade-offs after blending
        tradeoff_effects = self._apply_tradeoffs()
        return {
            "blended_traits": changes,
            "fitness": round(self.fitness, 6),
            "tradeoff_effects": tradeoff_effects,
        }

    # ------------------------------------------------------------------
    # Trade-off mechanics
    # ------------------------------------------------------------------

    def _apply_tradeoffs(self) -> Dict[str, Dict[str, float]]:
        """Apply trade-off penalties: high source_trait penalises target_trait.

        Returns dict of adjustments made.
        """
        effects: Dict[str, Dict[str, float]] = {}
        for source, target, penalty in TRADEOFF_RULES:
            src_val = self._traits.get(source, 0.0)
            if src_val > TRADEOFF_THRESHOLD and target in self._traits:
                excess = src_val - TRADEOFF_THRESHOLD
                old_target = self._traits[target]
                # Damping: reduce penalty as target approaches floor
                # to prevent trade-offs from crushing a trait to zero
                headroom = max(0.0, old_target - TRAIT_FLOOR)
                damping = min(1.0, headroom / 0.15) if headroom < 0.15 else 1.0
                reduction = excess * penalty * damping
                new_target = _soft_floor(old_target - reduction)
                self._traits[target] = new_target
                if abs(old_target - new_target) > 1e-8:
                    effects[f"{source}->{target}"] = {
                        "source_val": round(src_val, 6),
                        "excess": round(excess, 6),
                        "reduction": round(reduction, 6),
                        "damping": round(damping, 6),
                        "target_old": round(old_target, 6),
                        "target_new": round(new_target, 6),
                    }
        return effects


    # ------------------------------------------------------------------
    # Distance / Species
    # ------------------------------------------------------------------

    def distance(self, other: "Genome") -> float:
        """Euclidean distance between two genomes in trait-space.

        Uses *effective* (soft-capped) traits so the metric reflects actual
        phenotypic difference rather than raw unbounded values.
        """
        a = self.effective_traits
        b = other.effective_traits
        all_keys = sorted(set(a) | set(b))
        return math.sqrt(sum((a.get(k, 0.0) - b.get(k, 0.0)) ** 2 for k in all_keys))


def genome_distance(a: Genome, b: Genome) -> float:
    """Module-level convenience: Euclidean distance between two Genomes."""
    return a.distance(b)


# ── Module-level helpers ─────────────────────────────────────────────────

def _soft_cap(v: float) -> float:
    """Soft ceiling with diminishing returns above 1.0.

    Below 1.0 → identity.  Above 1.0 → 1.0 + ln(v) * SOFT_CEILING_SCALE.
    This means pushing a trait from 1.0 to 2.718 only yields ~0.3 extra.
    Floor is always 0.0.
    """
    if v <= 0.0:
        return 0.0
    if v <= 1.0:
        return v
    return 1.0 + math.log(v) * SOFT_CEILING_SCALE


def _soft_floor(v: float, lo: float = TRAIT_FLOOR) -> float:
    """Floor at TRAIT_FLOOR — no trait reaches zero.  No ceiling."""
    return max(lo, v)


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Legacy clamp — kept for external callers only."""
    return max(lo, min(hi, v))
