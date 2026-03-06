"""AL-01 v3.0 — Emergent Behavior Detection.

Detects unscripted behavioral strategies and lineage convergence/divergence
by analysing decision history, trait patterns, and population dynamics.

Detected Strategies
-------------------
* **energy-hoarder** — organism consistently stabilises and avoids
  costly mutations; high energy_efficiency trait; energy stays high.
* **explorer** — organism frequently mutates and adapts; high
  adaptability/creativity; energy volatile.
* **specialist** — one or two traits dominate; low trait variance.
* **generalist** — traits are evenly spread; high trait variance.
* **resilient** — maintains fitness despite scarcity events and entropy.

Lineage Metrics
---------------
* **convergence** — trait variance across population decreasing over time.
* **divergence** — trait variance increasing over time (speciation signal).
* **fitness spread** — difference between max and min fitness in population.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("al01.behavior")


# ── Strategy thresholds ──────────────────────────────────────────────────

STRATEGY_HOARDER_STABILIZE_RATIO = 0.6
"""If >60% of recent decisions are 'stabilize', classify as hoarder."""

STRATEGY_EXPLORER_MUTATE_RATIO = 0.4
"""If >40% of recent decisions are 'mutate' or 'adapt', classify as explorer."""

STRATEGY_SPECIALIST_VARIANCE_THRESHOLD = 0.01
"""If trait variance < this, organism is a specialist."""

STRATEGY_GENERALIST_VARIANCE_THRESHOLD = 0.03
"""If trait variance > this, organism is a generalist."""

CONVERGENCE_WINDOW = 20
"""Number of variance samples to track for convergence/divergence detection."""

CONVERGENCE_SLOPE_THRESHOLD = -0.0005
"""Population variance slope below this → convergence."""

DIVERGENCE_SLOPE_THRESHOLD = 0.0005
"""Population variance slope above this → divergence."""


class BehaviorProfile:
    """Tracks behavioral patterns for a single organism."""

    # v3.11: Strategy drift — reclassify every N decisions
    STRATEGY_DRIFT_INTERVAL: int = 50

    def __init__(self, organism_id: str, history_size: int = 50) -> None:
        self.organism_id = organism_id
        self._decision_history: deque = deque(maxlen=history_size)
        self._energy_history: deque = deque(maxlen=history_size)
        self._fitness_history: deque = deque(maxlen=history_size)
        self._strategies_detected: List[str] = []
        self._last_traits: Optional[Dict[str, float]] = None
        # v3.11: Strategy drift tracking
        self._decision_count: int = 0
        self._strategy_history: List[Dict[str, Any]] = []
        self._cached_strategy: Optional[Dict[str, Any]] = None
        self._last_drift_at: int = 0

    def record_decision(
        self,
        decision: str,
        energy: float,
        fitness: float,
        traits: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record a decision cycle outcome."""
        self._decision_history.append(decision)
        self._energy_history.append(energy)
        self._fitness_history.append(fitness)
        if traits:
            self._last_traits = dict(traits)
        self._decision_count += 1

        # v3.11: Strategy drift — reclassify every STRATEGY_DRIFT_INTERVAL decisions
        if self._decision_count - self._last_drift_at >= self.STRATEGY_DRIFT_INTERVAL:
            self._apply_strategy_drift()

    def classify_strategy(self) -> Dict[str, Any]:
        """Classify the organism's current behavioral strategy.

        Returns a dict with strategy names and confidence scores.
        """
        if len(self._decision_history) < 5:
            return {"strategy": "undetermined", "confidence": 0.0, "details": {}}

        total = len(self._decision_history)
        counts: Dict[str, int] = {}
        for d in self._decision_history:
            counts[d] = counts.get(d, 0) + 1

        stabilize_ratio = counts.get("stabilize", 0) / total
        mutate_ratio = (counts.get("mutate", 0) + counts.get("adapt", 0)) / total
        blend_ratio = counts.get("blend", 0) / total

        strategies: List[Tuple[str, float]] = []

        # Energy hoarder
        if stabilize_ratio >= STRATEGY_HOARDER_STABILIZE_RATIO:
            confidence = min(1.0, stabilize_ratio / 0.8)
            strategies.append(("energy-hoarder", confidence))

        # Explorer
        if mutate_ratio >= STRATEGY_EXPLORER_MUTATE_RATIO:
            confidence = min(1.0, mutate_ratio / 0.6)
            strategies.append(("explorer", confidence))

        # Specialist / Generalist based on trait variance
        if self._last_traits and len(self._last_traits) >= 2:
            trait_var = statistics.variance(self._last_traits.values())
            if trait_var < STRATEGY_SPECIALIST_VARIANCE_THRESHOLD:
                strategies.append(("specialist", 1.0 - trait_var / STRATEGY_SPECIALIST_VARIANCE_THRESHOLD))
            elif trait_var > STRATEGY_GENERALIST_VARIANCE_THRESHOLD:
                confidence = min(1.0, trait_var / (STRATEGY_GENERALIST_VARIANCE_THRESHOLD * 3))
                strategies.append(("generalist", confidence))

        # Resilient — fitness stays stable or improves despite pressure
        if len(self._fitness_history) >= 10:
            recent = list(self._fitness_history)[-10:]
            if all(f > 0.3 for f in recent):
                fitness_stability = 1.0 - (max(recent) - min(recent))
                if fitness_stability > 0.7:
                    strategies.append(("resilient", fitness_stability))

        if not strategies:
            return {
                "strategy": "neutral",
                "confidence": 0.5,
                "details": {
                    "stabilize_ratio": round(stabilize_ratio, 3),
                    "mutate_ratio": round(mutate_ratio, 3),
                    "blend_ratio": round(blend_ratio, 3),
                },
            }

        # Primary strategy is the one with highest confidence
        strategies.sort(key=lambda x: x[1], reverse=True)
        primary = strategies[0]
        self._strategies_detected = [s[0] for s in strategies]

        return {
            "strategy": primary[0],
            "confidence": round(primary[1], 3),
            "all_strategies": [
                {"name": s[0], "confidence": round(s[1], 3)}
                for s in strategies
            ],
            "details": {
                "stabilize_ratio": round(stabilize_ratio, 3),
                "mutate_ratio": round(mutate_ratio, 3),
                "blend_ratio": round(blend_ratio, 3),
                "decision_count": total,
            },
        }

    @property
    def strategies(self) -> List[str]:
        return list(self._strategies_detected)

    @property
    def strategy_history(self) -> List[Dict[str, Any]]:
        """v3.11: History of strategy reclassifications."""
        return list(self._strategy_history)

    def _apply_strategy_drift(self) -> None:
        """v3.11: Reclassify strategy based on the most recent trait cluster.

        Called automatically every ``STRATEGY_DRIFT_INTERVAL`` decisions.
        Records the transition for observability.
        """
        old = self._cached_strategy
        new = self.classify_strategy()
        old_name = old["strategy"] if old else "unknown"
        new_name = new["strategy"]
        self._cached_strategy = new
        self._last_drift_at = self._decision_count
        if old_name != new_name:
            self._strategy_history.append({
                "from": old_name,
                "to": new_name,
                "at_decision": self._decision_count,
                "confidence": new.get("confidence", 0.0),
            })
            logger.info(
                "[BEHAVIOR] Strategy drift %s: %s → %s (decision #%d)",
                self.organism_id, old_name, new_name, self._decision_count,
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "organism_id": self.organism_id,
            "decision_history_len": len(self._decision_history),
            "classification": self.classify_strategy(),
            "strategy_drift_history": self._strategy_history[-10:],
            "decision_count": self._decision_count,
        }


class PopulationBehaviorAnalyzer:
    """Tracks lineage convergence/divergence and population-level behavior.

    Call ``record_population_snapshot()`` each cycle with trait data from
    all living organisms.
    """

    def __init__(self, window_size: int = CONVERGENCE_WINDOW) -> None:
        self._profiles: Dict[str, BehaviorProfile] = {}
        self._variance_history: deque = deque(maxlen=window_size)
        # Historical per-trait variance for convergence tracking
        self._trait_variance_history: Dict[str, deque] = {}
        self._fitness_spread_history: deque = deque(maxlen=window_size)

    # ── Per-organism tracking ────────────────────────────────────────

    def get_or_create_profile(self, organism_id: str) -> BehaviorProfile:
        if organism_id not in self._profiles:
            self._profiles[organism_id] = BehaviorProfile(organism_id)
        return self._profiles[organism_id]

    def record_decision(
        self,
        organism_id: str,
        decision: str,
        energy: float,
        fitness: float,
        traits: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record a decision for behavior profiling."""
        profile = self.get_or_create_profile(organism_id)
        profile.record_decision(decision, energy, fitness, traits)

    def remove_organism(self, organism_id: str) -> None:
        """Remove a dead organism's profile."""
        self._profiles.pop(organism_id, None)

    # ── Population-level tracking ────────────────────────────────────

    def record_population_snapshot(
        self,
        population_traits: Dict[str, Dict[str, float]],
        population_fitness: Dict[str, float],
    ) -> Dict[str, Any]:
        """Record trait and fitness data for the whole population.

        Returns convergence/divergence analysis.
        """
        # Compute per-trait variance
        if len(population_traits) >= 2:
            all_trait_names: set = set()
            for traits in population_traits.values():
                all_trait_names.update(traits.keys())

            for trait_name in all_trait_names:
                values = [t.get(trait_name, 0.0) for t in population_traits.values()]
                if len(values) >= 2:
                    var = statistics.variance(values)
                else:
                    var = 0.0
                if trait_name not in self._trait_variance_history:
                    self._trait_variance_history[trait_name] = deque(maxlen=CONVERGENCE_WINDOW)
                self._trait_variance_history[trait_name].append(var)

            # Overall variance (mean of per-trait variances)
            all_vars = []
            for trait_name in all_trait_names:
                values = [t.get(trait_name, 0.0) for t in population_traits.values()]
                if len(values) >= 2:
                    all_vars.append(statistics.variance(values))
            overall_var = statistics.mean(all_vars) if all_vars else 0.0
            self._variance_history.append(overall_var)
        else:
            self._variance_history.append(0.0)

        # Fitness spread
        if population_fitness:
            fitness_values = list(population_fitness.values())
            spread = max(fitness_values) - min(fitness_values)
            self._fitness_spread_history.append(spread)
        else:
            self._fitness_spread_history.append(0.0)

        return self.convergence_analysis()

    def convergence_analysis(self) -> Dict[str, Any]:
        """Analyse whether the population is converging or diverging."""
        if len(self._variance_history) < 3:
            return {
                "status": "insufficient_data",
                "variance_samples": len(self._variance_history),
            }

        variances = list(self._variance_history)
        n = len(variances)

        # Simple linear regression slope
        x_mean = (n - 1) / 2.0
        y_mean = statistics.mean(variances)
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(variances))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator if denominator > 0 else 0.0

        if slope < CONVERGENCE_SLOPE_THRESHOLD:
            status = "converging"
        elif slope > DIVERGENCE_SLOPE_THRESHOLD:
            status = "diverging"
        else:
            status = "stable"

        # Per-trait convergence
        trait_status: Dict[str, str] = {}
        for trait_name, history in self._trait_variance_history.items():
            if len(history) >= 3:
                t_vars = list(history)
                t_n = len(t_vars)
                t_x_mean = (t_n - 1) / 2.0
                t_y_mean = statistics.mean(t_vars)
                t_num = sum((i - t_x_mean) * (v - t_y_mean) for i, v in enumerate(t_vars))
                t_den = sum((i - t_x_mean) ** 2 for i in range(t_n))
                t_slope = t_num / t_den if t_den > 0 else 0.0
                if t_slope < CONVERGENCE_SLOPE_THRESHOLD:
                    trait_status[trait_name] = "converging"
                elif t_slope > DIVERGENCE_SLOPE_THRESHOLD:
                    trait_status[trait_name] = "diverging"
                else:
                    trait_status[trait_name] = "stable"

        return {
            "status": status,
            "slope": round(slope, 8),
            "current_variance": round(variances[-1], 8) if variances else 0.0,
            "variance_samples": n,
            "trait_convergence": trait_status,
            "fitness_spread": round(
                self._fitness_spread_history[-1], 6
            ) if self._fitness_spread_history else 0.0,
        }

    def all_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get behavior classification for all tracked organisms."""
        return {oid: profile.to_dict() for oid, profile in self._profiles.items()}

    def population_strategy_distribution(self) -> Dict[str, int]:
        """Count how many organisms fall into each strategy category."""
        dist: Dict[str, int] = {}
        for profile in self._profiles.values():
            classification = profile.classify_strategy()
            strategy = classification.get("strategy", "unknown")
            dist[strategy] = dist.get(strategy, 0) + 1
        return dist

    def diversity_index(self) -> Dict[str, float]:
        """Compute Shannon and Simpson's diversity indices from strategy distribution.

        Shannon index: H = -Σ p_i * ln(p_i)
        Simpson's index: D = 1 - Σ p_i^2

        Both are 0 when population is monomorphic (single strategy)
        and increase with strategy diversity.
        """
        dist = self.population_strategy_distribution()
        total = sum(dist.values())
        if total == 0:
            return {"shannon": 0.0, "simpson": 0.0, "richness": 0, "evenness": 0.0}

        proportions = [count / total for count in dist.values()]
        richness = len(dist)

        # Shannon index
        shannon = -sum(p * math.log(p) for p in proportions if p > 0)

        # Simpson's index (1 - D form, where D = Σ p_i^2)
        simpson = 1.0 - sum(p ** 2 for p in proportions)

        # Evenness (Shannon / ln(richness)) — how evenly distributed strategies are
        evenness = (shannon / math.log(richness)) if richness > 1 else 0.0

        return {
            "shannon": round(shannon, 6),
            "simpson": round(simpson, 6),
            "richness": richness,
            "evenness": round(evenness, 6),
        }

    def summary(self) -> Dict[str, Any]:
        """Full behavior analysis summary."""
        return {
            "organism_profiles": self.all_profiles(),
            "strategy_distribution": self.population_strategy_distribution(),
            "diversity": self.diversity_index(),
            "convergence": self.convergence_analysis(),
            "tracked_organisms": len(self._profiles),
        }
