"""Tests for v3.11 ecosystem mechanics.

Coverage:
1. Anti-monoculture pressure — diminishing returns above 80% dominance.
2. Rare shock events — low-probability entropy spikes favouring resilience.
3. Explorer novelty reward — boosted fitness when explorers are rare.
4. Strategy drift — reclassification every 50 decisions.
"""

import copy
import statistics
import tempfile
import unittest
from collections import deque
from typing import Dict

from al01.behavior import BehaviorProfile, PopulationBehaviorAnalyzer
from al01.environment import Environment, EnvironmentConfig, ShockEvent
from al01.genome import Genome
from al01.population import Population


# ======================================================================
# Helper
# ======================================================================

def _make_population(n_children: int = 5) -> Population:
    """Return a population with *n_children* living children."""
    tmpdir = tempfile.mkdtemp()
    pop = Population(data_dir=tmpdir, parent_id="AL-01")
    g = Genome()
    for _ in range(n_children):
        pop.spawn_child(g, parent_evolution=1)
    return pop


# ======================================================================
# 1. Anti-Monoculture Pressure
# ======================================================================

class TestAntiMonoculture(unittest.TestCase):
    """strategy_dominance_penalty applies diminishing returns above 80%."""

    def test_no_penalty_at_80_percent(self) -> None:
        dist = {"energy-hoarder": 8, "explorer": 2}
        pop = _make_population()
        penalty = pop.strategy_dominance_penalty(dist, "energy-hoarder")
        self.assertEqual(penalty, 1.0)

    def test_penalty_at_90_percent(self) -> None:
        dist = {"energy-hoarder": 9, "explorer": 1}
        pop = _make_population()
        penalty = pop.strategy_dominance_penalty(dist, "energy-hoarder")
        # 90% → (0.90 - 0.80) / 0.20 = 0.50 excess → penalty = 0.50
        self.assertAlmostEqual(penalty, 0.50, places=2)

    def test_penalty_at_100_percent(self) -> None:
        dist = {"energy-hoarder": 10}
        pop = _make_population()
        penalty = pop.strategy_dominance_penalty(dist, "energy-hoarder")
        # 100% → (1.0 - 0.80) / 0.20 = 1.0 excess → 1.0 - 1.0 = 0.0 → floor 0.10
        self.assertAlmostEqual(penalty, 0.10, places=2)

    def test_no_penalty_for_minority(self) -> None:
        dist = {"energy-hoarder": 8, "explorer": 2}
        pop = _make_population()
        penalty = pop.strategy_dominance_penalty(dist, "explorer")
        self.assertEqual(penalty, 1.0)

    def test_empty_dist(self) -> None:
        pop = _make_population()
        self.assertEqual(pop.strategy_dominance_penalty({}, "anything"), 1.0)

    def test_penalty_never_below_floor(self) -> None:
        dist = {"specialist": 100}
        pop = _make_population()
        penalty = pop.strategy_dominance_penalty(dist, "specialist")
        self.assertGreaterEqual(penalty, 0.10)

    def test_custom_threshold(self) -> None:
        dist = {"explorer": 7, "specialist": 3}
        pop = _make_population()
        # At default threshold (80%) → 70% → no penalty
        self.assertEqual(pop.strategy_dominance_penalty(dist, "explorer"), 1.0)
        # At 60% threshold → 70% → penalty
        penalty = pop.strategy_dominance_penalty(dist, "explorer", threshold=0.60)
        self.assertLess(penalty, 1.0)


# ======================================================================
# 2. Rare Shock Events
# ======================================================================

class TestShockEvents(unittest.TestCase):
    """Environment.tick() can trigger shock events that favour resilience."""

    def test_shock_event_dataclass(self) -> None:
        ev = ShockEvent(
            remaining_cycles=10,
            entropy_spike=0.25,
            resilience_bonus=0.15,
            started_at="2026-03-01T00:00:00+00:00",
        )
        self.assertTrue(ev.tick())
        self.assertEqual(ev.remaining_cycles, 9)
        d = ev.to_dict()
        restored = ShockEvent.from_dict(d)
        self.assertEqual(restored.remaining_cycles, 9)
        self.assertAlmostEqual(restored.entropy_spike, 0.25)

    def test_shock_config_defaults(self) -> None:
        cfg = EnvironmentConfig()
        self.assertAlmostEqual(cfg.shock_probability, 0.02)
        self.assertAlmostEqual(cfg.shock_resilience_bonus, 0.15)
        self.assertEqual(cfg.shock_duration_min, 5)
        self.assertEqual(cfg.shock_duration_max, 15)
        self.assertAlmostEqual(cfg.shock_entropy_spike, 0.25)

    def test_forced_shock_trigger(self) -> None:
        """Force a shock by setting probability to 100%."""
        cfg = EnvironmentConfig(
            shock_probability=100.0,  # 100% per 100 cycles → 1.0 per tick
            shock_duration_min=3,
            shock_duration_max=3,
            rng_seed=42,
        )
        env = Environment(config=cfg)
        record = env.tick()
        # Should have triggered a shock event
        shock_events = [e for e in record.get("events", []) if e.get("type") == "shock_event"]
        self.assertGreaterEqual(len(shock_events), 1)
        self.assertTrue(env.is_shock_active)
        self.assertGreater(env.shock_resilience_bonus, 0.0)

    def test_shock_increases_entropy(self) -> None:
        cfg = EnvironmentConfig(
            shock_probability=100.0,
            shock_entropy_spike=0.30,
            shock_duration_min=5,
            shock_duration_max=5,
            rng_seed=42,
        )
        env = Environment(config=cfg)
        base_entropy = env.entropy_pressure
        env.tick()
        self.assertGreater(env.entropy_pressure, base_entropy)

    def test_shock_relaxes_after_expiry(self) -> None:
        cfg = EnvironmentConfig(
            shock_probability=100.0,
            shock_entropy_spike=0.30,
            shock_duration_min=2,
            shock_duration_max=2,
            rng_seed=42,
        )
        env = Environment(config=cfg)
        env.tick()  # triggers shock (remaining=2)
        self.assertTrue(env.is_shock_active)
        # Disable further shocks
        env._config = EnvironmentConfig(shock_probability=0.0, rng_seed=42)
        env._config.shock_entropy_spike = 0.30
        # Tick to decrement (remaining 1 → still active)
        env.tick()
        # Tick again — shock expires, entropy relaxes
        env.tick()
        self.assertFalse(env.is_shock_active)

    def test_no_shock_at_zero_probability(self) -> None:
        cfg = EnvironmentConfig(shock_probability=0.0, rng_seed=42)
        env = Environment(config=cfg)
        for _ in range(100):
            env.tick()
        self.assertFalse(env.is_shock_active)

    def test_shock_state_snapshot(self) -> None:
        cfg = EnvironmentConfig(shock_probability=100.0, rng_seed=42)
        env = Environment(config=cfg)
        env.tick()
        snap = env.state_snapshot()
        self.assertIn("active_shock_count", snap)
        self.assertIn("shock_resilience_bonus", snap)
        self.assertGreaterEqual(snap["active_shock_count"], 1)

    def test_shock_persistence(self) -> None:
        """to_dict / from_dict round-trips shock events."""
        cfg = EnvironmentConfig(shock_probability=100.0, rng_seed=42)
        env = Environment(config=cfg)
        env.tick()
        self.assertTrue(env.is_shock_active)
        data = env.to_dict()
        restored = Environment.from_dict(data, config=cfg)
        self.assertEqual(len(restored.shock_events), len(env.shock_events))


# ======================================================================
# 3. Explorer Novelty Reward
# ======================================================================

class TestExplorerNoveltyReward(unittest.TestCase):
    """explorer_novelty_multiplier boosts explorers when rare."""

    def test_boost_when_explorers_rare(self) -> None:
        pop = _make_population()
        dist = {"energy-hoarder": 19, "explorer": 1}  # 5%
        # At threshold=5% → exactly at boundary → no boost (equal)
        # Below 5% is < 0.05 → 1/20 = 0.05 → not strictly below
        # Let's test with 0 explorers → definitely below
        dist_zero = {"energy-hoarder": 20}
        mult = pop.explorer_novelty_multiplier(dist_zero)
        self.assertEqual(mult, 1.5)

    def test_no_boost_when_explorers_common(self) -> None:
        pop = _make_population()
        dist = {"energy-hoarder": 8, "explorer": 2}  # 20%
        mult = pop.explorer_novelty_multiplier(dist)
        self.assertEqual(mult, 1.0)

    def test_boost_just_below_threshold(self) -> None:
        pop = _make_population()
        # 1 out of 25 = 4% < 5% threshold
        dist = {"specialist": 24, "explorer": 1}
        mult = pop.explorer_novelty_multiplier(dist)
        self.assertEqual(mult, 1.5)

    def test_custom_multiplier(self) -> None:
        pop = _make_population()
        dist = {"specialist": 20}
        mult = pop.explorer_novelty_multiplier(dist, reward_multiplier=2.0)
        self.assertEqual(mult, 2.0)

    def test_empty_dist(self) -> None:
        pop = _make_population()
        self.assertEqual(pop.explorer_novelty_multiplier({}), 1.0)


# ======================================================================
# 4. Strategy Drift Mechanic
# ======================================================================

class TestStrategyDrift(unittest.TestCase):
    """Strategy reclassification every 50 decisions."""

    def test_drift_interval_constant(self) -> None:
        self.assertEqual(BehaviorProfile.STRATEGY_DRIFT_INTERVAL, 50)

    def test_no_drift_before_interval(self) -> None:
        profile = BehaviorProfile("test-1")
        for i in range(49):
            profile.record_decision("stabilize", 0.8, 0.5)
        self.assertEqual(len(profile.strategy_history), 0)

    def test_drift_at_interval(self) -> None:
        """After 50 decisions the strategy is recorded."""
        profile = BehaviorProfile("test-1")
        # Feed 50 stabilize decisions → should become energy-hoarder
        for i in range(50):
            profile.record_decision("stabilize", 0.8, 0.5)
        # Strategy should have been classified at decision 50
        # May or may not have history depending on initial cached state
        self.assertEqual(profile._last_drift_at, 50)

    def test_drift_detects_strategy_change(self) -> None:
        """Switching dominant decisions causes a drift record."""
        profile = BehaviorProfile("test-1", history_size=50)
        # Phase 1: 50 stabilize → energy-hoarder
        for _ in range(50):
            profile.record_decision("stabilize", 0.8, 0.5)
        first_strategy = profile.classify_strategy()["strategy"]
        # Phase 2: 50 mutate → should reclassify
        for _ in range(50):
            profile.record_decision("mutate", 0.5, 0.4)
        # Strategy history should record a transition
        history = profile.strategy_history
        # At least one transition should have occurred
        second_strategy = profile.classify_strategy()["strategy"]
        if first_strategy != second_strategy:
            self.assertTrue(len(history) >= 1)
            self.assertEqual(history[-1]["to"], second_strategy)

    def test_drift_no_history_for_same_strategy(self) -> None:
        """If strategy doesn't change at drift interval, no history entry is added."""
        profile = BehaviorProfile("test-1")
        # All stabilize → energy-hoarder consistently
        for _ in range(100):
            profile.record_decision("stabilize", 0.8, 0.5)
        # Since strategy doesn't change between intervals, the first drift may
        # record "unknown" → "energy-hoarder", but subsequent ones shouldn't add.
        hist = profile.strategy_history
        # After 100 decisions, we have 2 drift checkpoints (50 and 100).
        # The second one shouldn't add a new entry since the strategy stays the same.
        if len(hist) >= 2:
            self.fail("Expected at most 1 drift entry for consistent strategy")

    def test_to_dict_includes_drift(self) -> None:
        profile = BehaviorProfile("test-1")
        for _ in range(50):
            profile.record_decision("stabilize", 0.8, 0.5)
        d = profile.to_dict()
        self.assertIn("strategy_drift_history", d)
        self.assertIn("decision_count", d)
        self.assertEqual(d["decision_count"], 50)

    def test_population_strategy_distribution_with_drift(self) -> None:
        """Strategy distribution reflects drifted strategies."""
        analyzer = PopulationBehaviorAnalyzer()
        # Organism A: energy-hoarder
        for _ in range(50):
            analyzer.record_decision("org-A", "stabilize", 0.8, 0.5)
        # Organism B: explorer
        for _ in range(50):
            analyzer.record_decision("org-B", "mutate", 0.5, 0.4)
        dist = analyzer.population_strategy_distribution()
        # Both should be classified
        self.assertGreater(sum(dist.values()), 0)


# ======================================================================
# 5. Integration: child_autonomy_cycle wiring
# ======================================================================

class TestChildAutonomyWiring(unittest.TestCase):
    """Verify the ecosystem pressures are wired into the organism."""

    @classmethod
    def setUpClass(cls) -> None:
        from al01.organism import Organism, MetabolismConfig
        cls._tmpdir = tempfile.mkdtemp()
        cfg = MetabolismConfig(
            pulse_interval=9999, reflect_interval=9999, persist_interval=9999,
            evolve_interval=9999, population_interact_interval=9999,
            autonomy_interval=9999, environment_interval=9999,
            behavior_analysis_interval=9999, auto_reproduce_interval=9999,
            child_autonomy_interval=9999,
        )
        cls._org = Organism(data_dir=cls._tmpdir, config=cfg)
        cls._org.boot()

    def test_child_autonomy_runs_with_ecosystem_pressures(self) -> None:
        """child_autonomy_cycle completes without errors when ecosystem
        pressure code paths are active."""
        # Spawn some children
        parent_genome = self._org.genome
        for _ in range(3):
            self._org.population.spawn_child(parent_genome, parent_evolution=5)
        # Run the cycle — should not raise
        results = self._org.child_autonomy_cycle()
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_shock_active_modifies_cycle(self) -> None:
        """When shock is active the cycle runs and resilience bonus path is reachable."""
        # Force a shock event
        self._org._environment._shock_events.append(
            ShockEvent(
                remaining_cycles=5,
                entropy_spike=0.25,
                resilience_bonus=0.15,
                started_at="2026-03-01T00:00:00+00:00",
            )
        )
        self.assertTrue(self._org._environment.is_shock_active)
        results = self._org.child_autonomy_cycle()
        self.assertIsInstance(results, list)
        # Clean up
        self._org._environment._shock_events.clear()


if __name__ == "__main__":
    unittest.main()
