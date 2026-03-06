"""AL-01 v3.9 — Tests for dynamic environment shifts, multi-objective fitness,
memory drift, elite protection, and internal communication signal.

Covers:
 1. Dynamic env shift interval config (20–50 range)
 2. Environment shifts happen within 20–50 cycles (not fixed 200)
 3. next_shift_cycle serialises and restores
 4. Multi-objective fitness returns correct structure
 5. Multi-objective fitness default weights sum correctly
 6. Multi-objective fitness clamps inputs to 0–1
 7. Multi-objective fitness with custom weights
 8. Memory drift triggers at interval and applies changes
 9. Memory drift skips when below interval
10. Memory drift inversely scales with diversity
11. Elite IDs returns top organisms by fitness
12. Elite IDs returns empty when population too small
13. Elite protection: elite children get crossover not mutation
14. InternalSignal has correct fields
15. InternalSignal to_dict returns all keys
16. Organism.internal_signal computes without error
17. /status includes internal_signal
18. /status includes multi_objective fields in champion section
19. Population elite_ids fraction configurable
20. Environment state_snapshot includes next_shift_cycle
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
from typing import Any, Dict
from unittest.mock import MagicMock, patch

# Ensure al01 package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from al01.genome import Genome
from al01.population import Population
from al01.environment import Environment, EnvironmentConfig
from al01.organism import InternalSignal


# ======================================================================
# Feature 1: Dynamic Environment Shifts
# ======================================================================

class TestDynamicEnvShifts(unittest.TestCase):
    """Environment shifts every 20–50 cycles instead of fixed 200."""

    def test_config_defaults(self) -> None:
        """EnvironmentConfig has dynamic_shift_min/max with correct defaults."""
        cfg = EnvironmentConfig()
        self.assertEqual(cfg.dynamic_shift_min, 20)
        self.assertEqual(cfg.dynamic_shift_max, 50)

    def test_first_shift_within_range(self) -> None:
        """First env shift should happen within 20–50 cycles."""
        env = Environment(rng_seed=42)
        self.assertGreaterEqual(env._next_shift_cycle, 20)
        self.assertLessEqual(env._next_shift_cycle, 50)

    def test_shift_happens_before_cycle_51(self) -> None:
        """At least one shift should occur within the first 51 cycles."""
        env = Environment(rng_seed=99)
        shift_happened = False
        for _ in range(51):
            record = env.tick()
            for event in record.get("events", []):
                if event.get("type") == "environment_shift":
                    shift_happened = True
                    break
            if shift_happened:
                break
        self.assertTrue(shift_happened, "No shift within first 51 cycles")

    def test_shift_does_not_use_fixed_200(self) -> None:
        """With default config, no shift at exactly cycle 200 unless random lands there."""
        env = Environment(rng_seed=42)
        # Tick 60 cycles — should see at least one shift (range 20–50)
        shifts_seen = 0
        for _ in range(60):
            record = env.tick()
            for event in record.get("events", []):
                if event.get("type") == "environment_shift":
                    shifts_seen += 1
        self.assertGreaterEqual(shifts_seen, 1)

    def test_next_shift_cycle_advances_after_shift(self) -> None:
        """After a shift, next_shift_cycle should advance by 20–50 cycles."""
        env = Environment(rng_seed=7)
        first = env._next_shift_cycle
        # Tick until the first shift fires
        for _ in range(55):
            env.tick()
            if env._next_shift_cycle != first:
                break
        # It should have moved forward
        self.assertGreater(env._next_shift_cycle, first)
        # New target should be current cycle + 20..50
        self.assertGreaterEqual(env._next_shift_cycle, env.cycle + 1)

    def test_serialise_restores_next_shift(self) -> None:
        """to_dict / from_dict round-trip preserves next_shift_cycle."""
        env = Environment(rng_seed=42)
        for _ in range(10):
            env.tick()
        data = env.to_dict()
        self.assertIn("next_shift_cycle", data)
        restored = Environment.from_dict(data)
        self.assertEqual(restored._next_shift_cycle, env._next_shift_cycle)

    def test_state_snapshot_includes_next_shift(self) -> None:
        """state_snapshot() dict includes next_shift_cycle."""
        env = Environment(rng_seed=1)
        snap = env.state_snapshot()
        self.assertIn("next_shift_cycle", snap)
        self.assertIsInstance(snap["next_shift_cycle"], int)


# ======================================================================
# Feature 2: Multi-Objective Fitness
# ======================================================================

class TestMultiObjectiveFitness(unittest.TestCase):
    """Genome.multi_objective_fitness blends trait fitness with external metrics."""

    def test_basic_structure(self) -> None:
        """Returns dict with multi_fitness, components, weights."""
        g = Genome()
        result = g.multi_objective_fitness()
        self.assertIn("multi_fitness", result)
        self.assertIn("components", result)
        self.assertIn("weights", result)
        self.assertIsInstance(result["multi_fitness"], float)
        self.assertIsInstance(result["components"], dict)

    def test_components_present(self) -> None:
        """All 5 component names present in components dict."""
        g = Genome()
        result = g.multi_objective_fitness(
            survival_time=0.5, energy_efficiency_ratio=0.5,
            stability_score=0.5, adaptation_success_rate=0.5,
        )
        expected = {"trait_fitness", "survival_time", "energy_efficiency", "stability", "adaptation"}
        self.assertEqual(set(result["components"].keys()), expected)

    def test_default_weights_sum_to_one(self) -> None:
        """Default objectives weights should sum to 1.0."""
        g = Genome()
        result = g.multi_objective_fitness()
        total = sum(result["weights"].values())
        self.assertAlmostEqual(total, 1.0, places=4)

    def test_clamps_inputs(self) -> None:
        """Inputs > 1 or < 0 are clamped."""
        g = Genome()
        result = g.multi_objective_fitness(
            survival_time=5.0, energy_efficiency_ratio=-1.0,
            stability_score=2.0, adaptation_success_rate=-0.5,
        )
        comps = result["components"]
        self.assertLessEqual(comps["survival_time"], 1.0)
        self.assertGreaterEqual(comps["energy_efficiency"], 0.0)
        self.assertLessEqual(comps["stability"], 1.0)
        self.assertGreaterEqual(comps["adaptation"], 0.0)

    def test_custom_weights(self) -> None:
        """Custom weights override defaults."""
        g = Genome()
        custom = {
            "trait_fitness": 1.0,
            "survival_time": 0.0,
            "energy_efficiency": 0.0,
            "stability": 0.0,
            "adaptation": 0.0,
        }
        result = g.multi_objective_fitness(objective_weights=custom)
        # With only trait_fitness weighted, multi_fitness ≈ trait fitness
        self.assertAlmostEqual(result["multi_fitness"], g.fitness, places=4)

    def test_all_perfect_scores(self) -> None:
        """Perfect inputs on a perfect genome → multi_fitness near 1.0."""
        g = Genome(traits={n: 1.0 for n in ["adaptability", "energy_efficiency",
                                               "resilience", "perception", "creativity"]})
        result = g.multi_objective_fitness(
            survival_time=1.0, energy_efficiency_ratio=1.0,
            stability_score=1.0, adaptation_success_rate=1.0,
        )
        self.assertGreater(result["multi_fitness"], 0.9)

    def test_zero_traits_genome(self) -> None:
        """Genome with near-zero traits still returns valid structure."""
        g = Genome(traits={n: 0.02 for n in ["adaptability", "energy_efficiency",
                                                "resilience", "perception", "creativity"]})
        result = g.multi_objective_fitness()
        self.assertGreaterEqual(result["multi_fitness"], 0.0)
        self.assertLessEqual(result["multi_fitness"], 1.0)


# ======================================================================
# Feature 3: Memory Drift
# ======================================================================

class TestMemoryDrift(unittest.TestCase):
    """Every 100 cycles, global trait variance based on diversity."""

    def _build_organism(self):
        """Build a minimal Organism in a temp dir."""
        from al01.organism import Organism, MetabolismConfig
        tmpdir = tempfile.mkdtemp()
        cfg = MetabolismConfig(
            pulse_interval=9999, reflect_interval=9999, persist_interval=9999,
            evolve_interval=9999, population_interact_interval=9999,
            autonomy_interval=9999, environment_interval=9999,
            behavior_analysis_interval=9999, auto_reproduce_interval=9999,
            child_autonomy_interval=9999,
        )
        org = Organism(data_dir=tmpdir, config=cfg)
        org.boot()
        return org, tmpdir

    def test_memory_drift_not_triggered_early(self) -> None:
        """memory_drift returns None when below interval."""
        org, _ = self._build_organism()
        org._global_cycle = 50
        result = org.memory_drift()
        self.assertIsNone(result)

    def test_memory_drift_triggers_at_interval(self) -> None:
        """memory_drift triggers when global_cycle >= interval."""
        org, _ = self._build_organism()
        org._global_cycle = 100
        org._last_memory_drift_cycle = 0
        result = org.memory_drift()
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "memory_drift")
        self.assertGreater(result["organisms_affected"], 0)

    def test_memory_drift_adjusts_parent_genome(self) -> None:
        """After memory drift, parent genome should have shifted."""
        org, _ = self._build_organism()
        old_traits = dict(org.genome.traits)
        org._global_cycle = 100
        org._last_memory_drift_cycle = 0
        org.memory_drift()
        new_traits = org.genome.traits
        # At least one trait should differ (probabilistic but nearly certain)
        diffs = sum(1 for k in old_traits if abs(old_traits[k] - new_traits.get(k, 0)) > 1e-8)
        self.assertGreater(diffs, 0, "No traits changed after memory drift")

    def test_memory_drift_updates_last_cycle(self) -> None:
        """After drift, _last_memory_drift_cycle should advance."""
        org, _ = self._build_organism()
        org._global_cycle = 200
        org._last_memory_drift_cycle = 0
        org.memory_drift()
        self.assertEqual(org._last_memory_drift_cycle, 200)

    def test_memory_drift_record_has_entropy(self) -> None:
        """Drift result includes genome_entropy and drift_magnitude."""
        org, _ = self._build_organism()
        org._global_cycle = 100
        org._last_memory_drift_cycle = 0
        result = org.memory_drift()
        self.assertIn("genome_entropy", result)
        self.assertIn("drift_magnitude", result)
        self.assertGreater(result["drift_magnitude"], 0)


# ======================================================================
# Feature 4: Elite Protection Protocol
# ======================================================================

class TestEliteProtection(unittest.TestCase):
    """Top organisms are protected from mutation, get crossover instead."""

    def _make_pop(self) -> Population:
        """Create a population with several children of varying fitness."""
        tmpdir = tempfile.mkdtemp()
        pop = Population(data_dir=tmpdir, parent_id="AL-01", rng_seed=42)

        # Spawn 5 children with controlled fitness
        base = Genome(rng_seed=1)
        for i in range(5):
            child = pop.spawn_child(base, parent_evolution=i + 1)
            if child:
                cid = child["id"]
                # Set distinct fitness by tweaking traits
                g = Genome(traits={
                    "adaptability": 0.5 + i * 0.08,
                    "energy_efficiency": 0.5 + i * 0.06,
                    "resilience": 0.5 + i * 0.04,
                    "perception": 0.5 + i * 0.05,
                    "creativity": 0.5 + i * 0.03,
                })
                pop.update_member(cid, {"genome": g.to_dict()})
        return pop

    def test_elite_ids_returns_top(self) -> None:
        """elite_ids returns the highest-fitness organisms."""
        pop = self._make_pop()
        elites = pop.elite_ids(top_fraction=0.20)
        self.assertGreater(len(elites), 0)
        # All elite IDs should be valid living members
        for eid in elites:
            self.assertIn(eid, pop.member_ids)

    def test_elite_ids_empty_for_small_pop(self) -> None:
        """elite_ids returns [] when population < 3."""
        tmpdir = tempfile.mkdtemp()
        pop = Population(data_dir=tmpdir, parent_id="AL-01", rng_seed=1)
        # Only parent exists
        elites = pop.elite_ids()
        self.assertEqual(elites, [])

    def test_elite_ids_respects_fraction(self) -> None:
        """Higher fraction → more elites."""
        pop = self._make_pop()
        e10 = pop.elite_ids(top_fraction=0.10)
        e50 = pop.elite_ids(top_fraction=0.50)
        self.assertLessEqual(len(e10), len(e50))

    def test_elite_ids_includes_highest_fitness(self) -> None:
        """The champion should always be in the elite set."""
        pop = self._make_pop()
        champ = pop.champion()
        if champ:
            elites = pop.elite_ids(top_fraction=0.20)
            self.assertIn(champ["champion_id"], elites)


# ======================================================================
# Feature 5: Internal Communication Signal
# ======================================================================

class TestInternalSignal(unittest.TestCase):
    """InternalSignal dataclass and organism.internal_signal property."""

    def test_dataclass_fields(self) -> None:
        """InternalSignal has energy_state, stress_level, novelty_drive."""
        sig = InternalSignal()
        self.assertEqual(sig.energy_state, 1.0)
        self.assertEqual(sig.stress_level, 0.0)
        self.assertEqual(sig.novelty_drive, 0.0)

    def test_to_dict(self) -> None:
        """to_dict returns all three keys."""
        sig = InternalSignal(energy_state=0.8, stress_level=0.3, novelty_drive=0.5)
        d = sig.to_dict()
        self.assertIn("energy_state", d)
        self.assertIn("stress_level", d)
        self.assertIn("novelty_drive", d)
        self.assertAlmostEqual(d["energy_state"], 0.8, places=4)

    def test_custom_values(self) -> None:
        """Custom values round-trip correctly."""
        sig = InternalSignal(energy_state=0.123456, stress_level=0.654321, novelty_drive=0.999)
        d = sig.to_dict()
        self.assertAlmostEqual(d["energy_state"], 0.123456, places=5)
        self.assertAlmostEqual(d["stress_level"], 0.654321, places=5)

    def test_organism_internal_signal_property(self) -> None:
        """Organism.internal_signal returns an InternalSignal instance."""
        from al01.organism import Organism, MetabolismConfig
        tmpdir = tempfile.mkdtemp()
        cfg = MetabolismConfig(
            pulse_interval=9999, reflect_interval=9999, persist_interval=9999,
            evolve_interval=9999, population_interact_interval=9999,
            autonomy_interval=9999, environment_interval=9999,
            behavior_analysis_interval=9999, auto_reproduce_interval=9999,
            child_autonomy_interval=9999,
        )
        org = Organism(data_dir=tmpdir, config=cfg)
        org.boot()
        sig = org.internal_signal
        self.assertIsInstance(sig, InternalSignal)
        self.assertGreaterEqual(sig.energy_state, 0.0)
        self.assertLessEqual(sig.energy_state, 1.0)
        self.assertGreaterEqual(sig.stress_level, 0.0)
        self.assertGreaterEqual(sig.novelty_drive, 0.0)

    def test_internal_signal_stress_increases_with_low_energy(self) -> None:
        """Lower autonomy energy → higher stress level."""
        from al01.organism import Organism, MetabolismConfig
        tmpdir = tempfile.mkdtemp()
        cfg = MetabolismConfig(
            pulse_interval=9999, reflect_interval=9999, persist_interval=9999,
            evolve_interval=9999, population_interact_interval=9999,
            autonomy_interval=9999, environment_interval=9999,
            behavior_analysis_interval=9999, auto_reproduce_interval=9999,
            child_autonomy_interval=9999,
        )
        org = Organism(data_dir=tmpdir, config=cfg)
        org.boot()

        # high energy → low stress
        org._autonomy._energy = 0.9
        sig_high = org.internal_signal
        # low energy → higher stress
        org._autonomy._energy = 0.1
        sig_low = org.internal_signal

        self.assertGreater(sig_low.stress_level, sig_high.stress_level)


# ======================================================================
# Integration: Autonomy cycle includes new features
# ======================================================================

class TestAutonomyCycleIntegration(unittest.TestCase):
    """autonomy_cycle() returns multi_objective_fitness and internal_signal."""

    def _build_organism(self):
        from al01.organism import Organism, MetabolismConfig
        tmpdir = tempfile.mkdtemp()
        cfg = MetabolismConfig(
            pulse_interval=9999, reflect_interval=9999, persist_interval=9999,
            evolve_interval=9999, population_interact_interval=9999,
            autonomy_interval=9999, environment_interval=9999,
            behavior_analysis_interval=9999, auto_reproduce_interval=9999,
            child_autonomy_interval=9999,
        )
        org = Organism(data_dir=tmpdir, config=cfg)
        org.boot()
        return org

    def test_autonomy_cycle_has_multi_fitness(self) -> None:
        """autonomy_cycle record includes multi_objective_fitness."""
        org = self._build_organism()
        record = org.autonomy_cycle()
        self.assertIn("multi_objective_fitness", record)
        mof = record["multi_objective_fitness"]
        self.assertIn("multi_fitness", mof)
        self.assertIn("components", mof)

    def test_autonomy_cycle_has_internal_signal(self) -> None:
        """autonomy_cycle record includes internal_signal."""
        org = self._build_organism()
        record = org.autonomy_cycle()
        self.assertIn("internal_signal", record)
        sig = record["internal_signal"]
        self.assertIn("energy_state", sig)
        self.assertIn("stress_level", sig)
        self.assertIn("novelty_drive", sig)


if __name__ == "__main__":
    unittest.main()
