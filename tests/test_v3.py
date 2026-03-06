"""AL-01 v3.0 — Comprehensive test suite for all 7 new systems.

Tests cover:
1. Environment Model (resource pool, variables, scarcity, modulation)
2. Evolution Tracker (generation ID, genome hash, mutation log, CSV export)
3. Behavior Detection (strategy classification, convergence/divergence)
4. Experiment Protocol (seeded reproducibility, lifecycle, timing)
5. Population v3.0 (death, pruning, auto-reproduce, fitness history)
6. Autonomy v3.0 (env modifiers, death signalling, energy floor)
7. Organism v3.0 (full integration, autonomous cycles)
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import tempfile
import time
import unittest
from typing import Any, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from al01.environment import Environment, EnvironmentConfig, ScarcityEvent
from al01.evolution_tracker import EvolutionTracker, genome_hash
from al01.behavior import BehaviorProfile, PopulationBehaviorAnalyzer
from al01.experiment import ExperimentConfig, ExperimentProtocol, generate_experiment_id
from al01.genome import Genome
from al01.population import Population
from al01.autonomy import (
    AutonomyConfig, AutonomyEngine, AwarenessModel,
    DECISION_STABILIZE, DECISION_ADAPT, DECISION_MUTATE, DECISION_BLEND,
)


# ==================================================================
# 1. Environment Model
# ==================================================================

class TestEnvironmentConfig(unittest.TestCase):
    """EnvironmentConfig defaults are sane."""

    def test_defaults(self):
        cfg = EnvironmentConfig()
        self.assertEqual(cfg.resource_pool_max, 1000.0)
        self.assertEqual(cfg.temperature_initial, 0.5)
        self.assertGreater(cfg.scarcity_probability, 0)
        self.assertGreater(cfg.shift_interval, 0)


class TestEnvironmentInit(unittest.TestCase):
    """Environment initialises with correct state."""

    def test_initial_state(self):
        env = Environment(rng_seed=42)
        self.assertEqual(env.cycle, 0)
        self.assertAlmostEqual(env.resource_pool, 1000.0)
        self.assertAlmostEqual(env.temperature, 0.5)
        self.assertAlmostEqual(env.resource_abundance, 0.8)
        self.assertFalse(env.is_scarcity_active)

    def test_from_dict_roundtrip(self):
        env = Environment(rng_seed=42)
        for _ in range(10):
            env.tick()
        d = env.to_dict()
        env2 = Environment.from_dict(d)
        self.assertEqual(env2.cycle, env.cycle)
        self.assertAlmostEqual(env2.resource_pool, env.resource_pool, places=2)


class TestEnvironmentTick(unittest.TestCase):
    """Environment tick advances state correctly."""

    def test_cycle_increments(self):
        env = Environment(rng_seed=42)
        env.tick()
        self.assertEqual(env.cycle, 1)
        env.tick()
        self.assertEqual(env.cycle, 2)

    def test_variables_drift(self):
        env = Environment(rng_seed=42)
        initial_temp = env.temperature
        # After many ticks, temperature should have drifted.
        # Use 125 ticks (period/4*5) so sinusoidal is away from origin.
        for _ in range(125):
            env.tick()
        # Temperature should have moved from its initial value
        self.assertGreater(abs(env.temperature - initial_temp), 0.01)

    def test_resource_regeneration(self):
        env = Environment(rng_seed=42)
        env._resource_pool = 50.0  # deplete
        env.tick()
        self.assertGreater(env.resource_pool, 50.0)

    def test_resource_consume(self):
        env = Environment(rng_seed=42)
        consumed = env.consume_resources(10.0)
        self.assertAlmostEqual(consumed, 10.0)
        self.assertAlmostEqual(env.resource_pool, 990.0)

    def test_consume_more_than_available(self):
        env = Environment(rng_seed=42)
        env._resource_pool = 5.0
        consumed = env.consume_resources(10.0)
        self.assertAlmostEqual(consumed, 5.0)
        self.assertAlmostEqual(env.resource_pool, 0.0)


class TestEnvironmentScarcity(unittest.TestCase):
    """Scarcity events reduce resources and abundance."""

    def test_forced_scarcity(self):
        cfg = EnvironmentConfig(scarcity_probability=1.0)  # always triggers
        env = Environment(config=cfg, rng_seed=42)
        initial_pool = env.resource_pool
        env.tick()
        self.assertLess(env.resource_pool, initial_pool)
        self.assertTrue(env.is_scarcity_active)

    def test_scarcity_expires(self):
        cfg = EnvironmentConfig(
            scarcity_probability=1.0,
            scarcity_duration_min=3,
            scarcity_duration_max=3,
        )
        env = Environment(config=cfg, rng_seed=42)
        env.tick()  # triggers scarcity with duration=3
        # After tick: event was added then ticked once (remaining=2) → still active
        self.assertTrue(env.is_scarcity_active)
        # Prevent new events on subsequent ticks
        env._config.scarcity_probability = 0.0
        env.tick()  # remaining=1
        env.tick()  # remaining=0 → expired
        self.assertFalse(env.is_scarcity_active)


class TestEnvironmentShift(unittest.TestCase):
    """Periodic environment shifts change variables significantly."""

    def test_shift_at_interval(self):
        cfg = EnvironmentConfig(shift_interval=5)
        env = Environment(config=cfg, rng_seed=42)
        before = env.state_snapshot()
        for _ in range(5):
            env.tick()
        after = env.state_snapshot()
        # At least one variable should have shifted noticeably
        diffs = [
            abs(after["temperature"] - before["temperature"]),
            abs(after["entropy_pressure"] - before["entropy_pressure"]),
        ]
        self.assertTrue(any(d > 0.01 for d in diffs))


class TestEnvironmentModulation(unittest.TestCase):
    """Environment modulation APIs return valid values."""

    def test_mutation_cost_multiplier(self):
        env = Environment(rng_seed=42)
        mult = env.mutation_cost_multiplier()
        self.assertGreaterEqual(mult, 1.0)

    def test_energy_regen_rate(self):
        env = Environment(rng_seed=42)
        rate = env.energy_regen_rate()
        self.assertGreater(rate, 0.0)
        self.assertLessEqual(rate, 1.0)

    def test_trait_decay_multiplier(self):
        env = Environment(rng_seed=42)
        mult = env.trait_decay_multiplier()
        self.assertGreaterEqual(mult, 1.0)

    def test_fitness_noise_penalty(self):
        env = Environment(rng_seed=42)
        penalty = env.fitness_noise_penalty()
        self.assertGreaterEqual(penalty, 0.0)

    def test_survival_modifier(self):
        env = Environment(rng_seed=42)
        mod = env.survival_modifier()
        self.assertGreater(mod, 0.0)
        self.assertLessEqual(mod, 1.0)

    def test_env_trait_weight_modifiers(self):
        env = Environment(rng_seed=42)
        mods = env.env_trait_weight_modifiers()
        self.assertIn("resilience", mods)
        self.assertIn("adaptability", mods)
        self.assertIn("perception", mods)
        self.assertIn("energy_efficiency", mods)
        self.assertIn("creativity", mods)

    def test_state_hash_deterministic(self):
        env1 = Environment(rng_seed=42)
        env2 = Environment(rng_seed=42)
        self.assertEqual(env1.state_hash(), env2.state_hash())


# ==================================================================
# 2. Evolution Tracker
# ==================================================================

class TestGenomeHash(unittest.TestCase):
    """genome_hash produces deterministic, unique hashes."""

    def test_deterministic(self):
        traits = {"a": 0.5, "b": 0.3}
        h1 = genome_hash(traits)
        h2 = genome_hash(traits)
        self.assertEqual(h1, h2)

    def test_different_traits_different_hash(self):
        h1 = genome_hash({"a": 0.5, "b": 0.3})
        h2 = genome_hash({"a": 0.5, "b": 0.4})
        self.assertNotEqual(h1, h2)

    def test_order_independent(self):
        h1 = genome_hash({"a": 0.5, "b": 0.3})
        h2 = genome_hash({"b": 0.3, "a": 0.5})
        self.assertEqual(h1, h2)

    def test_hash_length(self):
        h = genome_hash({"x": 1.0})
        self.assertEqual(len(h), 16)


class TestEvolutionTracker(unittest.TestCase):
    """EvolutionTracker registers, records, and queries correctly."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.tracker = EvolutionTracker(data_dir=self._tmpdir)

    def test_register_organism(self):
        entry = self.tracker.register_organism(
            "AL-01", parent_id=None, traits={"a": 0.5}, cycle=0,
        )
        self.assertEqual(entry["generation_id"], 0)
        self.assertEqual(entry["organism_id"], "AL-01")
        self.assertTrue(len(entry["genome_hash"]) > 0)

    def test_child_increments_generation(self):
        self.tracker.register_organism("parent", None, {"a": 0.5}, 0)
        entry = self.tracker.register_organism("child1", "parent", {"a": 0.6}, 10)
        self.assertEqual(entry["generation_id"], 1)

    def test_grandchild_generation(self):
        self.tracker.register_organism("p", None, {"a": 0.5}, 0)
        self.tracker.register_organism("c1", "p", {"a": 0.6}, 10)
        entry = self.tracker.register_organism("gc1", "c1", {"a": 0.7}, 20)
        self.assertEqual(entry["generation_id"], 2)

    def test_record_mutation(self):
        self.tracker.register_organism("org", None, {"a": 0.5}, 0)
        self.tracker.record_mutation(
            "org", 5, {"a": {"old": 0.5, "new": 0.6}},
            0.5, 0.6, {"a": 0.6},
        )
        self.assertEqual(self.tracker.mutation_event_count(), 1)

    def test_record_fitness_trajectory(self):
        self.tracker.register_organism("org", None, {"a": 0.5}, 0)
        for i in range(10):
            self.tracker.record_fitness("org", i, 0.5 + i * 0.01, {"a": 0.5 + i * 0.01})
        trajectory = self.tracker.get_fitness_trajectory("org")
        self.assertEqual(len(trajectory), 10)

    def test_trait_variance(self):
        pop = {"org1": {"a": 0.3, "b": 0.5}, "org2": {"a": 0.7, "b": 0.5}}
        variances = self.tracker.trait_variance_across_population(pop)
        self.assertGreater(variances["a"], 0)
        self.assertAlmostEqual(variances["b"], 0.0)

    def test_record_death(self):
        self.tracker.register_organism("org", None, {"a": 0.5}, 0)
        self.tracker.record_death("org", 100, "energy_depleted")
        # Log file should have the death event
        with open(self.tracker.log_path) as f:
            lines = f.readlines()
        events = [json.loads(l) for l in lines]
        death_events = [e for e in events if e.get("event") == "death"]
        self.assertEqual(len(death_events), 1)


class TestEvolutionTrackerCSV(unittest.TestCase):
    """CSV export produces valid output."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.tracker = EvolutionTracker(data_dir=self._tmpdir)

    def test_fitness_csv(self):
        self.tracker.register_organism("org1", None, {"a": 0.5}, 0)
        self.tracker.record_fitness("org1", 1, 0.5)
        self.tracker.record_fitness("org1", 2, 0.6)
        csv_str = self.tracker.export_fitness_csv()
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        self.assertEqual(rows[0], ["organism_id", "cycle", "fitness"])
        self.assertEqual(len(rows), 3)  # header + 2 data

    def test_mutations_csv(self):
        self.tracker.register_organism("org1", None, {"a": 0.5}, 0)
        self.tracker.record_mutation("org1", 1, {}, 0.5, 0.6, {"a": 0.6})
        csv_str = self.tracker.export_mutations_csv()
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        self.assertEqual(len(rows), 2)  # header + 1

    def test_lineage_csv(self):
        self.tracker.register_organism("p", None, {"a": 0.5}, 0)
        self.tracker.register_organism("c", "p", {"a": 0.6}, 10)
        csv_str = self.tracker.export_lineage_csv()
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        self.assertEqual(len(rows), 3)  # header + 2


class TestEvolutionTrackerRecovery(unittest.TestCase):
    """Tracker rebuilds state from log on restart."""

    def test_rebuild_from_log(self):
        tmpdir = tempfile.mkdtemp()
        t1 = EvolutionTracker(data_dir=tmpdir)
        t1.register_organism("org1", None, {"a": 0.5}, 0)
        t1.record_mutation("org1", 1, {}, 0.5, 0.6, {"a": 0.6})
        t1.record_fitness("org1", 2, 0.6, {"a": 0.6})

        # New tracker from same dir
        t2 = EvolutionTracker(data_dir=tmpdir)
        self.assertIsNotNone(t2.get_lineage("org1"))
        self.assertEqual(t2.mutation_event_count(), 1)
        self.assertEqual(len(t2.get_fitness_trajectory("org1")), 1)


# ==================================================================
# 3. Behavior Detection
# ==================================================================

class TestBehaviorProfile(unittest.TestCase):
    """Individual behavior classification."""

    def test_undetermined_with_few_decisions(self):
        bp = BehaviorProfile("org1")
        bp.record_decision("stabilize", 0.9, 0.5)
        result = bp.classify_strategy()
        self.assertEqual(result["strategy"], "undetermined")

    def test_energy_hoarder_detection(self):
        bp = BehaviorProfile("org1")
        for _ in range(20):
            bp.record_decision("stabilize", 0.9, 0.5, {"a": 0.5, "b": 0.5})
        result = bp.classify_strategy()
        self.assertEqual(result["strategy"], "energy-hoarder")

    def test_explorer_detection(self):
        bp = BehaviorProfile("org1")
        for _ in range(20):
            bp.record_decision("mutate", 0.3, 0.5, {"a": 0.5, "b": 0.5})
        result = bp.classify_strategy()
        self.assertEqual(result["strategy"], "explorer")

    def test_specialist_detection(self):
        bp = BehaviorProfile("org1")
        # All traits very similar → specialist
        traits = {"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5, "e": 0.5}
        for _ in range(10):
            bp.record_decision("stabilize", 0.5, 0.5, traits)
        result = bp.classify_strategy()
        strategies = [s["name"] for s in result.get("all_strategies", [])]
        self.assertIn("specialist", strategies)

    def test_generalist_detection(self):
        bp = BehaviorProfile("org1")
        # Traits spread out → generalist
        traits = {"a": 0.1, "b": 0.9, "c": 0.5, "d": 0.2, "e": 0.8}
        for _ in range(10):
            bp.record_decision("stabilize", 0.5, 0.5, traits)
        result = bp.classify_strategy()
        strategies = [s["name"] for s in result.get("all_strategies", [])]
        self.assertIn("generalist", strategies)


class TestPopulationBehaviorAnalyzer(unittest.TestCase):
    """Population-level convergence/divergence detection."""

    def test_insufficient_data(self):
        pba = PopulationBehaviorAnalyzer()
        result = pba.convergence_analysis()
        self.assertEqual(result["status"], "insufficient_data")

    def test_records_population_snapshot(self):
        pba = PopulationBehaviorAnalyzer()
        pop_traits = {"a": {"x": 0.5}, "b": {"x": 0.6}}
        pop_fitness = {"a": 0.5, "b": 0.6}
        for _ in range(5):
            pba.record_population_snapshot(pop_traits, pop_fitness)
        result = pba.convergence_analysis()
        self.assertIn(result["status"], ("converging", "diverging", "stable"))

    def test_strategy_distribution(self):
        pba = PopulationBehaviorAnalyzer()
        for _ in range(20):
            pba.record_decision("org1", "stabilize", 0.9, 0.5)
            pba.record_decision("org2", "mutate", 0.3, 0.5)
        dist = pba.population_strategy_distribution()
        self.assertIn("energy-hoarder", dist)
        self.assertIn("explorer", dist)

    def test_remove_organism(self):
        pba = PopulationBehaviorAnalyzer()
        pba.record_decision("org1", "stabilize", 0.9, 0.5)
        pba.remove_organism("org1")
        self.assertEqual(len(pba.all_profiles()), 0)


# ==================================================================
# 4. Experiment Protocol
# ==================================================================

class TestExperimentConfig(unittest.TestCase):
    """ExperimentConfig defaults and seed derivation."""

    def test_defaults(self):
        cfg = ExperimentConfig()
        self.assertEqual(cfg.global_seed, 42)
        self.assertEqual(cfg.duration_days, 30)
        self.assertEqual(cfg.max_population, 60)

    def test_auto_id_generation(self):
        eid = generate_experiment_id(42)
        self.assertTrue(eid.startswith("EXP-"))
        self.assertEqual(len(eid), 16)  # "EXP-" + 12 chars


class TestExperimentProtocol(unittest.TestCase):
    """Experiment lifecycle management."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def test_start_stop(self):
        proto = ExperimentProtocol(
            ExperimentConfig(global_seed=42),
            data_dir=self._tmpdir,
        )
        result = proto.start()
        self.assertTrue(proto.active)
        self.assertTrue(result["active"])

        result = proto.stop(reason="test")
        self.assertFalse(proto.active)
        self.assertEqual(result["stop_reason"], "test")

    def test_seed_derivation(self):
        proto = ExperimentProtocol(
            ExperimentConfig(global_seed=42),
            data_dir=self._tmpdir,
        )
        self.assertIsNotNone(proto.environment_seed)
        self.assertIsNotNone(proto.genome_seed)
        self.assertEqual(proto.global_seed, 42)

    def test_reproducible_seeds(self):
        p1 = ExperimentProtocol(ExperimentConfig(global_seed=42), self._tmpdir)
        tmpdir2 = tempfile.mkdtemp()
        p2 = ExperimentProtocol(ExperimentConfig(global_seed=42), tmpdir2)
        self.assertEqual(p1.environment_seed, p2.environment_seed)
        self.assertEqual(p1.genome_seed, p2.genome_seed)

    def test_should_die(self):
        proto = ExperimentProtocol(
            ExperimentConfig(energy_death_threshold=0.0),
            data_dir=self._tmpdir,
        )
        self.assertTrue(proto.should_die(0.0))
        self.assertTrue(proto.should_die(-0.1))
        self.assertFalse(proto.should_die(0.01))

    def test_can_reproduce(self):
        proto = ExperimentProtocol(
            ExperimentConfig(
                reproduction_fitness_threshold=0.5,
                reproduction_fitness_cycles=3,
            ),
            data_dir=self._tmpdir,
        )
        self.assertFalse(proto.can_reproduce(0.5, 2))
        self.assertTrue(proto.can_reproduce(0.5, 3))
        self.assertFalse(proto.can_reproduce(0.4, 3))

    def test_should_prune(self):
        proto = ExperimentProtocol(
            ExperimentConfig(max_population=10),
            data_dir=self._tmpdir,
        )
        self.assertFalse(proto.should_prune(10))
        self.assertTrue(proto.should_prune(11))

    def test_snapshot(self):
        proto = ExperimentProtocol(
            ExperimentConfig(global_seed=42),
            data_dir=self._tmpdir,
        )
        proto.start()
        proto.snapshot(100, {"test": "data"})
        snap_path = os.path.join(
            proto.experiment_dir, "snapshot_000100.json"
        )
        self.assertTrue(os.path.exists(snap_path))

    def test_metadata_persistence(self):
        cfg = ExperimentConfig(global_seed=42, description="test run")
        proto1 = ExperimentProtocol(cfg, self._tmpdir)
        exp_id = proto1.experiment_id
        proto1.start()
        proto1.record_cycle(50)
        proto1._save_metadata()  # ensure state is flushed
        # Simulate restart with same experiment_id and data_dir
        proto2 = ExperimentProtocol(
            ExperimentConfig(
                experiment_id=exp_id,
                global_seed=42,
            ),
            self._tmpdir,
        )
        self.assertTrue(proto2.active)
        self.assertEqual(proto2.total_cycles, 50)


# ==================================================================
# 5. Population v3.0
# ==================================================================

class TestPopulationV3Init(unittest.TestCase):
    """Population initialises with v3.0 fields."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def test_parent_has_v3_fields(self):
        pop = Population(data_dir=self._tmpdir, parent_id="AL-01")
        parent = pop.get("AL-01")
        self.assertIn("generation_id", parent)
        self.assertEqual(parent["generation_id"], 0)
        self.assertIn("genome_hash", parent)
        self.assertIn("alive", parent)
        self.assertTrue(parent["alive"])
        self.assertIn("fitness_history", parent)
        self.assertIn("energy", parent)


class TestPopulationDeath(unittest.TestCase):
    """Organism death and removal."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.pop = Population(data_dir=self._tmpdir, parent_id="AL-01")

    def test_remove_member(self):
        death = self.pop.remove_member("AL-01", "energy_depleted")
        self.assertIsNotNone(death)
        self.assertEqual(death["cause"], "energy_depleted")
        parent = self.pop.get("AL-01")
        self.assertFalse(parent["alive"])

    def test_dead_not_in_member_ids(self):
        self.pop.remove_member("AL-01", "test")
        self.assertNotIn("AL-01", self.pop.member_ids)
        self.assertEqual(self.pop.size, 0)

    def test_dead_in_all_member_ids(self):
        self.pop.remove_member("AL-01", "test")
        self.assertIn("AL-01", self.pop.all_member_ids)
        self.assertEqual(self.pop.total_size, 1)


class TestPopulationPruning(unittest.TestCase):
    """Population pruning kills weakest members."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.pop = Population(data_dir=self._tmpdir, parent_id="AL-01")

    def test_prune_weakest(self):
        # Spawn some children with varying fitness
        parent = Genome(traits={"a": 0.5, "b": 0.5})
        for i in range(5):
            self.pop.spawn_child(parent, i)

        # 6 total (parent + 5 children), prune to 3
        # v3.14: Set max_population so floor (10%) doesn't block this prune
        self.pop.max_population = 10  # floor = 1
        deaths = self.pop.prune_weakest(3, min_keep=2)
        self.assertEqual(self.pop.size, 3)
        self.assertTrue(len(deaths) > 0)

    def test_prune_respects_min_keep(self):
        self.pop.prune_weakest(0, min_keep=1)
        self.assertGreaterEqual(self.pop.size, 1)


class TestPopulationAutoReproduce(unittest.TestCase):
    """Autonomous reproduction gating."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.pop = Population(data_dir=self._tmpdir, parent_id="AL-01")

    def test_no_reproduce_below_threshold(self):
        # Update consecutive count but with low fitness
        self.pop.update_consecutive_repro("AL-01", 0.3, threshold=0.5)
        child = self.pop.auto_reproduce("AL-01", 0.5, 1)
        self.assertIsNone(child)

    def test_reproduce_when_qualified(self):
        # Build up consecutive cycles
        for _ in range(5):
            self.pop.update_consecutive_repro("AL-01", 0.6, threshold=0.5)
        # Update genome fitness to be above threshold
        parent = self.pop.get("AL-01")
        genome = parent["genome"]
        genome["fitness"] = 0.6
        self.pop.update_member("AL-01", {"genome": genome})

        child = self.pop.auto_reproduce("AL-01", 0.5, 5)
        self.assertIsNotNone(child)
        self.assertEqual(child["generation_id"], 1)

    def test_consecutive_resets_on_dip(self):
        for _ in range(3):
            self.pop.update_consecutive_repro("AL-01", 0.6, threshold=0.5)
        self.pop.update_consecutive_repro("AL-01", 0.3, threshold=0.5)
        parent = self.pop.get("AL-01")
        self.assertEqual(parent["consecutive_above_repro"], 0)


class TestPopulationFitnessHistory(unittest.TestCase):
    """Independent fitness history per organism."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.pop = Population(data_dir=self._tmpdir, parent_id="AL-01")

    def test_record_fitness(self):
        for i in range(10):
            self.pop.record_fitness("AL-01", 0.5 + i * 0.01)
        parent = self.pop.get("AL-01")
        self.assertEqual(len(parent["fitness_history"]), 10)

    def test_fitness_history_capped(self):
        for i in range(200):
            self.pop.record_fitness("AL-01", 0.5)
        parent = self.pop.get("AL-01")
        self.assertLessEqual(len(parent["fitness_history"]), 100)


class TestPopulationSpawnV3(unittest.TestCase):
    """v3.0 spawn_child includes generation_id and genome_hash."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.pop = Population(data_dir=self._tmpdir, parent_id="AL-01")

    def test_child_has_v3_fields(self):
        parent_genome = Genome()
        child = self.pop.spawn_child(parent_genome, 10)
        self.assertEqual(child["generation_id"], 1)
        self.assertTrue(len(child["genome_hash"]) > 0)
        self.assertEqual(child["energy"], 0.8)
        self.assertTrue(child["alive"])
        self.assertIsNotNone(child.get("mutation_rate_offset"))

    def test_grandchild_generation(self):
        g = Genome()
        child = self.pop.spawn_child(g, 10, parent_id="AL-01")
        child_genome = Genome.from_dict(child["genome"])
        grandchild = self.pop.spawn_child(child_genome, 20, parent_id=child["id"])
        self.assertEqual(grandchild["generation_id"], 2)


class TestPopulationTraitVariance(unittest.TestCase):
    """Trait variance across living organisms."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.pop = Population(data_dir=self._tmpdir, parent_id="AL-01")

    def test_variance_with_single_member(self):
        var = self.pop.trait_variance()
        self.assertEqual(var, {})

    def test_variance_with_multiple_members(self):
        g = Genome()
        self.pop.spawn_child(g, 1)
        var = self.pop.trait_variance()
        self.assertIn("adaptability", var)


# ==================================================================
# 6. Autonomy v3.0
# ==================================================================

class TestAutonomyDeathSignalling(unittest.TestCase):
    """Autonomy engine signals death when energy depletes."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        cfg = AutonomyConfig(
            energy_initial=0.01,       # nearly dead
            energy_decay_per_cycle=0.02,  # will die in 1 cycle
            energy_min=0.0,            # no floor
        )
        self.engine = AutonomyEngine(data_dir=self._tmpdir, config=cfg)

    def test_organism_dies(self):
        record = self.engine.decide(0.5, 0.5, 0.1, 0)
        self.assertTrue(record.get("organism_died"))
        self.assertLessEqual(record["energy"], 0.0)


class TestAutonomyEnvModifiers(unittest.TestCase):
    """Environment modifiers affect decision cycle."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        cfg = AutonomyConfig(energy_initial=1.0, energy_min=0.0)
        self.engine = AutonomyEngine(data_dir=self._tmpdir, config=cfg)

    def test_high_mutation_cost(self):
        # With high mutation cost multiplier
        record = self.engine.decide(
            0.3, 0.5, 0.1, 0,
            env_modifiers={"mutation_cost_multiplier": 3.0},
        )
        # Should still make decisions
        self.assertIn(record["decision"], [DECISION_STABILIZE, DECISION_ADAPT, DECISION_MUTATE, DECISION_BLEND])
        self.assertIn("env_modifiers", record)

    def test_fitness_noise_penalty(self):
        record = self.engine.decide(
            0.5, 0.5, 0.1, 0,
            env_modifiers={"fitness_noise_penalty": 0.1},
        )
        # Effective fitness should be lower than raw
        self.assertLess(record["effective_fitness"], record["fitness"])


class TestAutonomyEnergyFloor(unittest.TestCase):
    """With energy_min=0.0, energy can reach 0 (death possible)."""

    def test_energy_can_reach_zero(self):
        tmpdir = tempfile.mkdtemp()
        cfg = AutonomyConfig(
            energy_initial=0.001,
            energy_decay_per_cycle=0.01,
            energy_min=0.0,
        )
        engine = AutonomyEngine(data_dir=tmpdir, config=cfg)
        record = engine.decide(0.5, 0.5, 0.1, 0)
        self.assertLessEqual(engine.energy, 0.01)


# ==================================================================
# 7. Organism v3.0 Integration (using tmpdir, no external deps)
# ==================================================================

class TestOrganismV3Version(unittest.TestCase):
    """Version is 3.0."""

    def test_version(self):
        from al01.organism import VERSION
        self.assertTrue(VERSION.startswith("3."))


class TestOrganismV3Properties(unittest.TestCase):
    """New v3.0 properties are accessible."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        from al01.organism import Organism
        self.org = Organism(data_dir=self._tmpdir)

    def test_environment_property(self):
        self.assertIsNotNone(self.org.environment)

    def test_evolution_tracker_property(self):
        self.assertIsNotNone(self.org.evolution_tracker)

    def test_behavior_analyzer_property(self):
        self.assertIsNotNone(self.org.behavior_analyzer)

    def test_global_cycle_starts_zero(self):
        self.assertEqual(self.org.global_cycle, 0)


class TestOrganismEnvironmentTick(unittest.TestCase):
    """Environment tick advances environment state."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        from al01.organism import Organism
        self.org = Organism(data_dir=self._tmpdir)

    def test_env_tick_increments_cycle(self):
        self.org.environment_tick()
        self.assertEqual(self.org.environment.cycle, 1)


class TestOrganismAutoReproduce(unittest.TestCase):
    """Auto-reproduce cycle works through organism."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        from al01.organism import Organism
        self.org = Organism(data_dir=self._tmpdir)

    def test_no_reproduce_initially(self):
        result = self.org.auto_reproduce_cycle()
        self.assertEqual(len(result), 0)


class TestOrganismBehaviorCycle(unittest.TestCase):
    """Behavior analysis cycle returns valid data."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        from al01.organism import Organism
        self.org = Organism(data_dir=self._tmpdir)

    def test_returns_analysis(self):
        result = self.org.behavior_analysis_cycle()
        # With single organism, insufficient data
        self.assertIn("status", result)


class TestOrganismGrowthMetricsV3(unittest.TestCase):
    """Growth metrics include v3.0 fields."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        from al01.organism import Organism
        self.org = Organism(data_dir=self._tmpdir)

    def test_environment_in_metrics(self):
        metrics = self.org.growth_metrics
        self.assertIn("environment", metrics)
        self.assertIn("resource_pool", metrics["environment"])

    def test_behavior_summary_in_metrics(self):
        metrics = self.org.growth_metrics
        self.assertIn("behavior_summary", metrics)

    def test_evolution_tracker_in_metrics(self):
        metrics = self.org.growth_metrics
        self.assertIn("evolution_tracker", metrics)
        self.assertIn("generation_counter", metrics["evolution_tracker"])


class TestOrganismAutonomyCycleV3(unittest.TestCase):
    """Autonomy cycle with v3.0 features."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        from al01.organism import Organism
        self.org = Organism(data_dir=self._tmpdir)

    def test_cycle_runs_and_returns_record(self):
        record = self.org.autonomy_cycle()
        self.assertIn("decision", record)
        self.assertIn("env_modifiers", record)
        self.assertGreater(self.org.global_cycle, 0)

    def test_cycle_records_fitness_in_tracker(self):
        self.org.autonomy_cycle()
        trajectory = self.org.evolution_tracker.get_fitness_trajectory("AL-01")
        self.assertGreater(len(trajectory), 0)

    def test_cycle_records_in_behavior_analyzer(self):
        self.org.autonomy_cycle()
        profiles = self.org.behavior_analyzer.all_profiles()
        self.assertIn("AL-01", profiles)


# ==================================================================
# Scarcity event unit tests
# ==================================================================

class TestScarcityEvent(unittest.TestCase):
    """ScarcityEvent tick and serialization."""

    def test_tick_countdown(self):
        e = ScarcityEvent(severity=0.5, remaining_cycles=3, started_at="now", abundance_reduction=0.2)
        self.assertTrue(e.tick())  # 2 remaining
        self.assertTrue(e.tick())  # 1 remaining
        self.assertFalse(e.tick())  # 0 → expired

    def test_roundtrip(self):
        e = ScarcityEvent(severity=0.5, remaining_cycles=3, started_at="ts", abundance_reduction=0.2)
        d = e.to_dict()
        e2 = ScarcityEvent.from_dict(d)
        self.assertAlmostEqual(e2.severity, 0.5)
        self.assertEqual(e2.remaining_cycles, 3)


# ==================================================================
# Determinism tests (reproducibility)
# ==================================================================

class TestEnvironmentDeterminism(unittest.TestCase):
    """Same seed produces same sequence."""

    def test_same_seed_same_state(self):
        env1 = Environment(rng_seed=123)
        env2 = Environment(rng_seed=123)
        for _ in range(50):
            env1.tick()
            env2.tick()
        self.assertAlmostEqual(env1.temperature, env2.temperature, places=6)
        self.assertAlmostEqual(env1.resource_pool, env2.resource_pool, places=2)
        self.assertAlmostEqual(env1.entropy_pressure, env2.entropy_pressure, places=6)

    def test_different_seed_different_state(self):
        env1 = Environment(rng_seed=1)
        env2 = Environment(rng_seed=2)
        for _ in range(50):
            env1.tick()
            env2.tick()
        # Very unlikely to be identical
        diff = abs(env1.entropy_pressure - env2.entropy_pressure)
        self.assertGreater(diff, 0.0)


class TestExperimentSeedDeterminism(unittest.TestCase):
    """Same global seed derives same sub-seeds."""

    def test_deterministic_derived_seeds(self):
        p1 = ExperimentProtocol(ExperimentConfig(global_seed=999), tempfile.mkdtemp())
        p2 = ExperimentProtocol(ExperimentConfig(global_seed=999), tempfile.mkdtemp())
        self.assertEqual(p1.environment_seed, p2.environment_seed)
        self.assertEqual(p1.genome_seed, p2.genome_seed)

    def test_different_global_seed_different_derived(self):
        p1 = ExperimentProtocol(ExperimentConfig(global_seed=1), tempfile.mkdtemp())
        p2 = ExperimentProtocol(ExperimentConfig(global_seed=2), tempfile.mkdtemp())
        self.assertNotEqual(p1.environment_seed, p2.environment_seed)


if __name__ == "__main__":
    unittest.main()
