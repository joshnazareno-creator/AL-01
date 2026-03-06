"""AL-01 v3.5 — Tests for fitness fix, child energy, energy floor, parent reserve,
per-cycle instrumentation, and GPT bridge top_members accuracy.

Covers:
1. GPT bridge top_members fitness reads from genome.fitness (not top-level)
2. GPT bridge top_members generation reads generation_id
3. Child energy updates during child_autonomy_cycle
4. Energy floor at 10% (organisms can't drop below)
5. Parent reproduction reserve (no repro below 25% energy)
6. CycleStats instrumentation (volatility, floor hits, efficiency, streaks)
7. Version bump to 3.5
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from al01.genome import Genome
from al01.population import Population
from al01.organism import Organism, MetabolismConfig, VERSION, CycleStats
from al01.autonomy import AutonomyConfig, AutonomyEngine
from al01.gpt_bridge import GPTBridge, GPTBridgeConfig


# ==================================================================
# Helper: build a minimal Organism for testing
# ==================================================================

def _make_organism(**kwargs) -> Organism:
    tmpdir = kwargs.pop("data_dir", tempfile.mkdtemp())
    return Organism(data_dir=tmpdir, config=MetabolismConfig(), **kwargs)


def _spawn_child(org: Organism, fitness_target: float = 0.5) -> str:
    """Spawn a child and register it in the tracker. Returns child_id."""
    g = org.genome
    child = org.population.spawn_child(g, 10)
    child_id = child["id"]
    org.evolution_tracker.register_organism(
        child_id, parent_id="AL-01",
        traits=child["genome"]["traits"], cycle=1,
    )
    return child_id


# ==================================================================
# 1. GPT Bridge — top_members fitness canonical
# ==================================================================

class TestBridgeFitnessCanonical(unittest.TestCase):
    """top_members fitness must come from member.genome.fitness, not top-level."""

    def setUp(self):
        self.org = _make_organism()
        self.child_id = _spawn_child(self.org)
        self.bridge = GPTBridge(self.org)

    def test_top_members_fitness_nonzero(self):
        """Fitness in top_members should match genome.fitness, not default 0."""
        raw = self.bridge._collect_state()
        top = raw["population"]["top_members"]
        # At least one member should have nonzero fitness
        fitnesses = [m["fitness"] for m in top]
        self.assertTrue(any(f > 0 for f in fitnesses),
                        f"All top_members fitness=0: {fitnesses}")

    def test_top_members_fitness_matches_genome(self):
        """Each top member's fitness should equal its genome.fitness."""
        raw = self.bridge._collect_state()
        for m in raw["population"]["top_members"]:
            member_record = self.org.population.get(m["id"])
            if member_record:
                expected = round(
                    member_record.get("genome", {}).get("fitness", 0), 4
                )
                self.assertAlmostEqual(m["fitness"], expected, places=4)

    def test_top_members_sorted_by_fitness_desc(self):
        """top_members should be sorted highest fitness first."""
        # Spawn multiple children for meaningful sort
        for _ in range(3):
            _spawn_child(self.org)
        raw = self.bridge._collect_state()
        fitnesses = [m["fitness"] for m in raw["population"]["top_members"]]
        self.assertEqual(fitnesses, sorted(fitnesses, reverse=True))

    def test_top_members_generation_field(self):
        """Generation must come from generation_id, not 'generation'."""
        raw = self.bridge._collect_state()
        for m in raw["population"]["top_members"]:
            member_record = self.org.population.get(m["id"])
            if member_record:
                expected_gen = member_record.get("generation_id", 0)
                self.assertEqual(m["generation"], expected_gen)


# ==================================================================
# 2. Child Energy Updates
# ==================================================================

class TestChildEnergy(unittest.TestCase):
    """Children should have energy updated during child_autonomy_cycle."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.org = Organism(data_dir=self._tmpdir, config=MetabolismConfig())
        self.child_id = _spawn_child(self.org)

    def test_child_starts_with_energy(self):
        member = self.org.population.get(self.child_id)
        self.assertAlmostEqual(member["energy"], 0.8, places=2)

    def test_child_energy_changes_after_cycle(self):
        """Energy should change (not stay at 0.8) after a child autonomy cycle."""
        energy_before = self.org.population.get(self.child_id)["energy"]
        self.org.child_autonomy_cycle()
        energy_after = self.org.population.get(self.child_id)["energy"]
        # Energy should have changed (decay + decision cost/gain)
        self.assertNotAlmostEqual(energy_before, energy_after, places=6,
                                  msg="Child energy wasn't updated during cycle")

    def test_child_energy_written_back(self):
        """The population record should contain updated energy."""
        self.org.child_autonomy_cycle()
        member = self.org.population.get(self.child_id)
        # Energy should be a float, not exactly 0.8
        self.assertIsInstance(member["energy"], float)
        # With energy_min=0.10, energy should be >= 0.10
        self.assertGreaterEqual(member["energy"], 0.10 - 0.001)

    def test_child_energy_clamped_above_floor(self):
        """Children's energy should never go below energy_min."""
        # Set child energy very low
        self.org.population.update_member(self.child_id, {"energy": 0.12})
        # Run multiple cycles
        for _ in range(5):
            self.org.child_autonomy_cycle()
        member = self.org.population.get(self.child_id)
        if member and member.get("alive", True):
            self.assertGreaterEqual(member["energy"], 0.10 - 0.001)


# ==================================================================
# 3. Energy Floor
# ==================================================================

class TestEnergyFloor(unittest.TestCase):
    """Energy floor should be 10%, preventing death spiral."""

    def test_default_energy_min(self):
        cfg = AutonomyConfig()
        self.assertAlmostEqual(cfg.energy_min, 0.10, places=2)

    def test_engine_energy_clamped_at_floor(self):
        """AutonomyEngine shouldn't let energy drop below energy_min."""
        cfg = AutonomyConfig(energy_min=0.10, energy_initial=0.12)
        engine = AutonomyEngine(data_dir=tempfile.mkdtemp(), config=cfg)
        # Force many costly decisions
        for _ in range(20):
            engine.decide(
                fitness=0.1, awareness=0.5,
                mutation_rate=0.1, pending_stimuli=0,
            )
        self.assertGreaterEqual(engine.energy, 0.10 - 0.001)

    def test_child_energy_respects_floor(self):
        """Children after many cycles should stay above floor."""
        org = _make_organism()
        child_id = _spawn_child(org)
        # Set child energy just above floor
        org.population.update_member(child_id, {"energy": 0.15})
        for _ in range(10):
            org.child_autonomy_cycle()
        member = org.population.get(child_id)
        if member and member.get("alive", True):
            self.assertGreaterEqual(member["energy"], 0.10 - 0.001)


# ==================================================================
# 4. Parent Reproduction Reserve
# ==================================================================

class TestParentReserve(unittest.TestCase):
    """AL-01 shouldn't reproduce when energy is below 25%."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.org = Organism(data_dir=self._tmpdir, config=MetabolismConfig())

    def test_no_repro_at_low_energy(self):
        """When parent energy is below 25%, reproduction should be blocked."""
        # Force energy low
        self.org._autonomy._energy = 0.20
        # Set evolution_count to a multiple of 10 (repro trigger)
        with self.org._state_lock:
            self.org._state["evolution_count"] = 20
        self.org._last_reproduce_at = 0
        initial_pop = self.org.population.size
        result = self.org.evolve_cycle()
        # Should NOT have reproduced
        self.assertFalse(result.get("reproduced", False),
                         "Reproduced despite energy < 25%")

    def test_repro_at_sufficient_energy(self):
        """When parent energy is above 25%, reproduction should proceed."""
        self.org._autonomy._energy = 0.50
        # Set evolution_count to 9 so the next evolve_cycle increments to 10
        with self.org._state_lock:
            self.org._state["evolution_count"] = 9
            self.org._state["stimuli"] = ["trigger_evolution"]
        self.org._last_reproduce_at = 0
        result = self.org.evolve_cycle()
        # evolution_count should now be 10, triggering reproduction
        self.assertTrue(result.get("reproduced", False),
                        "Failed to reproduce despite energy >= 25%")


# ==================================================================
# 5. CycleStats Instrumentation
# ==================================================================

class TestCycleStats(unittest.TestCase):
    """Test the CycleStats rolling-window instrumentation."""

    def test_initial_state(self):
        cs = CycleStats()
        self.assertEqual(cs.energy_volatility, 0.0)
        self.assertEqual(cs.floor_hits_rolling, 0)
        self.assertEqual(cs.efficiency_ratio, 0.0)
        self.assertEqual(cs.longest_alive_streak(), 0)

    def test_energy_volatility(self):
        cs = CycleStats()
        # Record constant deltas → volatility should be 0
        for _ in range(5):
            cs.record_energy_delta(-0.005, False)
        self.assertAlmostEqual(cs.energy_volatility, 0.0, places=6)

        # Record varying deltas → volatility should be nonzero
        cs2 = CycleStats()
        cs2.record_energy_delta(-0.01, False)
        cs2.record_energy_delta(0.02, False)
        cs2.record_energy_delta(-0.03, False)
        self.assertGreater(cs2.energy_volatility, 0)

    def test_floor_hits(self):
        cs = CycleStats()
        cs.record_energy_delta(-0.01, True)
        cs.record_energy_delta(-0.01, False)
        cs.record_energy_delta(-0.01, True)
        self.assertEqual(cs.floor_hits_rolling, 2)

    def test_window_size_limit(self):
        cs = CycleStats(window_size=3)
        cs.record_energy_delta(-0.01, True)
        cs.record_energy_delta(-0.01, True)
        cs.record_energy_delta(-0.01, True)
        cs.record_energy_delta(-0.01, False)  # pushes out oldest
        self.assertEqual(cs.floor_hits_rolling, 2)

    def test_alive_streaks(self):
        cs = CycleStats()
        cs.tick_alive("A")
        cs.tick_alive("A")
        cs.tick_alive("B")
        self.assertEqual(cs.longest_alive_streak(), 2)

    def test_record_death(self):
        cs = CycleStats()
        cs.tick_alive("A")
        cs.tick_alive("A")
        cs.record_death("A")
        self.assertEqual(cs.longest_alive_streak(), 0)

    def test_efficiency_ratio(self):
        cs = CycleStats()
        cs.record_efficiency(0.01, 0.005)
        cs.record_efficiency(0.01, 0.005)
        # Total spent = 0.02, total gained = 0.01 → ratio = 0.5
        self.assertAlmostEqual(cs.efficiency_ratio, 0.5, places=3)

    def test_efficiency_ratio_no_spend(self):
        cs = CycleStats()
        self.assertEqual(cs.efficiency_ratio, 0.0)

    def test_to_dict(self):
        cs = CycleStats()
        cs.record_energy_delta(-0.01, True)
        cs.tick_alive("X")
        d = cs.to_dict()
        self.assertIn("energy_volatility", d)
        self.assertIn("floor_hits_rolling", d)
        self.assertIn("efficiency_ratio", d)
        self.assertIn("longest_alive_streak", d)
        self.assertIn("window_size", d)
        self.assertEqual(d["floor_hits_rolling"], 1)
        self.assertEqual(d["longest_alive_streak"], 1)


# ==================================================================
# 6. CycleStats on Organism
# ==================================================================

class TestOrganismCycleStats(unittest.TestCase):
    """cycle_stats is wired to the Organism and populates during cycles."""

    def setUp(self):
        self.org = _make_organism()

    def test_cycle_stats_exists(self):
        self.assertIsInstance(self.org.cycle_stats, CycleStats)

    def test_cycle_stats_populated_after_autonomy(self):
        self.org.autonomy_cycle()
        cs = self.org.cycle_stats.to_dict()
        # After one cycle, should have at least some data
        self.assertIn("energy_volatility", cs)

    def test_cycle_stats_in_bridge_output(self):
        """GPT bridge narration includes cycle_stats."""
        bridge = GPTBridge(self.org)
        raw = bridge._collect_state()
        self.assertIn("cycle_stats", raw)
        self.assertIsInstance(raw["cycle_stats"], dict)


# ==================================================================
# 7. Version Bump
# ==================================================================

class TestVersionBump(unittest.TestCase):
    def test_version(self):
        self.assertEqual(VERSION, "3.9")


if __name__ == "__main__":
    unittest.main()
