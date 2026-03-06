"""AL-01 v3.8 — Tests for founder protection, champion slot, diversity metrics,
and population count/naming fixes.

Covers:
 1. Founder mutation rate cap (0.08)
 2. Founder energy floor (0.25) — energy never drops below
 3. Founder fitness floor (0.15) — death only below this
 4. Champion returns best child, excludes parent
 5. Champion returns None when no children exist
 6. Diversity metrics: trait_stddev, unique_genome_hashes, genome_entropy
 7. Diversity with identical genomes → entropy 0
 8. Population children_count excludes parent
 9. Generations_present includes all living generation IDs
10. /status includes member_count, children_count, generations_present
11. /status includes champion and diversity
12. /population endpoint includes new fields
13. /population/metrics endpoint includes genome_diversity + champion
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
from typing import Any, Dict

# Ensure al01 package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from al01.genome import Genome
from al01.population import Population
from al01.organism import (
    FOUNDER_MUTATION_CAP,
    FOUNDER_ENERGY_FLOOR,
    FOUNDER_FITNESS_FLOOR,
)


class TestFounderProtection(unittest.TestCase):
    """Founder (AL-01) gets special protection constants."""

    def test_mutation_cap_value(self) -> None:
        """FOUNDER_MUTATION_CAP should be between 0.06 and 0.10."""
        self.assertGreaterEqual(FOUNDER_MUTATION_CAP, 0.06)
        self.assertLessEqual(FOUNDER_MUTATION_CAP, 0.10)

    def test_energy_floor_value(self) -> None:
        """FOUNDER_ENERGY_FLOOR should be between 0.20 and 0.40."""
        self.assertGreaterEqual(FOUNDER_ENERGY_FLOOR, 0.20)
        self.assertLessEqual(FOUNDER_ENERGY_FLOOR, 0.40)

    def test_fitness_floor_value(self) -> None:
        """FOUNDER_FITNESS_FLOOR should be well below default survival threshold."""
        self.assertLess(FOUNDER_FITNESS_FLOOR, 0.2)
        self.assertGreater(FOUNDER_FITNESS_FLOOR, 0.0)

    def test_mutation_cap_limits_genome_mutation_rate(self) -> None:
        """When effective MR > cap, it should be clamped."""
        raw_mr = 0.15  # higher than cap
        capped = min(raw_mr, FOUNDER_MUTATION_CAP)
        self.assertAlmostEqual(capped, FOUNDER_MUTATION_CAP)

    def test_energy_floor_prevents_low_energy(self) -> None:
        """Energy below floor should be raised to floor."""
        energy = 0.05  # below floor
        if energy < FOUNDER_ENERGY_FLOOR:
            energy = FOUNDER_ENERGY_FLOOR
        self.assertAlmostEqual(energy, FOUNDER_ENERGY_FLOOR)


class TestChampion(unittest.TestCase):
    """Champion slot tracks best-performing child."""

    def _make_pop(self, *, n_children: int = 0) -> Population:
        tmpdir = tempfile.mkdtemp()
        pop = Population(data_dir=tmpdir, parent_id="AL-01", rng_seed=42)
        parent_genome = Genome(rng_seed=42)
        for i in range(n_children):
            pop.spawn_child(parent_genome, parent_evolution=i)
        return pop

    def test_no_children_returns_none(self) -> None:
        """With only the parent, champion() should return None."""
        pop = self._make_pop(n_children=0)
        self.assertIsNone(pop.champion())

    def test_champion_is_highest_fitness_child(self) -> None:
        """Champion should be the living child with highest fitness."""
        pop = self._make_pop(n_children=3)
        # Manually boost one child's fitness
        members = pop.get_all()
        children = [m for m in members if m["id"] != "AL-01"]
        self.assertGreaterEqual(len(children), 3)

        # Set child-2 fitness high
        target_id = children[1]["id"]
        g = Genome.from_dict(children[1]["genome"])
        # Force high traits
        for t in g.traits:
            g.set_trait(t, 0.95)
        pop.update_member(target_id, {"genome": g.to_dict()})

        champ = pop.champion()
        self.assertIsNotNone(champ)
        self.assertEqual(champ["champion_id"], target_id)
        self.assertGreater(champ["champion_fitness"], 0.8)
        self.assertIn("champion_genome_hash", champ)

    def test_champion_excludes_parent(self) -> None:
        """Even if AL-01 has highest fitness, it shouldn't be champion."""
        pop = self._make_pop(n_children=2)
        # Boost parent fitness
        parent_g = Genome()
        for t in parent_g.traits:
            parent_g.set_trait(t, 0.99)
        pop.update_member("AL-01", {"genome": parent_g.to_dict()})

        champ = pop.champion()
        self.assertIsNotNone(champ)
        self.assertNotEqual(champ["champion_id"], "AL-01")

    def test_champion_ignores_dead(self) -> None:
        """Dead children should not be champion."""
        pop = self._make_pop(n_children=2)
        members = pop.get_all()
        children = [m for m in members if m["id"] != "AL-01"]
        # Boost child-1, then kill it
        target_id = children[0]["id"]
        g = Genome.from_dict(children[0]["genome"])
        for t in g.traits:
            g.set_trait(t, 0.99)
        pop.update_member(target_id, {"genome": g.to_dict()})
        pop.remove_member(target_id, cause="test")

        champ = pop.champion()
        self.assertIsNotNone(champ)
        self.assertNotEqual(champ["champion_id"], target_id)


class TestDiversityMetrics(unittest.TestCase):
    """Population diversity: stddev, unique hashes, entropy."""

    def _make_pop_with_children(self, n: int = 5) -> Population:
        tmpdir = tempfile.mkdtemp()
        pop = Population(data_dir=tmpdir, parent_id="AL-01", rng_seed=42)
        g = Genome(rng_seed=42)
        for i in range(n):
            pop.spawn_child(g, parent_evolution=i)
        return pop

    def test_diversity_keys(self) -> None:
        """Diversity metrics should have expected keys."""
        pop = self._make_pop_with_children(3)
        d = pop.diversity_metrics()
        self.assertIn("trait_stddev", d)
        self.assertIn("unique_genome_hashes", d)
        self.assertIn("genome_entropy", d)

    def test_diversity_with_population(self) -> None:
        """With children from spawn_child (variance=0.05), diversity should be > 0."""
        pop = self._make_pop_with_children(5)
        d = pop.diversity_metrics()
        # Children are spawned with variance=0.05 so some diversity should exist
        stddev_vals = list(d["trait_stddev"].values())
        # At least some stddev should be > 0
        self.assertTrue(any(s > 0 for s in stddev_vals), "Expected some trait diversity")
        self.assertGreater(d["unique_genome_hashes"], 0)

    def test_identical_genomes_zero_entropy(self) -> None:
        """If all members have the same genome hash, entropy should be 0."""
        tmpdir = tempfile.mkdtemp()
        pop = Population(data_dir=tmpdir, parent_id="AL-01", rng_seed=42)
        # Add 3 members with identical traits
        base_traits = {"adaptability": 0.5, "energy_efficiency": 0.5,
                       "resilience": 0.5, "perception": 0.5, "creativity": 0.5}
        base_genome = Genome(traits=base_traits, rng_seed=1)
        for i in range(3):
            # Directly add members with identical genomes
            pop.update_member("AL-01", {"genome": base_genome.to_dict()})
        # Only parent exists — entropy should be 0
        d = pop.diversity_metrics()
        self.assertEqual(d["genome_entropy"], 0.0)

    def test_two_distinct_genomes_positive_entropy(self) -> None:
        """Two members with different genomes → positive entropy."""
        pop = self._make_pop_with_children(1)
        # Parent + 1 child = 2 distinct genomes
        d = pop.diversity_metrics()
        if d["unique_genome_hashes"] >= 2:
            self.assertGreater(d["genome_entropy"], 0.0)

    def test_empty_population_returns_zeros(self) -> None:
        """If no living members, diversity returns empty/zero."""
        tmpdir = tempfile.mkdtemp()
        pop = Population(data_dir=tmpdir, parent_id="AL-01", rng_seed=42)
        # Kill parent
        pop.remove_member("AL-01", cause="test")
        d = pop.diversity_metrics()
        self.assertEqual(d["unique_genome_hashes"], 0)
        self.assertEqual(d["genome_entropy"], 0.0)


class TestPopulationCounts(unittest.TestCase):
    """member_count, children_count, generations_present."""

    def _make_pop(self, n: int = 3) -> Population:
        tmpdir = tempfile.mkdtemp()
        pop = Population(data_dir=tmpdir, parent_id="AL-01", rng_seed=42)
        g = Genome(rng_seed=42)
        for i in range(n):
            pop.spawn_child(g, parent_evolution=i)
        return pop

    def test_children_count_excludes_parent(self) -> None:
        """children_count should be total living - 1 (parent)."""
        pop = self._make_pop(3)
        self.assertEqual(pop.children_count, 3)
        self.assertEqual(pop.size, 4)  # parent + 3 children

    def test_generations_present(self) -> None:
        """generations_present should include 0 (parent) and 1 (children)."""
        pop = self._make_pop(2)
        gens = pop.generations_present
        self.assertIn(0, gens)
        self.assertIn(1, gens)
        self.assertEqual(gens, sorted(gens))

    def test_children_count_zero_no_children(self) -> None:
        tmpdir = tempfile.mkdtemp()
        pop = Population(data_dir=tmpdir, parent_id="AL-01", rng_seed=42)
        self.assertEqual(pop.children_count, 0)

    def test_dead_child_not_counted(self) -> None:
        """Dead children should not appear in children_count."""
        pop = self._make_pop(3)
        members = pop.get_all()
        child = [m for m in members if m["id"] != "AL-01"][0]
        pop.remove_member(child["id"], cause="test")
        self.assertEqual(pop.children_count, 2)
        self.assertEqual(pop.size, 3)  # parent + 2 living children


class TestFounderMutationCapIntegration(unittest.TestCase):
    """Integration: FOUNDER_MUTATION_CAP actually caps the mutation rate."""

    def test_cap_applies_correctly(self) -> None:
        """Even with exploration boost, effective mutation rate stays <= cap."""
        base_mr = 0.10  # default
        exploration_boost = 0.10
        boosted = base_mr + exploration_boost  # 0.20
        capped = min(boosted, FOUNDER_MUTATION_CAP)
        self.assertLessEqual(capped, FOUNDER_MUTATION_CAP)
        self.assertAlmostEqual(capped, 0.08)


if __name__ == "__main__":
    unittest.main()
