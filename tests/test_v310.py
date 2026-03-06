"""Tests for v3.10 — Multi-Objective Fitness Refactor.

Validates:
1. Genome.fitness uses multi-objective formula when components are set.
2. Genome.fitness falls back to trait average when components are NOT set.
3. fitness_components dict is stored/inspected correctly.
4. Serialization round-trip preserves components.
5. autonomy_cycle sets components on parent genome.
6. No energy collapse, fitness spread, or stagnation spike over 25 cycles.
"""

import os
import tempfile
import unittest

from al01.genome import Genome, _soft_cap


# ======================================================================
# Unit: Genome fitness_components
# ======================================================================

class TestFitnessComponentsProperty(unittest.TestCase):
    """Genome stores and exposes fitness_components for inspection."""

    def test_no_components_returns_none(self) -> None:
        """Fresh genome has no components — returns None."""
        g = Genome()
        self.assertIsNone(g.fitness_components)

    def test_set_and_get_components(self) -> None:
        """set_fitness_components populates the dict."""
        g = Genome()
        g.set_fitness_components(
            survival=0.8, efficiency=0.6, stability=0.7, adaptation=0.5,
        )
        fc = g.fitness_components
        self.assertIsNotNone(fc)
        self.assertAlmostEqual(fc["survival"], 0.8, places=6)
        self.assertAlmostEqual(fc["efficiency"], 0.6, places=6)
        self.assertAlmostEqual(fc["stability"], 0.7, places=6)
        self.assertAlmostEqual(fc["adaptation"], 0.5, places=6)

    def test_components_clamped(self) -> None:
        """Values outside [0,1] are clamped."""
        g = Genome()
        g.set_fitness_components(
            survival=1.5, efficiency=-0.3, stability=2.0, adaptation=-1.0,
        )
        fc = g.fitness_components
        self.assertAlmostEqual(fc["survival"], 1.0)
        self.assertAlmostEqual(fc["efficiency"], 0.0)
        self.assertAlmostEqual(fc["stability"], 1.0)
        self.assertAlmostEqual(fc["adaptation"], 0.0)


# ======================================================================
# Unit: fitness property behaviour
# ======================================================================

class TestFitnessProperty(unittest.TestCase):
    """genome.fitness returns multi-objective when set, else trait average."""

    def test_fallback_to_trait_average(self) -> None:
        """Without components, fitness == trait_fitness (old formula)."""
        g = Genome()
        self.assertAlmostEqual(g.fitness, g.trait_fitness, places=8)

    def test_default_genome_fitness_approximately_half(self) -> None:
        """Default genome (all traits 0.5) still ~0.5 without components."""
        g = Genome()
        self.assertAlmostEqual(g.fitness, 0.5, places=2)

    def test_multi_objective_when_set(self) -> None:
        """With components set, fitness uses weighted formula."""
        g = Genome()
        g.set_fitness_components(
            survival=1.0, efficiency=1.0, stability=1.0, adaptation=1.0,
        )
        expected = 0.35 * 1.0 + 0.30 * 1.0 + 0.20 * 1.0 + 0.15 * 1.0
        self.assertAlmostEqual(g.fitness, expected, places=6)

    def test_weights_sum_to_one(self) -> None:
        """Multi-objective weights sum to 1.0."""
        total = sum(Genome.MO_WEIGHTS.values())
        self.assertAlmostEqual(total, 1.0, places=8)

    def test_specific_component_weights(self) -> None:
        """Each weight matches the spec: 0.35, 0.30, 0.20, 0.15."""
        w = Genome.MO_WEIGHTS
        self.assertAlmostEqual(w["survival"], 0.35)
        self.assertAlmostEqual(w["efficiency"], 0.30)
        self.assertAlmostEqual(w["stability"], 0.20)
        self.assertAlmostEqual(w["adaptation"], 0.15)

    def test_partial_components(self) -> None:
        """Mixed component values produce correct weighted average."""
        g = Genome()
        g.set_fitness_components(
            survival=0.8, efficiency=0.6, stability=0.4, adaptation=0.2,
        )
        expected = 0.35 * 0.8 + 0.30 * 0.6 + 0.20 * 0.4 + 0.15 * 0.2
        self.assertAlmostEqual(g.fitness, expected, places=6)

    def test_zero_components_yield_zero(self) -> None:
        """All-zero components -> fitness = 0."""
        g = Genome()
        g.set_fitness_components(
            survival=0.0, efficiency=0.0, stability=0.0, adaptation=0.0,
        )
        self.assertAlmostEqual(g.fitness, 0.0, places=8)


# ======================================================================
# Unit: trait_fitness always uses old formula
# ======================================================================

class TestTraitFitness(unittest.TestCase):
    """trait_fitness always returns old trait-average formula."""

    def test_trait_fitness_unchanged_by_components(self) -> None:
        """Setting components doesn't affect trait_fitness."""
        g = Genome()
        before = g.trait_fitness
        g.set_fitness_components(
            survival=0.1, efficiency=0.1, stability=0.1, adaptation=0.1,
        )
        after = g.trait_fitness
        self.assertAlmostEqual(before, after, places=8)

    def test_trait_fitness_matches_old_formula(self) -> None:
        """trait_fitness = mean of soft-capped values."""
        g = Genome(traits={"t1": 2.0, "t2": 0.5})
        expected = (_soft_cap(2.0) + _soft_cap(0.5)) / 2
        self.assertAlmostEqual(g.trait_fitness, expected, places=6)


# ======================================================================
# Unit: Serialization round-trip
# ======================================================================

class TestSerialization(unittest.TestCase):
    """to_dict / from_dict preserves fitness_components."""

    def test_to_dict_includes_fitness_components(self) -> None:
        """Serialized dict includes fitness_components when set."""
        g = Genome()
        g.set_fitness_components(
            survival=0.7, efficiency=0.5, stability=0.3, adaptation=0.9,
        )
        d = g.to_dict()
        self.assertIn("fitness_components", d)
        self.assertAlmostEqual(d["fitness_components"]["survival"], 0.7)
        self.assertAlmostEqual(d["fitness_components"]["adaptation"], 0.9)

    def test_to_dict_omits_components_when_unset(self) -> None:
        """Serialized dict does NOT include fitness_components when None."""
        g = Genome()
        d = g.to_dict()
        self.assertNotIn("fitness_components", d)

    def test_to_dict_includes_trait_fitness(self) -> None:
        """Serialized dict includes trait_fitness for inspection."""
        g = Genome()
        d = g.to_dict()
        self.assertIn("trait_fitness", d)
        self.assertAlmostEqual(d["trait_fitness"], g.trait_fitness, places=6)

    def test_round_trip_with_components(self) -> None:
        """from_dict restores fitness_components correctly."""
        g = Genome()
        g.set_fitness_components(
            survival=0.6, efficiency=0.4, stability=0.8, adaptation=0.2,
        )
        d = g.to_dict()
        g2 = Genome.from_dict(d)
        fc = g2.fitness_components
        self.assertIsNotNone(fc)
        self.assertAlmostEqual(fc["survival"], 0.6, places=5)
        self.assertAlmostEqual(fc["efficiency"], 0.4, places=5)
        self.assertAlmostEqual(fc["stability"], 0.8, places=5)
        self.assertAlmostEqual(fc["adaptation"], 0.2, places=5)
        self.assertAlmostEqual(g.fitness, g2.fitness, places=5)

    def test_round_trip_without_components(self) -> None:
        """from_dict with no components gives None fitness_components."""
        g = Genome()
        d = g.to_dict()
        g2 = Genome.from_dict(d)
        self.assertIsNone(g2.fitness_components)
        self.assertAlmostEqual(g.fitness, g2.fitness, places=6)


# ======================================================================
# Integration: autonomy_cycle sets components on parent
# ======================================================================

class TestAutonomyCycleComponents(unittest.TestCase):
    """autonomy_cycle injects fitness_components into the parent genome."""

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
        return org, tmpdir

    def test_autonomy_cycle_sets_components(self) -> None:
        """After one autonomy_cycle, genome has fitness_components."""
        org, _ = self._build_organism()
        org.autonomy_cycle()
        fc = org.genome.fitness_components
        self.assertIsNotNone(fc, "fitness_components should be set after autonomy_cycle")
        self.assertIn("survival", fc)
        self.assertIn("efficiency", fc)
        self.assertIn("stability", fc)
        self.assertIn("adaptation", fc)

    def test_record_includes_fitness_components(self) -> None:
        """autonomy_cycle record contains fitness_components key."""
        org, _ = self._build_organism()
        record = org.autonomy_cycle()
        self.assertIn("fitness_components", record)
        fc = record["fitness_components"]
        self.assertIsInstance(fc, dict)
        for key in ("survival", "efficiency", "stability", "adaptation"):
            self.assertIn(key, fc)


# ======================================================================
# Validation: 25-cycle run — no collapse, spread, no stagnation spike
# ======================================================================

class TestValidationRun(unittest.TestCase):
    """Run 25 cycles and check for stability."""

    def _build_organism(self):
        from al01.organism import Organism, MetabolismConfig
        tmpdir = tempfile.mkdtemp()
        cfg = MetabolismConfig(
            pulse_interval=9999, reflect_interval=9999, persist_interval=9999,
            evolve_interval=30, population_interact_interval=60,
            autonomy_interval=10, environment_interval=5,
            behavior_analysis_interval=9999, auto_reproduce_interval=15,
            child_autonomy_interval=10,
        )
        org = Organism(data_dir=tmpdir, config=cfg)
        org.boot()
        return org, tmpdir

    def test_25_cycle_no_collapse(self) -> None:
        """25 ticks: energy stays > 0.02, fitness > 0, no stagnation spike."""
        org, _ = self._build_organism()

        fitness_history = []
        energy_history = []
        stagnation_history = []

        for i in range(25):
            org.tick()
            gm = org.growth_metrics
            f = gm["fitness"]
            e = gm.get("genome", {}).get("traits", {}).get("energy_efficiency", 0.5)
            stag = gm.get("stagnation_count", 0)
            fitness_history.append(f)
            energy_history.append(e)
            stagnation_history.append(stag)

        # No energy collapse: energy_efficiency trait stays above 0.02
        self.assertTrue(
            all(e >= 0.02 for e in energy_history),
            f"Energy collapsed below floor: {min(energy_history):.4f}",
        )

        # Fitness doesn't collapse to zero
        self.assertTrue(
            all(f > 0.0 for f in fitness_history),
            f"Fitness collapsed to zero at some point",
        )

        # Trait variance stays nonzero (genome isn't dead)
        tv = org.growth_metrics.get("trait_variance", {})
        if tv:
            self.assertTrue(
                any(v > 0.0 for v in tv.values()),
                "All trait variance is zero - genome is frozen",
            )

        # No extreme stagnation spike
        max_stag = max(stagnation_history) if stagnation_history else 0
        self.assertLess(
            max_stag, 500,
            f"Stagnation spiked to {max_stag} - possible regression",
        )


if __name__ == "__main__":
    unittest.main()
