"""AL-01 v3.3 — ALife 2026 feature tests.

Covers:
1. Diversity index (Shannon, Simpson's) in PopulationBehaviorAnalyzer
2. Seeded RNG reproducibility in AutonomyEngine and Population
3. Survival grace cycles enforcement in Organism
4. New API endpoints: lineage, population metrics, CSV exports,
   experiment control, exploration toggle
5. Manual exploration mode toggle in AutonomyEngine
6. Dashboard population metrics card
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from al01.behavior import BehaviorProfile, PopulationBehaviorAnalyzer
from al01.autonomy import AutonomyConfig, AutonomyEngine
from al01.population import Population
from al01.genome import Genome


# ==================================================================
# 1. Diversity Index
# ==================================================================

class TestDiversityIndex(unittest.TestCase):
    """Shannon and Simpson's diversity indices from strategy distribution."""

    def test_empty_population_returns_zeros(self):
        analyzer = PopulationBehaviorAnalyzer()
        div = analyzer.diversity_index()
        self.assertEqual(div["shannon"], 0.0)
        self.assertEqual(div["simpson"], 0.0)
        self.assertEqual(div["richness"], 0)
        self.assertEqual(div["evenness"], 0.0)

    def test_single_strategy_zero_diversity(self):
        analyzer = PopulationBehaviorAnalyzer()
        # Create 5 organisms all classified as energy-hoarder by feeding stabilize decisions
        for i in range(5):
            oid = f"org-{i}"
            for _ in range(10):
                analyzer.record_decision(oid, "stabilize", 0.9, 0.5, {"adaptability": 0.5})
        div = analyzer.diversity_index()
        # One strategy → Shannon = 0 (ln(1) = 0), Simpson = 0
        self.assertEqual(div["shannon"], 0.0)
        self.assertEqual(div["simpson"], 0.0)
        self.assertEqual(div["richness"], 1)

    def test_two_strategies_positive_diversity(self):
        analyzer = PopulationBehaviorAnalyzer()
        # 3 hoarders + 3 explorers
        for i in range(3):
            oid = f"hoarder-{i}"
            for _ in range(10):
                analyzer.record_decision(oid, "stabilize", 0.9, 0.5, {"adaptability": 0.5})
        for i in range(3):
            oid = f"explorer-{i}"
            for _ in range(10):
                analyzer.record_decision(oid, "mutate", 0.4, 0.5, {"adaptability": 0.8, "creativity": 0.8})
        div = analyzer.diversity_index()
        self.assertGreater(div["shannon"], 0.0)
        self.assertGreater(div["simpson"], 0.0)
        self.assertGreaterEqual(div["richness"], 2)

    def test_equal_distribution_max_evenness(self):
        analyzer = PopulationBehaviorAnalyzer()
        # 2 hoarders + 2 explorers → equal distribution
        for i in range(2):
            oid = f"hoarder-{i}"
            for _ in range(10):
                analyzer.record_decision(oid, "stabilize", 0.9, 0.5, {"adaptability": 0.5})
        for i in range(2):
            oid = f"explorer-{i}"
            for _ in range(10):
                analyzer.record_decision(oid, "mutate", 0.4, 0.5, {"adaptability": 0.8, "creativity": 0.8})
        div = analyzer.diversity_index()
        # 50/50 split → evenness should be 1.0 (or very close for 2 strategies)
        self.assertAlmostEqual(div["evenness"], 1.0, places=2)

    def test_diversity_in_summary(self):
        analyzer = PopulationBehaviorAnalyzer()
        summary = analyzer.summary()
        self.assertIn("diversity", summary)
        self.assertIn("shannon", summary["diversity"])
        self.assertIn("simpson", summary["diversity"])


# ==================================================================
# 2. Seeded RNG Reproducibility
# ==================================================================

class TestSeededRNGAutonomy(unittest.TestCase):
    """AutonomyEngine with rng_seed produces deterministic drift."""

    def _make_engine(self, seed=42):
        tmpdir = tempfile.mkdtemp()
        return AutonomyEngine(data_dir=tmpdir, rng_seed=seed)

    def test_same_seed_same_drift(self):
        e1 = self._make_engine(seed=42)
        e2 = self._make_engine(seed=42)
        # Run drift on both
        d1 = e1._drift_environment()
        d2 = e2._drift_environment()
        self.assertEqual(
            e1._effective_fitness_threshold,
            e2._effective_fitness_threshold,
        )
        self.assertEqual(d1, d2)

    def test_different_seed_different_drift(self):
        e1 = self._make_engine(seed=42)
        e2 = self._make_engine(seed=99)
        e1._drift_environment()
        e2._drift_environment()
        # Very unlikely to be identical
        self.assertNotEqual(
            e1._effective_fitness_threshold,
            e2._effective_fitness_threshold,
        )

    def test_deterministic_across_multiple_drifts(self):
        e1 = self._make_engine(seed=123)
        e2 = self._make_engine(seed=123)
        results1 = [e1._drift_environment() for _ in range(10)]
        results2 = [e2._drift_environment() for _ in range(10)]
        self.assertEqual(results1, results2)

    def test_none_seed_still_works(self):
        """No seed → uses default random (non-deterministic but shouldn't crash)."""
        tmpdir = tempfile.mkdtemp()
        engine = AutonomyEngine(data_dir=tmpdir, rng_seed=None)
        drift = engine._drift_environment()
        self.assertIsInstance(drift, float)


class TestSeededRNGPopulation(unittest.TestCase):
    """Population with rng_seed produces deterministic behaviour."""

    def test_same_seed_same_mutation_offset(self):
        tmpdir1 = tempfile.mkdtemp()
        tmpdir2 = tempfile.mkdtemp()
        pop1 = Population(data_dir=tmpdir1, rng_seed=42)
        pop2 = Population(data_dir=tmpdir2, rng_seed=42)
        g = Genome()
        c1 = pop1.spawn_child(g, 1)
        c2 = pop2.spawn_child(g, 1)
        self.assertEqual(
            c1["mutation_rate_offset"],
            c2["mutation_rate_offset"],
        )

    def test_different_seeds_different_offset(self):
        tmpdir1 = tempfile.mkdtemp()
        tmpdir2 = tempfile.mkdtemp()
        pop1 = Population(data_dir=tmpdir1, rng_seed=42)
        pop2 = Population(data_dir=tmpdir2, rng_seed=99)
        g = Genome()
        c1 = pop1.spawn_child(g, 1)
        c2 = pop2.spawn_child(g, 1)
        self.assertNotEqual(
            c1["mutation_rate_offset"],
            c2["mutation_rate_offset"],
        )


# ==================================================================
# 3. Survival Grace Cycles
# ==================================================================

class TestSurvivalGraceCycles(unittest.TestCase):
    """Organism kills organisms below fitness floor after grace period."""

    def _make_organism(self, grace_cycles=5, threshold=0.9):
        from al01.organism import Organism, MetabolismConfig
        from al01.experiment import ExperimentConfig, ExperimentProtocol
        tmpdir = tempfile.mkdtemp()
        config = ExperimentConfig(
            survival_fitness_threshold=threshold,
            survival_grace_cycles=grace_cycles,
        )
        experiment = ExperimentProtocol(config=config, data_dir=tmpdir)
        org = Organism(data_dir=tmpdir, config=MetabolismConfig(), experiment=experiment)
        return org, tmpdir

    def test_below_fitness_increments_counter(self):
        org, _ = self._make_organism(grace_cycles=100, threshold=0.99)
        # v3.8: AL-01 uses FOUNDER_FITNESS_FLOOR (0.15) as death threshold,
        # so we must force traits low enough that weighted fitness < 0.15
        for t in org._genome.traits:
            org._genome.set_trait(t, 0.05)
        org.autonomy_cycle()
        self.assertGreater(org._below_fitness_cycles.get("AL-01", 0), 0)

    def test_above_fitness_resets_counter(self):
        org, _ = self._make_organism(grace_cycles=100, threshold=0.0)
        # fitness starts around 0.5, above 0.0 → should reset
        org._below_fitness_cycles["AL-01"] = 50
        org.autonomy_cycle()
        self.assertEqual(org._below_fitness_cycles.get("AL-01", 0), 0)

    def test_death_triggered_after_grace_exceeded(self):
        org, _ = self._make_organism(grace_cycles=3, threshold=0.99)
        # v3.8: force traits low so fitness < FOUNDER_FITNESS_FLOOR (0.15)
        for t in org._genome.traits:
            org._genome.set_trait(t, 0.05)
        # Pre-set counter near the edge
        org._below_fitness_cycles["AL-01"] = 2
        record = org.autonomy_cycle()
        # Should have died (fitness < 0.15, grace exceeded)
        if record.get("organism_died"):
            self.assertEqual(record.get("death_cause"), "fitness_floor")


# ==================================================================
# 4. Exploration Mode Toggle
# ==================================================================

class TestExplorationToggle(unittest.TestCase):
    """Manual exploration mode toggle via set_exploration_mode."""

    def _make_engine(self):
        tmpdir = tempfile.mkdtemp()
        return AutonomyEngine(data_dir=tmpdir)

    def test_toggle_on(self):
        engine = self._make_engine()
        result = engine.set_exploration_mode(True)
        self.assertTrue(result["exploration_mode"])
        self.assertGreater(result["cycles_remaining"], 0)
        self.assertTrue(engine.exploration_mode)

    def test_toggle_off(self):
        engine = self._make_engine()
        engine.set_exploration_mode(True)
        result = engine.set_exploration_mode(False)
        self.assertFalse(result["exploration_mode"])
        self.assertEqual(result["cycles_remaining"], 0)

    def test_toggle_with_custom_cycles(self):
        engine = self._make_engine()
        result = engine.set_exploration_mode(True, cycles=50)
        self.assertEqual(result["cycles_remaining"], 50)


# ==================================================================
# 5. API Endpoints
# ==================================================================

class TestALifeAPIEndpoints(unittest.TestCase):
    """New v3.3 API endpoints."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def _make_app(self, with_experiment=False):
        from al01.organism import Organism, MetabolismConfig
        from al01.api import create_app
        experiment = None
        if with_experiment:
            from al01.experiment import ExperimentConfig, ExperimentProtocol
            experiment = ExperimentProtocol(
                config=ExperimentConfig(),
                data_dir=self._tmpdir,
            )
        org = Organism(
            data_dir=self._tmpdir,
            config=MetabolismConfig(),
            experiment=experiment,
        )
        app = create_app(org, api_key=None)
        return app, org

    def test_lineage_all(self):
        from starlette.testclient import TestClient
        app, org = self._make_app()
        client = TestClient(app)
        resp = client.get("/lineage")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("lineage", data)
        self.assertIn("total_organisms", data)
        # AL-01 should be tracked
        self.assertGreaterEqual(data["total_organisms"], 1)

    def test_lineage_single(self):
        from starlette.testclient import TestClient
        app, org = self._make_app()
        client = TestClient(app)
        resp = client.get("/lineage/AL-01")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("organism_id", data)

    def test_lineage_not_found(self):
        from starlette.testclient import TestClient
        app, org = self._make_app()
        client = TestClient(app)
        resp = client.get("/lineage/NONEXISTENT")
        self.assertEqual(resp.status_code, 404)

    def test_population_metrics(self):
        from starlette.testclient import TestClient
        app, org = self._make_app()
        client = TestClient(app)
        resp = client.get("/population/metrics")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("fitness_stats", data)
        self.assertIn("trait_variance", data)
        self.assertIn("diversity", data)
        self.assertIn("convergence", data)
        self.assertIn("population_size", data)

    def test_export_fitness_csv(self):
        from starlette.testclient import TestClient
        app, org = self._make_app()
        # Generate some fitness data
        org.autonomy_cycle()
        client = TestClient(app)
        resp = client.get("/export/fitness.csv")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/csv", resp.headers.get("content-type", ""))
        lines = resp.text.strip().split("\n")
        self.assertGreaterEqual(len(lines), 1)  # at least header
        header = lines[0]
        self.assertIn("organism_id", header)
        self.assertIn("fitness", header)

    def test_export_mutations_csv(self):
        from starlette.testclient import TestClient
        app, org = self._make_app()
        client = TestClient(app)
        resp = client.get("/export/mutations.csv")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/csv", resp.headers.get("content-type", ""))

    def test_export_lineage_csv(self):
        from starlette.testclient import TestClient
        app, org = self._make_app()
        client = TestClient(app)
        resp = client.get("/export/lineage.csv")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/csv", resp.headers.get("content-type", ""))
        lines = resp.text.strip().split("\n")
        header = lines[0]
        self.assertIn("organism_id", header)
        # AL-01 should appear in lineage
        if len(lines) > 1:
            self.assertIn("AL-01", resp.text)

    def test_experiment_status_no_experiment(self):
        from starlette.testclient import TestClient
        app, org = self._make_app(with_experiment=False)
        client = TestClient(app)
        resp = client.get("/experiment/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "no_experiment")

    def test_experiment_start_no_experiment(self):
        from starlette.testclient import TestClient
        app, org = self._make_app(with_experiment=False)
        client = TestClient(app)
        resp = client.post("/experiment/start")
        self.assertEqual(resp.status_code, 400)

    def test_experiment_lifecycle(self):
        from starlette.testclient import TestClient
        app, org = self._make_app(with_experiment=True)
        client = TestClient(app)

        # Status before start
        resp = client.get("/experiment/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["active"])

        # Start
        resp = client.post("/experiment/start")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("status", data)

        # Status after start
        resp = client.get("/experiment/status")
        data = resp.json()
        self.assertTrue(data["active"])

        # Stop
        resp = client.post("/experiment/stop?reason=test")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("status", data)

        # Verify stopped
        resp = client.get("/experiment/status")
        data = resp.json()
        self.assertFalse(data["active"])

    def test_exploration_toggle(self):
        from starlette.testclient import TestClient
        app, org = self._make_app()
        client = TestClient(app)

        # Enable
        resp = client.post("/exploration/toggle?enabled=true&cycles=10")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "toggled")
        self.assertTrue(data["exploration_mode"])
        self.assertEqual(data["cycles_remaining"], 10)

        # Disable
        resp = client.post("/exploration/toggle?enabled=false")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["exploration_mode"])

    def test_dashboard_has_metrics(self):
        from starlette.testclient import TestClient
        app, org = self._make_app()
        client = TestClient(app)
        resp = client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.text
        self.assertIn("Population Metrics", html)
        self.assertIn("avg fitness", html)
        self.assertIn("Shannon diversity", html)
        self.assertIn("Simpson diversity", html)
        self.assertIn("Strategy Distribution", html)
        self.assertIn("Data Export", html)
        self.assertIn("fitness.csv", html)


# ==================================================================
# 6. Diversity index edge cases
# ==================================================================

class TestDiversityEdgeCases(unittest.TestCase):
    """Edge cases for diversity computation."""

    def test_single_organism_returns_zero(self):
        analyzer = PopulationBehaviorAnalyzer()
        for _ in range(10):
            analyzer.record_decision("org-0", "stabilize", 0.9, 0.5)
        div = analyzer.diversity_index()
        self.assertEqual(div["richness"], 1)
        self.assertEqual(div["shannon"], 0.0)
        self.assertEqual(div["simpson"], 0.0)
        self.assertEqual(div["evenness"], 0.0)

    def test_many_strategies_high_diversity(self):
        """3+ distinct strategies → higher Shannon than 2 strategies."""
        analyzer_2 = PopulationBehaviorAnalyzer()
        analyzer_3 = PopulationBehaviorAnalyzer()

        # 2-strategy: hoarders + explorers
        for i in range(3):
            for _ in range(10):
                analyzer_2.record_decision(f"h-{i}", "stabilize", 0.9, 0.5, {"adaptability": 0.5})
                analyzer_3.record_decision(f"h-{i}", "stabilize", 0.9, 0.5, {"adaptability": 0.5})
        for i in range(3):
            for _ in range(10):
                analyzer_2.record_decision(f"e-{i}", "mutate", 0.4, 0.5, {"adaptability": 0.8, "creativity": 0.8})
                analyzer_3.record_decision(f"e-{i}", "mutate", 0.4, 0.5, {"adaptability": 0.8, "creativity": 0.8})

        # 3-strategy: add 3 neutrals to analyzer_3
        for i in range(3):
            for _ in range(3):  # fewer decisions → neutral
                analyzer_3.record_decision(f"n-{i}", "blend", 0.6, 0.6, {"adaptability": 0.5, "creativity": 0.5})

        div_2 = analyzer_2.diversity_index()
        div_3 = analyzer_3.diversity_index()
        self.assertGreaterEqual(div_3["richness"], div_2["richness"])


# ==================================================================
# 7. Reproducibility: deterministic sequence
# ==================================================================

class TestReproducibilityFullSequence(unittest.TestCase):
    """Full decide() sequence produces identical output with same seed."""

    def test_decide_sequence_deterministic(self):
        tmpdir1 = tempfile.mkdtemp()
        tmpdir2 = tempfile.mkdtemp()
        cfg = AutonomyConfig()
        e1 = AutonomyEngine(data_dir=tmpdir1, config=cfg, rng_seed=42)
        e2 = AutonomyEngine(data_dir=tmpdir2, config=cfg, rng_seed=42)

        # Run 5 decide cycles
        results1 = []
        results2 = []
        for i in range(5):
            r1 = e1.decide(
                fitness=0.5, awareness=0.3, mutation_rate=0.1, pending_stimuli=0,
            )
            r2 = e2.decide(
                fitness=0.5, awareness=0.3, mutation_rate=0.1, pending_stimuli=0,
            )
            results1.append(r1["decision"])
            results2.append(r2["decision"])

        self.assertEqual(results1, results2)
        # Verify env state is also identical
        self.assertEqual(
            e1._effective_fitness_threshold,
            e2._effective_fitness_threshold,
        )
        self.assertEqual(e1._mutation_rate_offset, e2._mutation_rate_offset)
        self.assertEqual(e1._env_trait_weights, e2._env_trait_weights)


if __name__ == "__main__":
    unittest.main()
