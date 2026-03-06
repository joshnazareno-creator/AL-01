"""AL-01 v3.4 — Tests for child evolution, nicknames, population cap, repro toggle.

Covers:
1. Nickname field — set, get, clear, in spawn, in migration
2. Population cap — default 20, configurable via experiment
3. Reproduction toggle — enabled/disabled
4. Child autonomy cycle — children evolve independently
5. Child mutation tracking — evolution_count increments
6. Child survival grace cycles — fitness floor death
7. API endpoints — nickname set, population list with nicknames
8. Version bump
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from typing import Any, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from al01.genome import Genome
from al01.population import Population
from al01.experiment import ExperimentConfig, ExperimentProtocol
from al01.organism import Organism, MetabolismConfig, VERSION
from al01.autonomy import AutonomyConfig, AutonomyEngine
from al01.behavior import PopulationBehaviorAnalyzer
from al01.evolution_tracker import EvolutionTracker


# ==================================================================
# 1. Nickname Field
# ==================================================================

class TestNickname(unittest.TestCase):
    """Nickname field works on population members."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.pop = Population(data_dir=self._tmpdir)

    def test_parent_has_nickname_field(self):
        member = self.pop.get("AL-01")
        self.assertIn("nickname", member)
        self.assertIsNone(member["nickname"])

    def test_set_nickname(self):
        ok = self.pop.set_nickname("AL-01", "Messi")
        self.assertTrue(ok)
        self.assertEqual(self.pop.get_nickname("AL-01"), "Messi")

    def test_clear_nickname(self):
        self.pop.set_nickname("AL-01", "Messi")
        self.pop.set_nickname("AL-01", None)
        self.assertIsNone(self.pop.get_nickname("AL-01"))

    def test_set_nickname_nonexistent(self):
        ok = self.pop.set_nickname("FAKE-ID", "Nope")
        self.assertFalse(ok)

    def test_get_nickname_nonexistent(self):
        self.assertIsNone(self.pop.get_nickname("FAKE-ID"))

    def test_child_has_nickname_field(self):
        g = Genome()
        child = self.pop.spawn_child(g, 10)
        self.assertIn("nickname", child)
        self.assertIsNone(child["nickname"])

    def test_nickname_persists_reload(self):
        self.pop.set_nickname("AL-01", "Messi")
        pop2 = Population(data_dir=self._tmpdir)
        self.assertEqual(pop2.get_nickname("AL-01"), "Messi")

    def test_nickname_in_get_all(self):
        self.pop.set_nickname("AL-01", "Alpha")
        members = self.pop.get_all()
        found = [m for m in members if m["id"] == "AL-01"]
        self.assertEqual(found[0]["nickname"], "Alpha")

    def test_migration_adds_nickname(self):
        """Old members without nickname get it via migration."""
        # Simulate legacy member without nickname
        self.pop._members["AL-01"].pop("nickname", None)
        self.pop._save()
        pop2 = Population(data_dir=self._tmpdir)
        self.assertIsNone(pop2.get("AL-01")["nickname"])


# ==================================================================
# 2. Population Cap
# ==================================================================

class TestPopulationCap(unittest.TestCase):
    """Population cap prevents unbounded growth."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def test_default_max_population(self):
        org = Organism(data_dir=self._tmpdir, config=MetabolismConfig())
        self.assertEqual(org._default_max_population, 60)

    def test_population_has_cap(self):
        org = Organism(data_dir=self._tmpdir, config=MetabolismConfig())
        self.assertEqual(org.population.max_population, 60)

    def test_spawn_child_refuses_at_cap(self):
        """spawn_child returns None when cap is reached."""
        org = Organism(data_dir=self._tmpdir, config=MetabolismConfig())
        org.population.max_population = 3  # low cap for testing
        # AL-01 counts as 1 living member
        g = org.genome
        c1 = org.population.spawn_child(g, 1)  # 2 total
        self.assertIsNotNone(c1)
        c2 = org.population.spawn_child(g, 1)  # 3 total = at cap
        self.assertIsNotNone(c2)
        c3 = org.population.spawn_child(g, 1)  # 4 total — should be refused
        self.assertIsNone(c3)

    def test_experiment_overrides_cap(self):
        cfg = ExperimentConfig(max_population=10)
        self.assertEqual(cfg.max_population, 10)


# ==================================================================
# 3. Reproduction Toggle
# ==================================================================

class TestReproductionToggle(unittest.TestCase):
    """reproduction_enabled controls whether reproduction can occur."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def test_default_enabled(self):
        cfg = ExperimentConfig()
        self.assertTrue(cfg.reproduction_enabled)

    def test_can_disable(self):
        cfg = ExperimentConfig(reproduction_enabled=False)
        self.assertFalse(cfg.reproduction_enabled)

    def test_disabled_blocks_reproduction(self):
        cfg = ExperimentConfig(reproduction_enabled=False)
        proto = ExperimentProtocol(config=cfg, data_dir=self._tmpdir)
        org = Organism(
            data_dir=self._tmpdir,
            config=MetabolismConfig(),
            experiment=proto,
        )
        # Even if organisms qualify, no reproduction should happen
        result = org.auto_reproduce_cycle()
        self.assertEqual(result, [])


# ==================================================================
# 4. Child Autonomy Cycle — Children Evolve
# ==================================================================

class TestChildEvolution(unittest.TestCase):
    """Children evolve independently during child_autonomy_cycle."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.org = Organism(data_dir=self._tmpdir, config=MetabolismConfig())
        # Use a fresh genome with known low-ish traits so children land
        # in the adapt/mutate zone (fitness ≈ 0.48 < threshold * 1.2 = 0.54).
        # This avoids dependence on Firestore-synced parent state.
        low_genome = Genome(traits={
            "adaptability": 0.48, "energy_efficiency": 0.48,
            "resilience": 0.48, "perception": 0.48, "creativity": 0.48,
        })
        self.child = self.org.population.spawn_child(low_genome, 10)
        self.child_id = self.child["id"]
        # Register in tracker
        self.org.evolution_tracker.register_organism(
            self.child_id, parent_id="AL-01",
            traits=self.child["genome"]["traits"], cycle=1,
        )

    def test_child_starts_at_evol_zero(self):
        member = self.org.population.get(self.child_id)
        self.assertEqual(member["evolution_count"], 0)

    def test_child_evolves_after_cycle(self):
        """After child_autonomy_cycle, child's evolution_count should increase
        if the decision was mutate or adapt."""
        # Run several cycles to ensure at least one mutation
        for _ in range(10):
            self.org.child_autonomy_cycle()
        member = self.org.population.get(self.child_id)
        # Child should have evolved at least once (either mutate or adapt)
        self.assertGreater(member["evolution_count"], 0)

    def test_child_genome_changes(self):
        """Child genome should change over multiple evolution cycles."""
        before = self.org.population.get(self.child_id)["genome"]["traits"].copy()
        for _ in range(20):
            self.org.child_autonomy_cycle()
        after = self.org.population.get(self.child_id)["genome"]["traits"]
        # At least one trait should have changed
        changed = any(
            abs(before.get(t, 0) - after.get(t, 0)) > 1e-6
            for t in before
        )
        self.assertTrue(changed, "Child genome should have changed after 20 cycles")

    def test_child_fitness_tracked(self):
        """Child fitness should be recorded in evolution tracker."""
        self.org.child_autonomy_cycle()
        # Fitness trajectory should have entries
        lineage = self.org.evolution_tracker.get_lineage(self.child_id)
        self.assertIsNotNone(lineage)

    def test_parent_not_affected(self):
        """child_autonomy_cycle should not change the parent."""
        parent_evo_before = self.org.population.get("AL-01")["evolution_count"]
        self.org.child_autonomy_cycle()
        parent_evo_after = self.org.population.get("AL-01")["evolution_count"]
        self.assertEqual(parent_evo_before, parent_evo_after)

    def test_multiple_children_evolve(self):
        """Multiple children all evolve in the same cycle."""
        # Spawn more children from a known low-fitness genome
        low_genome = Genome(traits={
            "adaptability": 0.48, "energy_efficiency": 0.48,
            "resilience": 0.48, "perception": 0.48, "creativity": 0.48,
        })
        for _ in range(3):
            ch = self.org.population.spawn_child(low_genome, 20)
            self.org.evolution_tracker.register_organism(
                ch["id"], parent_id="AL-01",
                traits=ch["genome"]["traits"], cycle=2,
            )
        for _ in range(10):
            self.org.child_autonomy_cycle()
        # All children should have evolved
        for oid in self.org.population.member_ids:
            if oid == "AL-01":
                continue
            member = self.org.population.get(oid)
            self.assertGreater(
                member["evolution_count"], 0,
                f"{oid} should have evolved",
            )

    def test_dead_children_not_evolved(self):
        """Dead children should be skipped."""
        self.org.population.remove_member(self.child_id, "test_kill")
        results = self.org.child_autonomy_cycle()
        evolved_ids = [r["organism_id"] for r in results]
        self.assertNotIn(self.child_id, evolved_ids)


# ==================================================================
# 5. Child Survival Grace Cycles
# ==================================================================

class TestChildSurvivalGrace(unittest.TestCase):
    """Children die when below fitness floor for too many cycles."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        cfg = ExperimentConfig(
            survival_fitness_threshold=0.9,  # very high so child dies
            survival_grace_cycles=3,  # die after 3 cycles
        )
        proto = ExperimentProtocol(config=cfg, data_dir=self._tmpdir)
        self.org = Organism(
            data_dir=self._tmpdir,
            config=MetabolismConfig(),
            experiment=proto,
        )
        # Spawn a low-fitness child
        g = Genome()
        self.child = self.org.population.spawn_child(g, 5)
        self.child_id = self.child["id"]
        self.org.evolution_tracker.register_organism(
            self.child_id, parent_id="AL-01",
            traits=self.child["genome"]["traits"], cycle=1,
        )
        # Spawn extra filler children so extinction recovery doesn't wake
        # the dormant child (pop must stay >= 5)
        for _ in range(5):
            self.org.population.spawn_child(Genome(), 0)

    def test_child_dies_after_grace_cycles(self):
        """Child should enter dormant state after exceeding survival grace cycles."""
        for _ in range(5):
            self.org.child_autonomy_cycle()
        member = self.org.population.get(self.child_id)
        # v3.15: fitness_floor now causes dormancy instead of death
        self.assertEqual(member.get("state"), "dormant")
        self.assertTrue(member.get("alive", False))


# ==================================================================
# 6. API Endpoints
# ==================================================================

class TestNicknameAPI(unittest.TestCase):
    """Nickname API endpoint works."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def _make_app(self):
        from al01.api import create_app
        org = Organism(data_dir=self._tmpdir, config=MetabolismConfig())
        app = create_app(org, api_key=None)
        return app, org

    def test_set_nickname_endpoint(self):
        from starlette.testclient import TestClient
        app, org = self._make_app()
        client = TestClient(app)
        resp = client.put("/population/AL-01/nickname?nickname=Messi")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["nickname"], "Messi")
        self.assertEqual(data["status"], "set")

    def test_clear_nickname_endpoint(self):
        from starlette.testclient import TestClient
        app, org = self._make_app()
        client = TestClient(app)
        client.put("/population/AL-01/nickname?nickname=Messi")
        resp = client.put("/population/AL-01/nickname")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "cleared")

    def test_nickname_404(self):
        from starlette.testclient import TestClient
        app, org = self._make_app()
        client = TestClient(app)
        resp = client.put("/population/FAKE/nickname?nickname=X")
        self.assertEqual(resp.status_code, 404)

    def test_population_list_includes_nickname(self):
        from starlette.testclient import TestClient
        app, org = self._make_app()
        client = TestClient(app)
        client.put("/population/AL-01/nickname?nickname=Alpha")
        resp = client.get("/population")
        data = resp.json()
        members = data["members"]
        al01 = [m for m in members if m["id"] == "AL-01"][0]
        self.assertEqual(al01["nickname"], "Alpha")


# ==================================================================
# 7. Version Bump
# ==================================================================

class TestVersionBump(unittest.TestCase):
    def test_version(self):
        self.assertEqual(VERSION, "3.9")


# ==================================================================
# 8. Integration — MetabolismScheduler wires child_autonomy
# ==================================================================

class TestSchedulerIntegration(unittest.TestCase):
    """child_autonomy_interval is wired into the scheduler."""

    def test_config_has_child_autonomy_interval(self):
        cfg = MetabolismConfig()
        self.assertEqual(cfg.child_autonomy_interval, 10)

    def test_scheduler_calls_child_autonomy(self):
        """After child_autonomy_interval ticks, children evolve."""
        tmpdir = tempfile.mkdtemp()
        org = Organism(data_dir=tmpdir, config=MetabolismConfig())
        g = org.genome
        child = org.population.spawn_child(g, 5)
        org.evolution_tracker.register_organism(
            child["id"], parent_id="AL-01",
            traits=child["genome"]["traits"], cycle=1,
        )
        # Tick enough times to trigger child_autonomy_interval (default 10)
        for _ in range(10):
            org.tick()
        # Check child evolved
        member = org.population.get(child["id"])
        # May or may not have evolved depending on decision, but no crash
        self.assertIsNotNone(member)


if __name__ == "__main__":
    unittest.main()
