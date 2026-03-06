"""AL-01 v3.2 — Tests for the Genesis Vault (extinction recovery from frozen seed).

Covers:
1. GenesisVault creation and seed immutability
2. Vault file persistence and reload
3. Genome factory from seed
4. Automatic reseed when population hits 0
5. Reseed event logged in life_log
6. Reseed registered in evolution_tracker
7. Multiple reseeds track history
8. No reseed when population alive
9. Reseed counter persistence across reload
10. CLI commands (vault, vault-history)
11. API endpoints (/vault, /vault/seed, /vault/history, /vault/reseed)
12. Organism integration (genesis_vault property, check_extinction_reseed)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from copy import deepcopy
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from al01.genesis_vault import GenesisVault, _GENESIS_SEED
from al01.genome import DEFAULT_TRAITS, DEFAULT_MUTATION_RATE, DEFAULT_MUTATION_DELTA, Genome
from al01.population import Population
from al01.life_log import LifeLog
from al01.evolution_tracker import EvolutionTracker


# ==================================================================
# 1. GenesisVault creation & seed immutability
# ==================================================================

class TestGenesisVaultCreation(unittest.TestCase):
    """Vault creates a frozen seed on first init and never changes it."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.vault = GenesisVault(data_dir=self._tmpdir)

    def test_vault_created(self):
        self.assertTrue(os.path.exists(self.vault.vault_path))

    def test_seed_name(self):
        self.assertEqual(self.vault.seed["seed_name"], "AL-01 Genesis Seed")

    def test_seed_traits_match_defaults(self):
        for trait, value in DEFAULT_TRAITS.items():
            self.assertAlmostEqual(self.vault.seed["traits"][trait], value)

    def test_seed_mutation_rate(self):
        self.assertEqual(self.vault.seed["mutation_rate"], DEFAULT_MUTATION_RATE)

    def test_seed_mutation_delta(self):
        self.assertEqual(self.vault.seed["mutation_delta"], DEFAULT_MUTATION_DELTA)

    def test_seed_has_created_at(self):
        self.assertIn("created_at", self.vault.seed)
        self.assertIsNotNone(self.vault.seed["created_at"])

    def test_initial_reseed_count_zero(self):
        self.assertEqual(self.vault.reseed_count, 0)

    def test_initial_reseed_history_empty(self):
        self.assertEqual(self.vault.reseed_history, [])

    def test_seed_is_deep_copy(self):
        """Modifying the returned seed dict does not change vault internal state."""
        seed = self.vault.seed
        seed["traits"]["adaptability"] = 999.0
        self.assertAlmostEqual(self.vault.seed["traits"]["adaptability"], DEFAULT_TRAITS["adaptability"])

    def test_seed_traits_are_frozen(self):
        """Multiple reads return the same values."""
        s1 = self.vault.seed
        s2 = self.vault.seed
        self.assertEqual(s1["traits"], s2["traits"])


# ==================================================================
# 2. Vault file persistence and reload
# ==================================================================

class TestVaultPersistence(unittest.TestCase):
    """Vault persists to disk and can be reloaded."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def test_reload_from_disk(self):
        vault1 = GenesisVault(data_dir=self._tmpdir)
        seed1 = vault1.seed

        vault2 = GenesisVault(data_dir=self._tmpdir)
        seed2 = vault2.seed

        self.assertEqual(seed1["seed_name"], seed2["seed_name"])
        self.assertEqual(seed1["traits"], seed2["traits"])
        self.assertEqual(seed1["created_at"], seed2["created_at"])

    def test_vault_file_is_valid_json(self):
        vault = GenesisVault(data_dir=self._tmpdir)
        with open(vault.vault_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertIn("traits", data)
        self.assertIn("seed_name", data)

    def test_reload_preserves_reseed_count(self):
        vault = GenesisVault(data_dir=self._tmpdir)
        # Simulate a reseed by manually incrementing
        pop_dir = tempfile.mkdtemp()
        pop = Population(data_dir=pop_dir, parent_id="AL-01")
        # Kill the parent
        pop.remove_member("AL-01", cause="test")

        vault.check_and_reseed(population=pop, global_cycle=10)
        self.assertEqual(vault.reseed_count, 1)

        # Reload
        vault2 = GenesisVault(data_dir=self._tmpdir)
        self.assertEqual(vault2.reseed_count, 1)

    def test_corrupted_file_falls_back_to_default(self):
        vault_path = os.path.join(self._tmpdir, "genesis_vault.json")
        with open(vault_path, "w") as fh:
            fh.write("NOT VALID JSON {{{")

        vault = GenesisVault(data_dir=self._tmpdir)
        self.assertEqual(vault.seed["seed_name"], "AL-01 Genesis Seed")


# ==================================================================
# 3. Genome factory from seed
# ==================================================================

class TestGenomeFactory(unittest.TestCase):
    """create_genome_from_seed returns a fresh Genome with default traits."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.vault = GenesisVault(data_dir=self._tmpdir)

    def test_genome_type(self):
        genome = self.vault.create_genome_from_seed()
        self.assertIsInstance(genome, Genome)

    def test_genome_traits_match_seed(self):
        genome = self.vault.create_genome_from_seed()
        for trait, value in DEFAULT_TRAITS.items():
            self.assertAlmostEqual(genome.get_trait(trait), value)

    def test_genome_mutation_rate(self):
        genome = self.vault.create_genome_from_seed()
        self.assertEqual(genome.mutation_rate, DEFAULT_MUTATION_RATE)

    def test_each_genome_is_independent(self):
        g1 = self.vault.create_genome_from_seed()
        g2 = self.vault.create_genome_from_seed()
        # Mutate one, other should be unaffected
        g1.set_trait("adaptability", 0.99)
        self.assertAlmostEqual(g2.get_trait("adaptability"), DEFAULT_TRAITS["adaptability"])


# ==================================================================
# 4. Automatic reseed when population hits 0
# ==================================================================

class TestAutoReseed(unittest.TestCase):
    """check_and_reseed spawns a new organism when population is extinct."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._pop_dir = tempfile.mkdtemp()
        self.vault = GenesisVault(data_dir=self._tmpdir)
        self.pop = Population(data_dir=self._pop_dir, parent_id="AL-01")

    def _kill_all(self):
        for mid in list(self.pop.member_ids):
            self.pop.remove_member(mid, cause="test_extinction")

    def test_reseed_when_extinct(self):
        self._kill_all()
        self.assertEqual(self.pop.size, 0)

        result = self.vault.check_and_reseed(population=self.pop, global_cycle=42)

        self.assertIsNotNone(result)
        self.assertEqual(result["event"], "reseed")
        self.assertEqual(result["reseed_number"], 1)
        self.assertEqual(result["cycle"], 42)
        self.assertEqual(result["reason"], "population_extinct")

    def test_population_restored_after_reseed(self):
        self._kill_all()
        self.vault.check_and_reseed(population=self.pop, global_cycle=1)
        self.assertEqual(self.pop.size, 1)

    def test_reseed_organism_has_correct_id(self):
        self._kill_all()
        result = self.vault.check_and_reseed(population=self.pop, global_cycle=1)
        self.assertIn("AL-01-child-", result["organism_id"])

    def test_reseed_organism_is_alive(self):
        self._kill_all()
        result = self.vault.check_and_reseed(population=self.pop, global_cycle=1)
        member = self.pop.get(result["organism_id"])
        self.assertTrue(member["alive"])

    def test_reseed_seed_traits_in_record(self):
        self._kill_all()
        result = self.vault.check_and_reseed(population=self.pop, global_cycle=1)
        self.assertIn("seed_traits", result)
        for trait in DEFAULT_TRAITS:
            self.assertIn(trait, result["seed_traits"])

    def test_reseed_child_fitness_positive(self):
        self._kill_all()
        result = self.vault.check_and_reseed(population=self.pop, global_cycle=1)
        self.assertGreater(result["child_fitness"], 0)


# ==================================================================
# 5. Reseed event logged in life_log
# ==================================================================

class TestReseedLifeLog(unittest.TestCase):
    """Reseed events are appended to the life log."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._pop_dir = tempfile.mkdtemp()
        self._log_dir = tempfile.mkdtemp()
        self.vault = GenesisVault(data_dir=self._tmpdir)
        self.pop = Population(data_dir=self._pop_dir, parent_id="AL-01")
        self.life_log = LifeLog(data_dir=self._log_dir, organism_id="AL-01")

    def _kill_all(self):
        for mid in list(self.pop.member_ids):
            self.pop.remove_member(mid, cause="test")

    def test_reseed_event_in_log(self):
        self._kill_all()
        self.vault.check_and_reseed(
            population=self.pop, life_log=self.life_log, global_cycle=5,
        )
        events = self.life_log._read_tail(10)
        reseed_events = [e for e in events if e.get("event_type") == "reseed"]
        self.assertEqual(len(reseed_events), 1)

    def test_reseed_log_payload(self):
        self._kill_all()
        self.vault.check_and_reseed(
            population=self.pop, life_log=self.life_log, global_cycle=5,
        )
        events = self.life_log._read_tail(10)
        reseed_events = [e for e in events if e.get("event_type") == "reseed"]
        payload = reseed_events[0]["payload"]
        self.assertEqual(payload["event"], "reseed")
        self.assertEqual(payload["reseed_number"], 1)
        self.assertEqual(payload["reason"], "population_extinct")


# ==================================================================
# 6. Reseed registered in evolution_tracker
# ==================================================================

class TestReseedEvolutionTracker(unittest.TestCase):
    """Reseed organisms are registered in the evolution tracker."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._pop_dir = tempfile.mkdtemp()
        self._tracker_dir = tempfile.mkdtemp()
        self.vault = GenesisVault(data_dir=self._tmpdir)
        self.pop = Population(data_dir=self._pop_dir, parent_id="AL-01")
        self.tracker = EvolutionTracker(data_dir=self._tracker_dir)

    def _kill_all(self):
        for mid in list(self.pop.member_ids):
            self.pop.remove_member(mid, cause="test")

    def test_reseeded_organism_registered(self):
        self._kill_all()
        result = self.vault.check_and_reseed(
            population=self.pop, evolution_tracker=self.tracker, global_cycle=10,
        )
        lineage = self.tracker.get_lineage(result["organism_id"])
        self.assertIsNotNone(lineage)

    def test_reseeded_organism_has_no_parent(self):
        self._kill_all()
        result = self.vault.check_and_reseed(
            population=self.pop, evolution_tracker=self.tracker, global_cycle=10,
        )
        lineage = self.tracker.get_lineage(result["organism_id"])
        self.assertIsNone(lineage.get("parent_id"))


# ==================================================================
# 7. Multiple reseeds track history
# ==================================================================

class TestMultipleReseeds(unittest.TestCase):
    """Multiple extinctions result in incremented reseed counters."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._pop_dir = tempfile.mkdtemp()
        self.vault = GenesisVault(data_dir=self._tmpdir)
        self.pop = Population(data_dir=self._pop_dir, parent_id="AL-01")

    def _kill_all(self):
        for mid in list(self.pop.member_ids):
            self.pop.remove_member(mid, cause="test")

    def test_three_reseeds(self):
        for i in range(3):
            self._kill_all()
            result = self.vault.check_and_reseed(population=self.pop, global_cycle=i * 100)
            self.assertIsNotNone(result)
            self.assertEqual(result["reseed_number"], i + 1)

        self.assertEqual(self.vault.reseed_count, 3)

    def test_history_grows(self):
        for i in range(3):
            self._kill_all()
            self.vault.check_and_reseed(population=self.pop, global_cycle=i * 100)

        history = self.vault.reseed_history
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]["reseed_number"], 1)
        self.assertEqual(history[2]["reseed_number"], 3)

    def test_different_organisms_per_reseed(self):
        ids = []
        for i in range(3):
            self._kill_all()
            result = self.vault.check_and_reseed(population=self.pop, global_cycle=i * 100)
            ids.append(result["organism_id"])
        # All IDs should be unique
        self.assertEqual(len(set(ids)), 3)


# ==================================================================
# 8. No reseed when population alive
# ==================================================================

class TestNoReseedWhenAlive(unittest.TestCase):
    """check_and_reseed returns None if population has living members."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._pop_dir = tempfile.mkdtemp()
        self.vault = GenesisVault(data_dir=self._tmpdir)
        self.pop = Population(data_dir=self._pop_dir, parent_id="AL-01")

    def test_no_reseed_with_living_population(self):
        self.assertGreater(self.pop.size, 0)
        result = self.vault.check_and_reseed(population=self.pop, global_cycle=1)
        self.assertIsNone(result)

    def test_reseed_count_unchanged(self):
        self.vault.check_and_reseed(population=self.pop, global_cycle=1)
        self.assertEqual(self.vault.reseed_count, 0)


# ==================================================================
# 9. Status method
# ==================================================================

class TestVaultStatus(unittest.TestCase):
    """status() returns a complete diagnostic dict."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.vault = GenesisVault(data_dir=self._tmpdir)

    def test_status_keys(self):
        status = self.vault.status()
        expected_keys = {"seed_name", "created_at", "reseed_count", "seed_traits",
                         "mutation_rate", "mutation_delta", "last_reseed"}
        self.assertTrue(expected_keys.issubset(status.keys()))

    def test_status_no_last_reseed(self):
        status = self.vault.status()
        self.assertIsNone(status["last_reseed"])

    def test_status_after_reseed(self):
        pop_dir = tempfile.mkdtemp()
        pop = Population(data_dir=pop_dir, parent_id="AL-01")
        for mid in list(pop.member_ids):
            pop.remove_member(mid, cause="test")
        self.vault.check_and_reseed(population=pop, global_cycle=99)

        status = self.vault.status()
        self.assertEqual(status["reseed_count"], 1)
        self.assertIsNotNone(status["last_reseed"])
        self.assertEqual(status["last_reseed"]["cycle"], 99)


# ==================================================================
# 10. Organism integration
# ==================================================================

class TestOrganismIntegration(unittest.TestCase):
    """Organism exposes genesis_vault and check_extinction_reseed."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def _make_organism(self):
        from al01.organism import Organism, MetabolismConfig
        return Organism(
            data_dir=self._tmpdir,
            config=MetabolismConfig(),
        )

    def test_genesis_vault_property(self):
        org = self._make_organism()
        self.assertIsInstance(org.genesis_vault, GenesisVault)

    def test_check_extinction_reseed_no_action(self):
        """When population is alive and large enough, returns None."""
        org = self._make_organism()
        # Spawn enough children so pop >= 5 (extinction recovery threshold)
        from al01.genome import Genome
        from al01.population import BIRTH_COOLDOWN_CYCLES
        g = Genome()
        parent = org.population._members["AL-01"]
        for i in range(5):
            parent["last_birth_tick"] = -BIRTH_COOLDOWN_CYCLES * 2
            org.population.set_tick(i)
            org.population.spawn_child(g, parent_evolution=0, parent_id="AL-01")
        result = org.check_extinction_reseed()
        self.assertIsNone(result)

    def test_check_extinction_reseed_triggers(self):
        """When population is extinct, reseed fires."""
        org = self._make_organism()
        # Kill all living members
        for mid in list(org.population.member_ids):
            org.population.remove_member(mid, cause="test_extinction")
        self.assertEqual(org.population.size, 0)

        result = org.check_extinction_reseed()
        self.assertIsNotNone(result)
        self.assertEqual(result["event"], "reseed")
        self.assertEqual(org.population.size, 1)

    def test_handle_death_triggers_reseed(self):
        """When _handle_death causes extinction, vault auto-reseeds."""
        org = self._make_organism()
        # Ensure only AL-01 is alive
        self.assertEqual(org.population.size, 1)

        # Kill AL-01 via _handle_death
        org._handle_death("AL-01", "test_death")

        # Vault should have auto-reseeded
        self.assertGreater(org.population.size, 0)
        self.assertEqual(org.genesis_vault.reseed_count, 1)

    def test_reseed_logged_in_life_log(self):
        """Reseed event appears in the life log after extinction."""
        org = self._make_organism()
        for mid in list(org.population.member_ids):
            org.population.remove_member(mid, cause="test")
        org.check_extinction_reseed()

        events = org.life_log._read_tail(50)
        reseed_events = [e for e in events if e.get("event_type") == "reseed"]
        self.assertEqual(len(reseed_events), 1)

    def test_version_bumped(self):
        from al01.organism import VERSION
        self.assertEqual(VERSION, "3.9")


# ==================================================================
# 11. API endpoints
# ==================================================================

class TestAPIEndpoints(unittest.TestCase):
    """Genesis Vault API endpoints work correctly."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def _make_app(self):
        from al01.organism import Organism, MetabolismConfig
        from al01.api import create_app
        org = Organism(data_dir=self._tmpdir, config=MetabolismConfig())
        app = create_app(org, api_key=None)
        return app, org

    def test_vault_status_endpoint(self):
        from starlette.testclient import TestClient
        app, org = self._make_app()
        client = TestClient(app)
        resp = client.get("/vault")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["seed_name"], "AL-01 Genesis Seed")
        self.assertEqual(data["reseed_count"], 0)

    def test_vault_seed_endpoint(self):
        from starlette.testclient import TestClient
        app, org = self._make_app()
        client = TestClient(app)
        resp = client.get("/vault/seed")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("traits", data)
        for trait in DEFAULT_TRAITS:
            self.assertIn(trait, data["traits"])

    def test_vault_history_endpoint(self):
        from starlette.testclient import TestClient
        app, org = self._make_app()
        client = TestClient(app)
        resp = client.get("/vault/history")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["reseed_count"], 0)
        self.assertEqual(data["history"], [])

    def test_vault_reseed_endpoint_skips_when_alive(self):
        from starlette.testclient import TestClient
        app, org = self._make_app()
        client = TestClient(app)
        resp = client.post("/vault/reseed")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "skipped")

    def test_vault_reseed_endpoint_triggers_when_extinct(self):
        from starlette.testclient import TestClient
        app, org = self._make_app()
        client = TestClient(app)
        # Kill all
        for mid in list(org.population.member_ids):
            org.population.remove_member(mid, cause="test")
        resp = client.post("/vault/reseed")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "reseeded")
        self.assertIn("organism_id", data)


# ==================================================================
# 12. CLI commands
# ==================================================================

class TestCLICommands(unittest.TestCase):
    """CLI vault commands execute without errors."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        # Create a vault on disk for CLI to read
        self.vault = GenesisVault(data_dir=self._tmpdir)

    def test_cmd_vault_status_runs(self):
        from al01.cli import cmd_vault_status
        import argparse
        args = argparse.Namespace(data_dir=self._tmpdir)
        # Should not raise
        cmd_vault_status(args)

    def test_cmd_vault_history_runs_empty(self):
        from al01.cli import cmd_vault_history
        import argparse
        args = argparse.Namespace(data_dir=self._tmpdir)
        cmd_vault_history(args)

    def test_cmd_vault_history_runs_with_data(self):
        from al01.cli import cmd_vault_history
        import argparse

        # Create a reseed event
        pop_dir = tempfile.mkdtemp()
        pop = Population(data_dir=pop_dir, parent_id="AL-01")
        for mid in list(pop.member_ids):
            pop.remove_member(mid, cause="test")
        self.vault.check_and_reseed(population=pop, global_cycle=42)

        args = argparse.Namespace(data_dir=self._tmpdir)
        cmd_vault_history(args)


# ==================================================================
# 13. Edge cases
# ==================================================================

class TestEdgeCases(unittest.TestCase):
    """Edge cases and robustness."""

    def test_vault_in_nonexistent_dir(self):
        """Vault creates directory if it doesn't exist."""
        tmpdir = tempfile.mkdtemp()
        vault_dir = os.path.join(tmpdir, "deep", "nested", "dir")
        vault = GenesisVault(data_dir=vault_dir)
        self.assertTrue(os.path.exists(vault.vault_path))

    def test_reseed_with_no_optional_subsystems(self):
        """Reseed works even without life_log or evolution_tracker."""
        tmpdir = tempfile.mkdtemp()
        pop_dir = tempfile.mkdtemp()
        vault = GenesisVault(data_dir=tmpdir)
        pop = Population(data_dir=pop_dir, parent_id="AL-01")
        for mid in list(pop.member_ids):
            pop.remove_member(mid, cause="test")

        result = vault.check_and_reseed(population=pop, global_cycle=1)
        self.assertIsNotNone(result)
        self.assertEqual(pop.size, 1)

    def test_genesis_seed_constant_unchanged(self):
        """The module-level _GENESIS_SEED is not mutated."""
        tmpdir = tempfile.mkdtemp()
        pop_dir = tempfile.mkdtemp()
        vault = GenesisVault(data_dir=tmpdir)
        pop = Population(data_dir=pop_dir, parent_id="AL-01")
        for mid in list(pop.member_ids):
            pop.remove_member(mid, cause="test")

        original = deepcopy(_GENESIS_SEED)
        vault.check_and_reseed(population=pop, global_cycle=1)
        self.assertEqual(_GENESIS_SEED, original)

    def test_concurrent_vault_access(self):
        """Multiple vault instances reading same file don't corrupt."""
        tmpdir = tempfile.mkdtemp()
        v1 = GenesisVault(data_dir=tmpdir)
        v2 = GenesisVault(data_dir=tmpdir)
        self.assertEqual(v1.seed["traits"], v2.seed["traits"])


if __name__ == "__main__":
    unittest.main()
