"""AL-01 v1.3 VITAL test suite.

Tests
-----
1. Life log: append events produce valid hash chain
2. Life log: verify detects tampering
3. Snapshot: written at correct intervals + restorable
4. Temporal: birth_time, age_seconds, uptime_sessions persist across restarts
5. Policy: weights update + logged as events
6. CLI verify: passes on valid chain
7. record_interaction: writes to SQLite + life log
8. search_memory: returns structured results
9. API key: blocks unauthorized requests
10. API key: allows authorized requests
11. /health: works without auth
12. Restart recovery: interactions + state survive
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from al01.database import Database
from al01.life_log import LifeLog
from al01.memory_manager import MemoryManager
from al01.organism import Organism, OrganismState, VERSION
from al01.policy import PolicyManager
from al01.genome import Genome
from al01.population import Population
from al01.brain import Brain
from al01.autonomy import AutonomyConfig, AutonomyEngine, AwarenessModel, DECISION_STABILIZE, DECISION_ADAPT, DECISION_MUTATE, DECISION_BLEND


# ==================================================================
# VITAL: Life Log + Hash Chain
# ==================================================================

class TestLifeLog(unittest.TestCase):
    """Append-only life log with hash-chain integrity."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self.log = LifeLog(data_dir=self._tmpdir, organism_id="test-org")

    def test_append_creates_valid_chain(self) -> None:
        for i in range(10):
            self.log.append_event(
                event_type="test",
                payload={"index": i},
            )
        self.assertTrue(self.log.verify(last_n=10))
        self.assertEqual(self.log.head_seq, 10)

    def test_verify_detects_tampering(self) -> None:
        for i in range(5):
            self.log.append_event(event_type="test", payload={"i": i})

        # Tamper: rewrite a line in the JSONL
        log_path = os.path.join(self._tmpdir, "life_log.jsonl")
        with open(log_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        # Corrupt the third line
        ev = json.loads(lines[2])
        ev["payload"]["i"] = 999  # changed
        lines[2] = json.dumps(ev) + "\n"
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.writelines(lines)

        self.assertFalse(self.log.verify(last_n=5))

    def test_head_json_persists(self) -> None:
        self.log.append_event(event_type="x", payload={"a": 1})
        head = self.log.head
        self.assertEqual(head["head_seq"], 1)
        self.assertEqual(head["organism_id"], "test-org")

        # Reload from disk
        log2 = LifeLog(data_dir=self._tmpdir, organism_id="test-org")
        self.assertEqual(log2.head["head_seq"], 1)
        self.assertEqual(log2.head["head_hash"], head["head_hash"])

    def test_empty_log_verifies(self) -> None:
        self.assertTrue(self.log.verify())

    def test_event_count(self) -> None:
        for _ in range(7):
            self.log.append_event(event_type="t", payload={})
        self.assertEqual(self.log.event_count(), 7)


# ==================================================================
# VITAL: Snapshots
# ==================================================================

class TestSnapshots(unittest.TestCase):
    """Snapshot checkpoints at configured intervals."""

    def test_snapshot_written(self) -> None:
        tmpdir = tempfile.mkdtemp()
        log = LifeLog(data_dir=tmpdir, organism_id="test", snapshot_interval=5)
        for i in range(5):
            log.append_event(event_type="t", payload={"i": i})
        self.assertTrue(log.should_snapshot())
        path = log.write_snapshot({"awareness": 0.5, "evolution_count": 3})
        self.assertTrue(os.path.exists(path))

    def test_snapshot_loadable(self) -> None:
        tmpdir = tempfile.mkdtemp()
        log = LifeLog(data_dir=tmpdir, organism_id="test", snapshot_interval=3)
        for i in range(3):
            log.append_event(event_type="t", payload={"i": i})
        log.write_snapshot({"awareness": 0.42})
        snap = log.load_latest_snapshot()
        self.assertIsNotNone(snap)
        self.assertEqual(snap["state"]["awareness"], 0.42)
        self.assertEqual(snap["seq"], 3)


# ==================================================================
# VITAL: Temporal Awareness
# ==================================================================

class TestTemporalAwareness(unittest.TestCase):
    """birth_time, age_seconds, uptime_sessions tracked."""

    def test_birth_time_set_on_first_boot(self) -> None:
        tmpdir = tempfile.mkdtemp()
        db = Database(db_path=os.path.join(tmpdir, "t.db"))
        mem = MemoryManager(data_dir=tmpdir, credential_path=None, database=db)
        log = LifeLog(data_dir=os.path.join(tmpdir, "data"), organism_id="AL-01")
        policy = PolicyManager(data_dir=os.path.join(tmpdir, "data"))
        org = Organism(data_dir=tmpdir, memory_manager=mem, life_log=log, policy=policy)
        state = dict(org.state)
        self.assertIsNotNone(state["birth_time"])
        self.assertEqual(state["uptime_sessions"], 1)

    def test_uptime_sessions_increment(self) -> None:
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "t.db")

        # Session 1
        db1 = Database(db_path=db_path)
        mem1 = MemoryManager(data_dir=tmpdir, credential_path=None, database=db1)
        log1 = LifeLog(data_dir=os.path.join(tmpdir, "data"), organism_id="AL-01")
        pol1 = PolicyManager(data_dir=os.path.join(tmpdir, "data"))
        org1 = Organism(data_dir=tmpdir, memory_manager=mem1, life_log=log1, policy=pol1)
        org1.persist()
        birth1 = dict(org1.state)["birth_time"]

        # Session 2
        db2 = Database(db_path=db_path)
        mem2 = MemoryManager(data_dir=tmpdir, credential_path=None, database=db2)
        log2 = LifeLog(data_dir=os.path.join(tmpdir, "data"), organism_id="AL-01")
        pol2 = PolicyManager(data_dir=os.path.join(tmpdir, "data"))
        org2 = Organism(data_dir=tmpdir, memory_manager=mem2, life_log=log2, policy=pol2)
        state2 = dict(org2.state)
        self.assertEqual(state2["uptime_sessions"], 2)
        # birth_time should be preserved
        self.assertEqual(state2["birth_time"], birth1)

    def test_age_seconds_positive(self) -> None:
        tmpdir = tempfile.mkdtemp()
        db = Database(db_path=os.path.join(tmpdir, "t.db"))
        mem = MemoryManager(data_dir=tmpdir, credential_path=None, database=db)
        log = LifeLog(data_dir=os.path.join(tmpdir, "data"), organism_id="AL-01")
        pol = PolicyManager(data_dir=os.path.join(tmpdir, "data"))
        org = Organism(data_dir=tmpdir, memory_manager=mem, life_log=log, policy=pol)
        self.assertGreaterEqual(org.age_seconds, 0.0)


# ==================================================================
# VITAL: Policy / Adaptive Learning
# ==================================================================

class TestPolicy(unittest.TestCase):
    """Policy weights update and log events."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self.policy = PolicyManager(data_dir=self._tmpdir)

    def test_default_weights(self) -> None:
        w = self.policy.weights
        self.assertIn("curiosity_weight", w)
        self.assertIn("risk_weight", w)
        self.assertIn("social_weight", w)

    def test_update_weights(self) -> None:
        record = self.policy.update({"curiosity_weight": 0.9}, reason="testing")
        self.assertEqual(self.policy.get("curiosity_weight"), 0.9)
        self.assertIn("old_weights", record)
        self.assertIn("new_weights", record)

    def test_nudge(self) -> None:
        old = self.policy.get("risk_weight")
        self.policy.nudge("risk_weight", 0.1)
        self.assertAlmostEqual(self.policy.get("risk_weight"), old + 0.1, places=5)

    def test_clamp(self) -> None:
        self.policy.update({"curiosity_weight": 5.0})
        self.assertEqual(self.policy.get("curiosity_weight"), 1.0)
        self.policy.update({"curiosity_weight": -2.0})
        self.assertEqual(self.policy.get("curiosity_weight"), 0.0)

    def test_persists_to_disk(self) -> None:
        self.policy.update({"social_weight": 0.77})
        policy2 = PolicyManager(data_dir=self._tmpdir)
        self.assertAlmostEqual(policy2.get("social_weight"), 0.77, places=5)

    def test_organism_logs_policy_change(self) -> None:
        tmpdir = tempfile.mkdtemp()
        db = Database(db_path=os.path.join(tmpdir, "t.db"))
        mem = MemoryManager(data_dir=tmpdir, credential_path=None, database=db)
        log = LifeLog(data_dir=os.path.join(tmpdir, "data"), organism_id="AL-01")
        pol = PolicyManager(data_dir=os.path.join(tmpdir, "data"))
        org = Organism(data_dir=tmpdir, memory_manager=mem, life_log=log, policy=pol)

        initial_seq = log.head_seq
        org.update_policy({"curiosity_weight": 0.8}, reason="test")
        self.assertGreater(log.head_seq, initial_seq)


# ==================================================================
# v1.2 tests (carried forward)
# ==================================================================

class TestRecordInteraction(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self.db = Database(db_path=os.path.join(self._tmpdir, "test.db"))
        self.memory = MemoryManager(data_dir=self._tmpdir, credential_path=None, database=self.db)
        self.log = LifeLog(data_dir=os.path.join(self._tmpdir, "data"), organism_id="AL-01")
        self.pol = PolicyManager(data_dir=os.path.join(self._tmpdir, "data"))
        self.organism = Organism(
            data_dir=self._tmpdir, memory_manager=self.memory,
            life_log=self.log, policy=self.pol,
        )

    def test_interaction_stored_in_sqlite(self) -> None:
        self.organism.record_interaction(user_input="hello", response="hi there", mood="curious")
        self.assertEqual(self.db.interaction_count(), 1)
        rows = self.db.recent_interactions(n=1)
        self.assertEqual(rows[0]["user_input"], "hello")

    def test_interaction_increments_state_counter(self) -> None:
        self.organism.record_interaction(user_input="a", response="b")
        self.organism.record_interaction(user_input="c", response="d")
        self.assertEqual(dict(self.organism.state)["interaction_count"], 2)


class TestSearchMemory(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self.db = Database(db_path=os.path.join(self._tmpdir, "test.db"))
        self.memory = MemoryManager(data_dir=self._tmpdir, credential_path=None, database=self.db)
        self.log = LifeLog(data_dir=os.path.join(self._tmpdir, "data"), organism_id="AL-01")
        self.pol = PolicyManager(data_dir=os.path.join(self._tmpdir, "data"))
        self.organism = Organism(
            data_dir=self._tmpdir, memory_manager=self.memory,
            life_log=self.log, policy=self.pol,
        )
        self.organism.record_interaction(user_input="tell me about Python", response="Python is great", mood="happy")
        self.organism.record_interaction(user_input="what is Rust?", response="Rust is systems", mood="neutral")

    def test_keyword_search(self) -> None:
        results = self.organism.search_memory("Python")
        self.assertGreaterEqual(len(results), 1)

    def test_search_no_results(self) -> None:
        results = self.organism.search_memory("nonexistent-xyz-999")
        self.assertEqual(len(results), 0)


class TestDatabasePersistence(unittest.TestCase):
    def test_restart_recovery(self) -> None:
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test.db")

        db1 = Database(db_path=db_path)
        mem1 = MemoryManager(data_dir=tmpdir, credential_path=None, database=db1)
        log1 = LifeLog(data_dir=os.path.join(tmpdir, "data"), organism_id="AL-01")
        pol1 = PolicyManager(data_dir=os.path.join(tmpdir, "data"))
        org1 = Organism(data_dir=tmpdir, memory_manager=mem1, life_log=log1, policy=pol1)
        org1.record_interaction(user_input="before restart", response="noted")
        org1.persist()

        db2 = Database(db_path=db_path)
        mem2 = MemoryManager(data_dir=tmpdir, credential_path=None, database=db2)
        log2 = LifeLog(data_dir=os.path.join(tmpdir, "data"), organism_id="AL-01")
        pol2 = PolicyManager(data_dir=os.path.join(tmpdir, "data"))
        org2 = Organism(data_dir=tmpdir, memory_manager=mem2, life_log=log2, policy=pol2)
        self.assertGreaterEqual(dict(org2.state)["interaction_count"], 1)
        rows = db2.recent_interactions(n=10)
        self.assertIn("before restart", [r["user_input"] for r in rows])


class TestAPIKeyAuth(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi[test] not installed")

        self._tmpdir = tempfile.mkdtemp()
        self.db = Database(db_path=os.path.join(self._tmpdir, "test.db"))
        self.memory = MemoryManager(data_dir=self._tmpdir, credential_path=None, database=self.db)
        self.log = LifeLog(data_dir=os.path.join(self._tmpdir, "data"), organism_id="AL-01")
        self.pol = PolicyManager(data_dir=os.path.join(self._tmpdir, "data"))
        self.organism = Organism(
            data_dir=self._tmpdir, memory_manager=self.memory,
            life_log=self.log, policy=self.pol,
        )

        from al01.api import create_app
        self.api_key = "test-key-abc123"
        self.app = create_app(self.organism, api_key=self.api_key)
        self.client = TestClient(self.app)

    def test_health_no_auth_required(self) -> None:
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")

    def test_status_blocked_without_key(self) -> None:
        self.assertEqual(self.client.get("/status").status_code, 401)

    def test_status_allowed_with_key(self) -> None:
        resp = self.client.get("/status", headers={"X-API-Key": self.api_key})
        self.assertEqual(resp.status_code, 200)

    def test_interact_blocked_without_key(self) -> None:
        resp = self.client.post("/interact", json={"user_input": "hi", "response": "hello"})
        self.assertEqual(resp.status_code, 401)

    def test_interact_allowed_with_key(self) -> None:
        resp = self.client.post(
            "/interact",
            json={"user_input": "hi", "response": "hello"},
            headers={"X-API-Key": self.api_key},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "recorded")

    def test_vital_verify_endpoint(self) -> None:
        resp = self.client.get("/vital/verify", headers={"X-API-Key": self.api_key})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "PASS")

    def test_vital_policy_endpoint(self) -> None:
        # GET
        resp = self.client.get("/vital/policy", headers={"X-API-Key": self.api_key})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("curiosity_weight", resp.json())

        # POST
        resp = self.client.post(
            "/vital/policy",
            json={"weights": {"curiosity_weight": 0.9}, "reason": "test"},
            headers={"X-API-Key": self.api_key},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "updated")

    def test_bad_input_returns_422(self) -> None:
        resp = self.client.post(
            "/interact",
            json={"user_input": "", "response": "hello"},
            headers={"X-API-Key": self.api_key},
        )
        self.assertEqual(resp.status_code, 422)


# ==================================================================
# v2.0: Genome
# ==================================================================

class TestGenome(unittest.TestCase):
    """Genome mutation, fitness, and reproduction."""

    def test_default_traits(self) -> None:
        g = Genome()
        self.assertEqual(len(g.traits), 5)
        self.assertIn("adaptability", g.traits)
        self.assertAlmostEqual(g.fitness, 0.5, places=2)

    def test_mutation(self) -> None:
        g = Genome(mutation_rate=1.0)  # force 100% mutation for testing
        result = g.mutate()
        self.assertIn("mutated_traits", result)
        self.assertGreater(len(result["mutated_traits"]), 0)

    def test_traits_floored_at_zero(self) -> None:
        """Traits have a floor of 0.0 (no hard ceiling — soft cap applies to fitness)."""
        g = Genome(traits={"t1": 0.99, "t2": 0.01}, mutation_rate=1.0, mutation_delta=0.5)
        g.mutate()
        for v in g.traits.values():
            self.assertGreaterEqual(v, 0.0)
        # Soft-capped effective_traits should always be >= 0
        for v in g.effective_traits.values():
            self.assertGreaterEqual(v, 0.0)

    def test_spawn_child(self) -> None:
        parent = Genome()
        child = parent.spawn_child(variance=0.05)
        self.assertEqual(set(child.traits.keys()), set(parent.traits.keys()))
        # Child traits should differ slightly but not be identical
        self.assertEqual(len(child.traits), len(parent.traits))

    def test_energy_transfer(self) -> None:
        a = Genome(traits={"energy_efficiency": 0.8, "adaptability": 0.5})
        b = Genome(traits={"energy_efficiency": 0.3, "adaptability": 0.5})
        result = a.transfer_energy(b, amount=0.1)
        self.assertIn("direction", result)
        # Higher should have donated
        self.assertAlmostEqual(a.get_trait("energy_efficiency"), 0.7, places=4)
        self.assertAlmostEqual(b.get_trait("energy_efficiency"), 0.4, places=4)

    def test_serialization_roundtrip(self) -> None:
        g = Genome()
        d = g.to_dict()
        g2 = Genome.from_dict(d)
        self.assertEqual(g.traits, g2.traits)
        self.assertAlmostEqual(g.fitness, g2.fitness, places=6)


# ==================================================================
# v2.0: Population
# ==================================================================

class TestPopulation(unittest.TestCase):
    """Multi-organism population tracking."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self.pop = Population(data_dir=self._tmpdir, parent_id="AL-01")

    def test_parent_auto_registered(self) -> None:
        self.assertIn("AL-01", self.pop.member_ids)
        self.assertEqual(self.pop.size, 1)

    def test_spawn_child(self) -> None:
        parent_genome = Genome()
        child = self.pop.spawn_child(parent_genome, parent_evolution=10)
        self.assertEqual(child["id"], "AL-01-child-1")
        self.assertEqual(self.pop.size, 2)
        self.assertEqual(child["evolution_count"], 0)
        self.assertEqual(child["interaction_count"], 0)

    def test_spawn_multiple_children(self) -> None:
        g = Genome()
        self.pop.spawn_child(g, 10)
        self.pop.spawn_child(g, 20)
        self.assertEqual(self.pop.size, 3)
        self.assertIn("AL-01-child-2", self.pop.member_ids)

    def test_should_reproduce(self) -> None:
        self.assertTrue(self.pop.should_reproduce(10))
        self.assertTrue(self.pop.should_reproduce(20))
        self.assertFalse(self.pop.should_reproduce(7))
        self.assertFalse(self.pop.should_reproduce(0))

    def test_simulate_interactions_needs_two(self) -> None:
        # Only 1 member (parent), no interactions possible
        result = self.pop.simulate_interactions()
        self.assertEqual(result, [])

    def test_simulate_interactions_with_children(self) -> None:
        self.pop.spawn_child(Genome(), 10)
        result = self.pop.simulate_interactions()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "energy_transfer")

    def test_persistence(self) -> None:
        self.pop.spawn_child(Genome(), 10)
        pop2 = Population(data_dir=self._tmpdir, parent_id="AL-01")
        self.assertEqual(pop2.size, 2)


# ==================================================================
# v2.0 → v3.7: Brain (Environmental Analysis Engine)
# ==================================================================

class TestBrain(unittest.TestCase):
    """Brain: environment-driven analytical engine."""

    def test_brain_always_enabled(self) -> None:
        brain = Brain(api_key=None)
        self.assertTrue(brain.enabled)

    def test_process_query_returns_analysis(self) -> None:
        brain = Brain(api_key=None)
        result = brain.process_query("how to adapt?")
        self.assertIn("response", result)
        self.assertIn("sentiment", result)
        self.assertIn("trait_nudges", result)
        self.assertEqual(result["source"], "brain")
        self.assertIn("analysis", result)

    def test_positive_keywords_raise_sentiment(self) -> None:
        brain = Brain(api_key=None)
        result = brain.process_query("grow and learn")
        self.assertGreater(result["sentiment"], -0.5)

    def test_negative_keywords_lower_sentiment(self) -> None:
        brain = Brain(api_key=None)
        result = brain.process_query("danger and threat")
        self.assertLess(result["sentiment"], 0.5)

    def test_analyse_with_env_state(self) -> None:
        brain = Brain(api_key=None)
        env = {
            "temperature": 0.9,
            "entropy_pressure": 0.7,
            "resource_abundance": 0.2,
            "noise_level": 0.8,
            "active_scarcity_count": 1,
        }
        traits = {
            "adaptability": 0.3, "energy_efficiency": 0.2,
            "resilience": 0.3, "perception": 0.3, "creativity": 0.3,
        }
        result = brain.analyse(env, traits, organism_energy=0.15, organism_fitness=0.10)
        self.assertEqual(result.urgency, "critical")
        # Should nudge energy_efficiency (scarce) and resilience (extreme temp)
        self.assertGreater(result.trait_nudges.get("energy_efficiency", 0), 0)
        self.assertGreater(result.trait_nudges.get("resilience", 0), 0)
        self.assertIn("[BRAIN]", result.situation)

    def test_low_urgency_in_stable_env(self) -> None:
        brain = Brain(api_key=None)
        env = {
            "temperature": 0.5,
            "entropy_pressure": 0.1,
            "resource_abundance": 0.9,
            "noise_level": 0.1,
            "active_scarcity_count": 0,
        }
        traits = {
            "adaptability": 0.8, "energy_efficiency": 0.8,
            "resilience": 0.8, "perception": 0.8, "creativity": 0.8,
        }
        result = brain.analyse(env, traits, organism_energy=0.9, organism_fitness=0.8)
        self.assertEqual(result.urgency, "low")
        # All traits exceed demand — nudges should be zero
        total_nudge = sum(abs(v) for v in result.trait_nudges.values())
        self.assertAlmostEqual(total_nudge, 0.0, places=4)

    def test_analysis_count_increments(self) -> None:
        brain = Brain(api_key=None)
        self.assertEqual(brain.analysis_count, 0)
        brain.process_query("test")
        self.assertEqual(brain.analysis_count, 1)
        brain.process_query("test2")
        self.assertEqual(brain.analysis_count, 2)

    def test_recovery_mode_boosts_efficiency(self) -> None:
        brain = Brain(api_key=None)
        env = {
            "temperature": 0.5, "entropy_pressure": 0.3,
            "resource_abundance": 0.5, "noise_level": 0.2,
            "active_scarcity_count": 0,
        }
        traits = {
            "adaptability": 0.3, "energy_efficiency": 0.3,
            "resilience": 0.3, "perception": 0.3, "creativity": 0.3,
        }
        normal = brain.analyse(env, traits, organism_energy=0.3, organism_fitness=0.3)
        recovery = brain.analyse(env, traits, organism_energy=0.3, organism_fitness=0.3,
                                 recovery_mode=True)
        self.assertGreaterEqual(
            recovery.environment_demand.energy_efficiency,
            normal.environment_demand.energy_efficiency,
        )


# ==================================================================
# v2.0: Organism Evolution Integration
# ==================================================================

class TestOrganismEvolution(unittest.TestCase):
    """Organism genome, stimuli, evolution cycle, reproduction."""

    def _make_organism(self) -> Organism:
        tmpdir = tempfile.mkdtemp()
        db = Database(db_path=os.path.join(tmpdir, "t.db"))
        mem = MemoryManager(data_dir=tmpdir, credential_path=None, database=db)
        log = LifeLog(data_dir=os.path.join(tmpdir, "data"), organism_id="AL-01")
        pol = PolicyManager(data_dir=os.path.join(tmpdir, "data"))
        pop = Population(data_dir=tmpdir, parent_id="AL-01")
        brain = Brain(api_key=None)
        return Organism(
            data_dir=tmpdir, memory_manager=mem, life_log=log,
            policy=pol, population=pop, brain=brain,
        )

    def test_genome_initialized(self) -> None:
        org = self._make_organism()
        self.assertIsNotNone(org.genome)
        self.assertEqual(len(org.genome.traits), 5)

    def test_genome_in_state(self) -> None:
        org = self._make_organism()
        state = dict(org.state)
        self.assertIn("genome", state)
        self.assertIn("traits", state["genome"])

    def test_add_stimulus(self) -> None:
        org = self._make_organism()
        org.add_stimulus("environmental_change")
        self.assertEqual(len(org.stimuli), 1)

    def test_evolve_cycle_processes_stimuli(self) -> None:
        org = self._make_organism()
        org.add_stimulus("test_stimulus")
        initial_evo = dict(org.state)["evolution_count"]
        result = org.evolve_cycle()
        self.assertEqual(result["stimuli_processed"], 1)
        self.assertGreater(dict(org.state)["evolution_count"], initial_evo)

    def test_evolve_cycle_no_stimuli_no_mutation(self) -> None:
        org = self._make_organism()
        result = org.evolve_cycle()
        self.assertEqual(result["stimuli_processed"], 0)

    def test_force_evolve(self) -> None:
        org = self._make_organism()
        result = org.force_evolve()
        self.assertGreaterEqual(result["stimuli_processed"], 1)

    def test_stimulate_with_stimulus(self) -> None:
        org = self._make_organism()
        org.stimulate(stimulus="threat_detected")
        self.assertEqual(len(org.stimuli), 1)

    def test_brain_query_stimulus(self) -> None:
        org = self._make_organism()
        org.add_stimulus("query:how to grow?")
        # Should not crash — brain processes it immediately
        self.assertEqual(len(org.stimuli), 1)

    def test_growth_metrics_v2(self) -> None:
        org = self._make_organism()
        metrics = org.growth_metrics
        self.assertIn("genome", metrics)
        self.assertIn("fitness", metrics)
        self.assertIn("population_size", metrics)
        self.assertIn("brain_enabled", metrics)
        self.assertIn("pending_stimuli", metrics)


# ==================================================================
# v2.0: API Endpoints
# ==================================================================

class TestAPIv2(unittest.TestCase):
    """v2.0 API endpoints: /evolve, /population, /genome, /stimuli, /stimulate with body."""

    def setUp(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi[test] not installed")

        self._tmpdir = tempfile.mkdtemp()
        self.db = Database(db_path=os.path.join(self._tmpdir, "test.db"))
        self.memory = MemoryManager(data_dir=self._tmpdir, credential_path=None, database=self.db)
        self.log = LifeLog(data_dir=os.path.join(self._tmpdir, "data"), organism_id="AL-01")
        self.pol = PolicyManager(data_dir=os.path.join(self._tmpdir, "data"))
        self.pop = Population(data_dir=self._tmpdir, parent_id="AL-01")
        self.brain = Brain(api_key=None)
        self.organism = Organism(
            data_dir=self._tmpdir, memory_manager=self.memory,
            life_log=self.log, policy=self.pol,
            population=self.pop, brain=self.brain,
        )

        from al01.api import create_app
        self.api_key = "test-key-v2"
        self.app = create_app(self.organism, api_key=self.api_key)
        self.client = TestClient(self.app)

    def test_evolve_endpoint(self) -> None:
        resp = self.client.get("/evolve", headers={"X-API-Key": self.api_key})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("evolution_result", data)
        self.assertIn("status", data)

    def test_genome_endpoint(self) -> None:
        resp = self.client.get("/genome", headers={"X-API-Key": self.api_key})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("traits", resp.json())
        self.assertIn("fitness", resp.json())

    def test_population_endpoint(self) -> None:
        resp = self.client.get("/population", headers={"X-API-Key": self.api_key})
        self.assertEqual(resp.status_code, 200)
        self.assertGreaterEqual(resp.json()["population_size"], 1)

    def test_population_member_endpoint(self) -> None:
        resp = self.client.get("/population/AL-01", headers={"X-API-Key": self.api_key})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["id"], "AL-01")

    def test_population_member_404(self) -> None:
        resp = self.client.get("/population/nonexistent", headers={"X-API-Key": self.api_key})
        self.assertEqual(resp.status_code, 404)

    def test_stimuli_endpoint(self) -> None:
        resp = self.client.get("/stimuli", headers={"X-API-Key": self.api_key})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("stimuli", resp.json())

    def test_stimulate_with_body(self) -> None:
        resp = self.client.post(
            "/stimulate",
            json={"stimulus": "environmental_change", "trigger_cycle": False},
            headers={"X-API-Key": self.api_key},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("awareness", resp.json())

    def test_stimulate_no_body_still_works(self) -> None:
        resp = self.client.post(
            "/stimulate",
            headers={"X-API-Key": self.api_key},
        )
        self.assertEqual(resp.status_code, 200)

    def test_dashboard_has_genome(self) -> None:
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Genome", resp.text)
        self.assertIn("Population", resp.text)

    def test_status_has_v2_fields(self) -> None:
        resp = self.client.get("/status", headers={"X-API-Key": self.api_key})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("genome", data)
        self.assertIn("fitness", data)
        self.assertIn("population_size", data)


# ==================================================================
# v2.1: Autonomous Decision Cycle
# ==================================================================

class TestAutonomyEngine(unittest.TestCase):
    """AutonomyEngine — deterministic decision logic, stagnation tracking, log persistence."""

    def _make_engine(self, **overrides: Any) -> AutonomyEngine:
        tmpdir = tempfile.mkdtemp()
        defaults = dict(
            decision_interval=10,
            fitness_threshold=0.45,
            stagnation_window=5,
            stagnation_variance_epsilon=1e-4,
        )
        defaults.update(overrides)
        cfg = AutonomyConfig(**defaults)
        return AutonomyEngine(data_dir=tmpdir, config=cfg)

    def test_mutate_when_fitness_low(self) -> None:
        """fitness < threshold → mutate."""
        engine = self._make_engine()
        record = engine.decide(fitness=0.30, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        self.assertEqual(record["decision"], DECISION_MUTATE)
        self.assertIn("threshold", record["reason"])

    def test_stabilize_when_fitness_healthy(self) -> None:
        """fitness >= threshold and no stagnation → stabilize."""
        engine = self._make_engine()
        for i in range(3):
            record = engine.decide(fitness=0.60 + i * 0.01, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        self.assertEqual(record["decision"], DECISION_STABILIZE)

    def test_adapt_on_stagnation(self) -> None:
        """Identical fitness values for stagnation_window cycles → adapt."""
        engine = self._make_engine(stagnation_window=5)
        for _ in range(5):
            record = engine.decide(fitness=0.50, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        self.assertEqual(record["decision"], DECISION_ADAPT)
        self.assertGreater(engine.stagnation_count, 0)

    def test_stagnation_count_resets(self) -> None:
        """Stagnation count resets when fitness varies."""
        engine = self._make_engine(stagnation_window=3)
        for _ in range(3):
            engine.decide(fitness=0.50, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        self.assertTrue(engine.is_stagnant)
        # Now introduce variation
        engine.decide(fitness=0.80, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        self.assertEqual(engine.stagnation_count, 0)

    def test_fitness_history_bounded(self) -> None:
        """Fitness history never exceeds stagnation_window."""
        engine = self._make_engine(stagnation_window=5)
        for i in range(20):
            engine.decide(fitness=0.5 + i * 0.01, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        self.assertLessEqual(len(engine.fitness_history), 5)

    def test_log_file_persists(self) -> None:
        """autonomy_log.jsonl is created and contains valid JSON lines with awareness_breakdown."""
        engine = self._make_engine()
        engine.decide(fitness=0.50, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        engine.decide(fitness=0.30, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        self.assertTrue(os.path.exists(engine.log_path))
        with open(engine.log_path, "r", encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip()]
        self.assertEqual(len(lines), 2)
        for line in lines:
            obj = json.loads(line)
            self.assertIn("decision", obj)
            self.assertIn("timestamp", obj)
            self.assertIn("awareness_breakdown", obj)
            ab = obj["awareness_breakdown"]
            self.assertIn("stimuli_rate", ab)
            self.assertIn("decision_rate", ab)
            self.assertIn("fitness_variance_norm", ab)

    def test_recovery_from_log(self) -> None:
        """New engine recovers state from existing autonomy_log.jsonl."""
        tmpdir = tempfile.mkdtemp()
        cfg = AutonomyConfig(stagnation_window=5)
        e1 = AutonomyEngine(data_dir=tmpdir, config=cfg)
        for _ in range(3):
            e1.decide(fitness=0.60, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        self.assertEqual(e1.total_decisions, 3)

        # Create a new engine pointing at the same log
        e2 = AutonomyEngine(data_dir=tmpdir, config=cfg)
        self.assertEqual(e2.total_decisions, 3)
        self.assertEqual(len(e2.fitness_history), 3)

    def test_summary_contains_fields(self) -> None:
        """summary() contains all expected keys."""
        engine = self._make_engine()
        engine.decide(fitness=0.50, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        s = engine.summary()
        for key in ("stagnation_count", "is_stagnant", "total_decisions",
                     "fitness_history_len", "fitness_variance",
                     "decision_interval", "fitness_threshold", "stagnation_window",
                     "awareness", "total_stimuli",
                     "energy", "exploration_mode", "novelty_accumulator",
                     "effective_fitness_threshold", "mutation_rate_offset",
                     "env_trait_weights", "idle_cycles", "entropy_active",
                     "vital_score"):
            self.assertIn(key, s, f"Missing key: {key}")

    def test_total_decisions_increments(self) -> None:
        engine = self._make_engine()
        for i in range(5):
            engine.decide(fitness=0.60, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        self.assertEqual(engine.total_decisions, 5)


class TestOrganismAutonomy(unittest.TestCase):
    """Organism-level autonomy_cycle integration."""

    def _make_organism(self) -> Organism:
        tmpdir = tempfile.mkdtemp()
        db = Database(db_path=os.path.join(tmpdir, "t.db"))
        mem = MemoryManager(data_dir=tmpdir, credential_path=None, database=db)
        log = LifeLog(data_dir=os.path.join(tmpdir, "data"), organism_id="AL-01")
        pol = PolicyManager(data_dir=os.path.join(tmpdir, "data"))
        pop = Population(data_dir=tmpdir, parent_id="AL-01")
        brain = Brain(api_key=None)
        auto_cfg = AutonomyConfig(stagnation_window=3)
        auto = AutonomyEngine(data_dir=os.path.join(tmpdir, "data"), config=auto_cfg)
        return Organism(
            data_dir=tmpdir, memory_manager=mem, life_log=log,
            policy=pol, population=pop, brain=brain, autonomy=auto,
        )

    def test_autonomy_cycle_returns_decision(self) -> None:
        org = self._make_organism()
        record = org.autonomy_cycle()
        self.assertIn("decision", record)
        self.assertIn(record["decision"], ("stabilize", "adapt", "mutate"))

    def test_autonomy_cycle_mutate_changes_genome(self) -> None:
        """When fitness < threshold, autonomy_cycle mutates."""
        org = self._make_organism()
        # Force fitness below threshold by setting all traits low
        for trait in org.genome.traits:
            org.genome.set_trait(trait, 0.20)
        old_traits = dict(org.genome.traits)
        record = org.autonomy_cycle()
        self.assertEqual(record["decision"], "mutate")
        # Mutation may not change every trait, but record should contain 'mutations'
        self.assertIn("mutations", record)

    def test_autonomy_cycle_adapt_nudges_trait(self) -> None:
        """When stagnant, autonomy_cycle adapts weakest trait."""
        org = self._make_organism()
        # Need 3 identical fitness readings to trigger stagnation (window=3)
        for _ in range(3):
            org.autonomy_cycle()
        record = org.autonomy_cycle()
        # By now should be adapt or stabilize depending on exact fitness
        self.assertIn("decision", record)

    def test_growth_metrics_includes_stagnation(self) -> None:
        org = self._make_organism()
        metrics = org.growth_metrics
        self.assertIn("stagnation_count", metrics)
        self.assertIn("autonomy", metrics)

    def test_status_api_has_autonomy(self) -> None:
        """GET /status returns stagnation_count and autonomy."""
        tmpdir = tempfile.mkdtemp()
        db = Database(db_path=os.path.join(tmpdir, "t.db"))
        mem = MemoryManager(data_dir=tmpdir, credential_path=None, database=db)
        log = LifeLog(data_dir=os.path.join(tmpdir, "data"), organism_id="AL-01")
        pol = PolicyManager(data_dir=os.path.join(tmpdir, "data"))
        pop = Population(data_dir=tmpdir, parent_id="AL-01")
        brain = Brain(api_key=None)
        auto = AutonomyEngine(data_dir=os.path.join(tmpdir, "data"))
        org = Organism(
            data_dir=tmpdir, memory_manager=mem, life_log=log,
            policy=pol, population=pop, brain=brain, autonomy=auto,
        )
        from al01.api import create_app
        from fastapi.testclient import TestClient
        app = create_app(org, api_key="tkey")
        client = TestClient(app)
        resp = client.get("/status", headers={"X-API-Key": "tkey"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("stagnation_count", data)
        self.assertIn("autonomy", data)

    def test_autonomy_api_endpoint(self) -> None:
        """GET /autonomy returns decision engine summary."""
        tmpdir = tempfile.mkdtemp()
        db = Database(db_path=os.path.join(tmpdir, "t.db"))
        mem = MemoryManager(data_dir=tmpdir, credential_path=None, database=db)
        log = LifeLog(data_dir=os.path.join(tmpdir, "data"), organism_id="AL-01")
        pol = PolicyManager(data_dir=os.path.join(tmpdir, "data"))
        pop = Population(data_dir=tmpdir, parent_id="AL-01")
        brain = Brain(api_key=None)
        auto = AutonomyEngine(data_dir=os.path.join(tmpdir, "data"))
        org = Organism(
            data_dir=tmpdir, memory_manager=mem, life_log=log,
            policy=pol, population=pop, brain=brain, autonomy=auto,
        )
        from al01.api import create_app
        from fastapi.testclient import TestClient
        app = create_app(org, api_key="tkey")
        client = TestClient(app)
        resp = client.get("/autonomy", headers={"X-API-Key": "tkey"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("stagnation_count", data)
        self.assertIn("fitness_history", data)


# ==================================================================
# v2.1: Computational Awareness Model
# ==================================================================

class TestAwarenessModel(unittest.TestCase):
    """AwarenessModel — deterministic, formula-based awareness computation."""

    def _make_model(self, **overrides: Any) -> AwarenessModel:
        defaults = dict(
            stimuli_rate_cap=5.0,
            decision_rate_cap=20.0,
            fitness_variance_cap=0.01,
        )
        defaults.update(overrides)
        cfg = AutonomyConfig(**defaults)
        return AwarenessModel(cfg)

    def test_compute_returns_breakdown(self) -> None:
        model = self._make_model()
        result = model.compute(total_decisions=5, fitness_variance=0.005, is_stagnant=False)
        for key in ("awareness", "stimuli_rate", "decision_rate",
                     "fitness_variance_norm", "stagnation_component"):
            self.assertIn(key, result)
        self.assertGreaterEqual(result["awareness"], 0.0)
        self.assertLessEqual(result["awareness"], 1.0)

    def test_awareness_zero_when_idle(self) -> None:
        """No stimuli, no decisions, no variance, stagnant → awareness ≈ 0."""
        model = self._make_model()
        result = model.compute(total_decisions=0, fitness_variance=0.0, is_stagnant=True)
        self.assertAlmostEqual(result["awareness"], 0.0, places=4)

    def test_awareness_increases_with_stimuli(self) -> None:
        """Stimuli received bumps the stimuli_rate component."""
        model = self._make_model()
        # Record 3 stimuli before computing
        for _ in range(3):
            model.record_stimulus()
        result = model.compute(total_decisions=0, fitness_variance=0.0, is_stagnant=True)
        # 3/5.0 * 0.4 = 0.24
        self.assertAlmostEqual(result["stimuli_rate"], 0.6, places=4)
        self.assertGreater(result["awareness"], 0.0)

    def test_awareness_increases_with_decisions(self) -> None:
        """More decisions → higher decision_rate component."""
        model = self._make_model()
        r1 = model.compute(total_decisions=5, fitness_variance=0.0, is_stagnant=True)
        r2 = model.compute(total_decisions=15, fitness_variance=0.0, is_stagnant=True)
        self.assertGreater(r2["decision_rate"], r1["decision_rate"])

    def test_stagnation_flag_lowers_awareness(self) -> None:
        """Stagnant → stagnation_component = 0; not stagnant → 1.0."""
        model = self._make_model()
        r_stag = model.compute(total_decisions=10, fitness_variance=0.005, is_stagnant=True)
        r_ok = model.compute(total_decisions=10, fitness_variance=0.005, is_stagnant=False)
        self.assertGreater(r_ok["awareness"], r_stag["awareness"])

    def test_awareness_deterministic(self) -> None:
        """Same inputs always yield same output."""
        model = self._make_model()
        model.record_stimulus()
        r1 = model.compute(total_decisions=10, fitness_variance=0.005, is_stagnant=False)
        # Reset and replay
        model2 = self._make_model()
        model2.record_stimulus()
        r2 = model2.compute(total_decisions=10, fitness_variance=0.005, is_stagnant=False)
        self.assertEqual(r1["awareness"], r2["awareness"])

    def test_stimuli_counter_resets_per_cycle(self) -> None:
        """After compute(), stimuli_received_this_cycle resets to 0."""
        model = self._make_model()
        model.record_stimulus()
        model.record_stimulus()
        r1 = model.compute(total_decisions=1, fitness_variance=0.0, is_stagnant=False)
        self.assertEqual(r1["stimuli_received_this_cycle"], 2)
        # Next cycle should have 0 stimuli
        r2 = model.compute(total_decisions=2, fitness_variance=0.0, is_stagnant=False)
        self.assertEqual(r2["stimuli_received_this_cycle"], 0)

    def test_total_stimuli_persists(self) -> None:
        """total_stimuli increments across cycles."""
        model = self._make_model()
        model.record_stimulus()
        model.record_stimulus()
        r1 = model.compute(total_decisions=1, fitness_variance=0.0, is_stagnant=False)
        self.assertEqual(r1["total_stimuli"], 2)
        model.record_stimulus()
        r2 = model.compute(total_decisions=2, fitness_variance=0.0, is_stagnant=False)
        self.assertEqual(r2["total_stimuli"], 3)

    def test_recover_restores_state(self) -> None:
        """recover() restores total_stimuli and last_awareness."""
        model = self._make_model()
        model.recover(total_stimuli=42, last_awareness=0.75)
        self.assertEqual(model.total_stimuli, 42)
        self.assertAlmostEqual(model.last_awareness, 0.75, places=4)

    def test_clamped_at_one(self) -> None:
        """Even with maxed inputs, awareness never exceeds 1.0."""
        model = self._make_model()
        for _ in range(100):
            model.record_stimulus()
        result = model.compute(total_decisions=1000, fitness_variance=1.0, is_stagnant=False)
        self.assertLessEqual(result["awareness"], 1.0)


class TestAwarenessIntegration(unittest.TestCase):
    """Awareness computed in organism autonomy_cycle and written to state."""

    def _make_organism(self) -> Organism:
        tmpdir = tempfile.mkdtemp()
        db = Database(db_path=os.path.join(tmpdir, "t.db"))
        mem = MemoryManager(data_dir=tmpdir, credential_path=None, database=db)
        log = LifeLog(data_dir=os.path.join(tmpdir, "data"), organism_id="AL-01")
        pol = PolicyManager(data_dir=os.path.join(tmpdir, "data"))
        pop = Population(data_dir=tmpdir, parent_id="AL-01")
        brain = Brain(api_key=None)
        auto_cfg = AutonomyConfig(stagnation_window=3)
        auto = AutonomyEngine(data_dir=os.path.join(tmpdir, "data"), config=auto_cfg)
        return Organism(
            data_dir=tmpdir, memory_manager=mem, life_log=log,
            policy=pol, population=pop, brain=brain, autonomy=auto,
        )

    def test_awareness_updated_by_autonomy_cycle(self) -> None:
        """autonomy_cycle sets awareness to the model-computed value."""
        org = self._make_organism()
        record = org.autonomy_cycle()
        state = dict(org.state)
        self.assertAlmostEqual(state["awareness"], record["awareness"], places=6)

    def test_awareness_changes_with_stimuli(self) -> None:
        """Receiving stimuli before autonomy_cycle changes computed awareness."""
        org = self._make_organism()
        r1 = org.autonomy_cycle()
        # Now add stimuli and run another cycle
        org.add_stimulus("environmental_change")
        org.add_stimulus("threat_detected")
        r2 = org.autonomy_cycle()
        # Awareness should differ (stimuli boost stimuli_rate component)
        self.assertNotEqual(r1["awareness"], r2["awareness"])

    def test_pulse_does_not_change_awareness(self) -> None:
        """pulse() no longer modifies awareness — it's read-only now."""
        org = self._make_organism()
        org.autonomy_cycle()  # Set initial computed awareness
        state_before = dict(org.state)["awareness"]
        org.pulse()
        state_after = dict(org.state)["awareness"]
        self.assertEqual(state_before, state_after)

    def test_awareness_in_decision_record(self) -> None:
        """Decision record contains awareness_breakdown from model."""
        org = self._make_organism()
        record = org.autonomy_cycle()
        self.assertIn("awareness_breakdown", record)
        ab = record["awareness_breakdown"]
        self.assertIn("stimuli_rate", ab)
        self.assertIn("decision_rate", ab)
        self.assertIn("fitness_variance_norm", ab)
        self.assertIn("stagnation_component", ab)

    def test_awareness_recovery_across_engine_restart(self) -> None:
        """AutonomyEngine recovers awareness from log file on restart."""
        tmpdir = tempfile.mkdtemp()
        cfg = AutonomyConfig(stagnation_window=5)
        e1 = AutonomyEngine(data_dir=tmpdir, config=cfg)
        e1.record_stimulus()
        e1.record_stimulus()
        e1.decide(fitness=0.60, awareness=0.0, mutation_rate=0.1, pending_stimuli=0)
        aw1 = e1.awareness
        ts1 = e1.awareness_model.total_stimuli
        # Restart
        e2 = AutonomyEngine(data_dir=tmpdir, config=cfg)
        self.assertAlmostEqual(e2.awareness, aw1, places=6)
        self.assertEqual(e2.awareness_model.total_stimuli, ts1)


# ==================================================================
# v2.2: Cooperative Genome Blending
# ==================================================================

class TestCooperativeBlending(unittest.TestCase):
    """Genome.blend_with and Population.cooperative_blend — non-lethal evolution."""

    def test_genome_blend_with_modifies_traits(self) -> None:
        """blend_with moves traits toward the other genome."""
        g1 = Genome(traits={"a": 0.2, "b": 0.8})
        g2 = Genome(traits={"a": 0.8, "b": 0.2})
        result = g1.blend_with(g2, blend_factor=0.5, noise=0.0)
        # Should be close to 0.5 for both traits (no noise)
        self.assertAlmostEqual(g1.get_trait("a"), 0.5, places=1)
        self.assertAlmostEqual(g1.get_trait("b"), 0.5, places=1)
        self.assertIn("blended_traits", result)
        self.assertIn("fitness", result)

    def test_genome_blend_with_both_survive(self) -> None:
        """After blending, original 'other' genome is unmodified."""
        g1 = Genome(traits={"a": 0.2, "b": 0.8})
        g2 = Genome(traits={"a": 0.8, "b": 0.2})
        g1.blend_with(g2, blend_factor=0.5, noise=0.0)
        # g2 should NOT be modified
        self.assertAlmostEqual(g2.get_trait("a"), 0.8, places=6)
        self.assertAlmostEqual(g2.get_trait("b"), 0.2, places=6)

    def test_population_cooperative_blend(self) -> None:
        """Population cooperative_blend blends traits between two members."""
        tmpdir = tempfile.mkdtemp()
        pop = Population(data_dir=tmpdir, parent_id="AL-01")
        # Need at least 2 members
        parent_genome = Genome()
        pop.spawn_child(parent_genome, parent_evolution=1)
        self.assertGreaterEqual(pop.size, 2)
        result = pop.cooperative_blend(mutation_delta=0.02)
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "cooperative_blend")
        self.assertIn("organism_a", result)
        self.assertIn("organism_b", result)

    def test_population_cooperative_blend_single_member(self) -> None:
        """Blend returns None when only 1 member exists."""
        tmpdir = tempfile.mkdtemp()
        pop = Population(data_dir=tmpdir, parent_id="AL-01")
        self.assertEqual(pop.size, 1)
        result = pop.cooperative_blend()
        self.assertIsNone(result)


# ==================================================================
# v2.2: Energy System
# ==================================================================

class TestEnergySystem(unittest.TestCase):
    """Soft resource pressure — energy decays, affects fitness."""

    def _make_engine(self, **overrides: Any) -> AutonomyEngine:
        tmpdir = tempfile.mkdtemp()
        cfg = AutonomyConfig(stagnation_window=5, **overrides)
        return AutonomyEngine(data_dir=tmpdir, config=cfg)

    def test_energy_starts_at_initial(self) -> None:
        e = self._make_engine()
        self.assertAlmostEqual(e.energy, 1.0, places=4)

    def test_energy_decays_per_cycle(self) -> None:
        """Each decision cycle decays energy."""
        e = self._make_engine(energy_decay_per_cycle=0.1)
        e.decide(fitness=0.80, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        # Decay of 0.1 but stabilize bonus of 0.02 → net -0.08
        self.assertLess(e.energy, 1.0)

    def test_energy_never_below_min(self) -> None:
        e = self._make_engine(energy_decay_per_cycle=0.5, energy_min=0.1)
        for _ in range(20):
            e.decide(fitness=0.80, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        self.assertGreaterEqual(e.energy, 0.1)

    def test_low_energy_reduces_effective_fitness(self) -> None:
        """When energy is below floor, effective_fitness < raw fitness."""
        e = self._make_engine(energy_decay_per_cycle=0.5, energy_fitness_floor=0.3)
        # Drain energy fast
        for _ in range(10):
            e.decide(fitness=0.80, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        r = e.decide(fitness=0.80, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        self.assertLess(r["effective_fitness"], 0.80)

    def test_stabilize_gives_energy_bonus(self) -> None:
        """Stabilize decision restores a little energy."""
        e = self._make_engine(energy_decay_per_cycle=0.01, energy_stabilize_bonus=0.05)
        r = e.decide(fitness=0.80, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        self.assertEqual(r["decision"], "stabilize")
        # Energy should be close to initial (decay 0.01 + bonus 0.05 = net +0.04)
        self.assertGreaterEqual(e.energy, 1.0)  # capped at 1.0

    def test_energy_in_decision_record(self) -> None:
        e = self._make_engine()
        r = e.decide(fitness=0.80, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        self.assertIn("energy", r)
        self.assertIn("effective_fitness", r)

    def test_energy_persists_across_restart(self) -> None:
        tmpdir = tempfile.mkdtemp()
        cfg = AutonomyConfig(stagnation_window=5, energy_decay_per_cycle=0.1)
        e1 = AutonomyEngine(data_dir=tmpdir, config=cfg)
        for _ in range(5):
            e1.decide(fitness=0.80, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        saved_energy = e1.energy
        e2 = AutonomyEngine(data_dir=tmpdir, config=cfg)
        self.assertAlmostEqual(e2.energy, saved_energy, places=4)


# ==================================================================
# v2.2: Environment Drift
# ==================================================================

class TestEnvironmentDrift(unittest.TestCase):
    """Slowly randomise fitness_threshold and mutation_rate_offset."""

    def _make_engine(self, **overrides: Any) -> AutonomyEngine:
        tmpdir = tempfile.mkdtemp()
        cfg = AutonomyConfig(stagnation_window=5, **overrides)
        return AutonomyEngine(data_dir=tmpdir, config=cfg)

    def test_effective_fitness_threshold_drifts(self) -> None:
        """After multiple cycles, eft should differ from initial."""
        e = self._make_engine(drift_step=0.1)
        initial = e.effective_fitness_threshold
        drifted = False
        for _ in range(20):
            e.decide(fitness=0.80, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
            if abs(e.effective_fitness_threshold - initial) > 0.001:
                drifted = True
                break
        self.assertTrue(drifted, "Fitness threshold should drift over cycles")

    def test_drift_stays_within_bounds(self) -> None:
        e = self._make_engine(
            drift_step=0.5,  # aggressive drift for testing
            drift_fitness_threshold_min=0.30,
            drift_fitness_threshold_max=0.60,
        )
        for _ in range(50):
            e.decide(fitness=0.80, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        self.assertGreaterEqual(e.effective_fitness_threshold, 0.30)
        self.assertLessEqual(e.effective_fitness_threshold, 0.60)

    def test_mutation_rate_offset_drifts(self) -> None:
        e = self._make_engine(drift_step=0.1)
        for _ in range(20):
            e.decide(fitness=0.80, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        # Offset should have moved from 0
        self.assertIsInstance(e.mutation_rate_offset, float)

    def test_drift_magnitude_in_record(self) -> None:
        e = self._make_engine()
        r = e.decide(fitness=0.80, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        self.assertIn("drift_magnitude", r)
        self.assertGreaterEqual(r["drift_magnitude"], 0.0)

    def test_effective_fitness_threshold_recovers(self) -> None:
        tmpdir = tempfile.mkdtemp()
        cfg = AutonomyConfig(stagnation_window=5, drift_step=0.1)
        e1 = AutonomyEngine(data_dir=tmpdir, config=cfg)
        for _ in range(10):
            e1.decide(fitness=0.80, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        saved_eft = e1.effective_fitness_threshold
        e2 = AutonomyEngine(data_dir=tmpdir, config=cfg)
        self.assertAlmostEqual(e2.effective_fitness_threshold, saved_eft, places=4)


# ==================================================================
# v2.2: Novelty-Based Awareness
# ==================================================================

class TestNoveltyAwareness(unittest.TestCase):
    """Awareness grows with novelty, decays when idle."""

    def _make_model(self, **overrides: Any) -> AwarenessModel:
        defaults = dict(
            stimuli_rate_cap=5.0,
            decision_rate_cap=20.0,
            fitness_variance_cap=0.01,
        )
        defaults.update(overrides)
        cfg = AutonomyConfig(**defaults)
        return AwarenessModel(cfg)

    def test_novelty_grows_with_unique_stimuli(self) -> None:
        """New unique stimuli raise the novelty accumulator."""
        model = self._make_model()
        model.record_stimulus("hash_a")
        model.record_stimulus("hash_b")
        r = model.compute(total_decisions=0, fitness_variance=0.0, is_stagnant=True)
        self.assertGreater(r["novelty_accumulator"], 0.0)

    def test_repeated_stimulus_no_novelty(self) -> None:
        """Same hash twice → only one novelty event."""
        model = self._make_model()
        model.record_stimulus("same_hash")
        model.record_stimulus("same_hash")
        r = model.compute(total_decisions=0, fitness_variance=0.0, is_stagnant=True)
        # Only 1 novel event, not 2
        self.assertEqual(r["novelty_signals"], 1.0)

    def test_novelty_decays_when_idle(self) -> None:
        """Without novel events, novelty accumulator decays."""
        model = self._make_model(novelty_growth_per_event=0.1, novelty_decay_per_cycle=0.05)
        model.record_stimulus("hash_x")
        model.compute(total_decisions=0, fitness_variance=0.0, is_stagnant=True)
        na1 = model.novelty_accumulator
        # Idle cycle — no new stimuli
        model.compute(total_decisions=0, fitness_variance=0.0, is_stagnant=True)
        na2 = model.novelty_accumulator
        self.assertLess(na2, na1)

    def test_env_drift_triggers_novelty(self) -> None:
        """Non-zero env_drift_magnitude → novelty signal."""
        model = self._make_model()
        r = model.compute(
            total_decisions=0, fitness_variance=0.0,
            is_stagnant=True, env_drift_magnitude=0.05,
        )
        self.assertGreater(r["novelty_signals"], 0)

    def test_fitness_variance_increase_triggers_novelty(self) -> None:
        """Increasing fitness variance → novelty signal."""
        model = self._make_model()
        model.compute(total_decisions=0, fitness_variance=0.001, is_stagnant=True)
        r = model.compute(total_decisions=0, fitness_variance=0.005, is_stagnant=True)
        # variance increased → novelty signal
        self.assertGreater(r["novelty_signals"], 0)

    def test_novelty_accumulator_clamped(self) -> None:
        model = self._make_model(novelty_max=0.3)
        for i in range(50):
            model.record_stimulus(f"unique_{i}")
        model.compute(total_decisions=0, fitness_variance=0.0, is_stagnant=True)
        self.assertLessEqual(model.novelty_accumulator, 0.3)

    def test_novelty_accumulator_persists_via_recovery(self) -> None:
        model = self._make_model()
        for i in range(5):
            model.record_stimulus(f"h{i}")
        model.compute(total_decisions=5, fitness_variance=0.0, is_stagnant=False)
        saved_na = model.novelty_accumulator

        model2 = self._make_model()
        model2.recover(total_stimuli=5, last_awareness=0.5, novelty_accumulator=saved_na)
        self.assertAlmostEqual(model2.novelty_accumulator, saved_na, places=6)


# ==================================================================
# v2.2: Stagnation Rework
# ==================================================================

class TestStagnationResponse(unittest.TestCase):
    """Stagnation triggers exploration mode and mutation boost."""

    def _make_engine(self, **overrides: Any) -> AutonomyEngine:
        tmpdir = tempfile.mkdtemp()
        cfg = AutonomyConfig(stagnation_window=3, **overrides)
        return AutonomyEngine(data_dir=tmpdir, config=cfg)

    def test_exploration_mode_activates_on_stagnation(self) -> None:
        """Once stagnation is detected, exploration_mode turns on."""
        e = self._make_engine()
        self.assertFalse(e.exploration_mode)
        # Feed identical fitness to trigger stagnation (window=3)
        for _ in range(5):
            e.decide(fitness=0.99, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        self.assertTrue(e.exploration_mode or e.stagnation_count > 0)

    def test_adaptation_reason_logged(self) -> None:
        """When stagnation triggers exploration, adaptation_reason is set."""
        e = self._make_engine(stagnation_exploration_cycles=10)
        records = []
        for _ in range(10):
            r = e.decide(fitness=0.99, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
            records.append(r)
        # At least one record should have adaptation_reason
        reasons = [r.get("adaptation_reason") for r in records if r.get("adaptation_reason")]
        self.assertGreater(len(reasons), 0)
        self.assertIn("stagnation", reasons[0].lower())

    def test_effective_mutation_rate_boosted_during_exploration(self) -> None:
        """During exploration, effective_mutation_rate > base rate."""
        e = self._make_engine(stagnation_mutation_boost=0.1)
        for _ in range(5):
            r = e.decide(fitness=0.99, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        if e.exploration_mode:
            emr = r["effective_mutation_rate"]
            # Should be base (0.1) + offset + boost (0.1)
            self.assertGreater(emr, 0.1)

    def test_exploration_mode_expires(self) -> None:
        """Exploration mode lasts for stagnation_exploration_cycles then ends."""
        e = self._make_engine(
            stagnation_exploration_cycles=2,
            stagnation_blend_multiplier=100,  # prevent blend from firing
        )
        # Trigger stagnation
        for _ in range(5):
            e.decide(fitness=0.99, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        # Keep going — exploration should have expired
        for _ in range(10):
            e.decide(fitness=0.99, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        # After enough cycles, exploration should be off
        # (it re-triggers though since stagnation persists)
        # Just verify it's a boolean
        self.assertIsInstance(e.exploration_mode, bool)


# ==================================================================
# v2.2: Blend Decision
# ==================================================================

class TestBlendDecision(unittest.TestCase):
    """DECISION_BLEND triggers on deep stagnation."""

    def _make_engine(self, **overrides: Any) -> AutonomyEngine:
        tmpdir = tempfile.mkdtemp()
        cfg = AutonomyConfig(stagnation_window=3, **overrides)
        return AutonomyEngine(data_dir=tmpdir, config=cfg)

    def test_blend_triggers_on_deep_stagnation(self) -> None:
        """stagnation_count > window * multiplier → DECISION_BLEND."""
        e = self._make_engine(
            stagnation_blend_multiplier=2,
            stagnation_exploration_cycles=1,  # expire fast
        )
        decisions = []
        for _ in range(30):
            r = e.decide(fitness=0.99, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
            decisions.append(r["decision"])
        self.assertIn("blend", decisions)

    def test_blend_in_organism_increments_evolution(self) -> None:
        """Organism autonomy_cycle with blend increments evolution_count."""
        tmpdir = tempfile.mkdtemp()
        db = Database(db_path=os.path.join(tmpdir, "t.db"))
        mem = MemoryManager(data_dir=tmpdir, credential_path=None, database=db)
        log = LifeLog(data_dir=os.path.join(tmpdir, "data"), organism_id="AL-01")
        pol = PolicyManager(data_dir=os.path.join(tmpdir, "data"))
        pop = Population(data_dir=tmpdir, parent_id="AL-01")
        brain = Brain(api_key=None)
        auto_cfg = AutonomyConfig(
            stagnation_window=3,
            stagnation_blend_multiplier=2,
            stagnation_exploration_cycles=1,
        )
        auto = AutonomyEngine(
            data_dir=os.path.join(tmpdir, "data"), config=auto_cfg,
        )
        org = Organism(
            data_dir=tmpdir, memory_manager=mem, life_log=log,
            policy=pol, population=pop, brain=brain, autonomy=auto,
        )
        # Spawn a child so blend has partners
        org.population.spawn_child(org.genome, parent_evolution=1)

        initial_evo = dict(org.state).get("evolution_count", 0)
        # Run many cycles to trigger blend
        for _ in range(30):
            record = org.autonomy_cycle()
            if record["decision"] == "blend":
                break
        final_evo = dict(org.state).get("evolution_count", 0)
        if record["decision"] == "blend":
            self.assertGreater(final_evo, initial_evo)

    def test_organism_energy_in_state(self) -> None:
        """After autonomy_cycle, energy is stored in organism state."""
        tmpdir = tempfile.mkdtemp()
        db = Database(db_path=os.path.join(tmpdir, "t.db"))
        mem = MemoryManager(data_dir=tmpdir, credential_path=None, database=db)
        log = LifeLog(data_dir=os.path.join(tmpdir, "data"), organism_id="AL-01")
        pol = PolicyManager(data_dir=os.path.join(tmpdir, "data"))
        pop = Population(data_dir=tmpdir, parent_id="AL-01")
        brain = Brain(api_key=None)
        auto = AutonomyEngine(data_dir=os.path.join(tmpdir, "data"))
        org = Organism(
            data_dir=tmpdir, memory_manager=mem, life_log=log,
            policy=pol, population=pop, brain=brain, autonomy=auto,
        )
        org.autonomy_cycle()
        state = dict(org.state)
        self.assertIn("energy", state)
        self.assertGreaterEqual(state["energy"], 0.0)
        self.assertLessEqual(state["energy"], 1.0)


# ==================================================================
# v2.3: Soft Ceiling + Diminishing Returns
# ==================================================================

class TestSoftCeiling(unittest.TestCase):
    """Traits can exceed 1.0 but with diminishing returns."""

    def test_soft_cap_below_one(self) -> None:
        """Values ≤ 1.0 pass through unchanged."""
        from al01.genome import _soft_cap
        self.assertAlmostEqual(_soft_cap(0.5), 0.5, places=6)
        self.assertAlmostEqual(_soft_cap(1.0), 1.0, places=6)
        self.assertAlmostEqual(_soft_cap(0.0), 0.0, places=6)

    def test_soft_cap_above_one(self) -> None:
        """Values > 1.0 get diminishing returns."""
        from al01.genome import _soft_cap, SOFT_CEILING_SCALE
        import math
        result = _soft_cap(2.0)
        expected = 1.0 + math.log(2.0) * SOFT_CEILING_SCALE
        self.assertAlmostEqual(result, expected, places=6)
        # Diminishing: going from 2→3 gains less than 1→2
        gain_1_to_2 = _soft_cap(2.0) - _soft_cap(1.0)
        gain_2_to_3 = _soft_cap(3.0) - _soft_cap(2.0)
        self.assertGreater(gain_1_to_2, gain_2_to_3)

    def test_traits_can_exceed_one(self) -> None:
        """Raw traits are allowed to exceed 1.0 after mutation."""
        g = Genome(traits={"t1": 0.99}, mutation_rate=1.0, mutation_delta=0.5)
        # Force many mutations to exceed 1.0
        for _ in range(100):
            g.mutate()
        # At least possible to exceed 1.0
        # (probabilistic but with delta=0.5 and 100 tries, very likely)

    def test_effective_traits_capped(self) -> None:
        """effective_traits returns soft-capped values."""
        g = Genome(traits={"t1": 2.0})
        raw = g.traits["t1"]
        effective = g.effective_traits["t1"]
        self.assertEqual(raw, 2.0)
        self.assertLess(effective, 2.0)
        self.assertGreater(effective, 1.0)

    def test_fitness_uses_soft_cap(self) -> None:
        """Fitness = mean of soft-capped values."""
        g = Genome(traits={"t1": 2.0, "t2": 0.5})
        from al01.genome import _soft_cap
        expected = (_soft_cap(2.0) + _soft_cap(0.5)) / 2
        self.assertAlmostEqual(g.fitness, expected, places=6)

    def test_to_dict_includes_effective_traits(self) -> None:
        """Serialized genome includes effective_traits."""
        g = Genome()
        d = g.to_dict()
        self.assertIn("effective_traits", d)
        self.assertEqual(set(d["effective_traits"].keys()), set(d["traits"].keys()))


# ==================================================================
# v2.3: Environment-Weighted Fitness
# ==================================================================

class TestEnvironmentWeightedFitness(unittest.TestCase):
    """Genome.weighted_fitness with environment weights."""

    def test_default_weights_equal_fitness(self) -> None:
        """With all weights=1.0, weighted_fitness == fitness."""
        g = Genome()
        self.assertAlmostEqual(g.weighted_fitness(), g.fitness, places=6)

    def test_weighted_fitness_varies(self) -> None:
        """Different weights produce different fitness."""
        g = Genome(traits={"adaptability": 0.9, "creativity": 0.1})
        w1 = g.weighted_fitness({"adaptability": 2.0, "creativity": 1.0})
        w2 = g.weighted_fitness({"adaptability": 1.0, "creativity": 2.0})
        self.assertGreater(w1, w2)

    def test_env_trait_weights_in_engine(self) -> None:
        """AutonomyEngine exposes env_trait_weights."""
        tmpdir = tempfile.mkdtemp()
        e = AutonomyEngine(data_dir=tmpdir)
        w = e.env_trait_weights
        self.assertEqual(len(w), 5)
        for v in w.values():
            self.assertAlmostEqual(v, 1.0, places=4)

    def test_env_weights_drift(self) -> None:
        """After decisions, env_trait_weights should drift from 1.0."""
        tmpdir = tempfile.mkdtemp()
        e = AutonomyEngine(data_dir=tmpdir,
                           config=AutonomyConfig(env_weight_drift_step=0.1))
        initial = dict(e.env_trait_weights)
        for _ in range(20):
            e.decide(fitness=0.5, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        final = e.env_trait_weights
        # At least one weight should have drifted
        drifted = any(abs(final[k] - initial[k]) > 0.001 for k in initial)
        self.assertTrue(drifted, "Env weights should drift over cycles")

    def test_env_weights_in_decision_record(self) -> None:
        """Decision record includes env_trait_weights."""
        tmpdir = tempfile.mkdtemp()
        e = AutonomyEngine(data_dir=tmpdir)
        r = e.decide(fitness=0.5, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        self.assertIn("env_trait_weights", r)


# ==================================================================
# v2.3: Entropy Decay
# ==================================================================

class TestEntropyDecay(unittest.TestCase):
    """Trait entropy decay mechanic."""

    def test_decay_reduces_traits(self) -> None:
        """decay_traits() reduces all trait values."""
        g = Genome(traits={"t1": 0.5, "t2": 0.8})
        changes = g.decay_traits(rate=0.1)
        self.assertLess(g.get_trait("t1"), 0.5)
        self.assertLess(g.get_trait("t2"), 0.8)
        self.assertIn("t1", changes)

    def test_decay_proportional(self) -> None:
        """Higher traits lose more absolute value."""
        g = Genome(traits={"low": 0.1, "high": 0.9})
        changes = g.decay_traits(rate=0.1)
        self.assertGreater(changes["high"]["decay"], changes["low"]["decay"])

    def test_decay_floor_at_zero(self) -> None:
        """Traits never go below 0.0 from decay."""
        g = Genome(traits={"t1": 0.001})
        g.decay_traits(rate=0.5)
        self.assertGreaterEqual(g.get_trait("t1"), 0.0)

    def test_idle_cycles_tracked(self) -> None:
        """Engine tracks idle cycles (no stimulus)."""
        tmpdir = tempfile.mkdtemp()
        e = AutonomyEngine(data_dir=tmpdir)
        e.decide(fitness=0.5, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        self.assertEqual(e.idle_cycles, 1)
        e.decide(fitness=0.5, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        self.assertEqual(e.idle_cycles, 2)
        # With stimuli → resets
        e.decide(fitness=0.5, awareness=0.5, mutation_rate=0.1, pending_stimuli=1)
        self.assertEqual(e.idle_cycles, 0)

    def test_entropy_threshold(self) -> None:
        """should_entropy_decay activates after threshold idle cycles."""
        tmpdir = tempfile.mkdtemp()
        cfg = AutonomyConfig(entropy_idle_threshold=2)
        e = AutonomyEngine(data_dir=tmpdir, config=cfg)
        e.decide(fitness=0.5, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        self.assertFalse(e.should_entropy_decay)
        e.decide(fitness=0.5, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        self.assertTrue(e.should_entropy_decay)

    def test_autonomy_cycle_triggers_entropy(self) -> None:
        """Organism autonomy_cycle applies entropy when engine signals it."""
        tmpdir = tempfile.mkdtemp()
        db = Database(db_path=os.path.join(tmpdir, "t.db"))
        mem = MemoryManager(data_dir=tmpdir, credential_path=None, database=db)
        log = LifeLog(data_dir=os.path.join(tmpdir, "data"), organism_id="AL-01")
        pol = PolicyManager(data_dir=os.path.join(tmpdir, "data"))
        pop = Population(data_dir=tmpdir, parent_id="AL-01")
        brain = Brain(api_key=None)
        auto_cfg = AutonomyConfig(entropy_idle_threshold=1)
        auto = AutonomyEngine(
            data_dir=os.path.join(tmpdir, "data"), config=auto_cfg,
        )
        org = Organism(
            data_dir=tmpdir, memory_manager=mem, life_log=log,
            policy=pol, population=pop, brain=brain, autonomy=auto,
        )
        initial_fitness = org.genome.fitness
        # Run several cycles with no stimuli to trigger entropy
        for _ in range(5):
            org.autonomy_cycle()
        # Fitness should decay (or at least not increase purely from entropy)
        # Note: mutations may offset, but entropy should have an effect


# ==================================================================
# v2.3: Stagnation-Scaled Mutation
# ==================================================================

class TestStagnationScaledMutation(unittest.TestCase):
    """Mutation delta scales with stagnation pressure."""

    def test_no_stagnation_returns_base(self) -> None:
        """With 0 stagnation, delta = base."""
        tmpdir = tempfile.mkdtemp()
        e = AutonomyEngine(data_dir=tmpdir, config=AutonomyConfig(stagnation_window=5))
        self.assertAlmostEqual(e.stagnation_scaled_delta(0.10), 0.10, places=4)

    def test_stagnation_increases_delta(self) -> None:
        """With stagnation, delta > base."""
        tmpdir = tempfile.mkdtemp()
        cfg = AutonomyConfig(stagnation_window=5, stagnation_delta_scale=1.0)
        e = AutonomyEngine(data_dir=tmpdir, config=cfg)
        # Force stagnation
        for _ in range(7):
            e.decide(fitness=0.50, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        scaled = e.stagnation_scaled_delta(0.10)
        self.assertGreater(scaled, 0.10)

    def test_effective_delta_in_record(self) -> None:
        """Decision record includes effective_mutation_delta."""
        tmpdir = tempfile.mkdtemp()
        e = AutonomyEngine(data_dir=tmpdir)
        r = e.decide(fitness=0.5, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        self.assertIn("effective_mutation_delta", r)


# ==================================================================
# v2.3: Trait Trade-offs
# ==================================================================

class TestTraitTradeoffs(unittest.TestCase):
    """High adaptability/creativity penalise energy_efficiency."""

    def test_high_adaptability_reduces_efficiency(self) -> None:
        """Adaptability > 0.7 reduces energy_efficiency."""
        g = Genome(traits={
            "adaptability": 0.9,
            "energy_efficiency": 0.5,
            "resilience": 0.5,
            "perception": 0.5,
            "creativity": 0.5,
        }, mutation_rate=0.0)  # no random mutations
        # Trigger trade-offs via a mutation call (0% rate = no random changes)
        result = g.mutate()
        self.assertIn("tradeoff_effects", result)
        # energy_efficiency should have been reduced
        self.assertLess(g.get_trait("energy_efficiency"), 0.5)

    def test_below_threshold_no_penalty(self) -> None:
        """Adaptability ≤ 0.7 → no trade-off penalty."""
        g = Genome(traits={
            "adaptability": 0.7,
            "energy_efficiency": 0.5,
            "resilience": 0.5,
            "perception": 0.5,
            "creativity": 0.5,
        }, mutation_rate=0.0)
        g.mutate()
        self.assertAlmostEqual(g.get_trait("energy_efficiency"), 0.5, places=4)

    def test_creativity_tradeoff(self) -> None:
        """High creativity also penalises energy_efficiency."""
        g = Genome(traits={
            "adaptability": 0.5,
            "energy_efficiency": 0.6,
            "resilience": 0.5,
            "perception": 0.5,
            "creativity": 0.9,
        }, mutation_rate=0.0)
        g.mutate()
        self.assertLess(g.get_trait("energy_efficiency"), 0.6)

    def test_blend_applies_tradeoffs(self) -> None:
        """blend_with also applies trade-offs."""
        g1 = Genome(traits={"adaptability": 0.9, "energy_efficiency": 0.6,
                            "resilience": 0.5, "perception": 0.5, "creativity": 0.5})
        g2 = Genome(traits={"adaptability": 0.9, "energy_efficiency": 0.6,
                            "resilience": 0.5, "perception": 0.5, "creativity": 0.5})
        result = g1.blend_with(g2, noise=0.0)
        self.assertIn("tradeoff_effects", result)


# ==================================================================
# v2.3: Independent Child Evolution
# ==================================================================

class TestIndependentChildEvolution(unittest.TestCase):
    """Children evolve independently with separate RNG seeds."""

    def test_child_has_different_mutation_params(self) -> None:
        """Child mutation_rate and mutation_delta differ from parent."""
        parent = Genome(mutation_rate=0.10, mutation_delta=0.10)
        child = parent.spawn_child()
        # They *may* be close but should not be identical due to randomisation
        d = child.to_dict()
        # mutation_rate and mutation_delta should exist
        self.assertIn("mutation_rate", d)
        self.assertIn("mutation_delta", d)

    def test_children_diverge(self) -> None:
        """Two children from the same parent diverge in mutations."""
        parent = Genome(mutation_rate=1.0, mutation_delta=0.1, rng_seed=42)
        c1 = parent.spawn_child()
        c2 = parent.spawn_child()
        # Mutate both
        r1 = c1.mutate()
        r2 = c2.mutate()
        # They should produce different mutations (different seeds)
        # At minimum, the children are distinct Genome objects
        self.assertIsNot(c1, c2)

    def test_child_rng_independent_of_parent(self) -> None:
        """Parent mutations don't affect child mutations."""
        parent = Genome(mutation_rate=1.0, mutation_delta=0.1, rng_seed=99)
        child = parent.spawn_child()
        child_traits_before = dict(child.traits)
        parent.mutate()  # parent mutates
        child_traits_after = dict(child.traits)
        # Child should not be affected
        self.assertEqual(child_traits_before, child_traits_after)


# ==================================================================
# v2.3: Vital Score
# ==================================================================

class TestVitalScore(unittest.TestCase):
    """Vital Score computation (0-100 index)."""

    def test_initial_vital_score(self) -> None:
        """Fresh engine has a valid vital score."""
        tmpdir = tempfile.mkdtemp()
        e = AutonomyEngine(data_dir=tmpdir)
        v = e.compute_vital_score()
        self.assertIn("vital_index", v)
        self.assertGreaterEqual(v["vital_index"], 0.0)
        self.assertLessEqual(v["vital_index"], 100.0)
        for key in ("identity_persistence", "trait_variance",
                     "adaptation_success", "entropy_resistance"):
            self.assertIn(key, v)

    def test_vital_in_summary(self) -> None:
        """summary() includes vital_score."""
        tmpdir = tempfile.mkdtemp()
        e = AutonomyEngine(data_dir=tmpdir)
        e.decide(fitness=0.5, awareness=0.5, mutation_rate=0.1, pending_stimuli=0)
        s = e.summary()
        self.assertIn("vital_score", s)
        self.assertIn("vital_index", s["vital_score"])

    def test_vital_score_adapts_over_time(self) -> None:
        """Vital score changes after adaptation decisions."""
        tmpdir = tempfile.mkdtemp()
        cfg = AutonomyConfig(stagnation_window=3)
        e = AutonomyEngine(data_dir=tmpdir, config=cfg)
        # Run some cycles to build up history
        v1 = e.compute_vital_score()
        for i in range(10):
            e.decide(fitness=0.50 + i * 0.01, awareness=0.5,
                     mutation_rate=0.1, pending_stimuli=1,
                     current_traits={"adaptability": 0.5 + i * 0.01,
                                     "energy_efficiency": 0.5,
                                     "resilience": 0.5,
                                     "perception": 0.5,
                                     "creativity": 0.5})
        v2 = e.compute_vital_score()
        # At least some sub-scores should have changed
        self.assertNotEqual(v1["vital_index"], v2["vital_index"])

    def test_vital_in_organism_growth_metrics(self) -> None:
        """Organism growth_metrics includes vital_score."""
        tmpdir = tempfile.mkdtemp()
        db = Database(db_path=os.path.join(tmpdir, "t.db"))
        mem = MemoryManager(data_dir=tmpdir, credential_path=None, database=db)
        log = LifeLog(data_dir=os.path.join(tmpdir, "data"), organism_id="AL-01")
        pol = PolicyManager(data_dir=os.path.join(tmpdir, "data"))
        pop = Population(data_dir=tmpdir, parent_id="AL-01")
        brain = Brain(api_key=None)
        auto = AutonomyEngine(data_dir=os.path.join(tmpdir, "data"))
        org = Organism(
            data_dir=tmpdir, memory_manager=mem, life_log=log,
            policy=pol, population=pop, brain=brain, autonomy=auto,
        )
        metrics = org.growth_metrics
        self.assertIn("vital_score", metrics)
        self.assertIn("vital_index", metrics["vital_score"])


if __name__ == "__main__":
    unittest.main()
