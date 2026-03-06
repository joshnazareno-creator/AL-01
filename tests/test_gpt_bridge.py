"""AL-01 GPT Bridge — tests for narration, stimulus injection, rate limiting,
toggle, and API endpoints.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from typing import Any, Dict
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from al01.gpt_bridge import GPTBridge, GPTBridgeConfig
from al01.organism import Organism, VERSION
from al01.memory_manager import MemoryManager
from al01.database import Database
from al01.life_log import LifeLog
from al01.policy import PolicyManager


def _make_organism(tmpdir: str) -> Organism:
    db = Database(db_path=os.path.join(tmpdir, "t.db"))
    mem = MemoryManager(data_dir=tmpdir, credential_path=None, database=db)
    log = LifeLog(data_dir=os.path.join(tmpdir, "data"), organism_id="AL-01")
    pol = PolicyManager(data_dir=os.path.join(tmpdir, "data"))
    return Organism(data_dir=tmpdir, memory_manager=mem, life_log=log, policy=pol)


# ==================================================================
# 1. GPTBridgeConfig defaults
# ==================================================================

class TestGPTBridgeConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = GPTBridgeConfig()
        self.assertEqual(cfg.stimulus_rate_limit, 6)
        self.assertEqual(cfg.rate_window_seconds, 60.0)
        self.assertEqual(cfg.max_stimulus_length, 280)
        self.assertTrue(cfg.stimulus_enabled)
        self.assertEqual(cfg.narration_population_limit, 10)

    def test_custom(self):
        cfg = GPTBridgeConfig(stimulus_rate_limit=2, max_stimulus_length=100)
        self.assertEqual(cfg.stimulus_rate_limit, 2)
        self.assertEqual(cfg.max_stimulus_length, 100)


# ==================================================================
# 2. Narration
# ==================================================================

class TestNarration(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.org = _make_organism(self._tmpdir)
        self.bridge = GPTBridge(self.org)

    def test_narrate_returns_prose(self):
        result = self.bridge.narrate()
        self.assertIn("prose", result)
        self.assertIn("raw", result)
        self.assertIn("timestamp", result)
        self.assertIsInstance(result["prose"], str)
        self.assertGreater(len(result["prose"]), 50)

    def test_narrate_mentions_al01(self):
        result = self.bridge.narrate()
        self.assertIn("AL-01", result["prose"])

    def test_narrate_raw_has_core_fields(self):
        result = self.bridge.narrate()
        raw = result["raw"]
        for key in ["organism_state", "evolution_count", "awareness",
                     "energy", "fitness", "genome", "autonomy", "population",
                     "pending_stimuli", "loop_running"]:
            self.assertIn(key, raw, f"Missing key: {key}")

    def test_narrate_population_section(self):
        result = self.bridge.narrate()
        raw = result["raw"]
        pop = raw["population"]
        self.assertIn("living", pop)
        self.assertIn("total", pop)
        self.assertIn("top_members", pop)
        self.assertGreaterEqual(pop["living"], 1)  # at least AL-01

    def test_narrate_genome_traits(self):
        result = self.bridge.narrate()
        prose = result["prose"]
        # Should mention at least one trait name
        self.assertTrue(
            any(t in prose for t in ["adaptability", "resilience", "creativity",
                                      "energy_efficiency", "perception"]),
            "No genome trait found in prose",
        )

    def test_narrate_energy_words(self):
        self.assertEqual(GPTBridge._energy_word(0.9), "high")
        self.assertEqual(GPTBridge._energy_word(0.6), "moderate")
        self.assertEqual(GPTBridge._energy_word(0.3), "low")
        self.assertEqual(GPTBridge._energy_word(0.1), "critical")

    def test_narrate_fitness_words(self):
        self.assertEqual(GPTBridge._fitness_word(0.85), "excellent")
        self.assertEqual(GPTBridge._fitness_word(0.65), "good")
        self.assertEqual(GPTBridge._fitness_word(0.45), "average")
        self.assertEqual(GPTBridge._fitness_word(0.25), "below average")
        self.assertEqual(GPTBridge._fitness_word(0.1), "poor")

    def test_narrate_awareness_words(self):
        self.assertEqual(GPTBridge._awareness_word(0.9), "highly aware")
        self.assertEqual(GPTBridge._awareness_word(0.6), "moderately aware")
        self.assertEqual(GPTBridge._awareness_word(0.3), "somewhat aware")
        self.assertEqual(GPTBridge._awareness_word(0.1), "barely aware")


# ==================================================================
# 3. Stimulus injection
# ==================================================================

class TestStimulusInjection(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.org = _make_organism(self._tmpdir)
        self.bridge = GPTBridge(self.org, GPTBridgeConfig(stimulus_rate_limit=3))

    def test_accept_valid_stimulus(self):
        result = self.bridge.inject_stimulus("environmental_change")
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["injection_number"], 1)
        self.assertIn("queued_stimuli", result)

    def test_reject_empty_stimulus(self):
        result = self.bridge.inject_stimulus("")
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reason"], "empty_stimulus")

    def test_reject_whitespace_only(self):
        result = self.bridge.inject_stimulus("   ")
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reason"], "empty_stimulus")

    def test_reject_too_long(self):
        cfg = GPTBridgeConfig(max_stimulus_length=10)
        bridge = GPTBridge(self.org, cfg)
        result = bridge.inject_stimulus("x" * 20)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reason"], "too_long")

    def test_reject_when_disabled(self):
        cfg = GPTBridgeConfig(stimulus_enabled=False)
        bridge = GPTBridge(self.org, cfg)
        result = bridge.inject_stimulus("test")
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reason"], "stimulus_disabled")

    def test_rate_limit(self):
        cfg = GPTBridgeConfig(stimulus_rate_limit=2, rate_window_seconds=60.0)
        bridge = GPTBridge(self.org, cfg)
        r1 = bridge.inject_stimulus("one")
        r2 = bridge.inject_stimulus("two")
        r3 = bridge.inject_stimulus("three")
        self.assertEqual(r1["status"], "accepted")
        self.assertEqual(r2["status"], "accepted")
        self.assertEqual(r3["status"], "rejected")
        self.assertEqual(r3["reason"], "rate_limited")

    def test_rate_limit_window_expires(self):
        cfg = GPTBridgeConfig(stimulus_rate_limit=1, rate_window_seconds=0.2)
        bridge = GPTBridge(self.org, cfg)
        r1 = bridge.inject_stimulus("first")
        self.assertEqual(r1["status"], "accepted")
        # Immediately should be rate-limited
        r2 = bridge.inject_stimulus("second")
        self.assertEqual(r2["status"], "rejected")
        # Wait for window to expire
        time.sleep(0.3)
        r3 = bridge.inject_stimulus("third")
        self.assertEqual(r3["status"], "accepted")

    def test_stimulus_queued_in_organism(self):
        self.bridge.inject_stimulus("test_event")
        # The stimulate() method adds to the queue
        stimuli = self.org.stimuli
        self.assertIn("test_event", stimuli)

    def test_injection_counter_increments(self):
        self.bridge.inject_stimulus("a")
        self.bridge.inject_stimulus("b")
        self.assertEqual(self.bridge._total_injections, 2)


# ==================================================================
# 4. Status & toggle
# ==================================================================

class TestStatusAndToggle(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.org = _make_organism(self._tmpdir)
        self.bridge = GPTBridge(self.org)

    def test_status_fields(self):
        s = self.bridge.status()
        for key in ["stimulus_enabled", "rate_limit", "rate_window_seconds",
                     "injections_in_window", "total_injections", "total_rejections",
                     "max_stimulus_length", "log_entries", "timestamp"]:
            self.assertIn(key, s, f"Missing status key: {key}")

    def test_toggle_off_and_on(self):
        r = self.bridge.set_stimulus_enabled(False)
        self.assertFalse(r["stimulus_enabled"])
        self.assertTrue(r["previous"])
        # verify injection is now rejected
        inj = self.bridge.inject_stimulus("hey")
        self.assertEqual(inj["status"], "rejected")
        # toggle back on
        r2 = self.bridge.set_stimulus_enabled(True)
        self.assertTrue(r2["stimulus_enabled"])
        inj2 = self.bridge.inject_stimulus("hey")
        self.assertEqual(inj2["status"], "accepted")

    def test_rejection_counter(self):
        self.bridge.set_stimulus_enabled(False)
        self.bridge.inject_stimulus("a")
        self.bridge.inject_stimulus("b")
        self.assertEqual(self.bridge._total_rejections, 2)


# ==================================================================
# 5. Injection log
# ==================================================================

class TestInjectionLog(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.org = _make_organism(self._tmpdir)
        self.bridge = GPTBridge(self.org)

    def test_log_records_injection(self):
        self.bridge.inject_stimulus("stimulus_one")
        entries = self.bridge.recent_injections()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["text"], "stimulus_one")
        self.assertTrue(entries[0]["accepted"])

    def test_log_limit(self):
        bridge = GPTBridge(self.org, GPTBridgeConfig(stimulus_rate_limit=999))
        for i in range(10):
            bridge.inject_stimulus(f"s{i}")
        entries = bridge.recent_injections(limit=3)
        self.assertEqual(len(entries), 3)
        # Should be the 3 most recent
        self.assertEqual(entries[-1]["text"], "s9")

    def test_log_bounded(self):
        """Log doesn't grow unbounded."""
        bridge = GPTBridge(self.org, GPTBridgeConfig(stimulus_rate_limit=999))
        bridge._max_log_entries = 5
        for i in range(10):
            bridge.inject_stimulus(f"s{i}")
        self.assertLessEqual(len(bridge._injection_log), 5)


# ==================================================================
# 6. Organism integration (lazy property)
# ==================================================================

class TestOrganismIntegration(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.org = _make_organism(self._tmpdir)

    def test_gpt_bridge_property_exists(self):
        bridge = self.org.gpt_bridge
        self.assertIsInstance(bridge, GPTBridge)

    def test_gpt_bridge_is_lazy_singleton(self):
        b1 = self.org.gpt_bridge
        b2 = self.org.gpt_bridge
        self.assertIs(b1, b2)

    def test_narrate_through_organism(self):
        result = self.org.gpt_bridge.narrate()
        self.assertIn("prose", result)
        self.assertIn("AL-01", result["prose"])

    def test_inject_through_organism(self):
        result = self.org.gpt_bridge.inject_stimulus("hello_gpt")
        self.assertEqual(result["status"], "accepted")


# ==================================================================
# 7. API endpoints (via TestClient)
# ==================================================================

class TestGPTAPIEndpoints(unittest.TestCase):
    """Test the /gpt/* FastAPI endpoints through the test client."""

    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            raise unittest.SkipTest("fastapi[testclient] not installed")
        cls.TestClient = TestClient

    def setUp(self):
        from al01.api import create_app
        self._tmpdir = tempfile.mkdtemp()
        self.org = _make_organism(self._tmpdir)
        app = create_app(self.org, api_key="test-key")
        self.client = self.TestClient(app)
        self.headers = {"X-API-Key": "test-key"}

    def test_gpt_narrate(self):
        r = self.client.get("/gpt/narrate", headers=self.headers)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("prose", data)
        self.assertIn("AL-01", data["prose"])

    def test_gpt_narrate_requires_auth(self):
        r = self.client.get("/gpt/narrate")
        self.assertEqual(r.status_code, 401)

    def test_gpt_stimulus(self):
        r = self.client.post(
            "/gpt/stimulus",
            json={"text": "environmental_pressure"},
            headers=self.headers,
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "accepted")

    def test_gpt_stimulus_requires_auth(self):
        r = self.client.post("/gpt/stimulus", json={"text": "test"})
        self.assertEqual(r.status_code, 401)

    def test_gpt_stimulus_rejects_empty(self):
        r = self.client.post(
            "/gpt/stimulus",
            json={"text": ""},
            headers=self.headers,
        )
        # FastAPI validation catches min_length=1
        self.assertIn(r.status_code, [200, 422])

    def test_gpt_status(self):
        r = self.client.get("/gpt/status", headers=self.headers)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("stimulus_enabled", data)
        self.assertIn("total_injections", data)

    def test_gpt_toggle(self):
        r = self.client.post("/gpt/toggle?enabled=false", headers=self.headers)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertFalse(data["stimulus_enabled"])
        # verify stimulus is now rejected
        r2 = self.client.post(
            "/gpt/stimulus",
            json={"text": "should_fail"},
            headers=self.headers,
        )
        self.assertEqual(r2.json()["status"], "rejected")

    def test_gpt_log(self):
        # inject one stimulus first
        self.client.post(
            "/gpt/stimulus",
            json={"text": "logged_event"},
            headers=self.headers,
        )
        r = self.client.get("/gpt/log", headers=self.headers)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["entries"][0]["text"], "logged_event")

    def test_gpt_log_empty(self):
        r = self.client.get("/gpt/log", headers=self.headers)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["count"], 0)

    def test_gpt_openapi_json(self):
        """The /gpt/openapi.json endpoint returns a valid OpenAPI 3.1 spec."""
        r = self.client.get("/gpt/openapi.json")
        self.assertEqual(r.status_code, 200)
        spec = r.json()
        self.assertEqual(spec["openapi"], "3.1.0")
        self.assertIn("info", spec)
        self.assertEqual(spec["info"]["title"], "AL-01 GPT Bridge")
        # servers field must be present (ChatGPT Actions requirement)
        self.assertIn("servers", spec)
        self.assertIsInstance(spec["servers"], list)
        self.assertGreater(len(spec["servers"]), 0)
        self.assertIn("url", spec["servers"][0])
        # Must include bridge paths
        self.assertIn("/gpt/narrate", spec["paths"])
        self.assertIn("/gpt/stimulus", spec["paths"])
        # operationIds must match expected values
        self.assertEqual(spec["paths"]["/gpt/narrate"]["get"]["operationId"], "narrate")
        self.assertEqual(spec["paths"]["/gpt/stimulus"]["post"]["operationId"], "stimulus")
        # X-API-Key auth via securitySchemes (ChatGPT Actions requirement)
        self.assertIn("components", spec)
        schemes = spec["components"]["securitySchemes"]
        self.assertIn("apiKey", schemes)
        self.assertEqual(schemes["apiKey"]["type"], "apiKey")
        self.assertEqual(schemes["apiKey"]["name"], "X-API-Key")
        self.assertEqual(schemes["apiKey"]["in"], "header")
        # Top-level security must reference the scheme
        self.assertIn("security", spec)
        self.assertIn({"apiKey": []}, spec["security"])

    def test_gpt_openapi_no_auth_required(self):
        """The schema endpoint itself should be accessible without auth."""
        r = self.client.get("/gpt/openapi.json")
        self.assertEqual(r.status_code, 200)


# ==================================================================
# 8. Non-interference with evolution loop
# ==================================================================

class TestNonInterference(unittest.TestCase):
    """Verify the bridge never calls evolve_cycle or autonomy_cycle."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.org = _make_organism(self._tmpdir)
        self.bridge = GPTBridge(self.org)

    def test_narrate_does_not_mutate_genome(self):
        genome_before = self.org.genome.to_dict()
        self.bridge.narrate()
        genome_after = self.org.genome.to_dict()
        self.assertEqual(genome_before, genome_after)

    def test_narrate_does_not_increment_evolution(self):
        evo_before = self.org.state.get("evolution_count", 0)
        self.bridge.narrate()
        evo_after = self.org.state.get("evolution_count", 0)
        self.assertEqual(evo_before, evo_after)

    def test_stimulus_does_not_trigger_cycle(self):
        """inject_stimulus should NOT call evolve_cycle — stimulus is only queued."""
        evo_before = self.org.state.get("evolution_count", 0)
        self.bridge.inject_stimulus("test_event")
        evo_after = self.org.state.get("evolution_count", 0)
        self.assertEqual(evo_before, evo_after)

    def test_stimulus_does_not_change_genome(self):
        genome_before = self.org.genome.to_dict()
        self.bridge.inject_stimulus("pressure")
        genome_after = self.org.genome.to_dict()
        self.assertEqual(genome_before, genome_after)


if __name__ == "__main__":
    unittest.main()
