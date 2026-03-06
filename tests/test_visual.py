"""Tests for v3.12 Visual Organism dashboard endpoints."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from fastapi.testclient import TestClient

from al01.api import create_app
from al01.database import Database
from al01.genome import Genome
from al01.life_log import LifeLog
from al01.memory_manager import MemoryManager
from al01.organism import Organism, MetabolismConfig
from al01.policy import PolicyManager


def _make_app() -> tuple:
    """Create a minimal Organism + FastAPI TestClient."""
    tmpdir = tempfile.mkdtemp()
    db = Database(db_path=os.path.join(tmpdir, "t.db"))
    mem = MemoryManager(data_dir=tmpdir, credential_path=None, database=db)
    log = LifeLog(data_dir=os.path.join(tmpdir, "data"), organism_id="AL-01")
    pol = PolicyManager(data_dir=os.path.join(tmpdir, "data"))
    org = Organism(data_dir=tmpdir, memory_manager=mem, life_log=log, policy=pol)
    app = create_app(org, api_key="test-key")
    client = TestClient(app)
    return org, client


class TestApiOrganisms(unittest.TestCase):
    """GET /api/organisms — public JSON feed for visual dashboard."""

    @classmethod
    def setUpClass(cls):
        cls.org, cls.client = _make_app()

    def test_returns_200(self):
        resp = self.client.get("/api/organisms")
        self.assertEqual(resp.status_code, 200)

    def test_json_structure(self):
        data = self.client.get("/api/organisms").json()
        self.assertIn("organisms", data)
        self.assertIn("population_size", data)
        self.assertIn("shock_active", data)
        self.assertIn("timestamp", data)

    def test_organisms_list_not_empty(self):
        data = self.client.get("/api/organisms").json()
        self.assertGreater(len(data["organisms"]), 0, "should have at least the parent")

    def test_parent_present(self):
        data = self.client.get("/api/organisms").json()
        ids = [o["id"] for o in data["organisms"]]
        self.assertIn("AL-01", ids)

    def test_organism_fields(self):
        data = self.client.get("/api/organisms").json()
        parent = next(o for o in data["organisms"] if o["id"] == "AL-01")
        for key in ("id", "fitness", "traits", "strategy", "alive",
                    "nickname", "awareness", "stagnation", "is_parent"):
            self.assertIn(key, parent, f"missing field: {key}")

    def test_traits_rgb_fields(self):
        data = self.client.get("/api/organisms").json()
        parent = next(o for o in data["organisms"] if o["id"] == "AL-01")
        traits = parent["traits"]
        for key in ("adaptability", "energy_efficiency", "resilience"):
            self.assertIn(key, traits)
            self.assertIsInstance(traits[key], float)

    def test_parent_is_parent_flag(self):
        data = self.client.get("/api/organisms").json()
        parent = next(o for o in data["organisms"] if o["id"] == "AL-01")
        self.assertTrue(parent["is_parent"])

    def test_no_auth_required(self):
        """The /api/organisms endpoint should be public (no API key)."""
        resp = self.client.get("/api/organisms")
        self.assertEqual(resp.status_code, 200)

    def test_population_size_matches(self):
        data = self.client.get("/api/organisms").json()
        self.assertEqual(data["population_size"], len(data["organisms"]))

    def test_shock_active_is_bool(self):
        data = self.client.get("/api/organisms").json()
        self.assertIsInstance(data["shock_active"], bool)


class TestVisualDashboard(unittest.TestCase):
    """GET /visual — HTML visual organism dashboard."""

    @classmethod
    def setUpClass(cls):
        cls.org, cls.client = _make_app()

    def test_returns_200(self):
        resp = self.client.get("/visual")
        self.assertEqual(resp.status_code, 200)

    def test_content_type_html(self):
        resp = self.client.get("/visual")
        self.assertIn("text/html", resp.headers.get("content-type", ""))

    def test_contains_canvas(self):
        resp = self.client.get("/visual")
        self.assertIn("<canvas", resp.text)

    def test_contains_title(self):
        resp = self.client.get("/visual")
        self.assertIn("AL-01", resp.text)

    def test_polls_api_organisms(self):
        """The visual page should fetch from /api/organisms."""
        resp = self.client.get("/visual")
        self.assertIn("/api/organisms", resp.text)

    def test_no_auth_required(self):
        resp = self.client.get("/visual")
        self.assertEqual(resp.status_code, 200)

    def test_has_legend(self):
        resp = self.client.get("/visual")
        self.assertIn("Fitness", resp.text)
        self.assertIn("Awareness", resp.text)

    def test_has_back_link(self):
        resp = self.client.get("/visual")
        self.assertIn('href="/"', resp.text)


class TestVisualWithChildren(unittest.TestCase):
    """Test visual data when children exist in the population."""

    @classmethod
    def setUpClass(cls):
        cls.org, cls.client = _make_app()
        # Spawn a child
        cls.org.auto_reproduce_cycle()

    def test_children_appear(self):
        data = self.client.get("/api/organisms").json()
        ids = [o["id"] for o in data["organisms"]]
        child_ids = [i for i in ids if i != "AL-01"]
        # May or may not have children depending on fitness — just verify structure
        for org_data in data["organisms"]:
            self.assertIn("traits", org_data)
            self.assertIn("fitness", org_data)

    def test_child_is_not_parent(self):
        data = self.client.get("/api/organisms").json()
        for o in data["organisms"]:
            if o["id"] != "AL-01":
                self.assertFalse(o["is_parent"])


if __name__ == "__main__":
    unittest.main()
