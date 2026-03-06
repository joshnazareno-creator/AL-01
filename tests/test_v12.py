"""AL-01 v1.2 — minimum viable test suite.

Tests
-----
1. record_interaction writes to SQLite correctly
2. search_memory returns structured results
3. API key blocks unauthorized requests
4. API key allows authorized requests
5. /health endpoint works without auth
6. Database restart recovery (interactions survive)
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

# Ensure package is importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from al01.database import Database
from al01.memory_manager import MemoryManager
from al01.organism import Organism, OrganismState, VERSION


class TestRecordInteraction(unittest.TestCase):
    """record_interaction writes correctly to SQLite."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self.db = Database(db_path=os.path.join(self._tmpdir, "test.db"))
        self.memory = MemoryManager(
            data_dir=self._tmpdir,
            credential_path=None,
            database=self.db,
        )
        self.organism = Organism(
            data_dir=self._tmpdir, memory_manager=self.memory
        )

    def test_interaction_stored_in_sqlite(self) -> None:
        self.organism.record_interaction(
            user_input="hello",
            response="hi there",
            mood="curious",
        )
        count = self.db.interaction_count()
        self.assertEqual(count, 1)

        rows = self.db.recent_interactions(n=1)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["user_input"], "hello")
        self.assertEqual(row["response"], "hi there")
        self.assertEqual(row["mood"], "curious")

    def test_multiple_interactions(self) -> None:
        for i in range(5):
            self.organism.record_interaction(
                user_input=f"msg-{i}",
                response=f"reply-{i}",
            )
        self.assertEqual(self.db.interaction_count(), 5)
        recent = self.db.recent_interactions(n=3)
        self.assertEqual(len(recent), 3)
        # oldest-first order
        self.assertEqual(recent[0]["user_input"], "msg-2")
        self.assertEqual(recent[-1]["user_input"], "msg-4")

    def test_interaction_increments_state_counter(self) -> None:
        self.organism.record_interaction(user_input="a", response="b")
        self.organism.record_interaction(user_input="c", response="d")
        state = dict(self.organism.state)
        self.assertEqual(state["interaction_count"], 2)


class TestSearchMemory(unittest.TestCase):
    """search_memory returns structured results."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self.db = Database(db_path=os.path.join(self._tmpdir, "test.db"))
        self.memory = MemoryManager(
            data_dir=self._tmpdir,
            credential_path=None,
            database=self.db,
        )
        self.organism = Organism(
            data_dir=self._tmpdir, memory_manager=self.memory
        )
        # Seed data
        self.organism.record_interaction(
            user_input="tell me about Python",
            response="Python is great",
            mood="happy",
        )
        self.organism.record_interaction(
            user_input="what is Rust?",
            response="Rust is a systems language",
            mood="neutral",
        )
        self.organism.record_interaction(
            user_input="Python async patterns",
            response="Use asyncio",
            mood="focused",
        )

    def test_keyword_search(self) -> None:
        results = self.organism.search_memory("Python")
        self.assertGreaterEqual(len(results), 2)
        # Structured result keys
        for r in results:
            self.assertIn("user_input", r)
            self.assertIn("response", r)
            self.assertIn("mood", r)

    def test_search_mood_field(self) -> None:
        results = self.organism.search_memory("happy")
        self.assertGreaterEqual(len(results), 1)

    def test_search_no_results(self) -> None:
        results = self.organism.search_memory("nonexistent-xyz-999")
        self.assertEqual(len(results), 0)


class TestDatabasePersistence(unittest.TestCase):
    """Interactions survive process restart (simulated by creating new Organism)."""

    def test_restart_recovery(self) -> None:
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test.db")

        # First "session"
        db1 = Database(db_path=db_path)
        mem1 = MemoryManager(data_dir=tmpdir, credential_path=None, database=db1)
        org1 = Organism(data_dir=tmpdir, memory_manager=mem1)
        org1.record_interaction(user_input="before restart", response="noted")
        org1.persist()

        # Second "session" (new instances, same DB)
        db2 = Database(db_path=db_path)
        mem2 = MemoryManager(data_dir=tmpdir, credential_path=None, database=db2)
        org2 = Organism(data_dir=tmpdir, memory_manager=mem2)

        # Interaction count recovered
        state = dict(org2.state)
        self.assertGreaterEqual(state["interaction_count"], 1)

        # Interaction still in DB
        rows = db2.recent_interactions(n=10)
        inputs = [r["user_input"] for r in rows]
        self.assertIn("before restart", inputs)


class TestAPIKeyAuth(unittest.TestCase):
    """API key blocks unauthorized requests and allows authorized ones."""

    def setUp(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi[test] not installed")

        self._tmpdir = tempfile.mkdtemp()
        self.db = Database(db_path=os.path.join(self._tmpdir, "test.db"))
        self.memory = MemoryManager(
            data_dir=self._tmpdir,
            credential_path=None,
            database=self.db,
        )
        self.organism = Organism(
            data_dir=self._tmpdir, memory_manager=self.memory
        )

        from al01.api import create_app

        self.api_key = "test-key-abc123"
        self.app = create_app(self.organism, api_key=self.api_key)
        self.client = TestClient(self.app)

    def test_health_no_auth_required(self) -> None:
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")

    def test_status_blocked_without_key(self) -> None:
        resp = self.client.get("/status")
        self.assertEqual(resp.status_code, 401)

    def test_status_allowed_with_key(self) -> None:
        resp = self.client.get("/status", headers={"X-API-Key": self.api_key})
        self.assertEqual(resp.status_code, 200)

    def test_interact_blocked_without_key(self) -> None:
        resp = self.client.post(
            "/interact",
            json={"user_input": "hi", "response": "hello"},
        )
        self.assertEqual(resp.status_code, 401)

    def test_interact_allowed_with_key(self) -> None:
        resp = self.client.post(
            "/interact",
            json={"user_input": "hi", "response": "hello"},
            headers={"X-API-Key": self.api_key},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "recorded")

    def test_bad_input_returns_422(self) -> None:
        """Empty user_input should fail Pydantic validation."""
        resp = self.client.post(
            "/interact",
            json={"user_input": "", "response": "hello"},
            headers={"X-API-Key": self.api_key},
        )
        self.assertEqual(resp.status_code, 422)


if __name__ == "__main__":
    unittest.main()
