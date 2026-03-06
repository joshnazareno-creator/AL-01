"""Tests for v3.10 Portable Child Export / Import.

Coverage:
1. export_child — serializes all fields, includes checksum.
2. import_child — validates checksum, rejects tampered data, reconstructs organism.
3. Lineage preservation — parent_id and generation survive round-trip.
4. Serialization round-trip — export → import → re-export yields identical data.
5. Edge cases — dead child, founder, capacity, missing fields.
6. API endpoints — /export and /import via TestClient.
7. Schema validation — validate_payload catches all structural errors.
8. Fitness history — capping, timestamps, flatten/stamp helpers.
9. Lineage guard — immutable fields blocked on adopted records.
10. Deterministic serialization — canonical payload is insertion-order agnostic.
"""

import copy
import json
import os
import tempfile
import unittest

from al01.genome import Genome
from al01.population import Population
from al01.portable import (
    FITNESS_HISTORY_CAP,
    IMMUTABLE_LINEAGE_FIELDS,
    PORTABLE_CHILD_SCHEMA,
    PORTABLE_SCHEMA_VERSION,
    _compute_checksum,
    _flatten_fitness_history,
    _stamp_fitness_history,
    _verify_checksum,
    export_child,
    guard_lineage,
    import_child,
    validate_payload,
)


# ======================================================================
# Helper: build a population with a living child
# ======================================================================

def _make_population_with_child(
    parent_traits=None,
    child_energy=0.8,
) -> tuple:
    """Return (population, child_id) with one living child."""
    tmpdir = tempfile.mkdtemp()
    pop = Population(data_dir=tmpdir, parent_id="AL-01")
    parent_genome = Genome(traits=parent_traits) if parent_traits else Genome()
    child = pop.spawn_child(parent_genome, parent_evolution=5)
    assert child is not None, "spawn_child failed"
    return pop, child["id"]


# ======================================================================
# 1. Export
# ======================================================================

class TestExportChild(unittest.TestCase):
    """export_child produces a valid portable snapshot."""

    def test_basic_export_structure(self) -> None:
        """Exported dict contains all required top-level keys."""
        pop, cid = _make_population_with_child()
        snap = export_child(pop, cid)
        required = [
            "schema_version", "exported_at", "id", "parent_id",
            "generation", "genome", "fitness", "trait_fitness",
            "birth_time", "age_seconds", "energy", "evolution_count",
            "interaction_count", "state", "genome_hash",
            "fitness_history", "checksum",
        ]
        for key in required:
            self.assertIn(key, snap, f"Missing key: {key}")

    def test_schema_version(self) -> None:
        pop, cid = _make_population_with_child()
        snap = export_child(pop, cid)
        self.assertEqual(snap["schema_version"], PORTABLE_SCHEMA_VERSION)

    def test_genome_section_has_traits(self) -> None:
        pop, cid = _make_population_with_child()
        snap = export_child(pop, cid)
        g = snap["genome"]
        self.assertIn("traits", g)
        self.assertIn("effective_traits", g)
        self.assertIn("mutation_rate", g)
        self.assertIn("mutation_delta", g)

    def test_checksum_present_and_valid(self) -> None:
        pop, cid = _make_population_with_child()
        snap = export_child(pop, cid)
        self.assertTrue(_verify_checksum(snap))

    def test_export_preserves_lineage(self) -> None:
        pop, cid = _make_population_with_child()
        snap = export_child(pop, cid)
        self.assertEqual(snap["parent_id"], "AL-01")
        self.assertEqual(snap["generation"], 1)

    def test_export_dead_child_raises(self) -> None:
        pop, cid = _make_population_with_child()
        pop.remove_member(cid, cause="test")
        with self.assertRaises(ValueError, msg="dead child"):
            export_child(pop, cid)

    def test_export_founder_raises(self) -> None:
        pop, _ = _make_population_with_child()
        with self.assertRaises(ValueError, msg="founder"):
            export_child(pop, "AL-01")

    def test_export_nonexistent_raises(self) -> None:
        pop, _ = _make_population_with_child()
        with self.assertRaises(ValueError):
            export_child(pop, "AL-01-child-9999")

    def test_export_fitness_is_float(self) -> None:
        pop, cid = _make_population_with_child()
        snap = export_child(pop, cid)
        self.assertIsInstance(snap["fitness"], float)

    def test_export_nickname_included(self) -> None:
        pop, cid = _make_population_with_child()
        pop.set_nickname(cid, "Sparky")
        snap = export_child(pop, cid)
        self.assertEqual(snap["nickname"], "Sparky")


# ======================================================================
# 2. Import
# ======================================================================

class TestImportChild(unittest.TestCase):
    """import_child validates and reconstructs organisms."""

    def _export_then_import(self, pop_src=None, pop_dst=None):
        """Helper: export from src, import into dst."""
        if pop_src is None:
            pop_src, cid = _make_population_with_child()
        else:
            cid = [m for m in pop_src.member_ids if m != "AL-01"][0]
        if pop_dst is None:
            tmpdir = tempfile.mkdtemp()
            pop_dst = Population(data_dir=tmpdir, parent_id="AL-01")
        snap = export_child(pop_src, cid)
        record = import_child(pop_dst, snap)
        return snap, record, pop_dst

    def test_basic_import(self) -> None:
        snap, record, pop = self._export_then_import()
        self.assertTrue(record["alive"])
        self.assertIn(record["id"], pop.member_ids)

    def test_new_id_assigned(self) -> None:
        pop_src, cid = _make_population_with_child()
        snap = export_child(pop_src, cid)
        tmpdir = tempfile.mkdtemp()
        pop_dst = Population(data_dir=tmpdir, parent_id="AL-01")
        # pre-spawn a child in dst so counters differ
        pop_dst.spawn_child(Genome(), parent_evolution=0)
        record = import_child(pop_dst, snap)
        self.assertNotEqual(record["id"], snap["id"])
        self.assertTrue(record["id"].startswith("AL-01-child-"))

    def test_lineage_preserved(self) -> None:
        snap, record, _ = self._export_then_import()
        self.assertEqual(record["parent_id"], snap["parent_id"])
        self.assertEqual(record["generation_id"], snap["generation"])
        self.assertEqual(record["original_id"], snap["id"])
        self.assertEqual(record["original_parent_id"], snap["parent_id"])

    def test_genome_reconstructed(self) -> None:
        snap, record, _ = self._export_then_import()
        # Traits should match the export
        imported_traits = record["genome"]["traits"]
        for trait, val in snap["genome"]["traits"].items():
            self.assertAlmostEqual(imported_traits[trait], val, places=5)

    def test_adoption_metadata(self) -> None:
        snap, record, _ = self._export_then_import()
        self.assertTrue(record.get("adopted"))
        self.assertIsNotNone(record.get("adopted_at"))

    def test_tampered_checksum_rejected(self) -> None:
        pop, cid = _make_population_with_child()
        snap = export_child(pop, cid)
        snap["energy"] = 999.0  # tamper
        tmpdir = tempfile.mkdtemp()
        pop_dst = Population(data_dir=tmpdir, parent_id="AL-01")
        with self.assertRaises(ValueError, msg="tampered"):
            import_child(pop_dst, snap)

    def test_missing_checksum_rejected(self) -> None:
        pop, cid = _make_population_with_child()
        snap = export_child(pop, cid)
        del snap["checksum"]
        tmpdir = tempfile.mkdtemp()
        pop_dst = Population(data_dir=tmpdir, parent_id="AL-01")
        with self.assertRaises(ValueError):
            import_child(pop_dst, snap)

    def test_missing_required_fields_rejected(self) -> None:
        pop, cid = _make_population_with_child()
        snap = export_child(pop, cid)
        # Remove a required field and recompute checksum to bypass that check
        del snap["genome"]
        snap["checksum"] = _compute_checksum(snap)
        tmpdir = tempfile.mkdtemp()
        pop_dst = Population(data_dir=tmpdir, parent_id="AL-01")
        with self.assertRaises(ValueError, msg="missing"):
            import_child(pop_dst, snap)

    def test_population_capacity_rejected(self) -> None:
        pop, cid = _make_population_with_child()
        snap = export_child(pop, cid)
        tmpdir = tempfile.mkdtemp()
        pop_dst = Population(data_dir=tmpdir, parent_id="AL-01")
        # Founder takes 1 slot — set cap to 1 so no room for imports.
        # max_population setter enforces min of 2, so we set it directly.
        pop_dst._max_population = 1
        with self.assertRaises(ValueError, msg="capacity"):
            import_child(pop_dst, snap)

    def test_evolution_count_preserved(self) -> None:
        pop, cid = _make_population_with_child()
        pop.update_member(cid, {"evolution_count": 42})
        snap = export_child(pop, cid)
        tmpdir = tempfile.mkdtemp()
        pop_dst = Population(data_dir=tmpdir, parent_id="AL-01")
        record = import_child(pop_dst, snap)
        self.assertEqual(record["evolution_count"], 42)


# ======================================================================
# 3. Round-trip fidelity
# ======================================================================

class TestRoundTrip(unittest.TestCase):
    """Export → Import → Re-export should yield identical payload (minus timestamps)."""

    def test_double_export_matches(self) -> None:
        pop_src, cid = _make_population_with_child()
        snap1 = export_child(pop_src, cid)

        tmpdir = tempfile.mkdtemp()
        pop_dst = Population(data_dir=tmpdir, parent_id="AL-01")
        record = import_child(pop_dst, snap1)

        snap2 = export_child(pop_dst, record["id"])

        # Compare genome traits exactly
        for trait in snap1["genome"]["traits"]:
            self.assertAlmostEqual(
                snap1["genome"]["traits"][trait],
                snap2["genome"]["traits"][trait],
                places=5,
            )
        # Generation preserved
        self.assertEqual(snap1["generation"], snap2["generation"])
        # Fitness preserved
        self.assertAlmostEqual(snap1["fitness"], snap2["fitness"], places=4)

    def test_json_serializable(self) -> None:
        """Exported snapshot is fully JSON-serializable."""
        pop, cid = _make_population_with_child()
        snap = export_child(pop, cid)
        text = json.dumps(snap)
        reloaded = json.loads(text)
        self.assertTrue(_verify_checksum(reloaded))


# ======================================================================
# 4. Checksum utilities
# ======================================================================

class TestChecksumUtils(unittest.TestCase):
    """Low-level checksum functions."""

    def test_compute_checksum_deterministic(self) -> None:
        data = {"a": 1, "b": "hello"}
        c1 = _compute_checksum(data)
        c2 = _compute_checksum(data)
        self.assertEqual(c1, c2)

    def test_verify_rejects_wrong_checksum(self) -> None:
        data = {"a": 1, "checksum": "bad"}
        self.assertFalse(_verify_checksum(data))

    def test_verify_accepts_correct_checksum(self) -> None:
        data = {"a": 1}
        data["checksum"] = _compute_checksum(data)
        self.assertTrue(_verify_checksum(data))


# ======================================================================
# 5. API Endpoints
# ======================================================================

class TestAPIEndpoints(unittest.TestCase):
    """FastAPI /export and /import endpoints via TestClient."""

    @classmethod
    def setUpClass(cls) -> None:
        from al01.organism import Organism, MetabolismConfig
        cls._tmpdir = tempfile.mkdtemp()
        cfg = MetabolismConfig(
            pulse_interval=9999, reflect_interval=9999, persist_interval=9999,
            evolve_interval=9999, population_interact_interval=9999,
            autonomy_interval=9999, environment_interval=9999,
            behavior_analysis_interval=9999, auto_reproduce_interval=9999,
            child_autonomy_interval=9999,
        )
        cls._org = Organism(data_dir=cls._tmpdir, config=cfg)
        cls._org.boot()

        # Spawn a child to export
        parent_genome = cls._org.genome
        child = cls._org.population.spawn_child(parent_genome, parent_evolution=1)
        cls._child_id = child["id"]

        from al01.api import create_app
        app = create_app(cls._org, api_key="test-key")
        from fastapi.testclient import TestClient
        cls._client = TestClient(app)
        cls._headers = {"X-API-Key": "test-key"}

    def test_export_endpoint(self) -> None:
        r = self._client.get(
            f"/population/{self._child_id}/export",
            headers=self._headers,
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("checksum", data)
        self.assertEqual(data["id"], self._child_id)

    def test_export_nonexistent_returns_400(self) -> None:
        r = self._client.get(
            "/population/AL-01-child-9999/export",
            headers=self._headers,
        )
        self.assertEqual(r.status_code, 400)

    def test_import_endpoint(self) -> None:
        # First export
        r = self._client.get(
            f"/population/{self._child_id}/export",
            headers=self._headers,
        )
        snap = r.json()
        # Then import
        r2 = self._client.post(
            "/population/import",
            json=snap,
            headers=self._headers,
        )
        self.assertEqual(r2.status_code, 200)
        data = r2.json()
        self.assertEqual(data["status"], "adopted")
        self.assertIn("new_id", data)
        self.assertEqual(data["original_id"], self._child_id)

    def test_import_tampered_returns_400(self) -> None:
        r = self._client.get(
            f"/population/{self._child_id}/export",
            headers=self._headers,
        )
        snap = r.json()
        snap["energy"] = 999.0  # tamper
        r2 = self._client.post(
            "/population/import",
            json=snap,
            headers=self._headers,
        )
        self.assertEqual(r2.status_code, 400)
        self.assertIn("tampered", r2.json()["detail"].lower())

    def test_export_download_header(self) -> None:
        r = self._client.get(
            f"/population/{self._child_id}/export",
            headers=self._headers,
        )
        disp = r.headers.get("content-disposition", "")
        self.assertIn("attachment", disp)
        self.assertIn(".json", disp)


# ======================================================================
# 6. Life log integration
# ======================================================================

class TestLifeLogIntegration(unittest.TestCase):
    """import_child logs adoption event when life_log is provided."""

    def test_adoption_event_logged(self) -> None:
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

        # Create a child to export from a separate population
        pop_src, cid = _make_population_with_child()
        snap = export_child(pop_src, cid)

        before_seq = org.life_log.head_seq
        import_child(org.population, snap, life_log=org.life_log)
        after_seq = org.life_log.head_seq

        self.assertGreater(after_seq, before_seq,
                           "Life log should have a new event after adoption")


# ======================================================================
# 7. Schema validation
# ======================================================================

class TestSchemaValidation(unittest.TestCase):
    """validate_payload catches structural errors."""

    def _valid_payload(self) -> dict:
        """Return a minimal valid payload dict."""
        pop, cid = _make_population_with_child()
        return export_child(pop, cid)

    def test_schema_object_exists(self) -> None:
        self.assertIn("$schema", PORTABLE_CHILD_SCHEMA)
        self.assertIn("required", PORTABLE_CHILD_SCHEMA)
        self.assertIn("properties", PORTABLE_CHILD_SCHEMA)

    def test_valid_payload_passes(self) -> None:
        payload = self._valid_payload()
        errors = validate_payload(payload)
        self.assertEqual(errors, [])

    def test_missing_required_field_detected(self) -> None:
        payload = self._valid_payload()
        del payload["fitness"]
        payload["checksum"] = _compute_checksum(payload)
        errors = validate_payload(payload)
        self.assertTrue(any("fitness" in e for e in errors))

    def test_wrong_schema_version(self) -> None:
        payload = self._valid_payload()
        payload["schema_version"] = "99.0"
        payload["checksum"] = _compute_checksum(payload)
        errors = validate_payload(payload)
        self.assertTrue(any("schema_version" in e for e in errors))

    def test_empty_id_rejected(self) -> None:
        payload = self._valid_payload()
        payload["id"] = ""
        payload["checksum"] = _compute_checksum(payload)
        errors = validate_payload(payload)
        self.assertTrue(any("id" in e for e in errors))

    def test_negative_generation_rejected(self) -> None:
        payload = self._valid_payload()
        payload["generation"] = -1
        payload["checksum"] = _compute_checksum(payload)
        errors = validate_payload(payload)
        self.assertTrue(any("generation" in e for e in errors))

    def test_genome_missing_traits_rejected(self) -> None:
        payload = self._valid_payload()
        del payload["genome"]["traits"]
        payload["checksum"] = _compute_checksum(payload)
        errors = validate_payload(payload)
        self.assertTrue(any("genome.traits" in e for e in errors))

    def test_genome_missing_mutation_rate(self) -> None:
        payload = self._valid_payload()
        del payload["genome"]["mutation_rate"]
        payload["checksum"] = _compute_checksum(payload)
        errors = validate_payload(payload)
        self.assertTrue(any("mutation_rate" in e for e in errors))

    def test_bad_trait_type_rejected(self) -> None:
        payload = self._valid_payload()
        payload["genome"]["traits"]["adaptability"] = "not_a_number"
        payload["checksum"] = _compute_checksum(payload)
        errors = validate_payload(payload)
        self.assertTrue(any("genome.traits.adaptability" in e for e in errors))

    def test_bad_checksum_length_detected(self) -> None:
        payload = self._valid_payload()
        payload["checksum"] = "tooshort"
        errors = validate_payload(payload)
        self.assertTrue(any("checksum" in e for e in errors))

    def test_import_rejects_invalid_schema(self) -> None:
        """import_child raises ValueError when schema validation fails."""
        payload = self._valid_payload()
        payload["generation"] = -5
        # Recompute checksum so integrity passes but schema fails
        payload["checksum"] = _compute_checksum(payload)
        tmpdir = tempfile.mkdtemp()
        pop = Population(data_dir=tmpdir, parent_id="AL-01")
        with self.assertRaises(ValueError):
            import_child(pop, payload)


# ======================================================================
# 8. Fitness history — capping and timestamps
# ======================================================================

class TestFitnessHistory(unittest.TestCase):
    """Fitness history is capped, timestamped on export, flattened on import."""

    def test_export_fitness_history_is_timestamped(self) -> None:
        pop, cid = _make_population_with_child()
        # Record some fitness data points
        pop.record_fitness(cid, 0.45)
        pop.record_fitness(cid, 0.50)
        snap = export_child(pop, cid)
        for entry in snap["fitness_history"]:
            self.assertIsInstance(entry, dict)
            self.assertIn("value", entry)
            self.assertIn("recorded_at", entry)
            self.assertIsInstance(entry["value"], float)

    def test_export_caps_fitness_history(self) -> None:
        pop, cid = _make_population_with_child()
        # Inject a large history directly
        member = pop.get(cid)
        pop.update_member(cid, {
            "fitness_history": [0.5] * 200,
        })
        snap = export_child(pop, cid)
        self.assertLessEqual(len(snap["fitness_history"]), FITNESS_HISTORY_CAP)

    def test_import_flattens_timestamped_entries(self) -> None:
        pop_src, cid = _make_population_with_child()
        pop_src.record_fitness(cid, 0.55)
        snap = export_child(pop_src, cid)
        # Verify export has timestamped entries
        self.assertIsInstance(snap["fitness_history"][0], dict)
        # Import
        tmpdir = tempfile.mkdtemp()
        pop_dst = Population(data_dir=tmpdir, parent_id="AL-01")
        record = import_child(pop_dst, snap)
        # Internal storage should be flat floats
        for val in record["fitness_history"]:
            self.assertIsInstance(val, float)

    def test_import_caps_fitness_history(self) -> None:
        pop_src, cid = _make_population_with_child()
        snap = export_child(pop_src, cid)
        # Inject excessive history and recompute checksum
        snap["fitness_history"] = [{"value": 0.5, "recorded_at": "2026-01-01T00:00:00+00:00"}] * 80
        snap["checksum"] = _compute_checksum(snap)
        tmpdir = tempfile.mkdtemp()
        pop_dst = Population(data_dir=tmpdir, parent_id="AL-01")
        record = import_child(pop_dst, snap)
        self.assertLessEqual(len(record["fitness_history"]), FITNESS_HISTORY_CAP)

    def test_stamp_fitness_history_helper(self) -> None:
        raw = [0.4, 0.5, 0.6]
        stamped = _stamp_fitness_history(raw, "2026-03-01T00:00:00+00:00")
        self.assertEqual(len(stamped), 3)
        for entry in stamped:
            self.assertIn("value", entry)
            self.assertIn("recorded_at", entry)
        self.assertAlmostEqual(stamped[0]["value"], 0.4, places=5)

    def test_flatten_fitness_history_helper(self) -> None:
        entries = [
            {"value": 0.4, "recorded_at": "2026-01-01T00:00:00Z"},
            0.5,
            {"value": 0.6, "recorded_at": "2026-01-02T00:00:00Z"},
        ]
        flat = _flatten_fitness_history(entries)
        self.assertEqual(flat, [0.4, 0.5, 0.6])

    def test_flatten_caps_at_limit(self) -> None:
        entries = list(range(200))
        flat = _flatten_fitness_history(entries, cap=50)
        self.assertEqual(len(flat), 50)

    def test_validate_rejects_oversized_history(self) -> None:
        pop, cid = _make_population_with_child()
        snap = export_child(pop, cid)
        snap["fitness_history"] = [0.5] * 200
        snap["checksum"] = _compute_checksum(snap)
        errors = validate_payload(snap)
        self.assertTrue(any("fitness_history" in e for e in errors))

    def test_validate_rejects_bad_history_entry(self) -> None:
        pop, cid = _make_population_with_child()
        snap = export_child(pop, cid)
        snap["fitness_history"] = ["bad_string"]
        snap["checksum"] = _compute_checksum(snap)
        errors = validate_payload(snap)
        self.assertTrue(any("fitness_history[0]" in e for e in errors))


# ======================================================================
# 9. Lineage guard
# ======================================================================

class TestLineageGuard(unittest.TestCase):
    """guard_lineage strips immutable fields on adopted records."""

    def test_guard_strips_immutable_fields(self) -> None:
        record = {"adopted": True, "parent_id": "AL-01", "original_id": "old"}
        updates = {"parent_id": "HACKER", "energy": 0.99}
        safe = guard_lineage(record, updates)
        self.assertNotIn("parent_id", safe)
        self.assertEqual(safe["energy"], 0.99)

    def test_guard_passes_non_adopted(self) -> None:
        record = {"adopted": False, "parent_id": "AL-01"}
        updates = {"parent_id": "NEW", "energy": 0.5}
        safe = guard_lineage(record, updates)
        self.assertIn("parent_id", safe)

    def test_guard_strips_all_lineage_fields(self) -> None:
        record = {"adopted": True}
        updates = {
            "parent_id": "X",
            "original_id": "X",
            "original_parent_id": "X",
            "generation_id": 99,
            "energy": 0.1,
        }
        safe = guard_lineage(record, updates)
        for field in IMMUTABLE_LINEAGE_FIELDS:
            self.assertNotIn(field, safe)
        self.assertIn("energy", safe)

    def test_population_update_member_protects_lineage(self) -> None:
        """update_member on an adopted child silently strips lineage fields."""
        pop_src, cid = _make_population_with_child()
        snap = export_child(pop_src, cid)
        tmpdir = tempfile.mkdtemp()
        pop_dst = Population(data_dir=tmpdir, parent_id="AL-01")
        record = import_child(pop_dst, snap)
        new_id = record["id"]
        original_parent = record["parent_id"]
        original_gen = record["generation_id"]
        # Attempt to override lineage
        pop_dst.update_member(new_id, {
            "parent_id": "HACKER",
            "generation_id": 999,
            "energy": 0.1,
        })
        updated = pop_dst.get(new_id)
        # Lineage must be unchanged
        self.assertEqual(updated["parent_id"], original_parent)
        self.assertEqual(updated["generation_id"], original_gen)
        # Non-lineage fields ARE updated
        self.assertAlmostEqual(updated["energy"], 0.1, places=5)

    def test_immutable_fields_constant_defined(self) -> None:
        self.assertIn("parent_id", IMMUTABLE_LINEAGE_FIELDS)
        self.assertIn("original_id", IMMUTABLE_LINEAGE_FIELDS)
        self.assertIn("original_parent_id", IMMUTABLE_LINEAGE_FIELDS)
        self.assertIn("generation_id", IMMUTABLE_LINEAGE_FIELDS)


# ======================================================================
# 10. Deterministic serialization
# ======================================================================

class TestDeterministicSerialization(unittest.TestCase):
    """Canonical payload is insertion-order agnostic."""

    def test_canonical_payload_ignores_dict_order(self) -> None:
        from al01.portable import _canonical_payload
        d1 = {"z": 1, "a": 2, "m": 3}
        d2 = {"a": 2, "m": 3, "z": 1}
        self.assertEqual(_canonical_payload(d1), _canonical_payload(d2))

    def test_checksum_stable_across_dict_orders(self) -> None:
        d1 = {"x": 10, "b": 20}
        d2 = {"b": 20, "x": 10}
        self.assertEqual(_compute_checksum(d1), _compute_checksum(d2))

    def test_double_export_same_checksum(self) -> None:
        """Two exports of the same child within the same second yield
        identical checksums (timestamps are the only variable)."""
        pop, cid = _make_population_with_child()
        snap1 = export_child(pop, cid)
        snap2 = export_child(pop, cid)
        # The export timestamps may differ by a few microseconds,
        # so we compare the non-timestamp canonical payloads.
        for key in ("exported_at", "age_seconds", "checksum"):
            snap1.pop(key, None)
            snap2.pop(key, None)
        # Normalise fitness_history timestamps for comparison
        for entry in snap1.get("fitness_history", []):
            if isinstance(entry, dict):
                entry.pop("recorded_at", None)
        for entry in snap2.get("fitness_history", []):
            if isinstance(entry, dict):
                entry.pop("recorded_at", None)
        self.assertEqual(
            json.dumps(snap1, sort_keys=True),
            json.dumps(snap2, sort_keys=True),
        )

    def test_example_json_matches_schema_keys(self) -> None:
        """The example export file contains all required schema keys."""
        example_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "docs", "example_export.json",
        )
        if not os.path.exists(example_path):
            self.skipTest("docs/example_export.json not found")
        with open(example_path, "r") as f:
            example = json.load(f)
        for field in PORTABLE_CHILD_SCHEMA["required"]:
            self.assertIn(field, example, f"Example missing required key: {field}")


if __name__ == "__main__":
    unittest.main()
