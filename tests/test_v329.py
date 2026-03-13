"""AL-01 v3.29 — Tests for restart-safe ecology.

Covers:
 1. RESTART_RECOVERY_CYCLES constant exists and is positive
 2. _below_fitness_cycles is persisted via persist()
 3. _last_birth_cycle is persisted via persist()
 4. _conservation_mode is persisted via persist()
 5. environment_state is persisted via persist()
 6. boot() restores _below_fitness_cycles from state
 7. boot() restores _last_birth_cycle from state
 8. boot() restores _conservation_mode from state
 9. boot() restores environment state from persisted data
10. boot() activates restart recovery window
11. is_restart_recovery is True after boot(), False after enough cycles
12. restart_recovery_remaining decrements each autonomy_cycle tick
13. Recovery window reaches zero
14. Founder death is skipped during restart recovery
15. Child fitness_floor death is skipped during restart recovery
16. Population prune_weakest (pre-cap) is skipped during restart recovery
17. Deaths resume normally after recovery window expires
18. Persisted below_fitness_cycles survive full save/load/boot cycle
19. Environment from_dict round-trip preserves critical fields
20. _restart_recovery_remaining is not persisted (always fresh on boot)
21. Empty dicts round-trip cleanly
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Any, Dict

from al01.database import Database
from al01.life_log import LifeLog
from al01.memory_manager import MemoryManager
from al01.organism import (
    Organism,
    RESTART_RECOVERY_CYCLES,
)
from al01.policy import PolicyManager
from al01.environment import Environment, EnvironmentConfig
from al01.genome import Genome
from al01.population import Population


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _make_organism(tmp: str) -> Organism:
    """Create a minimal Organism in a temp directory."""
    db = Database(db_path=os.path.join(tmp, "t.db"))
    mem = MemoryManager(data_dir=tmp, credential_path=None, database=db)
    log = LifeLog(data_dir=os.path.join(tmp, "data"), organism_id="AL-01")
    pol = PolicyManager(data_dir=os.path.join(tmp, "data"))
    org = Organism(data_dir=tmp, memory_manager=mem, life_log=log, policy=pol)
    return org


def _read_state_file(tmp: str) -> Dict[str, Any]:
    """Read the raw state.json from a temp data dir."""
    state_path = os.path.join(tmp, "state.json")
    with open(state_path, "r") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════
# 1. Constant
# ══════════════════════════════════════════════════════════════

class TestRestartRecoveryConstant(unittest.TestCase):
    def test_constant_exists_and_positive(self):
        self.assertIsInstance(RESTART_RECOVERY_CYCLES, int)
        self.assertGreater(RESTART_RECOVERY_CYCLES, 0)

    def test_constant_is_reasonable(self):
        """Recovery window should be between 10 and 100 cycles."""
        self.assertGreaterEqual(RESTART_RECOVERY_CYCLES, 10)
        self.assertLessEqual(RESTART_RECOVERY_CYCLES, 100)


# ══════════════════════════════════════════════════════════════
# 2–5. Serialization includes new fields (via persist)
# ══════════════════════════════════════════════════════════════

class TestStateSerialization(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.org = _make_organism(self._tmp)
        self.org.boot()

    def test_below_fitness_cycles_persisted(self):
        self.org._below_fitness_cycles = {"child-1": 3, "child-2": 7}
        self.org.persist(force=True)
        raw = _read_state_file(self._tmp)
        self.assertEqual(raw["below_fitness_cycles"]["child-1"], 3)
        self.assertEqual(raw["below_fitness_cycles"]["child-2"], 7)

    def test_last_birth_cycle_persisted(self):
        self.org._last_birth_cycle = {"AL-01": 500}
        self.org.persist(force=True)
        raw = _read_state_file(self._tmp)
        self.assertEqual(raw["last_birth_cycle"]["AL-01"], 500)

    def test_conservation_mode_persisted(self):
        self.org._conservation_mode = {"child-1": True, "child-2": False}
        self.org.persist(force=True)
        raw = _read_state_file(self._tmp)
        self.assertTrue(raw["conservation_mode"]["child-1"])
        self.assertFalse(raw["conservation_mode"]["child-2"])

    def test_environment_state_persisted(self):
        for _ in range(10):
            self.org._environment.tick()
        self.org.persist(force=True)
        raw = _read_state_file(self._tmp)
        self.assertIn("environment_state", raw)
        env = raw["environment_state"]
        self.assertIn("cycle", env)
        self.assertIn("resource_pool", env)
        self.assertIn("temperature", env)

    def test_empty_dicts_persist_cleanly(self):
        self.org._below_fitness_cycles = {}
        self.org._last_birth_cycle = {}
        self.org._conservation_mode = {}
        self.org.persist(force=True)
        raw = _read_state_file(self._tmp)
        self.assertEqual(raw["below_fitness_cycles"], {})
        self.assertEqual(raw["last_birth_cycle"], {})
        self.assertEqual(raw["conservation_mode"], {})


# ══════════════════════════════════════════════════════════════
# 6–10. Boot restores persisted state
# ══════════════════════════════════════════════════════════════

class TestBootRestoration(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.org = _make_organism(self._tmp)
        self.org.boot()

    def test_boot_restores_below_fitness_cycles(self):
        self.org._below_fitness_cycles = {"child-a": 5}
        self.org.persist(force=True)
        org2 = _make_organism(self._tmp)
        org2.boot()
        self.assertEqual(org2._below_fitness_cycles.get("child-a"), 5)

    def test_boot_restores_last_birth_cycle(self):
        self.org._last_birth_cycle = {"AL-01": 999}
        self.org.persist(force=True)
        org2 = _make_organism(self._tmp)
        org2.boot()
        self.assertEqual(org2._last_birth_cycle.get("AL-01"), 999)

    def test_boot_restores_conservation_mode(self):
        self.org._conservation_mode = {"child-x": True}
        self.org.persist(force=True)
        org2 = _make_organism(self._tmp)
        org2.boot()
        self.assertTrue(org2._conservation_mode.get("child-x"))

    def test_boot_restores_environment_state(self):
        for _ in range(50):
            self.org._environment.tick()
        env_cycle_before = self.org._environment._cycle
        self.org.persist(force=True)
        org2 = _make_organism(self._tmp)
        org2.boot()
        self.assertEqual(org2._environment._cycle, env_cycle_before)

    def test_boot_activates_recovery_window(self):
        self.assertEqual(self.org._restart_recovery_remaining, RESTART_RECOVERY_CYCLES)


# ══════════════════════════════════════════════════════════════
# 11–13. Recovery window lifecycle
# ══════════════════════════════════════════════════════════════

class TestRecoveryWindow(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.org = _make_organism(self._tmp)
        self.org.boot()

    def test_is_restart_recovery_true_after_boot(self):
        self.assertTrue(self.org.is_restart_recovery)

    def test_restart_recovery_remaining_matches_constant(self):
        self.assertEqual(self.org.restart_recovery_remaining, RESTART_RECOVERY_CYCLES)

    def test_recovery_decrements_each_cycle(self):
        initial = self.org.restart_recovery_remaining
        self.org.autonomy_cycle()
        self.assertEqual(self.org.restart_recovery_remaining, initial - 1)

    def test_recovery_reaches_zero(self):
        for _ in range(RESTART_RECOVERY_CYCLES):
            self.org.autonomy_cycle()
        self.assertFalse(self.org.is_restart_recovery)
        self.assertEqual(self.org.restart_recovery_remaining, 0)

    def test_recovery_does_not_go_negative(self):
        for _ in range(RESTART_RECOVERY_CYCLES + 5):
            self.org.autonomy_cycle()
        self.assertEqual(self.org.restart_recovery_remaining, 0)


# ══════════════════════════════════════════════════════════════
# 14–16. Death guards during recovery
# ══════════════════════════════════════════════════════════════

class TestDeathGuardsDuringRecovery(unittest.TestCase):
    """All death/pruning paths are suppressed during recovery window."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.org = _make_organism(self._tmp)
        self.org.boot()

    def test_founder_death_skipped_during_recovery(self):
        """autonomy_cycle should skip AL-01 death during recovery
        and return restart_recovery=True in the record."""
        self.assertTrue(self.org.is_restart_recovery)
        # Force very low fitness on founder
        self.org._genome.set_trait("resilience", 0.0)
        self.org._genome.set_trait("adaptability", 0.0)
        self.org._genome.set_trait("efficiency", 0.0)
        self.org._genome.set_trait("curiosity", 0.0)
        # Run autonomy cycle — fitness floor check should be guarded
        result = self.org.autonomy_cycle()
        # Founder should survive — no death
        self.assertFalse(result.get("organism_died", False))
        self.assertTrue(result.get("restart_recovery", False))

    def test_child_death_skipped_during_recovery(self):
        """child_autonomy_cycle should not kill any children during recovery."""
        self.assertTrue(self.org.is_restart_recovery)
        # Spawn a child with terrible fitness
        g = Genome(rng_seed=99)
        g.set_trait("resilience", 0.0)
        g.set_trait("adaptability", 0.0)
        g.set_trait("efficiency", 0.0)
        g.set_trait("curiosity", 0.0)
        child = self.org._population.spawn_child(g, parent_evolution=1)
        assert child is not None
        child_id = child["id"]
        # Set many cycles below fitness floor
        self.org._below_fitness_cycles[child_id] = 999
        # Run child cycle — no death should happen
        results = self.org.child_autonomy_cycle()
        deaths = [r for r in results if r.get("decision") == "death"]
        self.assertEqual(len(deaths), 0)
        # Child should still exist
        self.assertIn(child_id, self.org._population.member_ids)

    def test_prune_cap_skipped_during_recovery(self):
        """population_interact should skip pre-cap pruning during recovery."""
        self.assertTrue(self.org.is_restart_recovery)
        # Spawn children first, then set cap low
        g = Genome(rng_seed=42)
        for _ in range(8):
            self.org._population.spawn_child(g, parent_evolution=1)
        self.org._population.max_population = 3
        self.assertGreater(self.org._population.size, 3)
        pop_before = self.org._population.size
        # Run population_interact — no pruning during recovery
        self.org.population_interact()
        self.assertEqual(self.org._population.size, pop_before)


# ══════════════════════════════════════════════════════════════
# 17. Deaths resume after recovery
# ══════════════════════════════════════════════════════════════

class TestDeathResumesAfterRecovery(unittest.TestCase):
    def test_child_death_resumes_after_recovery(self):
        """After recovery expires, child fitness_floor death should work."""
        tmp = tempfile.mkdtemp()
        org = _make_organism(tmp)
        org.boot()
        # Exhaust recovery window
        for _ in range(RESTART_RECOVERY_CYCLES):
            org.autonomy_cycle()
        self.assertFalse(org.is_restart_recovery)
        # Ensure there's room to spawn (autonomy cycles may have filled pop)
        org._population.max_population = 100
        # Spawn child with very low fitness — zero ALL traits
        g = Genome(rng_seed=99)
        for k in list(g.traits.keys()):
            g.set_trait(k, 0.0)
        child = org._population.spawn_child(g, parent_evolution=1)
        assert child is not None
        child_id = child["id"]
        # Set energy to 0 so weighted fitness is below threshold
        org._population.update_member(child_id, {"energy": 0.0})
        # Pre-set grace counter well above threshold
        org._below_fitness_cycles[child_id] = 999
        # Run child cycle — death should be possible now
        results = org.child_autonomy_cycle()
        deaths = [r for r in results if r.get("decision") == "death"]
        self.assertGreater(len(deaths), 0, "Child should die after recovery ends")

    def test_founder_fitness_floor_active_after_recovery(self):
        """After recovery, founder death path should be reachable."""
        tmp = tempfile.mkdtemp()
        org = _make_organism(tmp)
        org.boot()
        # Exhaust recovery
        for _ in range(RESTART_RECOVERY_CYCLES):
            org.autonomy_cycle()
        self.assertFalse(org.is_restart_recovery)
        # Run one more cycle — should NOT have restart_recovery flag
        result = org.autonomy_cycle()
        self.assertFalse(result.get("restart_recovery", False))


# ══════════════════════════════════════════════════════════════
# 18. Full save/load/boot round-trip
# ══════════════════════════════════════════════════════════════

class TestSaveLoadRoundTrip(unittest.TestCase):
    def test_below_fitness_survives_round_trip(self):
        tmp = tempfile.mkdtemp()
        org1 = _make_organism(tmp)
        org1.boot()
        org1._below_fitness_cycles = {"child-a": 3, "child-b": 12, "AL-01": 0}
        org1._last_birth_cycle = {"AL-01": 250, "child-a": 800}
        org1._conservation_mode = {"child-a": True, "child-b": False}
        org1.persist(force=True)

        org2 = _make_organism(tmp)
        org2.boot()
        self.assertEqual(org2._below_fitness_cycles["child-a"], 3)
        self.assertEqual(org2._below_fitness_cycles["child-b"], 12)
        self.assertEqual(org2._below_fitness_cycles["AL-01"], 0)
        self.assertEqual(org2._last_birth_cycle["AL-01"], 250)
        self.assertTrue(org2._conservation_mode["child-a"])
        self.assertFalse(org2._conservation_mode["child-b"])


# ══════════════════════════════════════════════════════════════
# 19. Environment round-trip
# ══════════════════════════════════════════════════════════════

class TestEnvironmentRoundTrip(unittest.TestCase):
    def test_from_dict_preserves_fields(self):
        cfg = EnvironmentConfig(resource_pool_min_floor=50.0)
        env = Environment(config=cfg)
        for _ in range(100):
            env.tick()
        d = env.to_dict()
        env2 = Environment.from_dict(d, config=cfg)
        self.assertEqual(env2._cycle, env._cycle)
        self.assertAlmostEqual(env2._resource_pool, env._resource_pool, places=4)
        self.assertAlmostEqual(env2._temperature, env._temperature, places=4)
        self.assertAlmostEqual(env2._entropy_pressure, env._entropy_pressure, places=4)
        self.assertAlmostEqual(env2._resource_abundance, env._resource_abundance, places=4)
        self.assertAlmostEqual(env2._noise_level, env._noise_level, places=4)

    def test_environment_restored_in_organism(self):
        tmp = tempfile.mkdtemp()
        org = _make_organism(tmp)
        org.boot()
        for _ in range(30):
            org._environment.tick()
        cycle_before = org._environment._cycle
        pool_before = org._environment._resource_pool
        org.persist(force=True)

        org2 = _make_organism(tmp)
        org2.boot()
        self.assertEqual(org2._environment._cycle, cycle_before)
        self.assertAlmostEqual(org2._environment._resource_pool, pool_before, places=4)


# ══════════════════════════════════════════════════════════════
# 20. Recovery is always fresh (not persisted)
# ══════════════════════════════════════════════════════════════

class TestRecoveryAlwaysFresh(unittest.TestCase):
    def test_recovery_resets_on_every_boot(self):
        """Even if recovery was 0 at persist time, boot sets it to full."""
        tmp = tempfile.mkdtemp()
        org = _make_organism(tmp)
        org.boot()
        for _ in range(RESTART_RECOVERY_CYCLES):
            org.autonomy_cycle()
        self.assertEqual(org._restart_recovery_remaining, 0)
        org.persist(force=True)
        org2 = _make_organism(tmp)
        org2.boot()
        self.assertEqual(org2._restart_recovery_remaining, RESTART_RECOVERY_CYCLES)


# ══════════════════════════════════════════════════════════════
# 21. Empty dicts round-trip
# ══════════════════════════════════════════════════════════════

class TestEmptyRoundTrip(unittest.TestCase):
    def test_empty_dicts_survive_boot(self):
        tmp = tempfile.mkdtemp()
        org = _make_organism(tmp)
        org.boot()
        org._below_fitness_cycles = {}
        org._last_birth_cycle = {}
        org._conservation_mode = {}
        org.persist(force=True)

        org2 = _make_organism(tmp)
        org2.boot()
        self.assertEqual(org2._below_fitness_cycles, {})
        self.assertEqual(org2._last_birth_cycle, {})
        self.assertEqual(org2._conservation_mode, {})


if __name__ == "__main__":
    unittest.main()
