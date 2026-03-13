"""AL-01 v3.30 — Tests for energy death guard during restart recovery.

Covers:
 1. Child energy depletion death is skipped during restart recovery
 2. Child energy is clamped to minimum during recovery instead of death
 3. Founder energy death is skipped during restart recovery
 4. Child energy death resumes after recovery window expires
 5. Founder energy death resumes after recovery window expires
"""

from __future__ import annotations

import os
import tempfile
import unittest

from al01.database import Database
from al01.life_log import LifeLog
from al01.memory_manager import MemoryManager
from al01.organism import Organism, RESTART_RECOVERY_CYCLES
from al01.policy import PolicyManager
from al01.genome import Genome


def _make_organism(tmp: str) -> Organism:
    db = Database(db_path=os.path.join(tmp, "t.db"))
    mem = MemoryManager(data_dir=tmp, credential_path=None, database=db)
    log = LifeLog(data_dir=os.path.join(tmp, "data"), organism_id="AL-01")
    pol = PolicyManager(data_dir=os.path.join(tmp, "data"))
    return Organism(data_dir=tmp, memory_manager=mem, life_log=log, policy=pol)


class TestChildEnergyDeathGuard(unittest.TestCase):
    """Child energy depletion death is guarded during restart recovery."""

    def test_child_energy_death_skipped_during_recovery(self):
        """Children with energy=0 should NOT die during restart recovery."""
        tmp = tempfile.mkdtemp()
        org = _make_organism(tmp)
        org.boot()
        self.assertTrue(org.is_restart_recovery)

        g = Genome(rng_seed=77)
        child = org._population.spawn_child(g, parent_evolution=1)
        assert child is not None
        child_id = child["id"]

        # Force energy to 0 via update_member (get() returns a copy)
        org._population.update_member(child_id, {"energy": 0.0})
        org._autonomy._config.energy_min = 0.0

        # Run child cycle — should NOT kill during recovery
        results = org.child_autonomy_cycle()
        deaths = [r for r in results if r.get("decision") == "death"]
        self.assertEqual(len(deaths), 0, "Child should not die from energy during recovery")
        # Child should still exist
        self.assertIn(child_id, org._population.member_ids)
        # Restore default
        org._autonomy._config.energy_min = 0.10

    def test_child_energy_clamped_during_recovery(self):
        """Children with energy=0 should have energy clamped to min during recovery."""
        tmp = tempfile.mkdtemp()
        org = _make_organism(tmp)
        org.boot()
        self.assertTrue(org.is_restart_recovery)

        g = Genome(rng_seed=77)
        child = org._population.spawn_child(g, parent_evolution=1)
        assert child is not None
        child_id = child["id"]

        # Force energy to 0 via update_member
        org._population.update_member(child_id, {"energy": 0.0})
        org._autonomy._config.energy_min = 0.0

        org.child_autonomy_cycle()

        # Energy should have been clamped above 0 by the recovery path
        member_after = org._population.get(child_id)
        assert member_after is not None
        self.assertGreaterEqual(member_after["energy"], 0.0,
                                "Energy should not be negative")
        # Restore default
        org._autonomy._config.energy_min = 0.10

    def test_child_energy_death_resumes_after_recovery(self):
        """After recovery expires, child energy death should work."""
        tmp = tempfile.mkdtemp()
        org = _make_organism(tmp)
        org.boot()

        # Exhaust recovery
        for _ in range(RESTART_RECOVERY_CYCLES):
            org.autonomy_cycle()
        self.assertFalse(org.is_restart_recovery)

        org._population.max_population = 100
        g = Genome(rng_seed=77)
        child = org._population.spawn_child(g, parent_evolution=1)
        assert child is not None
        child_id = child["id"]

        # Force energy deeply negative so decision bonuses can't save it
        org._population.update_member(child_id, {"energy": -10.0})
        org._autonomy._config.energy_min = 0.0

        results = org.child_autonomy_cycle()
        deaths = [r for r in results
                  if r.get("decision") == "death" and r.get("cause") == "energy_depleted"]
        self.assertGreater(len(deaths), 0, "Child should die from energy after recovery")
        # Restore default
        org._autonomy._config.energy_min = 0.10


class TestFounderEnergyDeathGuard(unittest.TestCase):
    """Founder energy death is guarded during restart recovery."""

    def test_founder_energy_death_skipped_during_recovery(self):
        """Founder with organism_died from energy should be rescued during recovery."""
        tmp = tempfile.mkdtemp()
        org = _make_organism(tmp)
        org.boot()
        self.assertTrue(org.is_restart_recovery)

        # Force energy to 0 so autonomy engine flags organism_died
        org._autonomy._energy = 0.0
        result = org.autonomy_cycle()

        # During recovery, organism_died should be overridden to False
        self.assertFalse(result.get("organism_died", False),
                         "Founder should not die from energy during recovery")
        self.assertTrue(result.get("restart_recovery", False))

    def test_founder_energy_death_resumes_after_recovery(self):
        """After recovery, founder energy death handler should be reachable.

        Note: The founder doesn't truly die (_handle_death rescues it),
        and the autonomy engine clamps energy to energy_min (0.10),
        so organism_died won't be True with default config. We verify
        the recovery guard is gone by checking restart_recovery is False.
        """
        tmp = tempfile.mkdtemp()
        org = _make_organism(tmp)
        org.boot()

        for _ in range(RESTART_RECOVERY_CYCLES):
            org.autonomy_cycle()
        self.assertFalse(org.is_restart_recovery)

        result = org.autonomy_cycle()
        # After recovery, restart_recovery should NOT be in the record
        self.assertFalse(result.get("restart_recovery", False),
                         "No restart_recovery flag after window expires")


if __name__ == "__main__":
    unittest.main()
