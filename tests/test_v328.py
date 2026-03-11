"""v3.28 — Permanent Child Death & Founder Protection tests.

Features:
 1. Children die permanently — no dormancy, no revival across any path
 2. AL-01 founder protection restored — emergency energy injection
 3. Dead-organism guard clauses on update_member / update_energy
 4. Startup validation: dead children in _members moved to graveyard
 5. rescue_from_graveyard only works for AL-01
 6. wake_dormant_cycle skips non-AL-01 organisms
 7. check_extinction_reseed does NOT wake dormant organisms
"""

import logging
import os
import tempfile

import pytest

from al01.environment import Environment, EnvironmentConfig
from al01.population import Population, BIRTH_COOLDOWN_CYCLES, LifecycleState
from al01.genome import Genome
from al01.organism import Organism
from al01.memory_manager import MemoryManager
from al01.database import Database
from al01.life_log import LifeLog
from al01.policy import PolicyManager


# ── Helpers ──────────────────────────────────────────────────────────────

def _env(pool_max=1000.0, pool_initial=1000.0, regen=5.0, **kw):
    cfg = EnvironmentConfig(
        resource_pool_max=pool_max,
        resource_pool_initial=pool_initial,
        resource_regen_rate=regen,
        scarcity_probability=0.0,
        shock_probability=0.0,
        **kw,
    )
    return Environment(config=cfg, rng_seed=42)


def _pop(tmp, max_pop=60, rng_seed=42):
    pop = Population(data_dir=tmp, parent_id="AL-01", rng_seed=rng_seed)
    pop.max_population = max_pop
    return pop


def _spawn_children(pop, n):
    parent = pop._members["AL-01"]
    genome = Genome.from_dict(parent["genome"])
    for i in range(n):
        parent["last_birth_tick"] = -BIRTH_COOLDOWN_CYCLES * 2
        pop.set_tick(i)
        pop.spawn_child(genome, parent_evolution=0, parent_id="AL-01")
    parent["last_birth_tick"] = -BIRTH_COOLDOWN_CYCLES
    pop.set_tick(0)


def _make_organism(tmp):
    db = Database(db_path=os.path.join(tmp, "t.db"))
    mem = MemoryManager(data_dir=tmp, credential_path=None, database=db)
    log = LifeLog(data_dir=os.path.join(tmp, "data"), organism_id="AL-01")
    pol = PolicyManager(data_dir=os.path.join(tmp, "data"))
    org = Organism(data_dir=tmp, memory_manager=mem, life_log=log, policy=pol)
    return org


def _cleanup_handlers():
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)


# ═══════════════════════════════════════════════════════════════════════
# 1. Child Permanent Death
# ═══════════════════════════════════════════════════════════════════════

class TestChildPermanentDeath:
    """v3.28: Children die permanently — skip dormancy, go straight to graveyard."""

    def test_child_death_skips_dormancy(self):
        """Child goes straight to graveyard, never enters dormant state."""
        tmp = tempfile.mkdtemp()
        try:
            org = _make_organism(tmp)
            org.boot()
            org._population.hardcore_extinction_mode = True
            _spawn_children(org._population, 3)
            child_ids = [m for m in org._population.member_ids if m != "AL-01"]
            target = child_ids[0]

            org._handle_death(target, "energy_depleted")

            # Child must NOT be in _members (not dormant)
            assert target not in org._population._members
            # Child must be in graveyard
            assert target in org._population._graveyard
            g = org._population._graveyard[target]
            assert g.get("lifecycle_state") == LifecycleState.DEAD
            assert g.get("alive") is False
        finally:
            _cleanup_handlers()

    def test_child_death_persists_across_save_load(self):
        """Dead child stays dead after population save/reload."""
        tmp = tempfile.mkdtemp()
        try:
            org = _make_organism(tmp)
            org.boot()
            org._population.hardcore_extinction_mode = True
            _spawn_children(org._population, 3)
            child_ids = [m for m in org._population.member_ids if m != "AL-01"]
            target = child_ids[0]

            org._handle_death(target, "fitness_floor")

            # Reload population from disk
            pop2 = Population(data_dir=tmp, parent_id="AL-01", rng_seed=42)
            pop2.max_population = 60

            # Dead child must NOT be in active members
            assert target not in pop2.member_ids
            assert target not in pop2._members
            # Dead child must be in graveyard
            assert target in pop2._graveyard
            assert pop2._graveyard[target].get("lifecycle_state") == LifecycleState.DEAD
        finally:
            _cleanup_handlers()

    def test_child_death_is_terminal_no_wake(self):
        """Dead child cannot be woken via wake_dormant."""
        tmp = tempfile.mkdtemp()
        try:
            org = _make_organism(tmp)
            org.boot()
            org._population.hardcore_extinction_mode = True
            _spawn_children(org._population, 3)
            child_ids = [m for m in org._population.member_ids if m != "AL-01"]
            target = child_ids[0]

            org._handle_death(target, "energy_depleted")

            # Attempt to wake — should fail since it's in graveyard, not dormant
            result = org._population.wake_dormant(target, energy_boost=0.5)
            assert result is None

            # Still in graveyard, not active
            assert target not in org._population._members
            assert target in org._population._graveyard
        finally:
            _cleanup_handlers()

    def test_multiple_children_die_permanently(self):
        """All children die permanently, population only has AL-01."""
        tmp = tempfile.mkdtemp()
        try:
            org = _make_organism(tmp)
            org.boot()
            org._population.hardcore_extinction_mode = True
            _spawn_children(org._population, 5)
            child_ids = [m for m in org._population.member_ids if m != "AL-01"]

            for cid in child_ids:
                org._handle_death(cid, "fitness_floor")

            # All original children should be in graveyard
            for cid in child_ids:
                assert cid in org._population._graveyard
                assert cid not in org._population._members
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 2. AL-01 Founder Protection
# ═══════════════════════════════════════════════════════════════════════

class TestFounderProtection:
    """v3.28: AL-01 gets founder protection — rescue instead of death."""

    def test_al01_survives_death(self):
        """AL-01 is rescued with energy injection instead of dying."""
        tmp = tempfile.mkdtemp()
        try:
            org = _make_organism(tmp)
            org.boot()

            org._handle_death("AL-01", "fitness_floor")
            member = org._population.get("AL-01")
            assert member is not None
            assert member.get("alive") is True
            assert member.get("lifecycle_state") != LifecycleState.DEAD
            assert org._founder_revival_count >= 1
            assert org._founder_recovery_mode is True
        finally:
            _cleanup_handlers()

    def test_al01_energy_restored_on_rescue(self):
        """AL-01 energy is set to FOUNDER_REVIVAL_ENERGY on rescue."""
        tmp = tempfile.mkdtemp()
        try:
            org = _make_organism(tmp)
            org.boot()
            # Drain energy
            org._population.update_energy("AL-01", 0.01)
            org._autonomy._energy = 0.01

            org._handle_death("AL-01", "energy_depleted")

            member = org._population.get("AL-01")
            assert member is not None, "AL-01 must survive founder rescue"
            assert member["energy"] >= 0.5  # FOUNDER_REVIVAL_ENERGY = 0.60
        finally:
            _cleanup_handlers()

    def test_al01_rescued_from_graveyard_on_boot(self):
        """AL-01 in graveyard is rescued to active members on boot."""
        tmp = tempfile.mkdtemp()
        try:
            org = _make_organism(tmp)
            org.boot()
            org._population.hardcore_extinction_mode = True

            # Manually kill AL-01 into graveyard (bypass founder protection
            # by using remove_member directly)
            org._population.remove_member("AL-01", cause="test_kill", death_cycle=0)
            assert "AL-01" not in org._population._members
            assert "AL-01" in org._population._graveyard

            # Create new organism using same data dir (simulates restart)
            org2 = _make_organism(tmp)
            org2.boot()

            # AL-01 should be rescued from graveyard
            member = org2._population.get("AL-01")
            assert member is not None
            assert member.get("alive") is True
        finally:
            _cleanup_handlers()

    def test_rescue_from_graveyard_rejects_children(self):
        """rescue_from_graveyard refuses to rescue non-AL-01 organisms."""
        tmp = tempfile.mkdtemp()
        try:
            pop = _pop(tmp)
            _spawn_children(pop, 3)
            child_ids = [m for m in pop.member_ids if m != "AL-01"]
            target = child_ids[0]

            # Kill the child
            pop.remove_member(target, cause="test_kill")
            assert target in pop._graveyard

            # Attempt rescue — must be rejected
            result = pop.rescue_from_graveyard(target)
            assert result is False
            assert target in pop._graveyard
            assert target not in pop._members
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 3. Dead-Organism Guard Clauses
# ═══════════════════════════════════════════════════════════════════════

class TestDeadOrganismGuards:
    """v3.28: update_member and update_energy reject dead organisms."""

    def test_update_member_rejects_dead(self):
        """update_member silently rejects updates for dead organisms."""
        tmp = tempfile.mkdtemp()
        try:
            pop = _pop(tmp)
            _spawn_children(pop, 1)
            child_ids = [m for m in pop.member_ids if m != "AL-01"]
            target = child_ids[0]

            # Manually set lifecycle to DEAD without removing from _members
            # (simulates a race condition or legacy data)
            pop._members[target]["lifecycle_state"] = LifecycleState.DEAD
            original_energy = pop._members[target]["energy"]

            pop.update_member(target, {"energy": 999.0})
            # Energy should NOT have changed
            assert pop._members[target]["energy"] == original_energy
        finally:
            _cleanup_handlers()

    def test_update_energy_rejects_dead(self):
        """update_energy silently rejects updates for dead organisms."""
        tmp = tempfile.mkdtemp()
        try:
            pop = _pop(tmp)
            _spawn_children(pop, 1)
            child_ids = [m for m in pop.member_ids if m != "AL-01"]
            target = child_ids[0]

            pop._members[target]["lifecycle_state"] = LifecycleState.DEAD
            original_energy = pop._members[target]["energy"]

            pop.update_energy(target, 999.0)
            assert pop._members[target]["energy"] == original_energy
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 4. Startup Validation
# ═══════════════════════════════════════════════════════════════════════

class TestStartupValidation:
    """v3.28: Dead children in _members are cleaned up on boot."""

    def test_dead_children_cleaned_on_boot(self):
        """Dead children found in _members on boot are moved to graveyard."""
        tmp = tempfile.mkdtemp()
        try:
            org = _make_organism(tmp)
            org.boot()
            _spawn_children(org._population, 3)
            child_ids = [m for m in org._population.member_ids if m != "AL-01"]
            target = child_ids[0]

            # Corrupt the data: mark child as DEAD but leave in _members
            org._population._members[target]["lifecycle_state"] = LifecycleState.DEAD
            org._population._members[target]["alive"] = False
            org._population._save()

            # Reboot
            org2 = _make_organism(tmp)
            org2.boot()

            # Dead child should have been moved to graveyard
            assert target not in org2._population._members
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 5. Extinction Recovery (no dormant waking)
# ═══════════════════════════════════════════════════════════════════════

class TestExtinctionRecoveryNoDormantWake:
    """v3.28: check_extinction_reseed no longer wakes dormant organisms."""

    def test_extinction_reseed_does_not_wake_dormant(self):
        """During extinction recovery, dormant organisms stay dormant."""
        tmp = tempfile.mkdtemp()
        try:
            org = _make_organism(tmp)
            org.boot()
            _spawn_children(org._population, 5)
            child_ids = [m for m in org._population.member_ids if m != "AL-01"]

            # Put first child dormant (manually — shouldn't happen in v3.28
            # but testing the guard)
            org._population.enter_dormant(child_ids[0], cause="test")
            assert org._population.dormant_count == 1

            # Kill others to trigger extinction threshold
            org._population.hardcore_extinction_mode = True
            for cid in child_ids[1:]:
                org._handle_death(cid, "test")

            # Run extinction recovery
            result = org.check_extinction_reseed()

            # The dormant child should NOT have been woken
            dormant_woken = result.get("dormant_woken", []) if result else []
            assert len(dormant_woken) == 0
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 6. Wake Dormant Cycle (AL-01 only)
# ═══════════════════════════════════════════════════════════════════════

class TestWakeDormantCycleAL01Only:
    """v3.28: wake_dormant_cycle only wakes AL-01, skips children."""

    def test_wake_dormant_skips_non_al01(self):
        """Children in dormant state are NOT woken by wake_dormant_cycle."""
        tmp = tempfile.mkdtemp()
        try:
            org = _make_organism(tmp)
            org.boot()
            _spawn_children(org._population, 3)
            child_ids = [m for m in org._population.member_ids if m != "AL-01"]

            # Manually dormant a child (shouldn't happen normally in v3.28)
            org._population.enter_dormant(child_ids[0], cause="test")
            assert org._population.dormant_count == 1

            # Set pool high enough to trigger waking
            org._environment._resource_pool = org._environment._config.resource_pool_max

            woke = org.wake_dormant_cycle()
            # No children should have been woken
            woke_ids = [w.get("organism_id") for w in woke]
            assert child_ids[0] not in woke_ids
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 7. Acceptance Test: Full lifecycle
# ═══════════════════════════════════════════════════════════════════════

class TestAcceptanceFullLifecycle:
    """v3.28 acceptance test: create child → death → save → reload → still dead."""

    def test_child_death_persists_full_cycle(self):
        """Create child, force death, save, reload, confirm still dead."""
        tmp = tempfile.mkdtemp()
        try:
            # Phase 1: Create and kill
            org = _make_organism(tmp)
            org.boot()
            org._population.hardcore_extinction_mode = True
            _spawn_children(org._population, 2)
            child_ids = [m for m in org._population.member_ids if m != "AL-01"]
            target = child_ids[0]
            surviving = child_ids[1]

            # Kill one child
            org._handle_death(target, "fitness_floor")
            assert target not in org._population._members
            assert target in org._population._graveyard

            # Phase 2: Reload from disk
            org2 = _make_organism(tmp)
            org2.boot()

            # Dead child stays dead
            assert target not in org2._population._members
            assert target not in org2._population.member_ids
            assert target in org2._population._graveyard

            # Surviving child stays alive
            assert surviving in org2._population._members

            # AL-01 stays alive
            al01 = org2._population.get("AL-01")
            assert al01 is not None
            assert al01.get("alive") is True
        finally:
            _cleanup_handlers()
