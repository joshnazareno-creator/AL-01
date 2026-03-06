"""v3.17 — Reproduction Safety: 500-cycle birth cooldown, event
idempotency, tick-ordering (death before reproduction), alive re-check
at spawn, and concurrency guard.

Acceptance tests:
 1. Dead parent cannot spawn even if reproduction chance is 100%.
 2. Alive parent spawns at most 1 child per BIRTH_COOLDOWN_CYCLES (500).
 3. Calling reproduction twice in same tick for same parent spawns only 1 child.
 4. Running two ticks concurrently does not duplicate children (lock + unique ids).
"""

import logging
import os
import tempfile
import threading

import pytest

from al01.environment import Environment, EnvironmentConfig
from al01.population import Population, ABSOLUTE_POPULATION_CAP, BIRTH_COOLDOWN_CYCLES
from al01.genome import Genome
from al01.organism import Organism, MetabolismScheduler, MetabolismConfig
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
    saved_tick = pop._current_tick
    for i in range(n):
        parent["last_birth_tick"] = -BIRTH_COOLDOWN_CYCLES * 2
        pop.set_tick(i)  # unique tick for idempotency
        pop.spawn_child(genome, parent_evolution=0, parent_id="AL-01")
    parent["last_birth_tick"] = -BIRTH_COOLDOWN_CYCLES
    pop.set_tick(saved_tick)


def _make_organism(tmp):
    db = Database(db_path=os.path.join(tmp, "t.db"))
    mem = MemoryManager(data_dir=tmp, credential_path=None, database=db)
    log = LifeLog(data_dir=os.path.join(tmp, "data"), organism_id="AL-01")
    pol = PolicyManager(data_dir=os.path.join(tmp, "data"))
    org = Organism(data_dir=tmp, memory_manager=mem, life_log=log, policy=pol)
    return org


def _cleanup_handlers():
    for h in logging.getLogger().handlers[:]:
        if isinstance(h, logging.FileHandler):
            h.close()
            logging.getLogger().removeHandler(h)


# ═══════════════════════════════════════════════════════════════════════
# 1. Dead parent cannot spawn
# ═══════════════════════════════════════════════════════════════════════

class TestDeadParentCannotSpawn:
    """Dead parent must never reproduce, regardless of pathway."""

    def test_dead_parent_blocked_by_spawn_child(self):
        """spawn_child refuses when parent is dead (alive=False)."""
        tmp = tempfile.mkdtemp()
        try:
            pop = _pop(tmp)
            parent = pop.get("AL-01")
            assert parent is not None
            genome = Genome.from_dict(parent["genome"])
            _spawn_children(pop, 3)
            child_ids = [mid for mid in pop.member_ids if mid != "AL-01"]
            target = child_ids[0]

            # Kill the child
            pop.remove_member(target, cause="test_kill")
            rec = pop.get(target)
            assert rec is not None
            assert rec["alive"] is False

            # Attempt spawn from dead parent — must fail
            pop.set_tick(999)
            rec2 = pop.get(target)
            assert rec2 is not None
            child_genome = Genome.from_dict(rec2["genome"])
            result = pop.spawn_child(child_genome, parent_evolution=0, parent_id=target)
            assert result is None
        finally:
            _cleanup_handlers()

    def test_dead_parent_blocked_by_auto_reproduce(self):
        """auto_reproduce refuses when parent is dead."""
        tmp = tempfile.mkdtemp()
        try:
            pop = _pop(tmp)
            _spawn_children(pop, 3)
            child_ids = [mid for mid in pop.member_ids if mid != "AL-01"]
            target = child_ids[0]

            # Set high fitness + consecutive_above_repro
            pop._members[target]["consecutive_above_repro"] = 10
            rec = pop.get(target)
            assert rec is not None
            g = Genome.from_dict(rec["genome"])
            for trait in g.traits:
                g.set_trait(trait, 0.95)
            pop.update_member(target, {"genome": g.to_dict()})

            # Kill
            pop.remove_member(target, cause="test_kill")
            rec2 = pop.get(target)
            assert rec2 is not None
            assert rec2["alive"] is False

            result = pop.auto_reproduce(target)
            assert result is None
        finally:
            _cleanup_handlers()

    def test_dormant_parent_blocked_by_spawn_child(self):
        """spawn_child refuses when parent is dormant."""
        tmp = tempfile.mkdtemp()
        try:
            pop = _pop(tmp)
            _spawn_children(pop, 3)
            child_ids = [mid for mid in pop.member_ids if mid != "AL-01"]
            target = child_ids[0]

            pop.enter_dormant(target, cause="test")
            rec = pop.get(target)
            assert rec is not None
            assert rec["state"] == "dormant"

            pop.set_tick(999)
            rec2 = pop.get(target)
            assert rec2 is not None
            child_genome = Genome.from_dict(rec2["genome"])
            result = pop.spawn_child(child_genome, parent_evolution=0, parent_id=target)
            assert result is None
        finally:
            _cleanup_handlers()

    def test_dead_parent_100pct_repro_still_blocked(self):
        """Even with 100% repro probability, dead parent cannot spawn."""
        tmp = tempfile.mkdtemp()
        try:
            org = _make_organism(tmp)
            org.boot()
            _spawn_children(org._population, 10)
            child_ids = [mid for mid in org._population.member_ids if mid != "AL-01"]
            target = child_ids[0]

            # Make target high fitness and eligible
            org._population._members[target]["consecutive_above_repro"] = 10
            rec = org._population.get(target)
            assert rec is not None
            g = Genome.from_dict(rec["genome"])
            for trait in g.traits:
                g.set_trait(trait, 0.99)
            org._population.update_member(target, {"genome": g.to_dict()})

            # Kill via death handler (dormant first, then dead)
            org._population.hardcore_extinction_mode = True
            org._handle_death(target, "test")
            org._handle_death(target, "test")
            rec2 = org._population.get(target)
            assert rec2 is not None
            assert rec2["alive"] is False

            # Try auto_reproduce — must return None
            result = org._population.auto_reproduce(target)
            assert result is None
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 2. Max 1 child per parent per BIRTH_COOLDOWN_CYCLES (birth lock)
# ═══════════════════════════════════════════════════════════════════════

class TestBirthLock:
    """Alive parent spawns at most 1 child per 500 cycles."""

    def test_one_child_per_tick(self):
        """Second spawn_child call on same parent in same tick is blocked."""
        tmp = tempfile.mkdtemp()
        try:
            pop = _pop(tmp)
            parent = pop.get("AL-01")
            assert parent is not None
            genome = Genome.from_dict(parent["genome"])

            pop.set_tick(10)
            first = pop.spawn_child(genome, parent_evolution=0, parent_id="AL-01")
            assert first is not None

            # Second spawn on same tick — must be blocked
            second = pop.spawn_child(genome, parent_evolution=0, parent_id="AL-01")
            assert second is None
        finally:
            _cleanup_handlers()

    def test_new_tick_allows_new_child(self):
        """After advancing past cooldown, parent can spawn again."""
        tmp = tempfile.mkdtemp()
        try:
            pop = _pop(tmp)
            parent = pop.get("AL-01")
            assert parent is not None
            genome = Genome.from_dict(parent["genome"])

            pop.set_tick(10)
            first = pop.spawn_child(genome, parent_evolution=0, parent_id="AL-01")
            assert first is not None

            pop.set_tick(510)  # 500-cycle cooldown
            second = pop.spawn_child(genome, parent_evolution=0, parent_id="AL-01")
            assert second is not None
        finally:
            _cleanup_handlers()

    def test_different_parents_same_tick(self):
        """Different parents can each spawn 1 child on the same tick."""
        tmp = tempfile.mkdtemp()
        try:
            pop = _pop(tmp)
            _spawn_children(pop, 2)
            child_ids = [mid for mid in pop.member_ids if mid != "AL-01"]
            assert len(child_ids) == 2

            pop.set_tick(20000)  # well past any cooldown
            m0 = pop.get(child_ids[0])
            m1 = pop.get(child_ids[1])
            assert m0 is not None and m1 is not None
            g0 = Genome.from_dict(m0["genome"])
            g1 = Genome.from_dict(m1["genome"])

            c1 = pop.spawn_child(g0, parent_evolution=0, parent_id=child_ids[0])
            c2 = pop.spawn_child(g1, parent_evolution=0, parent_id=child_ids[1])
            assert c1 is not None
            assert c2 is not None
            assert c1["id"] != c2["id"]
        finally:
            _cleanup_handlers()

    def test_last_birth_tick_field_set(self):
        """After spawning, parent's last_birth_tick is updated."""
        tmp = tempfile.mkdtemp()
        try:
            pop = _pop(tmp)
            parent = pop.get("AL-01")
            assert parent is not None
            genome = Genome.from_dict(parent["genome"])

            pop.set_tick(42)
            pop.spawn_child(genome, parent_evolution=0, parent_id="AL-01")
            rec = pop.get("AL-01")
            assert rec is not None
            assert rec["last_birth_tick"] == 42
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 3. Event idempotency
# ═══════════════════════════════════════════════════════════════════════

class TestEventIdempotency:
    """Calling reproduction twice in same tick for same parent spawns only 1."""

    def test_double_spawn_same_tick_idempotent(self):
        """Two spawn_child calls with same parent+tick produce only 1 child."""
        tmp = tempfile.mkdtemp()
        try:
            pop = _pop(tmp)
            parent = pop.get("AL-01")
            assert parent is not None
            genome = Genome.from_dict(parent["genome"])

            pop.set_tick(50)
            first = pop.spawn_child(genome, parent_evolution=0, parent_id="AL-01")
            second = pop.spawn_child(genome, parent_evolution=0, parent_id="AL-01")

            assert first is not None
            assert second is None
            # Only 2 members total: AL-01 + 1 child
            assert pop.size == 2
        finally:
            _cleanup_handlers()

    def test_idempotency_cleared_on_new_tick(self):
        """Processed events are cleared when set_tick advances past cooldown."""
        tmp = tempfile.mkdtemp()
        try:
            pop = _pop(tmp)
            parent = pop.get("AL-01")
            assert parent is not None
            genome = Genome.from_dict(parent["genome"])

            pop.set_tick(50)
            pop.spawn_child(genome, parent_evolution=0, parent_id="AL-01")
            assert pop.size == 2

            pop.set_tick(550)  # 500-cycle cooldown
            pop.spawn_child(genome, parent_evolution=0, parent_id="AL-01")
            assert pop.size == 3
        finally:
            _cleanup_handlers()

    def test_auto_reproduce_cycle_idempotent(self):
        """Running auto_reproduce_cycle twice on the same tick spawns no extras."""
        tmp = tempfile.mkdtemp()
        try:
            org = _make_organism(tmp)
            org.boot()
            _spawn_children(org._population, 3)

            # Make all children eligible
            for mid in org._population.member_ids:
                if mid == "AL-01":
                    continue
                org._population._members[mid]["consecutive_above_repro"] = 10
                mrec = org._population.get(mid)
                assert mrec is not None
                g = Genome.from_dict(mrec["genome"])
                for trait in g.traits:
                    g.set_trait(trait, 0.95)
                org._population.update_member(mid, {"genome": g.to_dict()})

            org._population.set_tick(100)
            first_run = org.auto_reproduce_cycle()

            pop_after_first = org._population.size
            # Second run on SAME tick — should produce 0 new children
            second_run = org.auto_reproduce_cycle()
            assert len(second_run) == 0
            assert org._population.size == pop_after_first
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 4. Concurrency guard
# ═══════════════════════════════════════════════════════════════════════

class TestConcurrencyGuard:
    """Running two ticks concurrently does not duplicate children."""

    def test_tick_lock_prevents_overlap(self):
        """Acquiring _tick_lock blocks a second concurrent tick."""
        tmp = tempfile.mkdtemp()
        try:
            org = _make_organism(tmp)
            org.boot()

            # Manually hold the tick lock
            assert org._tick_lock.acquire(blocking=False) is True
            # Second acquire should fail (non-blocking)
            assert org._tick_lock.acquire(blocking=False) is False
            org._tick_lock.release()
        finally:
            _cleanup_handlers()

    def test_concurrent_ticks_unique_children(self):
        """Two threads calling tick() produce unique child IDs (no duplicates)."""
        tmp = tempfile.mkdtemp()
        try:
            org = _make_organism(tmp)
            org.boot()
            _spawn_children(org._population, 3)

            # Make children eligible for reproduction
            for mid in org._population.member_ids:
                if mid == "AL-01":
                    continue
                org._population._members[mid]["consecutive_above_repro"] = 10
                mrec = org._population.get(mid)
                assert mrec is not None
                g = Genome.from_dict(mrec["genome"])
                for trait in g.traits:
                    g.set_trait(trait, 0.95)
                org._population.update_member(mid, {"genome": g.to_dict()})

            pop_before = org._population.size
            barrier = threading.Barrier(2)
            errors = []

            def run_tick():
                try:
                    barrier.wait(timeout=5)
                    org.tick()
                except Exception as e:
                    errors.append(e)

            t1 = threading.Thread(target=run_tick)
            t2 = threading.Thread(target=run_tick)
            t1.start()
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)

            assert not errors, f"Errors in threads: {errors}"

            # All IDs must be unique
            all_ids = org._population.member_ids
            assert len(all_ids) == len(set(all_ids))
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 5. Tick ordering — death before reproduction
# ═══════════════════════════════════════════════════════════════════════

class TestTickOrdering:
    """Death resolution happens before reproduction in scheduler tick."""

    def test_scheduler_child_autonomy_before_auto_reproduce(self):
        """In MetabolismScheduler.tick(), child_autonomy (death resolution)
        runs before auto_reproduce when both fire on the same tick."""
        calls = []

        class FakeOrganism:
            _population = type("P", (), {"set_tick": lambda self, t: None})()

            def pulse(self, **kw): calls.append("pulse")
            def reflect(self): calls.append("reflect")
            def persist(self): calls.append("persist")
            def environment_tick(self): calls.append("environment_tick")
            def evolve_cycle(self): calls.append("evolve_cycle")
            def population_interact(self): calls.append("population_interact")
            def autonomy_cycle(self): calls.append("autonomy_cycle")
            def auto_reproduce_cycle(self): calls.append("auto_reproduce_cycle")
            def child_autonomy_cycle(self): calls.append("child_autonomy_cycle")
            def behavior_analysis_cycle(self): calls.append("behavior_analysis_cycle")
            def rare_reproduction_cycle(self): calls.append("rare_reproduction_cycle")
            def novelty_stagnation_check(self): calls.append("novelty_stagnation_check")

        # Use intervals that make all fire on tick 1
        cfg = MetabolismConfig(
            pulse_interval=1,
            reflect_interval=1,
            persist_interval=1,
            environment_interval=1,
            evolve_interval=1,
            population_interact_interval=1,
            autonomy_interval=1,
            auto_reproduce_interval=1,
            child_autonomy_interval=1,
            behavior_analysis_interval=1,
            rare_reproduce_interval=1,
        )
        sched = MetabolismScheduler(FakeOrganism(), cfg)  # type: ignore[arg-type]
        sched.tick()

        # child_autonomy_cycle must come BEFORE auto_reproduce_cycle
        ca_idx = calls.index("child_autonomy_cycle")
        ar_idx = calls.index("auto_reproduce_cycle")
        assert ca_idx < ar_idx, (
            f"child_autonomy_cycle (idx={ca_idx}) must precede "
            f"auto_reproduce_cycle (idx={ar_idx}): {calls}"
        )

        # Also before rare_reproduction_cycle
        rr_idx = calls.index("rare_reproduction_cycle")
        assert ca_idx < rr_idx


# ═══════════════════════════════════════════════════════════════════════
# 6. set_tick clears per-tick state
# ═══════════════════════════════════════════════════════════════════════

class TestSetTick:
    """Population.set_tick advances tick and clears processed events."""

    def test_set_tick_clears_events(self):
        tmp = tempfile.mkdtemp()
        try:
            pop = _pop(tmp)
            pop.set_tick(1)
            pop._processed_repro_events.add("1:AL-01")
            assert len(pop._processed_repro_events) == 1

            pop.set_tick(2)
            assert len(pop._processed_repro_events) == 0
        finally:
            _cleanup_handlers()

    def test_set_tick_same_tick_no_clear(self):
        """Calling set_tick with the same tick doesn't clear events."""
        tmp = tempfile.mkdtemp()
        try:
            pop = _pop(tmp)
            pop.set_tick(5)
            pop._processed_repro_events.add("5:AL-01")

            pop.set_tick(5)
            assert len(pop._processed_repro_events) == 1
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 7. Migration — last_birth_tick field
# ═══════════════════════════════════════════════════════════════════════

class TestMigration:
    """Existing members get last_birth_tick=-BIRTH_COOLDOWN_CYCLES via migration."""

    def test_migration_adds_last_birth_tick(self):
        tmp = tempfile.mkdtemp()
        try:
            pop = _pop(tmp)
            # Remove the field to simulate legacy data
            for mid in pop._members:
                pop._members[mid].pop("last_birth_tick", None)

            pop._migrate_members()
            for mid in pop._members:
                assert pop._members[mid]["last_birth_tick"] == -BIRTH_COOLDOWN_CYCLES
        finally:
            _cleanup_handlers()
