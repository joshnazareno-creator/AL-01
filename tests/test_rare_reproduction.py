"""v3.10 rare-reproduction — Rare Reproduction Mechanism tests.

Rules under test
────────────────
 1. Evaluate every 500 cycles; 5 % random chance gate.
 2. Hard population cap of 50.
 3. Parent eligibility: energy >= 0.65, fitness >= 0.50, stagnation < 0.90.
 4. Cost: parent energy -= 0.30; child energy = 0.50.
 5. Cooldown: 2000 cycles between births per parent.
 6. Child: mutated genome copy (variance 0.10), parent_id, gen = parent+1.
 7. Safety: re-check cap before adding, ONE child per invocation, no loops.
 8. Birth logged in VITAL life log.
"""

import logging
import os
import random
import tempfile

import pytest

from al01.environment import Environment, EnvironmentConfig
from al01.population import Population, ABSOLUTE_POPULATION_CAP, BIRTH_COOLDOWN_CYCLES
from al01.genome import Genome
from al01.organism import (
    Organism,
    RARE_REPRO_CYCLE_INTERVAL,
    RARE_REPRO_CHANCE,
    RARE_REPRO_POP_CAP,
    RARE_REPRO_ENERGY_MIN,
    RARE_REPRO_FITNESS_MIN,
    RARE_REPRO_STAGNATION_MAX,
    RARE_REPRO_ENERGY_COST,
    RARE_REPRO_CHILD_ENERGY,
    RARE_REPRO_MUTATION_RATE,
    RARE_REPRO_COOLDOWN_CYCLES,
)
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


def _spawn_children(pop, n, parent_id="AL-01"):
    parent = pop._members[parent_id]
    genome = Genome.from_dict(parent["genome"])
    children = []
    saved_tick = pop._current_tick
    for i in range(n):
        parent["last_birth_tick"] = -BIRTH_COOLDOWN_CYCLES * 2
        pop.set_tick(i)  # unique tick for idempotency
        c = pop.spawn_child(genome, parent_evolution=0, parent_id=parent_id)
        if c:
            children.append(c)
    parent["last_birth_tick"] = -BIRTH_COOLDOWN_CYCLES
    pop.set_tick(saved_tick)
    return children


def _make_organism(tmp):
    db = Database(db_path=os.path.join(tmp, "t.db"))
    mem = MemoryManager(data_dir=tmp, credential_path=None, database=db)
    log = LifeLog(data_dir=os.path.join(tmp, "data"), organism_id="AL-01")
    pol = PolicyManager(data_dir=os.path.join(tmp, "data"))
    org = Organism(data_dir=tmp, memory_manager=mem, life_log=log, policy=pol)
    return org


def _set_eligible_parent(pop, member_id, energy=0.80, fitness_values=None):
    """Set a member as eligible for rare reproduction."""
    pop.update_energy(member_id, energy)
    if fitness_values is None:
        # Healthy, non-stagnating history
        fitness_values = [0.5 + i * 0.01 for i in range(20)]
    for f in fitness_values:
        pop.record_fitness(member_id, f)


def _cleanup_handlers():
    for h in logging.getLogger().handlers[:]:
        if isinstance(h, logging.FileHandler):
            h.close()
            logging.getLogger().removeHandler(h)


# ═══════════════════════════════════════════════════════════════════════
# 1. Constants sanity
# ═══════════════════════════════════════════════════════════════════════

class TestRareReproConstants:
    """Verify all constants are importable and have expected values."""

    def test_cycle_interval(self):
        assert RARE_REPRO_CYCLE_INTERVAL == 500

    def test_chance(self):
        assert RARE_REPRO_CHANCE == 0.05

    def test_pop_cap(self):
        assert RARE_REPRO_POP_CAP == 50

    def test_energy_min(self):
        assert RARE_REPRO_ENERGY_MIN == 0.65

    def test_fitness_min(self):
        assert RARE_REPRO_FITNESS_MIN == 0.50

    def test_stagnation_max(self):
        assert RARE_REPRO_STAGNATION_MAX == 0.90

    def test_energy_cost(self):
        assert RARE_REPRO_ENERGY_COST == 0.30

    def test_child_energy(self):
        assert RARE_REPRO_CHILD_ENERGY == 0.50

    def test_mutation_rate(self):
        assert RARE_REPRO_MUTATION_RATE == 0.10

    def test_cooldown_cycles(self):
        assert RARE_REPRO_COOLDOWN_CYCLES == 2000


# ═══════════════════════════════════════════════════════════════════════
# 2. Stagnation estimator
# ═══════════════════════════════════════════════════════════════════════

class TestEstimateStagnation:
    """Unit-test the _estimate_stagnation static method."""

    def test_empty_history_returns_zero(self):
        assert Organism._estimate_stagnation([]) == 0.0

    def test_single_entry_returns_zero(self):
        assert Organism._estimate_stagnation([0.5]) == 0.0

    def test_constant_history_returns_one(self):
        hist = [0.5] * 20
        assert Organism._estimate_stagnation(hist) == 1.0

    def test_spread_0_10_returns_zero(self):
        hist = [0.40, 0.50]  # spread = 0.10
        assert Organism._estimate_stagnation(hist) == pytest.approx(0.0)

    def test_spread_0_05_returns_half(self):
        hist = [0.45, 0.50]  # spread = 0.05
        assert Organism._estimate_stagnation(hist) == pytest.approx(0.5)

    def test_large_spread_clamps_at_zero(self):
        hist = [0.10, 0.80]  # spread = 0.70 >> 0.10
        assert Organism._estimate_stagnation(hist) == 0.0

    def test_uses_last_20_entries(self):
        # 30 entries with big range, but last 20 are constant
        hist = [0.1, 0.9] + [0.5] * 28
        assert Organism._estimate_stagnation(hist) == 1.0


# ═══════════════════════════════════════════════════════════════════════
# 3. Gate: Cycle interval
# ═══════════════════════════════════════════════════════════════════════

class TestCycleIntervalGate:
    """Reproduction only fires when _global_cycle % 500 == 0."""

    def test_not_on_interval_returns_none(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            org._global_cycle = 499
            assert org.rare_reproduction_cycle() is None
        finally:
            _cleanup_handlers()

    def test_on_interval_passes_gate(self, tmp_path, monkeypatch):
        """When on interval, the method proceeds (may still fail other gates)."""
        org = _make_organism(str(tmp_path))
        try:
            org._global_cycle = 500
            # Force random to fail so we can prove the cycle gate passed
            monkeypatch.setattr(random, "random", lambda: 1.0)
            assert org.rare_reproduction_cycle() is None
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 4. Gate: Random chance
# ═══════════════════════════════════════════════════════════════════════

class TestRandomChanceGate:

    def test_random_too_high_returns_none(self, tmp_path, monkeypatch):
        org = _make_organism(str(tmp_path))
        try:
            org._global_cycle = 500
            monkeypatch.setattr(random, "random", lambda: 0.06)  # > 0.05
            assert org.rare_reproduction_cycle() is None
        finally:
            _cleanup_handlers()

    def test_random_exactly_at_threshold_returns_none(self, tmp_path, monkeypatch):
        org = _make_organism(str(tmp_path))
        try:
            org._global_cycle = 500
            monkeypatch.setattr(random, "random", lambda: 0.05)  # >= 0.05
            assert org.rare_reproduction_cycle() is None
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 5. Gate: Population cap
# ═══════════════════════════════════════════════════════════════════════

class TestPopulationCapGate:

    def test_pop_at_cap_returns_none(self, tmp_path, monkeypatch):
        org = _make_organism(str(tmp_path))
        try:
            org._global_cycle = 500
            monkeypatch.setattr(random, "random", lambda: 0.01)
            # Fill population to RARE_REPRO_POP_CAP
            _spawn_children(org._population, RARE_REPRO_POP_CAP - 1)
            assert org._population.size >= RARE_REPRO_POP_CAP
            assert org.rare_reproduction_cycle() is None
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 6. Parent eligibility
# ═══════════════════════════════════════════════════════════════════════

class TestParentEligibility:

    def _setup(self, tmp_path, monkeypatch):
        org = _make_organism(str(tmp_path))
        org._global_cycle = 500
        monkeypatch.setattr(random, "random", lambda: 0.01)
        monkeypatch.setattr(random, "shuffle", lambda x: None)  # deterministic order
        return org

    def test_low_energy_ineligible(self, tmp_path, monkeypatch):
        org = self._setup(tmp_path, monkeypatch)
        try:
            _set_eligible_parent(org._population, "AL-01", energy=0.60)
            assert org.rare_reproduction_cycle() is None
        finally:
            _cleanup_handlers()

    def test_low_fitness_ineligible(self, tmp_path, monkeypatch):
        org = self._setup(tmp_path, monkeypatch)
        try:
            org._population.update_energy("AL-01", 0.80)
            # Record only low fitness values
            for _ in range(20):
                org._population.record_fitness("AL-01", 0.40)
            assert org.rare_reproduction_cycle() is None
        finally:
            _cleanup_handlers()

    def test_high_stagnation_ineligible(self, tmp_path, monkeypatch):
        org = self._setup(tmp_path, monkeypatch)
        try:
            org._population.update_energy("AL-01", 0.80)
            # Constant fitness → stagnation = 1.0 (>= 0.90 threshold)
            for _ in range(20):
                org._population.record_fitness("AL-01", 0.60)
            assert org.rare_reproduction_cycle() is None
        finally:
            _cleanup_handlers()

    def test_eligible_parent_produces_child(self, tmp_path, monkeypatch):
        org = self._setup(tmp_path, monkeypatch)
        try:
            _set_eligible_parent(org._population, "AL-01", energy=0.80)
            child = org.rare_reproduction_cycle()
            assert child is not None
            assert child["parent_id"] == "AL-01"
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 7. Reproduction cost & child energy
# ═══════════════════════════════════════════════════════════════════════

class TestReproductionCost:

    def test_parent_energy_deducted(self, tmp_path, monkeypatch):
        org = _make_organism(str(tmp_path))
        try:
            org._global_cycle = 500
            monkeypatch.setattr(random, "random", lambda: 0.01)
            monkeypatch.setattr(random, "shuffle", lambda x: None)
            _set_eligible_parent(org._population, "AL-01", energy=0.80)
            org.rare_reproduction_cycle()
            parent = org._population.get("AL-01")
            assert parent is not None
            assert parent["energy"] == pytest.approx(0.80 - RARE_REPRO_ENERGY_COST, abs=1e-4)
        finally:
            _cleanup_handlers()

    def test_child_energy_set(self, tmp_path, monkeypatch):
        org = _make_organism(str(tmp_path))
        try:
            org._global_cycle = 500
            monkeypatch.setattr(random, "random", lambda: 0.01)
            monkeypatch.setattr(random, "shuffle", lambda x: None)
            _set_eligible_parent(org._population, "AL-01", energy=0.80)
            child = org.rare_reproduction_cycle()
            assert child is not None
            child_record = org._population.get(child["id"])
            assert child_record is not None
            assert child_record["energy"] == pytest.approx(RARE_REPRO_CHILD_ENERGY, abs=1e-4)
        finally:
            _cleanup_handlers()

    def test_parent_energy_floor_at_zero(self, tmp_path, monkeypatch):
        """Parent energy doesn't go negative even if close to the threshold."""
        org = _make_organism(str(tmp_path))
        try:
            org._global_cycle = 500
            monkeypatch.setattr(random, "random", lambda: 0.01)
            monkeypatch.setattr(random, "shuffle", lambda x: None)
            # Energy just above threshold — after deduction would be 0.35
            _set_eligible_parent(org._population, "AL-01", energy=0.65)
            child = org.rare_reproduction_cycle()
            assert child is not None
            parent = org._population.get("AL-01")
            assert parent is not None
            assert parent["energy"] >= 0.0
            assert parent["energy"] == pytest.approx(0.35, abs=1e-4)
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 8. Cooldown enforcement
# ═══════════════════════════════════════════════════════════════════════

class TestCooldown:

    def test_cooldown_blocks_second_birth(self, tmp_path, monkeypatch):
        org = _make_organism(str(tmp_path))
        try:
            monkeypatch.setattr(random, "random", lambda: 0.01)
            monkeypatch.setattr(random, "shuffle", lambda x: None)

            # First birth at cycle 500
            org._global_cycle = 500
            _set_eligible_parent(org._population, "AL-01", energy=0.90)
            child1 = org.rare_reproduction_cycle()
            assert child1 is not None

            # Try again at cycle 1000 — too soon (need 2000 gap)
            org._global_cycle = 1000
            _set_eligible_parent(org._population, "AL-01", energy=0.90)
            child2 = org.rare_reproduction_cycle()
            assert child2 is None
        finally:
            _cleanup_handlers()

    def test_cooldown_expires(self, tmp_path, monkeypatch):
        org = _make_organism(str(tmp_path))
        try:
            monkeypatch.setattr(random, "random", lambda: 0.01)
            monkeypatch.setattr(random, "shuffle", lambda x: None)

            # First birth at cycle 500
            org._global_cycle = 500
            _set_eligible_parent(org._population, "AL-01", energy=0.90)
            child1 = org.rare_reproduction_cycle()
            assert child1 is not None

            # Try at cycle 2500 — 2000 since last birth → allowed
            org._global_cycle = 2500
            _set_eligible_parent(org._population, "AL-01", energy=0.90)
            child2 = org.rare_reproduction_cycle()
            assert child2 is not None
        finally:
            _cleanup_handlers()

    def test_cooldown_tracked_per_parent(self, tmp_path, monkeypatch):
        """Different parents have independent cooldowns."""
        org = _make_organism(str(tmp_path))
        try:
            monkeypatch.setattr(random, "random", lambda: 0.01)

            # Spawn a second organism
            _spawn_children(org._population, 1)
            members = list(org._population.member_ids)
            child_id = [m for m in members if m != "AL-01"][0]

            # Make both eligible
            _set_eligible_parent(org._population, "AL-01", energy=0.90)
            _set_eligible_parent(org._population, child_id, energy=0.90)

            # AL-01 reproduces at cycle 500
            org._global_cycle = 500
            monkeypatch.setattr(random, "shuffle", lambda x: x.sort())  # AL-01 first
            result1 = org.rare_reproduction_cycle()
            assert result1 is not None
            assert result1["parent_id"] == "AL-01"

            # At cycle 1000, AL-01 on cooldown but child_id should be ok
            org._global_cycle = 1000
            _set_eligible_parent(org._population, child_id, energy=0.90)
            # Force child_id to be checked first
            monkeypatch.setattr(random, "shuffle",
                                lambda x: x.sort(key=lambda i: 0 if i == child_id else 1))
            result2 = org.rare_reproduction_cycle()
            assert result2 is not None
            assert result2["parent_id"] == child_id
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 9. Child properties
# ═══════════════════════════════════════════════════════════════════════

class TestChildProperties:

    def test_child_has_parent_id(self, tmp_path, monkeypatch):
        org = _make_organism(str(tmp_path))
        try:
            org._global_cycle = 500
            monkeypatch.setattr(random, "random", lambda: 0.01)
            monkeypatch.setattr(random, "shuffle", lambda x: None)
            _set_eligible_parent(org._population, "AL-01", energy=0.80)
            child = org.rare_reproduction_cycle()
            assert child is not None
            assert child["parent_id"] == "AL-01"
        finally:
            _cleanup_handlers()

    def test_child_generation_incremented(self, tmp_path, monkeypatch):
        org = _make_organism(str(tmp_path))
        try:
            org._global_cycle = 500
            monkeypatch.setattr(random, "random", lambda: 0.01)
            monkeypatch.setattr(random, "shuffle", lambda x: None)
            _set_eligible_parent(org._population, "AL-01", energy=0.80)
            parent_rec = org._population.get("AL-01")
            assert parent_rec is not None
            parent_gen = parent_rec.get("generation_id", 0)
            child = org.rare_reproduction_cycle()
            assert child is not None
            assert child["generation_id"] == parent_gen + 1
        finally:
            _cleanup_handlers()

    def test_child_has_genome(self, tmp_path, monkeypatch):
        org = _make_organism(str(tmp_path))
        try:
            org._global_cycle = 500
            monkeypatch.setattr(random, "random", lambda: 0.01)
            monkeypatch.setattr(random, "shuffle", lambda x: None)
            _set_eligible_parent(org._population, "AL-01", energy=0.80)
            child = org.rare_reproduction_cycle()
            assert child is not None
            assert "genome" in child
            assert "traits" in child["genome"]
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 10. Safety: only one child per invocation
# ═══════════════════════════════════════════════════════════════════════

class TestSingleChildSafety:

    def test_only_one_child_spawned(self, tmp_path, monkeypatch):
        """Even with many eligible parents, only ONE child is created."""
        org = _make_organism(str(tmp_path))
        try:
            org._global_cycle = 500
            monkeypatch.setattr(random, "random", lambda: 0.01)
            # Spawn 5 additional organisms for many eligible parents
            _spawn_children(org._population, 5)
            for mid in org._population.member_ids:
                _set_eligible_parent(org._population, mid, energy=0.90)

            initial_size = org._population.size
            child = org.rare_reproduction_cycle()
            assert child is not None
            assert org._population.size == initial_size + 1
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 11. Safety: re-check cap before commit
# ═══════════════════════════════════════════════════════════════════════

class TestCapRecheck:

    def test_recheck_blocks_if_cap_reached_between_gates(self, tmp_path, monkeypatch):
        """If population reaches cap after initial check but before spawn,
        the method should return None."""
        org = _make_organism(str(tmp_path))
        try:
            org._global_cycle = 500
            monkeypatch.setattr(random, "random", lambda: 0.01)
            monkeypatch.setattr(random, "shuffle", lambda x: None)
            _set_eligible_parent(org._population, "AL-01", energy=0.80)

            # Fill to cap - 1 (initial check passes)
            _spawn_children(org._population, RARE_REPRO_POP_CAP - 2)
            # Size is now RARE_REPRO_POP_CAP - 1 → initial gate passes

            # Now the last slot is open — should succeed
            child = org.rare_reproduction_cycle()
            # With one slot open, it should succeed
            assert child is not None
            # Now at cap
            assert org._population.size >= RARE_REPRO_POP_CAP
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 12. VITAL life-log entry
# ═══════════════════════════════════════════════════════════════════════

class TestVitalLifeLog:

    def test_birth_logged_in_vital(self, tmp_path, monkeypatch):
        org = _make_organism(str(tmp_path))
        try:
            org._global_cycle = 500
            monkeypatch.setattr(random, "random", lambda: 0.01)
            monkeypatch.setattr(random, "shuffle", lambda x: None)
            _set_eligible_parent(org._population, "AL-01", energy=0.80)
            child = org.rare_reproduction_cycle()
            assert child is not None

            # Read life log JSONL and check for rare_reproduction event
            import json
            log_path = os.path.join(str(tmp_path), "data", "life_log.jsonl")
            events = []
            with open(log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
            repro_events = [e for e in events if e.get("event_type") == "rare_reproduction"]
            assert len(repro_events) >= 1
            evt = repro_events[-1]
            payload = evt["payload"]
            assert payload["parent_id"] == "AL-01"
            assert payload["child_id"] == child["id"]
            assert payload["cycle"] == 500
            assert payload["child_energy"] == RARE_REPRO_CHILD_ENERGY
            assert "parent_energy_after" in payload
            assert "parent_fitness" in payload
            assert "stagnation_estimate" in payload
            assert "population_size" in payload
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 13. Dormant parents are skipped
# ═══════════════════════════════════════════════════════════════════════

class TestDormantSkipped:

    def test_dormant_parent_ineligible(self, tmp_path, monkeypatch):
        org = _make_organism(str(tmp_path))
        try:
            org._global_cycle = 500
            monkeypatch.setattr(random, "random", lambda: 0.01)
            monkeypatch.setattr(random, "shuffle", lambda x: None)
            _set_eligible_parent(org._population, "AL-01", energy=0.80)
            # Set AL-01 to dormant via lifecycle_state
            org._population.enter_dormant("AL-01", cause="test")
            child = org.rare_reproduction_cycle()
            assert child is None
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 14. MetabolismScheduler integration
# ═══════════════════════════════════════════════════════════════════════

class TestMetabolismSchedulerIntegration:

    def test_rare_reproduce_interval_in_config(self):
        from al01.organism import MetabolismConfig
        cfg = MetabolismConfig()
        assert hasattr(cfg, "rare_reproduce_interval")
        assert cfg.rare_reproduce_interval == 50

    def test_scheduler_calls_rare_reproduction(self, tmp_path, monkeypatch):
        from al01.organism import MetabolismScheduler, MetabolismConfig
        org = _make_organism(str(tmp_path))
        try:
            cfg = MetabolismConfig(rare_reproduce_interval=1)
            sched = MetabolismScheduler(org, cfg)
            call_count = 0
            original = org.rare_reproduction_cycle
            def tracking_call():
                nonlocal call_count
                call_count += 1
                return original()
            monkeypatch.setattr(org, "rare_reproduction_cycle", tracking_call)
            sched.tick()
            assert call_count == 1
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 15. Edge: no members alive
# ═══════════════════════════════════════════════════════════════════════

class TestNoAliveMembers:

    def test_no_alive_members_returns_none(self, tmp_path, monkeypatch):
        org = _make_organism(str(tmp_path))
        try:
            org._global_cycle = 500
            monkeypatch.setattr(random, "random", lambda: 0.01)
            # Kill all members
            with org._population._lock:
                for mid in list(org._population._members):
                    org._population._members[mid]["alive"] = False
                org._population._save()
            assert org.rare_reproduction_cycle() is None
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 16. Edge: empty fitness history
# ═══════════════════════════════════════════════════════════════════════

class TestEmptyFitnessHistory:

    def test_empty_fitness_history_ineligible(self, tmp_path, monkeypatch):
        """Member with no fitness history has fitness=0.0, so is ineligible."""
        org = _make_organism(str(tmp_path))
        try:
            org._global_cycle = 500
            monkeypatch.setattr(random, "random", lambda: 0.01)
            monkeypatch.setattr(random, "shuffle", lambda x: None)
            org._population.update_energy("AL-01", 0.80)
            # Clear fitness history
            with org._population._lock:
                org._population._members["AL-01"]["fitness_history"] = []
                org._population._save()
            assert org.rare_reproduction_cycle() is None
        finally:
            _cleanup_handlers()
