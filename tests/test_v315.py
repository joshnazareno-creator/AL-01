"""v3.15 — Population Floor 20%, Emergency Regen 25%, Adaptive Metabolism,
Dormant State & Extinction Recovery Protocol tests.

Tests cover:
 1. Population floor raised from 10% → 20% of max_population
 2. Emergency regen threshold raised from 20% → 25%
 3. Adaptive metabolism — reduces metabolic cost during high scarcity
 4. Dormant state — organisms enter dormancy instead of immediate death
 5. Extinction recovery — if pop < 5, auto-seed 10 children from top fitness
 6. Wake dormant cycle — dormant organisms wake when conditions improve
 7. Integration tests: organism-level dormant + recovery flows
"""

import logging
import os
import tempfile

import pytest

from al01.environment import Environment, EnvironmentConfig
from al01.population import Population, ABSOLUTE_POPULATION_CAP, BIRTH_COOLDOWN_CYCLES
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
# 1. Population Floor — 20%
# ═══════════════════════════════════════════════════════════════════════

class TestPopulationFloor20:
    def test_floor_is_20_percent_of_max(self):
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp, max_pop=60)
            assert pop.min_population_floor == 12  # int(60 * 0.2)

    def test_floor_20_percent_various_caps(self):
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp, max_pop=40)
            assert pop.min_population_floor == 8   # int(40 * 0.2)
            pop.max_population = 20
            assert pop.min_population_floor == 4   # int(20 * 0.2)
            pop.max_population = 10
            assert pop.min_population_floor == 2   # int(10 * 0.2)

    def test_floor_minimum_is_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp, max_pop=2)
            # int(2 * 0.2) = 0, clamped to 1
            assert pop.min_population_floor >= 1

    def test_floor_blocks_at_20_percent(self):
        """Sequential kills stop at 20% floor."""
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp, max_pop=20)  # floor = 4
            _spawn_children(pop, 9)       # 10 total
            assert pop.size == 10
            assert pop.min_population_floor == 4
            killed = 0
            for mid in list(pop.member_ids):
                result = pop.remove_member(mid, cause="test")
                if result is not None:
                    killed += 1
            assert pop.size == 4
            assert killed == 6  # 10 - 4

    def test_prune_respects_20_percent_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp, max_pop=60)  # floor = 12
            _spawn_children(pop, 19)     # 20 total
            assert pop.size == 20
            deaths = pop.prune_weakest(max_size=3, min_keep=2)
            assert pop.size >= 12  # floor = 12


# ═══════════════════════════════════════════════════════════════════════
# 2. Emergency Regen Threshold — 25%
# ═══════════════════════════════════════════════════════════════════════

class TestEmergencyRegen25:
    def test_regen_triggered_below_25_percent(self):
        """Emergency regen activates when pool < 25%."""
        env = _env(pool_max=1000.0, pool_initial=200.0)  # 20% < 25%
        assert env.pool_fraction < 0.25
        regen = env.emergency_regenerate(population_size=5)
        assert regen > 0

    def test_regen_not_triggered_at_25_percent(self):
        """Emergency regen does NOT activate when pool == 25%."""
        env = _env(pool_max=1000.0, pool_initial=250.0)
        assert env.pool_fraction == pytest.approx(0.25, abs=0.001)
        regen = env.emergency_regenerate(population_size=5)
        assert regen == 0.0

    def test_regen_not_triggered_above_25_percent(self):
        env = _env(pool_max=1000.0, pool_initial=300.0)  # 30%
        regen = env.emergency_regenerate(population_size=5)
        assert regen == 0.0

    def test_state_snapshot_threshold(self):
        env = _env(pool_max=1000.0, pool_initial=200.0)
        snap = env.state_snapshot()
        assert snap["emergency_regen_threshold"] == 0.25
        assert snap["emergency_regen_active"] is True


# ═══════════════════════════════════════════════════════════════════════
# 3. Adaptive Metabolism
# ═══════════════════════════════════════════════════════════════════════

class TestAdaptiveMetabolism:
    def test_no_scarcity_returns_base_cost(self):
        """When not in scarcity, metabolic cost is base."""
        env = _env(pool_max=1000.0, pool_initial=1000.0)
        assert env.effective_metabolic_cost() == pytest.approx(1.0)

    def test_mild_scarcity_increases_cost(self):
        """During mild scarcity (severity < 0.6), cost INCREASES."""
        # severity needs to be low: pool needs to be just below threshold
        # scarcity_threshold = 0.25, so pool_fraction < 0.25
        # severity = 1.0 - (frac / 0.25)
        # For severity = 0.3: frac = 0.25 * (1 - 0.3) = 0.175 → pool = 175
        env = _env(pool_max=1000.0, pool_initial=175.0)
        assert env.is_scarcity_pressure
        severity = env.scarcity_severity
        assert severity < 0.6
        cost = env.effective_metabolic_cost()
        assert cost > 1.0  # base cost is 1.0, should be higher

    def test_high_scarcity_decreases_cost(self):
        """During high scarcity (severity >= 0.6), adaptive metabolism REDUCES cost."""
        # For severity = 0.8: frac = 0.25 * (1 - 0.8) = 0.05 → pool = 50
        env = _env(pool_max=1000.0, pool_initial=50.0)
        assert env.is_scarcity_pressure
        severity = env.scarcity_severity
        assert severity >= 0.6
        cost = env.effective_metabolic_cost()
        # At high severity, cost should be LOWER than the peak scarcity cost
        # The peak occurs at severity == 0.6 (adaptive_metabolism_threshold)
        peak_cost = 2.0 * (1.0 + 0.6 * 0.5)  # base * (1 + severity * (mult-1))
        assert cost < peak_cost

    def test_extreme_scarcity_near_floor(self):
        """At severity ≈ 1.0 (pool near zero), cost drops to adaptive_metabolism_floor."""
        env = _env(pool_max=1000.0, pool_initial=1.0)
        severity = env.scarcity_severity
        assert severity > 0.95
        cost = env.effective_metabolic_cost()
        # Should be near base * floor = 1.0 * 0.4 = 0.4
        assert cost < 0.75
        assert cost >= 1.0 * 0.4 - 0.1  # floor with tolerance

    def test_adaptive_metabolism_transition_is_smooth(self):
        """Cost at severity == threshold should equal the peak multiplier."""
        # At threshold=0.6: mild scarcity formula gives 1 + 0.6*0.5 = 1.3
        # High scarcity formula at threshold should also give 1.3 (continuity)
        env = _env(pool_max=1000.0, pool_initial=100.0,
                    scarcity_threshold=0.25)
        # Manually set pool to get exact severity = 0.6
        # frac = 0.25*(1-0.6) = 0.1 → pool = 100
        env._resource_pool = 100.0
        severity = env.scarcity_severity
        assert severity == pytest.approx(0.6, abs=0.01)
        cost = env.effective_metabolic_cost()
        # At the exact threshold, both formulas should agree
        expected = 1.0 * 1.5  # base * scarcity_metabolic_multiplier (peak)
        assert cost == pytest.approx(expected, abs=0.1)

    def test_state_snapshot_adaptive_metabolism_flag(self):
        """state_snapshot includes adaptive_metabolism_active."""
        env_high = _env(pool_max=1000.0, pool_initial=10.0)
        snap = env_high.state_snapshot()
        assert snap["adaptive_metabolism_active"] is True

        env_healthy = _env(pool_max=1000.0, pool_initial=800.0)
        snap = env_healthy.state_snapshot()
        assert snap["adaptive_metabolism_active"] is False

    def test_config_defaults(self):
        """Adaptive metabolism config fields have expected defaults."""
        cfg = EnvironmentConfig()
        assert cfg.adaptive_metabolism_threshold == 0.6
        assert cfg.adaptive_metabolism_floor == 0.4


# ═══════════════════════════════════════════════════════════════════════
# 4. Dormant State
# ═══════════════════════════════════════════════════════════════════════

class TestDormantState:
    def test_enter_dormant(self):
        """enter_dormant sets state='dormant' and preserves alive=True."""
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp, max_pop=60)
            _spawn_children(pop, 3)
            child_ids = [mid for mid in pop.member_ids if mid != "AL-01"]
            info = pop.enter_dormant(child_ids[0], cause="fitness_floor")
            assert info is not None
            assert info["cause"] == "fitness_floor"
            member = pop.get(child_ids[0])
            assert member["alive"] is True
            assert member["state"] == "dormant"

    def test_dormant_excluded_from_size(self):
        """Dormant organisms are NOT counted in .size."""
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp, max_pop=60)
            _spawn_children(pop, 5)
            assert pop.size == 6
            child_ids = [mid for mid in pop.member_ids if mid != "AL-01"]
            pop.enter_dormant(child_ids[0], cause="test")
            assert pop.size == 5  # one is dormant, not counted
            assert pop.dormant_count == 1
            assert pop.living_or_dormant_count == 6

    def test_dormant_excluded_from_member_ids(self):
        """Dormant organisms not in member_ids."""
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp, max_pop=60)
            _spawn_children(pop, 3)
            child_ids = [mid for mid in pop.member_ids if mid != "AL-01"]
            pop.enter_dormant(child_ids[0], cause="test")
            assert child_ids[0] not in pop.member_ids
            assert child_ids[0] in pop.dormant_ids

    def test_get_dormant(self):
        """get_dormant returns only dormant members."""
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp, max_pop=60)
            _spawn_children(pop, 3)
            child_ids = [mid for mid in pop.member_ids if mid != "AL-01"]
            pop.enter_dormant(child_ids[0], cause="test")
            pop.enter_dormant(child_ids[1], cause="test")
            dormant = pop.get_dormant()
            assert len(dormant) == 2

    def test_wake_dormant(self):
        """wake_dormant returns organism to active state with energy boost."""
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp, max_pop=60)
            _spawn_children(pop, 3)
            child_ids = [mid for mid in pop.member_ids if mid != "AL-01"]
            pop.enter_dormant(child_ids[0], cause="test")
            assert pop.dormant_count == 1

            wake = pop.wake_dormant(child_ids[0], energy_boost=0.3)
            assert wake is not None
            assert wake["organism_id"] == child_ids[0]
            assert pop.dormant_count == 0

            member = pop.get(child_ids[0])
            assert member["state"] == "idle"
            assert member["alive"] is True
            assert member["energy"] >= 0.3  # got boost

    def test_wake_non_dormant_returns_none(self):
        """wake_dormant on a non-dormant organism returns None."""
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp, max_pop=60)
            _spawn_children(pop, 1)
            child_ids = [mid for mid in pop.member_ids if mid != "AL-01"]
            result = pop.wake_dormant(child_ids[0])
            assert result is None

    def test_enter_dormant_already_dead_returns_none(self):
        """Can't enter dormant if already dead."""
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp, max_pop=60)
            pop.hardcore_extinction_mode = True
            _spawn_children(pop, 1)
            child_ids = [mid for mid in pop.member_ids if mid != "AL-01"]
            pop.remove_member(child_ids[0], cause="test")
            result = pop.enter_dormant(child_ids[0], cause="test")
            assert result is None

    def test_enter_dormant_twice_returns_existing_info(self):
        """Entering dormant twice returns existing dormant info."""
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp, max_pop=60)
            _spawn_children(pop, 1)
            child_ids = [mid for mid in pop.member_ids if mid != "AL-01"]
            info1 = pop.enter_dormant(child_ids[0], cause="test")
            info2 = pop.enter_dormant(child_ids[0], cause="test2")
            assert info1 is not None
            assert info2 is not None
            # Second call returns existing info (cause unchanged)
            assert pop.dormant_count == 1


# ═══════════════════════════════════════════════════════════════════════
# 5. Extinction Recovery Protocol
# ═══════════════════════════════════════════════════════════════════════

class TestExtinctionRecovery:
    def test_recovery_triggers_below_5(self):
        """When pop < 5, extinction recovery spawns new children."""
        tmp = tempfile.mkdtemp()
        try:
            org = _make_organism(tmp)
            org.boot()
            # Set up: 3 living organisms (below threshold of 5)
            _spawn_children(org._population, 2)  # AL-01 + 2 = 3
            assert org._population.size == 3

            # Manually trigger recovery
            result = org.check_extinction_reseed()
            assert result is not None
            assert result["event"] == "extinction_recovery"
            assert len(result["spawned"]) > 0
            assert org._population.size > 3
        finally:
            _cleanup_handlers()

    def test_recovery_does_not_trigger_at_5(self):
        """When pop == 5, no extinction recovery."""
        tmp = tempfile.mkdtemp()
        try:
            org = _make_organism(tmp)
            org.boot()
            _spawn_children(org._population, 4)  # AL-01 + 4 = 5
            assert org._population.size == 5
            result = org.check_extinction_reseed()
            assert result is None
        finally:
            _cleanup_handlers()

    def test_recovery_does_not_trigger_above_5(self):
        """When pop > 5, no extinction recovery."""
        tmp = tempfile.mkdtemp()
        try:
            org = _make_organism(tmp)
            org.boot()
            _spawn_children(org._population, 9)  # AL-01 + 9 = 10
            assert org._population.size == 10
            result = org.check_extinction_reseed()
            assert result is None
        finally:
            _cleanup_handlers()

    def test_recovery_uses_top_fitness_lineage(self):
        """Recovery seeds are based on top-fitness members."""
        tmp = tempfile.mkdtemp()
        try:
            org = _make_organism(tmp)
            org.boot()
            # Set high fitness on parent so it becomes a seed
            parent_genome = Genome.from_dict(org._population.get("AL-01")["genome"])
            parent_genome._traits["resilience"] = 0.9
            parent_genome._traits["adaptability"] = 0.9
            org._population.update_member("AL-01", {"genome": parent_genome.to_dict()})

            # Only 1 living → triggers recovery
            result = org.check_extinction_reseed()
            assert result is not None
            assert "AL-01" in result["seeds_used"]
        finally:
            _cleanup_handlers()

    def test_recovery_wakes_dormant(self):
        """Extinction recovery also wakes dormant organisms."""
        tmp = tempfile.mkdtemp()
        try:
            org = _make_organism(tmp)
            org.boot()
            _spawn_children(org._population, 3)
            child_ids = [mid for mid in org._population.member_ids if mid != "AL-01"]
            # Put 2 children dormant → 2 active (AL-01 + 1 child)
            org._population.enter_dormant(child_ids[0], cause="test")
            org._population.enter_dormant(child_ids[1], cause="test")
            assert org._population.size == 2  # < 5
            assert org._population.dormant_count == 2

            result = org.check_extinction_reseed()
            assert result is not None
            assert len(result["dormant_woken"]) == 2
        finally:
            _cleanup_handlers()

    def test_genesis_vault_fallback_on_full_extinction(self):
        """When pop == 0, genesis vault reseed still works."""
        tmp = tempfile.mkdtemp()
        try:
            org = _make_organism(tmp)
            org.boot()
            org._population.hardcore_extinction_mode = True
            # Kill everyone
            for mid in list(org._population.member_ids):
                org._population.remove_member(mid, cause="test")
            assert org._population.size == 0

            result = org.check_extinction_reseed()
            assert result is not None
            # Genesis vault reseed returns reseed record format
            assert result.get("event") == "reseed"
            assert org._population.size > 0
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 6. Top Fitness Members
# ═══════════════════════════════════════════════════════════════════════

class TestTopFitnessMembers:
    def test_top_fitness_returns_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp, max_pop=60)
            _spawn_children(pop, 5)
            # Set varying fitness
            for i, mid in enumerate(pop.member_ids):
                g = Genome.from_dict(pop.get(mid)["genome"])
                g._traits["resilience"] = 0.1 * (i + 1)
                pop.update_member(mid, {"genome": g.to_dict()})

            top = pop.top_fitness_members(n=3)
            assert len(top) == 3
            # Verify descending order
            fitnesses = [t.get("genome", {}).get("fitness", 0) for t in top]
            assert fitnesses == sorted(fitnesses, reverse=True)

    def test_top_fitness_includes_dormant(self):
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp, max_pop=60)
            _spawn_children(pop, 3)
            child_ids = [mid for mid in pop.member_ids if mid != "AL-01"]
            # Give one child high fitness then make it dormant
            g = Genome.from_dict(pop.get(child_ids[0])["genome"])
            g._traits["resilience"] = 0.99
            pop.update_member(child_ids[0], {"genome": g.to_dict()})
            pop.enter_dormant(child_ids[0], cause="test")

            top = pop.top_fitness_members(n=1, include_dormant=True)
            assert len(top) == 1
            assert top[0]["id"] == child_ids[0]

    def test_top_fitness_includes_recently_dead(self):
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp, max_pop=60)
            pop.hardcore_extinction_mode = True
            _spawn_children(pop, 2)
            child_ids = [mid for mid in pop.member_ids if mid != "AL-01"]
            # Give high fitness then kill
            g = Genome.from_dict(pop.get(child_ids[0])["genome"])
            g._traits["resilience"] = 0.99
            pop.update_member(child_ids[0], {"genome": g.to_dict()})
            pop.remove_member(child_ids[0], cause="test")

            top = pop.top_fitness_members(n=5)
            ids_in_top = [t["id"] for t in top]
            assert child_ids[0] in ids_in_top


# ═══════════════════════════════════════════════════════════════════════
# 7. Wake Dormant Cycle (Organism-level)
# ═══════════════════════════════════════════════════════════════════════

class TestWakeDormantCycle:
    def test_wake_when_pool_healthy(self):
        """Dormant organisms wake when environment is no longer in scarcity."""
        tmp = tempfile.mkdtemp()
        try:
            org = _make_organism(tmp)
            org.boot()
            _spawn_children(org._population, 5)
            child_ids = [mid for mid in org._population.member_ids if mid != "AL-01"]
            org._population.enter_dormant(child_ids[0], cause="test")
            assert org._population.dormant_count == 1

            # Pool is healthy (default 1000/1000)
            assert not org._environment.is_scarcity_pressure
            woke = org.wake_dormant_cycle()
            assert len(woke) == 1
            assert org._population.dormant_count == 0
        finally:
            _cleanup_handlers()

    def test_no_wake_during_scarcity(self):
        """Dormant organisms stay dormant during scarcity (unless pop critical)."""
        tmp = tempfile.mkdtemp()
        try:
            org = _make_organism(tmp)
            org.boot()
            _spawn_children(org._population, 9)  # 10 total → pop not critical
            child_ids = [mid for mid in org._population.member_ids if mid != "AL-01"]
            org._population.enter_dormant(child_ids[0], cause="test")

            # Force scarcity
            org._environment._resource_pool = 10.0  # 1% of max
            assert org._environment.is_scarcity_pressure

            woke = org.wake_dormant_cycle()
            assert len(woke) == 0  # still dormant
            assert org._population.dormant_count == 1
        finally:
            _cleanup_handlers()

    def test_wake_during_critical_pop_even_in_scarcity(self):
        """If pop < 5, dormant organisms wake even during scarcity."""
        tmp = tempfile.mkdtemp()
        try:
            org = _make_organism(tmp)
            org.boot()
            _spawn_children(org._population, 3)  # 4 total
            child_ids = [mid for mid in org._population.member_ids if mid != "AL-01"]
            org._population.enter_dormant(child_ids[0], cause="test")
            org._population.enter_dormant(child_ids[1], cause="test")
            # Now: 2 active (AL-01 + 1 child), 2 dormant → size=2 < 5

            # Force scarcity
            org._environment._resource_pool = 10.0
            assert org._environment.is_scarcity_pressure

            woke = org.wake_dormant_cycle()
            assert len(woke) == 2  # both wake because pop is critical
            assert org._population.dormant_count == 0
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 8. Integration — _handle_death uses dormant
# ═══════════════════════════════════════════════════════════════════════

class TestHandleDeathDormant:
    def test_fitness_floor_causes_dormancy(self):
        """_handle_death with fitness_floor puts child into dormant, not dead."""
        tmp = tempfile.mkdtemp()
        try:
            org = _make_organism(tmp)
            org.boot()
            _spawn_children(org._population, 5)
            child_ids = [mid for mid in org._population.member_ids if mid != "AL-01"]
            target = child_ids[0]

            org._handle_death(target, "fitness_floor")
            member = org._population.get(target)
            assert member is not None
            assert member["alive"] is True
            assert member["state"] == "dormant"
        finally:
            _cleanup_handlers()

    def test_energy_depleted_causes_dormancy_v316(self):
        """v3.16: _handle_death with energy_depleted now puts child into dormant."""
        tmp = tempfile.mkdtemp()
        try:
            org = _make_organism(tmp)
            org.boot()
            org._population.hardcore_extinction_mode = True
            _spawn_children(org._population, 5)
            child_ids = [mid for mid in org._population.member_ids if mid != "AL-01"]
            target = child_ids[0]

            org._handle_death(target, "energy_depleted")
            member = org._population.get(target)
            assert member is not None
            # v3.16: energy_depleted now triggers dormancy (not hard kill)
            assert member["alive"] is True
            assert member["state"] == "dormant"
        finally:
            _cleanup_handlers()

    def test_founder_does_not_go_dormant(self):
        """AL-01 (founder) is never put into dormant state."""
        tmp = tempfile.mkdtemp()
        try:
            org = _make_organism(tmp)
            org.boot()
            org._population.hardcore_extinction_mode = True

            org._handle_death("AL-01", "fitness_floor")
            member = org._population.get("AL-01")
            assert member is not None
            # Founder either died or was blocked by floor, but NOT dormant
            assert member.get("state") != "dormant"
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 9. API fields
# ═══════════════════════════════════════════════════════════════════════

class TestAPIFields:
    def test_population_has_dormant_properties(self):
        """Population exposes dormant_count and living_or_dormant_count."""
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp, max_pop=60)
            assert pop.dormant_count == 0
            assert pop.living_or_dormant_count == 1  # just AL-01

    def test_adaptive_metabolism_in_env_snapshot(self):
        """Environment state_snapshot includes adaptive_metabolism_active."""
        env = _env(pool_max=1000.0, pool_initial=10.0)
        snap = env.state_snapshot()
        assert "adaptive_metabolism_active" in snap
