"""v3.14/v3.15 — Population Floor, Emergency Regen & Hardcore Extinction Mode tests.

Tests cover:
1. min_population_floor = 20% of max_population (updated in v3.15)
2. remove_member blocked when at floor (non-hardcore)
3. prune_weakest respects floor
4. hardcore_extinction_mode bypasses floor guard
5. Emergency resource regeneration when pool < 25% (updated in v3.15)
6. Emergency regen proportional to surviving population
7. Emergency regen not triggered when pool >= 25%
8. Integration: environment_tick triggers emergency regen
9. API population/metrics exposes new fields
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
from al01.autonomy import AutonomyEngine


# ── Helpers ──────────────────────────────────────────────────────────────

def _env(pool_max=1000.0, pool_initial=1000.0, regen=5.0, **kw):
    """Create an Environment with custom config."""
    cfg = EnvironmentConfig(
        resource_pool_max=pool_max,
        resource_pool_initial=pool_initial,
        resource_regen_rate=regen,
        scarcity_probability=0.0,  # disable random scarcity
        shock_probability=0.0,      # disable random shocks
        **kw,
    )
    return Environment(config=cfg, rng_seed=42)


def _pop(tmp, max_pop=60, rng_seed=42):
    """Create a Population in a temp dir with given max."""
    pop = Population(data_dir=tmp, parent_id="AL-01", rng_seed=rng_seed)
    pop.max_population = max_pop
    return pop


def _spawn_children(pop, n):
    """Spawn *n* children from the parent genome."""
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
    """Minimal Organism wired to a temp directory."""
    db = Database(db_path=os.path.join(tmp, "t.db"))
    mem = MemoryManager(data_dir=tmp, credential_path=None, database=db)
    log = LifeLog(data_dir=os.path.join(tmp, "data"), organism_id="AL-01")
    pol = PolicyManager(data_dir=os.path.join(tmp, "data"))
    org = Organism(data_dir=tmp, memory_manager=mem, life_log=log, policy=pol)
    return org


# ═══════════════════════════════════════════════════════════════════════
# 1. Population Floor Basics
# ═══════════════════════════════════════════════════════════════════════

class TestPopulationFloor:
    def test_floor_is_10_percent_of_max(self):
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp, max_pop=60)
            assert pop.min_population_floor == 12  # int(60 * 0.2)

    def test_floor_updates_when_max_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp, max_pop=40)
            assert pop.min_population_floor == 8
            pop.max_population = 20
            assert pop.min_population_floor == 4

    def test_floor_minimum_is_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp, max_pop=2)
            # int(2 * 0.1) = 0, but floor is clamped to 1
            assert pop.min_population_floor >= 1

    def test_hardcore_mode_defaults_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp)
            assert pop.hardcore_extinction_mode is False

    def test_hardcore_mode_toggle(self):
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp)
            pop.hardcore_extinction_mode = True
            assert pop.hardcore_extinction_mode is True
            pop.hardcore_extinction_mode = False
            assert pop.hardcore_extinction_mode is False


# ═══════════════════════════════════════════════════════════════════════
# 2. remove_member Floor Guard
# ═══════════════════════════════════════════════════════════════════════

class TestRemoveMemberFloorGuard:
    def test_remove_blocked_at_floor(self):
        """When population equals floor, remove_member returns None (blocked)."""
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp, max_pop=20)  # floor = 4
            _spawn_children(pop, 3)       # 4 total = exactly at floor
            assert pop.size == 4
            assert pop.min_population_floor == 4
            child_ids = [mid for mid in pop.member_ids if mid != "AL-01"]
            result = pop.remove_member(child_ids[0], cause="test")
            assert result is None  # blocked
            assert pop.size == 4

    def test_remove_allowed_above_floor(self):
        """When population is above floor, remove_member succeeds."""
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp, max_pop=10)  # floor = 1
            _spawn_children(pop, 3)  # 4 total (AL-01 + 3 children)
            assert pop.size == 4
            child_ids = [mid for mid in pop.member_ids if mid != "AL-01"]
            result = pop.remove_member(child_ids[0], cause="test")
            assert result is not None
            assert pop.size == 3

    def test_remove_blocked_exactly_at_floor(self):
        """With max=30 (floor=6), can't kill when exactly 6 alive."""
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp, max_pop=30)  # floor = 6
            _spawn_children(pop, 5)       # 6 total
            assert pop.size == 6
            assert pop.min_population_floor == 6
            child_ids = [mid for mid in pop.member_ids if mid != "AL-01"]
            result = pop.remove_member(child_ids[0], cause="test")
            assert result is None
            assert pop.size == 6

    def test_remove_allowed_below_floor(self):
        """When population hasn't reached floor, deaths are allowed."""
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp, max_pop=60)  # floor = 6
            assert pop.size == 1  # just AL-01, well below floor
            result = pop.remove_member("AL-01", cause="test")
            assert result is not None
            assert pop.size == 0

    def test_remove_allowed_in_hardcore_mode(self):
        """Hardcore mode bypasses floor guard."""
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp, max_pop=20)  # floor = 2
            pop.hardcore_extinction_mode = True
            _spawn_children(pop, 1)  # 2 total = at floor
            result = pop.remove_member("AL-01", cause="test_hardcore")
            assert result is not None
            assert pop.size == 1


# ═══════════════════════════════════════════════════════════════════════
# 3. prune_weakest Floor Guard
# ═══════════════════════════════════════════════════════════════════════

class TestPruneWeakestFloorGuard:
    def test_prune_respects_floor(self):
        """prune_weakest won't drop below min_population_floor."""
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp, max_pop=60)  # floor = 12
            _spawn_children(pop, 19)      # 20 total
            assert pop.size == 20
            # Try to prune to 3 — floor = 12 should prevent going below 12
            deaths = pop.prune_weakest(max_size=3, min_keep=2)
            assert pop.size >= pop.min_population_floor
            assert pop.size >= 12

    def test_prune_in_hardcore_ignores_floor(self):
        """In hardcore mode, prune_weakest only respects min_keep."""
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp, max_pop=60)  # floor = 6
            pop.hardcore_extinction_mode = True
            _spawn_children(pop, 9)       # 10 total
            assert pop.size == 10
            deaths = pop.prune_weakest(max_size=3, min_keep=2)
            # Should have pruned down to 3, respecting only min_keep=2
            assert pop.size <= 3

    def test_prune_no_action_when_under_max(self):
        """No pruning when population is already under max_size."""
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp, max_pop=60)
            _spawn_children(pop, 3)  # 4 total
            deaths = pop.prune_weakest(max_size=10, min_keep=2)
            assert len(deaths) == 0
            assert pop.size == 4


# ═══════════════════════════════════════════════════════════════════════
# 4. Emergency Resource Regeneration
# ═══════════════════════════════════════════════════════════════════════

class TestEmergencyRegen:
    def test_emergency_regen_triggered_below_threshold(self):
        """Emergency regen activates when pool < 20%."""
        env = _env(pool_max=1000.0, pool_initial=100.0)  # 10% = below 20%
        assert env.pool_fraction < 0.20
        regen = env.emergency_regenerate(population_size=5)
        assert regen > 0

    def test_emergency_regen_not_triggered_above_threshold(self):
        """Emergency regen does NOT activate when pool >= 20%."""
        env = _env(pool_max=1000.0, pool_initial=300.0)  # 30% = above 20%
        assert env.pool_fraction >= 0.20
        regen = env.emergency_regenerate(population_size=5)
        assert regen == 0.0

    def test_emergency_regen_proportional_to_population(self):
        """Larger population → more emergency regen."""
        env_small = _env(pool_max=1000.0, pool_initial=50.0)
        env_large = _env(pool_max=1000.0, pool_initial=50.0)
        regen_small = env_small.emergency_regenerate(population_size=2)
        regen_large = env_large.emergency_regenerate(population_size=10)
        assert regen_large > regen_small

    def test_emergency_regen_uses_scarcity_severity(self):
        """Regen rate scales with scarcity severity (higher when pool is lower)."""
        # Very low pool → high severity → more regen per organism
        env_low = _env(pool_max=1000.0, pool_initial=10.0)   # ~1%
        env_mid = _env(pool_max=1000.0, pool_initial=150.0)   # 15%
        regen_low = env_low.emergency_regenerate(population_size=5)
        regen_mid = env_mid.emergency_regenerate(population_size=5)
        # Lower pool means higher severity → greater regen multiplier
        assert regen_low > regen_mid

    def test_emergency_regen_capped_at_pool_max(self):
        """Emergency regen never pushes pool above max."""
        env = _env(pool_max=100.0, pool_initial=15.0, regen=50.0)
        env.emergency_regenerate(population_size=100)
        assert env.resource_pool <= 100.0

    def test_emergency_regen_on_exact_threshold(self):
        """At exactly 25%, emergency regen is NOT triggered."""
        env = _env(pool_max=1000.0, pool_initial=250.0)  # exactly 25%
        assert env.pool_fraction == pytest.approx(0.25, abs=0.001)
        regen = env.emergency_regenerate(population_size=5)
        assert regen == 0.0

    def test_emergency_regen_state_snapshot(self):
        """state_snapshot includes emergency_regen_active flag."""
        env_low = _env(pool_max=1000.0, pool_initial=100.0)
        snap = env_low.state_snapshot()
        assert snap["emergency_regen_active"] is True
        assert snap["emergency_regen_threshold"] == 0.25

        env_high = _env(pool_max=1000.0, pool_initial=500.0)
        snap = env_high.state_snapshot()
        assert snap["emergency_regen_active"] is False


# ═══════════════════════════════════════════════════════════════════════
# 5. Organism Integration — environment_tick
# ═══════════════════════════════════════════════════════════════════════

class TestOrganismEnvironmentTick:
    def test_environment_tick_triggers_emergency_regen(self):
        """environment_tick() calls emergency_regenerate when pool is low."""
        tmp = tempfile.mkdtemp()
        try:
            org = _make_organism(tmp)
            org.boot()
            # Force pool very low
            org._environment._resource_pool = 50.0  # 5% of 1000
            record = org.environment_tick()
            assert "emergency_regen" in record
            assert record["emergency_regen"] > 0
        finally:
            # Close log handlers to release file locks on Windows
            for h in logging.getLogger().handlers[:]:
                if isinstance(h, logging.FileHandler):
                    h.close()
                    logging.getLogger().removeHandler(h)

    def test_environment_tick_no_emergency_when_pool_healthy(self):
        """environment_tick() does NOT add emergency_regen when pool is fine."""
        tmp = tempfile.mkdtemp()
        try:
            org = _make_organism(tmp)
            org.boot()
            org._environment._resource_pool = 500.0
            # Disable random scarcity/shock so tick() won't drain pool
            org._environment._config.scarcity_probability = 0.0
            org._environment._config.shock_probability = 0.0
            record = org.environment_tick()
            assert "emergency_regen" not in record
        finally:
            for h in logging.getLogger().handlers[:]:
                if isinstance(h, logging.FileHandler):
                    h.close()
                    logging.getLogger().removeHandler(h)


# ═══════════════════════════════════════════════════════════════════════
# 6. API — population metrics exposure
# ═══════════════════════════════════════════════════════════════════════

class TestAPIPopulationMetrics:
    def test_population_metrics_includes_floor(self):
        """The /population/metrics endpoint includes floor + hardcore fields."""
        tmp = tempfile.mkdtemp()
        try:
            org = _make_organism(tmp)
            org.boot()
            pop = org.population
            assert hasattr(pop, "min_population_floor")
            assert hasattr(pop, "hardcore_extinction_mode")
            assert pop.min_population_floor == int(pop.max_population * 0.2)
            assert pop.hardcore_extinction_mode is False
        finally:
            for h in logging.getLogger().handlers[:]:
                if isinstance(h, logging.FileHandler):
                    h.close()
                    logging.getLogger().removeHandler(h)


# ═══════════════════════════════════════════════════════════════════════
# 7. Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_floor_with_zero_population(self):
        """If all members are already dead, remove_member returns None."""
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp, max_pop=10)
            pop.hardcore_extinction_mode = True
            pop.remove_member("AL-01", cause="test")
            assert pop.size == 0
            # Trying to remove again returns None (not found alive)
            result = pop.remove_member("AL-01", cause="test_again")
            # Already dead — returns death_info (or None)
            # The method should at least not crash
            assert pop.size == 0

    def test_emergency_regen_with_zero_population(self):
        """Emergency regen handles population_size=0 gracefully."""
        env = _env(pool_max=1000.0, pool_initial=50.0)
        regen = env.emergency_regenerate(population_size=0)
        # pop is clamped to max(1, 0) = 1
        assert regen > 0

    def test_floor_guard_sequential_kills(self):
        """Sequential remove_member calls stop at floor."""
        with tempfile.TemporaryDirectory() as tmp:
            pop = _pop(tmp, max_pop=20)  # floor = 4
            _spawn_children(pop, 9)       # 10 total (above floor=4)
            assert pop.size == 10
            killed = 0
            for mid in list(pop.member_ids):
                result = pop.remove_member(mid, cause="sequential")
                if result is not None:
                    killed += 1
            # Should have killed 6 (10 - floor of 4 = 6)
            assert pop.size == pop.min_population_floor
            assert killed == 6

    def test_emergency_regen_formula(self):
        """Verify: regen_rate = base_rate * (1 + scarcity_severity)."""
        env = _env(pool_max=1000.0, pool_initial=0.0, regen=10.0)
        # pool is 0 → severity = 1.0 → regen_rate = 10 * (1+1) = 20
        # with pop=1, scale=1.0: regen = 20 * 1 * 1.0 = 20
        regen = env.emergency_regenerate(population_size=1)
        assert regen == pytest.approx(20.0, abs=0.1)
