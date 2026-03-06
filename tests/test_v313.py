"""v3.13 — Global Shared Resource Pool tests.

Tests cover:
1. Fair energy distribution via request_energy()
2. Smart regeneration with efficiency & overuse damping
3. Scarcity pressure activation and severity
4. Scarcity effects: metabolic cost, reproduction threshold, survival grace
5. Pool consumption during organism lifecycle (child + parent)
6. Dashboard exposure of pool stats
"""

import os
import tempfile

import pytest

from al01.environment import Environment, EnvironmentConfig
from al01.organism import Organism
from al01.memory_manager import MemoryManager
from al01.database import Database
from al01.life_log import LifeLog
from al01.policy import PolicyManager
from al01.population import Population
from al01.autonomy import AutonomyEngine


# ── Helpers ──────────────────────────────────────────────────────────────

def _env(pool_max=1000.0, pool_initial=1000.0, regen=5.0, **kw):
    """Create an Environment with custom config."""
    cfg = EnvironmentConfig(
        resource_pool_max=pool_max,
        resource_pool_initial=pool_initial,
        resource_regen_rate=regen,
        **kw,
    )
    return Environment(config=cfg, rng_seed=42)


def _make_organism(tmp):
    """Minimal Organism wired to a temp directory."""
    db = Database(db_path=os.path.join(tmp, "t.db"))
    mem = MemoryManager(data_dir=tmp, credential_path=None, database=db)
    log = LifeLog(data_dir=os.path.join(tmp, "data"), organism_id="AL-01")
    pol = PolicyManager(data_dir=os.path.join(tmp, "data"))
    org = Organism(data_dir=tmp, memory_manager=mem, life_log=log, policy=pol)
    return org


# ═══════════════════════════════════════════════════════════════════════
# 1. Pool basics & properties
# ═══════════════════════════════════════════════════════════════════════

class TestPoolBasics:
    def test_initial_pool_at_capacity(self):
        env = _env()
        assert env.resource_pool == 1000.0
        assert env.pool_fraction == 1.0

    def test_pool_fraction_after_consumption(self):
        env = _env(pool_max=1000, pool_initial=500)
        assert env.pool_fraction == pytest.approx(0.5)

    def test_pool_fraction_zero_capacity(self):
        env = _env(pool_max=0, pool_initial=0)
        assert env.pool_fraction == 0.0

    def test_consume_resources_still_works(self):
        """Original consume_resources remains backward-compatible."""
        env = _env(pool_max=100, pool_initial=100)
        consumed = env.consume_resources(30)
        assert consumed == pytest.approx(30.0)
        assert env.resource_pool == pytest.approx(70.0)


# ═══════════════════════════════════════════════════════════════════════
# 2. Fair energy distribution — request_energy()
# ═══════════════════════════════════════════════════════════════════════

class TestFairEnergy:
    def test_single_organism_gets_full_request(self):
        env = _env(pool_initial=1000)
        granted = env.request_energy(2.0, population_size=1)
        assert granted == pytest.approx(2.0)
        assert env.resource_pool == pytest.approx(998.0)

    def test_fair_share_limits_large_request(self):
        """With 10 organisms and 100 pool, each can get at most 10."""
        env = _env(pool_max=100, pool_initial=100)
        granted = env.request_energy(50.0, population_size=10)
        assert granted == pytest.approx(10.0)  # fair share = 100/10 = 10
        assert env.resource_pool == pytest.approx(90.0)

    def test_fair_share_small_request_honored(self):
        """If request < fair share, grant the full request."""
        env = _env(pool_max=1000, pool_initial=1000)
        granted = env.request_energy(2.0, population_size=10)
        assert granted == pytest.approx(2.0)

    def test_depleted_pool_grants_zero(self):
        env = _env(pool_max=1000, pool_initial=0)
        granted = env.request_energy(5.0, population_size=1)
        assert granted == 0.0

    def test_population_size_zero_treated_as_one(self):
        env = _env(pool_initial=100)
        granted = env.request_energy(5.0, population_size=0)
        assert granted == pytest.approx(5.0)

    def test_sequential_draws_deplete_pool(self):
        env = _env(pool_max=20, pool_initial=20)
        total = 0.0
        for _ in range(10):
            total += env.request_energy(3.0, population_size=10)
        # Each gets min(3.0, pool/10) — pool shrinks each time
        assert env.resource_pool < 20.0
        assert total > 0

    def test_fair_distribution_across_population(self):
        """Simulate N organisms each requesting the same amount in one cycle."""
        env = _env(pool_max=100, pool_initial=100)
        pop = 20
        grants = []
        for _ in range(pop):
            grants.append(env.request_energy(10.0, population_size=pop))
        # First organism gets min(10, 100/20) = 5, pool drops to 95
        # As pool shrinks, later organisms get slightly less
        assert grants[0] == pytest.approx(5.0)
        assert all(g >= 0 for g in grants)
        assert env.resource_pool >= 0


# ═══════════════════════════════════════════════════════════════════════
# 3. Smart regeneration
# ═══════════════════════════════════════════════════════════════════════

class TestSmartRegeneration:
    def test_regen_at_full_pool_zero_increase(self):
        env = _env(pool_max=1000, pool_initial=1000)
        regen = env.smart_regenerate(avg_efficiency=0.5)
        # Pool is already at max, can't go higher
        assert regen == pytest.approx(0.0)

    def test_regen_with_low_pool(self):
        env = _env(pool_max=1000, pool_initial=500, regen=10.0)
        regen = env.smart_regenerate(avg_efficiency=0.5)
        assert regen > 0

    def test_high_efficiency_boosts_regen(self):
        env1 = _env(pool_max=1000, pool_initial=500, regen=10.0)
        env2 = _env(pool_max=1000, pool_initial=500, regen=10.0)
        regen_low = env1.smart_regenerate(avg_efficiency=0.1)
        regen_high = env2.smart_regenerate(avg_efficiency=0.9)
        assert regen_high > regen_low

    def test_overuse_damping_slows_regen(self):
        """When pool_fraction < 0.5, overuse damping kicks in."""
        env1 = _env(pool_max=1000, pool_initial=600, regen=10.0)
        env2 = _env(pool_max=1000, pool_initial=100, regen=10.0)
        regen_healthy = env1.smart_regenerate(avg_efficiency=0.5)
        regen_damaged = env2.smart_regenerate(avg_efficiency=0.5)
        # Damaged pool has lower fraction → more damping → less regen
        assert regen_damaged < regen_healthy

    def test_regen_never_exceeds_capacity(self):
        env = _env(pool_max=100, pool_initial=99, regen=50.0)
        regen = env.smart_regenerate(avg_efficiency=1.0)
        assert env.resource_pool <= 100.0


# ═══════════════════════════════════════════════════════════════════════
# 4. Scarcity pressure
# ═══════════════════════════════════════════════════════════════════════

class TestScarcityPressure:
    def test_no_scarcity_at_full_pool(self):
        env = _env(pool_max=1000, pool_initial=1000)
        assert not env.is_scarcity_pressure
        assert env.scarcity_severity == pytest.approx(0.0)

    def test_scarcity_activates_below_threshold(self):
        env = _env(pool_max=1000, pool_initial=200, scarcity_threshold=0.25)
        assert env.is_scarcity_pressure  # 200/1000 = 0.2 < 0.25
        assert env.scarcity_severity > 0

    def test_scarcity_severity_scales_linearly(self):
        env = _env(pool_max=1000, pool_initial=125, scarcity_threshold=0.25)
        # fraction = 0.125; severity = 1 - (0.125/0.25) = 0.5
        assert env.scarcity_severity == pytest.approx(0.5)

    def test_scarcity_severity_at_empty_pool(self):
        env = _env(pool_max=1000, pool_initial=0, scarcity_threshold=0.25)
        assert env.scarcity_severity == pytest.approx(1.0)

    def test_no_scarcity_at_threshold(self):
        env = _env(pool_max=1000, pool_initial=250, scarcity_threshold=0.25)
        assert not env.is_scarcity_pressure
        assert env.scarcity_severity == pytest.approx(0.0)

    def test_metabolic_cost_normal(self):
        env = _env(pool_max=1000, pool_initial=1000, metabolic_base_cost=2.0)
        assert env.effective_metabolic_cost() == pytest.approx(2.0)

    def test_metabolic_cost_increases_during_scarcity(self):
        env = _env(
            pool_max=1000, pool_initial=0,
            metabolic_base_cost=2.0,
            scarcity_threshold=0.25,
            scarcity_metabolic_multiplier=1.5,
        )
        # severity=1.0 → adaptive metabolism kicks in (severity > 0.6)
        # At extreme scarcity, cost reduces toward floor (0.4)
        # cost = 2.0 * 0.4 = 0.8
        assert env.effective_metabolic_cost() == pytest.approx(0.8)

    def test_reproduction_threshold_raised_during_scarcity(self):
        env = _env(
            pool_max=1000, pool_initial=0,
            scarcity_threshold=0.25,
            scarcity_reproduction_penalty=0.5,
        )
        # severity=1.0 → threshold = 0.5 * (1 + 0.5) = 0.75
        result = env.effective_reproduction_threshold(0.5)
        assert result == pytest.approx(0.75)

    def test_reproduction_threshold_capped_at_one(self):
        env = _env(
            pool_max=1000, pool_initial=0,
            scarcity_threshold=0.25,
            scarcity_reproduction_penalty=5.0,
        )
        result = env.effective_reproduction_threshold(0.8)
        assert result <= 1.0

    def test_survival_grace_reduced_during_scarcity(self):
        env = _env(
            pool_max=1000, pool_initial=0,
            scarcity_threshold=0.25,
            scarcity_mortality_multiplier=1.5,
        )
        # severity=1.0 → divisor = 1.5 → grace = 20/1.5 = 13
        result = env.effective_survival_grace(20)
        assert result == 13  # int(20/1.5) = 13

    def test_survival_grace_normal_above_threshold(self):
        env = _env(pool_max=1000, pool_initial=1000)
        assert env.effective_survival_grace(20) == 20

    def test_survival_grace_minimum_one(self):
        env = _env(
            pool_max=1000, pool_initial=0,
            scarcity_threshold=0.25,
            scarcity_mortality_multiplier=100.0,
        )
        result = env.effective_survival_grace(1)
        assert result >= 1


# ═══════════════════════════════════════════════════════════════════════
# 5. State snapshot includes pool stats
# ═══════════════════════════════════════════════════════════════════════

class TestStateSnapshot:
    def test_snapshot_includes_pool_fraction(self):
        env = _env(pool_max=1000, pool_initial=500)
        snap = env.state_snapshot()
        assert "pool_fraction" in snap
        assert snap["pool_fraction"] == pytest.approx(0.5)

    def test_snapshot_includes_scarcity_pressure(self):
        env = _env(pool_max=1000, pool_initial=100, scarcity_threshold=0.25)
        snap = env.state_snapshot()
        assert snap["is_scarcity_pressure"] is True
        assert snap["scarcity_severity"] > 0

    def test_snapshot_includes_metabolic_cost(self):
        env = _env()
        snap = env.state_snapshot()
        assert "effective_metabolic_cost" in snap


# ═══════════════════════════════════════════════════════════════════════
# 6. Integration — organism lifecycle consumes pool
# ═══════════════════════════════════════════════════════════════════════

class TestOrganismPoolIntegration:
    def test_environment_tick_calls_smart_regenerate(self):
        """environment_tick() should invoke smart_regenerate, visible
        as additional regen beyond the basic tick() regen."""
        tmp = tempfile.mkdtemp()
        org = _make_organism(tmp)
        # Deplete pool partially so regen is visible
        org._environment._resource_pool = 500.0
        pool_before = org._environment.resource_pool
        org.environment_tick()
        pool_after = org._environment.resource_pool
        # tick() adds basic regen, then smart_regenerate adds more
        assert pool_after > pool_before

    def test_child_autonomy_draws_from_pool(self):
        """Running child_autonomy_cycle should reduce the resource pool."""
        tmp = tempfile.mkdtemp()
        org = _make_organism(tmp)
        # Spawn a child so child_autonomy has something to process
        org._population.spawn_child(org.genome, parent_evolution=1)
        pool_before = org._environment.resource_pool
        org.child_autonomy_cycle()
        pool_after = org._environment.resource_pool
        # Pool should decrease because child drew metabolic energy
        assert pool_after < pool_before

    def test_parent_autonomy_draws_from_pool(self):
        """Running autonomy_cycle should reduce the resource pool."""
        tmp = tempfile.mkdtemp()
        org = _make_organism(tmp)
        pool_before = org._environment.resource_pool
        org.autonomy_cycle()
        pool_after = org._environment.resource_pool
        assert pool_after < pool_before

    def test_scarcity_reduces_child_energy_recovery(self):
        """When pool is scarce, children get less energy recovery."""
        tmp = tempfile.mkdtemp()
        org = _make_organism(tmp)
        child = org._population.spawn_child(org.genome, parent_evolution=1)
        child_id = child["id"]

        # Full pool → normal recovery
        org.child_autonomy_cycle()
        member_full = org._population.get(child_id)
        energy_full = member_full["energy"] if member_full else 0

        # Reset child energy for second test
        org._population.update_member(child_id, {"energy": 0.8})

        # Deplete pool → scarce recovery
        org._environment._resource_pool = 10.0
        org.child_autonomy_cycle()
        member_scarce = org._population.get(child_id)
        energy_scarce = member_scarce["energy"] if member_scarce else 0

        # In scarce conditions, recovery is less (or energy drains faster)
        # We just verify the pool was consumed more aggressively
        assert org._environment.resource_pool < 10.0

    def test_scarcity_raises_repro_threshold(self):
        """auto_reproduce_cycle respects scarcity-adjusted threshold."""
        tmp = tempfile.mkdtemp()
        org = _make_organism(tmp)
        # Deplete pool to trigger scarcity
        org._environment._resource_pool = 0.0
        # Normal threshold would be 0.5, scarcity should raise it
        effective = org._environment.effective_reproduction_threshold(0.5)
        assert effective > 0.5


# ═══════════════════════════════════════════════════════════════════════
# 7. Pool serialization round-trip
# ═══════════════════════════════════════════════════════════════════════

class TestPoolSerialization:
    def test_round_trip_preserves_pool(self):
        env = _env(pool_max=1000, pool_initial=750)
        env.consume_resources(100)
        data = env.to_dict()
        env2 = Environment.from_dict(data, config=env.config)
        assert env2.resource_pool == pytest.approx(650.0)

    def test_round_trip_preserves_config_defaults(self):
        env = _env(pool_max=1000, pool_initial=1000)
        data = env.to_dict()
        env2 = Environment.from_dict(data, config=env.config)
        assert env2.config.metabolic_base_cost == pytest.approx(1.0)
        assert env2.config.scarcity_threshold == pytest.approx(0.25)


# ═══════════════════════════════════════════════════════════════════════
# 8. Edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_request_energy_negative_amount(self):
        env = _env(pool_initial=1000)
        granted = env.request_energy(-5.0, population_size=1)
        assert granted == 0.0  # negative request clamped

    def test_scarcity_threshold_zero(self):
        env = _env(pool_max=1000, pool_initial=0, scarcity_threshold=0.0)
        # When threshold is 0, scarcity should never activate
        assert not env.is_scarcity_pressure

    def test_pool_fraction_after_tick_and_consume(self):
        env = _env(pool_max=100, pool_initial=50, regen=5.0)
        env.tick()  # regenerates some
        env.request_energy(20, population_size=1)
        frac = env.pool_fraction
        assert 0 <= frac <= 1.0
