"""v3.18 — Ecosystem Stabilisation Rules.

Features:
 1. Resource pool minimum floor — pool never drops below configured floor.
 2. Reproduction cap enforcement — already existed (spawn_child caps); documented.
 3. Energy requirement for reproduction — auto, stability, lone-survivor all
    require minimum parent energy and deduct a cost after spawning.
 4. Reproduction probability scales with resource pool — pool_fraction is
    multiplied into base probability for stability, lone-survivor, and rare.
"""

import logging
import os
import random
import tempfile

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
        pop.set_tick(i)
        pop.spawn_child(genome, parent_evolution=0, parent_id="AL-01")
    parent["last_birth_tick"] = -BIRTH_COOLDOWN_CYCLES
    pop.set_tick(saved_tick)


def _make_organism(tmp, env=None):
    db = Database(db_path=os.path.join(tmp, "t.db"))
    mem = MemoryManager(data_dir=tmp, credential_path=None, database=db)
    log = LifeLog(data_dir=os.path.join(tmp, "data"), organism_id="AL-01")
    pol = PolicyManager(data_dir=os.path.join(tmp, "data"))
    org = Organism(data_dir=tmp, memory_manager=mem, life_log=log,
                   policy=pol, environment=env)
    return org


def _cleanup_handlers():
    for h in logging.getLogger().handlers[:]:
        if isinstance(h, logging.FileHandler):
            h.close()
            logging.getLogger().removeHandler(h)


# ═══════════════════════════════════════════════════════════════════════
# 1. Resource Pool Minimum Floor
# ═══════════════════════════════════════════════════════════════════════

class TestResourcePoolFloor:
    """Pool never drops below resource_pool_min_floor."""

    def test_default_floor_is_zero(self):
        """Default config has floor=0 for backward compatibility."""
        cfg = EnvironmentConfig()
        assert cfg.resource_pool_min_floor == 0.0

    def test_consume_respects_floor(self):
        """consume_resources cannot drain pool below the floor."""
        env = _env(pool_max=100, pool_initial=100, resource_pool_min_floor=50.0)
        consumed = env.consume_resources(80)
        assert consumed == pytest.approx(50.0)
        assert env.resource_pool == pytest.approx(50.0)

    def test_consume_when_pool_at_floor(self):
        """When pool equals floor, consume returns 0."""
        env = _env(pool_max=100, pool_initial=50, resource_pool_min_floor=50.0)
        consumed = env.consume_resources(10)
        assert consumed == pytest.approx(0.0)
        assert env.resource_pool == pytest.approx(50.0)

    def test_consume_partial_above_floor(self):
        """Consume up to the available amount above floor."""
        env = _env(pool_max=100, pool_initial=60, resource_pool_min_floor=50.0)
        consumed = env.consume_resources(20)
        assert consumed == pytest.approx(10.0)
        assert env.resource_pool == pytest.approx(50.0)

    def test_request_energy_respects_floor(self):
        """request_energy cannot drain pool below floor."""
        env = _env(pool_max=100, pool_initial=60, resource_pool_min_floor=50.0)
        granted = env.request_energy(20.0, population_size=1)
        assert granted == pytest.approx(10.0)
        assert env.resource_pool == pytest.approx(50.0)

    def test_request_energy_at_floor(self):
        """request_energy grants 0 when pool is at floor."""
        env = _env(pool_max=100, pool_initial=50, resource_pool_min_floor=50.0)
        granted = env.request_energy(5.0, population_size=1)
        assert granted == pytest.approx(0.0)

    def test_request_energy_fair_share_with_floor(self):
        """Fair share is computed from available (pool - floor), not total pool."""
        env = _env(pool_max=200, pool_initial=150, resource_pool_min_floor=50.0)
        # available = 150 - 50 = 100, fair_share = 100 / 10 = 10
        granted = env.request_energy(50.0, population_size=10)
        assert granted == pytest.approx(10.0)
        assert env.resource_pool == pytest.approx(140.0)

    def test_tick_enforces_floor(self):
        """After tick(), pool is at least floor even if regen is 0."""
        env = _env(pool_max=100, pool_initial=30, regen=0.0, resource_pool_min_floor=50.0)
        env.tick()
        assert env.resource_pool >= 50.0

    def test_zero_floor_allows_full_drain(self):
        """With floor=0, pool can reach 0 (backward compat)."""
        env = _env(pool_max=100, pool_initial=5, resource_pool_min_floor=0.0)
        consumed = env.consume_resources(10.0)
        assert consumed == pytest.approx(5.0)
        assert env.resource_pool == pytest.approx(0.0)

    def test_floor_does_not_block_regeneration(self):
        """Pool can regenerate above the floor via smart_regenerate."""
        env = _env(pool_max=1000, pool_initial=50, regen=10.0, resource_pool_min_floor=50.0)
        regen = env.smart_regenerate(avg_efficiency=0.5)
        assert regen > 0
        assert env.resource_pool > 50.0

    def test_floor_below_pool_has_no_effect_on_consume(self):
        """When pool is well above floor, consume behaves normally."""
        env = _env(pool_max=1000, pool_initial=1000, resource_pool_min_floor=50.0)
        consumed = env.consume_resources(100)
        assert consumed == pytest.approx(100.0)
        assert env.resource_pool == pytest.approx(900.0)


# ═══════════════════════════════════════════════════════════════════════
# 2. Reproduction Cap Enforcement (pre-existing — regression tests)
# ═══════════════════════════════════════════════════════════════════════

class TestReproductionCapEnforcement:
    """spawn_child enforces both dynamic and absolute population caps."""

    def test_spawn_refused_at_absolute_cap(self):
        """spawn_child refuses when pop >= ABSOLUTE_POPULATION_CAP."""
        tmp = tempfile.mkdtemp()
        try:
            pop = _pop(tmp, max_pop=ABSOLUTE_POPULATION_CAP + 10)
            _spawn_children(pop, ABSOLUTE_POPULATION_CAP - 1)  # fill to cap
            assert pop.size == ABSOLUTE_POPULATION_CAP

            parent = pop._members["AL-01"]
            parent["last_birth_tick"] = -BIRTH_COOLDOWN_CYCLES * 2
            pop.set_tick(99999)
            genome = Genome.from_dict(parent["genome"])
            child = pop.spawn_child(genome, parent_evolution=0, parent_id="AL-01")
            assert child is None
        finally:
            _cleanup_handlers()

    def test_spawn_refused_at_dynamic_cap(self):
        """spawn_child refuses when pop >= max_population."""
        tmp = tempfile.mkdtemp()
        try:
            pop = _pop(tmp, max_pop=5)
            _spawn_children(pop, 4)
            assert pop.size == 5

            parent = pop._members["AL-01"]
            parent["last_birth_tick"] = -BIRTH_COOLDOWN_CYCLES * 2
            pop.set_tick(99999)
            genome = Genome.from_dict(parent["genome"])
            child = pop.spawn_child(genome, parent_evolution=0, parent_id="AL-01")
            assert child is None
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 3. Energy Requirement for Reproduction
# ═══════════════════════════════════════════════════════════════════════

class TestReproductionEnergyGate:
    """All reproduction paths require minimum parent energy."""

    # -- auto_reproduce energy gate --

    def test_auto_repro_blocked_low_energy(self):
        """auto_reproduce returns None when parent energy < threshold."""
        tmp = tempfile.mkdtemp()
        try:
            pop = _pop(tmp)
            # Make AL-01 qualified: high fitness + consecutive
            pop._members["AL-01"]["consecutive_above_repro"] = 10
            rec = pop.get("AL-01")
            assert rec is not None
            g = Genome.from_dict(rec["genome"])
            for t in g.traits:
                g.set_trait(t, 0.95)
            pop.update_member("AL-01", {"genome": g.to_dict()})

            # Set energy below gate
            pop._members["AL-01"]["energy"] = 0.30

            pop.set_tick(BIRTH_COOLDOWN_CYCLES + 1)
            child = pop.auto_reproduce("AL-01", energy_min=0.50)
            assert child is None
        finally:
            _cleanup_handlers()

    def test_auto_repro_succeeds_high_energy(self):
        """auto_reproduce succeeds when parent has sufficient energy."""
        tmp = tempfile.mkdtemp()
        try:
            pop = _pop(tmp)
            pop._members["AL-01"]["consecutive_above_repro"] = 10
            rec = pop.get("AL-01")
            assert rec is not None
            g = Genome.from_dict(rec["genome"])
            for t in g.traits:
                g.set_trait(t, 0.95)
            pop.update_member("AL-01", {"genome": g.to_dict()})
            pop._members["AL-01"]["energy"] = 0.80

            pop.set_tick(BIRTH_COOLDOWN_CYCLES + 1)
            child = pop.auto_reproduce("AL-01", energy_min=0.50, energy_cost=0.20)
            assert child is not None
            # Parent energy deducted
            assert pop._members["AL-01"]["energy"] == pytest.approx(0.60)
        finally:
            _cleanup_handlers()

    def test_auto_repro_energy_cost_deducted(self):
        """After auto_reproduce, parent energy is reduced by cost."""
        tmp = tempfile.mkdtemp()
        try:
            pop = _pop(tmp)
            pop._members["AL-01"]["consecutive_above_repro"] = 10
            rec = pop.get("AL-01")
            assert rec is not None
            g = Genome.from_dict(rec["genome"])
            for t in g.traits:
                g.set_trait(t, 0.95)
            pop.update_member("AL-01", {"genome": g.to_dict()})
            pop._members["AL-01"]["energy"] = 1.0

            pop.set_tick(BIRTH_COOLDOWN_CYCLES + 1)
            pop.auto_reproduce("AL-01", energy_min=0.50, energy_cost=0.30)
            assert pop._members["AL-01"]["energy"] == pytest.approx(0.70)
        finally:
            _cleanup_handlers()

    # -- stability_reproduction energy gate --

    def test_stability_repro_blocked_low_energy(self):
        """stability_reproduction_cycle skips organisms with low energy."""
        tmp = tempfile.mkdtemp()
        try:
            env = _env(pool_max=1000, pool_initial=1000,
                       stability_repro_probability=1.0,
                       repro_energy_min_stability=0.60)
            org = _make_organism(tmp, env=env)
            org.boot()

            # Set AL-01 to high fitness but low energy
            rec = org._population.get("AL-01")
            assert rec is not None
            g = Genome.from_dict(rec["genome"])
            for t in g.traits:
                g.set_trait(t, 0.95)
            org._population.update_member("AL-01", {"genome": g.to_dict()})
            org._population._members["AL-01"]["energy"] = 0.30

            spawned = org.stability_reproduction_cycle()
            assert len(spawned) == 0
        finally:
            _cleanup_handlers()

    def test_stability_repro_succeeds_high_energy(self):
        """stability_reproduction_cycle spawns when energy is sufficient."""
        tmp = tempfile.mkdtemp()
        try:
            env = _env(pool_max=1000, pool_initial=1000,
                       stability_repro_probability=1.0,
                       repro_energy_min_stability=0.60,
                       repro_energy_cost=0.20)
            org = _make_organism(tmp, env=env)
            org.boot()

            rec = org._population.get("AL-01")
            assert rec is not None
            g = Genome.from_dict(rec["genome"])
            for t in g.traits:
                g.set_trait(t, 0.95)
            org._population.update_member("AL-01", {"genome": g.to_dict()})
            org._population._members["AL-01"]["energy"] = 0.90

            spawned = org.stability_reproduction_cycle()
            assert len(spawned) > 0
            # Energy deducted from parent
            assert org._population._members["AL-01"]["energy"] == pytest.approx(0.70)
        finally:
            _cleanup_handlers()

    # -- lone_survivor energy gate --

    def test_lone_survivor_blocked_low_energy(self):
        """lone_survivor_reproduction returns None when energy < threshold."""
        tmp = tempfile.mkdtemp()
        try:
            env = _env(pool_max=1000, pool_initial=1000,
                       lone_survivor_repro_probability=1.0,
                       repro_energy_min_lone_survivor=0.40)
            org = _make_organism(tmp, env=env)
            org.boot()
            assert org._population.size == 1

            org._population._members["AL-01"]["energy"] = 0.20
            result = org.lone_survivor_reproduction()
            assert result is None
        finally:
            _cleanup_handlers()

    def test_lone_survivor_succeeds_high_energy(self):
        """lone_survivor_reproduction spawns when energy >= threshold."""
        tmp = tempfile.mkdtemp()
        try:
            env = _env(pool_max=1000, pool_initial=1000,
                       lone_survivor_repro_probability=1.0,
                       repro_energy_min_lone_survivor=0.40,
                       repro_energy_cost=0.20)
            org = _make_organism(tmp, env=env)
            org.boot()
            assert org._population.size == 1

            org._population._members["AL-01"]["energy"] = 0.80
            result = org.lone_survivor_reproduction()
            assert result is not None
            assert org._population._members["AL-01"]["energy"] == pytest.approx(0.60)
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 4. Reproduction Probability Scales with Resource Pool
# ═══════════════════════════════════════════════════════════════════════

class TestReproductionProbabilityScaling:
    """Reproduction probability is multiplied by pool_fraction."""

    def test_stability_repro_probability_scaled(self):
        """At 50% pool, effective stability prob = base * 0.5."""
        tmp = tempfile.mkdtemp()
        try:
            # Pool at 50%
            env = _env(pool_max=1000, pool_initial=500,
                       stability_repro_probability=1.0,
                       stability_repro_pool_fraction=0.0,  # disable pool gate
                       repro_energy_min_stability=0.0)
            org = _make_organism(tmp, env=env)
            org.boot()

            # High fitness so threshold passes
            rec = org._population.get("AL-01")
            assert rec is not None
            g = Genome.from_dict(rec["genome"])
            for t in g.traits:
                g.set_trait(t, 0.95)
            org._population.update_member("AL-01", {"genome": g.to_dict()})

            # With pool_fraction=0.5 and base_prob=1.0,
            # effective_prob = 0.5. Over 100 runs, some should fail.
            successes = 0
            for trial in range(100):
                org._population.set_tick(trial * BIRTH_COOLDOWN_CYCLES + 1000)
                org._population._members["AL-01"]["last_birth_tick"] = -BIRTH_COOLDOWN_CYCLES * 2
                spawned = org.stability_reproduction_cycle()
                if spawned:
                    successes += 1
                    # Remove children to keep pop=1
                    for s in spawned:
                        org._population._members.pop(s["id"], None)

            # At 50% effective probability, expect ~50 successes (±20 for margin)
            assert 20 < successes < 80
        finally:
            _cleanup_handlers()

    def test_stability_repro_at_full_pool_no_reduction(self):
        """At 100% pool, effective prob = base prob (no reduction)."""
        tmp = tempfile.mkdtemp()
        try:
            env = _env(pool_max=1000, pool_initial=1000,
                       stability_repro_probability=1.0,
                       repro_energy_min_stability=0.0)
            org = _make_organism(tmp, env=env)
            org.boot()

            rec = org._population.get("AL-01")
            assert rec is not None
            g = Genome.from_dict(rec["genome"])
            for t in g.traits:
                g.set_trait(t, 0.95)
            org._population.update_member("AL-01", {"genome": g.to_dict()})

            org._population.set_tick(BIRTH_COOLDOWN_CYCLES + 1)
            spawned = org.stability_reproduction_cycle()
            # With 100% effective probability, should always spawn
            assert len(spawned) > 0
        finally:
            _cleanup_handlers()

    def test_lone_survivor_prob_scaled_by_pool(self):
        """lone_survivor effective probability = base * pool_fraction."""
        tmp = tempfile.mkdtemp()
        try:
            # Pool at 80% > 70% threshold
            env = _env(pool_max=1000, pool_initial=800,
                       lone_survivor_repro_probability=1.0,
                       repro_energy_min_lone_survivor=0.0,
                       repro_energy_cost=0.0)
            org = _make_organism(tmp, env=env)
            org.boot()
            assert org._population.size == 1

            successes = 0
            for trial in range(100):
                org._population.set_tick(trial * BIRTH_COOLDOWN_CYCLES + 1000)
                org._population._members["AL-01"]["last_birth_tick"] = -BIRTH_COOLDOWN_CYCLES * 2
                result = org.lone_survivor_reproduction()
                if result:
                    successes += 1
                    org._population._members.pop(result["id"], None)

            # effective_prob = 1.0 * 0.8 = 0.8 → expect ~80 successes
            assert 60 < successes < 95
        finally:
            _cleanup_handlers()

    def test_zero_pool_prevents_reproduction(self):
        """At pool=0 (floor=0), all probability-scaled repro returns 0."""
        env = _env(pool_max=1000, pool_initial=0, resource_pool_min_floor=0.0)
        # pool_fraction = 0 → effective_prob = anything * 0 = 0
        assert env.pool_fraction == 0.0


# ═══════════════════════════════════════════════════════════════════════
# 5. Config Defaults
# ═══════════════════════════════════════════════════════════════════════

class TestV318ConfigDefaults:
    """New v3.18 config fields have correct defaults."""

    def test_pool_floor_default(self):
        cfg = EnvironmentConfig()
        assert cfg.resource_pool_min_floor == 0.0

    def test_repro_energy_min_auto_default(self):
        cfg = EnvironmentConfig()
        assert cfg.repro_energy_min_auto == 0.50

    def test_repro_energy_min_stability_default(self):
        cfg = EnvironmentConfig()
        assert cfg.repro_energy_min_stability == 0.60

    def test_repro_energy_min_lone_survivor_default(self):
        cfg = EnvironmentConfig()
        assert cfg.repro_energy_min_lone_survivor == 0.40

    def test_repro_energy_cost_default(self):
        cfg = EnvironmentConfig()
        assert cfg.repro_energy_cost == 0.20

    def test_deduct_energy_helper(self):
        """Population.deduct_energy reduces member energy."""
        tmp = tempfile.mkdtemp()
        try:
            pop = _pop(tmp)
            pop._members["AL-01"]["energy"] = 1.0
            pop.deduct_energy("AL-01", 0.30)
            assert pop._members["AL-01"]["energy"] == pytest.approx(0.70)
        finally:
            _cleanup_handlers()

    def test_deduct_energy_floors_at_zero(self):
        """deduct_energy cannot make energy negative."""
        tmp = tempfile.mkdtemp()
        try:
            pop = _pop(tmp)
            pop._members["AL-01"]["energy"] = 0.10
            pop.deduct_energy("AL-01", 0.50)
            assert pop._members["AL-01"]["energy"] == pytest.approx(0.0)
        finally:
            _cleanup_handlers()

    def test_deduct_energy_missing_member(self):
        """deduct_energy silently ignores unknown organism_id."""
        tmp = tempfile.mkdtemp()
        try:
            pop = _pop(tmp)
            pop.deduct_energy("nonexistent", 1.0)  # should not raise
        finally:
            _cleanup_handlers()
