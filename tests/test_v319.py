"""v3.19 — Fix double mutation in rare reproduction.

Bug:
  rare_reproduction_cycle() called parent_genome.spawn_child(variance=0.10)
  to create a pre-mutated child genome, then passed that genome to
  population.spawn_child() which called .spawn_child(variance=0.05) AGAIN —
  resulting in double mutation (±10% then ±5%).

Fix:
  population.spawn_child() now accepts a mutation_variance parameter
  (default 0.05). rare_reproduction_cycle() passes the original parent
  genome with mutation_variance=RARE_REPRO_MUTATION_RATE (0.10), so
  mutation is applied exactly once.
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


def _make_organism(tmp, env=None):
    db = Database(db_path=os.path.join(tmp, "t.db"))
    mem = MemoryManager(data_dir=tmp, credential_path=None, database=db)
    log = LifeLog(data_dir=os.path.join(tmp, "data"), organism_id="AL-01")
    pol = PolicyManager(data_dir=os.path.join(tmp, "data"))
    org = Organism(data_dir=tmp, memory_manager=mem, life_log=log,
                   policy=pol, environment=env)
    return org


def _set_eligible_parent(pop, pid, energy=0.80):
    """Make a parent eligible for rare reproduction."""
    pop.update_energy(pid, energy)
    # Build a healthy, non-stagnating fitness history
    for f in [0.5 + i * 0.01 for i in range(20)]:
        pop.record_fitness(pid, f)


def _cleanup_handlers():
    for h in logging.getLogger().handlers[:]:
        if isinstance(h, logging.FileHandler):
            h.close()
            logging.getLogger().removeHandler(h)


# ═══════════════════════════════════════════════════════════════════════
# 1. population.spawn_child mutation_variance parameter
# ═══════════════════════════════════════════════════════════════════════

class TestSpawnChildVariance:
    """population.spawn_child() accepts mutation_variance parameter."""

    def test_default_variance_produces_small_drift(self):
        """Default mutation_variance=0.05 produces children within ±0.05 per trait."""
        tmp = tempfile.mkdtemp()
        try:
            pop = _pop(tmp)
            parent = pop.get("AL-01")
            assert parent is not None
            parent_genome = Genome.from_dict(parent["genome"])
            parent_traits = parent_genome.traits

            pop.set_tick(BIRTH_COOLDOWN_CYCLES + 1)
            child = pop.spawn_child(parent_genome, parent_evolution=0, parent_id="AL-01")
            assert child is not None

            child_genome = Genome.from_dict(child["genome"])
            for trait in parent_traits:
                diff = abs(child_genome.get_trait(trait) - parent_genome.get_trait(trait))
                # Two spawn_child layers are gone — now only one ±0.05
                # With soft_floor adjustments, allow small margin
                assert diff <= 0.06, f"{trait}: drift {diff} > 0.06"
        finally:
            _cleanup_handlers()

    def test_custom_variance_produces_larger_drift(self):
        """mutation_variance=0.10 allows up to ±0.10 per trait (single application)."""
        tmp = tempfile.mkdtemp()
        try:
            pop = _pop(tmp)
            parent = pop.get("AL-01")
            assert parent is not None
            parent_genome = Genome.from_dict(parent["genome"])

            pop.set_tick(BIRTH_COOLDOWN_CYCLES + 1)
            child = pop.spawn_child(parent_genome, parent_evolution=0,
                                    parent_id="AL-01", mutation_variance=0.10)
            assert child is not None

            child_genome = Genome.from_dict(child["genome"])
            for trait in parent_genome.traits:
                diff = abs(child_genome.get_trait(trait) - parent_genome.get_trait(trait))
                # Single ±0.10 application + soft_floor margin
                assert diff <= 0.11, f"{trait}: drift {diff} > 0.11"
        finally:
            _cleanup_handlers()

    def test_zero_variance_produces_near_clone(self):
        """mutation_variance=0 produces a child with identical traits."""
        tmp = tempfile.mkdtemp()
        try:
            pop = _pop(tmp)
            parent = pop.get("AL-01")
            assert parent is not None
            parent_genome = Genome.from_dict(parent["genome"])

            pop.set_tick(BIRTH_COOLDOWN_CYCLES + 1)
            child = pop.spawn_child(parent_genome, parent_evolution=0,
                                    parent_id="AL-01", mutation_variance=0.0)
            assert child is not None

            child_genome = Genome.from_dict(child["genome"])
            for trait in parent_genome.traits:
                assert child_genome.get_trait(trait) == pytest.approx(
                    parent_genome.get_trait(trait), abs=1e-6
                ), f"{trait}: expected clone, got drift"
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 2. Rare reproduction — single mutation only
# ═══════════════════════════════════════════════════════════════════════

class TestRareReproductionSingleMutation:
    """Rare reproduction applies mutation exactly once (variance=0.10)."""

    def test_rare_child_drift_bounded_by_single_0_10(self, tmp_path, monkeypatch):
        """Child traits should be within ±0.10 of parent (not ±0.15)."""
        org = _make_organism(str(tmp_path))
        try:
            org._global_cycle = 500
            monkeypatch.setattr(random, "random", lambda: 0.01)
            monkeypatch.setattr(random, "shuffle", lambda x: None)
            _set_eligible_parent(org._population, "AL-01", energy=0.80)

            parent = org._population.get("AL-01")
            assert parent is not None
            parent_genome = Genome.from_dict(parent["genome"])

            child = org.rare_reproduction_cycle()
            assert child is not None

            child_genome = Genome.from_dict(child["genome"])
            for trait in parent_genome.traits:
                diff = abs(child_genome.get_trait(trait) - parent_genome.get_trait(trait))
                # Single ±0.10 variance + soft_floor margin
                assert diff <= 0.11, (
                    f"{trait}: drift {diff:.4f} exceeds single-mutation "
                    f"bound of 0.11 (was double-mutated before fix)"
                )
        finally:
            _cleanup_handlers()

    def test_rare_child_not_double_mutated(self, tmp_path, monkeypatch):
        """Statistical test: over many runs, max drift should stay ≤ 0.10."""
        org = _make_organism(str(tmp_path))
        try:
            max_drift_seen = 0.0

            for trial in range(20):
                monkeypatch.setattr(random, "random", lambda: 0.01)
                monkeypatch.setattr(random, "shuffle", lambda x: None)
                _set_eligible_parent(org._population, "AL-01", energy=0.80)

                org._global_cycle = 500 + trial * RARE_REPRO_COOLDOWN_CYCLES

                parent = org._population.get("AL-01")
                assert parent is not None
                parent_genome = Genome.from_dict(parent["genome"])

                child = org.rare_reproduction_cycle()
                if child is None:
                    continue

                child_genome = Genome.from_dict(child["genome"])
                for trait in parent_genome.traits:
                    diff = abs(child_genome.get_trait(trait) - parent_genome.get_trait(trait))
                    if diff > max_drift_seen:
                        max_drift_seen = diff

                # Clean up child to keep pop small
                org._population._members.pop(child["id"], None)

            # With single ±0.10 variance, max drift should never exceed ~0.10
            # (soft_floor may add tiny margin). Double mutation would push to ~0.15+.
            assert max_drift_seen <= 0.11, (
                f"Max drift {max_drift_seen:.4f} suggests double mutation "
                f"is still occurring"
            )
        finally:
            _cleanup_handlers()

    def test_mutation_variance_constant_unchanged(self):
        """RARE_REPRO_MUTATION_RATE is still 0.10."""
        assert RARE_REPRO_MUTATION_RATE == 0.10

    def test_rare_child_has_parent_genome_base(self, tmp_path, monkeypatch):
        """Child genome should derive from parent, not from a pre-mutated copy."""
        org = _make_organism(str(tmp_path))
        try:
            org._global_cycle = 500
            monkeypatch.setattr(random, "random", lambda: 0.01)
            monkeypatch.setattr(random, "shuffle", lambda x: None)

            # Set parent traits to known values and make eligible
            _set_eligible_parent(org._population, "AL-01", energy=0.80)
            rec = org._population.get("AL-01")
            assert rec is not None
            g = Genome.from_dict(rec["genome"])
            g.set_trait("adaptability", 0.70)
            g.set_trait("energy_efficiency", 0.70)
            g.set_trait("resilience", 0.70)
            g.set_trait("perception", 0.70)
            g.set_trait("creativity", 0.70)
            org._population.update_member("AL-01", {"genome": g.to_dict()})

            child = org.rare_reproduction_cycle()
            assert child is not None

            child_genome = Genome.from_dict(child["genome"])
            # With single mutation from 0.70, all traits should be in [0.60, 0.80]
            for trait in child_genome.traits:
                val = child_genome.get_trait(trait)
                assert 0.58 <= val <= 0.82, (
                    f"{trait}={val:.4f} outside single-mutation range from 0.70"
                )
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 3. Other reproduction paths — still use default variance
# ═══════════════════════════════════════════════════════════════════════

class TestOtherReproductionVariance:
    """Auto, stability, and lone survivor still use default 0.05 variance."""

    def test_auto_reproduce_uses_default_variance(self):
        """auto_reproduce children have ≤0.05 drift per trait."""
        tmp = tempfile.mkdtemp()
        try:
            pop = _pop(tmp)
            parent = pop.get("AL-01")
            assert parent is not None
            pop._members["AL-01"]["consecutive_above_repro"] = 10
            pop._members["AL-01"]["energy"] = 0.80
            parent_genome = Genome.from_dict(parent["genome"])
            for t in parent_genome.traits:
                parent_genome.set_trait(t, 0.70)
            pop.update_member("AL-01", {"genome": parent_genome.to_dict()})

            pop.set_tick(BIRTH_COOLDOWN_CYCLES + 1)
            child = pop.auto_reproduce("AL-01", energy_min=0.50)
            assert child is not None

            parent_after = pop.get("AL-01")
            assert parent_after is not None
            pg = Genome.from_dict(parent_after["genome"])
            cg = Genome.from_dict(child["genome"])
            for trait in pg.traits:
                diff = abs(cg.get_trait(trait) - pg.get_trait(trait))
                assert diff <= 0.06, f"{trait}: drift {diff} > 0.06 for auto_reproduce"
        finally:
            _cleanup_handlers()

    def test_stability_repro_uses_default_variance(self):
        """stability_reproduction_cycle children have ≤0.05 drift."""
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
                g.set_trait(t, 0.70)
            org._population.update_member("AL-01", {"genome": g.to_dict()})

            org._population.set_tick(BIRTH_COOLDOWN_CYCLES + 1)
            spawned = org.stability_reproduction_cycle()
            assert len(spawned) > 0

            parent = org._population.get("AL-01")
            assert parent is not None
            pg = Genome.from_dict(parent["genome"])
            cg = Genome.from_dict(spawned[0]["genome"])
            for trait in pg.traits:
                diff = abs(cg.get_trait(trait) - pg.get_trait(trait))
                assert diff <= 0.06, f"{trait}: drift {diff} > 0.06 for stability_repro"
        finally:
            _cleanup_handlers()

    def test_lone_survivor_uses_default_variance(self):
        """lone_survivor_reproduction children have ≤0.05 drift."""
        tmp = tempfile.mkdtemp()
        try:
            env = _env(pool_max=1000, pool_initial=1000,
                       lone_survivor_repro_probability=1.0,
                       repro_energy_min_lone_survivor=0.0)
            org = _make_organism(tmp, env=env)
            org.boot()
            assert org._population.size == 1

            rec = org._population.get("AL-01")
            assert rec is not None
            g = Genome.from_dict(rec["genome"])
            for t in g.traits:
                g.set_trait(t, 0.70)
            org._population.update_member("AL-01", {"genome": g.to_dict()})

            result = org.lone_survivor_reproduction()
            assert result is not None

            parent = org._population.get("AL-01")
            assert parent is not None
            pg = Genome.from_dict(parent["genome"])
            cg = Genome.from_dict(result["genome"])
            for trait in pg.traits:
                diff = abs(cg.get_trait(trait) - pg.get_trait(trait))
                assert diff <= 0.06, f"{trait}: drift {diff} > 0.06 for lone_survivor"
        finally:
            _cleanup_handlers()
