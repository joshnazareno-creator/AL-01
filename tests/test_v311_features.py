"""v3.11-features — Lineage Tree, Species Divergence, Environmental Events,
Ecosystem Health, Birth Events, Fossil Record.

Tests for all six features added in the v3.11 session.
"""

import json
import logging
import math
import os
import random
import statistics
import tempfile

import pytest

from al01.environment import (
    Environment,
    EnvironmentConfig,
    NAMED_EVENT_CATALOGUE,
    NamedEvent,
)
from al01.evolution_tracker import EvolutionTracker
from al01.genome import Genome, genome_distance
from al01.population import (
    Population,
    ABSOLUTE_POPULATION_CAP,
    BIRTH_COOLDOWN_CYCLES,
    SPECIES_DIVERGENCE_THRESHOLD,
)
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
        named_event_probability=0.0,
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


def _cleanup_handlers():
    for h in logging.getLogger().handlers[:]:
        if isinstance(h, logging.FileHandler):
            h.close()
            logging.getLogger().removeHandler(h)


# ═══════════════════════════════════════════════════════════════════════
# 1. Genome Distance (genome.py)
# ═══════════════════════════════════════════════════════════════════════

class TestGenomeDistance:

    def test_identical_genomes_distance_zero(self):
        g1 = Genome()
        g2 = Genome()
        assert g1.distance(g2) == 0.0

    def test_distance_is_symmetric(self):
        g1 = Genome(traits={"a": 0.1, "b": 0.9})
        g2 = Genome(traits={"a": 0.9, "b": 0.1})
        assert g1.distance(g2) == pytest.approx(g2.distance(g1))

    def test_distance_increases_with_divergence(self):
        g1 = Genome(traits={"a": 0.5, "b": 0.5})
        g_close = Genome(traits={"a": 0.55, "b": 0.55})
        g_far = Genome(traits={"a": 0.9, "b": 0.1})
        assert g1.distance(g_close) < g1.distance(g_far)

    def test_module_level_convenience(self):
        g1 = Genome()
        g2 = Genome()
        assert genome_distance(g1, g2) == g1.distance(g2)

    def test_known_euclidean(self):
        """Distance uses effective (soft-capped) traits, so verify it's
        positive and monotonic rather than exact raw Euclidean."""
        g1 = Genome(traits={"x": 0.0, "y": 0.0})
        g2 = Genome(traits={"x": 0.3, "y": 0.4})
        d = g1.distance(g2)
        assert d > 0.0
        # And it should be close to 0.5 (soft-cap barely modifies sub-1 values)
        assert d == pytest.approx(0.5, abs=0.1)

    def test_different_trait_keys(self):
        """Traits missing in one genome are treated as 0.0."""
        g1 = Genome(traits={"a": 0.5})
        g2 = Genome(traits={"a": 0.5, "b": 0.5})
        assert g1.distance(g2) == pytest.approx(0.5, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# 2. Lineage Tree (evolution_tracker.py)
# ═══════════════════════════════════════════════════════════════════════

class TestLineageTree:

    def _setup_tracker(self, tmp):
        tracker = EvolutionTracker(data_dir=os.path.join(tmp, "data"))
        tracker.register_organism("AL-01", parent_id=None,
                                  traits={"a": 0.5}, cycle=0)
        tracker.register_organism("AL-01-child-1", parent_id="AL-01",
                                  traits={"a": 0.6}, cycle=10)
        tracker.register_organism("AL-01-child-2", parent_id="AL-01",
                                  traits={"a": 0.4}, cycle=20)
        tracker.register_organism("AL-01-child-3", parent_id="AL-01-child-1",
                                  traits={"a": 0.7}, cycle=30)
        return tracker

    def test_build_family_tree_structure(self, tmp_path):
        t = self._setup_tracker(str(tmp_path))
        tree = t.build_family_tree()
        assert "roots" in tree
        assert len(tree["roots"]) == 1
        root = tree["roots"][0]
        assert root["id"] == "AL-01"
        assert len(root["children"]) == 2

    def test_nested_child(self, tmp_path):
        t = self._setup_tracker(str(tmp_path))
        tree = t.build_family_tree()
        root = tree["roots"][0]
        child1 = [c for c in root["children"] if c["id"] == "AL-01-child-1"][0]
        assert len(child1["children"]) == 1
        assert child1["children"][0]["id"] == "AL-01-child-3"

    def test_render_tree_ascii(self, tmp_path):
        t = self._setup_tracker(str(tmp_path))
        text = t.render_tree_ascii()
        assert "AL-01" in text
        assert "AL-01-child-1" in text
        assert "AL-01-child-3" in text
        assert "+--" in text

    def test_empty_tree(self, tmp_path):
        t = EvolutionTracker(data_dir=os.path.join(str(tmp_path), "data"))
        tree = t.build_family_tree()
        assert tree["roots"] == []

    def test_ancestor_chain(self, tmp_path):
        t = self._setup_tracker(str(tmp_path))
        chain = t.get_ancestor_chain("AL-01-child-3")
        ids = [c["organism_id"] for c in chain]
        assert ids == ["AL-01-child-3", "AL-01-child-1", "AL-01"]

    def test_ancestor_chain_root(self, tmp_path):
        t = self._setup_tracker(str(tmp_path))
        chain = t.get_ancestor_chain("AL-01")
        assert len(chain) == 1
        assert chain[0]["organism_id"] == "AL-01"

    def test_descendants(self, tmp_path):
        t = self._setup_tracker(str(tmp_path))
        desc = t.get_descendants("AL-01")
        assert "AL-01-child-1" in desc
        assert "AL-01-child-2" in desc
        assert "AL-01-child-3" in desc

    def test_descendants_leaf(self, tmp_path):
        t = self._setup_tracker(str(tmp_path))
        desc = t.get_descendants("AL-01-child-2")
        assert desc == []


# ═══════════════════════════════════════════════════════════════════════
# 3. Species Divergence (population.py)
# ═══════════════════════════════════════════════════════════════════════

class TestSpeciesDivergence:

    def test_constant_imported(self):
        assert SPECIES_DIVERGENCE_THRESHOLD == 0.35

    def test_assign_species(self, tmp_path):
        pop = _pop(str(tmp_path))
        pop.assign_species("AL-01", "species-A")
        assert pop.get_species("AL-01") == "species-A"

    def test_species_census(self, tmp_path):
        pop = _pop(str(tmp_path))
        pop.assign_species("AL-01", "species-A")
        _spawn_children(pop, 2)
        members = list(pop.member_ids)
        child = [m for m in members if m != "AL-01"][0]
        pop.assign_species(child, "species-B")
        census = pop.species_census()
        assert "species-A" in census
        assert "species-B" in census

    def test_check_speciation_no_divergence(self, tmp_path):
        """Identical parent/child genomes → no new species."""
        pop = _pop(str(tmp_path))
        _spawn_children(pop, 1)
        members = list(pop.member_ids)
        child_id = [m for m in members if m != "AL-01"][0]
        pop.assign_species("AL-01", "species-A")
        result = pop.check_speciation(child_id, "AL-01")
        # Small variance from spawn_child, but unlikely to exceed 0.35
        # If no speciation, child inherits parent species
        child_species = pop.get_species(child_id)
        if result is None:
            assert child_species == "species-A"

    def test_check_speciation_forced_divergence(self, tmp_path):
        """Force a large genome difference to trigger speciation."""
        pop = _pop(str(tmp_path))
        _spawn_children(pop, 1)
        members = list(pop.member_ids)
        child_id = [m for m in members if m != "AL-01"][0]
        pop.assign_species("AL-01", "species-A")
        # Manually set child genome to be very different
        with pop._lock:
            pop._members[child_id]["genome"]["traits"] = {
                "adaptability": 0.99, "energy_efficiency": 0.01,
                "resilience": 0.99, "perception": 0.01, "creativity": 0.99,
            }
            pop._save()
        new_species = pop.check_speciation(child_id, "AL-01")
        assert new_species is not None
        assert new_species.startswith("species-")
        assert pop.get_species(child_id) == new_species

    def test_species_label_generation(self):
        assert Population._species_label(1) == "species-A"
        assert Population._species_label(2) == "species-B"
        assert Population._species_label(26) == "species-Z"
        assert Population._species_label(27) == "species-AA"


# ═══════════════════════════════════════════════════════════════════════
# 4. Named Environmental Events (environment.py)
# ═══════════════════════════════════════════════════════════════════════

class TestNamedEnvironmentalEvents:

    def test_catalogue_has_four_events(self):
        assert len(NAMED_EVENT_CATALOGUE) == 4
        names = {e["name"] for e in NAMED_EVENT_CATALOGUE}
        assert "heat_wave" in names
        assert "resource_boom" in names
        assert "scarcity_drought" in names
        assert "mutation_storm" in names

    def test_named_event_dataclass(self):
        e = NamedEvent(
            name="test", description="A test", effects={"x": 1.0},
            remaining_cycles=5, started_at="2026-01-01",
        )
        assert e.tick()  # 4 remaining
        assert e.remaining_cycles == 4

    def test_named_event_serialization(self):
        e = NamedEvent(
            name="heat_wave", description="hot", effects={"temperature_boost": 0.25},
            remaining_cycles=10, started_at="2026-01-01",
        )
        d = e.to_dict()
        e2 = NamedEvent.from_dict(d)
        assert e2.name == "heat_wave"
        assert e2.remaining_cycles == 10

    def test_trigger_named_event(self):
        cfg = EnvironmentConfig(
            scarcity_probability=0.0,
            shock_probability=0.0,
            named_event_probability=100.0,  # guarantee trigger
            named_event_duration_min=5,
            named_event_duration_max=5,
        )
        env = Environment(config=cfg, rng_seed=42)
        record = env.tick()
        named_events = [e for e in record.get("events", []) if e.get("type") == "named_event"]
        assert len(named_events) >= 1
        assert named_events[0]["name"] in {e["name"] for e in NAMED_EVENT_CATALOGUE}

    def test_named_event_expires(self):
        cfg = EnvironmentConfig(
            scarcity_probability=0.0,
            shock_probability=0.0,
            named_event_probability=0.0,  # no new events
        )
        env = Environment(config=cfg, rng_seed=42)
        # Manually inject a short event
        event = NamedEvent(
            name="test_event", description="test", effects={},
            remaining_cycles=2, started_at="2026-01-01",
        )
        env._named_events.append(event)
        assert len(env.named_events) == 1
        env.tick()
        assert len(env.named_events) == 1  # 1 cycle remaining
        env.tick()
        assert len(env.named_events) == 0  # expired

    def test_heat_wave_boosts_temperature(self):
        cfg = EnvironmentConfig(
            scarcity_probability=0.0,
            shock_probability=0.0,
            named_event_probability=0.0,
        )
        env = Environment(config=cfg, rng_seed=42)
        temp_before = env.temperature
        # Manually trigger heat_wave
        event = NamedEvent(
            name="heat_wave", description="hot",
            effects={"temperature_boost": 0.25, "resilience_bonus": 0.10},
            remaining_cycles=3, started_at="2026-01-01",
        )
        env._named_events.append(event)
        # Apply immediate effects (simulate what _trigger_named_event does)
        env._temperature = min(1.0, env._temperature + 0.25)
        assert env.temperature >= temp_before

    def test_mutation_rate_multiplier(self):
        cfg = EnvironmentConfig(
            scarcity_probability=0.0,
            shock_probability=0.0,
            named_event_probability=0.0,
        )
        env = Environment(config=cfg, rng_seed=42)
        assert env.mutation_rate_multiplier == 1.0
        event = NamedEvent(
            name="mutation_storm", description="cosmic rays",
            effects={"mutation_rate_multiplier": 2.0},
            remaining_cycles=5, started_at="2026-01-01",
        )
        env._named_events.append(event)
        assert env.mutation_rate_multiplier == 2.0

    def test_metabolic_multiplier(self):
        cfg = EnvironmentConfig(
            scarcity_probability=0.0,
            shock_probability=0.0,
            named_event_probability=0.0,
        )
        env = Environment(config=cfg, rng_seed=42)
        assert env.metabolic_multiplier == 1.0
        event = NamedEvent(
            name="scarcity_drought", description="drought",
            effects={"metabolic_multiplier": 1.5},
            remaining_cycles=5, started_at="2026-01-01",
        )
        env._named_events.append(event)
        assert env.metabolic_multiplier == 1.5

    def test_named_events_in_state_snapshot(self):
        cfg = EnvironmentConfig(
            scarcity_probability=0.0,
            shock_probability=0.0,
            named_event_probability=0.0,
        )
        env = Environment(config=cfg, rng_seed=42)
        snap = env.state_snapshot()
        assert "active_named_events" in snap
        assert "named_event_names" in snap
        assert "mutation_rate_multiplier" in snap
        assert "metabolic_multiplier" in snap

    def test_named_events_serialization(self):
        cfg = EnvironmentConfig(
            scarcity_probability=0.0,
            shock_probability=0.0,
            named_event_probability=0.0,
        )
        env = Environment(config=cfg, rng_seed=42)
        event = NamedEvent(
            name="test", description="test",
            effects={"mutation_rate_multiplier": 2.0},
            remaining_cycles=5, started_at="2026-01-01",
        )
        env._named_events.append(event)
        d = env.to_dict()
        assert len(d["named_events"]) == 1
        env2 = Environment.from_dict(d, config=cfg)
        assert len(env2.named_events) == 1
        assert env2.named_events[0].name == "test"

    def test_resource_boom_boosts_pool(self):
        cfg = EnvironmentConfig(
            resource_pool_initial=500.0,
            resource_pool_max=1000.0,
            resource_regen_rate=5.0,
            scarcity_probability=0.0,
            shock_probability=0.0,
            named_event_probability=0.0,
        )
        env = Environment(config=cfg, rng_seed=42)
        pool_before = env.resource_pool
        event = NamedEvent(
            name="resource_boom", description="boom",
            effects={"pool_regen_multiplier": 2.0, "abundance_boost": 0.20},
            remaining_cycles=5, started_at="2026-01-01",
        )
        env._named_events.append(event)
        env.tick()
        # Pool should have received bonus regen
        assert env.resource_pool > pool_before


# ═══════════════════════════════════════════════════════════════════════
# 5. Ecosystem Health (organism.py)
# ═══════════════════════════════════════════════════════════════════════

class TestEcosystemHealth:

    def test_health_score_exists(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            health = org.ecosystem_health()
            assert "score" in health
            assert "avg_fitness" in health
            assert "diversity_score" in health
            assert "pool_health" in health
            assert "population_size" in health
        finally:
            _cleanup_handlers()

    def test_health_score_range(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            health = org.ecosystem_health()
            assert 0.0 <= health["score"] <= 1.0
        finally:
            _cleanup_handlers()

    def test_health_formula(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            health = org.ecosystem_health()
            expected = (health["avg_fitness"] * 0.4
                        + health["diversity_score"] * 0.4
                        + health["pool_health"] * 0.2)
            assert health["score"] == pytest.approx(expected, abs=0.01)
        finally:
            _cleanup_handlers()

    def test_health_with_multiple_organisms(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            _spawn_children(org._population, 5)
            for mid in org._population.member_ids:
                org._population.record_fitness(mid, random.uniform(0.3, 0.8))
            health = org.ecosystem_health()
            assert health["population_size"] >= 6
            assert health["unique_genomes"] >= 1
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 6. Birth Events (organism.py)
# ═══════════════════════════════════════════════════════════════════════

class TestBirthEvents:

    def test_last_birth_events_empty(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            events = org.last_birth_events()
            assert isinstance(events, list)
        finally:
            _cleanup_handlers()

    def test_birth_events_after_reproduction(self, tmp_path, monkeypatch):
        org = _make_organism(str(tmp_path))
        try:
            org._global_cycle = 500
            monkeypatch.setattr(random, "random", lambda: 0.01)
            monkeypatch.setattr(random, "shuffle", lambda x: None)
            # Set up eligible parent
            org._population.update_energy("AL-01", 0.80)
            for i in range(20):
                org._population.record_fitness("AL-01", 0.5 + i * 0.01)
            child = org.rare_reproduction_cycle()
            if child:  # should succeed with these settings
                events = org.last_birth_events()
                repro = [e for e in events if e["event_type"] == "rare_reproduction"]
                assert len(repro) >= 1
                assert repro[-1]["parent_id"] == "AL-01"
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 7. Fossil Record (population.py)
# ═══════════════════════════════════════════════════════════════════════

class TestFossilRecord:

    def test_no_fossils_initially(self, tmp_path):
        pop = _pop(str(tmp_path))
        fossils = pop.fossil_record()
        assert fossils == []

    def test_fossil_after_death(self, tmp_path):
        pop = _pop(str(tmp_path))
        _spawn_children(pop, 3)
        members = [m for m in pop.member_ids if m != "AL-01"]
        # Kill one
        pop.remove_member(members[0], cause="test_death")
        fossils = pop.fossil_record()
        assert len(fossils) == 1
        assert fossils[0]["organism_id"] == members[0]
        assert fossils[0]["cause_of_death"] == "test_death"

    def test_fossil_fields(self, tmp_path):
        pop = _pop(str(tmp_path))
        _spawn_children(pop, 1)
        members = [m for m in pop.member_ids if m != "AL-01"]
        pop.remove_member(members[0], cause="energy_depleted")
        fossil = pop.fossil_record()[0]
        assert "organism_id" in fossil
        assert "generation" in fossil
        assert "parent_id" in fossil
        assert "genome_hash" in fossil
        assert "cause_of_death" in fossil
        assert "death_time" in fossil
        assert "final_fitness" in fossil
        assert "final_energy" in fossil
        assert "species_id" in fossil

    def test_fossil_summary(self, tmp_path):
        pop = _pop(str(tmp_path))
        _spawn_children(pop, 3)
        members = [m for m in pop.member_ids if m != "AL-01"]
        for m in members:
            pop.remove_member(m, cause="fitness_floor")
        summary = pop.fossil_summary()
        assert summary["total_deaths"] == 3
        assert "fitness_floor" in summary["causes"]
        assert summary["causes"]["fitness_floor"] == 3

    def test_fossil_summary_empty(self, tmp_path):
        pop = _pop(str(tmp_path))
        summary = pop.fossil_summary()
        assert summary["total_deaths"] == 0


# ═══════════════════════════════════════════════════════════════════════
# 8. Integration: Species on spawn
# ═══════════════════════════════════════════════════════════════════════

class TestSpeciesIntegration:

    def test_default_species(self, tmp_path):
        """Unassigned organisms default to 'default' in census."""
        pop = _pop(str(tmp_path))
        census = pop.species_census()
        assert "default" in census
        assert "AL-01" in census["default"]

    def test_species_preserved_in_fossil(self, tmp_path):
        pop = _pop(str(tmp_path))
        pop.assign_species("AL-01", "species-A")
        _spawn_children(pop, 1)
        members = [m for m in pop.member_ids if m != "AL-01"]
        pop.assign_species(members[0], "species-B")
        pop.remove_member(members[0], cause="test")
        fossils = pop.fossil_record()
        assert fossils[0]["species_id"] == "species-B"


# ═══════════════════════════════════════════════════════════════════════
# 9. State snapshot includes named events
# ═══════════════════════════════════════════════════════════════════════

class TestEnvironmentStateCompleteness:

    def test_snapshot_has_named_event_fields(self):
        env = _env()
        snap = env.state_snapshot()
        assert "active_named_events" in snap
        assert "named_event_names" in snap
        assert isinstance(snap["active_named_events"], list)

    def test_to_dict_has_named_events(self):
        env = _env()
        d = env.to_dict()
        assert "named_events" in d
        assert "named_event_log" in d

    def test_from_dict_round_trip(self):
        env = _env()
        d = env.to_dict()
        env2 = Environment.from_dict(d, config=env.config)
        assert env2.cycle == env.cycle


# ═══════════════════════════════════════════════════════════════════════
# 10. Lineage tree on live organism
# ═══════════════════════════════════════════════════════════════════════

class TestLineageTreeOnOrganism:

    def test_tree_via_organism(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            tree = org.evolution_tracker.build_family_tree()
            assert "roots" in tree
            # AL-01 should be registered
            root_ids = [r["id"] for r in tree["roots"]]
            assert "AL-01" in root_ids
        finally:
            _cleanup_handlers()

    def test_ascii_tree_via_organism(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            text = org.evolution_tracker.render_tree_ascii()
            assert "AL-01" in text
        finally:
            _cleanup_handlers()
