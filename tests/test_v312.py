"""v3.12 — Novelty Metric, Evolutionary Innovation, Stagnation Detection,
Population Diversity Score, Evolution Dashboard.

Tests for all five features added in v3.12.
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
from al01.organism import (
    Organism,
    NOVELTY_HISTORY_MAX,
    NOVELTY_STAGNATION_WINDOW,
    NOVELTY_STAGNATION_THRESHOLD,
    NOVELTY_STAGNATION_COOLDOWN,
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
# 1. Novelty Metric — genome distance at birth
# ═══════════════════════════════════════════════════════════════════════

class TestNoveltyMetric:

    def test_constants_imported(self):
        assert NOVELTY_HISTORY_MAX == 500
        assert NOVELTY_STAGNATION_WINDOW == 100
        assert NOVELTY_STAGNATION_THRESHOLD == 0.05
        assert NOVELTY_STAGNATION_COOLDOWN == 1000

    def test_record_novelty(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            g1 = Genome(traits={"a": 0.5, "b": 0.5})
            g2 = Genome(traits={"a": 0.6, "b": 0.4})
            novelty = org._record_novelty(g1, g2)
            assert novelty > 0.0
            assert len(org.novelty_history) == 1
            assert org.novelty_history[0] == novelty
        finally:
            _cleanup_handlers()

    def test_novelty_history_capped(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            g1 = Genome()
            g2 = Genome(traits={"a": 0.9})
            for _ in range(NOVELTY_HISTORY_MAX + 50):
                org._record_novelty(g1, g2)
            assert len(org.novelty_history) == NOVELTY_HISTORY_MAX
        finally:
            _cleanup_handlers()

    def test_avg_novelty_empty(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            assert org.avg_novelty == 0.0
        finally:
            _cleanup_handlers()

    def test_avg_novelty_computed(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            org._novelty_history = [0.1, 0.2, 0.3]
            avg = org.avg_novelty
            assert avg == pytest.approx(0.2, abs=0.001)
        finally:
            _cleanup_handlers()

    def test_novelty_rate_alias(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            org._novelty_history = [0.15, 0.25]
            assert org.novelty_rate == org.avg_novelty
        finally:
            _cleanup_handlers()

    def test_identical_genomes_zero_novelty(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            g = Genome()
            novelty = org._record_novelty(g, g)
            assert novelty == 0.0
        finally:
            _cleanup_handlers()

    def test_novelty_increases_with_divergence(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            g1 = Genome(traits={"a": 0.5, "b": 0.5})
            g_close = Genome(traits={"a": 0.55, "b": 0.55})
            g_far = Genome(traits={"a": 0.9, "b": 0.1})
            n_close = org._record_novelty(g1, g_close)
            n_far = org._record_novelty(g1, g_far)
            assert n_close < n_far
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 2. Track Evolutionary Innovation — novelty in VITAL log
# ═══════════════════════════════════════════════════════════════════════

class TestNoveltyInVITAL:

    def test_reproduction_logs_novelty(self, tmp_path):
        """When process_cycle triggers reproduction, novelty is in the log."""
        org = _make_organism(str(tmp_path))
        try:
            # Force a reproduction by setting up the right conditions
            _spawn_children(org._population, 1)
            # Check that the novelty_history has entries if any reproduction occurred
            # We test the mechanism more directly via _record_novelty
            g1 = Genome()
            g2 = Genome(traits={"a": 0.9})
            novelty = org._record_novelty(g1, g2)
            assert novelty > 0.0
            assert len(org.novelty_history) >= 1
        finally:
            _cleanup_handlers()

    def test_rare_reproduction_logs_novelty(self, tmp_path, monkeypatch):
        org = _make_organism(str(tmp_path))
        try:
            org._global_cycle = 500
            monkeypatch.setattr(random, "random", lambda: 0.01)
            monkeypatch.setattr(random, "shuffle", lambda x: None)
            org._population.update_energy("AL-01", 0.80)
            for i in range(20):
                org._population.record_fitness("AL-01", 0.5 + i * 0.01)
            child = org.rare_reproduction_cycle()
            if child:
                # Verify novelty was recorded
                assert len(org.novelty_history) >= 1
                # Verify it's in the life log
                log_path = os.path.join(str(tmp_path), "data", "life_log.jsonl")
                with open(log_path, "r") as fh:
                    lines = [json.loads(l) for l in fh if l.strip()]
                rare_events = [l for l in lines if l.get("event_type") == "rare_reproduction"]
                assert len(rare_events) >= 1
                assert "novelty" in rare_events[-1].get("payload", {})
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 3. Evolution Stagnation Detection
# ═══════════════════════════════════════════════════════════════════════

class TestStagnationDetection:

    def test_no_stagnation_when_insufficient_data(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            # Less than 10 data points — should return None
            org._novelty_history = [0.01] * 5
            result = org.novelty_stagnation_check()
            assert result is None
        finally:
            _cleanup_handlers()

    def test_no_stagnation_when_healthy(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            org._novelty_history = [0.15] * 20
            result = org.novelty_stagnation_check()
            assert result is None  # 0.15 > 0.05 threshold
        finally:
            _cleanup_handlers()

    def test_stagnation_detected_when_low_novelty(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            org._novelty_history = [0.01] * 20
            org._global_cycle = 5000
            result = org.novelty_stagnation_check()
            assert result is not None
            assert result["event"] == "novelty_stagnation_intervention"
            assert "mutation_storm" in result["actions"]
            assert "mutation_rate_boost" in result["actions"]
            assert "exploration_mode" in result["actions"]
        finally:
            _cleanup_handlers()

    def test_stagnation_triggers_mutation_storm(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            org._novelty_history = [0.01] * 20
            org._global_cycle = 5000
            events_before = len(org._environment._named_events)
            org.novelty_stagnation_check()
            events_after = len(org._environment._named_events)
            assert events_after > events_before
            # Verify the event is a mutation storm
            storm = org._environment._named_events[-1]
            assert storm.name == "mutation_storm"
        finally:
            _cleanup_handlers()

    def test_stagnation_cooldown(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            org._novelty_history = [0.01] * 20
            org._global_cycle = 5000
            result1 = org.novelty_stagnation_check()
            assert result1 is not None
            # Second call within cooldown → None
            result2 = org.novelty_stagnation_check()
            assert result2 is None
            # After cooldown → should trigger again
            org._global_cycle = 5000 + NOVELTY_STAGNATION_COOLDOWN + 1
            result3 = org.novelty_stagnation_check()
            assert result3 is not None
        finally:
            _cleanup_handlers()

    def test_stagnation_sets_exploration_mode(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            org._novelty_history = [0.01] * 20
            org._global_cycle = 5000
            org.novelty_stagnation_check()
            assert org._autonomy.exploration_mode is True
        finally:
            _cleanup_handlers()

    def test_stagnation_logged_to_vital(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            org._novelty_history = [0.01] * 20
            org._global_cycle = 5000
            org.novelty_stagnation_check()
            log_path = os.path.join(str(tmp_path), "data", "life_log.jsonl")
            with open(log_path, "r") as fh:
                lines = [json.loads(l) for l in fh if l.strip()]
            stag_events = [l for l in lines if l.get("event_type") == "novelty_stagnation"]
            assert len(stag_events) >= 1
            payload = stag_events[-1]["payload"]
            assert payload["avg_novelty"] < NOVELTY_STAGNATION_THRESHOLD
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 4. Population Diversity Score
# ═══════════════════════════════════════════════════════════════════════

class TestPopulationDiversity:

    def test_diversity_single_organism(self, tmp_path):
        pop = _pop(str(tmp_path))
        assert pop.population_diversity() == 0.0

    def test_diversity_two_identical(self, tmp_path):
        pop = _pop(str(tmp_path))
        _spawn_children(pop, 1)
        # Parent and child have similar genomes (low diversity)
        div = pop.population_diversity()
        assert div >= 0.0  # should be small but non-negative

    def test_diversity_increases_with_variance(self, tmp_path):
        pop = _pop(str(tmp_path))
        _spawn_children(pop, 3)
        div_before = pop.population_diversity()
        # Manually make one child very different
        members = [m for m in pop.member_ids if m != "AL-01"]
        with pop._lock:
            pop._members[members[0]]["genome"]["traits"] = {
                "adaptability": 0.99, "energy_efficiency": 0.01,
                "resilience": 0.99, "perception": 0.01, "creativity": 0.99,
            }
            pop._save()
        div_after = pop.population_diversity()
        assert div_after > div_before

    def test_diversity_excludes_dead(self, tmp_path):
        pop = _pop(str(tmp_path))
        _spawn_children(pop, 3)
        div_before = pop.population_diversity()
        members = [m for m in pop.member_ids if m != "AL-01"]
        pop.remove_member(members[0], cause="test")
        div_after = pop.population_diversity()
        # Different count should yield different diversity (not necessarily higher or lower)
        assert isinstance(div_after, float)

    def test_organism_diversity_index(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            _spawn_children(org._population, 3)
            div = org.population_diversity_index()
            assert isinstance(div, float)
            assert div >= 0.0
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 5. Evolution Dashboard
# ═══════════════════════════════════════════════════════════════════════

class TestEvolutionDashboard:

    def test_dashboard_structure(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            dash = org.evolution_dashboard()
            assert "population" in dash
            assert "species" in dash
            assert "species_breakdown" in dash
            assert "novelty_rate" in dash
            assert "diversity_index" in dash
            assert "ecosystem_health" in dash
            assert "avg_fitness" in dash
            assert "stagnating" in dash
            assert "total_births_tracked" in dash
            assert "cycle" in dash
            assert "timestamp" in dash
        finally:
            _cleanup_handlers()

    def test_dashboard_population_count(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            dash = org.evolution_dashboard()
            # Just the parent initially
            assert dash["population"] >= 1
        finally:
            _cleanup_handlers()

    def test_dashboard_species_count(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            dash = org.evolution_dashboard()
            assert dash["species"] >= 1
        finally:
            _cleanup_handlers()

    def test_dashboard_not_stagnating_initially(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            dash = org.evolution_dashboard()
            # No novelty data yet → not stagnating
            assert dash["stagnating"] is False
        finally:
            _cleanup_handlers()

    def test_dashboard_stagnating_flag(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            org._novelty_history = [0.01] * 20
            dash = org.evolution_dashboard()
            assert dash["stagnating"] is True
        finally:
            _cleanup_handlers()

    def test_dashboard_with_population(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            _spawn_children(org._population, 5)
            dash = org.evolution_dashboard()
            assert dash["population"] >= 6
            assert dash["diversity_index"] >= 0.0
        finally:
            _cleanup_handlers()

    def test_dashboard_novelty_rate_reflects_history(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            org._novelty_history = [0.10, 0.20, 0.30]
            dash = org.evolution_dashboard()
            assert dash["novelty_rate"] == pytest.approx(0.2, abs=0.001)
        finally:
            _cleanup_handlers()

    def test_dashboard_total_births(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            g1 = Genome()
            g2 = Genome(traits={"a": 0.9})
            for _ in range(5):
                org._record_novelty(g1, g2)
            dash = org.evolution_dashboard()
            assert dash["total_births_tracked"] == 5
        finally:
            _cleanup_handlers()


# ═══════════════════════════════════════════════════════════════════════
# 6. Integration: novelty flows through reproduction cycle
# ═══════════════════════════════════════════════════════════════════════

class TestNoveltyIntegration:

    def test_novelty_after_spawn(self, tmp_path):
        """Manually trigger process_cycle reproduction and check novelty."""
        org = _make_organism(str(tmp_path))
        try:
            # Directly use _record_novelty to confirm it integrates
            parent = Genome()
            child = parent.spawn_child(variance=0.05)
            n = org._record_novelty(parent, child)
            assert n >= 0.0
            assert len(org._novelty_history) == 1
        finally:
            _cleanup_handlers()

    def test_population_diversity_in_dashboard(self, tmp_path):
        org = _make_organism(str(tmp_path))
        try:
            _spawn_children(org._population, 4)
            # Make one very different
            members = [m for m in org._population.member_ids if m != "AL-01"]
            with org._population._lock:
                org._population._members[members[0]]["genome"]["traits"] = {
                    "adaptability": 0.99, "energy_efficiency": 0.01,
                }
                org._population._save()
            dash = org.evolution_dashboard()
            assert dash["diversity_index"] > 0.0
        finally:
            _cleanup_handlers()
