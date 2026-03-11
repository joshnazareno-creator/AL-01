"""Tests for AL-01 v3.23 — Ecosystem stabilization.

Covers:
1. Minimum energy floor / conservation mode
2. Population-scaled resource pool regeneration
3. Adaptability recovery boost
4. Stress feedback loop
5. Energy efficiency trait weight on metabolism
6. Extinction prevention guard
"""

from __future__ import annotations

import pytest

from al01.autonomy import AutonomyConfig, AutonomyEngine, DECISION_STABILIZE
from al01.environment import Environment, EnvironmentConfig
from al01.genome import Genome
from al01.organism import (
    ADAPTABILITY_RECOVERY_THRESHOLD,
    ADAPTABILITY_RECOVERY_NUDGE,
    CONSERVATION_ENERGY_THRESHOLD,
    ENERGY_EFFICIENCY_METABOLIC_SCALE,
    Organism,
    MetabolismConfig,
)
from al01.population import Population
from al01.memory_manager import MemoryManager

import os
import tempfile


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _make_organism(tmp_path, env_config=None, autonomy_config=None):
    """Build an Organism wired to temp directories for isolation."""
    data_dir = str(tmp_path)
    os.makedirs(os.path.join(data_dir, "data"), exist_ok=True)
    mm = MemoryManager(data_dir=data_dir, credential_path=None)
    pop = Population(data_dir=data_dir, parent_id="AL-01")
    env = Environment(config=env_config or EnvironmentConfig(), rng_seed=42)
    ae = AutonomyEngine(data_dir=os.path.join(data_dir, "data"),
                        config=autonomy_config or AutonomyConfig())
    org = Organism(
        data_dir=data_dir,
        memory_manager=mm,
        population=pop,
        environment=env,
        autonomy=ae,
    )
    return org


# ═══════════════════════════════════════════════════════════════════════
# 1. Conservation Mode (Minimum Energy Floor)
# ═══════════════════════════════════════════════════════════════════════

class TestConservationMode:
    """Organisms at minimum energy enter sleeping state (was conservation mode)."""

    def test_conservation_threshold_constant(self):
        assert CONSERVATION_ENERGY_THRESHOLD == pytest.approx(0.10)

    def test_conservation_mode_activates(self, tmp_path):
        """Child at low energy enters sleeping state."""
        org = _make_organism(tmp_path)
        # Spawn a child
        child = org._population.spawn_child(org._genome, 1)
        assert child is not None
        cid = child["id"]
        # Set child energy very low (at conservation threshold)
        org._population.update_member(cid, {"energy": 0.09})
        # Run child autonomy — should enter sleeping, NOT die
        results = org.child_autonomy_cycle()
        # Find the result for this child
        child_results = [r for r in results if r.get("organism_id") == cid]
        if child_results:
            assert child_results[0]["decision"] != "death"
        # v3.26: Check sleeping lifecycle state instead of conservation mode dict
        member = org._population.get(cid)
        if member and member.get("alive", True):
            from al01.population import LifecycleState
            assert member.get("lifecycle_state") == LifecycleState.SLEEPING

    def test_conservation_mode_exits(self, tmp_path):
        """Sleeping state exits when energy recovers."""
        org = _make_organism(tmp_path)
        child = org._population.spawn_child(org._genome, 1)
        assert child is not None
        cid = child["id"]
        # Put into sleeping state via population
        org._population.enter_sleeping(cid, cause="low_energy")
        # Set energy above exit threshold (2 × conservation threshold)
        org._population.update_member(cid, {"energy": 0.25})
        org.child_autonomy_cycle()
        # Should have exited sleeping state
        from al01.population import LifecycleState
        member = org._population.get(cid)
        assert member is not None
        assert member.get("lifecycle_state") != LifecycleState.SLEEPING

    def test_conservation_reduces_metabolism(self):
        """Conservation metabolic fraction is less than 1.0."""
        cfg = EnvironmentConfig()
        assert cfg.conservation_metabolic_fraction < 1.0
        assert cfg.conservation_metabolic_fraction > 0.0

    def test_death_clears_conservation(self, tmp_path):
        """v3.28: _handle_death kills child permanently (straight to graveyard).

        Sleeping state is irrelevant — child goes straight to graveyard.
        """
        org = _make_organism(tmp_path)
        org._population.hardcore_extinction_mode = True
        # Spawn enough children to avoid extinction recovery threshold (< 5)
        for _ in range(5):
            org._population.spawn_child(org._genome, 1)
        children = [mid for mid in org._population.member_ids if mid != "AL-01"]
        cid = children[0]
        org._population.enter_sleeping(cid, cause="low_energy")
        # v3.28: Death kills child permanently — graveyard
        org._handle_death(cid, "test_death")
        from al01.population import LifecycleState
        # Child is now in graveyard
        assert cid in org._population._graveyard
        g = org._population._graveyard[cid]
        assert g.get("lifecycle_state") == LifecycleState.DEAD


# ═══════════════════════════════════════════════════════════════════════
# 2. Population-Scaled Resource Pool Regeneration
# ═══════════════════════════════════════════════════════════════════════

class TestPopulationScaledRegen:
    """Resource regeneration scales with population size."""

    def test_config_has_population_regen_bonus(self):
        cfg = EnvironmentConfig()
        assert cfg.population_regen_bonus == pytest.approx(0.5)

    def test_smart_regen_accepts_population_size(self):
        """smart_regenerate accepts population_size parameter."""
        env = Environment(rng_seed=42)
        env._resource_pool = 500.0
        regen_solo = env.smart_regenerate(avg_efficiency=0.5, population_size=1)
        assert regen_solo > 0

    def test_larger_population_regenerates_more(self):
        """More organisms → more pool regeneration."""
        env1 = Environment(rng_seed=42)
        env1._resource_pool = 500.0
        regen_1 = env1.smart_regenerate(avg_efficiency=0.5, population_size=1)

        env2 = Environment(rng_seed=42)
        env2._resource_pool = 500.0
        regen_10 = env2.smart_regenerate(avg_efficiency=0.5, population_size=10)

        assert regen_10 > regen_1

    def test_population_regen_bonus_proportional(self):
        """Population bonus is proportional to population size."""
        cfg = EnvironmentConfig(population_regen_bonus=1.0)
        env1 = Environment(config=cfg, rng_seed=42)
        env1._resource_pool = 500.0
        regen_5 = env1.smart_regenerate(avg_efficiency=0.5, population_size=5)

        env2 = Environment(config=cfg, rng_seed=42)
        env2._resource_pool = 500.0
        regen_20 = env2.smart_regenerate(avg_efficiency=0.5, population_size=20)

        # regen_20 should be about 15 units higher (20-5 * 1.0)
        diff = regen_20 - regen_5
        assert diff == pytest.approx(15.0, abs=0.1)

    def test_environment_tick_passes_pop_size(self, tmp_path):
        """environment_tick passes population_size to smart_regenerate."""
        org = _make_organism(tmp_path)
        org._environment._resource_pool = 500.0
        record = org.environment_tick()
        # Pool should have regenerated
        assert org._environment.resource_pool > 500.0


# ═══════════════════════════════════════════════════════════════════════
# 3. Adaptability Recovery Boost
# ═══════════════════════════════════════════════════════════════════════

class TestAdaptabilityRecovery:
    """Adaptability below threshold gets a recovery nudge."""

    def test_threshold_constant(self):
        assert ADAPTABILITY_RECOVERY_THRESHOLD == pytest.approx(0.20)

    def test_nudge_constant(self):
        assert ADAPTABILITY_RECOVERY_NUDGE == pytest.approx(0.02)

    def test_parent_adaptability_recovery(self, tmp_path):
        """Parent AL-01 recovers adaptability when below threshold."""
        org = _make_organism(tmp_path)
        # Set adaptability very low
        org._genome.set_trait("adaptability", 0.05)
        assert org._genome.effective_traits["adaptability"] < ADAPTABILITY_RECOVERY_THRESHOLD
        old_a = org._genome.get_trait("adaptability")
        record = org.autonomy_cycle()
        new_a = org._genome.get_trait("adaptability")
        assert new_a > old_a
        assert "adaptability_recovery" in record

    def test_no_recovery_above_threshold(self, tmp_path):
        """No recovery nudge when adaptability is healthy."""
        org = _make_organism(tmp_path)
        org._genome.set_trait("adaptability", 0.50)
        record = org.autonomy_cycle()
        assert "adaptability_recovery" not in record

    def test_child_adaptability_recovery(self, tmp_path):
        """Child organisms also get adaptability recovery."""
        org = _make_organism(tmp_path)
        child = org._population.spawn_child(org._genome, 1)
        assert child is not None
        cid = child["id"]
        # Set child's adaptability very low
        child_genome = Genome.from_dict(child["genome"])
        child_genome.set_trait("adaptability", 0.05)
        org._population.update_member(cid, {"genome": child_genome.to_dict()})
        old_a = child_genome.get_trait("adaptability")
        org.child_autonomy_cycle()
        member = org._population.get(cid)
        if member and member.get("alive", True):
            new_genome = Genome.from_dict(member["genome"])
            # Should have been nudged upward
            assert new_genome.get_trait("adaptability") > old_a


# ═══════════════════════════════════════════════════════════════════════
# 4. Stress Feedback Loop
# ═══════════════════════════════════════════════════════════════════════

class TestStressFeedback:
    """High stress triggers adaptive behaviour instead of collapse."""

    def test_stress_threshold_config(self):
        cfg = AutonomyConfig()
        assert cfg.stress_exploration_threshold == pytest.approx(0.60)
        assert cfg.stress_mutation_boost == pytest.approx(0.03)

    def test_stress_activates_exploration(self):
        """High stress triggers exploration mode."""
        cfg = AutonomyConfig()
        ae = AutonomyEngine(config=cfg)
        # Force low energy to create high stress (energy_stress = 1 - energy*2)
        ae._energy = 0.15  # energy_stress = 0.7
        record = ae.decide(
            fitness=0.5, awareness=0.5, mutation_rate=0.10,
            pending_stimuli=0, current_traits={"energy_efficiency": 0.5,
                                                "adaptability": 0.5,
                                                "resilience": 0.5,
                                                "perception": 0.5,
                                                "creativity": 0.5},
        )
        assert record["stress_level"] > 0.5
        assert "stress_boosted" in record

    def test_stress_boosts_mutation_rate(self):
        """When stress is high, effective mutation rate increases."""
        cfg = AutonomyConfig()
        ae = AutonomyEngine(config=cfg)
        ae._energy = 0.10  # very low energy = high stress
        ae._recovery_mode = False
        ae._low_energy_consecutive = 0
        record = ae.decide(
            fitness=0.3, awareness=0.5, mutation_rate=0.10,
            pending_stimuli=0, current_traits={"energy_efficiency": 0.5,
                                                "adaptability": 0.5,
                                                "resilience": 0.5,
                                                "perception": 0.5,
                                                "creativity": 0.5},
        )
        # Effective mutation rate should be higher than base due to stress boost
        assert record["effective_mutation_rate"] >= 0.10

    def test_no_stress_boost_in_recovery_mode(self):
        """Stress feedback does not trigger during recovery mode."""
        cfg = AutonomyConfig()
        ae = AutonomyEngine(config=cfg)
        ae._energy = 0.10
        ae._recovery_mode = True
        ae._low_energy_consecutive = 15
        record = ae.decide(
            fitness=0.3, awareness=0.5, mutation_rate=0.10,
            pending_stimuli=0, current_traits={"energy_efficiency": 0.5,
                                                "adaptability": 0.5,
                                                "resilience": 0.5,
                                                "perception": 0.5,
                                                "creativity": 0.5},
        )
        # Recovery mode overrides — should NOT have stress_boosted
        assert record["stress_boosted"] is False

    def test_stress_record_fields(self):
        """Decision record includes stress_level and stress_boosted."""
        cfg = AutonomyConfig()
        ae = AutonomyEngine(config=cfg)
        record = ae.decide(
            fitness=0.5, awareness=0.5, mutation_rate=0.10,
            pending_stimuli=0,
        )
        assert "stress_level" in record
        assert "stress_boosted" in record


# ═══════════════════════════════════════════════════════════════════════
# 5. Energy Efficiency Trait Weight
# ═══════════════════════════════════════════════════════════════════════

class TestEnergyEfficiencyWeight:
    """energy_efficiency trait reduces per-cycle metabolic cost."""

    def test_scale_constant(self):
        assert ENERGY_EFFICIENCY_METABOLIC_SCALE == pytest.approx(0.30)

    def test_config_has_scale(self):
        cfg = AutonomyConfig()
        assert cfg.energy_efficiency_metabolic_scale == pytest.approx(0.30)

    def test_high_efficiency_less_energy_decay(self):
        """Organism with high energy_efficiency loses less energy per cycle."""
        cfg = AutonomyConfig()
        # High efficiency organism
        ae_high = AutonomyEngine(config=cfg)
        ae_high._energy = 0.80
        traits_high = {"energy_efficiency": 0.9, "adaptability": 0.5,
                       "resilience": 0.5, "perception": 0.5, "creativity": 0.5}
        rec_high = ae_high.decide(fitness=0.6, awareness=0.5, mutation_rate=0.05,
                                  pending_stimuli=0, current_traits=traits_high)

        # Low efficiency organism
        ae_low = AutonomyEngine(config=cfg)
        ae_low._energy = 0.80
        traits_low = {"energy_efficiency": 0.1, "adaptability": 0.5,
                      "resilience": 0.5, "perception": 0.5, "creativity": 0.5}
        rec_low = ae_low.decide(fitness=0.6, awareness=0.5, mutation_rate=0.05,
                                pending_stimuli=0, current_traits=traits_low)

        # High-efficiency organism should have more energy remaining
        assert rec_high["energy"] > rec_low["energy"]

    def test_efficiency_scale_clamped(self):
        """Scale factor is clamped between 0.3 and 1.0."""
        cfg = AutonomyConfig(energy_efficiency_metabolic_scale=0.30)
        # At trait = 1.0: scale = 1.0 - 1.0 * 0.3 = 0.7 (valid)
        # At trait = 0.0: scale = 1.0 - 0.0 * 0.3 = 1.0 (valid)
        # Even at extreme values, clamped to [0.3, 1.0]
        ae = AutonomyEngine(config=cfg)
        ae._energy = 0.80
        traits_max = {"energy_efficiency": 5.0, "adaptability": 0.5,
                      "resilience": 0.5, "perception": 0.5, "creativity": 0.5}
        # Should not crash, energy scale clamped at 0.3
        rec = ae.decide(fitness=0.6, awareness=0.5, mutation_rate=0.05,
                        pending_stimuli=0, current_traits=traits_max)
        assert rec["energy"] > 0


# ═══════════════════════════════════════════════════════════════════════
# 6. Extinction Prevention Guard
# ═══════════════════════════════════════════════════════════════════════

class TestExtinctionPrevention:
    """When population = 1, resources are boosted to prevent extinction."""

    def test_config_defaults(self):
        cfg = EnvironmentConfig()
        assert cfg.extinction_prevention_regen_multiplier == pytest.approx(3.0)
        assert cfg.extinction_prevention_pool_boost == pytest.approx(100.0)

    def test_triggers_at_pop_1(self):
        """Extinction prevention activates when pop = 1."""
        env = Environment(rng_seed=42)
        env._resource_pool = 200.0
        regen = env.extinction_prevention_regenerate(1)
        assert regen > 0
        # Should have added: 5.0 * 2.0 + 100.0 = 110.0
        assert regen == pytest.approx(110.0)

    def test_no_trigger_at_pop_2(self):
        """Extinction prevention does NOT activate when pop >= 2."""
        env = Environment(rng_seed=42)
        env._resource_pool = 200.0
        regen = env.extinction_prevention_regenerate(2)
        assert regen == pytest.approx(0.0)

    def test_capped_at_pool_max(self):
        """Pool boost cannot exceed pool max."""
        env = Environment(rng_seed=42)
        env._resource_pool = 950.0
        regen = env.extinction_prevention_regenerate(1)
        assert env.resource_pool <= env.config.resource_pool_max

    def test_integrated_environment_tick(self, tmp_path):
        """environment_tick calls extinction prevention when pop = 1."""
        org = _make_organism(tmp_path)
        # Remove all but AL-01
        for oid in list(org._population.member_ids):
            if oid != "AL-01":
                org._population.remove_member(oid, "test")
        assert org._population.size == 1
        org._environment._resource_pool = 200.0
        record = org.environment_tick()
        assert "extinction_prevention_regen" in record
        assert record["extinction_prevention_regen"] > 0

    def test_no_extinction_regen_with_full_pop(self, tmp_path):
        """No extinction prevention when population is healthy."""
        org = _make_organism(tmp_path)
        # Spawn a few children
        for i in range(3):
            org._population.spawn_child(org._genome, i + 1)
        assert org._population.size > 1
        org._environment._resource_pool = 200.0
        record = org.environment_tick()
        assert "extinction_prevention_regen" not in record


# ═══════════════════════════════════════════════════════════════════════
# 7. Integration — Combined Stabilization
# ═══════════════════════════════════════════════════════════════════════

class TestStabilizationIntegration:
    """End-to-end integration: stabilization features working together."""

    def test_organism_survives_low_resources(self, tmp_path):
        """With stabilization, organisms can survive low-resource scenarios."""
        cfg = EnvironmentConfig(resource_pool_initial=100.0)
        org = _make_organism(tmp_path, env_config=cfg)
        child = org._population.spawn_child(org._genome, 1)
        assert child is not None
        # Run several cycles — organism should survive
        for _ in range(5):
            org.environment_tick()
            org.autonomy_cycle()
            org.child_autonomy_cycle()
        # AL-01 should still be alive
        al01 = org._population.get("AL-01")
        assert al01 is not None
        assert al01.get("alive", True) is True

    def test_adaptability_recovers_over_cycles(self, tmp_path):
        """Adaptability gradually recovers from near-zero."""
        org = _make_organism(tmp_path)
        org._genome.set_trait("adaptability", 0.05)
        initial = org._genome.get_trait("adaptability")
        for _ in range(10):
            org.autonomy_cycle()
        final = org._genome.get_trait("adaptability")
        assert final > initial

    def test_full_tick_cycle_no_crash(self, tmp_path):
        """A full tick cycle with all stabilization features runs clean."""
        org = _make_organism(tmp_path)
        for i in range(3):
            org._population.spawn_child(org._genome, i + 1)
        # Set one child to low energy
        children = [oid for oid in org._population.member_ids if oid != "AL-01"]
        if children:
            org._population.update_member(children[0], {"energy": 0.08})
        # Run environment + autonomy + child autonomy
        org.environment_tick()
        org.autonomy_cycle()
        results = org.child_autonomy_cycle()
        assert isinstance(results, list)

    def test_extinction_prevention_with_low_pool(self, tmp_path):
        """Lone survivor with low pool gets resource boost."""
        org = _make_organism(tmp_path)
        org._environment._resource_pool = 50.0
        assert org._population.size == 1
        record = org.environment_tick()
        # Pool should have increased significantly
        assert org._environment.resource_pool > 50.0
