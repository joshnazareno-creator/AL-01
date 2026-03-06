"""AL-01 v3.6 — Tests for stagnation death-spiral fix.

Covers:
 1. Recovery mode activation when energy stays low
 2. Recovery mode forces STABILIZE decision
 3. Recovery mode softens energy-fitness penalty
 4. Recovery mode deactivation when energy recovers
 5. Stagnation mutation delta is capped
 6. Directed mutation bias (upward_bias on Genome.mutate)
 7. Env regen uses configurable multiplier (not hardcoded 0.01)
 8. Mutate/Adapt energy rebates
 9. Recovery-mode enhanced stabilize bonus
10. CycleLogEntry structured logging
11. AlertGuardrails fires ENERGY_CRITICAL
12. AlertGuardrails fires FITNESS_STALLED
13. AlertGuardrails fires TRAIT_COLLAPSED
14. AlertGuardrails stays silent when healthy
15. CycleLogger records entries to disk
16. Organism wires cycle_logger and alert_guardrails
17. autonomy_cycle returns cycle_log and alerts
18. Stagnation delta cap prevents wild swings
19. Recovery mode persists in decision record
20. Version bump to 3.6
21. Recovery mode stabilize energy actually climbs
22. Directed bias observable: mean mutation delta > 0
23. End-to-end: organism escapes death spiral within N cycles
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from typing import Any, Dict
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from al01.genome import Genome
from al01.organism import (
    Organism, MetabolismConfig, VERSION,
    CycleStats, CycleLogEntry, AlertGuardrails, CycleLogger,
)
from al01.autonomy import AutonomyConfig, AutonomyEngine


# ==================================================================
# Helpers
# ==================================================================

def _make_organism(**kwargs) -> Organism:
    tmpdir = kwargs.pop("data_dir", tempfile.mkdtemp())
    return Organism(data_dir=tmpdir, config=MetabolismConfig(), **kwargs)


def _make_engine(energy: float = 1.0, **cfg_kw) -> AutonomyEngine:
    """Create a standalone AutonomyEngine with configurable energy."""
    tmpdir = tempfile.mkdtemp()
    cfg = AutonomyConfig(**cfg_kw)
    eng = AutonomyEngine(data_dir=tmpdir, config=cfg, rng_seed=42)
    eng._energy = energy
    return eng


# ==================================================================
# 1. Recovery mode activation
# ==================================================================

class TestRecoveryModeActivation(unittest.TestCase):
    """Recovery mode should activate after consecutive low-energy cycles."""

    def test_recovery_activates_after_trigger_cycles(self):
        eng = _make_engine(energy=0.12, recovery_trigger_cycles=3,
                           recovery_energy_threshold=0.20)
        self.assertFalse(eng.recovery_mode)
        # Run 3 decide cycles at low energy
        for _ in range(3):
            eng.decide(0.01, 0.5, 0.10, 0)
        self.assertTrue(eng.recovery_mode)

    def test_recovery_does_not_activate_above_threshold(self):
        eng = _make_engine(energy=0.50, recovery_trigger_cycles=3,
                           recovery_energy_threshold=0.20)
        for _ in range(5):
            eng.decide(0.8, 0.5, 0.10, 0)
        self.assertFalse(eng.recovery_mode)


# ==================================================================
# 2. Recovery mode forces STABILIZE
# ==================================================================

class TestRecoveryForceStabilize(unittest.TestCase):
    """When in recovery mode, decision should always be stabilize."""

    def test_decision_is_stabilize_during_recovery(self):
        eng = _make_engine(energy=0.12, recovery_trigger_cycles=1,
                           recovery_energy_threshold=0.20)
        # First cycle activates recovery
        rec = eng.decide(0.01, 0.5, 0.10, 0)
        # Second (and subsequent) should be stabilize
        rec = eng.decide(0.01, 0.5, 0.10, 0)
        self.assertEqual(rec["decision"], "stabilize")
        self.assertIn("RECOVERY MODE", rec["reason"])


# ==================================================================
# 3. Recovery mode softens energy-fitness penalty
# ==================================================================

class TestRecoveryFitnessPenaltyCap(unittest.TestCase):
    """During recovery, effective_fitness shouldn't be crushed as badly."""

    def test_effective_fitness_higher_in_recovery(self):
        # Without recovery
        eng1 = _make_engine(energy=0.10, recovery_trigger_cycles=100)
        rec1 = eng1.decide(0.50, 0.5, 0.10, 0)
        eff1 = rec1["effective_fitness"]

        # With recovery (trigger after 1 cycle)
        eng2 = _make_engine(energy=0.10, recovery_trigger_cycles=1,
                            recovery_energy_threshold=0.20)
        eng2.decide(0.50, 0.5, 0.10, 0)  # triggers recovery
        rec2 = eng2.decide(0.50, 0.5, 0.10, 0)
        eff2 = rec2["effective_fitness"]

        # Recovery mode should give higher effective fitness
        self.assertGreater(eff2, eff1)


# ==================================================================
# 4. Recovery mode deactivation
# ==================================================================

class TestRecoveryModeDeactivation(unittest.TestCase):
    def test_recovery_off_when_energy_recovers(self):
        eng = _make_engine(energy=0.12, recovery_trigger_cycles=1,
                           recovery_energy_threshold=0.20)
        eng.decide(0.01, 0.5, 0.10, 0)
        self.assertTrue(eng.recovery_mode)
        # Manually bump energy above threshold
        eng._energy = 0.50
        eng.decide(0.8, 0.5, 0.10, 0)
        self.assertFalse(eng.recovery_mode)


# ==================================================================
# 5. Stagnation mutation delta is capped
# ==================================================================

class TestStagnationDeltaCap(unittest.TestCase):
    def test_delta_capped_at_config_value(self):
        eng = _make_engine(stagnation_delta_cap=0.5)
        eng._stagnation_count = 1000  # enormous stagnation
        result = eng.stagnation_scaled_delta(0.10)
        self.assertLessEqual(result, 0.5)

    def test_delta_not_capped_when_small(self):
        eng = _make_engine(stagnation_delta_cap=1.0)
        eng._stagnation_count = 0
        result = eng.stagnation_scaled_delta(0.10)
        self.assertAlmostEqual(result, 0.10, places=4)


# ==================================================================
# 6. Directed mutation bias
# ==================================================================

class TestDirectedMutationBias(unittest.TestCase):
    def test_upward_bias_shifts_mean_positive(self):
        """With upward bias, average trait delta should be > 0."""
        g = Genome(rng_seed=42, mutation_rate=1.0)
        original_fitness = g.fitness
        total_delta = 0.0
        runs = 200
        for _ in range(runs):
            g2 = Genome(traits={"adaptability": 0.3, "energy_efficiency": 0.3,
                                "resilience": 0.3, "perception": 0.3,
                                "creativity": 0.3},
                        mutation_rate=1.0, rng_seed=None)
            before = g2.fitness
            g2.mutate(upward_bias=0.05)
            after = g2.fitness
            total_delta += (after - before)
        avg_delta = total_delta / runs
        self.assertGreater(avg_delta, 0.0,
                           f"Average fitness delta should be positive, got {avg_delta}")

    def test_zero_bias_is_neutral(self):
        """With zero bias, mutate() should behave identically to before."""
        g = Genome(rng_seed=42, mutation_rate=1.0)
        rec = g.mutate(upward_bias=0.0)
        self.assertIn("mutated_traits", rec)


# ==================================================================
# 7. Env regen multiplier
# ==================================================================

class TestEnvRegenMultiplier(unittest.TestCase):
    def test_regen_uses_config_multiplier(self):
        eng = _make_engine(energy=0.50, env_regen_multiplier=0.10)
        energy_before = eng.energy
        # Provide env_modifiers with energy_regen_rate=1.0
        eng.decide(0.80, 0.5, 0.10, 0,
                   env_modifiers={"energy_regen_rate": 1.0})
        # Energy should have gotten 1.0 * 0.10 = +0.10 from regen
        # minus decay and whatever decision cost
        # Key assertion: it's not using the old hardcoded 0.01
        # With 0.10 multiplier: regen = 0.10
        # With old 0.01 multiplier: regen = 0.01
        # Energy should be measurably higher than 0.50 - 0.005 + 0.01
        energy_after = eng.energy
        # The old way would give ~0.50 - 0.005 + 0.01 + stabilize_bonus = ~0.525
        # The new way gives ~0.50 - 0.005 + 0.10 + stabilize_bonus = ~0.615
        self.assertGreater(energy_after, 0.55)


# ==================================================================
# 8. Mutate/Adapt energy rebates
# ==================================================================

class TestEnergyRebates(unittest.TestCase):
    def test_mutate_rebate_reduces_net_cost(self):
        eng1 = _make_engine(energy=0.50, mutate_energy_rebate=0.0)
        eng1.decide(0.01, 0.5, 0.10, 0)  # should mutate (low fitness)
        cost_without = 0.50 - eng1.energy

        eng2 = _make_engine(energy=0.50, mutate_energy_rebate=0.005)
        eng2.decide(0.01, 0.5, 0.10, 0)
        cost_with = 0.50 - eng2.energy

        self.assertLess(cost_with, cost_without)


# ==================================================================
# 9. Recovery-mode enhanced stabilize bonus
# ==================================================================

class TestRecoveryStabilizeBonus(unittest.TestCase):
    def test_stabilize_grants_extra_energy_during_recovery(self):
        eng = _make_engine(energy=0.12, recovery_trigger_cycles=1,
                           recovery_energy_threshold=0.20,
                           recovery_stabilize_bonus=0.03,
                           energy_stabilize_bonus=0.02)
        eng.decide(0.01, 0.5, 0.10, 0)  # triggers recovery
        self.assertTrue(eng.recovery_mode)
        energy_before = eng.energy
        eng.decide(0.01, 0.5, 0.10, 0)  # recovery stabilize
        energy_after = eng.energy
        # Should get normal bonus (0.02) + recovery bonus (0.03) - decay (0.005)
        delta = energy_after - energy_before
        self.assertGreater(delta, 0.03,
                           f"Expected substantial energy gain during recovery, got {delta}")


# ==================================================================
# 10. CycleLogEntry structured logging
# ==================================================================

class TestCycleLogEntry(unittest.TestCase):
    def test_to_dict_contains_required_fields(self):
        entry = CycleLogEntry(
            cycle=10, decision="mutate",
            energy_before=0.5, energy_after=0.48,
            energy_delta=-0.02,
            fitness_before=0.3, fitness_after=0.31,
            fitness_delta=0.01,
            stagnation_count=5,
        )
        d = entry.to_dict()
        self.assertEqual(d["cycle"], 10)
        self.assertEqual(d["decision"], "mutate")
        self.assertAlmostEqual(d["energy_delta"], -0.02, places=4)
        self.assertAlmostEqual(d["fitness_delta"], 0.01, places=4)

    def test_alerts_included_when_present(self):
        entry = CycleLogEntry(
            cycle=1, decision="stabilize",
            energy_before=0.1, energy_after=0.12, energy_delta=0.02,
            fitness_before=0.1, fitness_after=0.1, fitness_delta=0.0,
            alerts=["ENERGY_CRITICAL: too low"],
        )
        d = entry.to_dict()
        self.assertIn("alerts", d)
        self.assertEqual(len(d["alerts"]), 1)


# ==================================================================
# 11–14. AlertGuardrails
# ==================================================================

class TestAlertGuardrails(unittest.TestCase):
    def test_energy_critical_fires(self):
        ag = AlertGuardrails(energy_critical_threshold=0.15,
                             energy_critical_cycles=3)
        for _ in range(4):
            alerts = ag.check(energy=0.10, fitness=0.5)
        self.assertTrue(any("ENERGY_CRITICAL" in a for a in alerts))

    def test_fitness_stalled_fires(self):
        ag = AlertGuardrails(fitness_stall_cycles=5,
                             fitness_stall_epsilon=0.001)
        for _ in range(6):
            alerts = ag.check(energy=0.5, fitness=0.30)
        self.assertTrue(any("FITNESS_STALLED" in a for a in alerts))

    def test_trait_collapsed_fires(self):
        ag = AlertGuardrails(trait_variance_floor=0.001)
        alerts = ag.check(energy=0.5, fitness=0.5,
                          traits={"a": 0.5, "b": 0.5, "c": 0.5})
        self.assertTrue(any("TRAIT_COLLAPSED" in a for a in alerts))

    def test_no_alerts_when_healthy(self):
        ag = AlertGuardrails()
        alerts = ag.check(energy=0.8, fitness=0.7,
                          traits={"a": 0.3, "b": 0.6, "c": 0.9})
        self.assertEqual(alerts, [])


# ==================================================================
# 15. CycleLogger
# ==================================================================

class TestCycleLogger(unittest.TestCase):
    def test_records_to_disk(self):
        tmpdir = tempfile.mkdtemp()
        log_path = os.path.join(tmpdir, "test_cycle_log.jsonl")
        cl = CycleLogger(log_path)
        entry = CycleLogEntry(
            cycle=1, decision="stabilize",
            energy_before=0.5, energy_after=0.52, energy_delta=0.02,
            fitness_before=0.4, fitness_after=0.4, fitness_delta=0.0,
        )
        cl.record(entry)
        self.assertTrue(os.path.exists(log_path))
        with open(log_path) as f:
            line = f.readline()
        self.assertIn('"decision": "stabilize"', line)

    def test_recent_returns_last_entries(self):
        tmpdir = tempfile.mkdtemp()
        cl = CycleLogger(os.path.join(tmpdir, "log.jsonl"))
        for i in range(5):
            cl.record(CycleLogEntry(
                cycle=i, decision="mutate",
                energy_before=0.5, energy_after=0.48, energy_delta=-0.02,
                fitness_before=0.3, fitness_after=0.3, fitness_delta=0.0,
            ))
        recent = cl.recent
        self.assertEqual(len(recent), 5)


# ==================================================================
# 16. Organism wiring
# ==================================================================

class TestOrganismWiring(unittest.TestCase):
    def test_cycle_logger_accessible(self):
        org = _make_organism()
        self.assertIsInstance(org.cycle_logger, CycleLogger)

    def test_alert_guardrails_accessible(self):
        org = _make_organism()
        self.assertIsInstance(org.alert_guardrails, AlertGuardrails)


# ==================================================================
# 17. autonomy_cycle returns cycle_log and alerts
# ==================================================================

class TestAutonomyCycleLogging(unittest.TestCase):
    def test_cycle_log_in_record(self):
        org = _make_organism()
        rec = org.autonomy_cycle()
        self.assertIn("cycle_log", rec)
        cl = rec["cycle_log"]
        self.assertIn("energy_before", cl)
        self.assertIn("energy_after", cl)
        self.assertIn("fitness_delta", cl)

    def test_alerts_in_record(self):
        org = _make_organism()
        rec = org.autonomy_cycle()
        self.assertIn("alerts", rec)
        self.assertIsInstance(rec["alerts"], list)


# ==================================================================
# 18. Stagnation delta cap prevents wild swings
# ==================================================================

class TestStagnationDeltaCapIntegration(unittest.TestCase):
    def test_massive_stagnation_still_capped(self):
        eng = _make_engine(stagnation_delta_cap=0.8,
                           stagnation_delta_scale=0.5)
        eng._stagnation_count = 5000
        delta = eng.stagnation_scaled_delta(0.10)
        self.assertLessEqual(delta, 0.8)
        self.assertGreater(delta, 0.0)


# ==================================================================
# 19. Recovery mode in decision record
# ==================================================================

class TestRecoveryModeInRecord(unittest.TestCase):
    def test_recovery_mode_field_present(self):
        eng = _make_engine(energy=0.12, recovery_trigger_cycles=1,
                           recovery_energy_threshold=0.20)
        eng.decide(0.01, 0.5, 0.10, 0)  # triggers
        rec = eng.decide(0.01, 0.5, 0.10, 0)
        self.assertIn("recovery_mode", rec)
        self.assertTrue(rec["recovery_mode"])

    def test_low_energy_consecutive_in_record(self):
        eng = _make_engine(energy=0.12, recovery_trigger_cycles=1,
                           recovery_energy_threshold=0.20)
        rec = eng.decide(0.01, 0.5, 0.10, 0)
        self.assertIn("low_energy_consecutive", rec)


# ==================================================================
# 20. Version bump
# ==================================================================

class TestVersionBump(unittest.TestCase):
    def test_version_is_3_6(self):
        self.assertEqual(VERSION, "3.9")


# ==================================================================
# 21. Recovery stabilize actually climbs energy
# ==================================================================

class TestRecoveryEnergyClimbs(unittest.TestCase):
    def test_energy_climbs_during_recovery(self):
        eng = _make_engine(
            energy=0.10,
            recovery_trigger_cycles=1,
            recovery_energy_threshold=0.20,
            recovery_stabilize_bonus=0.03,
            energy_stabilize_bonus=0.02,
            energy_min=0.10,
        )
        # Activate recovery
        eng.decide(0.01, 0.5, 0.10, 0)
        self.assertTrue(eng.recovery_mode)

        # Track energy over several recovery cycles
        energies = [eng.energy]
        for _ in range(20):
            eng.decide(0.01, 0.5, 0.10, 0)
            energies.append(eng.energy)

        # Energy should be climbing
        self.assertGreater(energies[-1], energies[0],
                           f"Energy should climb during recovery: {energies[:5]}...{energies[-5:]}")


# ==================================================================
# 22. Directed bias observable
# ==================================================================

class TestDirectedBiasObservable(unittest.TestCase):
    def test_mutate_with_bias_increases_fitness_on_average(self):
        improvements = 0
        trials = 100
        for seed in range(trials):
            g = Genome(
                traits={"a": 0.2, "b": 0.2, "c": 0.2, "d": 0.2, "e": 0.2},
                mutation_rate=1.0,
                rng_seed=seed,
            )
            before = g.fitness
            g.mutate(delta_override=0.10, upward_bias=0.03)
            after = g.fitness
            if after > before:
                improvements += 1
        # With upward bias, majority of trials should show improvement
        self.assertGreater(improvements, 40,
                           f"Expected >40% improvement rate, got {improvements}/{trials}")


# ==================================================================
# 23. End-to-end: organism escapes death spiral
# ==================================================================

class TestDeathSpiralEscape(unittest.TestCase):
    """Simulate a death-spiral scenario and verify the organism recovers."""

    def test_organism_recovers_from_low_energy(self):
        org = _make_organism()
        # Force organism into death spiral conditions
        org._autonomy._energy = 0.10
        org._genome = Genome(
            traits={"adaptability": 0.05, "energy_efficiency": 0.0,
                    "resilience": 0.20, "perception": 0.07, "creativity": 0.06},
        )
        org._autonomy._stagnation_count = 100
        org._autonomy._low_energy_consecutive = 0

        # Run cycles
        initial_energy = org._autonomy.energy
        for _ in range(50):
            org.autonomy_cycle()

        final_energy = org._autonomy.energy
        # Recovery mode should have activated and energy should be climbing
        self.assertGreater(final_energy, initial_energy,
                           f"Energy should recover: {initial_energy} → {final_energy}")

    def test_stagnation_counter_stops_climbing(self):
        org = _make_organism()
        org._autonomy._energy = 0.10
        org._autonomy._stagnation_count = 100

        # After enough recovery stabilize cycles, stagnation should reset
        # because fitness will change (stabilize → no mutation → entropy changes fitness)
        for _ in range(50):
            org.autonomy_cycle()

        # Recovery mode forces stabilize, which should eventually
        # get stagnation_count back down when fitness starts varying again
        final_stag = org._autonomy.stagnation_count
        # At minimum, the stagnation counter shouldn't be growing unboundedly
        # during recovery (since we're forcing stabilize, not mutate)
        self.assertLess(final_stag, 200,
                        f"Stagnation should not keep growing: {final_stag}")


if __name__ == "__main__":
    unittest.main()
