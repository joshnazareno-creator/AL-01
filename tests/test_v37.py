"""AL-01 v3.7 — Tests for trait floor, variance kick, and stagnation breaker.

Covers:
 1. Trait floor prevents energy_efficiency from reaching zero
 2. Trade-off damping protects near-floor traits
 3. Entropy decay respects trait floor
 4. Variance kick fires when fitness variance is zero
 5. Variance kick cooldown prevents rapid re-fire
 6. Variance kick targets the weakest trait
 7. Genome set_trait respects floor
 8. Genome spawn_child respects floor
 9. Trade-off + decay cannot grind trait below floor
10. End-to-end: organism energy_efficiency stays above floor after many cycles
11. Stagnation breaker tier-1 re-triggers exploration
12. Stagnation breaker tier-3 shuffles traits
13. Stagnation breaker hard limit resets traits + clears history
14. Stagnation count clamped below hard limit after reset
15. Variance kick doubles magnitude at tier-2 stagnation
16. Exploration re-triggers every tier-1 interval during stagnation
17. End-to-end: stagnation never exceeds hard limit
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from typing import Any, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from al01.genome import Genome, TRAIT_FLOOR, DEFAULT_ENTROPY_RATE
from al01.autonomy import AutonomyConfig, AutonomyEngine


# ==================================================================
# Trait Floor Tests
# ==================================================================

class TestTraitFloor(unittest.TestCase):
    """Trait floor prevents any trait from reaching zero."""

    def test_init_traits_above_floor(self) -> None:
        """All default traits should be well above the floor."""
        g = Genome()
        for name, val in g.traits.items():
            self.assertGreaterEqual(val, TRAIT_FLOOR, f"{name} below floor")

    def test_set_trait_enforces_floor(self) -> None:
        """set_trait(name, 0.0) should clamp to TRAIT_FLOOR, not zero."""
        g = Genome()
        g.set_trait("energy_efficiency", 0.0)
        self.assertGreaterEqual(g.get_trait("energy_efficiency"), TRAIT_FLOOR)

    def test_set_trait_negative_enforces_floor(self) -> None:
        """set_trait(name, -1.0) should clamp to TRAIT_FLOOR."""
        g = Genome()
        g.set_trait("resilience", -1.0)
        self.assertGreaterEqual(g.get_trait("resilience"), TRAIT_FLOOR)

    def test_decay_respects_floor(self) -> None:
        """Entropy decay cannot push traits below floor."""
        g = Genome(traits={
            "adaptability": TRAIT_FLOOR + 0.001,
            "energy_efficiency": TRAIT_FLOOR + 0.001,
            "resilience": TRAIT_FLOOR + 0.001,
            "perception": TRAIT_FLOOR + 0.001,
            "creativity": TRAIT_FLOOR + 0.001,
        })
        # Decay 100 times — should never go below floor
        for _ in range(100):
            g.decay_traits(rate=0.1)
        for name, val in g.traits.items():
            self.assertGreaterEqual(val, TRAIT_FLOOR, f"{name} below floor after decay")

    def test_spawn_child_respects_floor(self) -> None:
        """Child genome traits should stay above floor."""
        g = Genome(traits={
            "adaptability": TRAIT_FLOOR,
            "energy_efficiency": TRAIT_FLOOR,
            "resilience": TRAIT_FLOOR,
            "perception": TRAIT_FLOOR,
            "creativity": TRAIT_FLOOR,
        })
        for _ in range(20):
            child = g.spawn_child(variance=0.1)
            for name, val in child.traits.items():
                self.assertGreaterEqual(val, TRAIT_FLOOR, f"child {name} below floor")


# ==================================================================
# Trade-off Damping Tests
# ==================================================================

class TestTradeoffDamping(unittest.TestCase):
    """Trade-off penalties are damped when target trait is near floor."""

    def test_tradeoff_cannot_crush_below_floor(self) -> None:
        """High adaptability shouldn't crush energy_efficiency to zero."""
        g = Genome(traits={
            "adaptability": 1.5,   # well above 0.7 threshold
            "energy_efficiency": 0.05,  # near floor
            "resilience": 0.5,
            "perception": 0.5,
            "creativity": 1.5,     # also above threshold
        })
        # Mutate many times — each applies tradeoffs
        for _ in range(50):
            g.mutate()
        self.assertGreaterEqual(
            g.get_trait("energy_efficiency"), TRAIT_FLOOR,
            "energy_efficiency crushed below floor by trade-offs"
        )

    def test_damping_reduces_penalty_near_floor(self) -> None:
        """Verify damping factor is applied when target is near floor."""
        g1 = Genome(traits={
            "adaptability": 1.0,
            "energy_efficiency": 0.5,  # plenty of headroom
            "resilience": 0.5, "perception": 0.5, "creativity": 0.5,
        })
        g2 = Genome(traits={
            "adaptability": 1.0,
            "energy_efficiency": 0.04,  # near floor
            "resilience": 0.5, "perception": 0.5, "creativity": 0.5,
        })
        # Apply tradeoffs to both
        effects1 = g1._apply_tradeoffs()
        effects2 = g2._apply_tradeoffs()
        # The near-floor genome (g2) should have a smaller reduction
        r1 = effects1.get("adaptability->energy_efficiency", {}).get("reduction", 0)
        r2 = effects2.get("adaptability->energy_efficiency", {}).get("reduction", 0)
        self.assertLess(r2, r1, "Damping should reduce penalty near floor")

    def test_repeated_tradeoffs_converge_above_floor(self) -> None:
        """Applying tradeoffs 100 times with high source trait stays above floor."""
        g = Genome(traits={
            "adaptability": 2.0,
            "energy_efficiency": 0.10,
            "resilience": 0.5,
            "perception": 0.5,
            "creativity": 2.0,
        })
        for _ in range(100):
            g._apply_tradeoffs()
        self.assertGreaterEqual(g.get_trait("energy_efficiency"), TRAIT_FLOOR)


# ==================================================================
# Variance Kick Tests
# ==================================================================

class TestVarianceKick(unittest.TestCase):
    """Variance kick breaks zero-variance stagnation lock."""

    def _make_engine(self, **overrides) -> AutonomyEngine:
        tmpdir = tempfile.mkdtemp()
        cfg_args = {
            "stagnation_window": 5,
            "variance_kick_threshold": 1e-6,
            "variance_kick_magnitude": 0.03,
            "variance_kick_cooldown": 3,
        }
        cfg_args.update(overrides)
        cfg = AutonomyConfig(**cfg_args)
        return AutonomyEngine(data_dir=os.path.join(tmpdir, "data"), config=cfg)

    def test_variance_kick_fires_on_flat_history(self) -> None:
        """When fitness history is flat (zero variance), kick should trigger."""
        eng = self._make_engine()
        # Fill fitness_history with identical values
        eng._fitness_history = [0.35] * 10
        g = Genome()
        self.assertTrue(eng.should_variance_kick())
        result = eng.apply_variance_kick(g)
        self.assertIsNotNone(result, "Kick should fire on flat history")
        self.assertIn("trait", result)
        self.assertIn("nudge", result)

    def test_variance_kick_does_not_fire_on_varied_history(self) -> None:
        """When fitness history has variance, kick should NOT fire."""
        eng = self._make_engine()
        eng._fitness_history = [0.3, 0.35, 0.4, 0.45, 0.5]
        g = Genome()
        self.assertFalse(eng.should_variance_kick())
        result = eng.apply_variance_kick(g)
        self.assertIsNone(result, "Kick should not fire on varied history")

    def test_variance_kick_cooldown(self) -> None:
        """After a kick, cooldown prevents immediate re-fire."""
        eng = self._make_engine()
        eng._fitness_history = [0.35] * 10
        g = Genome()
        result = eng.apply_variance_kick(g)
        self.assertIsNotNone(result)
        # Cooldown should be set
        self.assertEqual(eng._variance_kick_cooldown_remaining, 3)
        # Second kick should NOT fire
        result2 = eng.apply_variance_kick(g)
        self.assertIsNone(result2, "Kick should be on cooldown")

    def test_variance_kick_targets_weakest(self) -> None:
        """Kick should nudge the weakest trait."""
        eng = self._make_engine()
        eng._fitness_history = [0.35] * 10
        g = Genome(traits={
            "adaptability": 0.5,
            "energy_efficiency": 0.02,  # weakest
            "resilience": 0.5,
            "perception": 0.5,
            "creativity": 0.5,
        })
        result = eng.apply_variance_kick(g)
        self.assertIsNotNone(result)
        self.assertEqual(result["trait"], "energy_efficiency")
        self.assertGreater(result["new_value"], result["old_value"])

    def test_variance_kick_cooldown_ticks_down(self) -> None:
        """Cooldown decrements each decide() cycle."""
        eng = self._make_engine()
        eng._variance_kick_cooldown_remaining = 3
        traits = {"adaptability": 0.5, "energy_efficiency": 0.5,
                  "resilience": 0.5, "perception": 0.5, "creativity": 0.5}
        # Run decide once — cooldown should decrement
        eng.decide(fitness=0.5, awareness=0.5, mutation_rate=0.1,
                   pending_stimuli=0, current_traits=traits)
        self.assertEqual(eng._variance_kick_cooldown_remaining, 2)

    def test_should_variance_kick_insufficient_history(self) -> None:
        """Kick should not fire with insufficient history samples."""
        eng = self._make_engine()
        eng._fitness_history = [0.35, 0.35]  # only 2, need 5 (window)
        self.assertFalse(eng.should_variance_kick())


# ==================================================================
# End-to-End: Energy Efficiency Stays Above Floor
# ==================================================================

class TestEndToEndTraitFloor(unittest.TestCase):
    """Integration: energy_efficiency doesn't hit zero over many cycles."""

    def test_energy_efficiency_survives_heavy_pressure(self) -> None:
        """With high adaptability/creativity, energy_efficiency stays above floor."""
        g = Genome(traits={
            "adaptability": 1.0,
            "energy_efficiency": 0.15,
            "resilience": 0.5,
            "perception": 0.5,
            "creativity": 1.0,
        })
        for _ in range(200):
            g.mutate()
            g.decay_traits(rate=DEFAULT_ENTROPY_RATE)
        self.assertGreaterEqual(
            g.get_trait("energy_efficiency"), TRAIT_FLOOR,
            f"energy_efficiency fell below floor after 200 cycles"
        )

    def test_all_traits_above_floor_after_stress(self) -> None:
        """Every single trait stays above TRAIT_FLOOR after 200 stress cycles."""
        g = Genome(traits={
            "adaptability": 0.8,
            "energy_efficiency": 0.05,
            "resilience": 0.05,
            "perception": 0.05,
            "creativity": 0.8,
        })
        for _ in range(200):
            g.mutate()
            g.decay_traits(rate=0.01)
        for name, val in g.traits.items():
            self.assertGreaterEqual(val, TRAIT_FLOOR, f"{name} below floor")


# ==================================================================
# Stagnation Breaker Tests
# ==================================================================

class TestStagnationBreaker(unittest.TestCase):
    """Escalating stagnation breaker prevents stagnation > 800."""

    def _make_engine(self, **overrides) -> AutonomyEngine:
        tmpdir = tempfile.mkdtemp()
        cfg_args = {
            "stagnation_window": 5,
            "stagnation_tier1_threshold": 50,
            "stagnation_tier2_threshold": 200,
            "stagnation_tier3_threshold": 500,
            "stagnation_hard_limit": 800,
            "stagnation_hard_reset_spread": 0.15,
            "stagnation_tier3_shuffle_magnitude": 0.10,
            "variance_kick_threshold": 1e-6,
            "variance_kick_magnitude": 0.03,
            "variance_kick_cooldown": 3,
        }
        cfg_args.update(overrides)
        cfg = AutonomyConfig(**cfg_args)
        return AutonomyEngine(data_dir=os.path.join(tmpdir, "data"), config=cfg)

    def test_below_tier1_no_action(self) -> None:
        """Stagnation below 50 should not trigger breaker."""
        eng = self._make_engine()
        eng._stagnation_count = 30
        g = Genome()
        result = eng.break_stagnation(g)
        self.assertIsNone(result)

    def test_tier1_retriggers_exploration(self) -> None:
        """At stagnation=50, exploration should be re-triggered."""
        eng = self._make_engine()
        eng._stagnation_count = 50
        eng._exploration_mode = False
        g = Genome()
        result = eng.break_stagnation(g)
        self.assertIsNotNone(result)
        self.assertEqual(result["tier"], 1)
        self.assertEqual(result["action"], "exploration_retrigger")
        self.assertTrue(eng._exploration_mode)

    def test_tier1_at_100(self) -> None:
        """At stagnation=100, exploration re-triggers again."""
        eng = self._make_engine()
        eng._stagnation_count = 100
        eng._exploration_mode = False
        g = Genome()
        result = eng.break_stagnation(g)
        self.assertIsNotNone(result)
        self.assertEqual(result["tier"], 1)
        self.assertEqual(result["action"], "exploration_retrigger")

    def test_tier3_shuffles_all_traits(self) -> None:
        """At stagnation=500, all traits should be shuffled."""
        eng = self._make_engine()
        eng._stagnation_count = 500
        g = Genome(traits={
            "adaptability": 0.3, "energy_efficiency": 0.3,
            "resilience": 0.3, "perception": 0.3, "creativity": 0.3,
        })
        old_traits = dict(g.traits)
        result = eng.break_stagnation(g)
        self.assertIsNotNone(result)
        self.assertEqual(result["tier"], 3)
        self.assertEqual(result["action"], "trait_shuffle")
        # At least some traits should have changed
        new_traits = g.traits
        changes = sum(1 for t in old_traits if abs(old_traits[t] - new_traits[t]) > 1e-6)
        self.assertGreater(changes, 0, "Tier 3 should shuffle at least some traits")

    def test_hard_limit_resets_traits_and_history(self) -> None:
        """At stagnation=800, traits reset to near-defaults + history cleared."""
        eng = self._make_engine()
        eng._stagnation_count = 800
        eng._fitness_history = [0.35] * 10
        g = Genome(traits={
            "adaptability": 0.1, "energy_efficiency": 0.02,
            "resilience": 0.1, "perception": 0.1, "creativity": 0.1,
        })
        result = eng.break_stagnation(g)
        self.assertIsNotNone(result)
        self.assertEqual(result["tier"], 4)
        self.assertEqual(result["action"], "hard_reset")
        # Stagnation should be reset to 0
        self.assertEqual(eng._stagnation_count, 0)
        # Fitness history should be cleared
        self.assertEqual(len(eng._fitness_history), 0)
        # Traits should be near 0.5 (defaults ± 0.15)
        for name, val in g.traits.items():
            self.assertGreater(val, 0.2, f"{name} too low after hard reset")
            self.assertLess(val, 0.8, f"{name} too high after hard reset")

    def test_hard_limit_prevents_exceeding_1000(self) -> None:
        """After hard reset at 800, stagnation should never reach 1000."""
        eng = self._make_engine()
        g = Genome()
        # Simulate 1500 cycles of flat fitness
        max_stag_seen = 0
        for i in range(1500):
            eng._fitness_history = [0.35] * 10  # force flat
            eng._stagnation_count += 1
            max_stag_seen = max(max_stag_seen, eng._stagnation_count)
            eng.break_stagnation(g)
        self.assertLess(max_stag_seen, 1000,
                        f"Stagnation reached {max_stag_seen} — should be < 1000")


class TestVarianceKickTier2(unittest.TestCase):
    """Variance kick doubles magnitude during tier-2 stagnation."""

    def _make_engine(self) -> AutonomyEngine:
        tmpdir = tempfile.mkdtemp()
        cfg = AutonomyConfig(
            stagnation_window=5,
            stagnation_tier2_threshold=200,
            variance_kick_threshold=1e-6,
            variance_kick_magnitude=0.03,
            variance_kick_cooldown=3,
        )
        return AutonomyEngine(data_dir=os.path.join(tmpdir, "data"), config=cfg)

    def test_normal_magnitude_below_tier2(self) -> None:
        """Below tier-2 stagnation, variance kick uses normal magnitude."""
        eng = self._make_engine()
        eng._stagnation_count = 50
        eng._fitness_history = [0.35] * 10
        g = Genome()
        result = eng.apply_variance_kick(g)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["nudge"], 0.03, places=4)

    def test_doubled_magnitude_at_tier2(self) -> None:
        """At tier-2 stagnation (200+), variance kick magnitude doubles."""
        eng = self._make_engine()
        eng._stagnation_count = 200
        eng._fitness_history = [0.35] * 10
        g = Genome()
        result = eng.apply_variance_kick(g)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["nudge"], 0.06, places=4)


class TestExplorationRetrigger(unittest.TestCase):
    """Exploration mode re-triggers periodically during deep stagnation."""

    def _make_engine(self) -> AutonomyEngine:
        tmpdir = tempfile.mkdtemp()
        cfg = AutonomyConfig(
            stagnation_window=5,
            stagnation_tier1_threshold=50,
            stagnation_exploration_cycles=5,
        )
        return AutonomyEngine(data_dir=os.path.join(tmpdir, "data"), config=cfg)

    def test_exploration_retriggers_in_decide(self) -> None:
        """During deep stagnation, exploration should re-trigger periodically."""
        eng = self._make_engine()
        # Force stagnation state = 49, next cycle makes it 50
        eng._fitness_history = [0.35] * 10
        eng._stagnation_count = 49
        # Run one decide — should increment stagnation to 50 and retrigger
        traits = {"adaptability": 0.5, "energy_efficiency": 0.5,
                  "resilience": 0.5, "perception": 0.5, "creativity": 0.5}
        record = eng.decide(fitness=0.35, awareness=0.5, mutation_rate=0.1,
                           pending_stimuli=0, current_traits=traits)
        self.assertEqual(eng._stagnation_count, 50)
        # Exploration should have been triggered
        self.assertTrue(eng._exploration_mode)


class TestEndToEndStagnationLimit(unittest.TestCase):
    """Integration: stagnation never exceeds hard limit over extended simulation."""

    def test_stagnation_capped_simulation(self) -> None:
        """Simulate 2000 decide cycles with flat fitness — stagnation stays < 1000."""
        tmpdir = tempfile.mkdtemp()
        cfg = AutonomyConfig(
            stagnation_window=5,
            stagnation_tier1_threshold=50,
            stagnation_tier3_threshold=500,
            stagnation_hard_limit=800,
        )
        eng = AutonomyEngine(
            data_dir=os.path.join(tmpdir, "data"), config=cfg,
        )
        g = Genome()
        max_stag = 0
        traits = dict(g.traits)
        for _ in range(2000):
            eng.decide(fitness=0.35, awareness=0.3, mutation_rate=0.1,
                       pending_stimuli=0, current_traits=traits)
            # Apply breaker
            eng.break_stagnation(g)
            traits = dict(g.traits)
            max_stag = max(max_stag, eng.stagnation_count)
        self.assertLess(max_stag, 1000,
                        f"Max stagnation was {max_stag} — should be < 1000")


if __name__ == "__main__":
    unittest.main()
