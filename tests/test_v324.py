"""v3.24 — Wire missing reproduction paths into MetabolismScheduler.

Bug:
  lone_survivor_reproduction(), stability_reproduction_cycle(), and
  wake_dormant_cycle() were defined on Organism but never called by the
  MetabolismScheduler.tick() method.  AL-01 could never reproduce as a
  lone survivor because the code path was dead.

Fix:
  Added all three calls to MetabolismScheduler.tick() on the
  auto_reproduce_interval, after the existing auto_reproduce_cycle() call.
"""

import tempfile
from unittest.mock import MagicMock, patch, call

import pytest

from al01.database import Database
from al01.organism import Organism, MetabolismConfig, MetabolismScheduler


def _make_organism(tmp_path) -> Organism:
    d = str(tmp_path)
    db = Database(db_path=f"{d}/t.db")
    from al01.memory_manager import MemoryManager

    mm = MemoryManager(data_dir=d, credential_path=None, database=db)
    return Organism(data_dir=d, memory_manager=mm)


# ═══════════════════════════════════════════════════════════════════════
# 1. Scheduler wiring — methods are actually invoked
# ═══════════════════════════════════════════════════════════════════════

class TestSchedulerWiring:
    """Verify that all reproduction paths are called by the scheduler."""

    def test_lone_survivor_called_on_auto_repro_interval(self, tmp_path):
        """lone_survivor_reproduction() is invoked every auto_reproduce_interval ticks."""
        org = _make_organism(tmp_path)
        interval = org.config.auto_reproduce_interval
        with patch.object(org, "lone_survivor_reproduction") as mock_ls:
            # Run enough ticks to hit the interval
            for _ in range(interval):
                org.scheduler.tick()
            assert mock_ls.call_count >= 1

    def test_stability_reproduction_called_on_auto_repro_interval(self, tmp_path):
        """stability_reproduction_cycle() is invoked every auto_reproduce_interval ticks."""
        org = _make_organism(tmp_path)
        interval = org.config.auto_reproduce_interval
        with patch.object(org, "stability_reproduction_cycle") as mock_sr:
            for _ in range(interval):
                org.scheduler.tick()
            assert mock_sr.call_count >= 1

    def test_wake_dormant_called_on_auto_repro_interval(self, tmp_path):
        """wake_dormant_cycle() is invoked every auto_reproduce_interval ticks."""
        org = _make_organism(tmp_path)
        interval = org.config.auto_reproduce_interval
        with patch.object(org, "wake_dormant_cycle") as mock_wd:
            for _ in range(interval):
                org.scheduler.tick()
            assert mock_wd.call_count >= 1

    def test_auto_reproduce_still_called(self, tmp_path):
        """auto_reproduce_cycle() is still called (not regressed)."""
        org = _make_organism(tmp_path)
        interval = org.config.auto_reproduce_interval
        with patch.object(org, "auto_reproduce_cycle") as mock_ar:
            for _ in range(interval):
                org.scheduler.tick()
            assert mock_ar.call_count >= 1

    def test_rare_reproduction_still_called(self, tmp_path):
        """rare_reproduction_cycle() is still called on its own interval."""
        org = _make_organism(tmp_path)
        interval = org.config.rare_reproduce_interval
        with patch.object(org, "rare_reproduction_cycle") as mock_rr:
            for _ in range(interval):
                org.scheduler.tick()
            assert mock_rr.call_count >= 1


# ═══════════════════════════════════════════════════════════════════════
# 2. Ordering — death before reproduction
# ═══════════════════════════════════════════════════════════════════════

class TestReproductionOrdering:
    """Reproduction calls happen after child_autonomy (death resolution)."""

    def test_child_autonomy_before_lone_survivor(self, tmp_path):
        """child_autonomy_cycle runs before lone_survivor_reproduction."""
        org = _make_organism(tmp_path)
        call_order = []
        with patch.object(org, "child_autonomy_cycle", side_effect=lambda: call_order.append("child_autonomy")):
            with patch.object(org, "lone_survivor_reproduction", side_effect=lambda: call_order.append("lone_survivor")):
                # Use LCM of both intervals to ensure both fire
                interval = org.config.auto_reproduce_interval
                child_interval = org.config.child_autonomy_interval
                lcm = interval * child_interval  # guaranteed to hit both
                for _ in range(lcm):
                    org.scheduler.tick()
        assert "child_autonomy" in call_order
        assert "lone_survivor" in call_order
        # child_autonomy must appear before lone_survivor
        first_child = call_order.index("child_autonomy")
        first_lone = call_order.index("lone_survivor")
        assert first_child < first_lone

    def test_child_autonomy_before_stability_repro(self, tmp_path):
        """child_autonomy_cycle runs before stability_reproduction_cycle."""
        org = _make_organism(tmp_path)
        call_order = []
        with patch.object(org, "child_autonomy_cycle", side_effect=lambda: call_order.append("child_autonomy")):
            with patch.object(org, "stability_reproduction_cycle", side_effect=lambda: call_order.append("stability_repro")):
                interval = org.config.auto_reproduce_interval
                child_interval = org.config.child_autonomy_interval
                lcm = interval * child_interval
                for _ in range(lcm):
                    org.scheduler.tick()
        assert "child_autonomy" in call_order
        assert "stability_repro" in call_order
        first_child = call_order.index("child_autonomy")
        first_stab = call_order.index("stability_repro")
        assert first_child < first_stab
