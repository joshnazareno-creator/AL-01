"""AL-01 v3.0 — Population: multi-organism tracking with selection pressure.

v3.0 additions over v2.0:
* **Generation ID** per member (parent=0, children=parent+1).
* **Genome hash** (SHA-256) for lineage fingerprinting.
* **Independent fitness history** per member — separate trajectory.
* **Independent mutation_rate_offset** per member.
* **Death / removal** — ``remove_member()`` for energy-death or pruning.
* **Population pruning** — ``prune_weakest(max_size)`` drops lowest-fitness.
* **Trait variance** — ``trait_variance()`` across all living members.
* **Autonomous reproduction** — ``auto_reproduce()`` checks fitness
  threshold for N consecutive cycles without manual trigger.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import statistics
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from al01.genome import Genome

logger = logging.getLogger("al01.population")

# v3.13: Absolute hard ceiling that NO code path may exceed.
# This includes genesis vault reseeds, portable imports, and
# experiment config overrides.  AL-01 (parent) counts toward this.
ABSOLUTE_POPULATION_CAP: int = 60

# v3.17: Minimum cycles between births for any single parent.
# Applies to all reproduction paths (auto, rare, stability, lone-survivor).
BIRTH_COOLDOWN_CYCLES: int = 500

# v3.11: Species divergence threshold — Euclidean genome distance
SPECIES_DIVERGENCE_THRESHOLD: float = 0.35

# v3.26: Graveyard cap — maximum dead organisms kept in archive
GRAVEYARD_CAP: int = 200


class LifecycleState:
    """Canonical lifecycle states for organisms (v3.26).

    active   — alive and currently simulated
    sleeping — alive, low-activity conservation, auto-recoverable
    dormant  — suspended survival state, recoverable only under
               specific recovery rules
    dead     — terminal state, never auto-recovers
    """
    ACTIVE = "active"
    SLEEPING = "sleeping"
    DORMANT = "dormant"
    DEAD = "dead"

    _SIMULATED = frozenset({"active", "sleeping"})

    @classmethod
    def is_simulated(cls, state: str) -> bool:
        """Active or sleeping — participates in tick simulation."""
        return state in cls._SIMULATED

    @classmethod
    def is_alive(cls, state: str) -> bool:
        """Not dead (active, sleeping, or dormant)."""
        return state != cls.DEAD

    @classmethod
    def can_reproduce(cls, state: str) -> bool:
        """Only active organisms can reproduce."""
        return state == cls.ACTIVE


def _genome_hash(traits: Dict[str, float]) -> str:
    """SHA-256 hash of trait values, truncated to 16 hex chars."""
    canonical = json.dumps(
        {k: round(v, 6) for k, v in sorted(traits.items())},
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class Population:
    """Local population registry backed by ``population.json``.

    v3.0: each member record now includes:
    * generation_id, genome_hash, energy, fitness_history,
      mutation_rate_offset, alive flag, death info.
    """

    def __init__(self, data_dir: str = ".", parent_id: str = "AL-01", rng_seed: Optional[int] = None) -> None:
        self._data_dir = data_dir
        self._parent_id = parent_id
        self._path = os.path.join(data_dir, "population.json")
        self._lock = threading.RLock()
        self._child_counter: int = 0
        self._max_population: int = ABSOLUTE_POPULATION_CAP  # hard cap, overridable via Organism (clamped to ABSOLUTE_POPULATION_CAP)

        # v3.14: Population floor — never allow pop below 20% of max
        # unless hardcore_extinction_mode is explicitly enabled.
        self._hardcore_extinction_mode: bool = False

        # v3.15: Dormant organisms — soft-failure instead of death
        # Dormant organisms don't participate in reproduction or decisions
        # but are preserved for recovery when conditions improve.
        self._dormant_metabolic_fraction: float = 0.1  # dormant organisms use 10% metabolic cost

        # v3.17: Reproduction safety — birth lock + event idempotency
        self._current_tick: int = 0
        self._processed_repro_events: set = set()

        # Seeded RNG for reproducibility — interactions and spawning use this
        self._rng = random.Random(rng_seed)

        os.makedirs(data_dir, exist_ok=True)
        self._members: Dict[str, Dict[str, Any]] = self._load()

        # v3.26: Dead organism archive (persisted separately)
        self._graveyard_path = os.path.join(data_dir, "graveyard.json")
        self._graveyard: Dict[str, Dict[str, Any]] = self._load_graveyard()

        # Count existing children for numbering (scan both live and graveyard)
        for mid in list(self._members) + list(self._graveyard):
            if mid.startswith(f"{parent_id}-child-"):
                try:
                    num = int(mid.rsplit("-", 1)[-1])
                    self._child_counter = max(self._child_counter, num)
                except ValueError:
                    pass

        # Ensure the parent always exists in the registry
        # v3.26 fix: Check graveyard too — don't create a fresh parent
        # if the dead copy is in the graveyard (boot() handles revival).
        if parent_id not in self._members and parent_id not in self._graveyard:
            parent_genome = Genome()
            self._members[parent_id] = {
                "id": parent_id,
                "genome": parent_genome.to_dict(),
                "evolution_count": 0,
                "interaction_count": 0,
                "state": "idle",
                "created_at": _utc_now(),
                "parent_id": None,
                # v3.0 fields
                "generation_id": 0,
                "genome_hash": _genome_hash(parent_genome.traits),
                "energy": 1.0,
                "fitness_history": [],
                "mutation_rate_offset": 0.0,
                "alive": True,
                "death_info": None,
                "consecutive_above_repro": 0,
                "last_birth_tick": -BIRTH_COOLDOWN_CYCLES,
                "nickname": None,
                # v3.26 lifecycle
                "lifecycle_state": LifecycleState.ACTIVE,
                "below_fitness_cycles": 0,
            }

        self._migrate_members()

    def _migrate_members(self) -> None:
        """Add v3.0+ fields to any legacy member records.

        v3.26: Adds lifecycle_state, below_fitness_cycles.
        Moves dead organisms from _members to _graveyard.
        """
        changed = False
        dead_ids: list = []
        for mid, m in self._members.items():
            if "generation_id" not in m:
                m["generation_id"] = 0 if m.get("parent_id") is None else 1
                changed = True
            if "genome_hash" not in m:
                traits = m.get("genome", {}).get("traits", {})
                m["genome_hash"] = _genome_hash(traits) if traits else ""
                changed = True
            if "energy" not in m:
                m["energy"] = 1.0
                changed = True
            if "fitness_history" not in m:
                m["fitness_history"] = []
                changed = True
            if "mutation_rate_offset" not in m:
                m["mutation_rate_offset"] = 0.0
                changed = True
            if "alive" not in m:
                m["alive"] = True
                changed = True
            if "death_info" not in m:
                m["death_info"] = None
                changed = True
            if "consecutive_above_repro" not in m:
                m["consecutive_above_repro"] = 0
                changed = True
            if "last_birth_tick" not in m:
                m["last_birth_tick"] = -BIRTH_COOLDOWN_CYCLES
                changed = True
            if "nickname" not in m:
                m["nickname"] = None
                changed = True
            # v3.26: lifecycle_state + below_fitness_cycles
            if "lifecycle_state" not in m:
                if not m.get("alive", True):
                    m["lifecycle_state"] = LifecycleState.DEAD
                    dead_ids.append(mid)
                elif m.get("state") == "dormant":
                    m["lifecycle_state"] = LifecycleState.DORMANT
                else:
                    m["lifecycle_state"] = LifecycleState.ACTIVE
                changed = True
            elif m.get("lifecycle_state") == LifecycleState.DEAD:
                # Fix: organisms already marked dead but still in _members
                # (e.g., from crash before graveyard save) must be moved.
                dead_ids.append(mid)
            if "below_fitness_cycles" not in m:
                m["below_fitness_cycles"] = 0
                changed = True
        # v3.26: Move dead organisms to graveyard
        for mid in dead_ids:
            self._graveyard[mid] = self._members.pop(mid)
        if dead_ids:
            self._save_graveyard()
            logger.info("[POPULATION] Migrated %d dead organisms to graveyard", len(dead_ids))
        if changed:
            self._save()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def max_population(self) -> int:
        """Hard population cap."""
        return self._max_population

    @max_population.setter
    def max_population(self, value: int) -> None:
        self._max_population = max(2, min(value, ABSOLUTE_POPULATION_CAP))

    @property
    def min_population_floor(self) -> int:
        """v3.15: Minimum population floor = 20% of max_population.

        Population will not drop below this unless
        ``hardcore_extinction_mode`` is enabled.
        """
        return max(1, int(self._max_population * 0.2))

    @property
    def hardcore_extinction_mode(self) -> bool:
        """v3.14: When True, population may drop to zero (no floor protection)."""
        return self._hardcore_extinction_mode

    @hardcore_extinction_mode.setter
    def hardcore_extinction_mode(self, value: bool) -> None:
        self._hardcore_extinction_mode = bool(value)
        logger.info("[POPULATION] hardcore_extinction_mode set to %s", self._hardcore_extinction_mode)

    @property
    def size(self) -> int:
        """Number of active + sleeping members (excludes dormant)."""
        with self._lock:
            return sum(1 for m in self._members.values()
                       if LifecycleState.is_simulated(
                           m.get("lifecycle_state", LifecycleState.ACTIVE)))

    @property
    def living_or_dormant_count(self) -> int:
        """Number of living + dormant members (everything in the live registry)."""
        with self._lock:
            return len(self._members)

    @property
    def dormant_count(self) -> int:
        """Number of dormant members."""
        with self._lock:
            return sum(1 for m in self._members.values()
                       if m.get("lifecycle_state") == LifecycleState.DORMANT)

    @property
    def sleeping_count(self) -> int:
        """Number of sleeping members."""
        with self._lock:
            return sum(1 for m in self._members.values()
                       if m.get("lifecycle_state") == LifecycleState.SLEEPING)

    @property
    def total_size(self) -> int:
        """Total members including dead (graveyard)."""
        with self._lock:
            return len(self._members) + len(self._graveyard)

    @property
    def member_ids(self) -> List[str]:
        """IDs of active + sleeping members (excludes dormant)."""
        with self._lock:
            return [mid for mid, m in self._members.items()
                    if LifecycleState.is_simulated(
                        m.get("lifecycle_state", LifecycleState.ACTIVE))]

    @property
    def all_member_ids(self) -> List[str]:
        """IDs of all members including graveyard."""
        with self._lock:
            return list(self._members.keys()) + list(self._graveyard.keys())

    def get(self, organism_id: str) -> Optional[Dict[str, Any]]:
        """Look up by ID. Checks live registry first, then graveyard."""
        with self._lock:
            m = self._members.get(organism_id) or self._graveyard.get(organism_id)
            return dict(m) if m else None

    def get_live(self, organism_id: str) -> Optional[Dict[str, Any]]:
        """Look up by ID in live registry only (excludes graveyard)."""
        with self._lock:
            m = self._members.get(organism_id)
            return dict(m) if m else None

    def get_all(self) -> List[Dict[str, Any]]:
        """Get all active + sleeping members (excludes dormant and dead)."""
        with self._lock:
            return [dict(m) for m in self._members.values()
                    if LifecycleState.is_simulated(
                        m.get("lifecycle_state", LifecycleState.ACTIVE))]

    def get_all_including_dead(self) -> List[Dict[str, Any]]:
        """Get all members including graveyard."""
        with self._lock:
            result = [dict(m) for m in self._members.values()]
            result.extend(dict(m) for m in self._graveyard.values())
            return result

    def get_dormant(self) -> List[Dict[str, Any]]:
        """Get all dormant members."""
        with self._lock:
            return [dict(m) for m in self._members.values()
                    if m.get("lifecycle_state") == LifecycleState.DORMANT]

    def get_sleeping(self) -> List[Dict[str, Any]]:
        """v3.26: Get all sleeping members."""
        with self._lock:
            return [dict(m) for m in self._members.values()
                    if m.get("lifecycle_state") == LifecycleState.SLEEPING]

    def update_member(self, organism_id: str, updates: Dict[str, Any]) -> None:
        """Update fields for an existing member and persist.

        v3.28: Rejects updates for dead organisms (lifecycle_state == DEAD).

        Adopted children have immutable lineage fields (parent_id,
        original_id, original_parent_id, generation_id) which are
        silently stripped from *updates*.
        """
        with self._lock:
            if organism_id in self._members:
                member = self._members[organism_id]
                # v3.28: Guard — reject updates for dead organisms
                if member.get("lifecycle_state") == LifecycleState.DEAD:
                    logger.error(
                        "[ERROR] Dead organism %s attempted to update — rejected",
                        organism_id,
                    )
                    return
                # Protect immutable lineage fields on adopted children
                if member.get("adopted", False):
                    from al01.portable import IMMUTABLE_LINEAGE_FIELDS
                    updates = {
                        k: v for k, v in updates.items()
                        if k not in IMMUTABLE_LINEAGE_FIELDS
                    }
                member.update(updates)
                # Recalculate genome hash if genome was updated
                if "genome" in updates:
                    traits = updates["genome"].get("traits", {})
                    self._members[organism_id]["genome_hash"] = _genome_hash(traits)
                self._save()

    def set_nickname(self, organism_id: str, nickname: Optional[str]) -> bool:
        """Set or clear a nickname for an organism. Returns True if successful."""
        with self._lock:
            m = self._members.get(organism_id)
            if not m:
                return False
            m["nickname"] = nickname
            self._save()
        logger.info("[POPULATION] Nickname set: %s → %s", organism_id, nickname)
        return True

    def get_nickname(self, organism_id: str) -> Optional[str]:
        """Get the nickname for an organism."""
        with self._lock:
            m = self._members.get(organism_id)
            if not m:
                return None
            return m.get("nickname")

    def record_fitness(self, organism_id: str, fitness: float, max_history: int = 100) -> None:
        """Record a fitness data point for an organism's independent history."""
        with self._lock:
            m = self._members.get(organism_id)
            if not m or m.get("lifecycle_state") == LifecycleState.DEAD:
                return
            history = m.get("fitness_history", [])
            history.append(round(fitness, 6))
            if len(history) > max_history:
                history = history[-max_history:]
            m["fitness_history"] = history
            self._save()

    def update_energy(self, organism_id: str, energy: float) -> None:
        """Update energy for a member.

        v3.28: Rejects updates for dead organisms.
        """
        with self._lock:
            if organism_id in self._members:
                member = self._members[organism_id]
                if member.get("lifecycle_state") == LifecycleState.DEAD:
                    logger.error(
                        "[ERROR] Dead organism %s attempted energy update — rejected",
                        organism_id,
                    )
                    return
                member["energy"] = round(energy, 6)
                self._save()

    def update_consecutive_repro(
        self,
        organism_id: str,
        fitness: float,
        threshold: float = 0.5,
    ) -> int:
        """Track consecutive cycles above reproduction threshold.

        Returns the updated count.
        """
        with self._lock:
            m = self._members.get(organism_id)
            if not m or m.get("lifecycle_state") == LifecycleState.DEAD:
                return 0
            if fitness >= threshold:
                m["consecutive_above_repro"] = m.get("consecutive_above_repro", 0) + 1
            else:
                m["consecutive_above_repro"] = 0
            self._save()
            return m["consecutive_above_repro"]

    # ------------------------------------------------------------------
    # Death / Removal
    # ------------------------------------------------------------------

    def remove_member(self, organism_id: str, cause: str = "unknown",
                      death_cycle: int = 0) -> Optional[Dict[str, Any]]:
        """Kill an organism and move it to the graveyard.

        v3.14: Refuses to kill if it would drop population below
        ``min_population_floor`` (unless ``hardcore_extinction_mode`` is on).
        v3.26: Dead organisms are moved from _members to _graveyard.
        Death is terminal — dead organisms never auto-recover.

        Returns the death info, or None if not found / blocked by floor.
        """
        with self._lock:
            m = self._members.get(organism_id)
            if not m:
                # Already dead (in graveyard)?
                g = self._graveyard.get(organism_id)
                if g:
                    return g.get("death_info")
                return None
            if m.get("lifecycle_state") == LifecycleState.DEAD:
                return m.get("death_info")

            # v3.14: Population floor guard
            if not self._hardcore_extinction_mode:
                living_count = len(self._members)
                floor = self.min_population_floor
                if living_count >= floor and (living_count - 1) < floor:
                    logger.warning(
                        "[POPULATION] Floor guard: refusing death of %s (living=%d, floor=%d)",
                        organism_id, living_count, floor,
                    )
                    return None

            death_info = {
                "cause": cause,
                "death_time": _utc_now(),
                "death_cycle": death_cycle,
                "final_fitness": m.get("genome", {}).get("fitness", 0.0),
                "final_energy": m.get("energy", 0.0),
                "generation_id": m.get("generation_id", 0),
            }
            m["alive"] = False
            m["lifecycle_state"] = LifecycleState.DEAD
            m["death_info"] = death_info
            m["state"] = "dead"
            # v3.26: Move to graveyard
            self._graveyard[organism_id] = self._members.pop(organism_id)
            self._prune_graveyard()
            self._save()
            self._save_graveyard()

        logger.info(
            "[POPULATION] Death: %s cause=%s fitness=%.4f → graveyard",
            organism_id, cause, death_info["final_fitness"],
        )
        return death_info

    def enter_dormant(self, organism_id: str, cause: str = "unknown") -> Optional[Dict[str, Any]]:
        """v3.15: Put an organism into dormant state.

        Dormant organisms:
        - ``lifecycle_state = "dormant"``
        - Are still ``alive=True``
        - Do NOT participate in reproduction or evolution decisions
        - Can be woken only when specific recovery conditions are met

        v3.26: Uses lifecycle_state instead of state field.
        """
        with self._lock:
            m = self._members.get(organism_id)
            if not m:
                return None
            if m.get("lifecycle_state") == LifecycleState.DEAD:
                return None
            if m.get("lifecycle_state") == LifecycleState.DORMANT:
                return m.get("dormant_info")

            dormant_info = {
                "cause": cause,
                "dormant_since": _utc_now(),
                "fitness_at_dormancy": m.get("genome", {}).get("fitness", 0.0),
                "energy_at_dormancy": m.get("energy", 0.0),
                "generation_id": m.get("generation_id", 0),
            }
            m["lifecycle_state"] = LifecycleState.DORMANT
            m["state"] = "dormant"
            m["dormant_info"] = dormant_info
            m["consecutive_above_repro"] = 0
            self._save()

        logger.info(
            "[POPULATION] Dormant: %s cause=%s fitness=%.4f",
            organism_id, cause, dormant_info["fitness_at_dormancy"],
        )
        return dormant_info

    def enter_sleeping(self, organism_id: str, cause: str = "low_energy") -> Optional[Dict[str, Any]]:
        """v3.26: Put an organism into sleeping state (low-activity conservation).

        Sleeping organisms:
        - ``lifecycle_state = "sleeping"``
        - Are still ``alive=True``
        - Consume reduced metabolic cost
        - Auto-recover when energy exceeds threshold
        """
        with self._lock:
            m = self._members.get(organism_id)
            if not m:
                return None
            ls = m.get("lifecycle_state", LifecycleState.ACTIVE)
            if ls in (LifecycleState.DEAD, LifecycleState.DORMANT):
                return None
            if ls == LifecycleState.SLEEPING:
                return m.get("sleeping_info")

            sleeping_info = {
                "cause": cause,
                "sleeping_since": _utc_now(),
                "energy_at_sleep": m.get("energy", 0.0),
            }
            m["lifecycle_state"] = LifecycleState.SLEEPING
            m["state"] = "sleeping"
            m["sleeping_info"] = sleeping_info
            m["consecutive_above_repro"] = 0
            self._save()

        logger.info(
            "[POPULATION] Sleeping: %s cause=%s energy=%.4f",
            organism_id, cause, sleeping_info["energy_at_sleep"],
        )
        return sleeping_info

    def wake_dormant(self, organism_id: str, energy_boost: float = 0.3) -> Optional[Dict[str, Any]]:
        """v3.15: Wake a dormant organism back to active state.

        v3.26: Sets lifecycle_state to ACTIVE.
        """
        with self._lock:
            m = self._members.get(organism_id)
            if not m:
                return None
            if m.get("lifecycle_state") != LifecycleState.DORMANT:
                return None

            old_energy = m.get("energy", 0.0)
            new_energy = min(1.0, old_energy + energy_boost)
            m["lifecycle_state"] = LifecycleState.ACTIVE
            m["state"] = "idle"
            m["energy"] = round(new_energy, 6)
            dormant_info = m.pop("dormant_info", {})
            wake_record = {
                "organism_id": organism_id,
                "woke_at": _utc_now(),
                "dormant_since": dormant_info.get("dormant_since"),
                "energy_before": round(old_energy, 6),
                "energy_after": round(new_energy, 6),
            }
            self._save()

        logger.info(
            "[POPULATION] Woke dormant: %s energy=%.4f→%.4f",
            organism_id, old_energy, new_energy,
        )
        return wake_record

    def wake_sleeping(self, organism_id: str) -> Optional[Dict[str, Any]]:
        """v3.26: Wake a sleeping organism back to active state."""
        with self._lock:
            m = self._members.get(organism_id)
            if not m:
                return None
            if m.get("lifecycle_state") != LifecycleState.SLEEPING:
                return None

            m["lifecycle_state"] = LifecycleState.ACTIVE
            m["state"] = "idle"
            sleeping_info = m.pop("sleeping_info", {})
            wake_record = {
                "organism_id": organism_id,
                "woke_at": _utc_now(),
                "sleeping_since": sleeping_info.get("sleeping_since"),
            }
            self._save()

        logger.info("[POPULATION] Woke sleeping: %s", organism_id)
        return wake_record

    @property
    def dormant_ids(self) -> List[str]:
        """IDs of dormant members."""
        with self._lock:
            return [mid for mid, m in self._members.items()
                    if m.get("lifecycle_state") == LifecycleState.DORMANT]

    @property
    def sleeping_ids(self) -> List[str]:
        """IDs of sleeping members."""
        with self._lock:
            return [mid for mid, m in self._members.items()
                    if m.get("lifecycle_state") == LifecycleState.SLEEPING]

    def prune_weakest(self, max_size: int, min_keep: int = 2) -> List[Dict[str, Any]]:
        """Kill the weakest organisms to bring population down to *max_size*.

        Never kills below *min_keep* living organisms.
        v3.14: Also respects ``min_population_floor`` unless hardcore mode.
        Returns list of death records.
        """
        deaths: List[Dict[str, Any]] = []
        with self._lock:
            living = [
                (mid, m) for mid, m in self._members.items()
                if LifecycleState.is_alive(m.get("lifecycle_state", LifecycleState.ACTIVE))
            ]
            if len(living) <= max_size or len(living) <= min_keep:
                return deaths

            # Sort by fitness — lowest first
            living.sort(key=lambda x: x[1].get("genome", {}).get("fitness", 0.0))

            to_kill = len(living) - max_size
            # Respect minimum
            to_kill = min(to_kill, len(living) - min_keep)

            # v3.14: Respect population floor
            # Only enforced when population is at or above the floor.
            if not self._hardcore_extinction_mode:
                floor = self.min_population_floor
                if len(living) >= floor:
                    to_kill = min(to_kill, len(living) - floor)

            for i in range(to_kill):
                mid = living[i][0]
                # Don't kill outside the lock — mark inline
                m = self._members[mid]
                death_info = {
                    "cause": "population_pruning",
                    "death_time": _utc_now(),
                    "final_fitness": m.get("genome", {}).get("fitness", 0.0),
                    "final_energy": m.get("energy", 0.0),
                    "generation_id": m.get("generation_id", 0),
                }
                m["alive"] = False
                m["lifecycle_state"] = LifecycleState.DEAD
                m["death_info"] = death_info
                m["state"] = "dead"
                # v3.26: Move to graveyard
                self._graveyard[mid] = self._members.pop(mid)
                deaths.append({"organism_id": mid, **death_info})
                logger.info("[POPULATION] Pruned %s (fitness=%.4f)", mid, death_info["final_fitness"])

            if deaths:
                self._prune_graveyard()
                self._save()
                self._save_graveyard()
        return deaths

    # ------------------------------------------------------------------
    # Reproduction
    # ------------------------------------------------------------------

    def set_tick(self, tick: int) -> None:
        """Advance the population tick counter and clear per-tick state."""
        if tick != self._current_tick:
            self._current_tick = tick
            self._processed_repro_events.clear()

    def deduct_energy(self, organism_id: str, cost: float) -> None:
        """v3.18: Deduct *cost* energy from an organism (clamped to 0)."""
        with self._lock:
            m = self._members.get(organism_id)
            if m:
                m["energy"] = max(0.0, m.get("energy", 0.0) - cost)

    def spawn_child(self, parent_genome: Genome, parent_evolution: int,
                    parent_id: Optional[str] = None,
                    mutation_variance: float = 0.05) -> Optional[Dict[str, Any]]:
        """Create a new child organism from a parent genome.

        v3.0: child gets generation_id = parent+1, genome_hash,
        independent fitness_history, stochastic birth mutation.
        v3.4: Returns None if population cap reached.
        v3.17: Birth-lock, event idempotency, alive re-check.
        v3.19: mutation_variance parameter (default 0.05).
        """
        # Enforce hard cap
        if self.size >= self._max_population:
            logger.info("[POPULATION] Cap reached (%d/%d) — spawn refused",
                        self.size, self._max_population)
            return None
        # v3.13: Absolute ceiling — never exceed ABSOLUTE_POPULATION_CAP
        if self.size >= ABSOLUTE_POPULATION_CAP:
            logger.warning(
                "[POPULATION] ABSOLUTE CAP reached (%d/%d) — spawn blocked",
                self.size, ABSOLUTE_POPULATION_CAP,
            )
            return None
        with self._lock:
            # v3.17 guards only apply when parent_id is explicitly provided
            # (not for setup/manual spawns where parent_id=None)
            if parent_id:
                # Safety re-check — parent must be alive and active at spawn time
                parent_rec = self._members.get(parent_id)
                if not parent_rec or not LifecycleState.can_reproduce(
                        parent_rec.get("lifecycle_state", LifecycleState.ACTIVE)):
                    logger.info("[POPULATION] Parent %s not active (state=%s) — spawn blocked",
                                parent_id, parent_rec.get("lifecycle_state") if parent_rec else "missing")
                    return None

                # Birth cooldown — 500 cycles between births per parent
                last_bt = parent_rec.get("last_birth_tick", -BIRTH_COOLDOWN_CYCLES) if parent_rec else -BIRTH_COOLDOWN_CYCLES
                if self._current_tick - last_bt < BIRTH_COOLDOWN_CYCLES:
                    logger.info("[POPULATION] Birth-cooldown: %s last spawned at tick %d (current %d, need %d gap)",
                                parent_id, last_bt, self._current_tick, BIRTH_COOLDOWN_CYCLES)
                    return None

                # Event idempotency — same tick:parent never processed twice
                event_id = f"{self._current_tick}:{parent_id}"
                if event_id in self._processed_repro_events:
                    logger.info("[POPULATION] Idempotency: event %s already processed", event_id)
                    return None
                self._processed_repro_events.add(event_id)
            self._child_counter += 1
            child_id = f"{self._parent_id}-child-{self._child_counter}"

            effective_parent = parent_id or self._parent_id
            parent_gen = 0
            parent_record = self._members.get(effective_parent)
            if parent_record:
                parent_gen = parent_record.get("generation_id", 0)

            child_genome = parent_genome.spawn_child(variance=mutation_variance)
            child_traits = child_genome.traits

            child_record = {
                "id": child_id,
                "genome": child_genome.to_dict(),
                "evolution_count": 0,
                "interaction_count": 0,
                "state": "idle",
                "created_at": _utc_now(),
                "parent_id": effective_parent,
                "parent_evolution_at_birth": parent_evolution,
                # v3.0 fields
                "generation_id": parent_gen + 1,
                "genome_hash": _genome_hash(child_traits),
                "energy": 0.8,  # children start with less energy
                "fitness_history": [round(child_genome.fitness, 6)],
                "mutation_rate_offset": self._rng.uniform(-0.02, 0.02),
                "alive": True,
                "death_info": None,
                "consecutive_above_repro": 0,
                "last_birth_tick": -BIRTH_COOLDOWN_CYCLES,
                "nickname": None,
                # v3.26 lifecycle
                "lifecycle_state": LifecycleState.ACTIVE,
                "below_fitness_cycles": 0,
            }
            self._members[child_id] = child_record
            # v3.17: Record birth tick on parent
            if parent_id:
                parent_rec = self._members.get(parent_id)
                if parent_rec:
                    parent_rec["last_birth_tick"] = self._current_tick
            self._save()

        logger.info(
            "[POPULATION] Spawned %s gen=%d (fitness=%.4f) at parent evolution=%d",
            child_id, child_record["generation_id"],
            child_genome.fitness, parent_evolution,
        )
        return dict(child_record)

    def auto_reproduce(
        self,
        organism_id: str,
        fitness_threshold: float = 0.5,
        required_cycles: int = 5,
        energy_min: float = 0.50,
        energy_cost: float = 0.20,
    ) -> Optional[Dict[str, Any]]:
        """Check if organism qualifies for autonomous reproduction.

        Returns child record if reproduction triggered, else None.
        v3.18: Parent must have energy ≥ *energy_min*; costs *energy_cost*.
        """
        with self._lock:
            m = self._members.get(organism_id)
            if not m or not LifecycleState.can_reproduce(
                    m.get("lifecycle_state", LifecycleState.ACTIVE)):
                return None
            consecutive = m.get("consecutive_above_repro", 0)
            if consecutive < required_cycles:
                return None
            genome_data = m.get("genome", {})
            fitness = genome_data.get("fitness", 0.0)
            if fitness < fitness_threshold:
                return None
            # v3.18: Energy gate
            parent_energy = m.get("energy", 0.0)
            if parent_energy < energy_min:
                return None
            evo = m.get("evolution_count", 0)

        # Reset counter only after successful spawn
        parent_genome = Genome.from_dict(genome_data)
        child = self.spawn_child(parent_genome, evo, parent_id=organism_id)
        if child:
            with self._lock:
                self._members[organism_id]["consecutive_above_repro"] = 0
                # v3.18: Deduct reproduction energy cost from parent
                old_e = self._members[organism_id].get("energy", 0.0)
                self._members[organism_id]["energy"] = max(0.0, old_e - energy_cost)
                self._save()
        return child

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def top_fitness_members(self, n: int = 10, include_dormant: bool = True) -> List[Dict[str, Any]]:
        """Return top-N living members by fitness for reproduction seeding.

        v3.26: Dead organisms are excluded — only living organisms
        eligible for reproduction are considered.  Dormant organisms
        may be included (default) since they are alive and recoverable.
        """
        with self._lock:
            candidates = []
            for mid, m in self._members.items():
                ls = m.get("lifecycle_state", LifecycleState.ACTIVE)
                if ls == LifecycleState.DORMANT and not include_dormant:
                    continue
                candidates.append(dict(m))
        # Sort by fitness descending
        candidates.sort(key=lambda x: x.get("genome", {}).get("fitness", 0.0), reverse=True)
        return candidates[:n]

    def trait_variance(self) -> Dict[str, float]:
        """Per-trait variance across all living members."""
        with self._lock:
            living = [m for m in self._members.values()
                      if LifecycleState.is_alive(m.get("lifecycle_state", LifecycleState.ACTIVE))]
        if len(living) < 2:
            return {}
        all_trait_names: set = set()
        for m in living:
            traits = m.get("genome", {}).get("traits", {})
            all_trait_names.update(traits.keys())
        variances: Dict[str, float] = {}
        for trait in all_trait_names:
            vals = [m.get("genome", {}).get("traits", {}).get(trait, 0.0) for m in living]
            variances[trait] = round(statistics.variance(vals), 8) if len(vals) >= 2 else 0.0
        return variances

    def population_traits(self) -> Dict[str, Dict[str, float]]:
        """Get trait dicts for all living organisms (for behavior analysis)."""
        with self._lock:
            return {
                mid: m.get("genome", {}).get("traits", {})
                for mid, m in self._members.items()
                if LifecycleState.is_alive(m.get("lifecycle_state", LifecycleState.ACTIVE))
            }

    def population_fitness(self) -> Dict[str, float]:
        """Get fitness for all living organisms."""
        with self._lock:
            return {
                mid: m.get("genome", {}).get("fitness", 0.0)
                for mid, m in self._members.items()
                if LifecycleState.is_alive(m.get("lifecycle_state", LifecycleState.ACTIVE))
            }

    def champion(self) -> Optional[Dict[str, Any]]:
        """Return the best-performing *child* organism (excludes parent).

        Returns dict with champion_id, champion_fitness, champion_genome_hash,
        or None if no children exist.
        """
        with self._lock:
            best_id: Optional[str] = None
            best_fitness: float = -1.0
            best_hash: str = ""
            for mid, m in self._members.items():
                if mid == self._parent_id:
                    continue  # skip founder
                if not LifecycleState.is_simulated(m.get("lifecycle_state", LifecycleState.ACTIVE)):
                    continue
                fit = m.get("genome", {}).get("fitness", 0.0)
                if fit > best_fitness:
                    best_fitness = fit
                    best_id = mid
                    best_hash = m.get("genome_hash", "")
        if best_id is None:
            return None
        return {
            "champion_id": best_id,
            "champion_fitness": round(best_fitness, 6),
            "champion_genome_hash": best_hash,
        }

    def elite_ids(self, top_fraction: float = 0.10) -> List[str]:
        """Return IDs of elite (top N%) living organisms by fitness.

        v3.9 Elite Protection Protocol — elite organisms are shielded from
        mutation but can participate in crossover (blend_with).

        Args:
            top_fraction: fraction of population that counts as elite (default 10%).

        Returns:
            List of organism IDs.  May be empty if population is too small.
        """
        with self._lock:
            living = [
                (mid, m.get("genome", {}).get("fitness", 0.0))
                for mid, m in self._members.items()
                if LifecycleState.is_alive(m.get("lifecycle_state", LifecycleState.ACTIVE))
            ]
            # Need at least 3 living members for elite distinction to matter
            if len(living) < 3:
                return []
        # Sort by fitness descending
        living.sort(key=lambda x: x[1], reverse=True)
        n_elite = max(1, int(len(living) * top_fraction))
        return [mid for mid, _ in living[:n_elite]]

    def diversity_metrics(self) -> Dict[str, Any]:
        """Population diversity: trait stddev, unique genome hashes, entropy.

        Returns:
            trait_stddev: per-trait standard deviation across living members
            unique_genome_hashes: count of distinct genome hashes
            genome_entropy: Shannon entropy of genome hash distribution (bits)
        """
        with self._lock:
            living = [m for m in self._members.values()
                      if LifecycleState.is_alive(m.get("lifecycle_state", LifecycleState.ACTIVE))]
        if not living:
            return {"trait_stddev": {}, "unique_genome_hashes": 0, "genome_entropy": 0.0}

        # Per-trait standard deviation
        all_trait_names: set = set()
        for m in living:
            all_trait_names.update(m.get("genome", {}).get("traits", {}).keys())
        trait_stddev: Dict[str, float] = {}
        for trait in sorted(all_trait_names):
            vals = [m.get("genome", {}).get("traits", {}).get(trait, 0.0) for m in living]
            if len(vals) >= 2:
                trait_stddev[trait] = round(statistics.stdev(vals), 6)
            else:
                trait_stddev[trait] = 0.0

        # Unique genome hashes
        hashes = [m.get("genome_hash", "") for m in living]
        unique_hashes = len(set(hashes))

        # Shannon entropy of genome hash distribution
        n = len(hashes)
        freq: Dict[str, int] = {}
        for h in hashes:
            freq[h] = freq.get(h, 0) + 1
        entropy = 0.0
        for count in freq.values():
            p = count / n
            if p > 0:
                entropy -= p * math.log2(p)

        return {
            "trait_stddev": trait_stddev,
            "unique_genome_hashes": unique_hashes,
            "genome_entropy": round(entropy, 4),
        }

    @property
    def children_count(self) -> int:
        """Number of living children (excludes founder)."""
        with self._lock:
            return sum(
                1 for mid, m in self._members.items()
                if LifecycleState.is_alive(m.get("lifecycle_state", LifecycleState.ACTIVE)) and mid != self._parent_id
            )

    @property
    def generations_present(self) -> List[int]:
        """Sorted list of distinct generation IDs among living members."""
        with self._lock:
            gens = set()
            for m in self._members.values():
                if LifecycleState.is_alive(m.get("lifecycle_state", LifecycleState.ACTIVE)):
                    gens.add(m.get("generation_id", 0))
        return sorted(gens)

    # ------------------------------------------------------------------
    # v3.11: Ecosystem pressure helpers
    # ------------------------------------------------------------------

    def strategy_dominance_penalty(
        self,
        strategy_dist: Dict[str, int],
        organism_strategy: str,
        threshold: float = 0.80,
    ) -> float:
        """Anti-monoculture: diminishing returns when a strategy exceeds *threshold*.

        Returns a multiplicative penalty in (0, 1].  1.0 = no penalty.
        When the organism's strategy fraction exceeds *threshold*, the
        penalty scales linearly: ``1.0 - (fraction - threshold) / (1.0 - threshold)``.

        This does NOT punish the strategy outright — it merely reduces
        its marginal fitness advantage.
        """
        total = sum(strategy_dist.values())
        if total == 0:
            return 1.0
        count = strategy_dist.get(organism_strategy, 0)
        fraction = count / total
        if fraction <= threshold:
            return 1.0
        # Linear diminishing returns above threshold
        excess = (fraction - threshold) / (1.0 - threshold)
        return max(0.10, 1.0 - excess)  # floor at 0.10 — never zeroes out

    def explorer_novelty_multiplier(
        self,
        strategy_dist: Dict[str, int],
        explorer_threshold: float = 0.05,
        reward_multiplier: float = 1.5,
    ) -> float:
        """Novelty reward: boost explorer fitness when explorers are rare.

        If the fraction of explorers in the population is below
        *explorer_threshold*, return *reward_multiplier* (> 1).
        Otherwise return 1.0.
        """
        total = sum(strategy_dist.values())
        if total == 0:
            return 1.0
        explorer_count = strategy_dist.get("explorer", 0)
        explorer_fraction = explorer_count / total
        if explorer_fraction < explorer_threshold:
            return reward_multiplier
        return 1.0

    # ------------------------------------------------------------------
    # Interactions
    # ------------------------------------------------------------------

    def simulate_interactions(self) -> List[Dict[str, Any]]:
        """If population > 1, randomly transfer energy between living members."""
        with self._lock:
            ids = [mid for mid, m in self._members.items()
                   if LifecycleState.is_simulated(
                       m.get("lifecycle_state", LifecycleState.ACTIVE))]
            if len(ids) < 2:
                return []

            interactions: List[Dict[str, Any]] = []
            a_id, b_id = self._rng.sample(ids, 2)
            a_data = self._members[a_id]
            b_data = self._members[b_id]

            a_genome = Genome.from_dict(a_data.get("genome", {}))
            b_genome = Genome.from_dict(b_data.get("genome", {}))

            result = a_genome.transfer_energy(b_genome, amount=0.02)

            a_data["genome"] = a_genome.to_dict()
            b_data["genome"] = b_genome.to_dict()
            a_data["genome_hash"] = _genome_hash(a_genome.traits)
            b_data["genome_hash"] = _genome_hash(b_genome.traits)
            a_data["interaction_count"] = int(a_data.get("interaction_count", 0)) + 1
            b_data["interaction_count"] = int(b_data.get("interaction_count", 0)) + 1

            record = {
                "type": "energy_transfer",
                "organism_a": a_id,
                "organism_b": b_id,
                "result": result,
                "timestamp": _utc_now(),
            }
            interactions.append(record)
            self._save()

        if interactions:
            logger.info("[POPULATION] Interaction: %s <-> %s energy_transfer", a_id, b_id)
        return interactions

    def cooperative_blend(self, mutation_delta: float = 0.02) -> Optional[Dict[str, Any]]:
        """Blend traits between two random living population members."""
        with self._lock:
            ids = [mid for mid, m in self._members.items()
                   if LifecycleState.is_simulated(
                       m.get("lifecycle_state", LifecycleState.ACTIVE))]
            if len(ids) < 2:
                return None

            a_id, b_id = self._rng.sample(ids, 2)
            a_genome = Genome.from_dict(self._members[a_id].get("genome", {}))
            b_genome = Genome.from_dict(self._members[b_id].get("genome", {}))

            a_snapshot = Genome.from_dict(a_genome.to_dict())
            b_snapshot = Genome.from_dict(b_genome.to_dict())

            a_result = a_genome.blend_with(b_snapshot, noise=mutation_delta)
            b_result = b_genome.blend_with(a_snapshot, noise=mutation_delta)

            self._members[a_id]["genome"] = a_genome.to_dict()
            self._members[b_id]["genome"] = b_genome.to_dict()
            self._members[a_id]["genome_hash"] = _genome_hash(a_genome.traits)
            self._members[b_id]["genome_hash"] = _genome_hash(b_genome.traits)
            self._save()

        logger.info("[POPULATION] Cooperative blend: %s <-> %s", a_id, b_id)
        return {
            "type": "cooperative_blend",
            "organism_a": a_id,
            "organism_b": b_id,
            "a_result": a_result,
            "b_result": b_result,
            "timestamp": _utc_now(),
        }

    def should_reproduce(self, evolution_count: int) -> bool:
        """Check if parent should reproduce (every 10 evolutions)."""
        return evolution_count > 0 and evolution_count % 10 == 0

    # ------------------------------------------------------------------
    # Population Diversity Index (v3.12)
    # ------------------------------------------------------------------

    def population_diversity(self) -> float:
        """Average pairwise genome distance across all living organisms.

        Returns a float ≥ 0.  Higher = more diverse population.
        If fewer than 2 organisms, returns 0.0.
        """
        with self._lock:
            living = [
                m for m in self._members.values()
                if LifecycleState.is_simulated(
                    m.get("lifecycle_state", LifecycleState.ACTIVE))
            ]
        if len(living) < 2:
            return 0.0

        genomes = [Genome.from_dict(m.get("genome", {})) for m in living]
        total_dist = 0.0
        pairs = 0
        for i in range(len(genomes)):
            for j in range(i + 1, len(genomes)):
                total_dist += genomes[i].distance(genomes[j])
                pairs += 1
        return round(total_dist / pairs, 6) if pairs > 0 else 0.0

    # ------------------------------------------------------------------
    # Species Divergence (v3.11)
    # ------------------------------------------------------------------

    def assign_species(self, organism_id: str, species_id: str) -> None:
        """Assign a species label to an organism."""
        with self._lock:
            m = self._members.get(organism_id)
            if m:
                m["species_id"] = species_id
                self._save()

    def get_species(self, organism_id: str) -> Optional[str]:
        """Return the species_id for an organism, or None."""
        with self._lock:
            m = self._members.get(organism_id)
            if m:
                return m.get("species_id")
        return None

    def species_census(self) -> Dict[str, List[str]]:
        """Return {species_id: [organism_ids]} for all living members."""
        with self._lock:
            census: Dict[str, List[str]] = {}
            for mid, m in self._members.items():
                if not LifecycleState.is_alive(m.get("lifecycle_state", LifecycleState.ACTIVE)):
                    continue
                sid = m.get("species_id", "default")
                census.setdefault(sid, []).append(mid)
        return census

    def check_speciation(self, child_id: str, parent_id: str,
                         threshold: float = SPECIES_DIVERGENCE_THRESHOLD) -> Optional[str]:
        """Check if *child* has diverged from *parent* enough for a new species.

        If genome distance > threshold, assigns a new species_id to the child
        and returns it.  Otherwise inherits the parent's species and returns None.
        """
        with self._lock:
            child_m = self._members.get(child_id)
            parent_m = self._members.get(parent_id)
            if not child_m or not parent_m:
                return None

            child_genome = Genome.from_dict(child_m.get("genome", {}))
            parent_genome = Genome.from_dict(parent_m.get("genome", {}))
            dist = child_genome.distance(parent_genome)

            parent_species = parent_m.get("species_id", "species-A")

            if dist > threshold:
                # New species
                existing = set()
                for m in self._members.values():
                    s = m.get("species_id")
                    if s:
                        existing.add(s)
                # Generate species-B, species-C, ... species-Z, species-AA, ...
                idx = len(existing) + 1
                new_id = self._species_label(idx)
                while new_id in existing:
                    idx += 1
                    new_id = self._species_label(idx)
                child_m["species_id"] = new_id
                self._save()
                logger.info(
                    "[SPECIES] Divergence! %s → new species %s (dist=%.4f > %.4f)",
                    child_id, new_id, dist, threshold,
                )
                return new_id
            else:
                child_m["species_id"] = parent_species
                self._save()
                return None

    @staticmethod
    def _species_label(n: int) -> str:
        """Convert 1-based index to species label: 1→A, 2→B, ..., 26→Z, 27→AA."""
        result = []
        while n > 0:
            n -= 1
            result.append(chr(ord('A') + n % 26))
            n //= 26
        return "species-" + "".join(reversed(result))

    # ------------------------------------------------------------------
    # Fossil Record (v3.11)
    # ------------------------------------------------------------------

    def fossil_record(self) -> List[Dict[str, Any]]:
        """Return structured fossil records for all dead organisms.

        Each fossil contains the organism's identity, lineage, lifespan
        data and cause of death — suitable for historical analysis.
        """
        fossils: List[Dict[str, Any]] = []
        with self._lock:
            # v3.26: Dead organisms are in the graveyard, not _members
            for mid, m in self._graveyard.items():
                death_info = m.get("death_info", {})
                fossil = {
                    "organism_id": mid,
                    "generation": m.get("generation_id", 0),
                    "parent_id": m.get("parent_id"),
                    "species_id": m.get("species_id", "default"),
                    "genome_hash": m.get("genome_hash", ""),
                    "cause_of_death": death_info.get("cause", "unknown"),
                    "death_time": death_info.get("death_time"),
                    "final_fitness": death_info.get("final_fitness", 0.0),
                    "final_energy": death_info.get("final_energy", 0.0),
                    "fitness_history_length": len(m.get("fitness_history", [])),
                }
                fossils.append(fossil)
        # Sort by death_time ascending (oldest first)
        fossils.sort(key=lambda f: f.get("death_time") or "")
        return fossils

    def fossil_summary(self) -> Dict[str, Any]:
        """Aggregate statistics across the fossil record."""
        fossils = self.fossil_record()
        if not fossils:
            return {"total_deaths": 0, "causes": {}, "avg_final_fitness": 0.0,
                    "generations_lost": [], "species_extinct": []}

        causes: Dict[str, int] = {}
        gens: set = set()
        species_died: Dict[str, int] = {}
        fitnesses: List[float] = []
        for f in fossils:
            c = f["cause_of_death"]
            causes[c] = causes.get(c, 0) + 1
            gens.add(f["generation"])
            sid = f["species_id"]
            species_died[sid] = species_died.get(sid, 0) + 1
            fitnesses.append(f["final_fitness"])

        # Find species that are fully extinct (all members dead)
        living_species = set()
        with self._lock:
            for m in self._members.values():
                if LifecycleState.is_alive(m.get("lifecycle_state", LifecycleState.ACTIVE)):
                    living_species.add(m.get("species_id", "default"))
        extinct = [s for s in species_died if s not in living_species]

        return {
            "total_deaths": len(fossils),
            "causes": dict(sorted(causes.items(), key=lambda x: -x[1])),
            "avg_final_fitness": round(sum(fitnesses) / len(fitnesses), 4) if fitnesses else 0.0,
            "generations_lost": sorted(gens),
            "species_extinct": sorted(extinct),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if not os.path.exists(self._path):
            return {}
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, dict):
                return raw
        except Exception as exc:
            logger.warning("[POPULATION] Failed to load population.json: %s", exc)
        return {}

    def _save(self) -> None:
        try:
            with open(self._path, "w", encoding="utf-8") as fh:
                json.dump(self._members, fh, indent=2, default=str)
        except Exception as exc:
            logger.error("[POPULATION] Failed to save population.json: %s", exc)

    # v3.26: Graveyard persistence

    def _load_graveyard(self) -> Dict[str, Dict[str, Any]]:
        if not os.path.exists(self._graveyard_path):
            return {}
        try:
            with open(self._graveyard_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, dict):
                return raw
        except Exception as exc:
            logger.warning("[POPULATION] Failed to load graveyard.json: %s", exc)
        return {}

    def _save_graveyard(self) -> None:
        try:
            with open(self._graveyard_path, "w", encoding="utf-8") as fh:
                json.dump(self._graveyard, fh, indent=2, default=str)
        except Exception as exc:
            logger.error("[POPULATION] Failed to save graveyard.json: %s", exc)

    def rescue_from_graveyard(self, organism_id: str) -> bool:
        """v3.28: Rescue AL-01 from graveyard back to active _members.

        Only used for founder protection — children must never be rescued.
        Returns True if the organism was found in graveyard and restored.
        """
        if organism_id != "AL-01":
            logger.error(
                "[ERROR] rescue_from_graveyard called for non-founder %s — rejected",
                organism_id,
            )
            return False
        with self._lock:
            entry = self._graveyard.pop(organism_id, None)
            if entry is None:
                return False
            entry["alive"] = True
            entry["lifecycle_state"] = LifecycleState.ACTIVE
            entry["state"] = "idle"
            entry["energy"] = 0.6
            entry.pop("death_info", None)
            self._members[organism_id] = entry
            self._save()
            self._save_graveyard()
        logger.warning(
            "[POPULATION] Rescued %s from graveyard → active members",
            organism_id,
        )
        return True

    def _prune_graveyard(self) -> None:
        """Cap graveyard at GRAVEYARD_CAP, pruning the oldest entries."""
        if len(self._graveyard) <= GRAVEYARD_CAP:
            return
        sorted_ids = sorted(
            self._graveyard.keys(),
            key=lambda k: self._graveyard[k].get("death_info", {}).get("death_time", ""),
        )
        while len(self._graveyard) > GRAVEYARD_CAP:
            removed_id = sorted_ids.pop(0)
            del self._graveyard[removed_id]
            logger.info("[GRAVEYARD] Pruned oldest entry: %s", removed_id)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
