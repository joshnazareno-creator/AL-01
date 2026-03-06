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
        self._max_population: int = 50  # hard cap, overridable via Organism

        # Seeded RNG for reproducibility — interactions and spawning use this
        self._rng = random.Random(rng_seed)

        os.makedirs(data_dir, exist_ok=True)
        self._members: Dict[str, Dict[str, Any]] = self._load()

        # Count existing children for numbering
        for mid in self._members:
            if mid.startswith(f"{parent_id}-child-"):
                try:
                    num = int(mid.rsplit("-", 1)[-1])
                    self._child_counter = max(self._child_counter, num)
                except ValueError:
                    pass

        # Ensure the parent always exists in the registry
        if parent_id not in self._members:
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
                "nickname": None,
            }
            self._save()
        else:
            # Migrate existing members to v3.0 format
            self._migrate_members()

    def _migrate_members(self) -> None:
        """Add v3.0 fields to any legacy member records."""
        changed = False
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
            if "nickname" not in m:
                m["nickname"] = None
                changed = True
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
        self._max_population = max(2, value)

    @property
    def size(self) -> int:
        """Number of living members."""
        with self._lock:
            return sum(1 for m in self._members.values() if m.get("alive", True))

    @property
    def total_size(self) -> int:
        """Total members including dead."""
        with self._lock:
            return len(self._members)

    @property
    def member_ids(self) -> List[str]:
        """IDs of living members."""
        with self._lock:
            return [mid for mid, m in self._members.items() if m.get("alive", True)]

    @property
    def all_member_ids(self) -> List[str]:
        """IDs of all members including dead."""
        with self._lock:
            return list(self._members.keys())

    def get(self, organism_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            m = self._members.get(organism_id)
            return dict(m) if m else None

    def get_all(self) -> List[Dict[str, Any]]:
        """Get all living members."""
        with self._lock:
            return [dict(m) for m in self._members.values() if m.get("alive", True)]

    def get_all_including_dead(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(m) for m in self._members.values()]

    def update_member(self, organism_id: str, updates: Dict[str, Any]) -> None:
        """Update fields for an existing member and persist.

        Adopted children have immutable lineage fields (parent_id,
        original_id, original_parent_id, generation_id) which are
        silently stripped from *updates*.
        """
        with self._lock:
            if organism_id in self._members:
                member = self._members[organism_id]
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
            if not m or not m.get("alive", True):
                return
            history = m.get("fitness_history", [])
            history.append(round(fitness, 6))
            if len(history) > max_history:
                history = history[-max_history:]
            m["fitness_history"] = history
            self._save()

    def update_energy(self, organism_id: str, energy: float) -> None:
        """Update energy for a member."""
        with self._lock:
            if organism_id in self._members:
                self._members[organism_id]["energy"] = round(energy, 6)
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
            if not m or not m.get("alive", True):
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

    def remove_member(self, organism_id: str, cause: str = "unknown") -> Optional[Dict[str, Any]]:
        """Mark an organism as dead.  Does NOT delete from registry.

        Returns the death info, or None if not found.
        """
        with self._lock:
            m = self._members.get(organism_id)
            if not m:
                return None
            if not m.get("alive", True):
                return m.get("death_info")

            death_info = {
                "cause": cause,
                "death_time": _utc_now(),
                "final_fitness": m.get("genome", {}).get("fitness", 0.0),
                "final_energy": m.get("energy", 0.0),
                "generation_id": m.get("generation_id", 0),
            }
            m["alive"] = False
            m["death_info"] = death_info
            m["state"] = "dead"
            self._save()

        logger.info(
            "[POPULATION] Death: %s cause=%s fitness=%.4f",
            organism_id, cause, death_info["final_fitness"],
        )
        return death_info

    def prune_weakest(self, max_size: int, min_keep: int = 2) -> List[Dict[str, Any]]:
        """Kill the weakest organisms to bring population down to *max_size*.

        Never kills below *min_keep* living organisms.
        Returns list of death records.
        """
        deaths: List[Dict[str, Any]] = []
        with self._lock:
            living = [
                (mid, m) for mid, m in self._members.items()
                if m.get("alive", True)
            ]
            if len(living) <= max_size:
                return deaths

            # Sort by fitness — lowest first
            living.sort(key=lambda x: x[1].get("genome", {}).get("fitness", 0.0))

            to_kill = len(living) - max_size
            # Respect minimum
            to_kill = min(to_kill, len(living) - min_keep)

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
                m["death_info"] = death_info
                m["state"] = "dead"
                deaths.append({"organism_id": mid, **death_info})
                logger.info("[POPULATION] Pruned %s (fitness=%.4f)", mid, death_info["final_fitness"])

            if deaths:
                self._save()
        return deaths

    # ------------------------------------------------------------------
    # Reproduction
    # ------------------------------------------------------------------

    def spawn_child(self, parent_genome: Genome, parent_evolution: int,
                    parent_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Create a new child organism from a parent genome.

        v3.0: child gets generation_id = parent+1, genome_hash,
        independent fitness_history, stochastic birth mutation.
        v3.4: Returns None if population cap reached.
        """
        # Enforce hard cap
        if self.size >= self._max_population:
            logger.info("[POPULATION] Cap reached (%d/%d) — spawn refused",
                        self.size, self._max_population)
            return None
        with self._lock:
            self._child_counter += 1
            child_id = f"{self._parent_id}-child-{self._child_counter}"

            effective_parent = parent_id or self._parent_id
            parent_gen = 0
            parent_record = self._members.get(effective_parent)
            if parent_record:
                parent_gen = parent_record.get("generation_id", 0)

            child_genome = parent_genome.spawn_child(variance=0.05)
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
                "nickname": None,
            }
            self._members[child_id] = child_record
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
    ) -> Optional[Dict[str, Any]]:
        """Check if organism qualifies for autonomous reproduction.

        Returns child record if reproduction triggered, else None.
        """
        with self._lock:
            m = self._members.get(organism_id)
            if not m or not m.get("alive", True):
                return None
            consecutive = m.get("consecutive_above_repro", 0)
            if consecutive < required_cycles:
                return None
            genome_data = m.get("genome", {})
            fitness = genome_data.get("fitness", 0.0)
            if fitness < fitness_threshold:
                return None
            evo = m.get("evolution_count", 0)

        # Reset counter only after successful spawn
        parent_genome = Genome.from_dict(genome_data)
        child = self.spawn_child(parent_genome, evo, parent_id=organism_id)
        if child:
            with self._lock:
                self._members[organism_id]["consecutive_above_repro"] = 0
                self._save()
        return child

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def trait_variance(self) -> Dict[str, float]:
        """Per-trait variance across all living members."""
        with self._lock:
            living = [m for m in self._members.values() if m.get("alive", True)]
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
                if m.get("alive", True)
            }

    def population_fitness(self) -> Dict[str, float]:
        """Get fitness for all living organisms."""
        with self._lock:
            return {
                mid: m.get("genome", {}).get("fitness", 0.0)
                for mid, m in self._members.items()
                if m.get("alive", True)
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
                if not m.get("alive", True):
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
                if m.get("alive", True)
            ]
        if len(living) < 3:
            # Need at least 3 living members for elite distinction to matter
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
            living = [m for m in self._members.values() if m.get("alive", True)]
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
                if m.get("alive", True) and mid != self._parent_id
            )

    @property
    def generations_present(self) -> List[int]:
        """Sorted list of distinct generation IDs among living members."""
        with self._lock:
            gens = set()
            for m in self._members.values():
                if m.get("alive", True):
                    gens.add(m.get("generation_id", 0))
        return sorted(gens)

    # ------------------------------------------------------------------
    # Interactions
    # ------------------------------------------------------------------

    def simulate_interactions(self) -> List[Dict[str, Any]]:
        """If population > 1, randomly transfer energy between living members."""
        with self._lock:
            ids = [mid for mid, m in self._members.items() if m.get("alive", True)]
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
            ids = [mid for mid, m in self._members.items() if m.get("alive", True)]
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
