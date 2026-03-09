"""AL-01 v3.0 — Long-Term Evolution Tracker.

Tracks generation IDs, genome hashes (SHA-256), mutation event logs,
fitness trajectories, trait variance over time, and exports to CSV.

Every organism gets a generation_id (parent=0, children increment).
Every genome snapshot is hashed for lineage fingerprinting.
All mutation events, fitness readings, and trait snapshots are logged
to an append-only JSONL file for post-hoc analysis.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os

from al01.storage import rotate_jsonl
import statistics
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("al01.evolution_tracker")


def genome_hash(traits: Dict[str, float]) -> str:
    """SHA-256 hash of trait values, truncated to 16 hex chars.

    Deterministic: same traits → same hash regardless of dict order.
    """
    canonical = json.dumps(
        {k: round(v, 6) for k, v in sorted(traits.items())},
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class EvolutionTracker:
    """Central evolution tracking & analytics for the entire population.

    Stores data in ``evolution_log.jsonl`` (append-only) and provides
    analytical queries + CSV export.
    """

    def __init__(self, data_dir: str = ".") -> None:
        self._data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self._log_path = os.path.join(data_dir, "evolution_log.jsonl")

        # In-memory indexes (rebuilt from log on init)
        self._generation_counter: int = 0
        self._fitness_trajectories: Dict[str, List[Tuple[int, float]]] = {}
        # organism_id → list of (cycle, fitness)
        self._trait_snapshots: Dict[str, List[Tuple[int, Dict[str, float]]]] = {}
        # organism_id → list of (cycle, traits)
        self._mutation_events: List[Dict[str, Any]] = []
        self._lineage: Dict[str, Dict[str, Any]] = {}
        # organism_id → {generation_id, parent_id, genome_hash, birth_cycle}

        self._rebuild_from_log()

    # ── Public API ───────────────────────────────────────────────────

    @property
    def generation_counter(self) -> int:
        return self._generation_counter

    @property
    def log_path(self) -> str:
        return self._log_path

    def register_organism(
        self,
        organism_id: str,
        parent_id: Optional[str],
        traits: Dict[str, float],
        cycle: int = 0,
    ) -> Dict[str, Any]:
        """Register a new organism in the lineage.

        Returns the lineage entry with generation_id and genome_hash.
        """
        if parent_id and parent_id in self._lineage:
            gen = self._lineage[parent_id]["generation_id"] + 1
        else:
            gen = 0
        self._generation_counter = max(self._generation_counter, gen)

        ghash = genome_hash(traits)
        entry = {
            "organism_id": organism_id,
            "generation_id": gen,
            "parent_id": parent_id,
            "genome_hash": ghash,
            "birth_cycle": cycle,
            "birth_timestamp": _utc_now(),
        }
        self._lineage[organism_id] = entry

        self._append_log({
            "event": "register",
            **entry,
            "traits": {k: round(v, 6) for k, v in traits.items()},
        })

        logger.info(
            "[TRACKER] Registered %s gen=%d hash=%s parent=%s",
            organism_id, gen, ghash, parent_id,
        )
        return dict(entry)

    def record_mutation(
        self,
        organism_id: str,
        cycle: int,
        mutations: Dict[str, Any],
        fitness_before: float,
        fitness_after: float,
        traits_after: Dict[str, float],
    ) -> None:
        """Record a mutation event."""
        ghash = genome_hash(traits_after)
        event = {
            "event": "mutation",
            "organism_id": organism_id,
            "cycle": cycle,
            "mutations": mutations,
            "fitness_before": round(fitness_before, 6),
            "fitness_after": round(fitness_after, 6),
            "genome_hash": ghash,
            "traits_after": {k: round(v, 6) for k, v in traits_after.items()},
            "timestamp": _utc_now(),
        }
        self._mutation_events.append(event)
        self._append_log(event)

        # Update lineage hash
        if organism_id in self._lineage:
            self._lineage[organism_id]["genome_hash"] = ghash

    def record_fitness(
        self,
        organism_id: str,
        cycle: int,
        fitness: float,
        traits: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record a fitness data point for trajectory tracking."""
        if organism_id not in self._fitness_trajectories:
            self._fitness_trajectories[organism_id] = []
        self._fitness_trajectories[organism_id].append((cycle, round(fitness, 6)))

        # Keep last 1000 points per organism
        if len(self._fitness_trajectories[organism_id]) > 1000:
            self._fitness_trajectories[organism_id] = \
                self._fitness_trajectories[organism_id][-1000:]

        if traits:
            if organism_id not in self._trait_snapshots:
                self._trait_snapshots[organism_id] = []
            self._trait_snapshots[organism_id].append(
                (cycle, {k: round(v, 6) for k, v in traits.items()})
            )
            if len(self._trait_snapshots[organism_id]) > 500:
                self._trait_snapshots[organism_id] = \
                    self._trait_snapshots[organism_id][-500:]

        self._append_log({
            "event": "fitness",
            "organism_id": organism_id,
            "cycle": cycle,
            "fitness": round(fitness, 6),
            "traits": {k: round(v, 6) for k, v in traits.items()} if traits else None,
            "timestamp": _utc_now(),
        })

    def record_death(self, organism_id: str, cycle: int, cause: str) -> None:
        """Record an organism's death."""
        self._append_log({
            "event": "death",
            "organism_id": organism_id,
            "cycle": cycle,
            "cause": cause,
            "timestamp": _utc_now(),
        })
        logger.info("[TRACKER] Death: %s cause=%s cycle=%d", organism_id, cause, cycle)

    def record_reproduction(
        self,
        parent_id: str,
        child_id: str,
        cycle: int,
        child_traits: Dict[str, float],
    ) -> None:
        """Record a reproduction event."""
        self._append_log({
            "event": "reproduction",
            "parent_id": parent_id,
            "child_id": child_id,
            "cycle": cycle,
            "child_genome_hash": genome_hash(child_traits),
            "child_traits": {k: round(v, 6) for k, v in child_traits.items()},
            "timestamp": _utc_now(),
        })

    # ── Analytics ────────────────────────────────────────────────────

    def get_lineage(self, organism_id: str) -> Optional[Dict[str, Any]]:
        """Get lineage info for an organism."""
        return self._lineage.get(organism_id)

    def get_fitness_trajectory(self, organism_id: str) -> List[Tuple[int, float]]:
        """Get fitness over time for an organism."""
        return list(self._fitness_trajectories.get(organism_id, []))

    def get_all_lineages(self) -> Dict[str, Dict[str, Any]]:
        """Get full lineage map."""
        return {k: dict(v) for k, v in self._lineage.items()}

    # ── Lineage tree ─────────────────────────────────────────────────

    def build_family_tree(self) -> Dict[str, Any]:
        """Build a nested family-tree structure from the flat lineage map.

        Returns a list of root nodes (organisms with no parent or whose
        parent is not tracked).  Each node has::

            {"id": ..., "generation": ..., "genome_hash": ...,
             "birth_cycle": ..., "children": [<subtree>, ...]}
        """
        # Index children by parent_id
        children_of: Dict[str, List[str]] = {}
        for oid, info in self._lineage.items():
            pid = info.get("parent_id")
            if pid:
                children_of.setdefault(pid, []).append(oid)

        def _subtree(oid: str) -> Dict[str, Any]:
            info = self._lineage.get(oid, {})
            node: Dict[str, Any] = {
                "id": oid,
                "generation": info.get("generation_id", 0),
                "genome_hash": info.get("genome_hash", ""),
                "birth_cycle": info.get("birth_cycle", 0),
                "children": [],
            }
            for cid in sorted(children_of.get(oid, [])):
                node["children"].append(_subtree(cid))
            return node

        # Roots: organisms with no parent *or* parent not in lineage
        roots: List[str] = []
        for oid, info in self._lineage.items():
            pid = info.get("parent_id")
            if not pid or pid not in self._lineage:
                roots.append(oid)

        return {"roots": [_subtree(r) for r in sorted(roots)]}

    def render_tree_ascii(self) -> str:
        """Render a human-readable ASCII family tree.

        Example::

            AL-01
             +-- AL-01-child-1
             |    +-- AL-01-child-3
             +-- AL-01-child-2
        """
        tree = self.build_family_tree()
        lines: List[str] = []

        def _render(node: Dict[str, Any], prefix: str = "", is_last: bool = True) -> None:
            connector = "+-- " if prefix else ""
            lines.append(f"{prefix}{connector}{node['id']}")
            children = node.get("children", [])
            for i, child in enumerate(children):
                last = (i == len(children) - 1)
                if prefix:
                    new_prefix = prefix + ("     " if is_last else "|    ")
                else:
                    new_prefix = " "
                _render(child, new_prefix, last)

        for root in tree.get("roots", []):
            _render(root)
        return "\n".join(lines)

    def get_ancestor_chain(self, organism_id: str) -> List[Dict[str, Any]]:
        """Walk up from *organism_id* to the root, returning the chain."""
        chain: List[Dict[str, Any]] = []
        current = organism_id
        visited: set = set()
        while current and current not in visited:
            visited.add(current)
            info = self._lineage.get(current)
            if not info:
                break
            chain.append(dict(info))
            current = info.get("parent_id")
        return chain

    def get_descendants(self, organism_id: str) -> List[str]:
        """Return all descendant IDs of *organism_id* (breadth-first)."""
        children_of: Dict[str, List[str]] = {}
        for oid, info in self._lineage.items():
            pid = info.get("parent_id")
            if pid:
                children_of.setdefault(pid, []).append(oid)

        result: List[str] = []
        queue = list(children_of.get(organism_id, []))
        visited: set = set()
        while queue:
            oid = queue.pop(0)
            if oid in visited:
                continue
            visited.add(oid)
            result.append(oid)
            queue.extend(children_of.get(oid, []))
        return result

    def trait_variance_across_population(
        self,
        population_traits: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """Compute per-trait variance across all living organisms.

        *population_traits* maps organism_id → {trait_name: value}.
        """
        if len(population_traits) < 2:
            return {}

        all_trait_names: set = set()
        for traits in population_traits.values():
            all_trait_names.update(traits.keys())

        variances: Dict[str, float] = {}
        for trait in all_trait_names:
            values = [
                t.get(trait, 0.0) for t in population_traits.values()
            ]
            if len(values) >= 2:
                variances[trait] = round(statistics.variance(values), 8)
            else:
                variances[trait] = 0.0
        return variances

    def population_fitness_stats(
        self,
        population_fitness: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute fitness statistics across the population."""
        values = list(population_fitness.values())
        if not values:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "variance": 0.0, "count": 0}
        return {
            "mean": round(statistics.mean(values), 6),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
            "variance": round(statistics.variance(values), 8) if len(values) >= 2 else 0.0,
            "count": len(values),
        }

    def mutation_event_count(self) -> int:
        return len(self._mutation_events)

    def recent_mutations(self, n: int = 20) -> List[Dict[str, Any]]:
        return self._mutation_events[-n:]

    # ── CSV Export ───────────────────────────────────────────────────

    def export_fitness_csv(self) -> str:
        """Export all fitness trajectories as CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["organism_id", "cycle", "fitness"])
        for oid, trajectory in sorted(self._fitness_trajectories.items()):
            for cycle, fitness in trajectory:
                writer.writerow([oid, cycle, fitness])
        return output.getvalue()

    def export_mutations_csv(self) -> str:
        """Export all mutation events as CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "organism_id", "cycle", "fitness_before", "fitness_after",
            "genome_hash", "timestamp",
        ])
        for event in self._mutation_events:
            writer.writerow([
                event.get("organism_id", ""),
                event.get("cycle", 0),
                event.get("fitness_before", 0),
                event.get("fitness_after", 0),
                event.get("genome_hash", ""),
                event.get("timestamp", ""),
            ])
        return output.getvalue()

    def export_lineage_csv(self) -> str:
        """Export lineage data as CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "organism_id", "generation_id", "parent_id",
            "genome_hash", "birth_cycle", "birth_timestamp",
        ])
        for oid, info in sorted(self._lineage.items()):
            writer.writerow([
                oid,
                info.get("generation_id", 0),
                info.get("parent_id", ""),
                info.get("genome_hash", ""),
                info.get("birth_cycle", 0),
                info.get("birth_timestamp", ""),
            ])
        return output.getvalue()

    # ── Persistence ──────────────────────────────────────────────────

    def _append_log(self, record: Dict[str, Any]) -> None:
        try:
            rotate_jsonl(self._log_path)
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        except Exception as exc:
            logger.error("[TRACKER] Log write failed: %s", exc)

    def _rebuild_from_log(self) -> None:
        """Rebuild in-memory indexes from the JSONL log file."""
        if not os.path.exists(self._log_path):
            return
        try:
            with open(self._log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._index_entry(entry)
            logger.info(
                "[TRACKER] Rebuilt %d lineages, %d mutation events from log",
                len(self._lineage),
                len(self._mutation_events),
            )
        except Exception as exc:
            logger.error("[TRACKER] Rebuild failed: %s", exc)

    def _index_entry(self, entry: Dict[str, Any]) -> None:
        """Index a single log entry into in-memory structures."""
        event_type = entry.get("event")
        if event_type == "register":
            oid = entry.get("organism_id", "")
            self._lineage[oid] = {
                "organism_id": oid,
                "generation_id": entry.get("generation_id", 0),
                "parent_id": entry.get("parent_id"),
                "genome_hash": entry.get("genome_hash", ""),
                "birth_cycle": entry.get("birth_cycle", 0),
                "birth_timestamp": entry.get("birth_timestamp", ""),
            }
            self._generation_counter = max(
                self._generation_counter,
                entry.get("generation_id", 0),
            )
        elif event_type == "mutation":
            self._mutation_events.append(entry)
            oid = entry.get("organism_id", "")
            if oid in self._lineage:
                self._lineage[oid]["genome_hash"] = entry.get("genome_hash", "")
        elif event_type == "fitness":
            oid = entry.get("organism_id", "")
            cycle = entry.get("cycle", 0)
            fitness = entry.get("fitness", 0.0)
            if oid not in self._fitness_trajectories:
                self._fitness_trajectories[oid] = []
            self._fitness_trajectories[oid].append((cycle, fitness))
            traits = entry.get("traits")
            if traits:
                if oid not in self._trait_snapshots:
                    self._trait_snapshots[oid] = []
                self._trait_snapshots[oid].append((cycle, traits))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
