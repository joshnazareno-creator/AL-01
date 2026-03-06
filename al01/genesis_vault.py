"""AL-01 Genesis Vault — immutable seed template and extinction recovery.

The Genesis Vault stores a read-only "AL-01 Genesis Seed" — a pristine copy
of the default genome configuration.  When the living population drops to
zero (extinction), the vault automatically spawns a new generation from the
seed and logs a transparent "reseed event" so the system never pretends
it survived an extinction.

Files managed:
  data/genesis_vault.json  – immutable seed template (written once)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from al01.genome import DEFAULT_TRAITS, DEFAULT_MUTATION_RATE, DEFAULT_MUTATION_DELTA, Genome

logger = logging.getLogger("al01.genesis_vault")

# Immutable default seed — the "AL-01 Genesis Seed"
_GENESIS_SEED: Dict[str, Any] = {
    "seed_name": "AL-01 Genesis Seed",
    "traits": dict(DEFAULT_TRAITS),
    "mutation_rate": DEFAULT_MUTATION_RATE,
    "mutation_delta": DEFAULT_MUTATION_DELTA,
    "description": "Pristine genome template — used for extinction recovery.",
}


class GenesisVault:
    """Read-only vault holding the genesis seed for population recovery.

    The vault file (``genesis_vault.json``) is written once on first
    initialisation and never modified after that.  When the population
    reaches zero, ``check_and_reseed()`` spawns a new organism from the
    frozen seed and returns a reseed record.
    """

    def __init__(self, data_dir: str = "data") -> None:
        self._data_dir = data_dir
        self._vault_path = os.path.join(data_dir, "genesis_vault.json")
        self._lock = threading.RLock()
        self._reseed_count: int = 0
        self._reseed_history: list[Dict[str, Any]] = []

        os.makedirs(data_dir, exist_ok=True)
        self._seed: Dict[str, Any] = self._load_or_create_seed()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def seed(self) -> Dict[str, Any]:
        """Return a deep copy of the genesis seed (never the original)."""
        with self._lock:
            return deepcopy(self._seed)

    @property
    def reseed_count(self) -> int:
        """How many times the vault has reseeded the population."""
        with self._lock:
            return self._reseed_count

    @property
    def reseed_history(self) -> list[Dict[str, Any]]:
        """Chronological list of all reseed events."""
        with self._lock:
            return list(self._reseed_history)

    @property
    def vault_path(self) -> str:
        return self._vault_path

    # ------------------------------------------------------------------
    # Seed management (immutable after creation)
    # ------------------------------------------------------------------

    def _load_or_create_seed(self) -> Dict[str, Any]:
        """Load vault from disk, or write the default seed once."""
        if os.path.exists(self._vault_path):
            try:
                with open(self._vault_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict) and "traits" in data:
                    # Restore reseed history if present
                    self._reseed_count = data.get("reseed_count", 0)
                    self._reseed_history = data.get("reseed_history", [])
                    logger.info(
                        "[GENESIS VAULT] Loaded seed '%s' (reseeds=%d)",
                        data.get("seed_name", "unknown"), self._reseed_count,
                    )
                    return data
            except Exception as exc:
                logger.warning("[GENESIS VAULT] Failed to load vault: %s", exc)

        # First run — freeze the default seed
        seed = deepcopy(_GENESIS_SEED)
        seed["created_at"] = _utc_now()
        seed["reseed_count"] = 0
        seed["reseed_history"] = []
        self._persist_seed(seed)
        logger.info("[GENESIS VAULT] Created genesis seed '%s'", seed["seed_name"])
        return seed

    def _persist_seed(self, seed: Dict[str, Any]) -> None:
        """Write seed to disk (used only on creation + after reseed events)."""
        try:
            with open(self._vault_path, "w", encoding="utf-8") as fh:
                json.dump(seed, fh, indent=2, default=str)
        except Exception as exc:
            logger.error("[GENESIS VAULT] Failed to persist vault: %s", exc)

    # ------------------------------------------------------------------
    # Genome factory
    # ------------------------------------------------------------------

    def create_genome_from_seed(self) -> Genome:
        """Build a fresh Genome from the frozen seed template."""
        with self._lock:
            return Genome(
                traits=deepcopy(self._seed["traits"]),
                mutation_rate=self._seed.get("mutation_rate", DEFAULT_MUTATION_RATE),
                mutation_delta=self._seed.get("mutation_delta", DEFAULT_MUTATION_DELTA),
            )

    # ------------------------------------------------------------------
    # Reseed logic
    # ------------------------------------------------------------------

    def check_and_reseed(
        self,
        population: Any,
        evolution_tracker: Any = None,
        life_log: Any = None,
        behavior_analyzer: Any = None,
        global_cycle: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """If population is extinct (size == 0), reseed from the vault.

        Returns a reseed record dict if reseeding happened, or None.
        """
        if population.size > 0:
            return None

        with self._lock:
            self._reseed_count += 1
            reseed_num = self._reseed_count

        logger.warning(
            "[GENESIS VAULT] Population extinct — initiating reseed #%d from genesis seed",
            reseed_num,
        )

        # Build a fresh genome from the vault
        genesis_genome = self.create_genome_from_seed()

        # Temporarily raise cap for reseed (extinction recovery must succeed)
        old_cap = population.max_population
        population.max_population = max(old_cap, population.size + 5)

        # Spawn into population
        child_record = population.spawn_child(
            genesis_genome, parent_evolution=0, parent_id=None,
        )

        # Restore cap
        population.max_population = old_cap

        if child_record is None:
            logger.error("[GENESIS VAULT] Reseed spawn failed despite cap override")
            return None
        child_id = child_record["id"]

        # Build reseed event record
        reseed_event: Dict[str, Any] = {
            "event": "reseed",
            "reseed_number": reseed_num,
            "organism_id": child_id,
            "seed_name": self._seed.get("seed_name", "AL-01 Genesis Seed"),
            "cycle": global_cycle,
            "timestamp": _utc_now(),
            "seed_traits": deepcopy(self._seed["traits"]),
            "child_fitness": child_record.get("genome", {}).get("fitness", 0.0),
            "reason": "population_extinct",
        }

        # Register in evolution tracker (new lineage root — no parent)
        if evolution_tracker is not None:
            child_traits = child_record.get("genome", {}).get("traits", {})
            evolution_tracker.register_organism(
                child_id, parent_id=None, traits=child_traits,
                cycle=global_cycle,
            )

        # Log reseed event in life log
        if life_log is not None:
            life_log.append_event(
                event_type="reseed",
                payload=reseed_event,
            )

        # Clean up behavior analyzer state for dead population
        # (new organism will be tracked fresh)

        # Persist reseed history in vault file
        with self._lock:
            self._reseed_history.append(reseed_event)
            self._seed["reseed_count"] = self._reseed_count
            self._seed["reseed_history"] = self._reseed_history
            self._persist_seed(self._seed)

        logger.info(
            "[GENESIS VAULT] Reseed #%d complete: spawned %s (fitness=%.4f)",
            reseed_num, child_id, reseed_event["child_fitness"],
        )

        return reseed_event

    # ------------------------------------------------------------------
    # Status / Serialization
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Return vault status for API / diagnostics."""
        with self._lock:
            return {
                "seed_name": self._seed.get("seed_name", "unknown"),
                "created_at": self._seed.get("created_at"),
                "reseed_count": self._reseed_count,
                "seed_traits": deepcopy(self._seed["traits"]),
                "mutation_rate": self._seed.get("mutation_rate"),
                "mutation_delta": self._seed.get("mutation_delta"),
                "last_reseed": (
                    self._reseed_history[-1] if self._reseed_history else None
                ),
            }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
