"""AL-01 v3.10 — Portable Child Export / Import.

Serialize a child organism to a self-contained JSON snapshot that can be
downloaded and later *adopted* into another AL-01 instance.

Integrity is guaranteed by an HMAC-SHA256 checksum computed on a
**canonical** JSON string (``sort_keys=True``, ``checksum`` field
excluded).  If any field is tampered with, import will reject the file.

A formal JSON Schema (:data:`PORTABLE_CHILD_SCHEMA`) defines the
required fields and types.  The lightweight :func:`validate_payload`
validator enforces the schema before import.

Public API
----------
* ``export_child(population, child_id)`` → dict  (ready for JSON response)
* ``import_child(population, payload, life_log=None)`` → dict  (new member record)
* ``validate_payload(data)`` → list[str]  (schema error strings; empty = valid)
* ``guard_lineage(record, updates)`` → dict  (strip immutable fields)
* ``PORTABLE_SCHEMA_VERSION`` — bumped when the schema changes.
* ``PORTABLE_CHILD_SCHEMA`` — formal JSON Schema (v1.0).
* ``FITNESS_HISTORY_CAP`` — max entries retained in fitness_history.
* ``IMMUTABLE_LINEAGE_FIELDS`` — fields frozen after adoption.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from al01.genome import Genome

logger = logging.getLogger("al01.portable")

# ── Constants ─────────────────────────────────────────────────────────

# Schema version — bump when fields change so older exports can be migrated.
PORTABLE_SCHEMA_VERSION: str = "1.0"

# HMAC key — used to sign/verify export payloads.  Changing this will
# invalidate all previously exported snapshots.
_HMAC_KEY: bytes = b"AL-01-portable-child-v1"

# Max fitness-history entries retained on export / import.
FITNESS_HISTORY_CAP: int = 100

# Fields whose values are frozen once an adopted child record is created.
# These MUST NOT be overridden via update_member() or external mutation.
IMMUTABLE_LINEAGE_FIELDS: frozenset = frozenset({
    "parent_id",
    "original_id",
    "original_parent_id",
    "generation_id",
})

# ── Formal JSON Schema (v1.0) ────────────────────────────────────────

PORTABLE_CHILD_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "AL-01 Portable Child Export v1.0",
    "description": (
        "Self-contained snapshot of a child organism exported from an "
        "AL-01 population.  Integrity is guaranteed by an HMAC-SHA256 "
        "checksum computed on the canonical (sort_keys=True) JSON of "
        "all fields except 'checksum' itself."
    ),
    "type": "object",
    "required": [
        "schema_version",
        "id",
        "parent_id",
        "generation",
        "genome",
        "fitness",
        "trait_fitness",
        "checksum",
    ],
    "properties": {
        "schema_version": {
            "type": "string",
            "const": "1.0",
            "description": "Export schema version for migration support.",
        },
        "exported_at": {
            "type": "string",
            "format": "date-time",
            "description": "ISO-8601 UTC timestamp of export.",
        },
        "id": {
            "type": "string",
            "minLength": 1,
            "description": "Original runtime ID in the source population.",
        },
        "parent_id": {
            "type": "string",
            "minLength": 1,
            "description": "Parent organism ID (immutable after import).",
        },
        "generation": {
            "type": "integer",
            "minimum": 1,
            "description": "Lineage generation number.",
        },
        "genome": {
            "type": "object",
            "required": ["traits", "mutation_rate", "mutation_delta"],
            "properties": {
                "traits": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                    "description": "Raw trait values.",
                },
                "effective_traits": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                    "description": "Trait values after soft-ceiling.",
                },
                "mutation_rate": {
                    "type": "number",
                    "minimum": 0,
                },
                "mutation_delta": {
                    "type": "number",
                    "minimum": 0,
                },
                "fitness_components": {
                    "description": "Multi-objective fitness breakdown, or null.",
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "survival": {"type": "number"},
                                "efficiency": {"type": "number"},
                                "stability": {"type": "number"},
                                "adaptation": {"type": "number"},
                            },
                        },
                        {"type": "null"},
                    ],
                },
            },
            "additionalProperties": False,
        },
        "fitness": {
            "type": "number",
            "description": "Primary fitness score at time of export.",
        },
        "trait_fitness": {
            "type": "number",
            "description": "Trait-average fitness (soft-capped).",
        },
        "birth_time": {
            "oneOf": [{"type": "string", "format": "date-time"}, {"type": "null"}],
            "description": "ISO-8601 UTC birth timestamp.",
        },
        "age_seconds": {
            "type": "number",
            "minimum": 0,
            "description": "Computed age in seconds at export time.",
        },
        "energy": {
            "type": "number",
            "description": "Current energy level.",
        },
        "evolution_count": {
            "type": "integer",
            "minimum": 0,
        },
        "interaction_count": {
            "type": "integer",
            "minimum": 0,
        },
        "state": {
            "type": "string",
            "description": "Organism lifecycle state.",
        },
        "genome_hash": {
            "type": "string",
        },
        "fitness_history": {
            "type": "array",
            "maxItems": FITNESS_HISTORY_CAP,
            "description": "Timestamped fitness entries (capped at 100).",
            "items": {
                "oneOf": [
                    {"type": "number"},
                    {
                        "type": "object",
                        "required": ["value"],
                        "properties": {
                            "value": {"type": "number"},
                            "recorded_at": {
                                "type": "string",
                                "format": "date-time",
                            },
                        },
                        "additionalProperties": False,
                    },
                ],
            },
        },
        "nickname": {
            "oneOf": [{"type": "string"}, {"type": "null"}],
        },
        "checksum": {
            "type": "string",
            "minLength": 64,
            "maxLength": 64,
            "description": "HMAC-SHA256 hex digest of the canonical payload.",
        },
    },
    "additionalProperties": False,
}


# ── Schema Validation ─────────────────────────────────────────────────

def validate_payload(data: Dict[str, Any]) -> List[str]:
    """Validate an import payload against the v1.0 portable child schema.

    Returns a list of human-readable error strings.  An empty list means
    the payload is structurally valid.  This is a lightweight validator —
    it does *not* implement full JSON-Schema resolution.
    """
    errors: List[str] = []

    # ── Required top-level fields ─────────────────────────────────────
    for field in PORTABLE_CHILD_SCHEMA["required"]:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    if errors:
        return errors  # can't inspect further

    # ── Type / format checks ──────────────────────────────────────────
    if not isinstance(data.get("schema_version"), str):
        errors.append("schema_version must be a string")
    elif data["schema_version"] != PORTABLE_SCHEMA_VERSION:
        errors.append(
            f"Unsupported schema_version '{data['schema_version']}' "
            f"(expected '{PORTABLE_SCHEMA_VERSION}')"
        )

    if not isinstance(data.get("id"), str) or not data["id"]:
        errors.append("id must be a non-empty string")

    if not isinstance(data.get("parent_id"), str) or not data["parent_id"]:
        errors.append("parent_id must be a non-empty string")

    gen = data.get("generation")
    if not isinstance(gen, int) or gen < 1:
        errors.append("generation must be a positive integer")

    if not isinstance(data.get("fitness"), (int, float)):
        errors.append("fitness must be a number")

    if not isinstance(data.get("trait_fitness"), (int, float)):
        errors.append("trait_fitness must be a number")

    cs = data.get("checksum", "")
    if not isinstance(cs, str) or len(cs) != 64:
        errors.append("checksum must be a 64-character hex string")

    # ── Genome sub-object ─────────────────────────────────────────────
    genome = data.get("genome")
    if not isinstance(genome, dict):
        errors.append("genome must be an object")
    else:
        if "traits" not in genome:
            errors.append("genome.traits is required")
        elif not isinstance(genome["traits"], dict):
            errors.append("genome.traits must be an object")
        else:
            for k, v in genome["traits"].items():
                if not isinstance(v, (int, float)):
                    errors.append(f"genome.traits.{k} must be a number")

        for num_field in ("mutation_rate", "mutation_delta"):
            if num_field not in genome:
                errors.append(f"genome.{num_field} is required")
            elif not isinstance(genome[num_field], (int, float)):
                errors.append(f"genome.{num_field} must be a number")

    # ── fitness_history ───────────────────────────────────────────────
    fh = data.get("fitness_history")
    if fh is not None:
        if not isinstance(fh, list):
            errors.append("fitness_history must be an array")
        elif len(fh) > FITNESS_HISTORY_CAP:
            errors.append(
                f"fitness_history has {len(fh)} entries "
                f"(max {FITNESS_HISTORY_CAP})"
            )
        else:
            for i, entry in enumerate(fh):
                if isinstance(entry, (int, float)):
                    continue
                if isinstance(entry, dict) and "value" in entry:
                    if not isinstance(entry["value"], (int, float)):
                        errors.append(
                            f"fitness_history[{i}].value must be a number"
                        )
                    continue
                errors.append(
                    f"fitness_history[{i}] must be a number or "
                    f'{{"value": <number>, "recorded_at": <string>}}'
                )

    return errors


# ------------------------------------------------------------------
# Checksum Helpers
# ------------------------------------------------------------------

def _canonical_payload(data: Dict[str, Any]) -> str:
    """Deterministic JSON string used as the HMAC message.

    The *checksum* key is excluded before serialization; all remaining
    keys are sorted (``sort_keys=True``) so that dict insertion order
    never affects the digest.  ``default=str`` handles datetime objects.
    """
    filtered = {k: v for k, v in data.items() if k != "checksum"}
    return json.dumps(filtered, sort_keys=True, default=str)


def _compute_checksum(data: Dict[str, Any]) -> str:
    """HMAC-SHA256 hex digest of the canonical payload."""
    msg = _canonical_payload(data).encode("utf-8")
    return hmac.new(_HMAC_KEY, msg, hashlib.sha256).hexdigest()


def _verify_checksum(data: Dict[str, Any]) -> bool:
    """Return True if the checksum in *data* matches a fresh computation."""
    expected = data.get("checksum", "")
    actual = _compute_checksum(data)
    return hmac.compare_digest(expected, actual)


# ------------------------------------------------------------------
# Fitness-History Helpers
# ------------------------------------------------------------------

def _stamp_fitness_history(
    raw: list,
    export_time: str,
    cap: int = FITNESS_HISTORY_CAP,
) -> list:
    """Convert a flat-float fitness history into timestamped entries.

    Each bare float ``v`` becomes ``{"value": v, "recorded_at": export_time}``.
    Already-timestamped entries are kept as-is.  The list is truncated to
    the most recent *cap* entries.
    """
    trimmed = raw[-cap:] if len(raw) > cap else list(raw)
    stamped: list = []
    for entry in trimmed:
        if isinstance(entry, (int, float)):
            stamped.append({
                "value": round(float(entry), 6),
                "recorded_at": export_time,
            })
        elif isinstance(entry, dict) and "value" in entry:
            stamped.append(entry)
        else:
            logger.warning(
                "[PORTABLE] Skipping unrecognised fitness_history entry: %s",
                entry,
            )
    return stamped


def _flatten_fitness_history(
    entries: list,
    cap: int = FITNESS_HISTORY_CAP,
) -> list:
    """Normalise fitness-history entries back to a flat list of floats.

    Accepts both bare floats and ``{"value": float, ...}`` objects.
    Truncated to the most recent *cap* entries.
    """
    flat: list = []
    for entry in entries:
        if isinstance(entry, (int, float)):
            flat.append(round(float(entry), 6))
        elif isinstance(entry, dict) and "value" in entry:
            flat.append(round(float(entry["value"]), 6))
    return flat[-cap:]


# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------

def export_child(
    population: Any,
    child_id: str,
) -> Dict[str, Any]:
    """Export a living child organism as a portable JSON snapshot.

    Fitness history is capped to :data:`FITNESS_HISTORY_CAP` entries and
    each entry is stamped with ``{"value": float, "recorded_at": iso}``.

    Parameters
    ----------
    population : Population
        The population registry that owns the child.
    child_id : str
        ID of the child to export (e.g. ``"AL-01-child-3"``).

    Returns
    -------
    dict
        The full portable snapshot including ``checksum``.

    Raises
    ------
    ValueError
        If the child does not exist, is dead, or is the parent organism.
    """
    member = population.get(child_id)
    if member is None:
        raise ValueError(f"Child '{child_id}' not found in population")
    if not member.get("alive", True):
        raise ValueError(f"Child '{child_id}' is dead and cannot be exported")
    if member.get("parent_id") is None:
        raise ValueError(f"'{child_id}' is the founder organism — only children can be exported")

    genome_data = member.get("genome", {})
    genome = Genome.from_dict(genome_data)
    now = datetime.now(timezone.utc).isoformat()

    # Fitness history — cap + timestamp
    raw_history = member.get("fitness_history", [])
    stamped_history = _stamp_fitness_history(raw_history, now)

    payload: Dict[str, Any] = {
        "schema_version": PORTABLE_SCHEMA_VERSION,
        "exported_at": now,
        # Identity
        "id": child_id,
        "parent_id": member.get("parent_id"),
        "generation": member.get("generation_id", 1),
        # Genome
        "genome": {
            "traits": genome_data.get("traits", {}),
            "effective_traits": genome_data.get("effective_traits", {}),
            "mutation_rate": genome_data.get("mutation_rate", 0.10),
            "mutation_delta": genome_data.get("mutation_delta", 0.10),
            "fitness_components": genome_data.get("fitness_components"),
        },
        "fitness": round(genome.fitness, 6),
        "trait_fitness": round(genome.trait_fitness, 6),
        # Lifecycle
        "birth_time": member.get("created_at"),
        "age_seconds": _age_seconds(member.get("created_at")),
        "energy": member.get("energy", 0.8),
        "evolution_count": member.get("evolution_count", 0),
        "interaction_count": member.get("interaction_count", 0),
        "state": member.get("state", "idle"),
        # Lineage
        "genome_hash": member.get("genome_hash", ""),
        "fitness_history": stamped_history,
        "nickname": member.get("nickname"),
    }

    payload["checksum"] = _compute_checksum(payload)
    return payload


# ------------------------------------------------------------------
# Import (Adopt)
# ------------------------------------------------------------------

def import_child(
    population: Any,
    payload: Dict[str, Any],
    life_log: Optional[Any] = None,
) -> Dict[str, Any]:
    """Import (adopt) a previously exported child into the local population.

    Validation order:

    1. HMAC-SHA256 checksum integrity.
    2. Formal schema validation (required fields, types, ranges).
    3. Business-logic checks (population capacity).

    Lineage fields (``parent_id``, ``original_id``, ``original_parent_id``,
    ``generation_id``) are **immutable** after the adopted record is created.

    Parameters
    ----------
    population : Population
        Target population registry.
    payload : dict
        The portable snapshot (as returned by :func:`export_child`).
    life_log : LifeLog, optional
        If provided, the adoption event is appended to the log.

    Returns
    -------
    dict
        The new member record (with a fresh runtime ID).

    Raises
    ------
    ValueError
        If the checksum is invalid, schema validation fails, or capacity
        is exceeded.
    """
    # ── 1. Integrity check ────────────────────────────────────────────
    if not _verify_checksum(payload):
        raise ValueError("Checksum verification failed — file may have been tampered with")

    # ── 2. Schema validation ──────────────────────────────────────────
    errors = validate_payload(payload)
    if errors:
        raise ValueError(f"Schema validation failed: {'; '.join(errors)}")

    genome_section = payload["genome"]

    # ── 3. Reconstruct genome ─────────────────────────────────────────
    genome = Genome(
        traits=genome_section["traits"],
        mutation_rate=genome_section.get("mutation_rate", 0.10),
        mutation_delta=genome_section.get("mutation_delta", 0.10),
    )
    # Restore fitness components if present
    fc = genome_section.get("fitness_components")
    if fc and isinstance(fc, dict):
        genome.set_fitness_components(
            survival=fc.get("survival", 0.0),
            efficiency=fc.get("efficiency", 0.0),
            stability=fc.get("stability", 0.0),
            adaptation=fc.get("adaptation", 0.0),
        )

    # ── 4. Lineage (immutable after creation) ─────────────────────────
    original_id = payload["id"]
    original_parent = payload["parent_id"]
    generation = payload["generation"]

    # ── 5. Capacity check ─────────────────────────────────────────────
    if population.size >= population.max_population:
        raise ValueError(
            f"Population at capacity ({population.size}/{population.max_population}) "
            "— cannot adopt"
        )

    # ── 6. Mint new runtime ID (thread-safe) ──────────────────────────
    import threading
    with population._lock:
        population._child_counter += 1
        new_id = f"{population._parent_id}-child-{population._child_counter}"

    # ── 7. Flatten fitness history for internal storage ────────────────
    flat_history = _flatten_fitness_history(
        payload.get("fitness_history", []),
    )

    # ── 8. Build adopted record ───────────────────────────────────────
    now = datetime.now(timezone.utc).isoformat()

    adopted_record: Dict[str, Any] = {
        "id": new_id,
        "genome": genome.to_dict(),
        "evolution_count": payload.get("evolution_count", 0),
        "interaction_count": payload.get("interaction_count", 0),
        "state": "idle",
        "created_at": now,
        # Immutable lineage fields ──────────────────────────────────
        "parent_id": original_parent,
        "parent_evolution_at_birth": 0,
        "generation_id": generation,
        # ──────────────────────────────────────────────────────────
        "genome_hash": payload.get("genome_hash", ""),
        "energy": payload.get("energy", 0.8),
        "fitness_history": flat_history,
        "mutation_rate_offset": 0.0,
        "alive": True,
        "death_info": None,
        "consecutive_above_repro": 0,
        "nickname": payload.get("nickname"),
        # Adoption metadata
        "adopted": True,
        "adopted_at": now,
        "original_id": original_id,
        "original_parent_id": original_parent,
    }

    # ── 9. Register in population ─────────────────────────────────────
    with population._lock:
        population._members[new_id] = adopted_record
        population._save()

    # ── 10. Log adoption event ────────────────────────────────────────
    if life_log is not None:
        try:
            life_log.append_event(
                event_type="child_adopted",
                payload={
                    "new_id": new_id,
                    "original_id": original_id,
                    "original_parent_id": original_parent,
                    "generation": generation,
                    "fitness": round(genome.fitness, 6),
                    "schema_version": payload.get("schema_version", "unknown"),
                },
            )
        except Exception as exc:
            logger.warning("[PORTABLE] Failed to log adoption event: %s", exc)

    logger.info(
        "[PORTABLE] Adopted %s (was %s, parent=%s, gen=%d, fitness=%.4f)",
        new_id, original_id, original_parent, generation, genome.fitness,
    )
    return adopted_record


# ------------------------------------------------------------------
# Lineage Guard
# ------------------------------------------------------------------

def guard_lineage(
    record: Dict[str, Any],
    updates: Dict[str, Any],
) -> Dict[str, Any]:
    """Strip immutable lineage fields from *updates* if the record is adopted.

    Call this before applying ``population.update_member()`` on an adopted
    child to ensure lineage metadata cannot be overridden externally.

    Returns
    -------
    dict
        A copy of *updates* with immutable fields removed (if adopted).
    """
    if not record.get("adopted", False):
        return updates
    safe = {k: v for k, v in updates.items() if k not in IMMUTABLE_LINEAGE_FIELDS}
    stripped = set(updates.keys()) - set(safe.keys())
    if stripped:
        logger.warning(
            "[PORTABLE] Blocked mutation of immutable lineage fields: %s",
            stripped,
        )
    return safe


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _age_seconds(created_at: Optional[str]) -> float:
    """Compute age in seconds from an ISO timestamp, or 0.0 if unavailable."""
    if not created_at:
        return 0.0
    try:
        born = datetime.fromisoformat(created_at)
        if born.tzinfo is None:
            born = born.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - born).total_seconds())
    except Exception:
        return 0.0
