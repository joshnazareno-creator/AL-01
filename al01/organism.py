"""AL-01 organism: deterministic metabolism scheduler with configurable cadence."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MetabolismConfig:
    """Configurable cadence parameters for the metabolism scheduler.

    All intervals are expressed in ticks (calls to :py:meth:`MetabolismScheduler.tick`).

    Attributes:
        pulse_interval:     Ticks between each pulse call.
        reflect_interval:   Ticks between each reflect call.
        persist_interval:   Ticks between each forced state+memory persist.
        pulse_log_interval: A pulse observation is only appended to memory
                            every *pulse_log_interval* pulses (bloat protection).
    """

    pulse_interval: int = 1
    reflect_interval: int = 10
    persist_interval: int = 5
    pulse_log_interval: int = 5


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

class JsonStore:
    """JSON-backed key/value store.

    Defaults are created on first load when the backing file does not yet exist.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._data: Dict[str, Any] = self._load()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def save(self) -> None:
        dir_part = os.path.dirname(self._path)
        if dir_part:
            os.makedirs(dir_part, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self._path):
            with open(self._path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {}


class MemoryStore:
    """Append-only event log with tag indexing, backed by a single JSON file.

    File layout::

        {
          "log": [ <observation>, ... ],
          "tag_index": { "<tag>": [<log_index>, ...], ... }
        }

    Defaults (empty log + empty index) are created when the file does not exist.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        raw = self._load_raw()
        self._log: List[Dict[str, Any]] = raw.get("log", [])
        self._tag_index: Dict[str, List[int]] = raw.get("tag_index", {})

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def append(self, observation: Dict[str, Any]) -> None:
        """Append *observation* and update tag index."""
        idx = len(self._log)
        self._log.append(observation)
        for tag in observation.get("tags", []):
            self._tag_index.setdefault(tag, []).append(idx)

    def tail(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return the last *n* observations."""
        return self._log[-n:]

    def by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Return all observations carrying *tag*."""
        return [self._log[i] for i in self._tag_index.get(tag, [])]

    def save(self) -> None:
        dir_part = os.path.dirname(self._path)
        if dir_part:
            os.makedirs(dir_part, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump({"log": self._log, "tag_index": self._tag_index}, fh, indent=2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_raw(self) -> Dict[str, Any]:
        if os.path.exists(self._path):
            with open(self._path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {"log": [], "tag_index": {}}


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class MetabolismScheduler:
    """Deterministic metabolism scheduler.

    No threads are used.  The caller drives execution by invoking
    :py:meth:`tick` at whatever wall-clock cadence it desires.  All
    intervals are expressed in ticks.
    """

    def __init__(self, organism: "Organism", config: MetabolismConfig) -> None:
        self._organism = organism
        self._config = config
        self._tick_count: int = 0
        self._pulse_count: int = 0

    def tick(self) -> None:
        """Advance the scheduler by one tick."""
        self._tick_count += 1

        if self._tick_count % self._config.pulse_interval == 0:
            self._pulse_count += 1
            log_this_pulse = (self._pulse_count % self._config.pulse_log_interval == 0)
            self._organism.pulse(log_observation=log_this_pulse)

        if self._tick_count % self._config.reflect_interval == 0:
            self._organism.reflect()

        if self._tick_count % self._config.persist_interval == 0:
            self._organism.persist()


# ---------------------------------------------------------------------------
# Organism
# ---------------------------------------------------------------------------

class Organism:
    """AL-01 digital organism with deterministic metabolism.

    Parameters:
        data_dir: Directory used to store identity.json, state.json, and
                  memory.json.  Defaults to the current working directory.
        config:   :class:`MetabolismConfig` instance.  A default instance is
                  used when *None* is supplied.
    """

    def __init__(
        self,
        data_dir: str = ".",
        config: Optional[MetabolismConfig] = None,
    ) -> None:
        self._data_dir = data_dir
        self.config: MetabolismConfig = config or MetabolismConfig()

        self._identity = JsonStore(os.path.join(data_dir, "identity.json"))
        self._state = JsonStore(os.path.join(data_dir, "state.json"))
        self._memory = MemoryStore(os.path.join(data_dir, "memory.json"))

        self.scheduler = MetabolismScheduler(self, self.config)

        # Ensure required state fields exist with explicit defaults.
        if self._state.get("awareness") is None:
            self._state.set("awareness", 0.0)
        if self._state.get("evolution_count") is None:
            self._state.set("evolution_count", 0)
        if self._state.get("last_boot_utc") is None:
            self._state.set("last_boot_utc", None)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Externally-driven deterministic tick; delegates to the scheduler."""
        self.scheduler.tick()

    def boot(self) -> None:
        """Record startup observation, increment evolution_count, persist."""
        now = _utc_now()
        evolution = self._state.get("evolution_count", 0) + 1
        self._state.set("evolution_count", evolution)
        self._state.set("last_boot_utc", now)
        self._memory.append({
            "ts": now,
            "type": "observation",
            "content": f"Boot #{evolution} at {now}",
            "tags": ["boot"],
        })
        self.persist()

    def pulse(self, log_observation: bool = True) -> None:
        """Increase awareness by a small increment; optionally log observation.

        The *log_observation* flag is controlled by the scheduler according to
        :py:attr:`MetabolismConfig.pulse_log_interval` to prevent memory bloat.
        """
        awareness: float = self._state.get("awareness", 0.0)
        awareness = min(1.0, awareness + 0.01)
        self._state.set("awareness", awareness)

        if log_observation:
            now = _utc_now()
            self._memory.append({
                "ts": now,
                "type": "observation",
                "content": f"Pulse: awareness={awareness:.4f}",
                "tags": ["pulse"],
            })

    def reflect(self) -> None:
        """Synthesize a short reflection from the last 5 observations and store it."""
        recent = self._memory.tail(5)
        if not recent:
            return
        summary = "; ".join(e.get("content", "") for e in recent)
        now = _utc_now()
        self._memory.append({
            "ts": now,
            "type": "observation",
            "content": f"Reflection: [{summary}]",
            "tags": ["reflection"],
        })

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def persist(self) -> None:
        """Flush state and memory to disk."""
        self._state.save()
        self._memory.save()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
