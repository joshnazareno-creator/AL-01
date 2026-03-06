"""AL-01 VITAL — adaptive policy weights with change logging.

File: data/policy.json
Controls organism behaviour weights. Every change is logged as a life-log event.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger("al01.policy")

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "curiosity_weight": 0.5,
    "risk_weight": 0.3,
    "social_weight": 0.5,
}


class PolicyManager:
    """Manages adaptive learning weights stored in ``data/policy.json``."""

    def __init__(self, data_dir: str = "data") -> None:
        self._data_dir = data_dir
        self._path = os.path.join(data_dir, "policy.json")
        self._lock = threading.RLock()
        os.makedirs(data_dir, exist_ok=True)

        self._weights: Dict[str, float] = self._load()
        logger.info("[POLICY] Loaded weights: %s", self._weights)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def weights(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._weights)

    def get(self, key: str) -> float:
        with self._lock:
            return self._weights.get(key, 0.0)

    def update(self, changes: Dict[str, float], reason: str = "") -> Dict[str, Any]:
        """Apply weight changes and return a change record (to log as event)."""
        with self._lock:
            old = dict(self._weights)
            for key, value in changes.items():
                # Clamp to [0.0, 1.0]
                self._weights[key] = max(0.0, min(1.0, float(value)))
            new = dict(self._weights)
            self._save()

        change_record = {
            "old_weights": old,
            "new_weights": new,
            "changes": {k: v for k, v in changes.items()},
            "reason": reason,
        }
        logger.info("[POLICY] Weights updated: %s reason=%s", changes, reason)
        return change_record

    def nudge(self, key: str, delta: float, reason: str = "") -> Dict[str, Any]:
        """Increment/decrement a single weight by *delta*."""
        with self._lock:
            current = self._weights.get(key, 0.5)
        return self.update({key: current + delta}, reason=reason)

    def reset(self) -> Dict[str, Any]:
        """Reset all weights to defaults."""
        return self.update(dict(_DEFAULT_WEIGHTS), reason="reset to defaults")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> Dict[str, float]:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                if isinstance(raw, dict):
                    merged = dict(_DEFAULT_WEIGHTS)
                    merged.update({k: float(v) for k, v in raw.items()})
                    return merged
            except Exception as exc:
                logger.warning("[POLICY] Failed to load policy.json: %s", exc)
        return dict(_DEFAULT_WEIGHTS)

    def _save(self) -> None:
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(self._weights, fh, indent=2)
