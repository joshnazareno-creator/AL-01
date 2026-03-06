"""AL-01 GPT Bridge — translates organism state into natural language and
accepts controlled stimulus injection from a GPT agent.

Design principles:
  1. **Read-only state access** — reads /status, /autonomy, /population,
     /genome via the Organism object directly (no HTTP round-trip).
  2. **Natural-language narration** — converts numeric state into prose
     that a GPT system prompt can consume.
  3. **Controlled stimulus injection** — GPT can send stimuli that are
     rate-limited and queued for the *next* evolution cycle (never forces
     an immediate cycle, so the evolution loop timing stays untouched).
  4. **No interference** — the bridge never calls ``evolve_cycle()``,
     ``autonomy_cycle()``, or any mutation method.  It only reads state
     and queues stimuli through the existing ``stimulate()`` path.

Typical flow:
  GPT → POST /gpt/stimulus  (bridge queues it)
  GPT → GET  /gpt/narrate   (bridge reads state, returns prose)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("al01.gpt_bridge")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GPTBridgeConfig:
    """Tuneable knobs for the bridge layer."""
    # Maximum stimuli a GPT agent can inject per minute
    stimulus_rate_limit: int = 6
    # Window size in seconds for the rate limiter
    rate_window_seconds: float = 60.0
    # Maximum length (chars) of a single stimulus string
    max_stimulus_length: int = 280
    # Whether stimulus injection is enabled at all
    stimulus_enabled: bool = True
    # Maximum number of population members to include in narration
    narration_population_limit: int = 10


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class GPTBridge:
    """Stateless-ish bridge between a GPT agent and the AL-01 organism.

    Holds a reference to the live ``Organism`` instance and provides
    two main capabilities:

    * **narrate()** — snapshot of organism state as structured prose.
    * **inject_stimulus(text)** — rate-limited stimulus queue.
    """

    def __init__(
        self,
        organism: Any,  # al01.organism.Organism (avoid circular import)
        config: Optional[GPTBridgeConfig] = None,
    ) -> None:
        self._organism = organism
        self._config = config or GPTBridgeConfig()
        # Ring buffer of injection timestamps for rate limiting
        self._injection_timestamps: List[float] = []
        # Audit log of recent injections (kept in memory, bounded)
        self._injection_log: List[Dict[str, Any]] = []
        self._max_log_entries: int = 200
        # Stats
        self._total_injections: int = 0
        self._total_rejections: int = 0

    # ------------------------------------------------------------------
    # Public: narration
    # ------------------------------------------------------------------

    def narrate(self) -> Dict[str, Any]:
        """Build a natural-language narration of the current organism state.

        Returns a dict with ``prose`` (human-readable text), ``raw``
        (structured data the prose was derived from), and ``timestamp``.
        """
        raw = self._collect_state()
        prose = self._state_to_prose(raw)
        return {
            "prose": prose,
            "raw": raw,
            "timestamp": _utc_now(),
        }

    # ------------------------------------------------------------------
    # Public: stimulus injection
    # ------------------------------------------------------------------

    def inject_stimulus(self, text: str) -> Dict[str, Any]:
        """Queue a stimulus from GPT into AL-01's evolution pipeline.

        The stimulus is handed to ``organism.stimulate()`` which queues it
        for the *next* scheduled evolution cycle — the bridge never forces
        an immediate cycle, preserving loop timing.

        Returns a status dict (accepted / rejected + reason).
        """
        now = time.monotonic()

        # Gate: injection enabled?
        if not self._config.stimulus_enabled:
            self._total_rejections += 1
            return self._rejection("stimulus_disabled", "Stimulus injection is currently disabled.")

        # Gate: text length
        text = text.strip()
        if not text:
            self._total_rejections += 1
            return self._rejection("empty_stimulus", "Stimulus text must not be empty.")
        if len(text) > self._config.max_stimulus_length:
            self._total_rejections += 1
            return self._rejection(
                "too_long",
                f"Stimulus exceeds {self._config.max_stimulus_length} chars (got {len(text)}).",
            )

        # Gate: rate limit
        if not self._check_rate_limit(now):
            self._total_rejections += 1
            return self._rejection(
                "rate_limited",
                f"Rate limit: max {self._config.stimulus_rate_limit} stimuli "
                f"per {self._config.rate_window_seconds}s window.",
            )

        # --- Accepted ---
        # Queue the stimulus via the organism's existing pathway.
        # stimulate() bumps interaction_count and queues for next evolve_cycle.
        # We do NOT pass trigger_cycle — the API default trigger_cycle is for
        # /stimulate callers; here we call the lower-level method which only
        # queues.
        self._organism.stimulate(stimulus=text)

        self._injection_timestamps.append(now)
        self._total_injections += 1
        entry = {
            "text": text,
            "accepted": True,
            "timestamp": _utc_now(),
            "injection_number": self._total_injections,
        }
        self._append_log(entry)
        logger.info("[GPT-BRIDGE] Stimulus accepted (#%d): %.60s", self._total_injections, text)

        return {
            "status": "accepted",
            "injection_number": self._total_injections,
            "queued_stimuli": len(self._organism.stimuli),
            "timestamp": _utc_now(),
        }

    # ------------------------------------------------------------------
    # Public: configuration & stats
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Return bridge status and statistics."""
        now = time.monotonic()
        window = self._config.rate_window_seconds
        recent = sum(1 for t in self._injection_timestamps if now - t < window)
        return {
            "stimulus_enabled": self._config.stimulus_enabled,
            "rate_limit": self._config.stimulus_rate_limit,
            "rate_window_seconds": self._config.rate_window_seconds,
            "injections_in_window": recent,
            "total_injections": self._total_injections,
            "total_rejections": self._total_rejections,
            "max_stimulus_length": self._config.max_stimulus_length,
            "narration_population_limit": self._config.narration_population_limit,
            "log_entries": len(self._injection_log),
            "timestamp": _utc_now(),
        }

    def set_stimulus_enabled(self, enabled: bool) -> Dict[str, Any]:
        """Toggle stimulus injection on/off."""
        old = self._config.stimulus_enabled
        self._config.stimulus_enabled = enabled
        logger.info("[GPT-BRIDGE] stimulus_enabled: %s → %s", old, enabled)
        return {"stimulus_enabled": enabled, "previous": old}

    def recent_injections(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent injection log entries."""
        limit = max(1, min(limit, self._max_log_entries))
        return list(self._injection_log[-limit:])

    # ------------------------------------------------------------------
    # Internal: state collection
    # ------------------------------------------------------------------

    def _collect_state(self) -> Dict[str, Any]:
        """Gather a snapshot of organism state from live objects."""
        org = self._organism
        state = dict(org.state)

        # Core vitals
        evolution_count = state.get("evolution_count", 0)
        awareness = round(float(state.get("awareness", 0.0)), 4)
        energy = round(float(state.get("energy", 1.0)), 4)
        state_version = state.get("state_version", 0)

        # Genome
        genome_dict = org.genome.to_dict()
        fitness = round(org.genome.fitness, 4)

        # Autonomy summary
        autonomy = org.autonomy.summary()

        # Population snapshot (bounded)
        pop_members = org.population.get_all()
        living = [m for m in pop_members if m.get("alive", True)]
        living_sorted = sorted(
            living,
            key=lambda m: m.get("genome", {}).get("fitness", 0),
            reverse=True,
        )
        top_n = living_sorted[: self._config.narration_population_limit]

        # Pending stimuli count
        pending_stimuli = len(state.get("stimuli", []))

        return {
            "organism_state": org.organism_state,
            "evolution_count": evolution_count,
            "awareness": awareness,
            "energy": energy,
            "fitness": fitness,
            "state_version": state_version,
            "genome": genome_dict,
            "autonomy": {
                "decision_count": autonomy.get("total_decisions", 0),
                "stagnation_count": autonomy.get("stagnation_count", 0),
                "is_stagnant": autonomy.get("is_stagnant", False),
                "exploration_mode": autonomy.get("exploration_mode", False),
                "vital_score": autonomy.get("vital_score", 0),
                "idle_cycles": autonomy.get("idle_cycles", 0),
            },
            "population": {
                "living": len(living),
                "total": len(pop_members),
                "top_members": [
                    {
                        "id": m.get("id", "?"),
                        "nickname": m.get("nickname"),
                        "fitness": round(
                            m.get("genome", {}).get("fitness", 0), 4
                        ),
                        "energy": round(m.get("energy", 0), 4),
                        "evolution_count": m.get("evolution_count", 0),
                        "generation": m.get("generation_id", 0),
                        "alive": m.get("alive", True),
                    }
                    for m in top_n
                ],
            },
            "pending_stimuli": pending_stimuli,
            "cycle_stats": org.cycle_stats.to_dict(),
            "loop_running": org.loop_running,
        }

    # ------------------------------------------------------------------
    # Internal: prose generation
    # ------------------------------------------------------------------

    def _state_to_prose(self, raw: Dict[str, Any]) -> str:
        """Convert structured state into natural-language paragraphs."""
        lines: List[str] = []

        # --- Identity & vital signs ---
        state_label = raw["organism_state"]
        energy = raw["energy"]
        fitness = raw["fitness"]
        awareness = raw["awareness"]
        evo = raw["evolution_count"]
        sv = raw["state_version"]

        energy_word = self._energy_word(energy)
        fitness_word = self._fitness_word(fitness)
        awareness_word = self._awareness_word(awareness)

        lines.append(
            f"AL-01 is currently {state_label}. "
            f"It has completed {evo:,} evolution cycles (state version {sv:,}). "
            f"Energy is {energy_word} ({energy:.2%}), "
            f"fitness is {fitness_word} ({fitness:.4f}), "
            f"and awareness is {awareness_word} ({awareness:.4f})."
        )

        # --- Autonomy ---
        auto = raw["autonomy"]
        if auto["is_stagnant"]:
            lines.append(
                f"The autonomy engine has detected stagnation "
                f"({auto['stagnation_count']} stagnant cycles). "
                f"It may benefit from external stimulation."
            )
        elif auto["exploration_mode"]:
            lines.append(
                "The organism is in exploration mode — mutation rates are boosted "
                "and it is actively searching for novel trait combinations."
            )
        else:
            lines.append(
                f"The autonomy engine has made {auto['decision_count']:,} decisions "
                f"with no current stagnation."
            )

        # --- Genome traits ---
        traits = raw["genome"].get("effective_traits", raw["genome"].get("traits", {}))
        if traits:
            trait_parts = [f"{k}: {v:.3f}" for k, v in sorted(traits.items())]
            lines.append(f"Genome traits — {', '.join(trait_parts)}.")

        # --- Population ---
        pop = raw["population"]
        lines.append(
            f"Population: {pop['living']} living organism(s) "
            f"out of {pop['total']} total."
        )

        top = pop.get("top_members", [])
        if top:
            best = top[0]
            name = best.get("nickname") or best["id"]
            lines.append(
                f"Top performer: {name} "
                f"(fitness {best['fitness']:.4f}, "
                f"generation {best['generation']}, "
                f"{best['evolution_count']} evolutions)."
            )

        # --- Pending stimuli ---
        pending = raw["pending_stimuli"]
        if pending > 0:
            lines.append(
                f"There {'is' if pending == 1 else 'are'} "
                f"{pending} pending stimulus event(s) queued for the next evolution cycle."
            )
        else:
            lines.append("No pending stimuli — the organism is evolving autonomously.")

        # --- Cycle instrumentation ---
        cs = raw.get("cycle_stats", {})
        if cs:
            parts = []
            ev = cs.get("energy_volatility", 0)
            if ev > 0:
                parts.append(f"energy volatility {ev:.4f}")
            fh = cs.get("floor_hits_rolling", 0)
            if fh > 0:
                parts.append(f"{fh} energy-floor hits (rolling {cs.get('window_size', 20)})")
            er = cs.get("efficiency_ratio", 0)
            if er != 0:
                parts.append(f"efficiency ratio {er:.4f}")
            streak = cs.get("longest_alive_streak", 0)
            if streak > 0:
                parts.append(f"longest alive streak {streak} cycles")
            if parts:
                lines.append(f"Cycle stats — {', '.join(parts)}.")

        # --- Loop ---
        if not raw["loop_running"]:
            lines.append("⚠ The main evolution loop is NOT running.")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Prose helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _energy_word(energy: float) -> str:
        if energy >= 0.8:
            return "high"
        if energy >= 0.5:
            return "moderate"
        if energy >= 0.25:
            return "low"
        return "critical"

    @staticmethod
    def _fitness_word(fitness: float) -> str:
        if fitness >= 0.8:
            return "excellent"
        if fitness >= 0.6:
            return "good"
        if fitness >= 0.4:
            return "average"
        if fitness >= 0.2:
            return "below average"
        return "poor"

    @staticmethod
    def _awareness_word(awareness: float) -> str:
        if awareness >= 0.8:
            return "highly aware"
        if awareness >= 0.5:
            return "moderately aware"
        if awareness >= 0.25:
            return "somewhat aware"
        return "barely aware"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_rate_limit(self, now: float) -> bool:
        """Return True if an injection is allowed under the rate limit."""
        window = self._config.rate_window_seconds
        # Prune old timestamps outside the window
        self._injection_timestamps = [
            t for t in self._injection_timestamps if now - t < window
        ]
        return len(self._injection_timestamps) < self._config.stimulus_rate_limit

    def _rejection(self, code: str, message: str) -> Dict[str, Any]:
        """Standard rejection response."""
        return {
            "status": "rejected",
            "reason": code,
            "message": message,
            "timestamp": _utc_now(),
        }

    def _append_log(self, entry: Dict[str, Any]) -> None:
        """Append to bounded injection audit log."""
        self._injection_log.append(entry)
        if len(self._injection_log) > self._max_log_entries:
            self._injection_log = self._injection_log[-self._max_log_entries:]


# ---------------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
