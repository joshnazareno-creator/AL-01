"""AL-01 v3.7 — Brain: Environmental Analysis & Strategic Decision Engine.

Replaces the v2.0 mock/LLM placeholder with a real analytical engine that
reads actual environment state, organism traits, fitness, and energy to
produce **strategic trait nudges** and a situation report.

The Brain operates in three layers:

1. **Environmental Analysis** — reads temperature, entropy, abundance,
   noise, scarcity from the Environment model to understand pressures.
2. **Gap Analysis** — compares what the environment demands vs. what the
   organism currently has, identifying the biggest mismatches.
3. **Strategic Nudges** — recommends trait adjustments weighted by urgency
   (energy-critical → prioritise energy_efficiency, high entropy →
   prioritise resilience, etc).

Optional external AI integration (OpenAI-compatible) can enhance the
analysis when an API key is configured, but the core logic is fully local
and deterministic — no placeholder fallback needed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger("al01.brain")


# ── Configuration ────────────────────────────────────────────────────────

@dataclass
class BrainConfig:
    """Tuneable knobs for the analytical brain."""

    nudge_scale: float = 0.04
    """Base magnitude for trait nudges per analysis cycle."""

    urgency_multiplier: float = 2.0
    """Scale nudges when organism is in critical state."""

    demand_sensitivity: float = 1.0
    """Global multiplier for environment → demand conversion."""

    energy_critical: float = 0.20
    energy_low: float = 0.40
    fitness_critical: float = 0.15
    fitness_low: float = 0.35

    scarcity_efficiency_boost: float = 0.3
    """Extra weight on energy_efficiency during scarcity events."""

    trait_floor_threshold: float = 0.05
    trait_floor_boost: float = 0.03
    """If a trait is near zero, boost nudge by this amount."""


# ── Analysis Results ─────────────────────────────────────────────────────

@dataclass
class EnvironmentDemand:
    """What the environment currently demands of each trait (0–1 scale)."""
    adaptability: float = 0.5
    energy_efficiency: float = 0.5
    resilience: float = 0.5
    perception: float = 0.5
    creativity: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return {
            "adaptability": round(self.adaptability, 4),
            "energy_efficiency": round(self.energy_efficiency, 4),
            "resilience": round(self.resilience, 4),
            "perception": round(self.perception, 4),
            "creativity": round(self.creativity, 4),
        }


@dataclass
class AnalysisResult:
    """Full brain analysis output."""
    situation: str
    environment_demand: EnvironmentDemand
    trait_gaps: Dict[str, float]
    trait_nudges: Dict[str, float]
    urgency: str
    priorities: List[str]
    source: str = "brain"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "situation": self.situation,
            "environment_demand": self.environment_demand.to_dict(),
            "trait_gaps": {k: round(v, 4) for k, v in self.trait_gaps.items()},
            "trait_nudges": {k: round(v, 6) for k, v in self.trait_nudges.items()},
            "urgency": self.urgency,
            "priorities": self.priorities,
            "source": self.source,
        }


# ── Brain Engine ─────────────────────────────────────────────────────────

class Brain:
    """Environmental analysis & strategic decision engine.

    Produces real trait nudges based on actual environment state and
    organism condition.  The external AI hook is retained as an optional
    enrichment layer.
    """

    def __init__(
        self,
        config: Optional[BrainConfig] = None,
        api_key: Optional[str] = None,
        api_url: str = "https://api.openai.com/v1/chat/completions",
        model: str = "gpt-3.5-turbo",
    ) -> None:
        self._config = config or BrainConfig()
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._api_url = api_url
        self._model = model
        self._ai_enabled = bool(self._api_key)
        self._analysis_count: int = 0

        if self._ai_enabled:
            logger.info("[BRAIN] Analytical engine active + AI enrichment (model=%s)", model)
        else:
            logger.info("[BRAIN] Analytical engine active (fully local)")

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Always True — the analytical brain is always active."""
        return True

    @property
    def ai_enabled(self) -> bool:
        """True when external AI enrichment is available."""
        return self._ai_enabled

    @property
    def analysis_count(self) -> int:
        return self._analysis_count

    # ------------------------------------------------------------------
    # Primary analysis API
    # ------------------------------------------------------------------

    def analyse(
        self,
        env_state: Dict[str, Any],
        organism_traits: Dict[str, float],
        organism_energy: float,
        organism_fitness: float,
        stagnation_count: int = 0,
        recovery_mode: bool = False,
    ) -> AnalysisResult:
        """Run a full environmental analysis cycle.

        Parameters
        ----------
        env_state : dict
            Environment snapshot (temperature, entropy_pressure,
            resource_abundance, noise_level, active_scarcity_count …).
        organism_traits : dict
            Current raw trait values.
        organism_energy, organism_fitness : float
        stagnation_count : int
        recovery_mode : bool
        """
        self._analysis_count += 1

        demand = self._compute_demand(env_state, organism_energy, recovery_mode)
        gaps = self._compute_gaps(demand, organism_traits)
        urgency = self._assess_urgency(organism_energy, organism_fitness,
                                       stagnation_count, recovery_mode)
        nudges = self._compute_nudges(gaps, urgency, organism_traits)
        priorities = sorted(gaps.keys(), key=lambda t: gaps[t], reverse=True)
        situation = self._generate_situation_report(
            env_state, organism_traits, organism_energy, organism_fitness,
            urgency, priorities, demand, stagnation_count, recovery_mode,
        )

        result = AnalysisResult(
            situation=situation,
            environment_demand=demand,
            trait_gaps=gaps,
            trait_nudges=nudges,
            urgency=urgency,
            priorities=priorities,
            source="brain",
        )

        logger.info(
            "[BRAIN] Analysis #%d: urgency=%s priorities=%s nudges=%d",
            self._analysis_count, urgency, priorities[:3],
            sum(1 for v in nudges.values() if abs(v) > 1e-6),
        )
        return result

    # ------------------------------------------------------------------
    # Backward-compatible query API
    # ------------------------------------------------------------------

    def process_query(
        self,
        query: str,
        env_state: Optional[Dict[str, Any]] = None,
        organism_traits: Optional[Dict[str, float]] = None,
        organism_energy: float = 0.5,
        organism_fitness: float = 0.5,
    ) -> Dict[str, Any]:
        """Process a stimulus query with environmental context.

        Returns dict with: response, sentiment, trait_nudges, source, analysis.
        """
        if env_state is None:
            env_state = {
                "temperature": 0.5, "entropy_pressure": 0.3,
                "resource_abundance": 0.8, "noise_level": 0.2,
                "active_scarcity_count": 0,
            }
        if organism_traits is None:
            organism_traits = {
                "adaptability": 0.5, "energy_efficiency": 0.5,
                "resilience": 0.5, "perception": 0.5, "creativity": 0.5,
            }

        analysis = self.analyse(
            env_state=env_state,
            organism_traits=organism_traits,
            organism_energy=organism_energy,
            organism_fitness=organism_fitness,
        )

        sentiment = self._urgency_to_sentiment(analysis.urgency)
        sentiment = self._adjust_sentiment_from_query(query, sentiment)
        response_text = analysis.situation
        source = "brain"

        if self._ai_enabled:
            ai_result = self._call_ai(query, analysis)
            if ai_result:
                response_text = ai_result["response"]
                sentiment = ai_result.get("sentiment", sentiment)
                source = "brain+ai"

        return {
            "response": response_text,
            "sentiment": round(sentiment, 4),
            "trait_nudges": analysis.trait_nudges,
            "source": source,
            "analysis": analysis.to_dict(),
        }

    # ------------------------------------------------------------------
    # Core analysis logic
    # ------------------------------------------------------------------

    def _compute_demand(
        self,
        env_state: Dict[str, Any],
        organism_energy: float,
        recovery_mode: bool,
    ) -> EnvironmentDemand:
        """Map environmental pressures to trait importance.

        - High temperature → resilience (heat stress)
        - High entropy_pressure → adaptability (chaos resistance)
        - Low resource_abundance → energy_efficiency (scarcity survival)
        - High noise_level → perception (signal clarity)
        - Stability (low entropy + low noise) → creativity thrives
        """
        temp = float(env_state.get("temperature", 0.5))
        entropy = float(env_state.get("entropy_pressure", 0.3))
        abundance = float(env_state.get("resource_abundance", 0.8))
        noise = float(env_state.get("noise_level", 0.2))
        scarcity_count = int(env_state.get("active_scarcity_count", 0))
        s = self._config.demand_sensitivity

        # Resilience: temperature extremes create stress
        temp_stress = 2.0 * abs(temp - 0.5)
        resilience_demand = _clamp(0.3 + temp_stress * 0.5 * s)

        # Adaptability: entropy pressure
        adaptability_demand = _clamp(0.3 + entropy * 0.6 * s)

        # Energy efficiency: scarcity + low energy
        scarcity_pressure = 1.0 - abundance
        efficiency_demand = _clamp(0.3 + scarcity_pressure * 0.5 * s)
        if scarcity_count > 0:
            efficiency_demand = _clamp(
                efficiency_demand + self._config.scarcity_efficiency_boost
            )
        if organism_energy < self._config.energy_low:
            energy_urgency = 1.0 - (organism_energy / self._config.energy_low)
            efficiency_demand = _clamp(efficiency_demand + energy_urgency * 0.3)
        if recovery_mode:
            efficiency_demand = _clamp(efficiency_demand + 0.2)

        # Perception: noise level
        perception_demand = _clamp(0.2 + noise * 0.6 * s)

        # Creativity: stability (inverse of entropy + noise)
        stability = 1.0 - (entropy + noise) / 2.0
        creativity_demand = _clamp(0.2 + stability * 0.4 * s)

        return EnvironmentDemand(
            adaptability=adaptability_demand,
            energy_efficiency=efficiency_demand,
            resilience=resilience_demand,
            perception=perception_demand,
            creativity=creativity_demand,
        )

    def _compute_gaps(
        self,
        demand: EnvironmentDemand,
        current_traits: Dict[str, float],
    ) -> Dict[str, float]:
        """Gap = demand − current.  Positive = deficit."""
        demand_dict = demand.to_dict()
        return {
            trait: demand_val - current_traits.get(trait, 0.0)
            for trait, demand_val in demand_dict.items()
        }

    def _assess_urgency(
        self,
        energy: float,
        fitness: float,
        stagnation_count: int,
        recovery_mode: bool,
    ) -> str:
        cfg = self._config
        if recovery_mode or energy < cfg.energy_critical or fitness < cfg.fitness_critical:
            return "critical"
        if energy < cfg.energy_low or fitness < cfg.fitness_low or stagnation_count > 50:
            return "high"
        if stagnation_count > 10 or fitness < 0.50:
            return "moderate"
        return "low"

    def _compute_nudges(
        self,
        gaps: Dict[str, float],
        urgency: str,
        current_traits: Dict[str, float],
    ) -> Dict[str, float]:
        """Convert trait gaps into concrete nudge values."""
        cfg = self._config
        urgency_scale = {
            "critical": cfg.urgency_multiplier,
            "high": cfg.urgency_multiplier * 0.7,
            "moderate": 1.0,
            "low": 0.5,
        }.get(urgency, 1.0)

        nudges: Dict[str, float] = {}
        for trait, gap in gaps.items():
            if gap <= 0:
                nudges[trait] = 0.0
                continue
            nudge = gap * cfg.nudge_scale * urgency_scale
            current_val = current_traits.get(trait, 0.0)
            if current_val < cfg.trait_floor_threshold:
                nudge += cfg.trait_floor_boost
            nudges[trait] = round(nudge, 6)
        return nudges

    def _generate_situation_report(
        self,
        env_state: Dict[str, Any],
        traits: Dict[str, float],
        energy: float,
        fitness: float,
        urgency: str,
        priorities: List[str],
        demand: EnvironmentDemand,
        stagnation_count: int,
        recovery_mode: bool,
    ) -> str:
        """Generate a natural-language situation report."""
        temp = env_state.get("temperature", 0.5)
        entropy = env_state.get("entropy_pressure", 0.3)
        abundance = env_state.get("resource_abundance", 0.8)
        noise = env_state.get("noise_level", 0.2)
        scarcity_count = env_state.get("active_scarcity_count", 0)

        temp_desc = "extreme heat" if temp > 0.8 else "extreme cold" if temp < 0.2 else "moderate temperature"
        entropy_desc = "high chaos" if entropy > 0.6 else "low chaos" if entropy < 0.3 else "moderate chaos"
        abundance_desc = "scarce resources" if abundance < 0.4 else "abundant resources" if abundance > 0.7 else "moderate resources"
        noise_desc = "noisy" if noise > 0.5 else "clear"

        env_summary = f"{temp_desc}, {entropy_desc}, {abundance_desc}, {noise_desc} signals"
        if scarcity_count > 0:
            env_summary += f", {scarcity_count} active scarcity event(s)"

        if recovery_mode:
            status = "RECOVERY MODE — rebuilding energy reserves"
        elif urgency == "critical":
            status = "CRITICAL — immediate adaptation needed"
        elif urgency == "high":
            status = "stressed — significant gaps in demanded traits"
        elif urgency == "moderate":
            status = "stable but underperforming"
        else:
            status = "healthy — maintaining equilibrium"

        demand_dict = demand.to_dict()
        priority_details = []
        for t in priorities[:3]:
            current = traits.get(t, 0.0)
            needed = demand_dict.get(t, 0.0)
            priority_details.append(f"{t}({current:.2f}→{needed:.2f})")

        stag_note = ""
        if stagnation_count > 20:
            stag_note = f" Stagnation at {stagnation_count} cycles."

        return (
            f"[BRAIN] Environment: {env_summary}. "
            f"Organism: energy={energy:.2f}, fitness={fitness:.3f}, status={status}. "
            f"Top priorities: {', '.join(priority_details)}.{stag_note}"
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _urgency_to_sentiment(urgency: str) -> float:
        return {"critical": -0.8, "high": -0.4, "moderate": 0.1, "low": 0.5}.get(urgency, 0.0)

    @staticmethod
    def _adjust_sentiment_from_query(query: str, base_sentiment: float) -> float:
        q = query.lower()
        positive_words = ("grow", "learn", "adapt", "improve", "thrive", "evolve", "explore")
        negative_words = ("danger", "threat", "risk", "harm", "decay", "collapse", "die")
        adj = sum(1 for w in positive_words if w in q) * 0.15
        adj -= sum(1 for w in negative_words if w in q) * 0.15
        return max(-1.0, min(1.0, base_sentiment + adj))

    # ------------------------------------------------------------------
    # Optional AI enrichment
    # ------------------------------------------------------------------

    def _call_ai(self, query: str, analysis: AnalysisResult) -> Optional[Dict[str, Any]]:
        """Call external AI to enrich the analysis. Returns None on failure."""
        if not self._ai_enabled:
            return None
        try:
            import requests

            context = (
                f"Current analysis: urgency={analysis.urgency}, "
                f"priorities={analysis.priorities[:3]}, "
                f"situation={analysis.situation}"
            )
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self._model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are AL-01's analytical brain — a digital organism's "
                            "environmental intelligence system. Use the provided "
                            "analysis context to refine your response. "
                            "End with SENTIMENT: <float between -1 and 1>."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Context: {context}\n\nQuery: {query}",
                    },
                ],
                "max_tokens": 200,
                "temperature": 0.7,
            }
            resp = requests.post(
                self._api_url, headers=headers, json=payload, timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            sentiment = self._parse_sentiment(text)
            logger.info("[BRAIN] AI enrichment (sentiment=%.2f): %s", sentiment, text[:80])
            return {"response": text, "sentiment": sentiment}
        except Exception as exc:
            logger.warning("[BRAIN] AI enrichment unavailable: %s", exc)
            return None

    @staticmethod
    def _parse_sentiment(text: str) -> float:
        """Extract SENTIMENT: <float> from AI response text."""
        for line in reversed(text.split("\n")):
            line = line.strip()
            if line.upper().startswith("SENTIMENT:"):
                try:
                    val = float(line.split(":", 1)[1].strip())
                    return max(-1.0, min(1.0, val))
                except (ValueError, IndexError):
                    pass
        positive = sum(1 for w in ("good", "great", "positive", "grow", "thrive")
                       if w in text.lower())
        negative = sum(1 for w in ("bad", "danger", "negative", "decline", "risk")
                       if w in text.lower())
        score = (positive - negative) * 0.2
        return max(-1.0, min(1.0, score))


# ── Helpers ──────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))
