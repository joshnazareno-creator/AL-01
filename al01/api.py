"""AL-01 v3.3 FastAPI server — hardened HTTP interface with evolution, population, and AI brain."""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from pydantic import BaseModel, Field

from al01.organism import Organism, VERSION

logger = logging.getLogger("al01.api")

# ---------------------------------------------------------------------------
# Singleton reference — set by create_app() at startup
# ---------------------------------------------------------------------------
_organism: Optional[Organism] = None
_api_key: Optional[str] = None
_boot_time: float = time.monotonic()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class CommandPayload(BaseModel):
    command: str = Field(..., min_length=1, description="Command name")
    args: Optional[Dict[str, Any]] = Field(default=None, description="Optional arguments")


class InteractionPayload(BaseModel):
    user_input: str = Field(..., min_length=1, description="User input text")
    response: str = Field(..., min_length=1, description="Organism response text")
    mood: Optional[str] = Field(default=None, description="Mood label (optional)")


class StatusResponse(BaseModel):
    version: str
    organism_state: str
    evolution_count: int
    awareness: float
    state_version: int
    interaction_count: int
    last_boot_utc: Optional[str]
    loop_running: bool
    timestamp: str
    # v2.0
    genome: Optional[Dict[str, Any]] = None
    fitness: Optional[float] = None
    population_size: Optional[int] = None  # backward compat — same as member_count
    member_count: Optional[int] = None
    children_count: Optional[int] = None
    generations_present: Optional[List[int]] = None
    pending_stimuli: Optional[int] = None
    brain_enabled: Optional[bool] = None
    # v2.1 autonomy
    stagnation_count: int = 0
    autonomy: Optional[Dict[str, Any]] = None
    # v2.2 energy
    energy: float = 1.0
    # v3.8 champion + diversity
    champion: Optional[Dict[str, Any]] = None
    diversity: Optional[Dict[str, Any]] = None
    # v3.9 internal signal
    internal_signal: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    timestamp: str


class IdentityResponse(BaseModel):
    name: str
    description: str
    version: str
    runtime_version: str


class PolicyUpdate(BaseModel):
    weights: Dict[str, float] = Field(..., description="Weight key→value pairs")
    reason: str = Field("", description="Reason for the change")


class StimulusPayload(BaseModel):
    stimulus: str = Field(..., min_length=1, description="Stimulus event string (e.g. 'environmental_change')")
    trigger_cycle: bool = Field(True, description="If true, trigger an immediate evolution cycle")


class GPTStimulusPayload(BaseModel):
    text: str = Field(..., min_length=1, max_length=280, description="Stimulus text from GPT")


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

def _require_api_key(
    x_api_key: Optional[str] = Header(None),
    api_key: Optional[str] = Query(None, alias="api_key"),
) -> None:
    """Validate API key via ``X-API-Key`` header **or** ``?api_key=`` query param."""
    if _api_key is None:
        # No key configured — allow all requests (dev mode)
        return
    provided = x_api_key or api_key
    if provided is None or provided != _api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(organism: Organism, api_key: Optional[str] = None) -> FastAPI:
    """Build and return a FastAPI application wired to *organism*."""
    global _organism, _api_key, _boot_time
    _organism = organism
    _api_key = api_key
    _boot_time = time.monotonic()

    app = FastAPI(title="AL-01", version=VERSION)

    # --- CORS — required for ChatGPT Actions / browser callers ------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://chat.openai.com", "https://chatgpt.com", "*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Global exception handler ------------------------------------
    @app.exception_handler(Exception)
    async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("[API] Unhandled error: %s\n%s", exc, traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": "An unexpected error occurred."},
        )

    # ------------------------------------------------------------------
    # Shared status helper (used by /status JSON + / dashboard)
    # ------------------------------------------------------------------

    def _get_status() -> Dict[str, Any]:
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        state = dict(_organism.state)
        pop = _organism.population
        member_count = pop.size
        return {
            "version": VERSION,
            "organism_state": _organism.organism_state,
            "evolution_count": state.get("evolution_count", 0),
            "awareness": state.get("awareness", 0.0),
            "state_version": state.get("state_version", 0),
            "interaction_count": state.get("interaction_count", 0),
            "last_boot_utc": state.get("last_boot_utc"),
            "loop_running": _organism.loop_running,
            "timestamp": _utc_now(),
            # v2.0
            "genome": _organism.genome.to_dict(),
            "fitness": round(_organism.genome.fitness, 6),
            "population_size": member_count,   # backward compat
            "member_count": member_count,
            "children_count": pop.children_count,
            "generations_present": pop.generations_present,
            "pending_stimuli": len(state.get("stimuli", [])),
            "brain_enabled": _organism.brain.enabled,
            # v2.1 autonomy
            "stagnation_count": _organism.autonomy.stagnation_count,
            "autonomy": _organism.autonomy.summary(),
            # v2.2 energy
            "energy": round(float(state.get("energy", 1.0)), 6),
            # v3.8 champion + diversity
            "champion": pop.champion(),
            "diversity": pop.diversity_metrics(),
            # v3.9 internal signal
            "internal_signal": _organism.internal_signal.to_dict(),
        }

    # --- GET / (public HTML dashboard) --------------------------------
    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        """Public human-readable dashboard — no auth required."""
        s = _get_status()
        awareness_pct = round(s["awareness"] * 100, 2)
        bar_width = min(100, max(0, awareness_pct))
        loop_dot = "\u25cf" if s["loop_running"] else "\u25cb"
        loop_label = "running" if s["loop_running"] else "stopped"
        boot_time = s["last_boot_utc"] or "—"

        # Awareness breakdown components
        ab = s.get("autonomy", {})
        stag_icon = "⚠️" if ab.get("is_stagnant") else "✓"
        energy_val = s.get("energy", 1.0)
        energy_pct = round(energy_val * 100, 1)
        energy_bar = min(100, max(0, energy_pct))
        exploration_icon = "🔭" if ab.get("exploration_mode") else ""
        novelty_val = ab.get("novelty_accumulator", 0)

        # Genome traits table
        traits = s.get("genome", {}).get("traits", {})
        traits_rows = ""
        for tname, tval in traits.items():
            pct = round(tval * 100, 1)
            traits_rows += f'<tr><td>{tname}</td><td>{tval:.4f}</td><td><div class="bar-bg"><div class="bar-fg" style="width:{pct}%;background:#58a6ff"></div></div></td></tr>\n'

        # Population list
        pop_members = []
        if _organism:
            pop_members = _organism.population.get_all()
        pop_rows = ""
        for m in pop_members:
            mid = m.get("id", "?")
            mfit = m.get("genome", {}).get("fitness", 0)
            mevo = m.get("evolution_count", 0)
            mint = m.get("interaction_count", 0)
            mnick = m.get("nickname") or ""
            nick_badge = f' <span style="color:#e3b341;font-size:.7rem">({mnick})</span>' if mnick else ""
            pop_rows += f'<tr><td><a href="/population/{mid}?api_key={_api_key or ""}" style="color:#58a6ff;text-decoration:none">{mid}</a>{nick_badge}</td><td>{mfit:.4f}</td><td>{mevo}</td><td>{mint}</td></tr>\n'

        # History data for charts (from growth snapshots)
        history_json = "[]"
        if _organism:
            try:
                db = _organism._memory_manager.database
                conn = db._connect()
                rows = conn.execute(
                    "SELECT timestamp, awareness, evolution_count FROM growth_snapshots ORDER BY id DESC LIMIT 50"
                ).fetchall()
                conn.close()
                history = [{"t": r["timestamp"], "a": r["awareness"], "e": r["evolution_count"]} for r in reversed(rows)]
                history_json = json.dumps(history)
            except Exception:
                history_json = "[]"

        # v3.3: Population metrics for dashboard
        pop_fitness_map = {}
        metrics_avg = metrics_var = metrics_min = metrics_max = 0.0
        metrics_shannon = metrics_simpson = 0.0
        convergence_status = "n/a"
        strategy_dist: Dict[str, int] = {}
        if _organism:
            pop_fitness_map = _organism.population.population_fitness()
            if pop_fitness_map:
                fvals = list(pop_fitness_map.values())
                metrics_avg = sum(fvals) / len(fvals)
                metrics_min = min(fvals)
                metrics_max = max(fvals)
                if len(fvals) >= 2:
                    import statistics as _stats
                    metrics_var = _stats.variance(fvals)
            bsummary = _organism.behavior_analyzer.summary()
            div = bsummary.get("diversity", {})
            metrics_shannon = div.get("shannon", 0.0)
            metrics_simpson = div.get("simpson", 0.0)
            convergence_status = bsummary.get("convergence", {}).get("status", "n/a")
            strategy_dist = bsummary.get("strategy_distribution", {})

        strategy_rows = ""
        for sname, scount in strategy_dist.items():
            strategy_rows += f'<tr><td>{sname}</td><td>{scount}</td></tr>\n'

        # v3.8: Champion + diversity + population counts
        champion_data = s.get("champion")
        champion_row = "<tr><td>champion</td><td>— (no children)</td></tr>" if not champion_data else (
            f'<tr><td>\U0001F3C6 champion</td><td>{champion_data["champion_id"]} (fitness {champion_data["champion_fitness"]:.4f})</td></tr>'
        )
        diversity_data = s.get("diversity", {})
        trait_stddev = diversity_data.get("trait_stddev", {})
        unique_hashes = diversity_data.get("unique_genome_hashes", 0)
        genome_entropy = diversity_data.get("genome_entropy", 0.0)
        stddev_rows = ""
        for tname, sd in trait_stddev.items():
            stddev_rows += f'<tr><td>{tname}</td><td>{sd:.4f}</td></tr>\n'
        children_count = s.get("children_count", 0)
        generations = s.get("generations_present", [])

        # v3.9: Internal signal
        signal_data = s.get("internal_signal", {})
        signal_energy = signal_data.get("energy_state", 0.0)
        signal_stress = signal_data.get("stress_level", 0.0)
        signal_novelty = signal_data.get("novelty_drive", 0.0)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="5">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AL-01 Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{background:#0d1117;color:#c9d1d9;font-family:'Courier New',Courier,monospace;
        display:flex;justify-content:center;padding:1.5rem}}
  .wrap{{max-width:900px;width:100%}}
  h1{{color:#58a6ff;font-size:1.4rem;margin-bottom:.25rem}}
  h2{{color:#58a6ff;font-size:1rem;margin:1.2rem 0 .5rem;border-bottom:1px solid #21262d;padding-bottom:.25rem}}
  .sub{{color:#8b949e;font-size:.75rem;margin-bottom:1rem}}
  table{{width:100%;border-collapse:collapse;margin-bottom:.5rem}}
  td,th{{padding:.35rem .5rem;border-bottom:1px solid #21262d;font-size:.8rem;text-align:left}}
  td:first-child,th:first-child{{color:#8b949e;width:40%}}
  td:last-child{{color:#f0f6fc;text-align:right}}
  .bar-bg{{background:#21262d;border-radius:4px;height:8px;width:100%;margin-top:2px}}
  .bar-fg{{background:#3fb950;border-radius:4px;height:8px}}
  .dot{{font-size:1rem}}
  .dot.on{{color:#3fb950}}
  .dot.off{{color:#f85149}}
  .ts{{color:#484f58;font-size:.7rem;text-align:center;margin-top:1rem}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}}
  @media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}
  .card{{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:.8rem}}
  .stim-form{{margin-top:.8rem;display:flex;gap:.5rem}}
  .stim-input{{flex:1;background:#0d1117;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;
               padding:.4rem .6rem;font-size:.8rem;font-family:inherit}}
  .btn{{background:#238636;color:#f0f6fc;border:1px solid #2ea043;border-radius:6px;
        padding:.4rem 1rem;font-size:.8rem;font-family:inherit;cursor:pointer}}
  .btn:hover{{background:#2ea043}}
  .btn:active{{background:#196c2e}}
  .btn-blue{{background:#1f6feb;border-color:#388bfd}}
  .btn-blue:hover{{background:#388bfd}}
  canvas{{max-height:180px;margin-top:.4rem}}
  a{{color:#58a6ff}}
</style>
</head>
<body>
<div class="wrap">
  <h1>AL-01</h1>
  <div class="sub">persistent digital organism &mdash; v{s["version"]} &bull; fitness {s["fitness"]:.4f} &bull; members {s["member_count"]} (children {children_count})</div>

  <div class="grid">
    <div class="card">
      <h2>Core State</h2>
      <table>
        <tr><td>organism_state</td><td>{s["organism_state"]}</td></tr>
        <tr><td>evolution_count</td><td>{s["evolution_count"]}</td></tr>
        <tr><td>awareness</td><td>
          {awareness_pct}%
          <div class="bar-bg"><div class="bar-fg" style="width:{bar_width}%"></div></div>
        </td></tr>
        <tr><td>state_version</td><td>{s["state_version"]}</td></tr>
        <tr><td>interaction_count</td><td>{s["interaction_count"]}</td></tr>
        <tr><td>loop_running</td><td><span class="dot {"on" if s["loop_running"] else "off"}">{loop_dot}</span> {loop_label}</td></tr>
        <tr><td>pending_stimuli</td><td>{s["pending_stimuli"]}</td></tr>
        <tr><td>brain</td><td>{"🧠 AI-Enhanced" if _organism.brain.ai_enabled else "🧠 Analytical"}</td></tr>
        <tr><td>stagnation</td><td>{s["stagnation_count"]} {stag_icon}</td></tr>
        <tr><td>decisions</td><td>{ab.get("total_decisions", 0)}</td></tr>
        <tr><td>energy</td><td>
          {energy_pct}%
          <div class="bar-bg"><div class="bar-fg" style="width:{energy_bar}%;background:#f0883e"></div></div>
        </td></tr>
        <tr><td>last_boot_utc</td><td style="font-size:.7rem">{boot_time}</td></tr>
      </table>

      <h2 style="margin-top:.6rem">Awareness Breakdown</h2>
      <table>
        <tr><td>stimuli_rate</td><td style="color:#3fb950">× 0.4</td><td>{ab.get("awareness", 0):.4f}</td></tr>
        <tr><td>decision_rate</td><td style="color:#58a6ff">× 0.3</td><td>{ab.get("total_decisions", 0)}</td></tr>
        <tr><td>fitness_variance</td><td style="color:#d2a8ff">× 0.2</td><td>{ab.get("fitness_variance", 0):.6f}</td></tr>
        <tr><td>¬stagnation</td><td style="color:#f0883e">× 0.1</td><td>{"0.0" if ab.get("is_stagnant") else "1.0"}</td></tr>
        <tr><td>novelty</td><td style="color:#e3b341">+</td><td>{novelty_val:.4f}</td></tr>
        <tr><td>total_stimuli</td><td></td><td>{ab.get("total_stimuli", 0)}</td></tr>
      </table>

      <h2 style="margin-top:.6rem">Environment</h2>
      <table>
        <tr><td>eff. fitness threshold</td><td>{ab.get("effective_fitness_threshold", 0.45):.4f}</td></tr>
        <tr><td>mutation rate offset</td><td>{ab.get("mutation_rate_offset", 0):.4f}</td></tr>
        <tr><td>exploration mode</td><td>{exploration_icon} {"active" if ab.get("exploration_mode") else "off"}</td></tr>
      </table>

      <form class="stim-form" method="post" action="/stimulate?api_key={_api_key or ''}&_redirect=1">
        <button class="btn" type="submit">⚡ Stimulate</button>
      </form>
    </div>

    <div class="card">
      <h2>Genome (fitness {s["fitness"]:.4f})</h2>
      <table>
        <tr><th>Trait</th><th>Value</th><th></th></tr>
        {traits_rows}
      </table>

      <h2>Send Stimulus</h2>
      <form class="stim-form" id="stimForm">
        <input class="stim-input" type="text" id="stimInput" placeholder="e.g. environmental_change or query:how to adapt?">
        <button class="btn btn-blue" type="submit">Send</button>
      </form>
    </div>
  </div>

  <div class="grid" style="margin-top:1rem">
    <div class="card">
      <h2>Population ({s["member_count"]})</h2>
      <table>
        <tr><td>member_count</td><td>{s["member_count"]}</td></tr>
        <tr><td>children_count</td><td>{children_count}</td></tr>
        <tr><td>generations</td><td>{generations}</td></tr>
        {champion_row}
      </table>
      <table style="margin-top:.5rem">
        <tr><th>ID</th><th>Fitness</th><th>Evol</th><th>Int</th></tr>
        {pop_rows}
      </table>
    </div>

    <div class="card">
      <h2>Evolution History</h2>
      <canvas id="evoChart"></canvas>
    </div>
  </div>

  <div class="grid" style="margin-top:1rem">
    <div class="card">
      <h2>Population Metrics</h2>
      <table>
        <tr><td>avg fitness</td><td>{metrics_avg:.4f}</td></tr>
        <tr><td>min / max</td><td>{metrics_min:.4f} / {metrics_max:.4f}</td></tr>
        <tr><td>fitness variance</td><td>{metrics_var:.6f}</td></tr>
        <tr><td>Shannon diversity</td><td>{metrics_shannon:.4f}</td></tr>
        <tr><td>Simpson diversity</td><td>{metrics_simpson:.4f}</td></tr>
        <tr><td>convergence</td><td>{convergence_status}</td></tr>
      </table>
      <h2 style="margin-top:.6rem">Genome Diversity</h2>
      <table>
        <tr><td>unique genomes</td><td>{unique_hashes}</td></tr>
        <tr><td>genome entropy</td><td>{genome_entropy:.4f} bits</td></tr>
      </table>
      <h2 style="margin-top:.6rem">Internal Signal</h2>
      <table>
        <tr><td>energy_state</td><td>{signal_energy:.4f}</td></tr>
        <tr><td>stress_level</td><td>{signal_stress:.4f}</td></tr>
        <tr><td>novelty_drive</td><td>{signal_novelty:.4f}</td></tr>
      </table>
      <h2 style="margin-top:.6rem">Trait Std Dev</h2>
      <table>
        <tr><th>Trait</th><th>StdDev</th></tr>
        {stddev_rows}
      </table>
      <h2 style="margin-top:.6rem">Strategy Distribution</h2>
      <table>
        <tr><th>Strategy</th><th>Count</th></tr>
        {strategy_rows}
      </table>
    </div>

    <div class="card">
      <h2>Data Export</h2>
      <table>
        <tr><td><a href="/export/fitness.csv?api_key={_api_key or ''}">📊 fitness.csv</a></td><td>Fitness trajectories</td></tr>
        <tr><td><a href="/export/mutations.csv?api_key={_api_key or ''}">🧬 mutations.csv</a></td><td>Mutation events</td></tr>
        <tr><td><a href="/export/lineage.csv?api_key={_api_key or ''}">🌳 lineage.csv</a></td><td>Genealogy data</td></tr>
      </table>
    </div>
  </div>

  <div class="ts">refreshes every 5 s &mdash; {s["timestamp"]}</div>
</div>

<script>
// Stimulus form — POST to /stimulate with JSON body
document.getElementById('stimForm').addEventListener('submit', async function(e) {{
  e.preventDefault();
  const val = document.getElementById('stimInput').value.trim();
  if (!val) return;
  try {{
    await fetch('/stimulate?api_key={_api_key or ""}', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{stimulus: val, trigger_cycle: true}})
    }});
    document.getElementById('stimInput').value = '';
    location.reload();
  }} catch(err) {{ console.error(err); }}
}});

// Chart.js — evolution + awareness over time
const historyData = {history_json};
if (historyData.length > 0) {{
  const labels = historyData.map(d => d.t ? d.t.substring(11,19) : '');
  new Chart(document.getElementById('evoChart'), {{
    type: 'line',
    data: {{
      labels: labels,
      datasets: [
        {{
          label: 'Awareness',
          data: historyData.map(d => d.a),
          borderColor: '#3fb950',
          backgroundColor: 'rgba(63,185,80,0.1)',
          tension: 0.3,
          yAxisID: 'y',
        }},
        {{
          label: 'Evolution',
          data: historyData.map(d => d.e),
          borderColor: '#58a6ff',
          backgroundColor: 'rgba(88,166,255,0.1)',
          tension: 0.3,
          yAxisID: 'y1',
        }}
      ]
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ labels: {{ color: '#8b949e', font: {{ size: 10 }} }} }} }},
      scales: {{
        x: {{ ticks: {{ color: '#484f58', font: {{ size: 9 }} }}, grid: {{ color: '#21262d' }} }},
        y: {{ position: 'left', min: 0, max: 1, ticks: {{ color: '#3fb950', font: {{ size: 9 }} }}, grid: {{ color: '#21262d' }} }},
        y1: {{ position: 'right', ticks: {{ color: '#58a6ff', font: {{ size: 9 }} }}, grid: {{ drawOnChartArea: false }} }},
      }}
    }}
  }});
}}
</script>
</body>
</html>"""
        return HTMLResponse(content=html)

    # --- POST /stimulate ----------------------------------------------
    @app.post("/stimulate", dependencies=[Depends(_require_api_key)])
    def stimulate(
        _redirect: Optional[str] = Query(None, alias="_redirect"),
        payload: Optional[StimulusPayload] = None,
    ) -> Any:
        """External stimulus — accepts optional JSON body with 'stimulus' string.

        If ``stimulus`` is provided, it's queued for the next evolution cycle.
        If ``trigger_cycle`` is true and stimulus provided, an immediate
        evolution cycle is triggered.
        """
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")

        stimulus_text = payload.stimulus if payload else None
        _organism.stimulate(stimulus=stimulus_text)

        # Trigger immediate evolution if requested
        if payload and payload.trigger_cycle and stimulus_text:
            _organism.evolve_cycle()

        if _redirect:
            return RedirectResponse(url="/", status_code=303)
        return _get_status()

    # --- GET /evolve --------------------------------------------------
    @app.get("/evolve", dependencies=[Depends(_require_api_key)])
    def evolve() -> Dict[str, Any]:
        """Force an immediate evolution cycle (for testing).

        Mutates genome, increments evolution_count, returns new state.
        """
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        result = _organism.force_evolve()
        status = _get_status()
        return {
            "evolution_result": result,
            "status": status,
        }

    # --- GET /population ----------------------------------------------
    @app.get("/population", dependencies=[Depends(_require_api_key)])
    def population_list() -> Dict[str, Any]:
        """Return all organisms in the population."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        pop = _organism.population
        members = pop.get_all()
        return {
            "population_size": len(members),  # backward compat
            "member_count": len(members),
            "children_count": pop.children_count,
            "generations_present": pop.generations_present,
            "champion": pop.champion(),
            "members": members,
        }

    # --- GET /population/metrics --------------------------------------
    @app.get("/population/metrics", dependencies=[Depends(_require_api_key)])
    def population_metrics() -> Dict[str, Any]:
        """Return aggregate population metrics: fitness stats, variance, diversity, convergence."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        pop = _organism.population
        pop_fitness = pop.population_fitness()
        fitness_stats = _organism.evolution_tracker.population_fitness_stats(pop_fitness)
        trait_variance = pop.trait_variance()
        behavior_summary = _organism.behavior_analyzer.summary()
        return {
            "fitness_stats": fitness_stats,
            "trait_variance": trait_variance,
            "strategy_distribution": behavior_summary.get("strategy_distribution", {}),
            "diversity": behavior_summary.get("diversity", {}),
            "convergence": behavior_summary.get("convergence", {}),
            "tracked_organisms": behavior_summary.get("tracked_organisms", 0),
            "population_size": pop.size,  # backward compat
            "member_count": pop.size,
            "children_count": pop.children_count,
            "generations_present": pop.generations_present,
            "champion": pop.champion(),
            "genome_diversity": pop.diversity_metrics(),
            "timestamp": _utc_now(),
        }

    # --- GET /population/{organism_id} --------------------------------
    @app.get("/population/{organism_id}", dependencies=[Depends(_require_api_key)])
    def population_member(organism_id: str) -> Dict[str, Any]:
        """Return details for a specific organism in the population."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        member = _organism.population.get(organism_id)
        if member is None:
            raise HTTPException(status_code=404, detail=f"Organism '{organism_id}' not found")
        return member

    # --- PUT /population/{organism_id}/nickname -----------------------
    @app.put("/population/{organism_id}/nickname", dependencies=[Depends(_require_api_key)])
    def set_nickname(organism_id: str, nickname: Optional[str] = Query(None, description="Nickname (null to clear)")) -> Dict[str, Any]:
        """Set or clear a nickname for an organism."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        ok = _organism.population.set_nickname(organism_id, nickname)
        if not ok:
            raise HTTPException(status_code=404, detail=f"Organism '{organism_id}' not found")
        return {
            "organism_id": organism_id,
            "nickname": nickname,
            "status": "set" if nickname else "cleared",
        }

    # --- GET /population/{organism_id}/export -------------------------
    @app.get("/population/{organism_id}/export", dependencies=[Depends(_require_api_key)])
    def export_child(organism_id: str) -> JSONResponse:
        """Export a child organism as a downloadable JSON snapshot."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        from al01.portable import export_child as _export
        try:
            snapshot = _export(_organism.population, organism_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        headers = {
            "Content-Disposition": f'attachment; filename="{organism_id}.json"',
        }
        return JSONResponse(content=snapshot, headers=headers)

    # --- POST /population/import --------------------------------------
    @app.post("/population/import", dependencies=[Depends(_require_api_key)])
    async def import_child(request: Request) -> Dict[str, Any]:
        """Import (adopt) a previously exported child organism from JSON."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")
        from al01.portable import import_child as _import
        try:
            record = _import(
                _organism.population,
                payload,
                life_log=_organism.life_log,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {
            "status": "adopted",
            "new_id": record["id"],
            "original_id": record.get("original_id"),
            "generation": record.get("generation_id"),
            "fitness": record.get("genome", {}).get("fitness", 0.0),
        }

    # --- GET /genome --------------------------------------------------
    @app.get("/genome", dependencies=[Depends(_require_api_key)])
    def genome() -> Dict[str, Any]:
        """Return the current genome traits and fitness."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        return _organism.genome.to_dict()

    # --- GET /stimuli -------------------------------------------------
    @app.get("/stimuli", dependencies=[Depends(_require_api_key)])
    def stimuli_list() -> Dict[str, Any]:
        """Return pending stimuli queue."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        return {
            "stimuli": _organism.stimuli,
            "count": len(_organism.stimuli),
        }

    # --- GET /autonomy ------------------------------------------------
    @app.get("/autonomy", dependencies=[Depends(_require_api_key)])
    def autonomy_status() -> Dict[str, Any]:
        """Return autonomy engine state — stagnation, fitness history, decisions."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        summary = _organism.autonomy.summary()
        summary["fitness_history"] = _organism.autonomy.fitness_history
        return summary

    # --- GET /health --------------------------------------------------
    @app.get("/health", response_model=HealthResponse)
    def health() -> Dict[str, Any]:
        """Lightweight health check — no auth required."""
        return {
            "status": "ok",
            "version": VERSION,
            "uptime_seconds": round(time.monotonic() - _boot_time, 2),
            "timestamp": _utc_now(),
        }

    # --- GET /identity -----------------------------------------------
    @app.get("/identity", response_model=IdentityResponse, dependencies=[Depends(_require_api_key)])
    def identity() -> Dict[str, Any]:
        """Return identity.json contents + runtime VERSION."""
        identity_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "identity.json"
        )
        try:
            with open(identity_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            data = {"name": "AL-01", "description": "A persistent digital organism", "version": "unknown"}
        data["runtime_version"] = VERSION
        return data

    # --- GET /status --------------------------------------------------
    @app.get("/status", response_model=StatusResponse, dependencies=[Depends(_require_api_key)])
    def status() -> Dict[str, Any]:
        """Return the current organism state snapshot (JSON, auth required)."""
        return _get_status()

    # --- GET /growth --------------------------------------------------
    @app.get("/growth", dependencies=[Depends(_require_api_key)])
    def growth() -> Dict[str, Any]:
        """Return growth metrics + totals since first run."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        return _organism.growth_metrics

    # --- POST /command ------------------------------------------------
    @app.post("/command", dependencies=[Depends(_require_api_key)])
    def command(payload: CommandPayload) -> Dict[str, Any]:
        """Accept a JSON command, log it, and acknowledge."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")

        logger.info("[API] Command received: %s args=%s", payload.command, payload.args)

        _organism._memory_manager.write_memory(
            {
                "event_type": "api_command",
                "payload": {
                    "command": payload.command,
                    "args": payload.args or {},
                },
            }
        )

        return {
            "status": "accepted",
            "command": payload.command,
            "timestamp": _utc_now(),
        }

    # --- POST /interact -----------------------------------------------
    @app.post("/interact", dependencies=[Depends(_require_api_key)])
    def interact(payload: InteractionPayload) -> Dict[str, Any]:
        """Record a structured interaction (writes to SQLite + Firestore/JSON)."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")

        entry = _organism.record_interaction(
            user_input=payload.user_input,
            response=payload.response,
            mood=payload.mood,
        )
        return {
            "status": "recorded",
            "entry": entry,
            "timestamp": _utc_now(),
        }

    # --- GET /memory/recent -------------------------------------------
    @app.get("/memory/recent", dependencies=[Depends(_require_api_key)])
    def memory_recent(n: int = Query(5, ge=1, le=50)) -> Dict[str, Any]:
        """Return the last *n* interactions from SQLite."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        interactions = _organism.recent_interactions(n=n)
        return {
            "interactions": interactions,
            "count": len(interactions),
        }

    # --- GET /memory/search -------------------------------------------
    @app.get("/memory/search", dependencies=[Depends(_require_api_key)])
    def memory_search(
        keyword: str = Query(..., min_length=1),
        limit: int = Query(20, ge=1, le=100),
        since: Optional[str] = Query(None, description="ISO-8601 timestamp lower bound"),
    ) -> Dict[str, Any]:
        """Search memory entries by keyword across user_input, response, mood."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        results = _organism.search_memory(keyword, limit=limit, since_timestamp=since)
        return {
            "keyword": keyword,
            "results": results,
            "count": len(results),
        }

    # =================================================================
    # VITAL endpoints
    # =================================================================

    # --- GET /vital/status --------------------------------------------
    @app.get("/vital/status", dependencies=[Depends(_require_api_key)])
    def vital_status() -> Dict[str, Any]:
        """VITAL subsystem status: integrity, head, policy weights."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        head = _organism.life_log.head
        return {
            "integrity_status": _organism.life_log.integrity_status,
            "head_seq": head.get("head_seq", 0),
            "head_hash": head.get("head_hash"),
            "total_events": _organism.life_log.event_count(),
            "policy_weights": _organism.policy.weights,
            "birth_time": dict(_organism.state).get("birth_time"),
            "age_seconds": round(_organism.age_seconds, 2),
            "uptime_sessions": dict(_organism.state).get("uptime_sessions", 0),
        }

    # --- GET /vital/verify --------------------------------------------
    @app.get("/vital/verify", dependencies=[Depends(_require_api_key)])
    def vital_verify(
        last: int = Query(500, ge=1, le=10000),
    ) -> Dict[str, Any]:
        """Verify hash-chain integrity of the life log."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        return _organism.life_log.verify_full_report(last_n=last)

    # --- GET /vital/head ----------------------------------------------
    @app.get("/vital/head", dependencies=[Depends(_require_api_key)])
    def vital_head() -> Dict[str, Any]:
        """Return current life-log head."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        return _organism.life_log.head

    # --- GET /vital/policy --------------------------------------------
    @app.get("/vital/policy", dependencies=[Depends(_require_api_key)])
    def vital_policy_get() -> Dict[str, Any]:
        """Return current adaptive policy weights."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        return _organism.policy.weights

    # --- POST /vital/policy -------------------------------------------
    @app.post("/vital/policy", dependencies=[Depends(_require_api_key)])
    def vital_policy_update(payload: PolicyUpdate) -> Dict[str, Any]:
        """Update adaptive policy weights."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        record = _organism.update_policy(payload.weights, reason=payload.reason)
        return {
            "status": "updated",
            "change_record": record,
            "timestamp": _utc_now(),
        }

    # --- GET /snapshots ------------------------------------------------
    @app.get("/snapshots", dependencies=[Depends(_require_api_key)])
    def snapshot_list(
        limit: int = Query(20, ge=1, le=200),
        label: Optional[str] = Query(None),
        since: Optional[str] = Query(None, description="ISO-8601 lower bound"),
    ) -> Dict[str, Any]:
        """List recent state snapshots."""
        if _organism is None or _organism.snapshot_manager is None:
            raise HTTPException(status_code=500, detail="Snapshot manager not initialized")
        entries = _organism.snapshot_manager.list_snapshots(limit=limit, label=label, since=since)
        return {
            "snapshots": entries,
            "count": len(entries),
            "timestamp": _utc_now(),
        }

    # --- POST /snapshots -----------------------------------------------
    @app.post("/snapshots", dependencies=[Depends(_require_api_key)])
    def snapshot_take(
        label: Optional[str] = Query("api", description="Snapshot label"),
    ) -> Dict[str, Any]:
        """Take an immediate state snapshot."""
        if _organism is None or _organism.snapshot_manager is None:
            raise HTTPException(status_code=500, detail="Snapshot manager not initialized")
        entry = _organism.snapshot_manager.take_snapshot(label=label)
        return {
            "status": "created",
            "snapshot": entry,
            "timestamp": _utc_now(),
        }

    # --- GET /snapshots/status -----------------------------------------
    @app.get("/snapshots/status", dependencies=[Depends(_require_api_key)])
    def snapshot_status() -> Dict[str, Any]:
        """Return snapshot manager status."""
        if _organism is None or _organism.snapshot_manager is None:
            raise HTTPException(status_code=500, detail="Snapshot manager not initialized")
        return _organism.snapshot_manager.status()

    # --- GET /snapshots/latest -----------------------------------------
    @app.get("/snapshots/latest", dependencies=[Depends(_require_api_key)])
    def snapshot_latest() -> Dict[str, Any]:
        """Load the most recent snapshot."""
        if _organism is None or _organism.snapshot_manager is None:
            raise HTTPException(status_code=500, detail="Snapshot manager not initialized")
        snap = _organism.snapshot_manager.latest_snapshot()
        if snap is None:
            raise HTTPException(status_code=404, detail="No snapshots found")
        return snap

    # --- DELETE /snapshots/purge ---------------------------------------
    @app.delete("/snapshots/purge", dependencies=[Depends(_require_api_key)])
    def snapshot_purge(
        days: int = Query(30, ge=1, le=365, description="Delete snapshots older than N days"),
    ) -> Dict[str, Any]:
        """Purge snapshots older than N days."""
        if _organism is None or _organism.snapshot_manager is None:
            raise HTTPException(status_code=500, detail="Snapshot manager not initialized")
        deleted = _organism.snapshot_manager.purge_older_than(days)
        return {
            "status": "purged",
            "deleted_count": deleted,
            "retention_days": days,
            "timestamp": _utc_now(),
        }

    # ------------------------------------------------------------------
    # v3.2: Genesis Vault endpoints
    # ------------------------------------------------------------------

    @app.get("/vault", dependencies=[Depends(_require_api_key)])
    def vault_status() -> Dict[str, Any]:
        """Get Genesis Vault status and seed information."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        return _organism.genesis_vault.status()

    @app.get("/vault/seed", dependencies=[Depends(_require_api_key)])
    def vault_seed() -> Dict[str, Any]:
        """Get the frozen genesis seed template."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        seed = _organism.genesis_vault.seed
        return {
            "seed_name": seed.get("seed_name"),
            "traits": seed.get("traits"),
            "mutation_rate": seed.get("mutation_rate"),
            "mutation_delta": seed.get("mutation_delta"),
            "created_at": seed.get("created_at"),
        }

    @app.get("/vault/history", dependencies=[Depends(_require_api_key)])
    def vault_reseed_history() -> Dict[str, Any]:
        """Get the full reseed history."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        history = _organism.genesis_vault.reseed_history
        return {
            "reseed_count": _organism.genesis_vault.reseed_count,
            "history": history,
        }

    @app.post("/vault/reseed", dependencies=[Depends(_require_api_key)])
    def vault_force_reseed() -> Dict[str, Any]:
        """Force a reseed from the Genesis Vault (manual extinction recovery)."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        if _organism.population.size > 0:
            return {
                "status": "skipped",
                "reason": "population_alive",
                "population_size": _organism.population.size,
                "message": "Cannot reseed — population is not extinct.",
            }
        reseed = _organism.check_extinction_reseed()
        if reseed:
            return {"status": "reseeded", **reseed}
        return {"status": "no_action", "message": "Reseed check returned no action."}

    # ------------------------------------------------------------------
    # v3.3: ALife 2026 — Lineage, Metrics, Exports, Experiment, Exploration
    # ------------------------------------------------------------------

    # --- GET /lineage --------------------------------------------------
    @app.get("/lineage", dependencies=[Depends(_require_api_key)])
    def lineage_all() -> Dict[str, Any]:
        """Return the full genealogy tree for all tracked organisms."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        lineages = _organism.evolution_tracker.get_all_lineages()
        return {
            "lineage": lineages,
            "total_organisms": len(lineages),
            "timestamp": _utc_now(),
        }

    @app.get("/lineage/{organism_id}", dependencies=[Depends(_require_api_key)])
    def lineage_single(organism_id: str) -> Dict[str, Any]:
        """Return lineage info for a specific organism."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        info = _organism.evolution_tracker.get_lineage(organism_id)
        if info is None:
            raise HTTPException(status_code=404, detail=f"No lineage for '{organism_id}'")
        return info

    # --- CSV Export endpoints ------------------------------------------
    @app.get("/export/fitness.csv", dependencies=[Depends(_require_api_key)])
    def export_fitness_csv() -> PlainTextResponse:
        """Download fitness trajectories as CSV."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        csv_content = _organism.evolution_tracker.export_fitness_csv()
        return PlainTextResponse(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=fitness.csv"},
        )

    @app.get("/export/mutations.csv", dependencies=[Depends(_require_api_key)])
    def export_mutations_csv() -> PlainTextResponse:
        """Download mutation events as CSV."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        csv_content = _organism.evolution_tracker.export_mutations_csv()
        return PlainTextResponse(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=mutations.csv"},
        )

    @app.get("/export/lineage.csv", dependencies=[Depends(_require_api_key)])
    def export_lineage_csv() -> PlainTextResponse:
        """Download lineage data as CSV."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        csv_content = _organism.evolution_tracker.export_lineage_csv()
        return PlainTextResponse(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=lineage.csv"},
        )

    # --- Experiment endpoints ------------------------------------------
    @app.post("/experiment/start", dependencies=[Depends(_require_api_key)])
    def experiment_start() -> Dict[str, Any]:
        """Start the experiment protocol."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        if _organism.experiment is None:
            raise HTTPException(status_code=400, detail="No experiment configured")
        if _organism.experiment.active:
            return {"status": "already_active", **_organism.experiment.status()}
        return {"status": "started", **_organism.experiment.start()}

    @app.post("/experiment/stop", dependencies=[Depends(_require_api_key)])
    def experiment_stop(
        reason: str = Query("manual", description="Stop reason"),
    ) -> Dict[str, Any]:
        """Stop the experiment protocol."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        if _organism.experiment is None:
            raise HTTPException(status_code=400, detail="No experiment configured")
        if not _organism.experiment.active:
            return {"status": "not_active", **_organism.experiment.status()}
        return {"status": "stopped", **_organism.experiment.stop(reason=reason)}

    @app.get("/experiment/status", dependencies=[Depends(_require_api_key)])
    def experiment_status() -> Dict[str, Any]:
        """Get current experiment status."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        if _organism.experiment is None:
            return {"status": "no_experiment", "active": False}
        return _organism.experiment.status()

    # --- Exploration mode toggle ---------------------------------------
    @app.post("/exploration/toggle", dependencies=[Depends(_require_api_key)])
    def exploration_toggle(
        enabled: bool = Query(True, description="Enable or disable exploration mode"),
        cycles: int = Query(0, ge=0, description="Duration in cycles (0=default)"),
    ) -> Dict[str, Any]:
        """Manually toggle exploration mode (adaptive mutation boost)."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        result = _organism.autonomy.set_exploration_mode(enabled, cycles)
        return {
            "status": "toggled",
            **result,
            "timestamp": _utc_now(),
        }

    # ------------------------------------------------------------------
    # v3.4: GPT Bridge — natural-language narration + controlled stimulus
    # ------------------------------------------------------------------

    @app.get("/gpt/narrate", dependencies=[Depends(_require_api_key)])
    def gpt_narrate() -> Dict[str, Any]:
        """Return AL-01's current state as natural-language prose.

        Designed to be consumed by a GPT system prompt so the model
        understands what the organism is doing without parsing JSON.
        """
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        return _organism.gpt_bridge.narrate()

    @app.post("/gpt/stimulus", dependencies=[Depends(_require_api_key)])
    def gpt_stimulus(payload: GPTStimulusPayload) -> Dict[str, Any]:
        """Inject a rate-limited stimulus from GPT.

        The stimulus is queued for the next evolution cycle — it does
        **not** trigger an immediate cycle, so evolution timing is
        preserved.
        """
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        return _organism.gpt_bridge.inject_stimulus(payload.text)

    @app.get("/gpt/status", dependencies=[Depends(_require_api_key)])
    def gpt_bridge_status() -> Dict[str, Any]:
        """Return GPT bridge statistics — rate-limit window, injection count, etc."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        return _organism.gpt_bridge.status()

    @app.post("/gpt/toggle", dependencies=[Depends(_require_api_key)])
    def gpt_toggle_stimulus(
        enabled: bool = Query(True, description="Enable or disable GPT stimulus injection"),
    ) -> Dict[str, Any]:
        """Enable or disable GPT stimulus injection."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        return _organism.gpt_bridge.set_stimulus_enabled(enabled)

    @app.get("/gpt/log", dependencies=[Depends(_require_api_key)])
    def gpt_injection_log(
        limit: int = Query(20, ge=1, le=200, description="Number of recent entries"),
    ) -> Dict[str, Any]:
        """Return recent GPT stimulus injection log."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        entries = _organism.gpt_bridge.recent_injections(limit=limit)
        return {
            "entries": entries,
            "count": len(entries),
            "timestamp": _utc_now(),
        }

    # ------------------------------------------------------------------
    # OpenAPI 3.1 schema for ChatGPT Actions
    # ------------------------------------------------------------------

    @app.get("/gpt/openapi.json", include_in_schema=False)
    def gpt_openapi_schema() -> JSONResponse:
        """Return a ChatGPT-Actions-compatible OpenAPI 3.1.0 spec.

        Includes only the /gpt/* bridge endpoints with X-API-Key auth.
        The server URL is read from the NGROK_URL env var (or falls
        back to the request's own origin).
        """
        server_url = os.environ.get(
            "NGROK_URL", "https://zena-mistrustful-gloria.ngrok-free.dev"
        ).rstrip("/")

        spec: Dict[str, Any] = {
            "openapi": "3.1.0",
            "info": {
                "title": "AL-01 GPT Bridge",
                "description": (
                    "Read AL-01's live organism state as natural language "
                    "and inject controlled stimuli via a rate-limited bridge."
                ),
                "version": VERSION,
            },
            "servers": [{"url": server_url}],
            "components": {
                "schemas": {},
                "securitySchemes": {
                    "apiKey": {
                        "type": "apiKey",
                        "name": "X-API-Key",
                        "in": "header",
                    }
                }
            },
            "security": [{"apiKey": []}],
            "paths": {
                "/gpt/narrate": {
                    "get": {
                        "operationId": "narrate",
                        "summary": "Read AL-01 state as natural-language prose",
                        "description": (
                            "Returns the organism's current energy, fitness, "
                            "awareness, genome traits, population stats, and "
                            "autonomy status as human-readable text."
                        ),
                        "responses": {
                            "200": {
                                "description": "Current AL-01 state narration",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "prose": {
                                                    "type": "string",
                                                    "description": "Natural-language description of AL-01's state",
                                                },
                                                "raw": {
                                                    "type": "object",
                                                    "description": "Structured state data the prose was derived from",
                                                },
                                                "timestamp": {
                                                    "type": "string",
                                                    "format": "date-time",
                                                },
                                            },
                                        }
                                    }
                                },
                            }
                        },
                    }
                },
                "/gpt/stimulus": {
                    "post": {
                        "operationId": "stimulus",
                        "summary": "Inject a stimulus into AL-01",
                        "description": (
                            "Queue a rate-limited stimulus for the next evolution "
                            "cycle. Does NOT trigger an immediate cycle — evolution "
                            "timing is preserved. Max 6 per minute, 280 chars."
                        ),
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["text"],
                                        "properties": {
                                            "text": {
                                                "type": "string",
                                                "maxLength": 280,
                                                "description": "Stimulus event string (e.g. 'environmental_pressure')",
                                            }
                                        },
                                    }
                                }
                            },
                        },
                        "responses": {
                            "200": {
                                "description": "Stimulus accepted or rejected",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "status": {
                                                    "type": "string",
                                                    "enum": ["accepted", "rejected"],
                                                },
                                                "injection_number": {"type": "integer"},
                                                "queued_stimuli": {"type": "integer"},
                                                "reason": {"type": "string"},
                                                "message": {"type": "string"},
                                                "timestamp": {
                                                    "type": "string",
                                                    "format": "date-time",
                                                },
                                            },
                                        }
                                    }
                                },
                            }
                        },
                    }
                },
                "/gpt/status": {
                    "get": {
                        "operationId": "bridgeStatus",
                        "summary": "Check GPT bridge rate limits and statistics",
                        "responses": {
                            "200": {
                                "description": "Bridge status and stats",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "stimulus_enabled": {
                                                    "type": "boolean",
                                                    "description": "Whether GPT stimulus injection is currently enabled",
                                                },
                                                "rate_limit": {
                                                    "type": "integer",
                                                    "description": "Max stimuli allowed per rate window",
                                                },
                                                "rate_window_seconds": {
                                                    "type": "number",
                                                    "description": "Rate limit window size in seconds",
                                                },
                                                "injections_in_window": {
                                                    "type": "integer",
                                                    "description": "Number of injections in the current window",
                                                },
                                                "total_injections": {
                                                    "type": "integer",
                                                    "description": "Total stimuli injected since boot",
                                                },
                                                "total_rejections": {
                                                    "type": "integer",
                                                    "description": "Total stimuli rejected since boot",
                                                },
                                                "max_stimulus_length": {
                                                    "type": "integer",
                                                    "description": "Maximum allowed stimulus text length",
                                                },
                                                "log_entries": {
                                                    "type": "integer",
                                                    "description": "Number of entries in the injection audit log",
                                                },
                                                "timestamp": {
                                                    "type": "string",
                                                    "format": "date-time",
                                                },
                                            },
                                        }
                                    }
                                },
                            }
                        },
                    }
                },
            },
        }
        return JSONResponse(content=spec)

    return app


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
