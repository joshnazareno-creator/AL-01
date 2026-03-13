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
# Visual organism dashboard — static HTML served at /visual
# ---------------------------------------------------------------------------

_VISUAL_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AL-01 — Visual Organism</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:#0d1117;color:#c9d1d9;font-family:'Courier New',Courier,monospace;overflow:hidden}
  #header{position:fixed;top:0;left:0;right:0;z-index:10;background:rgba(13,17,23,0.92);
          padding:.6rem 1.2rem;display:flex;align-items:center;gap:1rem;
          border-bottom:1px solid #21262d;backdrop-filter:blur(8px)}
  #header h1{color:#58a6ff;font-size:1rem;white-space:nowrap}
  #header .meta{color:#8b949e;font-size:.7rem;display:flex;gap:1rem}
  #header .meta span{white-space:nowrap}
  #shock-banner{display:none;position:fixed;top:42px;left:0;right:0;z-index:9;
                background:rgba(248,81,73,0.15);color:#f85149;text-align:center;
                font-size:.75rem;padding:.3rem;border-bottom:1px solid #f8514950}
  canvas{display:block;width:100vw;height:100vh}
  #tooltip{display:none;position:fixed;z-index:20;background:#161b22ee;
           border:1px solid #30363d;border-radius:10px;padding:.7rem .9rem;
           font-size:.75rem;line-height:1.5;max-width:280px;pointer-events:none;
           box-shadow:0 6px 24px rgba(0,0,0,0.5);transition:opacity .15s ease}
  #tooltip .tt-id{color:#58a6ff;font-weight:bold;font-size:.8rem}
  #tooltip .tt-nick{color:#e3b341;font-size:.7rem}
  #tooltip .tt-strategy{color:#d2a8ff}
  #tooltip .tt-fitness{color:#3fb950}
  #tooltip .tt-energy{color:#f0883e}
  #tooltip .tt-evo{color:#a5d6ff}
  #tooltip .tt-row{display:flex;justify-content:space-between;gap:1rem}
  #tooltip .tt-label{color:#8b949e}
  #tooltip .tt-swatch{display:inline-block;width:10px;height:10px;border-radius:50%;
                       vertical-align:middle;margin-right:4px}
  #legend{position:fixed;bottom:12px;right:12px;z-index:10;background:rgba(22,27,34,0.9);
          border:1px solid #21262d;border-radius:8px;padding:.5rem .8rem;font-size:.65rem;
          color:#8b949e;line-height:1.7;backdrop-filter:blur(8px)}
  #legend b{color:#c9d1d9}
  #back-link{position:fixed;bottom:12px;left:12px;z-index:10;color:#58a6ff;
             font-size:.7rem;text-decoration:none}
  #back-link:hover{text-decoration:underline}
  #pool-bar{position:fixed;bottom:42px;left:12px;z-index:10;width:120px;height:6px;
            background:#21262d;border-radius:3px;overflow:hidden}
  #pool-fill{height:100%;border-radius:3px;transition:width .6s ease,background .6s ease}
  #pool-label{position:fixed;bottom:50px;left:12px;z-index:10;font-size:.6rem;color:#8b949e}
</style>
</head>
<body>
<div id="header">
  <h1>AL-01 Visual</h1>
  <div class="meta">
    <span id="pop-count">—</span>
    <span id="avg-fitness">—</span>
    <span id="pool-stat">—</span>
    <span id="update-ts">—</span>
  </div>
</div>
<div id="shock-banner">⚡ SHOCK EVENT ACTIVE — resilience favoured</div>
<canvas id="canvas"></canvas>
<div id="tooltip"></div>
<div id="legend">
  <b>Sphere Size</b> = Fitness &nbsp; <b>Pulse</b> = Energy<br>
  <b>Hue</b> = <span style="color:#ff6b6b">Adaptability</span> &nbsp;
  <b>Bright</b> = <span style="color:#6bff6b">Efficiency</span> &nbsp;
  <b>Rim</b> = <span style="color:#6b6bff">Resilience</span><br>
  <b>Particles</b> = Perception &nbsp; <b>Pattern</b> = Creativity<br>
  <b>Vibrate</b> = High Energy &nbsp; <b>&#x1F480;</b> = Dormant<br>
  <b>&#x1F451;</b> = Top 3 Fitness &nbsp; <b>&#x2B21;</b> = Parent AL-01<br>
  <b>Trail</b> = Movement Path &nbsp; <b>Heartbeat</b> = Alive pulse<br>
  <b>Green Field</b> = Resource Rich &nbsp; <b>Red Field</b> = Scarce
</div>
<a id="back-link" href="/">← Dashboard</a>
<div id="pool-label">🌍 Pool</div>
<div id="pool-bar"><div id="pool-fill"></div></div>

<script>
(function(){
  /* ================= CONSTANTS ================= */
  const POLL_MS = 3000;
  const BASE_R = 10;
  const SCALE_R = 40;
  const PADDING = 60;
  const GLOW_THRESHOLD = 0.5;
  const PULSE_INTENSITY = 0.06;
  const FLICKER_THRESHOLD = 0.70;
  const EVO_RING_INTERVAL = 250;
  const MAX_EVO_RINGS = 3;
  const LEADER_COUNT = 3;
  const HOVER_REPULSE_RADIUS = 120;
  const HOVER_REPULSE_FORCE = 30;
  const AURA_PARTICLE_COUNT = 60;
  const TRAIL_MAX = 8;
  const HEARTBEAT_PERIOD = 1.2;
  const MARBLE_SWIRL_SPEED = 0.25;
  const MARBLE_PARTICLE_MAX = 24;

  /* --- Spatial ecosystem constants --- */
  const DRIFT_SPEED = 18;           // base px/s organism movement
  const REPULSE_RADIUS_MULT = 3.5;  // repulsion starts at 3.5× organism radius
  const REPULSE_STRENGTH = 110;     // repulsion force (stronger to prevent clumping at walls)
  const RESOURCE_ATTRACT = 12;      // attraction toward resource-rich zones
  const WANDER_STRENGTH = 8;        // random wander force
  const VELOCITY_DAMPING = 0.92;    // friction per frame
  const MOVE_TRAIL_LEN = 30;        // movement trail point count
  const ENV_PARTICLE_COUNT = 100;   // floating env resource particles
  const RESOURCE_FIELD_RES = 8;     // grid cell size for field vis (lower=finer)

  /* --- Mobile-aware boundary constants (computed on resize) --- */
  let BOUNDS_MARGIN = 60;           // soft wall margin (recalculated per screen)
  let BOUNDS_FORCE = 90;            // wall avoidance force
  const BOUNDS_RAMP_ZONE = 2.0;     // avoidance ramps up over this × BOUNDS_MARGIN
  const CENTER_PULL = 4;            // gentle pull toward screen center
  const CENTER_PULL_EDGE = 0.35;    // fraction of half-dim where center pull activates

  /* --- Visual idle behaviour (visual-only, does not alter simulation) --- */
  const EXPLORE_WANDER_RATE = 0.6;  // rad/s wander angle change for explorers
  const EXPLORE_ARC_STRENGTH = 12;  // curved arc force for exploring organisms
  const REST_DRIFT_MULT = 0.08;     // resting organisms drift at 8% speed
  const REST_BREATHE_SPEED = 0.5;   // breathing oscillation Hz
  const REST_BREATHE_AMP = 0.03;    // breathing scale amplitude
  const REST_DIM_FACTOR = 0.78;     // dimming multiplier for resting organisms
  const EXPLORE_PAUSE_PROB = 0.002; // per-frame chance of pause-turn-continue
  const EXPLORE_PAUSE_DUR = 0.8;    // seconds to pause

  /* --- Visual render-size scaling (display only, no sim change) --- */
  const MIN_RENDER_RADIUS = 5;           // px — minimum bubble render size
  const MAX_RENDER_RADIUS_PCT = 0.065;   // fraction of shorter screen dim (~6.5%)
  const RENDER_BASE = 5;                 // base render radius (px)
  const RENDER_SCALE = 22;               // max additional px from fitness
  const POP_SCALE_THRESHOLD = 25;        // start shrinking above this count
  const POP_SCALE_MIN = 0.55;            // smallest pop-based multiplier
  const CROWDING_RADIUS_MULT = 2.5;      // distance for crowding check (× renderR)
  const CROWDING_SHRINK_MIN = 0.7;       // max shrink from crowding
  const CROWDING_ALPHA_MIN = 0.55;       // max alpha reduction from crowding

  /* ================= DOM REFS ================= */
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const tooltip = document.getElementById('tooltip');
  let organisms = [];
  let circles = [];
  let hoveredIdx = -1;
  let mouseX = -1, mouseY = -1;
  let dpr = window.devicePixelRatio || 1;
  let poolFraction = 1.0;
  let isScarcity = false;
  let shockActive = false;
  let leaderIds = new Set();
  let time = 0;
  let envParticles = [];   // floating resource particles in the environment
  let resourceField = [];  // 2D grid of resource concentration values

  /* ================= AURA PARTICLES ================= */
  let auraParticles = [];
  function initAura(){
    auraParticles = [];
    const W = window.innerWidth, H = window.innerHeight;
    for(let i = 0; i < AURA_PARTICLE_COUNT; i++){
      auraParticles.push({
        x: Math.random() * W,
        y: Math.random() * H,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.2 - 0.1,
        size: 1 + Math.random() * 2.5,
        alpha: 0.1 + Math.random() * 0.2,
        phase: Math.random() * Math.PI * 2
      });
    }
  }

  /* ================= RESIZE ================= */
  function resize(){
    const w = window.innerWidth, h = window.innerHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Mobile-aware boundary: bigger margin on small screens
    const shortDim = Math.min(w, h);
    if(shortDim < 500){
      BOUNDS_MARGIN = Math.max(50, shortDim * 0.14);
      BOUNDS_FORCE = 120;
    } else if(shortDim < 900){
      BOUNDS_MARGIN = Math.max(55, shortDim * 0.10);
      BOUNDS_FORCE = 100;
    } else {
      BOUNDS_MARGIN = 60;
      BOUNDS_FORCE = 90;
    }

    syncOrganisms();
    if(!auraParticles.length) initAura();
    initEnvParticles();
  }
  window.addEventListener('resize', resize);

  /* ================= COLOUR HELPERS ================= */
  function traitColor(t){
    const r = Math.round(Math.min(1, t.adaptability || 0) * 255);
    const g = Math.round(Math.min(1, t.energy_efficiency || 0) * 255);
    const b = Math.round(Math.min(1, t.resilience || 0) * 255);
    return {r, g, b, str: 'rgb('+r+','+g+','+b+')'};
  }
  function dominantTrait(t){
    const vals = [
      {name:'adaptability', v: t.adaptability||0},
      {name:'energy_efficiency', v: t.energy_efficiency||0},
      {name:'resilience', v: t.resilience||0},
      {name:'perception', v: t.perception||0},
      {name:'creativity', v: t.creativity||0}
    ];
    vals.sort((a,b)=>b.v - a.v);
    return vals[0].name;
  }

  /* ================= SPATIAL ECOSYSTEM ================= */
  function initEnvParticles(){
    const W = window.innerWidth, H = window.innerHeight;
    envParticles = [];
    for(let i = 0; i < ENV_PARTICLE_COUNT; i++){
      envParticles.push({
        x: Math.random() * W,
        y: Math.random() * H,
        vx: (Math.random()-0.5) * 6,
        vy: (Math.random()-0.5) * 6,
        size: 1.5 + Math.random() * 3,
        life: 0.5 + Math.random() * 0.5,
        phase: Math.random() * Math.PI * 2
      });
    }
  }

  // Build / update resource concentration field
  function updateResourceField(W, H){
    const cols = Math.ceil(W / RESOURCE_FIELD_RES);
    const rows = Math.ceil(H / RESOURCE_FIELD_RES);
    // Simple procedural field: higher near centre, perturbed by time + scarcity
    resourceField = [];
    const cx0 = W / 2, cy0 = H / 2;
    const maxDist = Math.sqrt(cx0*cx0 + cy0*cy0);
    for(let r = 0; r < rows; r++){
      const row = [];
      for(let c = 0; c < cols; c++){
        const px = c * RESOURCE_FIELD_RES + RESOURCE_FIELD_RES/2;
        const py = r * RESOURCE_FIELD_RES + RESOURCE_FIELD_RES/2;
        const dx = px - cx0, dy = py - cy0;
        const dist = Math.sqrt(dx*dx + dy*dy) / maxDist;
        // Base: radial falloff from centre
        let val = (1 - dist * 0.7) * poolFraction;
        // Perlin-ish noise via sin waves
        val += 0.15 * Math.sin(px*0.008 + time*0.15) * Math.cos(py*0.01 + time*0.12);
        val += 0.1 * Math.sin(px*0.015 - time*0.08) * Math.sin(py*0.012 + time*0.1);
        // Scarcity drains edges more
        if(isScarcity) val *= 0.4 + 0.6 * (1 - dist);
        row.push(Math.max(0, Math.min(1, val)));
      }
      resourceField.push(row);
    }
    return {cols, rows};
  }

  // Sample resource field at world position
  function sampleResource(px, py){
    if(!resourceField.length) return 0.5;
    const col = Math.floor(px / RESOURCE_FIELD_RES);
    const row = Math.floor(py / RESOURCE_FIELD_RES);
    if(row < 0 || row >= resourceField.length) return 0;
    if(col < 0 || col >= resourceField[0].length) return 0;
    return resourceField[row][col];
  }

  // Compute resource gradient at position (which direction has more resources)
  function resourceGradient(px, py){
    const step = RESOURCE_FIELD_RES * 2;
    const vl = sampleResource(px - step, py);
    const vr = sampleResource(px + step, py);
    const vu = sampleResource(px, py - step);
    const vd = sampleResource(px, py + step);
    return { gx: vr - vl, gy: vd - vu };
  }

  /* ================= VISUAL SCALING ================= */
  // Compute display-only render radius — does NOT affect simulation/physics.
  // Uses sqrt-compressed fitness, population scaling, and screen-size capping.
  function visualRadius(fitness, popCount){
    // 1. Sqrt-compressed fitness curve (0→0, 0.25→0.5, 1→1)
    const f = Math.sqrt(Math.max(0, Math.min(1, fitness || 0)));
    // 2. Base + scaled
    let r = RENDER_BASE + f * RENDER_SCALE;
    // 3. Population-aware shrink
    if(popCount > POP_SCALE_THRESHOLD){
      const excess = (popCount - POP_SCALE_THRESHOLD) / POP_SCALE_THRESHOLD;
      r *= Math.max(POP_SCALE_MIN, 1 - excess * 0.35);
    }
    // 4. Screen-size hard cap (mobile-first)
    const maxR = Math.min(window.innerWidth, window.innerHeight) * MAX_RENDER_RADIUS_PCT;
    r = Math.min(r, maxR);
    // 5. Floor
    return Math.max(MIN_RENDER_RADIUS, r);
  }

  // Per-frame crowding analysis — returns {shrink, alpha} per circle index
  let crowdingFactors = [];
  function computeCrowding(){
    crowdingFactors.length = circles.length;
    for(let i = 0; i < circles.length; i++){
      let neighbors = 0;
      let overlapSum = 0;
      const ci = circles[i];
      const rr = ci.renderR || ci.r;
      const threshold = rr * CROWDING_RADIUS_MULT;
      for(let j = 0; j < circles.length; j++){
        if(j === i) continue;
        const cj = circles[j];
        const dx = ci.x - cj.x;
        const dy = ci.y - cj.y;
        const dist = Math.sqrt(dx*dx + dy*dy);
        const combined = rr + (cj.renderR || cj.r);
        if(dist < combined * 1.6){
          neighbors++;
          overlapSum += 1 - dist / (combined * 1.6);
        }
      }
      // More neighbors + more overlap → more shrink
      const density = Math.min(1, overlapSum * 0.4);
      crowdingFactors[i] = {
        shrink: 1 - density * (1 - CROWDING_SHRINK_MIN),
        alpha:  1 - density * (1 - CROWDING_ALPHA_MIN)
      };
    }
  }

  // Sync organisms array to circles (spatial placement, preserve positions)
  function syncOrganisms(){
    const n = organisms.length;
    if(!n){ circles.length = 0; return; }
    const W = window.innerWidth;
    const H = window.innerHeight;
    const headerH = 50;

    // Determine leaders
    const sorted = organisms.slice().sort((a,b) => (b.fitness||0) - (a.fitness||0));
    leaderIds = new Set(sorted.slice(0, Math.min(LEADER_COUNT, n)).map(o => o.id));

    // Build id→circle map for existing circles
    const existing = {};
    for(let i = 0; i < circles.length; i++){
      if(circles[i] && circles[i].data) existing[circles[i].data.id] = circles[i];
    }

    const newCircles = [];
    for(let i = 0; i < n; i++){
      const o = organisms[i];
      const r = BASE_R + (o.fitness || 0) * SCALE_R;          // physics radius (unchanged)
      const vr = visualRadius(o.fitness, n);                   // display radius
      const c = traitColor(o.traits || {});
      const prev = existing[o.id];
      if(prev){
        // Update data but keep spatial position
        prev.targetR = r;
        prev.targetRenderR = vr;
        prev.color = c;
        prev.data = o;
        // Update visual idle state from real data
        prev.visualState = classifyVisualState(o);
        newCircles.push(prev);
      } else {
        // New organism: spawn biased toward interior (Gaussian-ish)
        const cx0 = W / 2, cy0 = (headerH + H) / 2;
        const spreadX = (W - BOUNDS_MARGIN * 4) * 0.35;
        const spreadY = (H - headerH - BOUNDS_MARGIN * 4) * 0.35;
        const spawnX = cx0 + (Math.random() + Math.random() - 1) * spreadX;
        const spawnY = cy0 + (Math.random() + Math.random() - 1) * spreadY;
        newCircles.push({
          x: Math.max(BOUNDS_MARGIN, Math.min(W - BOUNDS_MARGIN, spawnX)),
          y: Math.max(headerH + BOUNDS_MARGIN, Math.min(H - BOUNDS_MARGIN, spawnY)),
          vx: (Math.random()-0.5) * 6, vy: (Math.random()-0.5) * 6,
          r: r, targetR: r,
          renderR: vr, targetRenderR: vr,
          color: c, data: o,
          shimmerPhase: Math.random() * Math.PI * 2,
          crownAngle: Math.random() * Math.PI * 2,
          trail: [],
          heartbeatPhase: Math.random() * Math.PI * 2,
          wanderAngle: Math.random() * Math.PI * 2,
          moveTrail: [],
          visualState: classifyVisualState(o),
          pauseTimer: 0,
          wanderCurvature: (Math.random() - 0.5) * 0.5
        });
      }
    }
    circles = newCircles;
  }

  /* ================= VISUAL STATE CLASSIFICATION (visual-only) ================= */
  // Derives a display-only behaviour mode from real organism data.
  // Does NOT alter organism state, fitness, energy, or any simulation value.
  function classifyVisualState(o){
    if(o.state === 'dormant') return 'sleeping';
    const energy = o.energy || 0;
    const fitness = o.fitness || 0;
    const stagnation = o.stagnation || 0;
    // Low energy or high stagnation → resting
    if(energy < 0.25 || stagnation > 0.6) return 'resting';
    // Moderate energy, moderate fitness → exploring
    if(energy > 0.35 && fitness > 0.2) return 'exploring';
    // Default fallback
    return energy > 0.15 ? 'exploring' : 'resting';
  }

  // Smooth cubic wall-avoidance ramp: returns 0 far from wall, ramps to 1 at wall
  function wallRamp(pos, wallPos, margin){
    const dist = Math.abs(pos - wallPos);
    if(dist >= margin) return 0;
    const t = 1 - dist / margin;  // 0 at edge of zone, 1 at wall
    return t * t * (3 - 2 * t);   // smoothstep — no jitter, organic curve
  }

  /* ================= PHYSICS STEP ================= */
  function physicsStep(dt){
    const W = window.innerWidth;
    const H = window.innerHeight;
    const headerH = 50;
    const margin = BOUNDS_MARGIN;
    const rampZone = margin * BOUNDS_RAMP_ZONE; // avoidance activates further out
    const minY = headerH + margin;
    const maxY = H - margin;
    const minX = margin;
    const maxX = W - margin;
    const midX = W / 2, midY = (headerH + H) / 2;
    const halfW = (W - margin * 2) / 2;
    const halfH = (H - headerH - margin * 2) / 2;

    for(let i = 0; i < circles.length; i++){
      const c = circles[i];
      const o = c.data;
      const energy = o.energy || 0;
      const isDormant = o.state === 'dormant';
      const vs = c.visualState || 'exploring';
      const isResting = vs === 'resting' || vs === 'sleeping';

      // Speed varies by visual state
      let speed;
      if(isDormant) speed = DRIFT_SPEED * 0.08;
      else if(isResting) speed = DRIFT_SPEED * REST_DRIFT_MULT;
      else speed = DRIFT_SPEED * (0.4 + energy * 0.8);

      let fx = 0, fy = 0;

      // --- Pause-turn-continue for explorers ---
      if(!c.pauseTimer) c.pauseTimer = 0;
      if(c.pauseTimer > 0){
        c.pauseTimer -= dt;
        // During pause: only boundary + repulsion forces, no wander
      } else {
        // 1. Wander — organic curved steering
        if(!isResting){
          // Smooth curvature-based wander (arcs, not jitter)
          if(!c.wanderCurvature) c.wanderCurvature = (Math.random() - 0.5) * 0.5;
          // Slowly evolve curvature for natural-looking arcs
          c.wanderCurvature += (Math.random() - 0.5) * 0.3 * dt;
          c.wanderCurvature = Math.max(-1.2, Math.min(1.2, c.wanderCurvature));
          c.wanderAngle = (c.wanderAngle || 0) + c.wanderCurvature * EXPLORE_WANDER_RATE * dt;
          fx += Math.cos(c.wanderAngle) * EXPLORE_ARC_STRENGTH;
          fy += Math.sin(c.wanderAngle) * EXPLORE_ARC_STRENGTH;
          // Occasional pause-turn-continue
          if(Math.random() < EXPLORE_PAUSE_PROB){
            c.pauseTimer = EXPLORE_PAUSE_DUR * (0.5 + Math.random());
            c.wanderCurvature = (Math.random() - 0.5) * 1.0; // new direction after pause
          }
        } else {
          // Resting: very gentle float
          c.wanderAngle = (c.wanderAngle || 0) + (Math.random()-0.5) * 0.2 * dt;
          fx += Math.cos(c.wanderAngle) * WANDER_STRENGTH * 0.15;
          fy += Math.sin(c.wanderAngle) * WANDER_STRENGTH * 0.15;
        }
      }

      // 2. Resource attraction — move toward higher resource concentration
      if(!isDormant && !isResting){
        const grad = resourceGradient(c.x, c.y);
        fx += grad.gx * RESOURCE_ATTRACT * (1 + (1 - energy) * 2);
        fy += grad.gy * RESOURCE_ATTRACT * (1 + (1 - energy) * 2);
      }

      // 3. Organism repulsion — avoid overlap
      for(let j = 0; j < circles.length; j++){
        if(j === i) continue;
        const other = circles[j];
        const dx = c.x - other.x;
        const dy = c.y - other.y;
        const dist = Math.sqrt(dx*dx + dy*dy) || 1;
        const minDist = (c.r + other.r) * REPULSE_RADIUS_MULT;
        if(dist < minDist){
          const overlap = 1 - dist / minDist;
          const force = overlap * overlap * REPULSE_STRENGTH; // quadratic for smoother feel
          fx += (dx / dist) * force;
          fy += (dy / dist) * force;
        }
      }

      // 4. Hover repulsion
      if(hoveredIdx >= 0 && i !== hoveredIdx){
        const hc = circles[hoveredIdx];
        const dx = c.x - hc.x;
        const dy = c.y - hc.y;
        const dist = Math.sqrt(dx*dx+dy*dy) || 1;
        if(dist < HOVER_REPULSE_RADIUS){
          const force = (1 - dist/HOVER_REPULSE_RADIUS) * HOVER_REPULSE_FORCE * 2;
          fx += (dx/dist) * force;
          fy += (dy/dist) * force;
        }
      }

      // 5. Smooth boundary avoidance — cubic ramp (organic, no bounce)
      const rL = wallRamp(c.x, 0, rampZone);
      const rR = wallRamp(c.x, W, rampZone);
      const rT = wallRamp(c.y, headerH, rampZone);
      const rB = wallRamp(c.y, H, rampZone);
      fx += rL * BOUNDS_FORCE;           // push right when near left wall
      fx -= rR * BOUNDS_FORCE;           // push left when near right wall
      fy += rT * BOUNDS_FORCE;           // push down when near top
      fy -= rB * BOUNDS_FORCE;           // push up when near bottom

      // Also steer wanderAngle away from nearby walls (prevents aiming at wall)
      if(rL > 0.1 || rR > 0.1 || rT > 0.1 || rB > 0.1){
        const awayAngle = Math.atan2(midY - c.y, midX - c.x);
        const wallStrength = Math.max(rL, rR, rT, rB);
        c.wanderAngle = c.wanderAngle + (awayAngle - c.wanderAngle) * wallStrength * 0.3 * dt * 5;
      }

      // 6. Gentle center pull for organisms far from centre (prevents edge crowding)
      const offX = (c.x - midX) / halfW;  // -1..1
      const offY = (c.y - midY) / halfH;
      const edgeDist = Math.max(Math.abs(offX), Math.abs(offY));
      if(edgeDist > CENTER_PULL_EDGE){
        const pull = (edgeDist - CENTER_PULL_EDGE) / (1 - CENTER_PULL_EDGE);
        fx += (midX - c.x) / halfW * CENTER_PULL * pull * pull;
        fy += (midY - c.y) / halfH * CENTER_PULL * pull * pull;
      }

      // Apply forces to velocity
      c.vx = (c.vx || 0) + fx * dt;
      c.vy = (c.vy || 0) + fy * dt;

      // Clamp velocity
      const vel = Math.sqrt(c.vx*c.vx + c.vy*c.vy);
      const maxVel = speed * 3;
      if(vel > maxVel){
        c.vx = c.vx / vel * maxVel;
        c.vy = c.vy / vel * maxVel;
      }

      // Damping (stronger for resting organisms — settle faster)
      const damp = isResting ? 0.88 : VELOCITY_DAMPING;
      c.vx *= damp;
      c.vy *= damp;

      // Integrate position
      c.x += c.vx * dt;
      c.y += c.vy * dt;

      // Hard clamp (safety net — should rarely be hit with smooth forces)
      c.x = Math.max(margin * 0.3, Math.min(W - margin * 0.3, c.x));
      c.y = Math.max(headerH + 8, Math.min(H - 8, c.y));

      // Smooth radius (physics + render independently)
      c.r = lerp(c.r, c.targetR, 0.1);
      c.renderR = lerp(c.renderR || c.r, c.targetRenderR || c.targetR, 0.1);

      // Record movement trail
      if(!c.moveTrail) c.moveTrail = [];
      c.moveTrail.push({x: c.x, y: c.y, age: 0});
      if(c.moveTrail.length > MOVE_TRAIL_LEN) c.moveTrail.shift();
      for(let mt of c.moveTrail) mt.age += dt;
    }
  }

  /* ================= ENV PARTICLE PHYSICS ================= */
  function updateEnvParticles(dt, W, H){
    // Spawn replacements
    while(envParticles.length < ENV_PARTICLE_COUNT){
      envParticles.push({
        x: Math.random() * W, y: Math.random() * H,
        vx: (Math.random()-0.5)*6, vy: (Math.random()-0.5)*6,
        size: 1.5 + Math.random()*3,
        life: 0.5 + Math.random()*0.5,
        phase: Math.random() * Math.PI*2
      });
    }
    for(let i = envParticles.length - 1; i >= 0; i--){
      const p = envParticles[i];
      // Drift toward nearby organisms (attraction to life)
      let ax = 0, ay = 0;
      for(let c of circles){
        const dx = c.x - p.x, dy = c.y - p.y;
        const dist = Math.sqrt(dx*dx+dy*dy) || 1;
        if(dist < 150){
          const pull = (1 - dist/150) * 20 * (c.data.energy||0.5);
          ax += (dx/dist) * pull;
          ay += (dy/dist) * pull;
        }
        // Consumed: if particle very close to organism, fade it
        if(dist < c.r * 1.2){
          p.life -= dt * 3;
        }
      }
      p.vx += ax * dt;
      p.vy += ay * dt;
      p.vx *= 0.96; p.vy *= 0.96;
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      // Wrap
      if(p.x < 0) p.x = W; if(p.x > W) p.x = 0;
      if(p.y < 0) p.y = H; if(p.y > H) p.y = 0;
      p.life -= dt * 0.08;
      if(p.life <= 0){
        // Respawn — prefer resource-rich areas
        p.x = Math.random() * W; p.y = Math.random() * H;
        const res = sampleResource(p.x, p.y);
        if(res < 0.3 && Math.random() > res){
          // Try again in richer area (bias toward centre)
          p.x = W * 0.2 + Math.random() * W * 0.6;
          p.y = H * 0.2 + Math.random() * H * 0.6;
        }
        p.vx = (Math.random()-0.5)*6; p.vy = (Math.random()-0.5)*6;
        p.life = 0.5 + Math.random()*0.5;
        p.size = 1.5 + Math.random()*3;
      }
    }
  }

  /* ================= ANIMATION HELPERS ================= */
  function lerp(a, b, t){ return a + (b - a) * t; }

  /* Shimmer colour based on dominant trait */
  function shimmerColor(trait, phase){
    const t = Math.sin(phase) * 0.5 + 0.5;
    switch(trait){
      case 'adaptability':    return 'rgba(255,'+ Math.round(80+t*100) +',80,0.18)';
      case 'energy_efficiency': return 'rgba(80,'+Math.round(200+t*55)+',80,0.18)';
      case 'resilience':      return 'rgba(80,80,'+Math.round(200+t*55)+',0.18)';
      case 'perception':      return 'rgba('+Math.round(180+t*75)+',180,'+Math.round(220+t*35)+',0.15)';
      case 'creativity':      return 'rgba('+Math.round(220+t*35)+','+Math.round(160+t*60)+',255,0.15)';
      default:                return 'rgba(150,150,150,0.1)';
    }
  }

  /* ================= SEEDED RNG (from genome hash) ================= */
  function hashToSeed(id){
    let h = 0;
    const s = String(id);
    for(let i = 0; i < s.length; i++){
      h = ((h << 5) - h + s.charCodeAt(i)) | 0;
    }
    return Math.abs(h);
  }
  function seededRng(seed){
    let s = seed | 0 || 1;
    return function(){
      s ^= s << 13; s ^= s >> 17; s ^= s << 5;
      return ((s >>> 0) / 4294967296);
    };
  }

  /* ================= MARBLE PATTERN TYPES ================= */
  // Returns a function(ctx, cx, cy, r, time, rng, traits) that draws the pattern
  function getPatternDrawer(patternIdx){
    const drawers = [
      drawMarbleVeins,
      drawNebulaPattern,
      drawCrystalInclusions,
      drawGalaxyCluster,
      drawLiquidBands
    ];
    return drawers[patternIdx % drawers.length];
  }

  // Pattern 0: Swirling marble veins
  function drawMarbleVeins(ctx, cx, cy, r, t, rng, traits){
    const complexity = 3 + Math.floor((traits.creativity||0.5) * 6);
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI*2);
    ctx.clip();
    ctx.globalAlpha = 0.25;
    for(let v = 0; v < complexity; v++){
      const baseAngle = rng() * Math.PI * 2 + t * MARBLE_SWIRL_SPEED * (v%2===0?1:-1);
      const width = 0.5 + rng() * 2;
      const amp = r * (0.3 + rng() * 0.4);
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(255,255,255,' + (0.15 + rng()*0.2) + ')';
      ctx.lineWidth = width;
      for(let s = 0; s <= 20; s++){
        const frac = s / 20;
        const angle = baseAngle + frac * Math.PI * (1.5 + rng());
        const dist = frac * r * 0.9;
        const wobble = Math.sin(frac * Math.PI * (2+v) + t*0.5) * amp * 0.15;
        const px = cx + Math.cos(angle) * (dist + wobble);
        const py = cy + Math.sin(angle) * (dist + wobble);
        if(s===0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }
    ctx.restore();
  }

  // Pattern 1: Cloudy nebula
  function drawNebulaPattern(ctx, cx, cy, r, t, rng, traits){
    const density = 4 + Math.floor((traits.creativity||0.5) * 8);
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI*2);
    ctx.clip();
    for(let i = 0; i < density; i++){
      const angle = rng() * Math.PI * 2 + t * 0.15 * (i%2===0?1:-1);
      const dist = rng() * r * 0.7;
      const nx = cx + Math.cos(angle + t*0.1) * dist;
      const ny = cy + Math.sin(angle + t*0.1) * dist;
      const blobR = r * (0.2 + rng() * 0.35);
      const g = ctx.createRadialGradient(nx, ny, 0, nx, ny, blobR);
      const hue = (rng() * 60 - 30 + (traits.adaptability||0.5) * 360) % 360;
      g.addColorStop(0, 'hsla('+hue+',60%,70%,0.15)');
      g.addColorStop(1, 'hsla('+hue+',60%,40%,0)');
      ctx.beginPath();
      ctx.arc(nx, ny, blobR, 0, Math.PI*2);
      ctx.fillStyle = g;
      ctx.fill();
    }
    ctx.restore();
  }

  // Pattern 2: Crystal inclusions
  function drawCrystalInclusions(ctx, cx, cy, r, t, rng, traits){
    const count = 3 + Math.floor((traits.creativity||0.5) * 7);
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI*2);
    ctx.clip();
    ctx.globalAlpha = 0.3;
    for(let i = 0; i < count; i++){
      const angle = rng() * Math.PI * 2;
      const dist = rng() * r * 0.6;
      const px = cx + Math.cos(angle + t*0.08) * dist;
      const py = cy + Math.sin(angle + t*0.08) * dist;
      const sides = 3 + Math.floor(rng() * 4);
      const cSize = r * (0.08 + rng() * 0.15);
      const rot = rng() * Math.PI + t * 0.2;
      ctx.beginPath();
      for(let s = 0; s <= sides; s++){
        const a = rot + (Math.PI * 2 / sides) * s;
        const x = px + Math.cos(a) * cSize;
        const y = py + Math.sin(a) * cSize;
        if(s===0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fillStyle = 'rgba(255,255,255,' + (0.12 + rng()*0.1) + ')';
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,' + (0.2 + rng()*0.15) + ')';
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }
    ctx.restore();
  }

  // Pattern 3: Galaxy-like particle cluster
  function drawGalaxyCluster(ctx, cx, cy, r, t, rng, traits){
    const armCount = 2 + Math.floor((traits.creativity||0.5) * 3);
    const particles = 20 + Math.floor((traits.perception||0.5) * 40);
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI*2);
    ctx.clip();
    for(let arm = 0; arm < armCount; arm++){
      const armAngle = (Math.PI * 2 / armCount) * arm + t * MARBLE_SWIRL_SPEED * 0.5;
      for(let p = 0; p < Math.floor(particles/armCount); p++){
        const frac = (p + rng()) / (particles/armCount);
        const spiralAngle = armAngle + frac * Math.PI * 2.5;
        const dist = frac * r * 0.85;
        const spread = (rng()-0.5) * r * 0.15;
        const px = cx + Math.cos(spiralAngle) * dist + Math.cos(spiralAngle+1.57) * spread;
        const py = cy + Math.sin(spiralAngle) * dist + Math.sin(spiralAngle+1.57) * spread;
        const pSize = 0.5 + rng() * 1.5;
        const alpha = (1 - frac) * 0.4;
        ctx.beginPath();
        ctx.arc(px, py, pSize, 0, Math.PI*2);
        ctx.fillStyle = 'rgba(255,255,255,'+alpha+')';
        ctx.fill();
      }
    }
    ctx.restore();
  }

  // Pattern 4: Flowing liquid bands
  function drawLiquidBands(ctx, cx, cy, r, t, rng, traits){
    const bandCount = 3 + Math.floor((traits.creativity||0.5) * 5);
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI*2);
    ctx.clip();
    ctx.globalAlpha = 0.2;
    for(let b = 0; b < bandCount; b++){
      const yOff = (rng() - 0.5) * r * 1.6;
      const amp = r * (0.1 + rng() * 0.25);
      const freq = 1.5 + rng() * 2;
      const phase = rng() * Math.PI * 2 + t * MARBLE_SWIRL_SPEED;
      const width = r * (0.04 + rng() * 0.08);
      ctx.beginPath();
      for(let s = 0; s <= 30; s++){
        const frac = s / 30;
        const x = cx - r + frac * r * 2;
        const y = cy + yOff + Math.sin(frac * freq * Math.PI + phase) * amp;
        if(s===0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = 'rgba(255,255,255,' + (0.2 + rng()*0.15) + ')';
      ctx.lineWidth = width;
      ctx.lineCap = 'round';
      ctx.stroke();
    }
    ctx.restore();
  }

  /* ================= MARBLE INTERNAL PARTICLES ================= */
  // Per-organism floating internal particles (awareness / energy driven)
  function updateMarbleParticles(c, cx, cy, drawR, dt, traits){
    if(!c.marbleParticles) c.marbleParticles = [];
    const targetCount = Math.floor((traits.perception||0.5) * MARBLE_PARTICLE_MAX);
    // Spawn
    while(c.marbleParticles.length < targetCount){
      const angle = Math.random() * Math.PI * 2;
      const dist = Math.random() * drawR * 0.7;
      c.marbleParticles.push({
        angle: angle,
        dist: dist,
        speed: 0.2 + Math.random() * 0.5,
        size: 0.5 + Math.random() * 1.5,
        phase: Math.random() * Math.PI * 2,
        brightness: 0.3 + Math.random() * 0.7
      });
    }
    // Trim
    while(c.marbleParticles.length > targetCount && c.marbleParticles.length > 0){
      c.marbleParticles.pop();
    }
    // Update and draw
    for(let p of c.marbleParticles){
      p.angle += p.speed * dt;
      p.dist += Math.sin(p.phase + time) * 0.3;
      p.dist = Math.max(0, Math.min(drawR * 0.75, p.dist));
      const px = cx + Math.cos(p.angle) * p.dist;
      const py = cy + Math.sin(p.angle) * p.dist;
      const alpha = p.brightness * (0.3 + 0.2 * Math.sin(time * 2 + p.phase));
      ctx.beginPath();
      ctx.arc(px, py, p.size, 0, Math.PI*2);
      ctx.fillStyle = 'rgba(255,255,255,'+alpha+')';
      ctx.fill();
    }
  }

  /* ================= DRAW MARBLE ORGANISM ================= */
  function drawMarble(ctx, c, cx, cy, drawR, isDormant, time){
    const o = c.data;
    const traits = o.traits || {};
    const energy = o.energy || 0;
    const fitness = o.fitness || 0;

    // Deterministic seed from organism id
    const seed = hashToSeed(o.id || 'default');
    const rng = seededRng(seed);
    // Pick pattern type: determined by seed
    const patternIdx = seed % 5;

    // --- Trait-to-visual mapping ---
    // Adaptability → hue, Efficiency → brightness, Resilience → rim glow
    const hue = (traits.adaptability || 0.5) * 360;
    const brightness = 30 + (traits.energy_efficiency || 0.5) * 45;
    const saturation = 50 + (traits.creativity || 0.5) * 30;
    const rimStrength = (traits.resilience || 0.5);

    // === LAYER 0: Drop shadow ===
    if(!isDormant){
      ctx.save();
      ctx.beginPath();
      ctx.ellipse(cx + 2, cy + drawR * 0.9, drawR * 0.7, drawR * 0.18, 0, 0, Math.PI*2);
      ctx.fillStyle = 'rgba(0,0,0,0.25)';
      ctx.fill();
      ctx.restore();
    }

    // Clip to sphere
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, drawR, 0, Math.PI*2);
    ctx.clip();

    // === LAYER 1: Base color sphere (HSL from traits) ===
    if(isDormant){
      const bGrad = ctx.createRadialGradient(
        cx - drawR*0.3, cy - drawR*0.3, drawR*0.05,
        cx, cy, drawR
      );
      bGrad.addColorStop(0, 'hsl(0,0%,45%)');
      bGrad.addColorStop(0.6, 'hsl(0,0%,30%)');
      bGrad.addColorStop(1, 'hsl(0,0%,15%)');
      ctx.fillStyle = bGrad;
      ctx.fillRect(cx-drawR, cy-drawR, drawR*2, drawR*2);
    } else {
      const bGrad = ctx.createRadialGradient(
        cx - drawR*0.35, cy - drawR*0.35, drawR*0.05,
        cx + drawR*0.1, cy + drawR*0.1, drawR
      );
      bGrad.addColorStop(0, 'hsl('+hue+','+saturation+'%,'+(brightness+20)+'%)');
      bGrad.addColorStop(0.5, 'hsl('+hue+','+saturation+'%,'+brightness+'%)');
      bGrad.addColorStop(1, 'hsl('+hue+','+(saturation-10)+'%,'+(brightness-15)+'%)');
      ctx.fillStyle = bGrad;
      ctx.fillRect(cx-drawR, cy-drawR, drawR*2, drawR*2);
    }

    // === LAYER 2: Internal pattern (genome-determined) ===
    if(!isDormant){
      const drawer = getPatternDrawer(patternIdx);
      // Reset rng for consistency
      const patRng = seededRng(seed + 42);
      drawer(ctx, cx, cy, drawR, time, patRng, traits);
    }

    // === LAYER 3: Floating internal particles (perception-driven) ===
    if(!isDormant){
      updateMarbleParticles(c, cx, cy, drawR, 1/60, traits);
    }

    // === LAYER 4: Energy inner glow ===
    if(!isDormant && energy > 0.3){
      const glowIntensity = (energy - 0.3) / 0.7;
      const pulseGlow = glowIntensity * (0.8 + 0.2 * Math.sin(time * 2));
      const eGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, drawR * 0.8);
      eGrad.addColorStop(0, 'hsla('+((hue+30)%360)+',80%,70%,'+(pulseGlow*0.15)+')');
      eGrad.addColorStop(0.5, 'hsla('+hue+',60%,60%,'+(pulseGlow*0.06)+')');
      eGrad.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = eGrad;
      ctx.fillRect(cx-drawR, cy-drawR, drawR*2, drawR*2);
    }

    ctx.restore(); // End sphere clip

    // === LAYER 5: Rim lighting (resilience-driven) ===
    if(!isDormant){
      ctx.save();
      const rimGrad = ctx.createRadialGradient(cx, cy, drawR*0.85, cx, cy, drawR);
      const rimAlpha = 0.1 + rimStrength * 0.35;
      const pulseRim = rimAlpha * (0.85 + 0.15 * Math.sin(time * 1.5));
      rimGrad.addColorStop(0, 'rgba(0,0,0,0)');
      rimGrad.addColorStop(0.7, 'rgba(0,0,0,0)');
      rimGrad.addColorStop(1, 'hsla('+((hue+180)%360)+',50%,70%,'+pulseRim+')');
      ctx.beginPath();
      ctx.arc(cx, cy, drawR, 0, Math.PI*2);
      ctx.fillStyle = rimGrad;
      ctx.fill();
      ctx.restore();
    }

    // === LAYER 6: Glass sphere shading (curvature + depth) ===
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, drawR, 0, Math.PI*2);
    // Darkened edges for spherical depth
    const depthGrad = ctx.createRadialGradient(
      cx - drawR*0.2, cy - drawR*0.2, drawR*0.1,
      cx, cy, drawR
    );
    depthGrad.addColorStop(0, 'rgba(255,255,255,0.08)');
    depthGrad.addColorStop(0.6, 'rgba(0,0,0,0)');
    depthGrad.addColorStop(1, 'rgba(0,0,0,0.3)');
    ctx.fillStyle = depthGrad;
    ctx.fill();
    ctx.restore();

    // === LAYER 7: Primary specular highlight (top-left) ===
    ctx.save();
    const hlX = cx - drawR * 0.32;
    const hlY = cy - drawR * 0.32;
    const hlR = drawR * 0.45;
    const hlGrad = ctx.createRadialGradient(hlX, hlY, 0, hlX, hlY, hlR);
    hlGrad.addColorStop(0, 'rgba(255,255,255,0.55)');
    hlGrad.addColorStop(0.4, 'rgba(255,255,255,0.15)');
    hlGrad.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.beginPath();
    ctx.arc(cx, cy, drawR, 0, Math.PI*2);
    ctx.clip();
    ctx.beginPath();
    ctx.arc(hlX, hlY, hlR, 0, Math.PI*2);
    ctx.fillStyle = hlGrad;
    ctx.fill();
    ctx.restore();

    // === LAYER 8: Small secondary specular (glass reflection) ===
    ctx.save();
    const h2X = cx + drawR * 0.18;
    const h2Y = cy + drawR * 0.28;
    const h2R = drawR * 0.12;
    const h2Grad = ctx.createRadialGradient(h2X, h2Y, 0, h2X, h2Y, h2R);
    h2Grad.addColorStop(0, 'rgba(255,255,255,0.25)');
    h2Grad.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.beginPath();
    ctx.arc(cx, cy, drawR, 0, Math.PI*2);
    ctx.clip();
    ctx.beginPath();
    ctx.arc(h2X, h2Y, h2R, 0, Math.PI*2);
    ctx.fillStyle = h2Grad;
    ctx.fill();
    ctx.restore();

    // === LAYER 9: Glass edge highlight ring ===
    if(!isDormant){
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, drawR - 0.5, 0, Math.PI*2);
      ctx.strokeStyle = 'rgba(255,255,255,0.12)';
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.restore();
    }
  }

  /* ================= DRAW AURA FIELD + RESOURCE FIELD ================= */
  function drawAura(W, H, dt){
    // Background tint based on pool health
    const health = poolFraction;
    const bgR = Math.round(lerp(30, 8, health));
    const bgG = Math.round(lerp(10, 14, health));
    const bgB = Math.round(lerp(10, 20, health));
    ctx.fillStyle = 'rgb('+bgR+','+bgG+','+bgB+')';
    ctx.fillRect(0, 0, W, H);

    // Resource concentration field — faint green/red tint showing abundance/scarcity
    const fieldInfo = updateResourceField(W, H);
    if(resourceField.length){
      for(let r = 0; r < fieldInfo.rows; r++){
        for(let c = 0; c < fieldInfo.cols; c++){
          const val = resourceField[r][c];
          if(val < 0.01) continue;
          const px = c * RESOURCE_FIELD_RES;
          const py = r * RESOURCE_FIELD_RES;
          // Green = abundant, Red = scarce
          if(val > 0.45){
            const intensity = (val - 0.45) / 0.55;
            ctx.fillStyle = 'rgba(63,185,80,' + (intensity * 0.06) + ')';
          } else {
            const intensity = (0.45 - val) / 0.45;
            ctx.fillStyle = 'rgba(248,81,73,' + (intensity * 0.04) + ')';
          }
          ctx.fillRect(px, py, RESOURCE_FIELD_RES, RESOURCE_FIELD_RES);
        }
      }
    }

    // Ambient glow in centre proportional to avg fitness
    if(organisms.length){
      const avgFit = organisms.reduce((s,o)=>s+(o.fitness||0),0)/organisms.length;
      const glowR = Math.min(200, W * 0.4);
      const grad = ctx.createRadialGradient(W/2, H/2, 0, W/2, H/2, glowR);
      const intensity = Math.min(0.08, avgFit * 0.1);
      grad.addColorStop(0, 'rgba(88,166,255,'+intensity+')');
      grad.addColorStop(1, 'rgba(88,166,255,0)');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, W, H);
    }

    // Scarcity warning vignette
    if(isScarcity){
      const vignette = ctx.createRadialGradient(W/2, H/2, W*0.2, W/2, H/2, W*0.7);
      const pulse = 0.06 + Math.sin(time * 1.5) * 0.03;
      vignette.addColorStop(0, 'rgba(0,0,0,0)');
      vignette.addColorStop(1, 'rgba(248,81,73,'+pulse+')');
      ctx.fillStyle = vignette;
      ctx.fillRect(0, 0, W, H);
    }

    // Background aura particles
    for(let p of auraParticles){
      p.x += p.vx; p.y += p.vy;
      if(p.x < 0) p.x = W;
      if(p.x > W) p.x = 0;
      if(p.y < 0) p.y = H;
      if(p.y > H) p.y = 0;
      const flicker = 0.5 + 0.5 * Math.sin(time * 2 + p.phase);
      const a = p.alpha * flicker * health;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.fillStyle = isScarcity
        ? 'rgba(248,81,73,' + (a * 0.6) + ')'
        : 'rgba(88,166,255,' + a + ')';
      ctx.fill();
    }

    // Environment resource particles — floating energy motes
    updateEnvParticles(dt, W, H);
    for(let p of envParticles){
      const res = sampleResource(p.x, p.y);
      const flicker = 0.5 + 0.5 * Math.sin(time * 3 + p.phase);
      const alpha = p.life * flicker * 0.5 * (0.5 + res * 0.5);
      if(alpha < 0.01) continue;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size * (0.6 + res * 0.4), 0, Math.PI*2);
      if(isScarcity){
        ctx.fillStyle = 'rgba(255,160,80,' + alpha + ')';
      } else {
        const g = Math.round(200 + res * 55);
        ctx.fillStyle = 'rgba(80,' + g + ',120,' + alpha + ')';
      }
      ctx.fill();
    }
  }

  /* ================= DRAW EVOLUTION RINGS ================= */
  function drawEvoRings(cx, cy, baseR, evoCount, dt){
    const ringCount = Math.min(MAX_EVO_RINGS, Math.floor(evoCount / EVO_RING_INTERVAL));
    if(ringCount <= 0) return;
    for(let i = 0; i < ringCount; i++){
      const ringR = baseR + 6 + i * 5;
      const rot = time * (0.3 + i * 0.15) * (i % 2 === 0 ? 1 : -1);
      const dashLen = Math.PI * 2 * ringR * 0.06;
      const gapLen = dashLen * 1.5;
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(rot);
      ctx.setLineDash([dashLen, gapLen]);
      ctx.beginPath();
      ctx.arc(0, 0, ringR, 0, Math.PI * 2);
      const alpha = 0.3 + i * 0.12;
      const colors = ['#58a6ff','#d2a8ff','#3fb950'];
      ctx.strokeStyle = colors[i % 3];
      ctx.globalAlpha = alpha;
      ctx.lineWidth = 1.2;
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    }
  }

  /* ================= DRAW LEADER CROWN ================= */
  function drawCrown(cx, cy, baseR, rank, crownAngle){
    const orbitR = baseR + 10;
    const particleCount = 5 - rank;  // 1st=5, 2nd=4, 3rd=3
    const colors = ['#ffd700','#c0c0c0','#cd7f32'];
    const col = colors[rank] || '#ffd700';

    // Golden ring
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, baseR + 3, 0, Math.PI * 2);
    ctx.strokeStyle = col;
    ctx.globalAlpha = 0.5 + Math.sin(time * 2) * 0.15;
    ctx.lineWidth = 1.8;
    ctx.stroke();
    ctx.restore();

    // Orbiting particles
    for(let i = 0; i < particleCount; i++){
      const angle = crownAngle + (Math.PI * 2 / particleCount) * i;
      const px = cx + Math.cos(angle) * orbitR;
      const py = cy + Math.sin(angle) * orbitR;
      ctx.save();
      ctx.beginPath();
      ctx.arc(px, py, 2, 0, Math.PI * 2);
      ctx.fillStyle = col;
      ctx.globalAlpha = 0.7 + Math.sin(time * 3 + i) * 0.3;
      ctx.fill();
      ctx.restore();
    }
  }

  /* ================= MAIN DRAW LOOP ================= */
  let lastTime = performance.now();

  function draw(now){
    const dt = (now - lastTime) / 1000;
    lastTime = now;
    time += dt;

    const W = window.innerWidth, H = window.innerHeight;

    // 6. Environmental Aura Field
    drawAura(W, H, dt);

    // Build leader rank map
    const sorted = circles.slice().filter(c=>c.data.alive!==false)
      .sort((a,b)=>(b.data.fitness||0)-(a.data.fitness||0));
    const leaderRank = {};
    for(let i = 0; i < Math.min(LEADER_COUNT, sorted.length); i++){
      leaderRank[sorted[i].data.id] = i;
    }

    // Physics simulation step
    physicsStep(dt);

    // Crowding analysis for visual density management
    computeCrowding();

    // Draw movement trails (before organisms so trails appear behind)
    for(let i = 0; i < circles.length; i++){
      const c = circles[i];
      const mt = c.moveTrail;
      if(!mt || mt.length < 2) continue;
      const isDorm = c.data.state === 'dormant';
      ctx.save();
      ctx.lineCap = 'round';
      for(let t = 1; t < mt.length; t++){
        const frac = t / mt.length;
        const alpha = frac * (isDorm ? 0.04 : 0.12);
        const width = frac * c.r * 0.3;
        ctx.beginPath();
        ctx.moveTo(mt[t-1].x, mt[t-1].y);
        ctx.lineTo(mt[t].x, mt[t].y);
        ctx.strokeStyle = 'rgba('+c.color.r+','+c.color.g+','+c.color.b+','+alpha+')';
        ctx.lineWidth = Math.max(0.5, width);
        ctx.stroke();
      }
      ctx.restore();
    }

    for(let i = 0; i < circles.length; i++){
      const c = circles[i];
      const o = c.data;
      const energy = o.energy || 0;
      const evoCount = o.evolution_count || 0;
      const opacity = Math.max(0.25, 1.0 - (o.stagnation || 0) * 0.6);

      // 1. Energy-Based Pulse Animation (uses visual renderR, not physics r)
      const vs = c.visualState || 'exploring';
      const isResting = vs === 'resting' || vs === 'sleeping';
      // Resting organisms pulse calmer and breathe gently
      const energyFactor = isResting ? 0.4 + energy * 0.6 : 0.8 + energy * 2.2;
      const breathe = isResting ? REST_BREATHE_AMP * Math.sin(time * REST_BREATHE_SPEED * Math.PI * 2) : 0;
      const pulse = 1.0 + (isResting ? PULSE_INTENSITY * 0.35 : PULSE_INTENSITY) * Math.sin(time * energyFactor) + breathe;
      const baseVisR = c.renderR || c.r;
      // Apply crowding shrink
      const crowd = crowdingFactors[i] || {shrink:1, alpha:1};
      let drawR = Math.max(MIN_RENDER_RADIUS, baseVisR * pulse * crowd.shrink);

      // 4. High-Energy Vibration — organisms buzz with excess energy
      let flickerOffsetX = 0, flickerOffsetY = 0, flickerAlpha = 1.0;
      if(energy > FLICKER_THRESHOLD){
        const intensity = (energy - FLICKER_THRESHOLD) / (1 - FLICKER_THRESHOLD);
        flickerOffsetX = (Math.random() - 0.5) * intensity * 4;
        flickerOffsetY = (Math.random() - 0.5) * intensity * 4;
        flickerAlpha = 1.0;  // full brightness — they're thriving
      }

      // Dormant organisms: grey desaturated, low alpha, peaceful breathing
      let isDormant = o.state === 'dormant';
      if(isDormant){
        flickerAlpha = 0.35 + 0.05 * Math.sin(time * 0.8);
      } else if(isResting){
        // Resting: slightly dimmed, subtle breathing opacity
        flickerAlpha = REST_DIM_FACTOR + 0.04 * Math.sin(time * REST_BREATHE_SPEED * Math.PI * 2 + (c.shimmerPhase || 0));
      }

      const cx = c.x + flickerOffsetX;
      const cy = c.y + flickerOffsetY;

      ctx.save();
      ctx.globalAlpha = opacity * flickerAlpha * crowd.alpha;

      // 2. Awareness Halo System
      const awareness = o.awareness || 0;
      if(awareness > GLOW_THRESHOLD){
        const haloStrength = (awareness - GLOW_THRESHOLD) / (1 - GLOW_THRESHOLD);
        const haloR = drawR + 8 + haloStrength * 18;
        const grad = ctx.createRadialGradient(cx, cy, drawR * 0.8, cx, cy, haloR);
        grad.addColorStop(0, 'rgba('+c.color.r+','+c.color.g+','+c.color.b+','+(0.25 * haloStrength)+')');
        grad.addColorStop(0.5, 'rgba('+c.color.r+','+c.color.g+','+c.color.b+','+(0.08 * haloStrength)+')');
        grad.addColorStop(1, 'rgba('+c.color.r+','+c.color.g+','+c.color.b+',0)');
        ctx.beginPath();
        ctx.arc(cx, cy, haloR, 0, Math.PI * 2);
        ctx.fillStyle = grad;
        ctx.fill();
      }

      // 3. Evolution Rings
      drawEvoRings(cx, cy, drawR, evoCount, dt);

      // 5. Trait Shimmer Layer
      const dom = dominantTrait(o.traits || {});
      c.shimmerPhase += dt * 1.8;
      const shimGrad = ctx.createRadialGradient(
        cx - drawR * 0.3, cy - drawR * 0.3, 0,
        cx, cy, drawR
      );
      shimGrad.addColorStop(0, shimmerColor(dom, c.shimmerPhase));
      shimGrad.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.beginPath();
      ctx.arc(cx, cy, drawR, 0, Math.PI * 2);
      ctx.fillStyle = shimGrad;
      ctx.fill();

      // Energy trail particles — emit from high-energy organisms
      if(energy > 0.5 && !isDormant){
        const emitChance = energy * 0.4;
        if(Math.random() < emitChance){
          c.trail.push({
            x: cx + (Math.random()-0.5)*drawR*0.6,
            y: cy + (Math.random()-0.5)*drawR*0.6,
            vx: (Math.random()-0.5)*12,
            vy: (Math.random()-0.5)*12 - 8,
            life: 1.0,
            size: 1 + Math.random()*2.5
          });
          if(c.trail.length > TRAIL_MAX) c.trail.shift();
        }
      }
      // Draw + update trail particles
      for(let p = c.trail.length - 1; p >= 0; p--){
        const pt = c.trail[p];
        pt.x += pt.vx * dt;
        pt.y += pt.vy * dt;
        pt.vy += 15 * dt;  // gravity
        pt.life -= dt * 1.5;
        if(pt.life <= 0){ c.trail.splice(p, 1); continue; }
        ctx.save();
        ctx.globalAlpha = pt.life * 0.6 * opacity;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, pt.size * pt.life, 0, Math.PI*2);
        ctx.fillStyle = 'rgb('+c.color.r+','+c.color.g+','+c.color.b+')';
        ctx.fill();
        ctx.restore();
      }

      // Heartbeat ring — rhythmic expanding ring
      if(!isDormant){
        c.heartbeatPhase = (c.heartbeatPhase || 0) + dt / HEARTBEAT_PERIOD * Math.PI * 2;
        const hbWave = Math.pow(Math.max(0, Math.sin(c.heartbeatPhase)), 4);
        if(hbWave > 0.01){
          ctx.save();
          ctx.beginPath();
          const hbR = drawR + hbWave * 14;
          ctx.arc(cx, cy, hbR, 0, Math.PI*2);
          ctx.strokeStyle = 'rgba('+c.color.r+','+c.color.g+','+c.color.b+','+(hbWave*0.3)+')';
          ctx.lineWidth = 1.5;
          ctx.stroke();
          ctx.restore();
        }
      }

      // Main organism body — living marble renderer
      drawMarble(ctx, c, cx, cy, drawR, isDormant, time);

      // Fitness glow border — brighter outline for fitter organisms
      if(!isDormant && (o.fitness||0) > 0.3){
        const fitGlow = Math.min(1, ((o.fitness||0) - 0.3) / 0.7);
        ctx.save();
        ctx.beginPath();
        ctx.arc(cx, cy, drawR + 1, 0, Math.PI*2);
        ctx.strokeStyle = 'rgba(63,185,80,'+(fitGlow*0.5)+')';
        ctx.lineWidth = 1.5 + fitGlow;
        ctx.stroke();
        ctx.restore();
      }

      // Dormant skull marker
      if(isDormant){
        ctx.save();
        ctx.font = Math.max(10, drawR*0.7)+'px serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.globalAlpha = 0.6;
        ctx.fillText('\uD83D\uDC80', cx, cy);  // 💀
        ctx.restore();
      }

      // Parent marker — hexagon outline
      if(o.is_parent){
        ctx.strokeStyle = '#58a6ff';
        ctx.lineWidth = 2;
        ctx.shadowColor = '#58a6ff';
        ctx.shadowBlur = 6;
        ctx.beginPath();
        const hr = drawR + 4;
        for(let s = 0; s < 6; s++){
          const a = Math.PI / 3 * s - Math.PI / 2;
          const px = cx + hr * Math.cos(a);
          const py = cy + hr * Math.sin(a);
          if(s === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.closePath();
        ctx.stroke();
        ctx.shadowBlur = 0;
      }

      // 8. Leader Crown System
      if(leaderRank[o.id] !== undefined){
        c.crownAngle += dt * (1.5 + leaderRank[o.id] * 0.3);
        drawCrown(cx, cy, drawR, leaderRank[o.id], c.crownAngle);
      }

      // Hover highlight
      if(i === hoveredIdx){
        ctx.strokeStyle = '#58a6ff';
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.8 + Math.sin(time * 4) * 0.2;
        ctx.beginPath();
        ctx.arc(cx, cy, drawR + 3, 0, Math.PI * 2);
        ctx.stroke();
      }

      ctx.restore();

      // Label for parent
      if(o.is_parent){
        ctx.fillStyle = '#8b949e';
        ctx.font = '9px Courier New';
        ctx.textAlign = 'center';
        ctx.fillText('AL-01', cx, cy + drawR + 14);
      }
    }

    requestAnimationFrame(draw);
  }

  /* ================= HOVER / TOOLTIP ================= */
  canvas.addEventListener('mousemove', function(e){
    mouseX = e.clientX; mouseY = e.clientY;
    hoveredIdx = -1;
    for(let i = circles.length - 1; i >= 0; i--){
      const c = circles[i];
      const dx = mouseX - c.x, dy = mouseY - c.y;
      if(dx*dx + dy*dy <= (c.r + 6) * (c.r + 6)){
        hoveredIdx = i;
        break;
      }
    }
    if(hoveredIdx >= 0){
      const o = circles[hoveredIdx].data;
      const t = o.traits || {};
      const col = traitColor(t);
      const evoRings = Math.min(MAX_EVO_RINGS, Math.floor((o.evolution_count||0) / EVO_RING_INTERVAL));
      let h = '<div class="tt-id">' + o.id + '</div>';
      if(o.nickname) h += '<div class="tt-nick">' + o.nickname + '</div>';
      h += '<div class="tt-strategy">Strategy: ' + (o.strategy || '—') + '</div>';
      h += '<div class="tt-fitness">Fitness: ' + (o.fitness || 0).toFixed(4) + '</div>';
      h += '<div class="tt-energy">Energy: ' + ((o.energy||0)*100).toFixed(1) + '%</div>';
      h += '<div class="tt-evo">Evolutions: ' + (o.evolution_count||0) + (evoRings > 0 ? ' ('+evoRings+' ring'+(evoRings>1?'s':'')+')' : '') + '</div>';
      h += '<hr style="border:0;border-top:1px solid #21262d;margin:.3rem 0">';
      h += '<div class="tt-row"><span class="tt-label">Adaptability</span><span style="color:#ff6b6b">' + (t.adaptability||0).toFixed(3) + '</span></div>';
      h += '<div class="tt-row"><span class="tt-label">Efficiency</span><span style="color:#6bff6b">' + (t.energy_efficiency||0).toFixed(3) + '</span></div>';
      h += '<div class="tt-row"><span class="tt-label">Resilience</span><span style="color:#6b6bff">' + (t.resilience||0).toFixed(3) + '</span></div>';
      h += '<div class="tt-row"><span class="tt-label">Perception</span><span>' + (t.perception||0).toFixed(3) + '</span></div>';
      h += '<div class="tt-row"><span class="tt-label">Creativity</span><span>' + (t.creativity||0).toFixed(3) + '</span></div>';
      h += '<hr style="border:0;border-top:1px solid #21262d;margin:.3rem 0">';
      h += '<div class="tt-row"><span class="tt-label">Awareness</span><span>' + (o.awareness||0).toFixed(3) + '</span></div>';
      h += '<div class="tt-row"><span class="tt-label">Stagnation</span><span>' + (o.stagnation||0).toFixed(3) + '</span></div>';
      if(leaderIds.has(o.id)) h += '<div style="margin-top:.3rem;color:#ffd700">👑 Top Fitness</div>';
      h += '<div style="margin-top:.3rem"><span class="tt-swatch" style="background:'+col.str+'"></span> rgb(' + col.r + ',' + col.g + ',' + col.b + ')</div>';
      tooltip.innerHTML = h;
      tooltip.style.display = 'block';
      tooltip.style.opacity = '1';
      let tx = e.clientX + 14, ty = e.clientY + 14;
      if(tx + 290 > window.innerWidth) tx = e.clientX - 290;
      if(ty + 300 > window.innerHeight) ty = e.clientY - 300;
      tooltip.style.left = tx + 'px';
      tooltip.style.top = ty + 'px';
    } else {
      tooltip.style.opacity = '0';
      setTimeout(()=>{ if(hoveredIdx < 0) tooltip.style.display='none'; }, 150);
    }
  });
  canvas.addEventListener('mouseleave', function(){
    hoveredIdx = -1; mouseX = -1; mouseY = -1;
    tooltip.style.opacity = '0';
    setTimeout(()=>tooltip.style.display='none', 150);
  });

  /* ================= DATA POLLING ================= */
  async function poll(){
    try{
      const resp = await fetch('/api/organisms');
      const data = await resp.json();
      organisms = data.organisms || [];
      poolFraction = data.pool_fraction != null ? data.pool_fraction : 1.0;
      isScarcity = !!data.is_scarcity;
      shockActive = !!data.shock_active;

      document.getElementById('pop-count').textContent = 'Pop: ' + data.population_size;
      if(organisms.length){
        const avg = organisms.reduce((s,o)=>s+o.fitness, 0) / organisms.length;
        document.getElementById('avg-fitness').textContent = 'Avg Fitness: ' + avg.toFixed(4);
      }
      document.getElementById('pool-stat').textContent = 'Pool: ' + (poolFraction*100).toFixed(0) + '%';
      const ts = data.timestamp ? data.timestamp.substring(11,19) : '';
      document.getElementById('update-ts').textContent = ts;

      const banner = document.getElementById('shock-banner');
      banner.style.display = shockActive ? 'block' : 'none';

      // Pool bar
      const fill = document.getElementById('pool-fill');
      fill.style.width = (poolFraction*100) + '%';
      fill.style.background = isScarcity ? '#f85149' : '#3fb950';

      syncOrganisms();
    } catch(err){
      console.warn('poll error:', err);
    }
  }

  /* ================= BOOTSTRAP ================= */
  resize();
  poll().then(() => { requestAnimationFrame(draw); });
  setInterval(poll, POLL_MS);
})();
</script>
</body>
</html>"""


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

    # --- ngrok interstitial skip — lets API clients bypass warning page
    @app.middleware("http")
    async def _ngrok_skip_header(request: Request, call_next):
        response = await call_next(request)
        response.headers["ngrok-skip-browser-warning"] = "true"
        return response

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

      <h2 style="margin-top:.6rem">\U0001F30D Resource Pool</h2>
      <table>
        <tr><td>pool</td><td>{_organism.environment.resource_pool:.1f} / {_organism.environment.config.resource_pool_max:.0f}</td></tr>
        <tr><td>pool %</td><td><div class="bar-bg"><div class="bar-fg" style="width:{_organism.environment.pool_fraction*100:.0f}%;background:{'#f85149' if _organism.environment.is_scarcity_pressure else '#3fb950'}"></div></div></td></tr>
        <tr><td>scarcity pressure</td><td style="color:{'#f85149' if _organism.environment.is_scarcity_pressure else '#3fb950'}">{"⚠️ active (severity " + f"{_organism.environment.scarcity_severity:.0%}" + ")" if _organism.environment.is_scarcity_pressure else "✓ normal"}</td></tr>
        <tr><td>metabolic cost</td><td>{_organism.environment.effective_metabolic_cost():.2f}</td></tr>
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
      <h2 style="margin-top:.6rem">Visual</h2>
      <table>
        <tr><td><a href="/visual">🔬 Organism Visualizer</a></td><td>Live genome-colored circles</td></tr>
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

    # --- GET /api/organisms (public JSON — visual dashboard data) -----
    @app.get("/api/organisms")
    def api_organisms() -> Dict[str, Any]:
        """Public JSON feed for the visual organism dashboard.

        Returns per-organism data optimised for rendering:
        id, fitness, traits (adaptability/energy_efficiency/resilience),
        strategy, alive, nickname, awareness proxy, stagnation proxy.
        """
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        pop = _organism.population
        members = pop.get_all()
        ba = _organism.behavior_analyzer

        organisms = []
        for m in members:
            mid = m.get("id", "?")
            genome = m.get("genome", {})
            traits = genome.get("traits", {})
            fitness = genome.get("fitness", 0.0)
            alive = m.get("alive", True)
            lifecycle_state = m.get("lifecycle_state", "active")
            nickname = m.get("nickname")

            # Strategy from behavior profile
            profile = ba.get_or_create_profile(mid)
            classification = profile.classify_strategy()
            strategy = classification.get("strategy", "neutral")

            # Awareness proxy: for parent use autonomy awareness,
            # for children approximate from fitness history stability
            awareness = 0.0
            if mid == "AL-01":
                awareness = float(dict(_organism.state).get("awareness", 0.0))
            else:
                fh = m.get("fitness_history", [])
                if len(fh) >= 3:
                    recent = [float(f) if isinstance(f, (int, float)) else float(f.get("fitness", 0)) for f in fh[-5:]]
                    if recent:
                        avg = sum(recent) / len(recent)
                        awareness = min(1.0, avg)

            # Stagnation proxy: low fitness variance in recent history
            stagnation = 0.0
            fh = m.get("fitness_history", [])
            if len(fh) >= 5:
                recent_f = [float(f) if isinstance(f, (int, float)) else float(f.get("fitness", 0)) for f in fh[-10:]]
                if len(recent_f) >= 2:
                    import statistics as _st
                    var = _st.variance(recent_f)
                    # Low variance → high stagnation
                    stagnation = max(0.0, 1.0 - var * 100)

            # Energy: parent uses _state, children have member-level energy
            if mid == "AL-01":
                energy = float(dict(_organism.state).get("energy", 1.0))
            else:
                energy = float(m.get("energy", 0.8))

            # Evolution count
            if mid == "AL-01":
                evo_count = int(dict(_organism.state).get("evolution_count", 0))
            else:
                evo_count = int(m.get("evolution_count", 0))

            organisms.append({
                "id": mid,
                "fitness": round(fitness, 6),
                "traits": {
                    "adaptability": round(traits.get("adaptability", 0.0), 4),
                    "energy_efficiency": round(traits.get("energy_efficiency", 0.0), 4),
                    "resilience": round(traits.get("resilience", 0.0), 4),
                    "perception": round(traits.get("perception", 0.0), 4),
                    "creativity": round(traits.get("creativity", 0.0), 4),
                },
                "strategy": strategy,
                "alive": alive,
                "lifecycle_state": lifecycle_state,
                "nickname": nickname,
                "awareness": round(awareness, 4),
                "stagnation": round(stagnation, 4),
                "is_parent": mid == "AL-01",
                "energy": round(energy, 4),
                "evolution_count": evo_count,
            })

        # Environment shock status
        env = _organism.environment
        shock_active = env.is_shock_active if hasattr(env, "is_shock_active") else False

        # Resource pool info for environmental aura
        pool_fraction = env.pool_fraction if hasattr(env, "pool_fraction") else 1.0
        is_scarcity = env.is_scarcity_pressure if hasattr(env, "is_scarcity_pressure") else False

        return {
            "organisms": organisms,
            "population_size": len(organisms),
            "shock_active": shock_active,
            "pool_fraction": round(pool_fraction, 4),
            "is_scarcity": is_scarcity,
            "named_events": env.active_named_event_names if hasattr(env, "active_named_event_names") else [],
            "recent_births": _organism.last_birth_events(n=5) if _organism else [],
            "novelty_rate": _organism.novelty_rate if _organism else 0.0,
            "diversity_index": round(_organism.population_diversity_index(), 4) if _organism else 0.0,
            "timestamp": _utc_now(),
        }

    # --- GET /visual (public HTML — organism visualization) -----------
    @app.get("/visual", response_class=HTMLResponse)
    def visual_dashboard() -> HTMLResponse:
        """Visual organism dashboard — renders each organism as a genome-colored circle."""
        html = _VISUAL_DASHBOARD_HTML
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
            "min_population_floor": pop.min_population_floor,
            "hardcore_extinction_mode": pop.hardcore_extinction_mode,
            "dormant_count": pop.dormant_count,
            "living_or_dormant_count": pop.living_or_dormant_count,
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
    # v3.11: Lineage Tree, Species, Ecosystem Health, Fossils, Events
    # ------------------------------------------------------------------

    @app.get("/lineage/tree", dependencies=[Depends(_require_api_key)])
    def lineage_tree() -> Dict[str, Any]:
        """Return nested family-tree structure for all organisms."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        return _organism.evolution_tracker.build_family_tree()

    @app.get("/lineage/tree/ascii", dependencies=[Depends(_require_api_key)])
    def lineage_tree_ascii() -> Dict[str, str]:
        """Return ASCII-rendered family tree."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        return {"tree": _organism.evolution_tracker.render_tree_ascii()}

    @app.get("/lineage/{organism_id}/ancestors", dependencies=[Depends(_require_api_key)])
    def lineage_ancestors(organism_id: str) -> Dict[str, Any]:
        """Return ancestor chain from organism to root."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        chain = _organism.evolution_tracker.get_ancestor_chain(organism_id)
        return {"organism_id": organism_id, "ancestors": chain}

    @app.get("/lineage/{organism_id}/descendants", dependencies=[Depends(_require_api_key)])
    def lineage_descendants(organism_id: str) -> Dict[str, Any]:
        """Return all descendants of an organism."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        desc = _organism.evolution_tracker.get_descendants(organism_id)
        return {"organism_id": organism_id, "descendants": desc, "count": len(desc)}

    @app.get("/species", dependencies=[Depends(_require_api_key)])
    def species_census() -> Dict[str, Any]:
        """Return species census — which organisms belong to which species."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        census = _organism.population.species_census()
        return {
            "species": {sid: {"members": mids, "count": len(mids)}
                        for sid, mids in census.items()},
            "total_species": len(census),
            "timestamp": _utc_now(),
        }

    @app.get("/ecosystem/health", dependencies=[Depends(_require_api_key)])
    def ecosystem_health() -> Dict[str, Any]:
        """Return composite ecosystem health score and breakdown."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        return _organism.ecosystem_health()

    @app.get("/fossils", dependencies=[Depends(_require_api_key)])
    def fossil_record() -> Dict[str, Any]:
        """Return the fossil record — all dead organisms."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        fossils = _organism.population.fossil_record()
        return {"fossils": fossils, "count": len(fossils), "timestamp": _utc_now()}

    @app.get("/fossils/summary", dependencies=[Depends(_require_api_key)])
    def fossil_summary() -> Dict[str, Any]:
        """Return aggregate fossil statistics."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        return _organism.population.fossil_summary()

    @app.get("/environment/events", dependencies=[Depends(_require_api_key)])
    def environment_events() -> Dict[str, Any]:
        """Return active and historical named environmental events."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        env = _organism.environment
        return {
            "active": [e.to_dict() for e in env.named_events],
            "history": env.named_event_log[-50:],
            "active_count": len(env.named_events),
            "timestamp": _utc_now(),
        }

    @app.get("/births", dependencies=[Depends(_require_api_key)])
    def birth_events(
        limit: int = Query(10, ge=1, le=100, description="Number of recent births"),
    ) -> Dict[str, Any]:
        """Return recent birth events for visual layer."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        events = _organism.last_birth_events(n=limit)
        return {"births": events, "count": len(events), "timestamp": _utc_now()}

    # ------------------------------------------------------------------
    # v3.12: Evolution Dashboard, Novelty, Diversity
    # ------------------------------------------------------------------

    @app.get("/evolution/dashboard", dependencies=[Depends(_require_api_key)])
    def evolution_dashboard() -> Dict[str, Any]:
        """Return evolution dashboard: population, species, novelty rate,
        diversity index, ecosystem health, and stagnation status."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        return _organism.evolution_dashboard()

    @app.get("/evolution/novelty", dependencies=[Depends(_require_api_key)])
    def novelty_history(
        limit: int = Query(100, ge=1, le=500, description="Recent novelty scores"),
    ) -> Dict[str, Any]:
        """Return novelty score history and current average."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        history = _organism.novelty_history[-limit:]
        return {
            "novelty_rate": _organism.novelty_rate,
            "history": history,
            "count": len(history),
            "stagnating": (_organism.avg_novelty < 0.05
                           if len(_organism.novelty_history) >= 10 else False),
            "timestamp": _utc_now(),
        }

    @app.get("/evolution/diversity", dependencies=[Depends(_require_api_key)])
    def population_diversity() -> Dict[str, Any]:
        """Return population diversity index (avg pairwise genome distance)."""
        if _organism is None:
            raise HTTPException(status_code=500, detail="Organism not initialized")
        diversity = _organism.population_diversity_index()
        metrics = _organism.population.diversity_metrics()
        return {
            "diversity_index": round(diversity, 4),
            "unique_genomes": metrics.get("unique_genome_hashes", 0),
            "genome_entropy": metrics.get("genome_entropy", 0.0),
            "trait_stddev": metrics.get("trait_stddev", {}),
            "population_size": _organism.population.size,
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


# ---------------------------------------------------------------------------
# Default ASGI app instance — allows `uvicorn al01.api:app --reload`
# without requiring factory mode.  The factory create_app() is still the
# preferred entry-point when the caller supplies its own Organism.
# ---------------------------------------------------------------------------

def _build_default_app() -> FastAPI:
    import threading
    from contextlib import asynccontextmanager

    api_key = os.environ.get("AL01_API_KEY")
    organism = Organism()

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        """Boot the organism loop on startup, shut it down on exit."""
        organism.boot()
        interval = int(os.environ.get("AL01_LOOP_INTERVAL", "5"))
        organism._loop_stop_event.clear()
        organism._loop_thread = threading.Thread(
            target=organism._loop_worker,
            args=(max(1, interval), False),
            daemon=True,
            name="al01-run-loop",
        )
        organism._loop_thread.start()
        logger.info("[UVICORN] Organism loop started (interval=%ds)", interval)
        yield
        organism.shutdown()
        logger.info("[UVICORN] Organism shutdown complete")

    built_app = create_app(organism=organism, api_key=api_key)
    built_app.router.lifespan_context = _lifespan
    return built_app


app = _build_default_app()
