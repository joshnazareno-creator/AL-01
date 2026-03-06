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
  <b>Circle Size</b> = Fitness &nbsp; <b>Pulse</b> = Energy<br>
  <b>Color</b> = <span style="color:#ff6b6b">Adapt</span> /
                 <span style="color:#6bff6b">Efficiency</span> /
                 <span style="color:#6b6bff">Resilience</span><br>
  <b>Glow</b> = Awareness &nbsp; <b>Rings</b> = Evolutions<br>
  <b>Flicker</b> = Low Energy &nbsp; <b>👑</b> = Top 3 Fitness<br>
  <b>⬡</b> = Parent AL-01
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
  const FLICKER_THRESHOLD = 0.15;
  const EVO_RING_INTERVAL = 250;
  const MAX_EVO_RINGS = 3;
  const LEADER_COUNT = 3;
  const HOVER_REPULSE_RADIUS = 120;
  const HOVER_REPULSE_FORCE = 30;
  const AURA_PARTICLE_COUNT = 60;

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
    layoutGrid();
    if(!auraParticles.length) initAura();
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

  /* ================= GRID LAYOUT ================= */
  function layoutGrid(){
    const n = organisms.length;
    if(!n) return;
    const W = window.innerWidth;
    const H = window.innerHeight;
    const headerH = 50;
    const availW = W - PADDING * 2;
    const availH = H - headerH - PADDING * 2;
    const cols = Math.max(1, Math.ceil(Math.sqrt(n * (availW / availH))));
    const rows = Math.max(1, Math.ceil(n / cols));
    const cellW = availW / cols;
    const cellH = availH / rows;

    // Determine top-3 fitness leaders
    const sorted = organisms.slice().sort((a,b) => (b.fitness||0) - (a.fitness||0));
    leaderIds = new Set(sorted.slice(0, Math.min(LEADER_COUNT, n)).map(o => o.id));

    for(let i = 0; i < n; i++){
      const col = i % cols;
      const row = Math.floor(i / cols);
      const tx = PADDING + col * cellW + cellW / 2;
      const ty = headerH + PADDING + row * cellH + cellH / 2;
      const o = organisms[i];
      const r = BASE_R + (o.fitness || 0) * SCALE_R;
      const c = traitColor(o.traits || {});
      if(circles[i]){
        circles[i].targetX = tx;
        circles[i].targetY = ty;
        circles[i].targetR = r;
        circles[i].color = c;
        circles[i].data = o;
      } else {
        circles[i] = {x: tx, y: ty, targetX: tx, targetY: ty,
                       r: r, targetR: r, color: c, data: o,
                       shimmerPhase: Math.random() * Math.PI * 2,
                       crownAngle: Math.random() * Math.PI * 2};
      }
    }
    circles.length = n;
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

  /* ================= DRAW AURA FIELD ================= */
  function drawAura(W, H, dt){
    // Background tint based on pool health
    const health = poolFraction;
    const bgR = Math.round(lerp(30, 8, health));
    const bgG = Math.round(lerp(10, 14, health));
    const bgB = Math.round(lerp(10, 20, health));
    ctx.fillStyle = 'rgb('+bgR+','+bgG+','+bgB+')';
    ctx.fillRect(0, 0, W, H);

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

    // Floating particles
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

    for(let i = 0; i < circles.length; i++){
      const c = circles[i];
      const o = c.data;

      // 7. Micro-Interaction Physics — hover repulsion
      if(hoveredIdx >= 0 && i !== hoveredIdx){
        const hc = circles[hoveredIdx];
        const dx = c.targetX - hc.x;
        const dy = c.targetY - hc.y;
        const dist = Math.sqrt(dx*dx + dy*dy);
        if(dist < HOVER_REPULSE_RADIUS && dist > 0){
          const force = (1 - dist / HOVER_REPULSE_RADIUS) * HOVER_REPULSE_FORCE;
          const nx = dx / dist, ny = dy / dist;
          c.x += nx * force * dt * 2;
          c.y += ny * force * dt * 2;
        }
      }

      // Smooth interpolation
      c.x = lerp(c.x, c.targetX, 0.08);
      c.y = lerp(c.y, c.targetY, 0.08);
      c.r = lerp(c.r, c.targetR, 0.1);

      const energy = o.energy || 0;
      const evoCount = o.evolution_count || 0;
      const opacity = Math.max(0.25, 1.0 - (o.stagnation || 0) * 0.6);

      // 1. Energy-Based Pulse Animation
      const energyFactor = 0.8 + energy * 2.2;
      const pulse = 1.0 + PULSE_INTENSITY * Math.sin(time * energyFactor);
      let drawR = Math.max(3, c.r * pulse);

      // 4. Energy Stress Flicker
      let flickerOffsetX = 0, flickerOffsetY = 0, flickerAlpha = 1.0;
      if(energy < FLICKER_THRESHOLD && energy > 0){
        const severity = 1 - (energy / FLICKER_THRESHOLD);
        flickerOffsetX = (Math.random() - 0.5) * severity * 3;
        flickerOffsetY = (Math.random() - 0.5) * severity * 3;
        flickerAlpha = 0.7 + Math.random() * 0.3 * (1 - severity * 0.4);
      }

      const cx = c.x + flickerOffsetX;
      const cy = c.y + flickerOffsetY;

      ctx.save();
      ctx.globalAlpha = opacity * flickerAlpha;

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

      // Main circle body
      ctx.beginPath();
      ctx.arc(cx, cy, drawR, 0, Math.PI * 2);
      ctx.fillStyle = c.color.str;
      ctx.fill();

      // Inner highlight (specular)
      const hlGrad = ctx.createRadialGradient(
        cx - drawR * 0.25, cy - drawR * 0.25, 0,
        cx, cy, drawR
      );
      hlGrad.addColorStop(0, 'rgba(255,255,255,0.12)');
      hlGrad.addColorStop(0.6, 'rgba(255,255,255,0.02)');
      hlGrad.addColorStop(1, 'rgba(0,0,0,0.1)');
      ctx.beginPath();
      ctx.arc(cx, cy, drawR, 0, Math.PI * 2);
      ctx.fillStyle = hlGrad;
      ctx.fill();

      // Parent marker — hexagon outline
      if(o.is_parent){
        ctx.strokeStyle = '#f0f6fc';
        ctx.lineWidth = 2;
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

      layoutGrid();
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
