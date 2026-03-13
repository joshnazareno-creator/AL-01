# AL-01

A persistent, self-evolving digital organism ecosystem with autonomous reproduction, genome evolution, environmental pressure, and a live visual dashboard.

**v3.30** — 1273 tests | 23 modules | 64 API endpoints | Canvas organism visualizer

---

## What Is AL-01?

AL-01 is a bounded artificial life system that runs continuously, evolving a population of digital organisms through deterministic tick-based cycles. Each organism has a genome of five traits, an energy budget, fitness scoring, and behavioral strategies that emerge from environmental pressure.

Key properties:
- **Self-evolving** — organisms mutate, reproduce, and die without intervention
- **Environmentally bounded** — a shared resource pool creates real scarcity and competition
- **Persistent** — all state survives restarts via SQLite + Firestore/JSON
- **Observable** — full HTTP API, HTML dashboard, and live canvas visualizer
- **Reproducible** — seeded experiments with deterministic RNG

---

## Features

| Category | Capabilities |
|----------|-------------|
| **Genome** | 5 traits (adaptability, energy_efficiency, resilience, perception, creativity) with soft ceilings, trade-off rules, and mutation drift |
| **Population** | Up to 60 organisms, autonomous reproduction, selection pressure, death at zero energy, generational lineage tracking, species divergence (threshold 0.35), energy-gated reproduction with parent cost |
| **Environment** | Fluctuating temperature/entropy/resources/noise, scarcity events, global shared resource pool (1000 units, fair distribution, min floor), emergency regen, adaptive metabolism, resource-scaled reproduction probability |
| **Autonomy** | Decision engine (stabilize/adapt/mutate/blend), computational awareness model, auto mutation rate adjustment |
| **Behavior** | Emergent strategy detection: energy-hoarder, explorer, specialist, generalist, resilient |
| **Brain** | Optional OpenAI integration for strategic nudges; core logic is fully local and deterministic |
| **VITAL** | Append-only hash-chain life log with SHA-256 integrity verification |
| **Persistence** | SQLite primary store, Firestore cloud replication (optional), local JSON fallback (1000-entry cap), hourly snapshots with 30-day retention, absolute-path storage anchoring |
| **Stabilization** | Minimum energy floor with conservation mode, population-scaled resource regeneration, adaptability recovery boost, stress feedback loop, energy-efficiency-weighted metabolism, extinction prevention guard |
| **Evolution** | Novelty metric (genome distance), stagnation detection, population diversity scoring, evolution dashboard, innovation tracking |
| **Ecosystem** | Anti-monoculture culling, environmental shock events, dormant organisms, extinction recovery via genesis vault, ecosystem stabilisation (pool floor, energy gates, probability scaling), permanent child death (no dormancy/revival for non-founders), founder-only rescue from graveyard |
| **API** | 64 FastAPI endpoints — status, genome, population, lineage, species, fossils, evolution dashboard, novelty, diversity, export/import, experiment control, GPT bridge |
| **Visual** | Full-screen canvas dashboard with 12 animation systems (pulse, halo, rings, vibration, shimmer, aura, physics, crowns, energy trail, heartbeat, fitness glow, dormant state) + mobile-first organic movement, visual idle states (exploring/resting/sleeping), smooth wall avoidance, center-biased spawning |

---

## Project Structure

```
al01/
  organism.py          # Core organism — metabolism, evolution cycles, energy, state machine (3001 lines)
  api.py               # FastAPI server — 64 endpoints, HTML dashboards, GPT OpenAPI schema (2051 lines)
  autonomy.py          # Autonomous Decision Engine + Computational Awareness Model (1083 lines)
  environment.py       # Environment model — resource pool, fluctuating variables, scarcity (1165 lines)
  population.py        # Multi-organism registry — reproduction, death, pruning, cap=60 (1190 lines)
  portable.py          # Child export/import — HMAC-SHA256 integrity, JSON schema (581 lines)
  brain.py             # Strategic Decision Engine — env analysis, gap analysis, nudges (449 lines)
  memory_manager.py    # Local-first persistence + Firestore replication, 5000-entry cap (470 lines)
  life_log.py          # VITAL — append-only SHA-256 hash-chain log (436 lines)
  evolution_tracker.py # Generation IDs, genome hashing, mutation logs, CSV export (425 lines)
  snapshot_manager.py  # Hourly state snapshots with 30-day retention (411 lines)
  storage.py           # v3.22: Global storage configuration — absolute paths, JSONL rotation, snapshot cleanup
  genome.py            # 5-trait genome — mutation, trade-offs, soft ceilings (399 lines)
  gpt_bridge.py        # Natural language narration + stimulus injection for GPT (387 lines)
  behavior.py          # Emergent behavior detection + strategy classification (339 lines)
  experiment.py        # Seeded reproducible experiment protocol (300 lines)
  database.py          # SQLite persistence (interactions, snapshots, metadata) (293 lines)
  cli.py               # CLI tool — hash-chain verification, snapshot management (238 lines)
  genesis_vault.py     # Immutable seed template + extinction recovery (205 lines)
  policy.py            # Adaptive policy weights (curiosity, risk, social) (78 lines)
  identity.json        # Organism identity (name, description, version)
  __main__.py          # Boot sequence entrypoint (241 lines)
  __init__.py          # Package re-exports

tests/                 # 29 test files, 1089+ tests
  test_alife.py          # Core artificial life tests
  test_genesis_vault.py  # Genesis vault + extinction recovery
  test_gpt_bridge.py     # GPT bridge narration + stimulus
  test_portable.py       # Portable export/import schema
  test_rare_reproduction.py # Rare reproduction (5% gate, 2000-cycle cooldown)
  test_repair_vital.py   # VITAL hash-chain repair
  test_snapshots.py      # Snapshot manager
  test_v12.py            # v1.2 persistence + API auth
  test_v3.py             # v3.0 population + behavior
  test_v34.py            # v3.4 cycle stats
  test_v35.py            # v3.5 energy floor + parent reserve
  test_v36.py            # v3.6 brain rewrite
  test_v37.py            # v3.7 trait floor + variance kick
  test_v38.py            # v3.8 stagnation breaker + founder protection
  test_v39.py            # v3.9 multi-objective fitness + memory drift
  test_v310.py           # v3.10 portable child export/import
  test_v311.py           # v3.11 ecosystem mechanics (anti-monoculture, shock, explorer)
  test_v311_features.py  # v3.11 stability reproduction, family tree, species divergence
  test_v312.py           # v3.12 novelty metric, stagnation detection, evolution dashboard
  test_v313.py           # v3.13 global resource pool
  test_v314.py           # v3.14 population floor (20% of max)
  test_v315.py           # v3.15 dormant organisms, emergency regen, adaptive metabolism
  test_v316.py           # v3.16 dormant wake, stability repro, lone survivor
  test_v317.py           # v3.17 reproduction safety (500-cycle birth cooldown, idempotency, concurrency guard)
  test_v318.py           # v3.18 ecosystem stabilisation (pool floor, energy gates, probability scaling)
  test_v319.py           # v3.19 fix double mutation in rare reproduction
  test_v320.py           # v3.20 nuclear threshold for oversized memory.json
  test_v321.py           # v3.21 bounded memory, SQLite archival, tick snapshots
  test_v322.py           # v3.22 absolute-path storage, JSONL rotation, snapshot retention
  test_v323.py           # v3.23 ecosystem stabilization (conservation, regen, adaptability, stress)
  test_v324.py           # v3.24 wire missing reproduction paths into scheduler
  test_visual.py         # Visual dashboard + /api/organisms endpoint
  test_vital.py          # VITAL hash-chain integrity

data/
  life_log.jsonl       # Append-only hash-chain event log (VITAL)
  head.json            # Current chain head (seq, hash, organism_id)
  genesis_vault.json   # Immutable seed genome (written once)
  evolution_log.jsonl  # Evolution tracker events
  autonomy_log.jsonl   # Decision cycle events
  cycle_log.jsonl      # Structured per-cycle logs
  snapshots/           # Periodic state checkpoints

snapshots/
  hourly/              # Hourly auto-snapshots with manifest (30-day retention)

.env                   # API key config (not committed)
.env.example           # Template
```

---

## Quick Start

### Prerequisites

- **Python 3.12+** (tested on 3.14.2)
- No external services required — runs fully offline

### Install

```bash
# Clone
git clone https://github.com/joshn/AL-01.git
cd AL-01

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install fastapi uvicorn pydantic
```

**Optional dependencies:**

```bash
pip install firebase-admin   # Firestore cloud persistence
pip install openai           # AI brain integration
pip install httpx             # Required for running tests
pip install ngrok             # Tunnel to expose publicly
```

### Configure

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required: API key for authenticated endpoints
AL01_API_KEY=your-secret-key

# Optional: enables AI brain integration
OPENAI_API_KEY=sk-...

# Optional: override for GPT Actions OpenAPI spec
NGROK_URL=https://your-domain.ngrok-free.dev
```

Remove `AL01_API_KEY` or leave it empty to run in open/dev mode (no auth required).

### Run

```bash
python -m al01
```

The system will:

1. Load `.env` and configure logging (console + `al01.log`)
2. Initialize SQLite database (`al01.db`)
3. Initialize VITAL subsystems (life log + policy manager)
4. Initialize persistence (local JSON first, optional Firestore)
5. Initialize population registry (parent: AL-01)
6. Initialize brain (enabled if `OPENAI_API_KEY` set)
7. Initialize autonomy engine (decision interval: 10 ticks)
8. Boot organism (load prior state, increment evolution, persist)
9. Initialize snapshot manager (hourly, 30-day retention)
10. Initialize genesis vault (extinction recovery)
11. Start API server on **port 8000** (daemon thread)
12. Enter heartbeat loop (Ctrl+C to stop gracefully)

### Access

| URL | Description |
|-----|-------------|
| `http://localhost:8000/` | HTML dashboard — live charts, population stats, resource pool |
| `http://localhost:8000/visual` | Canvas organism visualizer — animated circles for each organism |
| `http://localhost:8000/health` | Health check (JSON) |
| `http://localhost:8000/api/organisms` | Raw organism data (JSON) |

---

## API Reference

All endpoints except those marked **Public** require the `X-API-Key` header (or `?api_key=` query param).

### Public Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | HTML dashboard |
| GET | `/visual` | Canvas organism visualizer |
| GET | `/api/organisms` | JSON feed — id, fitness, traits, energy, evolution_count, strategy, awareness, stagnation per organism + pool fraction |
| GET | `/health` | Health check — status, version, uptime |
| GET | `/gpt/openapi.json` | ChatGPT Actions-compatible OpenAPI 3.1.0 spec |

### Core State

| Method | Path | Description |
|--------|------|-------------|
| GET | `/status` | Full organism state snapshot |
| GET | `/identity` | Name, description, version |
| GET | `/growth` | Growth metrics + totals since first run |
| GET | `/genome` | Current genome — 5 traits, fitness, mutation rate |
| GET | `/stimuli` | Pending stimuli queue |
| GET | `/autonomy` | Autonomy engine state + awareness model |
| GET | `/evolve` | Trigger an evolution cycle |

### Interaction

| Method | Path | Description |
|--------|------|-------------|
| POST | `/interact` | Record a structured interaction (writes to SQLite + persistence) |
| POST | `/command` | Send a named command to the organism |
| POST | `/stimulate` | Inject a stimulus into the evolution loop |

### Population

| Method | Path | Description |
|--------|------|-------------|
| GET | `/population` | Full population registry |
| GET | `/population/metrics` | Population-wide metrics |
| GET | `/population/{id}` | Single organism details |
| PUT | `/population/{id}/nickname` | Set organism nickname |
| GET | `/population/{id}/export` | Export child as portable JSON (HMAC-signed) |
| POST | `/population/import` | Import/adopt a portable child |

### Memory

| Method | Path | Description |
|--------|------|-------------|
| GET | `/memory/recent?n=10` | Last *n* interactions (max 50) |
| GET | `/memory/search?keyword=...&limit=20` | Full-text search across interactions |

### VITAL (Integrity)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/vital/status` | VITAL subsystem status |
| GET | `/vital/verify` | Verify hash-chain integrity |
| GET | `/vital/head` | Current chain head |
| GET | `/vital/policy` | Current policy weights |
| POST | `/vital/policy` | Update policy weights |

### Snapshots

| Method | Path | Description |
|--------|------|-------------|
| GET | `/snapshots` | List all snapshots |
| POST | `/snapshots` | Trigger manual snapshot |
| GET | `/snapshots/status` | Snapshot manager status |
| GET | `/snapshots/latest` | Latest snapshot data |
| DELETE | `/snapshots/purge` | Purge old snapshots |

### Genesis Vault

| Method | Path | Description |
|--------|------|-------------|
| GET | `/vault` | Vault status |
| GET | `/vault/seed` | Pristine genesis seed template |
| GET | `/vault/history` | Reseed history |
| POST | `/vault/reseed` | Force extinction recovery reseed |

### Lineage & Export

| Method | Path | Description |
|--------|------|-------------|
| GET | `/lineage` | Full lineage tree |
| GET | `/lineage/{id}` | Lineage for specific organism |
| GET | `/lineage/tree` | Structured lineage tree (JSON) |
| GET | `/lineage/tree/ascii` | ASCII art lineage tree |
| GET | `/lineage/{id}/ancestors` | Ancestor chain for organism |
| GET | `/lineage/{id}/descendants` | Descendant tree for organism |
| GET | `/export/fitness.csv` | Fitness history CSV |
| GET | `/export/mutations.csv` | Mutation log CSV |
| GET | `/export/lineage.csv` | Lineage CSV |

### Experiments

| Method | Path | Description |
|--------|------|-------------|
| POST | `/experiment/start` | Start a seeded experiment |
| POST | `/experiment/stop` | Stop current experiment |
| GET | `/experiment/status` | Experiment status |
| POST | `/exploration/toggle` | Toggle exploration mode |

### GPT Bridge

| Method | Path | Description |
|--------|------|-------------|
| GET | `/gpt/narrate` | Natural language narration of organism state |
| POST | `/gpt/stimulus` | GPT stimulus injection (rate-limited) |
| GET | `/gpt/status` | GPT bridge status |
| POST | `/gpt/toggle` | Toggle GPT bridge on/off |
| GET | `/gpt/log` | GPT bridge event log |

### Species & Ecosystem

| Method | Path | Description |
|--------|------|-------------|
| GET | `/species` | Species registry — groups by genome distance > 0.35 |
| GET | `/ecosystem/health` | Ecosystem health summary (diversity, pool, pop metrics) |
| GET | `/fossils` | Fossil record — dead organisms with full history |
| GET | `/fossils/summary` | Fossil summary statistics |
| GET | `/environment/events` | Recent environment events (scarcity, named events) |
| GET | `/births` | Recent birth events with parent/child details |

### Evolution Intelligence (v3.12)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/evolution/dashboard` | Full evolution dashboard — fitness trends, stagnation, novelty, diversity |
| GET | `/evolution/novelty` | Novelty metric — genome distance from historical archive |
| GET | `/evolution/diversity` | Population diversity score and per-trait variance |

---

## Core Concepts

### Genome

Every organism has 5 traits, each starting at 0.5:

| Trait | Effect |
|-------|--------|
| `adaptability` | Speed of environmental adaptation |
| `energy_efficiency` | Metabolic cost reduction |
| `resilience` | Resistance to environmental stress |
| `perception` | Awareness of environment changes |
| `creativity` | Exploration of novel strategies |

Traits mutate each evolution cycle (`mutation_rate=0.10`, `mutation_delta=0.10`). A soft ceiling via logarithmic squash prevents runaway values. Trade-off rules: increasing adaptability or creativity penalizes energy_efficiency.

### Energy & Fitness

- Every organism has an **energy** level (0.0–1.0)
- Energy is consumed by metabolic cost each cycle, replenished from the shared resource pool
- **Fitness** is calculated from genome traits, environmental fit, and multi-objective scoring
- Death occurs at energy ≤ 0.0
- Founder (AL-01) has protective floors: energy ≥ 0.25, fitness ≥ 0.15

### Population & Reproduction

- **Hard cap: 60 organisms** (`ABSOLUTE_POPULATION_CAP`)
- **Population floor: 20% of max** (auto-maintained unless hardcore extinction mode)
- Reproduction requires fitness ≥ 0.5 maintained for 5 consecutive cycles
- **Energy gates** (v3.18): auto-reproduce requires energy ≥ 0.50, stability ≥ 0.60, lone survivor ≥ 0.40, rare ≥ 0.65; parent pays 0.20 energy cost per spawn
- **Probability scaling** (v3.18): reproduction probability is multiplied by `pool_fraction` — low resources make reproduction rare
- Children inherit parent genome + mutations, start with 0.8 energy
- **Permanent child death** (v3.28): non-founder organisms die permanently — no dormancy, no revival, no rescue from graveyard
- **Founder protection** (v3.28): AL-01 gets emergency energy injection at low energy, is the only organism eligible for graveyard rescue
- Selection pressure: organisms below fitness 0.2 get 20 grace cycles before removal
- Genesis vault auto-reseeds if population drops to zero
- **Species tracking**: organisms diverging > 0.35 genome distance get separate species IDs
- **Birth cooldown**: each parent can only reproduce once every 500 cycles (`BIRTH_COOLDOWN_CYCLES`)

**Reproduction types:**
| Type | Trigger |
|------|---------|
| Auto-reproduce | Fitness ≥ 0.5 for 5 consecutive cycles, energy ≥ 0.50, costs 0.20 energy |
| Rare reproduction | 5% × pool_fraction chance every 500 cycles, fitness ≥ 0.50, energy ≥ 0.65, costs 0.30, variance 0.10, 2000-cycle cooldown |
| Stability reproduction | Resource pool > 80%, fitness maintained, energy ≥ 0.60, costs 0.20, prob scaled by pool_fraction |
| Lone survivor | Only 1 organism alive + pool > 70%, energy ≥ 0.40, costs 0.20, prob scaled by pool_fraction |
| Extinction reseed | Population = 0, genesis vault reseed |

### Resource Pool

A global shared pool of 1000 energy units, regenerating +5 per cycle:

- **Minimum floor** (v3.18): pool never drops below `resource_pool_min_floor` (production: 50 units) — ensures baseline regeneration even under extreme consumption
- Energy is distributed fairly: `(pool − floor) / population_size` per organism
- Smart regeneration: rate scales with average population efficiency
- **Scarcity mode** activates when pool drops below 25%:
  - Metabolic cost increases 1.5×
  - Reproduction threshold rises 50%
  - Survival grace reduces by 1.5×
- **Emergency regen**: auto-top-up when pool < 25%
- **Adaptive metabolism**: cost reduction during high scarcity
- **Dormant state** (v3.15): organisms go dormant at very low pool, wake when pool > 50%
- Random scarcity events (2% per cycle) create environmental shocks
- **Named events**: mutation_storm, fitness_spike, entropy_burst, resource_boom

### Autonomy Engine

The decision engine runs every 10 ticks, choosing between:

- **Stabilize** — reduce mutation, conserve energy
- **Adapt** — moderate mutation toward environment fit
- **Mutate** — increase exploration
- **Blend** — mix strategies from successful neighbours

**Awareness model**: `stimuli_rate × 0.4 + decision_rate × 0.3 + fitness_var × 0.2 + (1−stagnation) × 0.1 + novelty_bonus`

### Behavior Strategies

Emergent strategies are detected from trait patterns:

| Strategy | Pattern |
|----------|---------|
| Energy Hoarder | High energy_efficiency, low adaptability |
| Explorer | High creativity + adaptability |
| Specialist | One dominant trait > 0.8 |
| Generalist | All traits balanced |
| Resilient | High resilience, high energy_efficiency |

---

## Visual Dashboard

The `/visual` page renders every organism as an animated circle on a full-screen canvas. It polls `/api/organisms` every 3 seconds.

### 12 Animation Systems

1. **Energy Pulse** — circle radius oscillates with a sine wave; frequency proportional to energy level; resting organisms pulse calmer (35% intensity) with gentle breathing oscillation
2. **Awareness Halo** — soft radial gradient glow for organisms with awareness > 0.5
3. **Evolution Rings** — rotating dashed rings appear every 250 evolutions (max 3), alternating direction
4. **High-Energy Vibration** — position jitter and buzz when energy > 70% (intensity scales with surplus)
5. **Trait Shimmer** — animated gradient overlay colored by dominant trait
6. **Environmental Aura** — background particles shift blue→red with resource pool health; scarcity triggers a red vignette
7. **Hover Physics** — hovering over one organism pushes nearby ones away
8. **Leader Crowns** — top 3 fitness organisms get gold/silver/bronze orbiting particles
9. **Energy Trail** — particle emitters on high-energy organisms (>50%), gravity-affected sparks in organism color
10. **Heartbeat Ring** — rhythmic expanding pulse ring on all living organisms
11. **Fitness Glow Border** — green outline that brightens with fitness (>0.3)
12. **Dormant State** — greyed-out, low-alpha, peaceful breathing opacity with 💀 skull marker

### Visual Idle States (visual-only, does not alter simulation)

Each organism is classified into a display-only behaviour mode derived from real data:

| State | Trigger | Visual Behaviour |
|-------|---------|------------------|
| **Exploring** | Energy > 0.35, fitness > 0.2 | Smooth curved arcs via curvature-based wander, occasional pause-turn-continue, resource attraction |
| **Resting** | Energy < 0.25 or stagnation > 0.6 | Very gentle float (8% speed), subtle breathing scale oscillation, dimmed opacity (78%), stronger damping |
| **Sleeping** | Dormant state | Near-stationary, grey desaturated, 💀 marker, minimal drift |

### Mobile-First Organic Movement

- **Smooth wall avoidance** — smoothstep cubic ramp replaces linear boundary force; organisms curve away from walls over a wide zone (2× margin)
- **Wander-angle steering** — when near walls, `wanderAngle` is blended toward screen center so organisms don't aim at walls
- **Center-biased spawning** — new organisms spawn from a Gaussian-ish distribution centered on screen, not uniform-random
- **Gentle center pull** — organisms in the outer 35% of the screen get a soft quadratic pull toward center
- **Mobile-aware margins** — boundary margin and force scale with screen size (larger margin on small screens)
- **Stronger repulsion** — quadratic overlap force (110 strength, 3.5× radius) prevents wall-piled clumping

### Visual Encoding

| Visual Property | Data Source |
|----------------|-------------|
| Circle size | Fitness |
| Circle color | RGB from adaptability / energy_efficiency / resilience |
| Circle fill | 3D gradient (highlight + shadow) with trait color |
| Pulse speed | Energy level |
| Glow intensity | Awareness |
| Ring count | Evolution count ÷ 250 |
| High-energy vibration | Energy > 70% |
| Background tint | Resource pool health |
| Crown rank | Top 3 fitness |
| Hexagon outline | Parent AL-01 (blue glow) |
| Opacity | Inversely proportional to stagnation |
| Energy trail | Particle sparks when energy > 50% |
| Heartbeat ring | Rhythmic expanding pulse on living organisms |
| Green border | Fitness > 0.3 (brightness scales with fitness) |
| Greyed + 💀 | Dormant state |
| Dimmed + breathing | Resting state (low energy or high stagnation) |
| Smooth arcs + pauses | Exploring state (healthy energy, active) |

---

## Persistence Architecture

```
                 ┌──────────────┐
                 │   Organism   │
                 └──────┬───────┘
                        │
              ┌─────────┴─────────┐
              │   MemoryManager   │
              └──┬──────────┬─────┘
                 │          │
         ┌───────┴──┐  ┌───┴────────┐
         │  SQLite  │  │  Firestore  │
         │ (al01.db)│  │  (cloud)    │
         └──────────┘  └──┬─────────┘
                          │ fallback
                    ┌─────┴──────┐
                    │ Local JSON │
                    │ state.json │
                    │ memory.json│
                    └────────────┘
```

- **SQLite** (`al01.db`) — primary structured store for interactions + growth snapshots
- **Firestore** — cloud persistence for state + memory events (optional, requires `ServiceAccountKey.json`)
- **Local JSON** — automatic fallback when Firestore is unavailable; all writes go local first; memory entries capped at 5000 to prevent file bloat
- **Snapshots** — hourly state snapshots with 30-day retention + Firestore backup sync
- **VITAL** — append-only hash-chain life log (`data/life_log.jsonl`) with SHA-256 integrity
- On restart, all state is recovered from SQLite + JSON — nothing is lost

---

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AL01_API_KEY` | No | API authentication key. If unset, runs in open/dev mode |
| `OPENAI_API_KEY` | No | Enables AI brain integration |
| `AL01_LOOP_INTERVAL` | No | Override heartbeat interval in seconds (default: 5) |
| `NGROK_URL` | No | Public URL for GPT Actions OpenAPI spec |

### MetabolismConfig

| Field | Default | Description |
|-------|---------|-------------|
| `pulse_interval` | 1 | Ticks between pulses |
| `reflect_interval` | 10 | Ticks between reflections |
| `persist_interval` | 5 | Ticks between state persists |
| `evolve_interval` | 30 | Ticks between evolution cycles |
| `autonomy_interval` | 10 | Ticks between autonomy decisions |
| `environment_interval` | 5 | Ticks between environment updates |
| `auto_reproduce_interval` | 15 | Ticks between reproduction checks |
| `child_autonomy_interval` | 10 | Ticks between child autonomy cycles |
| `behavior_analysis_interval` | 20 | Ticks between behavior analysis |
| `population_interact_interval` | 60 | Ticks between population interactions |

### ExperimentConfig

| Field | Default | Description |
|-------|---------|-------------|
| `global_seed` | 42 | RNG seed for reproducibility |
| `duration_days` | 30 | Experiment duration |
| `max_population` | 60 | Population ceiling |
| `min_population` | 2 | Minimum population before reseed consideration |
| `survival_fitness_threshold` | 0.2 | Below this fitness → grace countdown begins |
| `survival_grace_cycles` | 20 | Cycles before low-fitness organism is removed |
| `reproduction_fitness_threshold` | 0.5 | Minimum fitness to reproduce |
| `reproduction_fitness_cycles` | 5 | Consecutive cycles above threshold required |
| `energy_death_threshold` | 0.0 | Energy level that triggers death |

---

## Running Tests

```bash
pip install pytest httpx
python -m pytest tests/ -v
```

1241 tests across 34 files covering:
- Core organism lifecycle and state machine
- Population spawning, death, pruning, and cap enforcement
- Genome mutation, trade-offs, and soft ceilings
- Autonomy decisions and awareness model
- Environmental pressure and resource pool dynamics
- Behavior detection and strategy classification
- VITAL hash-chain integrity verification and repair
- Portable child export/import with HMAC validation
- Genesis vault extinction recovery
- GPT bridge narration and stimulus injection
- Snapshot management and retention
- API authentication, endpoint responses, and error handling
- Visual dashboard HTML structure and data feed
- Novelty metric, stagnation detection, and evolution dashboard (v3.12)
- Global resource pool mechanics (v3.13)
- Population floor enforcement (v3.14)
- Dormant organisms, emergency regen, adaptive metabolism (v3.15)
- Dormant wake, stability reproduction, lone survivor (v3.16)
- Reproduction safety: 500-cycle birth cooldown per parent, event idempotency, tick ordering, concurrency guard (v3.17)
- Ecosystem stabilisation: pool minimum floor, energy-gated reproduction, resource-scaled probability (v3.18)
- Fix double mutation in rare reproduction, spawn_child mutation_variance parameter (v3.19)
- Nuclear threshold: oversized memory.json (>500 MB) replaced instead of parsed to prevent OOM/freeze (v3.20)
- Bounded memory: rolling window cap (1000), SQLite archival, atomic writes, write throttling, tick snapshots (v3.21)
- Absolute-path storage, JSONL log rotation (50 MB), tick snapshot retention (200), RotatingFileHandler, reflection doubling fix, disk monitoring (v3.22)
- Ecosystem stabilization: conservation mode, population-scaled regen, adaptability recovery, stress feedback, energy-efficiency metabolism, extinction prevention guard (v3.23)
- Wire missing reproduction paths into scheduler: lone_survivor, stability_reproduction, wake_dormant (v3.24)
- Permanent child death, founder-only graveyard rescue, dead member startup cleanup, wake_dormant/check_extinction skip non-founder organisms (v3.28)
- Restart-safe ecology: persisted survival counters, environment state restoration, 50-cycle recovery window suppresses all death/pruning after boot (v3.29)
- Energy death guard: child and founder energy depletion deaths suppressed during restart recovery window (v3.30)
- Rare reproduction mechanics (5% gate, 2000-cycle cooldown)

---

## Deploying with ngrok

To expose AL-01 publicly (for ChatGPT Actions, external monitoring, etc.):

```bash
# Start AL-01
python -m al01

# In another terminal, tunnel port 8000
ngrok http 8000 --url your-domain.ngrok-free.dev
```

Set `NGROK_URL` in `.env` so the GPT-compatible OpenAPI spec at `/gpt/openapi.json` uses the correct public URL.

---

## Alternative: Run with uvicorn Directly

```bash
python -m uvicorn al01.api:app --host 0.0.0.0 --port 8000
```

This uses a default app instance that auto-initializes the organism. The module-level `app` object includes a lifespan context manager that boots the organism and starts the heartbeat loop automatically.

---

## CLI Tools

```bash
# Verify VITAL hash-chain integrity
python -m al01.cli verify --last 500
```

---

## Version History

| Version | Key Additions |
|---------|--------------|
| **v3.30** | Energy death guard during restart recovery — child energy-depletion death (`energy ≤ 0`) is now suppressed during the 50-cycle restart recovery window (energy clamped to `energy_min` instead of killing); founder energy-depletion death handler also guarded (overrides `organism_died` flag, prevents unnecessary rescue cascade); closes the last unguarded death path that could cause mass extinction on restart |
| **v3.29** | Restart-safe ecology — persists `_below_fitness_cycles`, `_last_birth_cycle`, `_conservation_mode`, and full environment state across restarts; boot restores all survival counters so grace periods are never lost; 50-cycle post-restart recovery window (`RESTART_RECOVERY_CYCLES`) suppresses fitness-floor death, population pruning, and founder death; `is_restart_recovery` / `restart_recovery_remaining` properties; environment `from_dict()` round-trip on every boot prevents trait-weight drift from causing artificial extinction; recovery countdown decrements each `autonomy_cycle` tick |
| **v3.28** | Permanent child death & founder protection — non-founder organisms die permanently (no dormancy, no revival, no rescue from graveyard); AL-01 founder gets emergency energy injection at critically low energy; dead-organism guard clauses on `update_member()`/`update_energy()`; startup validation moves dead children from `_members` to graveyard; `rescue_from_graveyard()` restricted to AL-01 only; `wake_dormant_cycle()` skips non-AL-01; `check_extinction_reseed()` does not wake dormant organisms. Visual dashboard: mobile-first organic movement with smoothstep wall avoidance, visual idle states (exploring/resting/sleeping), curvature-based wander with pause-turn-continue, center-biased spawning, gentle center pull, mobile-aware boundary margins |
| **v3.24** | Wire missing reproduction paths into MetabolismScheduler — `lone_survivor_reproduction()`, `stability_reproduction_cycle()`, and `wake_dormant_cycle()` were defined on Organism but never called by the tick loop; now invoked every `auto_reproduce_interval` ticks after `auto_reproduce_cycle()`, respecting death-before-reproduction ordering |
| **v3.23** | Ecosystem stabilization — minimum energy floor with conservation mode (10% threshold, 30% metabolism), population-scaled resource regeneration (`population_regen_bonus=0.5` per organism), adaptability recovery boost (nudge +0.02/cycle when trait < 0.20), stress feedback loop (stress > 0.60 → exploration + mutation boost), energy_efficiency trait reduces per-cycle energy decay by up to 30%, extinction prevention guard (pop=1 → 3× regen + 100 flat pool boost) |
| **v3.22** | Absolute-path storage anchoring — all file writes pinned to `D:\AL-01` via `al01/storage.py` module (overridable via `AL01_BASE_DIR` env var), RotatingFileHandler for al01.log (10 MB × 5 backups), JSONL log rotation for life_log/evolution_log/autonomy_log/cycle_log/experiment_log (50 MB × 3 backups), tick snapshot retention (keep newest 200), SQLite moved to `db/al01.db` with auto-migration, `tempfile.tempdir` redirected to `BASE_DIR/tmp/`, CLI defaults use absolute paths, per-event payload size cap (10 KB) to prevent reflection-doubling bug, periodic disk usage monitoring (warns at 10 GB), memory_manager backup/log paths absolutised |
| **v3.21** | Bounded memory system — rolling window cap reduced from 5000 → 1000 entries, memory events archived to SQLite `memory_events` table for long-term storage, atomic writes for memory.json (tmpfile + rename), write throttling (batch N writes before disk flush), periodic tick-based snapshots (`data/snapshots/snap_<tick>.json`), file-size auto-trim threshold lowered to 10 MB, `flush_memory()` on shutdown |
| **v3.20** | Nuclear threshold for oversized memory.json — files above 500 MB are replaced with empty entries instead of parsed (prevents OOM/freeze on startup); safety guard in `_load_local_memory_entries()` also refuses to parse files above the threshold |
| **v3.19** | Fix double mutation in rare reproduction — `rare_reproduction_cycle()` was applying `spawn_child(variance=0.10)` then `population.spawn_child()` applied `spawn_child(variance=0.05)` again; now `population.spawn_child()` accepts `mutation_variance` parameter (default 0.05), rare repro passes parent genome directly with `mutation_variance=0.10` for single-pass mutation |
| **v3.18** | Ecosystem stabilisation — resource pool minimum floor (`resource_pool_min_floor`, production: 50), energy-gated reproduction (auto ≥ 0.50, stability ≥ 0.60, lone survivor ≥ 0.40, cost 0.20), reproduction probability scales with `pool_fraction`, `deduct_energy()` helper |
| **v3.17** | 500-cycle birth cooldown per parent (BIRTH_COOLDOWN_CYCLES=500), event idempotency, tick ordering (death before reproduction), alive/dormant re-check at spawn, concurrency guard (threading.Lock wrapping tick) |
| **v3.16** | Dormant wake (pool > 50%), stability reproduction (pool > 80%), lone survivor reproduction (pool > 70%), carrying capacity scales with pool |
| **v3.15** | Dormant organisms, emergency regen (pool < 25%), adaptive metabolism (cost scales with scarcity) |
| **v3.14** | Population floor raised to 20% of max (auto-maintained) |
| **v3.13** | Absolute population cap enforcement (ABSOLUTE_POPULATION_CAP=60) |
| **v3.12** | Novelty metric (genome distance), evolutionary stagnation detection (window=100, threshold=0.05), population diversity scoring, evolution dashboard endpoint, innovation tracking in all 5 reproduction paths |
| **v3.11** | Anti-monoculture culling, environmental shock events, explorer bonus, stability reproduction, family tree, species divergence (threshold=0.35) |
| **v3.10** | Portable child export/import (HMAC-SHA256 signed JSON, v1.0 schema), multi-objective fitness (survival 0.35 + efficiency 0.30 + stability 0.20 + adaptation 0.15) |
| **v3.9** | Multi-objective fitness scoring, memory drift, 51→64 API endpoints, rare reproduction (5% gate, 2000-cycle cooldown) |
| **v3.8** | Stagnation breaker, founder protection (energy ≥ 0.25, fitness ≥ 0.15, mutation cap 0.08) |
| **v3.7** | Trait floor (≥ 0.02), variance kick (random perturbation to break zero-variance lock) |
| **v3.6** | Brain rewrite — deterministic local analysis with optional OpenAI, structured cycle logging |
| **v3.5** | Energy floor, parent reserve (cost 0.30 on reproduction) |
| **v3.4** | Cycle stats (per-cycle metrics tracking) |
| **v3.0** | Population system (multi-organism registry), behavior detection, selection pressure |
| **v1.2** | SQLite persistence, API authentication |

---

## Error Handling

| Code | Meaning |
|------|---------|
| 200 | Success |
| 401 | Missing or invalid API key |
| 422 | Request validation failed (bad input) |
| 500 | Internal server error (safe message returned) |

---

## License

This project is licensed under the MIT License.
Copyright (c) 2026 Joshua Randy Tablit Nazareno
