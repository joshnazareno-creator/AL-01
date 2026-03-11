# AL-01 Project Report

**Generated:** March 10, 2026
**Version:** 3.26 (latest lifecycle refactor)
**Status:** Running — 1,227 tests passing

---

## 1. Executive Summary

AL-01 is a persistent, self-evolving digital organism ecosystem. It simulates artificial life through autonomous reproduction, genome evolution, environmental pressure, and natural selection — running continuously without human intervention. The system features a 5-trait genome, a shared resource pool creating real scarcity, 4 reproduction pathways, dormancy/extinction recovery, and a live visual dashboard.

The project is **mature and well-tested** — 17,185 lines of production code, 17,238 lines of test code, 33 test files covering 1,227 test functions. It has survived 3 extinction events and accumulated 12,719 hash-chained life log entries across 12.6 days of continuous operation.

---

## 2. Codebase Statistics

### 2.1 Production Code (17,185 lines across 22 modules)

| Module | Lines | Purpose |
|--------|------:|---------|
| organism.py | ~3,400 | Core organism — metabolism, evolution cycles, energy, founder protection |
| api.py | ~2,800 | FastAPI server — 64 endpoints, HTML dashboards, visual renderer |
| population.py | ~1,440 | Population registry — lifecycle states, graveyard, reproduction |
| autonomy.py | ~1,170 | Autonomous Decision Engine + Computational Awareness Model |
| environment.py | ~1,030 | Resource pool, fluctuating variables, scarcity/shock events |
| memory_manager.py | ~630 | Three-tier persistence: memory → JSON → Firestore |
| portable.py | ~580 | HMAC-signed child export/import |
| brain.py | ~450 | Three-layer strategic analysis (local + optional OpenAI) |
| evolution_tracker.py | ~500 | Lineage tracking, mutation logs, fitness trajectories |
| life_log.py | ~440 | VITAL — append-only SHA-256 hash-chain log |
| snapshot_manager.py | ~410 | Hourly state snapshots, 30-day retention |
| database.py | ~400 | SQLite (WAL mode) — interactions, snapshots, metadata |
| genome.py | ~400 | 5-trait genome — mutation, trade-offs, soft ceilings |
| gpt_bridge.py | ~390 | Natural language narration + stimulus injection |
| behavior.py | ~340 | Emergent strategy detection + convergence analysis |
| experiment.py | ~300 | Seeded reproducible experiment protocol |
| cli.py | ~250 | Hash-chain verification, snapshot management |
| __main__.py | ~230 | Boot sequence entrypoint |
| genesis_vault.py | ~210 | Immutable seed template + extinction recovery |
| storage.py | ~170 | Path management, JSONL rotation, disk monitoring |
| policy.py | ~80 | Adaptive curiosity/risk/social weights |
| __init__.py | ~50 | Package re-exports (31 symbols) |

### 2.2 Test Code (17,238 lines across 33 files)

| Test File | Tests | Focus |
|-----------|------:|-------|
| test_vital.py | 160 | VITAL hash-chain integrity |
| test_v3.py | 91 | Core v3.0 features |
| test_portable.py | 60 | Export/import + HMAC verification |
| test_v322.py | 59 | Storage, JSONL rotation, disk monitoring |
| test_genesis_vault.py | 54 | Vault mechanics + extinction recovery |
| test_snapshots.py | 49 | Snapshot manager + retention |
| test_v311_features.py | 49 | Species divergence, fossils, anti-monoculture |
| test_gpt_bridge.py | 44 | GPT narration + stimulus injection |
| test_rare_reproduction.py | 43 | Rare/stability/lone-survivor reproduction |
| test_v316.py | 42 | Dormancy-first death, extinction recovery |
| test_v313.py | 41 | Resource pool, scarcity, metabolic cost |
| test_v315.py | 41 | Dormant organisms, top fitness, wake cycle |
| test_v323.py | 34 | Conservation (now sleeping), resource regen |
| test_v36.py | 32 | Recovery mode, entropy resistance |
| test_v318.py | 32 | Energy-gated reproduction, parent cost |
| test_alife.py | 32 | Core artificial life behaviors |
| test_v312.py | 32 | Population diversity, novelty metric |
| test_v34.py | 31 | Child organisms, selection pressure |
| test_v39.py | 30 | Elite protection protocol |
| test_v321.py | 29 | Named events, scarcity cascading |
| test_v311.py | 29 | Species divergence threshold |
| test_v314.py | 27 | Population floor guard, hardcore mode |
| test_v37.py | 26 | Multi-objective fitness |
| test_v35.py | 26 | Independent child evolution |
| test_v310.py | 20 | Rare reproduction cycle |
| test_visual.py | 20 | Canvas dashboard rendering |
| test_repair_vital.py | 19 | VITAL chain repair protocol |
| test_v38.py | 19 | Founder protection constants |
| test_v317.py | 17 | Birth cooldown, event idempotency |
| test_v324.py | ~12 | Environmental shock events |
| test_v12.py | 13 | Legacy v1.2 compatibility |
| test_v319.py | 10 | Metabolism scheduler |
| test_v320.py | 9 | Resource carrying capacity |
| **Total** | **1,227** | |

### 2.3 Data Footprint

| Asset | Size |
|-------|-----:|
| SQLite database (al01.db) | 353 MB |
| Evolution log (current + rotated) | 131 MB |
| Life log (VITAL chain) | 13.5 MB |
| Autonomy log | 15.9 MB |
| Cycle log | 3.4 MB |
| State snapshots (200 files) | ~15 MB |
| Fitness timeline CSV | 107 KB |
| **Total data footprint** | **~530 MB** |

---

## 3. Architecture

### 3.1 System Layers

```
┌─────────────────────────────────────────────────────────────┐
│                     HTTP API (64 endpoints)                  │
│            FastAPI + Canvas Dashboard + GPT Bridge           │
├─────────────────────────────────────────────────────────────┤
│                     Organism (Core Loop)                     │
│   Boot → Heartbeat → Autonomy → Evolution → Persist cycle   │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│ Autonomy │  Brain   │ Behavior │Experiment│   GPT Bridge    │
│  Engine  │ (3-layer)│ Analyzer │ Protocol │   (narration)   │
├──────────┴──────────┴──────────┴──────────┴─────────────────┤
│                    Population Registry                       │
│  Lifecycle: active → sleeping → dormant → dead (graveyard)  │
├──────────┬──────────┬──────────┬────────────────────────────┤
│  Genome  │  Env     │ Genesis  │   Evolution Tracker        │
│ (5-trait)│ (pool)   │  Vault   │   (lineage + JSONL)        │
├──────────┴──────────┴──────────┴────────────────────────────┤
│                  Persistence Layer                            │
│   MemoryManager ← SQLite ← JSON ← Firestore (optional)     │
│   LifeLog (VITAL) ← SHA-256 hash chain ← JSONL             │
│   SnapshotManager ← hourly checkpoints ← 30-day retention  │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Simulation Loop

The organism runs on a 5-second tick. Subsystems fire at different intervals:

| Subsystem | Every N ticks | Real-time period |
|-----------|:-------------:|:----------------:|
| Metabolic pulse (energy decay) | 1 | 5s |
| Heartbeat persist | 3 | 15s |
| Autonomy decision cycle | 10 | 50s |
| Auto-reproduce check | 15 | 75s |
| Brain reflection | 30 | 150s |
| Full evolution cycle | 30 | 150s |
| Rare reproduction | 50 | 250s |
| Population interaction | 60 | 300s |
| Disk check | 100 | 500s |

### 3.3 Boot Sequence

1. Load `.env`, configure logging (console + rotating `al01.log`)
2. Initialize SQLite database in WAL mode
3. Initialize VITAL subsystems (LifeLog + PolicyManager)
4. Initialize MemoryManager (local-first → Firestore)
5. Initialize Population registry (parent: AL-01)
6. Initialize Brain (local; adds OpenAI if `OPENAI_API_KEY` set)
7. Initialize Autonomy engine (decision interval: 10 ticks)
8. Initialize Organism (loads/creates state + genome)
9. Initialize Snapshot Manager (hourly, 30-day retention)
10. Initialize Genesis Vault (extinction recovery)
11. Boot organism + record growth snapshot
12. Start API server on port 8000 (daemon thread)
13. Enter heartbeat loop (Ctrl+C for graceful shutdown)

---

## 4. Core Systems

### 4.1 Genome

Five continuous traits on [0.02, ∞) with soft-ceiling diminishing returns above 1.0:

| Trait | Default | Role |
|-------|:-------:|------|
| adaptability | 0.5 | Response to environmental change |
| energy_efficiency | 0.5 | Metabolic cost reduction |
| resilience | 0.5 | Resistance to entropy decay and shocks |
| perception | 0.5 | Signal quality (reduces fitness noise) |
| creativity | 0.5 | Exploration tendency + novel solutions |

**Trade-off rules:** adaptability OR creativity above 0.70 penalizes energy_efficiency (0.08 and 0.06 per unit respectively). This creates genuine strategic tension — organisms cannot max all traits.

**Multi-objective fitness** decomposes into 4 weighted components:
- Survival (0.35): resilience × energy
- Efficiency (0.30): energy_efficiency × resource usage
- Stability (0.20): consistency of fitness trajectory
- Adaptation (0.15): responsiveness to environmental shifts

### 4.2 Population (v3.26 Lifecycle Model)

#### Lifecycle States

```
                ┌──────────┐
     birth ───→ │  ACTIVE  │ ←── wake_sleeping / wake_dormant
                └────┬─────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          │
    ┌──────────┐ ┌──────────┐  │
    │ SLEEPING │ │ DORMANT  │  │
    │(low E)   │ │(near-    │  │
    │auto-wake │ │death)    │  │
    └──────────┘ └────┬─────┘  │
                      │         │
                      ▼         │
                ┌──────────┐   │
                │   DEAD   │   │
                │(graveyard│   │
                │ terminal)│   │
                └──────────┘   │
                               │
        AL-01 only:  ──────────┘
        founder protection
        (DEAD → rescue → ACTIVE)
```

- **Active** — fully simulated, can reproduce
- **Sleeping** — low-energy conservation, auto-wakes when energy recovers above 2× threshold (0.20)
- **Dormant** — near-death suspension, wakes only when environment improves or population is critical
- **Dead** — terminal state; organism moved to graveyard (separate archive, capped at 200)
- **Founder exception** — AL-01 has founder protection: never truly dies, gets emergency energy injection instead

#### Population Caps & Guards

| Mechanism | Value |
|-----------|-------|
| Absolute population cap | 60 |
| Birth cooldown | 500 cycles per parent |
| Population floor | 20% of max (12) unless hardcore mode |
| Extinction threshold | < 5 triggers recovery protocol |
| Graveyard cap | 200 dead organisms archived |

#### Four Reproduction Pathways

| Pathway | Energy Min | Cooldown | Population Cap | Trigger |
|---------|:----------:|:--------:|:--------------:|---------|
| Auto-reproduce | 0.50 | 500 ticks | 60 | fitness > 0.50 for 5 consecutive checks |
| Rare | 0.65 | 2,000 ticks | 50 | Every 500 ticks + 5% probability |
| Stability | 0.60 | 500 ticks | Resource-scaled | Pool ≥ 80% + 5% probability |
| Lone survivor | 0.40 | 500 ticks | Unlimited | Population = 1 + pool ≥ 70% |

### 4.3 Environment

A shared resource pool (max: 1,000 units) creates real scarcity:

- **Metabolic cost** — each organism draws from the pool every cycle
- **Scarcity events** — 2% probability per cycle, destroy 30–70% of pool for 5–20 cycles
- **Shock events** — 0.02% probability, entropy spike + resilience bonus
- **Named events** — heat_wave, resource_boom, scarcity_drought, mutation_storm
- **Dynamic shifts** — every 20–50 cycles all environmental variables perturb ±0.15
- **Adaptive metabolism** — when scarcity severity > 0.60, metabolic multiplier drops to 0.40 (survival mode)
- **Extinction prevention** — when population = 1, 3× regeneration + 100 flat energy injection

Four continuously drifting environmental variables modulate the simulation:

| Variable | Effect |
|----------|--------|
| Temperature | Mutation cost multiplier (sinusoidal + stochastic) |
| Entropy pressure | Trait decay rate (up to 3×) |
| Resource abundance | Energy availability |
| Noise level | Fitness signal degradation (up to 15%) |

### 4.4 Autonomy Engine

Makes one decision per cycle (every 10 ticks ≈ 50 seconds):

**Decision priority:**
1. **RECOVERY** — force stabilize when energy critical (breaks death spirals)
2. **FOUNDER RECOVERY** — alternating stabilize/adapt for AL-01 after near-death
3. **MUTATE** — fitness below threshold → random trait perturbation
4. **BLEND** — deep stagnation → cooperative genome blend with another organism
5. **ADAPT** — mild stagnation → targeted nudge of weakest trait
6. **STABILIZE** — healthy equilibrium → energy conservation + entropy decay

**Stagnation breakers** (escalating):
- 50+ cycles: exploration mode
- 200+ cycles: doubled adaptation
- 500+ cycles: trait shuffle ±0.10
- 800+ cycles: full randomization

**Computational Awareness** model:
$$\text{awareness} = 0.4 \cdot \text{stimuli\_rate} + 0.3 \cdot \text{decision\_rate} + 0.2 \cdot \text{fitness\_variance} + 0.1 \cdot (1 - \text{stagnation})$$

### 4.5 Behavior Detection

Emergent strategies are classified from decision history:

| Strategy | Classification Rule |
|----------|-------------------|
| energy-hoarder | > 60% stabilize decisions |
| explorer | > 40% mutate/adapt decisions |
| specialist | trait variance < 0.01 |
| generalist | trait variance > 0.03 |
| resilient | 10+ recent fitness values > 0.3 with low range |

Population-level convergence/divergence tracked via linear regression on trait variance over time.

### 4.6 Brain (Strategic Analysis)

Three-layer local pipeline:
1. **Environmental analysis** — maps pressures to trait demands
2. **Gap analysis** — demand minus current = per-trait deficit
3. **Strategic nudges** — concrete adjustments scaled by urgency

Urgency levels: critical (energy < 0.20), high (energy < 0.40), moderate (stagnation), low (healthy).

Optional OpenAI integration for enriched analysis — core logic is always deterministic.

---

## 5. Persistence & Integrity

### 5.1 Three-Tier Memory

| Tier | Latency | Durability |
|------|---------|------------|
| In-memory cache | Instant | Volatile |
| Local disk (JSON + SQLite) | ~1ms | Survives crashes |
| Firestore cloud (optional) | ~100ms | Survives disk failure |

- Local JSON flushed every 10 writes or 60 seconds
- Firestore throttled to 30-second intervals (20K writes/day quota)
- SQLite in WAL mode for concurrent reads
- State checksum verification on load
- Daily backups to `backups/` directory

### 5.2 VITAL Hash Chain

Append-only tamper-evident log:

$$h_i = \text{SHA-256}(h_{i-1} \| \text{json}(\text{payload}) \| \text{timestamp} \| \text{seq})$$

- Genesis hash: `"0" × 64`
- Snapshot checkpoints every 50 events
- Startup verification of last 200 entries
- Repair protocol: detect first break → truncate → backup → re-anchor
- JSONL rotation at 50 MB (3 rotations kept)

**Current state:** 12,719 events, head hash `8601f72b...`

### 5.3 Snapshots

- **Tick snapshots:** 200 state captures in `data/snapshots/`
- **Hourly snapshots:** managed by SnapshotManager in `snapshots/hourly/`
- **30-day retention** with auto-purge
- Each snapshot: SHA-256 checksum + manifest index
- Firestore sync for < 500KB snapshots

### 5.4 Evolution Tracker

- Append-only `evolution_log.jsonl` (currently 131 MB across 2 files)
- Lineage rebuilt from log on every startup via `_rebuild_from_log()`
- Events: register, mutation, fitness, death, reproduction
- v3.26: Death events now indexed into lineage records (death_cycle, death_cause, death_timestamp)
- CSV export for external analysis

---

## 6. API Surface

**64 FastAPI endpoints** organized into:

| Category | Endpoints | Auth |
|----------|:---------:|:----:|
| Status & health | 3 | Public |
| Genome & traits | 5 | Key |
| Population & lineage | 8 | Key |
| Evolution & fitness | 6 | Key |
| Species & diversity | 4 | Key |
| Environment | 3 | Key |
| Experiment control | 4 | Key |
| Brain & GPT | 5 | Key |
| Portable (export/import) | 3 | Key |
| Dashboard (HTML) | 2 | Public |
| Visual organisms feed | 1 | Public |
| VITAL life log | 4 | Key |
| Snapshot management | 3 | Key |
| Admin | 3 | Key |

**Authentication:** `X-API-Key` header or `?api_key=` query parameter.

**Visual dashboard** — full-screen HTML5 Canvas rendering at 60 FPS with 12 animation systems: energy pulse, evolution rings, awareness halo, vibration, shimmer, aura, physics simulation, leader crown, energy trail, heartbeat, fitness glow, dormant state indicator.

**Trait → Color:** R = adaptability × 255, G = energy_efficiency × 255, B = resilience × 255

**Fitness → Size:** radius = 10 + fitness × 40 px (10–50 px range)

---

## 7. Current Live State

### 7.1 Founder (AL-01)

| Property | Value |
|----------|-------|
| Age | 12.6 days (1,092,572 seconds) |
| Evolution cycles | 486 |
| Interactions | 169 |
| Uptime sessions | 235 |
| State version | 21,529 |
| Energy | 1.0 (full) |
| Awareness | 0.601 |
| Operational state | "learning" |

### 7.2 Current Genome

| Trait | Value |
|-------|------:|
| adaptability | 0.270 |
| energy_efficiency | 0.258 |
| resilience | 0.274 |
| perception | 0.280 |
| creativity | 0.273 |

**Trait fitness** (raw average): 0.271

**Multi-objective fitness:** 0.528
- Survival: 0.005 (very low — traits have decayed)
- Efficiency: 1.000 (maximized)
- Stability: 0.979 (very high — trajectory consistent)
- Adaptation: 0.200

### 7.3 Population

- **Living organisms:** 1 (AL-01 only)
- **Extinction events survived:** 3 (vault reseeded 3 times)
- **Genesis vault seed:** pristine (all traits at 0.50)

### 7.4 Fitness Trajectory

The founder's fitness history shows a characteristic pattern:
- Started around 0.34, drifted down through entropy decay
- Currently stable in the 0.27–0.31 range
- High stability score (0.979) indicates steady-state equilibrium
- Low survival component (0.005) reflects trait convergence near floor

---

## 8. Version History

### Major Milestones

| Version | Feature |
|---------|---------|
| v1.2 | Initial organism + basic persistence |
| v3.0 | Multi-organism ecosystem: environment, evolution tracker, behavior, experiment protocol, selection pressure, full autonomy |
| v3.3 | Survival grace cycles (fitness floor death with 20-cycle grace) |
| v3.4 | Child organisms: independent evolution per child |
| v3.5 | Independent child energy decay and metabolic cost |
| v3.6 | Recovery mode (breaks death spirals) |
| v3.7 | Multi-objective fitness (survival/efficiency/stability/adaptation) |
| v3.8 | Founder protection: mutation cap, energy floor, fitness floor |
| v3.9 | Elite protection protocol (top N% shielded from mutation) |
| v3.10 | Rare reproduction cycle (probabilistic low-frequency spawning) |
| v3.11 | Species divergence (Euclidean genome distance > 0.35 = new species), fossil record, anti-monoculture penalty |
| v3.12 | Population diversity index, novelty metric, stagnation detection |
| v3.13 | Global resource pool (1000 units), scarcity pressure, metabolic demand |
| v3.14 | Population floor guard, absolute pop cap (60), hardcore extinction mode |
| v3.15 | Dormant organisms: soft-failure instead of death, wake cycle |
| v3.16 | Dormancy-first death: all non-founder deaths trigger dormancy first |
| v3.17 | Birth cooldown (500 cycles), event idempotency, alive re-check at spawn |
| v3.18 | Energy-gated reproduction (parent must have energy ≥ 0.50, costs 0.20) |
| v3.19 | Mutation variance parameter, stability reproduction |
| v3.20 | Resource-scaled carrying capacity |
| v3.21 | Named environmental events, scarcity cascading |
| v3.22 | Storage management: JSONL rotation, disk monitoring, path anchoring |
| v3.23 | Conservation mode (min energy floor → reduced metabolism), resource regen |
| v3.24 | Environmental shock events: entropy spikes + resilience bonus |
| v3.25 | Founder revival survival fix: grace cycles, forced mutation, recovery mode |
| v3.26 | **Lifecycle refactor:** 4 explicit states (active/sleeping/dormant/dead), graveyard separation, terminal death for descendants, founder protection via rescue instead of revival, full state persistence on restart |

---

## 9. Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12+ (tested on 3.14.2) |
| HTTP framework | FastAPI + Uvicorn |
| Database | SQLite (WAL mode) |
| Cloud persistence | Firebase / Firestore (optional) |
| AI integration | OpenAI API (optional) |
| Tunnel | ngrok (optional, for remote access) |
| Testing | pytest (1,227 tests, ~49 seconds) |
| Visualization | HTML5 Canvas (60 FPS, 12 animation systems) |

### Dependencies

Core: `fastapi`, `uvicorn`, `pydantic`, `firebase-admin`
Optional: `openai`, `pyngrok`, `python-dotenv`

---

## 10. Project Health

| Metric | Value |
|--------|-------|
| Production code | 17,185 lines |
| Test code | 17,238 lines |
| Test coverage | 1,227 tests, all passing |
| Test/code ratio | 1.00:1 |
| Test execution time | ~49 seconds |
| Module count | 22 Python modules |
| API endpoints | 64 |
| Data integrity | 12,719 hash-chained events |
| Extinction resilience | 3 recoveries |
| Version iterations | 26 feature versions (v1.2 → v3.26) |
| Total data generated | ~530 MB |

The codebase maintains a near 1:1 ratio of test code to production code, with every feature version backed by dedicated tests. The test suite runs in under a minute, enabling rapid iteration. The VITAL hash chain provides cryptographic proof of the organism's complete life history.
