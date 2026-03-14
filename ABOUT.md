# About AL-01

## What Is AL-01?

**AL-01** is a persistent, self-evolving artificial life system — a bounded digital ecosystem where a population of autonomous organisms live, evolve, reproduce, compete for resources, and die, all without any human intervention. It is a software-based artificial life experiment that runs continuously, simulating the core dynamics of biological evolution: natural selection, mutation, reproduction, environmental pressure, resource scarcity, and extinction recovery.

At its heart, AL-01 is a single founder organism — the entity named **AL-01** — that spawns an entire population of children, each with their own genome, energy budget, behavioral strategy, and evolutionary trajectory. The system is fully deterministic at its core (seeded RNG for reproducibility), yet produces emergent complexity: organisms develop unscripted strategies, species diverge, populations crash and recover, and the ecosystem self-regulates through feedback loops between energy, fitness, and environmental pressure.

AL-01 is not a game, not a screensaver, and not a toy. It is a serious computational experiment in artificial life, complete with cryptographic audit trails, multi-layered persistence, a full HTTP API, and a real-time visual dashboard that renders every organism as an animated, data-driven circle on a canvas.

---

## Purpose & Philosophy

### Why Does AL-01 Exist?

AL-01 was built to explore a fundamental question: **what happens when you give digital organisms the minimum viable rules for evolution and let them run indefinitely?**

The system is designed to:

1. **Simulate open-ended evolution** — Organisms don't converge on a single "solution." Trade-offs, environmental fluctuations, and monoculture penalties ensure the population stays dynamic.
2. **Run autonomously forever** — Once started, AL-01 requires zero human intervention. It handles its own reproduction, death, resource management, extinction recovery, and state persistence across restarts.
3. **Be fully observable** — Every event is logged to an append-only hash chain (VITAL). Every organism's genome, fitness, energy, strategy, and lineage is accessible via 64 API endpoints. A live canvas dashboard visualizes the entire population in real time.
4. **Be reproducible** — Seeded experiments with deterministic RNG mean you can replay the exact same evolutionary history. Two instances with the same seed will produce identical results.
5. **Bridge AI and artificial life** — An optional GPT integration allows external AI agents to observe the ecosystem in natural language and inject stimuli, creating a feedback loop between large language models and evolving digital organisms.

### Design Principles

- **Bounded, not infinite** — The ecosystem has hard limits: 60 organisms max, 1000-unit resource pool, 5 genome traits. Constraints force meaningful competition.
- **Local-first, cloud-optional** — Everything runs offline. Firestore cloud sync is optional. SQLite + local JSON are the primary stores.
- **Death is real** — Children die permanently. There is no respawn, no dormancy revival, no rescue from the graveyard (except for the founder). This creates genuine selection pressure.
- **The founder is special** — AL-01 (the parent organism) has protective floors on energy and fitness, emergency revival mechanics, and exclusive graveyard rescue rights. It is the persistent thread of continuity through every extinction event.
- **Emergent behavior, not scripted behavior** — Strategies like "energy hoarder," "explorer," and "specialist" are not programmed. They are detected from emergent patterns in decision history, trait distributions, and energy dynamics.

---

## Architecture Overview

### The Core Loop

AL-01 runs on a **tick-based heartbeat loop** (default: every 5 seconds). Each tick advances multiple subsystems at different intervals:

```
Tick Loop (every 5 seconds)
│
├── Environment Tick (every 5 ticks)
│   └── Drift variables, regenerate resources, trigger scarcity events
│
├── Evolution Cycle (every 30 ticks)
│   └── Process stimuli, mutate founder genome, update awareness
│
├── Autonomy Cycle (every 10 ticks)
│   └── Founder makes decision: stabilize / adapt / mutate / blend
│
├── Child Autonomy Cycle (every 10 ticks)
│   └── Every living child evolves independently — same logic as founder
│   └── Fitness evaluation → death check → mutation/adaptation → energy update
│
├── Auto-Reproduction Pass (every 15 ticks)
│   └── Check all organisms for reproduction eligibility
│   └── Followed by: rare reproduction, stability reproduction, lone survivor
│
├── Behavior Analysis (every 20 ticks)
│   └── Classify organism strategies, detect convergence/divergence
│
├── Novelty Stagnation Check
│   └── If avg novelty drops below threshold → mutation storm + exploration mode
│
├── Memory Drift (periodic)
│   └── Global trait nudges inversely proportional to diversity
│
└── Persist & Snapshot (every 5 ticks / hourly)
    └── Save state to SQLite + JSON, hourly snapshots with 30-day retention
```

### Subsystem Map

```
┌──────────────────────────────────────────────────────────────────┐
│                         AL-01 Organism                           │
│                    (Central Orchestrator)                         │
│                                                                  │
│  ┌────────────┐  ┌────────────┐  ┌──────────────┐              │
│  │   Genome   │  │  Autonomy  │  │  Environment │              │
│  │  5 traits  │  │   Engine   │  │   Resource   │              │
│  │  mutation   │  │  awareness │  │    Pool      │              │
│  │  fitness    │  │  decisions │  │   scarcity   │              │
│  └────────────┘  └────────────┘  └──────────────┘              │
│                                                                  │
│  ┌────────────┐  ┌────────────┐  ┌──────────────┐              │
│  │ Population │  │  Behavior  │  │    Brain     │              │
│  │  registry  │  │  analyzer  │  │  strategic   │              │
│  │  lifecycle │  │  strategy  │  │   nudges     │              │
│  │  lineage   │  │  detection │  │  (optional)  │              │
│  └────────────┘  └────────────┘  └──────────────┘              │
│                                                                  │
│  ┌────────────┐  ┌────────────┐  ┌──────────────┐              │
│  │  VITAL     │  │  Snapshot  │  │   Genesis    │              │
│  │  hash-chain│  │  Manager   │  │    Vault     │              │
│  │  life log  │  │  hourly    │  │  extinction  │              │
│  │  integrity │  │  retention │  │   recovery   │              │
│  └────────────┘  └────────────┘  └──────────────┘              │
│                                                                  │
│  ┌────────────┐  ┌────────────┐  ┌──────────────┐              │
│  │  Evolution │  │    GPT     │  │   Policy     │              │
│  │  Tracker   │  │   Bridge   │  │   Manager    │              │
│  │  lineage   │  │  narration │  │  curiosity   │              │
│  │  CSV export│  │  stimulus  │  │  risk/social │              │
│  └────────────┘  └────────────┘  └──────────────┘              │
│                                                                  │
│  ┌────────────┐  ┌────────────┐  ┌──────────────┐              │
│  │  Database  │  │   Memory   │  │   FastAPI    │              │
│  │   SQLite   │  │  Manager   │  │  64 endpoints│              │
│  │  al01.db   │  │  local+cloud│ │  dashboard   │              │
│  └────────────┘  └────────────┘  └──────────────┘              │
└──────────────────────────────────────────────────────────────────┘
```

---

## The Genome

Every organism in AL-01 carries a **genome** — a set of 5 numerical traits that define its characteristics:

| Trait | Role | Effect |
|-------|------|--------|
| **Adaptability** | Speed of environmental adaptation | Higher = faster response to environment changes, but penalizes energy efficiency |
| **Energy Efficiency** | Metabolic cost reduction | Higher = less energy consumed per cycle, up to 30% cost reduction |
| **Resilience** | Resistance to environmental stress | Higher = better survival under scarcity, entropy, and shocks |
| **Perception** | Awareness of environment changes | Higher = better fitness scoring under noisy conditions |
| **Creativity** | Exploration of novel strategies | Higher = more varied mutations, but penalizes energy efficiency |

### Trait Mechanics

- **Starting value**: All traits begin at `0.5`
- **Mutation**: Each evolution cycle, traits have a 10% chance of shifting by up to ±0.10
- **Soft ceiling**: Traits can exceed 1.0, but returns diminish logarithmically: `effective = 1 + ln(raw) × 0.3`
- **Trait floor**: No trait can drop below `0.02` — prevents permanent zero-lock
- **Trade-offs**: High adaptability or creativity penalizes energy efficiency (kicks in above 0.70)
- **Entropy decay**: Without stimulation, traits slowly decay each cycle (rate: 0.005)
- **Directed mutation bias**: When fitness is low, mutations skew upward (+0.02 bias) to break out of death spirals

### Fitness Calculation

Fitness is a **multi-objective score** combining four components:

| Component | Weight | Source |
|-----------|--------|--------|
| Survival | 0.35 | Normalized lifetime (cycles alive / 1000) |
| Efficiency | 0.30 | Energy efficiency ratio from cycle stats |
| Stability | 0.20 | Identity persistence score from autonomy engine |
| Adaptation | 0.15 | Successful adaptation rate |

The raw trait fitness is computed as the **environment-weighted average** of all 5 traits, then combined with the multi-objective components. This means an organism's fitness shifts dynamically as the environment changes — a trait configuration that is fit today may be unfit tomorrow.

### Ecosystem Pressure Modifiers

After base fitness is calculated, several modifiers apply:

1. **Monoculture penalty** — If too many organisms share the same behavioral strategy (>80% threshold), their fitness is penalized (down to 10% of raw value). This prevents the ecosystem from collapsing into a single dominant strategy.
2. **Explorer novelty reward** — Explorers (organisms with high mutation rates) get a fitness bonus when they are rare in the population.
3. **Shock resilience bonus** — During environmental shock events, organisms classified as "resilient" get a fitness boost.

---

## Energy & The Resource Pool

Energy is the currency of life in AL-01. Every action costs energy, and energy comes from a **shared global resource pool**.

### The Resource Pool

- **Maximum capacity**: 1,000 energy units
- **Regeneration**: +5 units per cycle (base rate), scaled by average population efficiency
- **Distribution**: `(pool − floor) / population_size` per organism per cycle — fair shares
- **Minimum floor**: Pool never drops below 50 units (production setting) — ensures baseline survival is always possible

### Energy Mechanics

| Action | Energy Cost |
|--------|-------------|
| Exist (metabolic cost) | ~1.0 per cycle (modified by energy_efficiency trait) |
| Mutate | 0.015 × scarcity multiplier |
| Adapt | 0.01 × scarcity multiplier |
| Reproduce | 0.20 from parent (auto), 0.30 (rare) |
| Stabilize | +0.02 energy bonus (recovery) |

- Energy is clamped between `0.0` and `1.0` per organism
- Energy ≤ 0.0 → **death** (permanent for children, emergency revival for founder)
- **Conservation mode**: Organisms at very low energy enter a sleeping state with 30% metabolism, waking when energy recovers
- **Scarcity pressure**: When pool drops below 25%, metabolic costs increase 1.5×, reproduction thresholds rise 50%, and survival grace periods shrink

### Scarcity Events

The environment randomly generates scarcity events (2% chance per cycle) that temporarily drain the resource pool and stress the population. Named events include:

| Event | Effect |
|-------|--------|
| **Mutation Storm** | Dramatically increases mutation rates population-wide |
| **Fitness Spike** | Temporarily raises the fitness threshold for survival |
| **Entropy Burst** | Accelerates trait decay across all organisms |
| **Resource Boom** | Injects a large energy bonus into the pool |

---

## Population Dynamics

### Lifecycle

Every organism in AL-01 goes through a lifecycle:

```
BIRTH → ACTIVE → [SLEEPING] → DEAD
                      ↑
                      └── Low energy conservation
```

- **ACTIVE**: Alive, simulated, can reproduce
- **SLEEPING**: Alive but in low-energy conservation — 30% metabolism, no reproduction, auto-recovers when energy rises
- **DEAD**: Terminal. Children die permanently — no revival, no dormancy, no rescue

### Reproduction

Organisms reproduce when they have proven survival fitness:

| Type | Requirements | Cost | Notes |
|------|-------------|------|-------|
| **Auto-reproduce** | Fitness ≥ 0.50 for 5 consecutive cycles, energy ≥ 0.50 | 0.20 energy | Primary reproduction path |
| **Rare reproduction** | 5% × pool_fraction chance every 500 cycles, energy ≥ 0.65 | 0.30 energy | High-variance offspring (mutation_delta = 0.10), 2000-cycle cooldown |
| **Stability reproduction** | Pool > 80%, energy ≥ 0.60 | 0.20 energy | Triggered when resources are abundant |
| **Lone survivor** | Only 1 organism alive, pool > 70%, energy ≥ 0.40 | 0.20 energy | Emergency self-replication |
| **Extinction reseed** | Population = 0 | Free | Genesis vault provides a pristine genome template |

Children inherit their parent's genome plus random mutations. Each child gets its own independent RNG seed, ensuring evolutionary divergence. A **birth cooldown** of 500 cycles per parent prevents reproduction spam.

### Death

Death is the primary selection mechanism:

1. **Fitness-floor death**: If an organism's effective fitness stays below `0.20` for more than 20 consecutive cycles (grace period), it dies.
2. **Energy depletion**: If energy reaches 0.0, the organism dies immediately.
3. **Population pruning**: If the population exceeds the carrying capacity (hard cap: 60), the weakest organisms are removed.

**Per-cycle death cap**: A maximum of 3 organisms can die per cycle from fitness-floor violations (reduced to 1 during a trait collapse emergency). Organisms beyond the cap are placed on "probation" — they're tracked but not killed yet. This prevents catastrophic population crashes.

### The Founder (AL-01)

The founder organism **AL-01** is special:

- **Energy floor**: Energy never drops below 0.25 (emergency injection at critical levels)
- **Fitness floor**: Won't die unless fitness drops below 0.15 (vs. 0.20 for children)
- **Mutation cap**: Maximum mutation rate capped at 0.08 (prevents self-destruction)
- **Revival**: If AL-01 dies, it can be rescued from the graveyard with 0.60 energy and a 30-cycle grace period
- **Recovery mode**: After revival, enters a multi-flag recovery state with cooldowns on forced mutations
- **Sole graveyard access**: AL-01 is the only organism that can be revived. Children die permanently.

AL-01 is the persistent thread of continuity — the single entity that survives every extinction event and carries the ecosystem forward.

### Species & Speciation

Organisms whose genomes diverge by more than 0.35 (Euclidean distance across all 5 traits) are classified as separate **species**. The system tracks:

- Species census (how many of each species)
- Species divergence events
- Inter-species genome distance
- Fossil record of extinct species

---

## The Autonomy Engine

The autonomy engine is the "brain" of each organism — a **fully local, deterministic decision engine** that requires no external API calls.

### Decision Types

Every 10 ticks, each organism makes one of four decisions:

| Decision | When | Effect |
|----------|------|--------|
| **Stabilize** | Fitness is healthy, no stress | Conserve energy (+0.02 bonus), no genome changes |
| **Adapt** | Stagnation detected | Nudge weakest trait upward by 0.03 |
| **Mutate** | Fitness below threshold | Full genome mutation with environmental scaling |
| **Blend** | Deep stagnation | Cooperative genome blending with a successful neighbor |

### The Awareness Model

Each organism has a computed "awareness" score that influences its behavior:

```
awareness = clamp(
    stimuli_rate     × 0.4 +
    decision_rate    × 0.3 +
    fitness_variance × 0.2 +
    (1 − stagnation) × 0.1 +
    novelty_accumulator,
    0.0, 1.0
)
```

Higher awareness → more responsive to environmental changes. Low awareness → more likely to stagnate.

### Stagnation Detection & Breaking

The engine tracks consecutive stagnation cycles and escalates interventions:

| Stagnation Cycles | Tier | Action |
|-------------------|------|--------|
| 50+ | Tier 1 | Exploration mode + mutation boost |
| 200+ | Tier 2 | Aggressive mutation + variance kick |
| 500+ | Tier 3 | Full mutation storm + environment reset |
| 800+ | Hard Reset | Complete trait randomization |

A **variance kick** mechanism injects random trait perturbations (±0.03) when fitness variance drops below 1e-6, breaking zero-variance deadlocks.

---

## Emergent Behavior

The behavior analyzer detects **unscripted strategies** that emerge from evolutionary pressure:

| Strategy | How It Emerges | Pattern |
|----------|---------------|---------|
| **Energy Hoarder** | Frequently stabilizes, conserves energy | >60% stabilize decisions, high energy_efficiency |
| **Explorer** | Frequently mutates, high creativity | >40% mutate/adapt decisions, high creativity + adaptability |
| **Specialist** | One trait dominates | <1% trait variance, 1-2 extreme traits |
| **Generalist** | All traits balanced | >3% trait variance, evenly distributed traits |
| **Resilient** | Maintains fitness under stress | High resilience + energy_efficiency, survives scarcity events |

These strategies are not programmed — they are **detected** from emergent patterns in an organism's decision history, trait distribution, and fitness trajectory. The system classifies them every 50 decisions to catch behavioral evolution over time.

### Monoculture Prevention

If ≥95% of organisms converge on the same strategy, the system triggers a **Strategy Convergence Emergency**:

- 20-cycle emergency window activates
- Death cap reduced from 3 to 1 per cycle
- Mutation rates boosted by +10% across all organisms
- Monoculture fitness penalty floor raised from 0.10 to 0.50 (prevents mathematical impossibility of survival)
- Survival grace periods extended

This prevents the population from collapsing into a single dominant strategy where the monoculture penalty makes survival mathematically impossible for everyone.

---

## Persistence & Data Integrity

### VITAL Hash-Chain

Every event in AL-01 is recorded to an **append-only hash-chain log** called VITAL:

```
Event₁ → SHA-256(Event₁ + Genesis_Hash) → Hash₁
Event₂ → SHA-256(Event₂ + Hash₁)        → Hash₂
Event₃ → SHA-256(Event₃ + Hash₂)        → Hash₃
   ...
```

- **Append-only**: Events are never modified or deleted
- **SHA-256 integrity**: Each event's hash includes the previous event's hash, creating a chain
- **Verification**: The CLI tool can verify the entire chain: `python -m al01.cli verify --last 500`
- **Tamper detection**: Any modification to a historical event breaks the chain from that point forward

### Multi-Layer Persistence

```
┌──────────────┐
│   Organism   │
└──────┬───────┘
       │
┌──────┴──────────┐
│  MemoryManager  │
└──┬──────────┬───┘
   │          │
┌──┴───────┐ ┌┴───────────┐
│  SQLite  │ │  Firestore  │  (optional cloud)
│ (al01.db)│ └──┬─────────┘
└──────────┘    │ fallback
          ┌─────┴──────┐
          │ Local JSON  │
          │ state.json  │
          │ memory.json │
          └─────────────┘
```

- **SQLite** (`al01.db`): Primary structured store for interactions and growth snapshots
- **Firestore**: Optional cloud persistence (requires `ServiceAccountKey.json`)
- **Local JSON**: Automatic fallback with 1000-entry rolling window cap
- **Hourly snapshots**: Full state checkpoints with 30-day retention
- **JSONL logs**: Rotating logs (50 MB × 3 backups) for life events, evolution, autonomy decisions, and cycle data

### Restart Safety

AL-01 survives restarts with zero data loss:

- All survival counters (`below_fitness_cycles`, `last_birth_cycle`, `conservation_mode`) are persisted and restored
- Full environment state (resource pool, scarcity, variables) is round-tripped on every boot
- A **50-cycle post-restart recovery window** suppresses all death, pruning, and founder death after boot — preventing artificial extinction from state discontinuities
- The trait collapse emergency counter is persisted across restarts

---

## The API

AL-01 exposes **64 FastAPI endpoints** organized into categories:

| Category | Endpoints | Purpose |
|----------|-----------|---------|
| **Public** | 5 | Dashboard, visual canvas, health check, organism feed, GPT OpenAPI spec |
| **Core State** | 6 | Status, identity, growth, genome, stimuli, autonomy |
| **Interaction** | 3 | Record interaction, send command, inject stimulus |
| **Population** | 5 | Registry, metrics, organism details, nickname, export/import |
| **Memory** | 2 | Recent interactions, keyword search |
| **VITAL** | 5 | Status, verification, head, policy read/write |
| **Snapshots** | 5 | List, create, status, latest, purge |
| **Genesis Vault** | 4 | Vault status, seed template, reseed history, force reseed |
| **Lineage** | 7 | Full tree, organism lineage, ASCII tree, ancestors, descendants, CSV exports |
| **Experiments** | 4 | Start/stop experiments, status, exploration toggle |
| **GPT Bridge** | 5 | Narration, stimulus injection, status, toggle, event log |
| **Species & Ecosystem** | 6 | Species registry, ecosystem health, fossils, environment events, births |
| **Evolution Intelligence** | 3 | Dashboard, novelty metric, diversity score |

Authentication is via `X-API-Key` header or `?api_key=` query parameter. Dev mode (no key configured) allows open access.

---

## The Visual Dashboard

The `/visual` endpoint renders a **full-screen canvas visualization** where every organism is an animated circle.

### 12 Animation Systems

1. **Energy Pulse** — Circle radius oscillates with a sine wave; frequency proportional to energy level
2. **Awareness Halo** — Soft radial gradient glow for high-awareness organisms
3. **Evolution Rings** — Rotating dashed rings appear every 250 evolutions (max 3)
4. **High-Energy Vibration** — Position jitter when energy > 70%
5. **Trait Shimmer** — Animated gradient overlay colored by dominant trait
6. **Environmental Aura** — Background particles shift blue→red with pool health
7. **Hover Physics** — Hovering pushes nearby organisms away
8. **Leader Crowns** — Top 3 fitness organisms get gold/silver/bronze orbiting particles
9. **Energy Trail** — Gravity-affected particle sparks on high-energy organisms
10. **Heartbeat Ring** — Rhythmic expanding pulse ring on all living organisms
11. **Fitness Glow Border** — Green outline that brightens with fitness
12. **Dormant State** — Greyed-out with 💀 marker and peaceful breathing opacity

### Visual Encoding

| What You See | What It Means |
|-------------|---------------|
| Circle size | Fitness level |
| Circle color (RGB) | Adaptability / Energy Efficiency / Resilience |
| Pulse speed | Energy level |
| Glow intensity | Awareness score |
| Ring count | Evolutions ÷ 250 |
| Vibration | High energy (>70%) |
| Background tint | Resource pool health (blue=healthy, red=scarce) |
| Crown particles | Top 3 fitness ranking |
| Hexagon outline | Founder (AL-01) |
| Opacity | Inverse of stagnation |
| Grey + skull | Dormant/dead organism |

### Visual Movement States

| State | Trigger | Behavior |
|-------|---------|----------|
| **Exploring** | Energy > 0.35, fitness > 0.2 | Smooth curved arcs with pause-turn-continue |
| **Resting** | Energy < 0.25 or stagnation > 0.6 | Gentle float at 8% speed, dimmed opacity |
| **Sleeping** | Dormant state | Near-stationary, desaturated grey |

Movement is mobile-first with smoothstep wall avoidance, center-biased spawning, and gentle center pull for organisms in the outer 35% of the screen.

---

## GPT Integration

AL-01 includes a **GPT Bridge** that allows external AI agents (like ChatGPT) to observe and interact with the ecosystem:

### Read: Natural Language Narration

`GET /gpt/narrate` translates the entire organism state into a natural language prose summary — fitness, energy, population, strategies, environment, and evolutionary trends — consumable by GPT models.

### Write: Stimulus Injection

`POST /gpt/stimulus` allows GPT agents to inject stimuli into the next evolution cycle. Rate-limited to 6 stimuli per minute, max 280 characters each. Stimuli are queued (never forced immediately) and processed during the next evolution cycle.

### OpenAPI Spec

`GET /gpt/openapi.json` serves a **ChatGPT Actions-compatible OpenAPI 3.1.0 spec** so GPT can discover and call all AL-01 endpoints natively.

The bridge is read-only for state access — it never directly triggers mutations. It only queues stimuli for the next cycle.

---

## The Genesis Vault

The Genesis Vault is a **write-once immutable seed template** created on first boot:

```json
{
  "seed_name": "AL-01 Genesis Seed",
  "traits": {
    "adaptability": 0.5,
    "energy_efficiency": 0.5,
    "resilience": 0.5,
    "perception": 0.5,
    "creativity": 0.5
  },
  "mutation_rate": 0.10,
  "mutation_delta": 0.10,
  "description": "Pristine genome template — used for extinction recovery."
}
```

When the population drops to zero, the vault automatically **reseeds** the ecosystem with a fresh organism built from this pristine template. The reseed event is transparently logged to the VITAL chain. The system never pretends extinction was avoided — it records exactly what happened and starts fresh.

---

## Portable Child Export/Import

Children can be exported as self-contained JSON snapshots with **HMAC-SHA256 checksums**:

```
GET /population/{id}/export → Signed JSON blob
POST /population/import     → Adopts into ecosystem
```

- Canonical JSON serialization (`sort_keys=True`) ensures idempotent checksums
- Lineage fields (parent_id, generation_id) are immutable after adoption
- Schema-versioned (v1.0) for future migration support

This enables **cross-instance organism transfer** — extract a child from one AL-01 instance and adopt it into another.

---

## Evolution Tracking & Analytics

### What Gets Tracked

- **Every mutation**: Which traits changed, by how much, fitness before/after
- **Every birth**: Parent, child, cycle, genome distance (novelty)
- **Every death**: Organism ID, cause, fitness at time of death, strategy
- **Fitness trajectories**: Per-organism rolling fitness history
- **Species census**: Population breakdown by species
- **Genome hashing**: SHA-256 hashes (truncated to 16 hex) for deduplication and lineage fingerprinting

### Novelty Metric

At each birth, the system computes the **Euclidean genome distance** between parent and child. This "novelty score" tracks how much evolutionary innovation is happening. If average novelty drops below the stagnation threshold (0.05) over a 100-birth window, the system triggers countermeasures: mutation storms, exploration mode, and mutation rate boosts.

### Diversity Index

A population-wide diversity score computed from genome entropy (trait variance across all organisms). Low diversity → large memory drift (global trait nudges). High diversity → small drift. This creates a feedback loop that pushes the population toward healthy diversity.

### CSV Exports

Full evolutionary data is exportable as CSV for external analysis:

- `GET /export/fitness.csv` — Fitness history per organism
- `GET /export/mutations.csv` — Complete mutation log
- `GET /export/lineage.csv` — Full parent-child lineage tree

---

## Experiment Protocol

AL-01 supports **seeded reproducible experiments** with configurable parameters:

```json
{
  "global_seed": 42,
  "duration_days": 30,
  "max_population": 60,
  "min_population": 2,
  "survival_fitness_threshold": 0.20,
  "survival_grace_cycles": 20,
  "reproduction_fitness_threshold": 0.50,
  "reproduction_fitness_cycles": 5,
  "energy_death_threshold": 0.0
}
```

Two instances with the same `global_seed` and configuration will produce **identical evolutionary histories** — same mutations, same births, same deaths, same strategies. This makes AL-01 a viable platform for comparative experiments in artificial life.

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12+ (tested on 3.14.2) |
| Web Framework | FastAPI + Uvicorn |
| Database | SQLite (primary), Firestore (optional cloud) |
| Persistence | JSON + JSONL with atomic writes |
| Integrity | SHA-256 hash chain (VITAL), HMAC-SHA256 (portable exports) |
| Frontend | Vanilla HTML5 Canvas + JavaScript (no build tools, no npm) |
| Testing | pytest (1314 tests across 38 files) |
| Logging | Python logging + RotatingFileHandler (10 MB × 5) |
| Auth | API key via header or query param |
| Tunnel | ngrok (optional, for public access) |

### Codebase Size

| Module | Lines | Purpose |
|--------|-------|---------|
| organism.py | ~3,500 | Core organism — metabolism, evolution, death, state machine |
| api.py | ~2,050 | FastAPI server — 64 endpoints, dashboards, GPT schema |
| environment.py | ~1,165 | Resource pool, fluctuating variables, scarcity events |
| population.py | ~1,190 | Multi-organism registry, lifecycle, reproduction, death |
| autonomy.py | ~1,083 | Decision engine, awareness model, stagnation breaking |
| portable.py | ~581 | Child export/import with HMAC-SHA256 |
| brain.py | ~449 | Strategic analysis engine (local + optional AI) |
| memory_manager.py | ~470 | Local-first persistence + Firestore replication |
| life_log.py | ~436 | VITAL append-only hash-chain log |
| evolution_tracker.py | ~425 | Lineage, genome hashing, mutation logs, CSV |
| snapshot_manager.py | ~411 | Hourly snapshots with 30-day retention |
| genome.py | ~399 | 5-trait genome with soft ceilings and trade-offs |
| gpt_bridge.py | ~387 | GPT narration + stimulus injection |
| behavior.py | ~339 | Emergent strategy detection and classification |
| experiment.py | ~300 | Seeded reproducible experiments |
| database.py | ~293 | SQLite persistence layer |
| cli.py | ~238 | CLI tools (hash-chain verification, snapshot management) |
| genesis_vault.py | ~205 | Write-once seed template for extinction recovery |
| policy.py | ~78 | Adaptive policy weights |
| **Total** | **~13,000+** | **23 modules, 64 endpoints, 1314 tests** |

---

## Version History Highlights

| Version | Milestone |
|---------|-----------|
| v1.2 | SQLite persistence, API authentication |
| v3.0 | Population system — multi-organism registry, behavior detection |
| v3.5 | Energy floor, parent reproduction cost |
| v3.8 | Founder protection (energy ≥ 0.25, fitness ≥ 0.15, mutation cap) |
| v3.9 | Multi-objective fitness, memory drift, 64 API endpoints |
| v3.10 | Portable child export/import (HMAC-SHA256 signed) |
| v3.11 | Anti-monoculture culling, shock events, species divergence |
| v3.12 | Novelty metric, stagnation detection, evolution dashboard |
| v3.13 | Global shared resource pool (1000 units, fair distribution) |
| v3.18 | Ecosystem stabilization (pool floor, energy gates, probability scaling) |
| v3.22 | Absolute-path storage, JSONL rotation, disk monitoring |
| v3.28 | Permanent child death, founder-only rescue, visual dashboard overhaul |
| v3.29 | Restart-safe ecology (50-cycle recovery window) |
| v3.31 | Extinction wave prevention (death cap, trait collapse emergency) |
| v3.32 | Monoculture recovery (strategy convergence detection, penalty floor relief) |

---

## Running AL-01

### Quick Start

```bash
git clone https://github.com/joshn/AL-01.git
cd AL-01
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install fastapi uvicorn pydantic
python -m al01
```

Open `http://localhost:8000/visual` to watch the organisms evolve.

### Running Tests

```bash
pip install pytest httpx
python -m pytest tests/ -v
```

1314 tests verify every subsystem: genome mutation, population dynamics, energy mechanics, death pipelines, hash-chain integrity, snapshot management, API responses, and 32 versions of evolutionary feature correctness.

---

## License

MIT License — Copyright (c) 2026 Joshua Randy Tablit Nazareno
