# AL-01: Artificial Life System Architecture

**A Continuous, Self-Sustaining Digital Organism Simulation**

---

## 1. Organism Lifecycle

### 1.1 Spawning

All organisms originate through one of two mechanisms:

**Genesis spawn.** When the population is empty, the Genesis Vault — an immutable seed record — instantiates a fresh organism with uniform trait values of 0.5 across all five genomic dimensions and a mutation rate and delta of 0.10. This organism is assigned generation_id = 0, no parent reference, and is registered in the evolution tracker. The vault records each reseed event with a monotonic counter, the instantiating cycle number, and an ISO-8601 timestamp.

**Reproductive spawn.** An existing organism produces a child. The child's genome is derived from the parent's genome with variance: each trait is perturbed by ±U(0, 0.05) (uniform random), the child's mutation rate and mutation delta each receive an independent perturbation of ±U(0, 0.02), floored at 0.01. The child receives an independent RNG seed, is assigned generation_id = parent.generation + 1, and is registered in the population and evolution tracker. The parent's energy is unaffected (energy cost is paid through the metabolic system, not direct transfer).

### 1.2 Evolution

Evolution occurs through two interacting systems:

**Trait mutation.** Every autonomy cycle (every 10 ticks ≈ 50 seconds), the autonomy engine may decide to mutate the organism. The decision depends on fitness relative to threshold, stagnation state, and exploration mode. When a mutation fires, each of the five traits independently faces a 10% probability of change (the mutation rate). If selected, the trait shifts by ±U(0, δ) where δ = 0.10 (the mutation delta). In exploration mode, the rate increases by +0.05; under stress, by +0.03. Trade-off rules are applied after mutation.

**Stagnation breakers.** If an organism's trait variance falls below 10⁻⁴ over its last 10 decisions, it is classified as stagnant. Escalating interventions fire at thresholds:

- **>50 cycles stagnant**: Activate exploration mode; double adaptation nudge
- **>200 cycles stagnant**: Triple adaptation nudge
- **>500 cycles stagnant**: Shuffle all traits by ±0.10
- **>800 cycles stagnant**: Full trait randomization; history cleared

### 1.3 Death

An organism dies under any of the following conditions:

- **Energy depletion.** Energy falls to ≤ 0.0.
- **Fitness floor.** Fitness remains below the death threshold (0.15 for the founder, 0.20 for children) for longer than a 20-cycle grace period.
- **Population management.** Population will not drop below 20% of the configured maximum, providing a floor guarantee.

Non-founder organisms enter dormancy before death: they are flagged dormant, consume 30% of normal metabolic cost, and can be awakened if the resource pool recovers to ≥50%. Dormant organisms appear visually as greyed-out bodies with a skull (💀) marker and breathing opacity between 0.30 and 0.40.

### 1.4 Reproduction

There are four distinct reproduction pathways, all checked every 15 ticks (~75 seconds):

| Pathway | Trigger Condition | Parent Energy Min | Cooldown | Pop Cap |
|---|---|---|---|---|
| Auto-reproduce | Fitness > 0.50 sustained for 5+ consecutive checks | 0.50 | 500 ticks | 60 |
| Rare reproduction | Every 500 ticks + 5% stochastic chance | 0.65 | 2,000 ticks | 50 |
| Stability reproduction | Resource pool ≥ 80% + 5% stochastic | 0.60 | 500 ticks | Resource-scaled |
| Lone survivor | Population = 1 AND pool ≥ 70% | 0.40 | 500 ticks | Unlimited |

A fifth system, wake_dormant_cycle(), reactivates dormant organisms when the resource pool is ≥50%.

---

## 2. Fitness Calculation

Fitness is computed through a layered pipeline, each stage refining the previous.

### Layer 1: Trait Fitness

The mean of all five soft-capped trait values:

    f_trait = (1/5) × Σ v̂_i

where the soft-cap function is:

    v̂ = v                          if v ≤ 1.0
    v̂ = 1.0 + ln(v) × 0.3         if v > 1.0

### Layer 2: Environment-Weighted Fitness

The environment assigns dynamic weights to each trait based on current conditions (e.g., resilience is weighted higher during temperature stress):

    f_weighted = Σ(v̂_i × w_i) / Σ(w_i)

### Layer 3: Multi-Objective Fitness (Primary)

The canonical fitness score blends genomic traits with lifecycle performance metrics:

    f_multi = 0.35 × s_survival + 0.30 × s_efficiency + 0.20 × s_stability + 0.15 × s_adaptation

Where:
- s_survival — normalized survival time (how long the organism has lived)
- s_efficiency — energy spent per unit fitness gained
- s_stability — consistency of fitness over recent cycles (low variance = high stability)
- s_adaptation — rate of positive fitness change in response to environmental shifts

All components are clamped to [0, 1].

### Layer 4: Noise Penalty

Environmental noise degrades the perceived fitness signal:

    f_effective = f - (n_noise × 0.15)

where n_noise ∈ [0, 1] is a drifting environmental variable.

### Layer 5: Energy Penalty

When energy drops below the fitness floor (0.30):

    f_final = f_effective × (e / 0.30)

This ensures organisms with depleted energy cannot maintain artificially high fitness.

---

## 3. Mutation System

### 3.1 Rate and Delta

Each organism carries its own heritable mutation parameters:

| Parameter | Default | Floor | Child Perturbation |
|---|---|---|---|
| Mutation rate | 0.10 (10% per trait) | 0.01 | ±U(0, 0.02) |
| Mutation delta | 0.10 (max shift) | 0.01 | ±U(0, 0.02) |

### 3.2 Mutation Application

When the autonomy engine selects "mutate":

1. For each of the 5 traits: with probability = mutation_rate, apply shift ±U(0, delta).
2. Apply trade-off rules (see §3.3).
3. Enforce floor of 0.02 on all traits.

Context-dependent rate modifiers:
- Exploration mode: +0.05 to rate
- Stress state: +0.03 to rate
- Stagnation scaling: delta multiplied by (1 + count × 0.5 / window), capped at 1.0

### 3.3 Trade-Off Rules

After any mutation or blending, pairwise trade-offs are enforced:

| Source Trait | Target Trait | Threshold | Penalty Coefficient |
|---|---|---|---|
| Adaptability | Energy Efficiency | 0.70 | 0.08 |
| Creativity | Energy Efficiency | 0.70 | 0.06 |

The penalty formula:

    reduction = (source - 0.70) × coeff × damping

where damping prevents extinction of the target:

    damping = min(1.0, max(0, target - 0.02) / 0.15)

As the target approaches its floor (0.02), damping → 0, halting further reduction.

### 3.4 Entropy Decay

Without external stimulus, traits decay passively:

    v(t+1) = max(0.02, v(t) - v(t) × 0.005 × (1 + p_entropy × 2.0))

At maximum entropy pressure (1.0), decay triples from 0.5% to 1.5% per cycle.

---

## 4. Energy System

### 4.1 Energy Gain

Energy is gained through the stabilize action selected by the autonomy engine:

- Stabilize yields +0.02 energy per execution.
- No passive energy regeneration exists; energy must be actively acquired through behavioral decisions.

### 4.2 Energy Consumption

Energy is consumed by metabolic cost and behavioral actions:

| Consumption Source | Cost |
|---|---|
| Base metabolic decay (per autonomy cycle) | 0.005 × (1 - e_eff × 0.30) |
| Mutate action | −0.010 energy (net, after partial regen) |
| Adapt action | −0.007 energy (net) |
| Blend action | −0.010 energy |
| Dormant metabolism | 30% of normal cost |

The energy efficiency trait directly reduces metabolic burn. At trait value 1.0, metabolic cost is reduced by 30%.

### 4.3 Environmental Modulation

Scarcity increases metabolic cost for active organisms:

- Mild scarcity (severity < 0.6): cost multiplier scales from 1.0 → 1.5
- Severe scarcity (severity ≥ 0.6): conservation mode activates, reducing multiplier to 0.4

### 4.4 Energy Floors and Recovery

- Hard floor: Energy cannot drop below 0.0 (death at ≤ 0).
- Soft floor: Energy minimum of 0.10 (below which recovery mode may activate).
- Recovery mode: After 10+ consecutive cycles below 0.20 energy, the organism enters recovery, forcing stabilize-only behavior and capping the fitness penalty to prevent a death spiral.

---

## 5. Reproduction Conditions (Detailed)

### 5.1 Auto-Reproduction

The primary pathway. Requires:

1. Weighted fitness ≥ 0.50 for 5 consecutive reproduction checks
2. Parent energy ≥ 0.50
3. ≥ 500 ticks since last reproduction
4. Population < 60

### 5.2 Rare Reproduction

A stochastic diversity mechanism. Requires:

1. ≥ 500 ticks since last reproduction check interval threshold (2,000 ticks between successes)
2. 5% random chance per check
3. Parent energy ≥ 0.65
4. Population < 50

### 5.3 Stability Reproduction

Rewards ecosystem health. Requires:

1. Resource pool ≥ 80% of maximum
2. 5% random chance per check
3. Parent energy ≥ 0.60
4. ≥ 500 ticks since last reproduction
5. Population cap is resource-scaled

### 5.4 Lone Survivor Reproduction

Emergency perpetuation. Requires:

1. Population = 1 (sole survivor)
2. Resource pool ≥ 70% of maximum
3. Parent energy ≥ 0.40 (lowest threshold)
4. ≥ 500 ticks since last reproduction
5. No population cap (unlimited spawning)

### 5.5 Scarcity Modulation

During scarcity events, reproduction thresholds increase:

    threshold_adjusted = threshold_base × (1 + severity × 0.5)

At maximum severity, reproduction requires 50% more fitness.

---

## 6. Environment Model

### 6.1 Resource Pool

The environment maintains a shared resource pool that governs ecosystem carrying capacity:

| Parameter | Value |
|---|---|
| Maximum pool | 1,000 units |
| Base regeneration | 5 units/cycle |
| Minimum floor | 50 units (prevents total collapse) |

Smart regeneration calculates actual regen per cycle as:

    r = r_base × f_efficiency × f_damping + r_pop_bonus

where f_efficiency ∈ [0.3, 1.5] is pool-health-dependent and f_damping ∈ [0.2, 1.0] prevents overshoot near capacity.

Extinction prevention: When population = 1, regeneration receives a 3× multiplier plus 100 flat units injected per cycle.

### 6.2 Environmental Variables

Four continuous variables drift in [0, 1] via bounded random walks:

| Variable | Effect | Drift Rate |
|---|---|---|
| Temperature | Sinusoidal base + stochastic noise; extremes stress resilience | ± random per cycle |
| Entropy pressure | Increases trait decay rate by up to 3×; drives adaptability demand | ± random per cycle |
| Abundance | Modulates resource availability and organism confidence | ± random per cycle |
| Noise | Degrades fitness signal by up to 15%; drives perception demand | ±0.01 per cycle |

Major environmental shifts occur every 20–50 cycles, applying a ±0.15 perturbation to each variable simultaneously.

### 6.3 Scarcity Events

- Trigger: 2% probability per cycle
- Effect: Destroys 30–70% of the resource pool
- Duration: 5–20 cycles
- Metabolic impact: Organisms face increased metabolic cost (up to 1.5×), then conservation mode (0.4×) if severity ≥ 0.6
- Reproduction impact: Fitness thresholds scaled up by (1 + severity × 0.5)

### 6.4 Shock Events

- Trigger: 0.02% probability per cycle (2 per 10,000 cycles expected)
- Duration: 5–15 cycles
- Effect: Entropy pressure spikes by +0.25; resilient organisms receive a +0.15 fitness bonus
- Purpose: Rewards organisms that invested in the resilience trait

### 6.5 Trait-Weighted Environmental Pressure

The environment dynamically adjusts trait importance:

| Environmental Condition | Favored Trait | Weight Modifier |
|---|---|---|
| High temperature stress | Resilience | +0.3 × temp_stress |
| High entropy | Adaptability | +0.4 × entropy |
| Scarcity | Energy Efficiency | +0.5 × scarcity |
| High noise | Perception | +0.6 × noise |
| High stability (low chaos) | Creativity | +0.4 × stability |

---

## 7. Evolutionary Pressure

### 7.1 Selection Mechanisms

AL-01 does not use tournament selection or explicit culling. Instead, Darwinian pressure emerges from the interaction of three systems:

**Energy-mediated survival.** Organisms with low energy efficiency burn energy faster. If they cannot stabilize before reaching 0.0, they die. This creates strong selection pressure for energy efficiency and, indirectly, against extreme adaptability and creativity (which penalize efficiency through trade-offs).

**Fitness-gated reproduction.** Only organisms sustaining fitness ≥ 0.50 for 5+ checks can auto-reproduce. Since fitness incorporates survival time, efficiency, stability, and adaptation success, this selects for generalist excellence rather than single-trait dominance.

**Environmental dynamism.** Drifting conditions shift which traits are rewarded. An organism optimized for a stable environment (high creativity) will suffer when entropy spikes. The continuous environmental change prevents any single phenotype from permanently dominating.

### 7.2 Why Certain Organisms Dominate

Dominance is transient and context-dependent:

- During scarcity: High energy efficiency organisms survive on reduced resources while others enter dormancy or die.
- During entropy spikes: High adaptability organisms maintain fitness while others decay.
- During shock events: High resilience organisms receive a direct +0.15 fitness bonus.
- During stability: High creativity organisms receive environmental weight boosts.
- At all times: The trade-off system prevents any single trait from being maximized without cost to energy efficiency, creating a perpetual balancing act.

The multi-objective fitness formula further ensures no single strategy dominates: an organism must also demonstrate survival longevity, metabolic efficiency, behavioral stability, and adaptive responsiveness.

---

## 8. Persistence Layer

### 8.1 State Storage

The system uses a three-tier persistence architecture:

**Tier 1 — In-Memory State.** The canonical organism state (_state dictionary) is held in RAM behind a thread lock. It includes all traits, energy, fitness, cycle counters, evolution history, and population membership.

**Tier 2 — Local Files (JSON + SQLite).**

- state.json: Full organism state, persisted every 3 heartbeats (15 seconds).
- memory.json: Bounded memory entries (capped to prevent runaway growth; nuclear threshold at 32 GB).
- al01.db (SQLite, WAL mode): Interactions, growth snapshots, memory events (indexed by timestamp and event type), and key-value metadata.
- data/life_log.jsonl: Append-only VITAL hash-chain.
- data/evolution_log.jsonl: Mutation events, fitness readings, births, deaths.
- data/cycle_log.jsonl: Per-cycle autonomy decisions.
- data/genesis_vault.json: Immutable seed and reseed history.

**Tier 3 — Remote Sync (Firestore).** When configured, the memory manager syncs state to Google Firestore (Spark-tier compatible). The snapshot manager uploads hourly snapshots (full state if <500 KB, preview otherwise).

### 8.2 VITAL: Verifiable Integrity Timeline and Audit Log

Every significant event (birth, death, mutation, reproduction, environmental shock) is appended to an immutable hash-chain. Each entry contains:

    h_i = SHA-256(h_(i-1) ‖ json(payload_i) ‖ t_i ‖ seq_i)

- h_0 = "0" × 64 (genesis hash)
- The chain is verified on startup (last 200 entries by default)
- Snapshots are written every 50 events to enable fast random-access verification
- If corruption is detected, the system re-anchors to the last valid entry and backs up the original

### 8.3 Lineage Tracking

The evolution tracker maintains per-organism lineage records:

- Generation ID: Incremented per lineage level (root = 0)
- Genome hash: SHA-256 of trait values, truncated to 16 hex characters
- Parent chain: Full ancestor traversal via parent_id references
- Fitness trajectory: Rolling buffer of last 1,000 fitness measurements per organism
- Trait snapshots: Rolling buffer of last 500 trait vectors per organism
- Events: All mutations, births, deaths, and fitness readings logged to evolution_log.jsonl

### 8.4 Snapshots

The snapshot manager runs on a background daemon thread:

- Interval: 1 hour
- Retention: 30-day rolling archive (auto-purged)
- Format: JSON with SHA-256 checksum of state payload
- Manifest: Index of up to 2,000 snapshot metadata entries for fast lookup
- Remote: Uploaded to Firestore when enabled

---

## 9. Simulation Loop

### 9.1 Tick Frequency

The simulation runs on a 5-second tick (INTERVAL = 5). A background thread (_loop_worker) executes one _run_cycle() per tick indefinitely until interrupted.

### 9.2 Per-Tick Execution

Each _run_cycle() call proceeds through the Metabolism Scheduler, which maintains independent counters for each subsystem:

| Subsystem | Interval (ticks) | Wall-Clock Period | Action |
|---|---|---|---|
| Pulse | 1 | 5s | Metabolic tick (energy accounting, state bookkeeping) |
| Heartbeat persist | 3 | 15s | Write state to disk |
| Autonomy cycle | 10 | 50s | Decision engine: mutate / adapt / stabilize / blend |
| Child autonomy | 10 | 50s | Same cycle applied to all child organisms |
| Auto-reproduce | 15 | 75s | Check 4 reproduction pathways + wake dormant |
| Pulse log | 15 | 75s | Log metrics to cycle_log.jsonl |
| Persist | 15 | 75s | Full state persist |
| Reflect | 30 | 150s | Brain strategic analysis (if AI enabled) |
| Evolve | 30 | 150s | Formal evolution pass (genome mutation via brain) |
| Rare reproduce | 50 | 250s | Stochastic rare reproduction check |
| Population interact | 60 | 300s | Genome blending between organisms |
| Memory snapshot | 100 | 500s | Bounded memory compaction snapshot |

### 9.3 State Merge

At the start of each cycle, the in-memory state is merged with the latest on-disk state (loaded from the memory manager). Evolution count and state version use max() to prevent regression. This allows external writes (e.g., API-triggered interactions) to be incorporated without data loss.

### 9.4 Shutdown Sequence

On Ctrl+C:

1. Record final growth snapshot
2. Execute organism.shutdown() (final persist, close subsystems)
3. Flush buffered memory entries to disk
4. Flush pending Firestore writes
5. Attempt daily backup if eligible
6. Log clean exit

---

## 10. Visualization System

The visual dashboard is a real-time HTML5 Canvas application served at /visual, rendering at 60 FPS with a 3-second data-polling interval.

### 10.1 Trait-to-Color Mapping

Each organism's body color is derived directly from three of its five traits mapped to RGB channels:

    R = floor(min(1, v_adaptability) × 255)
    G = floor(min(1, v_energy_efficiency) × 255)
    B = floor(min(1, v_resilience) × 255)

This produces an intuitive color space:

- Red-dominant → high adaptability
- Green-dominant → high energy efficiency
- Blue-dominant → high resilience
- Yellow → high adaptability + efficiency
- Cyan → high efficiency + resilience
- White → all traits near maximum

Perception and creativity do not map to the primary body color but influence other visual properties.

### 10.2 Fitness-to-Size Mapping

Circle radius scales linearly with fitness:

    r = 10 + f × 40 px

Fitness 0 → 10 px radius; fitness 1.0 → 50 px radius.

### 10.3 Body Rendering

Each organism body uses a 3D radial gradient (not flat fill):

- Inner highlight: lightened RGB (+18% specular)
- Mid-body: base RGB color
- Outer edge: darkened RGB (shadow)

### 10.4 Animation Systems (12 Total)

| Animation | Visual | Trigger |
|---|---|---|
| Energy pulse | Rhythmic size oscillation (±6%) | Always active; frequency = 0.8 + e × 2.2 Hz |
| Evolution rings | Up to 3 concentric rings at +6, +11, +16 px | Evolution count > 0 |
| Awareness halo | Semi-transparent outer glow | Awareness > 0.5 |
| High-energy vibration | Random position jitter ±(intensity × 2) px | Energy > 0.70 |
| Dormant state | Greyscale body, breathing opacity, skull emoji | Organism flagged dormant |
| Energy trail | Up to 8 trailing particles, gravity-affected, sized 1–3.5 px | Energy > 0.50 |
| Heartbeat ring | Expanding pulse ring, period 1.2s, sin⁴ envelope, max +14 px | Always active |
| Fitness glow | Green border outline, stroke width 1.5 + i × 1.0 px | Fitness > 0.30 |
| Leader crown | Orbiting particles (5 / 4 / 3 for rank 1–3) | Top 3 by fitness |
| Parent hexagon | Blue hexagonal outline with glow | Has children |
| Smooth interpolation | Position/size/color lerp at rate 0.08–0.10 | Always |
| Dominant trait bar | Color-coded background bar for highest trait | Always |

### 10.5 Information Display

- Header bar: Population count, mean fitness, resource pool percentage, cycle number
- Tooltip on hover: Organism ID, generation, all 5 trait values (color-coded), fitness, energy, awareness, strategy, parent ID, dominant trait
- Legend: Color channel mapping, ring meanings, vibration/dormant/trail explanations
