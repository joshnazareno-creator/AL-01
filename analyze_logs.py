"""AL-01 comprehensive log analysis — answering all data questions."""
import json, re, os, sys
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path(r"D:\AL-01\data")
LOG_FILE = Path(r"D:\AL-01\al01.log.1")
POP_FILE = Path(r"D:\AL-01\population.json")
STATE_FILE = Path(r"D:\AL-01\state.json")

# ── Cutoff date ──
CUTOFF = datetime(2025, 2, 25, tzinfo=timezone.utc)

print("=" * 70)
print("AL-01 COMPREHENSIVE DATA ANALYSIS")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════
# 1. TOTAL CYCLE COUNT SINCE FEB 25
# ═══════════════════════════════════════════════════════════════════════
print("\n┌─────────────────────────────────────────────────────┐")
print("│  1. TOTAL CYCLE COUNT                               │")
print("└─────────────────────────────────────────────────────┘")

cycle_count = 0
cycle_log_path = DATA_DIR / "cycle_log.jsonl"
first_ts = None
last_ts = None
if cycle_log_path.exists():
    with open(cycle_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                ts_str = rec.get("timestamp") or rec.get("ts")
                if ts_str:
                    if first_ts is None:
                        first_ts = ts_str
                    last_ts = ts_str
                cycle_count += 1
            except json.JSONDecodeError:
                continue

print(f"  Total cycle records: {cycle_count:,}")
print(f"  First record: {first_ts}")
print(f"  Last record:  {last_ts}")

# Also check state.json for global cycle
if STATE_FILE.exists():
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)
    print(f"  State global_cycle: {state.get('global_cycle', 'N/A')}")

# ═══════════════════════════════════════════════════════════════════════
# 2. PARSE AUTONOMY LOG — fitness, decisions, energy, revivals
# ═══════════════════════════════════════════════════════════════════════
print("\n┌─────────────────────────────────────────────────────┐")
print("│  2. PARSING AUTONOMY LOG...                         │")
print("└─────────────────────────────────────────────────────┘")

autonomy_path = DATA_DIR / "autonomy_log.jsonl"
decisions_all = []
fitness_timeline = []  # (index, fitness, energy, decision, recovery_mode)
decision_counter = Counter()

if autonomy_path.exists():
    with open(autonomy_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                decision = rec.get("decision", "unknown")
                fitness = rec.get("fitness", 0)
                eff_fitness = rec.get("effective_fitness", 0)
                energy = rec.get("energy", 0)
                recovery = rec.get("recovery_mode", False)
                founder_recovery = rec.get("founder_recovery_mode", False)
                founder_blocked = rec.get("founder_mutate_blocked", False)
                ts = rec.get("timestamp", "")
                reason = rec.get("reason", "")

                decisions_all.append(rec)
                fitness_timeline.append({
                    "idx": i,
                    "fitness": fitness,
                    "eff_fitness": eff_fitness,
                    "energy": energy,
                    "decision": decision,
                    "recovery": recovery,
                    "founder_recovery": founder_recovery,
                    "founder_blocked": founder_blocked,
                    "ts": ts,
                    "reason": reason,
                })
                decision_counter[decision] += 1
            except json.JSONDecodeError:
                continue

print(f"  Total autonomy records: {len(decisions_all):,}")

# ═══════════════════════════════════════════════════════════════════════
# 3. PARSE LOG FILE — revival events, death events
# ═══════════════════════════════════════════════════════════════════════
print("\n┌─────────────────────────────────────────────────────┐")
print("│  3. PARSING al01.log.1 FOR REVIVAL/DEATH EVENTS     │")
print("└─────────────────────────────────────────────────────┘")

revival_events = []
death_events = []
founder_death_events = []
boot_revivals = []
founder_recovery_logs = []
mutation_cap_logs = []

ts_pattern = re.compile(r'^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})')

if LOG_FILE.exists():
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            m = ts_pattern.match(line)
            ts_str = m.group(1) if m else None

            if "[REVIVE]" in line:
                revival_events.append({"ts": ts_str, "line": line.strip()})
            elif "[BOOT]" in line and "reviving" in line:
                boot_revivals.append({"ts": ts_str, "line": line.strip()})
            elif "[SURVIVAL]" in line and "triggering death" in line:
                death_events.append({"ts": ts_str, "line": line.strip(), "cause": "fitness_floor"})
            elif "[FOUNDER-DEATH]" in line:
                founder_death_events.append({"ts": ts_str, "line": line.strip()})
            elif "[FOUNDER-RECOVERY]" in line:
                founder_recovery_logs.append({"ts": ts_str, "line": line.strip()})
                if "mutation cap" in line.lower():
                    mutation_cap_logs.append({"ts": ts_str, "line": line.strip()})
            elif "energy_depleted" in line and "death" in line.lower():
                if "AL-01" in line:
                    death_events.append({"ts": ts_str, "line": line.strip(), "cause": "energy_depleted"})

# Deduplicate by timestamp
unique_revival_ts = set()
unique_revivals = []
for r in revival_events:
    if r["ts"] not in unique_revival_ts:
        unique_revival_ts.add(r["ts"])
        unique_revivals.append(r)

unique_death_ts = set()
unique_deaths = []
for d in death_events:
    if d["ts"] not in unique_death_ts:
        unique_death_ts.add(d["ts"])
        unique_deaths.append(d)

total_revivals = len(unique_revivals) + len(boot_revivals)

print(f"  In-cycle revivals (unique): {len(unique_revivals):,}")
print(f"  Boot revivals: {len(boot_revivals)}")
print(f"  Total revivals: {total_revivals:,}")
print(f"  Death events (unique): {len(unique_deaths):,}")
print(f"  Founder-death logs (v3.25): {len(founder_death_events)}")
print(f"  Founder-recovery logs (v3.25): {len(founder_recovery_logs)}")
print(f"  Mutation cap triggers (v3.25): {len(mutation_cap_logs)}")

# ═══════════════════════════════════════════════════════════════════════
# 4. FITNESS BEFORE/AFTER REVIVAL
# ═══════════════════════════════════════════════════════════════════════
print("\n┌─────────────────────────────────────────────────────┐")
print("│  4. AVG FITNESS BEFORE/AFTER REVIVAL                │")
print("└─────────────────────────────────────────────────────┘")

# Use the autonomy log fitness timeline to find fitness around revival-like
# patterns: fitness dropping to very low then jumping
revival_fitness_before = []
revival_fitness_after = []

# Detect revival patterns: energy jumps from below 0.3 to 0.5+
for i in range(1, len(fitness_timeline)):
    curr = fitness_timeline[i]
    prev = fitness_timeline[i-1]
    # Revival signature: energy jumps significantly upward in one step
    if curr["energy"] >= 0.45 and prev["energy"] < 0.3:
        revival_fitness_before.append(prev["fitness"])
        # Look at fitness 5 cycles after revival
        after_idx = min(i + 5, len(fitness_timeline) - 1)
        revival_fitness_after.append(fitness_timeline[after_idx]["fitness"])

if revival_fitness_before:
    avg_before = sum(revival_fitness_before) / len(revival_fitness_before)
    avg_after = sum(revival_fitness_after) / len(revival_fitness_after)
    print(f"  Revival events detected in autonomy log: {len(revival_fitness_before)}")
    print(f"  Avg fitness BEFORE revival: {avg_before:.4f}")
    print(f"  Avg fitness 5 cycles AFTER revival: {avg_after:.4f}")
    print(f"  Fitness change: {avg_after - avg_before:+.4f}")
else:
    print("  No revival patterns detected in autonomy log")

# ═══════════════════════════════════════════════════════════════════════
# 5. FITNESS FLOOR HITS vs RECOVERIES
# ═══════════════════════════════════════════════════════════════════════
print("\n┌─────────────────────────────────────────────────────┐")
print("│  5. FITNESS FLOOR (<0.15) HITS vs RECOVERY (>0.30)  │")
print("└─────────────────────────────────────────────────────┘")

floor_hits = 0
recoveries_above_030 = 0
below_floor = False

for entry in fitness_timeline:
    f = entry["fitness"]
    if f < 0.15:
        if not below_floor:
            floor_hits += 1
            below_floor = True
    elif f >= 0.30:
        if below_floor:
            recoveries_above_030 += 1
            below_floor = False
    # If between 0.15 and 0.30, stay in whatever state

print(f"  Times fitness dropped below 0.15: {floor_hits:,}")
print(f"  Times recovered above 0.30 from below: {recoveries_above_030:,}")
print(f"  Recovery rate: {recoveries_above_030/max(1,floor_hits)*100:.1f}%")
print(f"  Still below floor at end: {'Yes' if below_floor else 'No'}")

# ═══════════════════════════════════════════════════════════════════════
# 6. AVERAGE LIFESPAN (CYCLES) BETWEEN DEATHS
# ═══════════════════════════════════════════════════════════════════════
print("\n┌─────────────────────────────────────────────────────┐")
print("│  6. AVERAGE LIFESPAN BETWEEN DEATH EVENTS           │")
print("└─────────────────────────────────────────────────────┘")

# Use autonomy log to find death events (organism_died flag or energy → 0)
death_indices = []
for i, entry in enumerate(fitness_timeline):
    rec = decisions_all[i] if i < len(decisions_all) else {}
    if rec.get("organism_died", False):
        death_indices.append(i)

# Also detect implicitly: energy crash to near-zero
if not death_indices:
    for i, entry in enumerate(fitness_timeline):
        if entry["energy"] <= 0.1 and i > 0 and fitness_timeline[i-1]["energy"] > 0.2:
            death_indices.append(i)

if len(death_indices) >= 2:
    lifespans = [death_indices[i+1] - death_indices[i] for i in range(len(death_indices)-1)]
    avg_life = sum(lifespans) / len(lifespans)
    min_life = min(lifespans)
    max_life = max(lifespans)
    median_life = sorted(lifespans)[len(lifespans)//2]
    print(f"  Death events in autonomy log: {len(death_indices)}")
    print(f"  Average lifespan: {avg_life:.1f} cycles")
    print(f"  Median lifespan:  {median_life} cycles")
    print(f"  Min lifespan:     {min_life} cycles")
    print(f"  Max lifespan:     {max_life} cycles")
    # Last 10 lifespans
    print(f"  Last 10 lifespans: {lifespans[-10:]}")
else:
    print(f"  Death events detected: {len(death_indices)} (need >=2 for lifespan)")
    print(f"  Insufficient data for lifespan calculation")

# ═══════════════════════════════════════════════════════════════════════
# 7. MUTATION SUCCESS RATE
# ═══════════════════════════════════════════════════════════════════════
print("\n┌─────────────────────────────────────────────────────┐")
print("│  7. MUTATION SUCCESS RATE                            │")
print("└─────────────────────────────────────────────────────┘")

mutate_count = 0
mutate_improved = 0
mutate_degraded = 0
mutate_neutral = 0

for i in range(1, len(fitness_timeline)):
    if fitness_timeline[i-1]["decision"] == "mutate":
        mutate_count += 1
        delta = fitness_timeline[i]["fitness"] - fitness_timeline[i-1]["fitness"]
        if delta > 0.001:
            mutate_improved += 1
        elif delta < -0.001:
            mutate_degraded += 1
        else:
            mutate_neutral += 1

print(f"  Total mutations: {mutate_count:,}")
print(f"  Improved fitness: {mutate_improved:,} ({mutate_improved/max(1,mutate_count)*100:.1f}%)")
print(f"  Degraded fitness: {mutate_degraded:,} ({mutate_degraded/max(1,mutate_count)*100:.1f}%)")
print(f"  Neutral (< ±0.001): {mutate_neutral:,} ({mutate_neutral/max(1,mutate_count)*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════════════
# 8. DECISION DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════
print("\n┌─────────────────────────────────────────────────────┐")
print("│  8. DECISION TYPE DISTRIBUTION                      │")
print("└─────────────────────────────────────────────────────┘")

total_decisions = sum(decision_counter.values())
for dec, count in decision_counter.most_common():
    pct = count / max(1, total_decisions) * 100
    bar = "█" * int(pct / 2)
    print(f"  {dec:12s}: {count:>7,} ({pct:5.1f}%) {bar}")

# ═══════════════════════════════════════════════════════════════════════
# 9. SCARCITY AT DEATH EVENTS
# ═══════════════════════════════════════════════════════════════════════
print("\n┌─────────────────────────────────────────────────────┐")
print("│  9. SCARCITY AT DEATH EVENTS                        │")
print("└─────────────────────────────────────────────────────┘")

death_scarcity = []
for idx in death_indices:
    if idx < len(decisions_all):
        env_mods = decisions_all[idx].get("env_modifiers", {})
        cost_mult = env_mods.get("mutation_cost_multiplier", 1.0)
        death_scarcity.append(cost_mult)

if death_scarcity:
    avg_scarcity = sum(death_scarcity) / len(death_scarcity)
    max_scarcity = max(death_scarcity)
    above_1 = sum(1 for s in death_scarcity if s > 1.01)
    print(f"  Deaths with scarcity data: {len(death_scarcity)}")
    print(f"  Avg mutation_cost_multiplier at death: {avg_scarcity:.3f}")
    print(f"  Max mutation_cost_multiplier at death: {max_scarcity:.3f}")
    print(f"  Deaths during scarcity (mult > 1.01): {above_1} ({above_1/len(death_scarcity)*100:.1f}%)")
else:
    print("  No scarcity data available at death events")

# ═══════════════════════════════════════════════════════════════════════
# 10. CHILD LINEAGE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
print("\n┌─────────────────────────────────────────────────────┐")
print("│  10. CHILD LINEAGE & TRAIT ANALYSIS                 │")
print("└─────────────────────────────────────────────────────┘")

population = {}
if POP_FILE.exists():
    with open(POP_FILE, "r", encoding="utf-8") as f:
        population = json.load(f)

children = {}
al01_traits = None
for oid, data in population.items():
    if oid == "AL-01":
        al01_traits = data.get("traits", {})
    else:
        children[oid] = data

if al01_traits and children:
    # Generational distance
    max_gen = 0
    max_gen_child = None
    for oid, data in children.items():
        gen = data.get("generation", 0)
        if gen > max_gen:
            max_gen = gen
            max_gen_child = oid

    print(f"  Total children: {len(children)}")
    print(f"  AL-01 traits: {', '.join(f'{k}={v:.3f}' for k,v in al01_traits.items())}")
    if max_gen_child:
        child_data = children[max_gen_child]
        child_traits = child_data.get("traits", {})
        print(f"\n  Highest generation child: {max_gen_child} (gen {max_gen})")
        print(f"  Fitness: {child_data.get('fitness', 0):.4f}, Energy: {child_data.get('energy', 0):.4f}")
        print(f"  Traits: {', '.join(f'{k}={v:.3f}' for k,v in child_traits.items())}")
        # Trait divergence
        divs = []
        for trait in al01_traits:
            if trait in child_traits:
                delta = abs(child_traits[trait] - al01_traits[trait])
                divs.append((trait, delta, child_traits[trait] - al01_traits[trait]))
        divs.sort(key=lambda x: -x[1])
        print(f"  Trait divergence from AL-01:")
        for trait, delta, direction in divs:
            arrow = "↑" if direction > 0 else "↓"
            print(f"    {trait}: {delta:.4f} {arrow}")

    # Selection pressure: which traits are highest in fittest children?
    print(f"\n  TRAIT SELECTION PRESSURE (all children):")
    trait_names = list(al01_traits.keys())
    # Average trait by fitness quartile
    sorted_children = sorted(children.values(), key=lambda c: c.get("fitness", 0), reverse=True)
    top_q = sorted_children[:max(1, len(sorted_children)//4)]
    bot_q = sorted_children[-max(1, len(sorted_children)//4):]

    for trait in trait_names:
        top_avg = sum(c.get("traits", {}).get(trait, 0.5) for c in top_q) / len(top_q)
        bot_avg = sum(c.get("traits", {}).get(trait, 0.5) for c in bot_q) / len(bot_q)
        pressure = top_avg - bot_avg
        arrow = "↑" if pressure > 0.01 else ("↓" if pressure < -0.01 else "~")
        print(f"    {trait:20s}: top25%={top_avg:.3f}  bot25%={bot_avg:.3f}  pressure={pressure:+.3f} {arrow}")

    # Generation vs fitness correlation
    gens = []
    fits = []
    for c in children.values():
        g = c.get("generation", 0)
        f = c.get("fitness", 0)
        if g > 0:
            gens.append(g)
            fits.append(f)
    if len(gens) >= 3:
        n = len(gens)
        mean_g = sum(gens) / n
        mean_f = sum(fits) / n
        cov = sum((gens[i] - mean_g) * (fits[i] - mean_f) for i in range(n)) / n
        std_g = (sum((g - mean_g)**2 for g in gens) / n) ** 0.5
        std_f = (sum((f - mean_f)**2 for f in fits) / n) ** 0.5
        corr = cov / (std_g * std_f) if std_g > 0 and std_f > 0 else 0
        print(f"\n  Generation vs Fitness correlation: r = {corr:.3f}")
        print(f"    (n={n}, {'positive' if corr > 0.1 else 'negative' if corr < -0.1 else 'weak/none'})")
else:
    print("  Population data not available or no children")

# ═══════════════════════════════════════════════════════════════════════
# 11. BEFORE/AFTER PATCH COMPARISON
# ═══════════════════════════════════════════════════════════════════════
print("\n┌─────────────────────────────────────────────────────┐")
print("│  11. BEFORE vs AFTER RECOVERY PATCH (v3.25)         │")
print("└─────────────────────────────────────────────────────┘")

# Detect patch: look for founder_recovery_mode=True in autonomy log
patch_idx = None
for i, rec in enumerate(decisions_all):
    if rec.get("founder_recovery_mode") is not None:
        patch_idx = i
        break

if patch_idx is not None:
    pre_deaths = [d for d in death_indices if d < patch_idx]
    post_deaths = [d for d in death_indices if d >= patch_idx]

    print(f"  Patch detected at autonomy record #{patch_idx:,}")
    print(f"  Pre-patch cycles: {patch_idx:,}")
    print(f"  Post-patch cycles: {len(decisions_all) - patch_idx:,}")
    print(f"  Pre-patch deaths: {len(pre_deaths)}")
    print(f"  Post-patch deaths: {len(post_deaths)}")

    if len(pre_deaths) >= 2:
        pre_lifespans = [pre_deaths[i+1] - pre_deaths[i] for i in range(len(pre_deaths)-1)]
        print(f"  Pre-patch avg lifespan: {sum(pre_lifespans)/len(pre_lifespans):.1f} cycles")
    if len(post_deaths) >= 2:
        post_lifespans = [post_deaths[i+1] - post_deaths[i] for i in range(len(post_deaths)-1)]
        print(f"  Post-patch avg lifespan: {sum(post_lifespans)/len(post_lifespans):.1f} cycles")
    elif len(post_deaths) <= 1:
        cycles_since_last = len(decisions_all) - (post_deaths[0] if post_deaths else patch_idx)
        print(f"  Post-patch: {'1 death' if post_deaths else 'NO deaths'} — {cycles_since_last:,} cycles and counting")

    # Mutation cap triggers
    print(f"\n  Mutation cap triggers (v3.25): {len(mutation_cap_logs)}")
    for evt in mutation_cap_logs[:5]:
        print(f"    {evt['ts']}: {evt['line'][:100]}")
else:
    print("  Patch not yet active in autonomy log (AL-01 needs restart)")
    if death_indices:
        all_lifespans = [death_indices[i+1] - death_indices[i] for i in range(len(death_indices)-1)]
        if all_lifespans:
            print(f"  Current avg lifespan: {sum(all_lifespans)/len(all_lifespans):.1f} cycles")
            print(f"  This will be the pre-patch baseline for comparison")

# ═══════════════════════════════════════════════════════════════════════
# 12. FITNESS OVER TIME — CSV for graphing
# ═══════════════════════════════════════════════════════════════════════
print("\n┌─────────────────────────────────────────────────────┐")
print("│  12. FITNESS TIMELINE SAVED FOR GRAPHING             │")
print("└─────────────────────────────────────────────────────┘")

# Save sampled fitness timeline (every Nth point + all revival/death markers)
SAMPLE_RATE = max(1, len(fitness_timeline) // 2000)  # ~2000 points
sampled = []
revival_markers = []

for i, entry in enumerate(fitness_timeline):
    is_revival = False
    if i > 0:
        prev = fitness_timeline[i-1]
        if entry["energy"] >= 0.45 and prev["energy"] < 0.3:
            is_revival = True
            revival_markers.append(i)

    if i % SAMPLE_RATE == 0 or is_revival or i in death_indices:
        sampled.append({
            "cycle": i,
            "fitness": round(entry["fitness"], 4),
            "energy": round(entry["energy"], 4),
            "decision": entry["decision"],
            "is_revival": is_revival,
            "is_death": i in set(death_indices),
        })

csv_path = DATA_DIR / "fitness_timeline.csv"
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("cycle,fitness,energy,decision,is_revival,is_death\n")
    for s in sampled:
        f.write(f"{s['cycle']},{s['fitness']},{s['energy']},{s['decision']},{s['is_revival']},{s['is_death']}\n")

print(f"  Saved {len(sampled)} data points to {csv_path}")
print(f"  Revival markers: {len(revival_markers)}")
print(f"  Death markers: {len(death_indices)}")

# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Total cycles:          {len(decisions_all):,}")
print(f"  Total revivals:        {total_revivals:,}")
print(f"  Total deaths:          {len(unique_deaths):,}")
print(f"  Fitness floor hits:    {floor_hits:,}")
print(f"  Recovered above 0.30:  {recoveries_above_030:,}")
print(f"  Mutation success rate:  {mutate_improved/max(1,mutate_count)*100:.1f}%")
print(f"  Top decision:          {decision_counter.most_common(1)[0][0]} ({decision_counter.most_common(1)[0][1]:,})")
print(f"  Children alive:        {len(children)}")
if patch_idx is not None:
    print(f"  Patch active since:    record #{patch_idx:,}")
else:
    print(f"  Patch status:          Not yet active (needs restart)")
