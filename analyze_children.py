"""AL-01 child lineage analysis + fitness graph generation."""
import json, os
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(r"D:\AL-01\data")
POP_FILE = Path(r"D:\AL-01\population.json")

# ═══════════════════════════════════════════════════════════════════════
# CHILD LINEAGE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("CHILD LINEAGE & EVOLUTION ANALYSIS")
print("=" * 70)

with open(POP_FILE, "r", encoding="utf-8") as f:
    pop = json.load(f)

al01_data = pop.get("AL-01", {})
al01_genome = al01_data.get("genome", {})
al01_traits = al01_genome.get("traits", {})
al01_fitness = al01_genome.get("fitness", 0)

print(f"\n  AL-01 fitness: {al01_fitness:.4f}")
print(f"  AL-01 energy:  {al01_data.get('energy', 0):.4f}")
print(f"  AL-01 alive:   {al01_data.get('alive')}")
print(f"  AL-01 traits:")
for t, v in al01_traits.items():
    print(f"    {t:20s}: {v:.6f}")

# Collect children
children = {}
for oid, data in pop.items():
    if oid == "AL-01":
        continue
    genome = data.get("genome", {})
    children[oid] = {
        "id": oid,
        "generation": data.get("generation_id", 0),
        "fitness": genome.get("fitness", 0),
        "energy": data.get("energy", 0),
        "alive": data.get("alive", False),
        "state": data.get("state", "unknown"),
        "traits": genome.get("traits", {}),
        "parent": data.get("parent_id", ""),
    }

alive_children = {k: v for k, v in children.items() if v["alive"]}
dormant = {k: v for k, v in children.items() if v["state"] == "dormant"}
dead = {k: v for k, v in children.items() if not v["alive"] and v["state"] != "dormant"}

print(f"\n  Total children: {len(children)}")
print(f"  Alive:   {len(alive_children)}")
print(f"  Dormant: {len(dormant)}")
print(f"  Dead:    {len(dead)}")

# Highest generational distance
print(f"\n{'─'*60}")
print("HIGHEST GENERATIONAL DISTANCE FROM AL-01")
print(f"{'─'*60}")

max_gen = 0
max_gen_child = None
for oid, c in children.items():
    if c["generation"] > max_gen:
        max_gen = c["generation"]
        max_gen_child = oid

if max_gen_child:
    c = children[max_gen_child]
    print(f"  Child: {max_gen_child}")
    print(f"  Generation: {max_gen}")
    print(f"  Fitness: {c['fitness']:.4f}")
    print(f"  Alive: {c['alive']}, State: {c['state']}")
    print(f"  Trait divergence from AL-01:")
    divs = []
    for trait in al01_traits:
        if trait in c["traits"]:
            delta = c["traits"][trait] - al01_traits[trait]
            divs.append((trait, abs(delta), delta))
    divs.sort(key=lambda x: -x[1])
    for trait, abs_d, d in divs:
        arrow = "↑" if d > 0 else "↓"
        print(f"    {trait:20s}: AL-01={al01_traits[trait]:.4f} → child={c['traits'][trait]:.4f} (Δ={d:+.4f} {arrow})")

# Selection pressure
print(f"\n{'─'*60}")
print("TRAIT SELECTION PRESSURE (top 25% vs bottom 25% by fitness)")
print(f"{'─'*60}")

sorted_children = sorted(children.values(), key=lambda c: c["fitness"], reverse=True)
n = len(sorted_children)
q_size = max(1, n // 4)
top_q = sorted_children[:q_size]
bot_q = sorted_children[-q_size:]

trait_names = list(al01_traits.keys())
for trait in trait_names:
    top_avg = sum(c["traits"].get(trait, 0.5) for c in top_q) / len(top_q)
    bot_avg = sum(c["traits"].get(trait, 0.5) for c in bot_q) / len(bot_q)
    pressure = top_avg - bot_avg
    if pressure > 0.02:
        signal = "POSITIVE ↑"
    elif pressure < -0.02:
        signal = "NEGATIVE ↓"
    else:
        signal = "NEUTRAL  ~"
    bar_len = int(abs(pressure) * 100)
    bar = ("+" if pressure > 0 else "-") * min(40, bar_len)
    print(f"  {trait:20s}: top25%={top_avg:.3f}  bot25%={bot_avg:.3f}  Δ={pressure:+.4f}  {signal} {bar}")

print(f"\n  Top quartile avg fitness: {sum(c['fitness'] for c in top_q)/len(top_q):.4f}")
print(f"  Bottom quartile avg fitness: {sum(c['fitness'] for c in bot_q)/len(bot_q):.4f}")

# Generation vs fitness correlation
print(f"\n{'─'*60}")
print("GENERATION vs FITNESS CORRELATION")
print(f"{'─'*60}")

gens = [c["generation"] for c in children.values() if c["generation"] > 0]
fits = [c["fitness"] for c in children.values() if c["generation"] > 0]

if len(gens) >= 3:
    n = len(gens)
    mean_g = sum(gens) / n
    mean_f = sum(fits) / n
    cov = sum((gens[i] - mean_g) * (fits[i] - mean_f) for i in range(n)) / n
    std_g = (sum((g - mean_g)**2 for g in gens) / n) ** 0.5
    std_f = (sum((f - mean_f)**2 for f in fits) / n) ** 0.5
    corr = cov / (std_g * std_f) if std_g > 0 and std_f > 0 else 0
    print(f"  Pearson r = {corr:.4f} (n={n})")
    if corr > 0.3:
        print(f"  → Later generations tend to be FITTER (progressive evolution)")
    elif corr < -0.3:
        print(f"  → Later generations tend to be LESS fit (regression)")
    else:
        print(f"  → No strong linear relationship between generation and fitness")

# Per-generation stats
gen_fitness = defaultdict(list)
for c in children.values():
    gen_fitness[c["generation"]].append(c["fitness"])

print(f"\n  Per-generation fitness:")
for gen in sorted(gen_fitness.keys()):
    fits_g = gen_fitness[gen]
    avg = sum(fits_g) / len(fits_g)
    print(f"    Gen {gen}: n={len(fits_g):2d}, avg_fitness={avg:.4f}, range=[{min(fits_g):.4f}, {max(fits_g):.4f}]")

# ═══════════════════════════════════════════════════════════════════════
# EVOLUTION LOG ANALYSIS (births, trait drift)
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─'*60}")
print("EVOLUTION LOG — BIRTHS & DRIFT")
print(f"{'─'*60}")

evo_path = DATA_DIR / "evolution_log.jsonl"
evo2_path = DATA_DIR / "evolution_log.jsonl.1"

birth_count = 0
death_count_evo = 0
dormant_count = 0

for ep in [evo2_path, evo_path]:
    if ep.exists():
        with open(ep, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    evt = rec.get("event")
                    if evt == "birth":
                        birth_count += 1
                    elif evt == "death":
                        death_count_evo += 1
                    elif evt == "dormant":
                        dormant_count += 1
                except json.JSONDecodeError:
                    continue

print(f"  Total births recorded:  {birth_count}")
print(f"  Total deaths recorded:  {death_count_evo}")
print(f"  Total dormant entries:  {dormant_count}")
print(f"  Net population change:  {birth_count - death_count_evo:+d}")
