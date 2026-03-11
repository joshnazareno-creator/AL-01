"""Generate fitness-over-time graph with revival/death markers."""
import csv
from pathlib import Path

DATA_DIR = Path(r"D:\AL-01\data")
csv_path = DATA_DIR / "fitness_timeline.csv"
out_path = DATA_DIR / "fitness_over_time.png"

cycles, fitnesses, energies = [], [], []
revival_x, revival_y = [], []
death_x, death_y = [], []

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        c = int(row["cycle"])
        fit = float(row["fitness"])
        eng = float(row["energy"])
        cycles.append(c)
        fitnesses.append(fit)
        energies.append(eng)
        if row.get("is_revival") == "True":
            revival_x.append(c)
            revival_y.append(fit)
        if row.get("is_death") == "True":
            death_x.append(c)
            death_y.append(fit)

# Use Agg backend (no display needed)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(16, 7))

# Fitness line
ax.plot(cycles, fitnesses, color="#3b82f6", linewidth=0.6, alpha=0.85, label="Fitness", zorder=2)

# Energy (secondary, lighter)
ax.plot(cycles, energies, color="#22c55e", linewidth=0.4, alpha=0.45, label="Energy", zorder=1)

# Revival markers
if revival_x:
    ax.scatter(revival_x, revival_y, color="#f59e0b", s=12, marker="^",
               alpha=0.7, zorder=3, label=f"Revival ({len(revival_x)})")

# Death markers
if death_x:
    ax.scatter(death_x, death_y, color="#ef4444", s=12, marker="v",
               alpha=0.7, zorder=3, label=f"Death ({len(death_x)})")

ax.set_xlabel("Cycle", fontsize=12)
ax.set_ylabel("Fitness / Energy", fontsize=12)
ax.set_title("AL-01  Fitness Over Time  (with revival & death events)", fontsize=14, fontweight="bold")
ax.legend(loc="upper left", fontsize=10)
ax.set_ylim(-0.02, 1.05)
ax.grid(True, alpha=0.2)
ax.set_facecolor("#0f172a")
fig.patch.set_facecolor("#1e293b")
ax.title.set_color("white")
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_color("#334155")
ax.legend(loc="upper left", fontsize=10, facecolor="#1e293b", edgecolor="#334155", labelcolor="white")

fig.tight_layout()
fig.savefig(str(out_path), dpi=150)
print(f"Saved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
print(f"  Cycles: {cycles[0]} → {cycles[-1]} ({len(cycles)} data points)")
print(f"  Revivals plotted: {len(revival_x)}")
print(f"  Deaths plotted:   {len(death_x)}")
