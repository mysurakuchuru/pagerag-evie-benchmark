import csv
from pathlib import Path

import matplotlib.pyplot as plt


INPUT_FILE = Path("results/benchmark_summary.csv")
OUTPUT_FILE = Path("results/t4_query_latency_comparison.png")


with INPUT_FILE.open(newline="") as file:
    rows = list(csv.DictReader(file))

bge = next(row for row in rows if row["model"] == "BGE-M3-dense")
evie = next(row for row in rows if row["model"] == "EVIE-Preview-4.5B")

models = ["BGE-M3", "EVIE-Preview-4.5B"]

latencies = [
    float(bge["avg_query_latency_seconds"]),
    float(evie["avg_query_latency_seconds"]),
]

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(models, latencies)

ax.set_title("Average Query Latency on NVIDIA Tesla T4")
ax.set_ylabel("Seconds per query")

ax.bar_label(
    bars,
    labels=[f"{value:.4f} s" for value in latencies],
    padding=4,
)

ax.text(
    0.5,
    -0.14,
    "Measured implementation latency; EVIE scoring pipeline is not equally optimized.",
    transform=ax.transAxes,
    ha="center",
    fontsize=9,
)

fig.tight_layout()
fig.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")

print(f"Saved chart to {OUTPUT_FILE}")
