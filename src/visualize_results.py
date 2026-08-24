import csv
from pathlib import Path

import matplotlib.pyplot as plt


INPUT_FILE = Path("results/benchmark_summary.csv")
OUTPUT_FILE = Path("results/retrieval_quality_comparison.png")

METRICS = [
    ("hit_at_1", "Hit@1"),
    ("hit_at_5", "Hit@5"),
    ("hit_at_10", "Hit@10"),
    ("recall_at_5", "Recall@5"),
    ("recall_at_10", "Recall@10"),
    ("ndcg_at_5", "nDCG@5"),
    ("ndcg_at_10", "nDCG@10"),
    ("mrr", "MRR"),
]


with INPUT_FILE.open(newline="") as file:
    rows = list(csv.DictReader(file))

bge = next(row for row in rows if row["model"] == "BGE-M3-dense")
evie = next(row for row in rows if row["model"] == "EVIE-Preview-4.5B")

labels = [label for _, label in METRICS]
bge_values = [float(bge[key]) for key, _ in METRICS]
evie_values = [float(evie[key]) for key, _ in METRICS]

x = range(len(labels))
width = 0.36

fig, ax = plt.subplots(figsize=(12, 6))

bge_bars = ax.bar(
    [position - width / 2 for position in x],
    bge_values,
    width,
    label="BGE-M3 Dense",
)

evie_bars = ax.bar(
    [position + width / 2 for position in x],
    evie_values,
    width,
    label="EVIE-Preview-4.5B",
)

ax.set_title("Page-Level Retrieval Quality: BGE-M3 vs EVIE")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.set_xticks(list(x))
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.legend()

ax.bar_label(bge_bars, fmt="%.3f", padding=3, fontsize=8)
ax.bar_label(evie_bars, fmt="%.3f", padding=3, fontsize=8)

fig.tight_layout()
fig.savefig(OUTPUT_FILE, dpi=200)

print(f"Saved chart to {OUTPUT_FILE}")
