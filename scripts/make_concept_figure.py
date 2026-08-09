"""Generate the conceptual QuaRM workflow figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "figures" / "figure1_framework.png"


def box(ax, x, y, w, h, title, body, color):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.018", linewidth=1.4,
        edgecolor=color, facecolor="white"
    )
    ax.add_patch(patch)
    ax.text(x + 0.02, y + h - 0.06, title, fontsize=10, fontweight="bold", color=color, va="top")
    ax.text(x + 0.02, y + h - 0.14, body, fontsize=8, color="#30343B", va="top", linespacing=1.35)


fig, ax = plt.subplots(figsize=(12.5, 4.2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
blue, orange, purple, green = "#0072B2", "#D55E00", "#7A5195", "#009E73"
box(ax, 0.02, 0.27, 0.18, 0.48, "1  Reference contract", "Trusted training snapshot\nClean evaluation distribution\nLearner and bounded loss", blue)
box(ax, 0.27, 0.17, 0.20, 0.68, "2  Factorial interventions", "Every defect coalition\nMissing cells\nFeature noise\nTarget noise\nDuplicate rows\n\nPaired seeds and splits", orange)
box(ax, 0.54, 0.27, 0.18, 0.48, "3  Risk surface", "R(S) for all S\nMain effects\nNon-additive interactions\nRepeat-level observations", purple)
box(ax, 0.79, 0.17, 0.19, 0.68, "4  Decision evidence", "Exact debt allocation\nBootstrap intervals\nRanking probabilities\nRepair priorities\n\nAssumptions stay attached", green)
for x1, x2, color in [(0.20, 0.27, blue), (0.47, 0.54, orange), (0.72, 0.79, purple)]:
    ax.add_patch(FancyArrowPatch((x1 + 0.005, 0.51), (x2 - 0.005, 0.51), arrowstyle="-|>", mutation_scale=14, lw=1.6, color=color))
ax.text(0.5, 0.96, "QuaRM turns a quality checklist into an auditable counterfactual experiment", ha="center", va="top", fontsize=15, fontweight="bold", color="#20242A")
ax.text(0.5, 0.05, "The output is conditional on the reference, task, model, loss, mechanisms, severity, and randomization policy.", ha="center", fontsize=9, color="#50555D")
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(OUT)
