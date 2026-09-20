import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Configuration
# ============================================================
_cwd = Path.cwd()

OUTPUT_DIR = _cwd / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "densenet121_classwise_metrics.png"

# ============================================================
# DenseNet-121 Classification Results
# ============================================================

classes = [
    "MEL",
    "NV",
    "BCC",
    "AKIEC",
    "BKL",
    "DF",
    "VASC"
]

precision = np.array([
    0.9365,
    0.8838,
    0.9865,
    0.9777,
    0.9821,
    0.9940,
    0.9985
])

recall = np.array([
    0.9239,
    0.9761,
    0.9806,
    0.9806,
    0.9001,
    0.9955,
    0.9925
])

f1_score = np.array([
    0.9301,
    0.9277,
    0.9836,
    0.9792,
    0.9393,
    0.9948,
    0.9955
])

# Macro averages from the classification report
macro_precision = 0.9656
macro_recall = 0.9642
macro_f1 = 0.9643

# ============================================================
# Plot
# ============================================================

x = np.arange(len(classes))
width = 0.24

fig, ax = plt.subplots(figsize=(7.2, 4.6))

# Color palette suitable for academic figures
color_precision = "#4C78A8"
color_recall = "#F58518"
color_f1 = "#54A24B"

bars_precision = ax.bar(
    x - width,
    precision,
    width,
    label="Precision",
    color=color_precision
)

bars_recall = ax.bar(
    x,
    recall,
    width,
    label="Recall",
    color=color_recall
)

bars_f1 = ax.bar(
    x + width,
    f1_score,
    width,
    label="F1-Score",
    color=color_f1
)

# ============================================================
# Macro-average reference lines
# ============================================================

ax.axhline(
    macro_precision,
    color=color_precision,
    linestyle="--",
    linewidth=1.2,
    alpha=0.65
)

ax.axhline(
    macro_recall,
    color=color_recall,
    linestyle="--",
    linewidth=1.2,
    alpha=0.65
)

ax.axhline(
    macro_f1,
    color=color_f1,
    linestyle="--",
    linewidth=1.2,
    alpha=0.65
)

# ============================================================
# Formatting
# ============================================================

ax.set_xlabel(
    "Skin Lesion Class",
    fontsize=11
)

ax.set_ylabel(
    "Score",
    fontsize=11
)

ax.set_xticks(x)
ax.set_xticklabels(
    classes,
    fontsize=10
)

ax.set_ylim(0.85, 1.01)

ax.set_yticks(
    np.arange(0.85, 1.01, 0.025)
)

ax.tick_params(
    axis="y",
    labelsize=9
)

ax.legend(
    loc="lower right",
    fontsize=9,
    frameon=True
)

# Light horizontal grid for readability
ax.yaxis.grid(
    True,
    linestyle=":",
    linewidth=0.7,
    alpha=0.5
)

ax.set_axisbelow(True)

# Remove unnecessary top/right borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ============================================================
# Value labels
# ============================================================

def add_labels(bars):
    for bar in bars:
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.002,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90
        )

add_labels(bars_precision)
add_labels(bars_recall)
add_labels(bars_f1)

# ============================================================
# Save
# ============================================================

plt.tight_layout()

fig.savefig(
    OUTPUT_FILE,
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)

print("=" * 60)
print("DenseNet-121 class-wise metrics figure generated")
print("=" * 60)
print(f"Saved to: {OUTPUT_FILE}")