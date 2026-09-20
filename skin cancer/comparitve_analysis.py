import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Paths
# ============================================================

_cwd = Path.cwd()
ROOT = _cwd.parent

OUTPUT_DIR = ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "gan_vs_qgan_classification_comparison.png"

# ============================================================
# Data
# ============================================================

models = [
    "CNN",
    "EfficientNet-B3",
    "DenseNet-121",
    "ResNet-50"
]

# Classification metrics (%)
gan_accuracy  = [93.63, np.nan, 94.93, 94.91]
qgan_accuracy = [94.16, 94.38, 96.42, 96.04]

gan_precision  = [95.03, np.nan, 95.10, 95.07]
qgan_precision = [94.81, 94.76, 96.56, 96.08]

gan_f1  = [93.90, np.nan, 94.93, 94.93]
qgan_f1 = [94.27, 94.44, 96.43, 96.05]

# AUC
gan_auc  = [0.9963, np.nan, 0.9973, 0.9973]
qgan_auc = [0.9964, 0.9955, 0.9986, 0.9984]

# ============================================================
# Colors
# ============================================================

# GAN = lighter
# QGAN = darker

accuracy_gan  = "#A8D5C5"
accuracy_qgan = "#4FB394"

precision_gan  = "#FDB08A"
precision_qgan = "#F47F4C"

f1_gan  = "#AABBDD"
f1_qgan = "#7189B8"

auc_gan  = "#E6A1C9"
auc_qgan = "#D66BAA"

# ============================================================
# Figure
# ============================================================

fig, ax1 = plt.subplots(figsize=(11, 6.5))

x = np.arange(len(models))

# Thick bars
width = 0.20

# ============================================================
# BAR POSITIONS
# ============================================================

# 8 bars per classifier
positions = {
    "acc_gan":  x - 3.5 * width,
    "acc_qgan": x - 2.5 * width,

    "pre_gan":  x - 1.5 * width,
    "pre_qgan": x - 0.5 * width,

    "f1_gan":   x + 0.5 * width,
    "f1_qgan":  x + 1.5 * width,

    "auc_gan":  x + 2.5 * width,
    "auc_qgan": x + 3.5 * width,
}

# ============================================================
# LEFT AXIS
# ============================================================

# Use bottom=90 so the visual focuses on the actual differences
BOTTOM = 90

bars_acc_gan = ax1.bar(
    positions["acc_gan"],
    np.array(gan_accuracy) - BOTTOM,
    width,
    bottom=BOTTOM,
    color=accuracy_gan,
    edgecolor="black",
    linewidth=1.0,
    label="Accuracy (GAN)"
)

bars_acc_qgan = ax1.bar(
    positions["acc_qgan"],
    np.array(qgan_accuracy) - BOTTOM,
    width,
    bottom=BOTTOM,
    color=accuracy_qgan,
    edgecolor="black",
    linewidth=1.0,
    label="Accuracy (QGAN)"
)

bars_pre_gan = ax1.bar(
    positions["pre_gan"],
    np.array(gan_precision) - BOTTOM,
    width,
    bottom=BOTTOM,
    color=precision_gan,
    edgecolor="black",
    linewidth=1.0,
    label="Precision (GAN)"
)

bars_pre_qgan = ax1.bar(
    positions["pre_qgan"],
    np.array(qgan_precision) - BOTTOM,
    width,
    bottom=BOTTOM,
    color=precision_qgan,
    edgecolor="black",
    linewidth=1.0,
    label="Precision (QGAN)"
)

bars_f1_gan = ax1.bar(
    positions["f1_gan"],
    np.array(gan_f1) - BOTTOM,
    width,
    bottom=BOTTOM,
    color=f1_gan,
    edgecolor="black",
    linewidth=1.0,
    label="F1-score (GAN)"
)

bars_f1_qgan = ax1.bar(
    positions["f1_qgan"],
    np.array(qgan_f1) - BOTTOM,
    width,
    bottom=BOTTOM,
    color=f1_qgan,
    edgecolor="black",
    linewidth=1.0,
    label="F1-score (QGAN)"
)

# ============================================================
# CLASSIFICATION VALUE LABELS
# ============================================================

def add_labels_inside(bars, values):

    for bar, value in zip(bars, values):

        if np.isnan(value):
            continue

        # Put value slightly below top of bar
        y = value - 0.28

        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{value:.2f}",
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold",
            rotation=90
        )


add_labels_inside(bars_acc_gan, gan_accuracy)
add_labels_inside(bars_acc_qgan, qgan_accuracy)

add_labels_inside(bars_pre_gan, gan_precision)
add_labels_inside(bars_pre_qgan, qgan_precision)

add_labels_inside(bars_f1_gan, gan_f1)
add_labels_inside(bars_f1_qgan, qgan_f1)

# ============================================================
# RIGHT AXIS FOR AUC
# ============================================================

ax2 = ax1.twinx()

AUC_BOTTOM = 0.990

bars_auc_gan = ax2.bar(
    positions["auc_gan"],
    np.array(gan_auc) - AUC_BOTTOM,
    width,
    bottom=AUC_BOTTOM,
    color=auc_gan,
    edgecolor="black",
    linewidth=1.0,
    label="AUC (GAN)"
)

bars_auc_qgan = ax2.bar(
    positions["auc_qgan"],
    np.array(qgan_auc) - AUC_BOTTOM,
    width,
    bottom=AUC_BOTTOM,
    color=auc_qgan,
    edgecolor="black",
    linewidth=1.0,
    label="AUC (QGAN)"
)

# ============================================================
# AUC LABELS
# ============================================================

# Slightly stagger the labels so close values don't overlap
auc_offsets_gan = [0.00020, 0.00020, 0.00020, 0.00020]
auc_offsets_qgan = [0.00045, 0.00045, 0.00045, 0.00045]

for bar, value, offset in zip(
    bars_auc_gan,
    gan_auc,
    auc_offsets_gan
):

    if np.isnan(value):
        continue

    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        value + offset,
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        rotation=45
    )


for bar, value, offset in zip(
    bars_auc_qgan,
    qgan_auc,
    auc_offsets_qgan
):

    if np.isnan(value):
        continue

    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        value + offset,
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        rotation=45
    )

# ============================================================
# AXES
# ============================================================

ax1.set_ylim(90, 100.8)
ax2.set_ylim(0.990, 1.0012)

ax1.set_ylabel(
    "Accuracy / Precision / F1-score (%)",
    fontsize=12,
    fontweight="bold"
)

ax2.set_ylabel(
    "ROC-AUC",
    fontsize=12,
    fontweight="bold"
)

ax1.set_xlabel(
    "Classifier Model",
    fontsize=12,
    fontweight="bold"
)

ax1.set_xticks(x)
ax1.set_xticklabels(
    models,
    fontsize=11
)

# ============================================================
# TITLE
# ============================================================

ax1.set_title(
    "Comparison of Classification Performance Across GAN and QGAN",
    fontsize=15,
    fontweight="bold",
    pad=15
)

# ============================================================
# GRID
# ============================================================

ax1.grid(
    axis="y",
    linestyle="--",
    linewidth=0.6,
    alpha=0.35
)

ax1.set_axisbelow(True)

# ============================================================
# LEGEND
# ============================================================

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    handles1 + handles2,
    labels1 + labels2,
    ncol=3,
    fontsize=8.5,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    frameon=True
)

# ============================================================
# Layout
# ============================================================

plt.tight_layout()

# ============================================================
# Save
# ============================================================

plt.savefig(
    OUTPUT_FILE,
    dpi=600,
    bbox_inches="tight"
)

plt.show()

print(f"Figure saved to: {OUTPUT_FILE}")