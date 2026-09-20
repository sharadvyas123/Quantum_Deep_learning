# T-SNE.py

from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.models import densenet121, DenseNet121_Weights

from PIL import Image

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

_cwd = Path.cwd()

CHECKPOINT_DIR = _cwd / "checkpoints"
DENSENET_121_MODEL = CHECKPOINT_DIR / "densenet121_best.pt"

DATA_ROOT = _cwd / "dataset" / "processed_images"

RESULTS_DIR = _cwd / "results" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "densenet121_tsne_real_vs_synthetic.png"


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42
IMG_SIZE = 64
BATCH_SIZE = 64
NUM_CLASSES = 7

CLASS_NAMES = [
    "MEL",
    "NV",
    "BCC",
    "AKIEC",
    "BKL",
    "DF",
    "VASC",
]

MINORITY_CLASSES = [
    "MEL",
    "BCC",
    "AKIEC",
    "BKL",
    "DF",
    "VASC",
]

# Number of images used per source/class.
# This keeps the visualization balanced.
SAMPLES_PER_GROUP = 100

TSNE_PERPLEXITY = 30
TSNE_ITERATIONS = 2000

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Transform
# Same deterministic preprocessing used during evaluation
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

eval_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),
])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TSNEDataset(Dataset):

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label, source = self.samples[idx]

        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label, source


# ─────────────────────────────────────────────────────────────────────────────
# Utility: collect images
# ─────────────────────────────────────────────────────────────────────────────

def collect_images(directory):

    if not directory.exists():
        return []

    images = (
        list(directory.glob("*.jpg"))
        + list(directory.glob("*.jpeg"))
        + list(directory.glob("*.png"))
    )

    return sorted(images)


# ─────────────────────────────────────────────────────────────────────────────
# Build balanced t-SNE sample list
# ─────────────────────────────────────────────────────────────────────────────

def build_tsne_samples():

    samples = []

    for class_idx, class_name in enumerate(CLASS_NAMES):

        real_dir = DATA_ROOT / class_name / "real"

        real_images = collect_images(real_dir)

        if len(real_images) == 0:
            print(f"[WARN] No real images found for {class_name}")
            continue

        # ---------------------------------------------------------
        # Real images
        # ---------------------------------------------------------

        n_real = min(
            SAMPLES_PER_GROUP,
            len(real_images)
        )

        rng = random.Random(
            SEED + class_idx
        )

        real_selected = rng.sample(
            real_images,
            n_real
        )

        for path in real_selected:
            samples.append(
                (path, class_idx, "Real")
            )

        # ---------------------------------------------------------
        # Synthetic images
        # ---------------------------------------------------------

        if class_name in MINORITY_CLASSES:

            synthetic_dir = (
                DATA_ROOT / class_name / "normalized"
            )

            synthetic_images = collect_images(
                synthetic_dir
            )

            if len(synthetic_images) == 0:
                print(
                    f"[WARN] No synthetic images found for "
                    f"{class_name}"
                )
                continue

            n_synthetic = min(
                SAMPLES_PER_GROUP,
                len(synthetic_images)
            )

            synthetic_selected = rng.sample(
                synthetic_images,
                n_synthetic
            )

            for path in synthetic_selected:
                samples.append(
                    (path, class_idx, "Synthetic")
                )

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Build DenseNet-121
# ─────────────────────────────────────────────────────────────────────────────

def build_model(num_classes):

    model = densenet121(
        weights=DenseNet121_Weights.DEFAULT
    )

    in_features = model.classifier.in_features

    model.classifier = nn.Linear(
        in_features,
        num_classes
    )

    return model


print("\nLoading DenseNet-121...")

model = build_model(NUM_CLASSES).to(DEVICE)

checkpoint = torch.load(
    DENSENET_121_MODEL,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    f"Checkpoint loaded successfully "
    f"(epoch={checkpoint.get('epoch', 'N/A')}, "
    f"val_acc={checkpoint.get('val_acc', 'N/A')})"
)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(model, loader):

    all_features = []
    all_labels = []
    all_sources = []

    for imgs, labels, sources in loader:

        imgs = imgs.to(DEVICE)

        # DenseNet feature extractor
        x = model.features(imgs)

        x = F.relu(
            x,
            inplace=False
        )

        x = F.adaptive_avg_pool2d(
            x,
            (1, 1)
        )

        x = torch.flatten(
            x,
            1
        )

        all_features.append(
            x.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

        all_sources.extend(
            sources
        )

    features = np.concatenate(
        all_features,
        axis=0
    )

    labels = np.asarray(
        all_labels
    )

    sources = np.asarray(
        all_sources
    )

    return features, labels, sources


# ─────────────────────────────────────────────────────────────────────────────
# Prepare samples
# ─────────────────────────────────────────────────────────────────────────────

print("\nCollecting t-SNE samples...")

samples = build_tsne_samples()

print(
    f"Total samples selected: {len(samples)}"
)

for class_idx, class_name in enumerate(CLASS_NAMES):

    real_count = sum(
        1
        for _, label, source in samples
        if label == class_idx
        and source == "Real"
    )

    synthetic_count = sum(
        1
        for _, label, source in samples
        if label == class_idx
        and source == "Synthetic"
    )

    print(
        f"{class_name:>6}: "
        f"Real={real_count:3d}, "
        f"Synthetic={synthetic_count:3d}"
    )


dataset = TSNEDataset(
    samples,
    transform=eval_transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ─────────────────────────────────────────────────────────────────────────────
# Extract DenseNet features
# ─────────────────────────────────────────────────────────────────────────────

print("\nExtracting DenseNet-121 features...")

features, labels, sources = extract_features(
    model,
    loader
)

print(
    f"Feature matrix: {features.shape}"
)


# ─────────────────────────────────────────────────────────────────────────────
# PCA
# ─────────────────────────────────────────────────────────────────────────────

print("\nApplying PCA...")

n_components = min(
    50,
    features.shape[1],
    features.shape[0] - 1
)

pca = PCA(
    n_components=n_components,
    random_state=SEED
)

features_pca = pca.fit_transform(
    features
)

print(
    f"PCA output: {features_pca.shape}"
)


# ─────────────────────────────────────────────────────────────────────────────
# t-SNE
# ─────────────────────────────────────────────────────────────────────────────

print("\nRunning t-SNE...")

tsne = TSNE(
    n_components=2,
    perplexity=TSNE_PERPLEXITY,
    max_iter=TSNE_ITERATIONS,
    init="pca",
    learning_rate="auto",
    random_state=SEED
)

features_tsne = tsne.fit_transform(
    features_pca
)

print("t-SNE completed.")


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(
    figsize=(9, 7)
)

# Publication-friendly class colors
class_colors = {
    "MEL": "#4C78A8",
    "NV": "#F58518",
    "BCC": "#54A24B",
    "AKIEC": "#E45756",
    "BKL": "#B279A2",
    "DF": "#FF9DA6",
    "VASC": "#9D755D",
}

# Real = circles
# Synthetic = X
for class_idx, class_name in enumerate(CLASS_NAMES):

    color = class_colors[class_name]

    # Real
    mask_real = (
        (labels == class_idx)
        & (sources == "Real")
    )

    ax.scatter(
        features_tsne[mask_real, 0],
        features_tsne[mask_real, 1],
        s=32,
        alpha=0.65,
        marker="o",
        color=color,
        edgecolors="none",
        label=f"{class_name} - Real"
    )

    # Synthetic
    mask_synthetic = (
        (labels == class_idx)
        & (sources == "Synthetic")
    )

    if np.any(mask_synthetic):

        ax.scatter(
            features_tsne[mask_synthetic, 0],
            features_tsne[mask_synthetic, 1],
            s=48,
            alpha=0.85,
            marker="x",
            color=color,
            linewidths=1.3,
            label=f"{class_name} - Synthetic"
        )


ax.set_xlabel(
    "t-SNE Dimension 1",
    fontsize=12
)

ax.set_ylabel(
    "t-SNE Dimension 2",
    fontsize=12
)

ax.set_title(
    "t-SNE Visualization of DenseNet-121 Feature Representations",
    fontsize=13,
    pad=12
)

ax.tick_params(
    axis="both",
    labelsize=10
)

ax.grid(
    alpha=0.2,
    linewidth=0.7
)

ax.legend(
    fontsize=8,
    loc="best",
    ncol=2,
    frameon=True
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=600,
    bbox_inches="tight"
)

plt.close()

print(
    f"\nFigure saved to:\n{OUTPUT_FILE}"
)

# ─────────────────────────────────────────────────────────────────────────────
# PCA: 1024-D DenseNet features → 2-D
# ─────────────────────────────────────────────────────────────────────────────

print("\nRunning PCA...")

pca_2d = PCA(
    n_components=2,
    random_state=SEED
)

features_pca_2d = pca_2d.fit_transform(features)

explained_variance = pca_2d.explained_variance_ratio_ * 100

print(
    f"PC1 explained variance: {explained_variance[0]:.2f}%"
)

print(
    f"PC2 explained variance: {explained_variance[1]:.2f}%"
)

print(
    f"Total explained variance: "
    f"{explained_variance.sum():.2f}%"
)
# ─────────────────────────────────────────────────────────────────────────────
# PCA plot
# ─────────────────────────────────────────────────────────────────────────────

PCA_OUTPUT_FILE = (
    RESULTS_DIR / "densenet121_pca_real_vs_synthetic.png"
)

fig, ax = plt.subplots(
    figsize=(9, 7)
)

# Same class colors used for t-SNE
class_colors = {
    "MEL": "#4C78A8",
    "NV": "#F58518",
    "BCC": "#54A24B",
    "AKIEC": "#E45756",
    "BKL": "#B279A2",
    "DF": "#FF9DA6",
    "VASC": "#9D755D",
}

for class_idx, class_name in enumerate(CLASS_NAMES):

    color = class_colors[class_name]

    # ─────────────────────────────────────────────
    # Real
    # ─────────────────────────────────────────────

    mask_real = (
        (labels == class_idx)
        & (sources == "Real")
    )

    ax.scatter(
        features_pca_2d[mask_real, 0],
        features_pca_2d[mask_real, 1],
        s=32,
        alpha=0.65,
        marker="o",
        color=color,
        edgecolors="none",
        label=f"{class_name} - Real"
    )

    # ─────────────────────────────────────────────
    # Synthetic
    # ─────────────────────────────────────────────

    mask_synthetic = (
        (labels == class_idx)
        & (sources == "Synthetic")
    )

    if np.any(mask_synthetic):

        ax.scatter(
            features_pca_2d[mask_synthetic, 0],
            features_pca_2d[mask_synthetic, 1],
            s=48,
            alpha=0.85,
            marker="x",
            color=color,
            linewidths=1.3,
            label=f"{class_name} - Synthetic"
        )


ax.set_xlabel(
    f"Principal Component 1 "
    f"({explained_variance[0]:.2f}% variance)",
    fontsize=12
)

ax.set_ylabel(
    f"Principal Component 2 "
    f"({explained_variance[1]:.2f}% variance)",
    fontsize=12
)

ax.set_title(
    "PCA Visualization of DenseNet-121 Feature Representations",
    fontsize=13,
    pad=12
)

ax.tick_params(
    axis="both",
    labelsize=10
)

ax.grid(
    alpha=0.2,
    linewidth=0.7
)

ax.legend(
    fontsize=8,
    loc="best",
    ncol=2,
    frameon=True
)

plt.tight_layout()

plt.savefig(
    PCA_OUTPUT_FILE,
    dpi=600,
    bbox_inches="tight"
)

plt.close()

print(
    f"\nPCA figure saved to:\n{PCA_OUTPUT_FILE}"
)