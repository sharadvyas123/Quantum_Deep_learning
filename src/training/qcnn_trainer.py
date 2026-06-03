"""
src/training/qcnn_trainer.py
-----------------------------
Training loops for the QCNN (PyTorch) and CNN baseline (PyTorch),
plus callbacks, CSV logging, and comparison plotting.

Why PyTorch for both models?
  Using the same framework makes the comparison clean — same optimizer (Adam),
  same loss (BCELoss), same training loop. The only difference is
  whether the feature extractor is a QNode or a Conv2D.

PennyLane + PyTorch integration reference:
  https://docs.pennylane.ai/en/stable/introduction/interfaces/torch.html

Barren plateau mitigation reference:
  Grant et al. (2019) https://arxiv.org/abs/1903.05076
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# History recorder
# ---------------------------------------------------------------------------

class TrainHistory:
    """Records loss and accuracy per epoch for post-training analysis."""

    def __init__(self, label: str = "model"):
        self.label  = label
        self.epochs: list[int]   = []
        self.losses: list[float] = []
        self.accs:   list[float] = []

    def record(self, epoch: int, loss: float, acc: float):
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.accs.append(acc)

    def to_csv(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss", "accuracy"])
            writer.writerows(zip(self.epochs, self.losses, self.accs))
        print(f"  Saved log → {path}")


# ---------------------------------------------------------------------------
# Generic PyTorch training loop (shared by QCNN and CNN)
# ---------------------------------------------------------------------------

def train_loop(
    model:      nn.Module,
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    X_val:      np.ndarray,
    y_val:      np.ndarray,
    epochs:     int   = 30,
    lr:         float = 0.01,
    batch_size: int   = 16,
    label:      str   = "model",
    log_dir:    str | Path = "results/logs",
    run_name:   str   = "run",
    print_every: int  = 5,
) -> dict:
    """
    Generic binary classification training loop using BCELoss + Adam.

    Works for both QCNNModel and any nn.Module that outputs (batch, 1) sigmoid probs.

    Returns
    -------
    dict with keys: train_acc, val_acc, elapsed_s, history, n_params
    """
    device    = torch.device("cpu")   # quantum circuits always run on CPU
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_vl = torch.tensor(X_val,   dtype=torch.float32)
    y_vl = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    history = TrainHistory(label=label)
    t0 = time.perf_counter()

    print(f"\n── Training {label} ──")
    print(f"   Params  : {sum(p.numel() for p in model.parameters())}")
    print(f"   LR      : {lr}   Epochs: {epochs}   Batch: {batch_size}")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        avg_loss = epoch_loss / len(X_train)

        # Validation accuracy
        model.eval()
        with torch.no_grad():
            val_pred = (model(X_vl) > 0.5).float()
            val_acc  = (val_pred == y_vl).float().mean().item()

        history.record(epoch, avg_loss, val_acc)

        if epoch % print_every == 0 or epoch == epochs:
            print(f"  Epoch {epoch:4d}/{epochs} | loss {avg_loss:.4f} | val acc {val_acc:.3f}")

    elapsed = time.perf_counter() - t0

    # Final metrics
    model.eval()
    with torch.no_grad():
        tr_pred  = (model(X_tr) > 0.5).float()
        train_acc = (tr_pred == y_tr).float().mean().item()
        vl_pred  = (model(X_vl) > 0.5).float()
        val_acc_final = (vl_pred == y_vl).float().mean().item()

    print(f"\n  Train acc : {train_acc:.3f}")
    print(f"  Val acc   : {val_acc_final:.3f}")
    print(f"  Time      : {elapsed:.1f}s")

    history.to_csv(Path(log_dir) / f"{run_name}_history.csv")

    return {
        "train_acc": train_acc,
        "val_acc":   val_acc_final,
        "elapsed_s": elapsed,
        "history":   history,
        "n_params":  sum(p.numel() for p in model.parameters()),
    }


# ---------------------------------------------------------------------------
# Classical CNN baseline (PyTorch, same training loop)
# ---------------------------------------------------------------------------

class ClassicalCNN(nn.Module):
    """
    Minimal CNN for 8×8 single-channel images.
    Filter count is tuned to roughly match the QCNN parameter budget.

    Reference: LeCun et al. (1998) http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
    """

    def __init__(self, image_size: int = 8, n_params_target: int = 90):
        super().__init__()
        # Heuristic: pick filter count so total params ≈ n_params_target
        n_filters = max(2, int(np.sqrt(n_params_target / 20)))

        self.features = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                          # 8→4
            nn.Conv2d(n_filters, n_filters * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                          # 4→2
        )
        flat = (image_size // 4) ** 2 * (n_filters * 2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, image_size, image_size)  or  (batch, 1, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)     # add channel dim
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Comparison plotting
# ---------------------------------------------------------------------------

def plot_comparison(
    qcnn_results: dict,
    cnn_results:  dict,
    save_path:    Optional[str | Path] = None,
):
    """
    Three-panel comparison:
      Left  — loss curves
      Centre — accuracy curves
      Right  — final accuracy bar chart with param counts
    """
    fig = plt.figure(figsize=(15, 4))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    q_h = qcnn_results["history"]
    c_h = cnn_results["history"]

    # ── Loss ──────────────────────────────────────────────────────────────
    ax1.plot(q_h.epochs, q_h.losses, color="#4477AA", lw=1.5, label="QCNN")
    ax1.plot(c_h.epochs, c_h.losses, color="#CC6633", lw=1.5, ls="--", label="CNN")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE Loss")
    ax1.set_title("Training loss"); ax1.legend(); ax1.grid(alpha=0.3)

    # ── Accuracy ──────────────────────────────────────────────────────────
    ax2.plot(q_h.epochs, q_h.accs, color="#4477AA", lw=1.5, label="QCNN val acc")
    ax2.plot(c_h.epochs, c_h.accs, color="#CC6633", lw=1.5, ls="--", label="CNN val acc")
    ax2.axhline(0.5, color="gray", ls=":", alpha=0.5, label="chance")
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation accuracy"); ax2.legend(); ax2.grid(alpha=0.3)

    # ── Bar chart ─────────────────────────────────────────────────────────
    labels = ["QCNN", "Classical CNN"]
    vals   = [qcnn_results["val_acc"], cnn_results["val_acc"]]
    colors = ["#4477AA", "#CC6633"]
    bars = ax3.bar(labels, vals, color=colors, alpha=0.85, width=0.45)
    ax3.set_ylim(0, 1.15); ax3.set_ylabel("Test accuracy")
    ax3.set_title("Final test accuracy"); ax3.axhline(0.5, color="gray", ls=":", alpha=0.5)
    for bar, v, n in zip(bars, vals, [qcnn_results["n_params"], cnn_results["n_params"]]):
        ax3.text(bar.get_x() + bar.get_width() / 2, v + 0.03,
                 f"{v:.1%}", ha="center", fontsize=11)
        ax3.text(bar.get_x() + bar.get_width() / 2, 0.02,
                 f"{n} params", ha="center", fontsize=8, color="white")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved figure → {save_path}")

    plt.tight_layout()
    return fig