"""
src/utils/mnist_loader.py
--------------------------
MNIST preprocessing for QCNN vs CNN comparison.

Reduction strategy (standard in QML literature):
  1. Binary subset    — restrict to 2 digits (default 0 vs 1)
  2. Downsample       — resize 28×28 → 8×8 via average pooling
  3. Angle encoding   — normalise to [0, π] for RY rotation embedding
  4. CNN format       — keep 2D (8×8×1) for the classical baseline

References:
  Farhi & Neven (2018) — https://arxiv.org/abs/1802.06002
  Grant et al. (2018)  — https://arxiv.org/abs/1804.03680
  PennyLane QCNN demo  — https://pennylane.ai/qml/demos/tutorial_QCNN/
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import zoom


def load_mnist_binary(
    class_a: int = 0,
    class_b: int = 1,
    n_train: int = 200,
    n_test:  int = 50,
    image_size: int = 8,
    n_qubits:   int = 8,
    random_state: int = 42,
) -> dict:
    """
    Load MNIST, filter to two classes, downsample, and return both
    quantum (angle-encoded) and CNN (2D image) formats.

    Returns dict with:
      X_train_quantum  (n_train, n_qubits)     float64, values in [0, π]
      X_test_quantum   (n_test,  n_qubits)
      X_train_cnn      (n_train, image_size, image_size)   float32, [0,1]
      X_test_cnn       (n_test,  image_size, image_size)
      y_train, y_test  int arrays {0, 1}
      meta             dict
    """
    print(f"Loading MNIST (digits {class_a} vs {class_b})…")
    mnist   = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X_raw   = mnist.data
    y_raw   = mnist.target.astype(int)

    mask     = (y_raw == class_a) | (y_raw == class_b)
    X_filt   = X_raw[mask]
    y_filt   = (y_raw[mask] == class_b).astype(int)

    rng  = np.random.default_rng(random_state)
    idx  = rng.choice(len(X_filt), size=min(n_train + n_test, len(X_filt)), replace=False)
    X_sub, y_sub = X_filt[idx], y_filt[idx]

    X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
        X_sub, y_sub,
        train_size=n_train, test_size=n_test,
        stratify=y_sub, random_state=random_state,
    )

    X_train_img = _resize_batch(X_tr_raw, image_size)
    X_test_img  = _resize_batch(X_te_raw, image_size)

    # CNN format: normalised to [0, 1]
    X_train_cnn = (X_train_img / 255.0).astype(np.float32)
    X_test_cnn  = (X_test_img  / 255.0).astype(np.float32)

    # Quantum format: column-average → n_qubits features → [0, π]
    X_train_q = _to_quantum_features(X_train_img, n_qubits)
    X_test_q  = _to_quantum_features(X_test_img,  n_qubits)

    print(f"  Train {len(y_train)} | Test {len(y_test)}")
    print(f"  Quantum shape : {X_train_q.shape}  range [{X_train_q.min():.2f}, {X_train_q.max():.2f}]")
    print(f"  CNN shape     : {X_train_cnn.shape}")

    return {
        "X_train_quantum": X_train_q,
        "X_test_quantum":  X_test_q,
        "X_train_cnn":     X_train_cnn,
        "X_test_cnn":      X_test_cnn,
        "y_train":         y_train,
        "y_test":          y_test,
        "meta": {
            "class_a": class_a, "class_b": class_b,
            "image_size": image_size, "n_qubits": n_qubits,
            "n_train": len(y_train), "n_test": len(y_test),
        },
    }


def _resize_batch(X: np.ndarray, target: int) -> np.ndarray:
    n    = len(X)
    out  = np.zeros((n, target, target), dtype=np.float32)
    scale = target / 28
    for i, img in enumerate(X):
        out[i] = zoom(img.reshape(28, 28), scale, order=1)
    return out


def _to_quantum_features(X_img: np.ndarray, n_qubits: int) -> np.ndarray:
    """Column-average → n_qubits features → scale to [0, π]."""
    col_avg = X_img.mean(axis=1)          # (N, W)
    _, w    = col_avg.shape

    if w == n_qubits:
        features = col_avg
    elif w > n_qubits:
        step     = w // n_qubits
        features = np.stack(
            [col_avg[:, k*step:(k+1)*step].mean(1) for k in range(n_qubits)], axis=1
        )
    else:
        features = np.repeat(col_avg, (n_qubits // w) + 1, axis=1)[:, :n_qubits]

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    return scaler.fit_transform(features).astype(np.float64)