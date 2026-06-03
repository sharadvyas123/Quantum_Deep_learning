# Experiment 01 — QCNN vs Classical CNN on MNIST (PennyLane)

## Install

```bash
# From project root
uv add pennylane pennylane-lightning torch scikit-learn scipy matplotlib
```

`pennylane-lightning` — faster C++ simulator, drop-in replacement for `default.qubit`.

## Run

```bash
jupyter lab experiments/01_qcnn_mnist/qcnn_vs_cnn_mnist.ipynb
```

## Files

```
01_qcnn_mnist/
├── qcnn_vs_cnn_mnist.ipynb   ← Main notebook
├── config.py                  ← All hyperparameters
└── README.md

src/circuits/qcnn_layer.py       ← Conv + pool unitaries (PennyLane ops)
src/models/qcnn_classifier.py    ← QNode + qml.qnn.TorchLayer wrapper
src/training/qcnn_trainer.py     ← Shared PyTorch training loop + plots
src/utils/mnist_loader.py        ← MNIST downsampling + angle encoding
```

## Architecture

```
8 features → RY AngleEmbedding → Conv1 → Pool1 → Conv2 → Pool2 → <Z> → sigmoid → class
                                  (8 pairs)  (8→4q)  (4 pairs)  (4→2q)
```

Total trainable parameters: **92**

## Gradient methods

| `DIFF_METHOD` | Speed | Hardware-compatible | Notes |
|---|---|---|---|
| `backprop` | 3–5× faster | ✗ simulator only | Best for iterating locally |
| `adjoint` | fastest | ✗ simulator only | `default.qubit` only |

Change in `config.py`.

## Key references

| Paper | Link |
|---|---|
| Cong et al. (2019) — QCNN | https://arxiv.org/abs/1810.03912 |
| Hur et al. (2022) — QCNN on classical data | https://arxiv.org/abs/2108.00661 |
| Bergholm et al. (2018) — PennyLane | https://arxiv.org/abs/1811.04968 |
| Schuld et al. (2019) — Parameter-shift | https://arxiv.org/abs/1811.11184 |
| Grant et al. (2019) — Barren plateau init | https://arxiv.org/abs/1903.05076 |

Tutorials:
- https://pennylane.ai/qml/demos/tutorial_QCNN/
- https://docs.pennylane.ai/en/stable/introduction/interfaces/torch.html