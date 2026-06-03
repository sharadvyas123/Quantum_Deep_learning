"""
src/circuits/qcnn_layer.py
--------------------------
QCNN circuit primitives using PennyLane.

Architecture references:
  Cong et al. (2019) "Quantum Convolutional Neural Networks"
  https://arxiv.org/abs/1810.03912

  Hur et al. (2022) "Quantum Convolutional Neural Network for Classical Data Classification"
  https://arxiv.org/abs/2108.00661

PennyLane docs:
  https://docs.pennylane.ai/en/stable/introduction/circuits.html
  https://docs.pennylane.ai/en/stable/releases/changelog-0.36.0.html

The QCNN is built from:
  1. Angle embedding   — encodes each pixel as an RY rotation angle
  2. Conv unitary      — 2-qubit parameterised gate applied to adjacent qubit pairs
  3. Pool unitary      — 2-qubit gate that "measures" one qubit and feeds a
                         classically-controlled rotation to the other, halving
                         active qubit count per pooling step
"""

from __future__ import annotations
import numpy as np
import pennylane as qml


# ---------------------------------------------------------------------------
# Convolution unitary  (6 params — lightweight)
# ---------------------------------------------------------------------------

def conv_unitary(params: np.ndarray, wires: list):
    """
    2-qubit parameterised convolution gate — 6 parameters.

    Expressive enough for binary MNIST; faster than the full SU(4) 15-param gate.
    Equivalent to the 'simplified two-design' layer in PennyLane.

    Reference:
      Cerezo et al. (2021) "Variational quantum algorithms"
      https://arxiv.org/abs/2012.09265  (Sec. 4.1 on ansatz design)

    Parameters
    ----------
    params : array of shape (6,)
    wires  : [wire_0, wire_1]
    """
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[2], wires=wires[0])
    qml.RY(params[3], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(params[4], wires=wires[0])
    qml.RY(params[5], wires=wires[1])


# ---------------------------------------------------------------------------
# Pooling unitary  (3 params)
# ---------------------------------------------------------------------------

def pool_unitary(params: np.ndarray, wires: list):
    """
    2-qubit pooling gate — 3 parameters.

    Approximates a classically-controlled rotation (measure qubit 0,
    conditionally rotate qubit 1) using CRZ + CRX to stay differentiable.

    After pooling, qubit wires[0] is "discarded" (no further ops on it).
    The information is transferred into qubit wires[1].

    Reference:
      Hur et al. (2022) Appendix A — https://arxiv.org/abs/2108.00661
    """
    qml.CRZ(params[0], wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])
    qml.CRX(params[1], wires=[wires[0], wires[1]])
    qml.RZ(params[2], wires=wires[0])


# ---------------------------------------------------------------------------
# Full QCNN builder
# ---------------------------------------------------------------------------

def qcnn_circuit(inputs: np.ndarray, weights: np.ndarray, n_qubits: int = 8):
    """
    Full QCNN forward pass as a PennyLane QNode-compatible function.

    Structure for n_qubits=8:
      Encode (8) → Conv1 (8 pairs) → Pool1 (8→4) → Conv2 (4 pairs) → Pool2 (4→2)
      → measure Z on qubit 1 (survivor after pool2)

    Parameters
    ----------
    inputs   : (n_qubits,)   angle-encoded pixel features in [0, π]
    weights  : (total_weights,)  all trainable QCNN parameters (flat)
    n_qubits : int

    Returns
    -------
    Expectation value of PauliZ on the final output qubit (scalar ∈ [-1, 1])
    """
    PARAMS_CONV = 6
    PARAMS_POOL = 3

    # ── Angle embedding ────────────────────────────────────────────────────
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

    # ── Layer 1: Conv (circular) ───────────────────────────────────────────
    w_idx = 0
    for i in range(n_qubits):
        conv_unitary(
            weights[w_idx : w_idx + PARAMS_CONV],
            wires=[i, (i + 1) % n_qubits],
        )
        w_idx += PARAMS_CONV

    # ── Layer 1: Pool (0→1, 2→3, 4→5, 6→7)  survivors: 1,3,5,7 ──────────
    for i in range(0, n_qubits, 2):
        pool_unitary(
            weights[w_idx : w_idx + PARAMS_POOL],
            wires=[i, i + 1],
        )
        w_idx += PARAMS_POOL

    survivors1 = list(range(1, n_qubits, 2))   # [1, 3, 5, 7]

    # ── Layer 2: Conv (circular on survivors) ─────────────────────────────
    for k in range(len(survivors1)):
        q0 = survivors1[k]
        q1 = survivors1[(k + 1) % len(survivors1)]
        conv_unitary(
            weights[w_idx : w_idx + PARAMS_CONV],
            wires=[q0, q1],
        )
        w_idx += PARAMS_CONV

    # ── Layer 2: Pool (1→3, 5→7)  survivors: 3, 7 ────────────────────────
    for k in range(0, len(survivors1), 2):
        pool_unitary(
            weights[w_idx : w_idx + PARAMS_POOL],
            wires=[survivors1[k], survivors1[k + 1]],
        )
        w_idx += PARAMS_POOL

    # ── Measurement on final output qubit (qubit 3 after pool2) ───────────
    return qml.expval(qml.PauliZ(wires=survivors1[1]))   # qubit 3


def total_weights(n_qubits: int = 8) -> int:
    """Total number of trainable QCNN parameters."""
    PARAMS_CONV, PARAMS_POOL = 6, 3
    return (
        n_qubits       * PARAMS_CONV   # conv1
        + (n_qubits // 2) * PARAMS_POOL   # pool1
        + (n_qubits // 2) * PARAMS_CONV   # conv2
        + (n_qubits // 4) * PARAMS_POOL   # pool2
    )


def weight_breakdown(n_qubits: int = 8) -> dict:
    return {
        "conv_layer_1":  n_qubits        * 6,
        "pool_layer_1": (n_qubits // 2)  * 3,
        "conv_layer_2": (n_qubits // 2)  * 6,
        "pool_layer_2": (n_qubits // 4)  * 3,
        "total":        total_weights(n_qubits),
        "input_features": n_qubits,
    }