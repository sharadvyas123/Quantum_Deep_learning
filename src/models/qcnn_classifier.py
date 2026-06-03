# quantum neural network model
"""
src/models/qcnn_classifier.py
------------------------------
PennyLane QNode + PyTorch/NumPy hybrid training wrapper.

The QNode is constructed with the 'default.qubit' simulator backend.
For hardware runs, swap to 'qiskit.ibmq' or 'braket.aws.qubit'.

Key PennyLane concepts used here:
  - qml.QNode          — wraps a quantum circuit into a differentiable function
  - qml.device         — selects simulator / hardware backend
  - qml.qnn.TorchLayer — makes a QNode a torch.nn.Module (auto-differentiable)
  - Parameter-shift gradients — default for qubit devices; hardware-compatible

References:
  PennyLane QNN docs:
    https://docs.pennylane.ai/en/stable/introduction/interfaces/torch.html

  Bergholm et al. (2018) "PennyLane: Automatic differentiation of hybrid
  quantum-classical computations"  https://arxiv.org/abs/1811.04968

  Schuld et al. (2019) "Evaluating analytic gradients on quantum hardware"
  https://arxiv.org/abs/1811.11184   (parameter-shift rule)

  Grant et al. (2019) "An initialization strategy for addressing barren plateaus"
  https://arxiv.org/abs/1903.05076   (near-zero init strategy used here)
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from src.circuits.qcnn_layer import qcnn_circuit, total_weights, weight_breakdown


# ---------------------------------------------------------------------------
# Build QNode
# ---------------------------------------------------------------------------

def make_qnode(n_qubits: int = 8, diff_method: str = "parameter-shift") -> qml.QNode:
    """
    Create a PennyLane QNode for the QCNN circuit.

    Parameters
    ----------
    n_qubits     : int   Number of qubits (= number of input features)
    diff_method  : str   Gradient method. Options:
                           'parameter-shift'  — hardware-compatible, exact
                           'backprop'         — faster on simulator (default.qubit only)
                           'adjoint'          — fastest on default.qubit simulator

    Returns
    -------
    qml.QNode
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        return qcnn_circuit(inputs, weights, n_qubits=n_qubits)

    return circuit


# ---------------------------------------------------------------------------
# PyTorch hybrid model
# ---------------------------------------------------------------------------

class QCNNModel(nn.Module):
    """
    Hybrid quantum-classical model:
      QCNN QNode → scalar ∈ [-1,1] → linear rescale → sigmoid → P(class=1)

    The QNode is wrapped as a torch.nn.Module via qml.qnn.TorchLayer,
    which makes the quantum weights proper nn.Parameters tracked by autograd.

    Usage
    -----
    model = QCNNModel(n_qubits=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # training step
    pred = model(X_batch)           # (batch, 1)
    loss = criterion(pred, y_batch)
    loss.backward()
    optimizer.step()
    """

    def __init__(
        self,
        n_qubits:    int = 8,
        diff_method: str = "parameter-shift",
        init_std:    float = 0.1,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        n_w = total_weights(n_qubits)

        qnode = make_qnode(n_qubits, diff_method)

        # qml.qnn.TorchLayer registers 'weights' as nn.Parameter automatically
        # Near-zero init (init_std ~ 0.1) mitigates barren plateaus
        # Ref: Grant et al. 2019 — https://arxiv.org/abs/1903.05076
        weight_shapes = {"weights": (n_w,)}
        self.qlayer = qml.qnn.TorchLayer(
            qnode,
            weight_shapes,
            init_method=lambda t: nn.init.normal_(t, mean=0.0, std=init_std),
        )

        # Output ∈ [-1, 1] → shift/scale → sigmoid → probability
        self.post = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_qubits)
        out = torch.stack([self.qlayer(xi) for xi in x])  # (batch,)
        out = out.unsqueeze(-1)                            # (batch, 1)
        return self.post(out)                              # (batch, 1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            probs = self.forward(torch.tensor(X, dtype=torch.float32))
        return (probs.squeeze() > 0.5).int().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            return self.forward(torch.tensor(X, dtype=torch.float32)).squeeze().numpy()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float((preds == y).mean())

    def parameter_summary(self) -> dict:
        bd = weight_breakdown(self.n_qubits)
        bd["torch_params_total"] = sum(p.numel() for p in self.parameters())
        return bd

    def draw(self):
        """Print the underlying quantum circuit."""
        dev = qml.device("default.qubit", wires=self.n_qubits)
        dummy_inputs  = np.zeros(self.n_qubits)
        dummy_weights = np.zeros(total_weights(self.n_qubits))

        @qml.qnode(dev)
        def _draw():
            return qcnn_circuit(dummy_inputs, dummy_weights, n_qubits=self.n_qubits)

        print(qml.draw(_draw)())