# Quantum Deep Learning & Machine Learning Research Project

This project is a research-oriented exploration of Variational Quantum Circuits (VQCs) and Parameterized Quantum Circuits (PQCs) within Quantum Machine Learning (QML) workflows. The primary objective is to investigate optimization dynamics, training convergence characteristics, and practical feasibility on both local simulators and real quantum hardware.

---

## 🔬 Research Focus & Objectives

- **Variational Quantum Circuits (VQCs):** Designing and implementing trainable quantum circuit architectures using various parameterized ansatz structures (e.g., `RealAmplitudes`, `EfficientSU2`) and feature maps (e.g., `ZZFeatureMap`).
- **Optimization Dynamics:** Analyzing how quantum circuit parameters evolve during training under classical optimizers (e.g., COBYLA, SPSA, Adam).
- **Hybrid Quantum-Classical Workflows:** Constructing and evaluating training loops that combine quantum circuit execution with classical optimization updates.
- **Hardware Validation:** Transitioning models from noiseless/noisy classical simulators (Qiskit Aer) to execution on real IBM Quantum devices using the Qiskit Runtime API.

---

## 📈 Current Progress

### 1. Environment & Infrastructure
- Successfully configured Qiskit development environment with dependencies managed via `uv` / `pyproject.toml`.
- Integrated IBM Quantum account with the Qiskit Runtime API (`qiskit-ibm-runtime`).
- Verified access and selection mechanisms for various IBM QPU backends.

### 2. Simulator Execution
- Built and validated end-to-end QML execution pipelines locally using noiseless simulators (`StatevectorEstimator`, `qiskit-aer`).
- Confirmed parameter binding, circuit composition, and result expectation value collection.

### 3. Optimizer Study
- Implemented core exploratory pipelines comparing stochastic quantum optimization behavior and training stability.

---

## 🛠️ Technology Stack

- **Languages:** Python (>= 3.13)
- **Quantum Computing Framework:** Qiskit (>= 2.4.1)
- **Simulators:** Qiskit Aer (>= 0.17.2)
- **Runtime Services:** Qiskit IBM Runtime (>= 0.47.0)
- **Machine Learning Integration:** Qiskit Machine Learning (>= 0.9.0), Scikit-Learn
- **Scientific Computing & Visualization:** NumPy, Pandas, Matplotlib, PyLatexEnc

---

## 📁 Directory Structure

The repository is structured to maintain a clean boundary between exploratory notebooks, modularized library code, experimental results, and scientific notes:

```text
paper_learning/
├── src/                               # Reusable library code (Python package)
│   ├── circuits/                      # Circuit builders (ansatz, feature maps)
│   │   ├── ansatz.py                  
│   │   └── feature_maps.py            
│   ├── models/                        # Quantum Neural Network & Classifier models
│   │   ├── vqc_classifier.py          
│   │   └── qnn.py                     
│   ├── training/                      # Custom training loops and optimizers
│   │   ├── trainer.py                 
│   │   └── callbacks.py               
│   ├── backends/                      # Device & simulator backend management
│   │   ├── ibm_runtime.py             
│   │   └── simulator.py               
│   └── utils/                         # Shared plotting and evaluation utilities
│       ├── visualization.py           
│       ├── metrics.py                 
│       └── io.py                      
│
├── experiments/                       # Chronological research notebooks
│   ├── 01_vqc_basics/                 # Basic circuit design & parameter binding
│   ├── 02_pqc_training/               # Optimization curves and training runs
│   ├── 03_optimization_study/         # Optimizer benchmarks (COBYLA vs SPSA vs Adam)
│   ├── 04_hybrid_training/            # Integrated hybrid pipelines
│   └── 05_ibm_hardware/               # Real QPU deployment trials & noise analysis
│
├── data/                              # Datasets used for model benchmarking
│   ├── raw/                           
│   └── processed/                     
│
├── results/                           # Generated outputs (never tracked/edited manually)
│   ├── models/                        # Serialized weights (*.model)
│   ├── logs/                          # Trajectory CSVs and training metrics
│   └── figures/                       # High-res vector plots for the paper
│
├── papers/                            # PDF preprints and reference literature
├── scripts/                           # Standalone batch execution & evaluation scripts
└── notes/                             # Scientific logs, ideas, and hypotheses
    ├── experiment_log.md              
    ├── ideas.md                       
    └── observations.md                
```

---

## 🚀 Reorganization Guidelines

To transition current exploratory scripts into this clean structure:
1. **Source Code Extraction:** Extract helper logic from Jupyter notebooks into relevant files under `src/` (e.g., customized ansatz wrappers go to `src/circuits/ansatz.py`).
2. **Result Isolation:** Save model weights (like `.model` files) under `results/models/` and performance logs under `results/logs/` instead of cluttering workspace directories.
3. **Keep Notebooks Topic-Specific:** Organise future notebook sessions chronologically within numbered folders in `experiments/` to track research progression clearly.
