# ── Data ──────────────────────────────────────────────────────────────────
CLASS_A      = 0        # digit treated as class 0
CLASS_B      = 1        # digit treated as class 1
N_TRAIN      = 200      # increase to 500 for better results (slower)
N_TEST       = 50
IMAGE_SIZE   = 8        # resize MNIST to IMAGE_SIZE × IMAGE_SIZE
N_QUBITS     = 8        # must match IMAGE_SIZE for column-avg encoding
RANDOM_SEED  = 42
 
# ── QCNN (PennyLane) ─────────────────────────────────────────────────────
DIFF_METHOD  = "backprop"   # "backprop" is faster on simulator
QCNN_LR      = 0.01
QCNN_EPOCHS  = 30
QCNN_BATCH   = 16
INIT_STD     = 0.1    # near-zero init to mitigate barren plateaus
 
# ── Classical CNN (PyTorch) ───────────────────────────────────────────────
CNN_LR       = 0.01
CNN_EPOCHS   = 30
CNN_BATCH    = 16
 
# ── Output paths (relative to project root) ───────────────────────────────
LOG_DIR     = "results/logs"
FIGURES_DIR = "results/figures"
MODELS_DIR  = "results/models"