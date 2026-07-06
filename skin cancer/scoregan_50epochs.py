"""
scoregan_50epochs.py
====================
Generates 100 images per minority class using the 50-epoch Q-DCGAN checkpoints
(results/models/qdcgan_*.pt) and the shared Pix2Pix refiner (results/models/pix2pix.pt),
runs full pipeline (coarse → denoise → Reinhard colour-normalize), then scores
quality (SSIM, PSNR, FID-approx, Wasserstein) against real images.

Output images are saved to  skin cancer/eval_50epochs/{CLASS}/{stage}/
(completely separate from the main dataset dir — nothing in dataset/ is touched).
Metrics are written to  skin cancer/metrics_image_quality_50epochs.csv
"""

import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pennylane as qml
import torch
import torch.nn as nn
from PIL import Image
from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity as ski_ssim
from torch.amp import autocast

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent          # skin cancer/
MODELS_DIR  = SCRIPT_DIR.parent / "results" / "models"
DATASET_ROOT = SCRIPT_DIR / "dataset"
PROCESSED_DIR = DATASET_ROOT / "processed_images"     # real images live here
EVAL_DIR    = SCRIPT_DIR / "eval_50epochs"             # all generated output goes here
METRICS_CSV = SCRIPT_DIR / "metrics_image_quality_50epochs.csv"

MINORITY_CLASSES = ["MEL", "BCC", "AKIEC", "BKL", "DF", "VASC"]
IMG_SIZE  = 64
N_IMAGES  = 100    # images to generate + score per class
NZ        = 100    # noise dimension
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device : {DEVICE}")
print(f"Models : {MODELS_DIR}")
print(f"Output : {EVAL_DIR}")
print()

# ---------------------------------------------------------------------------
# Quantum layer definitions  (must match gan.ipynb exactly)
# ---------------------------------------------------------------------------
N_QUBITS = 4
N_LAYERS = 1

try:
    import pennylane_lightning 
    qdevice    = qml.device("lightning.qubit", wires=N_QUBITS)
    DIFF_METHOD = "adjoint"
    print("Using lightning.qubit")
except ImportError:
    qdevice    = qml.device("default.qubit", wires=N_QUBITS)
    DIFF_METHOD = "backprop"
    print("Falling back to default.qubit")


@qml.qnode(qdevice, interface="torch", diff_method=DIFF_METHOD)
def vqc_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="X")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


VQC_WEIGHT_SHAPES = {"weights": (N_LAYERS, N_QUBITS, 3)}


class HybridQuantumLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_qubits: int = N_QUBITS, n_layers: int = N_LAYERS):
        super().__init__()
        self.n_qubits = n_qubits
        self.pre = nn.Linear(in_features, n_qubits)
        self.qlayer = qml.qnn.TorchLayer(vqc_circuit, VQC_WEIGHT_SHAPES)
        self.post = nn.Linear(n_qubits, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        x = torch.tanh(x) * torch.pi
        original_device = x.device
        with autocast(device_type="cpu", enabled=False):
            x = self.qlayer(x.float().cpu())
        x = x.to(device=original_device)
        x = self.post(x)
        return x


# ---------------------------------------------------------------------------
# Q-DCGAN Generator — V1 architecture (matches the 50-epoch qdcgan_*.pt checkpoints)
#
# Key differences from the later V2:
#   - NO self.quantum inside the Generator; quantum features are pre-computed
#     externally by precompute_quantum_noise() and passed as x directly.
#   - Uses nn.ConvTranspose2d (not Upsample + Conv2d).
#
# Checkpoint key layout (from state_dict):
#   net.0  = ConvTranspose2d(128, 128, 4, 2, 1)  4x4  -> 8x8
#   net.1  = BatchNorm2d(128)
#   net.2  = LeakyReLU  (no params)
#   net.3  = ConvTranspose2d(128, 64, 4, 2, 1)   8x8  -> 16x16
#   net.4  = BatchNorm2d(64)
#   net.5  = LeakyReLU  (no params)
#   net.6  = ConvTranspose2d(64, 32, 4, 2, 1)    16x16-> 32x32
#   net.7  = BatchNorm2d(32)
#   net.8  = LeakyReLU  (no params)
#   net.9  = ConvTranspose2d(32, 3, 4, 2, 1)     32x32-> 64x64
#   net.10 = Tanh  (no params)
# ---------------------------------------------------------------------------
class QDCGANGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = pre-computed quantum features (B, 128*4*4 = 2048)
        # passed in from precompute_quantum_noise() — quantum circuit runs externally.
        x = x.view(-1, 128, 4, 4)
        return self.net(x)


# ---------------------------------------------------------------------------
# U-Net Generator (Pix2Pix refiner)  (matches gan.ipynb Cell 19)
# ---------------------------------------------------------------------------
class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.bottleneck = HybridQuantumLayer(8192, 8192)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.output = nn.Sequential(nn.Conv2d(64, 3, 3, 1, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        b = d4.flatten(1)
        b = self.bottleneck(b)
        b = b.view(-1, 512, 4, 4)

        u1 = self.up1(b)
        u1 = torch.cat([u1, d3], dim=1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d2], dim=1)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d1], dim=1)
        u4 = self.up4(u3)
        return self.output(u4)


# ---------------------------------------------------------------------------
# Helpers: quantum feature pre-computation
# ---------------------------------------------------------------------------
@torch.no_grad()
def precompute_quantum_noise(n_samples: int, nz: int = NZ, batch_size: int = 32) -> torch.Tensor:
    """Run the quantum sub-layer once on random noise → cache features."""
    layer = HybridQuantumLayer(nz, 128 * 4 * 4).to(DEVICE)
    all_feats = []
    for start in range(0, n_samples, batch_size):
        cur = min(batch_size, n_samples - start)
        noise = torch.randn(cur, nz, device=DEVICE)
        feats = layer(noise)
        all_feats.append(feats.cpu())
        if (start // batch_size) % 5 == 0:
            print(f"  quantum cache: {start + cur}/{n_samples}")
    del layer
    return torch.cat(all_feats, dim=0)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def load_qdcgan(class_name: str) -> nn.Module:
    path = MODELS_DIR / f"qdcgan_{class_name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found at {path}")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    gen = QDCGANGenerator().to(DEVICE)   # V1: no nz arg, no self.quantum
    gen.load_state_dict(ckpt["generator"])
    gen.eval()
    epoch = ckpt.get("epoch", "?")
    print(f"  Loaded Q-DCGAN [{class_name}] from {path.name}  (epoch {epoch})")
    return gen


def load_pix2pix() -> nn.Module:
    path = MODELS_DIR / "pix2pix.pt"
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found at {path}")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    gen = UNetGenerator().to(DEVICE)
    gen.load_state_dict(ckpt["generator"])
    gen.eval()
    epoch = ckpt.get("epoch", "?")
    print(f"  Loaded Pix2Pix from {path.name}  (epoch {epoch})")
    return gen


# ---------------------------------------------------------------------------
# Image generation helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_coarse(gen: nn.Module, class_name: str, n_images: int) -> Path:
    """Generate coarse Q-DCGAN images, save to eval_50epochs/{class_name}/augmented/."""
    out_dir = EVAL_DIR / class_name / "augmented"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [{class_name}] Pre-computing quantum features for {n_images} images…")
    q_feats = precompute_quantum_noise(n_images)

    gen.eval()
    saved = 0
    batch_size = 32
    while saved < n_images:
        cur = min(batch_size, n_images - saved)
        imgs = gen(q_feats[saved:saved + cur].to(DEVICE)).cpu()
        imgs = ((imgs + 1) / 2).clamp(0, 1)
        for i in range(cur):
            arr = (imgs[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(out_dir / f"aug_{saved + i}.jpg")
        saved += cur

    print(f"  [{class_name}] Saved {saved} coarse images → {out_dir}")
    return out_dir


@torch.no_grad()
def denoise_images(pix2pix_gen: nn.Module, class_name: str) -> Path:
    """Run coarse images through the Pix2Pix U-Net, save to .../denoised/."""
    import torchvision.transforms as T

    aug_dir = EVAL_DIR / class_name / "augmented"
    out_dir = EVAL_DIR / class_name / "denoised"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(aug_dir.glob("*.jpg"))
    if not paths:
        print(f"  [{class_name}] No augmented images to denoise.")
        return out_dir

    transform = T.Compose([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    pix2pix_gen.eval()

    batch_size = 16
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        batch = torch.stack(
            [transform(Image.open(p).convert("RGB")) for p in batch_paths]
        ).to(DEVICE)
        refined = pix2pix_gen(batch).cpu()
        refined = ((refined + 1) / 2).clamp(0, 1)
        for j, p in enumerate(batch_paths):
            arr = (refined[j].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(out_dir / p.name)

    print(f"  [{class_name}] Saved {len(paths)} denoised images → {out_dir}")
    return out_dir


def reinhard_color_transfer(source_rgb: np.ndarray, target_rgb: np.ndarray) -> np.ndarray:
    src_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    src_mean, src_std = src_lab.mean(axis=(0, 1)), src_lab.std(axis=(0, 1))
    tgt_mean, tgt_std = tgt_lab.mean(axis=(0, 1)), tgt_lab.std(axis=(0, 1))
    src_std = np.where(src_std == 0, 1e-6, src_std)
    result_lab = (src_lab - src_mean) * (tgt_std / src_std) + tgt_mean
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)


def normalize_class_colors(class_name: str) -> Path:
    """Reinhard color-transfer each denoised image using a random real image as target."""
    denoised_dir = EVAL_DIR / class_name / "denoised"
    real_dir     = PROCESSED_DIR / class_name / "real"   # read-only — not modified
    out_dir      = EVAL_DIR / class_name / "normalized"
    out_dir.mkdir(parents=True, exist_ok=True)

    denoised_paths = sorted(denoised_dir.glob("*.jpg"))
    real_paths     = sorted(real_dir.glob("*.jpg"))

    if not denoised_paths or not real_paths:
        print(f"  [{class_name}] Missing denoised or real images — skipping colour normalisation.")
        return out_dir

    for p in denoised_paths:
        src = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        tgt = cv2.cvtColor(cv2.imread(str(random.choice(real_paths))), cv2.COLOR_BGR2RGB)
        normalized = reinhard_color_transfer(src, tgt)
        cv2.imwrite(str(out_dir / p.name), cv2.cvtColor(normalized, cv2.COLOR_RGB2BGR))

    print(f"  [{class_name}] Saved {len(denoised_paths)} colour-normalised images → {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# Scoring helpers  (identical logic to score_gan.py)
# ---------------------------------------------------------------------------
def load_images_as_array(paths, size=IMG_SIZE):
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((size, size))
        imgs.append(np.asarray(img, dtype=np.float32) / 255.0)
    return np.stack(imgs) if imgs else np.zeros((0, size, size, 3), dtype=np.float32)


def ssim_score(a, b):
    return sum(ski_ssim(a[:, :, c], b[:, :, c], data_range=1.0) for c in range(3)) / 3


def psnr_score(a, b):
    mse = np.mean((a - b) ** 2)
    return 10 * math.log10(1.0 / mse) if mse > 1e-10 else 100.0


def fid_approx(fr, fg):
    fr, fg = fr.astype(np.float64), fg.astype(np.float64)
    mu_r, mu_g = fr.mean(0), fg.mean(0)
    t0 = time.time()
    sig_r = np.cov(fr, rowvar=False) + np.eye(fr.shape[1]) * 1e-6
    sig_g = np.cov(fg, rowvar=False) + np.eye(fg.shape[1]) * 1e-6
    print(f"    [fid_approx] covariance done, starting sqrtm on {sig_r.shape}…")
    sp = sqrtm(sig_r @ sig_g)
    print(f"    [fid_approx] sqrtm done in {time.time() - t0:.1f}s")
    if np.iscomplexobj(sp):
        sp = sp.real
    return float((mu_r - mu_g) @ (mu_r - mu_g) + np.trace(sig_r + sig_g - 2 * sp))


def wass(fr, fg):
    n = min(fr.shape[0], fg.shape[0])
    return float(np.mean([wasserstein_distance(fr[:n, d], fg[:n, d]) for d in range(fr.shape[1])]))


def evaluate_vs_real(class_name: str, stage_dir: Path, real_paths, n_eval=N_IMAGES):
    stage_paths = sorted(stage_dir.glob("*.jpg"))
    n = min(n_eval, len(real_paths), len(stage_paths))
    print(f"    scoring {n} pairs for {class_name}/{stage_dir.name}")
    if n < 2:
        print("    → skipped (< 2 samples)")
        return None

    real_imgs  = load_images_as_array(real_paths[:n])
    stage_imgs = load_images_as_array(stage_paths[:n])

    ss, ps = [], []
    for i in range(n):
        ss.append(ssim_score(real_imgs[i], stage_imgs[i]))
        ps.append(psnr_score(real_imgs[i], stage_imgs[i]))

    fr = real_imgs.reshape(n, -1)
    fg = stage_imgs.reshape(n, -1)

    return {
        "class":       class_name,
        "stage":       stage_dir.name,
        "n_samples":   n,
        "ssim":        round(float(np.mean(ss)), 4),
        "psnr":        round(float(np.mean(ps)), 2),
        "fid_approx":  round(fid_approx(fr, fg), 2),
        "wasserstein": round(wass(fr, fg), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0_total = time.time()

    # -----------------------------------------------------------------------
    # Phase 1 — Generate images for ALL classes first (fast GPU work)
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("  PHASE 1: Image Generation (all classes)")
    print("=" * 60)

    pix2pix_gen = load_pix2pix()
    print()

    # Track which classes succeeded so Phase 2 only scores those
    generated_classes = {}   # cls -> {"aug": Path, "denoised": Path, "norm": Path, "real": [Path]}

    for cls in MINORITY_CLASSES:
        print(f"--- [{cls}] Generating ---")

        real_dir   = PROCESSED_DIR / cls / "real"
        real_paths = sorted(real_dir.glob("*.jpg"))
        if not real_paths:
            print(f"  [{cls}] No real images found — skipping.\n")
            continue

        # Skip Stage 1 if augmented images already exist (resume support)
        aug_dir = EVAL_DIR / cls / "augmented"
        existing = list(aug_dir.glob("*.jpg")) if aug_dir.exists() else []
        if len(existing) >= N_IMAGES:
            print(f"  [{cls}] Stage 1 already done ({len(existing)} coarse images found) — skipping Q-DCGAN.")
        else:
            try:
                gen = load_qdcgan(cls)
            except FileNotFoundError as e:
                print(f"  ERROR: {e} — skipping class.\n")
                continue
            t1 = time.time()
            aug_dir = generate_coarse(gen, cls, N_IMAGES)
            del gen
            print(f"  Stage 1 done in {time.time() - t1:.1f}s")

        # Stage 2: Pix2Pix denoising
        denoised_dir = EVAL_DIR / cls / "denoised"
        existing_d = list(denoised_dir.glob("*.jpg")) if denoised_dir.exists() else []
        if len(existing_d) >= N_IMAGES:
            print(f"  [{cls}] Stage 2 already done ({len(existing_d)} denoised images found) — skipping.")
        else:
            t2 = time.time()
            denoised_dir = denoise_images(pix2pix_gen, cls)
            print(f"  Stage 2 done in {time.time() - t2:.1f}s")

        # Stage 3: Reinhard colour normalisation
        norm_dir = EVAL_DIR / cls / "normalized"
        existing_n = list(norm_dir.glob("*.jpg")) if norm_dir.exists() else []
        if len(existing_n) >= N_IMAGES:
            print(f"  [{cls}] Stage 3 already done ({len(existing_n)} normalised images found) — skipping.")
        else:
            t3 = time.time()
            norm_dir = normalize_class_colors(cls)
            print(f"  Stage 3 done in {time.time() - t3:.1f}s")

        generated_classes[cls] = {
            "aug":      EVAL_DIR / cls / "augmented",
            "denoised": EVAL_DIR / cls / "denoised",
            "norm":     EVAL_DIR / cls / "normalized",
            "real":     real_paths,
        }
        print(f"  [{cls}] Generation complete.\n")

    # Free the Pix2Pix model from GPU memory before scoring
    del pix2pix_gen
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    t_gen_done = time.time()
    print(f"\n{'='*60}")
    print(f"  PHASE 1 done in {t_gen_done - t0_total:.1f}s")
    print(f"  Generated classes: {list(generated_classes.keys())}")
    print(f"{'='*60}\n")

    # -----------------------------------------------------------------------
    # Phase 2 — Score ALL classes (slow FID / Wasserstein computation)
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("  PHASE 2: Scoring (all classes)")
    print(f"  Warning: FID-approx on 12288-dim features is slow.")
    print(f"  Estimated ~{len(generated_classes) * 3 * 6:.0f} min total (3 stages × 6 classes).")
    print("=" * 60)
    print()

    results = []

    for cls, dirs in generated_classes.items():
        print(f"--- [{cls}] Scoring ---")
        real_paths = dirs["real"]

        for stage_key, stage_dir in [("normalized", dirs["norm"])]:
            t_s = time.time()
            row = evaluate_vs_real(cls, stage_dir, real_paths, n_eval=N_IMAGES)
            if row:
                results.append(row)
                print(f"    [{cls}/{stage_key}] {row}  ({time.time()-t_s:.1f}s)")

            # Save partial results after every stage so progress isn't lost
            # if the script crashes midway through scoring.
            if results:
                pd.DataFrame(results).to_csv(METRICS_CSV, index=False)

        print()

    # --- Final save & report ---
    df = pd.DataFrame(results)
    df.to_csv(METRICS_CSV, index=False)
    print(f"\n{'='*60}")
    print(f"All done in {time.time() - t0_total:.1f}s")
    print(f"Metrics saved → {METRICS_CSV}")
    print()
    if not df.empty:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
