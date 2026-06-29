import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import time
from skimage.metrics import structural_similarity as ski_ssim
from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance
# ----------------------------------------------------------------------------
# Image Quality Evaluation — SSIM, PSNR, FID (pixel-space approx), Wasserstein
# Compares each pipeline stage (Coarse / Denoised / Normalized) against real
# images, per class.
#
# NOTE on fid_approx: this is a lightweight FID computed directly on flattened
# RAW PIXELS, not Inception-V3 embeddings like standard FID. It's useful as a
# fast, in-notebook relative signal (lower = closer to real) but is NOT
# comparable to FID numbers reported in papers, which require a pretrained
# feature extractor. Treat it as a rough proxy, not a publishable metric.
# ----------------------------------------------------------------------------

PROJECT_ROOT= Path().cwd()
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
DATASET_ROOT = PROJECT_ROOT / "dataset"
PROCESSED_DIR = DATASET_ROOT / "processed_images"

minority_classes = ["MEL", "BCC", "AKIEC", "BKL", "DF", "VASC"]

IMG_SIZE=64
N_EVAL = 100  # images per class per stage; capped to availability below


def load_images_as_array(paths, size=IMG_SIZE):
    """Loads a list of image paths as a (N, H, W, 3) float array in [0, 1]."""
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((size, size))
        imgs.append(np.asarray(img, dtype=np.float32) / 255.0)
    return np.stack(imgs) if imgs else np.zeros((0, size, size, 3), dtype=np.float32)


def ssim_score(a, b):
    s = 0
    for c in range(3):
        s += ski_ssim(a[:, :, c], b[:, :, c], data_range=1.0)
    return s / 3


def psnr_score(a, b):
    mse = np.mean((a - b) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0


def fid_approx(fr, fg):
    """Lightweight FID computed on raw flattened pixels (see note above)."""
    print(f"        [fid_approx] starting | feature dim = {fr.shape[1]} "
          f"(this is the likely bottleneck — covariance is {fr.shape[1]}x{fr.shape[1]})")
    t0 = time.time()

    fr = fr.astype(np.float64)
    fg = fg.astype(np.float64)
    mu_r, mu_g = fr.mean(0), fg.mean(0)
    print(f"        [fid_approx] means computed ({time.time() - t0:.2f}s elapsed)")

    t1 = time.time()
    sig_r = np.cov(fr, rowvar=False) + np.eye(fr.shape[1]) * 1e-6
    sig_g = np.cov(fg, rowvar=False) + np.eye(fg.shape[1]) * 1e-6
    print(f"        [fid_approx] covariance matrices computed in {time.time() - t1:.2f}s")

    diff = mu_r - mu_g

    t2 = time.time()
    print(f"        [fid_approx] starting sqrtm on {sig_r.shape} matrix — this is usually the slow part...")
    sp = sqrtm(sig_r @ sig_g)
    print(f"        [fid_approx] sqrtm finished in {time.time() - t2:.2f}s")

    if np.iscomplexobj(sp):
        sp = sp.real
    result = float(diff @ diff + np.trace(sig_r + sig_g - 2 * sp))
    print(f"        [fid_approx] done | total {time.time() - t0:.2f}s | value = {result:.2f}")
    return result


def wass(fr, fg):
    n = min(fr.shape[0], fg.shape[0])
    t0 = time.time()
    val = float(np.mean([wasserstein_distance(fr[:n, d], fg[:n, d]) for d in range(fr.shape[1])]))
    print(f"        [wasserstein] done in {time.time() - t0:.2f}s | value = {val:.4f}")
    return val


def evaluate_stage_vs_real(class_name, stage_dir_name, real_paths, n_eval=N_EVAL):
    """Computes SSIM/PSNR/FID-approx/Wasserstein for one (class, stage) pair vs real images."""
    print(f"    -> evaluating stage '{stage_dir_name}' for class '{class_name}'")
    stage_dir = PROCESSED_DIR / class_name / stage_dir_name
    stage_paths = sorted(stage_dir.glob("*.jpg")) if stage_dir.exists() else []
    print(f"       found {len(stage_paths)} images in {stage_dir}")

    n_real = min(n_eval, len(real_paths))
    n_stage = min(n_eval, len(stage_paths))
    n = min(n_real, n_stage)
    print(f"       using n={n} paired samples (n_real={n_real}, n_stage={n_stage})")

    if n < 2:
        print(f"       skipping — not enough samples (need >= 2, got {n})")
        return None  # not enough samples to compute meaningful metrics

    t_load = time.time()
    real_imgs = load_images_as_array(real_paths[:n])
    stage_imgs = load_images_as_array(stage_paths[:n])
    print(f"       loaded {n} real + {n} stage images in {time.time() - t_load:.2f}s")

    t_ssim_psnr = time.time()
    ss, ps = [], []
    for i in range(n):
        ss.append(ssim_score(real_imgs[i], stage_imgs[i]))
        ps.append(psnr_score(real_imgs[i], stage_imgs[i]))
        if (i + 1) % 20 == 0 or (i + 1) == n:
            print(f"       SSIM/PSNR progress: {i + 1}/{n}")
    print(f"       SSIM/PSNR done for all {n} pairs in {time.time() - t_ssim_psnr:.2f}s "
          f"(avg ssim={np.mean(ss):.4f}, avg psnr={np.mean(ps):.2f})")

    fr = real_imgs.reshape(n, -1)
    fg = stage_imgs.reshape(n, -1)

    fid_val = fid_approx(fr, fg)
    wass_val = wass(fr, fg)

    print(f"    <- finished stage '{stage_dir_name}' for class '{class_name}'\n")

    return {
        "class": class_name,
        "stage": stage_dir_name,
        "n_samples": n,
        "ssim": round(float(np.mean(ss)), 4),
        "psnr": round(float(np.mean(ps)), 2),
        "fid_approx": round(fid_val, 2),
        "wasserstein": round(wass_val, 4),
    }


# ----------------------------------------------------------------------------
# Run evaluation across all minority classes and all 3 pipeline stages
# ----------------------------------------------------------------------------
stages_to_eval = ["normalized"]  # Coarse, Denoised, Normalized
results = []

print(f"Starting evaluation | classes={minority_classes} | stages={stages_to_eval} | N_EVAL={N_EVAL}")
print(f"PROCESSED_DIR = {PROCESSED_DIR}\n")

t_start_all = time.time()

for cls_idx, cls in enumerate(minority_classes, start=1):
    print(f"=== [{cls_idx}/{len(minority_classes)}] Class: {cls} ===")
    t_class = time.time()

    real_dir = PROCESSED_DIR / cls / "real"
    real_paths = sorted(real_dir.glob("*.jpg")) if real_dir.exists() else []
    print(f"  real_dir = {real_dir} | found {len(real_paths)} real images")

    if not real_paths:
        print(f"[{cls}] No real images found — skipping evaluation.\n")
        continue

    for stage in stages_to_eval:
        t_stage = time.time()
        result = evaluate_stage_vs_real(cls, stage, real_paths, n_eval=N_EVAL)
        if result is not None:
            results.append(result)
            print(f"  [{cls}/{stage}] complete in {time.time() - t_stage:.2f}s -> {result}")
        else:
            print(f"  [{cls}] Skipping '{stage}' — not enough samples to evaluate "
                  f"({time.time() - t_stage:.2f}s)")

    print(f"=== Class {cls} done in {time.time() - t_class:.2f}s ===\n")

print(f"All classes processed in {time.time() - t_start_all:.2f}s total")

metrics_df = pd.DataFrame(results)
metrics_csv_path = PROJECT_ROOT / "metrics_image_quality.csv"
metrics_df.to_csv(metrics_csv_path, index=False)

print(f"\nSaved metrics to {metrics_csv_path}")
print("\nImage Quality Metrics (per class, per stage):")
print(metrics_df.to_string(index=False))