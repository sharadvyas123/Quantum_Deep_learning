"""
scoreganv2.py
=============
Unified image-quality scorer for the Quantum-DCGAN skin-cancer pipeline.

Metrics computed (real vs generated images, per minority class, "normalized" stage):
  1  SSIM          -- Structural Similarity Index (grayscale luminance, skimage)
  2  PSNR          -- Peak Signal-to-Noise Ratio  (uint8 / 255-scale, dB)
  3  MS-SSIM       -- Multi-Scale SSIM (3 scales, betas tuned for 64x64 images)
  4  LPIPS         -- Learned Perceptual Image Patch Similarity (AlexNet, [-1,1])
  5  sFID          -- Spatial FID via InceptionV3 Mixed_7c spatial features
  6  Wasserstein   -- Mean 1-D Wasserstein distance on flattened pixel vectors

Mathematical formulations
-------------------------
SSIM(x, y) = [2*mu_x*mu_y + C1][2*sigma_xy + C2] / [(mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2)]
    applied on luminance channel, averaged over the image window.

PSNR(x, y) = 20 * log10(MAX_I / sqrt(MSE(x, y)))   where MAX_I = 255
    (from psnr.ipynb, applied on uint8 images)

MS-SSIM(x, y) = [lM(x,y)]^alphaM * prod_j=1^M [cj(x,y)]^betaj [sj(x,y)]^gammaj
    with 3 scales (betas: 0.0448, 0.2856, 0.3001) to support 64x64 inputs.
    (from msssin.ipynb)

LPIPS(x, y) = sum_l w_l * ||phi_l(x) - phi_l(y)||_2^2  (AlexNet features, normalised)
    input normalised to [-1, 1].
    (from lpips.ipynb / msssin.ipynb)

sFID: standard Frechet Inception Distance using InceptionV3 Mixed_7c
    spatial features (pooled to 2048-d vector) instead of the final logit layer.
    FID(N(mu1,Sigma1), N(mu2,Sigma2)) = ||mu1-mu2||^2 + Tr(Sigma1+Sigma2 - 2*sqrt(Sigma1*Sigma2))
    (from sfid2.ipynb)

Wasserstein: mean over feature dimensions of 1-D Earth-Mover distances.
    (from score_gan.py)

Directory layout expected
-------------------------
  <script_dir>/dataset/processed_images/<CLASS>/<stage>/<image>.jpg
      where <CLASS> in minority_classes and <stage> in {"real", "normalized", ...}

Output
------
  <script_dir>/metrics_image_quality_v2.csv

Requirements
------------
  pip install torch torchvision lpips torchmetrics scikit-image scipy pillow tqdm numpy pandas
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import time
import warnings
from math import log10
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from scipy import linalg
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity as ski_ssim
from tqdm import tqdm


from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
import lpips as lpips_lib
# Suppress noisy deprecation warnings from older torchvision/lpips
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths  (all relative to THIS script file -- works wherever it is placed)
# ---------------------------------------------------------------------------
SCRIPT_DIR    = Path(__file__).resolve().parent          # skin cancer/
DATASET_ROOT  = SCRIPT_DIR / "dataset"
PROCESSED_DIR = DATASET_ROOT / "processed_images"       # real & generated images
METRICS_CSV   = SCRIPT_DIR / "metrics_image_quality_v2.csv"

MINORITY_CLASSES = ["MEL", "BCC", "AKIEC", "BKL", "DF", "VASC"]
STAGES_TO_EVAL   = ["normalized" , "denoised"]                       # extend as needed
IMG_SIZE         = 64      # resize target for all images (GAN output size)
N_EVAL           = 100     # max images per (class, stage) pair
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device        : {DEVICE}")
print(f"Processed dir : {PROCESSED_DIR}")
print(f"Output CSV    : {METRICS_CSV}")
print()

# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def load_images_as_array(paths, size: int = IMG_SIZE):
    """Return (N, H, W, 3) float32 array in [0, 1]."""
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
        imgs.append(np.asarray(img, dtype=np.float32) / 255.0)
    if not imgs:
        return np.zeros((0, size, size, 3), dtype=np.float32)
    return np.stack(imgs)


def load_images_as_uint8(paths, size: int = IMG_SIZE):
    """Return (N, H, W, 3) uint8 array in [0, 255] (for PSNR)."""
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
        imgs.append(np.asarray(img, dtype=np.uint8))
    if not imgs:
        return np.zeros((0, size, size, 3), dtype=np.uint8)
    return np.stack(imgs)


# ---------------------------------------------------------------------------
# Metric 1: SSIM  (from ssim.ipynb -- grayscale luminance)
# ---------------------------------------------------------------------------

def ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    SSIM on grayscale, exactly matching ssim.ipynb:
      img1 = cv2.imread(...)           -> BGR uint8
      img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
      score, _ = ssim(img1_gray, img2_gray, full=True)

    a, b: (H, W, 3) float32 in [0, 1]  (loaded via PIL, i.e. RGB order)
    We convert to BGR uint8 to exactly reproduce the cv2.imread output, then
    apply cv2.COLOR_BGR2GRAY -- this gives the same result as if the image
    had been read directly with cv2.imread.
    Returns scalar SSIM in [-1, 1].
    """
    # float32 RGB [0,1] -> uint8 BGR [0,255]  (mirrors cv2.imread channel order)
    a_bgr = (a[:, :, ::-1] * 255).astype(np.uint8)
    b_bgr = (b[:, :, ::-1] * 255).astype(np.uint8)

    a_gray = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY)

    # data_range=255 because cv2 gives uint8
    score, _ = ski_ssim(a_gray, b_gray, data_range=255, full=True)
    return float(score)


# ---------------------------------------------------------------------------
# Metric 2: PSNR  (from psnr.ipynb -- uint8 / 255-scale formula)
# ---------------------------------------------------------------------------

def psnr_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    PSNR = 20 * log10(255 / sqrt(MSE))  on uint8 images.

    a, b: (H, W, 3) uint8
    Returns PSNR in dB (100 dB sentinel for identical images).
    """
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0.0:
        return 100.0
    return 20.0 * log10(255.0 / np.sqrt(mse))


# ---------------------------------------------------------------------------
# Metrics 3 + 4: MS-SSIM and LPIPS  (from msssin.ipynb / lpips.ipynb)
# ---------------------------------------------------------------------------

def _build_ms_ssim_and_lpips(device):
    """
    Initialise MS-SSIM (3 scales) and LPIPS (AlexNet) metrics.

    MS-SSIM uses 3 scales with betas (0.0448, 0.2856, 0.3001) so it works
    on 64x64 images.  The standard 5-scale implementation requires ~160px+.
    LPIPS input must be in [-1, 1].
    """
    

    ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(
        data_range=1.0,
        betas=(0.0448, 0.2856, 0.3001),   # 3 scales -- required for 64x64
    ).to(device)

    lpips_metric = lpips_lib.LPIPS(net="alex").to(device).eval()
    return ms_ssim_metric, lpips_metric


_to_01 = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),          # -> [0, 1]
])


def _to_m11(t: torch.Tensor) -> torch.Tensor:
    """Rescale [0,1] -> [-1,1] for LPIPS."""
    return t * 2.0 - 1.0


def ms_ssim_lpips_scores(real_paths, fake_paths, ms_ssim_metric, lpips_metric,
                          n: int, device):
    """
    Compute mean MS-SSIM and mean LPIPS over n paired (real, fake) images.

    Returns (ms_ssim_mean, lpips_mean) -- NaN if n == 0.
    """
    ms_vals, lp_vals = [], []
    for rp, fp in zip(real_paths[:n], fake_paths[:n]):
        try:
            r_img = Image.open(rp).convert("RGB")
            f_img = Image.open(fp).convert("RGB")
        except Exception:
            continue

        r_t = _to_01(r_img).unsqueeze(0).to(device).clamp(0, 1)
        f_t = _to_01(f_img).unsqueeze(0).to(device).clamp(0, 1)

        # MS-SSIM (expects [0,1])
        try:
            val = ms_ssim_metric(r_t, f_t).item()
            if not np.isnan(val):
                ms_vals.append(val)
        except Exception:
            pass

        # LPIPS (expects [-1,1])
        try:
            val = lpips_metric(_to_m11(r_t), _to_m11(f_t)).item()
            if not np.isnan(val):
                lp_vals.append(val)
        except Exception:
            pass

    ms_mean = float(np.nanmean(ms_vals)) if ms_vals else float("nan")
    lp_mean = float(np.nanmean(lp_vals)) if lp_vals else float("nan")
    return ms_mean, lp_mean


# ---------------------------------------------------------------------------
# Metric 5: sFID  (from sfid2.ipynb -- InceptionV3 Mixed_7c spatial features)
# ---------------------------------------------------------------------------

def _build_inception(device):
    """
    Load InceptionV3, hooking Mixed_7c for 2048-D spatial feature vectors.

    Exactly mirrors sfid2.ipynb:
      inception.Mixed_7c.register_forward_hook(
          lambda m, i, o: setattr(inception, 'features', o)
      )
    Uses the modern weights API to suppress the 'pretrained' deprecation warning.
    """
    from torchvision.models import inception_v3, Inception_V3_Weights

    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                         transform_input=False)
    # Store hook output as model.features -- same attribute name as sfid2.ipynb
    model.Mixed_7c.register_forward_hook(
        lambda m, i, o: setattr(model, "features", o)
    )
    model.eval().to(device)
    return model


_inception_transform = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def get_inception_features(paths, model, device, desc="") -> np.ndarray:
    """
    Extract 2048-D spatial feature vectors from InceptionV3 Mixed_7c.

    Mirrors sfid2.ipynb: forward pass -> spatial mean-pool (H*W dims) -> 2048-D vec.
    """
    feats = []
    for p in tqdm(paths, desc=desc, leave=False):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        img_t = _inception_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(img_t)
            # model.features: (1, 2048, H, W)  -- set by the hook above
            feat = model.features.squeeze().cpu().numpy()
            # Spatial mean pooling: (2048, h, w) -> (2048,)  [mirrors sfid2.ipynb]
            feat = feat.reshape(feat.shape[0], -1).mean(axis=1)
        feats.append(feat)
    return np.array(feats) if feats else np.zeros((0, 2048))


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    """
    Frechet distance between two Gaussians N(mu1, Sigma1) and N(mu2, Sigma2).

    FID = ||mu1 - mu2||^2 + Tr(Sigma1 + Sigma2 - 2 * sqrt(Sigma1 @ Sigma2))

    Adapted from sfid2.ipynb using scipy linalg.sqrtm.
    """
    mu1    = np.atleast_1d(mu1)
    mu2    = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset  = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(
        diff.dot(diff)
        + np.trace(sigma1) + np.trace(sigma2)
        - 2 * np.trace(covmean)
    )
    return fid


# ---------------------------------------------------------------------------
# Metric 6: Wasserstein  (retained from score_gan.py)
# ---------------------------------------------------------------------------

def wasserstein_score(fr: np.ndarray, fg: np.ndarray) -> float:
    """
    Mean 1-D Wasserstein distance over all pixel/feature dimensions.

    fr, fg: (N, D) float arrays.
    """
    n = min(fr.shape[0], fg.shape[0])
    return float(np.mean([
        wasserstein_distance(fr[:n, d], fg[:n, d])
        for d in range(fr.shape[1])
    ]))


# ---------------------------------------------------------------------------
# Main per-(class, stage) evaluation
# ---------------------------------------------------------------------------

def evaluate_stage_vs_real(class_name: str, stage_dir_name: str,
                            real_paths: list, n_eval: int,
                            ms_ssim_metric, lpips_metric,
                            inception_model):
    """
    Compute all six metrics for one (class, stage) pair vs real images.

    Returns a dict of metrics or None if there are too few samples.
    """
    print(f"    -> evaluating stage='{stage_dir_name}' for class='{class_name}'")
    stage_dir   = PROCESSED_DIR / class_name / stage_dir_name
    stage_paths = sorted(stage_dir.glob("*.jpg")) if stage_dir.exists() else []
    print(f"       stage images found : {len(stage_paths)}  in  {stage_dir}")

    n_real  = min(n_eval, len(real_paths))
    n_stage = min(n_eval, len(stage_paths))
    n       = min(n_real, n_stage)
    print(f"       using n={n} paired samples (n_real={n_real}, n_stage={n_stage})")

    if n < 2:
        print(f"       SKIP -- need >= 2 samples, got {n}")
        return None

    rp = real_paths[:n]
    sp = list(stage_paths[:n])

    # ------------------------------------------------------------------
    # Metrics 1 (SSIM) + 2 (PSNR)
    # ------------------------------------------------------------------
    t0 = time.time()
    real_f32  = load_images_as_array(rp)
    stage_f32 = load_images_as_array(sp)
    real_u8   = load_images_as_uint8(rp)
    stage_u8  = load_images_as_uint8(sp)

    ssim_vals, psnr_vals = [], []
    for i in range(n):
        ssim_vals.append(ssim_score(real_f32[i], stage_f32[i]))
        psnr_vals.append(psnr_score(real_u8[i],  stage_u8[i]))
        if (i + 1) % 20 == 0 or (i + 1) == n:
            print(f"       SSIM/PSNR progress: {i+1}/{n}")

    ssim_mean = float(np.mean(ssim_vals))
    psnr_mean = float(np.mean(psnr_vals))
    print(f"       SSIM={ssim_mean:.4f}  PSNR={psnr_mean:.2f} dB  ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # Metrics 3 (MS-SSIM) + 4 (LPIPS)
    # ------------------------------------------------------------------
    t1 = time.time()
    ms_ssim_mean, lpips_mean = ms_ssim_lpips_scores(
        rp, sp, ms_ssim_metric, lpips_metric, n, DEVICE
    )
    print(f"       MS-SSIM={ms_ssim_mean:.4f}  LPIPS={lpips_mean:.4f}  ({time.time()-t1:.1f}s)")

    # ------------------------------------------------------------------
    # Metric 5 (sFID)
    # ------------------------------------------------------------------
    t2 = time.time()
    real_feats  = get_inception_features(rp, inception_model, DEVICE,
                                         desc=f"       sFID-real [{class_name}]")
    stage_feats = get_inception_features(sp, inception_model, DEVICE,
                                         desc=f"       sFID-fake [{class_name}]")

    if real_feats.shape[0] >= 2 and stage_feats.shape[0] >= 2:
        mu1, sig1 = np.mean(real_feats,  axis=0), np.cov(real_feats,  rowvar=False)
        mu2, sig2 = np.mean(stage_feats, axis=0), np.cov(stage_feats, rowvar=False)
        sfid_val  = calculate_frechet_distance(mu1, sig1, mu2, sig2)
    else:
        sfid_val = float("nan")
    print(f"       sFID={sfid_val:.2f}  ({time.time()-t2:.1f}s)")

    # ------------------------------------------------------------------
    # Metric 6 (Wasserstein -- on flattened pixel vectors)
    # ------------------------------------------------------------------
    t3 = time.time()
    fr = real_f32.reshape(n, -1)
    fg = stage_f32.reshape(n, -1)
    wass_val = wasserstein_score(fr, fg)
    print(f"       Wasserstein={wass_val:.6f}  ({time.time()-t3:.1f}s)")

    print(f"    <- done  '{stage_dir_name}' / '{class_name}'\n")

    return {
        "class":       class_name,
        "stage":       stage_dir_name,
        "n_samples":   n,
        "ssim":        round(ssim_mean,    4),
        "psnr_dB":     round(psnr_mean,    2),
        "ms_ssim":     round(ms_ssim_mean, 4) if not np.isnan(ms_ssim_mean) else None,
        "lpips":       round(lpips_mean,   4) if not np.isnan(lpips_mean)   else None,
        "sfid":        round(sfid_val,     2) if not np.isnan(sfid_val)     else None,
        "wasserstein": round(wass_val,     6),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Check heavy dependencies early
    try:
        import lpips          # noqa: F401
        import torchmetrics   # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"Missing dependency: {exc}\n"
            "Run:  pip install lpips torchmetrics"
        ) from exc

    print("Building MS-SSIM metric and LPIPS model ...")
    ms_ssim_metric, lpips_metric = _build_ms_ssim_and_lpips(DEVICE)

    print("Loading InceptionV3 for sFID ...")
    inception_model = _build_inception(DEVICE)
    print()

    results = []
    t_all = time.time()

    print(f"Starting evaluation")
    print(f"  classes  : {MINORITY_CLASSES}")
    print(f"  stages   : {STAGES_TO_EVAL}")
    print(f"  N_EVAL   : {N_EVAL}")
    print(f"  IMG_SIZE : {IMG_SIZE}")
    print()

    for idx, cls in enumerate(MINORITY_CLASSES, start=1):
        print(f"=== [{idx}/{len(MINORITY_CLASSES)}] Class: {cls} ===")
        t_cls = time.time()

        real_dir   = PROCESSED_DIR / cls / "real"
        real_paths = sorted(real_dir.glob("*.jpg")) if real_dir.exists() else []
        print(f"  real_dir : {real_dir}  |  {len(real_paths)} images found")

        if not real_paths:
            print(f"  [SKIP] No real images for {cls}\n")
            continue

        for stage in STAGES_TO_EVAL:
            t_stg  = time.time()
            result = evaluate_stage_vs_real(
                cls, stage, real_paths, N_EVAL,
                ms_ssim_metric, lpips_metric, inception_model
            )
            if result is not None:
                results.append(result)
                print(f"  [{cls}/{stage}] -> {result}  ({time.time()-t_stg:.1f}s)")
            else:
                print(f"  [{cls}/{stage}] skipped -- insufficient samples")

        print(f"=== Class {cls} done in {time.time()-t_cls:.1f}s ===\n")

    print(f"All classes processed in {time.time()-t_all:.1f}s total\n")

    # Save results
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(METRICS_CSV, index=False)
    print(f"Saved metrics to: {METRICS_CSV}")
    print("\nImage Quality Metrics (per class, per stage):")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
