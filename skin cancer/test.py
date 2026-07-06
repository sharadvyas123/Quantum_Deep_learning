import numpy as np
from PIL import Image
from pathlib import Path


PROJECT_ROOT= Path().cwd()
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
DATASET_ROOT = PROJECT_ROOT / "dataset"
PROCESSED_DIR = DATASET_ROOT / "processed_images"

minority_classes = ["MEL", "BCC", "AKIEC", "BKL", "DF", "VASC"]

def coarse_image_stats(class_name, n_sample=50):
    """Quick health check: how much variance/structure does this class's coarse
    Q-DCGAN output actually have, compared to its own real images?"""
    aug_dir = PROCESSED_DIR / class_name / "augmented"
    real_dir = PROCESSED_DIR / class_name / "real"

    aug_paths = sorted(aug_dir.glob("*.jpg"))[:n_sample]
    real_paths = sorted(real_dir.glob("*.jpg"))[:n_sample]

    if not aug_paths or not real_paths:
        print(f"[{class_name}] Missing augmented or real images.")
        return

    aug_imgs = np.stack([np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0 for p in aug_paths])
    real_imgs = np.stack([np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0 for p in real_paths])

    # Std across the BATCH dimension, averaged over pixels — low value = mode collapse /
    # all generated images look nearly identical to each other.
    aug_batch_std = aug_imgs.std(axis=0).mean()
    real_batch_std = real_imgs.std(axis=0).mean()

    # Per-image spatial std, averaged over the batch — low value = flat/textureless images
    # (like the checkerboard-grid problem), independent of class-to-class variation.
    aug_spatial_std = aug_imgs.reshape(len(aug_imgs), -1).std(axis=1).mean()
    real_spatial_std = real_imgs.reshape(len(real_imgs), -1).std(axis=1).mean()

    print(f"[{class_name}]")
    print(f"  Coarse — variance ACROSS images (mode collapse check): {aug_batch_std:.5f}  (real: {real_batch_std:.5f})")
    print(f"  Coarse — variance WITHIN each image (texture/detail):  {aug_spatial_std:.5f}  (real: {real_spatial_std:.5f})")
    print()


for cls in minority_classes:
    coarse_image_stats(cls)