import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import random


# ============================================================
# Configuration
# ============================================================

_cwd = Path.cwd()
ROOT = _cwd.parent

PROCESSED_DATA_DIR = _cwd / "dataset" / "processed_images"
METEDATA_FILE = PROCESSED_DATA_DIR / "final_labels_onehot.csv"

OUTPUT_DIR = ROOT / "results" / "figures"

OUTPUT_FILE_1 = OUTPUT_DIR / "qualitative_real.png"
OUTPUT_FILE_2 = OUTPUT_DIR / "qualitative_generated.png"

WANTED_CLASSES = ["MEL", "VASC", "DF", "BCC"]

# Fixed seed for reproducible qualitative samples
RANDOM_SEED = 42


# ============================================================
# Supported image formats
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg"
}


# ============================================================
# Utility Functions
# ============================================================

def get_images(directory):
    """
    Return all supported image files inside a directory.
    """

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory does not exist:\n{directory}"
        )

    images = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        raise FileNotFoundError(
            f"No images found in:\n{directory}"
        )

    return sorted(images)


def find_matching_image(directory, filename):
    """
    Find an image with the same filename in the target directory.
    """

    exact_path = directory / filename

    if exact_path.exists():
        return exact_path

    # Fallback: match by filename stem
    stem = Path(filename).stem

    candidates = [
        path
        for path in get_images(directory)
        if path.stem == stem
    ]

    if candidates:
        return candidates[0]

    return None


def select_images():
    """
    Select one real image from each requested class and
    find the corresponding normalized image.
    """

    random.seed(RANDOM_SEED)

    selected_real = {}
    selected_normalized = {}

    print("=" * 70)
    print("Selecting qualitative evaluation images")
    print("=" * 70)

    for class_name in WANTED_CLASSES:

        real_dir = (
            PROCESSED_DATA_DIR
            / class_name
            / "real"
        )

        normalized_dir = (
            PROCESSED_DATA_DIR
            / class_name
            / "normalized"
        )

        real_images = get_images(real_dir)

        # Select one random real image
        real_image = random.choice(real_images)

        # Try to find corresponding normalized image
        normalized_image = find_matching_image(
            normalized_dir,
            real_image.name
        )

        # Fallback if matching file does not exist
        if normalized_image is None:

            normalized_images = get_images(
                normalized_dir
            )

            normalized_image = random.choice(
                normalized_images
            )

            print(
                f"[WARNING] No matching normalized image "
                f"found for {real_image.name} in {class_name}. "
                f"Using a random normalized image."
            )

        selected_real[class_name] = real_image
        selected_normalized[class_name] = normalized_image

        print(f"\nClass: {class_name}")
        print(f"  Real       : {real_image.name}")
        print(f"  Normalized : {normalized_image.name}")

    print("\n" + "=" * 70)

    return selected_real, selected_normalized


# ============================================================
# Create Tight 2x2 Composite Image
# ============================================================

def create_composite(
    selected_images,
    output_file
):
    """
    Create a tightly packed 2x2 image grid.

    The output contains ONLY the four images.
    No titles, labels, axes, or surrounding whitespace.
    """

    # --------------------------------------------------------
    # Read images first
    # --------------------------------------------------------

    images = []

    for class_name in WANTED_CLASSES:

        image_path = selected_images[class_name]

        image = plt.imread(image_path)

        images.append(image)

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(4, 4),
        gridspec_kw={
            "wspace": 0,
            "hspace": 0
        }
    )

    axes = axes.flatten()

    # --------------------------------------------------------
    # Display images
    # --------------------------------------------------------

    for ax, image in zip(axes, images):

        ax.imshow(
            image,
            interpolation="nearest"
        )

        # Remove everything except the image
        ax.axis("off")

    # --------------------------------------------------------
    # Remove all padding
    # --------------------------------------------------------

    plt.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=1,
        wspace=0,
        hspace=0
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    fig.savefig(
        output_file,
        dpi=600,
        bbox_inches=None,
        pad_inches=0
    )

    plt.close(fig)

    print(f"[SAVED] {output_file}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Select corresponding samples
    # --------------------------------------------------------

    real_images, normalized_images = select_images()

    # --------------------------------------------------------
    # Generate REAL composite
    # --------------------------------------------------------

    create_composite(
        selected_images=real_images,
        output_file=OUTPUT_FILE_1
    )

    # --------------------------------------------------------
    # Generate PROPOSED FRAMEWORK composite
    # --------------------------------------------------------

    create_composite(
        selected_images=normalized_images,
        output_file=OUTPUT_FILE_2
    )

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("QUALITATIVE FIGURES GENERATED SUCCESSFULLY")
    print("=" * 70)

    print(f"Real figure            : {OUTPUT_FILE_1}")
    print(f"Proposed Framework     : {OUTPUT_FILE_2}")

    print("\nClasses included:")

    for class_name in WANTED_CLASSES:
        print(f"  - {class_name}")

    print("=" * 70)