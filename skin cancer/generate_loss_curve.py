import re
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# Configuration
# ============================================================
_cwd = Path.cwd()
ROOT = _cwd.parent
OUTPUT_DIR = ROOT / 'results' / 'figures'
LOG_DIR = ROOT/'results' / 'logs'
LOG_FILE = LOG_DIR / "server_training.log"
OUTPUT_FILE = OUTPUT_DIR / "pix2pix_training_loss.png"

# ============================================================
# Extract losses from training log
# ============================================================

def parse_training_log(log_file):

    pattern = re.compile(
        r"Epoch\s+(\d+)/\d+.*?"
        r"D_loss=([0-9.]+),\s*"
        r"G_loss=([0-9.]+)"
    )

    epochs = []
    d_losses = []
    g_losses = []

    with open(log_file, "r", encoding="utf-8") as file:

        for line in file:

            match = pattern.search(line)

            if match:

                epochs.append(int(match.group(1)))
                d_losses.append(float(match.group(2)))
                g_losses.append(float(match.group(3)))

    if len(epochs) == 0:
        raise ValueError("No training loss values found in the log.")

    return epochs, d_losses, g_losses


# ============================================================
# Generate publication-quality plot
# ============================================================

def generate_plot(epochs, d_losses, g_losses):

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 6.5),
        sharex=True
    )

    # --------------------------------------------------------
    # Generator Loss
    # --------------------------------------------------------

    axes[0].plot(
        epochs,
        g_losses,
        linewidth=2.0
    )

    axes[0].set_ylabel(
        "Generator Loss",
        fontsize=12
    )

    axes[0].set_title(
        "Generator and Discriminator Loss During Q-Pix2Pix Training",
        fontsize=13,
        pad=10
    )

    axes[0].grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.5
    )

    # Give a small margin around the actual values
    g_min = min(g_losses)
    g_max = max(g_losses)
    g_margin = (g_max - g_min) * 0.15

    axes[0].set_ylim(
        g_min - g_margin,
        g_max + g_margin
    )


    # --------------------------------------------------------
    # Discriminator Loss
    # --------------------------------------------------------

    axes[1].plot(
        epochs,
        d_losses,
        linewidth=2.0
    )

    axes[1].set_ylabel(
        "Discriminator Loss",
        fontsize=12
    )

    axes[1].set_xlabel(
        "Epoch",
        fontsize=12
    )

    axes[1].grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.5
    )

    d_min = min(d_losses)
    d_max = max(d_losses)
    d_margin = (d_max - d_min) * 0.15

    axes[1].set_ylim(
        d_min - d_margin,
        d_max + d_margin
    )


    # --------------------------------------------------------
    # Common X-axis
    # --------------------------------------------------------

    axes[1].set_xlim(
        min(epochs),
        max(epochs)
    )


    # --------------------------------------------------------
    # Final epoch markers
    # --------------------------------------------------------

    axes[0].scatter(
        epochs[-1],
        g_losses[-1],
        s=35,
        zorder=5
    )

    axes[1].scatter(
        epochs[-1],
        d_losses[-1],
        s=35,
        zorder=5
    )


    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FILE,
        dpi=600,
        bbox_inches="tight"
    )

    plt.show()

    print("=" * 60)
    print("Q-Pix2Pix Training Loss Figure")
    print("=" * 60)
    print(f"Epoch range       : {epochs[0]} - {epochs[-1]}")
    print(f"Final G loss      : {g_losses[-1]:.4f}")
    print(f"Final D loss      : {d_losses[-1]:.4f}")
    print(f"Saved to          : {OUTPUT_FILE}")
    print("=" * 60)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    epochs, d_losses, g_losses = parse_training_log(
        LOG_FILE
    )

    generate_plot(
        epochs,
        d_losses,
        g_losses
    )