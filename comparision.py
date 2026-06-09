"""
generate_lowlight_grid.py
--------------------------
Generates a 2-row comparison grid:
  Row 1: One original FairFace image per racial group (7 images)
  Row 2: The same images after low-light simulation (pixel * 0.2)

Output: lowlight_comparison.png  (drop it next to your .tex file)

Requirements:
    pip install opencv-python matplotlib numpy pandas

Usage:
    python generate_lowlight_grid.py \
        --csv  path/to/fairface_label_val.csv \
        --imgs path/to/fairface_val/         \
        --out  lowlight_comparison.png
"""

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ────────────────────────────────────────────────────────────────────
GAMMA        = 2.5          # darkening strength (higher = darker)
IMG_SIZE     = 224          # pixels for each cell in the grid
GROUPS = [
    "White",
    "East Asian",
    "Southeast Asian",
    "Latino_Hispanic",   # FairFace CSV spelling
    "Middle Eastern",
    "Black",
    "Indian",
]
DISPLAY_NAMES = [           # prettier labels for the figure
    "White",
    "East Asian",
    "SE Asian",
    "Latino /\nHispanic",
    "Middle\nEastern",
    "Black",
    "Indian",
]
# Highlight groups with the biggest low-light drop
HIGHLIGHT = {"Black", "Indian"}
# ─────────────────────────────────────────────────────────────────────────────


def simulate_low_light(img_bgr: np.ndarray) -> np.ndarray:
    """Simulate low-light by scaling pixel values by 0.2 (matches experiment code)."""
    return (img_bgr * 0.2).astype("uint8")


def load_one_per_group(csv_path: str, img_dir: str):
    """
    Read FairFace validation CSV, pick the first image for each racial group.
    Returns dict: group_name -> np.ndarray (BGR, IMG_SIZE x IMG_SIZE)
    """
    df = pd.read_csv(csv_path)

    # FairFace CSV columns: 'file', 'age', 'gender', 'race', 'service_test'
    # 'race' values match GROUPS list above
    samples = {}
    for group in GROUPS:
        rows = df[df["race"] == group]
        if rows.empty:
            raise ValueError(
                f"No rows found for group '{group}'. "
                f"Check that GROUPS matches the CSV 'race' column values.\n"
                f"Unique values in CSV: {df['race'].unique().tolist()}"
            )
        filename = rows.iloc[0]["file"]                 # e.g. "val/1.jpg"
        # The CSV 'file' column may include a subfolder prefix — strip it
        filename = os.path.basename(filename)
        filepath = os.path.join(img_dir, filename)
        img = cv2.imread(filepath)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {filepath}")
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        samples[group] = img

    return samples


def build_grid(samples: dict):
    """
    Build a 2 x 7 matplotlib figure.
    Row 0: original images
    Row 1: gamma-darkened images
    """
    n = len(GROUPS)
    fig, axes = plt.subplots(
        nrows=2, ncols=n,
        figsize=(n * 2.2, 5.5),
        gridspec_kw={"hspace": 0.08, "wspace": 0.04},
    )

    row_labels = ["Original", "Low-Light\n(× 0.2)"]

    for col, (group, display) in enumerate(zip(GROUPS, DISPLAY_NAMES)):
        orig = samples[group]
        dark = simulate_low_light(orig)

        for row, img in enumerate([orig, dark]):
            ax = axes[row, col]
            # OpenCV is BGR; matplotlib expects RGB
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_xticks([])
            ax.set_yticks([])

            # Red border for highlighted groups
            for spine in ax.spines.values():
                spine.set_linewidth(2.5)
                spine.set_edgecolor(
                    "#d62728" if group in HIGHLIGHT else "#cccccc"
                )

        # Column header (group name) — red text for highlighted groups
        color = "#d62728" if group in HIGHLIGHT else "black"
        axes[0, col].set_title(
            display,
            fontsize=8.5,
            fontweight="bold" if group in HIGHLIGHT else "normal",
            color=color,
            pad=4,
        )

    # Row labels on the left
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(
            label,
            fontsize=9,
            fontweight="bold",
            rotation=90,
            labelpad=6,
        )

    # Legend for highlighted groups
    patch = mpatches.Patch(
        edgecolor="#d62728", facecolor="none",
        linewidth=2.5,
        label="Largest low-light performance gap (Black, Indian)",
    )
    fig.legend(
        handles=[patch],
        loc="lower center",
        ncol=1,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "FairFace Validation Samples: Original vs. Low-Light Transformation",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )

    return fig


def main():
    global GAMMA

    CSV_PATH = "fairface_label_val.csv"
    IMGS_DIR = "val"
    OUT_FILE = "lowlight_comparison.png"
    GAMMA    = 2.5

    print("Loading one image per racial group...")
    samples = load_one_per_group(CSV_PATH, IMGS_DIR)
    print(f"  Loaded {len(samples)} groups: {list(samples.keys())}")

    print("Building comparison grid...")
    fig = build_grid(samples)

    fig.savefig(
        OUT_FILE,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"Saved → {OUT_FILE}")
    plt.close(fig)


if __name__ == "__main__":
    main()
