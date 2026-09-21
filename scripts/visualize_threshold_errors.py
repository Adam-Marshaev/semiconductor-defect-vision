from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.classical_cv.threshold_baseline import (
    ThresholdBaselineConfig,
    predict_mask,
)

DATASET_ROOT = Path("data/raw/carinthia-s/data")
MANIFEST_PATH = Path("data/processed/manifest.csv")
RESULTS_PATH = Path(
    "reports/baselines/threshold_validation_best_per_image.csv"
)
OUTPUT_DIR = Path(
    "reports/figures/readme/threshold_baseline"
)

MASK_THRESHOLD = 128

CONFIG = ThresholdBaselineConfig(
    blur_kernel=3,
    invert=True,
    morphology_kernel=5,
    morphology_iterations=1,
)


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image_file:
        return np.asarray(
            image_file.convert("L"),
            dtype=np.uint8,
        )


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as mask_file:
        gray = np.asarray(
            mask_file.convert("L"),
            dtype=np.uint8,
        )

    return gray >= MASK_THRESHOLD


def make_figure(
    sample_ids: list[str],
    name: str,
) -> None:
    manifest = pd.read_csv(
        MANIFEST_PATH
    ).set_index("sample_id")

    results = pd.read_csv(
        RESULTS_PATH
    ).set_index("sample_id")

    rows = len(sample_ids)

    fig, axes = plt.subplots(
        rows,
        4,
        figsize=(14, 3.2 * rows),
    )

    if rows == 1:
        axes = np.expand_dims(
            axes,
            axis=0,
        )

    for row_index, sample_id in enumerate(
        sample_ids
    ):
        metadata = manifest.loc[sample_id]
        metrics = results.loc[sample_id]

        image = load_image(
            DATASET_ROOT
            / metadata["image_path"]
        )

        target = load_mask(
            DATASET_ROOT
            / metadata["mask_path"]
        )

        prediction = predict_mask(
            image,
            CONFIG,
        )

        axes[row_index, 0].imshow(
            image,
            cmap="gray",
        )
        axes[row_index, 0].set_title(
            f"SEM | label {metadata['label']}"
        )

        axes[row_index, 1].imshow(
            target,
            cmap="gray",
        )
        axes[row_index, 1].set_title(
            "Ground truth"
        )

        axes[row_index, 2].imshow(
            prediction,
            cmap="gray",
        )
        axes[row_index, 2].set_title(
            "Otsu prediction"
        )

        axes[row_index, 3].imshow(
            image,
            cmap="gray",
        )
        axes[row_index, 3].imshow(
            prediction,
            alpha=0.35,
            cmap="autumn",
        )
        axes[row_index, 3].set_title(
            f"Dice={metrics['dice']:.3f} | "
            f"pred={metrics['predicted_fraction']:.4f}"
        )

        for column in range(4):
            axes[row_index, column].axis(
                "off"
            )

        axes[row_index, 0].set_ylabel(
            sample_id[:8],
            rotation=0,
            labelpad=35,
        )

    fig.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR / f"{name}.png"
    )

    fig.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Wrote {output_path}")


def main() -> None:
    results = pd.read_csv(
        RESULTS_PATH
    )

    nonempty = results[
        ~results["empty_target"]
    ]

    empty = results[
        results["empty_target"]
    ]

    worst = (
        nonempty.nsmallest(
            6,
            "dice",
        )["sample_id"]
        .tolist()
    )

    best = (
        nonempty.nlargest(
            6,
            "dice",
        )["sample_id"]
        .tolist()
    )

    false_positive = (
        empty[
            ~empty["empty_prediction"]
        ]
        .sort_values(
            "predicted_fraction",
            ascending=False,
        )
        ["sample_id"]
        .tolist()
    )

    make_figure(
        worst,
        "worst_nonempty",
    )

    make_figure(
        best,
        "best_nonempty",
    )

    make_figure(
        false_positive,
        "empty_false_positives",
    )


if __name__ == "__main__":
    main()
