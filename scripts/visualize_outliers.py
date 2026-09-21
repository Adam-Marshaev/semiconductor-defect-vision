from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

DATASET_ROOT = Path("data/raw/carinthia-s/data")
MANIFEST_PATH = Path("data/processed/manifest.csv")
OUTPUT_DIR = Path("reports/figures/readme/outliers")

MASK_THRESHOLD = 128

SAMPLES = {
    "label6_nonempty": "8b782473c99742e394df74b430412232",
    "largest_defect": "3531e9467b034961b1afe510af96fea1",
    "large_label3": "3b463b1344f14139961e9336ccf02e9e",
    "most_components": "ed11de8d30a443328f589c582a43d972",
    "second_most_components": "87355a31111a4174a68cb9dd82d5591c",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH)

    for name, sample_id in SAMPLES.items():
        matches = manifest[manifest["sample_id"] == sample_id]

        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one row for {sample_id}, found {len(matches)}"
            )

        row = matches.iloc[0]

        image_path = DATASET_ROOT / row["image_path"]
        mask_path = DATASET_ROOT / row["mask_path"]

        with Image.open(image_path) as image_file:
            image = np.asarray(image_file.convert("L"))

        with Image.open(mask_path) as mask_file:
            gray_mask = np.asarray(mask_file.convert("L"))

        mask = gray_mask >= MASK_THRESHOLD

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title(
            f"SEM image\nlabel={row['label']}"
        )
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title(
            f"Canonical mask\n"
            f"area={row['defect_fraction']:.4f}, "
            f"components={row['component_count']}"
        )
        axes[1].axis("off")

        axes[2].imshow(image, cmap="gray")
        axes[2].imshow(
            mask,
            alpha=0.35,
            cmap="autumn",
        )

        if not row["empty_mask"]:
            rectangle = Rectangle(
                (row["bbox_x"], row["bbox_y"]),
                row["bbox_width"],
                row["bbox_height"],
                fill=False,
                linewidth=1.5,
            )
            axes[2].add_patch(rectangle)

        axes[2].set_title("Mask overlay + bounding box")
        axes[2].axis("off")

        fig.suptitle(
            f"{name}: {sample_id}",
            fontsize=10,
        )

        fig.tight_layout()

        output_path = OUTPUT_DIR / f"{name}.png"

        fig.savefig(
            output_path,
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
