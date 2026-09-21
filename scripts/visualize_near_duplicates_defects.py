from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

DATASET_ROOT = Path("data/raw/carinthia-s/data")
MANIFEST_PATH = Path("data/processed/manifest.csv")
CANDIDATE_PATH = Path("reports/eda/near_duplicate_candidates_defect_classes.csv")

OUTPUT_DIR = Path("reports/figures/readme/near_duplicates_defects")

TOP_N = 10


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image_file:
        return np.asarray(image_file.convert("L"))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH)
    candidates = pd.read_csv(CANDIDATE_PATH).head(TOP_N)

    path_lookup = manifest.set_index("sample_id")["image_path"].to_dict()

    fig, axes = plt.subplots(
        TOP_N,
        2,
        figsize=(8, 3 * TOP_N),
    )

    for rank, row in enumerate(
        candidates.itertuples(index=False),
        start=1,
    ):
        image_a = load_image(
            DATASET_ROOT / path_lookup[row.sample_id_a]
        )

        image_b = load_image(
            DATASET_ROOT / path_lookup[row.sample_id_b]
        )

        ax_a = axes[rank - 1, 0]
        ax_b = axes[rank - 1, 1]

        ax_a.imshow(image_a, cmap="gray")
        ax_a.set_title(
            f"#{rank} A | label {row.label_a}\n"
            f"{row.sample_id_a}"
        )
        ax_a.axis("off")

        ax_b.imshow(image_b, cmap="gray")
        ax_b.set_title(
            f"#{rank} B | label {row.label_b}\n"
            f"similarity={row.cosine_similarity:.6f}\n"
            f"{row.sample_id_b}"
        )
        ax_b.axis("off")

    fig.suptitle(
        "Top Near-Duplicate Candidates",
        fontsize=14,
    )

    fig.tight_layout()

    output_path = OUTPUT_DIR / "top_10_pairs.png"

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
