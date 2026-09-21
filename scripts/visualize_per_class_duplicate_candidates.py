from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

DATASET_ROOT = Path("data/raw/carinthia-s/data")
MANIFEST_PATH = Path("data/processed/manifest.csv")

CSV_OUTPUT = Path(
    "reports/eda/per_class_near_duplicate_candidates.csv"
)
FIGURE_OUTPUT = Path(
    "reports/figures/readme/"
    "per_class_near_duplicate_candidates.png"
)

LABELS = [1, 2, 4, 5]
TOP_N = 3
RESIZE = 24


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image_file:
        return np.asarray(image_file.convert("L"))


def descriptor(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(
        image,
        (RESIZE, RESIZE),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)

    resized -= resized.mean()

    std = float(resized.std())

    if std > 1e-6:
        resized /= std

    vector = resized.ravel()

    norm = np.linalg.norm(vector)

    if norm > 0:
        vector /= norm

    return vector


def main() -> None:
    df = pd.read_csv(MANIFEST_PATH)

    records = []

    for label in LABELS:
        class_df = (
            df[df["label"] == label]
            .reset_index(drop=True)
        )

        images = []
        vectors = []

        for row in class_df.itertuples(index=False):
            image = load_image(
                DATASET_ROOT / row.image_path
            )

            images.append(image)
            vectors.append(descriptor(image))

        matrix = np.stack(vectors)

        similarities = matrix @ matrix.T

        rows, cols = np.triu_indices(
            len(class_df),
            k=1,
        )

        scores = similarities[rows, cols]

        order = np.argsort(-scores)[:TOP_N]

        for rank, index in enumerate(
            order,
            start=1,
        ):
            a = int(rows[index])
            b = int(cols[index])

            records.append(
                {
                    "label": label,
                    "rank": rank,
                    "sample_id_a": class_df.loc[
                        a, "sample_id"
                    ],
                    "sample_id_b": class_df.loc[
                        b, "sample_id"
                    ],
                    "similarity": float(
                        scores[index]
                    ),
                    "image_a": images[a],
                    "image_b": images[b],
                }
            )

    CSV_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        [
            {
                key: value
                for key, value in record.items()
                if key not in {"image_a", "image_b"}
            }
            for record in records
        ]
    ).to_csv(
        CSV_OUTPUT,
        index=False,
    )

    figure_rows = len(records)

    fig, axes = plt.subplots(
        figure_rows,
        2,
        figsize=(8, figure_rows * 3),
    )

    for row_index, record in enumerate(records):
        axes[row_index, 0].imshow(
            record["image_a"],
            cmap="gray",
        )

        axes[row_index, 0].set_title(
            f"Label {record['label']} "
            f"pair #{record['rank']} A\n"
            f"{record['sample_id_a']}"
        )

        axes[row_index, 0].axis("off")

        axes[row_index, 1].imshow(
            record["image_b"],
            cmap="gray",
        )

        axes[row_index, 1].set_title(
            f"B | similarity="
            f"{record['similarity']:.6f}\n"
            f"{record['sample_id_b']}"
        )

        axes[row_index, 1].axis("off")

    fig.tight_layout()

    FIGURE_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        FIGURE_OUTPUT,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Wrote {CSV_OUTPUT}")
    print(f"Wrote {FIGURE_OUTPUT}")


if __name__ == "__main__":
    main()
