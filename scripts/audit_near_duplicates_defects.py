from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import NearestNeighbors

DATASET_ROOT = Path("data/raw/carinthia-s/data")
MANIFEST_PATH = Path("data/processed/manifest.csv")
OUTPUT_PATH = Path("reports/eda/near_duplicate_candidates_defect_classes.csv")

RESIZE = 24
NEIGHBORS = 6
TOP_PAIRS = 100


def make_descriptor(image: np.ndarray) -> np.ndarray:
    """Create a compact brightness-normalized structural descriptor."""

    resized = cv2.resize(
        image,
        (RESIZE, RESIZE),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)

    mean = float(resized.mean())
    std = float(resized.std())

    if std > 1e-6:
        resized = (resized - mean) / std
    else:
        resized = resized - mean

    descriptor = resized.ravel()

    norm = np.linalg.norm(descriptor)

    if norm > 0:
        descriptor = descriptor / norm

    return descriptor


def main() -> None:
    df = pd.read_csv(MANIFEST_PATH)
    df = df[df["label"] != 6].reset_index(drop=True)

    descriptors = []

    total = len(df)

    for index, row in enumerate(
        df.itertuples(index=False),
        start=1,
    ):
        image_path = DATASET_ROOT / row.image_path

        with Image.open(image_path) as image_file:
            image = np.asarray(
                image_file.convert("L")
            )

        descriptors.append(
            make_descriptor(image)
        )

        if index % 500 == 0 or index == total:
            print(
                f"Built descriptor {index}/{total}"
            )

    matrix = np.stack(descriptors)

    print()
    print(
        f"Descriptor matrix: {matrix.shape}"
    )

    model = NearestNeighbors(
        n_neighbors=NEIGHBORS,
        metric="cosine",
        algorithm="brute",
    )

    model.fit(matrix)

    distances, indices = model.kneighbors(matrix)

    pairs = {}

    for source_index in range(len(df)):
        for rank in range(1, NEIGHBORS):
            target_index = int(
                indices[source_index, rank]
            )

            pair_key = tuple(
                sorted(
                    (
                        source_index,
                        target_index,
                    )
                )
            )

            distance = float(
                distances[source_index, rank]
            )

            if (
                pair_key not in pairs
                or distance < pairs[pair_key]
            ):
                pairs[pair_key] = distance

    records = []

    for (
        source_index,
        target_index,
    ), distance in pairs.items():
        source = df.iloc[source_index]
        target = df.iloc[target_index]

        records.append(
            {
                "sample_id_a": source["sample_id"],
                "label_a": int(source["label"]),
                "sample_id_b": target["sample_id"],
                "label_b": int(target["label"]),
                "same_label": (
                    int(source["label"])
                    == int(target["label"])
                ),
                "cosine_distance": distance,
                "cosine_similarity": 1.0 - distance,
            }
        )

    candidates = (
        pd.DataFrame(records)
        .sort_values(
            "cosine_distance",
            ascending=True,
        )
        .head(TOP_PAIRS)
    )

    candidates.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("NEAR-DUPLICATE CANDIDATES")
    print("=========================")
    print(
        f"candidate pairs saved: {len(candidates)}"
    )

    print()
    print("Top 20:")
    print(
        candidates.head(20).to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print(f"Report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
