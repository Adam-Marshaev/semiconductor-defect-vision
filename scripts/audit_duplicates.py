import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DATASET_ROOT = Path("data/raw/carinthia-s/data")
MANIFEST_PATH = Path("data/processed/manifest.csv")
OUTPUT_PATH = Path("reports/eda/exact_duplicate_groups.csv")


def hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(
        array.tobytes()
    ).hexdigest()


def main() -> None:
    df = pd.read_csv(MANIFEST_PATH)

    records = []

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

        records.append(
            {
                "sample_id": row.sample_id,
                "label": row.label,
                "image_hash": hash_array(image),
            }
        )

        if index % 500 == 0 or index == total:
            print(f"Hashed {index}/{total}")

    hashes = pd.DataFrame(records)

    group_sizes = (
        hashes.groupby("image_hash")
        .size()
        .rename("group_size")
    )

    duplicates = hashes.join(
        group_sizes,
        on="image_hash",
    )

    duplicates = duplicates[
        duplicates["group_size"] > 1
    ].sort_values(
        ["group_size", "image_hash"],
        ascending=[False, True],
    )

    duplicates.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    duplicate_groups = (
        duplicates["image_hash"].nunique()
    )

    duplicate_samples = len(duplicates)

    cross_label_groups = 0

    if not duplicates.empty:
        cross_label_groups = (
            duplicates.groupby("image_hash")["label"]
            .nunique()
            .gt(1)
            .sum()
        )

    print()
    print("EXACT DUPLICATE AUDIT")
    print("=====================")
    print(f"total samples: {len(df)}")
    print(
        "unique decoded-image hashes:",
        hashes["image_hash"].nunique(),
    )
    print(
        "duplicate groups:",
        duplicate_groups,
    )
    print(
        "samples participating in duplicate groups:",
        duplicate_samples,
    )
    print(
        "duplicate groups containing multiple labels:",
        cross_label_groups,
    )
    print()
    print(f"Report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
