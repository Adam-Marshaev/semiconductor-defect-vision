from pathlib import Path

import pandas as pd

MANIFEST_PATH = Path("data/processed/manifest.csv")
SPLITS_PATH = Path("data/processed/splits.csv")
LEAKAGE_GROUPS_PATH = Path(
    "data/processed/leakage_groups.csv"
)

EXPECTED_SPLIT_COUNTS = {
    "train": 3673,
    "val": 459,
    "test": 459,
}

EXPECTED_CLASS_SPLITS = {
    1: {"train": 44, "val": 5, "test": 6},
    2: {"train": 6, "val": 1, "test": 1},
    3: {"train": 3207, "val": 401, "test": 400},
    4: {"train": 231, "val": 29, "test": 29},
    5: {"train": 2, "val": 1, "test": 1},
    6: {"train": 183, "val": 22, "test": 22},
}


def test_splits_cover_manifest_exactly_once() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    splits = pd.read_csv(SPLITS_PATH)

    assert len(manifest) == 4591
    assert len(splits) == len(manifest)

    assert manifest["sample_id"].is_unique
    assert splits["sample_id"].is_unique

    assert set(splits["sample_id"]) == set(
        manifest["sample_id"]
    )


def test_split_and_class_counts_are_frozen() -> None:
    splits = pd.read_csv(SPLITS_PATH)

    observed_split_counts = (
        splits["split"].value_counts().to_dict()
    )

    assert (
        observed_split_counts
        == EXPECTED_SPLIT_COUNTS
    )

    for label, expected in (
        EXPECTED_CLASS_SPLITS.items()
    ):
        class_rows = splits[
            splits["label"] == label
        ]

        observed = (
            class_rows["split"]
            .value_counts()
            .to_dict()
        )

        assert observed == expected


def test_leakage_groups_do_not_cross_partitions() -> None:
    splits = pd.read_csv(SPLITS_PATH)
    groups = pd.read_csv(LEAKAGE_GROUPS_PATH)

    merged = groups.merge(
        splits[
            [
                "sample_id",
                "split",
            ]
        ],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )

    assert merged["split"].notna().all()

    partitions_per_group = (
        merged.groupby("group_id")["split"]
        .nunique()
    )

    assert (
        partitions_per_group <= 1
    ).all()
