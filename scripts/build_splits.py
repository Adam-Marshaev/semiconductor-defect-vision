from pathlib import Path

import numpy as np
import pandas as pd

MANIFEST_PATH = Path("data/processed/manifest.csv")
LEAKAGE_GROUPS_PATH = Path(
    "data/processed/leakage_groups.csv"
)

SPLITS_PATH = Path("data/processed/splits.csv")
SUMMARY_PATH = Path(
    "reports/eda/split_class_counts.csv"
)

RANDOM_SEED = 42

SPLIT_NAMES = (
    "train",
    "val",
    "test",
)

# Explicit integer allocation.
#
# Goals:
# - exactly 4591 assigned samples
# - approximately 80/10/10 overall
# - every class represented in every partition
# - ultra-rare label 5 receives 2/1/1
CLASS_TARGETS = {
    1: {
        "train": 44,
        "val": 5,
        "test": 6,
    },
    2: {
        "train": 6,
        "val": 1,
        "test": 1,
    },
    3: {
        "train": 3207,
        "val": 401,
        "test": 400,
    },
    4: {
        "train": 231,
        "val": 29,
        "test": 29,
    },
    5: {
        "train": 2,
        "val": 1,
        "test": 1,
    },
    6: {
        "train": 183,
        "val": 22,
        "test": 22,
    },
}


def load_leakage_groups() -> pd.DataFrame:
    if not LEAKAGE_GROUPS_PATH.exists():
        raise FileNotFoundError(
            f"Missing leakage-group file: "
            f"{LEAKAGE_GROUPS_PATH}"
        )

    groups = pd.read_csv(
        LEAKAGE_GROUPS_PATH
    )

    required = {
        "group_id",
        "sample_id",
        "reason",
    }

    missing = required - set(groups.columns)

    if missing:
        raise ValueError(
            "Leakage-group file is missing "
            f"columns: {sorted(missing)}"
        )

    if groups["sample_id"].duplicated().any():
        duplicates = groups.loc[
            groups["sample_id"].duplicated(
                keep=False
            ),
            "sample_id",
        ].tolist()

        raise ValueError(
            "Samples occur in multiple leakage "
            f"groups: {duplicates}"
        )

    return groups


def validate_targets(
    manifest: pd.DataFrame,
) -> None:
    observed = (
        manifest["label"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    expected = {
        label: sum(targets.values())
        for label, targets
        in CLASS_TARGETS.items()
    }

    if observed != expected:
        raise ValueError(
            "Manifest class counts do not match "
            "the frozen split targets.\n"
            f"Observed: {observed}\n"
            f"Expected: {expected}"
        )


def build_units(
    class_df: pd.DataFrame,
    leakage_lookup: dict[str, str],
) -> list[dict]:
    units: dict[str, list[str]] = {}

    for row in class_df.itertuples(
        index=False
    ):
        leakage_group = leakage_lookup.get(
            row.sample_id
        )

        if leakage_group is None:
            unit_id = (
                f"singleton:{row.sample_id}"
            )
        else:
            unit_id = (
                f"leakage:{leakage_group}"
            )

        units.setdefault(
            unit_id,
            [],
        ).append(
            row.sample_id
        )

    return [
        {
            "unit_id": unit_id,
            "sample_ids": sample_ids,
            "size": len(sample_ids),
        }
        for unit_id, sample_ids
        in units.items()
    ]


def assign_class(
    class_df: pd.DataFrame,
    targets: dict[str, int],
    leakage_lookup: dict[str, str],
    seed: int,
) -> dict[str, str]:
    units = build_units(
        class_df,
        leakage_lookup,
    )

    rng = np.random.default_rng(seed)

    # Randomize units first, then process larger
    # groups before singletons. Processing larger
    # groups first prevents a 2-sample group from
    # being stranded after capacities reach 1.
    randomized = [
        (
            float(rng.random()),
            unit,
        )
        for unit in units
    ]

    randomized.sort(
        key=lambda item: (
            -item[1]["size"],
            item[0],
        )
    )

    remaining = dict(targets)

    assignments: dict[str, str] = {}

    for _, unit in randomized:
        size = unit["size"]

        eligible = [
            split
            for split in SPLIT_NAMES
            if remaining[split] >= size
        ]

        if not eligible:
            raise RuntimeError(
                "Unable to place group "
                f"{unit['unit_id']} "
                f"with size {size}. "
                f"Remaining capacities: "
                f"{remaining}"
            )

        weights = np.array(
            [
                remaining[split]
                for split in eligible
            ],
            dtype=np.float64,
        )

        probabilities = (
            weights / weights.sum()
        )

        split = str(
            rng.choice(
                eligible,
                p=probabilities,
            )
        )

        for sample_id in unit[
            "sample_ids"
        ]:
            assignments[
                sample_id
            ] = split

        remaining[split] -= size

    if any(
        value != 0
        for value in remaining.values()
    ):
        raise RuntimeError(
            "Class assignment did not fill "
            f"targets exactly: {remaining}"
        )

    return assignments


def validate_splits(
    splits: pd.DataFrame,
    manifest: pd.DataFrame,
    leakage_groups: pd.DataFrame,
) -> None:
    if len(splits) != len(manifest):
        raise ValueError(
            "Split row count does not match "
            "manifest row count"
        )

    if splits["sample_id"].duplicated().any():
        raise ValueError(
            "Duplicate sample IDs in splits"
        )

    if set(splits["sample_id"]) != set(
        manifest["sample_id"]
    ):
        raise ValueError(
            "Split sample IDs do not exactly "
            "match manifest sample IDs"
        )

    observed_splits = set(
        splits["split"]
    )

    if observed_splits != set(
        SPLIT_NAMES
    ):
        raise ValueError(
            "Unexpected split names: "
            f"{observed_splits}"
        )

    # Validate class counts exactly.
    for label, targets in (
        CLASS_TARGETS.items()
    ):
        class_rows = splits[
            splits["label"] == label
        ]

        for split, expected in (
            targets.items()
        ):
            observed = int(
                (
                    class_rows["split"]
                    == split
                ).sum()
            )

            if observed != expected:
                raise ValueError(
                    f"Label {label}, "
                    f"{split}: expected "
                    f"{expected}, got "
                    f"{observed}"
                )

    # Every leakage group must exist entirely
    # within one partition.
    merged = leakage_groups.merge(
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

    split_counts = (
        merged.groupby(
            "group_id"
        )["split"]
        .nunique()
    )

    crossing = split_counts[
        split_counts > 1
    ]

    if not crossing.empty:
        raise ValueError(
            "Leakage groups cross split "
            f"boundaries: "
            f"{crossing.to_dict()}"
        )


def main() -> None:
    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    leakage_groups = (
        load_leakage_groups()
    )

    validate_targets(
        manifest
    )

    missing_group_samples = set(
        leakage_groups["sample_id"]
    ) - set(
        manifest["sample_id"]
    )

    if missing_group_samples:
        raise ValueError(
            "Leakage-group samples missing "
            "from manifest: "
            f"{sorted(missing_group_samples)}"
        )

    label_lookup = manifest.set_index(
        "sample_id"
    )["label"].to_dict()

    group_label_counts = (
        leakage_groups.assign(
            label=leakage_groups[
                "sample_id"
            ].map(label_lookup)
        )
        .groupby("group_id")["label"]
        .nunique()
    )

    mixed_label_groups = (
        group_label_counts[
            group_label_counts > 1
        ]
    )

    if not mixed_label_groups.empty:
        raise ValueError(
            "Leakage groups contain multiple "
            "labels: "
            f"{mixed_label_groups.to_dict()}"
        )

    leakage_lookup = dict(
        zip(
            leakage_groups[
                "sample_id"
            ],
            leakage_groups[
                "group_id"
            ],
            strict=True,
        )
    )

    all_assignments: dict[
        str,
        str,
    ] = {}

    for label in sorted(
        CLASS_TARGETS
    ):
        class_df = (
            manifest[
                manifest["label"] == label
            ]
            .reset_index(drop=True)
        )

        assignments = assign_class(
            class_df=class_df,
            targets=CLASS_TARGETS[
                label
            ],
            leakage_lookup=(
                leakage_lookup
            ),
            seed=(
                RANDOM_SEED + label
            ),
        )

        all_assignments.update(
            assignments
        )

    splits = manifest[
        [
            "sample_id",
            "label",
        ]
    ].copy()

    splits["split"] = splits[
        "sample_id"
    ].map(
        all_assignments
    )

    splits["leakage_group_id"] = (
        splits["sample_id"].map(
            leakage_lookup
        )
    )

    validate_splits(
        splits,
        manifest,
        leakage_groups,
    )

    split_order = pd.Categorical(
        splits["split"],
        categories=SPLIT_NAMES,
        ordered=True,
    )

    splits = (
        splits.assign(
            _split_order=split_order
        )
        .sort_values(
            [
                "_split_order",
                "label",
                "sample_id",
            ]
        )
        .drop(
            columns="_split_order"
        )
        .reset_index(drop=True)
    )

    SPLITS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    splits.to_csv(
        SPLITS_PATH,
        index=False,
    )

    summary = (
        splits.groupby(
            [
                "label",
                "split",
            ],
            observed=False,
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reindex(
            columns=SPLIT_NAMES,
            fill_value=0,
        )
    )

    summary["total"] = (
        summary.sum(axis=1)
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_PATH
    )

    totals = (
        splits["split"]
        .value_counts()
        .reindex(SPLIT_NAMES)
    )

    print("SPLIT VALIDATION PASSED")
    print("=======================")
    print()
    print(summary.to_string())
    print()
    print("TOTALS")
    print("======")

    for split in SPLIT_NAMES:
        count = int(
            totals[split]
        )

        fraction = (
            count / len(splits)
        )

        print(
            f"{split:5} "
            f"{count:4} "
            f"({fraction:.2%})"
        )

    print()
    print("LEAKAGE GROUPS")
    print("==============")

    grouped = splits[
        splits[
            "leakage_group_id"
        ].notna()
    ]

    print(
        grouped.to_string(
            index=False
        )
    )

    print()
    print(
        f"Random seed: {RANDOM_SEED}"
    )
    print(
        f"Wrote: {SPLITS_PATH}"
    )
    print(
        f"Wrote: {SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()
