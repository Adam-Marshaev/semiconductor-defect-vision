from pathlib import Path

import pandas as pd

INPUT_PATH = Path(
    "reports/training/"
    "unet_vs_segformer_validation.csv"
)

OUTPUT_PATH = Path(
    "reports/training/"
    "unet_vs_segformer_by_defect_size.csv"
)


def main() -> None:
    results = pd.read_csv(
        INPUT_PATH
    )

    nonempty = results.loc[
        results["target_pixels"] > 0
    ].copy()

    nonempty["size_group"] = pd.qcut(
        nonempty["target_pixels"],
        q=4,
        labels=[
            "Q1_smallest",
            "Q2_small",
            "Q3_large",
            "Q4_largest",
        ],
    )

    summary = (
        nonempty.groupby(
            "size_group",
            observed=True,
        )
        .agg(
            samples=(
                "sample_id",
                "count",
            ),
            min_target_pixels=(
                "target_pixels",
                "min",
            ),
            median_target_pixels=(
                "target_pixels",
                "median",
            ),
            max_target_pixels=(
                "target_pixels",
                "max",
            ),
            mean_dice_unet=(
                "dice_unet",
                "mean",
            ),
            mean_dice_segformer=(
                "dice_segformer",
                "mean",
            ),
            mean_delta_dice=(
                "delta_dice",
                "mean",
            ),
            mean_precision_unet=(
                "precision_unet",
                "mean",
            ),
            mean_precision_segformer=(
                "precision_segformer",
                "mean",
            ),
            mean_recall_unet=(
                "recall_unet",
                "mean",
            ),
            mean_recall_segformer=(
                "recall_segformer",
                "mean",
            ),
            improved_samples=(
                "delta_dice",
                lambda values: (
                    values > 0
                ).sum(),
            ),
            regressed_samples=(
                "delta_dice",
                lambda values: (
                    values < 0
                ).sum(),
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "SEGMENTATION BY DEFECT SIZE"
    )
    print(
        "==========================="
    )
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print(
        "Note: size groups are quartiles "
        "of validation target-mask area."
    )


if __name__ == "__main__":
    main()
