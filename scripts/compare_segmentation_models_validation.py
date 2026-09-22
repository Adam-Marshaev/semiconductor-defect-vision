from pathlib import Path

import pandas as pd

UNET_PATH = Path(
    "reports/training/unet_validation_per_image.csv"
)
SEGFORMER_PATH = Path(
    "reports/training/segformer_validation_per_image.csv"
)

OUTPUT_DIR = Path(
    "reports/training"
)

PER_IMAGE_OUTPUT = (
    OUTPUT_DIR
    / "unet_vs_segformer_validation.csv"
)

BY_CLASS_OUTPUT = (
    OUTPUT_DIR
    / "unet_vs_segformer_validation_by_class.csv"
)


def main() -> None:
    unet = pd.read_csv(
        UNET_PATH
    )

    segformer = pd.read_csv(
        SEGFORMER_PATH
    )

    merged = unet.merge(
        segformer,
        on=[
            "sample_id",
            "label",
        ],
        suffixes=(
            "_unet",
            "_segformer",
        ),
        validate="one_to_one",
    )

    for metric in [
        "dice",
        "iou",
        "precision",
        "recall",
    ]:
        merged[
            f"delta_{metric}"
        ] = (
            merged[
                f"{metric}_segformer"
            ]
            - merged[
                f"{metric}_unet"
            ]
        )

    merged[
        "target_pixels"
    ] = (
        merged["true_positive_unet"]
        + merged["false_negative_unet"]
    )

    merged[
        "prediction_pixels_unet"
    ] = (
        merged["true_positive_unet"]
        + merged["false_positive_unet"]
    )

    merged[
        "prediction_pixels_segformer"
    ] = (
        merged[
            "true_positive_segformer"
        ]
        + merged[
            "false_positive_segformer"
        ]
    )

    by_class = (
        merged.groupby("label")
        .agg(
            samples=("sample_id", "count"),
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
            median_delta_dice=(
                "delta_dice",
                "median",
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

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        PER_IMAGE_OUTPUT,
        index=False,
    )

    by_class.to_csv(
        BY_CLASS_OUTPUT,
        index=False,
    )

    nonempty = merged.loc[
        ~merged[
            "empty_target_unet"
        ]
    ]

    print(
        "U-NET VS SEGFORMER VALIDATION"
    )
    print(
        "============================"
    )
    print(
        "samples:",
        len(merged),
    )
    print(
        "mean U-Net Dice:",
        f"{merged['dice_unet'].mean():.6f}",
    )
    print(
        "mean SegFormer Dice:",
        f"{merged['dice_segformer'].mean():.6f}",
    )
    print(
        "mean Dice delta:",
        f"{merged['delta_dice'].mean():+.6f}",
    )
    print(
        "nonempty mean delta:",
        f"{nonempty['delta_dice'].mean():+.6f}",
    )

    print()
    print("BY CLASS")
    print("========")

    print(
        by_class.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    columns = [
        "sample_id",
        "label",
        "dice_unet",
        "dice_segformer",
        "delta_dice",
        "precision_unet",
        "precision_segformer",
        "recall_unet",
        "recall_segformer",
        "target_pixels",
    ]

    print()
    print("TOP 10 IMPROVEMENTS")
    print("===================")

    print(
        nonempty.sort_values(
            "delta_dice",
            ascending=False,
        )
        .head(10)[columns]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("TOP 10 REGRESSIONS")
    print("==================")

    print(
        nonempty.sort_values(
            "delta_dice",
        )
        .head(10)[columns]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )


if __name__ == "__main__":
    main()
