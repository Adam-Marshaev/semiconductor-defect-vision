import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.classical_cv.adaptive_threshold_baseline import (
    AdaptiveThresholdConfig,
    predict_mask,
)
from src.evaluation.segmentation_metrics import (
    compute_segmentation_metrics,
)

DATASET_ROOT = Path("data/raw/carinthia-s/data")
MANIFEST_PATH = Path("data/processed/manifest.csv")
SPLITS_PATH = Path("data/processed/splits.csv")

OUTPUT_DIR = Path("reports/baselines")

GRID_PATH = (
    OUTPUT_DIR
    / "adaptive_threshold_validation_grid.csv"
)
PER_IMAGE_PATH = (
    OUTPUT_DIR
    / "adaptive_threshold_validation_best_per_image.csv"
)
CLASS_SUMMARY_PATH = (
    OUTPUT_DIR
    / "adaptive_threshold_validation_best_by_class.csv"
)
BEST_CONFIG_PATH = (
    OUTPUT_DIR
    / "adaptive_threshold_validation_best_config.json"
)

MASK_THRESHOLD = 128


def build_configs() -> list[AdaptiveThresholdConfig]:
    configs = []

    for blur_kernel in [1, 3]:
        for block_size in [15, 31, 61]:
            for c in [2.0, 5.0, 10.0]:
                for morphology in [
                    "none",
                    "open",
                    "close",
                ]:
                    configs.append(
                        AdaptiveThresholdConfig(
                            blur_kernel=blur_kernel,
                            block_size=block_size,
                            c=c,
                            invert=True,
                            morphology=morphology,
                            morphology_kernel=3,
                        )
                    )

    return configs


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image_file:
        return np.asarray(
            image_file.convert("L"),
            dtype=np.uint8,
        )


def load_target(path: Path) -> np.ndarray:
    with Image.open(path) as mask_file:
        gray = np.asarray(
            mask_file.convert("L"),
            dtype=np.uint8,
        )

    return gray >= MASK_THRESHOLD


def summarize_config(
    rows: pd.DataFrame,
) -> dict:
    nonempty = rows[
        ~rows["empty_target"]
    ]

    empty = rows[
        rows["empty_target"]
    ]

    if len(empty) > 0:
        empty_target_accuracy = float(
            empty["empty_prediction"].mean()
        )
    else:
        empty_target_accuracy = float("nan")

    return {
        "mean_dice": float(
            rows["dice"].mean()
        ),
        "median_dice": float(
            rows["dice"].median()
        ),
        "mean_iou": float(
            rows["iou"].mean()
        ),
        "median_iou": float(
            rows["iou"].median()
        ),
        "mean_precision": float(
            rows["precision"].mean()
        ),
        "mean_recall": float(
            rows["recall"].mean()
        ),
        "mean_dice_nonempty": float(
            nonempty["dice"].mean()
        ),
        "mean_iou_nonempty": float(
            nonempty["iou"].mean()
        ),
        "empty_target_accuracy": (
            empty_target_accuracy
        ),
        "samples": len(rows),
        "nonempty_targets": len(nonempty),
        "empty_targets": len(empty),
    }


def main() -> None:
    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    splits = pd.read_csv(
        SPLITS_PATH
    )

    validation = manifest.merge(
        splits[
            [
                "sample_id",
                "split",
            ]
        ],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )

    validation = (
        validation[
            validation["split"] == "val"
        ]
        .reset_index(drop=True)
    )

    if len(validation) != 459:
        raise ValueError(
            "Expected 459 validation samples, "
            f"found {len(validation)}"
        )

    configs = build_configs()

    print(
        f"Validation samples: {len(validation)}"
    )
    print(
        f"Configurations: {len(configs)}"
    )
    print()

    records = []

    total = len(validation)

    for sample_index, row in enumerate(
        validation.itertuples(index=False),
        start=1,
    ):
        image = load_image(
            DATASET_ROOT / row.image_path
        )

        target = load_target(
            DATASET_ROOT / row.mask_path
        )

        for config_id, config in enumerate(
            configs
        ):
            prediction = predict_mask(
                image,
                config,
            )

            metrics = (
                compute_segmentation_metrics(
                    prediction,
                    target,
                )
            )

            records.append(
                {
                    "config_id": config_id,
                    "sample_id": row.sample_id,
                    "label": row.label,
                    "blur_kernel": (
                        config.blur_kernel
                    ),
                    "block_size": (
                        config.block_size
                    ),
                    "c": config.c,
                    "invert": config.invert,
                    "morphology": (
                        config.morphology
                    ),
                    "morphology_kernel": (
                        config.morphology_kernel
                    ),
                    "dice": metrics.dice,
                    "iou": metrics.iou,
                    "precision": (
                        metrics.precision
                    ),
                    "recall": metrics.recall,
                    "empty_target": (
                        metrics.empty_target
                    ),
                    "empty_prediction": (
                        metrics.empty_prediction
                    ),
                    "predicted_fraction": float(
                        prediction.mean()
                    ),
                }
            )

        if (
            sample_index % 50 == 0
            or sample_index == total
        ):
            print(
                f"Evaluated "
                f"{sample_index}/{total}"
            )

    results = pd.DataFrame(records)

    summaries = []

    for config_id, rows in results.groupby(
        "config_id"
    ):
        first = rows.iloc[0]

        summary = {
            "config_id": int(config_id),
            "blur_kernel": int(
                first["blur_kernel"]
            ),
            "block_size": int(
                first["block_size"]
            ),
            "c": float(
                first["c"]
            ),
            "invert": bool(
                first["invert"]
            ),
            "morphology": (
                first["morphology"]
            ),
            "morphology_kernel": int(
                first["morphology_kernel"]
            ),
        }

        summary.update(
            summarize_config(rows)
        )

        summaries.append(summary)

    grid = pd.DataFrame(summaries)

    grid = grid.sort_values(
        [
            "mean_dice",
            "mean_iou",
            "empty_target_accuracy",
        ],
        ascending=False,
    ).reset_index(drop=True)

    best = grid.iloc[0]

    best_config_id = int(
        best["config_id"]
    )

    best_rows = (
        results[
            results["config_id"]
            == best_config_id
        ]
        .sort_values("sample_id")
        .reset_index(drop=True)
    )

    class_summary = (
        best_rows.groupby("label")
        .agg(
            samples=("sample_id", "count"),
            mean_dice=("dice", "mean"),
            median_dice=("dice", "median"),
            mean_iou=("iou", "mean"),
            median_iou=("iou", "median"),
            mean_precision=(
                "precision",
                "mean",
            ),
            mean_recall=(
                "recall",
                "mean",
            ),
            mean_predicted_fraction=(
                "predicted_fraction",
                "mean",
            ),
            empty_targets=(
                "empty_target",
                "sum",
            ),
            empty_predictions=(
                "empty_prediction",
                "sum",
            ),
        )
        .reset_index()
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    grid.to_csv(
        GRID_PATH,
        index=False,
    )

    best_rows.to_csv(
        PER_IMAGE_PATH,
        index=False,
    )

    class_summary.to_csv(
        CLASS_SUMMARY_PATH,
        index=False,
    )

    best_config = {
        "selection_split": "val",
        "selection_metric": "mean_dice",
        "config_id": best_config_id,
        "blur_kernel": int(
            best["blur_kernel"]
        ),
        "block_size": int(
            best["block_size"]
        ),
        "c": float(
            best["c"]
        ),
        "invert": bool(
            best["invert"]
        ),
        "morphology": (
            best["morphology"]
        ),
        "morphology_kernel": int(
            best["morphology_kernel"]
        ),
        "mean_dice": float(
            best["mean_dice"]
        ),
        "mean_iou": float(
            best["mean_iou"]
        ),
        "mean_dice_nonempty": float(
            best["mean_dice_nonempty"]
        ),
        "mean_iou_nonempty": float(
            best["mean_iou_nonempty"]
        ),
        "empty_target_accuracy": float(
            best["empty_target_accuracy"]
        ),
    }

    BEST_CONFIG_PATH.write_text(
        json.dumps(
            best_config,
            indent=2,
        )
        + "\n"
    )

    print()
    print("TOP 10 CONFIGURATIONS")
    print("=====================")

    print(
        grid.head(10).to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("BEST CONFIGURATION")
    print("==================")

    print(
        json.dumps(
            best_config,
            indent=2,
        )
    )

    print()
    print("BEST CONFIGURATION BY CLASS")
    print("===========================")

    print(
        class_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("COMPARISON TO OTSU")
    print("==================")
    print("Otsu validation mean Dice: 0.455618")
    print(
        "Adaptive validation mean Dice: "
        f"{best['mean_dice']:.6f}"
    )
    print(
        "Difference: "
        f"{best['mean_dice'] - 0.45561829029080647:+.6f}"
    )

    print()
    print("TEST SET WAS NOT EVALUATED.")
    print()
    print(f"Wrote: {GRID_PATH}")
    print(f"Wrote: {PER_IMAGE_PATH}")
    print(f"Wrote: {CLASS_SUMMARY_PATH}")
    print(f"Wrote: {BEST_CONFIG_PATH}")


if __name__ == "__main__":
    main()
