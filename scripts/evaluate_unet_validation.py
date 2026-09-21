import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.carinthia_dataset import (
    CarinthiaSegmentationDataset,
)
from src.data.segmentation_transforms import (
    SegmentationTransform,
    SegmentationTransformConfig,
)
from src.evaluation.segmentation_metrics import (
    compute_segmentation_metrics,
)
from src.models.unet import UNet

DATASET_ROOT = Path(
    "data/raw/carinthia-s/data"
)
MANIFEST_PATH = Path(
    "data/processed/manifest.csv"
)
SPLITS_PATH = Path(
    "data/processed/splits.csv"
)
STATS_PATH = Path(
    "data/processed/training_statistics.json"
)
CHECKPOINT_PATH = Path(
    "models/unet_baseline_best.pt"
)

OUTPUT_DIR = Path(
    "reports/training"
)

PER_IMAGE_PATH = (
    OUTPUT_DIR
    / "unet_validation_per_image.csv"
)

CLASS_SUMMARY_PATH = (
    OUTPUT_DIR
    / "unet_validation_by_class.csv"
)

PREDICTION_THRESHOLD = 0.5
BATCH_SIZE = 8


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required"
        )

    statistics = json.loads(
        STATS_PATH.read_text()
    )

    transform = SegmentationTransform(
        SegmentationTransformConfig(
            image_mean=statistics[
                "image_mean"
            ],
            image_std=statistics[
                "image_std"
            ],
        ),
        training=False,
    )

    dataset = CarinthiaSegmentationDataset(
        dataset_root=DATASET_ROOT,
        manifest_path=MANIFEST_PATH,
        splits_path=SPLITS_PATH,
        split="val",
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    device = torch.device(
        "cuda"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=32,
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    records = []

    with torch.no_grad():
        for batch in loader:
            images = batch[
                "image"
            ].to(
                device,
                non_blocking=True,
            )

            logits = model(
                images
            )

            predictions = (
                torch.sigmoid(logits)
                >= PREDICTION_THRESHOLD
            ).cpu().numpy()

            targets = (
                batch["mask"]
                >= 0.5
            ).numpy()

            labels = batch[
                "label"
            ].tolist()

            sample_ids = batch[
                "sample_id"
            ]

            for index in range(
                len(sample_ids)
            ):
                metrics = (
                    compute_segmentation_metrics(
                        predictions[index, 0],
                        targets[index, 0],
                    )
                )

                records.append(
                    {
                        "sample_id": (
                            sample_ids[index]
                        ),
                        "label": int(
                            labels[index]
                        ),
                        "dice": metrics.dice,
                        "iou": metrics.iou,
                        "precision": (
                            metrics.precision
                        ),
                        "recall": (
                            metrics.recall
                        ),
                        "empty_target": (
                            metrics.empty_target
                        ),
                        "empty_prediction": (
                            metrics.empty_prediction
                        ),
                        "true_positive": (
                            metrics.true_positive
                        ),
                        "false_positive": (
                            metrics.false_positive
                        ),
                        "false_negative": (
                            metrics.false_negative
                        ),
                    }
                )

    results = pd.DataFrame(
        records
    )

    class_summary = (
        results.groupby("label")
        .agg(
            samples=("sample_id", "count"),
            mean_dice=("dice", "mean"),
            median_dice=("dice", "median"),
            min_dice=("dice", "min"),
            mean_iou=("iou", "mean"),
            mean_precision=(
                "precision",
                "mean",
            ),
            mean_recall=(
                "recall",
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

    results.to_csv(
        PER_IMAGE_PATH,
        index=False,
    )

    class_summary.to_csv(
        CLASS_SUMMARY_PATH,
        index=False,
    )

    print("U-NET VALIDATION ANALYSIS")
    print("========================")
    print(
        "checkpoint epoch:",
        checkpoint["epoch"],
    )
    print(
        "samples:",
        len(results),
    )

    print()
    print("BY CLASS")
    print("========")

    print(
        class_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    nonempty = results[
        ~results[
            "empty_target"
        ]
    ]

    print()
    print("WORST 15 NON-EMPTY")
    print("==================")

    columns = [
        "sample_id",
        "label",
        "dice",
        "iou",
        "precision",
        "recall",
        "false_positive",
        "false_negative",
    ]

    print(
        nonempty.nsmallest(
            15,
            "dice",
        )[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("OVERALL")
    print("=======")
    print(
        "mean Dice:",
        f"{results['dice'].mean():.6f}",
    )
    print(
        "mean IoU:",
        f"{results['iou'].mean():.6f}",
    )
    print(
        "non-empty mean Dice:",
        f"{nonempty['dice'].mean():.6f}",
    )

    empty = results[
        results[
            "empty_target"
        ]
    ]

    print(
        "empty-target accuracy:",
        f"{empty['empty_prediction'].mean():.6f}",
    )

    print()
    print(
        "TEST SET WAS NOT EVALUATED."
    )
    print(
        f"Wrote: {PER_IMAGE_PATH}"
    )
    print(
        f"Wrote: {CLASS_SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()
