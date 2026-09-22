from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.carinthia_dataset import (
    CarinthiaSegmentationDataset,
)
from src.evaluation.segmentation_metrics import (
    compute_segmentation_metrics,
)
from src.models.segformer import (
    SegFormerBinarySegmenter,
)

DATASET_ROOT = Path(
    "data/raw/carinthia-s/data"
)
MANIFEST_PATH = Path(
    "data/processed/manifest.csv"
)
SPLITS_PATH = Path(
    "data/processed/splits.csv"
)

CHECKPOINT_PATH = Path(
    "models/segformer_b0_best.pt"
)
CONFIG_PATH = Path(
    "configs/segformer_b0_binary_config.json"
)

OUTPUT_DIR = Path(
    "reports/test"
)
PER_IMAGE_PATH = (
    OUTPUT_DIR
    / "segformer_final_test_per_image.csv"
)
BY_CLASS_PATH = (
    OUTPUT_DIR
    / "segformer_final_test_by_class.csv"
)

BATCH_SIZE = 8
THRESHOLD = 0.5


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required"
        )

    device = torch.device(
        "cuda"
    )

    dataset = CarinthiaSegmentationDataset(
        dataset_root=DATASET_ROOT,
        manifest_path=MANIFEST_PATH,
        splits_path=SPLITS_PATH,
        split="test",
        transform=None,
    )

    if len(dataset) != 459:
        raise RuntimeError(
            "Expected 459 test samples, "
            f"found {len(dataset)}"
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model = SegFormerBinarySegmenter(
        pretrained=False,
        config_path=CONFIG_PATH,
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    records = []

    with torch.inference_mode():
        for batch in loader:
            images = batch[
                "image"
            ].to(
                device,
                non_blocking=True,
            )

            targets = (
                batch["mask"]
                >= 0.5
            ).numpy()

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                logits = model(
                    images
                )

                predictions = (
                    torch.sigmoid(logits)
                    >= THRESHOLD
                )

            predictions = (
                predictions
                .cpu()
                .numpy()
            )

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
                        predictions[
                            index,
                            0,
                        ],
                        targets[
                            index,
                            0,
                        ],
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
            samples=(
                "sample_id",
                "count",
            ),
            mean_dice=(
                "dice",
                "mean",
            ),
            median_dice=(
                "dice",
                "median",
            ),
            min_dice=(
                "dice",
                "min",
            ),
            mean_iou=(
                "iou",
                "mean",
            ),
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
        BY_CLASS_PATH,
        index=False,
    )

    nonempty = results.loc[
        ~results[
            "empty_target"
        ]
    ]

    empty = results.loc[
        results[
            "empty_target"
        ]
    ]

    print(
        "FINAL SEGFORMER-B0 TEST"
    )
    print(
        "======================="
    )

    print(
        "checkpoint epoch:",
        checkpoint["epoch"],
    )
    print(
        "samples:",
        len(results),
    )
    print(
        "precision:",
        "FP16 autocast",
    )
    print(
        "threshold:",
        THRESHOLD,
    )
    print()

    print(
        "mean Dice:",
        f"{results['dice'].mean():.6f}",
    )
    print(
        "mean IoU:",
        f"{results['iou'].mean():.6f}",
    )
    print(
        "nonempty mean Dice:",
        f"{nonempty['dice'].mean():.6f}",
    )

    if len(empty) > 0:
        empty_accuracy = (
            empty[
                "empty_prediction"
            ].mean()
        )

        print(
            "empty accuracy:",
            f"{empty_accuracy:.6f}",
        )

    print()
    print(
        "BY CLASS"
    )
    print(
        "========"
    )

    print(
        class_summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    print()
    print(
        "WORST 10 NONEMPTY"
    )
    print(
        "================="
    )

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
        nonempty.sort_values(
            "dice"
        )
        .head(10)[columns]
        .to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    print()
    print(
        f"Wrote: {PER_IMAGE_PATH}"
    )
    print(
        f"Wrote: {BY_CLASS_PATH}"
    )


if __name__ == "__main__":
    main()
