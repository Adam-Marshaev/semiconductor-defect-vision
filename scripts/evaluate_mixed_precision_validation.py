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

UNET_CHECKPOINT_PATH = Path(
    "models/unet_baseline_best.pt"
)
SEGFORMER_CHECKPOINT_PATH = Path(
    "models/segformer_b0_best.pt"
)

OUTPUT_PATH = Path(
    "reports/training/"
    "mixed_precision_validation.csv"
)

BATCH_SIZE = 8
THRESHOLD = 0.5


def create_model(
    model_name: str,
    device: torch.device,
) -> torch.nn.Module:
    if model_name == "unet":
        checkpoint = torch.load(
            UNET_CHECKPOINT_PATH,
            map_location=device,
            weights_only=False,
        )

        model = UNet(
            in_channels=1,
            out_channels=1,
            base_channels=32,
        )

    elif model_name == "segformer":
        checkpoint = torch.load(
            SEGFORMER_CHECKPOINT_PATH,
            map_location=device,
            weights_only=False,
        )

        model = SegFormerBinarySegmenter(
            model_name=checkpoint.get(
                "model_name",
                "nvidia/mit-b0",
            )
        )

    else:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model


def evaluate(
    model_name: str,
    precision: str,
    device: torch.device,
) -> dict[str, object]:
    if model_name == "unet":
        import json

        from src.data.segmentation_transforms import (
            SegmentationTransform,
            SegmentationTransformConfig,
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
    else:
        transform = None

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

    model = create_model(
        model_name,
        device,
    )

    use_fp16 = precision == "fp16"

    dice_values = []
    iou_values = []
    nonempty_dice = []

    empty_correct = 0
    empty_count = 0

    current_predictions = []

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(
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
                enabled=use_fp16,
            ):
                logits = model(
                    images
                )

                predictions = (
                    torch.sigmoid(logits)
                    >= THRESHOLD
                )

            predictions_cpu = (
                predictions
                .cpu()
                .numpy()
            )

            for index in range(
                predictions_cpu.shape[0]
            ):
                prediction = (
                    predictions_cpu[
                        index,
                        0,
                    ]
                )

                target = targets[
                    index,
                    0,
                ]

                metrics = (
                    compute_segmentation_metrics(
                        prediction,
                        target,
                    )
                )

                dice_values.append(
                    metrics.dice
                )
                iou_values.append(
                    metrics.iou
                )

                if not metrics.empty_target:
                    nonempty_dice.append(
                        metrics.dice
                    )
                else:
                    empty_count += 1

                    if metrics.empty_prediction:
                        empty_correct += 1

                current_predictions.append(
                    prediction
                )

    del model
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "precision": precision,
        "mean_dice": sum(
            dice_values
        ) / len(dice_values),
        "mean_iou": sum(
            iou_values
        ) / len(iou_values),
        "nonempty_mean_dice": sum(
            nonempty_dice
        ) / len(nonempty_dice),
        "empty_accuracy": (
            empty_correct
            / empty_count
        ),
        "predictions": (
            current_predictions
        ),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required"
        )

    device = torch.device("cuda")

    rows = []

    for model_name in [
        "unet",
        "segformer",
    ]:
        fp32 = evaluate(
            model_name,
            "fp32",
            device,
        )

        fp16 = evaluate(
            model_name,
            "fp16",
            device,
        )

        disagreement_pixels = 0
        total_pixels = 0

        for (
            fp32_prediction,
            fp16_prediction,
        ) in zip(
            fp32["predictions"],
            fp16["predictions"],
            strict=True,
        ):
            disagreement_pixels += int(
                (
                    fp32_prediction
                    != fp16_prediction
                ).sum()
            )

            total_pixels += int(
                fp32_prediction.size
            )

        disagreement_fraction = (
            disagreement_pixels
            / total_pixels
        )

        for result in [
            fp32,
            fp16,
        ]:
            rows.append(
                {
                    "model": result[
                        "model"
                    ],
                    "precision": result[
                        "precision"
                    ],
                    "mean_dice": result[
                        "mean_dice"
                    ],
                    "mean_iou": result[
                        "mean_iou"
                    ],
                    "nonempty_mean_dice": (
                        result[
                            "nonempty_mean_dice"
                        ]
                    ),
                    "empty_accuracy": (
                        result[
                            "empty_accuracy"
                        ]
                    ),
                    "fp16_vs_fp32_"
                    "disagreement_fraction": (
                        disagreement_fraction
                        if result[
                            "precision"
                        ]
                        == "fp16"
                        else 0.0
                    ),
                }
            )

        print()
        print(
            model_name.upper()
        )
        print(
            "FP32 Dice:",
            f"{fp32['mean_dice']:.6f}",
        )
        print(
            "FP16 Dice:",
            f"{fp16['mean_dice']:.6f}",
        )
        print(
            "Dice delta:",
            f"{fp16['mean_dice'] - fp32['mean_dice']:+.8f}",
        )
        print(
            "prediction pixel disagreement:",
            f"{disagreement_fraction:.8%}",
        )

    output = pd.DataFrame(
        rows
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "MIXED PRECISION VALIDATION"
    )
    print(
        "=========================="
    )

    print(
        output.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.8f}"
            ),
        )
    )

    print()
    print(
        f"Wrote: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
