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
    "segmentation_threshold_sweep.csv"
)

BATCH_SIZE = 8

THRESHOLDS = torch.arange(
    0.10,
    0.91,
    0.05,
)


def create_accumulator(
    threshold_count: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "sum_dice": torch.zeros(
            threshold_count,
            device=device,
        ),
        "sum_iou": torch.zeros(
            threshold_count,
            device=device,
        ),
        "sum_nonempty_dice": torch.zeros(
            threshold_count,
            device=device,
        ),
        "sum_precision": torch.zeros(
            threshold_count,
            device=device,
        ),
        "precision_count": torch.zeros(
            threshold_count,
            device=device,
        ),
        "sum_recall": torch.zeros(
            threshold_count,
            device=device,
        ),
        "recall_count": torch.zeros(
            threshold_count,
            device=device,
        ),
        "empty_correct": torch.zeros(
            threshold_count,
            device=device,
        ),
        "sample_count": torch.zeros(
            threshold_count,
            device=device,
        ),
        "nonempty_count": torch.zeros(
            threshold_count,
            device=device,
        ),
        "empty_count": torch.zeros(
            threshold_count,
            device=device,
        ),
    }


def update_accumulator(
    accumulator: dict[str, torch.Tensor],
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    thresholds: torch.Tensor,
) -> None:
    predictions = (
        probabilities.unsqueeze(0)
        >= thresholds.view(
            -1,
            1,
            1,
            1,
            1,
        )
    )

    targets_bool = (
        targets >= 0.5
    ).unsqueeze(0)

    reduction_dims = (
        2,
        3,
        4,
    )

    true_positive = (
        predictions
        & targets_bool
    ).sum(
        dim=reduction_dims
    ).float()

    false_positive = (
        predictions
        & ~targets_bool
    ).sum(
        dim=reduction_dims
    ).float()

    false_negative = (
        ~predictions
        & targets_bool
    ).sum(
        dim=reduction_dims
    ).float()

    target_pixels = (
        true_positive
        + false_negative
    )

    prediction_pixels = (
        true_positive
        + false_positive
    )

    dice_denominator = (
        2 * true_positive
        + false_positive
        + false_negative
    )

    dice = torch.where(
        dice_denominator > 0,
        (
            2 * true_positive
            / dice_denominator
        ),
        torch.ones_like(
            dice_denominator
        ),
    )

    iou_denominator = (
        true_positive
        + false_positive
        + false_negative
    )

    iou = torch.where(
        iou_denominator > 0,
        (
            true_positive
            / iou_denominator
        ),
        torch.ones_like(
            iou_denominator
        ),
    )

    precision_defined = (
        prediction_pixels > 0
    )

    recall_defined = (
        target_pixels > 0
    )

    precision = torch.where(
        precision_defined,
        true_positive
        / prediction_pixels.clamp_min(1),
        torch.zeros_like(
            true_positive
        ),
    )

    recall = torch.where(
        recall_defined,
        true_positive
        / target_pixels.clamp_min(1),
        torch.zeros_like(
            true_positive
        ),
    )

    nonempty = target_pixels > 0
    empty = ~nonempty

    empty_prediction = (
        prediction_pixels == 0
    )

    accumulator[
        "sum_dice"
    ] += dice.sum(dim=1)

    accumulator[
        "sum_iou"
    ] += iou.sum(dim=1)

    accumulator[
        "sum_nonempty_dice"
    ] += (
        dice
        * nonempty
    ).sum(dim=1)

    accumulator[
        "sum_precision"
    ] += (
        precision
        * precision_defined
    ).sum(dim=1)

    accumulator[
        "precision_count"
    ] += precision_defined.sum(
        dim=1
    )

    accumulator[
        "sum_recall"
    ] += (
        recall
        * recall_defined
    ).sum(dim=1)

    accumulator[
        "recall_count"
    ] += recall_defined.sum(
        dim=1
    )

    accumulator[
        "empty_correct"
    ] += (
        empty
        & empty_prediction
    ).sum(dim=1)

    batch_size = probabilities.shape[0]

    accumulator[
        "sample_count"
    ] += batch_size

    accumulator[
        "nonempty_count"
    ] += nonempty.sum(
        dim=1
    )

    accumulator[
        "empty_count"
    ] += empty.sum(
        dim=1
    )


def accumulator_to_rows(
    model_name: str,
    accumulator: dict[str, torch.Tensor],
    thresholds: torch.Tensor,
) -> list[dict[str, float | str]]:
    rows = []

    for index, threshold in enumerate(
        thresholds.tolist()
    ):
        sample_count = accumulator[
            "sample_count"
        ][index]

        nonempty_count = accumulator[
            "nonempty_count"
        ][index]

        empty_count = accumulator[
            "empty_count"
        ][index]

        rows.append(
            {
                "model": model_name,
                "threshold": threshold,
                "mean_dice": (
                    accumulator[
                        "sum_dice"
                    ][index]
                    / sample_count
                ).item(),
                "mean_iou": (
                    accumulator[
                        "sum_iou"
                    ][index]
                    / sample_count
                ).item(),
                "nonempty_mean_dice": (
                    accumulator[
                        "sum_nonempty_dice"
                    ][index]
                    / nonempty_count
                ).item(),
                "mean_precision": (
                    accumulator[
                        "sum_precision"
                    ][index]
                    / accumulator[
                        "precision_count"
                    ][index]
                ).item(),
                "mean_recall": (
                    accumulator[
                        "sum_recall"
                    ][index]
                    / accumulator[
                        "recall_count"
                    ][index]
                ).item(),
                "empty_accuracy": (
                    accumulator[
                        "empty_correct"
                    ][index]
                    / empty_count
                ).item(),
            }
        )

    return rows


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required"
        )

    device = torch.device(
        "cuda"
    )

    statistics = json.loads(
        STATS_PATH.read_text()
    )

    unet_transform = SegmentationTransform(
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

    unet_dataset = (
        CarinthiaSegmentationDataset(
            dataset_root=DATASET_ROOT,
            manifest_path=MANIFEST_PATH,
            splits_path=SPLITS_PATH,
            split="val",
            transform=unet_transform,
        )
    )

    segformer_dataset = (
        CarinthiaSegmentationDataset(
            dataset_root=DATASET_ROOT,
            manifest_path=MANIFEST_PATH,
            splits_path=SPLITS_PATH,
            split="val",
            transform=None,
        )
    )

    unet_loader = DataLoader(
        unet_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    segformer_loader = DataLoader(
        segformer_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    unet_checkpoint = torch.load(
        UNET_CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    unet = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=32,
    ).to(device)

    unet.load_state_dict(
        unet_checkpoint[
            "model_state_dict"
        ]
    )
    unet.eval()

    segformer_checkpoint = torch.load(
        SEGFORMER_CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    segformer = SegFormerBinarySegmenter(
        model_name=segformer_checkpoint.get(
            "model_name",
            "nvidia/mit-b0",
        )
    ).to(device)

    segformer.load_state_dict(
        segformer_checkpoint[
            "model_state_dict"
        ]
    )
    segformer.eval()

    thresholds = THRESHOLDS.to(
        device
    )

    unet_accumulator = (
        create_accumulator(
            len(THRESHOLDS),
            device,
        )
    )

    segformer_accumulator = (
        create_accumulator(
            len(THRESHOLDS),
            device,
        )
    )

    with torch.no_grad():
        for (
            unet_batch,
            segformer_batch,
        ) in zip(
            unet_loader,
            segformer_loader,
            strict=True,
        ):
            if (
                unet_batch["sample_id"]
                != segformer_batch[
                    "sample_id"
                ]
            ):
                raise RuntimeError(
                    "Validation loaders are "
                    "not aligned"
                )

            targets = (
                segformer_batch["mask"]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            unet_images = (
                unet_batch["image"]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            segformer_images = (
                segformer_batch["image"]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            unet_probabilities = (
                torch.sigmoid(
                    unet(
                        unet_images
                    )
                )
            )

            segformer_probabilities = (
                torch.sigmoid(
                    segformer(
                        segformer_images
                    )
                )
            )

            update_accumulator(
                unet_accumulator,
                unet_probabilities,
                targets,
                thresholds,
            )

            update_accumulator(
                segformer_accumulator,
                segformer_probabilities,
                targets,
                thresholds,
            )

    rows = accumulator_to_rows(
        "unet",
        unet_accumulator,
        thresholds,
    )

    rows.extend(
        accumulator_to_rows(
            "segformer",
            segformer_accumulator,
            thresholds,
        )
    )

    results = pd.DataFrame(
        rows
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "VALIDATION THRESHOLD SWEEP"
    )
    print(
        "=========================="
    )

    for model_name in [
        "unet",
        "segformer",
    ]:
        model_results = (
            results[
                results["model"]
                == model_name
            ]
            .sort_values(
                "threshold"
            )
        )

        print()
        print(
            model_name.upper()
        )
        print(
            model_results.to_string(
                index=False,
                float_format=lambda x: (
                    f"{x:.6f}"
                ),
            )
        )

        best = model_results.loc[
            model_results[
                "mean_dice"
            ].idxmax()
        ]

        print()
        print(
            "best threshold:",
            f"{best['threshold']:.2f}",
        )
        print(
            "best mean Dice:",
            f"{best['mean_dice']:.6f}",
        )


if __name__ == "__main__":
    main()
