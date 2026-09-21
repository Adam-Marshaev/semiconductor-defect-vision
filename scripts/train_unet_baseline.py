import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.carinthia_dataset import (
    CarinthiaSegmentationDataset,
)
from src.data.segmentation_transforms import (
    SegmentationTransform,
    SegmentationTransformConfig,
)
from src.models.unet import UNet
from src.training.segmentation_loss import (
    BCEDiceLoss,
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
STATS_PATH = Path(
    "data/processed/training_statistics.json"
)

MODEL_PATH = Path(
    "models/unet_baseline_best.pt"
)

REPORT_DIR = Path(
    "reports/training"
)
HISTORY_PATH = (
    REPORT_DIR / "unet_baseline_history.csv"
)
BEST_SUMMARY_PATH = (
    REPORT_DIR / "unet_baseline_best.json"
)


SEED = 42

BATCH_SIZE = 8
NUM_WORKERS = 4

MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

BASE_CHANNELS = 32
PREDICTION_THRESHOLD = 0.5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def batch_segmentation_statistics(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities
        >= PREDICTION_THRESHOLD
    )

    targets_bool = targets >= 0.5

    dims = (
        1,
        2,
        3,
    )

    intersection = (
        predictions
        & targets_bool
    ).sum(
        dim=dims
    ).float()

    predicted_pixels = (
        predictions.sum(
            dim=dims
        ).float()
    )

    target_pixels = (
        targets_bool.sum(
            dim=dims
        ).float()
    )

    dice_denominator = (
        predicted_pixels
        + target_pixels
    )

    dice = torch.where(
        dice_denominator == 0,
        torch.ones_like(
            dice_denominator
        ),
        (
            2.0 * intersection
            / dice_denominator
        ),
    )

    union = (
        predicted_pixels
        + target_pixels
        - intersection
    )

    iou = torch.where(
        union == 0,
        torch.ones_like(
            union
        ),
        intersection / union,
    )

    empty_target = (
        target_pixels == 0
    )

    empty_prediction = (
        predicted_pixels == 0
    )

    nonempty_target = (
        ~empty_target
    )

    return {
        "dice": dice,
        "iou": iou,
        "empty_target": empty_target,
        "empty_prediction": (
            empty_prediction
        ),
        "nonempty_target": (
            nonempty_target
        ),
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_function: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        masks = batch["mask"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(images)

        loss = loss_function(
            logits,
            masks,
        )

        loss.backward()

        optimizer.step()

        batch_size = images.shape[0]

        total_loss += (
            loss.detach().item()
            * batch_size
        )

        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    dice_values = []
    iou_values = []

    nonempty_dice_values = []
    nonempty_iou_values = []

    empty_targets = 0
    correctly_empty = 0

    for batch in loader:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        masks = batch["mask"].to(
            device,
            non_blocking=True,
        )

        logits = model(images)

        loss = loss_function(
            logits,
            masks,
        )

        statistics = (
            batch_segmentation_statistics(
                logits,
                masks,
            )
        )

        batch_size = images.shape[0]

        total_loss += (
            loss.detach().item()
            * batch_size
        )

        total_samples += batch_size

        dice = statistics[
            "dice"
        ].cpu()

        iou = statistics[
            "iou"
        ].cpu()

        nonempty = statistics[
            "nonempty_target"
        ].cpu()

        empty = statistics[
            "empty_target"
        ].cpu()

        empty_prediction = statistics[
            "empty_prediction"
        ].cpu()

        dice_values.append(
            dice
        )

        iou_values.append(
            iou
        )

        if nonempty.any():
            nonempty_dice_values.append(
                dice[nonempty]
            )

            nonempty_iou_values.append(
                iou[nonempty]
            )

        empty_targets += int(
            empty.sum()
        )

        correctly_empty += int(
            (
                empty
                & empty_prediction
            ).sum()
        )

    dice_tensor = torch.cat(
        dice_values
    )

    iou_tensor = torch.cat(
        iou_values
    )

    nonempty_dice = torch.cat(
        nonempty_dice_values
    )

    nonempty_iou = torch.cat(
        nonempty_iou_values
    )

    if empty_targets > 0:
        empty_accuracy = (
            correctly_empty
            / empty_targets
        )
    else:
        empty_accuracy = float(
            "nan"
        )

    return {
        "loss": (
            total_loss
            / total_samples
        ),
        "mean_dice": float(
            dice_tensor.mean()
        ),
        "mean_iou": float(
            iou_tensor.mean()
        ),
        "mean_dice_nonempty": float(
            nonempty_dice.mean()
        ),
        "mean_iou_nonempty": float(
            nonempty_iou.mean()
        ),
        "empty_target_accuracy": (
            empty_accuracy
        ),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for training"
        )

    set_seed(SEED)

    statistics = json.loads(
        STATS_PATH.read_text()
    )

    transform_config = (
        SegmentationTransformConfig(
            image_mean=statistics[
                "image_mean"
            ],
            image_std=statistics[
                "image_std"
            ],
            horizontal_flip_probability=0.5,
            vertical_flip_probability=0.5,
        )
    )

    train_transform = (
        SegmentationTransform(
            transform_config,
            training=True,
        )
    )

    val_transform = (
        SegmentationTransform(
            transform_config,
            training=False,
        )
    )

    train_dataset = (
        CarinthiaSegmentationDataset(
            dataset_root=DATASET_ROOT,
            manifest_path=MANIFEST_PATH,
            splits_path=SPLITS_PATH,
            split="train",
            transform=train_transform,
        )
    )

    val_dataset = (
        CarinthiaSegmentationDataset(
            dataset_root=DATASET_ROOT,
            manifest_path=MANIFEST_PATH,
            splits_path=SPLITS_PATH,
            split="val",
            transform=val_transform,
        )
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    device = torch.device(
        "cuda"
    )

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=BASE_CHANNELS,
    ).to(device)

    loss_function = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("U-NET BASELINE TRAINING")
    print("======================")
    print(
        "device:",
        torch.cuda.get_device_name(0),
    )
    print(
        "parameters:",
        f"{parameter_count:,}",
    )
    print(
        "train samples:",
        len(train_dataset),
    )
    print(
        "validation samples:",
        len(val_dataset),
    )
    print(
        "batch size:",
        BATCH_SIZE,
    )
    print(
        "max epochs:",
        MAX_EPOCHS,
    )
    print(
        "test set loaded:",
        False,
    )
    print()

    history = []

    best_dice = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    torch.cuda.reset_peak_memory_stats()

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
        )

        validation = validate(
            model=model,
            loader=val_loader,
            loss_function=loss_function,
            device=device,
        )

        learning_rate = float(
            optimizer.param_groups[0][
                "lr"
            ]
        )

        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "val_loss": validation[
                "loss"
            ],
            "val_mean_dice": validation[
                "mean_dice"
            ],
            "val_mean_iou": validation[
                "mean_iou"
            ],
            "val_mean_dice_nonempty": (
                validation[
                    "mean_dice_nonempty"
                ]
            ),
            "val_mean_iou_nonempty": (
                validation[
                    "mean_iou_nonempty"
                ]
            ),
            "val_empty_target_accuracy": (
                validation[
                    "empty_target_accuracy"
                ]
            ),
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss "
            f"{train_loss:.4f} | "
            f"val loss "
            f"{validation['loss']:.4f} | "
            f"Dice "
            f"{validation['mean_dice']:.4f} | "
            f"IoU "
            f"{validation['mean_iou']:.4f} | "
            f"nonempty Dice "
            f"{validation['mean_dice_nonempty']:.4f} | "
            f"empty acc "
            f"{validation['empty_target_accuracy']:.3f} | "
            f"lr {learning_rate:.2e}"
        )

        scheduler.step(
            validation[
                "mean_dice"
            ]
        )

        if (
            validation["mean_dice"]
            > best_dice
        ):
            best_dice = validation[
                "mean_dice"
            ]

            best_epoch = epoch

            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "val_metrics": (
                        validation
                    ),
                    "config": {
                        "seed": SEED,
                        "batch_size": (
                            BATCH_SIZE
                        ),
                        "base_channels": (
                            BASE_CHANNELS
                        ),
                        "learning_rate": (
                            LEARNING_RATE
                        ),
                        "weight_decay": (
                            WEIGHT_DECAY
                        ),
                        "prediction_threshold": (
                            PREDICTION_THRESHOLD
                        ),
                        "image_mean": (
                            statistics[
                                "image_mean"
                            ]
                        ),
                        "image_std": (
                            statistics[
                                "image_std"
                            ]
                        ),
                    },
                },
                MODEL_PATH,
            )

            print(
                "  -> saved new best model"
            )

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print()
            print(
                "Early stopping: "
                f"no improvement for "
                f"{EARLY_STOPPING_PATIENCE} epochs."
            )
            break

    with HISTORY_PATH.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys(),
        )

        writer.writeheader()
        writer.writerows(
            history
        )

    best_row = next(
        row
        for row in history
        if row["epoch"]
        == best_epoch
    )

    peak_memory_gib = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    best_summary = {
        "model": "compact_unet",
        "selection_split": "val",
        "selection_metric": (
            "mean_per_image_dice"
        ),
        "best_epoch": best_epoch,
        "best_validation_metrics": (
            best_row
        ),
        "parameters": parameter_count,
        "batch_size": BATCH_SIZE,
        "peak_allocated_vram_gib": (
            peak_memory_gib
        ),
        "test_evaluated": False,
        "checkpoint": str(
            MODEL_PATH
        ),
    }

    BEST_SUMMARY_PATH.write_text(
        json.dumps(
            best_summary,
            indent=2,
        )
        + "\n"
    )

    print()
    print("TRAINING COMPLETE")
    print("=================")
    print(
        "best epoch:",
        best_epoch,
    )
    print(
        "best validation Dice:",
        f"{best_dice:.6f}",
    )
    print(
        "Otsu validation Dice:",
        "0.455618",
    )
    print(
        "improvement over Otsu:",
        f"{best_dice - 0.45561829029080647:+.6f}",
    )
    print(
        "peak allocated VRAM:",
        f"{peak_memory_gib:.3f} GiB",
    )
    print(
        "test evaluated:",
        False,
    )
    print()
    print(
        f"Checkpoint: {MODEL_PATH}"
    )
    print(
        f"History: {HISTORY_PATH}"
    )
    print(
        f"Summary: {BEST_SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()
