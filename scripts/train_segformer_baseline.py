import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data.carinthia_dataset import (
    CarinthiaSegmentationDataset,
)
from src.data.segformer_transforms import (
    SegFormerTrainingTransform,
    SegFormerTransformConfig,
)
from src.evaluation.segmentation_metrics import (
    compute_segmentation_metrics,
)
from src.models.segformer import (
    SegFormerBinarySegmenter,
)
from src.training.segmentation_loss import (
    BCEDiceLoss,
)

SEED = 42

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
HISTORY_PATH = Path(
    "reports/training/segformer_b0_history.csv"
)
SUMMARY_PATH = Path(
    "reports/training/segformer_b0_best.json"
)

MODEL_NAME = "nvidia/mit-b0"

BATCH_SIZE = 8
NUM_WORKERS = 4

MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

PREDICTION_THRESHOLD = 0.5

BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5

UNET_VALIDATION_DICE = 0.949786
OTSU_VALIDATION_DICE = 0.45561829029080647


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
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

        loss = criterion(
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


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    dice_scores = []
    iou_scores = []

    nonempty_dice_scores = []
    nonempty_iou_scores = []

    empty_correct = 0
    empty_total = 0

    with torch.no_grad():
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

            loss = criterion(
                logits,
                masks,
            )

            batch_size = images.shape[0]

            total_loss += (
                loss.detach().item()
                * batch_size
            )
            total_samples += batch_size

            predictions = (
                torch.sigmoid(logits)
                >= PREDICTION_THRESHOLD
            ).cpu().numpy()

            targets = (
                masks >= 0.5
            ).cpu().numpy()

            for index in range(batch_size):
                metrics = (
                    compute_segmentation_metrics(
                        predictions[index, 0],
                        targets[index, 0],
                    )
                )

                dice_scores.append(
                    metrics.dice
                )
                iou_scores.append(
                    metrics.iou
                )

                if metrics.empty_target:
                    empty_total += 1

                    if metrics.empty_prediction:
                        empty_correct += 1
                else:
                    nonempty_dice_scores.append(
                        metrics.dice
                    )
                    nonempty_iou_scores.append(
                        metrics.iou
                    )

    return {
        "loss": total_loss / total_samples,
        "dice": float(
            np.mean(dice_scores)
        ),
        "iou": float(
            np.mean(iou_scores)
        ),
        "nonempty_dice": float(
            np.mean(nonempty_dice_scores)
        ),
        "nonempty_iou": float(
            np.mean(nonempty_iou_scores)
        ),
        "empty_accuracy": (
            empty_correct / empty_total
            if empty_total
            else float("nan")
        ),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for E004"
        )

    set_seed(SEED)

    device = torch.device("cuda")

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_transform = (
        SegFormerTrainingTransform(
            SegFormerTransformConfig(
                horizontal_flip_probability=0.5,
                vertical_flip_probability=0.5,
            )
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

    validation_dataset = (
        CarinthiaSegmentationDataset(
            dataset_root=DATASET_ROOT,
            manifest_path=MANIFEST_PATH,
            splits_path=SPLITS_PATH,
            split="val",
            transform=None,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    model = SegFormerBinarySegmenter(
        model_name=MODEL_NAME,
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    criterion = BCEDiceLoss(
        bce_weight=BCE_WEIGHT,
        dice_weight=DICE_WEIGHT,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = []

    best_validation_dice = -1.0
    best_epoch = 0

    epochs_without_improvement = 0

    torch.cuda.reset_peak_memory_stats(
        device
    )

    print("SEGFORMER-B0 BASELINE TRAINING")
    print("==============================")
    print(
        "device:",
        torch.cuda.get_device_name(device),
    )
    print(
        "pretrained encoder:",
        MODEL_NAME,
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
        len(validation_dataset),
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
        "initial learning rate:",
        f"{LEARNING_RATE:.2e}",
    )
    print(
        "test set loaded:",
        False,
    )
    print()

    training_start = time.perf_counter()

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        epoch_start = time.perf_counter()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation = validate(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(
            validation["dice"]
        )

        current_lr = optimizer.param_groups[
            0
        ]["lr"]

        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": validation[
                "loss"
            ],
            "val_dice": validation[
                "dice"
            ],
            "val_iou": validation[
                "iou"
            ],
            "val_nonempty_dice": validation[
                "nonempty_dice"
            ],
            "val_nonempty_iou": validation[
                "nonempty_iou"
            ],
            "val_empty_accuracy": validation[
                "empty_accuracy"
            ],
            "learning_rate": current_lr,
            "epoch_seconds": epoch_seconds,
        }

        history.append(record)

        print(
            f"Epoch {epoch:02d}"
            f" | train loss {train_loss:.4f}"
            f" | val loss {validation['loss']:.4f}"
            f" | Dice {validation['dice']:.4f}"
            f" | IoU {validation['iou']:.4f}"
            f" | nonempty Dice "
            f"{validation['nonempty_dice']:.4f}"
            f" | empty acc "
            f"{validation['empty_accuracy']:.3f}"
            f" | lr {current_lr:.2e}"
            f" | {epoch_seconds:.1f}s"
        )

        if (
            validation["dice"]
            > best_validation_dice
        ):
            best_validation_dice = (
                validation["dice"]
            )
            best_epoch = epoch

            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_name": MODEL_NAME,
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "validation": validation,
                    "prediction_threshold": (
                        PREDICTION_THRESHOLD
                    ),
                    "parameter_count": (
                        parameter_count
                    ),
                },
                CHECKPOINT_PATH,
            )

            print(
                "  -> saved new best model"
            )

        else:
            epochs_without_improvement += 1

        pd.DataFrame(
            history
        ).to_csv(
            HISTORY_PATH,
            index=False,
        )

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print()
            print(
                "Early stopping triggered."
            )
            break

    total_seconds = (
        time.perf_counter()
        - training_start
    )

    peak_vram_gib = (
        torch.cuda.max_memory_allocated(
            device
        )
        / (1024**3)
    )

    summary = {
        "experiment": "E004",
        "model": "SegFormer-B0",
        "pretrained_encoder": MODEL_NAME,
        "parameters": parameter_count,
        "batch_size": BATCH_SIZE,
        "initial_learning_rate": (
            LEARNING_RATE
        ),
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_dice": (
            best_validation_dice
        ),
        "unet_validation_dice": (
            UNET_VALIDATION_DICE
        ),
        "otsu_validation_dice": (
            OTSU_VALIDATION_DICE
        ),
        "difference_vs_unet": (
            best_validation_dice
            - UNET_VALIDATION_DICE
        ),
        "difference_vs_otsu": (
            best_validation_dice
            - OTSU_VALIDATION_DICE
        ),
        "peak_allocated_vram_gib": (
            peak_vram_gib
        ),
        "total_training_seconds": (
            total_seconds
        ),
        "prediction_threshold": (
            PREDICTION_THRESHOLD
        ),
        "test_evaluated": False,
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
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
        f"{best_validation_dice:.6f}",
    )
    print(
        "U-Net validation Dice:",
        f"{UNET_VALIDATION_DICE:.6f}",
    )
    print(
        "difference vs U-Net:",
        f"{best_validation_dice - UNET_VALIDATION_DICE:+.6f}",
    )
    print(
        "Otsu validation Dice:",
        f"{OTSU_VALIDATION_DICE:.6f}",
    )
    print(
        "difference vs Otsu:",
        f"{best_validation_dice - OTSU_VALIDATION_DICE:+.6f}",
    )
    print(
        "peak allocated VRAM:",
        f"{peak_vram_gib:.3f} GiB",
    )
    print(
        "total training time:",
        f"{total_seconds / 60:.1f} min",
    )
    print(
        "test evaluated:",
        False,
    )

    print()
    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )
    print(
        f"History: {HISTORY_PATH}"
    )
    print(
        f"Summary: {SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()
