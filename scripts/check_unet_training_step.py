import json
from pathlib import Path

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

BATCH_SIZE = 4


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this smoke test"
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
        training=True,
    )

    dataset = CarinthiaSegmentationDataset(
        dataset_root=DATASET_ROOT,
        manifest_path=MANIFEST_PATH,
        splits_path=SPLITS_PATH,
        split="train",
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    device = torch.device(
        "cuda"
    )

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=32,
    ).to(device)

    loss_function = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    batch = next(
        iter(loader)
    )

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

    torch.cuda.reset_peak_memory_stats()

    logits = model(images)

    loss = loss_function(
        logits,
        masks,
    )

    loss.backward()

    optimizer.step()

    torch.cuda.synchronize()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    peak_memory_gib = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    print("U-NET TRAINING STEP")
    print("===================")
    print(
        "device:",
        torch.cuda.get_device_name(0),
    )
    print(
        "input shape:",
        tuple(images.shape),
    )
    print(
        "output shape:",
        tuple(logits.shape),
    )
    print(
        "mask shape:",
        tuple(masks.shape),
    )
    print(
        "parameters:",
        f"{parameter_count:,}",
    )
    print(
        "loss:",
        f"{loss.detach().item():.6f}",
    )
    print(
        "peak allocated VRAM:",
        f"{peak_memory_gib:.3f} GiB",
    )

    assert logits.shape == masks.shape
    assert torch.isfinite(loss)

    print()
    print(
        "Forward/backward/optimizer "
        "step passed."
    )


if __name__ == "__main__":
    main()
