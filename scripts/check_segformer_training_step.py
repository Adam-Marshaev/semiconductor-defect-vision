from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data.carinthia_dataset import (
    CarinthiaSegmentationDataset,
)
from src.models.segformer import (
    SegFormerBinarySegmenter,
)
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

BATCH_SIZE = 8


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this smoke test"
        )

    device = torch.device("cuda")

    dataset = CarinthiaSegmentationDataset(
        dataset_root=DATASET_ROOT,
        manifest_path=MANIFEST_PATH,
        splits_path=SPLITS_PATH,
        split="train",
        transform=None,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    batch = next(iter(loader))

    images = batch["image"].to(
        device,
        non_blocking=True,
    )

    masks = batch["mask"].to(
        device,
        non_blocking=True,
    )

    torch.cuda.reset_peak_memory_stats(
        device
    )

    model = SegFormerBinarySegmenter().to(
        device
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    criterion = BCEDiceLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    model.train()

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

    torch.cuda.synchronize()

    peak_vram = (
        torch.cuda.max_memory_allocated(
            device
        )
        / (1024**3)
    )

    print("SEGFORMER TRAINING STEP")
    print("=======================")
    print(
        "device:",
        torch.cuda.get_device_name(device),
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
        f"{peak_vram:.3f} GiB",
    )

    print()
    print(
        "Forward/backward/optimizer step passed."
    )


if __name__ == "__main__":
    main()
