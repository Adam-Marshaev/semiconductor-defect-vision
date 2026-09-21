from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.carinthia_dataset import (
    CarinthiaSegmentationDataset,
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


def main() -> None:
    datasets = {
        split: CarinthiaSegmentationDataset(
            dataset_root=DATASET_ROOT,
            manifest_path=MANIFEST_PATH,
            splits_path=SPLITS_PATH,
            split=split,
        )
        for split in (
            "train",
            "val",
            "test",
        )
    }

    print("DATASET SIZES")
    print("=============")

    for split, dataset in (
        datasets.items()
    ):
        print(
            f"{split:5}: "
            f"{len(dataset)}"
        )

    train_loader = DataLoader(
        datasets["train"],
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    batch = next(
        iter(train_loader)
    )

    images = batch["image"]
    masks = batch["mask"]
    labels = batch["label"]

    print()
    print("FIRST TRAIN BATCH")
    print("=================")

    print(
        "image shape:",
        tuple(images.shape),
    )

    print(
        "image dtype:",
        images.dtype,
    )

    print(
        "image min/max:",
        float(images.min()),
        float(images.max()),
    )

    print(
        "mask shape:",
        tuple(masks.shape),
    )

    print(
        "mask dtype:",
        masks.dtype,
    )

    print(
        "mask unique:",
        sorted(
            torch.unique(
                masks
            ).tolist()
        ),
    )

    print(
        "labels:",
        labels.tolist(),
    )

    print(
        "sample IDs:",
        batch["sample_id"],
    )

    assert images.shape == (
        8,
        1,
        480,
        480,
    )

    assert masks.shape == (
        8,
        1,
        480,
        480,
    )

    assert images.dtype == (
        torch.float32
    )

    assert masks.dtype == (
        torch.float32
    )

    assert set(
        torch.unique(
            masks
        ).tolist()
    ) <= {
        0.0,
        1.0,
    }

    if torch.cuda.is_available():
        device = torch.device(
            "cuda"
        )

        images_gpu = images.to(
            device,
            non_blocking=True,
        )

        masks_gpu = masks.to(
            device,
            non_blocking=True,
        )

        print()
        print("GPU TRANSFER")
        print("============")

        print(
            "device:",
            images_gpu.device,
        )

        print(
            "image batch MB:",
            (
                images_gpu.numel()
                * images_gpu.element_size()
                / 1024**2
            ),
        )

        print(
            "mask batch MB:",
            (
                masks_gpu.numel()
                * masks_gpu.element_size()
                / 1024**2
            ),
        )

    print()
    print(
        "PyTorch data pipeline "
        "smoke test passed."
    )


if __name__ == "__main__":
    main()
