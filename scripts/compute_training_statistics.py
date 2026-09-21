import json
from pathlib import Path

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

OUTPUT_PATH = Path(
    "data/processed/training_statistics.json"
)

BATCH_SIZE = 16


def main() -> None:
    dataset = CarinthiaSegmentationDataset(
        dataset_root=DATASET_ROOT,
        manifest_path=MANIFEST_PATH,
        splits_path=SPLITS_PATH,
        split="train",
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    pixel_sum = 0.0
    pixel_squared_sum = 0.0
    pixel_count = 0

    foreground_pixels = 0.0
    mask_pixels = 0

    total_batches = len(loader)

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):
        images = batch["image"].double()
        masks = batch["mask"].double()

        pixel_sum += float(
            images.sum()
        )

        pixel_squared_sum += float(
            (images**2).sum()
        )

        pixel_count += images.numel()

        foreground_pixels += float(
            masks.sum()
        )

        mask_pixels += masks.numel()

        if (
            batch_index % 50 == 0
            or batch_index == total_batches
        ):
            print(
                f"Processed batch "
                f"{batch_index}/{total_batches}"
            )

    mean = pixel_sum / pixel_count

    variance = (
        pixel_squared_sum / pixel_count
        - mean**2
    )

    std = variance**0.5

    foreground_fraction = (
        foreground_pixels / mask_pixels
    )

    statistics = {
        "split": "train",
        "samples": len(dataset),
        "pixel_count": pixel_count,
        "image_mean": mean,
        "image_std": std,
        "foreground_fraction": (
            foreground_fraction
        ),
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            statistics,
            indent=2,
        )
        + "\n"
    )

    print()
    print("TRAINING STATISTICS")
    print("===================")
    print(
        f"samples: {len(dataset)}"
    )
    print(
        f"pixels: {pixel_count}"
    )
    print(
        f"image mean: {mean:.8f}"
    )
    print(
        f"image std: {std:.8f}"
    )
    print(
        "foreground fraction: "
        f"{foreground_fraction:.8f}"
    )
    print()
    print(f"Wrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
