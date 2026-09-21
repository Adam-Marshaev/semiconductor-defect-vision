from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

SplitName = Literal["train", "val", "test"]

PairedTransform = Callable[
    [torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]


class CarinthiaSegmentationDataset(Dataset):
    """PyTorch dataset for Carinthia-S semantic segmentation."""

    def __init__(
        self,
        dataset_root: Path,
        manifest_path: Path,
        splits_path: Path,
        split: SplitName,
        mask_threshold: int = 128,
        transform: PairedTransform | None = None,
    ) -> None:
        if split not in {
            "train",
            "val",
            "test",
        }:
            raise ValueError(
                f"Unknown split: {split}"
            )

        self.dataset_root = Path(
            dataset_root
        )
        self.mask_threshold = mask_threshold
        self.transform = transform

        manifest = pd.read_csv(
            manifest_path
        )

        splits = pd.read_csv(
            splits_path
        )

        required_manifest = {
            "sample_id",
            "label",
            "image_path",
            "mask_path",
        }

        missing_manifest = (
            required_manifest
            - set(manifest.columns)
        )

        if missing_manifest:
            raise ValueError(
                "Manifest missing columns: "
                f"{sorted(missing_manifest)}"
            )

        required_splits = {
            "sample_id",
            "split",
        }

        missing_splits = (
            required_splits
            - set(splits.columns)
        )

        if missing_splits:
            raise ValueError(
                "Splits file missing columns: "
                f"{sorted(missing_splits)}"
            )

        merged = manifest.merge(
            splits[
                [
                    "sample_id",
                    "split",
                ]
            ],
            on="sample_id",
            how="inner",
            validate="one_to_one",
        )

        self.samples = (
            merged[
                merged["split"] == split
            ]
            .reset_index(drop=True)
        )

        if self.samples.empty:
            raise ValueError(
                f"No samples found for split: {split}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor | str]:
        row = self.samples.iloc[index]

        image_path = (
            self.dataset_root
            / row["image_path"]
        )

        mask_path = (
            self.dataset_root
            / row["mask_path"]
        )

        if not image_path.exists():
            raise FileNotFoundError(
                image_path
            )

        if not mask_path.exists():
            raise FileNotFoundError(
                mask_path
            )

        with Image.open(
            image_path
        ) as image_file:
            image = np.array(
                image_file.convert("L"),
                dtype=np.float32,
                copy=True,
            )

        with Image.open(
            mask_path
        ) as mask_file:
            mask_gray = np.array(
                mask_file.convert("L"),
                dtype=np.uint8,
                copy=True,
            )

        if image.shape != mask_gray.shape:
            raise ValueError(
                "Image/mask shape mismatch: "
                f"{image.shape} != "
                f"{mask_gray.shape}"
            )

        image /= 255.0

        mask = (
            mask_gray
            >= self.mask_threshold
        ).astype(
            np.float32
        )

        image_tensor = (
            torch.from_numpy(image)
            .unsqueeze(0)
        )

        mask_tensor = (
            torch.from_numpy(mask)
            .unsqueeze(0)
        )

        if self.transform is not None:
            (
                image_tensor,
                mask_tensor,
            ) = self.transform(
                image_tensor,
                mask_tensor,
            )

        label_tensor = torch.tensor(
            int(row["label"]),
            dtype=torch.long,
        )

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": label_tensor,
            "sample_id": str(
                row["sample_id"]
            ),
        }
