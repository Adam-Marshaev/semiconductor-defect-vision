from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.carinthia_dataset import (
    CarinthiaSegmentationDataset,
)


def build_fake_dataset(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    root = tmp_path / "dataset"

    image_dir = root / "images"
    mask_dir = root / "masks"

    image_dir.mkdir(
        parents=True
    )
    mask_dir.mkdir(
        parents=True
    )

    rows = []

    split_rows = []

    sample_specs = [
        ("sample_a", 1, "train"),
        ("sample_b", 3, "val"),
        ("sample_c", 6, "test"),
    ]

    for index, (
        sample_id,
        label,
        split,
    ) in enumerate(sample_specs):
        image = np.full(
            (8, 8),
            40 + index * 50,
            dtype=np.uint8,
        )

        mask = np.zeros(
            (8, 8),
            dtype=np.uint8,
        )

        mask[2:5, 3:6] = 255

        image_path = (
            image_dir
            / f"{sample_id}.png"
        )

        mask_path = (
            mask_dir
            / f"{sample_id}.png"
        )

        Image.fromarray(
            image
        ).save(image_path)

        Image.fromarray(
            mask
        ).save(mask_path)

        rows.append(
            {
                "sample_id": sample_id,
                "label": label,
                "image_path": (
                    f"images/{sample_id}.png"
                ),
                "mask_path": (
                    f"masks/{sample_id}.png"
                ),
            }
        )

        split_rows.append(
            {
                "sample_id": sample_id,
                "split": split,
            }
        )

    manifest_path = (
        tmp_path / "manifest.csv"
    )

    splits_path = (
        tmp_path / "splits.csv"
    )

    pd.DataFrame(
        rows
    ).to_csv(
        manifest_path,
        index=False,
    )

    pd.DataFrame(
        split_rows
    ).to_csv(
        splits_path,
        index=False,
    )

    return (
        root,
        manifest_path,
        splits_path,
    )


def test_dataset_filters_split(
    tmp_path: Path,
) -> None:
    (
        root,
        manifest_path,
        splits_path,
    ) = build_fake_dataset(
        tmp_path
    )

    dataset = (
        CarinthiaSegmentationDataset(
            dataset_root=root,
            manifest_path=manifest_path,
            splits_path=splits_path,
            split="val",
        )
    )

    assert len(dataset) == 1

    sample = dataset[0]

    assert (
        sample["sample_id"]
        == "sample_b"
    )


def test_tensor_shapes_and_dtypes(
    tmp_path: Path,
) -> None:
    (
        root,
        manifest_path,
        splits_path,
    ) = build_fake_dataset(
        tmp_path
    )

    dataset = (
        CarinthiaSegmentationDataset(
            dataset_root=root,
            manifest_path=manifest_path,
            splits_path=splits_path,
            split="train",
        )
    )

    sample = dataset[0]

    image = sample["image"]
    mask = sample["mask"]
    label = sample["label"]

    assert isinstance(
        image,
        torch.Tensor,
    )
    assert isinstance(
        mask,
        torch.Tensor,
    )
    assert isinstance(
        label,
        torch.Tensor,
    )

    assert image.shape == (
        1,
        8,
        8,
    )

    assert mask.shape == (
        1,
        8,
        8,
    )

    assert image.dtype == torch.float32
    assert mask.dtype == torch.float32
    assert label.dtype == torch.long

    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0

    assert set(
        torch.unique(mask).tolist()
    ) <= {0.0, 1.0}


def test_label_is_preserved(
    tmp_path: Path,
) -> None:
    (
        root,
        manifest_path,
        splits_path,
    ) = build_fake_dataset(
        tmp_path
    )

    dataset = (
        CarinthiaSegmentationDataset(
            dataset_root=root,
            manifest_path=manifest_path,
            splits_path=splits_path,
            split="train",
        )
    )

    assert int(
        dataset[0]["label"]
    ) == 1


def test_invalid_split_raises(
    tmp_path: Path,
) -> None:
    (
        root,
        manifest_path,
        splits_path,
    ) = build_fake_dataset(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match="Unknown split",
    ):
        CarinthiaSegmentationDataset(
            dataset_root=root,
            manifest_path=manifest_path,
            splits_path=splits_path,
            split="invalid",  # type: ignore[arg-type]
        )
