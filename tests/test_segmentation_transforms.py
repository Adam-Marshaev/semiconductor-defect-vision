import pytest
import torch

from src.data.segmentation_transforms import (
    SegmentationTransform,
    SegmentationTransformConfig,
)


def test_validation_normalizes_without_spatial_change() -> None:
    image = torch.tensor(
        [
            [
                [0.2, 0.4],
                [0.6, 0.8],
            ]
        ],
        dtype=torch.float32,
    )

    mask = torch.tensor(
        [
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )

    transform = SegmentationTransform(
        SegmentationTransformConfig(
            image_mean=0.5,
            image_std=0.1,
        ),
        training=False,
    )

    transformed_image, transformed_mask = transform(
        image,
        mask,
    )

    expected_image = (
        image - 0.5
    ) / 0.1

    assert torch.allclose(
        transformed_image,
        expected_image,
    )

    assert torch.equal(
        transformed_mask,
        mask,
    )


def test_horizontal_flip_is_paired() -> None:
    image = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ]
    )

    mask = image.clone()

    transform = SegmentationTransform(
        SegmentationTransformConfig(
            image_mean=0.0,
            image_std=1.0,
            horizontal_flip_probability=1.0,
            vertical_flip_probability=0.0,
        ),
        training=True,
    )

    transformed_image, transformed_mask = transform(
        image,
        mask,
    )

    expected = torch.tensor(
        [
            [
                [3.0, 2.0, 1.0],
                [6.0, 5.0, 4.0],
            ]
        ]
    )

    assert torch.equal(
        transformed_image,
        expected,
    )

    assert torch.equal(
        transformed_mask,
        expected,
    )


def test_vertical_flip_is_paired() -> None:
    image = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ]
    )

    mask = image.clone()

    transform = SegmentationTransform(
        SegmentationTransformConfig(
            image_mean=0.0,
            image_std=1.0,
            horizontal_flip_probability=0.0,
            vertical_flip_probability=1.0,
        ),
        training=True,
    )

    transformed_image, transformed_mask = transform(
        image,
        mask,
    )

    expected = torch.tensor(
        [
            [
                [3.0, 4.0],
                [1.0, 2.0],
            ]
        ]
    )

    assert torch.equal(
        transformed_image,
        expected,
    )

    assert torch.equal(
        transformed_mask,
        expected,
    )


def test_invalid_standard_deviation_raises() -> None:
    with pytest.raises(
        ValueError,
        match="image_std must be positive",
    ):
        SegmentationTransform(
            SegmentationTransformConfig(
                image_mean=0.5,
                image_std=0.0,
            ),
            training=True,
        )
