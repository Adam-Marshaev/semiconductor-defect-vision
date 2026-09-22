import torch

from src.data.segformer_transforms import (
    SegFormerTrainingTransform,
    SegFormerTransformConfig,
)


def test_horizontal_flip_is_paired() -> None:
    image = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]]]
    )
    mask = torch.tensor(
        [[[0.0, 1.0], [1.0, 0.0]]]
    )

    transform = SegFormerTrainingTransform(
        SegFormerTransformConfig(
            horizontal_flip_probability=1.0,
            vertical_flip_probability=0.0,
        )
    )

    transformed_image, transformed_mask = transform(
        image,
        mask,
    )

    assert torch.equal(
        transformed_image,
        torch.flip(image, dims=[2]),
    )

    assert torch.equal(
        transformed_mask,
        torch.flip(mask, dims=[2]),
    )


def test_vertical_flip_is_paired() -> None:
    image = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]]]
    )
    mask = torch.tensor(
        [[[0.0, 1.0], [1.0, 0.0]]]
    )

    transform = SegFormerTrainingTransform(
        SegFormerTransformConfig(
            horizontal_flip_probability=0.0,
            vertical_flip_probability=1.0,
        )
    )

    transformed_image, transformed_mask = transform(
        image,
        mask,
    )

    assert torch.equal(
        transformed_image,
        torch.flip(image, dims=[1]),
    )

    assert torch.equal(
        transformed_mask,
        torch.flip(mask, dims=[1]),
    )


def test_no_flip_preserves_values() -> None:
    image = torch.rand(1, 8, 8)
    mask = torch.randint(
        0,
        2,
        (1, 8, 8),
    ).float()

    transform = SegFormerTrainingTransform(
        SegFormerTransformConfig(
            horizontal_flip_probability=0.0,
            vertical_flip_probability=0.0,
        )
    )

    transformed_image, transformed_mask = transform(
        image,
        mask,
    )

    assert torch.equal(
        transformed_image,
        image,
    )

    assert torch.equal(
        transformed_mask,
        mask,
    )
