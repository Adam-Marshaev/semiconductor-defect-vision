import numpy as np
import pytest

from src.inference.segmentation_predictor import (
    prepare_grayscale_tensor,
)


def test_uint8_image_is_scaled_to_unit_range() -> None:
    image = np.array(
        [
            [0, 255],
            [128, 64],
        ],
        dtype=np.uint8,
    )

    tensor = prepare_grayscale_tensor(
        image
    )

    assert tensor.shape == (
        1,
        1,
        2,
        2,
    )

    assert tensor.dtype.is_floating_point

    assert tensor[0, 0, 0, 0].item() == 0.0
    assert tensor[0, 0, 0, 1].item() == 1.0


def test_float_image_is_preserved() -> None:
    image = np.array(
        [
            [0.25, 0.75],
        ],
        dtype=np.float32,
    )

    tensor = prepare_grayscale_tensor(
        image
    )

    assert tensor.shape == (
        1,
        1,
        1,
        2,
    )

    assert tensor[0, 0, 0, 0].item() == pytest.approx(
        0.25
    )


def test_rejects_multichannel_image() -> None:
    image = np.zeros(
        (
            8,
            8,
            3,
        ),
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
        match="2D grayscale",
    ):
        prepare_grayscale_tensor(
            image
        )


def test_rejects_invalid_float_range() -> None:
    image = np.array(
        [
            [0.0, 2.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match=r"\[0, 1\]",
    ):
        prepare_grayscale_tensor(
            image
        )
