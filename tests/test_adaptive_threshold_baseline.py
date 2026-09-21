import numpy as np
import pytest

from src.classical_cv.adaptive_threshold_baseline import (
    AdaptiveThresholdConfig,
    predict_mask,
)


def test_prediction_shape_and_dtype() -> None:
    image = np.arange(
        81,
        dtype=np.uint8,
    ).reshape(9, 9)

    prediction = predict_mask(
        image,
        AdaptiveThresholdConfig(
            blur_kernel=1,
            block_size=3,
        ),
    )

    assert prediction.shape == image.shape
    assert prediction.dtype == bool


def test_even_block_size_raises() -> None:
    image = np.zeros(
        (9, 9),
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
        match="odd integer",
    ):
        predict_mask(
            image,
            AdaptiveThresholdConfig(
                block_size=4,
            ),
        )


def test_invalid_morphology_raises() -> None:
    image = np.zeros(
        (9, 9),
        dtype=np.uint8,
    )

    config = AdaptiveThresholdConfig(
        block_size=3,
        morphology="invalid",  # type: ignore[arg-type]
    )

    with pytest.raises(
        ValueError,
        match="Unknown morphology",
    ):
        predict_mask(
            image,
            config,
        )
