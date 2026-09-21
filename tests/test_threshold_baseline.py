import numpy as np
import pytest

from src.classical_cv.threshold_baseline import (
    ThresholdBaselineConfig,
    predict_mask,
)


def test_predict_mask_returns_boolean_array() -> None:
    image = np.array(
        [
            [0, 0, 255, 255],
            [0, 0, 255, 255],
            [0, 0, 255, 255],
            [0, 0, 255, 255],
        ],
        dtype=np.uint8,
    )

    config = ThresholdBaselineConfig(
        blur_kernel=1,
        morphology_kernel=0,
    )

    prediction = predict_mask(
        image,
        config,
    )

    assert prediction.shape == image.shape
    assert prediction.dtype == bool


def test_invert_changes_foreground_direction() -> None:
    image = np.array(
        [
            [0, 0, 255, 255],
            [0, 0, 255, 255],
        ],
        dtype=np.uint8,
    )

    normal = predict_mask(
        image,
        ThresholdBaselineConfig(
            blur_kernel=1,
            morphology_kernel=0,
            invert=False,
        ),
    )

    inverted = predict_mask(
        image,
        ThresholdBaselineConfig(
            blur_kernel=1,
            morphology_kernel=0,
            invert=True,
        ),
    )

    assert np.array_equal(
        normal,
        ~inverted,
    )


def test_even_blur_kernel_raises() -> None:
    image = np.zeros(
        (4, 4),
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
        match="positive odd",
    ):
        predict_mask(
            image,
            ThresholdBaselineConfig(
                blur_kernel=4
            ),
        )
