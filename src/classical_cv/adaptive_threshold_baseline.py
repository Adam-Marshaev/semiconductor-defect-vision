from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

MorphologyOperation = Literal[
    "none",
    "open",
    "close",
]


@dataclass(frozen=True)
class AdaptiveThresholdConfig:
    blur_kernel: int = 3
    block_size: int = 31
    c: float = 5.0
    invert: bool = True
    morphology: MorphologyOperation = "none"
    morphology_kernel: int = 3


def predict_mask(
    image: np.ndarray,
    config: AdaptiveThresholdConfig,
) -> np.ndarray:
    """Segment an image using local adaptive Gaussian thresholding."""

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2-D grayscale image, got shape {image.shape}"
        )

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if config.blur_kernel <= 0 or config.blur_kernel % 2 == 0:
        raise ValueError(
            "blur_kernel must be a positive odd integer"
        )

    if config.block_size <= 1 or config.block_size % 2 == 0:
        raise ValueError(
            "block_size must be an odd integer greater than 1"
        )

    if config.morphology not in {
        "none",
        "open",
        "close",
    }:
        raise ValueError(
            f"Unknown morphology operation: {config.morphology}"
        )

    blurred = cv2.GaussianBlur(
        image,
        (
            config.blur_kernel,
            config.blur_kernel,
        ),
        0,
    )

    threshold_type = (
        cv2.THRESH_BINARY_INV
        if config.invert
        else cv2.THRESH_BINARY
    )

    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        threshold_type,
        config.block_size,
        config.c,
    )

    if config.morphology != "none":
        if config.morphology_kernel <= 0:
            raise ValueError(
                "morphology_kernel must be positive"
            )

        kernel = np.ones(
            (
                config.morphology_kernel,
                config.morphology_kernel,
            ),
            dtype=np.uint8,
        )

        operation = {
            "open": cv2.MORPH_OPEN,
            "close": cv2.MORPH_CLOSE,
        }[config.morphology]

        binary = cv2.morphologyEx(
            binary,
            operation,
            kernel,
        )

    return binary > 0
