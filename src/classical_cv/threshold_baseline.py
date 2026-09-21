from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ThresholdBaselineConfig:
    blur_kernel: int = 5
    invert: bool = False
    morphology_kernel: int = 3
    morphology_iterations: int = 1


def predict_mask(
    image: np.ndarray,
    config: ThresholdBaselineConfig,
) -> np.ndarray:
    """Generate a binary defect mask using an Otsu threshold baseline."""

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

    blurred = cv2.GaussianBlur(
        image,
        (
            config.blur_kernel,
            config.blur_kernel,
        ),
        0,
    )

    threshold_type = cv2.THRESH_BINARY

    if config.invert:
        threshold_type = cv2.THRESH_BINARY_INV

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        threshold_type | cv2.THRESH_OTSU,
    )

    if config.morphology_kernel > 0:
        kernel = np.ones(
            (
                config.morphology_kernel,
                config.morphology_kernel,
            ),
            dtype=np.uint8,
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel,
            iterations=config.morphology_iterations,
        )

    return binary > 0
