from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SegmentationMetrics:
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    iou: float
    dice: float
    precision: float
    recall: float
    empty_target: bool
    empty_prediction: bool


def compute_segmentation_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> SegmentationMetrics:
    """Compute binary segmentation metrics for one image."""

    prediction = np.asarray(
        prediction,
        dtype=bool,
    )
    target = np.asarray(
        target,
        dtype=bool,
    )

    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target must have the same shape: "
            f"{prediction.shape} != {target.shape}"
        )

    true_positive = int(
        np.logical_and(
            prediction,
            target,
        ).sum()
    )

    false_positive = int(
        np.logical_and(
            prediction,
            ~target,
        ).sum()
    )

    false_negative = int(
        np.logical_and(
            ~prediction,
            target,
        ).sum()
    )

    true_negative = int(
        np.logical_and(
            ~prediction,
            ~target,
        ).sum()
    )

    union = (
        true_positive
        + false_positive
        + false_negative
    )

    if union == 0:
        iou = 1.0
    else:
        iou = true_positive / union

    dice_denominator = (
        2 * true_positive
        + false_positive
        + false_negative
    )

    if dice_denominator == 0:
        dice = 1.0
    else:
        dice = (
            2 * true_positive
            / dice_denominator
        )

    predicted_positive = (
        true_positive
        + false_positive
    )

    actual_positive = (
        true_positive
        + false_negative
    )

    if predicted_positive == 0:
        precision = float("nan")
    else:
        precision = (
            true_positive
            / predicted_positive
        )

    if actual_positive == 0:
        recall = float("nan")
    else:
        recall = (
            true_positive
            / actual_positive
        )

    return SegmentationMetrics(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
        iou=float(iou),
        dice=float(dice),
        precision=float(precision),
        recall=float(recall),
        empty_target=actual_positive == 0,
        empty_prediction=predicted_positive == 0,
    )
