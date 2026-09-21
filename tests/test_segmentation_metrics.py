import numpy as np
import pytest

from src.evaluation.segmentation_metrics import (
    compute_segmentation_metrics,
)


def test_perfect_nonempty_segmentation() -> None:
    target = np.array(
        [
            [0, 1],
            [0, 1],
        ],
        dtype=np.uint8,
    )

    prediction = target.copy()

    metrics = compute_segmentation_metrics(
        prediction,
        target,
    )

    assert metrics.true_positive == 2
    assert metrics.false_positive == 0
    assert metrics.false_negative == 0
    assert metrics.true_negative == 2

    assert metrics.iou == 1.0
    assert metrics.dice == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0


def test_both_empty_is_perfect_overlap() -> None:
    target = np.zeros(
        (2, 2),
        dtype=np.uint8,
    )

    prediction = target.copy()

    metrics = compute_segmentation_metrics(
        prediction,
        target,
    )

    assert metrics.iou == 1.0
    assert metrics.dice == 1.0

    assert np.isnan(
        metrics.precision
    )
    assert np.isnan(
        metrics.recall
    )

    assert metrics.empty_target
    assert metrics.empty_prediction


def test_false_positive_on_empty_target() -> None:
    target = np.zeros(
        (2, 2),
        dtype=np.uint8,
    )

    prediction = np.array(
        [
            [1, 0],
            [0, 0],
        ],
        dtype=np.uint8,
    )

    metrics = compute_segmentation_metrics(
        prediction,
        target,
    )

    assert metrics.true_positive == 0
    assert metrics.false_positive == 1

    assert metrics.iou == 0.0
    assert metrics.dice == 0.0

    assert metrics.precision == 0.0
    assert np.isnan(
        metrics.recall
    )


def test_complete_miss() -> None:
    target = np.array(
        [
            [1, 0],
            [0, 0],
        ],
        dtype=np.uint8,
    )

    prediction = np.zeros(
        (2, 2),
        dtype=np.uint8,
    )

    metrics = compute_segmentation_metrics(
        prediction,
        target,
    )

    assert metrics.false_negative == 1

    assert metrics.iou == 0.0
    assert metrics.dice == 0.0

    assert np.isnan(
        metrics.precision
    )
    assert metrics.recall == 0.0


def test_shape_mismatch_raises() -> None:
    prediction = np.zeros(
        (2, 2),
        dtype=np.uint8,
    )

    target = np.zeros(
        (3, 3),
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
        match="same shape",
    ):
        compute_segmentation_metrics(
            prediction,
            target,
        )
