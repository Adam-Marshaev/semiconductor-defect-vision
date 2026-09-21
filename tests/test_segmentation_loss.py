import pytest
import torch

from src.training.segmentation_loss import (
    BCEDiceLoss,
)


def test_loss_is_finite() -> None:
    logits = torch.randn(
        2,
        1,
        8,
        8,
    )

    targets = torch.randint(
        0,
        2,
        (2, 1, 8, 8),
        dtype=torch.float32,
    )

    loss = BCEDiceLoss()(
        logits,
        targets,
    )

    assert torch.isfinite(loss)
    assert float(loss) >= 0.0


def test_loss_backpropagates() -> None:
    logits = torch.randn(
        1,
        1,
        8,
        8,
        requires_grad=True,
    )

    targets = torch.zeros_like(
        logits
    )

    loss = BCEDiceLoss()(
        logits,
        targets,
    )

    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(
        logits.grad
    ).all()


def test_shape_mismatch_raises() -> None:
    logits = torch.zeros(
        1,
        1,
        8,
        8,
    )

    targets = torch.zeros(
        1,
        1,
        4,
        4,
    )

    with pytest.raises(
        ValueError,
        match="identical shapes",
    ):
        BCEDiceLoss()(
            logits,
            targets,
        )
