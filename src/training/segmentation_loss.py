import torch
from torch import nn
from torch.nn import functional as F


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()

        if bce_weight < 0 or dice_weight < 0:
            raise ValueError(
                "Loss weights must be non-negative"
            )

        if bce_weight + dice_weight <= 0:
            raise ValueError(
                "At least one loss weight must be positive"
            )

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                "Logits and targets must have identical shapes: "
                f"{tuple(logits.shape)} != {tuple(targets.shape)}"
            )

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
        )

        probabilities = torch.sigmoid(
            logits
        )

        dims = tuple(
            range(
                1,
                probabilities.ndim,
            )
        )

        intersection = (
            probabilities
            * targets
        ).sum(dim=dims)

        denominator = (
            probabilities.sum(dim=dims)
            + targets.sum(dim=dims)
        )

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            denominator
            + self.smooth
        )

        dice_loss = 1.0 - dice.mean()

        return (
            self.bce_weight * bce
            + self.dice_weight * dice_loss
        )
