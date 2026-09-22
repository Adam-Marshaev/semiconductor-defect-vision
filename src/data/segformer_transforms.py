from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SegFormerTransformConfig:
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5


class SegFormerTrainingTransform:
    """Paired spatial augmentation without image normalization."""

    def __init__(
        self,
        config: SegFormerTransformConfig,
    ) -> None:
        self.config = config

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            torch.rand(1).item()
            < self.config.horizontal_flip_probability
        ):
            image = torch.flip(
                image,
                dims=[2],
            )
            mask = torch.flip(
                mask,
                dims=[2],
            )

        if (
            torch.rand(1).item()
            < self.config.vertical_flip_probability
        ):
            image = torch.flip(
                image,
                dims=[1],
            )
            mask = torch.flip(
                mask,
                dims=[1],
            )

        return image, mask
