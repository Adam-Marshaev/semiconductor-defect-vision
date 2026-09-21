from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SegmentationTransformConfig:
    image_mean: float
    image_std: float
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5


class SegmentationTransform:
    """Apply paired spatial transforms and image normalization."""

    def __init__(
        self,
        config: SegmentationTransformConfig,
        training: bool,
    ) -> None:
        if config.image_std <= 0:
            raise ValueError(
                "image_std must be positive"
            )

        for name, probability in (
            (
                "horizontal_flip_probability",
                config.horizontal_flip_probability,
            ),
            (
                "vertical_flip_probability",
                config.vertical_flip_probability,
            ),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1"
                )

        self.config = config
        self.training = training

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image.shape != mask.shape:
            raise ValueError(
                "Image and mask must have the same shape: "
                f"{tuple(image.shape)} != {tuple(mask.shape)}"
            )

        if image.ndim != 3:
            raise ValueError(
                "Expected tensors with shape [C, H, W], "
                f"got {tuple(image.shape)}"
            )

        if self.training:
            if (
                torch.rand(1).item()
                < self.config.horizontal_flip_probability
            ):
                image = torch.flip(
                    image,
                    dims=(-1,),
                )
                mask = torch.flip(
                    mask,
                    dims=(-1,),
                )

            if (
                torch.rand(1).item()
                < self.config.vertical_flip_probability
            ):
                image = torch.flip(
                    image,
                    dims=(-2,),
                )
                mask = torch.flip(
                    mask,
                    dims=(-2,),
                )

        image = (
            image - self.config.image_mean
        ) / self.config.image_std

        return image, mask
