import torch

from src.models.unet import UNet


def test_unet_preserves_spatial_shape() -> None:
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=8,
    )

    x = torch.randn(
        2,
        1,
        64,
        64,
    )

    y = model(x)

    assert y.shape == (
        2,
        1,
        64,
        64,
    )


def test_unet_has_trainable_parameters() -> None:
    model = UNet(
        base_channels=8,
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert trainable > 0
