import torch

from scripts.predict_segmentation import (
    checkpoint_path,
    resolve_device,
    resolve_precision,
)


def test_checkpoint_paths() -> None:
    assert str(
        checkpoint_path("segformer")
    ) == "models/segformer_b0_best.pt"

    assert str(
        checkpoint_path("unet")
    ) == "models/unet_baseline_best.pt"


def test_cpu_auto_precision_is_fp32() -> None:
    device = torch.device("cpu")

    assert (
        resolve_precision(
            "auto",
            device,
        )
        == "fp32"
    )


def test_cuda_auto_precision_is_fp16() -> None:
    device = torch.device("cuda")

    assert (
        resolve_precision(
            "auto",
            device,
        )
        == "fp16"
    )


def test_explicit_fp32_on_cpu_is_allowed() -> None:
    device = torch.device("cpu")

    assert (
        resolve_precision(
            "fp32",
            device,
        )
        == "fp32"
    )


def test_auto_device_returns_valid_device() -> None:
    device = resolve_device(
        "auto"
    )

    assert device.type in {
        "cpu",
        "cuda",
    }
