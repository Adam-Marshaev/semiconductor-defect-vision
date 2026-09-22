import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.inference.segmentation_predictor import (
    SegmentationPredictor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run semiconductor defect segmentation "
            "on a grayscale SEM image."
        )
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Input image path.",
    )

    parser.add_argument(
        "--output-mask",
        type=Path,
        default=None,
        help=(
            "Output PNG mask path. "
            "Defaults to <image_stem>_mask.png."
        ),
    )

    parser.add_argument(
        "--output-probability",
        type=Path,
        default=None,
        help=(
            "Optional .npy path for the "
            "probability map."
        ),
    )

    parser.add_argument(
        "--model",
        choices=[
            "segformer",
            "unet",
        ],
        default="segformer",
    )

    parser.add_argument(
        "--precision",
        choices=[
            "auto",
            "fp32",
            "fp16",
        ],
        default="auto",
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cuda",
            "cpu",
        ],
        default="auto",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    return parser.parse_args()


def resolve_device(
    requested_device: str,
) -> torch.device:
    if requested_device == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(
        requested_device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is unavailable."
        )

    return device


def resolve_precision(
    requested_precision: str,
    device: torch.device,
) -> str:
    if requested_precision == "auto":
        if device.type == "cuda":
            return "fp16"

        return "fp32"

    if (
        requested_precision == "fp16"
        and device.type != "cuda"
    ):
        raise ValueError(
            "FP16 inference requires CUDA."
        )

    return requested_precision


def checkpoint_path(
    model: str,
) -> Path:
    if model == "segformer":
        return Path(
            "models/segformer_b0_best.pt"
        )

    if model == "unet":
        return Path(
            "models/unet_baseline_best.pt"
        )

    raise ValueError(
        f"Unsupported model: {model}"
    )


def main() -> None:
    args = parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(
            f"Input image not found: {args.image}"
        )

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1"
        )

    device = resolve_device(
        args.device
    )

    precision = resolve_precision(
        args.precision,
        device,
    )

    output_mask = (
        args.output_mask
        if args.output_mask is not None
        else args.image.with_name(
            f"{args.image.stem}_mask.png"
        )
    )

    predictor = SegmentationPredictor(
        model_type=args.model,
        checkpoint_path=checkpoint_path(
            args.model
        ),
        threshold=args.threshold,
        precision=precision,
        device=device,
    )

    prediction = predictor.predict_path(
        args.image
    )

    output_mask.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask_uint8 = (
        prediction.mask.astype(
            np.uint8
        )
        * 255
    )

    Image.fromarray(
        mask_uint8
    ).save(
        output_mask
    )

    if args.output_probability is not None:
        args.output_probability.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            args.output_probability,
            prediction.probability,
        )

    print(
        "SEM DEFECT SEGMENTATION"
    )
    print(
        "======================="
    )
    print(
        "input:",
        args.image,
    )
    print(
        "model:",
        args.model,
    )
    print(
        "device:",
        device,
    )
    print(
        "precision:",
        precision,
    )
    print(
        "threshold:",
        args.threshold,
    )
    print(
        "predicted defect pixels:",
        int(
            prediction.mask.sum()
        ),
    )
    print(
        "mask:",
        output_mask,
    )

    if args.output_probability is not None:
        print(
            "probability map:",
            args.output_probability,
        )


if __name__ == "__main__":
    main()
