import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image

from src.models.segformer import (
    SegFormerBinarySegmenter,
)
from src.models.unet import UNet

ModelType = Literal[
    "unet",
    "segformer",
]

Precision = Literal[
    "fp32",
    "fp16",
]


@dataclass(frozen=True)
class SegmentationPrediction:
    probability: np.ndarray
    mask: np.ndarray
    threshold: float


def prepare_grayscale_tensor(
    image: np.ndarray,
) -> torch.Tensor:
    if image.ndim != 2:
        raise ValueError(
            "Expected a 2D grayscale image"
        )

    if image.dtype == np.uint8:
        normalized = (
            image.astype(np.float32)
            / 255.0
        )
    else:
        normalized = image.astype(
            np.float32
        )

        if (
            normalized.min() < 0.0
            or normalized.max() > 1.0
        ):
            raise ValueError(
                "Floating-point images must "
                "be in the range [0, 1]"
            )

    return torch.from_numpy(
        np.ascontiguousarray(normalized)
    ).unsqueeze(0).unsqueeze(0)


def load_grayscale_image(
    path: Path,
) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(
            image.convert("L"),
            dtype=np.uint8,
        ).copy()


class SegmentationPredictor:
    def __init__(
        self,
        model_type: ModelType,
        checkpoint_path: Path,
        *,
        threshold: float = 0.5,
        device: torch.device | None = None,
        precision: Precision = "fp32",
        statistics_path: Path = Path(
            "data/processed/"
            "training_statistics.json"
        ),
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be "
                "between 0 and 1"
            )

        if precision not in (
            "fp32",
            "fp16",
        ):
            raise ValueError(
                f"Unsupported precision: {precision}"
            )

        self.model_type = model_type
        self.threshold = threshold
        self.precision = precision

        self.device = (
            device
            if device is not None
            else torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        if (
            self.precision == "fp16"
            and self.device.type != "cuda"
        ):
            raise ValueError(
                "FP16 inference requires CUDA"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if model_type == "unet":
            statistics = json.loads(
                statistics_path.read_text()
            )

            self.image_mean = float(
                statistics["image_mean"]
            )
            self.image_std = float(
                statistics["image_std"]
            )

            model = UNet(
                in_channels=1,
                out_channels=1,
                base_channels=32,
            )

        elif model_type == "segformer":
            self.image_mean = None
            self.image_std = None

            model_name = checkpoint.get(
                "model_name",
                "nvidia/mit-b0",
            )

            model = (
                SegFormerBinarySegmenter(
                    model_name=model_name,
                )
            )

        else:
            raise ValueError(
                f"Unsupported model type: "
                f"{model_type}"
            )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model = model.to(
            self.device
        )
        self.model.eval()

    def _preprocess(
        self,
        image: np.ndarray,
    ) -> torch.Tensor:
        tensor = prepare_grayscale_tensor(
            image
        )

        if self.model_type == "unet":
            if (
                self.image_mean is None
                or self.image_std is None
            ):
                raise RuntimeError(
                    "U-Net normalization "
                    "statistics unavailable"
                )

            tensor = (
                tensor
                - self.image_mean
            ) / self.image_std

        return tensor.to(
            self.device
        )

    def predict_array(
        self,
        image: np.ndarray,
    ) -> SegmentationPrediction:
        tensor = self._preprocess(
            image
        )

        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=(
                    self.precision
                    == "fp16"
                ),
            ),
        ):
            logits = self.model(
                tensor
            )

            probability = torch.sigmoid(
                logits
            )[0, 0]

        probability_array = (
            probability
            .float()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        mask = (
            probability_array
            >= self.threshold
        )

        return SegmentationPrediction(
            probability=probability_array,
            mask=mask,
            threshold=self.threshold,
        )

    def predict_path(
        self,
        image_path: Path,
    ) -> SegmentationPrediction:
        image = load_grayscale_image(
            image_path
        )

        return self.predict_array(
            image
        )
