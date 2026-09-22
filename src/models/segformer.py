import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    SegformerConfig,
    SegformerForSemanticSegmentation,
)


class SegFormerBinarySegmenter(nn.Module):
    """SegFormer-B0 adapted for binary SEM defect segmentation."""

    def __init__(
        self,
        model_name: str = "nvidia/mit-b0",
    ) -> None:
        super().__init__()

        config = SegformerConfig.from_pretrained(
            model_name
        )
        config.num_labels = 1
        config.id2label = {0: "defect"}
        config.label2id = {"defect": 0}

        self.backbone = (
            SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                config=config,
                ignore_mismatched_sizes=True,
            )
        )

        self.register_buffer(
            "image_mean",
            torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
            persistent=False,
        )

        self.register_buffer(
            "image_std",
            torch.tensor(
                [0.229, 0.224, 0.225],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(
                "Expected images with shape [B, 1, H, W]"
            )

        if images.shape[1] != 1:
            raise ValueError(
                "Expected single-channel grayscale images"
            )

        original_size = images.shape[-2:]

        pixel_values = images.repeat(
            1,
            3,
            1,
            1,
        )

        pixel_values = (
            pixel_values - self.image_mean
        ) / self.image_std

        outputs = self.backbone(
            pixel_values=pixel_values,
        )

        logits = F.interpolate(
            outputs.logits,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )

        return logits
