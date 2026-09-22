import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data.segmentation_transforms import (
    SegmentationTransform,
    SegmentationTransformConfig,
)
from src.models.segformer import (
    SegFormerBinarySegmenter,
)
from src.models.unet import UNet

DATASET_ROOT = Path(
    "data/raw/carinthia-s/data"
)
MANIFEST_PATH = Path(
    "data/processed/manifest.csv"
)
COMPARISON_PATH = Path(
    "reports/training/"
    "unet_vs_segformer_validation.csv"
)

STATS_PATH = Path(
    "data/processed/training_statistics.json"
)

UNET_CHECKPOINT_PATH = Path(
    "models/unet_baseline_best.pt"
)
SEGFORMER_CHECKPOINT_PATH = Path(
    "models/segformer_b0_best.pt"
)

OUTPUT_PATH = Path(
    "reports/figures/readme/"
    "unet_vs_segformer_validation.png"
)

PREDICTION_THRESHOLD = 0.5
TOP_N = 5


def load_raw_image(
    path: Path,
) -> np.ndarray:
    with Image.open(path) as image_file:
        return np.asarray(
            image_file.convert("L"),
            dtype=np.uint8,
        ).copy()


def load_mask(
    path: Path,
) -> np.ndarray:
    with Image.open(path) as mask_file:
        mask = np.asarray(
            mask_file.convert("L"),
            dtype=np.uint8,
        ).copy()

    return mask >= 128


def load_models(
    device: torch.device,
) -> tuple[
    UNet,
    SegFormerBinarySegmenter,
]:
    unet_checkpoint = torch.load(
        UNET_CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    unet = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=32,
    ).to(device)

    unet.load_state_dict(
        unet_checkpoint[
            "model_state_dict"
        ]
    )

    segformer_checkpoint = torch.load(
        SEGFORMER_CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model_name = segformer_checkpoint.get(
        "model_name",
        "nvidia/mit-b0",
    )

    segformer = SegFormerBinarySegmenter(
        model_name=model_name,
    ).to(device)

    segformer.load_state_dict(
        segformer_checkpoint[
            "model_state_dict"
        ]
    )

    unet.eval()
    segformer.eval()

    return unet, segformer


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required"
        )

    manifest = (
        pd.read_csv(MANIFEST_PATH)
        .set_index("sample_id")
    )

    comparison = pd.read_csv(
        COMPARISON_PATH
    )

    nonempty = comparison.loc[
        ~comparison[
            "empty_target_unet"
        ]
    ]

    improvements = (
        nonempty.sort_values(
            "delta_dice",
            ascending=False,
        )
        .head(TOP_N)
        .copy()
    )

    regressions = (
        nonempty.sort_values(
            "delta_dice",
        )
        .head(TOP_N)
        .copy()
    )

    examples = pd.concat(
        [
            improvements.assign(
                comparison_group="Improvement"
            ),
            regressions.assign(
                comparison_group="Regression"
            ),
        ],
        ignore_index=True,
    )

    statistics = json.loads(
        STATS_PATH.read_text()
    )

    unet_transform = (
        SegmentationTransform(
            SegmentationTransformConfig(
                image_mean=statistics[
                    "image_mean"
                ],
                image_std=statistics[
                    "image_std"
                ],
            ),
            training=False,
        )
    )

    device = torch.device(
        "cuda"
    )

    unet, segformer = load_models(
        device
    )

    row_count = len(examples)

    fig, axes = plt.subplots(
        row_count,
        4,
        figsize=(
            14,
            row_count * 3.1,
        ),
    )

    with torch.no_grad():
        for row_index, result in (
            examples.iterrows()
        ):
            sample_id = result[
                "sample_id"
            ]

            metadata = manifest.loc[
                sample_id
            ]

            image = load_raw_image(
                DATASET_ROOT
                / metadata["image_path"]
            )

            target = load_mask(
                DATASET_ROOT
                / metadata["mask_path"]
            )

            raw_image_tensor = (
                torch.from_numpy(
                    image.astype(
                        np.float32
                    )
                    / 255.0
                )
                .unsqueeze(0)
            )

            dummy_mask = (
                torch.from_numpy(
                    target.astype(
                        np.float32
                    )
                )
                .unsqueeze(0)
            )

            unet_image, _ = (
                unet_transform(
                    raw_image_tensor.clone(),
                    dummy_mask,
                )
            )

            unet_logits = unet(
                unet_image
                .unsqueeze(0)
                .to(device)
            )

            segformer_logits = (
                segformer(
                    raw_image_tensor
                    .unsqueeze(0)
                    .to(device)
                )
            )

            unet_prediction = (
                torch.sigmoid(
                    unet_logits
                )[0, 0]
                .cpu()
                .numpy()
                >= PREDICTION_THRESHOLD
            )

            segformer_prediction = (
                torch.sigmoid(
                    segformer_logits
                )[0, 0]
                .cpu()
                .numpy()
                >= PREDICTION_THRESHOLD
            )

            group = result[
                "comparison_group"
            ]

            delta = result[
                "delta_dice"
            ]

            axes[
                row_index,
                0,
            ].imshow(
                image,
                cmap="gray",
            )

            axes[
                row_index,
                0,
            ].set_title(
                f"{group} | "
                f"label {int(result['label'])}\n"
                f"ΔDice={delta:+.3f}"
            )

            axes[
                row_index,
                1,
            ].imshow(
                target,
                cmap="gray",
            )

            axes[
                row_index,
                1,
            ].set_title(
                "Ground truth"
            )

            axes[
                row_index,
                2,
            ].imshow(
                unet_prediction,
                cmap="gray",
            )

            axes[
                row_index,
                2,
            ].set_title(
                "U-Net\n"
                f"D={result['dice_unet']:.3f} "
                f"P={result['precision_unet']:.3f} "
                f"R={result['recall_unet']:.3f}"
            )

            axes[
                row_index,
                3,
            ].imshow(
                segformer_prediction,
                cmap="gray",
            )

            axes[
                row_index,
                3,
            ].set_title(
                "SegFormer\n"
                f"D={result['dice_segformer']:.3f} "
                f"P={result['precision_segformer']:.3f} "
                f"R={result['recall_segformer']:.3f}"
            )

            for column in range(4):
                axes[
                    row_index,
                    column,
                ].axis("off")

            axes[
                row_index,
                0,
            ].set_ylabel(
                sample_id[:8],
                rotation=0,
                labelpad=35,
            )

    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        OUTPUT_PATH,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Wrote: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
