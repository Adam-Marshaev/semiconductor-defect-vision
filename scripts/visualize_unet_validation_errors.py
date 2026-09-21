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
from src.models.unet import UNet

DATASET_ROOT = Path(
    "data/raw/carinthia-s/data"
)
MANIFEST_PATH = Path(
    "data/processed/manifest.csv"
)
RESULTS_PATH = Path(
    "reports/training/unet_validation_per_image.csv"
)
STATS_PATH = Path(
    "data/processed/training_statistics.json"
)
CHECKPOINT_PATH = Path(
    "models/unet_baseline_best.pt"
)

OUTPUT_PATH = Path(
    "reports/figures/readme/unet/"
    "worst_validation_predictions.png"
)

PREDICTION_THRESHOLD = 0.5
TOP_N = 10


def load_raw_image(
    path: Path,
) -> np.ndarray:
    with Image.open(path) as image_file:
        return np.asarray(
            image_file.convert("L"),
            dtype=np.uint8,
        )


def load_mask(
    path: Path,
) -> np.ndarray:
    with Image.open(path) as mask_file:
        mask = np.asarray(
            mask_file.convert("L"),
            dtype=np.uint8,
        )

    return mask >= 128


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required"
        )

    manifest = (
        pd.read_csv(MANIFEST_PATH)
        .set_index("sample_id")
    )

    results = pd.read_csv(
        RESULTS_PATH
    )

    worst = (
        results[
            ~results["empty_target"]
        ]
        .nsmallest(
            TOP_N,
            "dice",
        )
        .reset_index(drop=True)
    )

    statistics = json.loads(
        STATS_PATH.read_text()
    )

    transform = SegmentationTransform(
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

    device = torch.device(
        "cuda"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=32,
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    fig, axes = plt.subplots(
        TOP_N,
        4,
        figsize=(14, TOP_N * 3.2),
    )

    for row_index, result in worst.iterrows():
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

        image_tensor = (
            torch.from_numpy(
                image.astype(
                    np.float32
                )
                / 255.0
            )
            .unsqueeze(0)
        )

        dummy_mask = torch.from_numpy(
            target.astype(
                np.float32
            )
        ).unsqueeze(0)

        image_tensor, _ = transform(
            image_tensor,
            dummy_mask,
        )

        with torch.no_grad():
            logits = model(
                image_tensor
                .unsqueeze(0)
                .to(device)
            )

            probability = (
                torch.sigmoid(logits)
                [0, 0]
                .cpu()
                .numpy()
            )

        prediction = (
            probability
            >= PREDICTION_THRESHOLD
        )

        axes[row_index, 0].imshow(
            image,
            cmap="gray",
        )
        axes[row_index, 0].set_title(
            f"SEM | label {int(result['label'])}"
        )

        axes[row_index, 1].imshow(
            target,
            cmap="gray",
        )
        axes[row_index, 1].set_title(
            "Ground truth"
        )

        axes[row_index, 2].imshow(
            prediction,
            cmap="gray",
        )
        axes[row_index, 2].set_title(
            f"Prediction\n"
            f"Dice={result['dice']:.3f}"
        )

        axes[row_index, 3].imshow(
            image,
            cmap="gray",
        )
        axes[row_index, 3].imshow(
            target,
            alpha=0.25,
            cmap="Greens",
        )
        axes[row_index, 3].imshow(
            prediction,
            alpha=0.35,
            cmap="autumn",
        )
        axes[row_index, 3].set_title(
            f"P={result['precision']:.3f} "
            f"R={result['recall']:.3f}"
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
