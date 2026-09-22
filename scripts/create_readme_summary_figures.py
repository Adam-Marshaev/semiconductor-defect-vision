from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MODEL_RESULTS_PATH = Path(
    "reports/model_comparison.csv"
)

GPU_RESULTS_PATH = Path(
    "reports/benchmark/v100_segmentation_gpu.csv"
)

OUTPUT_DIR = Path(
    "reports/figures/readme"
)


def create_validation_figure() -> None:
    results = pd.read_csv(
        MODEL_RESULTS_PATH
    )

    names = [
        "OpenCV Otsu",
        "U-Net",
        "SegFormer-B0",
    ]

    dice = [
        float(
            results.loc[
                results["model"]
                == "OpenCV_Otsu",
                "validation_dice",
            ].iloc[0]
        ),
        float(
            results.loc[
                results["model"]
                == "U-Net",
                "validation_dice",
            ].iloc[0]
        ),
        float(
            results.loc[
                results["model"]
                == "SegFormer-B0",
                "validation_dice",
            ].iloc[0]
        ),
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    bars = ax.bar(
        names,
        dice,
    )

    ax.set_ylabel(
        "Validation Dice"
    )
    ax.set_title(
        "Segmentation Model Comparison"
    )
    ax.set_ylim(
        0.0,
        1.0,
    )

    for bar, value in zip(
        bars,
        dice,
        strict=True,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
        )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "validation_dice_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def create_throughput_figure() -> None:
    results = pd.read_csv(
        GPU_RESULTS_PATH
    )

    fp16 = results.loc[
        results["precision"]
        == "fp16"
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    for model_name, display_name in [
        ("unet", "U-Net"),
        ("segformer", "SegFormer-B0"),
    ]:
        model = (
            fp16.loc[
                fp16["model"]
                == model_name
            ]
            .sort_values(
                "batch_size"
            )
        )

        ax.plot(
            model["batch_size"],
            model[
                "throughput_images_per_s"
            ],
            marker="o",
            label=display_name,
        )

    ax.set_xlabel(
        "Batch size"
    )
    ax.set_ylabel(
        "Throughput (images/s)"
    )
    ax.set_title(
        "V100 FP16 Inference Scaling"
    )

    ax.set_xticks(
        [1, 8, 16, 32]
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "v100_fp16_throughput.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    create_validation_figure()
    create_throughput_figure()

    print(
        "Wrote:",
        OUTPUT_DIR
        / "validation_dice_comparison.png",
    )

    print(
        "Wrote:",
        OUTPUT_DIR
        / "v100_fp16_throughput.png",
    )


if __name__ == "__main__":
    main()
