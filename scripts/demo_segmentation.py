import argparse
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.inference.segmentation_predictor import SegmentationPredictor

DATA_ROOT = Path("data/raw/carinthia-s/data")

METRICS_PATH = Path(
    "reports/training/segformer_validation_per_image.csv"
)

MANIFEST_PATH = Path(
    "data/processed/manifest.csv"
)

SPLITS_PATH = Path(
    "data/processed/splits.csv"
)

DEFAULT_CHECKPOINT = Path(
    "models/segformer_b0_best.pt"
)

MODEL_LIMITATION_SAMPLE_ID = (
    "4e336e23d47246ab894bdf66f17d65ea"
)

ANNOTATION_DISAGREEMENT_SAMPLE_ID = (
    "4e0c40ae83b142f0acacdfb56b447ed8"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive semiconductor defect "
            "segmentation portfolio demo."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
    )

    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16"],
        default="fp16",
    )


    return parser.parse_args()


def load_grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(
            image.convert("L"),
            dtype=np.uint8,
        )


def load_reference_mask(
    path: Path,
) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(
            image.convert("L"),
            dtype=np.uint8,
        )

    return mask >= 128


def select_demo_cases() -> list[dict]:
    metrics = pd.read_csv(
        METRICS_PATH
    )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    splits = pd.read_csv(
        SPLITS_PATH
    )

    validation_ids = set(
        splits.loc[
            splits["split"] == "val",
            "sample_id",
        ]
    )

    metrics = metrics.loc[
        metrics["sample_id"].isin(
            validation_ids
        )
    ].copy()

    manifest_columns = [
        "sample_id",
        "label",
        "image_path",
        "mask_path",
        "defect_pixels",
        "defect_fraction",
        "component_count",
    ]

    merged = metrics.merge(
        manifest[
            manifest_columns
        ],
        on=[
            "sample_id",
            "label",
        ],
        how="left",
        validate="one_to_one",
    )

    nonempty = merged.loc[
        ~merged["empty_target"]
    ].copy()

    nonempty = nonempty.loc[
        nonempty["defect_pixels"] > 0
    ]

    selected_ids: set[str] = set()

    # --------------------------------------------------------------
    # Example 1: easy high-quality prediction with representative size.
    # --------------------------------------------------------------

    easy_pool = nonempty.loc[
        (nonempty["dice"] >= 0.97)
        & (
            nonempty[
                "defect_fraction"
            ].between(
                0.01,
                0.15,
            )
        )
    ].copy()

    if easy_pool.empty:
        easy_pool = nonempty.copy()

    median_pixels = float(
        easy_pool[
            "defect_pixels"
        ].median()
    )

    easy_pool[
        "size_distance"
    ] = (
        easy_pool["defect_pixels"]
        - median_pixels
    ).abs()

    easy = (
        easy_pool.sort_values(
            [
                "dice",
                "size_distance",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .iloc[0]
    )

    selected_ids.add(
        str(easy["sample_id"])
    )

    # --------------------------------------------------------------
    # Example 2: visually complex prediction with many components.
    # --------------------------------------------------------------

    complex_pool = nonempty.loc[
        (
            nonempty["dice"]
            >= 0.90
        )
        & (
            ~nonempty[
                "sample_id"
            ].isin(selected_ids)
        )
    ].copy()

    complex_case = (
        complex_pool.sort_values(
            [
                "component_count",
                "dice",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .iloc[0]
    )

    selected_ids.add(
        str(
            complex_case[
                "sample_id"
            ]
        )
    )

    # --------------------------------------------------------------
    # Example 3: smallest well-segmented defect.
    # --------------------------------------------------------------

    tiny_pool = nonempty.loc[
        (
            nonempty["dice"]
            >= 0.80
        )
        & (
            ~nonempty[
                "sample_id"
            ].isin(selected_ids)
        )
    ].copy()

    tiny = (
        tiny_pool.sort_values(
            [
                "defect_pixels",
                "dice",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .iloc[0]
    )

    # --------------------------------------------------------------
    # Examples 4 and 5: manually reviewed examples.
    # --------------------------------------------------------------

    def get_curated_case(
        sample_id: str,
    ) -> pd.Series:
        matches = nonempty.loc[
            nonempty["sample_id"]
            == sample_id
        ]

        if len(matches) != 1:
            raise ValueError(
                "Expected exactly one validation "
                f"sample for {sample_id}, "
                f"found {len(matches)}."
            )

        return matches.iloc[0]

    model_limitation = get_curated_case(
        MODEL_LIMITATION_SAMPLE_ID
    )

    annotation_disagreement = (
        get_curated_case(
            ANNOTATION_DISAGREEMENT_SAMPLE_ID
        )
    )

    return [
        {
            "category": "EASY SUCCESS",
            "row": easy,
        },
        {
            "category": (
                "COMPLEX DEFECT — SUCCESS"
            ),
            "row": complex_case,
        },
        {
            "category": (
                "TINY DEFECT — SUCCESS"
            ),
            "row": tiny,
        },
        {
            "category": (
                "MODEL LIMITATION — "
                "PARTIAL UNDER-SEGMENTATION"
            ),
            "row": model_limitation,
        },
        {
            "category": (
                "MODEL–ANNOTATION DISAGREEMENT"
            ),
            "row": annotation_disagreement,
        },
    ]


def warm_up_predictor(
    predictor: SegmentationPredictor,
    image_path: Path,
    device: torch.device,
    iterations: int = 5,
) -> None:
    for _ in range(iterations):
        predictor.predict_path(
            image_path
        )

    if device.type == "cuda":
        torch.cuda.synchronize()


def measure_prediction(
    predictor: SegmentationPredictor,
    image_path: Path,
    device: torch.device,
):
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = perf_counter()

    prediction = (
        predictor.predict_path(
            image_path
        )
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    latency_ms = (
        perf_counter() - start
    ) * 1000.0

    return (
        prediction,
        latency_ms,
    )


def main() -> None:
    args = parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{args.checkpoint}"
        )

    if (
        args.device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    if (
        args.precision == "fp16"
        and args.device != "cuda"
    ):
        raise ValueError(
            "FP16 inference requires CUDA."
        )

    device = torch.device(
        args.device
    )

    cases = select_demo_cases()

    for case in cases:
        row = case["row"]

        case["image_path"] = (
            DATA_ROOT
            / str(
                row["image_path"]
            )
        )

        case["mask_path"] = (
            DATA_ROOT
            / str(
                row["mask_path"]
            )
        )

    print()
    print(
        "SELECTED DEMO CASES"
    )
    print(
        "==================="
    )

    for index, case in enumerate(
        cases,
        start=1,
    ):
        row = case["row"]

        print()
        print(
            f"{index}. "
            f"{case['category']}"
        )

        print(
            "   sample:",
            row["sample_id"],
        )

        print(
            "   label:",
            int(row["label"]),
        )

        print(
            "   validation Dice:",
            f"{row['dice']:.4f}",
        )

        print(
            "   defect pixels:",
            int(
                row[
                    "defect_pixels"
                ]
            ),
        )

        print(
            "   components:",
            int(
                row[
                    "component_count"
                ]
            ),
        )

    predictor = SegmentationPredictor(
        model_type="segformer",
        checkpoint_path=(
            args.checkpoint
        ),
        threshold=args.threshold,
        precision=args.precision,
        device=device,
    )

    print()
    print("Warming up GPU...")

    warm_up_predictor(
        predictor,
        cases[0]["image_path"],
        device,
    )

    print("Ready.")
    print()

    prediction_cache = {}

    modes = [
        "RAW SEM",
        "MODEL SEGMENTATION",
        "INSPECTION RESULT",
    ]

    state = {
        "sample": 0,
        "mode": 0,
    }

    fig, ax = plt.subplots(
        figsize=(10, 8),
    )

    fig.patch.set_facecolor(
        "#0d1117"
    )

    ax.set_facecolor(
        "#0d1117"
    )

    fig.subplots_adjust(
        left=0.04,
        right=0.96,
        bottom=0.11,
        top=0.86,
    )

    try:
        fig.canvas.manager.set_window_title(
            "Semiconductor Defect Vision"
        )
    except AttributeError:
        pass

    title_text = fig.text(
        0.5,
        0.95,
        "",
        ha="center",
        va="top",
        fontsize=20,
        color="white",
        weight="bold",
    )

    subtitle_text = fig.text(
        0.5,
        0.905,
        "",
        ha="center",
        va="top",
        fontsize=12,
        color="#b7c0ca",
    )

    footer_text = fig.text(
        0.5,
        0.035,
        (
            "SPACE: change view    "
            "← / →: previous / next image    "
            "Q: quit"
        ),
        ha="center",
        fontsize=10,
        color="#8b949e",
    )

    def ensure_prediction(
        sample_index: int,
    ):
        case = cases[
            sample_index
        ]

        sample_id = str(
            case["row"][
                "sample_id"
            ]
        )

        if (
            sample_id
            not in prediction_cache
        ):
            prediction, latency_ms = (
                measure_prediction(
                    predictor,
                    case[
                        "image_path"
                    ],
                    device,
                )
            )

            prediction_cache[
                sample_id
            ] = {
                "prediction": (
                    prediction
                ),
                "latency_ms": (
                    latency_ms
                ),
            }

        return prediction_cache[
            sample_id
        ]

    def render() -> None:
        ax.clear()

        ax.set_facecolor(
            "#0d1117"
        )

        case = cases[
            state["sample"]
        ]

        row = case["row"]

        image = load_grayscale(
            case["image_path"]
        )

        mode = modes[
            state["mode"]
        ]

        title_text.set_text(
            case["category"]
        )

        subtitle_text.set_text(
            
                f"{mode}    •    "
                f"Example "
                f"{state['sample'] + 1}"
                f"/{len(cases)}"
            
        )

        ax.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=255,
        )

        if (
            state["mode"] >= 1
        ):
            result = ensure_prediction(
                state["sample"]
            )

            prediction = result[
                "prediction"
            ]

            mask = (
                prediction.mask
                .astype(bool)
            )

            overlay = np.ma.masked_where(
                ~mask,
                mask.astype(float),
            )

            ax.imshow(
                overlay,
                cmap="autumn",
                alpha=0.38,
                interpolation="nearest",
            )

        if (
            state["mode"] == 2
        ):
            result = ensure_prediction(
                state["sample"]
            )

            prediction = result[
                "prediction"
            ]

            mask = (
                prediction.mask
                .astype(bool)
            )

            latency_ms = float(
                result[
                    "latency_ms"
                ]
            )

            if mask.any():
                ax.contour(
                    mask.astype(
                        np.uint8
                    ),
                    levels=[0.5],
                    colors="cyan",
                    linewidths=1.5,
                )

            defect_pixels = int(
                mask.sum()
            )

            defect_fraction = (
                defect_pixels
                / mask.size
            )

            inspection_text = (
                "SegFormer-B0\n"
                f"{args.precision.upper()} "
                f"• threshold "
                f"{args.threshold:.2f}\n"
                f"Defect pixels: "
                f"{defect_pixels:,}\n"
                f"Defect area: "
                f"{defect_fraction:.2%}\n"
                f"Latency: "
                f"{latency_ms:.2f} ms"
            )

            show_reference = (
                case["category"]
                in {
                    (
                        "MODEL LIMITATION — "
                        "PARTIAL UNDER-SEGMENTATION"
                    ),
                    (
                        "MODEL–ANNOTATION "
                        "DISAGREEMENT"
                    ),
                }
            )

            if show_reference:
                reference = (
                    load_reference_mask(
                        case[
                            "mask_path"
                        ]
                    )
                )

                if reference.any():
                    ax.contour(
                        reference.astype(
                            np.uint8
                        ),
                        levels=[0.5],
                        colors="yellow",
                        linewidths=1.5,
                    )

                inspection_text += (
                    "\n\nValidation comparison\n"
                    f"Dice: "
                    f"{row['dice']:.3f}\n"
                    f"Precision: "
                    f"{row['precision']:.3f}\n"
                    f"Recall: "
                    f"{row['recall']:.3f}\n"
                    "Cyan: model prediction\n"
                    "Yellow: dataset annotation"
                )

                if case["category"].startswith(
                    "MODEL LIMITATION"
                ):
                    inspection_text += (
                        "\n\nModel captures the main"
                        "\ndefect but misses some"
                        "\nsatellite defect regions."
                    )

                else:
                    inspection_text += (
                        "\n\nModel identifies additional"
                        "\nvisually apparent defect"
                        "\nstructure beyond the"
                        "\ndataset annotation."
                    )

            ax.text(
                0.025,
                0.025,
                inspection_text,
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                fontsize=11,
                color="white",
                bbox={
                    "boxstyle": (
                        "round,pad=0.55"
                    ),
                    "facecolor": (
                        "#111820"
                    ),
                    "edgecolor": (
                        "#30363d"
                    ),
                    "alpha": 0.90,
                },
            )

        ax.axis("off")

        footer_text.set_visible(
            True
        )

        fig.canvas.draw_idle()

    def on_key(event) -> None:
        if event.key in {
            " ",
            "space",
        }:
            state["mode"] = (
                state["mode"] + 1
            ) % len(modes)

            render()

        elif event.key == "right":
            state["sample"] = (
                state["sample"] + 1
            ) % len(cases)

            state["mode"] = 0

            render()

        elif event.key == "left":
            state["sample"] = (
                state["sample"] - 1
            ) % len(cases)

            state["mode"] = 0

            render()

        elif event.key in {
            "q",
            "escape",
        }:
            plt.close(fig)

    fig.canvas.mpl_connect(
        "key_press_event",
        on_key,
    )

    render()

    plt.show()


if __name__ == "__main__":
    main()
