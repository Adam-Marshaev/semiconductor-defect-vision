import shutil
import subprocess
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

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

CHECKPOINT_PATH = Path(
    "models/segformer_b0_best.pt"
)

FRAME_DIR = Path(
    "reports/demo_frames"
)

OUTPUT_PATH = Path(
    "reports/semiconductor_defect_demo.mp4"
)

MODEL_LIMITATION_SAMPLE_ID = (
    "4e336e23d47246ab894bdf66f17d65ea"
)

ANNOTATION_DISAGREEMENT_SAMPLE_ID = (
    "4e0c40ae83b142f0acacdfb56b447ed8"
)


# Exact durations requested by the user.
CASE_DURATIONS = [
    (12, 10, 10),  # Intro + easy example
    (4, 4, 4),     # Complex example
    (4, 4, 4),     # Tiny example
    (4, 4, 8),     # Model limitation
    (10, 10, 10),  # Annotation disagreement
]

OUTRO_DURATION = 15


def load_grayscale(
    path: Path,
) -> np.ndarray:
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

    merged = metrics.merge(
        manifest[
            [
                "sample_id",
                "label",
                "image_path",
                "mask_path",
                "defect_pixels",
                "defect_fraction",
                "component_count",
            ]
        ],
        on=[
            "sample_id",
            "label",
        ],
        how="left",
        validate="one_to_one",
    )

    nonempty = merged.loc[
        (~merged["empty_target"])
        & (merged["defect_pixels"] > 0)
    ].copy()

    selected_ids: set[str] = set()

    # Easy success.
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

    # Complex successful example.
    complex_pool = nonempty.loc[
        (nonempty["dice"] >= 0.90)
        & (
            ~nonempty[
                "sample_id"
            ].isin(selected_ids)
        )
    ]

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

    # Tiny successful example.
    tiny_pool = nonempty.loc[
        (nonempty["dice"] >= 0.80)
        & (
            ~nonempty[
                "sample_id"
            ].isin(selected_ids)
        )
    ]

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

    def get_case(
        sample_id: str,
    ) -> pd.Series:
        matches = nonempty.loc[
            nonempty["sample_id"]
            == sample_id
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Expected one validation sample "
                f"for {sample_id}, found "
                f"{len(matches)}."
            )

        return matches.iloc[0]

    limitation = get_case(
        MODEL_LIMITATION_SAMPLE_ID
    )

    disagreement = get_case(
        ANNOTATION_DISAGREEMENT_SAMPLE_ID
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
            "row": limitation,
        },
        {
            "category": (
                "MODEL–ANNOTATION DISAGREEMENT"
            ),
            "row": disagreement,
        },
    ]


def warm_up(
    predictor: SegmentationPredictor,
    image_path: Path,
) -> None:
    for _ in range(5):
        predictor.predict_path(
            image_path
        )

    torch.cuda.synchronize()


def timed_prediction(
    predictor: SegmentationPredictor,
    image_path: Path,
):
    torch.cuda.synchronize()

    start = perf_counter()

    prediction = predictor.predict_path(
        image_path
    )

    torch.cuda.synchronize()

    latency_ms = (
        perf_counter() - start
    ) * 1000.0

    return prediction, latency_ms


def make_figure():
    fig, ax = plt.subplots(
        figsize=(16, 9),
        dpi=120,
    )

    fig.patch.set_facecolor(
        "#0d1117"
    )

    ax.set_facecolor(
        "#0d1117"
    )

    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.04,
        top=0.86,
    )

    return fig, ax


def add_header(
    fig,
    category: str,
    mode: str,
    index: int,
) -> None:
    fig.text(
        0.5,
        0.955,
        category,
        ha="center",
        va="top",
        fontsize=23,
        color="white",
        weight="bold",
    )

    fig.text(
        0.5,
        0.905,
        (
            f"{mode}   •   "
            f"Example {index}/5"
        ),
        ha="center",
        va="top",
        fontsize=14,
        color="#b7c0ca",
    )


def render_frame(
    case: dict,
    index: int,
    mode: str,
    prediction,
    latency_ms: float,
    output_path: Path,
) -> None:
    row = case["row"]

    image_path = (
        DATA_ROOT
        / str(row["image_path"])
    )

    mask_path = (
        DATA_ROOT
        / str(row["mask_path"])
    )

    image = load_grayscale(
        image_path
    )

    mask = prediction.mask.astype(
        bool
    )

    fig, ax = make_figure()

    add_header(
        fig,
        case["category"],
        mode,
        index,
    )

    ax.imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=255,
    )

    if mode in {
        "MODEL SEGMENTATION",
        "INSPECTION RESULT",
    }:
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

    if mode == "INSPECTION RESULT":
        if mask.any():
            ax.contour(
                mask.astype(
                    np.uint8
                ),
                levels=[0.5],
                colors="cyan",
                linewidths=1.7,
            )

        defect_pixels = int(
            mask.sum()
        )

        defect_fraction = (
            defect_pixels
            / mask.size
        )

        info = (
            "SegFormer-B0\n"
            "FP16 • threshold 0.50\n"
            f"Defect pixels: "
            f"{defect_pixels:,}\n"
            f"Defect area: "
            f"{defect_fraction:.2%}\n"
            f"Latency: "
            f"{latency_ms:.2f} ms"
        )

        show_reference = (
            index in {4, 5}
        )

        if show_reference:
            reference = (
                load_reference_mask(
                    mask_path
                )
            )

            if reference.any():
                ax.contour(
                    reference.astype(
                        np.uint8
                    ),
                    levels=[0.5],
                    colors="yellow",
                    linewidths=1.7,
                )

            info += (
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

            if index == 4:
                info += (
                    "\n\nModel captures the main"
                    "\ndefect but misses some"
                    "\nsatellite defect regions."
                )

            if index == 5:
                info += (
                    "\n\nModel identifies additional"
                    "\nvisually apparent satellite"
                    "\ndefect regions beyond the"
                    "\ndataset annotation."
                )

        ax.text(
            0.025,
            0.025,
            info,
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=12,
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

    fig.savefig(
        output_path,
        dpi=120,
        facecolor=fig.get_facecolor(),
    )

    plt.close(fig)


def render_outro(
    output_path: Path,
) -> None:
    fig = plt.figure(
        figsize=(16, 9),
        dpi=120,
        facecolor="#0d1117",
    )

    fig.text(
        0.5,
        0.67,
        "Semiconductor Defect Vision System",
        ha="center",
        color="white",
        fontsize=31,
        weight="bold",
    )

    fig.text(
        0.5,
        0.53,
        "SegFormer-B0  •  FP16  •  NVIDIA V100",
        ha="center",
        color="#b7c0ca",
        fontsize=20,
    )

    fig.text(
        0.5,
        0.43,
        "Final test Dice: 0.957  •  "
        "End-to-end inference: ~6.3 ms/image",
        ha="center",
        color="white",
        fontsize=18,
    )

    fig.text(
        0.5,
        0.29,
        (
            "github.com/Adam-Marshaev/"
            "semiconductor-defect-vision"
        ),
        ha="center",
        color="#58a6ff",
        fontsize=17,
    )

    fig.savefig(
        output_path,
        dpi=120,
        facecolor=fig.get_facecolor(),
    )

    plt.close(fig)


def encode_video(
    frames: list[
        tuple[Path, int]
    ],
) -> None:
    command = [
        "ffmpeg",
        "-y",
    ]

    for frame_path, duration in frames:
        command.extend(
            [
                "-loop",
                "1",
                "-framerate",
                "30",
                "-t",
                str(duration),
                "-i",
                str(frame_path),
            ]
        )

    filter_parts = []

    for index in range(
        len(frames)
    ):
        filter_parts.append(
            
                f"[{index}:v]"
                "setpts=PTS-STARTPTS"
                f"[v{index}]"
            
        )

    concat_inputs = "".join(
        f"[v{index}]"
        for index in range(
            len(frames)
        )
    )

    filter_parts.append(
        
            f"{concat_inputs}"
            f"concat=n={len(frames)}:"
            "v=1:a=0,"
            "format=yuv420p[vout]"
        
    )

    command.extend(
        [
            "-filter_complex",
            ";".join(
                filter_parts
            ),
            "-map",
            "[vout]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(OUTPUT_PATH),
        ]
    )

    print()
    print("Encoding video...")

    subprocess.run(
        command,
        check=True,
    )



def main() -> None:
    if shutil.which(
        "ffmpeg"
    ) is None:
        raise RuntimeError(
            "ffmpeg is required. Install "
            "with: sudo apt install ffmpeg"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this demo."
        )

    FRAME_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove old rendered frames.
    for old_frame in FRAME_DIR.glob(
        "*.png"
    ):
        old_frame.unlink()

    cases = select_demo_cases()

    for case in cases:
        row = case["row"]

        case["image_path"] = (
            DATA_ROOT
            / str(
                row["image_path"]
            )
        )

    print("DEMO VIDEO")
    print("==========")

    for index, case in enumerate(
        cases,
        start=1,
    ):
        row = case["row"]

        print(
            f"{index}. "
            f"{case['category']}"
        )

        print(
            "   sample:",
            row["sample_id"],
        )

    predictor = SegmentationPredictor(
        model_type="segformer",
        checkpoint_path=(
            CHECKPOINT_PATH
        ),
        threshold=0.5,
        precision="fp16",
        device=torch.device(
            "cuda"
        ),
    )

    print()
    print("Warming up GPU...")

    warm_up(
        predictor,
        cases[0]["image_path"],
    )

    frames: list[
        tuple[Path, int]
    ] = []

    modes = [
        "RAW SEM",
        "MODEL SEGMENTATION",
        "INSPECTION RESULT",
    ]

    for index, case in enumerate(
        cases,
        start=1,
    ):
        print()
        print(
            f"Running example "
            f"{index}/5..."
        )

        prediction, latency_ms = (
            timed_prediction(
                predictor,
                case["image_path"],
            )
        )

        print(
            "latency:",
            f"{latency_ms:.2f} ms",
        )

        durations = (
            CASE_DURATIONS[
                index - 1
            ]
        )

        for mode_index, mode in enumerate(
            modes
        ):
            frame_path = (
                FRAME_DIR
                / (
                    f"{index:02d}_"
                    f"{mode_index:02d}.png"
                )
            )

            render_frame(
                case=case,
                index=index,
                mode=mode,
                prediction=prediction,
                latency_ms=latency_ms,
                output_path=frame_path,
            )

            frames.append(
                (
                    frame_path,
                    durations[
                        mode_index
                    ],
                )
            )

    outro_path = (
        FRAME_DIR
        / "99_outro.png"
    )

    render_outro(
        outro_path
    )

    frames.append(
        (
            outro_path,
            OUTRO_DURATION,
        )
    )

    encode_video(
        frames
    )

    print()
    print("DONE")
    print("====")
    print(
        "Video:",
        OUTPUT_PATH,
    )
    total_duration = (
        sum(
            sum(durations)
            for durations in CASE_DURATIONS
        )
        + OUTRO_DURATION
    )

    print(
        "Duration:",
        f"{total_duration} seconds",
    )
    print(
        "Resolution: 1920x1080"
    )
    print(
        "Audio: none"
    )


if __name__ == "__main__":
    main()
