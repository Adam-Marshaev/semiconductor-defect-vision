import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.inference.segmentation_predictor import (
    SegmentationPredictor,
)

DATASET_ROOT = Path(
    "data/raw/carinthia-s/data"
)
MANIFEST_PATH = Path(
    "data/processed/manifest.csv"
)
SPLITS_PATH = Path(
    "data/processed/splits.csv"
)

UNET_CHECKPOINT_PATH = Path(
    "models/unet_baseline_best.pt"
)
SEGFORMER_CHECKPOINT_PATH = Path(
    "models/segformer_b0_best.pt"
)

OUTPUT_PATH = Path(
    "reports/benchmark/"
    "v100_segmentation_end_to_end.csv"
)

WARMUP_IMAGES = 25
TIMED_IMAGES = 200


def validation_image_paths() -> list[Path]:
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

    validation = manifest.loc[
        manifest["sample_id"].isin(
            validation_ids
        )
    ].sort_values(
        "sample_id"
    )

    return [
        DATASET_ROOT
        / image_path
        for image_path in validation[
            "image_path"
        ]
    ]


def benchmark_predictor(
    *,
    model_type: str,
    checkpoint_path: Path,
    precision: str,
    image_paths: list[Path],
    device: torch.device,
) -> dict[str, object]:
    predictor = SegmentationPredictor(
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        threshold=0.5,
        precision=precision,
        device=device,
    )

    warmup_paths = [
        image_paths[
            index % len(image_paths)
        ]
        for index in range(
            WARMUP_IMAGES
        )
    ]

    timed_paths = [
        image_paths[
            index % len(image_paths)
        ]
        for index in range(
            TIMED_IMAGES
        )
    ]

    for image_path in warmup_paths:
        predictor.predict_path(
            image_path
        )

    torch.cuda.synchronize()

    latencies_ms = []

    torch.cuda.reset_peak_memory_stats(
        device
    )

    for image_path in timed_paths:
        start = time.perf_counter()

        prediction = (
            predictor.predict_path(
                image_path
            )
        )

        torch.cuda.synchronize()

        end = time.perf_counter()

        if prediction.mask.shape != (
            480,
            480,
        ):
            raise RuntimeError(
                "Unexpected prediction shape"
            )

        latencies_ms.append(
            (end - start) * 1000.0
        )

    latency_array = np.asarray(
        latencies_ms,
        dtype=np.float64,
    )

    mean_latency = float(
        latency_array.mean()
    )

    peak_memory = (
        torch.cuda.max_memory_allocated(
            device
        )
        / (1024**2)
    )

    return {
        "model": model_type,
        "precision": precision,
        "timed_images": TIMED_IMAGES,
        "mean_latency_ms": (
            mean_latency
        ),
        "median_latency_ms": float(
            np.median(
                latency_array
            )
        ),
        "p95_latency_ms": float(
            np.percentile(
                latency_array,
                95,
            )
        ),
        "min_latency_ms": float(
            latency_array.min()
        ),
        "max_latency_ms": float(
            latency_array.max()
        ),
        "throughput_images_per_s": (
            1000.0
            / mean_latency
        ),
        "peak_allocated_mib": (
            peak_memory
        ),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required"
        )

    device = torch.device(
        "cuda"
    )

    image_paths = (
        validation_image_paths()
    )

    if len(image_paths) != 459:
        raise RuntimeError(
            "Expected 459 validation images, "
            f"found {len(image_paths)}"
        )

    configurations = [
        (
            "unet",
            UNET_CHECKPOINT_PATH,
            "fp32",
        ),
        (
            "unet",
            UNET_CHECKPOINT_PATH,
            "fp16",
        ),
        (
            "segformer",
            SEGFORMER_CHECKPOINT_PATH,
            "fp32",
        ),
        (
            "segformer",
            SEGFORMER_CHECKPOINT_PATH,
            "fp16",
        ),
    ]

    rows = []

    print(
        "END-TO-END SEGMENTATION BENCHMARK"
    )
    print(
        "================================="
    )
    print(
        "GPU:",
        torch.cuda.get_device_name(
            device
        ),
    )
    print(
        "validation images:",
        len(image_paths),
    )
    print(
        "timed images/config:",
        TIMED_IMAGES,
    )

    for (
        model_type,
        checkpoint_path,
        precision,
    ) in configurations:
        print()
        print(
            f"{model_type} {precision}...",
            flush=True,
        )

        result = benchmark_predictor(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            precision=precision,
            image_paths=image_paths,
            device=device,
        )

        rows.append(
            result
        )

        print(
            "mean:",
            f"{result['mean_latency_ms']:.3f} ms",
        )
        print(
            "median:",
            f"{result['median_latency_ms']:.3f} ms",
        )
        print(
            "p95:",
            f"{result['p95_latency_ms']:.3f} ms",
        )
        print(
            "throughput:",
            f"{result['throughput_images_per_s']:.1f} img/s",
        )

        del result

        torch.cuda.empty_cache()

    output = pd.DataFrame(
        rows
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        output.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )

    print()
    print(
        f"Wrote: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
