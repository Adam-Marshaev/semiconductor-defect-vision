import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.carinthia_dataset import (
    CarinthiaSegmentationDataset,
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
SPLITS_PATH = Path(
    "data/processed/splits.csv"
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
    "reports/benchmark/"
    "v100_segmentation_gpu.csv"
)

BATCH_SIZES = [
    1,
    8,
    16,
    32,
]

PRECISIONS = [
    "fp32",
    "fp16",
]

WARMUP_ITERATIONS = 20
TIMED_ITERATIONS = 50
PREDICTION_THRESHOLD = 0.5


def parameter_count(
    model: torch.nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def create_model(
    model_name: str,
    device: torch.device,
) -> torch.nn.Module:
    if model_name == "unet":
        checkpoint = torch.load(
            UNET_CHECKPOINT_PATH,
            map_location=device,
            weights_only=False,
        )

        model = UNet(
            in_channels=1,
            out_channels=1,
            base_channels=32,
        )

    elif model_name == "segformer":
        checkpoint = torch.load(
            SEGFORMER_CHECKPOINT_PATH,
            map_location=device,
            weights_only=False,
        )

        pretrained_name = checkpoint.get(
            "model_name",
            "nvidia/mit-b0",
        )

        model = (
            SegFormerBinarySegmenter(
                model_name=pretrained_name,
            )
        )

    else:
        raise ValueError(
            f"Unsupported model: "
            f"{model_name}"
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(device)
    model.eval()

    return model


def create_base_inputs() -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    dataset = CarinthiaSegmentationDataset(
        dataset_root=DATASET_ROOT,
        manifest_path=MANIFEST_PATH,
        splits_path=SPLITS_PATH,
        split="val",
        transform=None,
    )

    raw_image = dataset[0][
        "image"
    ].unsqueeze(0)

    statistics = json.loads(
        STATS_PATH.read_text()
    )

    unet_image = (
        raw_image
        - float(
            statistics["image_mean"]
        )
    ) / float(
        statistics["image_std"]
    )

    return (
        unet_image,
        raw_image,
    )


def run_forward(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    use_fp16: bool,
) -> torch.Tensor:
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=use_fp16,
    ):
        logits = model(
            inputs
        )

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities
            >= PREDICTION_THRESHOLD
        )

    return predictions


def benchmark_configuration(
    model: torch.nn.Module,
    base_input: torch.Tensor,
    model_name: str,
    precision: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    use_fp16 = (
        precision == "fp16"
    )

    inputs = (
        base_input.repeat(
            batch_size,
            1,
            1,
            1,
        )
        .to(
            device,
            non_blocking=False,
        )
    )

    try:
        with torch.inference_mode():
            for _ in range(
                WARMUP_ITERATIONS
            ):
                run_forward(
                    model,
                    inputs,
                    use_fp16=use_fp16,
                )

        torch.cuda.synchronize()

        baseline_memory = (
            torch.cuda.memory_allocated(
                device
            )
        )

        torch.cuda.reset_peak_memory_stats(
            device
        )

        starts = []
        ends = []

        with torch.inference_mode():
            for _ in range(
                TIMED_ITERATIONS
            ):
                start = torch.cuda.Event(
                    enable_timing=True
                )
                end = torch.cuda.Event(
                    enable_timing=True
                )

                start.record()

                run_forward(
                    model,
                    inputs,
                    use_fp16=use_fp16,
                )

                end.record()

                starts.append(
                    start
                )
                ends.append(
                    end
                )

        torch.cuda.synchronize()

        latencies_ms = np.asarray(
            [
                start.elapsed_time(end)
                for start, end in zip(
                    starts,
                    ends,
                    strict=True,
                )
            ],
            dtype=np.float64,
        )

        mean_batch_latency = float(
            latencies_ms.mean()
        )

        peak_memory = (
            torch.cuda.max_memory_allocated(
                device
            )
        )

        return {
            "model": model_name,
            "precision": precision,
            "batch_size": batch_size,
            "status": "ok",
            "mean_batch_latency_ms": (
                mean_batch_latency
            ),
            "median_batch_latency_ms": float(
                np.median(latencies_ms)
            ),
            "p95_batch_latency_ms": float(
                np.percentile(
                    latencies_ms,
                    95,
                )
            ),
            "mean_latency_per_image_ms": (
                mean_batch_latency
                / batch_size
            ),
            "throughput_images_per_s": (
                batch_size
                * 1000.0
                / mean_batch_latency
            ),
            "peak_allocated_mib": (
                peak_memory
                / (1024**2)
            ),
            "incremental_peak_mib": (
                (
                    peak_memory
                    - baseline_memory
                )
                / (1024**2)
            ),
        }

    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()

        return {
            "model": model_name,
            "precision": precision,
            "batch_size": batch_size,
            "status": "oom",
            "mean_batch_latency_ms": np.nan,
            "median_batch_latency_ms": np.nan,
            "p95_batch_latency_ms": np.nan,
            "mean_latency_per_image_ms": np.nan,
            "throughput_images_per_s": np.nan,
            "peak_allocated_mib": np.nan,
            "incremental_peak_mib": np.nan,
        }

    finally:
        del inputs


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required"
        )

    device = torch.device(
        "cuda"
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(
            device
        ),
    )
    print(
        "PyTorch:",
        torch.__version__,
    )
    print(
        "warmup iterations:",
        WARMUP_ITERATIONS,
    )
    print(
        "timed iterations:",
        TIMED_ITERATIONS,
    )

    (
        unet_base_input,
        segformer_base_input,
    ) = create_base_inputs()

    rows = []

    for model_name in [
        "unet",
        "segformer",
    ]:
        print()
        print(
            f"Loading {model_name}..."
        )

        model = create_model(
            model_name,
            device,
        )

        parameters = parameter_count(
            model
        )

        if model_name == "unet":
            base_input = (
                unet_base_input
            )
        else:
            base_input = (
                segformer_base_input
            )

        print(
            "parameters:",
            f"{parameters:,}",
        )

        for precision in PRECISIONS:
            for batch_size in BATCH_SIZES:
                print(
                    f"{model_name:10s} "
                    f"{precision:4s} "
                    f"batch={batch_size:2d}",
                    end=" ... ",
                    flush=True,
                )

                result = (
                    benchmark_configuration(
                        model=model,
                        base_input=base_input,
                        model_name=model_name,
                        precision=precision,
                        batch_size=batch_size,
                        device=device,
                    )
                )

                result[
                    "parameters"
                ] = parameters

                rows.append(
                    result
                )

                if (
                    result["status"]
                    == "ok"
                ):
                    print(
                        f"{result['mean_batch_latency_ms']:.3f} ms, "
                        f"{result['throughput_images_per_s']:.1f} img/s"
                    )
                else:
                    print(
                        result["status"]
                    )

        del model
        gc.collect()
        torch.cuda.empty_cache()

    results = pd.DataFrame(
        rows
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "GPU INFERENCE BENCHMARK"
    )
    print(
        "======================="
    )

    print(
        results.to_string(
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
