from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

DATASET_ROOT = Path("data/raw/carinthia-s/data")
MANIFEST_PATH = Path("data/processed/manifest.csv")

OUTPUT_PATH = Path(
    "reports/figures/readme/near_duplicates/"
    "suspected_duplicate_full_resolution.png"
)

SAMPLE_A = "6830f9ceb59a485681c6f5392493edc9"
SAMPLE_B = "b76d11f40521489ca4d2bf47804d81d1"


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image_file:
        return np.asarray(
            image_file.convert("L"),
            dtype=np.float32,
        )


def standardize(image: np.ndarray) -> np.ndarray:
    std = float(image.std())

    if std < 1e-8:
        return image - image.mean()

    return (image - image.mean()) / std


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = standardize(a).ravel()
    b_flat = standardize(b).ravel()

    return float(
        np.mean(a_flat * b_flat)
    )


def warp_image(
    image: np.ndarray,
    shift_x: float,
    shift_y: float,
) -> np.ndarray:
    matrix = np.array(
        [
            [1.0, 0.0, shift_x],
            [0.0, 1.0, shift_y],
        ],
        dtype=np.float32,
    )

    height, width = image.shape

    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def fit_contrast(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> tuple[float, float, float, float]:
    x = reference.ravel().astype(np.float64)
    y = candidate.ravel().astype(np.float64)

    design = np.column_stack(
        [
            x,
            np.ones_like(x),
        ]
    )

    scale, offset = np.linalg.lstsq(
        design,
        y,
        rcond=None,
    )[0]

    prediction = scale * x + offset

    residual = y - prediction

    rmse = float(
        np.sqrt(
            np.mean(residual**2)
        )
    )

    total_variance = float(
        np.sum(
            (y - y.mean()) ** 2
        )
    )

    residual_variance = float(
        np.sum(residual**2)
    )

    if total_variance > 0:
        r_squared = 1.0 - (
            residual_variance / total_variance
        )
    else:
        r_squared = 0.0

    return (
        float(scale),
        float(offset),
        rmse,
        r_squared,
    )


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)

    lookup = manifest.set_index(
        "sample_id"
    )["image_path"].to_dict()

    image_a = load_image(
        DATASET_ROOT / lookup[SAMPLE_A]
    )

    image_b = load_image(
        DATASET_ROOT / lookup[SAMPLE_B]
    )

    raw_correlation = correlation(
        image_a,
        image_b,
    )

    standardized_a = standardize(
        image_a
    ).astype(np.float32)

    standardized_b = standardize(
        image_b
    ).astype(np.float32)

    shift, phase_response = cv2.phaseCorrelate(
        standardized_a,
        standardized_b,
    )

    shift_x, shift_y = shift

    # Test both possible shift directions and retain
    # whichever produces the stronger correlation.
    aligned_positive = warp_image(
        image_b,
        shift_x,
        shift_y,
    )

    aligned_negative = warp_image(
        image_b,
        -shift_x,
        -shift_y,
    )

    positive_correlation = correlation(
        image_a,
        aligned_positive,
    )

    negative_correlation = correlation(
        image_a,
        aligned_negative,
    )

    if positive_correlation >= negative_correlation:
        aligned_b = aligned_positive
        selected_shift = (
            shift_x,
            shift_y,
        )
        aligned_correlation = positive_correlation
    else:
        aligned_b = aligned_negative
        selected_shift = (
            -shift_x,
            -shift_y,
        )
        aligned_correlation = negative_correlation

    (
        contrast_scale,
        contrast_offset,
        contrast_rmse,
        contrast_r_squared,
    ) = fit_contrast(
        image_a,
        aligned_b,
    )

    predicted_b = (
        contrast_scale * image_a
        + contrast_offset
    )

    residual = np.abs(
        aligned_b - predicted_b
    )

    print("FULL-RESOLUTION PAIR INSPECTION")
    print("===============================")
    print(f"sample A: {SAMPLE_A}")
    print(f"sample B: {SAMPLE_B}")
    print()
    print(
        f"raw pixel correlation: "
        f"{raw_correlation:.6f}"
    )
    print(
        "phase-correlation shift:",
        f"({shift_x:.3f}, {shift_y:.3f})",
    )
    print(
        f"phase response: "
        f"{phase_response:.6f}"
    )
    print(
        "selected alignment shift:",
        f"({selected_shift[0]:.3f}, "
        f"{selected_shift[1]:.3f})",
    )
    print(
        f"aligned pixel correlation: "
        f"{aligned_correlation:.6f}"
    )
    print()
    print(
        "Best linear contrast model:"
    )
    print(
        "B ≈ "
        f"{contrast_scale:.6f} * A "
        f"+ {contrast_offset:.6f}"
    )
    print(
        f"contrast-model R^2: "
        f"{contrast_r_squared:.6f}"
    )
    print(
        f"contrast-model RMSE: "
        f"{contrast_rmse:.6f}"
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 10),
    )

    axes[0, 0].imshow(
        image_a,
        cmap="gray",
    )
    axes[0, 0].set_title("Sample A")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(
        image_b,
        cmap="gray",
    )
    axes[0, 1].set_title("Sample B")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(
        aligned_b,
        cmap="gray",
    )
    axes[1, 0].set_title(
        "Sample B after alignment"
    )
    axes[1, 0].axis("off")

    axes[1, 1].imshow(
        residual,
        cmap="gray",
    )
    axes[1, 1].set_title(
        "Residual after alignment + contrast fit"
    )
    axes[1, 1].axis("off")

    fig.tight_layout()

    fig.savefig(
        OUTPUT_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(f"Figure: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
