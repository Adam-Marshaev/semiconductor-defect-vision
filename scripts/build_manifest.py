import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

DATASET_ROOT = Path("data/raw/carinthia-s/data")
CSV_PATH = DATASET_ROOT / "carinthia-s.csv"

OUTPUT_DIR = Path("data/processed")
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"
SUMMARY_PATH = OUTPUT_DIR / "validation_summary.json"

MASK_THRESHOLD = 128

EXPECTED_COLUMNS = {
    "image_path",
    "mask_path",
    "filename",
    "label",
}


def load_binary_mask(path: Path) -> tuple[np.ndarray, dict]:
    """Load a raw mask and convert it to a canonical binary representation."""

    with Image.open(path) as mask_file:
        original_mode = mask_file.mode
        gray = np.asarray(mask_file.convert("L"))

    raw_unique_values = np.unique(gray)
    has_intermediate_values = bool(
        np.any((raw_unique_values > 0) & (raw_unique_values < 255))
    )

    binary = (gray >= MASK_THRESHOLD).astype(np.uint8)

    metadata = {
        "mask_original_mode": original_mode,
        "mask_raw_unique_count": len(raw_unique_values),
        "mask_has_intermediate_values": has_intermediate_values,
    }

    return binary, metadata


def analyze_mask(binary: np.ndarray) -> dict:
    """Calculate geometry from a canonical binary mask."""

    defect_pixels = int(binary.sum())
    total_pixels = int(binary.size)

    if defect_pixels == 0:
        return {
            "defect_pixels": 0,
            "defect_fraction": 0.0,
            "component_count": 0,
            "bbox_x": None,
            "bbox_y": None,
            "bbox_width": None,
            "bbox_height": None,
            "empty_mask": True,
        }

    num_labels, _, _, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    ys, xs = np.nonzero(binary)

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    return {
        "defect_pixels": defect_pixels,
        "defect_fraction": defect_pixels / total_pixels,
        "component_count": num_labels - 1,
        "bbox_x": x_min,
        "bbox_y": y_min,
        "bbox_width": x_max - x_min + 1,
        "bbox_height": y_max - y_min + 1,
        "empty_mask": False,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(CSV_PATH, sep=";")

    missing_columns = EXPECTED_COLUMNS - set(metadata.columns)

    if missing_columns:
        raise RuntimeError(
            f"CSV is missing required columns: {sorted(missing_columns)}"
        )

    records = []
    errors = []

    total = len(metadata)

    for index, row in enumerate(metadata.itertuples(index=False), start=1):
        image_path = DATASET_ROOT / row.image_path
        mask_path = DATASET_ROOT / row.mask_path

        try:
            if not image_path.exists():
                raise FileNotFoundError(image_path)

            if not mask_path.exists():
                raise FileNotFoundError(mask_path)

            with Image.open(image_path) as image_file:
                image_mode = image_file.mode
                image = np.asarray(image_file.convert("L"))

            binary_mask, mask_metadata = load_binary_mask(mask_path)

            if image.ndim != 2:
                raise ValueError(
                    f"Expected 2-D image, got shape {image.shape}"
                )

            if binary_mask.ndim != 2:
                raise ValueError(
                    f"Expected 2-D binary mask, got shape {binary_mask.shape}"
                )

            if image.shape != binary_mask.shape:
                raise ValueError(
                    f"Shape mismatch: {image.shape} vs {binary_mask.shape}"
                )

            mask_stats = analyze_mask(binary_mask)

            records.append(
                {
                    "sample_id": row.filename,
                    "label": int(row.label),
                    "image_path": row.image_path,
                    "mask_path": row.mask_path,
                    "image_original_mode": image_mode,
                    "image_width": int(image.shape[1]),
                    "image_height": int(image.shape[0]),
                    "image_dtype": str(image.dtype),
                    "image_min": int(image.min()),
                    "image_max": int(image.max()),
                    "mask_width": int(binary_mask.shape[1]),
                    "mask_height": int(binary_mask.shape[0]),
                    "mask_threshold": MASK_THRESHOLD,
                    **mask_metadata,
                    **mask_stats,
                }
            )

        except (OSError, ValueError, cv2.error) as exc:
            errors.append(
                {
                    "sample_id": row.filename,
                    "error": str(exc),
                }
            )

        if index % 500 == 0 or index == total:
            print(f"Processed {index}/{total}")

    manifest = pd.DataFrame(records)
    manifest.to_csv(MANIFEST_PATH, index=False)

    class_counts = {
        str(label): int(count)
        for label, count in (
            manifest["label"].value_counts().sort_index().items()
        )
    }

    dimension_counts = {
        f"{width}x{height}": int(count)
        for (width, height), count in (
            manifest[["image_width", "image_height"]]
            .value_counts()
            .items()
        )
    }

    mask_mode_counts = {
        str(mode): int(count)
        for mode, count in (
            manifest["mask_original_mode"].value_counts().items()
        )
    }

    summary = {
        "csv_rows": len(metadata),
        "validated_samples": len(manifest),
        "validation_errors": len(errors),
        "mask_threshold": MASK_THRESHOLD,
        "class_counts": class_counts,
        "image_dimensions": dimension_counts,
        "mask_original_modes": mask_mode_counts,
        "masks_with_intermediate_values": int(
            manifest["mask_has_intermediate_values"].sum()
        ),
        "empty_masks": int(manifest["empty_mask"].sum()),
        "nonempty_masks": int((~manifest["empty_mask"]).sum()),
        "mean_defect_fraction": float(
            manifest["defect_fraction"].mean()
        ),
        "median_defect_fraction": float(
            manifest["defect_fraction"].median()
        ),
        "errors": errors,
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2)
    )

    print()
    print(f"Validated samples: {len(manifest)}")
    print(f"Validation errors: {len(errors)}")
    print(f"Manifest written to: {MANIFEST_PATH}")
    print(f"Summary written to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
