from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MANIFEST_PATH = Path("data/processed/manifest.csv")

REPORT_DIR = Path("reports/eda")
FIGURE_DIR = Path("reports/figures/readme")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MANIFEST_PATH)

    print("samples:", len(df))
    print("columns:", len(df.columns))

    # ---------------------------------------------------------
    # Class summary
    # ---------------------------------------------------------

    class_summary = (
        df.groupby("label")
        .agg(
            samples=("sample_id", "count"),
            empty_masks=("empty_mask", "sum"),
            mean_defect_fraction=("defect_fraction", "mean"),
            median_defect_fraction=("defect_fraction", "median"),
            mean_components=("component_count", "mean"),
            median_components=("component_count", "median"),
        )
        .reset_index()
    )

    class_summary["sample_fraction"] = (
        class_summary["samples"] / len(df)
    )

    class_summary.to_csv(
        REPORT_DIR / "class_summary.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Defect-area statistics by class
    # ---------------------------------------------------------

    defect_summary = (
        df.groupby("label")["defect_fraction"]
        .describe(
            percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        )
        .reset_index()
    )

    defect_summary.to_csv(
        REPORT_DIR / "defect_fraction_by_class.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Connected-component statistics
    # ---------------------------------------------------------

    component_summary = (
        df.groupby("label")["component_count"]
        .describe(
            percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        )
        .reset_index()
    )

    component_summary.to_csv(
        REPORT_DIR / "component_summary_by_class.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Bounding-box statistics
    # ---------------------------------------------------------

    nonempty = df[~df["empty_mask"]].copy()

    nonempty["bbox_area"] = (
        nonempty["bbox_width"] * nonempty["bbox_height"]
    )

    nonempty["bbox_area_fraction"] = (
        nonempty["bbox_area"]
        / (nonempty["image_width"] * nonempty["image_height"])
    )

    bbox_summary = (
        nonempty.groupby("label")[
            [
                "bbox_width",
                "bbox_height",
                "bbox_area_fraction",
            ]
        ]
        .agg(["mean", "median", "min", "max"])
    )

    bbox_summary.to_csv(
        REPORT_DIR / "bbox_summary_by_class.csv"
    )

    # ---------------------------------------------------------
    # Class-distribution figure
    # ---------------------------------------------------------

    class_counts = (
        df["label"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots()

    class_counts.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title("Carinthia-S Class Distribution")
    ax.set_xlabel("Numeric label")
    ax.set_ylabel("Samples")

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "class_distribution.png",
        dpi=160,
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # Defect-fraction distribution
    # ---------------------------------------------------------

    fig, ax = plt.subplots()

    ax.hist(
        df["defect_fraction"],
        bins=60,
    )

    ax.set_title("Defect Pixel Fraction")
    ax.set_xlabel("Fraction of image pixels labeled defect")
    ax.set_ylabel("Images")

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "defect_fraction_distribution.png",
        dpi=160,
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # Defect fraction by class
    # ---------------------------------------------------------

    groups = [
        df.loc[df["label"] == label, "defect_fraction"]
        for label in sorted(df["label"].unique())
    ]

    labels = sorted(df["label"].unique())

    fig, ax = plt.subplots()

    ax.boxplot(
        groups,
        tick_labels=labels,
        showfliers=False,
    )

    ax.set_title("Defect Fraction by Class")
    ax.set_xlabel("Numeric label")
    ax.set_ylabel("Defect pixel fraction")

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "defect_fraction_by_class.png",
        dpi=160,
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # Compact terminal summary
    # ---------------------------------------------------------

    print()
    print("CLASS SUMMARY")
    print("=============")
    print(
        class_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("OVERALL DEFECT FRACTION")
    print("=======================")
    print(df["defect_fraction"].describe())

    print()
    print("CONNECTED COMPONENTS")
    print("====================")
    print(df["component_count"].describe())

    print()
    print("Outputs written under:")
    print(REPORT_DIR)
    print(FIGURE_DIR)


if __name__ == "__main__":
    main()
