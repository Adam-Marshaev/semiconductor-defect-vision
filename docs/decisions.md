# Project Decisions

This document records important technical decisions and the evidence behind
them.

## D001 - Preserve Raw Dataset Files

Decision:

Treat `data/raw/` as immutable.

Reason:

Raw files provide the source-of-truth dataset. Any normalization, resizing,
splitting, annotation transformation, or derived metadata must be
reproducible without modifying the source data.

Derived artifacts belong under `data/processed/`.

---

## D002 - Use a Canonical Dataset Manifest

Decision:

Use `data/processed/manifest.csv` as the common metadata interface for EDA,
training, classical CV, evaluation, and error analysis.

Reason:

This prevents individual scripts from independently interpreting the raw
filesystem and creates one validated sample definition.

---

## D003 - Normalize Masks to Binary at Threshold 128

Decision:

Convert masks to grayscale and define canonical foreground as:

`mask >= 128`

Reason:

The raw annotations are semantically binary but have heterogeneous PNG
storage representations.

Observed raw storage includes:

- grayscale (`L`)
- RGB
- RGBA
- 395 grayscale masks containing intermediate pixel values

Dataset-wide analysis showed that 0 and 255 dominate the annotations and
that intermediate values affect primarily boundary pixels.

Threshold sensitivity between 1 and 254 caused relatively modest changes to
foreground area.

Threshold 128 provides a deterministic midpoint-based conversion without
modifying the original masks.

---

## D004 - Preserve Original Mask Encoding Metadata

Decision:

The manifest records fields describing the original mask representation,
including storage mode and whether intermediate values were present.

Reason:

Normalization should not erase provenance.

This also enables later analysis of whether model behavior differs for masks
that originally contained soft boundary values.

---

## D005 - Do Not Use Accuracy Alone for Classification

Decision:

Classification evaluation must emphasize metrics such as macro F1 and
per-class recall rather than raw accuracy alone.

Reason:

Class distribution is extremely imbalanced.

Label 3 contains 4,008 of 4,591 samples, approximately 87.3% of the dataset.

A trivial majority-class classifier would therefore achieve misleadingly
high accuracy.

---

## D006 - Do Not Finalize Dataset Splits Yet

Decision:

Do not create the final train/validation/test split until duplicate,
near-duplicate, and grouping/leakage analysis is complete.

Reason:

Random sample-level splitting can produce optimistic evaluation if highly
similar or related SEM acquisitions appear in multiple splits.

The extremely small minority classes also make naive stratified splitting
problematic.

---

## D007 - Treat Segmentation as the Primary Modeling Task

Decision:

Semantic segmentation remains the strongest primary task.

Classification remains useful but should be interpreted cautiously.

Reason:

Several classification labels contain very few examples:

- label 2: 8 samples
- label 5: 4 samples

This makes statistically strong six-class classification conclusions
difficult.

The dataset provides pixel-level masks for all 4,591 samples, making
segmentation the more natural central task.

---

## D008 - Do Not Manually Correct Expert Ground Truth

Decision:

Preserve expert-provided Carinthia-S masks even when visual inspection suggests
that a visible feature may extend slightly beyond the annotated region.

Reason:

The project should evaluate against the published dataset's expert-validated
ground truth rather than introduce undocumented project-specific relabeling.

The single non-empty label-6 sample was tested across mask thresholds from 1
through 254. Its 50-pixel mask and 16 x 8 bounding box were invariant, proving
that the observed visual discrepancy is not caused by our thresholding rule.

---

## D009 - Near-Duplicate Samples Must Be Grouped During Splitting

Decision:

Samples determined to represent the same physical SEM defect or acquisition
field must be assigned as a group during train/validation/test splitting.

Known group:

- `6830f9ceb59a485681c6f5392493edc9`
- `b76d11f40521489ca4d2bf47804d81d1`

Reason:

Allowing contrast-varied or otherwise near-duplicate views of the same physical
structure to appear in different partitions would leak highly specific visual
information from training into validation or test data and inflate measured
generalization performance.
