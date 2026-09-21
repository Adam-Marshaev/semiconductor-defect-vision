# Project Log

## 2026-09-21 - Environment and Repository Setup

Configured Ubuntu ML development environment.

Verified:

- Ubuntu 22.04.5 LTS
- Python 3.10.12
- PyTorch 2.14.0+cu126
- CUDA runtime 12.6
- NVIDIA Tesla V100 SXM2 32GB
- compute capability 7.0
- CUDA tensor computation successful

Created modular repository structure with:

- `src/`
- `scripts/`
- `tests/`
- `configs/`
- `data/`
- `reports/`
- `models/`
- `notebooks/`

Environment tests pass.

## 2026-09-21 - Carinthia-S Acquisition

Downloaded official Carinthia-S artifacts from Zenodo.

Verified artifact MD5 hashes.

Extracted dataset locally under:

`data/raw/carinthia-s/`

Confirmed:

- 4,591 images
- 4,591 masks
- one-to-one filename-stem matching
- no missing image/mask pairings
- all inspected files readable

## 2026-09-21 - Metadata Audit

Discovered that `carinthia-s.csv` is semicolon-delimited.

CSV contains:

- image path
- mask path
- filename/sample identifier
- numeric label

Confirmed strong class imbalance:

- label 1: 55
- label 2: 8
- label 3: 4,008
- label 4: 289
- label 5: 4
- label 6: 227

## 2026-09-21 - Mask Representation Audit

Initial validator incorrectly assumed every mask was a 2-D `{0,255}` PNG.

Audit found:

- 4,587 L-mode masks
- 2 RGB masks
- 2 RGBA masks
- 395 grayscale masks with intermediate boundary values
- all masks are 480 x 480

Performed threshold-sensitivity analysis.

Selected grayscale threshold >=128 as the canonical binary-mask
normalization rule.

Raw masks remain unchanged.

## 2026-09-21 - Manifest Validation

Implemented:

`scripts/build_manifest.py`

Generated:

- `data/processed/manifest.csv`
- `data/processed/validation_summary.json`

Final validation:

- 4,591 / 4,591 samples validated
- 0 validation errors
- 226 empty masks
- 4,365 non-empty masks
- mean defect fraction: ~2.58%
- median defect fraction: ~1.69%

Next:

Begin exploratory data analysis and leakage/duplicate investigation before
finalizing train/validation/test splits.

## 2026-09-21 - Initial EDA and Duplicate Audit

Generated class-level and dataset-level EDA from the canonical manifest.

Key findings:

- label 3 represents approximately 87.3% of samples
- labels 1 and 4 have substantially larger defect areas than label 3
- label 1 has more connected defect components on average
- median defect area is approximately 1.69% of image pixels
- maximum observed defect fraction is approximately 84.6%
- maximum connected-component count is 45
- 226 of 227 label-6 masks are empty

Performed decoded-image SHA-256 duplicate audit:

- 4,591 images
- 4,591 unique hashes
- zero exact duplicate groups

Next:

- inspect segmentation and geometry outliers
- investigate the single non-empty label-6 sample
- perform near-duplicate analysis
- investigate dataset grouping/leakage risk
- finalize train/validation/test methodology only afterward

## 2026-09-21 - Qualitative Outlier Review

Visually reviewed selected EDA outliers.

Findings:

- the largest-mask sample represents a genuine large defect occupying most of
  the SEM field of view
- the 45-component sample contains many visually distinct defects, supporting
  the connected-component result
- the single non-empty label-6 sample contains a very small visible feature
- the visible extent of that feature appears slightly larger than the canonical
  mask-derived bounding box

The label-6 case will receive targeted raw-mask threshold analysis before the
mask normalization policy is considered fully finalized.

No geometric outliers have been removed.

## 2026-09-21 - Leakage Audit Completed

Completed exact-duplicate and near-duplicate analysis.

Results:

- zero exact decoded-image duplicates
- coarse similarity search performed across the full dataset
- defect-only and per-class similarity searches performed
- strongest candidate pairs manually reviewed
- one confirmed two-image near-duplicate group identified in label 4
- all other reviewed high-similarity candidates represented distinct physical
  images

The confirmed pair has been recorded as a leakage group and must remain within
one dataset partition.

The project is now ready to define a reproducible train/validation/test split.

## 2026-09-21 - Dataset Split Frozen

Created deterministic leakage-aware train/validation/test partitions using
`scripts/build_splits.py`.

Final sizes:

- train: 3,673
- validation: 459
- test: 459

All 4,591 samples are assigned exactly once.

All six labels are represented in every partition.

The known two-image near-duplicate group was kept entirely within the training
partition.

Random seed: 42.

The resulting partition definition is stored in:

`data/processed/splits.csv`

Future models will consume this frozen split rather than generating their own
random partitions.

## 2026-09-21 - Classical Segmentation Baseline

Implemented and unit-tested an OpenCV Otsu segmentation baseline.

Evaluated 24 preprocessing/threshold/morphology configurations on the frozen
validation partition only.

Selected configuration:

- 3 x 3 Gaussian blur
- inverted Otsu threshold
- 5 x 5 morphological opening

Validation performance:

- mean Dice: 0.4556
- mean IoU: 0.3232
- non-empty mean Dice: 0.4442
- empty-target accuracy: 68.2%

Performance varies strongly by defect class, with near-zero Dice for labels 1
and 2 and substantially stronger performance for label 4.

The test partition remains untouched.

Next:

Perform qualitative validation error analysis before finalizing the classical
baseline.

## 2026-09-21 - Otsu Error Analysis

Performed qualitative inspection of validation predictions from the selected
classical Otsu baseline.

Observed:

- broad block-like false-positive regions on some empty label-6 images
- generally correct but incomplete defect coverage in strong predictions
- missing boundary pixels in otherwise good segmentations
- near-complete misses on several small defects
- near-complete failure on at least one large label-1 defect

The failures show that global intensity thresholding is not sufficient to
represent all defect morphologies.

Next:

Evaluate a local adaptive-thresholding baseline on validation data only.

## 2026-09-21 - Adaptive Threshold Baseline

Evaluated 54 adaptive Gaussian threshold configurations on validation data.

Best adaptive validation mean Dice:

- 0.4078

This was below the selected global Otsu baseline:

- Otsu: 0.4556
- adaptive: 0.4078

Adaptive thresholding improved some low-contrast defect cases but generated
foreground on every empty validation image.

Conclusion:

Local intensity information alone is also insufficient for robust segmentation.

The Otsu configuration remains the selected classical-CV baseline.

Next:

Begin the learned segmentation phase using PyTorch while preserving the same
frozen dataset split and evaluation definitions.

## 2026-09-21 - Compact U-Net Baseline Trained

Completed the first learned segmentation experiment.

Compact U-Net:

- 7.76M parameters
- 20 epochs
- AdamW
- BCE + Dice loss
- batch size 8
- FP32 training
- horizontal/vertical flip augmentation

Best result occurred at epoch 20:

- validation Dice: 0.9498
- validation IoU: 0.9149
- non-empty Dice: 0.9473
- empty-target accuracy: 100%

This improves validation Dice by approximately 0.494 over the selected Otsu
baseline.

Peak allocated V100 memory was approximately 5.04 GiB.

The test partition remains untouched.

Next:

Perform per-image and per-class validation error analysis using the saved best
checkpoint.
