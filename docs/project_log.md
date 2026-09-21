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
