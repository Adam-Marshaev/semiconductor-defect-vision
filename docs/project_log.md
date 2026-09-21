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
