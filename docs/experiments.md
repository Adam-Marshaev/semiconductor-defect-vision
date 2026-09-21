# Experiment Results

This document records model and algorithm experiments using the frozen
Carinthia-S dataset partition.

The test partition is reserved for final evaluation after configuration and
model selection on the validation partition.

## E001 - Classical Otsu Segmentation Baseline

### Method

A classical image-processing segmentation baseline was evaluated using:

1. grayscale SEM input
2. Gaussian smoothing
3. global Otsu thresholding
4. optional threshold inversion
5. optional morphological opening

A 24-configuration grid was evaluated exclusively on the frozen validation
partition.

Selection metric:

- mean per-image Dice over the complete validation partition

### Selected Configuration

- Gaussian blur kernel: 3 x 3
- inverted Otsu threshold: yes
- morphological opening kernel: 5 x 5
- morphology iterations: 1

### Validation Results

- mean Dice: 0.455618
- median Dice: 0.486280
- mean IoU: 0.323185
- median IoU: 0.321248
- mean Dice on non-empty targets: 0.444231
- mean IoU on non-empty targets: 0.305130
- empty-target accuracy: 0.681818

There were:

- 459 validation samples
- 437 non-empty targets
- 22 empty targets

15 of the 22 empty targets were correctly predicted as empty.

### Per-Class Validation Dice

| Label | Samples | Mean Dice |
|------:|--------:|----------:|
| 1 | 5 | 0.011773 |
| 2 | 1 | 0.003507 |
| 3 | 401 | 0.432493 |
| 4 | 29 | 0.686836 |
| 5 | 1 | 0.718654 |
| 6 | 22 | 0.681818 |

Labels 2 and 5 contain only one validation example each, so their class-level
results cannot be treated as stable estimates of generalization.

### Interpretation

The classical baseline exhibits strong morphology-dependent performance.

Labels 1 and 2 are almost entirely missed by the selected thresholding
pipeline, while label 4 is segmented comparatively well.

The selected 5 x 5 morphological opening slightly reduces Dice on non-empty
targets compared with otherwise similar configurations, but substantially
improves handling of empty targets.

Without morphology, closely related configurations produced zero correctly
empty predictions.

The selected configuration correctly predicts 15 of 22 empty validation masks.
The remaining seven label-6 failures contain substantial false-positive
foreground. Based on the aggregate predicted foreground fraction, these failed
empty-mask cases predict approximately 37% foreground on average.

This demonstrates an important weakness of global intensity thresholding:
when image intensity structure violates its foreground/background assumption,
failure can be catastrophic rather than incremental.

### Test Status

The test partition has NOT been evaluated.

Configuration selection and ongoing qualitative analysis use validation data
only.

### Qualitative Validation Error Analysis

Selected best-case, worst-case, and empty-target false-positive predictions
were manually inspected.

Three major behaviors were observed.

#### Empty-Target False Positives

Several label-6 images exhibit a large block-like or rectangular separation in
the Otsu prediction despite appearing mostly uniform gray to visual inspection.

The thresholding pipeline classifies one broad image region as foreground even
though no defect is annotated.

Interpretation:

Global Otsu thresholding is sensitive to broad intensity variation across the
SEM field. It can mistake low-frequency illumination or acquisition-related
intensity structure for defect foreground.

The physical origin of this intensity structure has not been established, so
it is not currently labeled as a specific acquisition artifact.

#### Strong Predictions

The best predictions generally identify the correct defect structure and cover
most of its area.

However, some defect boundaries and thin edge regions are omitted.

This is consistent with both threshold mismatch near low-contrast boundaries
and the use of a 5 x 5 morphological opening, which can remove legitimate thin
foreground structures.

#### Worst Predictions

Several worst-performing images are predicted as almost entirely background
despite containing visible defects.

Most of these contain relatively small defects, but a large label-1 defect is
also almost completely missed.

Therefore failure cannot be explained only by defect size.

The result indicates that defect identity is not consistently represented by
one global bright/dark intensity relationship. Defects can have insufficient
global contrast, different local contrast behavior, or intensity relationships
that violate the assumptions of global Otsu thresholding.

### Consequence

A second classical baseline will test local adaptive thresholding.

The goal is to determine whether using neighborhood-relative intensity rather
than one global image threshold improves robustness to broad SEM intensity
variation and low-contrast defects.

The test partition remains untouched.

## E002 - Adaptive Local Threshold Segmentation

### Motivation

Qualitative analysis of the global Otsu baseline showed several failure modes:

- broad intensity variation could produce large false-positive regions
- low-contrast defects could be completely missed
- defect boundaries were sometimes incompletely segmented

Adaptive thresholding was evaluated to test whether neighborhood-relative
intensity could improve segmentation when global foreground/background
separation was insufficient.

### Method

Adaptive Gaussian thresholding was evaluated using a validation-only grid over:

- Gaussian blur kernel: 1, 3
- adaptive block size: 15, 31, 61
- C: 2, 5, 10
- morphology: none, opening, closing

Threshold polarity was fixed to inverted based on the prior Otsu experiment.

A total of 54 configurations were evaluated exclusively on the frozen
validation partition.

### Selected Configuration

- Gaussian blur: 3 x 3
- adaptive block size: 61
- C: 5
- inverted threshold: yes
- morphology: 3 x 3 opening

### Validation Results

- mean Dice: 0.407831
- median Dice: 0.439864
- mean IoU: 0.265803
- mean Dice on non-empty targets: 0.428362
- mean IoU on non-empty targets: 0.279184
- empty-target accuracy: 0.000000

The method failed to produce an empty prediction for all 22 empty validation
targets.

### Per-Class Mean Dice

| Label | Samples | Mean Dice |
|------:|--------:|----------:|
| 1 | 5 | 0.224076 |
| 2 | 1 | 0.540439 |
| 3 | 401 | 0.426278 |
| 4 | 29 | 0.486457 |
| 5 | 1 | 0.488935 |
| 6 | 22 | 0.000000 |

Labels 2 and 5 each contain only one validation example and therefore cannot
support stable class-level conclusions.

### Comparison with Global Otsu

Global Otsu validation mean Dice:

0.455618

Adaptive threshold validation mean Dice:

0.407831

Difference:

-0.047787

Adaptive thresholding substantially improved the observed Dice for label 1
and the single label-2 validation example, suggesting that neighborhood-relative
contrast can recover some structures missed by global thresholding.

However, performance decreased substantially for label 4 and all empty
label-6 images produced false-positive foreground.

### Interpretation

Adaptive thresholding addresses some failures caused by weak global contrast,
but introduces strong sensitivity to normal SEM texture.

Because a local threshold is computed throughout the image, the method tends
to find foreground structure even in defect-free images.

Together, E001 and E002 show that defect segmentation cannot be represented
reliably by intensity alone:

- global intensity separation misses some valid defects
- local intensity separation over-segments normal SEM texture

The global Otsu pipeline from E001 remains the selected classical-CV baseline.

### Test Status

The test partition has NOT been evaluated.

The classical baseline is frozen based only on validation results.
