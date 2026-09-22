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

## E003 - Compact U-Net Segmentation Baseline

### Architecture

A compact U-Net was trained for semantic defect segmentation.

Configuration:

- input channels: 1
- output channels: 1
- base channels: 32
- trainable parameters: 7,762,465
- input resolution: 480 x 480

### Data Pipeline

Training partition:

- 3,673 samples

Validation partition:

- 459 samples

Test partition:

- not loaded or evaluated

Image preprocessing:

- grayscale input scaled to [0, 1]
- standardized using training-only statistics
- training mean: 0.37754703
- training standard deviation: 0.04998759

Training augmentation:

- random horizontal flip
- random vertical flip

Validation augmentation:

- none

### Optimization

- optimizer: AdamW
- initial learning rate: 1e-3
- weight decay: 1e-4
- batch size: 8
- loss: 0.5 BCE + 0.5 Dice loss
- maximum epochs: 20
- validation model-selection metric: mean per-image Dice
- prediction threshold: 0.5
- random seed: 42

ReduceLROnPlateau reduced the learning rate during training:

- 1e-3
- 5e-4
- 2.5e-4

### Best Validation Result

Best epoch:

20

Metrics:

- mean Dice: 0.949786
- mean IoU: approximately 0.9149
- non-empty mean Dice: approximately 0.9473
- empty-target accuracy: 1.000
- validation loss: 0.0307

All 22 empty validation targets were predicted empty at the 0.5 prediction
threshold.

Peak allocated GPU memory:

- 5.042 GiB

GPU:

- NVIDIA Tesla V100-SXM2-32GB

### Comparison with Classical Baseline

Selected global Otsu baseline validation Dice:

0.455618

Compact U-Net validation Dice:

0.949786

Absolute Dice improvement:

+0.494168

The learned model therefore substantially outperformed both classical
intensity-thresholding baselines on the same frozen validation partition.

### Test Status

The test partition has NOT been evaluated.

The saved best checkpoint is:

`models/unet_baseline_best.pt`

Further analysis will use validation data before any final held-out test
evaluation.

### E003 Qualitative Error Analysis

The best U-Net checkpoint was evaluated per image on the frozen validation
partition.

Validation summary:

- mean Dice: 0.949786
- mean IoU: 0.914906
- non-empty mean Dice: 0.947258
- empty-target accuracy: 1.000000

Per-class mean Dice:

- label 1: 0.782835 (5 samples)
- label 2: 0.905150 (1 sample)
- label 3: 0.946832 (401 samples)
- label 4: 0.982951 (29 samples)
- label 5: 0.947489 (1 sample)
- label 6: 1.000000 (22 samples)

The lowest-scoring label-3 samples frequently showed very high precision and
very low recall. Qualitative inspection showed that several such cases involved
tight predictions around visually apparent defects while the reference mask
covered a broader region.

Label-1 predictions showed a related pattern: the reference mask can cover a
general clustered region while the network traces individual visible defect
structures more tightly.

Some genuine false negatives were also observed.

These findings reinforce the need to interpret Dice together with visual
inspection and precision/recall rather than treating every reference-mask
disagreement as the same type of failure.

The supplied masks remain unchanged and continue to serve as the canonical
quantitative evaluation targets.

## E004 - SegFormer-B0 Segmentation Baseline

SegFormer-B0 was fine-tuned using the frozen Carinthia-S training and validation
partitions.

Configuration:

- pretrained encoder: nvidia/mit-b0
- trainable parameters: 3,714,401
- input: grayscale SEM replicated to three channels
- input resolution: 480 x 480
- batch size: 8
- optimizer: AdamW
- initial learning rate: 1e-4
- weight decay: 1e-4
- loss: 0.5 BCE + 0.5 Dice
- prediction threshold: 0.5
- training augmentation: horizontal and vertical flips
- model selection metric: validation mean Dice
- maximum epochs: 20

Training was interrupted after epoch 7 and resumed from the saved checkpoint.
Resume support was subsequently added to the training pipeline.

ReduceLROnPlateau reduced the learning rate from:

- 1e-4
- to 5e-5

Best result occurred at epoch 19.

Best validation metrics:

- mean Dice: 0.955009
- mean IoU: approximately 0.9241
- non-empty mean Dice: approximately 0.9527
- empty-target accuracy: 1.000

Peak allocated GPU memory:

- 2.455 GiB

Total recorded epoch training time:

- approximately 24.5 minutes

Comparison:

- Otsu validation Dice: 0.455618
- U-Net validation Dice: 0.949786
- SegFormer-B0 validation Dice: 0.955009
- SegFormer improvement over U-Net: +0.005223
- SegFormer improvement over Otsu: +0.499391

The test partition remains untouched.

## E004 validation error analysis — U-Net vs SegFormer-B0

SegFormer-B0 and U-Net were evaluated on the same frozen 459-image
validation split at a probability threshold of 0.5. The test split remained
untouched.

Overall validation performance:

- U-Net mean Dice: 0.949786
- SegFormer-B0 mean Dice: 0.955009
- Dice difference: +0.005223 for SegFormer
- Nonempty mean Dice difference: +0.005486

Per-class mean Dice differences favored SegFormer for classes 1-4, were equal
for empty-mask class 6, and favored U-Net on the single class-5 validation
sample. Class-1 and class-5 conclusions are limited by their very small
validation sample counts.

Across the 437 nonempty validation images, SegFormer improved Dice on 310
samples and regressed on 127.

Qualitative inspection showed that SegFormer often improved recall by covering
more of the annotated defect region. On several class-1 examples, U-Net
produced tight masks around individual small defect structures while SegFormer
produced broader, sometimes connected masks that more closely matched the
ground-truth annotation, which itself represented a broader defect region.

For label-3 improvements, SegFormer sometimes recovered visible portions of
the defect missed by U-Net.

Most regressions were visually small differences. SegFormer was sometimes
tighter around the main annotated region. In one visually inspected case,
U-Net detected a small satellite feature that SegFormer omitted, but that
feature was also absent from the ground-truth mask; therefore this example is
not evidence that U-Net better recovers annotated satellite defects.

Defect-area quartile analysis showed that SegFormer achieved higher mean Dice
in every validation target-area quartile:

- Q1 smallest: 0.910439 -> 0.915688
- Q2 small:    0.957777 -> 0.965415
- Q3 large:    0.955563 -> 0.958890
- Q4 largest:  0.965592 -> 0.971324

Thus the validation evidence does not support a general claim that U-Net
performs better on small-defect images.

Some metric differences reflect agreement with annotation morphology rather
than an unambiguous difference in visually ideal defect boundaries. Canonical
ground-truth masks were not modified.

## Validation probability-threshold analysis

A validation-only probability-threshold sweep from 0.10 through 0.90 in
increments of 0.05 was performed for both U-Net and SegFormer-B0. The test
split remained untouched.

U-Net:
- threshold 0.50 mean Dice: 0.949786
- best tested threshold: 0.20
- best mean Dice: 0.950336
- improvement over 0.50: +0.000550

SegFormer-B0:
- threshold 0.50 mean Dice: 0.955009
- best tested threshold: 0.45
- best mean Dice: 0.955049
- improvement over 0.50: +0.000040

Empty-mask accuracy remained 1.0 for both models across every tested threshold.

Lowering the U-Net threshold increased recall while reducing precision, but the
resulting mean-Dice gain was only 0.00055. SegFormer performance was especially
flat around 0.40-0.55.

Because threshold optimization produced negligible validation improvement and
could encourage overfitting to the validation split, the common probability
threshold of 0.50 was retained for both models.

## V100 GPU-resident segmentation inference benchmark

U-Net and SegFormer-B0 were benchmarked on the NVIDIA Tesla V100 using
480x480 inputs. Measurements isolate GPU-resident inference and include model
forward pass, sigmoid, and binary thresholding. Disk I/O, image decoding,
host-to-device transfer, and CPU preprocessing are excluded.

Hardware during this benchmark:
- NVIDIA Tesla V100-SXM2-32GB
- 150 W power limit
- PCIe Gen3 x8 negotiated link
- zero PCIe AER correctable errors before and after the benchmark

Each configuration used 20 warmup iterations followed by 50 timed iterations.

Key results:

Batch 1:
- U-Net FP32: 8.303 ms, 120.4 images/s
- U-Net FP16 autocast: 5.107 ms, 195.8 images/s
- SegFormer-B0 FP32: 5.370 ms, 186.2 images/s
- SegFormer-B0 FP16 autocast: 4.300 ms, 232.6 images/s

Batch 32:
- U-Net FP32: 203.899 ms/batch, 156.9 images/s
- U-Net FP16 autocast: 110.767 ms/batch, 288.9 images/s
- SegFormer-B0 FP32: 113.994 ms/batch, 280.7 images/s
- SegFormer-B0 FP16 autocast: 62.066 ms/batch, 515.6 images/s

At batch 32, SegFormer-B0 delivered approximately 1.79x the throughput of
U-Net in both FP32 and FP16 autocast. FP16 autocast improved batch-32
throughput by approximately 1.84x for both architectures.

SegFormer-B0 also has fewer parameters (3.71M vs 7.76M) and higher validation
Dice (0.9550 vs 0.9498), making it the current leading deployment candidate.

Peak allocated memory did not strictly follow parameter count. SegFormer used
less memory than U-Net for FP32 batch-32 inference, but more memory under FP16
autocast. This indicates that activation/workspace requirements are important
in addition to model parameter size.

These GPU-resident results should not be interpreted as complete end-to-end
application latency. CPU preprocessing, image decoding, data transfer, and
postprocessing will be benchmarked separately.

## Mixed-precision validation

FP16 automatic mixed precision was evaluated on the complete frozen validation
split for both segmentation models before selecting a deployment precision.
The test split remained untouched.

U-Net:
- FP32 mean Dice: 0.949786
- FP16 mean Dice: 0.949794
- Dice difference: +0.000008
- FP16-vs-FP32 binary prediction disagreement: 0.000161%

SegFormer-B0:
- FP32 mean Dice: 0.955009
- FP16 mean Dice: 0.955007
- Dice difference: -0.000002
- FP16-vs-FP32 binary prediction disagreement: 0.000191%

Empty-mask accuracy remained 1.0 for both models.

These results show effectively equivalent segmentation quality between FP32
and FP16 autocast at the selected 0.5 probability threshold. Combined with
the substantial V100 inference speedups measured in the GPU benchmark, FP16
autocast is the preferred deployment precision for subsequent inference
engineering.

## End-to-end segmentation inference benchmark

A warm-cache single-image application benchmark was performed using the
production SegmentationPredictor interface. Model initialization was excluded
because a deployed inference service would load the model once and reuse it.

The measured path included:

JPEG file read and decode
-> grayscale conversion
-> NumPy/tensor preparation
-> model-specific normalization
-> host-to-device transfer
-> model inference
-> sigmoid and probability threshold
-> device-to-host probability transfer
-> binary mask returned to the caller

Each configuration used 25 warmup images followed by 200 timed validation
images. The test split remained untouched.

Results:

U-Net FP32:
- mean latency: 10.200 ms
- median latency: 10.148 ms
- p95 latency: 11.036 ms
- throughput: 98.0 images/s

U-Net FP16 autocast:
- mean latency: 6.669 ms
- median latency: 6.817 ms
- p95 latency: 6.960 ms
- throughput: 149.9 images/s

SegFormer-B0 FP32:
- mean latency: 6.636 ms
- median latency: 6.522 ms
- p95 latency: 7.305 ms
- throughput: 150.7 images/s

SegFormer-B0 FP16 autocast:
- mean latency: 6.293 ms
- median latency: 6.254 ms
- p95 latency: 7.168 ms
- throughput: 158.9 images/s

SegFormer-B0 FP16 autocast is the current preferred deployment configuration.
It combines the highest validation segmentation quality with the lowest
measured mean end-to-end single-image latency.

The gap between GPU-only and end-to-end latency demonstrates measurable
preprocessing, transfer, and result-materialization overhead, but the
application is not dominated by those costs.

These measurements are warm-cache workstation measurements rather than raw
storage-I/O benchmarks.

The V100 remained at a 150 W power limit and recorded zero PCIe AER errors
before and after the benchmark.

## Self-contained SegFormer deployment initialization

The SegFormer deployment path was changed so that production inference no
longer initializes the architecture from the Hugging Face Hub.

The exact binary SegFormer-B0 architecture configuration used for training was
exported to:

configs/segformer_b0_binary_config.json

Training behavior remains unchanged and can still initialize from the
nvidia/mit-b0 pretrained encoder. Production inference instead constructs the
architecture from the local JSON configuration and then loads the trained
checkpoint state dictionary.

Parity was verified on validation sample
a388040b307b4184a532d3e5dda02ffc:

- previous/reference FP32 predicted pixels: 22817
- local-config FP32 predicted pixels: 22817
- previous/reference FP16 predicted pixels: 22809
- local-config offline FP16 predicted pixels: 22809

FP16 inference was additionally tested with HF_HUB_OFFLINE=1 and completed
successfully. The returned probability map remained float32.

The production inference path therefore requires the Transformers Python
implementation but does not require network access or runtime model downloads.

## Final frozen SegFormer-B0 test evaluation

After completing all model selection, threshold analysis, mixed-precision validation, inference engineering, and deployment benchmarking, the frozen SegFormer-B0 configuration was evaluated once on the previously untouched 459-image test split.

The final configuration was:

- SegFormer-B0
- best validation-selected checkpoint from epoch 19
- local architecture configuration and local trained checkpoint
- FP16 automatic mixed precision
- probability threshold 0.50
- 480x480 grayscale input
- no additional test-driven tuning

Final test performance:

- mean Dice: 0.957080
- mean IoU: 0.925138
- nonempty mean Dice: 0.954919
- empty-mask accuracy: 1.000000

For comparison, the same frozen FP16 configuration achieved validation mean Dice 0.955007 and nonempty mean Dice 0.952742. Test performance therefore remained consistent with validation performance and showed no evidence of a substantial generalization drop.

Test mean Dice by class:

- class 1: 0.841441 across 6 samples
- class 2: 0.831222 across 1 sample
- class 3: 0.955995 across 400 samples
- class 4: 0.968714 across 29 samples
- class 5: 0.929305 across 1 sample
- class 6: 1.000000 across 22 samples

Class-2 and class-5 results should not be interpreted as robust estimates of class-level performance because each class contains only one test sample.

The remaining low-Dice cases include both under-segmentation and over-segmentation failure modes. No model, threshold, preprocessing, or training changes were made after observing the test results.

## Dockerized GPU inference reproducibility

A production inference container was built using Python 3.10, PyTorch
2.14.0+cu126, Transformers 5.17.0, and the local SegFormer-B0 architecture
configuration.

The trained checkpoint is mounted into the container at runtime rather than
baked into the image.

GPU access through NVIDIA Container Toolkit was verified inside the container:

- PyTorch: 2.14.0+cu126
- CUDA available: true
- GPU: Tesla V100-SXM2-32GB
- compute capability: 7.0

The container was configured with HF_HUB_OFFLINE=1 and
TRANSFORMERS_OFFLINE=1, confirming that production inference does not require
runtime Hugging Face Hub access.

Reproducibility was verified on validation sample
a388040b307b4184a532d3e5dda02ffc.

FP16 predicted foreground pixels:

- host production predictor: 22809
- host CLI: 22809
- Docker GPU CLI: 22809

The Docker-generated mask was a 480x480 uint8 PNG containing only values
0 and 255 and exactly 22809 foreground pixels.

PCIe AER counters remained at zero after the containerized GPU inference test.
