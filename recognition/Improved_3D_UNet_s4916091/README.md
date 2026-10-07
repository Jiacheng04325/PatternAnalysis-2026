# Residual 3D Improved U-Net for HipMRI Pelvic Organ Segmentation

## Project Overview

This project develops a 3D medical image segmentation prototype for automatic pelvic organ contouring in prostate radiotherapy planning.
The system segments the body, bone, bladder, rectum, and prostate from 3D HipMRI volumes.
Its intended users are medical imaging specialists who need an initial contour draft that can be reviewed and corrected before treatment planning.

The project compares a standard 3D U-Net baseline with a residual 3D Improved U-Net using deep supervision.
The main engineering question is whether improved 3D spatial context reduces clinically important boundary errors at the prostate base and apex enough to justify additional GPU memory and inference time.

## Feasibility Review

### User Need and Scope

Manual pelvic organ contouring is time-consuming and requires experienced clinical staff.
An automatic contouring assistant may reduce this workload, but inaccurate prostate, bladder, or rectum boundaries can affect radiotherapy planning.
The prototype will therefore produce draft segmentations for expert review rather than autonomous clinical decisions.

The project uses the HipMRI Study dataset located at:

```text
/home/groups/comp3710/HipMRI_Study_open
```

The implementation will use PyTorch and will be developed as a Hard Difficulty project.

### Acceptance Criteria

1. The train, validation, and test sets must have no patient overlap, and all longitudinal scans from one patient must remain in one split.
2. The proposed model must achieve a mean prostate Dice score of at least 0.70 on held-out patients.
3. Compared with the standard 3D U-Net baseline, the proposed model should improve prostate base and apex Dice by at least 0.02 or reduce prostate HD95 by at least 10 percent.
4. Peak GPU memory must remain at or below 24 GB, and inference must complete within 30 seconds per volume on one Rangpur A100 GPU.
5. The evaluation must include three to five representative failure cases with input images, ground-truth masks, predictions, false-positive regions, and false-negative regions.

If the proposed model does not provide a meaningful boundary improvement for its additional resource cost, the final recommendation will favour the simpler baseline.

### Model Choice and Course Concepts

The baseline will be a standard 3D U-Net, and the proposed Hard Difficulty model will be a residual 3D Improved U-Net with deep supervision.
Both models will use volumetric convolution, encoder-decoder stages, and skip connections to learn spatial context across MRI slices.
Residual units are intended to improve gradient propagation, while deep supervision will encourage useful predictions at multiple decoder resolutions.
A combined Dice and Cross-Entropy objective will address severe class imbalance without removing voxel-level classification supervision.

### Preliminary Feasibility Evidence

The initial audit found 211 matched MRI-label pairs from 38 patients, with one to eight longitudinal scans per patient.
All image-label pairs have matching shapes and affine matrices, all use the LPS orientation, and every label volume contains identifiers 0 to 5.
A total of 210 volumes have shape `256 x 256 x 128`, while `K019_Week1` has 144 slices in both its image and label.
The smallest class represents approximately 0.10 percent of all voxels, confirming the need for an imbalance-aware objective and foreground-aware sampling.

### Risks, Budget, and Fallback

The main risks are GPU memory limits, severe class imbalance, inconsistent physical spacing, and leakage across longitudinal scans.
The initial budget is one short CPU data check and one A100 smoke test before baseline training.
Long experiments will use Slurm batch jobs with explicit memory and time limits.
If memory use is excessive, patch size and base channel width will be reduced while retaining the residual and deep-supervision design.
If common-spacing resampling removes useful detail, native-resolution patch training will be evaluated as the fallback.
The next experiment will verify the label mapping, generate a deterministic patient-level split, and run one training and validation batch through the baseline.

## Dataset Audit

The initial audit identified 211 MRI volumes from 38 patients.
Each patient has between one and eight longitudinal scans.
The MRI directory and label directory each contain 211 files, and every image has a matching label.

All image-label pairs have matching shapes and affine matrices.
All volumes use the LPS orientation.
A total of 210 volumes have shape `256 x 256 x 128`.
One image-label pair, `K019_Week1`, has shape `256 x 256 x 144`.

The through-plane voxel spacing is approximately 1.56 mm for every volume.
The in-plane spacing has nine observed values between approximately 1.4062 mm and 1.8750 mm.
The most common spacing is approximately `1.6797 x 1.6797 x 1.56 mm`.

Every label volume contains identifiers 0 to 5.
The dataset contains background and five foreground structures: body, bone, urinary bladder, rectum, and prostate.
The exact numeric mapping of foreground identifiers will be verified against an authoritative dataset description before training.

The overall voxel ratios for labels 0 to 5 are approximately:

| Label | Voxel ratio |
|---|---:|
| 0 | 60.353% |
| 1 | 35.458% |
| 2 | 3.370% |
| 3 | 0.574% |
| 4 | 0.144% |
| 5 | 0.100% |

The severe foreground imbalance, especially for the smallest structures, will be addressed through a combined Dice and Cross-Entropy objective.

## Proposed Method

The baseline will be a standard 3D U-Net with volumetric convolution, encoder-decoder stages, and skip connections.

The proposed model will add:

- Residual convolution blocks to improve gradient propagation.
- Deep supervision at multiple decoder resolutions.
- A combined multi-class Dice and Cross-Entropy loss.
- Mixed-precision training to reduce memory use.

Both models will use the same patient split, preprocessing, augmentations, training budget, and evaluation code.
This controlled comparison will isolate the effects of residual learning and deep supervision.

Initial experiments will use 3D patches and batch size one.
Patch size and channel width will be selected through a measured GPU memory smoke test.
Full-volume inference will use overlapping sliding-window predictions if a full volume does not fit safely in memory.

## Evaluation Plan

The main metrics will be per-class Dice and Intersection-over-Union.
The project will additionally report HD95 to measure clinically important boundary errors.
Prostate performance will be analysed separately for base, middle, and apex slices.

The resource evaluation will report:

- Total parameter count.
- Peak GPU memory.
- Training time per epoch.
- Inference time per volume.

The qualitative evaluation will examine disconnected predictions, leakage into nearby organs, missed prostate tissue, and failures on base and apex slices.

## Implementation Risks and Compute Plan

The main risks are GPU memory limits, severe class imbalance, inconsistent physical spacing, and data leakage across longitudinal scans.

The initial compute budget is one A100 GPU, a short smoke test, and then a controlled baseline run before full proposed-model training.
Long training runs will use Slurm batch jobs.
Short debugging sessions will use limited interactive allocations.

If memory use is excessive, the first fallback will reduce patch size and base channel width while preserving the residual and deep-supervision design.
If class imbalance prevents learning of small organs, the loss weighting and foreground sampling strategy will be adjusted using explicit measured thresholds.
If spatial resampling removes clinically important detail, native-resolution patch training will be compared against common-spacing preprocessing.

The next experiment will verify the label mapping, generate a deterministic patient-level split, and run one training and validation batch through the standard 3D U-Net.

## Planned Repository Structure

```text
modules.py
dataset.py
train.py
predict.py
README.md
```

Additional evaluation and Slurm scripts may be added when required.

## Data Use and Acknowledgement

The dataset must not be committed to this repository.
Model checkpoints must not be committed to this repository.
The HipMRI Study data will be acknowledged and cited according to its data-use agreement.

Dataset reference:

- Dowling, J. and Greer, P. Labelled weekly MR images of the male pelvis. CSIRO Data Access Portal. DOI: 10.25919/45T8-P065.

## Artificial Intelligence Usage Disclosure

OpenAI ChatGPT and Codex were used to help interpret the assessment specification, plan the data audit, draft documentation, and review implementation decisions.

All AI-assisted outputs will be manually inspected.
Code will be tested using controlled smoke tests, unit-level checks where meaningful, held-out patient evaluation, and visual inspection of segmentation outputs.
The student remains responsible for every implementation decision, metric, result, and claim in this project.
