AFRMNet: Adaptive Feature Regulation and Matching Network for Dense Workpiece Detection in Industrial Scenes

Official repository for the research paper:

AFRMNet: Adaptive Feature Regulation and Matching Network for Dense Workpiece Detection in Industrial Scenes

This work investigates dense industrial workpiece detection under challenging conditions, including severe stacking, complex background interference, metallic reflections, and ambiguous object boundaries.

Overview

Accurate detection of densely arranged industrial workpieces is essential for automated inspection, workpiece counting, and intelligent manufacturing.

However, dense industrial scenes present several challenges:

Severe object stacking and overlap.
Complex background interference.
Ambiguous object boundaries.
Similar localization quality among neighboring candidate predictions.
Unreliable positive-sample assignment during detector training.
Difficulty in preserving fine-grained boundary information while modeling broader contextual dependencies.

To address these challenges, we propose AFRMNet (Adaptive Feature Regulation and Matching Network), a unified framework that jointly improves feature representation and positive-sample assignment for dense industrial workpiece detection.

Methodology

AFRMNet integrates three key components:

1. Gaussian-Enhanced Dynamic Matching (GEDM)

GEDM is designed to improve positive-sample assignment in densely stacked industrial scenes.

It incorporates Gaussian-modulated IoU and center-distance priors, together with IoU-guided adaptive weighting, to distinguish candidate predictions with comparable overlap quality but different geometric reliability.

This design aims to provide more reliable positive-sample supervision during detector training.

2. Instance-Aware Fusion Module (IAFM)

IAFM enhances cross-scale feature aggregation by adaptively recalibrating spatial and channel responses.

It is designed to:

Reduce irrelevant background interference.
Suppress inter-instance feature coupling.
Improve the representation of workpiece regions.
Preserve instance-level and boundary-related information during feature fusion.
3. Cross-Granularity Parallel Attention (CGPA)

CGPA is integrated into the Adaptive Regulation Detection Head (ARD-Head).

It employs parallel dilated-convolution branches and spatial-channel interactions to capture multi-scale contextual information while preserving fine-grained boundary cues.

The module aims to improve feature discriminability in densely arranged workpiece scenes.

Dataset

Experiments are conducted on the WPCD (Workpiece Counting and Detection) dataset.

The dataset contains 1,036 high-resolution images, with an average of more than 117 annotated workpieces per image.

The dataset is divided into:

Training images: 718
Test images: 318

The dataset is used for academic research in dense industrial workpiece detection.

Please refer to the original dataset source for access and usage information.

Experimental Results

The proposed AFRMNet is evaluated under different density levels using the WPCD dataset.

The evaluation includes:

Comparison with representative object detectors.
Comparison with alternative attention mechanisms.
Comparison with different sample-assignment strategies.
Comparison with feature-fusion modules.
Component-wise ablation studies.
Qualitative feature-response analysis.

The reported results demonstrate the effectiveness of AFRMNet for dense industrial workpiece detection, particularly in challenging density levels.

Detailed quantitative results and analysis are provided in the associated manuscript.

Code and Model Availability

The complete source code, model configurations, trained weights, and instructions for reproducing the experiments are currently under preparation.

The complete implementation and associated research materials will be released after the manuscript is accepted for publication.

The repository will be updated with the following materials in a future release:

AFRMNet source code.
GEDM implementation.
IAFM implementation.
CGPA implementation.
Model configuration files.
Training and evaluation scripts.
Pretrained model weights.
Reproduction instructions.
Citation

If you find this work useful, please cite the associated paper after its publication.

The citation information will be updated upon publication.

Acknowledgments

This work was supported by the relevant research funding agencies and research facilities acknowledged in the associated manuscript.

Status

Under development.

The repository currently provides project information only. The complete source code and trained models will be released in a future update.
