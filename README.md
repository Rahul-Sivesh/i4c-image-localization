# AI-Based Restoration of Degraded Images

### i4C Hackathon — Semiconductor Image Restoration

> **An AI-based image restoration system designed to recover high-quality visual information from degraded semiconductor images.**

---

##  1. Project Overview

Semiconductor manufacturing and inspection systems rely on high-quality images to observe microscopic structures and identify fine-scale features.

However, images captured during the imaging process can suffer from:

- Noise and speckle degradation
- Reduced spatial resolution
- Loss of fine structural details
- Blur and acquisition-related degradation
- Poor image quality under challenging imaging conditions

These degradations can make important semiconductor features difficult to observe and analyze.

Our project addresses this challenge using a **deep-learning-based image restoration pipeline**.

The system learns from paired degraded and ground-truth images and attempts to reconstruct a higher-quality image that is closer to the original ground-truth image.

### Overall Objective

The primary objective is to develop an AI model that can:

1. Accept a degraded low-resolution image.
2. Restore the lost image information.
3. Increase the spatial resolution by 2×.
4. Reduce unwanted noise.
5. Preserve important structural details.
6. Produce a restored image visually and quantitatively closer to the ground truth.

The overall concept is:

```text
Degraded / Low-Resolution Image
              │
              ▼
      Deep Learning Model
              │
              ▼
    Restored High-Resolution
           Image
              │
              ▼
       Ground Truth
       Comparison
              │
              ▼
    PSNR / SSIM / LPIPS
```


##  2. Problem Statement

### AI-Based Restoration of Degraded Images

Semiconductor imaging systems can produce images that are degraded by noise and reduced spatial resolution. Such degradation can obscure important structural information and reduce the usefulness of the images for inspection and analysis.

The objective of this project is to develop an AI-based restoration model that takes a degraded image as input and reconstructs a higher-quality image that is closer to its corresponding ground-truth image.

The restoration task focuses on addressing three major degradation aspects:

### 1. Speckle / Noise Degradation

Noise present in the input image can hide fine structural details and reduce visual clarity.

The model is trained to suppress unwanted noise while preserving meaningful image structures.

### 2. Gaussian-Type Noise

Additional noise can introduce unwanted intensity variations throughout the image.

The restoration network learns to distinguish useful image information from degradation and reconstruct a cleaner representation.

### 3. Super-Resolution

The degraded input images used in the project have a spatial resolution of:

```text
128 × 128
```
##  3. Dataset & Data Preparation

The project uses paired image data consisting of degraded low-resolution images and their corresponding high-quality ground-truth images.

Each valid pair contains:

```text
NoisyLR Image  →  Ground Truth Image
  128 × 128          256 × 256
```
##  4. Bicubic ×2 Baseline

Before developing the deep-learning restoration model, a conventional **Bicubic ×2 interpolation** method was implemented as the baseline.

The purpose of the baseline is to establish a reference performance and determine whether the proposed deep-learning approach provides a meaningful improvement over conventional image upscaling.

---

###  Baseline Pipeline

The degraded input image is first resized from:

```text
128 × 128
```
##  5. RestorationNet V1 — Proposed Model

To overcome the limitations of conventional Bicubic interpolation, we developed a lightweight convolutional image-restoration network named **RestorationNet V1**.

The model is designed to learn the mapping:

```text
NoisyLR 128 × 128
        │
        ▼
  RestorationNet V1
        │
        ▼
Restored Image 256 × 256
```
##  6. Training Methodology

RestorationNet V1 was trained using supervised learning with paired degraded and ground-truth images.

The model learns a mapping from:

```text
NoisyLR Image
128 × 128
     │
     ▼
RestorationNet V1
     │
     ▼
Predicted Restored Image
256 × 256
     │
     ▼
Ground Truth
256 × 256
```
##  7. Model V1 — Test Results

After training, the best-performing RestorationNet V1 checkpoint was evaluated on the held-out test set.

The test set contains:

```text
320 image pairs
```
##  8. Model V2 — Loss Function Improvement

Although RestorationNet V1 produced a significant improvement over the Bicubic baseline, visual evaluation indicated that some restored images could become overly smooth.

To address this limitation, a second training experiment was designed with an improved loss formulation.

The objective of V2 was to combine:

- Pixel-level accuracy
- Structural similarity

---

###  V1 vs V2 Loss Strategy

| Version | Loss Strategy | Main Objective |
|---|---|---|
| **V1** | L1 Loss | Pixel-level reconstruction |
| **V2** | L1 + SSIM Loss | Pixel accuracy + structural preservation |

The V2 approach combines the strengths of both losses.

---

###  V2 Loss Function

The proposed V2 loss is formulated as:

```text
L_total = L1 + λ × L_SSIM
```
##  9. Evaluation Metrics

The restoration quality of the proposed model was evaluated using three complementary image-quality metrics:

1. **PSNR**
2. **SSIM**
3. **LPIPS**

Using multiple metrics provides a more complete evaluation because image restoration quality cannot be represented by a single numerical measure.

---

### 9.1 PSNR — Peak Signal-to-Noise Ratio

PSNR measures the similarity between the restored image and the ground-truth image based on pixel-level reconstruction error.

The PSNR value is expressed in decibels (dB).

```text
Higher PSNR → Better pixel-level reconstruction
```
##  10. Visual Results & Qualitative Analysis

In addition to quantitative evaluation, visual comparisons were performed to understand how effectively the proposed model restores degraded images.

The restored output from RestorationNet V1 was compared with both the Bicubic baseline and the corresponding ground-truth image.

---

###  Visual Comparison Pipeline

The images were compared in the following order:

```text
NoisyLR Input
     │
     ▼
Bicubic ×2
     │
     ▼
RestorationNet V1
     │
     ▼
Ground Truth
```
##  11. Inference & Deployment

After training and evaluation, the trained restoration model is prepared for standalone inference.

The inference stage is designed to take degraded test images as input, process them using the trained restoration network, and generate restored outputs.

---

###  Inference Pipeline

```text
Test NoisyLR Images
        │
        ▼
   Load .npy Files
        │
        ▼
   Normalization
        │
        ▼
   Trained Model
        │
        ▼
RestorationNet V1/V2
        │
        ▼
   2× Upscaling
        │
        ▼
Post-processing
        │
        ▼
Restored Images
        │
        ▼
   Output Directory
```

##  12. Limitations & Future Scope

Although RestorationNet V1 demonstrated a significant improvement over the Bicubic baseline, several areas remain for further improvement.

---

### 12.1 Current Limitations

#### Fine-Detail Preservation

Some restored outputs may appear smoother than the corresponding ground-truth images.

This indicates that further improvements are required for recovering very fine image details.

---

####  Degradation Generalization

The model was trained using the degradation characteristics represented in the available dataset.

Performance on degradation types or distributions that are significantly different from the training data may vary.

---

####  Scale Generalization

The validated training configuration is:

```text
128 × 128  →  256 × 256
```
##  13. Conclusion

This project presents an AI-based approach for restoring degraded semiconductor images using deep learning.

The development process progressed systematically from dataset analysis and conventional baseline evaluation to the design, training and evaluation of a dedicated restoration network.

The overall journey was:

```text
Dataset Analysis
       ↓
Dataset Verification
       ↓
Bicubic ×2 Baseline
       ↓
RestorationNet V1
       ↓
Model Training
       ↓
Test Evaluation
       ↓
PSNR / SSIM / LPIPS Analysis
       ↓
V2 Loss Function Experiment
       ↓
Inference & Deployment Preparation
```
##  14. Team & Contact

### Team Byte Forge

We are a multidisciplinary student team working on AI-based image restoration for semiconductor imaging applications.

| S. No. | Member Name | Role | Email |
|:---:|---|---|---|
| 1 | **Indra Priyadharshani M G** | Team Lead | indrapriyadharhanimg.ec24@bitsathy.ac.in |
| 2 | **Praaveen Hari G S** | Model Development | praaveenharigs.ec24@bitsathy.ac.in |
| 3 | **Rahul Sivesh S** | Data & Evaluation | rahulsiveshs.ec24@bitsathy.ac.in |
| 4 | **Hari Haran V T** | Documentation & Presentation | hariharanvt.ec25@bitsathy.ac.in |

---


### **AI-Based Restoration of Degraded Images**


> *Restoring degraded images with AI for clearer and more reliable semiconductor imaging.*
