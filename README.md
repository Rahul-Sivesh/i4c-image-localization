# AI-Based Restoration of Degraded Images

### i4C Hackathon — Semiconductor Image Restoration

> **An AI-based image restoration system designed to recover high-quality visual information from degraded semiconductor images.**

---

## 📌 1. Project Overview

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
