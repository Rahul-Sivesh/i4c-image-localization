#!/usr/bin/env python
"""
run.py — Semicon Hackathon 2026 submission entry point.

Usage:
    python run.py <input-dir> <output-dir>

Reads every .npy file from <input-dir>, restores it with RestorationNetV1/V2
(2x super-resolution + denoising), and writes one restored .npy file per
input file into <output-dir>, using the same filename.

No internet access, API keys, or manual configuration are required. The
model weights are loaded from a local file bundled with this repository.
"""

import os
import sys
import glob

import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Path to the bundled model weights (relative to this script's location,
# so it works regardless of the current working directory).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "weights", "model_v2_best.pth")

# Normalization constants used at training time (must match training).
NOISY_MEAN = 0.433536
NOISY_STD = 0.284787


# --------------------------------------------------------------------------
# Model definition (must match the architecture used for the saved weights)
# --------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class RestorationNetV1(nn.Module):
    """Same architecture used for both V1 and V2 checkpoints (V2 differs
    only in the loss function used during training, not the architecture)."""

    def __init__(self, num_features=64, num_blocks=8):
        super().__init__()
        self.head = nn.Conv2d(1, num_features, 3, padding=1)
        self.body = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)]
        )
        self.body_conv = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )
        self.tail = nn.Conv2d(num_features, 1, 3, padding=1)

    def forward(self, x):
        features = self.head(x)
        body = self.body(features)
        body = self.body_conv(body)
        features = features + body
        features = self.upsample(features)
        output = self.tail(features)
        return torch.sigmoid(output)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def load_model(device):
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Model weights not found at {CHECKPOINT_PATH}. "
            "Make sure weights/model_v2_best.pth is included in the repository."
        )

    model = RestorationNetV1().to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

    # Support both a raw state_dict and a dict wrapping it under
    # "model_state_dict" (matches how the checkpoint was saved).
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval()
    return model


def restore_single(model, lr_array, device):
    """Run restoration on a single (H, W) or (H, W, 1) NoisyLR array.
    Returns a float32 (H2, W2) array in [0, 1] with no NaN/Inf."""

    arr = np.asarray(lr_array, dtype=np.float32)

    # Squeeze a trailing channel dim of size 1 down to (H, W).
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D (H, W) or (H, W, 1) array, got shape {arr.shape}")

    # Inputs are expected to already be roughly in [0, 1] (as in training data).
    # Normalize using the same statistics used during training.
    arr_norm = (arr - NOISY_MEAN) / NOISY_STD

    tensor = (
        torch.from_numpy(arr_norm)
        .float()
        .unsqueeze(0)  # batch
        .unsqueeze(0)  # channel
        .to(device)
    )

    with torch.no_grad():
        output = model(tensor)

    restored = output.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

    # Safety: clip to [0, 1] and replace any NaN/Inf (model already applies
    # sigmoid, so this is a defensive guard rather than an expected case).
    restored = np.nan_to_num(restored, nan=0.0, posinf=1.0, neginf=0.0)
    restored = np.clip(restored, 0.0, 1.0)

    return restored


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isdir(input_dir):
        print(f"Error: input directory does not exist: {input_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(device)
    print(f"Loaded model weights from: {CHECKPOINT_PATH}")

    input_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))

    if not input_files:
        print(f"Warning: no .npy files found in {input_dir}")
        sys.exit(0)

    print(f"Found {len(input_files)} .npy files. Running restoration...")

    for i, in_path in enumerate(input_files):
        filename = os.path.basename(in_path)
        out_path = os.path.join(output_dir, filename)

        lr_array = np.load(in_path)
        restored = restore_single(model, lr_array, device)

        np.save(out_path, restored)

        if (i + 1) % 50 == 0 or (i + 1) == len(input_files):
            print(f"  Processed {i + 1}/{len(input_files)}: {filename} -> shape {restored.shape}")

    print(f"Done. Restored outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
