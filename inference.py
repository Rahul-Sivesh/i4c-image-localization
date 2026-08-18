"""
inference.py — KLA evaluation script.

Loads the trained model and runs inference on all images in a test
directory, writing restored outputs to an output directory. This script
is meant to be run AS-IS by KLA's benchmarking team, with no manual edits
required beyond providing --test_dir and --output_dir.

Usage:
    python inference.py --test_dir /path/to/test_images --output_dir /path/to/save/outputs

By default, model weights are loaded from ./weights/model_v2_best.pth,
relative to this script's own location — so as long as the repo's folder
structure is kept intact after cloning, no --checkpoint argument is needed.
An explicit --checkpoint path can still be passed to override this.

Each .npy file in --test_dir is treated as a single-channel NoisyLR image,
normalized the same way as during training, restored by the model, and
saved to --output_dir as a .npy file with the same filename, values
clamped to [0, 1] float32.

End-to-end runtime (disk read -> preprocessing -> GPU transfer -> model
execution -> GPU->CPU transfer -> postprocessing -> disk write) is
measured and reported at the end of the run, along with the batch size
used, so timing is reproducible and doesn't need to be measured externally.

KNOWN LIMITATION: the model was trained exclusively on 128x128 -> 256x256
(2x) pairs. It is fully convolutional and will run on any input
resolution (including a 256x256 -> 512x512 case) without error, but
performance at untrained scales has not been validated and may be worse
than at the trained 128->256 scale. See README for details.
"""

import os
import time
import argparse

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Normalization (must match train.py / evaluate.py)
# ---------------------------------------------------------------------------
NOISY_MEAN = 0.433536
NOISY_STD = 0.284787


# ---------------------------------------------------------------------------
# Model (same as train.py)
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class RestorationNetV1(nn.Module):
    def __init__(self, num_features=64, num_blocks=8):
        super().__init__()
        self.head = nn.Conv2d(1, num_features, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_blocks)])
        self.body_conv = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )
        self.tail = nn.Conv2d(num_features, 1, kernel_size=3, padding=1)

    def forward(self, x):
        features = self.head(x)
        body = self.body(features)
        body = self.body_conv(body)
        features = features + body
        features = self.upsample(features)
        output = self.tail(features)
        return torch.sigmoid(output)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model = RestorationNetV1(num_features=64, num_blocks=8).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {checkpoint.get('epoch', '?')}, "
          f"val_psnr {checkpoint.get('val_psnr', float('nan')):.3f} dB)")

    os.makedirs(args.output_dir, exist_ok=True)

    test_files = sorted(f for f in os.listdir(args.test_dir) if f.endswith(".npy"))
    n_files = len(test_files)
    print(f"Found {n_files} test files to process. Batch size: {args.batch_size}")

    # Full end-to-end timing starts here: disk read, preprocessing, transfer,
    # model execution, transfer back, postprocessing, and disk write are all
    # included in this single wall-clock measurement.
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.time()

    processed = 0
    for batch_start in range(0, n_files, args.batch_size):
        batch_files = test_files[batch_start: batch_start + args.batch_size]

        # --- Disk read + preprocessing ---
        batch_arrays = []
        for filename in batch_files:
            lr = np.load(os.path.join(args.test_dir, filename)).astype(np.float32)
            lr_norm = (lr - NOISY_MEAN) / NOISY_STD
            batch_arrays.append(lr_norm)

        batch_np = np.stack(batch_arrays, axis=0)                       # [B, H, W]
        batch_tensor = torch.from_numpy(batch_np).unsqueeze(1).to(device)  # [B, 1, H, W]

        # --- Model execution ---
        pred = model(batch_tensor)
        pred = torch.clamp(pred, 0.0, 1.0)

        # --- GPU->CPU transfer + postprocessing + disk write ---
        pred_np = pred.squeeze(1).cpu().numpy().astype(np.float32)      # [B, H, W]
        for i, filename in enumerate(batch_files):
            np.save(os.path.join(args.output_dir, filename), pred_np[i])

        processed += len(batch_files)
        if processed % 50 == 0 or processed == n_files:
            print(f"Processed {processed}/{n_files}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_end = time.time()

    total_time = t_end - t_start
    per_image = total_time / max(n_files, 1)
    print("\n" + "=" * 50)
    print("END-TO-END RUNTIME (disk I/O + preprocessing + model + postprocessing + save)")
    print("=" * 50)
    print(f"Total images     : {n_files}")
    print(f"Batch size        : {args.batch_size}")
    print(f"Total time        : {total_time:.3f} s")
    print(f"Time per image    : {per_image * 1000:.2f} ms")
    print(f"Throughput        : {n_files / total_time:.2f} images/sec")
    print(f"Device            : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"\nDone. Restored outputs saved to: {args.output_dir}")


def get_default_checkpoint_path():
    """Default checkpoint location: weights/model_v2_best.pth,
    relative to this script's own directory (works after a fresh git clone,
    regardless of the caller's current working directory)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "weights", "model_v2_best.pth")


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a test set of NoisyLR .npy images")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to directory of input .npy test images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save restored .npy outputs")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to trained model checkpoint (.pth). "
             "Defaults to weights/model_v2_best.pth next to this script."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for GPU inference (default: 16). Reduce if you hit GPU memory limits."
    )
    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = get_default_checkpoint_path()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found at: {args.checkpoint}\n"
            f"Expected the trained weights at 'weights/model_v2_best.pth' "
            f"relative to this script, or pass --checkpoint explicitly."
        )

    return args


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
