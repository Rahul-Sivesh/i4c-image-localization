"""
train.py — reproducible training script for the KLA restoration challenge.

Trains RestorationNetV1 (residual CNN + PixelShuffle 2x upsampling) to jointly
denoise and super-resolve degraded semiconductor inspection images, using a
combined L1 + SSIM loss. Reproduces the submitted checkpoint (model_v2_best.pth)
from scratch, given the same dataset and split file.

Usage:
    python train.py --data_dir /path/to/Semicon/train --split_csv /path/to/dataset_split.csv \
                     --output_dir /path/to/weights --epochs 30

Expects the following folder structure under --data_dir:
    data_dir/
        GT/        <- clean ground-truth .npy images
        NoisyLR/   <- degraded, low-resolution .npy images (same filenames as GT)

And a --split_csv with columns "filename" and "split" (values: train/val/test),
so training never touches validation or test data (no leakage into model
selection, as required).
"""

import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42


def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Normalization (must match inference.py / evaluate.py)
# Computed from the actual training dataset — see README for details.
# ---------------------------------------------------------------------------
NOISY_MEAN = 0.433536
NOISY_STD = 0.284787


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SemiconductorDataset(Dataset):
    def __init__(self, data_dir, split_csv, split="train"):
        self.gt_dir = os.path.join(data_dir, "GT")
        self.lr_dir = os.path.join(data_dir, "NoisyLR")

        df = pd.read_csv(split_csv)
        self.filenames = df[df["split"] == split]["filename"].tolist()
        assert len(self.filenames) > 0, f"No files found for split='{split}' in {split_csv}"
        print(f"[{split}] {len(self.filenames)} samples")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        lr = np.load(os.path.join(self.lr_dir, fname)).astype(np.float32)
        gt = np.load(os.path.join(self.gt_dir, fname)).astype(np.float32)

        # Normalize input using dataset-derived stats. NOT clipped — this
        # preserves the out-of-[0,1]-range values caused by speckle noise,
        # which the model must learn to correct for.
        lr_norm = (lr - NOISY_MEAN) / NOISY_STD

        lr_tensor = torch.from_numpy(lr_norm).unsqueeze(0).float()  # [1, H, W]
        gt_tensor = torch.from_numpy(gt).unsqueeze(0).float()       # [1, 2H, 2W]
        return lr_tensor, gt_tensor


# ---------------------------------------------------------------------------
# Model (must match inference.py / evaluate.py exactly)
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
# SSIM (single-channel, differentiable) — used in the combined loss
# ---------------------------------------------------------------------------
def _gaussian_window(window_size, sigma, device):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g.unsqueeze(0) * g.unsqueeze(1)
    return window_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, W, W]


def ssim_loss(pred, target, window_size=11, sigma=1.5):
    device = pred.device
    window = _gaussian_window(window_size, sigma, device)
    pad = window_size // 2

    mu_pred = F.conv2d(pred, window, padding=pad)
    mu_target = F.conv2d(target, window, padding=pad)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, window, padding=pad) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, window, padding=pad) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window, padding=pad) - mu_pred_target

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
    return ssim_map.mean()


def restoration_loss_v2(prediction, target, alpha=1.0, beta=0.2):
    """Combined L1 + SSIM loss. L1 is robust to speckle-noise outlier pixels;
    the SSIM term discourages the over-smoothing that pure pixel-wise losses
    tend to produce, directly targeting the 'don't blur to denoise' objective."""
    l1 = F.l1_loss(prediction, target)
    ssim = ssim_loss(prediction, target)
    total_loss = alpha * l1 + beta * (1 - ssim)
    return total_loss, l1, ssim


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def calculate_psnr(pred, target):
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 100.0
    return 10 * np.log10(1.0 / mse)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = SemiconductorDataset(args.data_dir, args.split_csv, split="train")
    val_ds = SemiconductorDataset(args.data_dir, args.split_csv, split="val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = RestorationNetV1(num_features=64, num_blocks=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "model_v2_best.pth")

    best_psnr = -float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_psnr = 0.0, 0.0
        for lr_img, gt_img in train_loader:
            lr_img, gt_img = lr_img.to(device), gt_img.to(device)
            pred = model(lr_img)
            loss, l1_val, ssim_val = restoration_loss_v2(pred, gt_img)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_psnr += calculate_psnr(pred.detach(), gt_img)

        train_loss = total_loss / len(train_loader)
        train_psnr = total_psnr / len(train_loader)

        model.eval()
        val_loss, val_psnr = 0.0, 0.0
        with torch.no_grad():
            for lr_img, gt_img in val_loader:
                lr_img, gt_img = lr_img.to(device), gt_img.to(device)
                pred = model(lr_img)
                loss, _, _ = restoration_loss_v2(pred, gt_img)
                val_loss += loss.item()
                val_psnr += calculate_psnr(pred, gt_img)
        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)

        scheduler.step(val_psnr)
        print(f"Epoch [{epoch:02d}/{args.epochs}] | Train Loss: {train_loss:.4f} | "
              f"Train PSNR: {train_psnr:.3f} dB | Val PSNR: {val_psnr:.3f} dB | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_psnr": val_psnr,
            }, checkpoint_path)
            print(f"  -> saved new best checkpoint to {checkpoint_path} (val PSNR {val_psnr:.3f} dB)")

    print(f"\nTraining complete. Best validation PSNR: {best_psnr:.3f} dB")
    print(f"Best checkpoint saved to: {checkpoint_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the restoration model (RestorationNetV1, L1+SSIM loss)")
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Path to data root containing GT/ and NoisyLR/ subfolders")
    parser.add_argument("--split_csv", type=str, required=True,
                         help="Path to dataset_split.csv (columns: filename, split)")
    parser.add_argument("--output_dir", type=str, default="weights",
                         help="Directory to save the best checkpoint (model_v2_best.pth)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
