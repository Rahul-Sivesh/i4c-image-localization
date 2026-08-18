
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import lpips


# ============================================================
# NORMALIZATION USED DURING TRAINING
# ============================================================

NOISY_MEAN = 0.433536
NOISY_STD = 0.284787


# ============================================================
# MODEL
# Same architecture used for V2
# ============================================================

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

        self.head = nn.Conv2d(
            1, num_features, kernel_size=3, padding=1
        )

        self.body = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)]
        )

        self.body_conv = nn.Conv2d(
            num_features, num_features, kernel_size=3, padding=1
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(
                num_features,
                num_features * 4,
                kernel_size=3,
                padding=1,
            ),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )

        self.tail = nn.Conv2d(
            num_features, 1, kernel_size=3, padding=1
        )

    def forward(self, x):
        features = self.head(x)

        body = self.body(features)
        body = self.body_conv(body)

        features = features + body
        features = self.upsample(features)

        output = self.tail(features)

        return torch.sigmoid(output)


# ============================================================
# CHECKPOINT
# ============================================================

def get_default_checkpoint():

    script_dir = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(
        script_dir,
        "checkpoints",
        "model_v2_best.pth"
    )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Evaluate wafer image restoration model"
    )

    parser.add_argument(
        "--lr_dir",
        type=str,
        required=True,
        help="Directory containing degraded NoisyLR .npy images"
    )

    parser.add_argument(
        "--gt_dir",
        type=str,
        required=True,
        help="Directory containing corresponding GT .npy images"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model_v2_best.pth"
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optional limit on number of images"
    )

    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = get_default_checkpoint()

    return args


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(checkpoint_path, device):

    print("=" * 60)
    print("LOADING MODEL")
    print("=" * 60)

    model = RestorationNetV1().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    if "model_state_dict" in checkpoint:

        model.load_state_dict(checkpoint["model_state_dict"])

        print(
            "Checkpoint epoch:",
            checkpoint.get("epoch", "unknown")
        )

        if "val_psnr" in checkpoint:
            print(
                f"Validation PSNR: {checkpoint['val_psnr']:.3f} dB"
            )

    else:
        model.load_state_dict(checkpoint)

    model.eval()

    print("Model loaded successfully.")

    return model


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, lr_dir, gt_dir, device, lpips_model, max_images=None):

    lr_files = sorted(
        f for f in os.listdir(lr_dir) if f.endswith(".npy")
    )

    gt_files = sorted(
        f for f in os.listdir(gt_dir) if f.endswith(".npy")
    )

    common_files = sorted(set(lr_files) & set(gt_files))

    if max_images is not None:
        common_files = common_files[:max_images]

    print()
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)
    print("Images:", len(common_files))
    print("Device:", device)

    psnr_values = []
    ssim_values = []
    lpips_values = []

    for index, filename in enumerate(common_files):

        lr = np.load(
            os.path.join(lr_dir, filename)
        ).astype(np.float32)

        lr_norm = (lr - NOISY_MEAN) / NOISY_STD

        lr_tensor = torch.from_numpy(lr_norm).float()
        lr_tensor = lr_tensor.unsqueeze(0).unsqueeze(0).to(device)

        gt = np.load(
            os.path.join(gt_dir, filename)
        ).astype(np.float32)

        prediction = model(lr_tensor)
        prediction = torch.clamp(prediction, 0.0, 1.0)

        prediction_np = prediction.squeeze().cpu().numpy()

        psnr = peak_signal_noise_ratio(
            gt, prediction_np, data_range=1.0
        )

        ssim = structural_similarity(
            gt, prediction_np, data_range=1.0
        )

        pred_lpips = torch.from_numpy(prediction_np).float()
        pred_lpips = pred_lpips.unsqueeze(0).unsqueeze(0)

        gt_lpips = torch.from_numpy(gt).float()
        gt_lpips = gt_lpips.unsqueeze(0).unsqueeze(0)

        pred_lpips = pred_lpips.repeat(1, 3, 1, 1)
        gt_lpips = gt_lpips.repeat(1, 3, 1, 1)

        pred_lpips = pred_lpips * 2.0 - 1.0
        gt_lpips = gt_lpips * 2.0 - 1.0

        pred_lpips = pred_lpips.to(device)
        gt_lpips = gt_lpips.to(device)

        lpips_score = lpips_model(
            pred_lpips,
            gt_lpips
        ).item()

        psnr_values.append(psnr)
        ssim_values.append(ssim)
        lpips_values.append(lpips_score)

        if (index + 1) % 50 == 0:
            print(f"Processed {index + 1}/{len(common_files)}")

    print()
    print("=" * 60)
    print("FINAL EVALUATION RESULTS")
    print("=" * 60)

    print(f"Mean PSNR  : {np.mean(psnr_values):.6f} dB")
    print(f"Mean SSIM  : {np.mean(ssim_values):.6f}")
    print(f"Mean LPIPS : {np.mean(lpips_values):.6f}")

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_model(args.checkpoint, device)

    print()
    print("Loading LPIPS...")

    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    evaluate(
        model=model,
        lr_dir=args.lr_dir,
        gt_dir=args.gt_dir,
        device=device,
        lpips_model=lpips_model,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
