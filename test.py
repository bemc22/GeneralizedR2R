#!/usr/bin/env python
"""
Testing script for Generalized R2R (Recorrupted-to-Recorrupted) Self-Supervised Denoising.

This script evaluates trained denoising models on test datasets, computing PSNR and SSIM
metrics and optionally visualizing results.

Usage:
    # Test with pretrained weights from HuggingFace
    python test.py --noise-type gaussian --noise-level 0.1
    python test.py --noise-type poisson --noise-level 0.1
    python test.py --noise-type gamma --noise-level 5.0

    # Test with local checkpoint
    python test.py --noise-type gaussian --checkpoint ./ckpts/model.pth

    # Test and save visualizations
    python test.py --noise-type gaussian --save-images --output-dir ./results

Reference:
    Generalized Recorrupted-to-Recorrupted: Self-Supervised Learning Beyond Gaussian Noise
    CVPR 2025
"""

import os
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

import deepinv as dinv
from deepinv.datasets import SimpleFastMRISliceDataset
from deepinv.utils.demo import get_data_home
from deepinv.loss import PSNR, SSIM


# Pretrained weights from HuggingFace
PRETRAINED_WEIGHTS = {
    "gaussian": "artifact_r2r_gaussian.pth",
    "poisson": "artifact_r2r_poisson.pth",
    "gamma": "artifact_r2r_gamma.pth",
}


def get_weights_url(file_name):
    """Get HuggingFace URL for pretrained weights."""
    return (
        "https://huggingface.co/bemc22/"
        + "GeneralizedR2R"
        + "/resolve/main/"
        + file_name
        + "?download=true"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test a denoising model trained with Generalized R2R",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Noise configuration
    parser.add_argument(
        "--noise-type",
        type=str,
        default="gaussian",
        choices=["gaussian", "poisson", "gamma"],
        help="Type of noise distribution"
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=None,
        help="Noise level (default: 0.1 for gaussian/poisson, 5.0 for gamma)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="Alpha parameter for R2R loss (used when adapting model)"
    )

    # Model configuration
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint. If not provided, downloads pretrained weights."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model architecture (default: auto-detect based on noise type)"
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset to use (default: fastmri for gaussian, div2k for poisson/gamma)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory for datasets"
    )
    parser.add_argument(
        "--n-test-samples",
        type=int,
        default=None,
        help="Number of test samples to evaluate (default: all)"
    )

    # Output configuration
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save visualization images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory for saving results"
    )
    parser.add_argument(
        "--show-images",
        action="store_true",
        help="Display images during evaluation"
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable CUDA even if available"
    )
    parser.add_argument(
        "--batch-ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific batch IDs to evaluate (default: first 5)"
    )

    args = parser.parse_args()

    # Set default noise level based on noise type
    if args.noise_level is None:
        if args.noise_type == "gamma":
            args.noise_level = 5.0
        else:
            args.noise_level = 0.1

    # Set default dataset based on noise type
    if args.dataset is None:
        if args.noise_type == "gaussian":
            args.dataset = "fastmri"
        else:
            args.dataset = "div2k"

    # Set default model based on dataset
    if args.model is None:
        if args.dataset == "fastmri":
            args.model = "dncnn"
        else:
            args.model = "drunet"

    # Set default alpha for gamma
    if args.noise_type == "gamma" and args.alpha == 0.2:
        args.alpha = 0.5

    return args


def get_noise_model(noise_type, noise_level):
    """Create noise model based on noise type."""
    if noise_type == "gaussian":
        noise_model = dinv.physics.GaussianNoise(noise_level)
    elif noise_type == "poisson":
        noise_model = dinv.physics.PoissonNoise(noise_level)
        noise_model.sigma = noise_level
    elif noise_type == "gamma":
        noise_model = dinv.physics.GammaNoise(noise_level)
        noise_model.sigma = noise_level
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    return noise_model


def get_dataset(args, physics, device):
    """Load or generate test dataset."""
    data_dir = Path(args.data_dir)
    measurement_dir = data_dir / "measurements"

    if args.dataset == "fastmri":
        dataset_name = "mri_gaussian_denoising"
        train_dataset_name = "fastmri_knee_singlecoil"
        operation = f"Denoising_{args.noise_level}"
        save_dir = measurement_dir / train_dataset_name / operation
        img_size = 256

        transform = transforms.Compose([transforms.Resize(img_size)])

        if not os.path.exists(save_dir / f"{dataset_name}0.h5"):
            print(f"Generating dataset at {save_dir}...")
            train_data = SimpleFastMRISliceDataset(
                get_data_home(), transform=transform, train_percent=0.5, train=True, download=True
            )
            test_data = SimpleFastMRISliceDataset(
                get_data_home(), transform=transform, train_percent=0.5, train=False
            )

            dataset_path = dinv.datasets.generate_dataset(
                train_dataset=train_data,
                test_dataset=test_data,
                physics=physics,
                device=device,
                save_dir=save_dir,
                train_datapoints=900,
                num_workers=args.num_workers,
                dataset_filename=str(dataset_name),
            )
        else:
            dataset_path = save_dir / f"{dataset_name}0.h5"

        n_channels = 2

    elif args.dataset == "div2k":
        if args.noise_type == "poisson":
            dataset_name = "div2k_poisson"
        elif args.noise_type == "gamma":
            dataset_name = "div2k_gamma"
        else:
            dataset_name = "div2k_denoising"

        operation = f"Denoising_{args.noise_level}"
        save_dir = measurement_dir / "div2k" / operation

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop((224, 224))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop((512, 512))
        ])

        if not os.path.exists(save_dir / f"{dataset_name}0.h5"):
            print(f"Generating dataset at {save_dir}...")
            train_data = dinv.datasets.DIV2K(
                root=data_dir, mode="train", transform=transform, download=True
            )
            test_data = dinv.datasets.DIV2K(
                root=data_dir, mode="val", transform=test_transform, download=True
            )

            dataset_path = dinv.datasets.generate_dataset(
                train_dataset=train_data,
                test_dataset=test_data,
                physics=physics,
                device=device,
                save_dir=save_dir,
                train_datapoints=100,
                num_workers=args.num_workers,
                dataset_filename=str(dataset_name),
            )
        else:
            dataset_path = save_dir / f"{dataset_name}0.h5"

        n_channels = 3
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    test_dataset = dinv.datasets.HDF5Dataset(path=dataset_path, train=False)
    return test_dataset, n_channels


def get_model(args, n_channels, noise_model, device):
    """Create and load model."""
    if args.model == "dncnn":
        backbone = dinv.models.DnCNN(
            in_channels=n_channels,
            out_channels=n_channels,
            pretrained=None,
            depth=7,
        ).to(device)
    elif args.model == "drunet":
        backbone = dinv.models.DRUNet(
            in_channels=n_channels,
            out_channels=n_channels,
            pretrained=None,
            nc=[16, 32, 64, 128]
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = dinv.models.ArtifactRemoval(backbone)

    # Adapt model for R2R
    r2r_loss = dinv.loss.R2RLoss(noise_model=noise_model, alpha=args.alpha)
    model = r2r_loss.adapt_model(model)

    # Load weights
    if args.checkpoint is not None:
        print(f"Loading checkpoint from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)
    else:
        # Download pretrained weights
        filename = PRETRAINED_WEIGHTS.get(args.noise_type)
        if filename is None:
            raise ValueError(f"No pretrained weights for noise type: {args.noise_type}")

        url = get_weights_url(filename)
        print(f"Downloading pretrained weights: {filename}")
        ckpt = torch.hub.load_state_dict_from_url(
            url, map_location=lambda storage, loc: storage, file_name=filename
        )
        model.load_state_dict(ckpt["state_dict"])

    model.eval()
    return model


def visualize_results(x, y, x_net, psnr_val, psnr_baseline, args, batch_id, output_dir=None):
    """Visualize and optionally save denoising results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Convert tensors to numpy for plotting
    x_plot = x.squeeze().cpu().detach().numpy()
    y_plot = y.squeeze().cpu().detach().numpy()
    x_est = x_net.squeeze().cpu().detach().numpy()

    # Handle different channel configurations
    if x_plot.ndim == 3 and x_plot.shape[0] in [2, 3]:
        x_plot = x_plot.transpose(1, 2, 0)
        y_plot = y_plot.transpose(1, 2, 0)
        x_est = x_est.transpose(1, 2, 0)

    # Normalize for display
    x_plot = (x_plot - x_plot.min()) / (x_plot.max() - x_plot.min() + 1e-8)
    x_est = (x_est - x_est.min()) / (x_est.max() - x_est.min() + 1e-8)

    # Apply log transform for Poisson/Gamma noise for better visualization
    if args.noise_type in ["poisson", "gamma"]:
        y_plot = np.log(1 + y_plot)
    y_plot = (y_plot - y_plot.min()) / (y_plot.max() - y_plot.min() + 1e-8)

    # Flip for display
    x_plot = x_plot[::-1, ...]
    y_plot = y_plot[::-1, ...]
    x_est = x_est[::-1, ...]

    # Determine colormap and channel
    is_grayscale = x_plot.ndim == 2 or x_plot.shape[-1] == 2
    if is_grayscale:
        cmap = "gray"
        if x_plot.ndim == 3:
            x_plot = x_plot[..., 0]
            y_plot = y_plot[..., 0]
            x_est = x_est[..., 0]
    else:
        cmap = None

    # Plot noisy image
    axes[0].imshow(y_plot, origin="lower", cmap=cmap)
    axes[0].set_title("Noisy Image")
    axes[0].set_xlabel(f"{psnr_baseline:.2f}dB")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Plot denoised image
    axes[1].imshow(x_est, origin="lower", cmap=cmap)
    axes[1].set_title("GR2R Denoised")
    axes[1].set_xlabel(f"{psnr_val:.2f}dB")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Plot reference image
    axes[2].imshow(x_plot, origin="lower", cmap=cmap)
    axes[2].set_title("Reference")
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    plt.tight_layout()

    if output_dir is not None:
        output_path = output_dir / f"{args.noise_type}_sample_{batch_id}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if args.show_images:
        plt.show()
    else:
        plt.close()


def evaluate_sample(model, physics, x, y, device):
    """Evaluate a single sample and return metrics."""
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)

    with torch.no_grad():
        x_net = model(y, physics)

    psnr_val = PSNR()(x=x, x_net=x_net, y=y, physics=physics).item()
    ssim_val = SSIM()(x=x, x_net=x_net, y=y, physics=physics).item()
    psnr_baseline = PSNR()(x=x, x_net=y).item()

    return x_net, psnr_val, ssim_val, psnr_baseline


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Setup device
    if args.no_cuda or not torch.cuda.is_available():
        device = "cpu"
    else:
        device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Setup output directory
    output_dir = None
    if args.save_images:
        output_dir = Path(args.output_dir) / args.noise_type
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create noise model and physics
    noise_model = get_noise_model(args.noise_type, args.noise_level)
    physics = dinv.physics.Denoising(noise_model=noise_model)

    print(f"\n{'='*60}")
    print(f"Testing GR2R Denoising Model")
    print(f"{'='*60}")
    print(f"Noise type: {args.noise_type}")
    print(f"Noise level: {args.noise_level}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"{'='*60}\n")

    # Load dataset
    test_dataset, n_channels = get_dataset(args, physics, device)
    print(f"Test samples: {len(test_dataset)}")

    # Load model
    model = get_model(args, n_channels, noise_model, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Determine which samples to evaluate
    if args.batch_ids is not None:
        batch_ids = args.batch_ids
    elif args.n_test_samples is not None:
        batch_ids = list(range(min(args.n_test_samples, len(test_dataset))))
    else:
        batch_ids = list(range(min(5, len(test_dataset))))

    # Evaluate
    all_psnr = []
    all_ssim = []
    all_psnr_baseline = []

    print(f"\nEvaluating {len(batch_ids)} samples...\n")

    for batch_id in batch_ids:
        x, y = test_dataset[batch_id]

        x_net, psnr_val, ssim_val, psnr_baseline = evaluate_sample(
            model, physics, x, y, device
        )

        all_psnr.append(psnr_val)
        all_ssim.append(ssim_val)
        all_psnr_baseline.append(psnr_baseline)

        print(f"Sample {batch_id:3d}: PSNR={psnr_val:.2f}dB (baseline: {psnr_baseline:.2f}dB), SSIM={ssim_val:.4f}")

        # Visualize if requested
        if args.save_images or args.show_images:
            x_tensor = x.unsqueeze(0).to(device)
            y_tensor = y.unsqueeze(0).to(device)
            visualize_results(
                x_tensor, y_tensor, x_net,
                psnr_val, psnr_baseline,
                args, batch_id, output_dir
            )

    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Samples evaluated: {len(batch_ids)}")
    print(f"Average PSNR: {np.mean(all_psnr):.2f}dB (baseline: {np.mean(all_psnr_baseline):.2f}dB)")
    print(f"Average SSIM: {np.mean(all_ssim):.4f}")
    print(f"PSNR improvement: {np.mean(all_psnr) - np.mean(all_psnr_baseline):.2f}dB")
    print(f"{'='*60}")

    if args.save_images:
        print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
