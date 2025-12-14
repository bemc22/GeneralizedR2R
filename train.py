#!/usr/bin/env python
"""
Training script for Generalized R2R (Recorrupted-to-Recorrupted) Self-Supervised Denoising.

This script trains a denoising model using the GR2R loss function, which enables
self-supervised learning across various noise distributions (Gaussian, Poisson, Gamma).

Usage:
    python train.py --noise-type gaussian --noise-level 0.1 --epochs 800
    python train.py --noise-type poisson --noise-level 0.1 --epochs 800
    python train.py --noise-type gamma --noise-level 5.0 --epochs 800

Reference:
    Generalized Recorrupted-to-Recorrupted: Self-Supervised Learning Beyond Gaussian Noise
    CVPR 2025
"""

import os
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import deepinv as dinv
from deepinv.datasets import SimpleFastMRISliceDataset
from deepinv.utils.demo import get_data_home
from deepinv.loss import PSNR, SSIM
from deepinv.training import train


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a denoising model with Generalized R2R loss",
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
        help="Alpha parameter for R2R loss"
    )

    # Training configuration
    parser.add_argument(
        "--epochs",
        type=int,
        default=800,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for optimizer"
    )

    # Loss configuration
    parser.add_argument(
        "--loss",
        type=str,
        default="gr2r_mse",
        choices=["gr2r_mse", "sup", "neigh"],
        help="Loss function: gr2r_mse (Generalized R2R), sup (Supervised), neigh (Neighbor2Neighbor)"
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default="fastmri",
        choices=["fastmri", "div2k"],
        help="Dataset to use for training"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=256,
        help="Image size for training"
    )
    parser.add_argument(
        "--n-train-images",
        type=int,
        default=900,
        help="Maximum number of training images"
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="dncnn",
        choices=["dncnn", "drunet"],
        help="Model architecture"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=7,
        help="Depth for DnCNN model"
    )

    # Output configuration
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory for datasets"
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="./ckpts",
        help="Directory for saving checkpoints"
    )
    parser.add_argument(
        "--ckpt-interval",
        type=int,
        default=1,
        help="Checkpoint saving interval (epochs)"
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
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name (default: denoising-{noise_type})"
    )
    parser.add_argument(
        "--freq-plot",
        type=int,
        default=200,
        help="Frequency of plotting during training"
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable CUDA even if available"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Set default noise level based on noise type
    if args.noise_level is None:
        if args.noise_type == "gamma":
            args.noise_level = 5.0
        else:
            args.noise_level = 0.1

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
    """Load or generate dataset based on configuration."""
    data_dir = Path(args.data_dir)
    measurement_dir = data_dir / "measurements"

    if args.dataset == "fastmri":
        dataset_name = "mri_denoising"
        train_dataset_name = "fastmri_knee_singlecoil"
        operation = f"Denoising_{args.noise_level}"
        save_dir = measurement_dir / train_dataset_name / operation

        transform = transforms.Compose([transforms.Resize(args.img_size)])

        # Check if dataset already exists
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
                train_datapoints=args.n_train_images,
                num_workers=args.num_workers,
                dataset_filename=str(dataset_name),
            )
        else:
            dataset_path = save_dir / f"{dataset_name}0.h5"

        n_channels = 2  # real + imaginary for MRI

    elif args.dataset == "div2k":
        dataset_name = "div2k_denoising"
        operation = f"Denoising_{args.noise_level}"
        save_dir = measurement_dir / "div2k" / operation

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop((args.img_size, args.img_size))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop((512, 512))
        ])

        # Check if dataset already exists
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
                train_datapoints=args.n_train_images,
                num_workers=args.num_workers,
                dataset_filename=str(dataset_name),
            )
        else:
            dataset_path = save_dir / f"{dataset_name}0.h5"

        n_channels = 3  # RGB
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_dataset = dinv.datasets.HDF5Dataset(path=dataset_path, train=True)
    test_dataset = dinv.datasets.HDF5Dataset(path=dataset_path, train=False)

    return train_dataset, test_dataset, n_channels


def get_model(args, n_channels, device):
    """Create model based on configuration."""
    if args.model == "dncnn":
        backbone = dinv.models.DnCNN(
            in_channels=n_channels,
            out_channels=n_channels,
            pretrained=None,
            depth=args.depth,
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

    # Print model info
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model.upper()}, Parameters: {n_params:,}")

    return model


def get_loss(args, noise_model):
    """Create loss function based on configuration."""
    if args.loss == "sup":
        return dinv.loss.SupLoss()
    elif args.loss == "gr2r_mse":
        r2r_loss = dinv.loss.R2RLoss(noise_model=noise_model, alpha=args.alpha)
        return [r2r_loss]
    elif args.loss == "neigh":
        neigh_loss = dinv.loss.Neighbor2Neighbor()
        return [neigh_loss]
    else:
        raise ValueError(f"Unknown loss: {args.loss}")


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Setup device
    if args.no_cuda or not torch.cuda.is_available():
        device = "cpu"
    else:
        device = dinv.utils.get_freer_gpu()
    print(f"Using device: {device}")

    # Setup paths
    project_name = f"denoising-{args.noise_type}"
    ckpt_dir = Path(args.ckpt_dir) / project_name / f"noise={args.noise_level}" / args.loss
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Create noise model and physics
    noise_model = get_noise_model(args.noise_type, args.noise_level)
    physics = dinv.physics.Denoising(noise_model=noise_model)

    print(f"Noise type: {args.noise_type}")
    print(f"Noise level: {args.noise_level}")
    print(f"Alpha: {args.alpha}")
    print(f"Loss: {args.loss}")

    # Load dataset
    train_dataset, test_dataset, n_channels = get_dataset(args, physics, device)
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False
    )

    # Create model
    model = get_model(args, n_channels, device)

    # Create loss
    loss = get_loss(args, noise_model)

    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.epochs * 0.8) + 1
    )

    # Setup wandb if enabled
    wandb_setup = None
    if args.wandb:
        wandb_project = args.wandb_project or project_name
        wandb_setup = {
            "project": wandb_project,
            "config": vars(args),
            "name": f"{args.loss}-{args.noise_level}",
        }

    # Create trainer
    trainer = dinv.Trainer(
        metrics=[PSNR(), SSIM()],
        model=model,
        physics=physics,
        epochs=args.epochs,
        scheduler=scheduler,
        losses=loss,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=test_dataloader,
        plot_images=True,
        device=device,
        save_path=ckpt_dir,
        verbose=args.verbose,
        wandb_vis=args.wandb,
        wandb_setup=wandb_setup,
        show_progress_bar=True,
        ckp_interval=args.ckpt_interval,
        freq_plot=args.freq_plot,
    )

    # Train the model
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Checkpoints will be saved to: {ckpt_dir}")
    model = trainer.train()

    # Evaluate on test set
    print("\nEvaluating model on test dataset...")
    trainer.freq_plot = 1
    trainer.test(test_dataloader)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
