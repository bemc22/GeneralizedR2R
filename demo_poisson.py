r"""
Self-supervised learning with Generalized Recorrupted-to-Recovered (GR2R) 
====================================================================================================

This example shows you how to train a reconstruction network for an denoising problem on a fully self-supervised way, i.e., using corrupted measurement data only.
"""

import os
import deepinv as dinv
from torch.utils.data import DataLoader
import torchvision
import torch
from pathlib import Path
from torchvision import transforms
from deepinv.optim.prior import PnP
from deepinv.utils.demo import load_dataset, load_degradation
import wandb
import argparse
from torch.utils.data import DataLoader, random_split
from deepinv.loss import PSNR, SSIM, Loss, SupLoss

from deepinv.training import train
# ---------------------------------------------------------------
# Setup the training parameters
# ---------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Self-supervised learning with Generalized Recorrupted-to-Recovered (GR2R)"
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=50,
    help="input batch size for training (default: 32)",
)
parser.add_argument(
    "--loss",
    default="sup",
    choices=["sup", "neigh", "er", "er_mse"],
    help="Loss function to use (default: sup)",
)
parser.add_argument(
    "--epochs", type=int, default=1000, help="number of epochs to train (default: 1200)"
)
parser.add_argument(
    "--lr", type=float, default=1e-4, help="learning rate (default: 5e-4)"
)
parser.add_argument(
    "--noise", type=float, default=0.1, help="noise level (default: 0.01)"
)

parser.add_argument(
    "--trial", type=int, default=0, help="trial id (default: 0)"
)

parser.add_argument(
    "--alpha", type=float, default=0.5, help="alpha value for the R2R loss (default: 0.1)"
)


# ---------------------------------------------------------------
# Setup paths for data loading and results.
# ---------------------------------------------------------------

BASE_DIR = Path(".")
# PROJECT_NAME = "denoising-poisson"
PROJECT_NAME = "denoising-poisson"
ORIGINAL_DATA_DIR =  Path("./data")
DATA_DIR = ORIGINAL_DATA_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results"
DEG_DIR = BASE_DIR / "degradations"
CKPT_DIR = BASE_DIR / "ckpts" / PROJECT_NAME

# Set the global random seed from pytorch to ensure reproducibility of the example.
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"


def main(args):
    # print  all the arguments
    print(args)

    trial_id = args.trial
    torch.manual_seed(trial_id)

    run_name = f"{args.loss}-{args.noise}"

    wandb_setup = {
        "project": PROJECT_NAME,
        "config": args,
        "name": run_name,
    }

    operation = f"Denoising_{args.noise}"
    train_dataset_name = "div2k"
    # # ----------------------------------------------------------------------------------
    # Generate a dataset of knee images and load it.
    # ----------------------------------------------------------------------------------

    # mask = load_degradation("mri_mask_128x128.npy", ORIGINAL_DATA_DIR)

    # defined physics
    # physics = dinv.physics.MRI(mask=mask, device=device, noise_model=dinv.physics.GaussianNoise(args.noise) )
    physics = dinv.physics.Denoising(noise_model=dinv.physics.PoissonNoise(args.noise))

    # Use parallel dataloader if using a GPU to fasten training,
    # otherwise, as all computes are on CPU, use synchronous data loading.
    num_workers = 0 if torch.cuda.is_available() else 0
    n_images_max = 5000

    my_dataset_name = "div2k_poisson"
    measurement_dir = DATA_DIR / train_dataset_name / operation

    # check if the dataset is already generated
    # if not, generate it
    if not os.path.exists(measurement_dir / f"{my_dataset_name}0.h5"):

        img_size = 224
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop((img_size, img_size))
            ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop((512, 512))
        ])

        train_dataset    =  dinv.datasets.DIV2K(root=ORIGINAL_DATA_DIR, mode="train", transform=transform, download=True)
        test_dataset     =  dinv.datasets.DIV2K(root=ORIGINAL_DATA_DIR, mode="val", transform=test_transform, download=True)

        deepinv_datasets_path = dinv.datasets.generate_dataset(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            physics=physics,
            device=device,
            save_dir=measurement_dir,
            train_datapoints=n_images_max,
            num_workers=num_workers,
            dataset_filename=str(my_dataset_name),
        )

    else:      
        deepinv_datasets_path = measurement_dir / f"{my_dataset_name}0.h5"

    train_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=True)
    test_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=False)


    # Set up the reconstruction network
    # ---------------------------------------------------------------
    #
    # As a reconstruction network, we use an unrolled network (half-quadratic splitting)
    # with a trainable denoising prior based on the DnCNN architecture.

    n_channels = 3  

    
    model = dinv.models.DRUNet( in_channels=n_channels,
                                out_channels=n_channels,
                                pretrained=None,
                                nc=[16, 32, 64, 128],
                                train=True,
                                last_act='relu').to(device)

    # print number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")


    # Set up the training parameters
    # --------------------------------------------

    epochs        = args.epochs
    learning_rate = args.lr
    batch_size    = args.batch_size # if torch.cuda.is_available() else 1
    noise_level   = args.noise
    

    if args.loss == "sup":   # SUPERVISED LOSS
        loss = dinv.loss.SupLoss()

    elif args.loss == "er": # GENERALIZED R2R LOSS - NLL VARIANT

        def poisson_nll_loss(y_pred, y_true):
            return torch.nn.functional.poisson_nll_loss(y_pred / noise_level, 
                                                        y_true / noise_level,  
                                                        log_input=False, 
                                                        full=False, 
                                                        eps=1e-4)        

        r2r_loss = dinv.loss.R2RPoissonLoss(metric=poisson_nll_loss, gain=noise_level, p=0.1)
        loss     = [ r2r_loss ]
        model    = r2r_loss.adapt_model(model)

    elif args.loss == "er_mse": # GENERALIZED R2R LOSS - MSE VARIANT

        r2r_loss = dinv.loss.R2RPoissonLoss(metric=torch.nn.MSELoss(), gain=noise_level, p=args.alpha)
        loss     = [ r2r_loss ]
        model    = r2r_loss.adapt_model(model, MC_samples=10)
    
    elif args.loss == "neigh": # NEIGHBORHOOD LOSS

        neigh_loss = dinv.loss.Neighbor2Neighbor()
        loss = [ neigh_loss ]


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.8) + 1)

    verbose   = True
    wandb_vis = True


    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, num_workers=num_workers, shuffle=False
    )

    trainer = dinv.Trainer(
        metrics=[ PSNR(), SSIM() ],
        model=model,
        physics=physics,
        epochs=epochs,
        scheduler=scheduler,
        losses=loss,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        plot_images=True,
        device=device,
        save_path=Path(CKPT_DIR / f"noise={args.noise}"  / args.loss ),
        verbose=verbose,
        wandb_vis=False,
        show_progress_bar=True,
        ckp_interval=1,
        wandb_setup=None,
        eval_dataloader=test_dataloader,
        freq_plot=100,
    )

    model = trainer.train()

    # Evaluate the model on the test dataset
    print("EVALUATING THE MODEL ON THE TEST DATASET")
    trainer.test(test_dataloader)



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

