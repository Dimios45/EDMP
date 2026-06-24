"""
Training script for GPD (Guided Polynomial Diffusion).

Trains a TemporalUNet on Bernstein control-point trajectories (7 x 8)
instead of full waypoint trajectories (7 x 50).

Pre-requisites:
  1. Download the MPInets training dataset (train.hdf5).
  2. Run preprocessing:
       python -m gpd.preprocess_data \
           --input /path/to/train.hdf5 \
           --output /path/to/gpd_train.hdf5
  3. Run this script.

The model is saved to ./models/GPDModel{T}_N{n_control}/.
"""

import torch
import torch.nn as nn
import time
import numpy as np
import os
import argparse

from gpd.dataset import BernsteinTrajectoryDataset
from diffusion.models.temporalunet import TemporalUNet

import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',    required=True,
                        help='Path to preprocessed HDF5 (gpd_train.hdf5)')
    parser.add_argument('--model_dir',  default='./models/',
                        help='Directory to save model weights')
    parser.add_argument('--T',          type=int, default=64,
                        help='Diffusion timesteps (paper uses 64)')
    parser.add_argument('--n_control',  type=int, default=8,
                        help='Number of Bernstein control points')
    parser.add_argument('--epochs',     type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--variance_thresh', type=float, default=0.02,
                        help='Terminal beta for linear schedule (D1: 0.08 for T=64)')
    parser.add_argument('--schedule',   default='linear', choices=['linear', 'cosine'])
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--wandb',      action='store_true')
    parser.add_argument('--no_wandb',   dest='wandb', action='store_false')
    parser.set_defaults(wandb=True)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ---- Dataset ----
    dataset = BernsteinTrajectoryDataset(args.dataset,
                                         n_diffusion_steps=args.T,
                                         variance_thresh=args.variance_thresh,
                                         schedule=args.schedule)
    print(f"Schedule: {args.schedule} | variance_thresh={args.variance_thresh}")

    # ---- Model ----
    # n_control = 8 is much smaller than traj_len = 50 so the UNet dims
    # are scaled down to avoid over-parameterisation.
    model_name = os.path.join(args.model_dir,
                               f'GPDModel{args.T}_N{args.n_control}')
    os.makedirs(args.model_dir, exist_ok=True)

    model_dims   = (32, 64, 128, 256)
    time_dim     = 32
    denoiser = TemporalUNet(
        model_name=model_name,
        input_dim=7,
        time_dim=time_dim,
        dims=model_dims,
        device=device,
    )
    denoiser.to(device)

    optimizer = torch.optim.Adam(denoiser.parameters(), lr=args.lr)
    loss_fn   = nn.MSELoss()

    # ---- W&B ----
    if args.wandb:
        wandb.init(project='GPD_denoiser', name=f'GPD_T{args.T}_N{args.n_control}')
        wandb.config.update({
            'T': args.T, 'n_control': args.n_control,
            'epochs': args.epochs, 'batch_size': args.batch_size,
            'model_dims': model_dims,
        })

    # ---- Training loop ----
    for e in range(len(denoiser.losses), args.epochs):
        denoiser.train(True)

        X, Y_true, t = dataset.generate_training_batch(args.batch_size)
        X      = X.to(device)
        Y_true = Y_true.to(device)
        t      = t.to(device)

        Y_pred = denoiser(X, t)

        optimizer.zero_grad()
        loss = loss_fn(Y_pred, Y_true)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        denoiser.losses = np.append(denoiser.losses, loss_val)

        print(f"\rEpoch {e:6d}  loss: {loss_val:.6f}", end="")

        if args.wandb:
            wandb.log({'epoch': e, 'loss': loss_val})

        denoiser.save()
        if e % 1000 == 0:
            denoiser.save_checkpoint(e)
            print()  # newline after checkpoint

    print("\nTraining complete.")


if __name__ == '__main__':
    main()
