"""
Train a scene-conditioned GPD denoiser (D3) with classifier-free guidance.

Conditioning = DeepSets obstacle-set encoder added to the time embedding.
During training the scene is dropped to a learned null embedding with prob
`cfg_dropout`, enabling CFG at inference: eps = eps_u + w (eps_c - eps_u).

  python train_gpd_cond.py --cp gpd_train_combined.hdf5 \
      --scenes scenes_unique.hdf5 --T 64 --n_control 8 --epochs 60000
"""
import argparse, os, numpy as np, torch, torch.nn as nn
import wandb

from gpd.dataset import ConditionalBernsteinDataset
from gpd.conditional_unet import ConditionalTemporalUNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cp', default='gpd_train_combined.hdf5')
    p.add_argument('--scenes', default='scenes_unique.hdf5')
    p.add_argument('--model_dir', default='./models/')
    p.add_argument('--T', type=int, default=64)
    p.add_argument('--n_control', type=int, default=8)
    p.add_argument('--epochs', type=int, default=60000)
    p.add_argument('--batch_size', type=int, default=2048)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--cfg_dropout', type=float, default=0.15)
    p.add_argument('--variance_thresh', type=float, default=0.02)
    p.add_argument('--schedule', default='linear', choices=['linear', 'cosine'])
    p.add_argument('--wandb', action='store_true'); p.add_argument('--no_wandb', dest='wandb', action='store_false')
    p.set_defaults(wandb=True)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    ds = ConditionalBernsteinDataset(args.cp, args.scenes, n_diffusion_steps=args.T,
                                     variance_thresh=args.variance_thresh,
                                     schedule=args.schedule)

    model_name = os.path.join(args.model_dir, f'GPDCondModel{args.T}_N{args.n_control}')
    os.makedirs(args.model_dir, exist_ok=True)
    model = ConditionalTemporalUNet(model_name=model_name, input_dim=7, time_dim=32,
                                    dims=(32, 64, 128, 256), device=device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    if args.wandb:
        wandb.init(project='GPD_denoiser', name=f'GPDCond_T{args.T}_N{args.n_control}')
        wandb.config.update({'T': args.T, 'n_control': args.n_control,
                             'epochs': args.epochs, 'batch_size': args.batch_size,
                             'cfg_dropout': args.cfg_dropout})

    for e in range(len(model.losses), args.epochs):
        model.train(True)
        X, Y, t, O, Mk = ds.generate_training_batch(args.batch_size)
        X, Y, t = X.to(device), Y.to(device), t.to(device)
        O, Mk = O.to(device), Mk.to(device)
        drop = (torch.rand(X.shape[0], device=device) < args.cfg_dropout)

        pred = model(X, t, obs=O, mask=Mk, drop=drop)
        opt.zero_grad()
        loss = loss_fn(pred, Y)
        loss.backward()
        opt.step()

        lv = loss.item()
        model.losses = np.append(model.losses, lv)
        print(f"\rEpoch {e:6d}  loss: {lv:.6f}", end="")
        if args.wandb:
            wandb.log({'epoch': e, 'loss': lv})
        model.save()
        if e % 1000 == 0:
            model.save_checkpoint(e); print()

    print("\nTraining complete.")


if __name__ == '__main__':
    main()
