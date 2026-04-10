"""
GPU-native guided denoising for EDMP.

Drops the 510 numpy CPU↔GPU round-trips per scene (255 steps × 2)
by keeping X_t on GPU throughout, identical to what we did for GPD.
"""

import numpy as np
import torch
from diffusion.diffusion import Diffusion


def denoise_guided_gpu(diffuser: Diffusion, model, guide,
                       traj_len: int, num_channels: int,
                       guidance_schedule: np.ndarray,
                       batch_size: int = 1,
                       start: np.ndarray = None,
                       goal:  np.ndarray = None,
                       condition: bool = True,
                       benchmarking: bool = False) -> np.ndarray:
    """
    GPU-native replacement for Diffusion.denoise_guided().
    Keeps X_t on GPU for the full denoising loop; only the guidance
    gradient touches CPU (guide.get_gradient uses numpy in/out).

    Returns (batch_size, num_channels, traj_len) numpy array.
    """
    dev = diffuser.device

    # Pre-compute variance schedule tensors on GPU
    alpha_gpu     = torch.tensor(diffuser.alpha,     dtype=torch.float32, device=dev)
    alpha_bar_gpu = torch.tensor(diffuser.alpha_bar, dtype=torch.float32, device=dev)
    beta_gpu      = torch.tensor(diffuser.beta,      dtype=torch.float32, device=dev)
    gs_gpu        = torch.tensor(guidance_schedule,  dtype=torch.float32, device=dev)

    start_t = torch.tensor(start, dtype=torch.float32, device=dev)
    goal_t  = torch.tensor(goal,  dtype=torch.float32, device=dev)

    lower = torch.tensor([-166,-101,-166,-176,-166, -1,-166],
                         dtype=torch.float32, device=dev) * (torch.pi / 180)
    upper = torch.tensor([ 166, 101, 166,  -4, 166,215, 166],
                         dtype=torch.float32, device=dev) * (torch.pi / 180)

    # Initialise trajectory on GPU
    X_t = torch.randn(batch_size, num_channels, traj_len, device=dev)
    if condition:
        X_t[:, :, 0]  = start_t
        X_t[:, :, -1] = goal_t

    model.train(False)
    period = 2

    for t in range(diffuser.T, 0, -1):
        if benchmarking:
            print(f"\rDenoising: {t} ", end="")

        with torch.no_grad():
            t_in = torch.tensor([float(t)], dtype=torch.float32, device=dev)
            eps  = model(X_t, t_in)
            eps  = torch.nan_to_num(eps, nan=0.0, posinf=0.0, neginf=0.0)
            eps  = torch.clamp(eps, -5.0, 5.0)

            a  = alpha_gpu[t - 1].clamp(min=1e-8)
            ab = alpha_bar_gpu[t - 1].clamp(min=1e-8, max=1.0 - 1e-8)
            b  = beta_gpu[t - 1]
            z  = torch.randn_like(X_t)
            if t == 1:
                z.zero_()
            X_t = (X_t - ((1 - a) / torch.sqrt(1 - ab)) * eps) / torch.sqrt(a) + b * z
            X_t = torch.nan_to_num(X_t, nan=0.0, posinf=0.0, neginf=0.0)
            X_t = torch.clamp(X_t, -10.0, 10.0)

        # Gradient guidance
        if (t % period) < (period / 2) and t >= 5:
            with torch.no_grad():
                interior = torch.clamp(X_t[:, :, 1:-1], lower[:, None], upper[:, None])

            grad_np = guide.get_gradient(
                interior.cpu().numpy(), start[:], goal[:], t
            )

            with torch.no_grad():
                grad_t = torch.tensor(grad_np, dtype=torch.float32, device=dev)
                scale  = gs_gpu[:, t - 1].view(batch_size, 1, 1)
                X_t[:, :, 1:-1] -= scale * grad_t

        with torch.no_grad():
            if condition:
                X_t[:, :, 0]  = start_t
                X_t[:, :, -1] = goal_t

    return X_t.cpu().numpy()
