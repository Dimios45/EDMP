"""
Polynomial diffusion for GPD.

Extends the base Diffusion class with a denoise_guided_poly method that:
  1. Runs the diffusion denoising loop in Bernstein control-point space.
  2. Expands control points to waypoints for cost gradient computation.
  3. Pre-conditions the gradient with B^T before applying it to control points.
  4. Enforces start/goal boundary conditions at every step.
"""

import numpy as np
import torch

from diffusion.diffusion import Diffusion
from gpd.bernstein import BernsteinLayer


class PolynomialDiffusion(Diffusion):

    def __init__(self, T: int, device: str, n_control: int = 8,
                 traj_len: int = 50, variance_thresh: float = 0.02):
        super().__init__(T=T, device=device, variance_thresh=variance_thresh)
        self.bern = BernsteinLayer(traj_len=traj_len, n_control=n_control)
        self.n_control = n_control
        self.traj_len  = traj_len

    # ------------------------------------------------------------------
    # Guided denoising in control-point space
    # ------------------------------------------------------------------

    def denoise_guided_poly(self, model, guide, num_channels: int,
                            guidance_schedule: np.ndarray,
                            batch_size: int = 1,
                            start: np.ndarray = None,
                            goal:  np.ndarray = None,
                            condition: bool = True,
                            benchmarking: bool = False,
                            extra_candidate_steps: int = 0,
                            smoothness_weight: float = 0.0,
                            scene_obs: np.ndarray = None,
                            scene_mask: np.ndarray = None,
                            cfg_weight: float = 1.0) -> np.ndarray:
        """
        GPU-native guided denoising in Bernstein control-point space.

        Keeps alpha_t on GPU throughout — no numpy round-trips inside the loop.
        Only the guidance gradient temporarily touches CPU (via guide.get_gradient).

        Returns:
            trajectories : (batch_size, 7, traj_len) numpy array of waypoints
        """
        dev = self.device

        # Pre-compute variance schedule on GPU
        alpha_gpu     = torch.tensor(self.alpha,     dtype=torch.float32, device=dev)
        alpha_bar_gpu = torch.tensor(self.alpha_bar, dtype=torch.float32, device=dev)
        beta_gpu      = torch.tensor(self.beta,      dtype=torch.float32, device=dev)
        B_gpu         = torch.tensor(self.bern.B,    dtype=torch.float32, device=dev)  # (N, M)
        gs_gpu        = torch.tensor(guidance_schedule, dtype=torch.float32, device=dev)  # (B, T)

        # Second-difference (curvature) operator on control points, for the
        # smoothness-constrained representation experiment. L: (M-2, M) with
        # [1,-2,1] stencil; penalty = ||L alpha||^2 -> grad = 2 (L^T L) alpha.
        # Re-imposes the low-degree smoothness prior that a larger n_control
        # otherwise discards (tests the "compression is a regularizer" thesis).
        LtL_gpu = None
        if smoothness_weight > 0.0 and self.n_control >= 3:
            M = self.n_control
            L = np.zeros((M - 2, M), dtype=np.float32)
            for i in range(M - 2):
                L[i, i:i + 3] = [1.0, -2.0, 1.0]
            LtL_gpu = torch.tensor(2.0 * (L.T @ L), dtype=torch.float32, device=dev)  # (M, M)

        start_t = torch.tensor(start, dtype=torch.float32, device=dev)  # (7,)
        goal_t  = torch.tensor(goal,  dtype=torch.float32, device=dev)  # (7,)

        lower = torch.tensor([-166, -101, -166, -176, -166,  -1, -166],
                             dtype=torch.float32, device=dev) * (torch.pi / 180)
        upper = torch.tensor([ 166,  101,  166,   -4,  166, 215,  166],
                             dtype=torch.float32, device=dev) * (torch.pi / 180)

        # Initialise control points on GPU
        alpha_t = torch.randn(batch_size, num_channels, self.n_control, device=dev)
        if condition:
            alpha_t[:, :, 0]  = start_t
            alpha_t[:, :, -1] = goal_t

        model.train(False)
        period = 2
        collected = []   # for GPDS: trajectories from the last few denoising steps

        # ----- Scene conditioning (D3): precompute scene embedding once -----
        cond_emb = None
        if scene_obs is not None:
            with torch.no_grad():
                o = torch.tensor(scene_obs, dtype=torch.float32, device=dev)
                m = torch.tensor(scene_mask, dtype=torch.bool, device=dev)
                if o.dim() == 2:                       # (S,F) -> (1,S,F)
                    o = o.unsqueeze(0); m = m.unsqueeze(0)
                emb = model.scene_embed(o, m)          # (1, time_dim)
                cond_emb = emb.expand(batch_size, -1).contiguous()

        for t in range(self.T, 0, -1):
            if benchmarking:
                print(f"\rDenoising: {t} ", end="")

            # ----- Model forward — fully on GPU, no numpy -----
            with torch.no_grad():
                t_in = torch.full((batch_size,), float(t), dtype=torch.float32, device=dev)
                if cond_emb is None:
                    eps = model(alpha_t, t_in)         # unconditional model
                else:
                    # Classifier-free guidance: eps_u + w (eps_c - eps_u)
                    eps_c = model(alpha_t, t_in, scene_emb=cond_emb)
                    eps_u = model(alpha_t, t_in)       # null embedding path
                    eps = eps_u + cfg_weight * (eps_c - eps_u)
                eps  = torch.nan_to_num(eps, nan=0.0, posinf=0.0, neginf=0.0)
                # Clip eps to prevent gradient explosions
                eps  = torch.clamp(eps, -5.0, 5.0)

                # p_sample_using_posterior on GPU
                a  = alpha_gpu[t - 1].clamp(min=1e-8)
                ab = alpha_bar_gpu[t - 1].clamp(min=1e-8, max=1.0 - 1e-8)
                b  = beta_gpu[t - 1]
                z  = torch.randn_like(alpha_t)
                if t == 1:
                    z.zero_()
                alpha_t = (alpha_t - ((1 - a) / torch.sqrt(1 - ab)) * eps) \
                          / torch.sqrt(a) + b * z
                alpha_t = torch.nan_to_num(alpha_t, nan=0.0, posinf=0.0, neginf=0.0)
                alpha_t = torch.clamp(alpha_t, -10.0, 10.0)

            # ----- Gradient guidance -----
            if (t % period) < (period / 2) and t >= 5:
                # Expand to waypoints; clip before FK to avoid invalid joint angles
                with torch.no_grad():
                    q_t       = alpha_t @ B_gpu.T                     # (batch, 7, N)
                    q_int_gpu = torch.clamp(q_t[:, :, 1:-1],
                                            lower[:, None], upper[:, None])

                # guide.get_gradient: numpy in/out, GPU compute internally
                grad_q_np = guide.get_gradient(
                    q_int_gpu.cpu().numpy(), start[:], goal[:], t
                )  # (batch, 7, N-2)

                with torch.no_grad():
                    full_grad = torch.zeros(batch_size, num_channels,
                                            self.traj_len, device=dev)
                    full_grad[:, :, 1:-1] = torch.tensor(
                        grad_q_np, dtype=torch.float32, device=dev)
                    # Precondition: dJ/d_alpha = full_grad_q @ B
                    grad_alpha = full_grad @ B_gpu                    # (batch, 7, M)
                    # Zero boundary gradients — start/goal control points are
                    # pinned; updating them wastes computation and is undone below
                    grad_alpha[:, :, 0]  = 0.0
                    grad_alpha[:, :, -1] = 0.0
                    scale = gs_gpu[:, t - 1].view(batch_size, 1, 1)
                    alpha_t -= scale * grad_alpha

            # ----- Smoothness (curvature) regularization on control points -----
            if LtL_gpu is not None:
                with torch.no_grad():
                    grad_smooth = alpha_t @ LtL_gpu              # (batch, 7, M)
                    grad_smooth[:, :, 0]  = 0.0                  # don't move pinned
                    grad_smooth[:, :, -1] = 0.0                  # start/goal points
                    alpha_t -= smoothness_weight * grad_smooth

            # Re-pin boundary conditions (also guards against numerical drift)
            with torch.no_grad():
                if condition:
                    alpha_t[:, :, 0]  = start_t
                    alpha_t[:, :, -1] = goal_t

            # GPDS: collect candidate trajectories from the last few steps
            if extra_candidate_steps > 0 and t <= extra_candidate_steps:
                with torch.no_grad():
                    collected.append((alpha_t @ B_gpu.T).cpu().numpy())

        # Expand final control points to waypoints
        with torch.no_grad():
            trajectories = (alpha_t @ B_gpu.T).cpu().numpy()  # (batch, 7, N)
        if collected:
            # (n_steps * batch, 7, N) — diverse candidate pool for stitching
            return np.concatenate(collected, axis=0)
        return trajectories

    # ------------------------------------------------------------------
    # Training sample generation in control-point space
    # ------------------------------------------------------------------

    def generate_q_sample_poly(self, alpha0: np.ndarray,
                                condition: bool = True):
        """
        Like generate_q_sample but operates on Bernstein control points.

        Args:
            alpha0    : (batch, 7, n_control) clean control-point trajectories
            condition : whether to pin boundary control points

        Returns:
            X          : (batch, 7, n_control) noisy control points  [tensor]
            Y          : (batch, 7, n_control) noise                 [tensor]
            time_steps : (batch,) diffusion timestep                 [tensor]
        """
        b, c, m = alpha0.shape
        time_steps = np.random.randint(1, self.T + 1, size=(b,))
        eps = np.random.multivariate_normal(
            mean=np.zeros(m), cov=np.eye(m), size=(b, c)
        )

        alpha_t, _, _ = self.q_sample_from_x0(alpha0, time_steps, eps)

        if condition:
            alpha_t[:, :, 0]  = alpha0[:, :, 0]
            alpha_t[:, :, -1] = alpha0[:, :, -1]

        X = torch.tensor(alpha_t, dtype=torch.float32)
        Y = torch.tensor(eps,     dtype=torch.float32)
        t = torch.tensor(time_steps, dtype=torch.float32)
        return X, Y, t

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clip_joints(joints: np.ndarray) -> np.ndarray:
        """Clip interior waypoints to Franka joint limits."""
        lower = np.array([-166, -101, -166, -176, -166,   -1, -166],
                         dtype=np.float32) * (np.pi / 180)
        upper = np.array([ 166,  101,  166,  -4,   166,  215,  166],
                         dtype=np.float32) * (np.pi / 180)
        return np.clip(joints,
                       lower[np.newaxis, :, np.newaxis],
                       upper[np.newaxis, :, np.newaxis])
