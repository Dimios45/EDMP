"""
PyTorch dataset for GPD training.

Reads a preprocessed HDF5 file produced by gpd/preprocess_data.py.
Each call to generate_training_batch returns a random mini-batch of
noisy control-point trajectories ready for the diffusion training objective.
"""

import h5py
import numpy as np
import torch


class BernsteinTrajectoryDataset:

    def __init__(self, hdf5_path: str, n_diffusion_steps: int,
                 variance_thresh: float = 0.02, schedule: str = 'linear'):
        """
        Args:
            hdf5_path        : path to HDF5 produced by preprocess_data.py
            n_diffusion_steps: T (number of diffusion steps)
            variance_thresh  : terminal beta for the linear schedule (D1: tune
                               so alpha_bar_T -> 0 for the short T=64 chain)
            schedule         : 'linear' or 'cosine'
        """
        self.T               = n_diffusion_steps
        self.path            = hdf5_path
        self.variance_thresh = variance_thresh
        self.schedule        = schedule

        with h5py.File(hdf5_path, 'r') as f:
            self.N          = f['control_points'].shape[0]
            self.n_control  = f['control_points'].shape[2]
            self.n_channels = f['control_points'].shape[1]

        print(f"Dataset: {self.N} trajectories, "
              f"n_control={self.n_control}, channels={self.n_channels}")

        # Load entirely into RAM if feasible (< ~4 GB)
        with h5py.File(hdf5_path, 'r') as f:
            nbytes = f['control_points'].nbytes
            if nbytes < 4 * (1 << 30):
                self._data = f['control_points'][:]  # (N, 7, M) float32
                self._loaded = True
                print(f"Loaded {nbytes / 1e6:.1f} MB into RAM.")
            else:
                self._data   = None
                self._loaded = False

    def _get_batch_alpha(self, idx: np.ndarray) -> np.ndarray:
        if self._loaded:
            return self._data[idx].astype(np.float64)
        with h5py.File(self.path, 'r') as f:
            return f['control_points'][idx].astype(np.float64)

    def generate_training_batch(self, batch_size: int):
        """
        Returns (X, Y, t_steps) ready for diffusion training.

        X      : (B, 7, M) noisy control points  [float32 tensor]
        Y      : (B, 7, M) noise                 [float32 tensor]
        t_steps: (B,)      diffusion timestep     [float32 tensor]
        """
        idx    = np.random.randint(0, self.N, size=(batch_size,))
        alpha0 = self._get_batch_alpha(idx)              # (B, 7, M) float64
        B, C, M = alpha0.shape

        t_steps = np.random.randint(1, self.T + 1, size=(B,))

        # Variance schedule (D1: calibrated to the short chain). Must match the
        # schedule used by PolynomialDiffusion at inference time.
        if self.schedule == 'cosine':
            f = np.cos((np.arange(self.T + 1) / self.T + 0.008)
                       / 1.008 * np.pi / 2) ** 2
            abar_full = f / f[0]                               # alpha_bar_0..T
            full_alpha_bar = abar_full[1:]                     # length T
        else:
            beta = np.linspace(0, self.variance_thresh, self.T + 1)[1:]
            full_alpha_bar = np.array([np.prod(1 - beta[:t])
                                       for t in range(1, self.T + 1)])
        alpha_bar = full_alpha_bar[t_steps - 1]

        eps = np.random.randn(B, C, M)

        ab  = alpha_bar[:, np.newaxis, np.newaxis]
        alpha_t = np.sqrt(ab) * alpha0 + np.sqrt(1 - ab) * eps

        # Pin boundary conditions (start = alpha[:, :, 0], goal = alpha[:, :, -1])
        alpha_t[:, :, 0]  = alpha0[:, :, 0]
        alpha_t[:, :, -1] = alpha0[:, :, -1]

        X = torch.tensor(alpha_t, dtype=torch.float32)
        Y = torch.tensor(eps,     dtype=torch.float32)
        t = torch.tensor(t_steps, dtype=torch.float32)
        return X, Y, t
