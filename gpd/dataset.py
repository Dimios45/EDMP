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

    def __init__(self, hdf5_path: str, n_diffusion_steps: int):
        """
        Args:
            hdf5_path        : path to HDF5 produced by preprocess_data.py
            n_diffusion_steps: T (number of diffusion steps)
        """
        self.T    = n_diffusion_steps
        self.path = hdf5_path

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

        # Linear variance schedule (same as EDMP)
        beta      = np.linspace(0, 0.02, self.T + 1)[1:]
        alpha_bar = np.array([np.prod(1 - beta[:t]) for t in t_steps])

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
