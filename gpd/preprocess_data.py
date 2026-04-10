"""
Preprocess MPInets HDF5 training data into Bernstein control-point format.

Usage:
    python -m gpd.preprocess_data \
        --input /path/to/mpinets/train.hdf5 \
        --output /path/to/gpd_train.hdf5 \
        --n_control 8 \
        --traj_len 50

The input HDF5 is expected to have a dataset called 'joint_angles' of shape
(N, 7, 50) or (N, 50, 7).  Adjust --transpose if needed.

Output HDF5 has:
    'control_points' : (N, 7, n_control)  float32 Bernstein coefficients
    'joint_angles'   : (N, 7, traj_len)   float32 original waypoints (kept for
                       verification)
"""

import argparse
import h5py
import numpy as np
from tqdm import tqdm

from gpd.bernstein import BernsteinLayer


def preprocess(input_path: str, output_path: str,
               n_control: int = 8, traj_len: int = 50,
               batch: int = 8192, transpose: bool = False,
               key: str = 'hybrid_solutions'):

    bern = BernsteinLayer(traj_len=traj_len, n_control=n_control)

    with h5py.File(input_path, 'r') as f_in:
        traj_data = f_in[key]  # (N, 7, 50) or (N, 50, 7)
        N = traj_data.shape[0]
        print(f"Input shape: {traj_data.shape}")

        with h5py.File(output_path, 'w') as f_out:
            ds_cp = f_out.create_dataset(
                'control_points', shape=(N, 7, n_control), dtype='float32'
            )
            ds_jt = f_out.create_dataset(
                'joint_angles',   shape=(N, 7, traj_len), dtype='float32'
            )

            for start in tqdm(range(0, N, batch), desc="Fitting Bernstein"):
                end  = min(start + batch, N)
                q_np = traj_data[start:end][:]   # (B, 7, 50) or (B, 50, 7)

                if transpose:
                    q_np = q_np.transpose(0, 2, 1)  # → (B, 7, 50)

                q_np = q_np.astype(np.float64)
                alpha = bern.to_control_np(q_np)   # (B, 7, n_control)

                ds_cp[start:end] = alpha.astype(np.float32)
                ds_jt[start:end] = q_np.astype(np.float32)

    print(f"Saved preprocessed dataset to {output_path}")
    print(f"  N={N}  control_points shape: ({N}, 7, {n_control})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',     required=True)
    parser.add_argument('--output',    required=True)
    parser.add_argument('--n_control', type=int, default=8)
    parser.add_argument('--traj_len',  type=int, default=50)
    parser.add_argument('--batch',     type=int, default=8192)
    parser.add_argument('--key',       type=str, default='hybrid_solutions',
                        help='HDF5 dataset key containing trajectories')
    parser.add_argument('--transpose', action='store_true',
                        help='Transpose input from (N,50,7) to (N,7,50)')
    args = parser.parse_args()

    preprocess(args.input, args.output, args.n_control,
               args.traj_len, args.batch, args.transpose, args.key)
