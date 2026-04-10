"""
Merge two preprocessed Bernstein HDF5 files into one.

Usage:
    python -m gpd.merge_datasets \
        --inputs ./gpd_train.hdf5 ./gpd_train_global.hdf5 \
        --output ./gpd_train_combined.hdf5
"""

import argparse
import h5py
import numpy as np
from tqdm import tqdm


def merge(inputs: list, output: str, chunk: int = 8192):
    # Read shapes
    shapes = []
    for path in inputs:
        with h5py.File(path, 'r') as f:
            shapes.append(f['control_points'].shape)
    print("Input shapes:", shapes)

    N_total   = sum(s[0] for s in shapes)
    n_joints  = shapes[0][1]
    n_control = shapes[0][2]

    with h5py.File(output, 'w') as f_out:
        ds = f_out.create_dataset(
            'control_points', shape=(N_total, n_joints, n_control),
            dtype='float32', chunks=(min(chunk, N_total), n_joints, n_control)
        )

        offset = 0
        for path, shape in zip(inputs, shapes):
            N = shape[0]
            with h5py.File(path, 'r') as f_in:
                src = f_in['control_points']
                for start in tqdm(range(0, N, chunk), desc=f"Merging {path}"):
                    end = min(start + chunk, N)
                    ds[offset + start : offset + end] = src[start:end]
            offset += N

    print(f"Saved combined dataset to {output}  (N={N_total})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    merge(args.inputs, args.output)
