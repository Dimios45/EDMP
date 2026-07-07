"""
Build per-scene obstacle tensors (scene-conditioned GPD).

Each scene -> fixed set of obstacle primitive slots (40 cuboid + 12 cylinder):
  feat (12) = [is_cuboid, is_cylinder, cx,cy,cz, ex,ey,ez, qx,qy,qz,qw]
    cuboid:   type [1,0], center, half-extents (dims/2), quaternion
    cylinder: type [0,1], center, [radius, height, 0], quaternion
  padding slots are all-zero; a separate mask marks real obstacles.
DeepSets encoder is permutation-invariant + masked, so fixed slots are fine.

Stored PER UNIQUE SCENE (N rows). The combined CP dataset is
hybrid_solutions(N) ++ global_solutions(N), both sharing the same obstacle
arrays per index, so the dataset maps combined_idx -> idx % N.

Output: scenes_unique.hdf5 with
  'obstacles' (N, 52, 12) float32
  'mask'      (N, 52)     bool
"""
import argparse, h5py, numpy as np
from tqdm import tqdm

FEAT = 12


def build(raw_path: str, out_path: str, batch: int = 8192):
    with h5py.File(raw_path, 'r') as f:
        N = f['cuboid_centers'].shape[0]
        ncub = f['cuboid_centers'].shape[1]   # 40
        ncyl = f['cylinder_centers'].shape[1] # 12
        S = ncub + ncyl
        print(f"raw N={N}, slots: {ncub} cuboid + {ncyl} cylinder = {S}")

        with h5py.File(out_path, 'w') as fo:
            obs_ds = fo.create_dataset('obstacles', (N, S, FEAT), dtype='float32')
            msk_ds = fo.create_dataset('mask', (N, S), dtype='bool')

            for s in tqdm(range(0, N, batch), desc="scenes"):
                e = min(s + batch, N); b = e - s
                cub_c = f['cuboid_centers'][s:e]        # (b,40,3)
                cub_d = f['cuboid_dims'][s:e]           # (b,40,3)
                cub_q = f['cuboid_quaternions'][s:e]    # (b,40,4)
                cyl_c = f['cylinder_centers'][s:e]      # (b,12,3)
                cyl_r = f['cylinder_radii'][s:e]        # (b,12,1)
                cyl_h = f['cylinder_heights'][s:e]      # (b,12,1)
                cyl_q = f['cylinder_quaternions'][s:e]  # (b,12,4)

                out = np.zeros((b, S, FEAT), np.float32)
                # --- cuboid block [0:40] ---
                cub_real = cub_d.sum(axis=2) > 1e-6                 # (b,40)
                out[:, :ncub, 0] = 1.0                              # is_cuboid
                out[:, :ncub, 2:5] = cub_c
                out[:, :ncub, 5:8] = cub_d * 0.5
                out[:, :ncub, 8:12] = cub_q
                # --- cylinder block [40:52] ---
                cyl_real = cyl_r[:, :, 0] > 1e-6                    # (b,12)
                out[:, ncub:, 1] = 1.0                              # is_cylinder
                out[:, ncub:, 2:5] = cyl_c
                out[:, ncub:, 5] = cyl_r[:, :, 0]
                out[:, ncub:, 6] = cyl_h[:, :, 0]
                out[:, ncub:, 8:12] = cyl_q

                msk = np.concatenate([cub_real, cyl_real], axis=1) # (b,52)
                out[~msk] = 0.0                                     # zero padding rows

                obs_ds[s:e] = out
                msk_ds[s:e] = msk

    print(f"Saved {out_path}: obstacles (N={N}, {S}, {FEAT})")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='/mnt/hdd/mpinets_hybrid_training_data/train/train.hdf5')
    p.add_argument('--output', default='scenes_unique.hdf5')
    args = p.parse_args()
    build(args.input, args.output)
