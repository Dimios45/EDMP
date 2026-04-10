"""
Inference script for GPD (Guided Polynomial Diffusion).

Usage:
    python infer_gpd.py -c ./benchmark/cfgs/cfg_gpd.yaml

Key differences from infer_serial.py:
  - Diffusion runs in Bernstein control-point space (7 x n_control).
  - Gradients are pre-conditioned with the Bernstein matrix B.
  - Stitching assembles a collision-free trajectory from K candidates.
"""

from lib import *
from diffusion.models.temporalunet import TemporalUNet
from gpd.diffusion import PolynomialDiffusion
from gpd.stitch    import stitch
from datasets.load_test_dataset import TestDataset

import argparse
import os
import time
import numpy as np
import torch

from autolab_core import YamlConfig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='GPD Inference')
    parser.add_argument('-c', '--cfg_path', type=str,
                        default='./benchmark/cfgs/cfg_gpd.yaml')
    args = parser.parse_args()

    cfg = YamlConfig(args.cfg_path)

    # ---- Params ----
    device      = cfg['model']['device'] if torch.cuda.is_available() else 'cpu'
    T           = cfg['model']['T']
    n_control   = cfg['model']['n_control']
    traj_len    = cfg['model']['traj_len']
    num_channels = cfg['model']['num_channels']
    use_stitch  = cfg['model'].get('use_stitch', True)

    # ---- Dataset ----
    dataset = TestDataset(cfg['dataset']['dataset_type'],
                          d_path=cfg['dataset']['path'])
    print(f"Benchmarking dataset: {cfg['dataset']['dataset_type']}"
          f"\tScene types: {cfg['dataset']['scene_types']}"
          f"\tguides: {cfg['guide']['guides']}")

    # ---- Environment & diffuser ----
    env      = RobotEnvironment(gui=cfg['general']['gui'])
    diffuser = PolynomialDiffusion(T=T, device=device,
                                    n_control=n_control, traj_len=traj_len)

    # ---- Load TemporalUNet trained on (7, n_control) ----
    model_name = cfg['model']['model_dir'] + f'GPDModel{T}_N{n_control}'
    if not os.path.exists(model_name):
        print(f"Model not found at {model_name}. Run train_gpd.py first.")
        _ = input("Press anything to exit")
        exit()

    denoiser = TemporalUNet(
        model_name=model_name,
        input_dim=num_channels,
        time_dim=32,
        dims=(32, 64, 128, 256),
        device=device,
    )

    # ---- Guide configs ----
    guides              = cfg['guide']['guides']
    guide_dpath         = cfg['guide']['guide_path']
    batch_size_per_guide = cfg['guide']['batch_size_per_guide']
    num_guides          = len(guides)
    total_batch_size    = num_guides * batch_size_per_guide

    guide_cfgs = {
        'batch_size_per_guide': batch_size_per_guide,
        'total_batch_size':     total_batch_size,
        'clearance':            np.zeros((total_batch_size, T)),
        'expansion':            np.zeros((total_batch_size, T)),
        'guidance_method':      np.zeros((total_batch_size,)),
        'grad_norm':            np.zeros((total_batch_size,)),
        'guidance_schedule':    np.zeros((total_batch_size, T)),
        'volume_trust_region':  np.zeros((total_batch_size,)),
    }

    for i, g_id in enumerate(guides):
        g_cfg = YamlConfig(guide_dpath + f'cfgs/guide{g_id}.yaml')
        sl = slice(i * batch_size_per_guide, (i + 1) * batch_size_per_guide)

        guide_cfgs['clearance'][sl, :] = np.linspace(
            g_cfg['hyperparameters']['obstacle_clearance']['range'][0],
            g_cfg['hyperparameters']['obstacle_clearance']['range'][1], T)

        o_e = g_cfg['hyperparameters']['obstacle_expansion']
        T_orig = 255  # guide configs were written for T=255
        for isr_key, val_key in [('isr1', 'val1'), ('isr2', 'val2'),
                                  ('isr3', 'val3')]:
            r = o_e[isr_key]
            # Scale range indices from T_orig to T and clamp to [0, T]
            r0 = int(np.clip(round(r[0] * T / T_orig), 0, T))
            r1 = int(np.clip(round(r[1] * T / T_orig), 0, T))
            n  = abs(r1 - r0)
            if n > 0:
                guide_cfgs['expansion'][sl, r0:r1] = np.linspace(
                    o_e[val_key][0], o_e[val_key][1], num=n)

        guide_cfgs['guidance_method'][sl] = (
            1 if g_cfg['hyperparameters']['guidance_method'] == 'sv' else 0)
        guide_cfgs['grad_norm'][sl] = (
            1 if g_cfg['hyperparameters']['grad_norm'] else 0)
        guide_cfgs['guidance_schedule'][sl, :] = (
            (1.4 + np.arange(T) / T)
            if g_cfg['hyperparameters']['guidance_schedule']['type'] == 'varying'
            else g_cfg['hyperparameters']['guidance_schedule']['scale_val'])
        guide_cfgs['volume_trust_region'][sl] = (
            g_cfg['hyperparameters']['volume_trust_region'])

    # ---- Benchmark loop ----
    t_success = 0
    for scene_type in cfg['dataset']['scene_types']:
        for i in range(dataset.data_nums[scene_type]):
            print(f"Scene num: {i + 1}\tSuccess_rate: {t_success}/{i}")
            env.clear_obstacles()
            env.go_home()

            (obstacle_config, cuboid_config, cylinder_config,
             num_cuboids, num_cylinders,
             start_joints, all_ik_goals) = dataset.fetch_data(
                scene_num=i, scene_type=scene_type)

            guide = IntersectionVolumeGuide(
                obstacle_config=obstacle_config,
                device=device,
                guide_cfgs=guide_cfgs,
                batch_size=total_batch_size,
            )
            metrics_calculator = MetricsCalculator(guide)

            # ---- IK filtering (identical to infer_serial.py) ----
            st3 = time.time()
            volumes = guide.cost(
                torch.tensor(all_ik_goals.reshape((-1, 7, 1)), device=device),
                0, batch_size=all_ik_goals.shape[0]
            ).sum(axis=(1, 2)).cpu().numpy()
            min_vol = np.min(volumes)
            indices = np.argsort(volumes)
            rearranged = volumes[indices]
            goal_joints = all_ik_goals[indices]
            goal_joints = goal_joints[rearranged < min_vol + 0.0008]
            ideal_ind   = np.argmin(
                np.linalg.norm(start_joints - goal_joints, axis=1))
            goal_joints = goal_joints[ideal_ind]
            print(f"IK time: {time.time() - st3:.3f}s")

            # ---- Polynomial diffusion ----
            start_time = time.time()
            st4 = time.time()

            trajectories = diffuser.denoise_guided_poly(
                model=denoiser,
                guide=guide,
                num_channels=num_channels,
                guidance_schedule=guide_cfgs['guidance_schedule'],
                batch_size=total_batch_size,
                start=start_joints,
                goal=goal_joints,
                condition=True,
                benchmarking=True,
            )
            print(f"\nDenoising time: {time.time() - st4:.3f}s")

            # ---- Trajectory selection / stitching ----
            if use_stitch:
                trajectory = stitch(trajectories, guide, device)
            else:
                trajectory = guide.choose_best_trajectory(
                    start_joints, goal_joints, trajectories)

            # Clip to Franka joint limits (Bernstein can overshoot between
            # control points; stitched linear bridges can too)
            lower = np.array([-166, -101, -166, -176, -166,   -1, -166],
                             dtype=np.float32) * (np.pi / 180)
            upper = np.array([ 166,  101,  166,   -4,  166,  215,  166],
                             dtype=np.float32) * (np.pi / 180)
            trajectory = np.clip(trajectory,
                                 lower[:, np.newaxis],
                                 upper[:, np.newaxis])

            print(f"Planning Time: {time.time() - start_time:.3f}s")

            # ---- Spawn obstacles and evaluate ----
            if num_cuboids > 0:
                env.spawn_collision_cuboids(cuboid_config)
            if num_cylinders > 0:
                env.spawn_collision_cylinders(cylinder_config)

            success = env.benchmark_trajectory(trajectory)
            t_success += success
            print(f"Success: {success}")
