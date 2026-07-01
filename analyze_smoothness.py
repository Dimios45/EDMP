"""
#4: Mechanistic figure for the compression-as-regularizer claim.

Quantifies *why* the compact n=8 prior plans better than the higher-capacity n=16
prior despite n=16's lower reconstruction error: the compact basis yields smoother
(lower-jerk) raw samples. We also report the swept clearance margin (sphere-SDF)
to show trajopt repair *increases* margin. Produces a table + a histogram PNG.

Usage: python analyze_smoothness.py [per_type=25]
Outputs: results/smoothness_stats.json, results/smoothness_jerk.png
"""
import os, sys, json, numpy as np, torch
os.environ.setdefault('OMP_NUM_THREADS', '1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import run_worker as RW
from datasets.load_test_dataset import TestDataset
from lib.environment import RobotEnvironment
from lib.guide import IntersectionVolumeGuide
from gpd.diffusion import PolynomialDiffusion
from diffusion.models.temporalunet import TemporalUNet
from gpd.sphere_collision import SphereSDFCost
from gpd.refine import trajopt_refine

DEV = RW.DEVICE
PER_TYPE = int(sys.argv[1]) if len(sys.argv) > 1 else 25
SCENE_TYPES = ['tabletop', 'cubby', 'merged_cubby', 'dresser']


def jerk_rms(traj):           # traj (7, M): RMS 3rd time-difference (rad/step^3)
    j = np.diff(traj, n=3, axis=1)
    return float(np.sqrt((j ** 2).mean()))


def accel_rms(traj):
    a = np.diff(traj, n=2, axis=1)
    return float(np.sqrt((a ** 2).mean()))


def load(n_control):
    diff = PolynomialDiffusion(T=64, device=DEV, n_control=n_control, traj_len=50,
                               variance_thresh=0.02)
    mdir = f'./models/GPDModel64_N{n_control}'
    model = TemporalUNet(model_name=mdir, input_dim=7, time_dim=32,
                         dims=(32, 64, 128, 256), device=DEV)
    return diff, model


gc = RW.build_guide_cfgs([1], 64, 32); bs = gc['total_batch_size']
ds = TestDataset('hybrid', d_path='datasets/'); env = RobotEnvironment(gui=False)
d8, m8 = load(8); d16, m16 = load(16)

rows = {'n8': {'jerk': [], 'accel': [], 'margin': []},
        'n16': {'jerk': [], 'accel': [], 'margin': []},
        'n8_repaired': {'jerk': [], 'accel': [], 'margin': []}}

for st in SCENE_TYPES:
    cap = min(PER_TYPE, ds.data_nums[st])
    for i in range(cap):
        env.clear_obstacles(); env.go_home()
        obs, cub, cyl, ncub, ncyl, sj, aik = ds.fetch_data(i, st)
        g = IntersectionVolumeGuide(obs, DEV, gc, bs); gj = RW.filter_ik(g, aik, sj)
        if ncub > 0: env.spawn_collision_cuboids(cub)
        if ncyl > 0: env.spawn_collision_cylinders(cyl)
        sph = SphereSDFCost(g, DEV)

        def margin(traj):       # min signed clearance over the path (m); <0 = penetrate
            jt = torch.tensor(traj[None], dtype=torch.float32, device=DEV)
            return float(sph.clearance(jt).min())

        for tag, dd, mm in [('n8', d8, m8), ('n16', d16, m16)]:
            trajs = dd.denoise_guided_poly(model=mm, guide=g, num_channels=7,
                        guidance_schedule=gc['guidance_schedule'], batch_size=bs,
                        start=sj, goal=gj, condition=True)
            traj = g.choose_best_trajectory(sj, gj, trajs)         # raw prior output
            rows[tag]['jerk'].append(jerk_rms(traj))
            rows[tag]['accel'].append(accel_rms(traj))
            rows[tag]['margin'].append(margin(traj))
            if tag == 'n8':
                rep = trajopt_refine(traj, g, DEV, iters=60, collision_fn=env.configs_free)
                rows['n8_repaired']['jerk'].append(jerk_rms(rep))
                rows['n8_repaired']['accel'].append(accel_rms(rep))
                rows['n8_repaired']['margin'].append(margin(rep))
    print(f"[{st}] done", flush=True)

summ = {}
print(f"\n{'model':14} {'jerk_rms':>10} {'accel_rms':>10} {'margin(m)':>10}")
for tag, d in rows.items():
    j, a, m = np.array(d['jerk']), np.array(d['accel']), np.array(d['margin'])
    summ[tag] = {'jerk_mean': j.mean(), 'jerk_std': j.std(),
                 'accel_mean': a.mean(), 'margin_mean': m.mean(), 'n': len(j)}
    print(f"{tag:14} {j.mean():10.4f} {a.mean():10.4f} {m.mean():10.4f}")
json.dump(summ, open('results/smoothness_stats.json', 'w'), indent=2)

plt.figure(figsize=(6, 4))
for tag, c in [('n8', 'tab:blue'), ('n16', 'tab:red')]:
    plt.hist(rows[tag]['jerk'], bins=30, alpha=0.55, label=f'{tag} prior', color=c)
plt.xlabel('joint jerk RMS (rad/step$^3$)'); plt.ylabel('count')
plt.title('Compact n=8 prior yields smoother samples than n=16')
plt.legend(); plt.tight_layout()
plt.savefig('results/smoothness_jerk.png', dpi=130)
print("saved results/smoothness_stats.json + results/smoothness_jerk.png")
