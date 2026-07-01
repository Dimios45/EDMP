"""
Failure-mode breakdown of the GPD baseline (uncond N8 + guidance + faithful
RRT stitch) on hybrid. Categorizes each scene:
  goal_infeasible : selected goal IK config collides (or start collides)
  success         : benchmark_trajectory == 1
  path_collision  : stitched path has statically-colliding waypoints
  dynamic_only    : path statically free but execution collides (discretization)
"""
import os, sys, json, numpy as np, torch
os.environ.setdefault('OMP_NUM_THREADS', '1')

import run_worker as RW
from datasets.load_test_dataset import TestDataset
from lib.environment import RobotEnvironment
from gpd.diffusion import PolynomialDiffusion
from diffusion.models.temporalunet import TemporalUNet
from lib.guide import IntersectionVolumeGuide
from gpd.stitch import stitch
from gpd.refine import trajopt_refine

DEV = RW.DEVICE
PER_TYPE = int(sys.argv[1]) if len(sys.argv) > 1 else 50
MODE     = sys.argv[2] if len(sys.argv) > 2 else 'base'   # 'base' | 'd6' (apply repair)
SCENE_TYPES = ['tabletop', 'cubby', 'merged_cubby', 'dresser']
lower = np.array([-166,-101,-166,-176,-166,-1,-166], np.float32)*(np.pi/180)
upper = np.array([166,101,166,-4,166,215,166], np.float32)*(np.pi/180)

diff = PolynomialDiffusion(T=64, device=DEV, n_control=8, traj_len=50, variance_thresh=0.02)
model = TemporalUNet(model_name='./models/GPDModel64_N8', input_dim=7, time_dim=32,
                     dims=(32,64,128,256), device=DEV)
gc = RW.build_guide_cfgs([1], 64, 32)
total_bs = gc['total_batch_size']
ds = TestDataset('hybrid', d_path='datasets/')
env = RobotEnvironment(gui=False)

cats = {}
def bump(st, c): cats.setdefault(st, {}).setdefault(c, 0); cats[st][c]+=1

for st in SCENE_TYPES:
    cap = min(PER_TYPE, ds.data_nums[st])
    for i in range(cap):
        env.clear_obstacles(); env.go_home()
        obs_cfg, cub, cyl, ncub, ncyl, start_j, all_ik = ds.fetch_data(i, st)
        guide = IntersectionVolumeGuide(obs_cfg, DEV, gc, total_bs)
        goal_j = RW.filter_ik(guide, all_ik, start_j)
        trajs = diff.denoise_guided_poly(model=model, guide=guide, num_channels=7,
                    guidance_schedule=gc['guidance_schedule'], batch_size=total_bs,
                    start=start_j, goal=goal_j, condition=True)
        if ncub > 0: env.spawn_collision_cuboids(cub)
        if ncyl > 0: env.spawn_collision_cylinders(cyl)
        gfree = bool(env.configs_free(goal_j[None])[0])
        sfree = bool(env.configs_free(start_j[None])[0])
        traj = stitch(trajs, guide, DEV, use_rrt=True, collision_fn=env.configs_free)
        traj = np.clip(traj, lower[:,None], upper[:,None])
        if MODE == 'd6':
            traj = trajopt_refine(traj, guide, DEV, iters=60,
                                  collision_fn=env.configs_free)
            traj = np.clip(traj, lower[:,None], upper[:,None])
        static_free = env.configs_free(traj.T)          # (Nout,)
        n_col = int((~static_free).sum())
        succ = int(env.benchmark_trajectory(traj))
        if not (gfree and sfree):
            c = 'goal_infeasible'
        elif succ:
            c = 'success'
        elif n_col > 0:
            c = 'path_collision'
        else:
            c = 'dynamic_only'
        bump(st, c)
    print(f"[{st}] {cats[st]}", flush=True)

# aggregate
agg = {}
for st in SCENE_TYPES:
    for c, n in cats[st].items():
        agg[c] = agg.get(c, 0) + n
tot = sum(agg.values())
print("\n=== AGGREGATE (n=%d) ===" % tot)
for c in ['success','goal_infeasible','path_collision','dynamic_only']:
    print(f"  {c:18s} {agg.get(c,0):4d}  ({100*agg.get(c,0)/tot:.1f}%)")
outf = 'results/failure_modes.json' if MODE=='base' else f'results/failure_modes_{MODE}.json'
json.dump({'mode':MODE,'per_type':cats,'agg':agg}, open(outf,'w'), indent=2)
print("saved", outf)
