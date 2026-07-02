"""
Continuous-feasibility audit for guided diffusion planners (parameterized).

Measures continuous SR@R (fraction of plans collision-free under R samples per edge,
via densify + faithful PyBullet configs_free) for a planner's base vs post-repair
pipeline, across resolutions R. Reveals how much "waypoint-free" success is actually
swept-colliding, and that repair makes success resolution-flat.

Usage:
  python continuous_audit.py --method gpd  --dataset hybrid --per_type 25
  python continuous_audit.py --method gpd  --dataset global --per_type 25
  python continuous_audit.py --method edmp --dataset hybrid --per_type 15
Output: results/continuous_audit_<method>_<dataset>.json
"""
import os, sys, json, argparse, numpy as np, torch
os.environ.setdefault('OMP_NUM_THREADS', '1')

import run_worker as RW
from datasets.load_test_dataset import TestDataset
from lib.environment import RobotEnvironment
from lib.guide import IntersectionVolumeGuide
from gpd.diffusion import PolynomialDiffusion
from diffusion.models.temporalunet import TemporalUNet
from diffusion import Diffusion
from gpd.edmp_gpu import denoise_guided_gpu
from gpd.stitch import stitch
from gpd.refine import trajopt_refine, densify

SCENE_TYPES = ['tabletop', 'cubby', 'merged_cubby', 'dresser']
RES = [1, 2, 4, 8, 16, 32]
lower = np.array([-166,-101,-166,-176,-166,-1,-166], np.float32)*(np.pi/180)
upper = np.array([166,101,166,-4,166,215,166], np.float32)*(np.pi/180)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', default='gpd', choices=['gpd', 'edmp'])
    ap.add_argument('--dataset', default='hybrid', choices=['global', 'hybrid', 'both'])
    ap.add_argument('--per_type', type=int, default=25)
    ap.add_argument('--guides', default='1')
    ap.add_argument('--batch_per', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    DEV = RW.DEVICE
    guides = [int(g) for g in args.guides.split(',')]

    if args.method == 'gpd':
        diff = PolynomialDiffusion(T=64, device=DEV, n_control=8, traj_len=50, variance_thresh=0.02)
        model = TemporalUNet(model_name='./models/GPDModel64_N8', input_dim=7, time_dim=32,
                             dims=(32,64,128,256), device=DEV)
        gc = RW.build_guide_cfgs(guides, 64, args.batch_per)
        modes = ['base', 'repair_s1', 'dense_s4']
    else:
        diff = Diffusion(T=255, device=DEV)
        model = TemporalUNet(model_name='./models/TemporalUNetModel255_N50', input_dim=7,
                             time_dim=32, dims=(32,64,128,256,512,512), device=DEV)
        gc = RW.build_guide_cfgs(RW.EDMP_GUIDES, 255, RW.EDMP_BATCH_PER)
        modes = ['base', 'repair_s1']
    total_bs = gc['total_batch_size']
    ds = TestDataset(args.dataset, d_path='datasets/'); env = RobotEnvironment(gui=False)

    free_at = {m: {r: 0 for r in RES} for m in modes}
    dyn = {m: 0 for m in modes}; n_tot = 0

    def continuous_free(traj):
        return {r: bool(env.configs_free(densify(traj, r).T if r > 1 else traj.T).all()) for r in RES}

    for st in SCENE_TYPES:
        cap = min(args.per_type, ds.data_nums[st])
        for i in range(cap):
            env.clear_obstacles(); env.go_home()
            obs, cub, cyl, ncub, ncyl, sj, aik = ds.fetch_data(i, st)
            guide = IntersectionVolumeGuide(obs, DEV, gc, total_bs)
            gj = RW.filter_ik(guide, aik, sj)
            if args.method == 'gpd':
                trajs = diff.denoise_guided_poly(model=model, guide=guide, num_channels=7,
                            guidance_schedule=gc['guidance_schedule'], batch_size=total_bs,
                            start=sj, goal=gj, condition=True)
            else:
                trajs = denoise_guided_gpu(diffuser=diff, model=model, guide=guide,
                            traj_len=50, num_channels=7, guidance_schedule=gc['guidance_schedule'],
                            batch_size=total_bs, start=sj, goal=gj, condition=True, benchmarking=False)
            if ncub > 0: env.spawn_collision_cuboids(cub)
            if ncyl > 0: env.spawn_collision_cylinders(cyl)
            if args.method == 'gpd':
                base = np.clip(stitch(trajs, guide, DEV, use_rrt=True, collision_fn=env.configs_free),
                               lower[:,None], upper[:,None])
            else:
                base = np.clip(guide.choose_best_trajectory(sj, gj, trajs), lower[:,None], upper[:,None])
            out = {'base': base}
            out['repair_s1'] = trajopt_refine(base, guide, DEV, iters=60, edge_samples=1,
                                              collision_fn=env.configs_free, out_densify=1)
            if 'dense_s4' in modes:
                out['dense_s4'] = trajopt_refine(base, guide, DEV, iters=60, edge_samples=4,
                                                collision_fn=env.configs_free, out_densify=1)
            for m in modes:
                cf = continuous_free(out[m])
                for r in RES: free_at[m][r] += int(cf[r])
                dyn[m] += int(env.benchmark_trajectory(out[m]))
            n_tot += 1
        print(f"[{st}] done ({n_tot})", flush=True)

    res = {'method': args.method, 'dataset': args.dataset, 'n': n_tot, 'resolutions': RES,
           'continuous_SR': {m: {r: round(100*free_at[m][r]/n_tot,1) for r in RES} for m in modes},
           'dynamic_SR': {m: round(100*dyn[m]/n_tot,1) for m in modes}}
    print(f"\n=== {args.method}/{args.dataset} continuous SR@R (n={n_tot}) ===")
    print(f"{'R':>4} " + " ".join(f"{m:>10}" for m in modes))
    for r in RES: print(f"{r:>4} " + " ".join(f"{res['continuous_SR'][m][r]:>10}" for m in modes))
    print("dyn  " + " ".join(f"{res['dynamic_SR'][m]:>10}" for m in modes))
    outf = f'results/continuous_audit_{args.method}_{args.dataset}.json'
    json.dump(res, open(outf, 'w'), indent=2); print("saved", outf)


if __name__ == '__main__':
    main()
