"""
RRT-only baseline: solve each scene from scratch with RRT-Connect against the faithful
collision query (no diffusion, no repair). The attribution control for the completeness
fallback -- if this alone reaches ~90%, the learned prior is not contributing.

Sharded like run_worker.py; writes results/<dir>/gpd_w<k>.json (merge_results-compatible).
"""
import argparse, json, time, os
import numpy as np, torch
os.environ.setdefault('OMP_NUM_THREADS', '1')
import run_worker as RW
from datasets.load_test_dataset import TestDataset
from lib.environment import RobotEnvironment
from lib.guide import IntersectionVolumeGuide
from gpd.stitch import rrt_connect
from gpd.refine import densify

SCENE_TYPES = ['tabletop', 'cubby', 'merged_cubby', 'dresser']
lower = np.array([-166,-101,-166,-176,-166,-1,-166], np.float32)*(np.pi/180)
upper = np.array([166,101,166,-4,166,215,166], np.float32)*(np.pi/180)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='hybrid', choices=['global','hybrid','both'])
    ap.add_argument('--per_type', type=int, default=100)
    ap.add_argument('--num_workers', type=int, default=1)
    ap.add_argument('--worker_id', type=int, default=0)
    ap.add_argument('--rrt_iters', type=int, default=6000)
    ap.add_argument('--rrt_time', type=float, default=10.0)
    ap.add_argument('--densify', type=int, default=4)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    DEV = RW.DEVICE
    gc = RW.build_guide_cfgs([1], 64, 32)
    ds = TestDataset(args.dataset, d_path='datasets/'); env = RobotEnvironment(gui=False)
    res = []
    for st in SCENE_TYPES:
        cap = min(args.per_type, ds.data_nums[st])
        for i in RW.get_worker_indices(cap, args.worker_id, args.num_workers):
            env.clear_obstacles(); env.go_home()
            obs, cub, cyl, ncub, ncyl, sj, aik = ds.fetch_data(i, st)
            guide = IntersectionVolumeGuide(obs, DEV, gc, gc['total_batch_size'])
            gj = RW.filter_ik(guide, aik, sj)
            if ncub > 0: env.spawn_collision_cuboids(cub)
            if ncyl > 0: env.spawn_collision_cylinders(cyl)
            t0 = time.time()
            path = rrt_connect(np.asarray(sj, np.float32), np.asarray(gj, np.float32),
                               guide, DEV, max_iter=args.rrt_iters, seed=int(i),
                               collision_fn=env.configs_free, max_time=args.rrt_time)
            succ = 0
            if path is not None and len(path) >= 2:
                traj = np.clip(densify(np.stack(path).T, args.densify), lower[:,None], upper[:,None])
                succ = int(env.benchmark_trajectory(traj))
            pt = time.time() - t0
            res.append({'scene_type': st, 'scene_num': int(i), 'success': succ,
                        'plan_time': round(pt, 3)})
            print(f"  [W{args.worker_id}][{st}:{i+1:4d}] success={succ} t={pt:.2f}s", flush=True)
    with open(args.out, 'w') as f:
        json.dump({'method': 'gpd', 'worker_id': args.worker_id,
                   'num_workers': args.num_workers, 'results': res}, f, indent=2)


if __name__ == '__main__':
    main()
