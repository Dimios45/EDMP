"""
Parallel worker for benchmark.

Each worker handles a non-overlapping slice of scenes, identified by worker_id.
The 1800 scenes are distributed round-robin across num_workers so each worker
gets a contiguous block within each scene-type.

Usage:
    python run_worker.py --method gpd --worker_id 0 --num_workers 4 \
                         --out results/gpd_w0.json

Scene layout (FULL_CAPS = [600, 300, 300, 600]):
    tabletop  [0..599]   — 600 scenes
    cubby     [0..299]   — 300 scenes
    merged_cubby [0..299]
    dresser   [0..599]   — 600 scenes
"""

import argparse, json, time
import numpy as np
import torch

from autolab_core import YamlConfig
from lib import *
from diffusion import *
from datasets.load_test_dataset import TestDataset
from diffusion.models.temporalunet import TemporalUNet as TemporalUNetGPD
from gpd.diffusion import PolynomialDiffusion
from gpd.stitch    import stitch
from gpd.edmp_gpu  import denoise_guided_gpu

# ── Static config ──────────────────────────────────────────────────────────────
SCENE_TYPES  = ['tabletop', 'cubby', 'merged_cubby', 'dresser']
FULL_CAPS    = [600, 300, 300, 600]
MINI_CAPS    = [ 13,  12,  13,  12]
DATASET_TYPE = 'hybrid'
DATASET_PATH = './datasets/'
GUI          = False
DEVICE       = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# EDMP
EDMP_T         = 255
EDMP_TRAJ_LEN  = 50
EDMP_MODEL_DIR = './models/'
EDMP_GUIDES    = [1, 2, 3, 4, 5, 10, 11, 13, 14, 16, 18, 21]
EDMP_BATCH_PER = 10

# GPD
GPD_T          = 64
GPD_N_CONTROL  = 8
GPD_TRAJ_LEN   = 50
GPD_MODEL_DIR  = './models/'
GPD_GUIDES     = [1, 2, 3, 4, 5, 10, 11, 13, 14, 16, 18, 21]
GPD_BATCH_PER  = 10
USE_STITCH     = True   # if False, select best single candidate (paper GPD-NG, no stitching)
USE_RRT        = True   # when stitching: RRT-Connect bridges (True) or linear bridges (False)
USE_PB_COLLISION = False  # stitch/RRT collision check: PyBullet (True) or AABB proxy (False)
GPD_EXTRA_STEPS  = 0      # GPDS: collect candidates from the last N denoising steps (paper uses 5)
GPD_GUIDANCE_SCALE = 1.0  # multiplier on the guidance schedule (test stronger/weaker guidance)
GPD_VAR_THRESH = 0.02   # inference noise schedule terminal beta (must match training; D1: 0.08)

GUIDE_PATH     = './guides/'
T_ORIG         = 255


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_guide_cfgs(guides, T, batch_per):
    n   = len(guides)
    tot = n * batch_per
    cfgs = {
        'batch_size_per_guide': batch_per,
        'total_batch_size':     tot,
        'clearance':            np.zeros((tot, T)),
        'expansion':            np.zeros((tot, T)),
        'guidance_method':      np.zeros((tot,)),
        'grad_norm':            np.zeros((tot,)),
        'guidance_schedule':    np.zeros((tot, T)),
        'volume_trust_region':  np.zeros((tot,)),
    }
    for i, g_id in enumerate(guides):
        g  = YamlConfig(GUIDE_PATH + f'cfgs/guide{g_id}.yaml')
        sl = slice(i * batch_per, (i + 1) * batch_per)
        cfgs['clearance'][sl, :] = np.linspace(
            g['hyperparameters']['obstacle_clearance']['range'][0],
            g['hyperparameters']['obstacle_clearance']['range'][1], T)
        o_e = g['hyperparameters']['obstacle_expansion']
        for ik, vk in [('isr1','val1'),('isr2','val2'),('isr3','val3')]:
            r  = o_e[ik]
            r0 = int(np.clip(round(r[0] * T / T_ORIG), 0, T))
            r1 = int(np.clip(round(r[1] * T / T_ORIG), 0, T))
            if r1 > r0:
                cfgs['expansion'][sl, r0:r1] = np.linspace(
                    o_e[vk][0], o_e[vk][1], num=r1 - r0)
        cfgs['guidance_method'][sl]    = 1 if g['hyperparameters']['guidance_method'] == 'sv' else 0
        cfgs['grad_norm'][sl]          = 1 if g['hyperparameters']['grad_norm'] else 0
        cfgs['guidance_schedule'][sl, :] = (
            (1.4 + np.arange(T) / T)
            if g['hyperparameters']['guidance_schedule']['type'] == 'varying'
            else g['hyperparameters']['guidance_schedule']['scale_val'])
        cfgs['volume_trust_region'][sl] = g['hyperparameters']['volume_trust_region']
    return cfgs


def filter_ik(guide, all_ik_goals, start_joints):
    vols = guide.cost(
        torch.tensor(all_ik_goals.reshape((-1, 7, 1)), device=DEVICE),
        0, batch_size=all_ik_goals.shape[0]
    ).sum(axis=(1, 2)).cpu().numpy()
    mn  = np.min(vols)
    idx = np.argsort(vols)
    gj  = all_ik_goals[idx][vols[idx] < mn + 0.0008]
    ii  = np.argmin(np.linalg.norm(start_joints - gj, axis=1))
    return gj[ii]


def get_worker_indices(cap, worker_id, num_workers):
    """Return the sorted list of scene indices this worker is responsible for."""
    return list(range(worker_id, cap, num_workers))


# ── EDMP worker ────────────────────────────────────────────────────────────────

def run_edmp_worker(dataset, env, caps, worker_id, num_workers, out_path):
    diffuser   = Diffusion(T=EDMP_T, device=DEVICE)
    model_name = EDMP_MODEL_DIR + f'TemporalUNetModel{EDMP_T}_N{EDMP_TRAJ_LEN}'
    denoiser   = TemporalUNet(model_name=model_name, input_dim=7, time_dim=32,
                              dims=(32, 64, 128, 256, 512, 512), device=DEVICE)
    guide_cfgs = build_guide_cfgs(EDMP_GUIDES, EDMP_T, EDMP_BATCH_PER)
    total_bs   = guide_cfgs['total_batch_size']

    scene_results, t_success = [], 0
    for scene_type, cap in zip(SCENE_TYPES, caps):
        indices = get_worker_indices(min(cap, dataset.data_nums[scene_type]),
                                     worker_id, num_workers)
        for i in indices:
            env.clear_obstacles(); env.go_home()
            (obs_cfg, cub_cfg, cyl_cfg, n_cub, n_cyl,
             start_j, all_ik) = dataset.fetch_data(i, scene_type)

            guide  = IntersectionVolumeGuide(obs_cfg, DEVICE, guide_cfgs, total_bs)
            goal_j = filter_ik(guide, all_ik, start_j)

            t0 = time.time()
            trajs = denoise_guided_gpu(
                diffuser=diffuser, model=denoiser, guide=guide,
                traj_len=EDMP_TRAJ_LEN, num_channels=7,
                guidance_schedule=guide_cfgs['guidance_schedule'],
                batch_size=total_bs, start=start_j, goal=goal_j,
                condition=True, benchmarking=False)
            plan_t = time.time() - t0

            traj = guide.choose_best_trajectory(start_j, goal_j, trajs)
            if n_cub > 0: env.spawn_collision_cuboids(cub_cfg)
            if n_cyl > 0: env.spawn_collision_cylinders(cyl_cfg)
            success  = int(env.benchmark_trajectory(traj))
            t_success += success

            r = {'scene_type': scene_type, 'scene_num': i,
                 'success': success, 'plan_time': round(plan_t, 3)}
            scene_results.append(r)
            print(f"  [W{worker_id}][{scene_type}:{i+1:4d}]  success={success}  "
                  f"SR={t_success}/{len(scene_results)}  t={plan_t:.2f}s", flush=True)

        with open(out_path, 'w') as f:
            json.dump({'method': 'edmp', 'worker_id': worker_id,
                       'num_workers': num_workers, 'results': scene_results}, f, indent=2)

    return scene_results


# ── GPD worker ─────────────────────────────────────────────────────────────────

def run_gpd_worker(dataset, env, caps, worker_id, num_workers, out_path):
    lower = np.array([-166,-101,-166,-176,-166, -1,-166], np.float32) * (np.pi/180)
    upper = np.array([ 166, 101, 166,  -4, 166,215, 166], np.float32) * (np.pi/180)

    diffuser   = PolynomialDiffusion(T=GPD_T, device=DEVICE,
                                     n_control=GPD_N_CONTROL, traj_len=GPD_TRAJ_LEN,
                                     variance_thresh=GPD_VAR_THRESH)
    model_name = GPD_MODEL_DIR + f'GPDModel{GPD_T}_N{GPD_N_CONTROL}'
    denoiser   = TemporalUNetGPD(model_name=model_name, input_dim=7, time_dim=32,
                                  dims=(32, 64, 128, 256), device=DEVICE)
    guide_cfgs = build_guide_cfgs(GPD_GUIDES, GPD_T, GPD_BATCH_PER)
    guide_cfgs['guidance_schedule'] = guide_cfgs['guidance_schedule'] * GPD_GUIDANCE_SCALE
    total_bs   = guide_cfgs['total_batch_size']

    scene_results, t_success = [], 0
    for scene_type, cap in zip(SCENE_TYPES, caps):
        indices = get_worker_indices(min(cap, dataset.data_nums[scene_type]),
                                     worker_id, num_workers)
        for i in indices:
            env.clear_obstacles(); env.go_home()
            (obs_cfg, cub_cfg, cyl_cfg, n_cub, n_cyl,
             start_j, all_ik) = dataset.fetch_data(i, scene_type)

            guide  = IntersectionVolumeGuide(obs_cfg, DEVICE, guide_cfgs, total_bs)
            goal_j = filter_ik(guide, all_ik, start_j)

            t0    = time.time()
            trajs = diffuser.denoise_guided_poly(
                model=denoiser, guide=guide, num_channels=7,
                guidance_schedule=guide_cfgs['guidance_schedule'],
                batch_size=total_bs, start=start_j, goal=goal_j,
                condition=True, benchmarking=False,
                extra_candidate_steps=GPD_EXTRA_STEPS)

            # Spawn obstacles before stitch so a PyBullet collision checker (if
            # used) sees them; harmless for the AABB/no-stitch paths.
            if n_cub > 0: env.spawn_collision_cuboids(cub_cfg)
            if n_cyl > 0: env.spawn_collision_cylinders(cyl_cfg)

            if USE_STITCH:
                cfn = env.configs_free if USE_PB_COLLISION else None
                traj = stitch(trajs, guide, DEVICE, use_rrt=USE_RRT, collision_fn=cfn)
            else:
                traj = guide.choose_best_trajectory(start_j, goal_j, trajs)
            traj = np.clip(traj, lower[:, None], upper[:, None])
            plan_t = time.time() - t0

            success  = int(env.benchmark_trajectory(traj))
            t_success += success

            r = {'scene_type': scene_type, 'scene_num': i,
                 'success': success, 'plan_time': round(plan_t, 3)}
            scene_results.append(r)
            print(f"  [W{worker_id}][{scene_type}:{i+1:4d}]  success={success}  "
                  f"SR={t_success}/{len(scene_results)}  t={plan_t:.2f}s", flush=True)

        with open(out_path, 'w') as f:
            json.dump({'method': 'gpd', 'worker_id': worker_id,
                       'num_workers': num_workers, 'results': scene_results}, f, indent=2)

    return scene_results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method',      choices=['edmp', 'gpd'], required=True)
    parser.add_argument('--worker_id',   type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--full',        action='store_true',
                        help='Run all 1800 scenes (default: 50-scene mini)')
    parser.add_argument('--per_type',    type=int, default=None,
                        help='Balanced N scenes per type (capped at availability)')
    parser.add_argument('--caps',        default=None,
                        help='Explicit per-type caps "tabletop,cubby,merged_cubby,dresser" '
                             '(natural distribution = 600,300,300,600)')
    parser.add_argument('--guides',      default=None,
                        help='Comma-separated guide ids for GPD (e.g. "1" for GPD-1G)')
    parser.add_argument('--batch_per',   type=int, default=None,
                        help='Candidates per guide (K = n_guides * batch_per)')
    parser.add_argument('--no_stitch',   action='store_true',
                        help='Disable stitching; pick best single candidate (paper GPD-NG)')
    parser.add_argument('--linear_stitch', action='store_true',
                        help='Use linear bridges instead of RRT-Connect when stitching')
    parser.add_argument('--pybullet_collision', action='store_true',
                        help='Use faithful PyBullet collision checking in stitch/RRT')
    parser.add_argument('--extra_cand_steps', type=int, default=None,
                        help='GPDS: collect candidates from last N denoising steps (paper=5)')
    parser.add_argument('--dataset', default=None, choices=['global', 'hybrid', 'both'],
                        help='Solvability test set (paper reports each separately)')
    parser.add_argument('--guidance_scale', type=float, default=None,
                        help='Multiplier on the guidance schedule (test stronger guidance)')
    parser.add_argument('--model_dir',   default=None,
                        help='Override model dir (e.g. ./models_d1/ for an ablation model)')
    parser.add_argument('--variance_thresh', type=float, default=None,
                        help='Inference noise schedule terminal beta (match training)')
    parser.add_argument('--out',         default=None)
    args = parser.parse_args()

    if args.out is None:
        args.out = f'results/{args.method}_w{args.worker_id}.json'

    # ---- Apply GPD config overrides (module globals read by run_gpd_worker) ----
    global GPD_GUIDES, GPD_BATCH_PER, USE_STITCH, USE_RRT, USE_PB_COLLISION, GPD_MODEL_DIR, GPD_VAR_THRESH, GPD_EXTRA_STEPS, GPD_GUIDANCE_SCALE
    if args.guides is not None:
        GPD_GUIDES = [int(g) for g in args.guides.split(',')]
    if args.batch_per is not None:
        GPD_BATCH_PER = args.batch_per
    if args.no_stitch:
        USE_STITCH = False
    if args.linear_stitch:
        USE_RRT = False
    if args.pybullet_collision:
        USE_PB_COLLISION = True
    if args.extra_cand_steps is not None:
        GPD_EXTRA_STEPS = args.extra_cand_steps
    if args.guidance_scale is not None:
        GPD_GUIDANCE_SCALE = args.guidance_scale
    if args.model_dir is not None:
        GPD_MODEL_DIR = args.model_dir
    if args.variance_thresh is not None:
        GPD_VAR_THRESH = args.variance_thresh

    caps    = FULL_CAPS if args.full else MINI_CAPS
    if args.per_type is not None:
        caps = [min(args.per_type, c) for c in FULL_CAPS]
    if args.caps is not None:
        caps = [int(c) for c in args.caps.split(',')]
    dataset_type = args.dataset if args.dataset is not None else DATASET_TYPE
    dataset = TestDataset(dataset_type, d_path=DATASET_PATH)
    env     = RobotEnvironment(gui=GUI)

    print(f"Worker {args.worker_id}/{args.num_workers} | method={args.method} | "
          f"full={args.full} | out={args.out}")
    if args.method == 'gpd':
        print(f"  GPD config | guides={GPD_GUIDES} | batch_per={GPD_BATCH_PER} | "
              f"K={len(GPD_GUIDES)*GPD_BATCH_PER} | stitch={USE_STITCH} | "
              f"rrt={USE_RRT} | model_dir={GPD_MODEL_DIR} | caps={caps}")

    if args.method == 'edmp':
        run_edmp_worker(dataset, env, caps, args.worker_id, args.num_workers, args.out)
    else:
        run_gpd_worker(dataset, env, caps, args.worker_id, args.num_workers, args.out)


if __name__ == '__main__':
    main()
