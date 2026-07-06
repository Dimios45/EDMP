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

import argparse, json, time, os
import numpy as np
import torch

from autolab_core import YamlConfig
from lib import *
from diffusion import *
from datasets.load_test_dataset import TestDataset
from diffusion.models.temporalunet import TemporalUNet as TemporalUNetGPD
from gpd.diffusion import PolynomialDiffusion
from gpd.conditional_unet import ConditionalTemporalUNet
from gpd.stitch    import stitch, rrt_connect
from gpd.refine    import trajopt_refine
from gpd.sphere_collision import SphereSDFCost
from gpd.select import oracle_select, resample_to
from gpd.learned_repair import LearnedRepair, repair_batch
from gpd.refine    import densify
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
GPD_EXTRA_STEPS  = 0      # GPD-stitched: collect candidates from the last N denoising steps (paper uses 5)
GPD_GUIDANCE_SCALE = 1.0  # multiplier on the guidance schedule (test stronger/weaker guidance)
GPD_SMOOTHNESS = 0.0      # Smoothness penalty: control-point curvature penalty per denoising step
GPD_CONDITIONAL = False   # Scene conditioning: scene-conditioned denoiser (CFG)
GPD_CFG_WEIGHT  = 2.0     # Classifier-free guidance weight
USE_TRAJOPT     = False   # Post-hoc trajectory-optimization feasibility repair
TRAJOPT_ITERS   = 60
TRAJOPT_DENSIFY = 3       # Output densification factor (overshoot-free execution)
TRAJOPT_MID_W   = 1.0     # Edge-midpoint collision-cost weight (swept-feasibility)
GPD_SEED        = None    # multi-seed CIs: RNG seed for diffusion sampling (None=unseeded)
NAIVE_SEED      = None     # control: replace diffusion prior w/ 'linear' seed, same downstream
SPHERE_COST     = False    # Sphere-SDF (oriented-box) repair objective vs AABB proxy
SPHERE_GUIDANCE = False    # sphere-SDF gradient as the diffusion guidance signal (reproduction study)
SMC_EVERY       = 0       # SMC particle steering: resample every N steps in 2nd half of chain (0=off)
SMC_TEMP        = 1.0     # SMC softmax temperature on sphere-SDF penetration potential
RRT_FALLBACK    = False   # completeness fallback: RRT-Connect solve when the learned plan fails
RRT_FB_ITERS    = 2000    # RRT-Connect iteration budget for the fallback solve
RRT_FB_DENSIFY  = 4       # densification factor for the RRT fallback path
REPAIR_EDGE_SAMPLES = 1  # continuous-repair: sub-configs sampled per edge (1=midpoint)
ORACLE_SELECT   = False   # verifier: pick best-of-K candidate by faithful oracle (not guide cost)
SAVE_REPAIR_PAIRS = False # emit (seed -> repaired) trajectory pairs for learned-repair training
LEARNED_REPAIR  = False   # C: one-shot learned repair operator (optionally + trajopt polish)
_LR_CACHE = {}
GPD_VAR_THRESH = 0.02   # inference noise schedule terminal beta (must match training; schedule-ablation value 0.08)


def _lr_model(device):
    if 'm' not in _LR_CACHE:
        _LR_CACHE['m'] = LearnedRepair().load(device).to(device).eval()
    return _LR_CACHE['m']

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
    if GPD_SEED is not None:
        s = GPD_SEED + worker_id
        torch.manual_seed(s); np.random.seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
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
            if USE_TRAJOPT:
                # Cross-planner: same PRESTO-style repair as GPD, on EDMP's output (shows
                # the feasibility lever generalizes beyond the GPD prior).
                el = np.array([-166,-101,-166,-176,-166,-1,-166], np.float32)*(np.pi/180)
                eu = np.array([166,101,166,-4,166,215,166], np.float32)*(np.pi/180)
                cfn = env.configs_free if USE_PB_COLLISION else None
                traj = trajopt_refine(traj, guide, DEVICE, iters=TRAJOPT_ITERS,
                                      mid_w=TRAJOPT_MID_W, out_densify=TRAJOPT_DENSIFY,
                                      edge_samples=REPAIR_EDGE_SAMPLES, collision_fn=cfn)
                traj = np.clip(traj, el[:, None], eu[:, None])
                plan_t = time.time() - t0
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

def build_scene_tensor(cub_cfg, cyl_cfg, n_cub, n_cyl):
    """Build the (52,12) obstacle feature tensor + (52,) mask matching
    gpd/preprocess_scenes.py. Inputs use fetch_data's post-roll xyzw quats;
    roll back (+1) to the train.hdf5 (w-first) convention used at training.
    feat = [is_cuboid, is_cylinder, cx,cy,cz, ex,ey,ez, qx,qy,qz,qw]."""
    S, F = 52, 12
    obs  = np.zeros((S, F), np.float32)
    mask = np.zeros((S,),  bool)
    # cuboids -> slots [0:40]
    for j in range(min(n_cub, 40)):
        c = cub_cfg[j]
        q = np.roll(c[3:7], 1)                       # xyzw -> training convention
        obs[j] = [1, 0, c[0], c[1], c[2],
                  c[7]/2, c[8]/2, c[9]/2, q[0], q[1], q[2], q[3]]
        mask[j] = True
    # cylinders -> slots [40:52]
    for j in range(min(n_cyl, 12)):
        c = cyl_cfg[j]
        q = np.roll(c[3:7], 1)
        obs[40 + j] = [0, 1, c[0], c[1], c[2],
                       c[7], c[8], 0.0, q[0], q[1], q[2], q[3]]
        mask[40 + j] = True
    return obs, mask


def run_gpd_worker(dataset, env, caps, worker_id, num_workers, out_path):
    lower = np.array([-166,-101,-166,-176,-166, -1,-166], np.float32) * (np.pi/180)
    upper = np.array([ 166, 101, 166,  -4, 166,215, 166], np.float32) * (np.pi/180)

    if GPD_SEED is not None:
        s = GPD_SEED + worker_id          # decorrelate workers; reproducible per seed
        torch.manual_seed(s); np.random.seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    diffuser   = PolynomialDiffusion(T=GPD_T, device=DEVICE,
                                     n_control=GPD_N_CONTROL, traj_len=GPD_TRAJ_LEN,
                                     variance_thresh=GPD_VAR_THRESH)
    if GPD_CONDITIONAL:
        model_name = GPD_MODEL_DIR + f'GPDCondModel{GPD_T}_N{GPD_N_CONTROL}'
        denoiser   = ConditionalTemporalUNet(model_name=model_name, input_dim=7,
                                             time_dim=32, dims=(32, 64, 128, 256),
                                             device=DEVICE)
    else:
        model_name = GPD_MODEL_DIR + f'GPDModel{GPD_T}_N{GPD_N_CONTROL}'
        denoiser   = TemporalUNetGPD(model_name=model_name, input_dim=7, time_dim=32,
                                      dims=(32, 64, 128, 256), device=DEVICE)
    guide_cfgs = build_guide_cfgs(GPD_GUIDES, GPD_T, GPD_BATCH_PER)
    guide_cfgs['guidance_schedule'] = guide_cfgs['guidance_schedule'] * GPD_GUIDANCE_SCALE
    total_bs   = guide_cfgs['total_batch_size']

    scene_results, t_success = [], 0
    _pair_buf = []
    for scene_type, cap in zip(SCENE_TYPES, caps):
        indices = get_worker_indices(min(cap, dataset.data_nums[scene_type]),
                                     worker_id, num_workers)
        for i in indices:
            env.clear_obstacles(); env.go_home()
            (obs_cfg, cub_cfg, cyl_cfg, n_cub, n_cyl,
             start_j, all_ik) = dataset.fetch_data(i, scene_type)

            guide  = IntersectionVolumeGuide(obs_cfg, DEVICE, guide_cfgs, total_bs)
            goal_j = filter_ik(guide, all_ik, start_j)

            sc_obs = sc_mask = None
            if GPD_CONDITIONAL:
                sc_obs, sc_mask = build_scene_tensor(cub_cfg, cyl_cfg, n_cub, n_cyl)

            t0    = time.time()
            if NAIVE_SEED == 'linear':
                # Control: discard the diffusion prior, seed with a straight line
                # in joint space, then run the IDENTICAL stitch+trajopt downstream.
                seed = np.linspace(np.asarray(start_j, np.float32),
                                   np.asarray(goal_j,  np.float32),
                                   diffuser.traj_len).T            # (7, traj_len)
                trajs = seed[None]                                  # (1, 7, traj_len)
            else:
                sph_guide = SphereSDFCost(guide, DEVICE) if SPHERE_GUIDANCE else None
                smc_c = SphereSDFCost(guide, DEVICE) if SMC_EVERY > 0 else None
                trajs = diffuser.denoise_guided_poly(
                    model=denoiser, guide=guide, num_channels=7,
                    guidance_schedule=guide_cfgs['guidance_schedule'],
                    batch_size=total_bs, start=start_j, goal=goal_j,
                    condition=True, benchmarking=False,
                    extra_candidate_steps=GPD_EXTRA_STEPS,
                    smoothness_weight=GPD_SMOOTHNESS,
                    scene_obs=sc_obs, scene_mask=sc_mask, cfg_weight=GPD_CFG_WEIGHT,
                    sphere_cost=sph_guide,
                    smc_cost=smc_c, smc_every=SMC_EVERY, smc_temp=SMC_TEMP)

            # Spawn obstacles before stitch so a PyBullet collision checker (if
            # used) sees them; harmless for the AABB/no-stitch paths.
            if n_cub > 0: env.spawn_collision_cuboids(cub_cfg)
            if n_cyl > 0: env.spawn_collision_cylinders(cyl_cfg)

            t_a = time.time()
            cfn = env.configs_free if (USE_PB_COLLISION or ORACLE_SELECT) else None
            if ORACLE_SELECT and cfn is not None:
                traj = oracle_select(trajs, cfn)[0]          # best-of-K by faithful oracle
            elif USE_STITCH:
                traj = stitch(trajs, guide, DEVICE, use_rrt=USE_RRT, collision_fn=cfn)
            else:
                traj = guide.choose_best_trajectory(start_j, goal_j, trajs)
            traj = np.clip(traj, lower[:, None], upper[:, None])
            t_b = time.time(); seed_traj = traj.copy()
            if LEARNED_REPAIR:
                m = _lr_model(DEVICE)
                sc_o, sc_m = build_scene_tensor(cub_cfg, cyl_cfg, n_cub, n_cyl)
                traj = np.clip(repair_batch(m, resample_to(seed_traj, 50), sc_o, sc_m, DEVICE),
                               lower[:, None], upper[:, None])
                if USE_TRAJOPT:                      # warm-start hybrid: learned + trajopt polish
                    cobj = SphereSDFCost(guide, DEVICE) if SPHERE_COST else None
                    traj = np.clip(trajopt_refine(traj, guide, DEVICE, iters=TRAJOPT_ITERS,
                                      mid_w=TRAJOPT_MID_W, out_densify=1,
                                      edge_samples=REPAIR_EDGE_SAMPLES,
                                      cost_obj=cobj, collision_fn=cfn),
                                   lower[:, None], upper[:, None])
                traj = densify(traj, TRAJOPT_DENSIFY)
            elif USE_TRAJOPT:
                cobj = SphereSDFCost(guide, DEVICE) if SPHERE_COST else None
                traj = trajopt_refine(traj, guide, DEVICE, iters=TRAJOPT_ITERS,
                                      mid_w=TRAJOPT_MID_W, out_densify=TRAJOPT_DENSIFY,
                                      edge_samples=REPAIR_EDGE_SAMPLES,
                                      cost_obj=cobj, collision_fn=cfn)
                traj = np.clip(traj, lower[:, None], upper[:, None])
            t_c = time.time(); plan_t = t_c - t0

            success  = int(env.benchmark_trajectory(traj))

            # Completeness fallback: if the learned plan would still fail, solve the
            # scene from scratch with RRT-Connect against the faithful oracle. On
            # solvable scenes a complete planner catches most learned-planner misses.
            used_fallback = 0
            if RRT_FALLBACK and not success:
                path = rrt_connect(start_j, goal_j, guide, DEVICE,
                                   max_iter=RRT_FB_ITERS, seed=int(i),
                                   collision_fn=env.configs_free)
                if path is not None and len(path) >= 2:
                    rtraj = np.clip(densify(np.stack(path).T, RRT_FB_DENSIFY),
                                    lower[:, None], upper[:, None])
                    if int(env.benchmark_trajectory(rtraj)):
                        traj, success, used_fallback = rtraj, 1, 1
            t_c2 = time.time()
            plan_t = t_c2 - t0
            t_success += success

            if SAVE_REPAIR_PAIRS and USE_TRAJOPT:
                sc_o, sc_m = build_scene_tensor(cub_cfg, cyl_cfg, n_cub, n_cyl)
                _pair_buf.append({'scene_type': scene_type, 'scene_num': int(i),
                    'seed': resample_to(seed_traj, 50).tolist(),
                    'repaired': resample_to(traj, 50).tolist(),
                    'scene_obs': sc_o.tolist(), 'scene_mask': [bool(b) for b in sc_m],
                    'success': success})

            r = {'scene_type': scene_type, 'scene_num': i,
                 'success': success, 'plan_time': round(plan_t, 3),
                 't_diffuse': round(t_a - t0, 3), 't_select': round(t_b - t_a, 3),
                 't_repair': round(t_c - t_b, 3), 'fallback': used_fallback}
            scene_results.append(r)
            print(f"  [W{worker_id}][{scene_type}:{i+1:4d}]  success={success}  "
                  f"SR={t_success}/{len(scene_results)}  t={plan_t:.2f}s", flush=True)

        with open(out_path, 'w') as f:
            json.dump({'method': 'gpd', 'worker_id': worker_id,
                       'num_workers': num_workers, 'results': scene_results}, f, indent=2)

    if SAVE_REPAIR_PAIRS and _pair_buf:
        os.makedirs('results/repair_pairs', exist_ok=True)
        with open(f'results/repair_pairs/w{worker_id}.jsonl', 'w') as f:
            for p in _pair_buf:
                f.write(json.dumps(p) + '\n')

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
                        help='Comma-separated guide ids for GPD (e.g. "1" for a single-guide config)')
    parser.add_argument('--batch_per',   type=int, default=None,
                        help='Candidates per guide (K = n_guides * batch_per)')
    parser.add_argument('--no_stitch',   action='store_true',
                        help='Disable stitching; pick best single candidate (paper GPD-NG)')
    parser.add_argument('--linear_stitch', action='store_true',
                        help='Use linear bridges instead of RRT-Connect when stitching')
    parser.add_argument('--pybullet_collision', action='store_true',
                        help='Use faithful PyBullet collision checking in stitch/RRT')
    parser.add_argument('--extra_cand_steps', type=int, default=None,
                        help='GPD-stitched: collect candidates from last N denoising steps (paper=5)')
    parser.add_argument('--dataset', default=None, choices=['global', 'hybrid', 'both'],
                        help='Solvability test set (paper reports each separately)')
    parser.add_argument('--guidance_scale', type=float, default=None,
                        help='Multiplier on the guidance schedule (test stronger guidance)')
    parser.add_argument('--model_dir',   default=None,
                        help='Override model dir (e.g. ./models_alt/ for an ablation model)')
    parser.add_argument('--n_control',   type=int, default=None,
                        help='Number of Bernstein control points (capacity sweep; must match the model)')
    parser.add_argument('--smoothness_weight', type=float, default=None,
                        help='smoothness penalty: control-point curvature penalty per denoising step')
    parser.add_argument('--conditional', action='store_true',
                        help='scene-conditioned denoiser (GPDCondModel) with CFG')
    parser.add_argument('--cfg_weight', type=float, default=None,
                        help='classifier-free guidance weight')
    parser.add_argument('--trajopt', action='store_true',
                        help='post-hoc trajectory-optimization feasibility repair')
    parser.add_argument('--trajopt_iters', type=int, default=None,
                        help='max trajopt iterations per scene')
    parser.add_argument('--trajopt_densify', type=int, default=None,
                        help='output densification factor')
    parser.add_argument('--trajopt_mid_w', type=float, default=None,
                        help='edge-midpoint collision-cost weight')
    parser.add_argument('--seed', type=int, default=None,
                        help='RNG seed for diffusion sampling (multi-seed CIs)')
    parser.add_argument('--naive_seed', default=None, choices=['linear'],
                        help='control: replace diffusion prior with a linear seed')
    parser.add_argument('--sphere_cost', action='store_true',
                        help='sphere-SDF oriented-box repair objective (vs AABB proxy)')
    parser.add_argument('--sphere_guidance', action='store_true',
                        help='sphere-SDF gradient as the diffusion guidance signal')
    parser.add_argument('--repair_edge_samples', type=int, default=None,
                        help='continuous-repair: sub-configs per edge in the trajopt objective (1=midpoint)')
    parser.add_argument('--oracle_select', action='store_true',
                        help='select best-of-K candidate by the faithful PyBullet oracle')
    parser.add_argument('--rrt_fallback', action='store_true',
                        help='completeness fallback: RRT-Connect solve when the learned plan fails')
    parser.add_argument('--rrt_fb_iters', type=int, default=None,
                        help='RRT-Connect iteration budget for the fallback solve')
    parser.add_argument('--learned_repair', action='store_true',
                        help='C: one-shot learned repair operator (models/LearnedRepair)')
    parser.add_argument('--smc_every', type=int, default=None,
                        help='SMC particle steering: resample every N denoising steps (0=off)')
    parser.add_argument('--smc_temp', type=float, default=None,
                        help='SMC softmax temperature on the sphere-SDF potential')
    parser.add_argument('--save_repair_pairs', action='store_true',
                        help='log (seed -> trajopt-repaired) pairs to results/repair_pairs/')
    parser.add_argument('--variance_thresh', type=float, default=None,
                        help='Inference noise schedule terminal beta (match training)')
    parser.add_argument('--out',         default=None)
    args = parser.parse_args()

    if args.out is None:
        args.out = f'results/{args.method}_w{args.worker_id}.json'

    # ---- Apply GPD config overrides (module globals read by run_gpd_worker) ----
    global GPD_GUIDES, GPD_BATCH_PER, USE_STITCH, USE_RRT, USE_PB_COLLISION, GPD_MODEL_DIR, GPD_VAR_THRESH, GPD_EXTRA_STEPS, GPD_GUIDANCE_SCALE, GPD_N_CONTROL, GPD_SMOOTHNESS, GPD_CONDITIONAL, GPD_CFG_WEIGHT, USE_TRAJOPT, TRAJOPT_ITERS, TRAJOPT_DENSIFY, TRAJOPT_MID_W, GPD_SEED, NAIVE_SEED, SPHERE_COST, SPHERE_GUIDANCE, REPAIR_EDGE_SAMPLES, ORACLE_SELECT, SAVE_REPAIR_PAIRS, LEARNED_REPAIR, SMC_EVERY, SMC_TEMP, RRT_FALLBACK, RRT_FB_ITERS
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
    if args.n_control is not None:
        GPD_N_CONTROL = args.n_control
    if args.smoothness_weight is not None:
        GPD_SMOOTHNESS = args.smoothness_weight
    if args.conditional:
        GPD_CONDITIONAL = True
    if args.cfg_weight is not None:
        GPD_CFG_WEIGHT = args.cfg_weight
    if args.trajopt:
        USE_TRAJOPT = True
    if args.trajopt_iters is not None:
        TRAJOPT_ITERS = args.trajopt_iters
    if args.trajopt_densify is not None:
        TRAJOPT_DENSIFY = args.trajopt_densify
    if args.trajopt_mid_w is not None:
        TRAJOPT_MID_W = args.trajopt_mid_w
    if args.seed is not None:
        GPD_SEED = args.seed
    if args.naive_seed is not None:
        NAIVE_SEED = args.naive_seed
    if args.sphere_cost:
        SPHERE_COST = True
    if args.sphere_guidance:
        SPHERE_GUIDANCE = True
    if args.repair_edge_samples is not None:
        REPAIR_EDGE_SAMPLES = args.repair_edge_samples
    if args.smc_every is not None:
        SMC_EVERY = args.smc_every
    if args.smc_temp is not None:
        SMC_TEMP = args.smc_temp
    if args.rrt_fallback:
        RRT_FALLBACK = True
    if args.rrt_fb_iters is not None:
        RRT_FB_ITERS = args.rrt_fb_iters
    if args.learned_repair:
        LEARNED_REPAIR = True
    if args.oracle_select:
        ORACLE_SELECT = True
    if args.save_repair_pairs:
        SAVE_REPAIR_PAIRS = True

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
              f"rrt={USE_RRT} | n_control={GPD_N_CONTROL} | model_dir={GPD_MODEL_DIR} | caps={caps}")

    if args.method == 'edmp':
        run_edmp_worker(dataset, env, caps, args.worker_id, args.num_workers, args.out)
    else:
        run_gpd_worker(dataset, env, caps, args.worker_id, args.num_workers, args.out)


if __name__ == '__main__':
    main()
