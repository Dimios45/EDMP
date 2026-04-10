"""
Head-to-head comparison: EDMP vs GPD.

Usage:
    python compare.py --method both --full      # all 1800 scenes (~12h total)
    python compare.py --method gpd  --full      # GPD only (~45 min)
    python compare.py --method edmp --full      # EDMP only (~11h)
    python compare.py --method both             # 50-scene mini benchmark
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

# ── Static config ──────────────────────────────────────────────────────────────
SCENE_TYPES  = ['tabletop', 'cubby', 'merged_cubby', 'dresser']
FULL_CAPS    = [600, 300, 300, 600]   # all 1800 scenes
MINI_CAPS    = [ 13,  12,  13,  12]  # 50 scenes
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

GUIDE_PATH     = './guides/'
T_ORIG         = 255   # original T for which guide yamls were written


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


def print_partial(scene_results, label):
    sr  = sum(r['success'] for r in scene_results)
    tot = len(scene_results)
    avg = np.mean([r['plan_time'] for r in scene_results])
    print(f"\n{label}: {sr}/{tot} = {100*sr/tot:.1f}%  avg {avg:.2f}s/scene\n")


# ── EDMP runner ────────────────────────────────────────────────────────────────

def run_edmp(dataset, env, caps, out_path):
    print("\n" + "="*60)
    total_scenes = sum(min(c, dataset.data_nums[st])
                       for st, c in zip(SCENE_TYPES, caps))
    print(f"RUNNING EDMP  ({total_scenes} scenes)")
    print("="*60)

    diffuser   = Diffusion(T=EDMP_T, device=DEVICE)
    model_name = EDMP_MODEL_DIR + f'TemporalUNetModel{EDMP_T}_N{EDMP_TRAJ_LEN}'
    denoiser   = TemporalUNet(model_name=model_name, input_dim=7, time_dim=32,
                              dims=(32, 64, 128, 256, 512, 512), device=DEVICE)
    guide_cfgs = build_guide_cfgs(EDMP_GUIDES, EDMP_T, EDMP_BATCH_PER)
    total_bs   = guide_cfgs['total_batch_size']

    scene_results, t_success = [], 0
    for scene_type, cap in zip(SCENE_TYPES, caps):
        for i in range(min(cap, dataset.data_nums[scene_type])):
            env.clear_obstacles(); env.go_home()
            (obs_cfg, cub_cfg, cyl_cfg, n_cub, n_cyl,
             start_j, all_ik) = dataset.fetch_data(i, scene_type)

            guide  = IntersectionVolumeGuide(obs_cfg, DEVICE, guide_cfgs, total_bs)
            goal_j = filter_ik(guide, all_ik, start_j)

            t0   = time.time()
            trajs = diffuser.denoise_guided(
                model=denoiser, guide=guide, batch_size=total_bs,
                traj_len=EDMP_TRAJ_LEN, num_channels=7, condition=True,
                benchmarking=True, start=start_j, goal=goal_j,
                guidance_schedule=guide_cfgs['guidance_schedule'])
            plan_t = time.time() - t0

            traj = guide.choose_best_trajectory(start_j, goal_j, trajs)
            if n_cub > 0: env.spawn_collision_cuboids(cub_cfg)
            if n_cyl > 0: env.spawn_collision_cylinders(cyl_cfg)
            success  = int(env.benchmark_trajectory(traj))
            t_success += success

            r = {'scene_type': scene_type, 'scene_num': i,
                 'success': success, 'plan_time': round(plan_t, 3)}
            scene_results.append(r)
            print(f"  [{scene_type}:{i+1:4d}]  success={success}  "
                  f"SR={t_success}/{len(scene_results)}  t={plan_t:.2f}s")

        # checkpoint after each scene type
        with open(out_path, 'w') as f:
            json.dump({'edmp': scene_results}, f, indent=2)

    print_partial(scene_results, "EDMP final")
    return scene_results


# ── GPD runner ─────────────────────────────────────────────────────────────────

def run_gpd(dataset, env, caps, out_path):
    print("\n" + "="*60)
    total_scenes = sum(min(c, dataset.data_nums[st])
                       for st, c in zip(SCENE_TYPES, caps))
    print(f"RUNNING GPD  ({total_scenes} scenes)")
    print("="*60)

    lower = np.array([-166,-101,-166,-176,-166, -1,-166], np.float32) * (np.pi/180)
    upper = np.array([ 166, 101, 166,  -4, 166,215, 166], np.float32) * (np.pi/180)

    diffuser   = PolynomialDiffusion(T=GPD_T, device=DEVICE,
                                     n_control=GPD_N_CONTROL, traj_len=GPD_TRAJ_LEN)
    model_name = GPD_MODEL_DIR + f'GPDModel{GPD_T}_N{GPD_N_CONTROL}'
    denoiser   = TemporalUNetGPD(model_name=model_name, input_dim=7, time_dim=32,
                                  dims=(32, 64, 128, 256), device=DEVICE)
    guide_cfgs = build_guide_cfgs(GPD_GUIDES, GPD_T, GPD_BATCH_PER)
    total_bs   = guide_cfgs['total_batch_size']

    scene_results, t_success = [], 0
    for scene_type, cap in zip(SCENE_TYPES, caps):
        for i in range(min(cap, dataset.data_nums[scene_type])):
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
                condition=True, benchmarking=True)
            plan_t = time.time() - t0

            traj = stitch(trajs, guide, DEVICE)
            traj = np.clip(traj, lower[:, None], upper[:, None])

            if n_cub > 0: env.spawn_collision_cuboids(cub_cfg)
            if n_cyl > 0: env.spawn_collision_cylinders(cyl_cfg)
            success  = int(env.benchmark_trajectory(traj))
            t_success += success

            r = {'scene_type': scene_type, 'scene_num': i,
                 'success': success, 'plan_time': round(plan_t, 3)}
            scene_results.append(r)
            print(f"  [{scene_type}:{i+1:4d}]  success={success}  "
                  f"SR={t_success}/{len(scene_results)}  t={plan_t:.2f}s")

        # checkpoint after each scene type
        with open(out_path, 'w') as f:
            json.dump({'gpd': scene_results}, f, indent=2)

    print_partial(scene_results, "GPD final")
    return scene_results


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(edmp_r, gpd_r):
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    rows = []
    for method, rr in [('EDMP', edmp_r), ('GPD', gpd_r)]:
        sr  = sum(r['success'] for r in rr)
        tot = len(rr)
        avg = np.mean([r['plan_time'] for r in rr])
        rows.append((method, sr, tot, avg))
        print(f"  {method:6s}  {sr:4d}/{tot}  ({100*sr/tot:5.1f}%)  "
              f"avg {avg:.2f}s/scene")

    speedup = rows[0][3] / rows[1][3]
    print(f"\n  Speedup:  {speedup:.1f}×  faster with GPD")

    # per scene-type breakdown
    print("\n  Per scene-type:")
    for st in SCENE_TYPES:
        for method, rr in [('EDMP', edmp_r), ('GPD', gpd_r)]:
            sub = [r for r in rr if r['scene_type'] == st]
            if sub:
                sr = sum(r['success'] for r in sub)
                print(f"    {method:6s} {st:15s}  {sr}/{len(sub)} "
                      f"({100*sr/len(sub):.1f}%)")
    print("="*60)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['both', 'edmp', 'gpd'], default='both')
    parser.add_argument('--full',   action='store_true',
                        help='Run all 1800 scenes (default: 50-scene mini)')
    parser.add_argument('--out',    default='compare_results.json')
    args = parser.parse_args()

    caps    = FULL_CAPS if args.full else MINI_CAPS
    dataset = TestDataset(DATASET_TYPE, d_path=DATASET_PATH)
    env     = RobotEnvironment(gui=GUI)

    edmp_r, gpd_r = None, None

    if args.method in ('both', 'edmp'):
        edmp_r = run_edmp(dataset, env, caps, args.out.replace('.json', '_edmp.json'))

    if args.method in ('both', 'gpd'):
        gpd_r  = run_gpd(dataset, env, caps, args.out.replace('.json', '_gpd.json'))

    # Merge and save final
    result = {}
    if edmp_r: result['edmp'] = edmp_r
    if gpd_r:  result['gpd']  = gpd_r
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nFull results saved → {args.out}")

    if edmp_r and gpd_r:
        print_summary(edmp_r, gpd_r)


if __name__ == '__main__':
    main()
