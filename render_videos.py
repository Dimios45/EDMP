"""
Render MP4 inference videos for EDMP and GPD.
One video per scene type per method → 8 videos in assets/videos/

Uses PyBullet DIRECT mode + getCameraImage() for headless rendering.
Encodes with OpenCV VideoWriter (mp4v codec).

Usage:
    python render_videos.py
    python render_videos.py --scene_idx 0  # which scene index per type
"""

import argparse, os, time
import numpy as np
import torch
import cv2

import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data

from autolab_core import YamlConfig
from lib import *
from diffusion import *
from datasets.load_test_dataset import TestDataset
from diffusion.models.temporalunet import TemporalUNet as TemporalUNetGPD
from gpd.diffusion import PolynomialDiffusion
from gpd.stitch    import stitch
from gpd.edmp_gpu  import denoise_guided_gpu

os.makedirs('assets/videos', exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
SCENE_TYPES  = ['tabletop', 'cubby', 'merged_cubby', 'dresser']
DATASET_TYPE = 'hybrid'
DATASET_PATH = './datasets/'
DEVICE       = 'cuda:0' if torch.cuda.is_available() else 'cpu'

EDMP_T, EDMP_TRAJ_LEN, EDMP_MODEL_DIR = 255, 50, './models/'
EDMP_GUIDES, EDMP_BATCH_PER = [1,2,3,4,5,10,11,13,14,16,18,21], 10
GPD_T, GPD_N_CONTROL, GPD_TRAJ_LEN, GPD_MODEL_DIR = 64, 8, 50, './models/'
GPD_GUIDES,  GPD_BATCH_PER  = [1,2,3,4,5,10,11,13,14,16,18,21], 10
GUIDE_PATH, T_ORIG = './guides/', 255

# Video params
FPS          = 30
INTERP_STEPS = 8      # frames between waypoints → smooth motion
IMG_W, IMG_H = 640, 480

LOWER = np.array([-166,-101,-166,-176,-166,-1,-166], np.float32) * (np.pi/180)
UPPER = np.array([ 166, 101, 166,  -4, 166,215,166], np.float32) * (np.pi/180)


# ── Camera setup ───────────────────────────────────────────────────────────────
def make_camera(client):
    view = client.computeViewMatrix(
        cameraEyePosition    = [1.8, -1.2, 1.4],
        cameraTargetPosition = [0.2,  0.0, 0.5],
        cameraUpVector       = [0,    0,   1],
    )
    proj = client.computeProjectionMatrixFOV(
        fov=52, aspect=IMG_W/IMG_H, nearVal=0.1, farVal=10.0
    )
    return view, proj


def capture_frame(client, view, proj):
    _, _, rgba, _, _ = client.getCameraImage(
        IMG_W, IMG_H, view, proj,
        renderer=p.ER_TINY_RENDERER,
        flags=p.ER_NO_SEGMENTATION_MASK,
    )
    frame = np.array(rgba, dtype=np.uint8).reshape(IMG_H, IMG_W, 4)
    return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)


# ── PyBullet scene helpers ─────────────────────────────────────────────────────
def setup_scene(client, obs_cfg, cub_cfg, cyl_cfg, n_cub, n_cyl, colors):
    """Load obstacles into client."""
    ids = []
    if n_cub > 0:
        for i in range(n_cub):
            cuid = client.createCollisionShape(p.GEOM_BOX,
                halfExtents=cub_cfg[i, 7:] / 2)
            vuid = client.createVisualShape(p.GEOM_BOX,
                halfExtents=cub_cfg[i, 7:] / 2,
                rgbaColor=[*colors, 1.0])
            ids.append(client.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=cuid,
                baseVisualShapeIndex=vuid,
                basePosition=cub_cfg[i, :3],
                baseOrientation=cub_cfg[i, 3:7]))
    if n_cyl > 0:
        for i in range(n_cyl):
            cuid = client.createCollisionShape(p.GEOM_CYLINDER,
                radius=cyl_cfg[i, 7], height=cyl_cfg[i, 8])
            vuid = client.createVisualShape(p.GEOM_CYLINDER,
                radius=cyl_cfg[i, 7], length=cyl_cfg[i, 8],
                rgbaColor=[*colors, 1.0])
            ids.append(client.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=cuid,
                baseVisualShapeIndex=vuid,
                basePosition=cyl_cfg[i, :3],
                baseOrientation=cyl_cfg[i, 3:7]))
    return ids


def set_joints(client, robot, joints_7dof):
    active = [j for j in range(client.getNumJoints(robot))
              if client.getJointInfo(robot, j)[2] == client.JOINT_REVOLUTE]
    for i, ji in enumerate(active):
        client.resetJointState(robot, ji, joints_7dof[i])


def build_bullet_env():
    """Create a headless PyBullet client with Franka loaded."""
    client = bc.BulletClient(p.DIRECT)
    client.setAdditionalSearchPath(pybullet_data.getDataPath())
    client.setGravity(0, 0, -9.8)
    client.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    robot = client.loadURDF('franka_panda/panda.urdf',
                            basePosition=(0, 0, 0), useFixedBase=True)
    # Load a floor plane for aesthetics
    client.loadURDF('plane.urdf', basePosition=(0, 0, -0.001), useFixedBase=True)
    return client, robot


# ── Overlay helpers ────────────────────────────────────────────────────────────
def add_overlay(frame, method, scene_type, wp_idx, n_wp, success):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    # Semi-transparent top bar
    cv2.rectangle(overlay, (0, 0), (w, 56), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    color_method = (80, 200, 80) if method == 'GPD' else (80, 130, 255)
    cv2.putText(frame, method, (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color_method, 2, cv2.LINE_AA)
    cv2.putText(frame, f'{scene_type}', (w//2 - 90, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2, cv2.LINE_AA)

    # Progress bar
    bar_x, bar_y, bar_w, bar_h = 12, h - 20, w - 24, 8
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (60, 60, 60), -1)
    filled = int(bar_w * wp_idx / max(n_wp - 1, 1))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h),
                  color_method, -1)

    if success is not None:
        label = 'SUCCESS' if success else 'COLLISION'
        col   = (60, 220, 60) if success else (60, 60, 220)
        cv2.putText(frame, label, (w - 160, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
    return frame


# ── Guide config builder ───────────────────────────────────────────────────────
def build_guide_cfgs(guides, T, batch_per):
    n, tot = len(guides), len(guides) * batch_per
    cfgs = {
        'batch_size_per_guide': batch_per, 'total_batch_size': tot,
        'clearance':           np.zeros((tot, T)),
        'expansion':           np.zeros((tot, T)),
        'guidance_method':     np.zeros((tot,)),
        'grad_norm':           np.zeros((tot,)),
        'guidance_schedule':   np.zeros((tot, T)),
        'volume_trust_region': np.zeros((tot,)),
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
            r0 = int(np.clip(round(r[0]*T/T_ORIG), 0, T))
            r1 = int(np.clip(round(r[1]*T/T_ORIG), 0, T))
            if r1 > r0:
                cfgs['expansion'][sl, r0:r1] = np.linspace(o_e[vk][0], o_e[vk][1], r1-r0)
        cfgs['guidance_method'][sl]      = 1 if g['hyperparameters']['guidance_method']=='sv' else 0
        cfgs['grad_norm'][sl]            = 1 if g['hyperparameters']['grad_norm'] else 0
        cfgs['guidance_schedule'][sl, :] = (
            (1.4 + np.arange(T)/T)
            if g['hyperparameters']['guidance_schedule']['type']=='varying'
            else g['hyperparameters']['guidance_schedule']['scale_val'])
        cfgs['volume_trust_region'][sl]  = g['hyperparameters']['volume_trust_region']
    return cfgs


def filter_ik(guide, all_ik_goals, start_joints):
    vols = guide.cost(
        torch.tensor(all_ik_goals.reshape((-1,7,1)), device=DEVICE),
        0, batch_size=all_ik_goals.shape[0]
    ).sum(axis=(1,2)).cpu().numpy()
    mn  = np.min(vols)
    idx = np.argsort(vols)
    gj  = all_ik_goals[idx][vols[idx] < mn + 0.0008]
    ii  = np.argmin(np.linalg.norm(start_joints - gj, axis=1))
    return gj[ii]


# ── Main render loop ───────────────────────────────────────────────────────────
def render_trajectory(client, robot, view, proj, trajectory,
                      out_path, method, scene_type):
    """
    Render a (7, N) trajectory to MP4.
    Interpolates between waypoints for smooth motion.
    Returns success bool (no joint limit violations).
    """
    N = trajectory.shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (IMG_W, IMG_H))

    success = True
    frames = []

    for wi in range(N):
        q = trajectory[:, wi]
        if np.any(q < LOWER) or np.any(q > UPPER):
            success = False
        set_joints(client, robot, q)
        client.stepSimulation()

        # Interpolate frames between waypoints
        n_frames = INTERP_STEPS if wi < N - 1 else INTERP_STEPS * 2  # linger at goal
        for _ in range(n_frames):
            frame = capture_frame(client, view, proj)
            frame = add_overlay(frame, method, scene_type, wi, N, None)
            frames.append(frame)
            writer.write(frame)

    # Freeze final frame with result
    final_frame = frames[-1].copy()
    final_frame = add_overlay(final_frame, method, scene_type, N-1, N, success)
    for _ in range(FPS * 2):  # 2-second hold
        writer.write(final_frame)

    writer.release()
    print(f"  Saved {out_path}  ({N} waypoints, success={success})")
    return success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_idx', type=int, default=0,
                        help='Scene index within each scene type to render')
    args = parser.parse_args()

    dataset    = TestDataset(DATASET_TYPE, d_path=DATASET_PATH)
    scene_idx  = args.scene_idx

    # ── Load EDMP model ──────────────────────────────────────────────────────
    print("Loading EDMP model...")
    edmp_diffuser  = Diffusion(T=EDMP_T, device=DEVICE)
    edmp_model     = TemporalUNet(
        model_name=EDMP_MODEL_DIR+f'TemporalUNetModel{EDMP_T}_N{EDMP_TRAJ_LEN}',
        input_dim=7, time_dim=32, dims=(32,64,128,256,512,512), device=DEVICE)
    edmp_guide_cfgs = build_guide_cfgs(EDMP_GUIDES, EDMP_T, EDMP_BATCH_PER)
    edmp_total_bs   = edmp_guide_cfgs['total_batch_size']

    # ── Load GPD model ───────────────────────────────────────────────────────
    print("Loading GPD model...")
    gpd_diffuser = PolynomialDiffusion(T=GPD_T, device=DEVICE,
                                       n_control=GPD_N_CONTROL, traj_len=GPD_TRAJ_LEN)
    gpd_model    = TemporalUNetGPD(
        model_name=GPD_MODEL_DIR+f'GPDModel{GPD_T}_N{GPD_N_CONTROL}',
        input_dim=7, time_dim=32, dims=(32,64,128,256), device=DEVICE)
    gpd_guide_cfgs = build_guide_cfgs(GPD_GUIDES, GPD_T, GPD_BATCH_PER)
    gpd_total_bs   = gpd_guide_cfgs['total_batch_size']

    for scene_type in SCENE_TYPES:
        print(f"\n{'='*60}")
        print(f"Scene type: {scene_type}  index: {scene_idx}")
        print('='*60)

        # Fetch scene data
        (obs_cfg, cub_cfg, cyl_cfg, n_cub, n_cyl,
         start_j, all_ik) = dataset.fetch_data(scene_idx, scene_type)

        # ── EDMP inference ───────────────────────────────────────────────────
        print("  EDMP: planning...")
        guide_e = IntersectionVolumeGuide(obs_cfg, DEVICE, edmp_guide_cfgs, edmp_total_bs)
        goal_j  = filter_ik(guide_e, all_ik, start_j)

        t0 = time.time()
        trajs_e = denoise_guided_gpu(
            diffuser=edmp_diffuser, model=edmp_model, guide=guide_e,
            traj_len=EDMP_TRAJ_LEN, num_channels=7,
            guidance_schedule=edmp_guide_cfgs['guidance_schedule'],
            batch_size=edmp_total_bs, start=start_j, goal=goal_j,
            condition=True, benchmarking=False)
        traj_e = guide_e.choose_best_trajectory(start_j, goal_j, trajs_e)
        traj_e = np.clip(traj_e, LOWER[:, None], UPPER[:, None])
        print(f"  EDMP: planned in {time.time()-t0:.2f}s")

        # ── GPD inference ────────────────────────────────────────────────────
        print("  GPD:  planning...")
        guide_g = IntersectionVolumeGuide(obs_cfg, DEVICE, gpd_guide_cfgs, gpd_total_bs)
        t0 = time.time()
        trajs_g = gpd_diffuser.denoise_guided_poly(
            model=gpd_model, guide=guide_g, num_channels=7,
            guidance_schedule=gpd_guide_cfgs['guidance_schedule'],
            batch_size=gpd_total_bs, start=start_j, goal=goal_j,
            condition=True, benchmarking=False)
        traj_g = stitch(trajs_g, guide_g, DEVICE)
        traj_g = np.clip(traj_g, LOWER[:, None], UPPER[:, None])
        print(f"  GPD:  planned in {time.time()-t0:.2f}s")

        # ── Render EDMP video ────────────────────────────────────────────────
        client, robot = build_bullet_env()
        view, proj    = make_camera(client)
        obs_color     = [0.95, 0.75, 0.20]   # warm yellow for obstacles
        obs_ids = setup_scene(client, obs_cfg, cub_cfg, cyl_cfg,
                              n_cub, n_cyl, obs_color)

        out_edmp = f'assets/videos/edmp_{scene_type}.mp4'
        render_trajectory(client, robot, view, proj, traj_e,
                          out_edmp, 'EDMP', scene_type)
        client.disconnect()

        # ── Render GPD video ─────────────────────────────────────────────────
        client, robot = build_bullet_env()
        view, proj    = make_camera(client)
        obs_ids = setup_scene(client, obs_cfg, cub_cfg, cyl_cfg,
                              n_cub, n_cyl, obs_color)

        out_gpd = f'assets/videos/gpd_{scene_type}.mp4'
        render_trajectory(client, robot, view, proj, traj_g,
                          out_gpd, 'GPD', scene_type)
        client.disconnect()

    print("\nAll videos saved to assets/videos/:")
    for f in sorted(os.listdir('assets/videos')):
        size = os.path.getsize(f'assets/videos/{f}') // 1024
        print(f"  assets/videos/{f}  ({size} KB)")


if __name__ == '__main__':
    main()
