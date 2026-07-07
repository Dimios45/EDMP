"""
Render the four MPiNets environment types (Franka Panda + obstacles) for the
qualitative figure. Headless via PyBullet's TinyRenderer (DIRECT mode).
Outputs results/env_<type>.png and a 1x4 montage results/env_montage.png.
"""
import os, sys, numpy as np
os.environ.setdefault('OMP_NUM_THREADS', '1')
import pybullet as p
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import run_worker as RW
from datasets.load_test_dataset import TestDataset
from lib.environment import RobotEnvironment
from lib.guide import IntersectionVolumeGuide

SCENE_TYPES = ['tabletop', 'cubby', 'merged_cubby', 'dresser']
IDX = {'tabletop': 2, 'cubby': 1, 'merged_cubby': 1, 'dresser': 3}   # picked for clarity
W, H = 680, 620
ds = TestDataset('hybrid', d_path='datasets/')
env = RobotEnvironment(gui=False)
cli = env.client_id
_OBS = []
YELLOW = [0.95, 0.78, 0.18, 0.42]   # translucent so the robot shows through


def spawn_translucent(cub, cyl, ncub, ncyl):
    for i in range(ncub):
        v = cli.createVisualShape(p.GEOM_BOX, halfExtents=cub[i, 7:] / 2, rgbaColor=YELLOW)
        _OBS.append(cli.createMultiBody(baseVisualShapeIndex=v,
                    basePosition=cub[i, :3], baseOrientation=cub[i, 3:7]))
    for i in range(ncyl):
        v = cli.createVisualShape(p.GEOM_CYLINDER, radius=cyl[i, 7], length=cyl[i, 8], rgbaColor=YELLOW)
        _OBS.append(cli.createMultiBody(baseVisualShapeIndex=v,
                    basePosition=cyl[i, :3], baseOrientation=cyl[i, 3:7]))


def clear_translucent():
    while _OBS:
        cli.removeBody(_OBS.pop())


def set_config(q):
    for i, j in enumerate(env.joints):
        cli.resetJointState(env.manipulator, j, float(q[i]))


def snap(target, dist, yaw, pitch):
    view = cli.computeViewMatrixFromYawPitchRoll(target, dist, yaw, pitch, 0, 2)
    proj = cli.computeProjectionMatrixFOV(fov=55, aspect=W / H, nearVal=0.1, farVal=5)
    img = cli.getCameraImage(W, H, view, proj, renderer=p.ER_TINY_RENDERER,
                             lightDirection=[1, 1, 1.4])
    rgb = np.reshape(img[2], (H, W, 4))[:, :, :3].astype(np.uint8)
    return rgb


gc = RW.build_guide_cfgs([1], 64, 32); bs = gc['total_batch_size']
# per-scene camera (target, dist, yaw, pitch) framing the reach into the workspace
CAM = {'tabletop':     ([0.30, 0.0, 0.35], 2.1, 55, -30),
       'cubby':        ([0.45, 0.0, 0.55], 2.3, 35, -18),
       'merged_cubby': ([0.45, 0.0, 0.55], 2.4, 40, -20),
       'dresser':      ([0.35, 0.0, 0.55], 2.2, 35, -22)}
imgs = {}
for st in SCENE_TYPES:
    clear_translucent(); env.go_home()
    obs, cub, cyl, ncub, ncyl, sj, aik = ds.fetch_data(IDX[st], st)
    guide = IntersectionVolumeGuide(obs, RW.DEVICE, gc, bs)
    gj = RW.filter_ik(guide, aik, sj)          # goal: arm reaching into the scene
    spawn_translucent(cub, cyl, ncub, ncyl)
    set_config(gj)
    t, d, y, pi = CAM[st]
    rgb = snap(target=t, dist=d, yaw=y, pitch=pi)
    imgs[st] = rgb
    plt.imsave(f'results/env_{st}.png', rgb)
    print(f'rendered {st} (cuboids={ncub} cylinders={ncyl})', flush=True)

fig, ax = plt.subplots(1, 4, figsize=(13, 3.0))
titles = {'tabletop': 'Tabletop', 'cubby': 'Cubby',
          'merged_cubby': 'Merged cubby', 'dresser': 'Dresser'}
for a, st in zip(ax, SCENE_TYPES):
    a.imshow(imgs[st]); a.set_title(titles[st], fontsize=13); a.axis('off')
plt.tight_layout()
plt.savefig('results/env_montage.png', dpi=140, bbox_inches='tight')
print('saved results/env_montage.png')
