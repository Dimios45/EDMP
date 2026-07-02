"""
Oracle-guided candidate selection for guided diffusion planners.

Our study found the *faithful* collision oracle (PyBullet), not the differentiable
guide cost, gates planning success. This module selects among the K diffusion
candidates by that faithful oracle on a densified path, rather than by the guide's
(loose, differentiable) swept/intersection-volume cost — a verifier/best-of-K
test-time-selection lever that directly exploits that finding.
"""
import numpy as np
from gpd.refine import densify


def oracle_select(trajs, collision_fn, densify_factor: int = 4):
    """trajs: (K,7,N) candidate waypoint trajectories. collision_fn: (M,7)->bool free
    mask (faithful). Returns (best_traj (7,N), best_free_fraction, best_k). Best =
    highest fraction of collision-free configs along the densified path."""
    K = trajs.shape[0]
    best_k, best_frac = 0, -1.0
    for k in range(K):
        cfgs = densify(trajs[k], densify_factor).T          # ((N-1)*f+1, 7)
        frac = float(collision_fn(cfgs).mean())             # fraction collision-free
        if frac > best_frac:
            best_frac, best_k = frac, k
        if best_frac >= 1.0:                                 # early exit: fully free
            break
    return trajs[best_k], best_frac, best_k


def resample_to(traj, L: int = 50):
    """Resample a (7,N) trajectory to (7,L) by linear interpolation (for fixed-length
    learned-repair datasets)."""
    N = traj.shape[1]
    if N == L:
        return traj.astype(np.float32)
    xs = np.linspace(0.0, 1.0, N); xt = np.linspace(0.0, 1.0, L)
    return np.stack([np.interp(xt, xs, traj[j]) for j in range(traj.shape[0])]).astype(np.float32)
