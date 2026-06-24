"""
Stitching algorithm for GPD (Algorithm 2 from the paper).

Stitches K candidate trajectories into one collision-free trajectory by walking
the lowest-cost candidate forward and, on hitting a collision, bridging to a
collision-free waypoint on another candidate.

Two bridge modes:
  * use_rrt=True  (default): bridge with a self-contained RRT-Connect local
    planner that uses the guide's intersection-volume cost as the collision
    checker (this matches the paper's Algorithm 2, which uses RRT-Connect).
  * use_rrt=False: bridge with a 3-point linear interpolation (the previous
    behaviour; fast but can cut through obstacles).

If RRT-Connect fails for a bridge it falls back to the linear bridge.

Invariant: current_waypoint always strictly increases -> guaranteed termination.
"""

import numpy as np
import torch


# Franka Panda joint limits (rad) — used as RRT sampling bounds.
_LOWER = np.array([-166, -101, -166, -176, -166,  -1, -166], np.float32) * (np.pi / 180)
_UPPER = np.array([ 166,  101,  166,   -4,  166, 215,  166], np.float32) * (np.pi / 180)


def _swept_volume_costs(trajectories: np.ndarray, guide,
                        device: str) -> np.ndarray:
    """Total swept-volume cost per trajectory (K,). Zero clearance (t=0)."""
    K, _, N = trajectories.shape
    jt = torch.tensor(trajectories[:, :, 1:-1], dtype=torch.float32, device=device)
    start_t = torch.tensor(trajectories[0, :, 0],  dtype=torch.float32, device=device)
    goal_t  = torch.tensor(trajectories[0, :, -1], dtype=torch.float32, device=device)
    guide.define_obstacles(guide.obstacle_config, t=0, batch_size=K)
    with torch.no_grad():
        sv = guide.swept_volume_cost(jt, start_t, goal_t, t=0, batch_size=K)
        return sv.sum(dim=(1, 2)).cpu().numpy()


def _collision_mask(trajectories: np.ndarray, guide,
                    device: str, collision_fn=None) -> np.ndarray:
    """Per-waypoint collision flag, shape (K, N). True = colliding."""
    K, _, N = trajectories.shape
    if collision_fn is not None:
        # configs (K, 7, N) -> (K*N, 7); collision_fn returns free mask
        cfgs = trajectories.transpose(0, 2, 1).reshape(K * N, 7)
        free = collision_fn(cfgs).reshape(K, N)
        return ~free
    jt = torch.tensor(trajectories, dtype=torch.float32, device=device)
    guide.define_obstacles(guide.obstacle_config, t=0, batch_size=K)
    with torch.no_grad():
        vols = guide.cost(jt, t=0, batch_size=K)
        per_wp = vols.sum(dim=-1).cpu().numpy()
    return per_wp > 0.0


def _linear_bridge(q_a: np.ndarray, q_b: np.ndarray,
                   n_steps: int = 3) -> np.ndarray:
    """Linear interpolation waypoints between q_a and q_b (exclusive)."""
    alphas = np.linspace(0, 1, n_steps + 2)[1:-1]
    return np.stack([q_a + a * (q_b - q_a) for a in alphas])  # (n, 7)


# ----------------------------------------------------------------------------
# RRT-Connect local planner (collision checker = guide intersection volume)
# ----------------------------------------------------------------------------

def _configs_free(configs: np.ndarray, guide, device: str,
                  collision_fn=None) -> np.ndarray:
    """
    Collision-free mask for a batch of configurations.

    If collision_fn is given (e.g. env.configs_free, faithful PyBullet check),
    use it. Otherwise fall back to the guide AABB intersection-volume proxy.
    """
    if collision_fn is not None:
        return np.asarray(collision_fn(configs), dtype=bool)
    M = configs.shape[0]
    within = np.all((configs >= _LOWER) & (configs <= _UPPER), axis=1)  # (M,)
    jt = torch.tensor(configs.reshape(M, 7, 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        vols = guide.cost(jt, t=0, batch_size=M)          # (M, 1, no*nl)
        collide = (vols.sum(dim=-1).reshape(M) > 0.0).cpu().numpy()
    return within & (~collide)


def _edge_free(q_a: np.ndarray, q_b: np.ndarray, guide, device: str,
               res: float = 0.1, collision_fn=None) -> bool:
    """True if the straight line q_a->q_b is collision-free at resolution `res`."""
    d = float(np.max(np.abs(q_b - q_a)))
    n = max(2, int(np.ceil(d / res)) + 1)
    ts = np.linspace(0.0, 1.0, n)[:, None]               # (n,1)
    pts = q_a[None, :] * (1 - ts) + q_b[None, :] * ts    # (n,7)
    return bool(_configs_free(pts, guide, device, collision_fn).all())


def _trace(nodes, parents, idx):
    """Return path [root, ..., nodes[idx]] by walking parent pointers."""
    path = []
    while idx != -1:
        path.append(nodes[idx])
        idx = parents[idx]
    return path[::-1]


def rrt_connect(q_start: np.ndarray, q_goal: np.ndarray, guide, device: str,
                step: float = 0.3, res: float = 0.1, max_iter: int = 400,
                seed: int = 0, collision_fn=None):
    """
    Bidirectional RRT-Connect between two collision-free configs.

    Returns a list of configs [q_start, ..., q_goal] (collision-free, oriented
    start->goal) or None if no connection was found within max_iter.
    """
    # Fast path: direct straight line is already free.
    if _edge_free(q_start, q_goal, guide, device, res, collision_fn):
        return [q_start, q_goal]

    rng = np.random.default_rng(seed)

    def extend(nodes, parents, q_target):
        # nearest node
        arr = np.stack(nodes)                              # (n,7)
        i = int(np.argmin(np.sum((arr - q_target) ** 2, axis=1)))
        q_near = nodes[i]
        d = float(np.linalg.norm(q_target - q_near))
        if d <= step:
            q_new = q_target.copy()
        else:
            q_new = q_near + (step / d) * (q_target - q_near)
        if _edge_free(q_near, q_new, guide, device, res, collision_fn):
            nodes.append(q_new)
            parents.append(i)
            reached = bool(np.allclose(q_new, q_target, atol=1e-6))
            return ('reached' if reached else 'advanced'), len(nodes) - 1
        return 'trapped', i

    def connect(nodes, parents, q_target):
        status = 'advanced'
        while status == 'advanced':
            status, idx = extend(nodes, parents, q_target)
        return status, idx

    A_nodes, A_par = [q_start.copy()], [-1]
    B_nodes, B_par = [q_goal.copy()],  [-1]
    a_is_start = True

    for _ in range(max_iter):
        q_rand = rng.uniform(_LOWER, _UPPER).astype(np.float32)
        sA, iA = extend(A_nodes, A_par, q_rand)
        if sA != 'trapped':
            q_new = A_nodes[iA]
            sB, iB = connect(B_nodes, B_par, q_new)
            if sB == 'reached':
                pathA = _trace(A_nodes, A_par, iA)         # rootA .. q_new
                pathB = _trace(B_nodes, B_par, iB)         # rootB .. q_new
                full = pathA + pathB[::-1][1:]             # rootA .. q_new .. rootB
                if not a_is_start:                          # ensure start->goal order
                    full = full[::-1]
                # Orient to start (guards against any residual ambiguity)
                if np.linalg.norm(full[0] - q_start) > np.linalg.norm(full[-1] - q_start):
                    full = full[::-1]
                return full
        # swap trees
        A_nodes, B_nodes = B_nodes, A_nodes
        A_par, B_par = B_par, A_par
        a_is_start = not a_is_start

    return None


def _rrt_bridge(q_a, q_b, guide, device, max_iter, seed, collision_fn=None):
    """RRT-Connect interior waypoints between q_a and q_b (exclusive), or None."""
    path = rrt_connect(q_a, q_b, guide, device, max_iter=max_iter, seed=seed,
                       collision_fn=collision_fn)
    if path is None or len(path) <= 2:
        return None
    return np.stack(path[1:-1])                            # (n,7) interior only


def stitch(trajectories: np.ndarray, guide,
           device: str, bridge_steps: int = 3,
           dist_threshold: float = 2.0, use_rrt: bool = True,
           rrt_max_iter: int = 400, collision_fn=None) -> np.ndarray:
    """
    Stitch K candidate trajectories into one collision-free trajectory.

    Args:
        trajectories   : (K, 7, N) output of denoise_guided_poly
        guide          : IntersectionVolumeGuide
        device         : torch device string
        bridge_steps   : linear-bridge interpolation points (fallback)
        dist_threshold : max joint-space distance (rad) to accept a stitch
                         target (relaxed for RRT, which can connect farther)
        use_rrt        : bridge with RRT-Connect (True) or linear (False)
        rrt_max_iter   : RRT-Connect iteration budget per bridge

    Returns:
        stitched : (7, N_out)  or falls back to the best single trajectory
    """
    K, _, N = trajectories.shape

    # Sanity-check for NaN — fall back if denoising produced garbage
    if not np.isfinite(trajectories).all():
        finite_mask = np.isfinite(trajectories).all(axis=(1, 2))
        if not finite_mask.any():
            return np.zeros((7, N), dtype=np.float32)
        trajectories = trajectories[finite_mask]
        K = trajectories.shape[0]
        if K == 1:
            return trajectories[0]

    costs    = _swept_volume_costs(trajectories, guide, device)
    best_idx = int(np.argmin(costs))
    collision = _collision_mask(trajectories, guide, device, collision_fn)  # (K, N)

    # Fast path: best trajectory is entirely clean
    if not collision[best_idx].any():
        return trajectories[best_idx]

    waypoints    = []
    current_traj = best_idx
    current_wp   = 0
    last_safe    = trajectories[best_idx][:, 0]    # start config (pinned, safe)
    bridge_seed  = 0

    while current_wp < N:
        traj = trajectories[current_traj]          # (7, N)

        first_col = N
        for wi in range(current_wp, N):
            if collision[current_traj, wi]:
                first_col = wi
                break

        if first_col == N:
            waypoints.append(traj[:, current_wp:].T)
            break

        if first_col > current_wp:
            waypoints.append(traj[:, current_wp:first_col].T)
            last_safe = traj[:, first_col - 1]

        # Search other trajectories for a forward, collision-free, nearby target
        anchor = last_safe
        best_alt_traj = -1
        best_alt_wp   = N
        best_alt_dist = np.inf
        for alt in range(K):
            if alt == current_traj:
                continue
            alt_traj = trajectories[alt]
            for wi in range(first_col, N):
                if collision[alt, wi]:
                    continue
                d = np.linalg.norm(alt_traj[:, wi] - anchor)
                if d < dist_threshold and d < best_alt_dist:
                    best_alt_dist = d
                    best_alt_traj = alt
                    best_alt_wp   = wi
                    break

        if best_alt_traj >= 0:
            target = trajectories[best_alt_traj, :, best_alt_wp]
            bridge = None
            if use_rrt:
                bridge = _rrt_bridge(anchor, target, guide, device,
                                     rrt_max_iter, bridge_seed, collision_fn)
                bridge_seed += 1
            if bridge is None:                     # linear fallback (or use_rrt=False)
                bridge = _linear_bridge(anchor, target, bridge_steps)
            if len(bridge) > 0:
                waypoints.append(bridge)
            current_traj = best_alt_traj
            current_wp   = best_alt_wp
            last_safe    = target
        else:
            # No valid stitch target: accept the colliding waypoint and advance
            waypoints.append(traj[:, first_col:first_col + 1].T)
            current_wp = first_col + 1

    if not waypoints:
        return trajectories[best_idx]

    combined = np.concatenate(waypoints, axis=0).T     # (7, N_out)
    return combined
