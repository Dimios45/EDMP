"""
Post-hoc trajectory-optimization refinement for GPD (PRESTO/DRAFTO-style
continuous-feasibility repair, warm-started from the diffusion+stitch output).

Motivation (failure-mode analysis): the dominant baseline failure is *dynamic*
collision — the stitched path is collision-free at every waypoint but the robot
sweeps through obstacles *between* waypoints during execution. So we (a) optimize
the interior waypoints to drive the guide's differentiable collision cost to zero
at waypoints AND edge midpoints, with a smoothness term and pinned endpoints;
(b) verify on a densified (edge-sampled) faithful collision check; (c) optionally
return a densified trajectory so the executor tracks it without overshoot.
"""
import numpy as np
import torch

_LOWER = torch.tensor([-166,-101,-166,-176,-166,-1,-166], dtype=torch.float32) * (np.pi/180)
_UPPER = torch.tensor([ 166, 101, 166,  -4, 166,215, 166], dtype=torch.float32) * (np.pi/180)


def densify(traj: np.ndarray, factor: int) -> np.ndarray:
    """Linearly interpolate (7, N) -> (7, ~factor*N) for smooth execution."""
    if factor <= 1:
        return traj
    N = traj.shape[1]
    xs = np.linspace(0.0, 1.0, N)
    xt = np.linspace(0.0, 1.0, (N - 1) * factor + 1)
    return np.stack([np.interp(xt, xs, traj[j]) for j in range(traj.shape[0])])


def _edge_configs(traj_np: np.ndarray, sub: int) -> np.ndarray:
    """All configs along the path including `sub` samples per edge: (M,7)."""
    return densify(traj_np, sub).T


def trajopt_refine(traj: np.ndarray, guide, device: str,
                   iters: int = 60, lr: float = 0.02, smooth_w: float = 0.05,
                   mid_w: float = 1.0, cost_obj=None,
                   collision_fn=None, check_every: int = 5,
                   out_densify: int = 3, verify_sub: int = 4) -> np.ndarray:
    """
    traj : (7, N) stitched/clipped waypoint trajectory.
    Returns a refined (and densified) trajectory; collision-free under the
    densified faithful check when repair succeeds, else best effort.
    cost_obj : optional object with .cost(jt,t,batch_size) -> (B,N) differentiable
               collision cost (e.g. SphereSDFCost). Defaults to `guide` (#5 ablation).
    """
    N = traj.shape[1]
    if N < 3:
        return traj
    if cost_obj is None:
        cost_obj = guide
    lower = _LOWER.to(device)[:, None]
    upper = _UPPER.to(device)[:, None]

    full = torch.tensor(traj, dtype=torch.float32, device=device)   # (7, N)
    start = full[:, :1].clone(); goal = full[:, -1:].clone()
    q_int = full[:, 1:-1].clone().requires_grad_(True)             # (7, N-2)

    guide.define_obstacles(guide.obstacle_config, t=0, batch_size=1)
    opt = torch.optim.Adam([q_int], lr=lr)

    def assemble():
        return torch.cat([start, q_int, goal], dim=1)             # (7, N)

    def free_densified(traj_np):
        if collision_fn is None:
            return False
        return bool(collision_fn(_edge_configs(traj_np, verify_sub)).all())

    # Fast exit only if already edge-free (catches swept collisions, not just waypoints).
    cur = assemble().detach().cpu().numpy()
    if free_densified(cur):
        return densify(cur, out_densify)

    for it in range(iters):
        jt = assemble().unsqueeze(0)                               # (1,7,N)
        mid = 0.5 * (jt[:, :, 1:] + jt[:, :, :-1])                 # edge midpoints
        col = cost_obj.cost(jt, t=0, batch_size=1).sum()
        if mid_w > 0.0:
            col = col + mid_w * cost_obj.cost(mid, t=0, batch_size=1).sum()
        acc = jt[0, :, 2:] - 2 * jt[0, :, 1:-1] + jt[0, :, :-2]
        loss = col + smooth_w * (acc ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            q_int.clamp_(lower, upper)
        if collision_fn is not None and (it + 1) % check_every == 0:
            cur = assemble().detach().cpu().numpy()
            if free_densified(cur):
                return densify(cur, out_densify)

    return densify(assemble().detach().cpu().numpy(), out_densify)
