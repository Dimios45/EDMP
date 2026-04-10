"""
Stitching algorithm for GPD (Algorithm 2 from the paper).

Invariant: current_waypoint always strictly increases → guaranteed termination.

Strategy:
  1. Pick candidate with lowest swept-volume cost.
  2. Walk forward through it waypoint by waypoint.
  3. When a collision is found, search ALL other trajectories for the
     earliest (lowest index ≥ current_waypoint) collision-free waypoint
     that is also close in joint space to the current position.
  4. If a good stitch point is found, bridge to it and continue from there.
  5. If none found, just accept the colliding waypoint and advance by 1.
     This guarantees the loop always terminates in at most N iterations.
"""

import numpy as np
import torch


def _swept_volume_costs(trajectories: np.ndarray, guide,
                        device: str) -> np.ndarray:
    """
    Total swept-volume cost per trajectory (K,).
    Uses zero clearance (t=0) for final selection.
    """
    K, _, N = trajectories.shape
    jt = torch.tensor(trajectories[:, :, 1:-1], dtype=torch.float32,
                      device=device)
    start_t = torch.tensor(trajectories[0, :, 0],  dtype=torch.float32,
                            device=device)
    goal_t  = torch.tensor(trajectories[0, :, -1], dtype=torch.float32,
                            device=device)
    guide.define_obstacles(guide.obstacle_config, t=0, batch_size=K)
    with torch.no_grad():
        sv = guide.swept_volume_cost(jt, start_t, goal_t, t=0,
                                     batch_size=K)
        return sv.sum(dim=(1, 2)).cpu().numpy()  # (K,)


def _collision_mask(trajectories: np.ndarray, guide,
                    device: str) -> np.ndarray:
    """
    Per-waypoint collision flag, shape (K, N).
    A waypoint is colliding if its intersection-volume cost > 0.
    """
    K, _, N = trajectories.shape
    jt = torch.tensor(trajectories, dtype=torch.float32, device=device)
    guide.define_obstacles(guide.obstacle_config, t=0, batch_size=K)
    with torch.no_grad():
        vols = guide.cost(jt, t=0, batch_size=K)   # (K, N, no*nl)
        per_wp = vols.sum(dim=-1).cpu().numpy()      # (K, N)
    return per_wp > 0.0


def _linear_bridge(q_a: np.ndarray, q_b: np.ndarray,
                   n_steps: int = 3) -> np.ndarray:
    """Linear interpolation waypoints between q_a and q_b (exclusive)."""
    alphas = np.linspace(0, 1, n_steps + 2)[1:-1]
    return np.stack([q_a + a * (q_b - q_a) for a in alphas])  # (n, 7)


def stitch(trajectories: np.ndarray, guide,
           device: str, bridge_steps: int = 3,
           dist_threshold: float = 1.0) -> np.ndarray:
    """
    Stitch K candidate trajectories into one collision-free trajectory.

    Args:
        trajectories    : (K, 7, N) output of denoise_guided_poly
        guide           : IntersectionVolumeGuide
        device          : torch device string
        bridge_steps    : linear interpolation points when stitching
        dist_threshold  : max joint-space distance (rad) to accept a stitch

    Returns:
        stitched : (7, N_out) or falls back to the best single trajectory
    """
    K, _, N = trajectories.shape

    # Sanity-check for NaN — fall back if denoising produced garbage
    if not np.isfinite(trajectories).all():
        finite_mask = np.isfinite(trajectories).all(axis=(1, 2))
        if not finite_mask.any():
            # All NaN — return zeros as last resort
            return np.zeros((7, N), dtype=np.float32)
        trajectories = trajectories[finite_mask]
        K = trajectories.shape[0]
        if K == 1:
            return trajectories[0]

    # Pick best trajectory by total swept-volume cost
    costs    = _swept_volume_costs(trajectories, guide, device)  # (K,)
    best_idx = int(np.argmin(costs))

    collision = _collision_mask(trajectories, guide, device)  # (K, N)

    # Fast path: best trajectory is entirely clean
    if not collision[best_idx].any():
        return trajectories[best_idx]

    # ----- Stitching loop — current_waypoint strictly increases -----
    waypoints   = []          # list of (n, 7) arrays
    current_traj = best_idx
    current_wp   = 0          # always moves forward

    while current_wp < N:
        traj = trajectories[current_traj]   # (7, N)

        # Find first collision from current_wp onwards
        first_col = N
        for wi in range(current_wp, N):
            if collision[current_traj, wi]:
                first_col = wi
                break

        if first_col == N:
            # Remaining segment is clean — take it and finish
            waypoints.append(traj[:, current_wp:].T)   # (n, 7)
            break

        # Append clean prefix (may be empty if collision at current_wp)
        if first_col > current_wp:
            waypoints.append(traj[:, current_wp:first_col].T)

        # Search other trajectories for a stitch point:
        #   - Must be collision-free
        #   - Must be AHEAD of first_col (so we make forward progress)
        #   - Must be close in joint space
        anchor  = traj[:, first_col]   # (7,)
        best_alt_traj = -1
        best_alt_wp   = N
        best_alt_dist = np.inf

        for alt in range(K):
            if alt == current_traj:
                continue
            alt_traj = trajectories[alt]
            # Only consider waypoints beyond first_col (forward progress)
            for wi in range(first_col, N):
                if collision[alt, wi]:
                    continue
                d = np.linalg.norm(alt_traj[:, wi] - anchor)
                if d < dist_threshold and d < best_alt_dist:
                    best_alt_dist = d
                    best_alt_traj = alt
                    best_alt_wp   = wi
                    break  # take earliest good waypoint in this alt traj

        if best_alt_traj >= 0:
            # Bridge from anchor to the stitch point
            bridge = _linear_bridge(
                anchor,
                trajectories[best_alt_traj, :, best_alt_wp],
                bridge_steps
            )
            waypoints.append(bridge)
            current_traj = best_alt_traj
            current_wp   = best_alt_wp
            # current_wp >= first_col > previous current_wp → guaranteed advance
        else:
            # No valid stitch found: accept the colliding waypoint and move on
            waypoints.append(traj[:, first_col:first_col + 1].T)
            current_wp = first_col + 1
            # +1 guarantees strict forward progress

    if not waypoints:
        return trajectories[best_idx]

    combined = np.concatenate(waypoints, axis=0).T   # (7, N_out)
    return combined
