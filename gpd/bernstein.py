"""
Bernstein polynomial utilities for GPD.

Trajectory representation:  q = alpha @ B.T
  alpha : (batch, 7, n_control)   — control points in joint space
  B     : (traj_len, n_control)   — Bernstein basis matrix
  q     : (batch, 7, traj_len)    — waypoint trajectory

Boundary property (degree = n_control - 1):
  B[0,  0]             = 1  → q[:, :, 0]  = alpha[:, :, 0]   (start)
  B[-1, n_control - 1] = 1  → q[:, :, -1] = alpha[:, :, -1]  (goal)

So start/goal conditioning is simply:
  alpha[:, :, 0]  = start
  alpha[:, :, -1] = goal

Gradient preconditioning (chain rule):
  dJ/d_alpha = (dJ/d_q) @ B      shape: (batch, 7, n_control)
"""

from math import comb
import numpy as np
import torch


def make_bernstein_matrix(traj_len: int, n_control: int) -> np.ndarray:
    """
    Returns B of shape (traj_len, n_control), float64.
    B[i, j] = C(degree, j) * t_i^j * (1 - t_i)^(degree - j)
    where t_i = i / (traj_len - 1).
    """
    degree = n_control - 1
    t = np.linspace(0.0, 1.0, traj_len)
    B = np.zeros((traj_len, n_control), dtype=np.float64)
    for j in range(n_control):
        B[:, j] = comb(degree, j) * (t ** j) * ((1.0 - t) ** (degree - j))
    return B


def make_bernstein_pinv(B: np.ndarray) -> np.ndarray:
    """
    Returns the Moore-Penrose pseudoinverse of B: shape (n_control, traj_len).
    Used for least-squares fitting: alpha^T = pinv(B) @ q^T
    i.e., alpha = q @ pinv(B).T    (batch, 7, traj_len) @ (traj_len, n_control)
    """
    return np.linalg.pinv(B)


def fit_bernstein(q: np.ndarray, B_pinv: np.ndarray) -> np.ndarray:
    """
    Fit Bernstein control points to a trajectory via least squares.

    Args:
        q      : (7, traj_len) or (batch, 7, traj_len) — waypoint trajectory
        B_pinv : (n_control, traj_len) — pseudoinverse of B

    Returns:
        alpha  : same leading dims as q but last dim = n_control
    """
    # alpha = q @ B_pinv.T  →  (... , n_control)
    alpha = q @ B_pinv.T
    # Enforce boundary conditions (Bernstein already satisfies them
    # approximately from the fit; this makes them exact).
    alpha[..., 0]  = q[..., 0]
    alpha[..., -1] = q[..., -1]
    return alpha


class BernsteinLayer:
    """
    Holds pre-computed B and B_pinv for a fixed (traj_len, n_control) pair.
    Provides numpy and torch conversions.
    """

    def __init__(self, traj_len: int = 50, n_control: int = 8):
        self.traj_len  = traj_len
        self.n_control = n_control
        self.B      = make_bernstein_matrix(traj_len, n_control)   # (N, M)
        self.B_pinv = make_bernstein_pinv(self.B)                  # (M, N)

    def to_waypoints_np(self, alpha: np.ndarray) -> np.ndarray:
        """alpha: (..., 7, M)  →  q: (..., 7, N)"""
        return alpha @ self.B.T

    def to_control_np(self, q: np.ndarray) -> np.ndarray:
        """q: (..., 7, N)  →  alpha: (..., 7, M)"""
        return fit_bernstein(q, self.B_pinv)

    def to_waypoints_torch(self, alpha: torch.Tensor, device=None) -> torch.Tensor:
        """alpha: (..., 7, M)  →  q: (..., 7, N)"""
        B_t = torch.tensor(self.B, dtype=torch.float32,
                           device=device or alpha.device)
        return alpha @ B_t.T

    def precondition_gradient(self, grad_q: np.ndarray) -> np.ndarray:
        """
        Convert waypoint-space gradient to control-point-space gradient.
        grad_q : (..., 7, N)
        returns : (..., 7, M)   via chain rule: dJ/d_alpha = grad_q @ B
        """
        return grad_q @ self.B
