"""
#5: Exact-er GPU-batched differentiable collision cost for trajopt repair.

The guide's `cost()` uses an AABB-vs-AABB intersection volume: after FK it bounds
each *rotated* robot-link box by an axis-aligned box AND uses the obstacles'
axis-aligned bounds — a double inflation that makes the differentiable objective
loose. Here we instead (a) approximate each link by a chain of spheres whose
centers are placed *exactly* by FK (no robot-side AABB inflation), and (b) score
penetration against the obstacles' true *oriented* boxes (no obstacle-side AABB
inflation). Penetration is a smooth, differentiable, zero-when-free objective —
a drop-in replacement for `guide.cost(jt, t=0, batch_size=1)` in `trajopt_refine`.

Reuses the guide for FK (`get_link_transform`), link box dims, and obstacle
geometry (`obstacle_config` = [cx,cy,cz, qx,qy,qz,qw, sx,sy,sz], full extents).
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


class SphereSDFCost:
    def __init__(self, guide, device, spheres_per_link_axis=2):
        self.guide = guide
        self.device = device
        self._build_robot_spheres(spheres_per_link_axis)
        self._build_obstacles()

    # ---- robot model: spheres along each link box's long axis ------------------
    def _build_robot_spheres(self, dens):
        dims = self.guide.link_dimensions.detach().cpu().numpy()   # (9,3) full extents
        centers, radii, link_id = [], [], []
        for i, d in enumerate(dims):
            order = np.argsort(d)[::-1]            # long..short axis
            la, m1, m2 = order[0], order[1], order[2]
            L = float(d[la])
            r = 0.5 * float(np.sqrt(d[m1] ** 2 + d[m2] ** 2))   # circumscribe cross-section
            r = max(r, 1e-3)
            K = max(1, int(np.ceil(L / max(r, 1e-3))) * dens // 2 + 1)
            pos = np.linspace(-(L / 2), (L / 2), K) if K > 1 else np.array([0.0])
            for p in pos:
                c = np.zeros(3, np.float32); c[la] = p
                centers.append(c); radii.append(r); link_id.append(i)
        # homogeneous sphere centers in their link frames: (S,4)
        c = np.array(centers, np.float32)
        self.sph_local = torch.tensor(
            np.concatenate([c, np.ones((len(c), 1), np.float32)], axis=1),
            device=self.device)                                   # (S,4)
        self.sph_r   = torch.tensor(np.array(radii, np.float32), device=self.device)   # (S,)
        self.sph_lid = torch.tensor(np.array(link_id, np.int64),  device=self.device)  # (S,)
        self.S = len(centers)

    # ---- obstacles: oriented boxes (center, world->local rot, half-extents) -----
    def _build_obstacles(self):
        cfg = np.asarray(self.guide.obstacle_config)              # (no,10)
        cen = cfg[:, :3].astype(np.float32)
        Rb  = np.stack([R.from_quat(cfg[i, 3:7]).as_matrix() for i in range(len(cfg))]).astype(np.float32)
        he  = (cfg[:, 7:] * 0.5).astype(np.float32)               # half-extents
        self.obs_c  = torch.tensor(cen, device=self.device)       # (no,3)
        self.obs_Rt = torch.tensor(Rb,  device=self.device).transpose(1, 2)  # world->local (no,3,3)
        self.obs_he = torch.tensor(he,  device=self.device)       # (no,3)
        self.no = len(cfg)

    def cost(self, joint_input, t=0, batch_size=1):
        """joint_input (B,7,N) -> per-(B,N) penetration sum, shape (B,N) so .sum() works
        like guide.cost. Differentiable in joint_input."""
        joints = self.guide.rearrange_joints(joint_input)         # (B,N,7)
        lt = self.guide.get_link_transform(joints)                # (B,N,9,4,4)
        B, N = lt.shape[0], lt.shape[1]
        # world sphere centers: pick each sphere's link transform, apply to local center
        T = lt[:, :, self.sph_lid, :, :]                          # (B,N,S,4,4)
        pw = (T @ self.sph_local.view(1, 1, self.S, 4, 1)).squeeze(-1)[..., :3]  # (B,N,S,3)
        # into each obstacle's local frame: (B,N,S,no,3)
        rel = pw.unsqueeze(3) - self.obs_c.view(1, 1, 1, self.no, 3)
        pl  = torch.einsum('bnsoi,oij->bnsoj', rel, self.obs_Rt)  # world->local
        q   = pl.abs() - self.obs_he.view(1, 1, 1, self.no, 3)
        outside = torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1)
        inside  = torch.clamp(q.max(dim=-1).values, max=0.0)
        sdf = outside + inside                                    # (B,N,S,no) point->box
        pen = torch.clamp(self.sph_r.view(1, 1, self.S, 1) - sdf, min=0.0)
        return pen.sum(dim=(2, 3))                                # (B,N)

    @torch.no_grad()
    def clearance(self, joint_input):
        """Signed min surface clearance per waypoint (B,N): sdf - radius, minimized
        over spheres x obstacles. >0 = free margin, <0 = deepest penetration."""
        joints = self.guide.rearrange_joints(joint_input)
        lt = self.guide.get_link_transform(joints)
        B, N = lt.shape[0], lt.shape[1]
        T = lt[:, :, self.sph_lid, :, :]
        pw = (T @ self.sph_local.view(1, 1, self.S, 4, 1)).squeeze(-1)[..., :3]
        rel = pw.unsqueeze(3) - self.obs_c.view(1, 1, 1, self.no, 3)
        pl  = torch.einsum('bnsoi,oij->bnsoj', rel, self.obs_Rt)
        q   = pl.abs() - self.obs_he.view(1, 1, 1, self.no, 3)
        outside = torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1)
        inside  = torch.clamp(q.max(dim=-1).values, max=0.0)
        sdf = outside + inside - self.sph_r.view(1, 1, self.S, 1)
        return sdf.amin(dim=(2, 3))                               # (B,N)
