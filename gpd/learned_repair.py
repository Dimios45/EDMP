"""
Learned amortized repair operator: map (stitched seed trajectory + obstacle set) to a
feasible trajectory in ONE forward pass, distilling the iterative trajopt repair. Reuses
the masked-DeepSets scene encoder from the conditioning extension. Predicts a residual
(endpoints pinned) so it defaults to identity. No diffusion retraining.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from gpd.conditional_unet import DeepSetsSceneEncoder


class LearnedRepair(nn.Module):
    def __init__(self, traj_len: int = 50, scene_dim: int = 64, hidden: int = 128,
                 model_dir: str = './models/LearnedRepair'):
        super().__init__()
        self.L = traj_len
        self.scene_enc = DeepSetsSceneEncoder(feat_dim=12, hidden=128, out_dim=scene_dim)
        cin = 7 + scene_dim
        self.net = nn.Sequential(
            nn.Conv1d(cin, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, 7, 5, padding=2),
        )
        self.model_dir = model_dir

    def forward(self, seed, obs, mask):
        """seed (B,7,L), obs (B,52,12), mask (B,52) -> repaired (B,7,L)."""
        B, _, L = seed.shape
        s = self.scene_enc(obs, mask).unsqueeze(-1).expand(-1, -1, L)  # (B,scene_dim,L)
        res = self.net(torch.cat([seed, s], dim=1))                   # (B,7,L)
        keep = torch.ones(L, device=seed.device); keep[0] = 0.0; keep[-1] = 0.0
        return seed + res * keep.view(1, 1, L)                         # pin endpoints

    def save(self):
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(self.model_dir, 'weights.pt'))

    def load(self, device='cpu'):
        self.load_state_dict(torch.load(os.path.join(self.model_dir, 'weights.pt'),
                                        map_location=device))
        return self


@torch.no_grad()
def repair_batch(model, seed_np, obs_np, mask_np, device):
    """Convenience one-shot repair for a single (7,L) trajectory + scene arrays."""
    seed = torch.tensor(seed_np[None], dtype=torch.float32, device=device)
    obs = torch.tensor(obs_np[None], dtype=torch.float32, device=device)
    mask = torch.tensor(mask_np[None], dtype=torch.bool, device=device)
    return model(seed, obs, mask)[0].cpu().numpy()
