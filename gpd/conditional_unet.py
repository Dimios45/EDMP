"""
Scene-conditioned TemporalUNet for D3 (classifier-free guided polynomial diffusion).

Conditioning is injected by ADDING a scene embedding to the time embedding, so it
flows through the existing FiLM machinery in every Down/Middle/Up block with no
block changes. A learned null embedding supports classifier-free guidance (CFG).

Scene encoder: masked DeepSets over obstacle primitives (permutation-invariant).
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.models.blocks import (TimeEmbedding, DownSampler, MiddleBlock,
                                     UpSampler, Conv1dBlock)


class DeepSetsSceneEncoder(nn.Module):
    """Masked DeepSets: phi per obstacle -> masked mean-pool -> rho -> scene emb."""

    def __init__(self, feat_dim: int = 12, hidden: int = 128, out_dim: int = 32):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # obs: (B, S, feat)  mask: (B, S) bool/float
        h = self.phi(obs)                                   # (B, S, hidden)
        m = mask.float().unsqueeze(-1)                      # (B, S, 1)
        h = h * m
        pooled = h.sum(dim=1) / m.sum(dim=1).clamp(min=1.0) # masked mean (B, hidden)
        return self.rho(pooled)                             # (B, out_dim)


class ConditionalTemporalUNet(nn.Module):

    def __init__(self, model_name, input_dim, time_dim, device,
                 dims=(32, 64, 128, 256), scene_feat=12, scene_hidden=128):
        super().__init__()
        self.device = device
        full = [input_dim, *dims]

        self.time_embedding = TimeEmbedding(time_dim, device)

        self.down_samplers = nn.ModuleList([])
        for i in range(len(full) - 2):
            self.down_samplers.append(DownSampler(full[i], full[i + 1], time_dim))
        self.down_samplers.append(DownSampler(full[-2], full[-1], time_dim, is_last=True))

        self.middle_block = MiddleBlock(full[-1], time_dim)

        self.up_samplers = nn.ModuleList([])
        for i in range(len(full) - 1, 1, -1):
            self.up_samplers.append(UpSampler(full[i - 1], full[i], time_dim))

        self.final_conv = nn.Sequential(
            Conv1dBlock(full[1], full[1], kernel_size=5),
            nn.Conv1d(full[1], input_dim, kernel_size=1))

        # --- scene conditioning ---
        self.scene_encoder = DeepSetsSceneEncoder(scene_feat, scene_hidden, time_dim)
        self.null_scene = nn.Parameter(torch.zeros(time_dim))  # learned CFG null

        # --- persistence (load AFTER all submodules exist) ---
        self.model_name = model_name
        self.weights_path = os.path.join(model_name, "weights_latest.pt")
        self.losses_path = os.path.join(model_name, "losses.npy")
        if not os.path.exists(model_name):
            os.makedirs(model_name, exist_ok=True)
            self.losses = np.array([])
        elif os.path.exists(self.weights_path) and os.path.exists(self.losses_path):
            self.load()
        else:
            self.losses = np.array([])

        self.to(device)

    def scene_embed(self, obs, mask, drop=None):
        """Encode scenes; rows where drop=True use the learned null embedding (CFG)."""
        emb = self.scene_encoder(obs, mask)                 # (B, time_dim)
        if drop is not None:
            d = drop.view(-1, 1).float()
            emb = (1.0 - d) * emb + d * self.null_scene.unsqueeze(0)
        return emb

    def forward(self, x, t, obs=None, mask=None, scene_emb=None, drop=None):
        """
        x         : (B, C, horizon)
        t         : (B,) timesteps
        obs/mask  : (B,S,feat)/(B,S) obstacle set  (or pass precomputed scene_emb)
        scene_emb : (B, time_dim) precomputed scene embedding (overrides obs/mask)
        drop      : (B,) bool, replace scene with null embedding (CFG dropout)
        """
        input_horizon = x.shape[2]
        time_emb = self.time_embedding(t)                   # (B, time_dim)

        if scene_emb is None:
            if obs is None:                                 # fully unconditional
                scene_emb = self.null_scene.unsqueeze(0).expand(x.shape[0], -1)
            else:
                scene_emb = self.scene_embed(obs, mask, drop)
        emb = time_emb + scene_emb                          # combined conditioning

        h_list = []
        for ds in self.down_samplers:
            x, h = ds(x, emb)
            h_list.append(h)
        x = self.middle_block(x, emb)
        for us in self.up_samplers:
            h_temp = h_list.pop()
            x = us(x, h_temp, emb)
            target = h_list[-1].shape[2] if h_list else input_horizon
            x = self._match_horizon(x, target)
        return self.final_conv(x)

    @staticmethod
    def _match_horizon(x, target):
        h = x.shape[2]
        if h == target:
            return x
        if h > target:
            return x[:, :, :target]
        return F.pad(x, (0, target - h))

    def save(self):
        torch.save(self.state_dict(), self.weights_path)
        np.save(self.losses_path, self.losses)

    def save_checkpoint(self, checkpoint):
        torch.save(self.state_dict(), self.model_name + "/weights_" + str(checkpoint) + ".pt")
        np.save(self.model_name + "/latest_checkpoint.npy", checkpoint)

    def load(self):
        self.losses = np.load(self.losses_path)
        self.load_state_dict(torch.load(self.weights_path))
        print("Loaded conditional model at " + str(self.losses.size) + " epochs")
