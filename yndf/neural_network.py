"""Defines neural network components for Nethack agent (updated to use vector_fields + search_scores)."""

from typing import Dict
import gymnasium as gym
import torch
from torch import nn
import torch.nn.functional as F

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class NethackFeaturesExtractor(BaseFeaturesExtractor):
    """Extract features from Nethack observations for the agent."""
    def __init__(self,
                 observation_space: gym.spaces.Dict,
                 num_glyphs: int = 6000,
                 glyph_embed_dim: int = 32,
                 hidden_layers: int = 1,
                 hidden_dim: int = 256,
                 local_view: int = 7,
                 local_mlp_hidden: int = 128,
                 vf_hidden: int = 64,       # small head for vector_fields
                 pad_token: int = 0,        # used when cropping off-map
                 zero_center: bool = True   # zero out the agent's center tile in local branch
                 ):
        super().__init__(observation_space, features_dim=hidden_dim)

        # --- Shapes from obs space ---
        assert isinstance(observation_space, gym.spaces.Dict)
        self.local_view = local_view
        self.local_sz = local_view * local_view
        self.zero_center = zero_center
        self.pad_token = pad_token

        n_vector_fields = int(observation_space["vector_fields"].shape[0])

        # --- Global map path (glyphs + visited) ---
        self.embedding = nn.Embedding(num_glyphs, glyph_embed_dim)

        in_channels = glyph_embed_dim + 1  # +1 channel for visited_mask
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 21×79 → 11×40
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 11×40 → 6×20
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 21, 79)
            cnn_out = self.cnn(dummy).view(1, -1).size(1)

        # --- Local 7x7 branch: glyph-embedding patch + 7x7 search_scores ---
        # Input dim = glyph_embed_dim * V*V  +  V*V (search scores flattened)
        local_in = glyph_embed_dim * self.local_sz + self.local_sz
        self.local_mlp = nn.Sequential(
            nn.Linear(local_in, local_mlp_hidden),
            nn.ReLU(inplace=True),
        )

        # --- Vector fields head ---
        self.vf_head = nn.Sequential(
            nn.Linear(n_vector_fields, vf_hidden),
            nn.ReLU(inplace=True),
        )

        # --- Final fusion ---
        # +2 for normalized agent_yx, +8 for wavefront, +local_mlp_hidden, +vf_hidden
        fusion_in = cnn_out + 2 + 8 + local_mlp_hidden + vf_hidden
        layers = [nn.Linear(fusion_in, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        self.fc = nn.Sequential(*layers)

    # --------- Helpers ---------

    def _crop_local_ids(self, glyphs: torch.Tensor, agent_yx_int: torch.Tensor) -> torch.Tensor:
        """Return a (B, V, V) tensor of glyph ids centered on the agent (with padding)."""
        r = self.local_view // 2
        padded = F.pad(glyphs, (r, r, r, r), value=self.pad_token)  # (B, H+2r, W+2r)

        patches = []
        for b in range(glyphs.shape[0]):
            y, x = agent_yx_int[b].tolist()
            y += r
            x += r
            patches.append(padded[b, y - r:y + r + 1, x - r:x + r + 1])
        return torch.stack(patches, dim=0)  # (B, V, V)

    # --------- Forward ---------

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass to extract features from the observation."""
        # Global map features (glyphs + visited)
        glyphs = obs["glyphs"].long()                # (B, 21, 79)
        seen = obs["visited_mask"].float().unsqueeze(1)
        g_emb = self.embedding(glyphs).permute(0, 3, 1, 2)  # (B, D, 21, 79)
        x = torch.cat([g_emb, seen], dim=1)
        x = self.cnn(x).flatten(1)

        # Agent extras
        agent_yx_int = obs["agent_yx"].long()
        agent_yx = obs["agent_yx"].float()
        agent_yx[:, 0] /= 20.0   # normalize to [0,1]
        agent_yx[:, 1] /= 78.0

        wavefront = obs["wavefront"].float()         # (B, 8), values in {-1,0,1} -> float

        # Local patch: glyph embeddings + search_scores (already 7x7 from obs)
        local_ids = self._crop_local_ids(glyphs, agent_yx_int)   # (B, V, V)
        local_emb = self.embedding(local_ids)                    # (B, V, V, D)
        if self.zero_center:
            c = self.local_view // 2
            local_emb[:, c, c, :] = 0.0

        local_emb_flat = local_emb.reshape(local_emb.size(0), -1)          # (B, V*V*D)

        # search_scores are provided by the env as a (B, 7, 7) float in [0,1]
        search_scores = obs["search_scores"].float().reshape(glyphs.size(0), -1)  # (B, V*V)

        local_cat = torch.cat([local_emb_flat, search_scores], dim=1)
        local_feat = self.local_mlp(local_cat)                              # (B, local_mlp_hidden)

        # Vector fields head
        vf = obs["vector_fields"].float()                                   # (B, N_fields)
        vf_feat = self.vf_head(vf)                                          # (B, vf_hidden)

        # Fuse
        feats = torch.cat([x, agent_yx, wavefront, local_feat, vf_feat], dim=1)
        return self.fc(feats)


class NethackMaskablePolicy(MaskableActorCriticPolicy):
    """Custom policy for Nethack that uses the NethackFeaturesExtractor."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("features_extractor_class", NethackFeaturesExtractor)
        kwargs.setdefault("net_arch", [])  # linear heads by default
        super().__init__(*args, **kwargs)
