"""Defines neural network components for Nethack agent."""

from typing import Dict
import gymnasium as gym
import torch
from torch import nn

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class NethackFeaturesExtractor(BaseFeaturesExtractor):
    """Extract features from Nethack observations for the agent."""
    def __init__(self, observation_space: gym.spaces.Dict, num_glyphs: int = 6000, glyph_embed_dim: int = 32,
                 hidden_layers: int = 1, hidden_dim: int = 256):
        super().__init__(observation_space, features_dim=hidden_dim)
        self.embedding = nn.Embedding(num_glyphs, glyph_embed_dim)

        in_channels = glyph_embed_dim + 1  # +1 channel for visited_mask
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 21×79 → 11×40
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # 11×40 → 6×20
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 21, 79)
            cnn_out = self.cnn(dummy).view(1, -1).size(1)

        # +2 for agent_yx, +8 for wavefront
        layers = []
        layers.append(nn.Linear(cnn_out + 10, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        self.fc = nn.Sequential(*layers)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass to extract features from the observation."""
        glyphs = obs["glyphs"].long()
        seen = obs["visited_mask"].float().unsqueeze(1)

        g_emb = self.embedding(glyphs).permute(0, 3, 1, 2)
        x = torch.cat([g_emb, seen], dim=1)
        x = self.cnn(x).flatten(1)

        agent_yx = obs["agent_yx"].float()
        agent_yx[:, 0] /= 20.0
        agent_yx[:, 1] /= 78.0

        wavefront = obs["wavefront"].float()  # shape: (B, 8), values 0/1

        return self.fc(torch.cat([x, agent_yx, wavefront], dim=1))

class NethackMaskablePolicy(MaskableActorCriticPolicy):
    """Custom policy for Nethack that uses the NethackFeaturesExtractor."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("features_extractor_class", NethackFeaturesExtractor)
        kwargs.setdefault("net_arch", [])
        super().__init__(*args, **kwargs)
