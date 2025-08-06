# nethack_nav_training_with_masker.py
# pylint: disable=invalid-name,missing-docstring
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from nle import nethack

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from message_counter import MessageCounter

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

VISITED_REWARD = 0.01  # reward for visiting a new cell

# ───────────────────────────────────────────────────────────────────────────────
# 1.  Gym wrapper that: (a) keeps seen mask, (b) extracts (x,y) coords,
#     (c) narrows the action space to 8 moves + '>'
# ───────────────────────────────────────────────────────────────────────────────
MOVE_ACTIONS = tuple(nethack.CompassDirection)          # 8 directions
DESCEND_ACTION = (nethack.MiscDirection.DOWN,)            # '>'
ACTIONS = MOVE_ACTIONS + DESCEND_ACTION                  # total 9 actions


class NetHackObsWrapper(gym.ObservationWrapper):
    """Convert NLE observation → dict(glyphs, visited_mask, agent_yx)."""

    def __init__(self, env: gym.Env, stone_glyph: int | None = None) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Dict(
            glyphs=self.env.observation_space["glyphs"],
            visited_mask=gym.spaces.Box(0, 1, shape=(21, 79), dtype=np.uint8),
            agent_yx=gym.spaces.Box(
                low=np.array([0, 0]), high=np.array([78, 20]), dtype=np.int16
            ),
        )

        self._descend_only = np.zeros(len(ACTIONS), dtype=bool)
        self._descend_only[ACTIONS.index(nethack.MiscDirection.DOWN)] = True
        self._all_but_descend = np.ones(len(ACTIONS), dtype=bool)
        self._all_but_descend[ACTIONS.index(nethack.MiscDirection.DOWN)] = False

        # override action_space to 9 discrete actions
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self._visited_mask: np.ndarray = np.zeros((21, 79), dtype=np.uint8)
        self._exit: tuple[int, int] | None = None  # (y, x) coords of the exit
        self._is_on_exit = False
        self._message_counter : MessageCounter = None

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        info.update(obs)

        self._visited_mask.fill(0)
        self._exit = None
        self._is_on_exit = False

        wrapped_obs = self.observation(obs)
        info["action_mask"] = self.action_masks()
        self._message_counter : MessageCounter = MessageCounter()
        return wrapped_obs, info

    def step(self, action):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)

        message = obs['message'].tobytes().decode('utf-8').rstrip('\x00')
        if message:
            self._message_counter.add(message)

        if self._exit is None:
            index = np.where(obs['chars'] == ord('>'))
            if index is not None and len(index[0]) > 0 and len(index[1]) > 0:
                self._exit = index[0][0], index[1][0]

        if self._exit is not None:
            agent_yx = self._get_yx(obs)
            self._is_on_exit = agent_yx[0] == self._exit[0] and agent_yx[1] == self._exit[1]

            info['exit'] = self._exit
            info['is_on_exit'] = self._is_on_exit

        info.update(obs)
        wrapped_obs = self.observation(obs)
        info["action_mask"] = self.action_masks()

        return wrapped_obs, reward, terminated, truncated, info

    def observation(self, observation):
        glyphs = observation["glyphs"]
        agent_yx = self._get_yx(observation)
        self._visited_mask[agent_yx] = 1

        return {
            "glyphs": glyphs.astype(np.int32),
            "visited_mask": self._visited_mask.copy(),
            "agent_yx": agent_yx,
        }

    def _get_yx(self, observation):
        # blstats[0], blstats[1] are x,y coords
        blstats = observation["blstats"]
        return blstats[1], blstats[0]

    def action_masks(self):
        return self._descend_only if self._is_on_exit else self._all_but_descend

class NetHackRewardWrapper(gym.Wrapper):
    """Convert NLE reward to a more useful form."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._last_visited = 0
        self._steps_since_new = 0

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        self._last_visited = obs["visited_mask"].sum()
        self._steps_since_new = 0

        return obs, info

    def step(self, action):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)

        reward = 0.0

        # did we visit a new cell?
        visited = obs["visited_mask"].sum()
        if visited > self._last_visited:
            reward += VISITED_REWARD
            self._steps_since_new = 0
        else:
            self._steps_since_new += 1
            truncated |= self._steps_since_new > 100

        self._last_visited = visited

        # did we reach the exit?
        if info.get('is_on_exit', False):
            reward += 0.5

        return obs, reward, terminated, truncated, info

# ───────────────────────────────────────────────────────────────────────────────
# 2.  Torch extractor (minus FoV channel)
# ───────────────────────────────────────────────────────────────────────────────
class NavFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        num_glyphs: int = 6000,
        glyph_embed_dim: int = 32,
        hidden_dim: int = 256,
    ):
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

        self.fc = nn.Sequential(
            nn.Linear(cnn_out + 2, hidden_dim),  # +2 for agent_yx
            nn.ReLU(inplace=True),
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore[override]
        glyphs = obs["glyphs"].long()
        seen = obs["visited_mask"].float().unsqueeze(1)

        g_emb = self.embedding(glyphs).permute(0, 3, 1, 2)
        x = torch.cat([g_emb, seen], dim=1)

        x = self.cnn(x).flatten(1)

        agent_yx = obs["agent_yx"].float()
        agent_yx[:, 0 ] /= 20.0
        agent_yx[:, 1] /= 78.0

        return self.fc(torch.cat([x, agent_yx], dim=1))


# ───────────────────────────────────────────────────────────────────────────────
# 3.  Custom policy that uses the extractor; no extra MLP layers added
# ───────────────────────────────────────────────────────────────────────────────
class NavMaskablePolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("features_extractor_class", NavFeaturesExtractor)
        kwargs.setdefault("net_arch", [])
        super().__init__(*args, **kwargs)


# ───────────────────────────────────────────────────────────────────────────────
# 4.  Make environment, wrap, train with ActionMasker
# ───────────────────────────────────────────────────────────────────────────────


def make_env(rank: int):
    """
    Utility for multiproc env creation.
    Each worker gets its own random seed.
    """
    def _init():
        env = gym.make("NetHackScore-v0", actions=ACTIONS)
        env = NetHackObsWrapper(env)
        env = NetHackRewardWrapper(env)
        # IMPORTANT: your wrappers/env must implement `action_masks()`
        return env
    return _init

def main(total_timesteps: int = 1_000_000):
    num_cpu = 8
    envs = SubprocVecEnv([make_env(i) for i in range(num_cpu)], start_method="fork")
    envs = VecMonitor(envs)

    model = MaskablePPO(
        policy=NavMaskablePolicy,
        env=envs,
        verbose=1,
        batch_size=1024,
        n_steps=4096 // num_cpu,
        tensorboard_log="./nle_nav_tb",
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save("ppo_nethack_nav")

    print("Training finished and model saved.")


if __name__ == "__main__":
    main()
