"""Wrapper to convert NLE observations to the one expected by the agent."""

import gymnasium as gym
import numpy as np

from yndf.nethack_state import NethackState

class NethackObsWrapper(gym.Wrapper):
    """Convert NLE observation → dict(glyphs, visited_mask, agent_yx)."""
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Dict(
            glyphs=self.env.observation_space["glyphs"],
            visited_mask=gym.spaces.Box(0, 1, shape=(21, 79), dtype=np.uint8),
            agent_yx=gym.spaces.Box(
                low=np.array([0, 0]), high=np.array([20, 78]), dtype=np.int16
            ),
        )

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        wrapped_obs = self._wrap_observation(obs, info)
        return wrapped_obs, info

    def step(self, action):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        wrapped_obs = self._wrap_observation(obs, info)
        return wrapped_obs, reward, terminated, truncated, info

    def _wrap_observation(self, obs, info):
        state : NethackState = info["state"]

        glyphs = obs["glyphs"].astype(np.int16)
        agent_yx = np.array(state.player.position, dtype=np.int16)
        visited = state.visited.copy()
        return {
            "glyphs": glyphs,
            "visited_mask": visited,
            "agent_yx": agent_yx,
        }
