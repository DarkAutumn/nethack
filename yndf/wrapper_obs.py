"""Wrapper to convert NLE observations to the one expected by the agent."""

import gymnasium as gym
import numpy as np

from yndf.nethack_state import NethackState

CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
DIRECTIONS = CARDINALS + DIAGONALS

class NethackObsWrapper(gym.Wrapper):
    """Convert NLE observation → dict(glyphs, visited_mask, agent_yx)."""
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Dict(
            glyphs=self.env.observation_space["glyphs"],
            visited_mask=gym.spaces.Box(0, 1, shape=(21, 79), dtype=np.uint8),
            agent_yx=gym.spaces.Box(low=np.array([0, 0]), high=np.array([20, 78]), dtype=np.int16),
            wavefront=gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.int8),
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
        visited = (state.floor.properties & state.floor.VISITED) != 0
        return {
            "glyphs": glyphs,
            "visited_mask": visited.astype(np.uint8),
            "agent_yx": agent_yx,
            "wavefront": self._calculate_wavefront_hints(state),
        }

    def _calculate_wavefront_hints(self, state: NethackState) -> np.ndarray:
        wf = state.floor.wavefront
        y, x = state.player.position
        curr = int(wf[y, x])  # avoid unsigned underflow
        hints = np.zeros(8, dtype=np.int8)

        for i, (dy, dx) in enumerate(DIRECTIONS):
            ny, nx = y + dy, x + dx
            if 0 <= ny < wf.shape[0] and 0 <= nx < wf.shape[1]:
                v = int(wf[ny, nx])
                hints[i] = 1 if v < curr else (-1 if v > curr else 0)
        return hints
