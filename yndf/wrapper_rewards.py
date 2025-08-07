"""A wrapper that calculates rewards for nethack gameplay."""

import gymnasium as gym
from yndf.nethack_state import NethackState

VISITED_REWARD = 0.1

class NethackRewardWrapper(gym.Wrapper):
    """Convert NLE reward to a more useful form."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._last_visited = 0
        self._steps_since_new = 0

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        state : NethackState = info["state"]
        self._last_visited = state.visited.sum()
        self._steps_since_new = 0

        return obs, info

    def step(self, action):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)

        reward = 0.0
        state: NethackState = info["state"]

        # did we visit a new cell?
        visited = state.visited.sum()
        if visited > self._last_visited:
            reward += VISITED_REWARD
            self._steps_since_new = 0
        else:
            self._steps_since_new += 1
            truncated |= self._steps_since_new > 100

        self._last_visited = visited

        # did we run into a wall?
        if info.get("message", "") in ("It's solid stone.", "It's a wall."):
            reward = -0.01
            info['action_mask'][action] = False  # disable this action

        # did we reach the exit?
        if info.get('is_on_exit', False):
            reward += 0.5

        return obs, reward, terminated, truncated, info
