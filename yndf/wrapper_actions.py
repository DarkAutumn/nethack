"""Action wrapper for Nethack environments to mask out certain actions."""

import gymnasium as gym
import numpy as np
from nle import nethack

from yndf.nethack_state import NethackState

class NethackActionWrapper(gym.Wrapper):
    """Convert NLE observation → dict(glyphs, visited_mask, agent_yx)."""

    def __init__(self, env : gym.Env) -> None:
        super().__init__(env)
        actions = env.unwrapped.actions
        if nethack.MiscDirection.DOWN not in actions:
            self._descend_only = np.ones(len(actions), dtype=bool)
        else:
            self._descend_only = np.zeros(len(actions), dtype=bool)
            self._descend_only[actions.index(nethack.MiscDirection.DOWN)] = True

        self._all_but_descend = np.ones(len(actions), dtype=bool)
        if nethack.MiscDirection.DOWN in actions:
            self._all_but_descend[actions.index(nethack.MiscDirection.DOWN)] = False

        self._state: NethackState = None

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        self._state: NethackState = info["state"]
        info["action_mask"] = self.action_masks()
        return obs, info

    def step(self, action):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._state: NethackState = info["state"]
        info["action_mask"] = self.action_masks()
        return obs, reward, terminated, truncated, info

    def action_masks(self):
        """Return the action mask for the current state."""
        return self._descend_only.copy() if self._state.is_player_on_exit else self._all_but_descend.copy()
