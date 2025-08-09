"""Action wrapper for Nethack environments to mask out certain actions."""

import gymnasium as gym
import numpy as np
from nle import nethack

from yndf.nethack_state import NethackState

from yndf.movement import DIRECTION_MAP, can_move, adjacent_to, CLOSED_DOORS

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
        self._action_directions = {}
        for direction, (dy, dx) in DIRECTION_MAP.items():
            if direction not in actions:
                continue
            self._action_directions[actions.index(direction)] = (dy, dx)

        self._kick_index = actions.index(nethack.Command.KICK) if nethack.Command.KICK in actions else None

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
        mask = self._descend_only.copy() if self._state.is_player_on_exit else self._all_but_descend.copy()

        # Apply movement direction masks
        for index, (dy, dx) in self._action_directions.items():
            ny, nx = self._state.player.position[0] + dy, self._state.player.position[1] + dx
            if not can_move(self._state.floor_glyphs, self._state.player.position, (ny, nx)):
                mask[index] = False
            elif (ny, nx) in self._state.locked_doors:
                assert index != self._kick_index
                mask[index] = False

        if self._kick_index is not None:
            # Check if the player can kick
            mask[self._kick_index] = adjacent_to(self._state.floor_glyphs, *self._state.player.position, CLOSED_DOORS)

        return mask
