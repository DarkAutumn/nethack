"""Action wrapper for Nethack environments to mask out certain actions."""

import gymnasium as gym
import numpy as np
from nle import nethack

from yndf.nethack_state import NethackState

from yndf.movement import DIRECTION_MAP, DIRECTION_TO_ACTION, DIRECTIONS, can_move, CLOSED_DOORS, manhattan_distance

class UserInputAction:
    """A class to represent user input actions."""
    def __init__(self, action: int):
        self.action = action
        self.chr = chr(action)

    def __repr__(self):
        return f"UserInputAction(action={self.action}, chr='{self.chr}')"

class NethackActionWrapper(gym.Wrapper):
    """Convert NLE observation → dict(glyphs, visited_mask, agent_yx)."""

    def __init__(self, env : gym.Env, actions) -> None:
        super().__init__(env)

        self.action_space = gym.spaces.Discrete(len(actions))

        self._state: NethackState = None
        self._action_directions = {}
        for direction, (dy, dx) in DIRECTION_MAP.items():
            if direction not in actions:
                continue
            self._action_directions[actions.index(direction)] = (dy, dx)

        self.kick_index = self._get_index_or_none(nethack.Command.KICK, actions)
        self.search_index = self._get_index_or_none(nethack.Command.SEARCH, actions)
        self._descend_index = self._get_index_or_none(nethack.MiscDirection.DOWN, actions)

        self._unwrapped_actions = self.unwrapped.actions
        self.model_actions = actions
        self._action_map = {}
        for i, action in enumerate(actions):
            self._action_map[i] = self._unwrapped_actions.index(action)

        self._disallowed_moves = []

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        self._state: NethackState = info["state"]
        info["action_mask"] = self.action_masks()
        return obs, info

    def step(self, action):  # type: ignore[override]
        action = self._translate_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._state: NethackState = info["state"]

        info["action_mask"] = self.action_masks()
        return obs, reward, terminated, truncated, info

    def _get_index_or_none(self, action, actions):
        if action in actions:
            return actions.index(action)
        return None

    def _translate_action(self, action):
        if isinstance(action, UserInputAction):
            #if action.action not in
            unwrapped_actions = self.unwrapped.actions
            if action.action not in unwrapped_actions:
                print(f"Invalid action: {action}. Must be one of {unwrapped_actions}.")
                return None

            action = unwrapped_actions.index(action.action)
        else:
            action = self._action_map.get(int(action), action)

        return action

    def action_masks(self):
        """Return the action mask for the current state."""
        mask = np.ones(self.action_space.n, dtype=bool)
        if self._descend_index is not None:
            mask[self._descend_index] = self._state.is_player_on_exit

        # Apply movement direction masks
        boulders = [boulder
                    for boulder in self._state.stuck_boulders
                    if boulder.player_position == self._state.player.position]

        for index, (dy, dx) in self._action_directions.items():
            ny, nx = self._state.player.position[0] + dy, self._state.player.position[1] + dx
            if not can_move(self._state.floor_glyphs, self._state.player.position, (ny, nx)):
                mask[index] = False
            elif (ny, nx) in self._state.locked_doors:
                assert index != self.kick_index
                mask[index] = False
            elif boulders and any(boulder.boulder_position == (ny, nx) for boulder in boulders):
                # If the player is trying to move a stuck boulder, we need to disallow that action
                mask[index] = False

        if self.kick_index is not None:
            # Check if the player can kick
            can_kick = any(True
                           for pos in self._state.locked_doors
                           if manhattan_distance(pos, self._state.player.position) == 1)

            mask[self.kick_index] = can_kick

        if self._disallowed_moves:
            self._disallowed_moves = [dm for dm in self._disallowed_moves if dm.is_still_in_effect(self._state)]

            for disallowed in self._disallowed_moves:
                if disallowed.does_apply(self._state):
                    # If the player is trying to move a boulder, we need to disallow that action
                    index = self.model_actions.index(disallowed.action)
                    mask[index] = False

        return mask

    def translate_to_keycode(self, action: int) -> str:
        """Translate an action index to a keypress character."""
        if isinstance(action, UserInputAction):
            return action.action

        action = self._unwrapped_actions[action]
        return action.value

    def get_valid_kick_actions(self, state : NethackState) -> bool:
        """Check if the given position is adjacent to a closed door."""
        glyphs = state.floor_glyphs
        y, x = state.player.position
        result = []
        for dy, dx in DIRECTIONS:
            ny, nx = y + dy, x + dx
            if (0 <= ny < glyphs.shape[0] and 0 <= nx < glyphs.shape[1]):
                if glyphs[ny, nx] in CLOSED_DOORS:
                    direction = DIRECTION_TO_ACTION[(dy, dx)]
                    action = self.model_actions.index(direction)
                    result.append(action)

        return result
