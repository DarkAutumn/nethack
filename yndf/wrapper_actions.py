"""Action wrapper for Nethack environments to mask out certain actions."""

import gymnasium as gym
import numpy as np
from nle import nethack

from yndf.nethack_state import NethackState

from yndf.movement import DIRECTION_MAP, can_move, adjacent_to, CLOSED_DOORS

BOULDER_GLYPH = 2353
BOULDER_MESSAGE = "You try to move the boulder, but in vain."

class UserInputAction:
    """A class to represent user input actions."""
    def __init__(self, action: int):
        self.action = action
        self.chr = chr(action)

    def __repr__(self):
        return f"UserInputAction(action={self.action}, chr='{self.chr}')"

class DisallowedMove:
    """A way to prevent a movement when certain conditions are met."""
    def __init__(self, position, action):
        self.position = position
        self.action = action

    def is_still_in_effect(self, _: NethackState) -> bool:
        """Check if the rule is in effect."""
        return True

    def does_apply(self, _: NethackState) -> bool:
        """Check if the rule applies."""
        return True

class BoulderMove(DisallowedMove):
    """A way to prevent a boulder move."""
    def __init__(self, position, action):
        super().__init__(position, action)
        dy, dx = DIRECTION_MAP[action]
        self._boulder_position = position[0] + dy, position[1] + dx

    def is_still_in_effect(self, state: NethackState) -> bool:
        """Check if the rule is in effect."""
        return state.glyphs[self._boulder_position] == BOULDER_GLYPH

    def does_apply(self, state: NethackState) -> bool:
        """Check if the rule applies."""
        return state.player.position == self.position and state.glyphs[self._boulder_position] == BOULDER_GLYPH

class NethackActionWrapper(gym.Wrapper):
    """Convert NLE observation → dict(glyphs, visited_mask, agent_yx)."""

    def __init__(self, env : gym.Env, actions) -> None:
        super().__init__(env)

        self.action_space = gym.spaces.Discrete(len(actions))

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

        if self._state.message == BOULDER_MESSAGE:
            # If the player tried to move a boulder, we need to disallow that action
            disallowed = BoulderMove(self._state.player.position, self.model_actions[action])
            self._disallowed_moves.append(disallowed)

        info["action_mask"] = self.action_masks()
        return obs, reward, terminated, truncated, info

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
