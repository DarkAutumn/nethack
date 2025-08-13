"""Action wrapper for Nethack environments to mask out certain actions."""

import gymnasium as gym
import numpy as np
from nle import nethack

from yndf.nethack_level import GLYPH_TABLE
from yndf.nethack_state import NethackState
from yndf.nethack_level import DungeonLevel

DIRECTION_MAP = {
    nethack.CompassDirection.NW : (-1, -1),
    nethack.CompassDirection.N : (-1, 0),
    nethack.CompassDirection.NE : (-1, 1),
    nethack.CompassDirection.W : (0, -1),
    nethack.CompassDirection.E : (0, 1),
    nethack.CompassDirection.SW : (1, -1),
    nethack.CompassDirection.S : (1, 0),
    nethack.CompassDirection.SE : (1, 1),
}

DIRECTION_TO_ACTION = {
    (-1, -1): nethack.CompassDirection.NW,
    (-1, 0): nethack.CompassDirection.N,
    (-1, 1): nethack.CompassDirection.NE,
    (0, -1): nethack.CompassDirection.W,
    (0, 1): nethack.CompassDirection.E,
    (1, -1): nethack.CompassDirection.SW,
    (1, 0): nethack.CompassDirection.S,
    (1, 1): nethack.CompassDirection.SE,
}

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

        self._action_index_by_token = {tok: i for i, tok in enumerate(self.model_actions)}

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        self._state: NethackState = info["state"]
        info["action_mask"] = self.action_masks()
        return obs, info

    def step(self, action):  # type: ignore[override]
        if action == self.search_index or (isinstance(action, UserInputAction) and action.chr == 's'):
            self.unwrapped.nethack.step(ord('n'))
            self.unwrapped.nethack.step(ord('2'))
            self.unwrapped.nethack.step(ord('2'))

        elif action == self.kick_index or (isinstance(action, UserInputAction) and action.chr == 'k'):
            actions = self.get_valid_kick_actions(self._state)
            act = self.model_actions[action].value
            self.unwrapped.nethack.step(act)
            action = actions[0]

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
        h, w = self._state.floor.properties.shape
        y, x = self._state.player.position

        props      = self._state.floor.properties
        passable   = (props & GLYPH_TABLE.PASSABLE) != 0
        open_door  = (props & GLYPH_TABLE.OPEN_DOOR) != 0
        closed_door= (props & GLYPH_TABLE.CLOSED_DOOR) != 0
        locked     = (props & self._state.floor.LOCKED_DOOR) != 0

        mask = np.ones(self.action_space.n, dtype=bool)

        # Descend only if standing on an exit
        if self._descend_index is not None:
            on_exit = (props[y, x] & GLYPH_TABLE.DESCEND_LOCATION) != 0
            mask[self._descend_index] = bool(on_exit)

        # Movement actions
        for index, (dy, dx) in self._action_directions.items():
            ny, nx = y + dy, x + dx
            # in-bounds?
            if not (0 <= ny < h and 0 <= nx < w):
                mask[index] = False
                continue

            # Block diagonal through an open door (at either endpoint)
            if dy != 0 and dx != 0 and (open_door[y, x] or open_door[ny, nx]):
                mask[index] = False
                continue

            # Allow if destination is actually passable,
            # OR it's a closed door that is not locked (moving opens it)
            if passable[ny, nx] or (closed_door[ny, nx] and not locked[ny, nx]):
                mask[index] = True
            else:
                mask[index] = False

        # Kick only if there is a locked door in a cardinal neighbor
        if self.kick_index is not None:
            mask[self.kick_index] = len(self.get_valid_kick_actions(self._state)) > 0

        if self.search_index is not None:
            mask[self.search_index] = self._get_search_mask(self._state)

        return mask

    def _get_search_mask(self, state: NethackState) -> bool:
        """Check if the player can search."""
        floor = state.floor
        pos = state.player.position

        if floor.wavefront[pos] <= 2:
            return False

        if floor.search_count[pos] >= 22:
            return False

        if floor.search_score[pos] < 0.2:
            return False

        if (floor.properties[pos] & DungeonLevel.WALLS_ADJACENT) == 0:
            return False

        if ((floor.properties & GLYPH_TABLE.DESCEND_LOCATION) != 0).any():
            return False

        if floor.num_enemies > 0:
            return False

        return True

    def translate_to_keycode(self, action: int) -> str:
        """Translate an action index to a keypress character."""
        if isinstance(action, UserInputAction):
            return action.action

        action = self._unwrapped_actions[action]


        return action.value

    def get_valid_kick_actions(self, state : NethackState) -> list[int]:
        """Return action indices for directions with a locked door adjacent (8 directions)."""
        props = state.floor.properties
        h, w = props.shape
        y, x = state.player.position

        # Locked closed door mask
        locked_door = (props & state.floor.LOCKED_DOOR) != 0

        result: list[int] = []
        for dy, dx in self._action_directions.values():  # includes all 8 directions
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and locked_door[ny, nx]:
                token = DIRECTION_TO_ACTION[(dy, dx)]
                result.append(self._action_index_by_token[token])

        return result
