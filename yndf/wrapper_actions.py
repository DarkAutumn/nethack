"""Action wrapper for Nethack environments to mask out certain actions."""

import gymnasium as gym
import numpy as np
from nle import nethack

from yndf.nethack_level import GLYPH_TABLE
from yndf.nethack_state import NethackState
from yndf.nethack_level import DungeonLevel

COORDINATE_MAP = {
    nethack.CompassDirection.NW : (-1, -1),
    nethack.CompassDirection.N : (-1, 0),
    nethack.CompassDirection.NE : (-1, 1),
    nethack.CompassDirection.W : (0, -1),
    nethack.CompassDirection.E : (0, 1),
    nethack.CompassDirection.SW : (1, -1),
    nethack.CompassDirection.S : (1, 0),
    nethack.CompassDirection.SE : (1, 1),
}

COORDINATE_TO_ACTION = {
    (-1, -1): nethack.CompassDirection.NW,
    (-1, 0): nethack.CompassDirection.N,
    (-1, 1): nethack.CompassDirection.NE,
    (0, -1): nethack.CompassDirection.W,
    (0, 1): nethack.CompassDirection.E,
    (1, -1): nethack.CompassDirection.SW,
    (1, 0): nethack.CompassDirection.S,
    (1, 1): nethack.CompassDirection.SE,
}

DIRECTIONS = tuple(nethack.CompassDirection) + (nethack.MiscDirection.WAIT,)
DIRECTION_TO_INDEX = {dir: i for i, dir in enumerate(DIRECTIONS)}
VERBS = (nethack.Command.MOVE, nethack.Command.KICK, nethack.Command.SEARCH, nethack.MiscDirection.DOWN)
VERB_TO_INDEX = {verb: i for i, verb in enumerate(VERBS)}

SEARCH_COUNT = 22

class UserInputAction:
    """A class to represent user input actions."""
    def __init__(self, action: int):
        self.action = action
        self.chr = chr(action)

    def __repr__(self):
        return f"UserInputAction(action={self.action}, chr='{self.chr}')"

class NethackActionWrapper(gym.Wrapper):
    """Convert NLE observation → dict(glyphs, visited_mask, agent_yx)."""

    def __init__(self, env : gym.Env) -> None:
        super().__init__(env)

        self.action_space = gym.spaces.MultiDiscrete([len(VERBS), len(DIRECTIONS)])
        self._state: NethackState = None
        self._action_to_index = {}

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        self._state: NethackState = info["state"]
        return obs, info

    def step(self, action):  # type: ignore[override]
        verb, direction, index = self._action_to_verb(action)

        # search is always an 22 turn search
        search_count = self._state.floor.search_count[self._state.player.position]
        if verb == nethack.Command.SEARCH and search_count < SEARCH_COUNT:
            remaining_search = SEARCH_COUNT - search_count
            if remaining_search > 1:
                remaining_search = min(remaining_search, SEARCH_COUNT)

                self.unwrapped.nethack.step(ord('n'))
                for c in str(remaining_search):
                    self.unwrapped.nethack.step(ord(c))

        elif verb == nethack.Command.KICK:
            self.unwrapped.nethack.step(nethack.Command.KICK.value)
            index = self._get_env_index(direction)

        obs, reward, terminated, truncated, info = self.env.step(index)
        state = info["state"]

        if verb == nethack.Command.SEARCH:
            state.floor.search_count[self._state.player.position] += state.time - self._state.time

        self._state: NethackState = state
        return obs, reward, terminated, truncated, info

    def _action_to_verb(self, action):
        """Convert action from the model/user into a verb, direction, and environment index."""
        if isinstance(action, UserInputAction):
            verb = None
            direction = None

            if action.chr == 's':
                verb = nethack.Command.SEARCH
            elif action.chr == '>':
                verb = nethack.MiscDirection.DOWN
            elif action.action == 4:
                verb = nethack.Command.KICK
            elif action.action in DIRECTIONS:
                verb = nethack.Command.MOVE
                if action == nethack.MiscDirection.WAIT:
                    direction = action
                else:
                    direction = nethack.CompassDirection(action.action)

            index = self.unwrapped.actions.index(action.action)

        else:
            verb = VERBS[action[0]]
            direction = DIRECTIONS[action[1]]

            verb_or_direction = verb if verb != nethack.Command.MOVE else direction

            index = self._get_env_index(verb_or_direction)

        return verb, direction, index

    def _get_env_index(self, verb_or_direction):
        if (index := self._action_to_index.get(verb_or_direction)) is None:
            index = self.unwrapped.actions.index(verb_or_direction)
            self._action_to_index[verb_or_direction] = index
        return index

    def action_masks(self):
        """Return the action mask for the current state."""

        verb_mask = np.zeros((len(VERBS),), dtype=bool)
        direction_mask = np.zeros((len(VERBS), len(DIRECTIONS)), dtype=bool)

        move_index = VERB_TO_INDEX[nethack.Command.MOVE]
        move_mask = self._get_move_mask()
        direction_mask[move_index] = move_mask
        verb_mask[move_index] = move_mask.any()

        kick_index = VERB_TO_INDEX[nethack.Command.KICK]
        kick_mask = self._get_kick_mask()
        direction_mask[kick_index] = kick_mask
        verb_mask[kick_index] = kick_mask.any()

        verb_mask[VERB_TO_INDEX[nethack.Command.SEARCH]] = self._can_search(self._state)
        verb_mask[VERB_TO_INDEX[nethack.MiscDirection.DOWN]] = self._state.floor.exits[self._state.player.position]

        return verb_mask, direction_mask

    def _get_kick_mask(self) -> np.ndarray:
        """Return the kick mask for the current state."""
        # don't allow kicking pets, walls, or open air
        floor = self._state.floor
        can_kick = floor.objects | floor.enemies | floor.corpses | floor.closed_doors

        kick_mask = np.zeros((len(DIRECTIONS),), dtype=bool)
        for index, direction in enumerate(DIRECTIONS):
            # cannot kick yourself
            if direction == nethack.MiscDirection.WAIT:
                kick_mask[index] = False

            else:
                dy, dx = COORDINATE_MAP[direction]
                y, x = self._state.player.position[0] + dy, self._state.player.position[1] + dx

                if not (0 <= y < floor.properties.shape[0] and 0 <= x < floor.properties.shape[1]):
                    kick_mask[index] = False

                else:
                    kick_mask[index] = can_kick[y, x]

        return kick_mask

    def _get_move_mask(self) -> np.ndarray:
        """Return the movement mask for the current state."""
        direction_mask = np.zeros((len(DIRECTIONS),), dtype=bool)
        for index, direction in enumerate(DIRECTIONS):
            # For now, we disable waiting to compare to previous models
            if direction == nethack.MiscDirection.WAIT:
                direction_mask[index] = False

            else:
                y, x = self._state.player.position
                dy, dx = COORDINATE_MAP[direction]
                ny, nx = y + dy, x + dx

                direction_mask[index] = self._can_move(y, x, ny, nx, dy, dx)

        return direction_mask

    def _can_move(self, y, x, ny, nx, dy, dx) -> bool:
        """Check if the player can move from one position to another."""
        floor = self._state.floor
        open_door = floor.open_doors

        # in-bounds?
        if not (0 <= ny < open_door.shape[0] and 0 <= nx < open_door.shape[1]):
            return False

        # Block diagonal through an open door (at either endpoint)
        if (dy != 0 and dx != 0 and (open_door[y, x] or open_door[ny, nx])):
            return False

        # Allow movement into closed doors that aren't locked.
        if floor.closed_doors[ny, nx]:
            return not floor.locked_doors[ny, nx]

        # Otherwise, check if the tile is passable.
        return floor.passable[ny, nx]

    def _can_search(self, state: NethackState) -> bool:
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

        return floor.num_enemies == 0
