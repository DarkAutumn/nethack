"""A wrapper to expose NethackState object as info['state']."""

import gymnasium as gym
from yndf.wrapper_actions import COORDINATE_MAP
from yndf.nethack_state import NethackState

class NethackStateWrapper(gym.Wrapper):
    """Wraps the NLE environment to maintain the current state of the game."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

        self._current_state: NethackState | None = None

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        character = self.unwrapped.character
        self._current_state = NethackState(obs, info, character, None)
        info['state'] = self._current_state
        return obs, info

    def step(self, action):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)

        how_died = self.unwrapped.nethack.how_done().name.lower() if info['end_status'] == 1 else None
        character = self.unwrapped.character
        self._current_state = NethackState(obs, info, how_died, character, self._current_state)
        info['state'] = self._current_state

        if self._current_state.message == "This door is locked.":
            pos = self._get_target_position(action)
            if pos is not None:
                self._current_state.add_locked_door(pos)

        if self._current_state.message == "You try to move the boulder, but in vain." or \
               "Perhaps that's why you cannot move it." in self._current_state.message:
            # If the player tried to move a boulder and can't, we need to disallow that action
            pos = self._get_target_position(action)
            if pos is not None:
                self._current_state.add_stuck_boulder(self._current_state.player.position, pos)

        return obs, reward, terminated, truncated, info

    def _get_target_position(self, action):
        actions = self.env.unwrapped.actions
        direction = COORDINATE_MAP.get(actions[action], None)
        if direction is None:
            return None

        pos = (self._current_state.player.position[0] + direction[0],
                   self._current_state.player.position[1] + direction[1])

        return pos
