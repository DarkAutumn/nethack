"""A wrapper to expose NethackState object as info['state']."""

import gymnasium as gym
from yndf.movement import DIRECTION_MAP
from yndf.nethack_state import NethackState

class NethackStateWrapper(gym.Wrapper):
    """Wraps the NLE environment to maintain the current state of the game."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

        self._current_state: NethackState | None = None

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        self._current_state = NethackState(obs, info)
        info['state'] = self._current_state
        return obs, info

    def step(self, action):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._current_state = NethackState(obs, info, self._current_state)
        info['state'] = self._current_state

        if self._current_state.message == "This door is locked.":
            pos = self._get_target_position(action)
            self._current_state.add_locked_door(pos)

        if self._current_state.message == "You can't move diagonally into an intact doorway.":
            pos = self._get_target_position(action)
            self._current_state.add_open_door(pos)

        if self._current_state.message == "You can't move diagonally out of an intact doorway.":
            self._current_state.add_open_door(self._current_state.player.position)

        return obs, reward, terminated, truncated, info

    def _get_target_position(self, action):
        actions = self.env.unwrapped.actions
        direction = DIRECTION_MAP[actions[action]]
        pos = (self._current_state.player.position[0] + direction[0],
                   self._current_state.player.position[1] + direction[1])

        return pos
