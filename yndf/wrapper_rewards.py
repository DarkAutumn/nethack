"""A wrapper that calculates rewards for nethack gameplay."""

from enum import Enum
import gymnasium as gym
import numpy as np
from nle import nethack
from yndf.nethack_state import NethackState
from yndf.movement import GlyphKind, SolidGlyphs

class Reward:
    """A simple class to represent a reward with a name and value."""
    def __init__(self, name : str, value : float, max_value : float = None):
        self.name = name
        self.value = value
        self.max_value = max_value

    def __mul__(self, other: float) -> 'Reward':
        new_value = self.value * other
        if self.max_value is not None:
            new_value = min(new_value, self.max_value)
        return Reward(self.name, new_value, self.max_value)

class Rewards:
    """Enum for different types of rewards for the agent."""
    HURT = Reward("took-damage", -0.05)
    KILL = Reward("kill-enemy", 0.05)
    DESCENDED = Reward("descended", 0.5)
    SECRET = Reward("secret", 0.1)
    DEATH = Reward("death", -5.0)
    LEVEL_UP = Reward("level-up", 0.5)
    GOLD = Reward("gold", 0.05)
    SCORE = Reward("score", 0.01)
    REVEALED_TILE = Reward("revealed-tile", 0.01, max_value=0.05)
    SUCCESS = Reward("success", 5.0)

class Endings(Enum):
    """Enum for different types of endings."""
    SUCCESS = 0
    DEATH = 1
    NO_DISCOVERY = 2
    NO_PATH = 3


class NethackRewardWrapper(gym.Wrapper):
    """Convert NLE reward to a more useful form."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._steps_since_new = 0
        self._prev : NethackState = None
        self._has_search = nethack.Command.SEARCH in env.unwrapped.actions

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        self._prev = info["state"]
        self._steps_since_new = 0

        return obs, info

    def step(self, action):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)

        reward_list = []
        state: NethackState = info["state"]

        terminated, truncated = self._check_endings(terminated, truncated, info, state, reward_list)
        if not terminated and not truncated:
            self._check_state_changes(reward_list, self._prev, state)
            self._check_revealed_tiles(reward_list, self._prev, state)

        reward = 0.0
        if  reward_list:
            details = info["rewards"] = {}
            for r in reward_list:
                reward += r.value
                details[r.name] = details.get(r.name, 0.0) + r.value

        self._prev = state
        return obs, reward, terminated, truncated, info

    def _check_revealed_tiles(self, reward_list, prev, state):
        """Check if any new tiles were revealed."""
        prev_stones = (prev.floor_glyphs == SolidGlyphs.S_stone.value).sum()
        new_stones = (state.floor_glyphs == SolidGlyphs.S_stone.value).sum()

        revealed = prev_stones - new_stones
        if revealed > 0:
            reward_list.append(Rewards.REVEALED_TILE * revealed)
            self._steps_since_new = 0

    def _check_state_changes(self, reward_list, prev, state):
        if prev.player.depth < state.player.depth:
            reward_list.append(Rewards.DESCENDED)
            self._steps_since_new = 0

        if prev.player.hp > state.player.hp:
            reward_list.append(Rewards.HURT)

        if prev.player.exp < state.player.exp:
            reward_list.append(Rewards.KILL)
            self._steps_since_new = 0

        if prev.player.level < state.player.level:
            reward_list.append(Rewards.LEVEL_UP)

        if prev.player.gold < state.player.gold:
            reward_list.append(Rewards.GOLD)
            self._steps_since_new = 0

            # only reward score if we missed to rewarding something that increases it
        if prev.player.score < state.player.score and not reward_list:
            reward_list.append(Rewards.SCORE)

    def _check_endings(self, terminated, truncated, info, state, reward_list):
        if info['end_status'] == 1: # death
            reward_list.append(Rewards.DEATH)
            info['ending'] = Endings.DEATH
            terminated = True
            truncated = False

        else:
            # did we visit a new cell?
            if state.visited.sum() == self._prev.visited.sum():
                self._steps_since_new += 1
                if self._steps_since_new > 100:
                    truncated = self._steps_since_new > 100
                    info['ending'] = Endings.NO_DISCOVERY

            # do we have no way to make progress?
            if not self._has_search:
                if not np.isin(state.glyph_kinds, [GlyphKind.EXIT.value, GlyphKind.FRONTIER.value]).any():
                    terminated = True
                    info['ending'] = Endings.SUCCESS
                    reward_list.append(Rewards.SUCCESS)

        return terminated, truncated
