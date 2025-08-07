"""A wrapper that calculates rewards for nethack gameplay."""

from enum import Enum
import gymnasium as gym
from yndf.nethack_state import NethackState

class Reward:
    """A simple class to represent a reward with a name and value."""
    def __init__(self, name : str, value : float):
        self.name = name
        self.value = value

class Rewards:
    """Enum for different types of rewards for the agent."""
    HURT = Reward("took-damage", -0.05)
    KILL = Reward("kill-enemy", 0.05)
    WALL_HIT = Reward("wall-hit", -0.01)
    DESCENDED = Reward("descended", 0.5)
    SECRET = Reward("secret", 0.1)
    DEATH = Reward("death", -5.0)
    LEVEL_UP = Reward("level-up", 0.5)
    SCORE = Reward("score", 0.01)

class Endings(Enum):
    """Enum for different types of endings."""
    DEATH = 1
    NO_DISCOVERY = 2

class NethackRewardWrapper(gym.Wrapper):
    """Convert NLE reward to a more useful form."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._steps_since_new = 0
        self._prev = None

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        state : NethackState = info["state"]
        self._prev = state
        self._steps_since_new = 0

        return obs, info

    def step(self, action):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)

        reward_list = []
        state: NethackState = info["state"]

        if info['end_status'] == 1: # death
            reward_list.append(Rewards.DEATH)
            info['ending'] = Endings.DEATH
            terminated = True
            truncated = False

        else:
            # did we run into a wall?
            if info.get("message", "") in ("It's solid stone.", "It's a wall."):
                reward_list.append(Rewards.WALL_HIT)
                info['action_mask'][action] = False  # disable this action

            if self._prev.player.depth < state.player.depth:
                reward_list.append(Rewards.DESCENDED)
                self._steps_since_new = 0

            if self._prev.player.hp > state.player.hp:
                reward_list.append(Rewards.HURT)

            if self._prev.player.exp < state.player.exp:
                reward_list.append(Rewards.KILL)

            if self._prev.player.level < state.player.level:
                reward_list.append(Rewards.LEVEL_UP)

            # only reward score if we missed to rewarding something that increases it
            if self._prev.player.score < state.player.score and not reward_list:
                reward_list.append(Rewards.SCORE)

            # did we visit a new cell?
            if state.visited.sum() == self._prev.visited.sum():
                self._steps_since_new += 1
                if self._steps_since_new > 100:
                    truncated = self._steps_since_new > 100
                    info['ending'] = Endings.NO_DISCOVERY

        reward = 0.0
        if  reward_list:
            details = info["rewards"] = {}
            for r in reward_list:
                reward += r.value
                details[r.name] = details.get(r.name, 0.0) + r.value

        self._prev = state
        return obs, reward, terminated, truncated, info
