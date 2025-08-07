"""A wrapper that calculates rewards for nethack gameplay."""

from enum import Enum
import gymnasium as gym
from yndf.nethack_state import NethackState
class Rewards(Enum):
    """Enum for different types of rewards for the agent."""
    VISITED = 0.1
    WALL_HIT = -0.01
    EXIT_REACHED = 0.5
    DEATH = -5.0

class Endings(Enum):
    """Enum for different types of endings."""
    DEATH = 1
    NO_DISCOVERY = 2

class NethackRewardWrapper(gym.Wrapper):
    """Convert NLE reward to a more useful form."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._last_visited = 0
        self._steps_since_new = 0

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        state : NethackState = info["state"]
        self._last_visited = state.visited.sum()
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
            # did we visit a new cell?
            visited = state.visited.sum()
            if visited > self._last_visited:
                reward_list.append(Rewards.VISITED)
                self._steps_since_new = 0
            else:
                self._steps_since_new += 1
                if self._steps_since_new > 100:
                    truncated = self._steps_since_new > 100
                    info['ending'] = Endings.NO_DISCOVERY

            self._last_visited = visited

            # did we run into a wall?
            if info.get("message", "") in ("It's solid stone.", "It's a wall."):
                reward_list.append(Rewards.WALL_HIT)
                info['action_mask'][action] = False  # disable this action

            # did we reach the exit?
            if info.get('is_on_exit', False):
                reward_list.append(Rewards.EXIT_REACHED)

        reward = 0.0
        if  reward_list:
            details = info["rewards"] = {}
            for r in reward_list:
                reward += r.value
                details[r.name] = details.get(r.name, 0.0) + r.value

        return obs, reward, terminated, truncated, info
