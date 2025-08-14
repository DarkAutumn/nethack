"""A wrapper that calculates rewards for nethack gameplay."""

import gymnasium as gym
from nle import nethack
import numpy as np
from yndf.endings import  MaxTimestepsReached, NoForwardPathWithoutSearching
from yndf.nethack_level import GLYPH_TABLE
from yndf.nethack_state import NethackState

class Reward:
    """A simple class to represent a reward with a name and value."""
    def __init__(self, name : str, value : float, max_value : float = None):
        self.name = name
        self.value = value
        self.max_value = max_value

    def __mul__(self, other: float) -> 'Reward':
        if other == 1:
            return self

        new_value = self.value * other
        if self.max_value is not None:
            new_value = min(new_value, self.max_value)
        return Reward(self.name, new_value, self.max_value)

class Rewards:
    """Enum for different types of rewards for the agent."""
    STEP = Reward("step", -0.0015)
    HURT = Reward("took-damage", -0.05)
    KILL = Reward("kill-enemy", 0.5)
    DESCENDED = Reward("descended", 1.0)
    DIED = Reward("died", -5.0)
    STARVED = Reward("starved", 1.0)   # until we can eat, this is a reward for making it this long
    LEVEL_UP = Reward("level-up", 0.5)
    GOLD = Reward("gold", 0.05)
    SCORE = Reward("score", 0.01) # any increase in score not rewarded by something else
    REVEALED_TILE = Reward("revealed-tile", 0.01, max_value=0.05)
    REACHED_FRONTIER = Reward("reached-frontier", 0.05)
    SUCCESS = Reward("success", 1.0)  # mini-scenario completed
    SEARCH_SUCCESS = Reward("search-success", 0.01, max_value=0.05)
    SEARCHED_GOOD_SPOT = Reward("searched-good-spot", 0.02)
    WASTED_SEARCH = Reward("wasted-search", -0.05)

DEATH_PENALTIES = {
    "died": Rewards.DIED,
    "starved": Rewards.STARVED,
}

class NethackRewardWrapper(gym.Wrapper):
    """Convert NLE reward to a more useful form."""
    def __init__(self, env: gym.Env, has_search: bool = False) -> None:
        super().__init__(env)
        self._prev : NethackState = None
        self._has_search = has_search

        self.endings = [MaxTimestepsReached()]
        if not has_search:
            self.endings.append(NoForwardPathWithoutSearching())

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        self._prev = info["state"]

        for ending in self.endings:
            ending.reset(self._prev)

        return obs, info

    def step(self, action):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        action_is_search = action == self.unwrapped.actions.index(nethack.Command.SEARCH)

        state: NethackState = info["state"]
        time_passed = state.time - self._prev.time

        reward_list = [Rewards.STEP * time_passed]

        self._check_state_changes(reward_list, self._prev, state)
        self._check_revealed_tiles(reward_list, self._prev, state, action_is_search)
        terminated, truncated, reason = self._check_endings(state)
        if terminated or truncated:
            assert reason is not None, "Ending condition should have a reason."
            if (death_penalty := DEATH_PENALTIES.get(reason, None)) is not None:
                reward_list.append(death_penalty)

            info['ending'] = reason

        reward = 0.0
        details = info["rewards"] = {}
        for r in reward_list:
            reward += r.value
            details[r.name] = details.get(r.name, 0.0) + r.value

        self._prev = state
        return obs, reward, terminated, truncated, info

    def _check_revealed_tiles(self, reward_list, prev : NethackState, state : NethackState, action_is_search: bool):
        """Check if any new tiles were revealed."""
        revealed = self._prev.floor.stone_tile_count - state.floor.stone_tile_count
        if revealed > 0:
            reward_list.append(Rewards.REVEALED_TILE * revealed)
        else:
            # give a larger reward for grabbing items off of the floor, which is effectively what
            # this is checking
            prev_visited = (prev.floor.properties & prev.floor.VISITED) != 0
            if prev.floor.wavefront[state.player.position] == 0 and not prev_visited[state.player.position]:
                reward_list.append(Rewards.REACHED_FRONTIER)

        if action_is_search:
            prev_floor = self._prev.floor
            possible = (prev_floor.properties & GLYPH_TABLE.STONE) != 0
            possible |= (prev_floor.properties & GLYPH_TABLE.WALL) != 0

            actual = (state.floor.properties & GLYPH_TABLE.STONE) != 0
            actual |= (state.floor.properties & GLYPH_TABLE.WALL) != 0

            revealed = np.sum(~actual & possible)
            if revealed > 0:
                value = Rewards.SEARCH_SUCCESS.value + min(Rewards.REVEALED_TILE.value * revealed, 0.2)
                reward_list.append(Reward(Rewards.SEARCH_SUCCESS.name, value))

            elif state.floor.search_score[state.player.position] > 0.6:
                reward_list.append(Rewards.SEARCHED_GOOD_SPOT)

            elif state.floor.search_score[state.player.position] < 0.3:
                reward_list.append(Rewards.WASTED_SEARCH)

    def _check_state_changes(self, reward_list, prev : NethackState, state : NethackState):
        if prev.player.depth < state.player.depth:
            reward_list.append(Rewards.DESCENDED)

        if prev.player.hp > state.player.hp:
            reward_list.append(Rewards.HURT)

        if prev.player.exp < state.player.exp:
            reward_list.append(Rewards.KILL)

        if prev.player.level < state.player.level:
            reward_list.append(Rewards.LEVEL_UP)

        if prev.player.gold < state.player.gold:
            reward_list.append(Rewards.GOLD)

            # only reward score if we missed to rewarding something that increases it
        if prev.player.score < state.player.score and not reward_list:
            reward_list.append(Rewards.SCORE)

    def _check_endings(self, state : NethackState):
        """Check if any ending conditions have been met.  Returns (terminated, truncated, reason)."""
        # No way to continue after a game over, so it's not an Ending condition
        if state.game_over:
            return True, False, state.how_died

        for ending in self.endings:
            if ending.enabled:
                ending.step(state)
                if ending.terminated or ending.truncated:
                    return ending.terminated, ending.truncated, ending.name

        return False, False, None

    def merge_reward_info(self, info: dict, new_info: dict):
        """Merge new reward information into the existing info dictionary."""
        if "rewards" not in info:
            info["rewards"] = {}
        for key, value in new_info.get("rewards", {}).items():
            if key in info["rewards"]:
                info["rewards"][key] += value
            else:
                info["rewards"][key] = value

        if "ending" in new_info:
            info["ending"] = new_info["ending"]
