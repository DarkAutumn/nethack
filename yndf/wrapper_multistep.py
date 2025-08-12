import gymnasium as gym

from yndf.movement import SolidGlyphs
from yndf.nethack_state import NethackState

from yndf.wrapper_actions import NethackActionWrapper, UserInputAction
from yndf.wrapper_rewards import NethackRewardWrapper

class NethackMultistepActionWrapper(gym.Wrapper):
    """Convert NLE observation → dict(glyphs, visited_mask, agent_yx)."""

    def __init__(self, env: gym.Env, actions : NethackActionWrapper, rewards : NethackRewardWrapper) -> None:
        super().__init__(env)
        self.actions : NethackActionWrapper = actions
        self.rewards : NethackRewardWrapper = rewards
        self.max_search = 22

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            return obs, reward, terminated, truncated, info

        if isinstance(action, UserInputAction):
            action = self.actions.model_actions.index(action.action)

        if action == self.actions.kick_index:
            # Check if the player can kick
            start_state : NethackState = info['state']
            valid_kick_actions = self.actions.get_valid_kick_actions(start_state)

            for kick_action in valid_kick_actions:
                obs, reward2, terminated, truncated, info2 = self.env.step(kick_action)
                break
            else:
                # No valid kick action found, return the original step
                return obs, reward, terminated, truncated, info

            reward = reward + reward2
            self.rewards.merge_reward_info(info, info2)

        if action == self.actions.search_index:
            # Check if the player can search
            start_state : NethackState = info['state']
            start_state.search_state.search_counts[start_state.player.position] += 1

            # we already searched once at the top of the method
            already_searched = start_state.search_state.search_counts[start_state.player.position]
            if already_searched < self.max_search:
                times_to_search = self.max_search - already_searched

                state = start_state
                for _ in range(times_to_search):
                    # If there are visible enemies, we can't search
                    if state.visible_enemies or terminated or truncated:
                        break

                    # did we reveal any new tiles?
                    prev_stones = (start_state.floor_glyphs == SolidGlyphs.S_stone.value).sum()
                    new_stones = (state.floor_glyphs == SolidGlyphs.S_stone.value).sum()
                    if prev_stones > new_stones:
                        break

                    obs, reward2, terminated, truncated, info2 = self.env.step(self.actions.search_index)
                    state = info2['state']
                    state.search_state.search_counts[state.player.position] += 1
                    reward += reward2
                    self.rewards.merge_reward_info(info, info2)

        return obs, reward, terminated, truncated, info
