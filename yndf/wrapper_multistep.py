import gymnasium as gym

from yndf.nethack_level import GLYPH_TABLE, DungeonLevel
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
            floor : DungeonLevel = start_state.floor
            floor.search_count[start_state.player.position] += 1

            # we already searched once at the top of the method
            already_searched = floor.search_count[start_state.player.position]
            if already_searched < self.max_search:
                self.rewards.no_discovery.disable()
                times_to_search = self.max_search - already_searched

                state = start_state
                for _ in range(times_to_search):
                    # If there are visible enemies, we can't search
                    if state.floor.num_enemies or terminated or truncated:
                        break

                    # did we reveal any new tiles?
                    stones = (start_state.floor.properties & GLYPH_TABLE.STONE) != 0
                    stones &= (state.floor.properties & GLYPH_TABLE.STONE) == 0
                    if stones.any():
                        break

                    obs, reward2, terminated, truncated, info2 = self.env.step(self.actions.search_index)
                    state = info2['state']
                    floor.search_count[state.player.position] += 1
                    reward += reward2
                    self.rewards.merge_reward_info(info, info2)


                self.rewards.no_discovery.enable()

        return obs, reward, terminated, truncated, info
