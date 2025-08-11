import gymnasium as gym

from yndf.nethack_state import NethackState

from yndf.wrapper_actions import NethackActionWrapper
from yndf.wrapper_rewards import NethackRewardWrapper

class NethackMultistepActionWrapper(gym.Wrapper):
    """Convert NLE observation → dict(glyphs, visited_mask, agent_yx)."""

    def __init__(self, env: gym.Env, actions : NethackActionWrapper, rewards : NethackRewardWrapper) -> None:
        super().__init__(env)
        self.actions : NethackActionWrapper = actions
        self.rewards : NethackRewardWrapper = rewards

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            return obs, reward, terminated, truncated, info

        if action == self.actions.kick_index:
            # Check if the player can kick
            state : NethackState = info['state']
            valid_kick_actions = self.actions.get_valid_kick_actions(state)

            for kick_action in valid_kick_actions:
                obs, reward2, terminated, truncated, info2 = self.env.step(kick_action)
                break
            else:
                # No valid kick action found, return the original step
                return obs, reward, terminated, truncated, info

            reward = reward + reward2
            self.rewards.merge_reward_info(info, info2)

        return obs, reward, terminated, truncated, info
