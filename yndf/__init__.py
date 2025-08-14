"""yndf package initialization."""

import gymnasium as gym
import nle

from yndf.nethack_state import NethackState
from yndf.wrapper_obs import NethackObsWrapper
from yndf.wrapper_rewards import NethackRewardWrapper
from yndf.wrapper_actions import NethackActionWrapper
from yndf.wrapper_state import NethackStateWrapper
from yndf.wrapper_replay import NethackReplayWrapper

from yndf.neural_network import NethackMaskablePolicy

def create_env(**kwargs) -> gym.Env:
    """Create a Nethack environment with the necessary wrappers."""
    env = gym.make("NetHackScore-v0", actions=nle.nethack.ACTIONS, max_episode_steps=20_000)

    actions = kwargs.get("actions", env.unwrapped.actions)
    has_search = nle.nethack.Command.SEARCH in actions

    if (replay_dir := kwargs.get("replay_dir", None)):
        env = NethackReplayWrapper(env, replay_dir)

    env = NethackStateWrapper(env)
    env = NethackObsWrapper(env)
    env = NethackRewardWrapper(env, has_search)
    env = NethackActionWrapper(env, actions)

    return env

gym.register(id="YenderFlow-v0", entry_point="yndf:create_env")

__all__ = [
    NethackState.__name__,
    NethackMaskablePolicy.__name__,
    ]
