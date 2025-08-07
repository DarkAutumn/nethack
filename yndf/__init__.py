"""yndf package initialization."""

import gymnasium as gym
import nle

from yndf.wrapper_obs import NethackObsWrapper
from yndf.wrapper_rewards import NethackRewardWrapper
from yndf.wrapper_actions import NethackActionWrapper
from yndf.wrapper_state import NethackStateWrapper

from yndf.neural_network import NethackMaskablePolicy

def create_env(**kwargs) -> gym.Env:
    """Create a Nethack environment with the necessary wrappers."""
    env = gym.make("NetHackScore-v0", **kwargs)
    env = NethackStateWrapper(env)
    env = NethackObsWrapper(env)
    env = NethackActionWrapper(env)
    env = NethackRewardWrapper(env)
    return env

gym.register(id="YenderFlow-v0", entry_point="yndf:create_env")

__all__ = [NethackMaskablePolicy.__name__]
