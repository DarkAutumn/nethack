"""A wrapper that profiles step and reset methods of the environment."""

import cProfile

import gymnasium as gym

class ProfilingWrapper(gym.Wrapper):
    """A wrapper that profiles the step and reset methods of the environment."""

    def __init__(self, env: gym.Env, profiler: cProfile.Profile) -> None:
        super().__init__(env)

        self.profiler = profiler

    def reset(self, **kwargs):
        self.profiler.enable()
        obs = super().reset(**kwargs)
        self.profiler.disable()
        return obs

    def step(self, action):
        self.profiler.enable()
        result = super().step(action)
        self.profiler.disable()
        return result
