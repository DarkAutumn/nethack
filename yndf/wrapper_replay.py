from typing import List
import gymnasium as gym
from yndf.nethack_state import NethackState
from yndf.replays import save_replay
from yndf.wrapper_actions import NethackActionWrapper
from yndf.replays import StepRecord

class NethackReplayWrapper(gym.Wrapper):
    """Wraps the environment to save replays of the game steps."""

    def __init__(self, env: gym.Env, action_wrapper: NethackActionWrapper) -> None:
        super().__init__(env)
        self._steps : List[StepRecord] = []
        self._action_wrapper = action_wrapper

    def reset(self, **kwargs):
        # don't save replay on reset
        self._steps.clear()

        obs, info = self.env.reset(**kwargs)
        assert len(self._steps) == 0, "Replay steps should be cleared."
        self.add_step(obs, None, info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.add_step(obs, action, info)

        if terminated or truncated:
            self._save_replay()

        return obs, reward, terminated, truncated, info

    def add_step(self, obs, action, info):
        """Add a step to the replay."""
        if action is not None:
            action = self._action_wrapper.translate_to_keycode(action)
        state : NethackState = info["state"]

        step_record = StepRecord(action, state.original.observation, state.original.info, obs, info)
        self._steps.append(step_record)

    def _save_replay(self):
        # Save the replay to a file
        if self._steps:
            replay_dir = "replays"
            save_replay(replay_dir, self._steps)
            self._steps.clear()
