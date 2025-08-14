from typing import List
import gymnasium as gym
from yndf.replays import save_replay
from yndf.replays import StepRecord

class NethackReplayWrapper(gym.Wrapper):
    """Wraps the environment to save replays of the game steps."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._steps : List[StepRecord] = []

    def reset(self, **kwargs):
        # if we reset the game without terminating/truncating, only save if we made some progress
        # we don't need a 0 or 1 step game
        if len(self._steps) > 1:
            self._save_replay()

        obs, info = self.env.reset(**kwargs)

        self._steps.clear()
        self._steps.append(StepRecord(None, obs, info))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        action_keycode = self.unwrapped.actions[action].value
        if info['end_status'] == 1: # game_over
            info_copy = info.copy()
            info_copy['how_done'] = self.unwrapped.nethack.how_done().name
            self._steps.append(StepRecord(action_keycode, obs, info_copy))
        else:
            self._steps.append(StepRecord(action_keycode, obs, info))

        if terminated or truncated:
            self._save_replay()

        return obs, reward, terminated, truncated, info

    def _save_replay(self):
        # Save the replay to a file
        if self._steps:
            replay_dir = "replays"
            save_replay(replay_dir, self._steps)
            self._steps.clear()
