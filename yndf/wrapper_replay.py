import os
from typing import Dict
from pathlib import Path
import datetime as dt
import pickle
import subprocess
import tempfile
import threading

import gymnasium as gym
import numpy as np

GROWTH = 1.5
MIN_CAP = 32
class NethackReplayWrapper(gym.Wrapper):
    """Wraps the environment to save replays of the game steps."""

    def __init__(self, env: gym.Env, replay_dir: str | os.PathLike[str]) -> None:
        super().__init__(env)
        self._history: Dict[str, np.ndarray] = {}
        self._step = 0
        self._replay_dir = Path(replay_dir)

    def reset(self, **kwargs):
        # if we reset the game without terminating/truncating, only save if we made some progress
        # we don't need a 0 or 1 step game
        if self._step > 1:
            self._save_and_clear("reset")

        obs, info = self.env.reset(**kwargs)
        self._step = 0
        self._add_step(None, obs)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        action_keycode = self.unwrapped.actions[action].value

        self._add_step(action_keycode, obs)
        if terminated or truncated:
            done = None
            if info['end_status'] == 1: # game_over
                done = self.unwrapped.nethack.how_done().name
            elif info['end_status'] == -1: #aborted
                done = "aborted"
            self._save_and_clear(done)

        return obs, reward, terminated, truncated, info

    def _save_and_clear(self, ending: str):
        assert "ending" not in self._history, "Cannot have 'ending' as a recorded key."
        completed = {'ending': ending}
        for k, arr in self._history.items():
            completed[k] = arr[:self._step]

        self._step = 0

        self.__save(completed)

    def __save(self, completed):
        path = Path(self._replay_dir)
        path.mkdir(parents=True, exist_ok=True)

        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        pid = os.getpid()
        tid = threading.get_ident()
        final_name = f"replay-{ts}-{pid}-{tid}.pkl"
        final_path = path / final_name

        fd, tmp_str = tempfile.mkstemp(prefix=f".{final_name}.", suffix=".tmp", dir=path)
        tmp_path = Path(tmp_str)

        try:
            with os.fdopen(fd, "wb") as fobj:
                pickle.dump(completed, fobj, protocol=pickle.HIGHEST_PROTOCOL)
                fobj.flush()
                os.fsync(fobj.fileno())
            os.replace(tmp_path, final_path)

            # Best-effort fsync on the directory entry
            try:
                dir_fd = os.open(path, os.O_DIRECTORY)
            except OSError:
                dir_fd = None
            if dir_fd is not None:
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)

        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

        try:
            proc = subprocess.Popen(
                ["bzip2", "--", str(final_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,  # detach from our process group/terminal
                close_fds=True,          # don't leak FDs into the compressor
            )

            # Reap the child in a daemon thread so it doesn't linger as a zombie
            threading.Thread(target=proc.wait, daemon=True).start()

        except FileNotFoundError:
            # bzip2 not installed; silently skip background compression
            pass

        except Exception: # pylint: disable=broad-exception-caught
            # Swallow errors to avoid affecting the caller’s flow
            pass


    def _add_step(self, action_keycode: int, obs: Dict[str, np.ndarray]) -> None:
        """
        Append one observation dict to the batched buffers in self._history.
        Grows capacity geometrically (×1.5) when needed.
        """

        if not self._history:
            # First use: allocate arrays with an initial capacity.
            init_cap = max(MIN_CAP, 1)
            self._history = {
                k: np.empty((init_cap, *v.shape), dtype=v.dtype) for k, v in obs.items()
            }
            self._history["action"] = np.empty((init_cap,), dtype=np.int16)

        # Derive current capacity from any buffer we have.
        example_key = next(iter(self._history))
        cap = self._history[example_key].shape[0]

        # Grow if the next write would exceed capacity.
        if self._step >= cap:
            new_cap = max(int(cap * GROWTH), self._step + 1, MIN_CAP)
            for k, arr in self._history.items():
                grown = np.empty((new_cap, *arr.shape[1:]), dtype=arr.dtype)
                if self._step > 0:
                    grown[:self._step] = arr[:self._step]
                self._history[k] = grown
            cap = new_cap  # (not strictly needed, kept for clarity)

        # Insert this step.
        self._history["action"][self._step] = action_keycode if action_keycode is not None else -1
        for k, v in obs.items():
            dest = self._history[k]
            if dest.shape[1:] != v.shape:
                raise ValueError(
                    f"Shape mismatch for '{k}': expected {dest.shape[1:]}, got {v.shape}"
                )
            self._history[k][self._step] = v

        self._step += 1
