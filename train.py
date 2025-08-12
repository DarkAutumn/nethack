"""Trains a Maskable PPO agent for Nethack navigation tasks."""

from __future__ import annotations

import argparse
import cProfile
from collections import Counter
from pathlib import Path
import pstats
from typing import Callable

import gymnasium as gym
from nle import nethack
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

import yndf
from yndf.wrapper_profiler import ProfilingWrapper

# ----------------------------- Actions ------------------------------------- #
MOVE_ACTIONS = tuple(nethack.CompassDirection)       # 8 directions
DESCEND_ACTION = (nethack.MiscDirection.DOWN,)       # '>'
OTHER_ACTIONS = (nethack.Command.KICK, nethack.Command.SEARCH)
ACTIONS = MOVE_ACTIONS + DESCEND_ACTION + OTHER_ACTIONS

# Target total rollout size per update across all envs.
ROLLOUT_TARGET = 4096
DEFAULT_BATCH_SIZE = 1024

class PeriodicCheckpointCallback(BaseCallback):
    """Save model checkpoints every `save_every` timesteps.

    Files are written to `save_dir / f"{model_name}_{num_timesteps}.zip"`.
    """
    def __init__(self, save_every: int, save_dir: Path, model_name: str, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.save_every = int(save_every)
        self.save_dir = Path(save_dir)
        self.model_name = model_name
        self._last_saved_step = 0

    def _on_training_start(self) -> bool:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        return True

    def _on_step(self) -> bool:
        steps = int(self.model.num_timesteps)
        if steps - self._last_saved_step >= self.save_every:
            path = self.save_dir / f"{self.model_name}_{steps}.zip"
            self.model.save(str(path))
            self._last_saved_step = steps
            if self.verbose:
                print(f"[checkpoint] saved {path}")
        return True

class InfoCountsLogger(BaseCallback):
    """Log counts of various info keys during training."""
    def __init__(self, log_every: int, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.log_every = log_every
        self._last_log_step = 0
        self._emitted = set()
        self._counters = {}
        self._values = {}
        self._averages = {}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos:
            # `infos` is list[dict] for vec envs; may be dict for non-vec
            if isinstance(infos, dict):
                infos_iter = (infos,)
            else:
                infos_iter = infos

            for info in infos_iter:
                if not info:
                    continue

                if (ending := info.get("ending", None)) is not None:
                    self._counters.setdefault("endings", []).append(ending)
                    state = info.get("state", None)
                    if state is not None:
                        self._averages.setdefault("counts/depth", []).append(state.player.depth)

                if (rewards := info.get("rewards", None)) is not None:
                    for name, value in rewards.items():
                        key = f"rewards/{name}"
                        self._values[key] = self._values.get(key, 0.0) + value

        return True

    def _on_rollout_end(self) -> bool:
        if self.model.num_timesteps - self._last_log_step < self.log_every:
            return True

        self._last_log_step = int(self.model.num_timesteps)

        curr_emitted = set()
        for base_name, values in self._counters.items():
            counter = Counter(values)
            for name, count in counter.items():
                key = f"{base_name}/{name}"
                self.logger.record(key, count / len(values) if values else 0)
                curr_emitted.add(key)
                self._emitted.add(key)

        for name, values in self._averages.items():
            if values:
                avg = sum(values) / len(values)
                self.logger.record(name, avg)
                curr_emitted.add(name)
                self._emitted.add(name)

        for key, value in self._values.items():
            self.logger.record(key, value)
            curr_emitted.add(key)
            self._emitted.add(key)

        missing = self._emitted - curr_emitted
        for key in missing:
            self.logger.record(key, 0)

        # Reset for the next rollout window
        for v in self._counters.values():
            v.clear()

        for k in self._values:
            self._values[k] = 0.0

        return True

def main(
    total_timesteps: int,
    parallel: int = 12,
    output_dir: str = "models/",
    log_dir: str = "logs/",
    name: str = "nethack",
    save_replays: bool = True,
    profile: bool = False,
) -> None:
    """Train an agent to play NetHack.

    Args:
        total_timesteps: Total environment steps to train for.
        parallel: Number of parallel environments. If 1, uses a single env
                  (no SubprocVecEnv). Default is 12.
        output_dir: Directory to save trained models. Default "models/".
        log_dir: Directory for TensorBoard logs. Default "logs/".
    """
    out_path = Path(output_dir)
    log_path = Path(log_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)

    assert not profile or parallel <= 1, "Profiling requires single-threaded execution (parallel=1)."

    def _make_env(_: int) -> Callable[[], gym.Env]:
        """Factory for creating a single environment instance."""
        def _init() -> gym.Env:
            return gym.make("YenderFlow-v0", actions=ACTIONS, save_replays=save_replays)
        return _init

    profiler = None
    if parallel <= 1:
        n_envs = 1
        env = _make_env(0)()
        if profile:
            profiler = cProfile.Profile()
            env = ProfilingWrapper(env, profiler)

    else:
        n_envs = parallel
        env = SubprocVecEnv([_make_env(i) for i in range(n_envs)], start_method="fork")
        env = VecMonitor(env)

    # Keep total rollout size roughly constant across different parallelism.
    n_steps_per_env = max(1, ROLLOUT_TARGET // n_envs)
    batch_size = min(DEFAULT_BATCH_SIZE, n_steps_per_env * n_envs)
    model_file_name_base = f"{name}_{total_timesteps}"

    print(
        f"Starting training: timesteps={total_timesteps}, "
        f"parallel_envs={n_envs}, n_steps/env={n_steps_per_env}, "
        f"batch_size={batch_size}, output='{out_path}', logs='{log_path}'"
    )
    save_callback = PeriodicCheckpointCallback(
        save_every=100_000,
        save_dir=out_path,
        model_name=model_file_name_base,
        verbose=1,
    )

    info_callback = InfoCountsLogger(log_every=100_000, verbose=1)
    callbacks = CallbackList([save_callback, info_callback])

    model = MaskablePPO(
        policy=yndf.NethackMaskablePolicy,
        env=env,
        verbose=1,
        batch_size=batch_size,
        n_steps=n_steps_per_env,
        tensorboard_log=str(log_path),
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callbacks)
    model.save(str(out_path / model_file_name_base))

    print("Training finished and model saved.")
    if profiler:
        profiler.disable()
        profiler.dump_stats(str(out_path / f"{model_file_name_base}_profile.prof"))
        print(f"Profiling data saved to {out_path / f'{model_file_name_base}_profile.prof'}")
        stats = pstats.Stats(profiler)
        stats.strip_dirs().sort_stats('cumulative').print_stats(50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Maskable PPO on YenderFlow NetHack.")
    parser.add_argument(
        "timesteps",
        type=int,
        help="Total number of environment steps to train for (required).",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=12,
        help="Number of parallel envs (default: 12). Use 1 to disable SubprocVecEnv.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/",
        help='Directory to save trained models (default: "models/").',
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/",
        help='Directory for TensorBoard logs (default: "logs/").',
    )
    parser.add_argument(
        "--save-replays",
        type=bool,
        help="Save replays of the training episodes.",
        default=True,
    )
    parser.add_argument(
        "--name",
        type=str,
        default="nethack",
        help="Base name for the model files (default: 'nethack').",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling of the training process (single-threaded).",
        default=False,
    )

    args = parser.parse_args()
    if args.profile:
        args.parallel = 1  # Profiling requires single-threaded execution

    main(
        total_timesteps=args.timesteps,
        parallel=args.parallel,
        output_dir=args.output,
        name=args.name,
        log_dir=args.log_dir,
        save_replays=args.save_replays,
        profile=args.profile
    )
