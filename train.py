"""Trains a Maskable PPO agent for Nethack navigation tasks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import gymnasium as gym
from nle import nethack
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

import yndf

# ----------------------------- Actions ------------------------------------- #
MOVE_ACTIONS = tuple(nethack.CompassDirection)       # 8 directions
DESCEND_ACTION = (nethack.MiscDirection.DOWN,)       # '>'
OTHER_ACTIONS = (nethack.Command.KICK,)
ACTIONS = MOVE_ACTIONS + DESCEND_ACTION + OTHER_ACTIONS

# Target total rollout size per update across all envs.
ROLLOUT_TARGET = 4096
DEFAULT_BATCH_SIZE = 1024


def main(
    total_timesteps: int,
    parallel: int = 12,
    output_dir: str = "models/",
    log_dir: str = "logs/",
    save_replays: bool = True
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


    def _make_env(_: int) -> Callable[[], gym.Env]:
        """Factory for creating a single environment instance."""
        def _init() -> gym.Env:
            return gym.make("YenderFlow-v0", actions=ACTIONS, save_replays=save_replays)
        return _init

    if parallel <= 1:
        n_envs = 1
        env = _make_env(0)()
    else:
        n_envs = parallel
        env = SubprocVecEnv([_make_env(i) for i in range(n_envs)], start_method="fork")

    env = VecMonitor(env)

    # Keep total rollout size roughly constant across different parallelism.
    n_steps_per_env = max(1, ROLLOUT_TARGET // n_envs)
    batch_size = min(DEFAULT_BATCH_SIZE, n_steps_per_env * n_envs)

    model = MaskablePPO(
        policy=yndf.NethackMaskablePolicy,
        env=env,
        verbose=1,
        batch_size=batch_size,
        n_steps=n_steps_per_env,
        tensorboard_log=str(log_path),
    )

    print(
        f"Starting training: timesteps={total_timesteps}, "
        f"parallel_envs={n_envs}, n_steps/env={n_steps_per_env}, "
        f"batch_size={batch_size}, output='{out_path}', logs='{log_path}'"
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(str(out_path / f"ppo_nethack_nav_{total_timesteps}"))

    print("Training finished and model saved.")


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

    args = parser.parse_args()

    main(
        total_timesteps=args.timesteps,
        parallel=args.parallel,
        output_dir=args.output,
        log_dir=args.log_dir,
        save_replays=args.save_replays,
    )
