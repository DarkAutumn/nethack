#!/usr/bin/env python3
"""Evaluate a trained MaskablePPO model on YenderFlow NetHack.

Runs full episodes (terminated or truncated) in parallel until the requested
number of episodes completes, collecting EpisodeResult for each.

Example:
    python evaluate.py /path/to/model.zip --episodes 100 --parallel 10
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import time
from typing import List

import gymnasium as gym
from sb3_contrib.ppo_mask import MaskablePPO

from debugger import get_action_masker
from train import ACTIONS

# Your project code should be importable (policy, env registration, etc.)

@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """One completed episode's summary."""
    duration : float  # Time taken for the episode
    total_reward: float
    steps: int
    max_depth: int
    ending : str = None  # Ending type, e.g. "success", "death", etc.

def evaluate_model(model_path: Path, episodes : int, deterministic: bool,
                   save_replays: bool, quiet: bool) -> List[EpisodeResult]:
    """Run full episodes in parallel until `episodes` have completed.

    Args:
        model_path: Path to saved .zip model.
        episodes: Total number of completed episodes to gather.
        deterministic: Use deterministic actions.
        save_replays: Pass through to envs.
        quiet: Suppress output except for final results.

    Returns:
        Exactly `episodes` EpisodeResult entries.
    """
    if episodes <= 0:
        raise ValueError("episodes must be > 0")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    results = []
    env = gym.make("YenderFlow-v0", actions=ACTIONS, save_replays=save_replays)
    model = MaskablePPO.load(model_path, env=env)
    action_masker = get_action_masker(env)

    for ep in range(episodes):
        start_time = time.time()
        obs, info = env.reset()

        terminated = truncated = False
        total_reward = 0.0
        steps = 0
        max_depth = 0
        ending = None

        while not (terminated or truncated):
            action_masks = action_masker.action_masks()
            action = model.predict(obs, deterministic=deterministic, action_masks=action_masks)[0]
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1
            max_depth = max(max_depth, info['state'].player.depth)

            if terminated or truncated:
                ending = info.get('ending', 'unknown')

        duration = time.time() - start_time
        if not quiet:
            print(f"Episode {ep + 1}/{episodes} - Duration: {duration:.2f}s, "
                  f"Reward: {total_reward:.2f}, Steps: {steps}, "
                  f"Max Depth: {max_depth}, Ending: {ending}")

        results.append(EpisodeResult(duration, total_reward, steps, max_depth, ending))

    env.close()
    return results

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate a MaskablePPO model on YenderFlow NetHack.")
    parser.add_argument("model", type=Path, help="Path to the saved model .zip file.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of completed episodes to run (default: 100).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions (default is deterministic).",
    )
    parser.add_argument(
        "--save-replays",
        action="store_true",
        help="Pass save_replays=True to evaluation envs.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output except for final results.",
    )


    args = parser.parse_args()

    results = evaluate_model(
        model_path=args.model,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        save_replays=args.save_replays,
        quiet=args.quiet
    )


    total_episodes = len(results)
    avg_reward = sum(r.total_reward for r in results) / total_episodes
    max_reward = max(r.total_reward for r in results)
    min_reward = min(r.total_reward for r in results)
    avg_steps = sum(r.steps for r in results) / total_episodes
    avg_max_depth = sum(r.max_depth for r in results) / total_episodes
    max_depth = max(r.max_depth for r in results)
    endings = Counter(r.ending for r in results if r.ending is not None)
    print(f"Completed {total_episodes} episodes:")
    print(f" - Average reward: {avg_reward:.2f}")
    print(f" - Max reward: {max_reward:.2f}")
    print(f" - Min reward: {min_reward:.2f}")
    print(f" - Average steps: {avg_steps:.2f}")
    print(f" - Average max depth: {avg_max_depth:.2f}")
    print(f" - Max depth: {max_depth:.2f}")
    print(" - Endings:")
    for ending, count in endings.items():
        print(f"   - {ending}: {count}")


if __name__ == "__main__":
    main()
