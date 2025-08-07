#!/usr/bin/env python3
import time
import gymnasium as gym
import yndf
from sb3_contrib import MaskablePPO

from train import ACTIONS

def _main(model_path: str, episodes: int = 1, render_delay: float = 0.05):
    env = gym.make("YenderFlow-v0", actions=ACTIONS)
    model = MaskablePPO.load(model_path, env=env)

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        terminated, truncated = False, False
        total_reward = 0.0

        print(f"\n=== Starting episode {ep} ===\n")
        while not terminated and not truncated:
            # Predict a maskable action
            action, _states = model.predict(obs, deterministic=True)
            # Step the env
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            print("\033[H\033[J", end="")
            ansi_frame = env.render()
            print(ansi_frame, end="", flush=True)
            print("Action taken:", ACTIONS[action].name)

            time.sleep(render_delay)

        score = info.get("score", total_reward)
        print(f"\n=== Episode {ep} finished — score: {score} ===\n")

    env.close()

if __name__ == "__main__":
    _main("ppo_nethack_nav", episodes=10)
