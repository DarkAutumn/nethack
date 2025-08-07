"""Trains a Maskable PPO agent for Nethack navigation tasks."""
import gymnasium as gym
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from nle import nethack

import yndf

MOVE_ACTIONS = tuple(nethack.CompassDirection) # 8 directions
DESCEND_ACTION = (nethack.MiscDirection.DOWN,) # '>'
ACTIONS = MOVE_ACTIONS + DESCEND_ACTION

def _make_env(_):
    def _init():
        return gym.make("YenderFlow-v0", actions=ACTIONS)
    return _init

def main(total_timesteps: int = 100_000, multiprocessing: bool = True):
    """Trains an agent to play nethack."""
    if multiprocessing:
        num_cpu = 8
        envs = SubprocVecEnv([_make_env(i) for i in range(num_cpu)], start_method="fork")
        envs = VecMonitor(envs)
    else:
        num_cpu = 1
        envs = gym.make("YenderFlow-v0", actions=ACTIONS)

    model = MaskablePPO(
        policy=yndf.NethackMaskablePolicy,
        env=envs,
        verbose=1,
        batch_size=1024,
        n_steps=4096 // num_cpu,
        tensorboard_log="./logs",
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save("models/ppo_nethack_nav")

    print("Training finished and model saved.")

if __name__ == "__main__":
    main()
