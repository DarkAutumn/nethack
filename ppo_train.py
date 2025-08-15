import argparse
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv

from tqdm import tqdm

import yndf

from models import NethackPolicy

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
LOG = logging.getLogger("ppo_train")

# -------------------------
# Utilities / Helpers
# -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_env_thunk(env_id: str, seed: int, idx: int) -> Any:
    def _thunk() -> gym.Env:
        env = gym.make(env_id)
        env.reset(seed=seed + idx)
        return env
    return _thunk

def get_action_masks_batch(envs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Fetch batched action masks from a vectorized env."""

    results = envs.call("action_masks")
    verb_masks = np.stack([r[0] for r in results], axis=0)  # (B,V)
    direction_masks = np.stack([r[1] for r in results], axis=0)  # (B,V,D)

    return verb_masks, direction_masks

def pack_action_batch(verbs: np.ndarray, dirs: np.ndarray) -> Any:
    """Pack (verb, dir) batch into the environment's expected action format."""
    return list(zip(verbs.tolist(), dirs.tolist()))

# -------------------------
# Storage (rollouts)
# -------------------------

@dataclass
class RolloutBatch:
    obs: Dict[str, torch.Tensor]          # dict of tensors, flattened (T*B,...)
    action_verb: torch.Tensor             # (T*B,)
    action_dir: torch.Tensor              # (T*B,)
    old_logprob: torch.Tensor             # (T*B,)
    value: torch.Tensor                   # (T*B,)
    advantages: torch.Tensor              # (T*B,)
    returns: torch.Tensor                 # (T*B,)
    verb_mask: torch.Tensor               # (T*B, A)  mask at action time
    dir_mask_selected: torch.Tensor       # (T*B, 9)  mask row for chosen verb at action time
    requires_dir: torch.Tensor            # (T*B,) bool

class RolloutBuffer:
    def __init__(self, num_steps: int, num_envs: int, obs_example: Dict[str, np.ndarray], num_verbs: int, device: torch.device) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.num_verbs = num_verbs

        # Preallocate storage
        self.obs: Dict[str, List[np.ndarray]] = {k: [] for k in obs_example.keys()}
        self.action_verb: List[np.ndarray] = []
        self.action_dir: List[np.ndarray] = []
        self.old_logprob: List[np.ndarray] = []
        self.value: List[np.ndarray] = []
        self.rewards: List[np.ndarray] = []
        self.dones: List[np.ndarray] = []
        self.verb_mask: List[np.ndarray] = []
        self.dir_mask_selected: List[np.ndarray] = []
        self.requires_dir: List[np.ndarray] = []

    def add(
        self,
        obs: Dict[str, np.ndarray],
        action_verb: np.ndarray,
        action_dir: np.ndarray,
        old_logprob: np.ndarray,
        value: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        verb_mask: np.ndarray,            # (B,A)
        dir_mask_selected: np.ndarray,    # (B,9)
        requires_dir: np.ndarray,         # (B,)
    ) -> None:
        for k, v in obs.items():
            self.obs[k].append(np.asarray(v))
        self.action_verb.append(np.asarray(action_verb))
        self.action_dir.append(np.asarray(action_dir))
        self.old_logprob.append(np.asarray(old_logprob))
        self.value.append(np.asarray(value))
        self.rewards.append(np.asarray(reward))
        self.dones.append(np.asarray(done))
        self.verb_mask.append(np.asarray(verb_mask))
        self.dir_mask_selected.append(np.asarray(dir_mask_selected))
        self.requires_dir.append(np.asarray(requires_dir))

    def compute_gae(self, last_value: np.ndarray, gamma: float, gae_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
        # Shapes: lists length T of arrays (B,)
        T = len(self.rewards)
        B = self.rewards[0].shape[0]
        advantages = np.zeros((T, B), dtype=np.float32)
        lastgaelam = np.zeros((B,), dtype=np.float32)
        for t in reversed(range(T)):
            nextnonterminal = 1.0 - self.dones[t].astype(np.float32)
            next_values = last_value if t == T - 1 else self.value[t + 1]
            delta = self.rewards[t].astype(np.float32) + gamma * next_values * nextnonterminal - self.value[t].astype(np.float32)
            lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + np.asarray(self.value, dtype=np.float32)
        return advantages, returns

    def to_batches(self) -> RolloutBatch:
        # Flatten T and B
        def _stack_flat(key: str) -> torch.Tensor:
            arr = np.stack(self.obs[key], axis=0)  # (T,B,...)
            T, B = arr.shape[:2]
            arr = arr.reshape(T * B, *arr.shape[2:])
            return torch.as_tensor(arr, device=self.device)

        obs_torch: Dict[str, torch.Tensor] = {k: _stack_flat(k) for k in self.obs.keys()}

        def _stack1(lst: List[np.ndarray]) -> torch.Tensor:
            arr = np.stack(lst, axis=0)  # (T,B,...)
            T, B = arr.shape[:2]
            arr = arr.reshape(T * B, *arr.shape[2:])
            return torch.as_tensor(arr, device=self.device)

        action_verb = _stack1(self.action_verb).long()
        action_dir = _stack1(self.action_dir).long()
        old_logprob = _stack1(self.old_logprob).float()
        value = _stack1(self.value).float()

        # Advantages/returns are computed after rollout
        advantages_np = getattr(self, "_advantages_np")
        returns_np = getattr(self, "_returns_np")
        advantages = torch.as_tensor(advantages_np.reshape(-1), device=self.device)
        returns = torch.as_tensor(returns_np.reshape(-1), device=self.device)

        verb_mask = _stack1(self.verb_mask).bool()
        dir_mask_selected = _stack1(self.dir_mask_selected).bool()
        requires_dir = _stack1(self.requires_dir).bool().squeeze(-1)

        return RolloutBatch(
            obs=obs_torch,
            action_verb=action_verb,
            action_dir=action_dir,
            old_logprob=old_logprob,
            value=value,
            advantages=advantages,
            returns=returns,
            verb_mask=verb_mask,
            dir_mask_selected=dir_mask_selected,
            requires_dir=requires_dir,
        )

    def set_advantages_returns(self, advantages: np.ndarray, returns: np.ndarray) -> None:
        self._advantages_np = advantages  # (T,B)
        self._returns_np = returns        # (T,B)

# -------------------------
# Training
# -------------------------

@dataclass
class PPOArgs:
    env_id: str = "YenderFlow-v0"
    total_timesteps: int = 2_000_000
    num_envs: int = 20
    num_steps: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef_verb: float = 0.02
    ent_coef_dir: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    glyph_vocab_size: int = 6000
    exp_name: str = "ppo_verb_dir"
    track_tb: bool = True

def parse_args() -> PPOArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default=PPOArgs.env_id)
    parser.add_argument("--total-timesteps", type=int, default=PPOArgs.total_timesteps)
    parser.add_argument("--num-envs", type=int, default=PPOArgs.num_envs)
    parser.add_argument("--num-steps", type=int, default=PPOArgs.num_steps)
    parser.add_argument("--learning-rate", type=float, default=PPOArgs.learning_rate)
    parser.add_argument("--gamma", type=float, default=PPOArgs.gamma)
    parser.add_argument("--gae-lambda", type=float, default=PPOArgs.gae_lambda)
    parser.add_argument("--num-minibatches", type=int, default=PPOArgs.num_minibatches)
    parser.add_argument("--update-epochs", type=int, default=PPOArgs.update_epochs)
    parser.add_argument("--clip-coef", type=float, default=PPOArgs.clip_coef)
    parser.add_argument("--ent-coef-verb", type=float, default=PPOArgs.ent_coef_verb)
    parser.add_argument("--ent-coef-dir", type=float, default=PPOArgs.ent_coef_dir)
    parser.add_argument("--vf-coef", type=float, default=PPOArgs.vf_coef)
    parser.add_argument("--max-grad-norm", type=float, default=PPOArgs.max_grad_norm)
    parser.add_argument("--seed", type=int, default=PPOArgs.seed)
    parser.add_argument("--glyph-vocab-size", type=int, default=PPOArgs.glyph_vocab_size)
    parser.add_argument("--exp-name", type=str, default=PPOArgs.exp_name)
    parser.add_argument("--no-tb", action="store_true", help="Disable TensorBoard logging")
    parser.add_argument("--device", type=str, default=PPOArgs.device)
    args_ns = parser.parse_args()
    args = PPOArgs(
        env_id=args_ns.env_id,
        total_timesteps=args_ns.total_timesteps,
        num_envs=args_ns.num_envs,
        num_steps=args_ns.num_steps,
        learning_rate=args_ns.learning_rate,
        gamma=args_ns.gamma,
        gae_lambda=args_ns.gae_lambda,
        num_minibatches=args_ns.num_minibatches,
        update_epochs=args_ns.update_epochs,
        clip_coef=args_ns.clip_coef,
        ent_coef_verb=args_ns.ent_coef_verb,
        ent_coef_dir=args_ns.ent_coef_dir,
        vf_coef=args_ns.vf_coef,
        max_grad_norm=args_ns.max_grad_norm,
        seed=args_ns.seed,
        glyph_vocab_size=args_ns.glyph_vocab_size,
        exp_name=args_ns.exp_name,
        track_tb=not args_ns.no_tb,
        device=args_ns.device,
    )
    return args

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # Create vectorized envs
    if args.num_envs > 1:
        envs = AsyncVectorEnv([make_env_thunk(args.env_id, args.seed, i) for i in range(args.num_envs)])
    else:
        envs = SyncVectorEnv([make_env_thunk(args.env_id, args.seed, 0)])

    # Reset an get an initial observation to infer shapes
    obs_np, _ = envs.reset()
    # Initial masks to infer num_verbs
    verb_mask_np, dir_mask_np = get_action_masks_batch(envs)
    num_verbs = verb_mask_np.shape[1]
    num_directions = dir_mask_np.shape[2]
    LOG.info("Num verbs: %d, Directions: %d", num_verbs, num_directions)

    policy = NethackPolicy(num_verbs=num_verbs, glyph_vocab_size=args.glyph_vocab_size).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)

    writer: SummaryWriter | None = None
    if args.track_tb:
        logdir = os.path.join("runs", f"{args.exp_name}_{int(time.time())}")
        writer = SummaryWriter(log_dir=logdir)
        LOG.info("TensorBoard logging to: %s", logdir)

    # Rollout buffer
    rb = RolloutBuffer(args.num_steps, args.num_envs, obs_example=obs_np, num_verbs=num_verbs, device=device)

    global_step = 0
    start_time = time.time()
    next_obs = obs_np

    progress = tqdm(total=args.total_timesteps)
    while global_step < args.total_timesteps:
        # Collect a rollout of num_steps
        for step in range(args.num_steps):
            global_step += args.num_envs
            progress.update(args.num_envs)

            # Fetch masks for each env
            verb_mask_np, dir_mask_np = get_action_masks_batch(envs)  # (B,A), (B,A,9)

            # Convert obs + masks to torch
            obs_t: Dict[str, torch.Tensor] = {k: torch.as_tensor(next_obs[k], device=device) for k in next_obs}
            verb_mask = torch.as_tensor(verb_mask_np, device=device, dtype=torch.bool)
            dir_mask = torch.as_tensor(dir_mask_np, device=device, dtype=torch.bool)

            with torch.no_grad():
                act_verb_t, act_dir_t, logprob_t, value_t, _, _ = policy.get_action_and_value(
                    obs_t, verb_mask, dir_mask, action_verb=None, action_dir=None, deterministic=False
                )
                # For storage and env stepping, move to CPU numpy
                act_verb = act_verb_t.cpu().numpy()
                act_dir = act_dir_t.cpu().numpy()
                logprob = logprob_t.cpu().numpy()
                value = value_t.cpu().numpy()
                # Save selected dir mask row and requires_dir flag
                batch_idx = np.arange(args.num_envs)
                dir_mask_selected = dir_mask_np[batch_idx, act_verb, :]  # (B,9)
                requires_dir = dir_mask_selected.any(axis=1)

            # Step the envs
            env_actions = pack_action_batch(act_verb, act_dir)
            # pylint: disable=unbalanced-tuple-unpacking
            next_obs, rewards, terminations, truncations, infos = envs.step(env_actions)
            dones = np.logical_or(terminations, truncations)

            # Bookkeeping
            rb.add(
                obs=obs_np,
                action_verb=act_verb,
                action_dir=act_dir,
                old_logprob=logprob,
                value=value,
                reward=rewards,
                done=dones,
                verb_mask=verb_mask_np,
                dir_mask_selected=dir_mask_selected,
                requires_dir=requires_dir.astype(np.bool_),
            )

            # Logging episodic returns if provided
            if writer is not None and isinstance(infos, list):
                for info in infos:
                    ep = info.get("episode")
                    if ep is not None:
                        writer.add_scalar("charts/episodic_return", ep["r"], global_step)
                        writer.add_scalar("charts/episodic_length", ep["l"], global_step)

            obs_np = next_obs  # store for next add()

        # Bootstrap value for GAE
        with torch.no_grad():
            obs_t_final = {k: torch.as_tensor(next_obs[k], device=device) for k in next_obs}
            verb_mask_np, dir_mask_np = get_action_masks_batch(envs)
            verb_mask_final = torch.as_tensor(verb_mask_np, device=device, dtype=torch.bool)
            dir_mask_final = torch.as_tensor(dir_mask_np, device=device, dtype=torch.bool)
            _, _, _, last_value_t, _, _ = policy.get_action_and_value(
                obs_t_final, verb_mask_final, dir_mask_final, action_verb=None, action_dir=None, deterministic=True
            )
            last_value = last_value_t.cpu().numpy()

        advantages, returns = rb.compute_gae(last_value, gamma=args.gamma, gae_lambda=args.gae_lambda)
        rb.set_advantages_returns(advantages, returns)
        batch = rb.to_batches()

        # Normalize advantages
        adv = batch.advantages
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        # PPO update
        bsz = batch.action_verb.shape[0]  # T*B
        minibatch_size = bsz // args.num_minibatches
        idxs = np.arange(bsz)

        for _ in range(args.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, bsz, minibatch_size):
                mb_idx = idxs[start : start + minibatch_size]

                mb_obs = {k: v[mb_idx] for k, v in batch.obs.items()}
                mb_action_verb = batch.action_verb[mb_idx]
                mb_action_dir = batch.action_dir[mb_idx]
                mb_old_logprob = batch.old_logprob[mb_idx]
                mb_adv = adv[mb_idx]
                mb_returns = batch.returns[mb_idx]
                mb_values = batch.value[mb_idx]
                mb_verb_mask = batch.verb_mask[mb_idx]
                mb_dir_mask_selected = batch.dir_mask_selected[mb_idx]

                new_logprob, new_value, ent_verb, ent_dir, requires_dir = policy.evaluate_actions(
                    mb_obs, mb_verb_mask, mb_dir_mask_selected, mb_action_verb, mb_action_dir
                )

                # Policy loss
                logratio = new_logprob - mb_old_logprob
                ratio = torch.exp(logratio)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clip like PPO)
                v_loss_unclipped = (new_value - mb_returns) ** 2
                v_clipped = mb_values + torch.clamp(new_value - mb_values, -args.clip_coef, args.clip_coef)
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # Entropy bonuses
                # Average dir entropy over samples that required a dir
                dir_mask = requires_dir.float()
                dir_entropy_mean = (ent_dir * dir_mask).sum() / (dir_mask.sum() + 1e-8)
                entropy_loss = -(args.ent_coef_verb * ent_verb.mean() + args.ent_coef_dir * dir_entropy_mean)

                loss = pg_loss + args.vf_coef * v_loss + entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

            # End of epoch
        # End of update

        # Logging
        if writer is not None:
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/entropy_verb", ent_verb.mean().item(), global_step)
            writer.add_scalar("losses/entropy_dir", dir_entropy_mean.item(), global_step)
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            sps = int((global_step) / (time.time() - start_time))
            writer.add_scalar("charts/SPS", sps, global_step)

        # Reset buffer for next rollout
        rb = RolloutBuffer(args.num_steps, args.num_envs, obs_example=next_obs, num_verbs=num_verbs, device=device)

    # Cleanup
    envs.close()
    progress.close()
    if writer is not None:
        writer.close()

if __name__ == "__main__":
    main()
