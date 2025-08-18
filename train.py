import argparse
import logging
import os
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from tqdm import tqdm

import yndf
from models import NethackPolicy, ModelSaver

# ---- your callbacks are assumed to exist in this file/module ----
# class PeriodicCheckpointCallback(BaseCallback): ...
# class InfoCountsLogger(BaseCallback): ...
# ---------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
LOG = logging.getLogger("ppo_train")

# -------------------------
# Utilities / Helpers
# -------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _make_env_thunk(env_id: str, seed: int, idx: int) -> Any:
    def _thunk() -> gym.Env:
        env = gym.make(env_id)
        if seed is not None:
            env.reset(seed=seed + idx)
        else:
            env.reset()
        return env
    return _thunk

def get_action_masks_batch(envs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Fetch batched action masks from a vectorized env."""
    results = envs.call("action_masks")
    verb_masks = np.stack([r[0] for r in results], axis=0)       # (B,V)
    direction_masks = np.stack([r[1] for r in results], axis=0)  # (B,V,D)
    return verb_masks, direction_masks

def pack_action_batch(verbs: np.ndarray, dirs: np.ndarray) -> Any:
    """Pack (verb, dir) batch into the environment's expected action format."""
    return list(zip(verbs.tolist(), dirs.tolist()))

def derive_invariant_batch(args: 'PPOArgs') -> tuple[int, int, int, int, int]:
    """
    Returns:
      num_steps, num_minibatches, minibatch_size, batch_target, batch_actual
    """
    batch_target = args.batch_size if args.batch_size is not None else args.num_envs * args.num_steps
    num_steps = max(args.min_rollout_len, batch_target // args.num_envs)  # steps per env per update
    batch_actual = args.num_envs * num_steps

    if args.minibatch_size is not None and args.minibatch_size > 0:
        desired = max(1, int(args.minibatch_size))
        nm = max(1, int(round(batch_actual / desired)))
        while batch_actual % nm != 0 and nm > 1:
            nm -= 1
        num_minibatches = nm
        minibatch_size = batch_actual // num_minibatches
    else:
        num_minibatches = args.num_minibatches
        minibatch_size = batch_actual // num_minibatches

    return num_steps, num_minibatches, minibatch_size, batch_target, batch_actual

# -------------------------
# Storage (rollouts)
# -------------------------

@dataclass
class RolloutBatch:
    """A batch of rollouts for training."""
    obs: Dict[str, torch.Tensor]          # dict of tensors, flattened (T*B,...)
    action_verb: torch.Tensor             # (T*B,)
    action_dir: torch.Tensor              # (T*B,)
    old_logprob: torch.Tensor             # (T*B,)
    value: torch.Tensor                   # (T*B,)
    advantages: torch.Tensor              # (T*B,)
    returns: torch.Tensor                 # (T*B,)
    verb_mask: torch.Tensor               # (T*B, A)
    dir_mask_selected: torch.Tensor       # (T*B, 9)
    requires_dir: torch.Tensor            # (T*B,) bool

class RolloutBuffer:
    """A buffer to store rollouts for training."""
    def __init__(self, num_steps: int, num_envs: int, obs_example: Dict[str, np.ndarray], num_verbs: int,
                 device: torch.device) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.num_verbs = num_verbs

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
        self._advantages_np = None
        self._returns_np = None

    def add(self, obs: Dict[str, np.ndarray], action_verb: np.ndarray, action_dir: np.ndarray, old_logprob: np.ndarray,
            value: np.ndarray, reward: np.ndarray, done: np.ndarray, verb_mask: np.ndarray,
            dir_mask_selected: np.ndarray, requires_dir: np.ndarray) -> None:
        """Add a rollout step to the buffer."""
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
        """Compute generalized advantage estimation (GAE)."""
        t = len(self.rewards)
        b = self.rewards[0].shape[0]
        advantages = np.zeros((t, b), dtype=np.float32)
        lastgaelam = np.zeros((b,), dtype=np.float32)
        for t in reversed(range(t)):
            nextnonterminal = 1.0 - self.dones[t].astype(np.float32)
            next_values = last_value if t == t - 1 else self.value[t + 1]
            float_rewards = self.rewards[t].astype(np.float32)
            delta = float_rewards + gamma * next_values * nextnonterminal - self.value[t].astype(np.float32)
            lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam

        returns = advantages + np.asarray(self.value, dtype=np.float32)
        self._advantages_np = advantages
        self._returns_np = returns

    def to_batches(self) -> RolloutBatch:
        """Convert the rollout buffer to a batch."""
        def _stack_flat(key: str) -> torch.Tensor:
            arr = np.stack(self.obs[key], axis=0)  # (T,B,...)
            t, b = arr.shape[:2]
            arr = arr.reshape(t * b, *arr.shape[2:])
            return torch.as_tensor(arr, device=self.device)

        obs_torch: Dict[str, torch.Tensor] = {k: _stack_flat(k) for k in self.obs.keys()}

        def _stack1(lst: List[np.ndarray]) -> torch.Tensor:
            arr = np.stack(lst, axis=0)  # (T,B,...)
            t, b = arr.shape[:2]
            arr = arr.reshape(t * b, *arr.shape[2:])
            return torch.as_tensor(arr, device=self.device)

        action_verb = _stack1(self.action_verb).long()
        action_dir = _stack1(self.action_dir).long()
        old_logprob = _stack1(self.old_logprob).float()
        value = _stack1(self.value).float()

        advantages_np = getattr(self, "_advantages_np")
        returns_np = getattr(self, "_returns_np")
        advantages = torch.as_tensor(advantages_np.reshape(-1), device=self.device)
        returns = torch.as_tensor(returns_np.reshape(-1), device=self.device)

        verb_mask = _stack1(self.verb_mask).bool()
        dir_mask_selected = _stack1(self.dir_mask_selected).bool()
        requires_dir = _stack1(self.requires_dir).bool().squeeze(-1)

        return RolloutBatch(obs=obs_torch, action_verb=action_verb, action_dir=action_dir, old_logprob=old_logprob,
                            value=value, advantages=advantages, returns=returns, verb_mask=verb_mask,
                            dir_mask_selected=dir_mask_selected, requires_dir=requires_dir)

# -

class PeriodicCheckpointCallback:
    """Save model checkpoints every `save_every` timesteps."""
    def __init__(self, save_every: int, save_dir: Path, model_name: str) -> None:
        self.save_every = int(save_every)
        self.save_dir = Path(save_dir)
        self.model_name = model_name
        self.next_save_step = self.save_every
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(self, steps, model) -> bool:
        """Saves the model after a certain number of steps."""
        if steps >= self.next_save_step:
            path = self.save_dir / f"{self.model_name}_{steps}.zip"
            model.save(str(path))
            self.next_save_step += self.save_every

class InfoCountsLogger:
    """Log counts of various info keys during training."""
    def __init__(self, log_every: int, logger) -> None:
        self.logger = logger
        self._log_every = log_every
        self._next_log_step = self._log_every
        self._short_log_every = 10_000
        self._next_short_log_step = self._short_log_every
        self._emitted = set()
        self._counters = {}
        self._values = {}
        self._booleans = {}
        self._averages = {}
        self._averages_short = {}

    def _build_dict(self, d, i):
        result = {}
        for k, v in d.items():
            presence = '_' + k
            if presence in d:
                if d[presence][i]:
                    if isinstance(v, dict):
                        result[k] = self._build_dict(v, i)
                    else:
                        result[k] = v[i]
        return result

    def on_step(self, infos) -> bool:
        """Process a batch of info dictionaries."""
        count = len(next(iter(infos.values())))
        for i in range(count):
            info = self._build_dict(infos, i)
            if not info:
                continue

            state : yndf.NethackState = info.get("state", None)
            if state is not None:
                self._add_boolean("metrics/idle-moves", state.idle_action)

            if (total_reward := info.get("total_reward", None)) is not None:
                self._averages_short.setdefault("rollout/ep_rew_mean", []).append(total_reward)

            if (total_steps := info.get("total_steps", None)) is not None:
                self._averages_short.setdefault("metrics/total_steps", []).append(total_steps)

            if (ending := info.get("ending", None)) is not None:
                self._counters.setdefault("endings", []).append(ending)
                if state is not None:
                    self._averages.setdefault("metrics/depth", []).append(state.player.depth)
                    self._averages.setdefault("metrics/time", []).append(state.time)
                    self._averages.setdefault("metrics/score", []).append(state.player.score)
                    self._averages.setdefault("metrics/level", []).append(state.player.level)

                    self._averages.setdefault(f"times/{ending}", []).append(state.time)

            if (rewards := info.get("rewards", None)) is not None:
                for name, value in rewards.items():
                    key = f"rewards/{name}"
                    self._values[key] = self._values.get(key, 0.0) + value

        return True

    def _add_boolean(self, key: str, value: Optional[bool]) -> None:
        if value is not None:
            v = self._booleans.get(key, (0, 0))
            self._booleans[key] = (v[0] + (1 if value else 0), v[1] + 1)


    def should_emit_short(self, steps):
        """Whether we should emit short logs."""
        return self._next_short_log_step <= steps

    def should_emit_long(self, steps):
        """Whether we should emit long logs."""
        return self._next_log_step <= steps

    def emit_short_running(self, steps) -> bool:
        """Emit short logs."""
        self._emit_averages(steps, self._averages_short, None, None)
        self._next_short_log_step += self._short_log_every

    def emit_long_running(self, steps) -> bool:
        """Emit long logs."""
        self._next_log_step += self._log_every

        curr_emitted = set()
        for base_name, values in self._counters.items():
            counter = Counter(values)
            for name, count in counter.items():
                key = f"{base_name}/{name}"
                self.logger.add_scalar(key, count / len(values) if values else 0, steps)
                curr_emitted.add(key)
                self._emitted.add(key)

        self._emit_averages(steps, self._averages, curr_emitted, self._emitted)

        for key, value in self._values.items():
            self.logger.add_scalar(key, value, steps)
            curr_emitted.add(key)
            self._emitted.add(key)

        for key, value in self._booleans.items():
            self.logger.add_scalar(key, value[0] / value[1] if value[1] > 0 else 0, steps)
            curr_emitted.add(key)
            self._emitted.add(key)

        missing = self._emitted - curr_emitted
        for key in missing:
            self.logger.add_scalar(key, 0, steps)

        # Reset for the next rollout window
        for v in self._counters.values():
            v.clear()

        for k in self._values:
            self._values[k] = 0.0

        for k in self._booleans:
            self._booleans[k] = (0, 0)

    def _emit_averages(self, steps, averages, emitted, all_emitted):
        for name, values in averages.items():
            if values:
                avg = sum(values) / len(values)
                self.logger.add_scalar(name, avg, steps)
                if emitted:
                    emitted.add(name)
                if all_emitted:
                    all_emitted.add(name)

        for value in averages.values():
            value.clear()


# -------------------------
# Training
# -------------------------

@dataclass
class PPOArgs:
    """Parameters for PPO training."""
    env_id: str = "YenderFlow-v0"
    total_timesteps: int = 25_000_000
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
    seed: int = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    exp_name: str = "nethack"

    # logging / saving
    track_tb: bool = True
    model_name: str = "nethack"
    save_every: int = 100_000                 # checkpoint cadence in timesteps
    log_every: int | None = None              # if None, logs once per update at rollout end

    # batch geometry
    batch_size: int | None = None
    min_rollout_len: int = 64
    minibatch_size: int | None = 256


def _parse_args() -> PPOArgs:
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
    parser.add_argument("--exp-name", type=str, default=PPOArgs.exp_name)

    # new knobs
    parser.add_argument("--model-name", type=str, default=None,
                        help="Name used for checkpoints and TensorBoard logs (default: exp_name)")
    parser.add_argument("--save-every", type=int, default=PPOArgs.save_every,
                        help="Checkpoint cadence in timesteps for PeriodicCheckpointCallback")
    parser.add_argument("--log-every", type=int, default=None,
                        help="If set, InfoCountsLogger logs every N timesteps on rollout end; "
                             "default is once per update (num_envs * num_steps).")

    parser.add_argument("--no-tb", action="store_true", help="Disable TensorBoard logging")
    parser.add_argument("--device", type=str, default=PPOArgs.device)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Target total batch per update. If omitted, uses num_envs * num_steps.")
    parser.add_argument("--min-rollout-len", type=int, default=PPOArgs.min_rollout_len,
                        help="Minimum steps per env per update (floor).")
    parser.add_argument("--minibatch-size", type=int, default=PPOArgs.minibatch_size,
                        help="Desired minibatch size. If provided, num_minibatches will be derived.")

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
        exp_name=args_ns.exp_name,
        track_tb=not args_ns.no_tb,
        device=args_ns.device,
        batch_size=args_ns.batch_size,
        min_rollout_len=args_ns.min_rollout_len,
        minibatch_size=args_ns.minibatch_size,
        model_name=(args_ns.model_name or args_ns.exp_name),
        save_every=args_ns.save_every,
        log_every=args_ns.log_every,
    )
    return args

def train(args : PPOArgs) -> None:
    """Train the PPO agent."""
    if args.seed is not None:
        _set_seed(args.seed)
    device = torch.device(args.device)

    # Create vectorized envs
    envs = _create_envs(args)

    # Reset and get an initial observation to infer shapes
    obs_np, _ = envs.reset()
    verb_mask_np, dir_mask_np = get_action_masks_batch(envs)
    num_verbs = verb_mask_np.shape[1]
    num_directions = dir_mask_np.shape[2]
    LOG.info("Num verbs: %d, Directions: %d", num_verbs, num_directions)

    policy = NethackPolicy(num_verbs=num_verbs, glyph_vocab_size=yndf.GLYPH_MAX).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)

    writer: SummaryWriter | None = None
    if args.track_tb:
        logdir = os.path.join("logs", args.model_name)
        Path(logdir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=logdir)
        LOG.info("TensorBoard logging to: %s", logdir)

    # Derive invariant-batch rollout sizes
    num_steps, num_minibatches, mb_size, batch_target, batch_actual = derive_invariant_batch(args)
    args.num_steps = num_steps
    args.num_minibatches = num_minibatches

    LOG.info(
        "Invariant batch config: target=%d -> actual=%d = %d envs * %d steps; "
        "minibatch_size=%d; num_minibatches=%d",
        batch_target, batch_actual, args.num_envs, args.num_steps, mb_size, args.num_minibatches
    )

    # Rollout buffer
    rb = RolloutBuffer(args.num_steps, args.num_envs, obs_example=obs_np, num_verbs=num_verbs, device=device)

    # ---- set up SB3-style callbacks via adapters ----
    global_step = 0
    def _get_step() -> int:
        return global_step

    model_adapter = ModelSaver(policy, optimizer, args, get_step=_get_step)

    checkpoints = PeriodicCheckpointCallback(save_every=int(args.save_every), save_dir=Path("models"),
                                             model_name=args.model_name)
    info_logger = InfoCountsLogger(100_000, writer)

    start_time = time.time()
    next_obs = obs_np

    progress = tqdm(total=args.total_timesteps)
    while global_step < args.total_timesteps:
        # Collect a rollout of num_steps
        t = 0
        while t < args.num_steps:
            try:
                # Fetch masks (this may also fail if a worker just died)
                verb_mask_np, dir_mask_np = get_action_masks_batch(envs)  # (B,A), (B,A,9)

                # Convert obs + masks to torch
                obs_t: Dict[str, torch.Tensor] = {k: torch.as_tensor(next_obs[k], device=device)
                                                for k in next_obs}
                verb_mask = torch.as_tensor(verb_mask_np, device=device, dtype=torch.bool)
                dir_mask = torch.as_tensor(dir_mask_np, device=device, dtype=torch.bool)

                with torch.no_grad():
                    act_verb_t, act_dir_t, logprob_t, value_t, _, _ = policy.get_action_and_value(
                        obs_t, verb_mask, dir_mask, action_verb=None, action_dir=None, deterministic=False
                    )
                    act_verb = act_verb_t.cpu().numpy()
                    act_dir = act_dir_t.cpu().numpy()
                    logprob = logprob_t.cpu().numpy()
                    value = value_t.cpu().numpy()
                    batch_idx = np.arange(args.num_envs)
                    dir_mask_selected = dir_mask_np[batch_idx, act_verb, :]  # (B,9)
                    requires_dir = dir_mask_selected.any(axis=1)

                env_actions = pack_action_batch(act_verb, act_dir)

                # ---- The only place we advance counters is AFTER a successful step ----
                next_obs, rewards, terminations, truncations, infos = envs.step(env_actions)

                global_step += args.num_envs
                progress.update(args.num_envs)

                dones = np.logical_or(terminations, truncations)
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

                # episodic logging (unchanged)
                if writer is not None and isinstance(infos, list):
                    for info in infos:
                        ep = info.get("episode")
                        if ep is not None:
                            writer.add_scalar("charts/episodic_return", ep["r"], global_step)
                            writer.add_scalar("charts/episodic_length", ep["l"], global_step)

                info_logger.on_step(infos)
                obs_np = next_obs  # store for next add()
                t += 1  # we successfully collected one step across all envs

            except (EOFError, BrokenPipeError) as exc:
                LOG.exception("Vector env step failed (%s). Recovering and restarting rollout...", type(exc).__name__)
                envs, next_obs = _recover_envs_and_restart_rollout(args, envs)

                # discard current rollout and restart inner loop
                rb = RolloutBuffer(args.num_steps, args.num_envs, obs_example=next_obs,
                                num_verbs=num_verbs, device=device)
                obs_np = next_obs
                t = 0
                continue

            except Exception as exc:  # If you prefer to catch only env transport errors, remove this.
                # You may want to re-raise logic/assertion errors to uncover bugs.
                LOG.exception("Unexpected error during rollout: %s", exc)
                raise

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

        rb.compute_gae(last_value, gamma=args.gamma, gae_lambda=args.gae_lambda)
        batch = rb.to_batches()

        # Normalize advantages
        adv = batch.advantages
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        # PPO update
        bsz = batch.action_verb.shape[0]  # T*B
        minibatch_size = bsz // args.num_minibatches
        assert bsz % args.num_minibatches == 0, f"Batch {bsz} not divisible by num_minibatches {args.num_minibatches}"
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
                v_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()

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

        if checkpoints.next_save_step <= global_step:
            checkpoints.save(global_step, model_adapter)

        if info_logger.should_emit_short(global_step):
            info_logger.emit_short_running(global_step)

        if info_logger.should_emit_long(global_step):
            info_logger.emit_long_running(global_step)

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

    envs.close()
    progress.close()
    if writer is not None:
        writer.close()

def _recover_envs_and_restart_rollout(args, envs):
    """Close broken envs, recreate, reset, return (envs, next_obs)."""
    try:
        envs.close()
    except Exception:  # pylint: disable=broad-except
        pass

    # If you want unique RNG streams after a crash, pass a reseed offset into your thunks.
    # For example, change make_env_thunk to accept seed_offset and do: base_seed + seed_offset + i
    envs = _create_envs(args)
    next_obs, _ = envs.reset()
    LOG.warning("Recovered vector envs. Discarded current rollout.")
    return envs, next_obs

def _create_envs(args):
    if args.num_envs > 1:
        envs = AsyncVectorEnv([_make_env_thunk(args.env_id, args.seed, i) for i in range(args.num_envs)])
    else:
        envs = SyncVectorEnv([_make_env_thunk(args.env_id, args.seed, 0)])
    return envs

if __name__ == "__main__":
    train(_parse_args())
