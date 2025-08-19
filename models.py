import random
import time
from typing import Any, Callable, Dict, Optional, Tuple

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from yndf import GLYPH_MAX
from yndf.wrapper_actions import DIRECTIONS, VERBS

DIRECTION_COUNT = len(DIRECTIONS)

def _masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply a boolean mask to logits by setting invalid positions to a large negative value.

    Args:
        logits: (..., K) float tensor of unnormalized logits.
        mask:   (..., K) boolean tensor, True where valid, False where invalid.
    Returns:
        logits_masked with invalid entries set to a large negative number.
    """
    # We use a finite sentinel instead of -inf to avoid NaNs in softmax on some accelerators.
    invalid = ~mask
    if invalid.any():
        min_val = torch.finfo(logits.dtype).min if logits.dtype.is_floating_point else -1e9
        logits = logits.masked_fill(invalid, min_val)
    return logits

class NethackPolicy(nn.Module):
    """Shared encoder + (verb head, verb-conditioned direction head) + value head.

    Observation dict expected (batch-first):
      - glyphs: Long[B, H, W]  (tile glyph ids; will be modulo'd into glyph_vocab_size to be safe)
      - visited_mask: Float[B, H, W] in {0,1}
      - agent_yx: Float[B, 2] (y, x) in absolute grid coords; normalized internally to [-1, 1]
      - wavefront: Float[B, 8] (directional features; e.g., rays/occupancy/LOS cues)
      - vector_fields: Float[B, F] (scalar global features)
      - search_scores: Float[B, 7, 7]
      - prev_action:   Long[B, 2]  (prev_verb, prev_dir). Accepts sentinel (-1) or “last index” coding.
      - message_text:  Byte/Long[B, L] (ascii bytes 255; padded to fixed L)

    The model does not assume fixed H, W but is tuned to ~21x79.
    """

    def __init__(
        self,
        num_verbs: int,
        glyph_vocab_size: int = 6000,
        glyph_embed_dim: int = 32,
        trunk_hidden_dim: int = 256,
        verb_embed_dim: int = 32,
        use_agent_onehot: bool = True,
    ) -> None:
        super().__init__()
        self.num_verbs = num_verbs
        self.glyph_vocab_size = glyph_vocab_size
        self.glyph_embed_dim = glyph_embed_dim
        self.trunk_hidden_dim = trunk_hidden_dim
        self.verb_embed_dim = verb_embed_dim
        self.use_agent_onehot = use_agent_onehot

        # --- Map encoder ---
        self.glyph_embedding = nn.Embedding(self.glyph_vocab_size, self.glyph_embed_dim)
        # Conv expects channels-first; we'll build channels as [glyph_embed, visited, (agent_onehot)]
        # Start with a light CNN; we rely on global average pooling rather than strided downsamples
        self.conv1 = nn.Conv2d(self.glyph_embed_dim + (1 + int(self.use_agent_onehot)), 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # --- Small heads for non-spatial features ---
        self.fc_search = nn.Sequential(
            nn.Linear(7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.fc_wavefront = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        # vector_fields size is not hard-coded; we infer at first forward
        self.fc_vector_fields_1 = nn.Linear(7, 32)  # default, will be reshaped if needed on first call
        self.fc_vector_fields_2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_agent = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        # --- Fusion MLP to trunk ---
        #  We'll infer vector_fields dim and build a fusion layer on first call if dims mismatch
        # [map]64 + [agent]32 + [wave]32 + [vec]32 + [search]64 + [prev_action]32 + [msg]32
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32 + 32 + 32 + 64 + 32 + 32, self.trunk_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.trunk_hidden_dim, self.trunk_hidden_dim),
            nn.ReLU(inplace=True),
        )

        # --- Heads ---
        self.verb_head = nn.Linear(self.trunk_hidden_dim, self.num_verbs)
        self.verb_embedding = nn.Embedding(self.num_verbs, self.verb_embed_dim)
        self.dir_head = nn.Sequential(
            nn.Linear(self.trunk_hidden_dim + self.verb_embed_dim, self.trunk_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.trunk_hidden_dim // 2, DIRECTION_COUNT),
        )
        self.value_head = nn.Linear(self.trunk_hidden_dim, 1)
        self.prev_verb_emb = nn.Embedding(self.num_verbs + 1, 16)
        self.prev_dir_emb  = nn.Embedding(DIRECTION_COUNT + 1, 16)
        self.fc_prev_action = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.msg_byte_embed = nn.Embedding(255, 16)
        self.msg_cnn = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1),
        )

        self.msg_proj = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )

        self.rnn = nn.LSTM(
            input_size=self.trunk_hidden_dim,
            hidden_size=self.trunk_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

    def init_rnn_state(self, batch_size: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero (h, c) with shapes (num_layers, B, H)."""
        dev = torch.device(device)
        h = torch.zeros(1, batch_size, self.trunk_hidden_dim, device=dev)
        c = torch.zeros(1, batch_size, self.trunk_hidden_dim, device=dev)
        return h, c

    @staticmethod
    def _mask_rnn_state(
        state: tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """mask: Bool/Float [B]. 1=keep, 0=reset. Returns (h*, c*) with shape [L=1, B, H]."""
        if mask is None:
            return state
        h, c = state  # both [1, B, H]
        m = mask.float()
        if m.dim() == 0:
            # single element -> shape [1,1,1]
            m = m.view(1, 1, 1)
        elif m.dim() == 1:
            # [B] -> [1,B,1]
            m = m.view(1, -1, 1)
        elif m.dim() == 2:
            # [1,B] -> [1,B,1]
            m = m.unsqueeze(-1)
        else:
            # fallback: take batch as last known dim
            m = m.view(1, m.shape[-1], 1)
        return h * m, c * m

    def _apply_rnn(self,
                trunk_step: torch.Tensor,
                rnn_state: Optional[tuple[torch.Tensor, torch.Tensor]],
                rnn_mask: Optional[torch.Tensor]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        trunk_step: (B, H) fused features for the *current* step.
        rnn_state:  (h, c) or None → zeros.
        rnn_mask:   [B] 1=keep state, 0=reset this batch element.
        Returns: (B, H) updated features and next (h, c).
        """
        B = trunk_step.size(0)
        if rnn_state is None:
            h0, c0 = self.init_rnn_state(B, trunk_step.device)
        else:
            h0, c0 = self._mask_rnn_state(rnn_state, rnn_mask)
        # Add time dimension (sequence length = 1)
        out, (h1, c1) = self.rnn(trunk_step.unsqueeze(1), (h0, c0))  # out: [B,1,H]
        return out.squeeze(1), (h1, c1)

    @staticmethod
    def _normalize_agent_coords(agent_yx: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # agent_yx is (B,2) with [y, x] in [0..H-1], [0..W-1]
        y = agent_yx[..., 0] / max(height - 1, 1)
        x = agent_yx[..., 1] / max(width - 1, 1)
        # scale to [-1, 1]
        y = y * 2.0 - 1.0
        x = x * 2.0 - 1.0
        return torch.stack([y, x], dim=-1)

    def _encode_map(
        self, glyphs: torch.Tensor, visited_mask: torch.Tensor, agent_yx: torch.Tensor
    ) -> torch.Tensor:
        """Encode map with glyph embeddings + visited + optional agent one-hot; returns pooled (B, 64)."""
        batch, height, width = glyphs.shape
        # Safety: modulo to embedding size; clamp negatives
        glyphs = glyphs % self.glyph_vocab_size
        emb = self.glyph_embedding(glyphs)  # (B, H, W, E)
        emb = emb.permute(0, 3, 1, 2)  # (B, E, H, W)

        visited = visited_mask.unsqueeze(1).float()  # (B,1,H,W)

        if self.use_agent_onehot:
            agent_map = torch.zeros((batch, 1, height, width), dtype=emb.dtype, device=emb.device)
            ay = agent_yx[..., 0].long().clamp_(0, height - 1)
            ax = agent_yx[..., 1].long().clamp_(0, width - 1)
            agent_map[torch.arange(batch, device=emb.device), 0, ay, ax] = 1.0
            x = torch.cat([emb, visited, agent_map], dim=1)
        else:
            x = torch.cat([emb, visited], dim=1)

        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        # Global average pooling over HxW
        x = F.adaptive_avg_pool2d(x, output_size=1).squeeze(-1).squeeze(-1)  # (B, 64)
        return x

    def _maybe_rebuild_vector_layers(self, vector_fields: torch.Tensor) -> None:
        # If vector_fields dim != 7 (default), rebuild the first layer to match
        in_dim = vector_fields.shape[-1]
        if self.fc_vector_fields_1.in_features != in_dim:
            self.fc_vector_fields_1 = nn.Linear(in_dim, 32).to(vector_fields.device)
            # Keep fc_vector_fields_2 as defined (it takes 32 in)

    def encode_trunk(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode the trunk of the model using the provided observations."""
        glyphs = obs["glyphs"].long()
        visited = obs["visited_mask"].float()
        agent_yx = obs["agent_yx"].float()
        wavefront = obs["wavefront"].float()
        vector_fields = obs["vector_fields"].float()
        search_scores = obs["search_scores"].float()
        prev_action = obs["prev_action"].long()
        message_text = obs["message"]

        batch, height, width = glyphs.shape
        map_feat = self._encode_map(glyphs, visited, agent_yx)  # (B,64)

        agent_norm = self._normalize_agent_coords(agent_yx, height, width)
        agent_feat = self.fc_agent(agent_norm)  # (B,32)

        wave_feat = self.fc_wavefront(wavefront)  # (B,32)

        self._maybe_rebuild_vector_layers(vector_fields)
        vec_feat = self.fc_vector_fields_2(self.fc_vector_fields_1(vector_fields))  # (B,32)

        search_flat = search_scores.view(batch, -1)
        search_feat = self.fc_search(search_flat)  # (B,64)

        # Accept either: negative sentinel (-1) or "last index" sentinel.
        prev_verb_raw = prev_action[..., 0]
        prev_dir_raw  = prev_action[..., 1]
        prev_verb_idx = torch.where(prev_verb_raw >= 0, prev_verb_raw,
                                    torch.full_like(prev_verb_raw, self.num_verbs))
        prev_dir_idx  = torch.where(prev_dir_raw >= 0, prev_dir_raw,
                                    torch.full_like(prev_dir_raw, DIRECTION_COUNT))
        prev_verb_vec = self.prev_verb_emb(prev_verb_idx)  # (B,16)
        prev_dir_vec  = self.prev_dir_emb(prev_dir_idx)    # (B,16)
        prev_feat = self.fc_prev_action(torch.cat([prev_verb_vec, prev_dir_vec], dim=-1))  # (B,32)

        msg_bytes = message_text.long()                    # (B, L)
        msg_emb = self.msg_byte_embed(msg_bytes)           # (B, L, 16)
        msg_emb = msg_emb.permute(0, 2, 1).contiguous()    # (B, 16, L)
        msg_feat_text = self.msg_cnn(msg_emb).squeeze(-1)  # (B, 64)
        msg_feat = self.msg_proj(msg_feat_text)             # (B, 32)

        fused = torch.cat([map_feat, agent_feat, wave_feat, vec_feat, search_feat, prev_feat, msg_feat], dim=-1)
        trunk = self.fusion(fused)  # (B, hidden)
        return trunk

    def forward(self,
                obs: Dict[str, torch.Tensor],
                rnn_state: Optional[tuple[torch.Tensor, torch.Tensor]],
                rnn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for encoding observations."""
        trunk = self.encode_trunk(obs)
        trunk, next_state = self._apply_rnn(trunk, rnn_state, rnn_mask)
        verb_logits = self.verb_head(trunk)
        value = self.value_head(trunk).squeeze(-1)
        neutral_emb = torch.zeros((trunk.size(0), self.verb_embed_dim), device=trunk.device, dtype=trunk.dtype)
        dir_logits = self.dir_head(torch.cat([trunk, neutral_emb], dim=-1))
        return verb_logits, dir_logits, value, next_state

    def get_action_and_value(
            self,
            obs: Dict[str, torch.Tensor],
            verb_mask: torch.Tensor,
            dir_mask_per_verb: torch.Tensor,
            rnn_state: Optional[tuple[torch.Tensor, torch.Tensor]],
            rnn_mask: Optional[torch.Tensor] = None,
            action_verb: Optional[torch.Tensor] = None,
            action_dir: Optional[torch.Tensor] = None,
            deterministic: bool = False,
        ):
        trunk = self.encode_trunk(obs)
        trunk, next_state = self._apply_rnn(trunk, rnn_state, rnn_mask)

        verb_logits = self.verb_head(trunk)
        verb_logits = _masked_logits(verb_logits, verb_mask)
        verb_dist = torch.distributions.Categorical(logits=verb_logits)

        if action_verb is None:
            action_verb = torch.argmax(verb_logits, dim=-1) if deterministic else verb_dist.sample()
        logprob_verb = verb_dist.log_prob(action_verb)
        entropy_verb = verb_dist.entropy()

        batch = trunk.shape[0]
        verb_emb = self.verb_embedding(action_verb).detach()
        dir_logits = self.dir_head(torch.cat([trunk, verb_emb], dim=-1))

        batch_idx = torch.arange(batch, device=trunk.device)
        dir_mask = dir_mask_per_verb[batch_idx, action_verb]
        requires_dir = dir_mask.any(dim=-1)

        dir_logits_masked = _masked_logits(dir_logits, dir_mask)
        dir_dist = torch.distributions.Categorical(logits=dir_logits_masked)

        if action_dir is None:
            sampled = torch.argmax(dir_logits_masked, dim=-1) if deterministic else dir_dist.sample()
            action_dir = torch.where(requires_dir, sampled, torch.full_like(sampled, -1))

        logprob_dir = torch.where(requires_dir, dir_dist.log_prob(action_dir.clamp(min=0)),
                                  torch.zeros_like(logprob_verb))
        entropy_dir = torch.where(requires_dir, dir_dist.entropy(), torch.zeros_like(entropy_verb))

        logprob_sum = logprob_verb + logprob_dir
        value = self.value_head(trunk).squeeze(-1)
        return action_verb, action_dir, logprob_sum, value, entropy_verb, entropy_dir, next_state

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        verb_mask: torch.Tensor,
        dir_mask_selected: torch.Tensor,
        action_verb: torch.Tensor,
        action_dir: torch.Tensor,
        rnn_state: Optional[tuple[torch.Tensor, torch.Tensor]],
        rnn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        trunk = self.encode_trunk(obs)
        trunk, next_state = self._apply_rnn(trunk, rnn_state, rnn_mask)

        verb_logits = self.verb_head(trunk)
        verb_logits = _masked_logits(verb_logits, verb_mask)
        verb_dist = torch.distributions.Categorical(logits=verb_logits)
        logprob_verb = verb_dist.log_prob(action_verb)
        entropy_verb = verb_dist.entropy()

        verb_emb = self.verb_embedding(action_verb).detach()
        dir_logits = self.dir_head(torch.cat([trunk, verb_emb], dim=-1))

        requires_dir = dir_mask_selected.any(dim=-1)
        dir_logits_masked = _masked_logits(dir_logits, dir_mask_selected)
        dir_dist = torch.distributions.Categorical(logits=dir_logits_masked)

        logprob_dir = torch.where(requires_dir, dir_dist.log_prob(action_dir.clamp(min=0)), torch.zeros_like(logprob_verb))
        entropy_dir = torch.where(requires_dir, dir_dist.entropy(), torch.zeros_like(entropy_verb))

        logprob_sum = logprob_verb + logprob_dir
        value = self.value_head(trunk).squeeze(-1)
        return logprob_sum, value, entropy_verb, entropy_dir, requires_dir, next_state

# -------------------------
# SB3-callback compatibility adapters
# -------------------------

class ModelSaver:
    """
    Minimal object that exposes the attributes SB3 callbacks expect:
    - num_timesteps
    - save(path)
    - logger  (not used here)
    """
    def __init__(self, policy: nn.Module, optimizer, args_obj, get_step: Callable[[], int]):
        self._policy = policy
        self._optimizer = optimizer
        self._args = args_obj
        self._get_step = get_step

    @property
    def num_timesteps(self) -> int:
        """Number of timesteps."""
        return int(self._get_step())

    def save(self, path: str) -> None:
        """Save training state to the given path (extension can be .zip/.pt/etc.)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "policy_state_dict": self._policy.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "args": asdict(self._args),
            "num_timesteps": self.num_timesteps,
            "timestamp": int(time.time()),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state().tolist(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }

        torch.save(state, str(p))

    @staticmethod
    def load_checkpoint(path: str, device: str | torch.device = "cpu") -> dict[str, Any]:
        """Low-level: just read the checkpoint dict."""
        return torch.load(path, map_location=device)

    @staticmethod
    def load_as_inference(path: str,
                          policy_builder: Callable[[dict[str, Any]], nn.Module],
                          device: str | torch.device = "cpu",
                          n_envs: int = 1) -> "InferenceAdapter":
        """
        High-level: build a policy via `policy_builder(args_dict)`,
        load weights, and wrap with an SB3-like .predict().
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        args_dict: dict[str, Any] = ckpt.get("args", {})
        policy = policy_builder(args_dict).to(device)
        policy.load_state_dict(ckpt["policy_state_dict"])
        policy.eval()
        adapter = InferenceAdapter(policy, device=device)
        adapter.reset(n_envs=n_envs)
        return adapter


class InferenceAdapter:
    """SB3-like wrapper with recurrent state.
    - Maintains (h, c) across .predict() calls.
    - Accepts per-env dones to reset state rows.
    """
    def __init__(self, policy: nn.Module, device: str | torch.device = "cpu") -> None:
        self.policy = policy
        self.device = torch.device(device)
        self.rnn_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None  # (h, c) with shapes (1, B, H)
        self._num_envs: Optional[int] = None

    @staticmethod
    def _to_tensor(x: Any, device: torch.device, unsqueeze: bool = False) -> Any:
        """Recursively move obs/masks to a torch tensor on device."""

        result = None
        if torch.is_tensor(x):
            result = x.to(device)
            if unsqueeze:
                result = result.unsqueeze(0)
        elif isinstance(x, np.ndarray):
            result = torch.from_numpy(x).to(device)
            if unsqueeze:
                result = result.unsqueeze(0)
        elif isinstance(x, (list, tuple)):
            result = type(x)(InferenceAdapter._to_tensor(v, device, unsqueeze=unsqueeze) for v in x)
        elif isinstance(x, dict):
            result = {k: InferenceAdapter._to_tensor(v, device, unsqueeze=unsqueeze) for k, v in x.items()}
        else:
            result = torch.as_tensor(x, device=device)
            if unsqueeze:
                result = result.unsqueeze(0)

        return result

    def reset(self, n_envs: Optional[int] = None) -> None:
        """Reset the recurrent state. If n_envs is given, re-init to that batch size."""
        if n_envs is not None:
            self._num_envs = int(n_envs)
            self.rnn_state = self.policy.init_rnn_state(self._num_envs, self.device)
        else:
            self.rnn_state = None

    def reset_done(self, dones: Any) -> None:
        """Reset only rows where dones==True."""
        if self.rnn_state is None:
            return
        d = torch.as_tensor(dones, device=self.device).bool().view(-1)  # [B]
        if d.any():
            # rnn_mask 1=keep, 0=reset
            keep = (~d).float().view(1, -1, 1)  # [1,B,1]
            h, c = self.rnn_state
            self.rnn_state = (h * keep, c * keep)

    @staticmethod
    def _batch_size_from_obs(obs_t: Any) -> int:
        if isinstance(obs_t, dict):
            for v in obs_t.values():
                if torch.is_tensor(v) and v.dim() >= 1:
                    return int(v.shape[0])
        if torch.is_tensor(obs_t) and obs_t.dim() >= 1:
            return int(obs_t.shape[0])
        raise ValueError("Cannot infer batch size from obs.")

    @torch.inference_mode()
    def predict(self,
                obs: Any,
                deterministic: bool,
                action_masks: Tuple[torch.Tensor, torch.Tensor] | None,
                unsqueeze: bool,
                dones: Any | None = None,
                force_reset: bool = False) -> Tuple[Tuple[int, int], Any]:
        """Mimics stable baselines behavior."""
        obs_t = InferenceAdapter._to_tensor(obs, self.device, unsqueeze)

        # Infer batch size and ensure rnn_state exists and matches shape
        batch_size = self._batch_size_from_obs(obs_t)
        if self.rnn_state is None or force_reset:
            self._num_envs = batch_size
            self.rnn_state = self.policy.init_rnn_state(batch_size, self.device)
        else:
            # If batch size changed, re-init
            h, _ = self.rnn_state
            if h.size(1) != batch_size:
                self._num_envs = batch_size
                self.rnn_state = self.policy.init_rnn_state(batch_size, self.device)
        # Build rnn_mask: 1=keep, 0=reset
        if dones is None:
            rnn_mask = torch.ones(batch_size, device=self.device, dtype=torch.float32)
        else:
            d = torch.as_tensor(dones, device=self.device).bool().view(-1)
            rnn_mask = (~d).float()

        verb_mask_t: Optional[torch.Tensor] = None
        dir_mask_t: Optional[torch.Tensor] = None
        if action_masks is not None:
            verb_mask_t, dir_mask_t = action_masks
            verb_mask_t = InferenceAdapter._to_tensor(verb_mask_t, self.device, unsqueeze)
            dir_mask_t = InferenceAdapter._to_tensor(dir_mask_t, self.device, unsqueeze)

        ret = self.policy.get_action_and_value(
            obs_t,
            verb_mask_t,
            dir_mask_t,
            action_verb=None,
            action_dir=None,
            deterministic=deterministic,
            rnn_state=self.rnn_state,
            rnn_mask=rnn_mask,
        )

        # Support both 6-tuple (old) and 7-tuple (new with next_state)
        if isinstance(ret, tuple) and len(ret) == 7:
            act_verb_t, act_dir_t, _, _, _, _, next_state = ret
            self.rnn_state = next_state
        else:
            act_verb_t, act_dir_t, _, _, _, _ = ret  # keep prior state

        act_verb = act_verb_t.detach().cpu().item()
        act_dir = act_dir_t.detach().cpu().item()

        return (act_verb, act_dir), None


def load_model(model_path, device = None):
    """Loads a model with a stable-baselines like interface."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    def policy_builder(_: dict) -> torch.nn.Module:
        # pylint: disable=no-member
        return NethackPolicy(num_verbs=len(VERBS), glyph_vocab_size=GLYPH_MAX).to(device)

    return ModelSaver.load_as_inference(model_path, policy_builder, device=device)
