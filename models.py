from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Directions = 9  # 8 compass + HERE/self

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
        # We'll infer vector_fields dim and build a fusion layer on first call if dims mismatch
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32 + 32 + 32 + 64, self.trunk_hidden_dim),  # [map]64 + [agent]32 + [wave]32 + [vec]32 + [search]64
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
            nn.Linear(self.trunk_hidden_dim // 2, Directions),
        )
        self.value_head = nn.Linear(self.trunk_hidden_dim, 1)

    @staticmethod
    def _normalize_agent_coords(agent_yx: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # agent_yx is (B,2) with [y, x] in [0..H-1], [0..W-1]
        y = agent_yx[..., 0] / max(H - 1, 1)
        x = agent_yx[..., 1] / max(W - 1, 1)
        # scale to [-1, 1]
        y = y * 2.0 - 1.0
        x = x * 2.0 - 1.0
        return torch.stack([y, x], dim=-1)

    def _encode_map(
        self, glyphs: torch.Tensor, visited_mask: torch.Tensor, agent_yx: torch.Tensor
    ) -> torch.Tensor:
        """Encode map with glyph embeddings + visited + optional agent one-hot; returns pooled (B, 64)."""
        B, H, W = glyphs.shape
        # Safety: modulo to embedding size; clamp negatives
        glyphs = glyphs % self.glyph_vocab_size
        emb = self.glyph_embedding(glyphs)  # (B, H, W, E)
        emb = emb.permute(0, 3, 1, 2)  # (B, E, H, W)

        visited = visited_mask.unsqueeze(1).float()  # (B,1,H,W)

        if self.use_agent_onehot:
            agent_map = torch.zeros((B, 1, H, W), dtype=emb.dtype, device=emb.device)
            ay = agent_yx[..., 0].long().clamp_(0, H - 1)
            ax = agent_yx[..., 1].long().clamp_(0, W - 1)
            agent_map[torch.arange(B, device=emb.device), 0, ay, ax] = 1.0
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
        glyphs = obs["glyphs"].long()
        visited = obs["visited_mask"].float()
        agent_yx = obs["agent_yx"].float()
        wavefront = obs["wavefront"].float()
        vector_fields = obs["vector_fields"].float()
        search_scores = obs["search_scores"].float()

        B, H, W = glyphs.shape
        map_feat = self._encode_map(glyphs, visited, agent_yx)  # (B,64)

        agent_norm = self._normalize_agent_coords(agent_yx, H, W)
        agent_feat = self.fc_agent(agent_norm)  # (B,32)

        wave_feat = self.fc_wavefront(wavefront)  # (B,32)

        self._maybe_rebuild_vector_layers(vector_fields)
        vec_feat = self.fc_vector_fields_2(self.fc_vector_fields_1(vector_fields))  # (B,32)

        search_flat = search_scores.view(B, -1)
        search_feat = self.fc_search(search_flat)  # (B,64)

        fused = torch.cat([map_feat, agent_feat, wave_feat, vec_feat, search_feat], dim=-1)  # (B,224)
        trunk = self.fusion(fused)  # (B, hidden)
        return trunk

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trunk = self.encode_trunk(obs)
        verb_logits = self.verb_head(trunk)  # (B, A)
        value = self.value_head(trunk).squeeze(-1)  # (B,)
        # Direction logits require a chosen verb; we return a 'template' projected with a neutral verb embedding (zeros)
        # to enable optional analysis; normal action selection should call get_action_and_value(...)
        neutral_emb = torch.zeros((trunk.size(0), self.verb_embed_dim), device=trunk.device, dtype=trunk.dtype)
        dir_logits = self.dir_head(torch.cat([trunk, neutral_emb], dim=-1))  # (B, 9)
        return verb_logits, dir_logits, value

    def get_action_and_value(
        self,
        obs: Dict[str, torch.Tensor],
        verb_mask: torch.Tensor,          # (B, A) bool
        dir_mask_per_verb: torch.Tensor,  # (B, A, 9) bool; row all-False means 'no direction arg' for that verb
        action_verb: Optional[torch.Tensor] = None,   # (B,)
        action_dir: Optional[torch.Tensor] = None,    # (B,)
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or evaluate) actions and return logprobs and value.

        Returns:
            action_verb: (B,)
            action_dir:  (B,)  (-1 where no direction required)
            logprob_sum:(B,)  log pi(a) + [log pi(d | a) if required]
            value:      (B,)
            entropy_verb:(B,)
            entropy_dir: (B,)  (0 where dir not required)
        """
        trunk = self.encode_trunk(obs)  # (B, hidden)

        # Verb distribution (masked)
        verb_logits = self.verb_head(trunk)  # (B, A)
        verb_logits = _masked_logits(verb_logits, verb_mask)
        verb_dist = torch.distributions.Categorical(logits=verb_logits)

        if action_verb is None:
            if deterministic:
                action_verb = torch.argmax(verb_logits, dim=-1)
            else:
                action_verb = verb_dist.sample()
        logprob_verb = verb_dist.log_prob(action_verb)
        entropy_verb = verb_dist.entropy()

        # Direction distribution conditioned on chosen verb
        B = trunk.shape[0]
        verb_emb = self.verb_embedding(action_verb).detach()  # stop-gradient trick
        dir_logits = self.dir_head(torch.cat([trunk, verb_emb], dim=-1))  # (B, 9)

        # Gather the direction mask row for each chosen verb
        batch_idx = torch.arange(B, device=trunk.device)
        dir_mask = dir_mask_per_verb[batch_idx, action_verb]  # (B, 9) bool
        requires_dir = dir_mask.any(dim=-1)  # (B,)

        # For samples that require a direction, create a masked categorical
        dir_logits_masked = _masked_logits(dir_logits, dir_mask)
        dir_dist = torch.distributions.Categorical(logits=dir_logits_masked)

        if action_dir is None:
            # If no direction required, set to -1 (sentinel); else sample/argmax
            if deterministic:
                sampled = torch.argmax(dir_logits_masked, dim=-1)
            else:
                sampled = dir_dist.sample()
            action_dir = torch.where(requires_dir, sampled, torch.full_like(sampled, -1))

        # Logprob and entropy: zero where dir not required
        logprob_dir = torch.where(
            requires_dir,
            dir_dist.log_prob(action_dir.clamp(min=0)),  # clamp harmless when requires_dir False
            torch.zeros_like(logprob_verb),
        )
        entropy_dir = torch.where(
            requires_dir,
            dir_dist.entropy(),
            torch.zeros_like(entropy_verb),
        )

        logprob_sum = logprob_verb + logprob_dir
        value = self.value_head(trunk).squeeze(-1)
        return action_verb, action_dir, logprob_sum, value, entropy_verb, entropy_dir

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        verb_mask: torch.Tensor,          # (B, A) bool at the time actions were taken
        dir_mask_selected: torch.Tensor,  # (B, 9) bool (mask row for the chosen verb at that time)
        action_verb: torch.Tensor,        # (B,)
        action_dir: torch.Tensor,         # (B,) -1 if no dir required
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute logprobs and entropies for a batch of given actions (for PPO updates).

        Returns:
            logprob_sum: (B,)
            value:       (B,)
            entropy_verb:(B,)
            entropy_dir: (B,) (0 where not required)
            requires_dir:(B,) bool
        """
        trunk = self.encode_trunk(obs)

        # Verb
        verb_logits = self.verb_head(trunk)
        verb_logits = _masked_logits(verb_logits, verb_mask)
        verb_dist = torch.distributions.Categorical(logits=verb_logits)
        logprob_verb = verb_dist.log_prob(action_verb)
        entropy_verb = verb_dist.entropy()

        # Dir conditioned on given verb
        verb_emb = self.verb_embedding(action_verb).detach()
        dir_logits = self.dir_head(torch.cat([trunk, verb_emb], dim=-1))  # (B,9)

        requires_dir = dir_mask_selected.any(dim=-1)
        dir_logits_masked = _masked_logits(dir_logits, dir_mask_selected)
        dir_dist = torch.distributions.Categorical(logits=dir_logits_masked)

        logprob_dir = torch.where(
            requires_dir,
            dir_dist.log_prob(action_dir.clamp(min=0)),
            torch.zeros_like(logprob_verb),
        )
        entropy_dir = torch.where(
            requires_dir,
            dir_dist.entropy(),
            torch.zeros_like(entropy_verb),
        )

        logprob_sum = logprob_verb + logprob_dir
        value = self.value_head(trunk).squeeze(-1)
        return logprob_sum, value, entropy_verb, entropy_dir, requires_dir
