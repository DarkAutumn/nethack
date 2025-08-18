"""Wrapper to convert NLE observations to the one expected by the agent."""

import gymnasium as gym
import numpy as np

from yndf.nethack_level import GlyphLookupTable, PassableGlyphs
from yndf.nethack_state import NethackState

CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
DIRECTIONS = CARDINALS + DIAGONALS

FIELD_ENEMIES_PRESENT = 0
FIELD_IN_ROOM = 1
FIELD_IN_CORRIDOR = 2
FIELD_AT_DEAD_END = 3
FIELD_PERCENT_LEVEL_EXPLORED = 4
FIELD_FRONTIER_REACHABLE = 5
FIELD_HP_RATIO = 6
MAX_FIELD = 7


class NethackObsWrapper(gym.Wrapper):
    """Convert NLE observation → dict(glyphs, visited_mask, agent_yx)."""
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Dict(
            glyphs=self.env.observation_space["glyphs"],
            visited_mask=gym.spaces.Box(0, 1, shape=(21, 79), dtype=np.uint8),
            agent_yx=gym.spaces.Box(low=np.array([0, 0]), high=np.array([20, 78]), dtype=np.int16),
            wavefront=gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.int8),
            vector_fields=gym.spaces.Box(0, 1, shape=(MAX_FIELD, ), dtype=np.float32),
            search_scores=gym.spaces.Box(0, 1, shape=(7, 7), dtype=np.float32),
        )

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        wrapped_obs = self._wrap_observation(obs, info)
        return wrapped_obs, info

    def step(self, action):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        wrapped_obs = self._wrap_observation(obs, info)
        return wrapped_obs, reward, terminated, truncated, info

    def _wrap_observation(self, obs, info):
        state : NethackState = info["state"]

        glyphs = obs["glyphs"].astype(np.int16)
        for pet in np.argwhere(state.floor.pet_mask):
            glyphs[pet[0], pet[1]] = PassableGlyphs.S_room.value  # Remove pet glyphs


        agent_yx = np.array(state.player.position, dtype=np.int16)
        search_scores = self._get_padded_7x7(state.floor.search_score, state.player.position, r = 3)
        mult = (self._get_padded_7x7(state.floor.search_count, state.player.position, r = 3) < 22).astype(np.float32)
        search_scores *= mult

        return {
            "glyphs": glyphs,
            "visited_mask": state.floor.visited_mask.astype(np.uint8),
            "agent_yx": agent_yx,
            "wavefront": self._calculate_wavefront_hints(state),
            "vector_fields": self._get_vector_fields(state),
            "search_scores": search_scores,
        }

    def _get_vector_fields(self, state: NethackState) -> np.ndarray:
        """Get a 5x5xMAX_FIELD array of binary features around the player."""
        floor = state.floor
        fields = np.zeros((MAX_FIELD,), dtype=np.float32)

        fields[FIELD_ENEMIES_PRESENT] = 1.0 if floor.num_enemies > 0 else 0.0
        fields[FIELD_IN_ROOM] = 1.0 if floor.properties[state.player.position] & GlyphLookupTable.DUNGEON_FLOOR else 0.0
        fields[FIELD_IN_CORRIDOR] = 1.0 if floor.properties[state.player.position] & GlyphLookupTable.CORRIDOR else 0.0
        fields[FIELD_AT_DEAD_END] = 1.0 if floor.properties[state.player.position] & floor.DEAD_END else 0.0

        unexplored = np.sum((floor.properties & floor.UNSEEN_STONE) != 0)
        total = floor.glyphs.shape[0] * floor.glyphs.shape[1]
        fields[FIELD_PERCENT_LEVEL_EXPLORED] = 1.0 - (unexplored / total)

        # 0.5 if nearby frontier, 0.0 if not
        if floor.wavefront[state.player.position] < 12:
            # Close by reachable frontier
            fields[FIELD_FRONTIER_REACHABLE] = 1.0
        elif ((floor.properties & floor.FRONTIER) != 0).any():
            fields[FIELD_FRONTIER_REACHABLE] = 0.5

        fields[FIELD_HP_RATIO] = state.player.hp / state.player.hp_max if state.player.hp_max > 0 else 0.0

        return fields

    def _calculate_wavefront_hints(self, state: NethackState) -> np.ndarray:
        wf = state.floor.wavefront
        y, x = state.player.position
        curr = int(wf[y, x])  # avoid unsigned underflow
        hints = np.zeros(8, dtype=np.int8)

        for i, (dy, dx) in enumerate(DIRECTIONS):
            ny, nx = y + dy, x + dx
            if 0 <= ny < wf.shape[0] and 0 <= nx < wf.shape[1]:
                v = int(wf[ny, nx])
                hints[i] = 1 if v < curr else (-1 if v > curr else 0)
        return hints

    def _get_padded_7x7(self, arr: np.ndarray, pos: tuple[int, int], r: int) -> np.ndarray:
        y, x = pos
        ap = np.pad(arr, ((r, r), (r, r)), mode="constant")  # zero-pad
        return ap[y : y + 2*r + 1, x : x + 2*r + 1]
