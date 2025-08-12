"""The state of nethack at the current timestep.  All wrappers should interact with this instead of using
observation directly."""

import math
from typing import Optional, Tuple
from nle import nethack
import numpy as np

from yndf.movement import CLOSED_DOORS, OPEN_DOORS, GlyphKind, PassableGlyphs, SolidGlyphs, \
                    calculate_wavefront_and_glyph_kinds

CARDINALS: Tuple[Tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))

# Tunables (safe defaults)
_MAX_WEDGE_DEPTH = 12     # how far "behind the wall" to estimate unknown mass
_HALF_WEDGE_WIDTH = 3     # lateral half-width of the wedge
_UNKNOWN_NORM = 20.0      # unknown tiles to reach score multiplier ~1.0
_SEARCH_DECAY_TAU = 6.0   # e^{-searched / tau} decay

AGENT_GLYPH = 333

def _perps(dy: int, dx: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    # Two perpendicular unit vectors to (dy, dx)
    return (dx, -dy), (-dx, dy)

class NethackPlayer:
    """Player state in Nethack."""
    # pylint: disable=no-member

    def __init__(self, obs):
        blstats = obs['blstats']
        self.position = (int(blstats[1]), int(blstats[0]))  # (y, x) coords of the player
        self.score = blstats[nethack.NLE_BL_SCORE]
        self.gold = blstats[nethack.NLE_BL_GOLD]
        self.hp = blstats[nethack.NLE_BL_HP]
        self.hp_max = blstats[nethack.NLE_BL_HPMAX]
        self.level = blstats[nethack.NLE_BL_XP]
        self.depth = blstats[nethack.NLE_BL_DEPTH]
        self.exp = blstats[nethack.NLE_BL_EXP]
        self.hunger = blstats[nethack.NLE_BL_HUNGER]
        self._conditions = blstats[nethack.NLE_BL_CONDITION]

    @property
    def is_confused(self):
        """Check if the player is confused."""
        return bool(self._conditions & nethack.BL_MASK_CONF)

    @property
    def is_blind(self):
        """Check if the player is blind."""
        return bool(self._conditions & nethack.BL_MASK_BLIND)

    @property
    def is_deaf(self):
        """Check if the player is deaf."""
        return bool(self._conditions & nethack.BL_MASK_DEAF)

    @property
    def is_flying(self):
        """Check if the player is flying."""
        return bool(self._conditions & nethack.BL_MASK_FLY)

    @property
    def is_food_poisoned(self):
        """Check if the player is food poisoned."""
        return bool(self._conditions & nethack.BL_MASK_FOODPOIS)

    @property
    def is_hallucinating(self):
        """Check if the player is hallucinating."""
        return bool(self._conditions & nethack.BL_MASK_HALLU)

    @property
    def is_levitating(self):
        """Check if the player is levitating."""
        return bool(self._conditions & nethack.BL_MASK_LEV)

    @property
    def is_riding(self):
        """Check if the player is riding."""
        return bool(self._conditions & nethack.BL_MASK_RIDE)

    @property
    def is_slime(self):
        """Check if the player is slimed."""
        return bool(self._conditions & nethack.BL_MASK_SLIME)

    @property
    def is_stone(self):
        """Check if the player is petrified."""
        return bool(self._conditions & nethack.BL_MASK_STONE)

    @property
    def is_strangled(self):
        """Check if the player is strangled."""
        return bool(self._conditions & nethack.BL_MASK_STRNGL)

    @property
    def is_stunned(self):
        """Check if the player is stunned."""
        return bool(self._conditions & nethack.BL_MASK_STUN)

    @property
    def is_terminally_ill(self):
        """Check if the player is terminally ill."""
        return bool(self._conditions & nethack.BL_MASK_TERMILL)

    def as_dict(self):
        """Return player attributes as a dictionary."""
        status_flags = [
            ("confused", self.is_confused),
            ("blind", self.is_blind),
            ("deaf", self.is_deaf),
            ("flying", self.is_flying),
            ("food_poisoned", self.is_food_poisoned),
            ("hallucinating", self.is_hallucinating),
            ("levitating", self.is_levitating),
            ("riding", self.is_riding),
            ("slime", self.is_slime),
            ("stone", self.is_stone),
            ("strangled", self.is_strangled),
            ("stunned", self.is_stunned),
            ("terminally_ill", self.is_terminally_ill),
        ]

        status = " ".join(name for name, flag in status_flags if flag) or "healthy"
        return {
            "x": int(self.position[1]),
            "y": int(self.position[0]),
            "score": self.score,
            "gold": self.gold,
            "hp": self.hp,
            "hp_max": self.hp_max,
            "level": self.level,
            "depth": self.depth,
            "exp": self.exp,
            "hunger": self.hunger,
            "status": status,
        }

class OriginalObservationInfo:
    """Original NLE observation."""
    # pylint: disable=no-member

    def __init__(self, obs, info):
        self.observation = obs.copy()
        self.info = info.copy()
class StuckBoulder:
    """A boulder that is stuck in a position."""
    BOULDER_GLYPH = 2353
    def __init__(self, player_position, boulder_position):
        self.player_position = player_position
        self.boulder_position = boulder_position

    def __repr__(self):
        return f"StuckBoulder(player_position={self.player_position}, boulder_position={self.boulder_position})"

class NethackState:
    """World state in Nethack."""
    # pylint: disable=no-member

    def __init__(self, obs, info, prev: Optional['NethackState'] = None):
        self.original = OriginalObservationInfo(obs, info)

        self.player = NethackPlayer(obs)

        self.message = obs['message'].tobytes().decode('utf-8').rstrip('\x00')
        self.time = obs['blstats'][nethack.NLE_BL_TIME]
        self.floor_glyphs = self._create_floor_glyphs(self.glyphs, prev)

        prev_is_usable = prev is not None and prev.player.depth == self.player.depth
        self.searched_tiles = prev.searched_tiles.copy() if prev_is_usable else np.zeros((21, 79), dtype=np.uint8)

        self.visited = prev.visited.copy() if prev_is_usable else np.zeros((21, 79), dtype=np.uint8)
        self.visited[self.player.position] = 1

        self.stuck_boulders = prev.stuck_boulders.copy() if prev_is_usable else []
        for boulder in self.stuck_boulders:
            if self.glyphs[boulder.boulder_position] != StuckBoulder.BOULDER_GLYPH:
                self.stuck_boulders.remove(boulder)

        unpassable = [boulder.boulder_position
                      for boulder in self.stuck_boulders
                      if boulder.player_position == self.player.position] if self.stuck_boulders else []

        wavefront, glyph_kinds = calculate_wavefront_and_glyph_kinds(self.glyphs, self.floor_glyphs,
                                                                     self.visited, unpassable)
        self.wavefront = wavefront
        self.glyph_kinds = glyph_kinds

        self.found_exits = prev.found_exits.copy() if prev_is_usable else []
        exits = glyph_kinds == GlyphKind.EXIT.value
        for pos in np.argwhere(exits):
            pos = (int(pos[0]), int(pos[1]))
            if pos not in self.found_exits:
                self.found_exits.append(pos)

        self.locked_doors = prev.locked_doors.copy() if prev_is_usable else []
        for lock in self.locked_doors:
            if self.floor_glyphs[lock] not in CLOSED_DOORS:
                self.locked_doors.remove(lock)

        self.open_doors_not_visible = prev.open_doors_not_visible.copy() if prev_is_usable else []
        for door in self.open_doors_not_visible:
            # we expect the door to be covered by something other than geometry
            if nethack.glyph_is_cmap(self.floor_glyphs[door]):
                self.open_doors_not_visible.remove(door)
                self.floor_glyphs[door] = self.glyphs[door]
            else:
                self.floor_glyphs[door] = OPEN_DOORS[0]  # assume it's an open door

        self.search_state = SearchState(self, prev.search_state if prev_is_usable else None)

    @property
    def visible_enemies(self):
        """Check if there are any visible enemies."""
        is_mon = np.vectorize(nethack.glyph_is_monster)
        is_pet = np.vectorize(nethack.glyph_is_pet)

        mask = is_mon(self.glyphs) & ~is_pet(self.glyphs) & (self.glyphs != AGENT_GLYPH)
        return np.any(mask)

    @property
    def tty_chars(self):
        """Return the TTY characters from the observation."""
        return self.original.observation['tty_chars']

    @property
    def tty_colors(self):
        """Return the TTY colors from the observation."""
        return self.original.observation['tty_colors']

    @property
    def chars(self):
        """Return the characters from the observation."""
        return self.original.observation['chars']

    @property
    def glyphs(self):
        """Return the glyphs from the observation."""
        return self.original.observation['glyphs']

    def get_screen_description(self, pos):
        """Returns a string description of the glyph at the given position."""
        y, x = pos
        return self.original.observation['screen_descriptions'][y, x].tobytes().decode('utf-8').rstrip('\x00')

    def add_locked_door(self, pos):
        """Add a locked door position to the state."""
        if pos not in self.locked_doors:
            assert self.floor_glyphs[pos] in CLOSED_DOORS
            self.locked_doors.append(pos)

    def add_open_door(self, pos):
        """Add an open door position to the state."""
        if pos not in self.open_doors_not_visible:
            if not nethack.glyph_is_cmap(self.floor_glyphs[pos]):
                self.open_doors_not_visible.append(pos)

    def add_stuck_boulder(self, player_position, boulder_position):
        """Add a stuck boulder to the state."""
        if not any(boulder.boulder_position == boulder_position and boulder.player_position == player_position
                   for boulder in self.stuck_boulders):
            self.stuck_boulders.append(StuckBoulder(player_position, boulder_position))

    @property
    def is_player_on_exit(self):
        """Check if the player is on the exit."""
        return self.player.position in self.found_exits


    def as_dict(self):
        """Return the state as a dictionary."""
        result = {
            "time": self.time,
            "message": self.message,
            "found_exits": self.found_exits,
        }
        result.update(self.player.as_dict())
        return result

    def _create_floor_glyphs(self, glyphs: np.ndarray, prev: Optional['NethackState']) -> np.ndarray:
        """Create a 2D array of floor glyphs, where each cell contains the glyph without characters."""
        if prev is not None and self.player.depth != prev.player.depth:
            prev = None

        floor_glyphs = glyphs.copy()
        if prev is None:
            return floor_glyphs

        for y in range(glyphs.shape[0]):
            for x in range(glyphs.shape[1]):
                glyph = glyphs[y, x]
                if nethack.glyph_is_monster(glyph) or nethack.glyph_is_object(glyph):
                    prev_glyph = prev.floor_glyphs[y, x]
                    if prev_glyph != SolidGlyphs.S_stone.value:
                        floor_glyphs[y, x] = prev_glyph

        return floor_glyphs

class SearchState:
    """A state that tracks search progress."""
    def __init__(self, state: NethackState, prev_search_state: Optional['SearchState'] = None):
        self._state = state
        self.search_counts = prev_search_state.search_counts.copy() if prev_search_state \
                             else np.zeros_like(state.floor_glyphs, dtype=np.uint16)
        self.search_scores = self._calculate_search_score()

    @staticmethod
    def _is_corridor(g: int) -> bool:
        return g in (PassableGlyphs.S_corr.value, PassableGlyphs.S_litcorr.value)

    @staticmethod
    def _is_room_floor(g: int) -> bool:
        return g in (PassableGlyphs.S_room.value, PassableGlyphs.S_darkroom.value)

    @staticmethod
    def _is_open_door(g: int) -> bool:
        return g in (PassableGlyphs.S_vodoor.value, PassableGlyphs.S_hodoor.value, PassableGlyphs.S_ndoor.value)

    @staticmethod
    def _is_closed_door(g: int) -> bool:
        return g in (PassableGlyphs.S_vcdoor.value, PassableGlyphs.S_hcdoor.value)

    @staticmethod
    def _is_wall(g: int) -> bool:
        # Walls/corners/T-walls + closed drawbridges (act like walls for “search a wall”)
        return g in (
            SolidGlyphs.S_vwall.value, SolidGlyphs.S_hwall.value,
            SolidGlyphs.S_tlcorn.value, SolidGlyphs.S_trcorn.value,
            SolidGlyphs.S_blcorn.value, SolidGlyphs.S_brcorn.value,
            SolidGlyphs.S_crwall.value, SolidGlyphs.S_tuwall.value,
            SolidGlyphs.S_tdwall.value, SolidGlyphs.S_tlwall.value,
            SolidGlyphs.S_trwall.value, SolidGlyphs.S_vcdbridge.value,
            SolidGlyphs.S_hcdbridge.value
        )

    @staticmethod
    def _is_stone(g: int) -> bool:
        return g == SolidGlyphs.S_stone.value

    def _is_barrier(self, g: int) -> bool:
        # Something you could plausibly reveal a secret through (room wall or rock)
        return self._is_wall(g) or self._is_stone(g)

    def _is_standable(self, g: int) -> bool:
        # Where the agent can stand to perform a search
        if self._is_closed_door(g):
            return False
        return (
            self._is_room_floor(g) or self._is_corridor(g) or self._is_open_door(g) or
            g in (
                PassableGlyphs.S_upstair.value, PassableGlyphs.S_dnstair.value,
                PassableGlyphs.S_upladder.value, PassableGlyphs.S_dnladder.value,
                PassableGlyphs.S_altar.value, PassableGlyphs.S_throne.value,
                PassableGlyphs.S_fountain.value, PassableGlyphs.S_ice.value,
                PassableGlyphs.S_vodbridge.value, PassableGlyphs.S_hodbridge.value,
                PassableGlyphs.S_water.value
            )
        )

    # ---- local geometry helpers ------------------------------------------- #

    def _count_cardinal_corridors(self, y: int, x: int) -> int:
        shp_h, shp_w = self._state.floor_glyphs.shape
        cnt = 0
        for dy, dx in CARDINALS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < shp_h and 0 <= nx < shp_w and self._is_corridor(self._state.floor_glyphs[ny, nx]):
                cnt += 1
        return cnt

    def _wall_run_length(self, wy: int, wx: int, n_dy: int, n_dx: int) -> int:
        """
        Length of contiguous wall along the axis perpendicular to the normal (n_dy, n_dx),
        measured through the wall cell at (wy, wx). Includes this cell in the count.
        """
        floor_glyphs = self._state.floor_glyphs
        shp_h, shp_w = floor_glyphs.shape
        if not self._is_wall(floor_glyphs[wy, wx]):
            return 0
        length = 1
        p1, p2 = _perps(n_dy, n_dx)
        for py, px in (p1, p2):
            step = 1
            while True:
                ny, nx = wy + py * step, wx + px * step
                if not (0 <= ny < shp_h and 0 <= nx < shp_w):
                    break
                if not self._is_wall(floor_glyphs[ny, nx]):
                    break
                length += 1
                step += 1
        return length

    def _unknown_mass_wedge(self, wy: int, wx: int, n_dy: int, n_dx: int,
                            depth: int = _MAX_WEDGE_DEPTH,
                            half_width: int = _HALF_WEDGE_WIDTH) -> int:
        """
        Count UNSEEN tiles in a forward wedge starting one cell *behind* the wall/rock at (wy, wx),
        along the normal (n_dy, n_dx).
        """
        shp_h, shp_w = self._state.glyph_kinds.shape
        perps = _perps(n_dy, n_dx)
        total = 0
        for k in range(1, depth + 1):
            by, bx = wy + n_dy * k, wx + n_dx * k
            for w in range(-half_width, half_width + 1):
                py, px = perps[0]
                ny, nx = by + py * w, bx + px * w
                if 0 <= ny < shp_h and 0 <= nx < shp_w and self._state.glyph_kinds[ny, nx] == GlyphKind.UNSEEN.value:
                    total += 1
        return total

    # ---- main scoring ------------------------------------------------------ #

    def _calculate_search_score(self) -> np.ndarray:
        """
        Return a 7x7 float32 grid (centered on the agent) where each cell gives the
        *standing* value of running a 22s search from that tile, in [0,1].

        Heuristics:
          - corridor dead-end tips score highest if there is unknown mass behind the tip
          - standing next to long room walls with unknown behind scores well
          - multiplied by unknown mass factor and decayed by how often we already searched there
        """
        floor_glyphs = self._state.floor_glyphs
        shp_h, shp_w = floor_glyphs.shape
        ay, ax = self._state.player.position
        scores = np.zeros((7, 7), dtype=np.float32)

        searched_map = self.search_counts
        assert searched_map.shape == floor_glyphs.shape

        for oy in range(-3, 4):
            for ox in range(-3, 4):
                ty, tx = ay + oy, ax + ox
                if not (0 <= ty < shp_h and 0 <= tx < shp_w):
                    continue

                g_here = floor_glyphs[ty, tx]
                if not self._is_standable(g_here):
                    continue

                best = 0.0

                # 1) Corridor dead-end tip
                if self._is_corridor(g_here):
                    corr_neighbors = []
                    for dy, dx in CARDINALS:
                        ny, nx = ty + dy, tx + dx
                        if 0 <= ny < shp_h and 0 <= nx < shp_w and self._is_corridor(floor_glyphs[ny, nx]):
                            corr_neighbors.append((dy, dx))
                    if len(corr_neighbors) <= 1:
                        # look in directions that are NOT corridor (the blocked faces)
                        for dy, dx in CARDINALS:
                            if (dy, dx) in corr_neighbors:
                                continue
                            wy, wx = ty + dy, tx + dx
                            if not (0 <= wy < shp_h and 0 <= wx < shp_w):
                                continue
                            if not self._is_barrier(self._state.floor_glyphs[wy, wx]):
                                continue
                            unknown = self._unknown_mass_wedge(wy, wx, dy, dx)
                            uf = min(1.0, unknown / _UNKNOWN_NORM)
                            best = max(best, 1.0 * uf)  # dead-end base = 1.0

                # 2) Adjacent wall/rock with unknown behind (room-wall & rock cases)
                for dy, dx in CARDINALS:
                    wy, wx = ty + dy, tx + dx
                    if not (0 <= wy < shp_h and 0 <= wx < shp_w):
                        continue
                    wg = self._state.floor_glyphs[wy, wx]
                    if not self._is_barrier(wg):
                        continue

                    unknown = self._unknown_mass_wedge(wy, wx, dy, dx)
                    if unknown <= 0:
                        continue

                    uf = min(1.0, unknown / _UNKNOWN_NORM)

                    if self._is_wall(wg):
                        run = self._wall_run_length(wy, wx, dy, dx)
                        base = 0.7 if run >= 3 else 0.4
                        if run >= 5:
                            base = min(0.9, base + 0.05)
                    else:
                        # plain rock/stone (e.g., end of a rock corridor)
                        base = 0.5

                    best = max(best, base * uf)

                # 3) Downweight by how often we already searched *on this standing tile*
                searched = int(searched_map[ty, tx]) if searched_map is not None else 0
                decay = math.exp(-searched / _SEARCH_DECAY_TAU) if searched > 0 else 1.0
                score = max(0.0, min(1.0, best * decay))

                scores[oy + 3, ox + 3] = score

        return scores
