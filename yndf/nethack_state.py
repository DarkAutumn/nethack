"""The state of nethack at the current timestep.  All wrappers should interact with this instead of using
observation directly."""

from typing import Optional
from nle import nethack
import numpy as np

from yndf.movement import CLOSED_DOORS, OPEN_DOORS, GlyphKind, calculate_wavefront_and_glyph_kinds

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

class NethackState:
    """World state in Nethack."""
    # pylint: disable=no-member

    def __init__(self, obs, info, prev: Optional['NethackState'] = None):
        self.original = OriginalObservationInfo(obs, info)

        self.player = NethackPlayer(obs)

        self.message = obs['message'].tobytes().decode('utf-8').rstrip('\x00')
        self.time = obs['blstats'][nethack.NLE_BL_TIME]
        self.floor_glyphs = self._create_floor_glyphs(self.glyphs, prev)

        if prev is not None and self.player.depth == prev.player.depth:
            self.visited = prev.visited.copy()
        else:
            self.visited: np.ndarray = np.zeros((21, 79), dtype=np.uint8)

        self.visited[self.player.position] = 1

        wavefront, glyph_kinds = calculate_wavefront_and_glyph_kinds(self.floor_glyphs, self.visited)
        self.wavefront = wavefront
        self.glyph_kinds = glyph_kinds

        self.found_exits = prev.found_exits.copy() if prev is not None else []
        exits = glyph_kinds == GlyphKind.EXIT.value
        for pos in np.argwhere(exits):
            pos = (int(pos[0]), int(pos[1]))
            if pos not in self.found_exits:
                self.found_exits.append(pos)

        prev_is_usable = prev is not None and prev.player.depth == self.player.depth

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
                    floor_glyphs[y, x] = prev.floor_glyphs[y, x]

        return floor_glyphs
