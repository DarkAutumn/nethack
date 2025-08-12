"""The state of nethack at the current timestep.  All wrappers should interact with this instead of using
observation directly."""

from typing import Optional, Tuple
from nle import nethack

from yndf.nethack_level import GLYPH_TABLE, DungeonLevel

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

        prev_is_usable = prev is not None and prev.player.depth == self.player.depth

        self.stuck_boulders = prev.stuck_boulders.copy() if prev_is_usable else []
        for boulder in self.stuck_boulders:
            if self.glyphs[boulder.boulder_position] != StuckBoulder.BOULDER_GLYPH:
                self.stuck_boulders.remove(boulder)

        unpassable = [boulder.boulder_position
                      for boulder in self.stuck_boulders
                      if boulder.player_position == self.player.position] if self.stuck_boulders else []

        self.locked_doors = prev.locked_doors.copy() if prev_is_usable else []
        self.floor = DungeonLevel(self.glyphs, unpassable, self.locked_doors, prev.floor if prev_is_usable else None)

        self.game_over = info['end_status'] == 1  # death

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
            assert (self.floor.properties[pos] & GLYPH_TABLE.CLOSED_DOOR) != 0
            self.locked_doors.append(pos)

    def add_stuck_boulder(self, player_position, boulder_position):
        """Add a stuck boulder to the state."""
        if not any(boulder.boulder_position == boulder_position and boulder.player_position == player_position
                   for boulder in self.stuck_boulders):
            self.stuck_boulders.append(StuckBoulder(player_position, boulder_position))

    def as_dict(self):
        """Return the state as a dictionary."""
        result = {
            "time": self.time,
            "message": self.message,
        }
        result.update(self.player.as_dict())
        return result
