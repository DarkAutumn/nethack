"""The state of nethack at the current timestep.  All wrappers should interact with this instead of using
observation directly."""

from typing import Optional
from nle import nethack
import numpy as np

from yndf.wavefront import GlyphKind, calculate_wavefront_and_glyph_kinds

class NethackPlayer:
    """Player state in Nethack."""
    # pylint: disable=no-member

    def __init__(self, obs):
        self.position = (int(obs['blstats'][1]), int(obs['blstats'][0]))  # (y, x) coords of the player
        self.score = obs['blstats'][nethack.NLE_BL_SCORE]
        self.gold = obs['blstats'][nethack.NLE_BL_GOLD]
        self.hp = obs['blstats'][nethack.NLE_BL_HP]
        self.hp_max = obs['blstats'][nethack.NLE_BL_HPMAX]
        self.level = obs['blstats'][nethack.NLE_BL_XP]
        self.depth = obs['blstats'][nethack.NLE_BL_DEPTH]
        self.exp = obs['blstats'][nethack.NLE_BL_EXP]
        self.hunger = obs['blstats'][nethack.NLE_BL_HUNGER]
        self._conditions = obs['blstats'][nethack.NLE_BL_CONDITION]

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

class NethackState:
    """World state in Nethack."""
    # pylint: disable=no-member

    def __init__(self, obs, prev : Optional['NethackState'] = None):
        self.player = NethackPlayer(obs)
        self.message = obs['message'].tobytes().decode('utf-8').rstrip('\x00')
        self.time = obs['blstats'][nethack.NLE_BL_TIME]
        self.tty_chars = obs['tty_chars']
        self.tty_colors = obs['tty_colors']
        self.chars = obs['chars']
        self.glyphs = obs['glyphs']

        if prev is not None and self.player.depth == prev.player.depth:
            self.visited = prev.visited.copy()
        else:
            self.visited: np.ndarray = np.zeros((21, 79), dtype=np.uint8)

        self.visited[self.player.position] = 1

        wavefront, glyph_kinds = calculate_wavefront_and_glyph_kinds(self.glyphs, self.visited)
        self.wavefront = wavefront
        self.glyph_kinds = glyph_kinds

        self.found_exits = []
        if prev is not None and prev.found_exits:
            self.found_exits = prev.found_exits.copy()
        else:
            exits = glyph_kinds == GlyphKind.EXIT.value
            for pos in np.argwhere(exits):
                self.found_exits.append((pos[0], pos[1]))

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
