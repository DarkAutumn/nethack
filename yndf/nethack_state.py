"""The state of nethack at the current timestep.  All wrappers should interact with this instead of using
observation directly."""

from typing import Optional
from nle import nethack
import numpy as np

class NethackPlayer:
    """Player state in Nethack."""
    # pylint: disable=no-member

    def __init__(self, obs):
        self.position = obs['blstats'][1], obs['blstats'][0]  # (y, x) coords of the player
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

class NethackState:
    """World state in Nethack."""
    # pylint: disable=no-member

    def __init__(self, obs, prev : Optional['NethackState'] = None):
        self.player = NethackPlayer(obs)
        self.message = obs['message'].tobytes().decode('utf-8').rstrip('\x00')
        self.time = obs['blstats'][nethack.NLE_BL_TIME]

        self.exit = None
        if prev is not None and prev.exit is not None:
            self.exit = prev.exit
        else:
            index = obs['chars'] == ord('>')
            if index is not None and len(index[0]) > 0 and len(index[1]) > 0:
                self.exit = index[0][0], index[1][0]

        if prev is not None:
            self.visited = prev.visited.copy()
        else:

            self.visited: np.ndarray = np.zeros((21, 79), dtype=np.uint8)

        self.visited[self.player.position] = 1

    @property
    def is_player_on_exit(self):
        """Check if the player is on the exit."""
        if self.exit is None:
            return False

        agent_yx = self.player.position
        return agent_yx == self.exit
