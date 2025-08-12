"""Conditions for the end of an episode."""

from yndf.nethack_level import GLYPH_TABLE, DungeonLevel
from yndf.nethack_state import NethackState

class Ending:
    """A class to check whether an ending condition has been met."""
    # pylint: disable=unused-argument

    def __init__(self, name: str):
        self.name = name
        self._enabled = True
        self._terminated = False
        self._truncated = False

    def reset(self, state : NethackState) -> None:
        """Reset for a new episode."""
        self._terminated = False
        self._truncated = False

    def step(self, state : NethackState) -> None:
        """Check if the ending condition has been met."""
        self._terminated = False
        self._truncated = False

    @property
    def terminated(self) -> bool:
        """Check if the ending condition is terminated."""
        return self._terminated

    @property
    def truncated(self) -> bool:
        """Check if the ending condition is truncated."""
        return self._truncated

    @property
    def enabled(self) -> bool:
        """Check if this ending condition is enabled."""
        return self._enabled

    def disable(self):
        """Disable this ending condition."""
        self._enabled = False

    def enable(self):
        """Enable this ending condition."""
        self._enabled = True

class NoForwardPathWithoutSearching(Ending):
    """An ending condition that checks for no forward path without searching."""
    def __init__(self):
        super().__init__("no-forward-path-without-searching")

    def step(self, state : NethackState) -> None:
        """Check if there is no forward path without searching."""
        super().step(state)

        exits_or_frontiers = (state.level.properties & (GLYPH_TABLE.DESCEND_LOCATION | DungeonLevel.FRONTIER)) != 0
        self._terminated = not exits_or_frontiers.any()

class NoDiscovery(Ending):
    """An ending condition that checks for no discovery."""
    def __init__(self, max_steps: int = 100):
        super().__init__("no-discovery")

        assert max_steps > 1
        self.max_steps = max_steps
        self._steps_since_new = 0
        self._prev : NethackState = None

    def reset(self, state : NethackState) -> None:
        """Reset for a new episode."""
        super().reset(state)

        self._steps_since_new = 0
        self._prev = state

    def step(self, state : NethackState) -> None:
        """Check if there has been no discovery."""
        super().step(state)

        if self._prev.player.depth < state.player.depth:
            self._steps_since_new = 0

        elif self._prev.player.exp < state.player.exp:
            self._steps_since_new = 0

        elif self._prev.player.gold < state.player.gold:
            self._steps_since_new = 0

        elif self._prev.floor.stone_tile_count > state.floor.stone_tile_count:
            self._steps_since_new = 0

        else:
            self._steps_since_new += 1

        self._truncated = self._steps_since_new > self.max_steps
        self._prev = state
