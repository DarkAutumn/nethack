"""Conditions for the end of an episode."""

from yndf.nethack_level import GLYPH_TABLE, DungeonLevel
from yndf.nethack_state import NethackState

class Ending:
    """A class to check whether an ending condition has been met."""
    # pylint: disable=unused-argument

    def __init__(self, name: str):
        self.name = name
        self._enabled = 1
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
        return self._enabled > 0

    def disable(self):
        """Disable this ending condition."""
        self._enabled -= 1

    def enable(self):
        """Enable this ending condition."""
        self._enabled += 1

class NoForwardPathWithoutSearching(Ending):
    """An ending condition that checks for no forward path without searching."""
    def __init__(self):
        super().__init__("no-forward-path-without-searching")

    def step(self, state : NethackState) -> None:
        """Check if there is no forward path without searching."""
        super().step(state)

        exits_or_frontiers = (state.level.properties & (GLYPH_TABLE.DESCEND_LOCATION | DungeonLevel.FRONTIER)) != 0
        self._terminated = not exits_or_frontiers.any()

class MaxTimestepsReached(Ending):
    """An ending condition that checks for max timesteps reached."""
    def __init__(self, max_steps: int = 15_000):
        super().__init__("max-timesteps-reached")

        assert max_steps > 1
        self.max_steps = max_steps

    def step(self, state : NethackState) -> None:
        """Check if there has been no discovery."""
        super().step(state)

        if state.time > self.max_steps:
            self._truncated = True
