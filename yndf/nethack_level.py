from enum import Enum
from collections import deque
from typing import Tuple
from nle import nethack
import numpy as np

def _bit(value: int) -> np.uint32:
    """Return a bitmask for the given value."""
    return np.uint32(1 << value)

class SolidGlyphs(Enum):
    """Enum for solid glyphs in Nethack."""
    # pylint:disable=no-member,invalid-name
    S_stone     =  0 + nethack.GLYPH_CMAP_OFF
    S_vwall     =  1 + nethack.GLYPH_CMAP_OFF
    S_hwall     =  2 + nethack.GLYPH_CMAP_OFF
    S_tlcorn    =  3 + nethack.GLYPH_CMAP_OFF
    S_trcorn    =  4 + nethack.GLYPH_CMAP_OFF
    S_blcorn    =  5 + nethack.GLYPH_CMAP_OFF
    S_brcorn    =  6 + nethack.GLYPH_CMAP_OFF
    S_crwall    =  7 + nethack.GLYPH_CMAP_OFF
    S_tuwall    =  8 + nethack.GLYPH_CMAP_OFF
    S_tdwall    =  9 + nethack.GLYPH_CMAP_OFF
    S_tlwall    = 10 + nethack.GLYPH_CMAP_OFF
    S_trwall    = 11 + nethack.GLYPH_CMAP_OFF
    S_bars      = 17 + nethack.GLYPH_CMAP_OFF # KMH -- iron bars
    S_tree      = 18 + nethack.GLYPH_CMAP_OFF # KMH
    S_sink      = 30 + nethack.GLYPH_CMAP_OFF
    S_pool      = 32 + nethack.GLYPH_CMAP_OFF
    S_lava      = 34 + nethack.GLYPH_CMAP_OFF
    S_vcdbridge = 37 + nethack.GLYPH_CMAP_OFF # closed drawbridge, vertical wall
    S_hcdbridge = 38 + nethack.GLYPH_CMAP_OFF # closed drawbridge, horizontal wall

class PassableGlyphs(Enum):
    """Enum for passable glyphs in Nethack."""
    # pylint:disable=no-member,invalid-name
    S_ndoor     = 12 + nethack.GLYPH_CMAP_OFF
    S_vodoor    = 13 + nethack.GLYPH_CMAP_OFF
    S_hodoor    = 14 + nethack.GLYPH_CMAP_OFF
    S_vcdoor    = 15 + nethack.GLYPH_CMAP_OFF
    S_hcdoor    = 16 + nethack.GLYPH_CMAP_OFF
    S_room      = 19 + nethack.GLYPH_CMAP_OFF
    S_darkroom  = 20 + nethack.GLYPH_CMAP_OFF
    S_corr      = 21 + nethack.GLYPH_CMAP_OFF
    S_litcorr   = 22 + nethack.GLYPH_CMAP_OFF
    S_upstair   = 23 + nethack.GLYPH_CMAP_OFF
    S_dnstair   = 24 + nethack.GLYPH_CMAP_OFF
    S_upladder  = 25 + nethack.GLYPH_CMAP_OFF
    S_dnladder  = 26 + nethack.GLYPH_CMAP_OFF
    S_altar     = 27 + nethack.GLYPH_CMAP_OFF
    S_grave     = 28 + nethack.GLYPH_CMAP_OFF
    S_throne    = 29 + nethack.GLYPH_CMAP_OFF
    S_fountain  = 31 + nethack.GLYPH_CMAP_OFF
    S_ice       = 33 + nethack.GLYPH_CMAP_OFF
    S_vodbridge = 35 + nethack.GLYPH_CMAP_OFF
    S_hodbridge = 36 + nethack.GLYPH_CMAP_OFF
    S_air       = 39 + nethack.GLYPH_CMAP_OFF
    S_cloud     = 40 + nethack.GLYPH_CMAP_OFF
    S_water     = 41 + nethack.GLYPH_CMAP_OFF

class GlyphLookupTable:
    """A class to manage the glyph table."""
    BODY = _bit(1)
    CMAP = _bit(2)
    DETECTED_MONSTER = _bit(3)
    INVISIBLE = _bit(4)
    MONSTER = _bit(5)
    NORMAL_MONSTER = _bit(6)
    OBJECT = _bit(7)
    PET = _bit(8)
    RIDDEN_MONSTER = _bit(9)
    STATUE = _bit(10)
    SWALLOW = _bit(11)
    TRAP = _bit(12)
    WARNING = _bit(13)
    PASSABLE = _bit(14)
    DESCEND_LOCATION = _bit(15)
    WALL = _bit(16)
    FLOOR = _bit(17)
    CORRIDOR = _bit(18)
    OPEN_DOOR = _bit(19)
    CLOSED_DOOR = _bit(20)
    STONE = _bit(21)

    MAX = _bit(22)

    FLOOR_MASK = CMAP | WALL | FLOOR | CORRIDOR | OPEN_DOOR | CLOSED_DOOR | DESCEND_LOCATION | STONE | TRAP

    OVERLAY_MASK = MONSTER | NORMAL_MONSTER | PET | RIDDEN_MONSTER | DETECTED_MONSTER | INVISIBLE | BODY | OBJECT \
                    | STATUE | SWALLOW | WARNING

    def __init__(self):
        # pylint: disable=no-member,too-many-branches
        self.properties = np.zeros((nethack.NO_GLYPH,), dtype=np.uint32)

        descend_glyphs = (PassableGlyphs.S_dnstair.value, PassableGlyphs.S_dnladder.value)
        floor_glyphs = (PassableGlyphs.S_room.value, PassableGlyphs.S_darkroom.value)
        corridor_glyphs = (PassableGlyphs.S_corr.value, PassableGlyphs.S_litcorr.value)
        open_door_glyphs = (PassableGlyphs.S_ndoor.value, PassableGlyphs.S_vodoor.value,
                            PassableGlyphs.S_hodoor.value)
        closed_door_glyphs = (PassableGlyphs.S_vcdoor.value, PassableGlyphs.S_hcdoor.value)

        for glyph in range(self.properties.shape[0]):
            properties = self._get_basic_properties(glyph)

            if SolidGlyphs.S_vwall.value <= glyph <= SolidGlyphs.S_trwall.value:
                properties |= self.WALL

            if glyph in descend_glyphs:
                properties |= self.DESCEND_LOCATION

            if glyph in floor_glyphs:
                properties |= self.FLOOR

            if glyph in corridor_glyphs:
                properties |= self.CORRIDOR

            if glyph in open_door_glyphs:
                properties |= self.OPEN_DOOR

            if glyph in closed_door_glyphs:
                properties |= self.CLOSED_DOOR

            if glyph == SolidGlyphs.S_stone.value:
                properties |= self.STONE

            if glyph not in SolidGlyphs:
                properties |= self.PASSABLE

            self.properties[glyph] = properties

    def _get_basic_properties(self, glyph):
        # pylint: disable=no-member,too-many-branches
        properties = 0
        if nethack.glyph_is_body(glyph):
            properties |= self.BODY
        if nethack.glyph_is_cmap(glyph):
            properties |= self.CMAP
        if nethack.glyph_is_detected_monster(glyph):
            properties |= self.DETECTED_MONSTER
        if nethack.glyph_is_invisible(glyph):
            properties |= self.INVISIBLE
        if nethack.glyph_is_monster(glyph):
            properties |= self.MONSTER
        if nethack.glyph_is_normal_monster(glyph):
            properties |= self.NORMAL_MONSTER
        if nethack.glyph_is_object(glyph):
            properties |= self.OBJECT
        if nethack.glyph_is_pet(glyph):
            properties |= self.PET
        if nethack.glyph_is_ridden_monster(glyph):
            properties |= self.RIDDEN_MONSTER
        if nethack.glyph_is_statue(glyph):
            properties |= self.STATUE
        if nethack.glyph_is_swallow(glyph):
            properties |= self.SWALLOW
        if nethack.glyph_is_trap(glyph):
            properties |= self.TRAP
        if nethack.glyph_is_warning(glyph):
            properties |= self.WARNING

        return properties

GLYPH_TABLE = GlyphLookupTable()

class DungeonLevel:
    """A class to represent the current state of the dungeon floor."""
    VISITED = GLYPH_TABLE.MAX
    FRONTIER = GLYPH_TABLE.MAX << 1
    TARGET = GLYPH_TABLE.MAX << 2

    def __init__(self, glyphs: np.ndarray, unpassable : np.ndarray, prev : 'DungeonLevel' = None):
        self.glyphs = glyphs

        self.properties = GLYPH_TABLE.properties[glyphs] & (GLYPH_TABLE.MAX - 1)

        # If we have a previous state, use that to determine what's under movable glyphs
        if prev is not None:
            overlay = (self.properties & GLYPH_TABLE.OVERLAY_MASK) != 0
            if overlay.any():
                prev_floor = prev.properties & GLYPH_TABLE.FLOOR_MASK
                self.properties[overlay] |= prev_floor[overlay]

            # Mark visited tiles
            self.properties |= (prev.properties & self.VISITED)

            # If any monster was on a tile, but now it has objects on it, mark as unvisited so we don't skip the item
            visited_with_monster = (prev.properties & self.VISITED | GLYPH_TABLE.MONSTER) != 0
            objects  = (self.properties & GLYPH_TABLE.OBJECT) != 0
            unvisit  = visited_with_monster & objects
            self.properties[unvisit] &= ~self.VISITED

        self.properties[unpassable] &= ~GLYPH_TABLE.PASSABLE

        frontier_mask = self._calculate_frontier_mask()
        self.properties[frontier_mask] |= self.FRONTIER

        target_mask = self._get_target_mask()
        self.properties[target_mask] |= self.TARGET

        self.wavefront = self._calculate_wavefront()

    def mark_visited(self, position: Tuple[int, int], visited=True) -> None:
        """Mark a tile as visited."""
        if visited:
            self.properties[position] |= self.VISITED
        else:
            self.properties[position] &= ~self.VISITED

    def _calculate_frontier_mask(self) -> np.ndarray:
        props = self.properties
        visited = (props & self.VISITED) != 0
        stone   = (props & GLYPH_TABLE.STONE) != 0
        passbl  = (props & GLYPH_TABLE.PASSABLE) != 0

        def any_neighbor(a: np.ndarray) -> np.ndarray:
            """8-neighbor OR; returns True where any neighbor of a is True."""
            out = np.zeros_like(a, dtype=bool)
            # cardinal
            out[1:,  :] |= a[:-1, :]   # N
            out[:-1, :] |= a[1:,  :]   # S
            out[:, 1:]  |= a[:, :-1]   # W
            out[:, :-1] |= a[:, 1:]    # E

            # diagonals
            out[1:, 1:]  |= a[:-1, :-1]  # NW
            out[1:, :-1] |= a[:-1, 1:]   # NE
            out[:-1, 1:] |= a[1:,  :-1]  # SW
            out[:-1, :-1]|= a[1:,  1:]   # SE
            return out

        visited_nbr    = any_neighbor(visited)
        unseen_stone   = stone & ~visited_nbr
        near_unseen    = any_neighbor(unseen_stone)

        frontier_mask  = passbl & ~visited & near_unseen
        return frontier_mask

    def _get_target_mask(self):
        """Calculate the wavefront targets for the current state."""
        passable = (self.properties & GLYPH_TABLE.PASSABLE) != 0
        visited = (self.properties & self.VISITED) != 0
        non_monster_non_cmap = (self.properties & (GLYPH_TABLE.CMAP | GLYPH_TABLE.MONSTER)) == 0

        interesting_objects = ~visited & non_monster_non_cmap & passable

        frontier = (self.properties & self.FRONTIER) != 0
        target_mask = interesting_objects | frontier

        # Only push to exits if we don't have any interesting objects or unexplored rooms
        open_doors_or_floors = (self.properties & (GLYPH_TABLE.OPEN_DOOR | GLYPH_TABLE.FLOOR)) != 0
        interesting_frontier = frontier & open_doors_or_floors
        if not interesting_frontier.any():
            exits = (self.properties & GLYPH_TABLE.DESCEND_LOCATION) != 0
            target_mask |= exits

        return target_mask

    def _calculate_wavefront(self) -> np.ndarray:
        # pylint: disable=too-many-locals
        wavefront_max = 255
        h, w = self.glyphs.shape
        wave = np.full((h, w), wavefront_max, dtype=np.uint8)

        passable = (self.properties & GLYPH_TABLE.PASSABLE) != 0
        targets  = (self.properties & self.TARGET) != 0
        q = deque()

        ys, xs = np.nonzero(targets & passable)
        wave[ys, xs] = 0
        q.extend(zip(ys, xs))

        open_doors = (self.properties & GLYPH_TABLE.OPEN_DOOR) != 0
        card = ((0,1), (1,0), (0,-1), (-1,0))
        diag = ((1,1), (1,-1), (-1,1), (-1,-1))

        while q:
            y, x = q.popleft()
            new_wave = int(wave[y, x]) + 1

            # cardinals
            for dy, dx in card:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and passable[ny, nx] and wave[ny, nx] > new_wave:
                    wave[ny, nx] = new_wave
                    if new_wave < wavefront_max:
                        q.append((ny, nx))

            # diagonals (no diagonal through open doors)
            if not open_doors[y, x]:
                for dy, dx in diag:
                    ny, nx = y + +dy, x + dx
                    if (0 <= ny < h and 0 <= nx < w and passable[ny, nx]
                            and not open_doors[ny, nx] and wave[ny, nx] > new_wave):
                        wave[ny, nx] = new_wave
                        if new_wave < wavefront_max:
                            q.append((ny, nx))
        return wave
