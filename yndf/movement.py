from enum import Enum
from typing import List, Tuple
import numpy as np
import nle

class SolidGlyphs(Enum):
    """Enum for solid glyphs in Nethack."""
    # pylint:disable=no-member,invalid-name
    S_stone     =  0 + nle.nethack.GLYPH_CMAP_OFF
    S_vwall     =  1 + nle.nethack.GLYPH_CMAP_OFF
    S_hwall     =  2 + nle.nethack.GLYPH_CMAP_OFF
    S_tlcorn    =  3 + nle.nethack.GLYPH_CMAP_OFF
    S_trcorn    =  4 + nle.nethack.GLYPH_CMAP_OFF
    S_blcorn    =  5 + nle.nethack.GLYPH_CMAP_OFF
    S_brcorn    =  6 + nle.nethack.GLYPH_CMAP_OFF
    S_crwall    =  7 + nle.nethack.GLYPH_CMAP_OFF
    S_tuwall    =  8 + nle.nethack.GLYPH_CMAP_OFF
    S_tdwall    =  9 + nle.nethack.GLYPH_CMAP_OFF
    S_tlwall    = 10 + nle.nethack.GLYPH_CMAP_OFF
    S_trwall    = 11 + nle.nethack.GLYPH_CMAP_OFF
    #S_vcdoor    = 15 + nle.nethack.GLYPH_CMAP_OFF # closed door, vertical wall
    #S_hcdoor    = 16 + nle.nethack.GLYPH_CMAP_OFF # closed door, horizontal wall
    S_bars      = 17 + nle.nethack.GLYPH_CMAP_OFF # KMH -- iron bars
    S_tree      = 18 + nle.nethack.GLYPH_CMAP_OFF # KMH
    S_sink      = 30 + nle.nethack.GLYPH_CMAP_OFF
    S_pool      = 32 + nle.nethack.GLYPH_CMAP_OFF
    S_lava      = 34 + nle.nethack.GLYPH_CMAP_OFF
    S_vcdbridge = 37 + nle.nethack.GLYPH_CMAP_OFF # closed drawbridge, vertical wall
    S_hcdbridge = 38 + nle.nethack.GLYPH_CMAP_OFF # closed drawbridge, horizontal wall

class PassableGlyphs(Enum):
    """Enum for passable glyphs in Nethack."""
    # pylint:disable=no-member,invalid-name
    S_ndoor     = 12 + nle.nethack.GLYPH_CMAP_OFF
    S_vodoor    = 13 + nle.nethack.GLYPH_CMAP_OFF
    S_hodoor    = 14 + nle.nethack.GLYPH_CMAP_OFF
    S_vcdoor    = 15 + nle.nethack.GLYPH_CMAP_OFF
    S_hcdoor    = 16 + nle.nethack.GLYPH_CMAP_OFF
    S_room      = 19 + nle.nethack.GLYPH_CMAP_OFF
    S_darkroom  = 20 + nle.nethack.GLYPH_CMAP_OFF
    S_corr      = 21 + nle.nethack.GLYPH_CMAP_OFF
    S_litcorr   = 22 + nle.nethack.GLYPH_CMAP_OFF
    S_upstair   = 23 + nle.nethack.GLYPH_CMAP_OFF
    S_dnstair   = 24 + nle.nethack.GLYPH_CMAP_OFF
    S_upladder  = 25 + nle.nethack.GLYPH_CMAP_OFF
    S_dnladder  = 26 + nle.nethack.GLYPH_CMAP_OFF
    S_altar     = 27 + nle.nethack.GLYPH_CMAP_OFF
    S_grave     = 28 + nle.nethack.GLYPH_CMAP_OFF
    S_throne    = 29 + nle.nethack.GLYPH_CMAP_OFF
    S_fountain  = 31 + nle.nethack.GLYPH_CMAP_OFF
    S_ice       = 33 + nle.nethack.GLYPH_CMAP_OFF
    S_vodbridge = 35 + nle.nethack.GLYPH_CMAP_OFF
    S_hodbridge = 36 + nle.nethack.GLYPH_CMAP_OFF
    S_air       = 39 + nle.nethack.GLYPH_CMAP_OFF
    S_cloud     = 40 + nle.nethack.GLYPH_CMAP_OFF
    S_water     = 41 + nle.nethack.GLYPH_CMAP_OFF

PLAYER_GLYPH = 333

DIRECTION_MAP = {
    nle.nethack.CompassDirection.NW : (-1, -1),
    nle.nethack.CompassDirection.N : (-1, 0),
    nle.nethack.CompassDirection.NE : (-1, 1),
    nle.nethack.CompassDirection.W : (0, -1),
    nle.nethack.CompassDirection.E : (0, 1),
    nle.nethack.CompassDirection.SW : (1, -1),
    nle.nethack.CompassDirection.S : (1, 0),
    nle.nethack.CompassDirection.SE : (1, 1),
}

DIRECTIONS = list(DIRECTION_MAP.values())

class GlyphKind(Enum):
    """Enum for mapping glyphs to their walkable state."""
    PASSABLE = 1
    UNPASSABLE = 2
    UNSEEN = 3
    FRONTIER = 4
    EXIT = 5

UNPASSABLE_WAVEFRONT = 1_000_000

def _manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Calculate the Manhattan distance between two points."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def _cannot_move_diagonal(glyph: int) -> bool:
    """Check if a glyph is an open door."""
    return glyph in (PassableGlyphs.S_vodoor.value, PassableGlyphs.S_hodoor.value)

def can_move(glyphs : np.ndarray, from_pos : Tuple[int, int], to_pos : Tuple[int, int]) -> bool:
    """Check if the player can move onto a given glyph."""
    if not (0 <= to_pos[0] < glyphs.shape[0] and 0 <= to_pos[1] < glyphs.shape[1]):
        return False

    glyph = glyphs[to_pos[0], to_pos[1]]
    if glyph in SolidGlyphs:
        return False

    # can't move diagonally through an open door, can open a door diagnonally
    if _cannot_move_diagonal(glyph) or _cannot_move_diagonal(glyphs[from_pos[0], from_pos[1]]):
        if _manhattan_distance(from_pos, to_pos) == 2:
            # can't move diagonally through a door
            return False

    return True

def adjacent_to_closed_door(glyphs : np.ndarray, y: int, x: int) -> bool:
    """Check if the given position is adjacent to a closed door."""
    for dy, dx in DIRECTIONS:
        ny, nx = y + dy, x + dx
        if (0 <= ny < glyphs.shape[0] and 0 <= nx < glyphs.shape[1]):
            if glyphs[ny, nx] in (PassableGlyphs.S_vcdoor.value, PassableGlyphs.S_hcdoor.value):
                return True
    return False

def has_surrounding(glyphs : np.ndarray, visited : np.ndarray, y: int, x: int, func : callable) -> bool:
    """Check if the surrounding glyphs match the specified kind."""
    for dy, dx in DIRECTIONS:
        ny, nx = y + dy, x + dx
        if (0 <= ny < glyphs.shape[0] and 0 <= nx < glyphs.shape[1]):
            visit = visited[ny, nx]
            glyph = glyphs[ny, nx]
            if func(glyph, visit):
                return True

    return False

def calculate_glyph_kinds(glyphs : np.ndarray, visited : np.ndarray) -> np.ndarray:
    """Get a map of unseen stone tiles."""
    glyph_kinds = np.zeros_like(visited, dtype=np.uint8)
    for y in range(visited.shape[0]):
        for x in range(visited.shape[1]):
            if glyphs[y, x] == SolidGlyphs.S_stone.value:
                if has_surrounding(glyphs, visited, y, x, lambda g, v: g in PassableGlyphs and v):
                    # If we've concretely seen a stone tile, it is unpassable
                    glyph_kinds[y, x] = GlyphKind.UNPASSABLE.value
                else:
                    glyph_kinds[y, x] = GlyphKind.UNSEEN.value
            elif glyphs[y, x] in SolidGlyphs:
                glyph_kinds[y, x] = GlyphKind.UNPASSABLE.value
            elif glyphs[y, x] in (PassableGlyphs.S_dnladder.value, PassableGlyphs.S_dnstair.value):
                glyph_kinds[y, x] = GlyphKind.EXIT.value
            else:
                glyph_kinds[y, x] = GlyphKind.PASSABLE.value


    # Now find frontier tiles. Replace passable tiles that are adjacent to UNSEEN.
    for y in range(visited.shape[0]):
        for x in range(visited.shape[1]):
            if glyph_kinds[y, x] != GlyphKind.PASSABLE.value:
                continue

            if has_surrounding(glyph_kinds, visited, y, x, lambda g, v: g == GlyphKind.UNSEEN.value and not v):
                glyph_kinds[y, x] = GlyphKind.FRONTIER.value

    return glyph_kinds

def calculate_wavefront(glyphs: np.ndarray, glyph_kinds: np.ndarray, targets: List[tuple[int, int]]) -> np.ndarray:
    """Calculate the wavefront from a list of target locations.  Target values start at 0 and increase by 1 for each
    step away from the target running through PASSABLE and FRONTIER tiles."""
    wavefront = np.full(glyph_kinds.shape, UNPASSABLE_WAVEFRONT, dtype=np.uint32)
    queue = []
    for target in targets:
        y, x = target
        if 0 <= y < wavefront.shape[0] and 0 <= x < wavefront.shape[1]:
            wavefront[y, x] = 0
            queue.append((y, x))

    while queue:
        y, x = queue.pop(0)
        current_value = wavefront[y, x]

        for dy, dx in DIRECTIONS:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < wavefront.shape[0] and 0 <= nx < wavefront.shape[1]):
                continue

            if glyph_kinds[ny, nx] not in (GlyphKind.PASSABLE.value, GlyphKind.FRONTIER.value):
                continue

            if not can_move(glyphs, (y, x), (ny, nx)):
                continue

            if wavefront[ny, nx] > current_value + 1:
                wavefront[ny, nx] = current_value + 1
                queue.append((ny, nx))

    return wavefront

def calculate_wavefront_and_glyph_kinds(glyphs: np.ndarray, visited: np.ndarray) -> np.ndarray:
    """Calculate the wavefront from a NethackState and a list of target locations."""
    glyph_kinds = calculate_glyph_kinds(glyphs, visited)
    targets = np.argwhere((glyph_kinds == GlyphKind.EXIT.value) | (glyph_kinds == GlyphKind.FRONTIER.value))
    wavefront = calculate_wavefront(glyphs, glyph_kinds, targets)
    return wavefront, glyph_kinds
