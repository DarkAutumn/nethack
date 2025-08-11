from enum import Enum
from typing import Iterable, List, Tuple
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

CLOSED_DOORS = (PassableGlyphs.S_vcdoor.value, PassableGlyphs.S_hcdoor.value)
OPEN_DOORS = (PassableGlyphs.S_vodoor.value, PassableGlyphs.S_hodoor.value)

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

def adjacent_to(glyphs : np.ndarray, y: int, x: int, kind) -> bool:
    """Check if the given position is adjacent to a closed door."""
    for dy, dx in DIRECTIONS:
        ny, nx = y + dy, x + dx
        if (0 <= ny < glyphs.shape[0] and 0 <= nx < glyphs.shape[1]):
            if glyphs[ny, nx] in kind:
                return True
    return False

def _neighbors_any(mask: np.ndarray,
                   directions: Iterable[Tuple[int, int]],
                   out: np.ndarray | None = None) -> np.ndarray:
    """
    For each cell, returns True if ANY neighbor (in `directions`) has mask==True.
    No wraparound; out-of-bounds neighbors are ignored.
    Reuses `out` if provided (must be same shape, bool).
    """
    h, w = mask.shape
    res = out if out is not None else np.zeros((h, w), dtype=bool)
    res[...] = False

    for dy, dx in directions:
        # source slices
        if dy > 0:
            sy = slice(dy, h)
            dy_ = slice(0, h - dy)
        elif dy < 0:
            sy = slice(0, h + dy)
            dy_ = slice(-dy, h)
        else:
            sy = slice(0, h)
            dy_ = slice(0, h)

        if dx > 0:
            sx = slice(dx, w)
            dx_ = slice(0, w - dx)
        elif dx < 0:
            sx = slice(0, w + dx)
            dx_ = slice(-dx, w)
        else:
            sx = slice(0, w)
            dx_ = slice(0, w)

        res[dy_, dx_] |= mask[sy, sx]
    return res


def calculate_glyph_kinds(glyphs: np.ndarray, visited: np.ndarray) -> np.ndarray:
    """Calculate the glyph kinds for a given glyphs array and visited mask."""
    # Ensure booleans are booleans
    visited_bool = visited.astype(bool, copy=False)

    # ---- Precompute membership masks (uses enum integer codes) ----
    stone_val = SolidGlyphs.S_stone.value
    solid_vals = np.array([g.value for g in SolidGlyphs], dtype=glyphs.dtype)
    exit_vals  = np.array([PassableGlyphs.S_dnladder.value,
                           PassableGlyphs.S_dnstair.value], dtype=glyphs.dtype)
    passable_vals = np.array([g.value for g in PassableGlyphs], dtype=glyphs.dtype)

    is_stone            = glyphs == stone_val
    is_solid_nonstone   = np.isin(glyphs, solid_vals) & ~is_stone
    is_exit             = np.isin(glyphs, exit_vals)
    is_passable_glyph   = np.isin(glyphs, passable_vals)

    # ---- Neighbor queries (vectorized) ----
    # For stones: if any neighbor is (passable AND visited) -> UNPASSABLE else UNSEEN
    mask_passable_and_visited = is_passable_glyph & visited_bool
    seen_from_neighbor = _neighbors_any(mask_passable_and_visited, DIRECTIONS)

    # ---- Initialize result as PASSABLE; then override by rule priority ----
    gk = np.full(glyphs.shape, GlyphKind.PASSABLE.value, dtype=np.uint8)

    # Exits
    gk[is_exit] = GlyphKind.EXIT.value

    # Non-stone solids are always UNPASSABLE
    gk[is_solid_nonstone] = GlyphKind.UNPASSABLE.value

    # Stone special rule
    stone_seen    = is_stone & seen_from_neighbor
    stone_unseen  = is_stone & ~seen_from_neighbor # pylint:disable=invalid-unary-operand-type
    gk[stone_seen]   = GlyphKind.UNPASSABLE.value
    gk[stone_unseen] = GlyphKind.UNSEEN.value

    # ---- Frontier pass: PASSABLE tiles that touch UNSEEN & not visited ----
    unseen_and_unvisited = (gk == GlyphKind.UNSEEN.value) & ~visited_bool
    # Reuse a workspace array to avoid an extra allocation:
    work = np.zeros_like(visited_bool)
    touches_unseen = _neighbors_any(unseen_and_unvisited, DIRECTIONS, out=work)
    frontier_mask = (gk == GlyphKind.PASSABLE.value) & touches_unseen & ~visited_bool
    gk[frontier_mask] = GlyphKind.FRONTIER.value

    return gk

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

def calculate_wavefront_and_glyph_kinds(glyphs: np.ndarray, floor_glyphs: np.ndarray,
                                        visited: np.ndarray) -> np.ndarray:
    """Calculate the wavefront from a NethackState and a list of target locations."""

    # pylint: disable=no-member
    glyph_kinds = calculate_glyph_kinds(floor_glyphs, visited)

    # existing targets (shape: [N, 2])
    targets = np.argwhere(
        (glyph_kinds == GlyphKind.EXIT.value) |
        (glyph_kinds == GlyphKind.FRONTIER.value)
    )

    # BONUS: vectorized way (faster than Python loops)
    is_cmap = np.vectorize(nle.nethack.glyph_is_cmap)
    is_mon  = np.vectorize(nle.nethack.glyph_is_monster)
    mask = (~visited.astype(bool)) & (~is_cmap(glyphs)) & (~is_mon(glyphs))
    extra = np.argwhere(mask)  # shape: [M, 2], dtype=int64

    if extra.size:
        targets = np.concatenate([targets, extra], axis=0)

    wavefront = calculate_wavefront(floor_glyphs, glyph_kinds, targets)
    return wavefront, glyph_kinds
