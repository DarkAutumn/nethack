from enum import Enum
from collections import deque
from functools import cached_property
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
    PLAYER = _bit(22)

    UNUSED_BIT = 23

    FLOOR_MASK = CMAP | WALL | FLOOR | CORRIDOR | OPEN_DOOR | CLOSED_DOOR | DESCEND_LOCATION | STONE | TRAP

    OVERLAY_MASK = MONSTER | NORMAL_MONSTER | PET | RIDDEN_MONSTER | DETECTED_MONSTER | INVISIBLE | BODY | OBJECT \
                    | STATUE | SWALLOW | WARNING | PLAYER

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
        if glyph == 333:
            properties |= self.PLAYER
        else:
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

def _sat2d(a_bool: np.ndarray) -> np.ndarray:
    """Summed-area table with top/left zero padding. sat.shape = (H+1, W+1)."""
    sat = np.pad(a_bool.astype(np.int32), ((1,0), (1,0)))
    np.cumsum(sat, axis=0, out=sat)
    np.cumsum(sat, axis=1, out=sat)
    return sat

def _rect_sum(sat: np.ndarray, y0, y1, x0, x1):
    """
    Sum over half-open rectangle [y0:y1, x0:x1] for broadcastable y*/x* arrays.
    All indices are already clamped into [0..H] / [0..W].
    """
    return sat[y1, x1] - sat[y0, x1] - sat[y1, x0] + sat[y0, x0]

def _unseen_mass_by_dir(unseen_map : np.ndarray, depth: int, half_width: int):
    """
    Compute unseen mass 'behind' each cell for the 4 cardinals.
    Returns (north, south, west, east) int32 arrays, same shape as props.
    """
    # pylint: disable=too-many-locals
    h, w = unseen_map.shape
    unseen = unseen_map
    sat = _sat2d(unseen)

    y = np.arange(h)[:, None]  # shape (H,1)
    x = np.arange(w)[None, :]  # shape (1,W)

    # Common clamped spans across the perpendicular axis
    y0_band = np.clip(y - half_width, 0, h)
    y1_band = np.clip(y + half_width + 1, 0, h)
    x0_band = np.clip(x - half_width, 0, w)
    x1_band = np.clip(x + half_width + 1, 0, w)

    # EAST: rows [y0_band:y1_band], cols [x+1 : x+depth+1]
    x0_e = x + 1
    x1_e = np.minimum(w, x + depth + 1)
    east  = _rect_sum(sat, y0_band, y1_band, x0_e, x1_e)

    # WEST: rows [y0_band:y1_band], cols [x-depth : x]
    x0_w = np.maximum(0, x - depth)
    x1_w = x
    west  = _rect_sum(sat, y0_band, y1_band, x0_w, x1_w)

    # SOUTH: rows [y+1 : y+depth+1], cols [x0_band:x1_band]
    y0_s = y + 1
    y1_s = np.minimum(h, y + depth + 1)
    south = _rect_sum(sat, y0_s, y1_s, x0_band, x1_band)

    # NORTH: rows [y-depth : y], cols [x0_band:x1_band]
    y0_n = np.maximum(0, y - depth)
    y1_n = y
    north = _rect_sum(sat, y0_n, y1_n, x0_band, x1_band)

    return north, south, west, east  # int32 counts

def _shift_n(a):  # has True if neighbor to the NORTH is True
    out = np.zeros_like(a, dtype=bool)
    out[1:, :] = a[:-1, :]
    return out

def _shift_s(a):
    out = np.zeros_like(a, dtype=bool)
    out[:-1, :] = a[1:, :]
    return out

def _shift_w(a):
    out = np.zeros_like(a, dtype=bool)
    out[:, 1:] = a[:, :-1]
    return out

def _shift_e(a):
    out = np.zeros_like(a, dtype=bool)
    out[:, :-1] = a[:, 1:]
    return out

def _shift_nw(a):
    out = np.zeros_like(a, dtype=bool)
    out[1:, 1:] = a[:-1, :-1]
    return out

def _shift_ne(a):
    out = np.zeros_like(a, dtype=bool)
    out[1:, :-1] = a[:-1, 1:]
    return out

def _shift_sw(a):
    out = np.zeros_like(a, dtype=bool)
    out[:-1, 1:] = a[1:, :-1]
    return out

def _shift_se(a):
    out = np.zeros_like(a, dtype=bool)
    out[:-1, :-1] = a[1:, 1:]
    return out

class DungeonLevel:
    """A class to represent the current state of the dungeon floor."""
    UNSEEN_STONE = _bit(GLYPH_TABLE.UNUSED_BIT)
    VISITED = _bit(GLYPH_TABLE.UNUSED_BIT + 1)
    FRONTIER = _bit(GLYPH_TABLE.UNUSED_BIT + 2)
    TARGET = _bit(GLYPH_TABLE.UNUSED_BIT + 3)
    LOCKED_DOOR = _bit(GLYPH_TABLE.UNUSED_BIT + 4)
    WALLS_ADJACENT = _bit(GLYPH_TABLE.UNUSED_BIT + 5)
    DEAD_END = _bit(GLYPH_TABLE.UNUSED_BIT + 6)

    def __init__(self, glyphs: np.ndarray, unpassable, locked, prev : 'DungeonLevel' = None):
        self.glyphs = glyphs

        self.properties = GLYPH_TABLE.properties[glyphs] & ((1 << GLYPH_TABLE.UNUSED_BIT) - 1)
        player = (self.properties & GLYPH_TABLE.PLAYER) != 0
        self.properties[player] |= self.VISITED

        # If we have a previous state, use that to determine what's under movable glyphs
        if prev is not None:
            overlay = (self.properties & GLYPH_TABLE.OVERLAY_MASK) != 0
            if overlay.any():
                prev_floor = prev.properties & GLYPH_TABLE.FLOOR_MASK
                self.properties[overlay] |= prev_floor[overlay]

            # Mark visited tiles
            self.properties |= (prev.properties & self.VISITED)

            # If any monster was on a tile, but now it has objects on it, mark as unvisited so we don't skip the item
            prev_visited = (prev.properties & self.VISITED) != 0
            prev_monster = (prev.properties & GLYPH_TABLE.MONSTER) != 0
            objects_now  = (self.properties & GLYPH_TABLE.OBJECT) != 0
            unvisit = prev_visited & prev_monster & objects_now
            self.properties[unvisit] &= ~self.VISITED

        for pos in locked:
            if self.properties[pos] & GLYPH_TABLE.CLOSED_DOOR:
                self.properties[pos] |= self.LOCKED_DOOR

        for pos in unpassable:
            self.properties[pos] &= ~GLYPH_TABLE.PASSABLE

        walls_adjacent = self._calculate_walls_adjacent_mask()
        self.properties[walls_adjacent] |= self.WALLS_ADJACENT

        unseen_stone_mask = self._calculate_unseen_stone()
        self.properties[unseen_stone_mask] |= self.UNSEEN_STONE

        frontier_mask = self._calculate_frontier_mask()
        self.properties[frontier_mask] |= self.FRONTIER

        target_mask = self._get_target_mask()
        self.properties[target_mask] |= self.TARGET

        dead_end_mask = self._calculate_dead_end_mask()
        self.properties[dead_end_mask] |= self.DEAD_END

        self.search_count = prev.search_count if prev else np.zeros_like(self.glyphs, dtype=np.uint8)
        self.search_score = self._compute_search_score()
        self.wavefront = self._calculate_wavefront()

    @cached_property
    def num_enemies(self):
        """Count of visible enemies on the level."""
        enemies = (self.properties & (GLYPH_TABLE.MONSTER | GLYPH_TABLE.PET)) == GLYPH_TABLE.MONSTER
        return np.sum(enemies)

    @cached_property
    def stone_tile_count(self):
        """Count of stone tiles on the level."""
        stone = (self.properties & GLYPH_TABLE.STONE) != 0
        return np.sum(stone)

    def _calculate_walls_adjacent_mask(self) -> np.ndarray:
        walls = (self.properties & (GLYPH_TABLE.WALL | GLYPH_TABLE.STONE)) != 0
        adj = np.zeros_like(walls, dtype=bool)
        # cardinals only (no wraparound)
        adj[1:,  :] |= walls[:-1,  :]  # north neighbor is wall
        adj[:-1, :] |= walls[1:,   :]  # south
        adj[:, 1:]  |= walls[:,  :-1]  # west
        adj[:, :-1] |= walls[:,   1:]  # east
        return adj

    def _calculate_unseen_stone(self):
        visited = (self.properties & self.VISITED) != 0
        stone   = (self.properties & GLYPH_TABLE.STONE) != 0
        visited_nbr    = self._any_neighbor(visited)
        unseen_stone   = stone & ~visited_nbr
        return unseen_stone

    def _any_neighbor(self, a: np.ndarray) -> np.ndarray:
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

    def _calculate_frontier_mask(self) -> np.ndarray:
        props = self.properties
        passbl  = (props & GLYPH_TABLE.PASSABLE) != 0
        visited = (props & self.VISITED) != 0
        unseen_stone = (props & self.UNSEEN_STONE) != 0

        near_unseen    = self._any_neighbor(unseen_stone)

        frontier_mask  = passbl & ~visited & near_unseen
        return frontier_mask

    def _get_target_mask(self):
        """Calculate the wavefront targets for the current state."""
        passable = (self.properties & GLYPH_TABLE.PASSABLE) != 0
        visited = (self.properties & self.VISITED) != 0
        objects = (self.properties & GLYPH_TABLE.OBJECT) != 0

        interesting_objects = ~visited & objects & passable

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


    def _compute_search_score(self, depth=7, half_width=3) -> np.ndarray:
        props = self.properties
        h, w = props.shape

        # masks
        dead_end = (props & self.DEAD_END) != 0
        passable = (props & GLYPH_TABLE.PASSABLE) != 0
        corridor = (props & GLYPH_TABLE.CORRIDOR) != 0
        floor    = (props & GLYPH_TABLE.FLOOR)    != 0
        wall     = (props & GLYPH_TABLE.WALL)     != 0
        stone    = (props & GLYPH_TABLE.STONE)    != 0
        has_adj_wall = (props & self.WALLS_ADJACENT) != 0
        barrier  = wall | stone

        # neighbor masks
        wall_n, wall_s, wall_w, wall_e = _shift_n(wall), _shift_s(wall), _shift_w(wall), _shift_e(wall)
        corr_n, corr_s, corr_w, corr_e = _shift_n(corridor), _shift_s(corridor), _shift_w(corridor), _shift_e(corridor)

        # diagonal-only access gate (optional but per your spec)
        card_pass = _shift_n(passable) | _shift_s(passable) | _shift_w(passable) | _shift_e(passable)
        diag_pass = _shift_nw(passable) | _shift_ne(passable) | _shift_sw(passable) | _shift_se(passable)
        diag_only = (~card_pass) & diag_pass

        # unseen mass (per direction) from the wall cell adjacent to (y,x)
        unseen = (props & self.UNSEEN_STONE) != 0  # your UNSEEN flag over stone
        m_n0, m_s0, m_w0, m_e0 = _unseen_mass_by_dir(unseen, depth, half_width)
        m_n = np.zeros((h, w), np.int32)
        m_n[1:,  :] = (m_n0[:-1,  :] * barrier[:-1,  :])
        m_s = np.zeros((h, w), np.int32)
        m_s[:-1, :] = (m_s0[1:,   :] * barrier[1:,   :])
        m_w = np.zeros((h, w), np.int32)
        m_w[:, 1:]  = (m_w0[:,  :-1] * barrier[:,  :-1])
        m_e = np.zeros((h, w), np.int32)
        m_e[:, :-1] = (m_e0[:,   1:] * barrier[:,   1:])
        mass_any = np.maximum.reduce([m_n, m_s, m_w, m_e])

        # wall run-length of the adjacent wall in each direction
        run_h, run_v = self._calculate_wall_run_lengths(wall)
        r_n = np.zeros((h, w), np.int16)
        r_n[1:,  :] = (run_h[:-1,  :] * wall[:-1,  :])
        r_s = np.zeros((h, w), np.int16)
        r_s[:-1, :] = (run_h[1:,   :] * wall[1:,   :])
        r_w = np.zeros((h, w), np.int16)
        r_w[:, 1:]  = (run_v[:,  :-1] * wall[:,  :-1])
        r_e = np.zeros((h, w), np.int16)
        r_e[:, :-1] = (run_v[:,   1:] * wall[:,   1:])

        # ---------- Base patterns (per-tile, take max) ----------
        score = np.zeros((h, w), dtype=np.float32)

        # 1) Corridor dead-end tip (you face the dead end): 1.0
        #    Count *corridor* neighbors (cardinals).
        base_dead = dead_end.astype(np.float32)

        # use the largest unseen mass behind any adjacent barrier face
        factor_dead = np.minimum(1.0, mass_any.astype(np.float32) / 20.0)
        score = np.maximum(score, base_dead * factor_dead)

        # 2) Room wall with ≥3 contiguous wall tiles and unknown beyond: 0.7
        #    Check per direction from a ROOM tile.
        room = floor  # your FLOOR flag denotes room floor
        cond_n = room & wall_n & (r_n >= 3) & (m_n > 0)
        cond_s = room & wall_s & (r_s >= 3) & (m_s > 0)
        cond_w = room & wall_w & (r_w >= 3) & (m_w > 0)
        cond_e = room & wall_e & (r_e >= 3) & (m_e > 0)

        sc_n = np.where(cond_n, 0.7 * np.minimum(1.0, m_n.astype(np.float32) / 20.0), 0.0)
        sc_s = np.where(cond_s, 0.7 * np.minimum(1.0, m_s.astype(np.float32) / 20.0), 0.0)
        sc_w = np.where(cond_w, 0.7 * np.minimum(1.0, m_w.astype(np.float32) / 20.0), 0.0)
        sc_e = np.where(cond_e, 0.7 * np.minimum(1.0, m_e.astype(np.float32) / 20.0), 0.0)
        score = np.maximum.reduce([score, sc_n, sc_s, sc_w, sc_e])

        # 3) Corridor bend with unknown behind the outer wall: 0.5
        ns = corr_n | corr_s
        ew = corr_w | corr_e
        straight = (corr_n & corr_s) | (corr_w & corr_e)
        bend = corridor & ns & ew & ~straight

        outer_n = bend & (~corr_n) & wall_n & (m_n > 0)
        outer_s = bend & (~corr_s) & wall_s & (m_s > 0)
        outer_w = bend & (~corr_w) & wall_w & (m_w > 0)
        outer_e = bend & (~corr_e) & wall_e & (m_e > 0)

        sc_n = np.where(outer_n, 0.5 * np.minimum(1.0, m_n.astype(np.float32) / 20.0), 0.0)
        sc_s = np.where(outer_s, 0.5 * np.minimum(1.0, m_s.astype(np.float32) / 20.0), 0.0)
        sc_w = np.where(outer_w, 0.5 * np.minimum(1.0, m_w.astype(np.float32) / 20.0), 0.0)
        sc_e = np.where(outer_e, 0.5 * np.minimum(1.0, m_e.astype(np.float32) / 20.0), 0.0)
        score = np.maximum.reduce([score, sc_n, sc_s, sc_w, sc_e])

        # 4) Long room wall with no visible doors on that wall and unknown behind:
        #    0.6 + 0.05×(wall_run_length≥5), clipped to 0.9
        #    (Doors break wall runs anyway, so "no doors" implied by the run.)
        long_n = room & wall_n & (m_n > 0)
        long_s = room & wall_s & (m_s > 0)
        long_w = room & wall_w & (m_w > 0)
        long_e = room & wall_e & (m_e > 0)

        base_n = 0.6 + 0.05 * (r_n >= 5)
        base_s = 0.6 + 0.05 * (r_s >= 5)
        base_w = 0.6 + 0.05 * (r_w >= 5)
        base_e = 0.6 + 0.05 * (r_e >= 5)

        base_n = np.minimum(base_n, 0.9).astype(np.float32)
        base_s = np.minimum(base_s, 0.9).astype(np.float32)
        base_w = np.minimum(base_w, 0.9).astype(np.float32)
        base_e = np.minimum(base_e, 0.9).astype(np.float32)

        sc_n = np.where(long_n, base_n * np.minimum(1.0, m_n.astype(np.float32) / 20.0), 0.0)
        sc_s = np.where(long_s, base_s * np.minimum(1.0, m_s.astype(np.float32) / 20.0), 0.0)
        sc_w = np.where(long_w, base_w * np.minimum(1.0, m_w.astype(np.float32) / 20.0), 0.0)
        sc_e = np.where(long_e, base_e * np.minimum(1.0, m_e.astype(np.float32) / 20.0), 0.0)
        score = np.maximum.reduce([score, sc_n, sc_s, sc_w, sc_e])

        # ---------- global gates ----------
        score[~has_adj_wall] = 0.0
        score[diag_only] = 0.0

        return score.astype(np.float32)

    def _calculate_dead_end_mask(self) -> np.ndarray:
        corr = (self.properties & GLYPH_TABLE.CORRIDOR) != 0
        passable = (self.properties & GLYPH_TABLE.PASSABLE) != 0

        # count cardinal corridor neighbors
        nbh = np.zeros_like(corr, dtype=np.uint8)
        nbh[1:,  :] += passable[:-1, :]
        nbh[:-1, :] += passable[ 1:, :]
        nbh[:, 1:]  += passable[:, :-1]
        nbh[:, :-1] += passable[:,  1:]

        return corr & (nbh <= 1)

    def _calculate_unseen_mass_adjacent(self, depth=7, half_width=3):
        # Precompute once per frame
        unseen = (self.properties & self.UNSEEN_STONE) != 0
        north, south, west, east = _unseen_mass_by_dir(unseen, depth, half_width)

        barrier = (self.properties & (GLYPH_TABLE.WALL | GLYPH_TABLE.STONE)) != 0

        # Pull the neighbor’s mass in each direction (align back onto the center cell)
        mass_from_n = np.zeros(self.properties.shape, dtype=np.int32)
        mass_from_s = np.zeros(self.properties.shape, dtype=np.int32)
        mass_from_w = np.zeros(self.properties.shape, dtype=np.int32)
        mass_from_e = np.zeros(self.properties.shape, dtype=np.int32)

        # neighbor above/below/left/right must be a barrier
        mass_from_n[1:,  :] = (north[:-1,  :] * barrier[:-1,  :])
        mass_from_s[:-1, :] = (south[1:,   :] * barrier[1:,   :])
        mass_from_w[:, 1:]  = (west[:,  :-1] * barrier[:,  :-1])
        mass_from_e[:, :-1] = (east[:,  1:]  * barrier[:,   1:])

        max_mass = np.maximum.reduce([mass_from_n, mass_from_s, mass_from_w, mass_from_e])
        return max_mass


    def _calculate_wall_run_lengths(self, walls : np.ndarray):
        h, w = walls.shape

        # --- horizontal runs (along x) ---
        lab_h = np.cumsum(~walls, axis=1)                                # constant within True-runs
        gid_h = lab_h + (np.arange(h, dtype=np.int64)[:, None] * (w+1)) # unique per row
        cnt_h = np.bincount(gid_h[walls].ravel(), minlength=h*(w+1))
        run_h = np.zeros_like(lab_h, dtype=np.int16)
        run_h[walls] = cnt_h[gid_h[walls]].astype(np.int16, copy=False)

        # --- vertical runs (along y) ---
        lab_v = np.cumsum(~walls, axis=0)
        gid_v = lab_v + (np.arange(w, dtype=np.int64)[None, :] * (h+1)) # unique per col
        cnt_v = np.bincount(gid_v[walls].ravel(), minlength=w*(h+1))
        run_v = np.zeros_like(lab_v, dtype=np.int16)
        run_v[walls] = cnt_v[gid_v[walls]].astype(np.int16, copy=False)

        return run_h, run_v

    def _calculate_adj_max_wall_len(self):
        walls = (self.properties & GLYPH_TABLE.WALL) != 0
        run_h, run_v = self._calculate_wall_run_lengths(walls)

        # For a passable tile at (y,x), the neighbor wall at:
        #  - North uses run_h at (y-1,x)  (wall face is horizontal)
        #  - South uses run_h at (y+1,x)
        #  - West  uses run_v at (y,x-1)  (wall face is vertical)
        #  - East  uses run_v at (y,x+1)

        north = np.zeros(walls.shape, dtype=np.int16)
        south = np.zeros(walls.shape, dtype=np.int16)
        west  = np.zeros(walls.shape, dtype=np.int16)
        east  = np.zeros(walls.shape, dtype=np.int16)

        north[1:,  :] = (run_h[:-1,  :] * walls[:-1,  :])
        south[:-1, :] = (run_h[1:,   :] * walls[1:,   :])
        west[:, 1:]   = (run_v[:,  :-1] * walls[:,  :-1])
        east[:, :-1]  = (run_v[:,   1:] * walls[:,   1:])

        max_adjacent = np.maximum.reduce([north, south, west, east])
        return max_adjacent
