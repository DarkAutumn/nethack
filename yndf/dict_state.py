from typing import Any, Dict, Tuple
import numpy as np

from yndf.nethack_level import GLYPH_TABLE
from yndf.nethack_state import NethackState

def _get_dungeon_status(state: NethackState) -> Dict[str, any]:
    return {
        "depth": state.player.depth,
        "monster-level" : state.monster_level,
        "time" : state.time
    }

def _get_player_status(state: NethackState) -> Dict[str, any]:
    player = state.player

    inventory = []
    for letter, name in player.inventory.items():
        inventory.append({"id": letter, "name": name})

    result = {
        "position" : player.position,
        "hp" : player.hp,
        "max"
        "str" : f"{player.str25}/{player.str125}",
        "dex" : player.dex,
        "con" : player.con,
        "intel" : player.intel,
        "wis" : player.wis,
        "cha" : player.cha,
        "ac" : player.ac,
        "level" : player.level,
        "exp" : player.exp,
        "hunger" : player.hunger.name,
        "status" : player.conditions,
        "gold" : player.gold,
        "inventory" : inventory,
        "can-move" : _get_movable_directions(state)
    }

    return result

def _get_relative_position(player_pos : Tuple[int, int], obj_pos : Tuple[int, int]) -> Tuple[int, int]:
    return (int(obj_pos[0] - player_pos[0]), int(obj_pos[1] - player_pos[1]))

def _get_floor_status(state: NethackState) -> Dict[str, any]:
    floor = state.floor
    result = {}

    if floor.enemies.any():
        result["enemies"] = [_get_object(state, "e", idx, pos) for idx, pos in enumerate(np.argwhere(floor.enemies))]

    if floor.pet_mask.any():
        result["pets"] = [_get_object(state, "p", idx, pos) for idx, pos in enumerate(np.argwhere(floor.pet_mask))]

    if floor.descend_mask.any():
        result["stairs-down"] = [_get_object(state, "s", idx, pos)
                                 for idx, pos in enumerate(np.argwhere(floor.descend_mask))]

    if floor.closed_doors.any():
        result["closed-doors"] = [_get_object(state, "c", idx, pos)
                                  for idx, pos in enumerate(np.argwhere(floor.closed_doors))]

    if floor.open_doors.any():
        result["open-doors"] = [_get_object(state, "o", idx, pos)
                                for idx, pos in enumerate(np.argwhere(floor.open_doors))]

    if floor.locked_doors.any():
        result["locked-doors"] = [_get_object(state, "l", idx, pos)
                                   for idx, pos in enumerate(np.argwhere(floor.locked_doors))]

    if floor.frontier.any():
        result["frontier"] = [_get_object(state, "f", idx, pos)
                               for idx, pos in enumerate(np.argwhere(floor.frontier))]

    search_targets = (floor.search_score > 0.2) & (floor.search_count < 30) & ~floor.frontier
    if search_targets.any():
        result["search-targets"] = [_get_search_targets(state, "st", idx, pos)
                                     for idx, pos in enumerate(np.argwhere(search_targets))]

    interest_mask = GLYPH_TABLE.BODY | GLYPH_TABLE.OBJECT
    of_interest = (floor.properties & interest_mask) != 0
    if of_interest.any():
        result["other"] = [_get_object(state, "i", idx, pos) for idx, pos in enumerate(np.argwhere(of_interest))]

    return result

DIRECTIONS = {
    (-1, -1) : "nw",
    (-1, 0) : "n",
    (-1, 1) : "ne",
    (0, -1) : "w",
    (0, 1) : "e",
    (1, -1) : "sw",
    (1, 0) : "s",
    (1, 1) : "se"
}

def _get_movable_directions(state : NethackState):
    result = []
    y, x = state.player.position
    for (dy, dx), direction in DIRECTIONS.items():
        ny, nx = y + dy, x + dx
        if _can_move(state, y, x, ny, nx, dy, dx):
            result.append(direction)

    return result

def _can_move(state: NethackState, y, x, ny, nx, dy, dx) -> bool:
    """Check if the player can move from one position to another."""
    floor = state.floor
    open_door = floor.open_doors

    # in-bounds?
    if not (0 <= ny < open_door.shape[0] and 0 <= nx < open_door.shape[1]):
        return False

    # Block diagonal through an open door (at either endpoint)
    if (dy != 0 and dx != 0 and (open_door[y, x] or open_door[ny, nx])):
        return False

    # Allow movement into closed doors that aren't locked.
    if floor.closed_doors[ny, nx]:
        return not floor.locked_doors[ny, nx]

    # Otherwise, check if the tile is passable.
    return floor.passable[ny, nx]


def _get_search_targets(state : NethackState, obj_prefix, idx, pos):
    result = {
        "id": f"{obj_prefix}_{idx}",
        "search-count" : int(state.floor.search_count[pos[0], pos[1]]),
        "search-score" : int(state.floor.search_score[pos[0], pos[1]]),
        "position": (int(pos[0]), int(pos[1])),
        "rel-pos": _get_relative_position(state.player.position, pos),
    }

    return result


def _get_object(state : NethackState, obj_prefix, idx, pos):
    result = {
        "id": f"{obj_prefix}_{idx}",
        "desc": state.get_screen_description(pos),
        "position": (int(pos[0]), int(pos[1])),
        "rel-pos": _get_relative_position(state.player.position, pos),
    }

    return result

def _get_history(messages, actions):
    return {
        "messages": list(messages),
        "actions": list(actions)
    }

def get_status_dict(state: NethackState, messages, actions) -> Dict[str, Any]:
    result = {
        "player": _get_player_status(state),
        "dungeon": _get_dungeon_status(state),
        "floor": _get_floor_status(state),
        "history" : _get_history(messages, actions)
    }

    if state.message:
        result["message"] = state.message

    return result
