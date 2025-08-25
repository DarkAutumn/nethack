from typing import Any, Dict, Tuple
import numpy as np

from yndf.nethack_level import GLYPH_TABLE, DungeonLevel
from yndf.nethack_state import NethackState

def _get_dungeon_status(state: NethackState) -> Dict[str, any]:
    return {
        "depth": state.player.depth,
        "monster-level" : state.monster_level
    }

def _get_player_status(state: NethackState) -> Dict[str, any]:
    player = state.player
    result = {
        "position" : player.position,
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
        "inventory" : player.inventory
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

    search_targets = floor.search_score > 0.2
    if search_targets.any():
        result["search-targets"] = [_get_search_targets(state, "st", idx, pos)
                                     for idx, pos in enumerate(np.argwhere(search_targets))]

    interest_mask = GLYPH_TABLE.BODY | GLYPH_TABLE.OBJECT
    of_interest = (floor.properties & interest_mask) != 0
    if of_interest.any():
        result["other"] = [_get_object(state, "i", idx, pos) for idx, pos in enumerate(np.argwhere(of_interest))]

    return result

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

def get_status_dict(state: NethackState) -> Dict[str, Any]:
    result = {
        "player": _get_player_status(state),
        "dungeon": _get_dungeon_status(state),
        "floor": _get_floor_status(state),
    }

    if state.message:
        result["message"] = state.message

    return result
