from functools import partial
from flax.struct import dataclass
import jax.numpy as jnp
import jax
from typing import List
import json
import pickle
import enum

from game.resources import RESOURCE_TO_IDX
from dataclasses import fields, is_dataclass
from .techs import Technologies
from game.constants import DEAD_CITY_HEAL_LEVEL


@dataclass
class Units:
    """
    Holds all units in the game

    shapes are like (6, 250) == (num_players, 250)

    Attributes:
        military (bool)
        unit_type: e.g., settler, warrior, worker, etc
        unit_rowcol
        unit_ap
        engaged_for_n_turns: the number of turns a unit is committed to a specific type of action. E.g., a worker is spending
            4 turns on a mine, a caravann is spending 12 turns on a trade route. During this time, the unit cannot be moved
            or commit to another type of action.
        engaged_action_id: the action ID that should be completed once the given unit's engagement ends. E.g., finishing an improvement
        current_idx (int): used to add and delete units to the game. points to the current first **open** idx

        NOTE: The following fields are only used by caravan-type units. We do not need a "trade_from_player_int"
            field, as this can easily be inferred from the idx of "trade_to_player_int"
        trade_to_player_int: [i, j] is from player i to player j
        trade_to_city_int: [k] is to city k of player j (as determined by previous field)
        trade_from_city_int: [l] if from city l of player i (as determined by 2-prev field)
        trade_yields:  [i, j] is yields to player i and j (as determined by 3-prev field)
            [standard 8 yields, influence, religious pressure]
    """
    military: jnp.ndarray
    unit_type: jnp.ndarray
    unit_rowcol: jnp.ndarray
    unit_ap: jnp.ndarray
    engaged_for_n_turns: jnp.ndarray
    engaged_action_id: jnp.ndarray
    trade_to_player_int: jnp.ndarray
    trade_to_city_int: jnp.ndarray
    trade_from_city_int: jnp.ndarray
    trade_yields: jnp.ndarray
    combat_bonus_accel: jnp.ndarray
    health: jnp.ndarray

    @classmethod
    def create(cls, num_players, max_units, settler_rowcols, warrior_rowcols):
        """
        This method is meant to be invoked at map-creation time.

        Both engaged* arrays need to be signed due to the way we check for unit movements in 
        GameState.step_units
        """

        military = jnp.zeros(shape=(num_players, max_units), dtype=jnp.uint8)
        unit_type = jnp.zeros(shape=(num_players, max_units), dtype=jnp.int32)
        unit_rowcol = jnp.zeros(shape=(num_players, max_units, 2), dtype=jnp.int32)
        unit_ap = jnp.zeros(shape=(num_players, max_units), dtype=jnp.int32)
        engaged_for_n_turns = jnp.zeros(shape=(num_players, max_units), dtype=jnp.int8)
        engaged_action_id = jnp.zeros(shape=(num_players, max_units), dtype=jnp.int8)
        trade_to_player_int = jnp.zeros(shape=(num_players, max_units), dtype=jnp.uint8)
        trade_to_city_int = jnp.zeros(shape=(num_players, max_units), dtype=jnp.uint8)
        trade_from_city_int = jnp.zeros(shape=(num_players, max_units), dtype=jnp.uint8)
        trade_yields = jnp.zeros(shape=(num_players, max_units, 2, 10), dtype=jnp.float32)  # from-to ([0,1] axis 2) format
        combat_bonus_accel = jnp.ones(shape=(num_players, max_units), dtype=jnp.float32)
        health = jnp.ones(shape=(num_players, max_units), dtype=jnp.float32)

        # Player civs will begin the game with a single settler and warrior
        for i, rowcol in enumerate(settler_rowcols):
            unit_type = unit_type.at[i, 0].set(UnitTypeTable.settler)
            unit_rowcol = unit_rowcol.at[i, 0].set(rowcol)
            unit_ap = unit_ap.at[i, 0].set(UnitAPTable.settler)

        for i, rowcol in enumerate(warrior_rowcols):
            military = military.at[i, 1].set(1)
            unit_type = unit_type.at[i, 1].set(UnitTypeTable.warrior)
            unit_rowcol = unit_rowcol.at[i, 1].set(rowcol)
            unit_ap = unit_ap.at[i, 1].set(UnitAPTable.warrior)

        return cls(
            military=military,
            unit_type=unit_type,
            unit_rowcol=unit_rowcol,
            unit_ap=unit_ap,
            engaged_for_n_turns=engaged_for_n_turns,
            engaged_action_id=engaged_action_id,
            trade_to_player_int=trade_to_player_int,
            trade_to_city_int=trade_to_city_int,
            trade_from_city_int=trade_from_city_int,
            trade_yields=trade_yields,
            combat_bonus_accel=combat_bonus_accel,
            health=health
        )


class UnitTypeTable(enum.IntEnum):
    """
    This value will let us know what action categories to mask during unit movement
    This value will be used to idx UnitActionCategoryMask[unit_type - 1]
    """
    settler = 1
    warrior = 2
    worker = 3
    archer = 4
    caravan = 5
    chariot_archer = 6
    pikeman = 7
    scout = 8
    spearman = 9
    catapult = 10
    composite_bowman = 11
    horseman = 12
    swordsman = 13
    crossbowman = 14
    knight = 15
    longswordsman = 16
    trebuchet = 17
    cannon = 18
    lancer = 19
    musketman = 20
    airship = 21
    artillery = 22
    cavalry = 23
    expeditionary_force = 24
    gatlinggun = 25
    rifleman = 26
    anti_tank_rifle = 27
    infantry = 28
    landship = 29
    machine_gun = 30
    anti_tank_gun = 31
    bazooka = 32
    helicopter_gunship = 33
    marine = 34
    rocket_artillery = 35
    tank = 36
    xcom_squad = 37


class UnitAPTable(enum.IntEnum):
    settler = 2
    warrior = 2
    worker = 2

class UnitActionCategories(enum.IntEnum):
    """This is only to be used to set up action spaces for units in units.py"""
    move = 0, 0
    settle = 1, 1
    farm = 2, 1
    pasture = 3, 1
    mine = 4, 1
    fishing_boat = 5, 1
    plantation = 6, 1
    camp = 7, 1
    quarry = 8, 1
    lumber_mill = 9, 1
    fort = 10, 1
    trading_post = 11, 1
    road = 12, 1
    chop_forest = 13, 1
    chop_jungle = 14, 1
    clear_marsh = 15, 1
    move_caravan = 16, 2
    send_caravan = 17, 2
    combat = 18, 0
    
    def __new__(cls, value, ap_adj):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.ap_adj = ap_adj
        return obj
    

UnitActionCategoryAPAdj = jnp.array([x.ap_adj for x in UnitActionCategories])


class GameUnits(enum.IntEnum):
    """
    name = id, prod_cost, faith_cost, combat, ap, prereq, obsolete, prereq_res, range

    Some of these units will have an obsolete_tech of "future_tech". Please ensure that "future_tech"
    is never completable, otherwise many untils will be erroneously unbuildable

    0: non-mil (need this s.t. the lookup does not erroneously increment mil/non-mil stack  combat type)
    1: basic unit
    2: mounted
    3: anti-mounted (+50%)
    4: siege (+100%)
    5: armored
    6: anti-armor (+100%)
    """
    settler = 1, 73, 0, 0, 2, Technologies.agriculture._value_, Technologies.future_tech._value_, [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [], 0, 0
    warrior = 2, 40, 80, 8, 2, Technologies.agriculture._value_, Technologies.iron_working._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [], 1, 1
    worker = 3, 48, 0, 0, 2, Technologies.agriculture._value_, Technologies.future_tech._value_, [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], [], 0, 0
    archer = 4, 40, 80, 5, 2, Technologies.archery._value_, Technologies.construction._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [], 2, 1
    caravan = 5, 75, 0, 0, 1, Technologies.animal_husbandry._value_, Technologies.future_tech._value_, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0], [], 0, 0
    chariot_archer = 6, 56, 112, 6, 4, Technologies.the_wheel._value_, Technologies.chivalry._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [], 2, 2
    pikeman = 7, 90, 180, 16, 2, Technologies.civil_service._value_, Technologies.metallurgy._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [], 1, 3
    scout = 8, 25, 75, 5, 3, Technologies.agriculture._value_, Technologies.education._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [], 1,  1
    spearman = 9, 56, 112, 11, 2, Technologies.bronze_working._value_, Technologies.civil_service._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [], 1, 3
    catapult = 10, 75, 150, 7, 2, Technologies.mathematics._value_, Technologies.physics._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [], 2, 4
    composite_bowman = 11, 75, 150, 7, 2, Technologies.construction._value_, Technologies.machinery._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [], 2,  1
    horseman = 12, 75, 150, 12, 4, Technologies.horseback_riding._value_, Technologies.chivalry._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [RESOURCE_TO_IDX["horses"]], 1, 2
    swordsman = 13, 75, 150, 14, 2, Technologies.iron_working._value_, Technologies.steel._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [RESOURCE_TO_IDX["iron"]], 1, 1
    crossbowman = 14, 120, 240, 13, 2, Technologies.machinery._value_, Technologies.industrialization._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [], 2, 1
    knight = 15, 120, 240, 20, 4, Technologies.chivalry._value_, Technologies.military_science._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [RESOURCE_TO_IDX["horses"]], 1, 2
    longswordsman = 16, 120, 240, 21, 2, Technologies.steel._value_, Technologies.gunpowder._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [RESOURCE_TO_IDX["iron"]], 1, 1
    trebuchet = 17, 120, 240, 12, 2, Technologies.physics._value_, Technologies.chemistry._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [], 2, 4
    cannon = 18, 185, 370, 14, 2, Technologies.chemistry._value_, Technologies.dynamite._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [], 2, 4
    lancer = 19, 185, 370, 25, 4, Technologies.metallurgy._value_, Technologies.railroad._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [], 1, 2
    musketman = 20, 150, 300, 24, 2, Technologies.gunpowder._value_, Technologies.rifling._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [], 1, 1
    airship = 21, 150, 0, 15, 6, Technologies.steam_power._value_, Technologies.computers._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [], 0, 1
    artillery = 22, 250, 500, 21, 2, Technologies.dynamite._value_, Technologies.rocketry._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [], 3, 4
    cavalry = 23, 225, 450, 34, 4, Technologies.military_science._value_, Technologies.combustion._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [RESOURCE_TO_IDX["horses"]], 1, 2
    expeditionary_force = 24, 270, 700, 40, 3, Technologies.biology._value_, Technologies.penicilin._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [], 1, 1
    gatlinggun = 25, 225, 450, 30, 2, Technologies.industrialization._value_, Technologies.ballistics._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [], 1, 1
    rifleman = 26, 225, 450, 34, 2, Technologies.rifling._value_, Technologies.replaceable_parts._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [], 1, 1
    anti_tank_rifle = 27, 300, 600, 30, 2, Technologies.railroad._value_, Technologies.combined_arms._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [], 1, 6
    infantry = 28, 420, 840, 70, 2, Technologies.electronics._value_, Technologies.mobile_tactics._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [], 1, 1
    landship = 29, 390, 200, 60, 4, Technologies.combustion._value_, Technologies.combined_arms._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [RESOURCE_TO_IDX["oil"]], 1, 5
    machine_gun = 30, 390, 780, 50, 2, Technologies.ballistics._value_, Technologies.nuclear_fission._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [], 1, 1
    anti_tank_gun = 31, 360, 720, 50, 2, Technologies.combined_arms._value_, Technologies.future_tech._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [], 1, 6
    bazooka = 32, 450, 900, 80, 2, Technologies.nuclear_fission._value_, Technologies.future_tech._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [], 2, 1
    helicopter_gunship = 33, 510, 1020, 70, 8, Technologies.computers._value_, Technologies.future_tech._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [RESOURCE_TO_IDX["aluminium"]], 1, 5
    marine = 34, 450, 800, 80, 3, Technologies.penicilin._value_, Technologies.future_tech._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [], 1, 1
    rocket_artillery = 35, 510, 1020, 45, 2, Technologies.rocketry._value_, Technologies.future_tech._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [], 2, 1
    tank = 36, 450, 900, 70, 5, Technologies.combined_arms._value_, Technologies.future_tech._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [RESOURCE_TO_IDX["oil"]], 1, 5
    xcom_squad = 37, 510, 800, 100, 2, Technologies.nanotechnology._value_, Technologies.future_tech._value_, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [], 1, 1

    def __new__(cls, value: int, cost: int, faith_cost: int, combat: int, ap: int, prereq: int, obsolete: int, a_cat_mask: jnp.ndarray, prereq_res: List[int], _range: int, combat_type: int):
        obj = int.__new__(cls, value)     # create the real `int`
        obj._value_ = value
        obj.cost = cost 
        obj.faith_cost = faith_cost
        obj.combat = combat
        obj.ap = ap
        obj.prereq = prereq
        obj.obsolete = obsolete
        obj.a_cat_mask = a_cat_mask
        obj.prereq_res = prereq_res
        obj.range = _range
        obj.combat_type = combat_type
        return obj



# This will need to be indexed as unitID - 1
# refer to ALL_ACTION_FUNCTIONS in action_space.py
# [identity, settle, improvement?]
UnitActionCategoryMask = jnp.concatenate([jnp.array(x.a_cat_mask)[None] for x in GameUnits], axis=0)

ALL_UNIT_COST = jnp.array([x.cost * 0.67 for x in GameUnits])
ALL_UNIT_COMBAT = jnp.array([x.combat for x in GameUnits])
ALL_UNIT_AP = jnp.array([x.ap for x in GameUnits])
ALL_UNIT_TECH_PREREQ = jnp.array([x.prereq for x in GameUnits])
ALL_UNIT_RANGE = jnp.array([x.range for x in GameUnits])
ALL_UNIT_COMBAT_TYPE = jnp.array([x.combat_type for x in GameUnits])

# We use the comat  type of a unit as the index in the following array. The value at that index represents
# the unit combat type that this indexing compat type gets a bonus against!
ALL_UNIT_COMBAT_TYPE_GOOD_AGAINST = jnp.array([0, 0, 0, 2, 0, 0, 5])
ALL_UNIT_COMBAT_BONUS = jnp.array([1, 1, 1, 1.5, 2.0, 1, 2.0])

NUM_UNITS = len(GameUnits)

def _check_prereq_units(
        techs: jnp.ndarray,
        city_owned_res: jnp.ndarray,
        req_indices_tech: tuple[int, ...],
        req_indices_res: tuple[int, ...],
        obsolete_indices_tech: tuple[int, ...]
    ):
    req = jnp.asarray(req_indices_tech, dtype=jnp.int32)
    tech_prereq = jnp.all(techs[req] == 1)
        
    # The resources are +1, so need to -1
    if len(req_indices_res) == 0:
        res_prereq = True
    else:
        req = jnp.asarray(req_indices_res, dtype=jnp.int32)
        res_prereq = jnp.all(city_owned_res[req - 1] > 0)

    req = jnp.asarray(obsolete_indices_tech, dtype=jnp.int32)
    not_obsolete = jnp.all(techs[req] == 0)
    return tech_prereq & res_prereq & not_obsolete


ALL_UNIT_PREREQ_FN = []
for unit in GameUnits:
    fn = partial(
        _check_prereq_units,
        req_indices_tech=unit.prereq,
        req_indices_res=unit.prereq_res,
        obsolete_indices_tech=unit.obsolete
    )
    ALL_UNIT_PREREQ_FN.append(fn)

"""
In comparison to buildings, adding units to the game is simple. 
As this is being deployed within a switch statement, let's input the smallest atomic objects...
"""

def add_unit_to_game_minimal(player_units, player_trade_routes, city_rowcol, unit_type):
    _military = (ALL_UNIT_COMBAT[unit_type] > 0).astype(player_units.military.dtype)
    _unit_type = (unit_type + 1).astype(player_units.unit_type.dtype)
    _unit_rowcol = city_rowcol
    _unit_ap = (ALL_UNIT_AP[unit_type]).astype(player_units.unit_ap.dtype)
    return _military, _unit_type, _unit_rowcol, _unit_ap, player_trade_routes


def add_unit_to_game(player_units, player_trade_routes, city_rowcol, unit_type):
    """
    We'll just need some quick check for traderoute like unit_type == GameUnits["caravan"]._value_ or something
    This function should never be called if we are at the max number of units.
    """
    slot_to_use = player_units.unit_type.argmin()
    _military = player_units.military.at[slot_to_use].set(
        (ALL_UNIT_COMBAT[unit_type] > 0).astype(player_units.military.dtype)
    )
    # unit type is always + 1, as 0 indicates "no units"
    _unit_type = player_units.unit_type.at[slot_to_use].set(
        (unit_type + 1).astype(player_units.unit_type.dtype)
    )
    _unit_rowcol = player_units.unit_rowcol.at[slot_to_use].set(
        city_rowcol
    )
    _unit_ap = player_units.unit_ap.at[slot_to_use].set(
        (ALL_UNIT_AP[unit_type]).astype(player_units.unit_ap.dtype)
    )

    new_player_units = player_units.replace(
        military=_military,
        unit_type=_unit_type,
        unit_rowcol=_unit_rowcol,
        unit_ap=_unit_ap
    )
    return new_player_units, player_trade_routes


def kill_units(game):
    """
    This function operates on a per-game basis
    """
    health_mask = game.units.health <= 0

    updated_units = jax.tree_map(
        lambda x: x * jnp.reshape(~health_mask, health_mask.shape + (1,) * (x.ndim - health_mask.ndim)),
        game.units,
    )
    
    # Need to set back to one: combat_bonus_accel, health
    updated_units.replace(
        combat_bonus_accel=updated_units.combat_bonus_accel + (1 * health_mask),
        health=updated_units.health + (1 * health_mask)
    )

    return game.replace(units=updated_units)

def _is_additional_yield_map(x, *, path):
    """Helper to identify additional_yield_map by path."""
    return path[-1].key == 'additional_yield_map' if path else False


def transfer_cities(game, player_id):
    """
    If city.hp < 0, then:
        if not cap:
            if (game.player_cities.city_ids > 0).sum() > 0:
                absorb city
            else:
                delete city
        else:
            log sacked
    """
    are_slots_available = (game.player_cities.city_ids[player_id[0]] < 1).sum() > 0
    available_slot = game.player_cities.city_ids[player_id[0]].argmin()
    
    # cities spawn with 2 HP, so this mask is safe
    dead_city_mask = game.player_cities.hp < 0
    is_cap_bool = (game.player_cities.city_ids * dead_city_mask).sum() == 1
    dead_player_id = (game.player_cities.city_ids * dead_city_mask).sum(-1).argmax() 
    dead_city_idx = dead_city_mask[dead_player_id].argmax()

    # The easiest thing to do is to reach into the `initial_state_cache` objects
    pre_cities = game.player_cities
    initial_cache_cities = game.initial_state_cache.player_cities

    def replace_dead_cities(pre_cities, initial_cache_cities, dead_city_mask):
        """
        Replace dead cities with initial state values using dataclass fields.
        
        Args:
            pre_cities: Current cities dataclass
            initial_cache_cities: Initial state cities dataclass  
            dead_city_mask: Boolean/binary mask array of shape (6, max_num_cities)
        
        Returns:
            New cities dataclass with dead cities replaced
        """
        def process_field(pre_val, initial_val, field_name):
            # Skip additional_yield_map
            if field_name == 'additional_yield_map':
                return pre_val
            
            # Handle nested dataclass (religion_info)
            if is_dataclass(pre_val):
                # Recursively process nested dataclass fields
                return type(pre_val)(**{
                    nested_f.name: process_field(
                        getattr(pre_val, nested_f.name),
                        getattr(initial_val, nested_f.name),
                        nested_f.name  # Pass nested field name (though not used currently)
                    )
                    for nested_f in fields(pre_val)
                })
            
            # Handle regular array fields - apply mask
            extra_dims = pre_val.ndim - dead_city_mask.ndim
            mask_expanded = dead_city_mask.reshape(
                dead_city_mask.shape + (1,) * extra_dims
            )
            return jnp.where(mask_expanded, initial_val, pre_val)
        
        return type(pre_cities)(**{
            f.name: process_field(
                getattr(pre_cities, f.name),
                getattr(initial_cache_cities, f.name),
                f.name
            )
            for f in fields(pre_cities)
        }) 

    def absorb_city(cities, from_player, from_idx, to_player, to_idx):
        """Move city data from one player/slot to another."""
        def move_field(field_val, field_name, initial_val):
            if field_name == 'additional_yield_map':
                return field_val  # Don't move this
            
            if is_dataclass(field_val):
                # Recursively handle nested dataclass
                # Pass the corresponding nested initial values
                return type(field_val)(**{
                    nested_f.name: move_field(
                        getattr(field_val, nested_f.name),
                        nested_f.name,
                        getattr(initial_val, nested_f.name)  # Pass nested initial values
                    )
                    for nested_f in fields(field_val)
                })
            
            # Copy data from dead city to killer's slot
            field_val = field_val.at[to_player, to_idx].set(
                field_val[from_player, from_idx]
            )
            # Clear the dead city's slot using initial values
            field_val = field_val.at[from_player, from_idx].set(
                initial_val[from_player, from_idx]
            )
            return field_val
        
        return type(cities)(**{
            f.name: move_field(
                getattr(cities, f.name),
                f.name,
                getattr(initial_cache_cities, f.name)  # Pass top-level initial values
            )
            for f in fields(cities)
        })

    # Branch 1: Capital was sacked - log it but don't delete
    def handle_capital_sacked():
        new_has_sacked = game.has_sacked.at[player_id[0], dead_player_id].set(True)
        new_hp = game.player_cities.hp.at[dead_player_id, dead_city_idx].set(DEAD_CITY_HEAL_LEVEL)
        return game.replace(
            has_sacked=new_has_sacked,
            player_cities=game.player_cities.replace(
                hp=new_hp
            )
        )

    # Branch 2: Non-capital city was killed
    def handle_non_capital():
        # Sub-branch 2a: Absorb the city if slots available
        def absorb():
            absorbed_cities = absorb_city(
                pre_cities, 
                dead_player_id, 
                dead_city_idx,
                player_id[0], 
                available_slot
            )
            # Still need to clear additional yields for dead city's tiles
            tiles_owned_by_dead_cities = ((pre_cities.ownership_map > 1) * dead_city_mask[..., None, None]).sum(1)[..., None]
            mixed_additional_yields = jnp.where(tiles_owned_by_dead_cities, initial_cache_cities.additional_yield_map, pre_cities.additional_yield_map)
            absorbed_cities = absorbed_cities.replace(additional_yield_map=mixed_additional_yields)
            absorbed_cities = absorbed_cities.replace(
                hp=absorbed_cities.hp.at[player_id[0], available_slot].set(DEAD_CITY_HEAL_LEVEL)
            )
            return game.replace(player_cities=absorbed_cities)
        
        # Sub-branch 2b: Delete the city if no slots
        def delete():
            deleted_cities = replace_dead_cities(pre_cities, initial_cache_cities, dead_city_mask)
            tiles_owned_by_dead_cities = ((pre_cities.ownership_map > 1) * dead_city_mask[..., None, None]).sum(1)[..., None]
            mixed_additional_yields = jnp.where(tiles_owned_by_dead_cities, initial_cache_cities.additional_yield_map, pre_cities.additional_yield_map)
            deleted_cities = deleted_cities.replace(additional_yield_map=mixed_additional_yields)
            return game.replace(player_cities=deleted_cities)
        
        return jax.lax.cond(
            are_slots_available,
            absorb,
            delete
        )

    def do_nothing():
        return game

    # Main branch: If dead city => is it a capital?, else => identity
    return jax.lax.cond(
        dead_city_mask.sum() > 0,
        lambda: jax.lax.cond(
            is_cap_bool,
            handle_capital_sacked,
            handle_non_capital
        ),
        do_nothing
    )
