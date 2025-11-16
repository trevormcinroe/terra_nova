from flax import struct
from dataclasses import fields
from functools import partial
import jax.numpy as jnp
import jax
import enum
import sys

from game.buildings import GameBuildings, BLDG_IS_NAT_WONDER, BLDG_IS_WORLD_WONDER 
from game.constants import ANCIENT_ERA_IDX, CLASSICAL_ERA_IDX, DESERT_IDX, FOOD_IDX, FOREST_IDX, GOLD_IDX, HOLY_CITY_PRESSURE_ACCEL, JUNGLE_IDX, MEDIEVAL_ERA_IDX, OASIS_IDX, PASS_THRUS, PROD_IDX, RELIGIOUS_PRESSURE_THRESHOLD, RELIGIOUS_PRESSURE_BASES, RELIGIOUS_PRESSURE_DIST_MAX, TAKE_WITH_PLAYER_ID, TAKE_WITH_PLAYER_ID_AND_CITY_INT, TO_ZERO_OUT_FOR_RELIGION_STEP, TO_ZERO_OUT_FOR_RELIGION_STEP_ONLY_MAPS, TO_ZERO_OUT_FOR_RELIGION_STEP_SANS_MAPS, TUNDRA_IDX, make_update_fn_religion
from game.improvements import Improvements
from game.resources import RESOURCE_TO_IDX


class ReligiousTenets(enum.IntEnum):
    """
    name: id, category
    pantheon: first tenet chosen
    founder: chosen when religion founded
    """
    altars_of_worship = 0, "pantheon"
    ancestor_worship = 1, "pantheon"
    dance_of_the_aurora = 2, "pantheon"
    desert_folklore = 3, "pantheon"
    divine_judgement = 4, "pantheon"
    earth_mother = 5, "pantheon"
    god_of_craftsmen = 6, "pantheon"
    god_of_the_open_sky = 7, "pantheon"
    god_of_the_sea = 8, "pantheon"
    god_of_war = 9, "pantheon"
    god_king = 10, "pantheon"
    goddess_of_festivals = 11, "pantheon"
    goddess_of_love = 12, "pantheon"
    goddess_of_protection = 13, "pantheon"
    goddess_of_the_fields = 14, "pantheon"
    goddess_of_the_hunt = 15, "pantheon"
    harvest_festival = 16, "pantheon"
    messenger_of_the_gods = 17, "pantheon"
    mystic_rituals = 18, "pantheon"
    oceans_bounty = 19, "pantheon"
    one_with_nature = 20, "pantheon"
    oral_tradition = 21, "pantheon"
    rain_dancing = 22, "pantheon"
    religious_idols = 23, "pantheon"
    religious_settlements = 24, "pantheon"
    rite_of_spring = 25, "pantheon"
    ritual_sacrifice = 26, "pantheon"
    sacred_path = 27, "pantheon"
    seafood_rituals = 28, "pantheon"
    spirit_animals = 29, "pantheon"
    spirit_trees = 30, "pantheon"
    starlight_guidance = 31, "pantheon"
    stone_circles = 32, "pantheon"
    sun_god = 33, "pantheon"
    tears_of_the_gods = 34, "pantheon"
    vision_quests = 35, "pantheon"
    works_spirituals = 36, "pantheon"

    ceremonial_burial = 37, "founder"
    church_property = 38, "founder"
    dawah = 39, "founder"
    initiation_rites = 40, "founder"
    messiah = 41, "founder"
    missionary_zeal = 42, "founder"
    mithraea = 43, "founder"
    religious_unity = 44, "founder"
    salat = 45, "founder"
    tithe = 46, "founder"
    world_church = 47, "founder"
    zakat = 48, "founder"

    cathedrals = 49, "follower"
    choral_music = 50, "follower"
    devoted_elite = 51, "follower"
    devout_performers = 52, "follower"
    divine_inspiration = 53, "follower"
    feed_the_world = 54, "follower"
    followers_of_the_refined_crafts = 55, "follower"
    gurdwaras = 56, "follower"
    guruship = 57, "follower"
    holy_warriors = 58, "follower"
    liturgical_drama = 59, "follower"
    mandirs = 60, "follower"
    mosques = 61, "follower"
    pagodas = 62, "follower"
    peace_gardens = 63, "follower"
    religious_art = 64, "follower"
    religious_center = 64, "follower"
    religious_community = 65, "follower"
    sacred_waters = 66, "follower"
    synagogues = 67, "follower"
    viharas = 68, "follower"

    defender_of_the_faith = 69, "enhancer"
    dharma = 70, "enhancer"
    disciples = 71, "enhancer"
    hajj = 72, "enhancer"
    jizya = 73, "enhancer"
    just_war = 74, "enhancer"
    karma = 75, "enhancer"
    kotel = 76, "enhancer"
    pilgrimage = 77, "enhancer"
    promised_land = 78, "enhancer"
    religious_troubadours = 79, "enhancer"
    sanctified_innovations = 80, "enhancer"
    unity_of_the_prophets = 81, "enhancer"

    apostolic_palace = 82, "reformation"
    city_of_god = 83, "reformation"
    houses_of_worship = 84, "reformation"
    indulgences = 85, "reformation"
    jesuit_education = 86, "reformation"
    sacred_sites = 87, "reformation"
    swords_into_plowshares = 88, "reformation"
    underground_sect = 89, "reformation"
    work_ethic = 90, "reformation"
    
    def __new__(cls, value: int, category: str):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.category = category
        return obj


MAX_IDX_PANTHEON = jnp.array([p.category == "pantheon" for p in ReligiousTenets]).sum()
MAX_IDX_FOUNDER = jnp.array([p.category == "founder" for p in ReligiousTenets]).sum() + MAX_IDX_PANTHEON
MAX_IDX_FOLLOWER = jnp.array([p.category == "follower" for p in ReligiousTenets]).sum() + MAX_IDX_FOUNDER
MAX_IDX_ENHANCER = jnp.array([p.category == "enhancer" for p in ReligiousTenets]).sum() + MAX_IDX_FOLLOWER
MAX_IDX_REFORMATION = jnp.array([p.category == "reformation" for p in ReligiousTenets]).sum() + MAX_IDX_ENHANCER


"""
The below are "on-selection" effects from religious tenets.
Due to the way religious bonuses are applied, each of these functions should operate from a per-city perspective!
"""

def add_religious_tenet(religious_tenets, player_id, tenet_idx):
    """
    This function should only be called from within GameState.step_religion()

    Religious tenets are managed on a per-player basis, so this function should never be called from 
    within one of the below functions. Instead, use add_religious_tenet_to_city.
    """
    return religious_tenets.at[player_id[0], tenet_idx].set(1)

def add_religious_tenet_deprecated(game, player_id, tenet_idx):
    """
    Religious tenets are managed on a per-player basis, so this function should never be called from 
    within one of the below functions. Instead, use add_religious_tenet_to_city.

    This function should only be called from within GameState.step_religion()
    """
    religious_tenets = game.religious_tenets.at[
        player_id[0], tenet_idx
    ].set(1)
    return game.replace(religious_tenets=religious_tenets)

def add_religious_tenet_to_city(religion_info, tenet):
    """"""
    religious_tenets_per_city = religion_info.religious_tenets_per_city.at[
        ReligiousTenets[tenet]._value_
    ].set(1)
    return religion_info.replace(religious_tenets_per_city=religious_tenets_per_city)


def add_bldg_yields_for_existing_bldg(religion_info, bldgs_owned, to_add, bldg_idx):
    """Called from a vmapped over cities context"""
    b_already_owned = bldgs_owned[bldg_idx] == 1
    new_bldg_yields = religion_info.building_yields + to_add * b_already_owned
    return religion_info.replace(building_yields=new_bldg_yields)

def do_nothing(religion_info, *args):
    return religion_info

@partial(jax.jit, static_argnames=("bool_map_generator"))
def add_yield_tiles_to_given_city(religion_info, game, to_add, bool_map_generator):
    """
    bool_map_generator should return:
        (1) bool_map: 1 if condition met for adding yields -- related to game_map_rowcols
        (2) game_map_rowcols: rowcols of owned tiles by the city
        (3) bool_city_center: 1 if condition met for adding yeilds -- related to game_map_rowcols_city_center
        (4) game_map_rowcols_city_center: rowcols of the city center
    """
    bool_map, game_map_rowcols, bool_city_center, game_map_rowcols_city_center = bool_map_generator(game)
    _additional_yield_map = religion_info.additional_yield_map

    # (36,) => (36,7)
    city_ring_to_set = bool_map[:, None] * to_add[None]
    city_center_to_set = bool_city_center * to_add

    _additional_yield_map = _additional_yield_map.at[game_map_rowcols[:, 0], game_map_rowcols[:, 1]].add(city_ring_to_set)
    _additional_yield_map = _additional_yield_map.at[
        game_map_rowcols_city_center[0], game_map_rowcols_city_center[1]
    ].add(city_center_to_set)

    return religion_info.replace(additional_yield_map=_additional_yield_map)



"""
To avoid memory blowup issues, each function will form the following map (religion_info, game, player_id) -> (religion_info)
    If we return the game object, then XLA avoid a broadcast, which would be disasterous memory-wise.

All of the information in religion_info is on a per-city perspective!
"""
def _altars_of_worship(religion_info, game, player_id):
    """
    +20% prod towards ancient, classical, medieval wonders
    """
    religion_info = add_religious_tenet_to_city(religion_info, "altars_of_worship")
    wonder_accel = religion_info.wonder_accel.at[ANCIENT_ERA_IDX].add(0.2).at[CLASSICAL_ERA_IDX].add(0.2).at[MEDIEVAL_ERA_IDX].add(0.2)
    return religion_info.replace(wonder_accel=wonder_accel)

def _ancestor_worship(religion_info, game, player_id):
    """
    +2 culture from shrines
    """
    religion_info = add_religious_tenet_to_city(religion_info, "ancestor_worship")
    shrine_idx = GameBuildings["shrine"]._value_
    to_add = jnp.array([0, 0, 0, 0, 2, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        shrine_idx
    )
    return religion_info

def _dance_of_the_aurora(religion_info, game, player_id):
    """
    +1 faith from tundra tiles
    """
    religion_info = add_religious_tenet_to_city(religion_info, "dance_of_the_aurora")
    
    def bool_map_generator(game):
        this_city_rowcols = game.player_cities__potential_owned_rowcols
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities__ownership_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        tundra = game.terrain_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] == TUNDRA_IDX

        this_city_center = game.player_cities__city_rowcols
        center_tundra = game.terrain_map[this_city_center[0], this_city_center[1]] == TUNDRA_IDX
        return this_city_currently_owned * tundra, game_map_rowcols, center_tundra, this_city_center

    to_add = jnp.array([0, 0, 0, 1, 0, 0, 0])
    religion_info = add_yield_tiles_to_given_city(religion_info, game, to_add, bool_map_generator)
    return religion_info

def _desert_folklore(religion_info, game, player_id):
    """
    +1 faith from desert tiles
    """
    religion_info = add_religious_tenet_to_city(religion_info, "desert_folklore")
    
    def bool_map_generator(game):
        this_city_rowcols = game.player_cities__potential_owned_rowcols
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities__ownership_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2
        
        tundra = game.terrain_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] == DESERT_IDX

        this_city_center = game.player_cities__city_rowcols
        center_tundra = game.terrain_map[this_city_center[0], this_city_center[1]] == DESERT_IDX
        return this_city_currently_owned * tundra, game_map_rowcols, center_tundra, this_city_center

    to_add = jnp.array([0, 0, 0, 1, 0, 0, 0])
    religion_info = add_yield_tiles_to_given_city(religion_info, game, to_add, bool_map_generator)
    return religion_info

def _divine_judgement(religion_info, game, player_id):
    """
    +2 faith from barracks and courthouses
    +2 happiness form the palace
    """
    religion_info = add_religious_tenet_to_city(religion_info, "divine_judgement")
    barracks_idx = GameBuildings["barracks"]._value_
    to_add = jnp.array([0, 0, 0, 0, 2, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        barracks_idx
    )

    b_idx = GameBuildings["courthouse"]._value_
    to_add = jnp.array([0, 0, 0, 0, 2, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )
    
    b_idx = GameBuildings["palace"]._value_
    to_add = jnp.array([0, 0, 0, 0, 0, 0, 2, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )

    return religion_info

def _earth_mother(religion_info, game, player_id):
    """
    +1 faith from mines
    +2 faith from iron
    """
    religion_info = add_religious_tenet_to_city(religion_info, "earth_mother")

    def bool_map_generator(game):
        this_city_rowcols = game.player_cities__potential_owned_rowcols
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities__ownership_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2

        iron = RESOURCE_TO_IDX["iron"]
        
        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        res = this_city_resources == iron

        this_city_center = game.player_cities__city_rowcols
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        center_res = this_city_center_resources == iron
        
        can_see_resource_bool = game.visible_resources_map_players[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] > 0
        can_see_resource_center_bool = game.visible_resources_map_players[
            this_city_center[0], this_city_center[1]
        ] > 0

        return this_city_currently_owned * res * can_see_resource_bool, game_map_rowcols, center_res  * can_see_resource_center_bool, this_city_center

    to_add = jnp.array([0, 0, 0, 2, 0, 0, 0])
    religion_info = add_yield_tiles_to_given_city(religion_info, game, to_add, bool_map_generator)
    return religion_info

def _god_of_craftsmen(religion_info, game, player_id):
    """
    +1 production in every city
    """
    religion_info = add_religious_tenet_to_city(religion_info, "god_of_craftsmen")
    new_yields = religion_info.building_yields + jnp.array([0, 1, 0, 0, 0, 0, 0, 0])
    return religion_info.replace(building_yields=new_yields)


def _god_of_the_open_sky(religion_info, game, player_id):
    """
    +1 culture from pastures
    """
    religion_info = add_religious_tenet_to_city(religion_info, "god_of_the_open_sky")
    
    def bool_map_generator(game):
        this_city_rowcols = game.player_cities__potential_owned_rowcols
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities__ownership_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2

        pastures = game.improvement_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] == (Improvements["pasture"]._value_ + 1)

        citywide_bool = this_city_currently_owned & pastures

        this_city_center = game.player_cities__city_rowcols
        this_city_center_resources = game.improvement_map[this_city_center[0], this_city_center[1]]
        center_res = this_city_center_resources == (Improvements["pasture"]._value_ + 1)
        

        return citywide_bool, game_map_rowcols, center_res, this_city_center

    to_add = jnp.array([0, 0, 0, 0, 1, 0, 0])
    religion_info = add_yield_tiles_to_given_city(religion_info, game, to_add, bool_map_generator)
    return religion_info

def _god_of_the_sea(religion_info, game, player_id):
    """
    +1 faith, +1 culture from fish, whales, crabs, pearls, coral, atolls
    """
    religion_info = add_religious_tenet_to_city(religion_info, "god_of_the_sea")

    def bool_map_generator(game):
        this_city_rowcols = game.player_cities__potential_owned_rowcols
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities__ownership_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2

        fish = RESOURCE_TO_IDX["fish"]
        whales = RESOURCE_TO_IDX["whales"]
        crabs = RESOURCE_TO_IDX["crabs"]
        pearls = RESOURCE_TO_IDX["pearls"]
        coral = RESOURCE_TO_IDX["coral"]
        
        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        res = (this_city_resources == fish) | (this_city_resources == whales) | (this_city_resources == crabs) |(this_city_resources == pearls) | (this_city_resources == coral) 

        this_city_center = game.player_cities__city_rowcols
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        center_res = (this_city_center_resources == fish) | (this_city_center_resources == whales) | (this_city_center_resources == crabs) | (this_city_center_resources == pearls) | (this_city_center_resources == coral)
        
        can_see_resource_bool = game.visible_resources_map_players[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] > 0
        can_see_resource_center_bool = game.visible_resources_map_players[
            this_city_center[0], this_city_center[1]
        ] > 0

        return this_city_currently_owned * res * can_see_resource_bool, game_map_rowcols, center_res  * can_see_resource_center_bool, this_city_center

    to_add = jnp.array([0, 0, 0, 1, 1, 0, 0])
    religion_info = add_yield_tiles_to_given_city(religion_info, game, to_add, bool_map_generator)

    return religion_info

def _god_of_war(religion_info, game, player_id):
    """
    +2 prod from barracks
    """
    religion_info = add_religious_tenet_to_city(religion_info, "god_of_war")
    b_idx = GameBuildings["barracks"]._value_
    to_add = jnp.array([0, 2, 0, 0, 0, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )
    return religion_info

def _god_king(religion_info, game, player_id):
    """
    +1 prod, food, gold, culture, science, happiness, faith from palace
    """
    religion_info = add_religious_tenet_to_city(religion_info, "god_king")
    b_idx = GameBuildings["palace"]._value_
    to_add = jnp.array([1, 1, 1, 1, 1, 1, 1, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )
    return religion_info

def _goddess_of_festivals(religion_info, game, player_id):
    """
    +1 prod from sugar, wine, spices
    """
    religion_info = add_religious_tenet_to_city(religion_info, "goddess_of_festivals")

    def bool_map_generator(game):
        this_city_rowcols = game.player_cities__potential_owned_rowcols
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities__ownership_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2

        sugar = RESOURCE_TO_IDX["sugar"]
        wine = RESOURCE_TO_IDX["wine"]
        spices = RESOURCE_TO_IDX["spices"]
        
        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        res = (this_city_resources == sugar) | (this_city_resources == wine) | (this_city_resources == spices)

        this_city_center = game.player_cities__city_rowcols
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        center_res = (this_city_center_resources == sugar) | (this_city_center_resources == wine) | (this_city_center_resources == spices)
        
        can_see_resource_bool = game.visible_resources_map_players[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] > 0
        can_see_resource_center_bool = game.visible_resources_map_players[
            this_city_center[0], this_city_center[1]
        ] > 0

        return this_city_currently_owned * res * can_see_resource_bool, game_map_rowcols, center_res  * can_see_resource_center_bool, this_city_center

    to_add = jnp.array([0, 0, 0, 1, 1, 0, 0])
    religion_info = add_yield_tiles_to_given_city(religion_info, game, to_add, bool_map_generator)
    return religion_info

def _goddess_of_love(religion_info, game, player_id):
    """
    +1 happiness in cities with 3+ pop
    """
    religion_info = add_religious_tenet_to_city(religion_info, "goddess_of_love")
    to_add = jnp.array([0, 0, 0, 0, 0, 0, 1, 0])
    to_add_bool = game.player_cities__population >= 3
    new_yields = religion_info.building_yields + (to_add * to_add_bool)
    return religion_info.replace(building_yields=new_yields)

def _goddess_of_protection(religion_info, game, player_id):
    """
    +1 faith, +1 culture from walls
    +50% city bombardment str
    """
    religion_info = add_religious_tenet_to_city(religion_info, "goddess_of_protection")
    b_idx = GameBuildings["walls"]._value_
    to_add = jnp.array([0, 0, 0, 1, 1, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )
    new_accel = religion_info.city_ranged_strength_accel + 0.5
    return religion_info.replace(city_ranged_strength_accel=new_accel)

def _goddess_of_the_fields(religion_info, game, player_id):
    """
    +2 faith from cotton, incense, tea, coffee
    """
    religion_info = add_religious_tenet_to_city(religion_info, "goddess_of_the_fields")
    
    def bool_map_generator(game):
        this_city_rowcols = game.player_cities__potential_owned_rowcols
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities__ownership_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2

        fish = RESOURCE_TO_IDX["cotton"]
        whales = RESOURCE_TO_IDX["incense"]
        crabs = RESOURCE_TO_IDX["tea"]
        pearls = RESOURCE_TO_IDX["coffee"]
        
        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        res = (this_city_resources == fish) | (this_city_resources == whales) | (this_city_resources == crabs) |(this_city_resources == pearls) 

        this_city_center = game.player_cities__city_rowcols
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        center_res = (this_city_center_resources == fish) | (this_city_center_resources == whales) | (this_city_center_resources == crabs) | (this_city_center_resources == pearls)
        
        can_see_resource_bool = game.visible_resources_map_players[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] > 0
        can_see_resource_center_bool = game.visible_resources_map_players[
            this_city_center[0], this_city_center[1]
        ] > 0

        return this_city_currently_owned * res * can_see_resource_bool, game_map_rowcols, center_res  * can_see_resource_center_bool, this_city_center

    to_add = jnp.array([0, 0, 0, 2, 0, 0, 0])
    religion_info = add_yield_tiles_to_given_city(religion_info, game, to_add, bool_map_generator)
    return religion_info


def _goddess_of_the_hunt(religion_info, game, player_id):
    """
    +1 food from camps
    """
    religion_info = add_religious_tenet_to_city(religion_info, "goddess_of_the_hunt")
    
    def bool_map_generator(game):
        this_city_rowcols = game.player_cities__potential_owned_rowcols
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities__ownership_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2

        pastures = game.improvement_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] == (Improvements["camp"]._value_ + 1)

        citywide_bool = this_city_currently_owned & pastures

        this_city_center = game.player_cities__city_rowcols
        this_city_center_resources = game.improvement_map[this_city_center[0], this_city_center[1]]
        center_res = this_city_center_resources == (Improvements["camp"]._value_ + 1)

        return citywide_bool, game_map_rowcols, center_res, this_city_center

    to_add = jnp.array([1, 0, 0, 0, 0, 0, 0])
    religion_info = add_yield_tiles_to_given_city(religion_info, game, to_add, bool_map_generator)
    return religion_info

def _harvest_festival(religion_info, game, player_id):
    """
    +1 food from wheat, maize
    """
    religion_info = add_religious_tenet_to_city(religion_info, "harvest_festival")

    def bool_map_generator(game):
        this_city_rowcols = game.player_cities__potential_owned_rowcols
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities__ownership_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2

        fish = RESOURCE_TO_IDX["wheat"]
        whales = RESOURCE_TO_IDX["maize"]
        
        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        res = (this_city_resources == fish) | (this_city_resources == whales) 

        this_city_center = game.player_cities__city_rowcols
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        center_res = (this_city_center_resources == fish) | (this_city_center_resources == whales)
        
        can_see_resource_bool = game.visible_resources_map_players[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] > 0
        can_see_resource_center_bool = game.visible_resources_map_players[
            this_city_center[0], this_city_center[1]
        ] > 0

        return this_city_currently_owned * res * can_see_resource_bool, game_map_rowcols, center_res  * can_see_resource_center_bool, this_city_center

    to_add = jnp.array([1, 0, 0, 0, 0, 0, 0])
    religion_info = add_yield_tiles_to_given_city(religion_info, game, to_add, bool_map_generator)
    return religion_info

def _messenger_of_the_gods(religion_info, game, player_id):
    """
    +2 science, +1 faith in cities with a city connection
    """
    religion_info = add_religious_tenet_to_city(religion_info, "messenger_of_the_gods")
    to_add = jnp.array([0, 0, 0, 1, 0, 2, 0, 0]) * game.is_connected_to_cap
    new_bldg_yields = religion_info.building_yields + to_add
    return religion_info.replace(building_yields=new_bldg_yields)

def _mystic_rituals(religion_info, game, player_id):
    """
    +1 food, +1 culture from silk, dyes, perfume, tobacco
    """
    religion_info = add_religious_tenet_to_city(religion_info, "mystic_rituals")

    def bool_map_generator(game):
        this_city_rowcols = game.player_cities__potential_owned_rowcols
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities__ownership_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2

        fish = RESOURCE_TO_IDX["silk"]
        whales = RESOURCE_TO_IDX["dyes"]
        crabs = RESOURCE_TO_IDX["perfume"]
        pearls = RESOURCE_TO_IDX["tobacco"]
        
        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        res = (this_city_resources == fish) | (this_city_resources == whales) | (this_city_resources == crabs) |(this_city_resources == pearls) 

        this_city_center = game.player_cities__city_rowcols
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        center_res = (this_city_center_resources == fish) | (this_city_center_resources == whales) | (this_city_center_resources == crabs) | (this_city_center_resources == pearls)
        
        can_see_resource_bool = game.visible_resources_map_players[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] > 0
        can_see_resource_center_bool = game.visible_resources_map_players[
            this_city_center[0], this_city_center[1]
        ] > 0

        return this_city_currently_owned * res * can_see_resource_bool, game_map_rowcols, center_res  * can_see_resource_center_bool, this_city_center

    to_add = jnp.array([1, 0, 0, 0, 1, 0, 0])
    religion_info = add_yield_tiles_to_given_city(religion_info, game, to_add, bool_map_generator)
    return religion_info

def _oceans_bounty(religion_info, game, player_id):
    """
    +1 prod from fishing boats and atolls
    """
    religion_info = add_religious_tenet_to_city(religion_info, "oceans_bounty")

    def bool_map_generator(game):
        this_city_rowcols = game.player_cities__potential_owned_rowcols
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities__ownership_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2

        pastures = game.improvement_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] == (Improvements["fishing_boat"]._value_ + 1)

        citywide_bool = this_city_currently_owned & pastures

        this_city_center = game.player_cities__city_rowcols
        this_city_center_resources = game.improvement_map[this_city_center[0], this_city_center[1]]
        center_res = this_city_center_resources == (Improvements["fishing_boat"]._value_ + 1)
        
        return citywide_bool, game_map_rowcols, center_res, this_city_center

    to_add = jnp.array([0, 1, 0, 0, 0, 0, 0])
    religion_info = add_yield_tiles_to_given_city(religion_info, game, to_add, bool_map_generator)

    return religion_info

def _one_with_nature(religion_info, game, player_id):
    """
    +5 faith from nat wonders
    """
    religion_info = add_religious_tenet_to_city(religion_info, "one_with_nature")

    def bool_map_generator(game):
        this_city_rowcols = game.player_cities__potential_owned_rowcols
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities__ownership_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2

        this_city_nw = game.nw_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] > 0
        this_city_center = game.player_cities__city_rowcols
        this_city_center_nw = game.nw_map[this_city_center[0], this_city_center[1]] > 0

        return this_city_currently_owned * this_city_nw, game_map_rowcols, this_city_center_nw, this_city_center

    to_add = jnp.array([0, 0, 0, 5, 0, 0, 0])
    religion_info = add_yield_tiles_to_given_city(religion_info, game, to_add, bool_map_generator)
    return religion_info

def _oral_tradition(religion_info, game, player_id):
    """
    +1 culture from plantation improvements
    """
    religion_info = add_religious_tenet_to_city(religion_info, "oral_tradition")

    def bool_map_generator(game):
        this_city_rowcols = game.player_cities__potential_owned_rowcols
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities__ownership_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2

        pastures = game.improvement_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] == (Improvements["plantation"]._value_ + 1)

        citywide_bool = this_city_currently_owned & pastures

        this_city_center = game.player_cities__city_rowcols
        this_city_center_resources = game.improvement_map[this_city_center[0], this_city_center[1]]
        center_res = this_city_center_resources == (Improvements["plantation"]._value_ + 1)
        
        return citywide_bool, game_map_rowcols, center_res, this_city_center

    to_add = jnp.array([0, 0, 0, 0, 1, 0, 0])
    religion_info = add_yield_tiles_to_given_city(religion_info, game, to_add, bool_map_generator)
    return religion_info

def _rain_dancing(religion_info, game, player_id):
    """
    +2 faith, +1 culture from lakes and oases
    """
    religion_info = add_religious_tenet_to_city(religion_info, "rain_dancing")
    
    def bool_map_generator(game):
        this_city_rowcols = game.player_cities__potential_owned_rowcols
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities__ownership_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2

        lakes = game.lake_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] == 1
        oasis = game.feature_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] == OASIS_IDX
        this_city_bool = lakes | oasis

        this_city_center = game.player_cities__city_rowcols
        lakes = game.lake_map[this_city_center[0], this_city_center[1]] == 1
        oasis = game.feature_map[this_city_center[0], this_city_center[1]] == OASIS_IDX
        this_city_center_bool = lakes | oasis

        return this_city_currently_owned * this_city_bool, game_map_rowcols, this_city_center_bool, this_city_center

    to_add = jnp.array([0, 0, 0, 2, 1, 0, 0])
    religion_info = add_yield_tiles_to_given_city(religion_info, game, to_add, bool_map_generator)
    return religion_info

def _religious_idols(religion_info, game, player_id):
    """
    +1 faith, +1 culture from copper, silver, gold
    """
    religion_info = add_religious_tenet_to_city(religion_info, "religious_idols")
    
    def bool_map_generator(game):
        this_city_rowcols = game.player_cities__potential_owned_rowcols
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities__ownership_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2

        fish = RESOURCE_TO_IDX["copper"]
        whales = RESOURCE_TO_IDX["silver"]
        crabs = RESOURCE_TO_IDX["gold"]
        
        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        res = (this_city_resources == fish) | (this_city_resources == whales) | (this_city_resources == crabs)

        this_city_center = game.player_cities__city_rowcols
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        center_res = (this_city_center_resources == fish) | (this_city_center_resources == whales) | (this_city_center_resources == crabs)
        
        can_see_resource_bool = game.visible_resources_map_players[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] > 0
        can_see_resource_center_bool = game.visible_resources_map_players[
            this_city_center[0], this_city_center[1]
        ] > 0

        return this_city_currently_owned * res * can_see_resource_bool, game_map_rowcols, center_res  * can_see_resource_center_bool, this_city_center

    to_add = jnp.array([0, 0, 0, 1, 1, 0, 0])
    religion_info = add_yield_tiles_to_given_city(religion_info, game, to_add, bool_map_generator)

    return religion_info

def _religious_settlements(religion_info, game, player_id):
    """
    +1 food from cities
    +25% border expansion rate
    """
    religion_info = add_religious_tenet_to_city(religion_info, "religious_settlements")
    new_yields = religion_info.building_yields + jnp.array([1, 0, 0, 0, 0, 0, 0, 0])
    new_accel = religion_info.border_growth_accel + 0.25
    return religion_info.replace(building_yields=new_yields, border_growth_accel=new_accel)

def _rite_of_spring(religion_info, game, player_id):
    """
    +1 culture from deer, bison
    """
    religion_info = add_religious_tenet_to_city(religion_info, "rite_of_spring")

    def bool_map_generator(game):
        this_city_rowcols = game.player_cities__potential_owned_rowcols
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities__ownership_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2

        fish = RESOURCE_TO_IDX["deer"]
        whales = RESOURCE_TO_IDX["bison"]
        
        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        res = (this_city_resources == fish) | (this_city_resources == whales) 

        this_city_center = game.player_cities__city_rowcols
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        center_res = (this_city_center_resources == fish) | (this_city_center_resources == whales)
        
        can_see_resource_bool = game.visible_resources_map_players[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] > 0
        can_see_resource_center_bool = game.visible_resources_map_players[
            this_city_center[0], this_city_center[1]
        ] > 0

        return this_city_currently_owned * res * can_see_resource_bool, game_map_rowcols, center_res  * can_see_resource_center_bool, this_city_center

    to_add = jnp.array([0, 0, 0, 0, 1, 0, 0])
    religion_info = add_yield_tiles_to_given_city(religion_info, game, to_add, bool_map_generator)
    return religion_info

def _ritual_sacrifice(religion_info, game, player_id):
    """
    +1 culture, +2 gold from ivory, furs, truffles
    """
    religion_info = add_religious_tenet_to_city(religion_info, "ritual_sacrifice")

    def bool_map_generator(game):
        this_city_rowcols = game.player_cities__potential_owned_rowcols
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities__ownership_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2

        fish = RESOURCE_TO_IDX["ivory"]
        whales = RESOURCE_TO_IDX["furs"]
        crabs = RESOURCE_TO_IDX["truffles"]
        
        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        res = (this_city_resources == fish) | (this_city_resources == whales) | (this_city_resources == crabs) 

        this_city_center = game.player_cities__city_rowcols
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        center_res = (this_city_center_resources == fish) | (this_city_center_resources == whales) | (this_city_center_resources == crabs)
        
        can_see_resource_bool = game.visible_resources_map_players[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] > 0
        can_see_resource_center_bool = game.visible_resources_map_players[
            this_city_center[0], this_city_center[1]
        ] > 0

        return this_city_currently_owned * res * can_see_resource_bool, game_map_rowcols, center_res  * can_see_resource_center_bool, this_city_center

    to_add = jnp.array([0, 0, 2, 0, 1, 0, 0])
    religion_info = add_yield_tiles_to_given_city(religion_info, game, to_add, bool_map_generator)
    return religion_info

def _sacred_path(religion_info, game, player_id):
    """
    +1 culture from jungles, forest
    +1 culture from rubber, hardwood
    """
    religion_info = add_religious_tenet_to_city(religion_info, "sacred_path")

    def bool_map_generator(game):
        this_city_rowcols = game.player_cities__potential_owned_rowcols
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities__ownership_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2

        fish = RESOURCE_TO_IDX["rubber"]
        whales = RESOURCE_TO_IDX["hardwood"]
        
        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        res = (this_city_resources == fish) | (this_city_resources == whales) 

        this_city_center = game.player_cities__city_rowcols
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        center_res = (this_city_center_resources == fish) | (this_city_center_resources == whales)
        
        can_see_resource_bool = game.visible_resources_map_players[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] > 0
        can_see_resource_center_bool = game.visible_resources_map_players[
            this_city_center[0], this_city_center[1]
        ] > 0

        return this_city_currently_owned * res * can_see_resource_bool, game_map_rowcols, center_res  * can_see_resource_center_bool, this_city_center

    to_add = jnp.array([0, 0, 0, 0, 1, 0, 0])
    religion_info = add_yield_tiles_to_given_city(religion_info, game, to_add, bool_map_generator)

    def bool_map_generator(game):
        this_city_rowcols = game.player_cities__potential_owned_rowcols
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities__ownership_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2

        jungle = game.feature_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] == JUNGLE_IDX 
        forest = game.feature_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] == FOREST_IDX
        this_city_bool = jungle | forest

        this_city_center = game.player_cities__city_rowcols
        jungle = game.feature_map[this_city_center[0], this_city_center[1]] == JUNGLE_IDX 
        forest = game.feature_map[this_city_center[0], this_city_center[1]] == FOREST_IDX
        city_center_bool = jungle | forest
        
        return this_city_currently_owned * this_city_bool, game_map_rowcols, city_center_bool, this_city_center

    to_add = jnp.array([0, 0, 0, 0, 1, 0, 0])
    religion_info = add_yield_tiles_to_given_city(religion_info, game, to_add, bool_map_generator)

    return religion_info

def _seafood_rituals(religion_info, game, player_id):
    """
    +1 food, +1 culture, +1 faith from fish
    """
    religion_info = add_religious_tenet_to_city(religion_info, "seafood_rituals")

    def bool_map_generator(game):
        this_city_rowcols = game.player_cities__potential_owned_rowcols
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities__ownership_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2

        fish = RESOURCE_TO_IDX["fish"]
        
        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        res = (this_city_resources == fish)

        this_city_center = game.player_cities__city_rowcols
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        center_res = (this_city_center_resources == fish)
        
        can_see_resource_bool = game.visible_resources_map_players[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] > 0
        can_see_resource_center_bool = game.visible_resources_map_players[
            this_city_center[0], this_city_center[1]
        ] > 0

        return this_city_currently_owned * res * can_see_resource_bool, game_map_rowcols, center_res  * can_see_resource_center_bool, this_city_center

    to_add = jnp.array([1, 0, 0, 1, 1, 0, 0])
    religion_info = add_yield_tiles_to_given_city(religion_info, game, to_add, bool_map_generator)
    return religion_info

def _spirit_animals(religion_info, game, player_id):
    """
    +1 faith from horses, deer, bison
    """
    religion_info = add_religious_tenet_to_city(religion_info, "spirit_animals")

    def bool_map_generator(game):
        this_city_rowcols = game.player_cities__potential_owned_rowcols
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities__ownership_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2

        fish = RESOURCE_TO_IDX["horses"]
        whales = RESOURCE_TO_IDX["deer"]
        crabs = RESOURCE_TO_IDX["bison"]
        
        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        res = (this_city_resources == fish) | (this_city_resources == whales) | (this_city_resources == crabs) 

        this_city_center = game.player_cities__city_rowcols
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        center_res = (this_city_center_resources == fish) | (this_city_center_resources == whales) | (this_city_center_resources == crabs)
        
        can_see_resource_bool = game.visible_resources_map_players[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] > 0
        can_see_resource_center_bool = game.visible_resources_map_players[
            this_city_center[0], this_city_center[1]
        ] > 0

        return this_city_currently_owned * res * can_see_resource_bool, game_map_rowcols, center_res  * can_see_resource_center_bool, this_city_center

    to_add = jnp.array([0, 0, 0, 1, 0, 0, 0])
    religion_info = add_yield_tiles_to_given_city(religion_info, game, to_add, bool_map_generator)
    return religion_info

def _spirit_trees(religion_info, game, player_id):
    """
    +1 faith, +1 food from lumbermills
    """
    religion_info = add_religious_tenet_to_city(religion_info, "spirit_trees")

    def bool_map_generator(game):
        this_city_rowcols = game.player_cities__potential_owned_rowcols
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities__ownership_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2

        pastures = game.improvement_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] == (Improvements["lumber_mill"]._value_ + 1)

        citywide_bool = this_city_currently_owned & pastures

        this_city_center = game.player_cities__city_rowcols
        this_city_center_resources = game.improvement_map[this_city_center[0], this_city_center[1]]
        center_res = this_city_center_resources == (Improvements["lumber_mill"]._value_ + 1)
        
        return citywide_bool, game_map_rowcols, center_res, this_city_center

    to_add = jnp.array([1, 0, 0, 1, 0, 0, 0])
    religion_info = add_yield_tiles_to_given_city(religion_info, game, to_add, bool_map_generator)
    return religion_info

def _starlight_guidance(religion_info, game, player_id):
    """
    +1 culture, +1 faith, +1 happiness from lighthouses
    """
    religion_info = add_religious_tenet_to_city(religion_info, "starlight_guidance")
    b_idx = GameBuildings["lighthouse"]._value_
    to_add = jnp.array([0, 0, 0, 1, 1, 0, 1, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )
    return religion_info


def _stone_circles(religion_info, game, player_id):
    """
    +1 prod, +1 faith from quarries
    +1 faith from marble, obsidian
    """
    religion_info = add_religious_tenet_to_city(religion_info, "stone_circles")

    def bool_map_generator(game):
        this_city_rowcols = game.player_cities__potential_owned_rowcols
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities__ownership_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2

        pastures = game.improvement_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] == (Improvements["quarry"]._value_ + 1)

        citywide_bool = this_city_currently_owned & pastures

        this_city_center = game.player_cities__city_rowcols
        this_city_center_resources = game.improvement_map[this_city_center[0], this_city_center[1]]
        center_res = this_city_center_resources == (Improvements["quarry"]._value_ + 1)
        
        return citywide_bool, game_map_rowcols, center_res, this_city_center

    to_add = jnp.array([0, 1, 0, 1, 0, 0, 0])
    religion_info = add_yield_tiles_to_given_city(religion_info, game, to_add, bool_map_generator)

    def bool_map_generator(game):
        this_city_rowcols = game.player_cities__potential_owned_rowcols
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities__ownership_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2

        fish = RESOURCE_TO_IDX["marble"]
        whales = RESOURCE_TO_IDX["obsidian"]
        
        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        res = (this_city_resources == fish) | (this_city_resources == whales)

        this_city_center = game.player_cities__city_rowcols
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        center_res = (this_city_center_resources == fish) | (this_city_center_resources == whales)
        
        can_see_resource_bool = game.visible_resources_map_players[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] > 0
        can_see_resource_center_bool = game.visible_resources_map_players[
            this_city_center[0], this_city_center[1]
        ] > 0

        return this_city_currently_owned * res * can_see_resource_bool, game_map_rowcols, center_res  * can_see_resource_center_bool, this_city_center

    to_add = jnp.array([0, 0, 0, 1, 0, 0, 0])
    religion_info = add_yield_tiles_to_given_city(religion_info, game, to_add, bool_map_generator)
    return religion_info


def _sun_god(religion_info, game, player_id):
    """
    +1 food from bananas, citrus, olives, coconuts, cocoa
    """
    religion_info = add_religious_tenet_to_city(religion_info, "sun_god")

    def bool_map_generator(game):
        this_city_rowcols = game.player_cities__potential_owned_rowcols
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities__ownership_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2

        fish = RESOURCE_TO_IDX["banana"]
        whales = RESOURCE_TO_IDX["citrus"]
        crabs = RESOURCE_TO_IDX["olives"]
        pearls = RESOURCE_TO_IDX["coconut"]
        coral = RESOURCE_TO_IDX["chocolate"]
        
        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        res = (this_city_resources == fish) | (this_city_resources == whales) | (this_city_resources == crabs) |(this_city_resources == pearls) | (this_city_resources == coral) 

        this_city_center = game.player_cities__city_rowcols
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        center_res = (this_city_center_resources == fish) | (this_city_center_resources == whales) | (this_city_center_resources == crabs) | (this_city_center_resources == pearls) | (this_city_center_resources == coral)
        
        can_see_resource_bool = game.visible_resources_map_players[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] > 0
        can_see_resource_center_bool = game.visible_resources_map_players[
            this_city_center[0], this_city_center[1]
        ] > 0

        return this_city_currently_owned * res * can_see_resource_bool, game_map_rowcols, center_res  * can_see_resource_center_bool, this_city_center

    to_add = jnp.array([1, 0, 0, 0, 0, 0, 0])
    religion_info = add_yield_tiles_to_given_city(religion_info, game, to_add, bool_map_generator)
    return religion_info

def _tears_of_the_gods(religion_info, game, player_id):
    """
    +1 faith, +2 gold from gems, amber, lapis, jade
    """
    religion_info = add_religious_tenet_to_city(religion_info, "tears_of_the_gods")

    def bool_map_generator(game):
        this_city_rowcols = game.player_cities__potential_owned_rowcols
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities__ownership_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2

        fish = RESOURCE_TO_IDX["gems"]
        whales = RESOURCE_TO_IDX["amber"]
        crabs = RESOURCE_TO_IDX["lapis"]
        pearls = RESOURCE_TO_IDX["jade"]
        
        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        res = (this_city_resources == fish) | (this_city_resources == whales) | (this_city_resources == crabs) |(this_city_resources == pearls) 

        this_city_center = game.player_cities__city_rowcols
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        center_res = (this_city_center_resources == fish) | (this_city_center_resources == whales) | (this_city_center_resources == crabs) | (this_city_center_resources == pearls)
        
        can_see_resource_bool = game.visible_resources_map_players[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] > 0
        can_see_resource_center_bool = game.visible_resources_map_players[
            this_city_center[0], this_city_center[1]
        ] > 0

        return this_city_currently_owned * res * can_see_resource_bool, game_map_rowcols, center_res  * can_see_resource_center_bool, this_city_center

    to_add = jnp.array([0, 0, 2, 1, 0, 0, 0])
    religion_info = add_yield_tiles_to_given_city(religion_info, game, to_add, bool_map_generator)
    return religion_info

def _vision_quests(religion_info, game, player_id):
    """
    +1 happiness, +1 gold from shrines
    """
    religion_info = add_religious_tenet_to_city(religion_info, "vision_quests")
    
    b_idx = GameBuildings["shrine"]._value_
    to_add = jnp.array([0, 0, 1, 0, 0, 0, 1, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )
    return religion_info

def _works_spirituals(religion_info, game, player_id):
    """
    +1 faith from plantation luxuries and bananas
    """
    religion_info = add_religious_tenet_to_city(religion_info, "works_spirituals")

    def bool_map_generator(game):
        this_city_rowcols = game.player_cities__potential_owned_rowcols
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities__ownership_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2

        fish = RESOURCE_TO_IDX["dyes"]
        whales = RESOURCE_TO_IDX["coconut"]
        crabs = RESOURCE_TO_IDX["tobacco"]
        pearls = RESOURCE_TO_IDX["olives"]
        coral = RESOURCE_TO_IDX["sugar"]
        citrus = RESOURCE_TO_IDX["citrus"]
        cotton = RESOURCE_TO_IDX["cotton"]
        incense = RESOURCE_TO_IDX["incense"]
        coffee = RESOURCE_TO_IDX["coffee"]
        silk = RESOURCE_TO_IDX["silk"]
        perfume = RESOURCE_TO_IDX["perfume"]
        spices = RESOURCE_TO_IDX["spices"]
        chocolate = RESOURCE_TO_IDX["chocolate"]
        rubber = RESOURCE_TO_IDX["rubber"]
        tea = RESOURCE_TO_IDX["tea"]
        banana = RESOURCE_TO_IDX["banana"]
        
        this_city_resources = game.all_resource_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]]
        res = (this_city_resources == fish) | (this_city_resources == whales) | (this_city_resources == crabs) |(this_city_resources == pearls) | (this_city_resources == coral) | (this_city_resources == citrus) | (this_city_resources == cotton) | (this_city_resources == incense) | (this_city_resources == coffee) | (this_city_resources == silk) | (this_city_resources == perfume) | (this_city_resources == spices) | (this_city_resources == chocolate) | (this_city_resources == rubber) | (this_city_resources == tea) | (this_city_resources == banana) 

        this_city_center = game.player_cities__city_rowcols
        this_city_center_resources = game.all_resource_map[this_city_center[0], this_city_center[1]]
        center_res = (this_city_center_resources == fish) | (this_city_center_resources == whales) | (this_city_center_resources == crabs) | (this_city_center_resources == pearls) | (this_city_center_resources == coral) | (this_city_center_resources == citrus) | (this_city_center_resources  == cotton) | (this_city_center_resources == incense) | (this_city_center_resources == coffee) | (this_city_center_resources == silk) | (this_city_center_resources == perfume) | (this_city_center_resources == spices) | (this_city_center_resources == chocolate) | (this_city_center_resources == rubber) | (this_city_center_resources == tea) | (this_city_center_resources == banana)
        
        can_see_resource_bool = game.visible_resources_map_players[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] > 0
        can_see_resource_center_bool = game.visible_resources_map_players[
            this_city_center[0], this_city_center[1]
        ] > 0

        return this_city_currently_owned * res * can_see_resource_bool, game_map_rowcols, center_res  * can_see_resource_center_bool, this_city_center

    to_add = jnp.array([0, 0, 0, 1, 0, 0, 0])
    religion_info = add_yield_tiles_to_given_city(religion_info, game, to_add, bool_map_generator)
    return religion_info


def _ceremonial_burial(religion_info, game, player_id):
    """
    +1 happiness for every 2 cities following this religion
    +1 happiness from the palace
    """
    religion_info = add_religious_tenet_to_city(religion_info, "ceremonial_burial")
    n_cities_following = (game.player_cities__religion_info__religious_population[:, :, player_id[0]] > 0).sum() // 2

    new_yields = religion_info.building_yields + jnp.array([0, 0, 0, 0, 0, 0, n_cities_following, 0])
    religion_info = religion_info.replace(building_yields=new_yields)

    b_idx = GameBuildings["palace"]._value_
    to_add = jnp.array([0, 0, 0, 0, 0, 0, 1, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )
    return religion_info

def _church_property(religion_info, game, player_id):
    """
    +1 gold from each city following this religion
    +1 faith in each city following this religion
    """
    religion_info = add_religious_tenet_to_city(religion_info, "church_property")
    n_cities_following = (game.player_cities__religion_info__religious_population[:, :, player_id[0]] > 0).sum()

    new_yields = religion_info.building_yields + jnp.array([0, 0, n_cities_following, n_cities_following, 0, 0, 0, 0])
    religion_info = religion_info.replace(building_yields=new_yields)
    return religion_info

def _dawah(religion_info, game, player_id):
    """
    +4 culture from the palace
    """
    religion_info = add_religious_tenet_to_city(religion_info, "dawah")
    b_idx = GameBuildings["palace"]._value_
    to_add = jnp.array([0, 0, 0, 0, 4, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )
    return religion_info


def _initiation_rites(religion_info, game, player_id):
    """
    +2 gold from shrines and temples
    """
    religion_info = add_religious_tenet_to_city(religion_info, "initiation_rites")
    b_idx = GameBuildings["shrine"]._value_
    to_add = jnp.array([0, 0, 2, 0, 0, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )
    b_idx = GameBuildings["temple"]._value_
    to_add = jnp.array([0, 0, 2, 0, 0, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )
    return religion_info


def _messiah(religion_info, game, player_id):
    """
    Great prophets are earned with 25% less faith
    """
    religion_info = add_religious_tenet_to_city(religion_info, "messiah")
    return religion_info

def _missionary_zeal(religion_info, game, player_id):
    """
    missionaries cost 50% less faith
    +1 religion spread to missionaries
    """
    religion_info = add_religious_tenet_to_city(religion_info, "missionary_zeal")
    new_spreads = religion_info.missionary_spreads + 1

    return religion_info.replace(missionary_spreads=new_spreads)


def _mithraea(religion_info, game, player_id):
    """
    +1% gold per follower, max 20%
    """
    religion_info = add_religious_tenet_to_city(religion_info, "mithraea")
    n_followers = game.player_cities__religion_info__religious_population[:, :, player_id[0]].sum()
    n_followers = jnp.minimum(n_followers, 20) / 100
    new_accel = religion_info.citywide_yield_accel.at[GOLD_IDX].add(n_followers)
    return religion_info.replace(citywide_yield_accel=new_accel)

def _religious_unity(religion_info, game, player_id):
    """
    Relgion spreads to friendly CS 2x rate
    +45 influence resting point with CS that follow this religion (defer this to cs inf step) -- removed for simplicity!
    """
    religion_info = add_religious_tenet_to_city(religion_info, "religious_unity")
    new_accel = religion_info.cs_perturn_influence_accel + 1
    return religion_info.replace(cs_perturn_influence_accel=new_accel)

def _salat(religion_info, game, player_id):
    """
    +4 prod from palace
    """
    religion_info = add_religious_tenet_to_city(religion_info, "salat")
    b_idx = GameBuildings["palace"]._value_
    to_add = jnp.array([0, 4, 0, 0, 0, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )

    return religion_info

def _tithe(religion_info, game, player_id):
    """
    +1 gold for ever 4 followers of this religion
    """
    religion_info = add_religious_tenet_to_city(religion_info, "tithe")
    n_followers = game.player_cities__religion_info__religious_population[:, :, player_id[0]].sum() // 4
    new_yields = religion_info.building_yields + jnp.array([0, 0, n_followers, 0, 0, 0, 0, 0])
    religion_info = religion_info.replace(building_yields=new_yields)
    return religion_info

def _world_church(religion_info, game, player_id):
    """
    +1 culture from every 6 followers of this religion
    """
    religion_info = add_religious_tenet_to_city(religion_info, "world_church")
    n_followers = game.player_cities__religion_info__religious_population[:, :, player_id[0]].sum() // 6
    new_yields = religion_info.building_yields + jnp.array([0, 0, 0, 0, n_followers, 0, 0, 0])
    religion_info = religion_info.replace(building_yields=new_yields)

    return religion_info

def _zakat(religion_info, game, player_id):
    """
    +8 gold from the palace
    """
    religion_info = add_religious_tenet_to_city(religion_info, "zakat")
    b_idx = GameBuildings["palace"]._value_
    to_add = jnp.array([0, 0, 8, 0, 0, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )

    return religion_info

def _cathedrals(religion_info, game, player_id):
    """
    Use 100 faith to buy catherals
    """
    religion_info = add_religious_tenet_to_city(religion_info, "cathedrals")

    return religion_info

def _choral_music(religion_info, game, player_id):
    """
    +3 culture form temples
    """
    religion_info = add_religious_tenet_to_city(religion_info, "choral_music")
    b_idx = GameBuildings["temple"]._value_
    to_add = jnp.array([0, 0, 0, 0, 3, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )

    return religion_info

def _devoted_elite(religion_info, game, player_id):
    """
    +3 prod, +3 gold, -1 unhappiness in cities with >= 10 pop
    """
    religion_info = add_religious_tenet_to_city(religion_info, "devoted_elite")
    to_add_bool = game.player_cities__population >= 10
    to_add  = jnp.array([0, 3, 3, 0, 0, 0, 1, 0])
    new_yields = religion_info.building_yields + (to_add * to_add_bool)
    return religion_info.replace(building_yields=new_yields)

def _devout_performers(religion_info, game, player_id):
    """
    +2 culture from each circus
    +1 culture from each colosseum
    +3 happiness, +4 culture from Circus Maximus
    """
    religion_info = add_religious_tenet_to_city(religion_info, "devout_performers")
    b_idx = GameBuildings["circus"]._value_
    to_add = jnp.array([0, 0, 0, 0, 2, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )

    b_idx = GameBuildings["colosseum"]._value_
    to_add = jnp.array([0, 0, 0, 0, 0, 0, 1, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )

    b_idx = GameBuildings["circus_maximus"]._value_
    to_add = jnp.array([0, 0, 0, 0, 4, 0, 3, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )

    return religion_info

def _divine_inspiration(religion_info, game, player_id):
    """
    +1 faith, +1 culture from national, natural, world wonders
    """
    religion_info = add_religious_tenet_to_city(religion_info, "divine_inspiration")
    
    def bool_map_generator(game):
        this_city_rowcols = game.player_cities__potential_owned_rowcols
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities__ownership_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2

        this_city_nw = game.nw_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] > 0

        this_city_center = game.player_cities__city_rowcols
        this_city_center_nw = game.nw_map[this_city_center[0], this_city_center[1]] > 0

        return this_city_currently_owned * this_city_nw, game_map_rowcols, this_city_center_nw, this_city_center

    to_add = jnp.array([0, 0, 0, 1, 1, 0, 0])
    religion_info = add_yield_tiles_to_given_city(religion_info, game, to_add, bool_map_generator)

    return religion_info

def _feed_the_world(religion_info, game, player_id):
    """
    +1 food from shrines
    +2 food from temples 
    """
    religion_info = add_religious_tenet_to_city(religion_info, "feed_the_world")
    b_idx = GameBuildings["shrine"]._value_
    to_add = jnp.array([1, 0, 0, 0, 0, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )

    b_idx = GameBuildings["temple"]._value_
    to_add = jnp.array([2, 0, 0, 0, 0, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )
    return religion_info

def _followers_of_the_refined_crafts(religion_info, game, player_id):
    """
    +2 gold, +2 faith, +1 culture from mints, gem cutters, breweries, groceries, censer makers, textile mills
    """
    religion_info = add_religious_tenet_to_city(religion_info, "followers_of_the_refined_crafts")
    b_idx = GameBuildings["mint"]._value_
    to_add = jnp.array([0, 0, 2, 2, 1, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )
    b_idx = GameBuildings["gemcutter"]._value_
    to_add = jnp.array([0, 0, 2, 2, 1, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )
    b_idx = GameBuildings["brewery"]._value_
    to_add = jnp.array([0, 0, 2, 2, 1, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )
    b_idx = GameBuildings["grocer"]._value_
    to_add = jnp.array([0, 0, 2, 2, 1, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )
    b_idx = GameBuildings["censer"]._value_
    to_add = jnp.array([0, 0, 2, 2, 1, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )
    b_idx = GameBuildings["textile"]._value_
    to_add = jnp.array([0, 0, 2, 2, 1, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )

    return religion_info

def _gurdwaras(religion_info, game, player_id):
    """
    180 faith to buy gurdwara
    """
    religion_info = add_religious_tenet_to_city(religion_info, "gurdwaras")

    return religion_info

def _guruship(religion_info, game, player_id):
    """
    +2 food, +1 prod, +1 gold id any city with specialist
    We defer the application of this bonus to GameState.step_specialists_and_great_people()
    """
    religion_info = add_religious_tenet_to_city(religion_info, "guruship")

    return religion_info

def _holy_warriors(religion_info, game, player_id):
    """
    Barracks, armory, military academy give +1 production each
    """
    religion_info = add_religious_tenet_to_city(religion_info, "holy_warriors")
    b_idx = GameBuildings["barracks"]._value_
    to_add = jnp.array([0, 1, 0, 0, 0, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )
    b_idx = GameBuildings["armory"]._value_
    to_add = jnp.array([0, 1, 0, 0, 0, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )
    b_idx = GameBuildings["military_academy"]._value_
    to_add = jnp.array([0, 1, 0, 0, 0, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )
    return religion_info

def _liturgical_drama(religion_info, game, player_id):
    """
    +2 faith, +1 culture from amphitheaters, opera houses
    """
    religion_info = add_religious_tenet_to_city(religion_info, "liturgical_drama")
    b_idx = GameBuildings["amphitheater"]._value_
    to_add = jnp.array([0, 0, 0, 2, 1, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )
    b_idx = GameBuildings["opera_house"]._value_
    to_add = jnp.array([0, 0, 0, 2, 1, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )

    return religion_info

def _mandirs(religion_info, game, player_id):
    """
    Can use faith to purchase mandirs
    """
    religion_info = add_religious_tenet_to_city(religion_info, "mandirs")

    return religion_info

def _mosques(religion_info, game, player_id):
    """
    Can use faith to purchase mosques
    """
    religion_info = add_religious_tenet_to_city(religion_info, "mosques")

    return religion_info

def _pagodas(religion_info, game, player_id):
    """
    Can use faith to purchase pagodas
    """
    religion_info = add_religious_tenet_to_city(religion_info, "pagodas")

    return religion_info

def _peace_gardens(religion_info, game, player_id):
    """
    +1 happiness from gardens
    +2 gold, +1 prod from watermills
    """
    religion_info = add_religious_tenet_to_city(religion_info, "peace_gardens")
    b_idx = GameBuildings["garden"]._value_
    to_add = jnp.array([0, 0, 0, 0, 0, 0, 1, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )
    b_idx = GameBuildings["watermill"]._value_
    to_add = jnp.array([0, 1, 2, 0, 0, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )

    return religion_info

def _religious_art(religion_info, game, player_id):
    """
    +5 culture, +5 tourism from Hermitage
    """
    religion_info = add_religious_tenet_to_city(religion_info, "religious_art")
    b_idx = GameBuildings["hermitage"]._value_
    to_add = jnp.array([0, 0, 0, 0, 5, 0, 0, 5])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )

    return religion_info

def _religious_center(religion_info, game, player_id):
    """
    +1 happiness, +1 gold, +1 production from temples
    """
    religion_info = add_religious_tenet_to_city(religion_info, "religious_center")
    b_idx = GameBuildings["temple"]._value_
    to_add = jnp.array([0, 1, 1, 0, 0, 0, 1, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )

    return religion_info

def _religious_community(religion_info, game, player_id):
    """
    +1% prod per follower, max of 10%
    """
    religion_info = add_religious_tenet_to_city(religion_info, "religious_community")
    n_followers = game.player_cities__religion_info__religious_population[:, :, player_id[0]].sum()
    n_followers = jnp.minimum(n_followers, 10) / 100
    new_accel = religion_info.citywide_yield_accel.at[PROD_IDX].add(n_followers)
    return religion_info.replace(citywide_yield_accel=new_accel)

def _sacred_waters(religion_info, game, player_id):
    """
    +1 happiness for cities on rivers
    +2 faith, +1 food from gardens
    """
    religion_info = add_religious_tenet_to_city(religion_info, "sacred_waters")
    b_idx = GameBuildings["garden"]._value_
    to_add = jnp.array([0, 0, 0, 0, 2, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )
    
    on_river = game.edge_river_map[game.player_cities__city_rowcols[0], game.player_cities__city_rowcols[1]].sum() > 0
    to_add = jnp.array([0, 0, 0, 0, 0, 0, 1, 0]) * on_river
    new_yields = religion_info.building_yields + to_add
    return religion_info.replace(building_yields=new_yields)

def _synagogues(religion_info, game, player_id):
    """
    Can use faith to purchase synagogues
    """
    religion_info = add_religious_tenet_to_city(religion_info, "synagogues")

    return religion_info

def _viharas(religion_info, game, player_id):
    """
    Can use faith to purchase viharas
    """
    religion_info = add_religious_tenet_to_city(religion_info, "viharas")

    return religion_info


def _defender_of_the_faith(religion_info, game, player_id):
    """
    +1 faith, +1 Culture Walls, Castle, Arsenal
    """
    religion_info = add_religious_tenet_to_city(religion_info, "defender_of_the_faith")

    b_idx = GameBuildings["walls"]._value_
    to_add = jnp.array([0, 0, 0, 1, 1, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )

    b_idx = GameBuildings["castle"]._value_
    to_add = jnp.array([0, 0, 0, 1, 1, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )

    b_idx = GameBuildings["arsenal"]._value_
    to_add = jnp.array([0, 0, 0, 1, 1, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )

    return religion_info


def _dharma(religion_info, game, player_id):
    """
    +10 science from grand temple
    """
    religion_info = add_religious_tenet_to_city(religion_info, "dharma")
    b_idx = GameBuildings["grand_temple"]._value_
    to_add = jnp.array([0, 0, 0, 0, 0, 10, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )

    return religion_info

def _disciples(religion_info, game, player_id):
    """
    +2 Gold and Culture from Temples
    """
    religion_info = add_religious_tenet_to_city(religion_info, "disciples")
    b_idx = GameBuildings["temple"]._value_
    to_add = jnp.array([0, 0, 2, 0, 2, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )
    return religion_info

def _hajj(religion_info, game, player_id):
    """
    +3 prod, +3 food, +3 culture from palace
    """
    religion_info = add_religious_tenet_to_city(religion_info, "hajj")
    b_idx = GameBuildings["palace"]._value_
    to_add = jnp.array([3, 3, 0, 0, 3, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )

    return religion_info

def _jizya(religion_info, game, player_id):
    """
    +6 prod, +6 gold from Grand Temple
    """
    religion_info = add_religious_tenet_to_city(religion_info, "jizya")
    b_idx = GameBuildings["grand_temple"]._value_
    to_add = jnp.array([0, 6, 6, 0, 0, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )

    return religion_info

def _just_war(religion_info, game, player_id):
    """
    +20% combat strength near enemy cities that follow this religion
    """
    religion_info = add_religious_tenet_to_city(religion_info, "just_war")

    return religion_info

def _karma(religion_info, game, player_id):
    """
    +8 culture in the holy city
    """
    religion_info = add_religious_tenet_to_city(religion_info, "karma")
    b_idx = GameBuildings["palace"]._value_
    to_add = jnp.array([0, 0, 0, 0, 8, 0, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )

    return religion_info
 
def _kotel(religion_info, game, player_id):
    """
    +6 happiness from the palace
    """
    religion_info = add_religious_tenet_to_city(religion_info, "kotel")
    b_idx = GameBuildings["palace"]._value_
    to_add = jnp.array([0, 0, 0, 0, 0, 0, 6, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )
    return religion_info

    game = add_religious_tenet_to_city(religion_info, "kotel")
    return game

def _pilgrimage(religion_info, game, player_id):
    """
    +4 faith, +3 culture from every foreign city following this religion
    """
    religion_info = add_religious_tenet_to_city(religion_info, "pilgrimage")
    # All cities - player cities
    has_maj = game.player_cities__religion_info__religious_population.sum(-1) > 0

    n_cities = ((game.player_cities__religion_info__religious_population.argmax(-1) == player_id) * has_maj).at[player_id[0]].set(0).sum()
    to_add = jnp.array([0, 0, 0, 4 * n_cities, 3 * n_cities, 0, 0, 0])
    new_yields = religion_info.building_yields + to_add
    return religion_info.replace(building_yields=new_yields)

def _promised_land(religion_info, game, player_id):
    """
    Religion spreads 100% faster to your cities and cities following your religion
    For the sake of simplicity, we have nerfed to 25% faster, but towards everyone
    """
    religion_info = add_religious_tenet_to_city(religion_info, "promised_land")
    # Array is only (6, max_num_cities) within fn context
    new_pressure = religion_info.player_perturn_influence_accel.at[player_id[0]].add(0.25)
    return religion_info.replace(player_perturn_influence_accel=new_pressure)

def _religious_troubadours(religion_info, game, player_id):
    """
    +3 faith fro every trade route sent to foreign cities (defd to action_space.py)
    +1 trade routes
    """
    religion_info = add_religious_tenet_to_city(religion_info, "religious_troubadours")
    return religion_info


def _sanctified_innovations(religion_info, game, player_id):
    """
    +1 prod, +1 science, +1 faith from Nat wonders
    """
    religion_info = add_religious_tenet_to_city(religion_info, "sanctified_innovations")
    # Buildings owned is coming through already indexed by player_id and city_int, as we are on a per-city basis
    n_nat_wonders = ((game.player_cities__buildings_owned == 1) & BLDG_IS_NAT_WONDER).sum()
    to_add = jnp.array([0, 1, 0, 1, 0, 1, 0, 0]) * n_nat_wonders
    new_yields = religion_info.building_yields + to_add
    return religion_info.replace(building_yields=new_yields)


def _unity_of_the_prophets(religion_info, game, player_id):
    """
    +4 faith, +2 prod from improved plantation luxuries
    Inquisitors and great prophets reduce this religion by half instead of eliminating
    """
    religion_info = add_religious_tenet_to_city(religion_info, "unity_of_the_prophets")
    
    def bool_map_generator(game):
        this_city_rowcols = game.player_cities__potential_owned_rowcols
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        this_city_currently_owned = game.player_cities__ownership_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] >= 2

        pastures = game.improvement_map[
            game_map_rowcols[:, 0], game_map_rowcols[:, 1]
        ] == (Improvements["plantation"]._value_ + 1)

        citywide_bool = this_city_currently_owned & pastures

        this_city_center = game.player_cities__city_rowcols
        this_city_center_resources = game.improvement_map[this_city_center[0], this_city_center[1]]
        center_res = this_city_center_resources == (Improvements["plantation"]._value_ + 1)
        
        return citywide_bool, game_map_rowcols, center_res, this_city_center

    to_add = jnp.array([0, 2, 0, 4, 0, 0, 0])
    religion_info = add_yield_tiles_to_given_city(religion_info, game, to_add, bool_map_generator)
    return religion_info


def _apostolic_palace(religion_info, game, player_id):
    """
    +5 prod, +5 gold, +5 faith, +5 culture, +5 food, +5 science from Grand Temple
    """
    religion_info = add_religious_tenet_to_city(religion_info, "apostolic_palace")
    b_idx = GameBuildings["grand_temple"]._value_
    to_add = jnp.array([5, 5, 5, 5, 5, 5, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        b_idx
    )
    return religion_info
 

def _city_of_god(religion_info, game, player_id):
    """
    +3 food, +3 science from World Wonders
    Free great prophet
    """
    religion_info = add_religious_tenet_to_city(religion_info, "city_of_god")
    n_nat_wonders = ((game.player_cities__buildings_owned == 1) & BLDG_IS_WORLD_WONDER).sum()
    to_add = jnp.array([3, 0, 0, 0, 0, 3, 0, 0]) * n_nat_wonders
    new_yields = religion_info.building_yields + to_add
    return religion_info.replace(building_yields=new_yields)

def _houses_of_worship(religion_info, game, player_id):
    """
    Can buy art hall writing aufitorium, vocal chamber with faith
    """
    religion_info = add_religious_tenet_to_city(religion_info, "houses_of_worship")
    return religion_info

def _indulgences(religion_info, game, player_id):
    """
    +1 food, +1 prod, +1 culture, +1 faith, +1 gold in cities following this religion
    """
    religion_info = add_religious_tenet_to_city(religion_info, "indulgences")
    # We're only ever in this fn if the given city is following this religion
    new_yields = religion_info.building_yields + jnp.array([1, 1, 1, 1, 1, 0, 0, 0])
    return religion_info.replace(building_yields=new_yields)

def _jesuit_education(religion_info, game, player_id):
    """
    Can buy libraries, universities, observatories, public schools, research labs with faith
    +2 faith, +1 science from science buildings
    """
    religion_info = add_religious_tenet_to_city(religion_info, "jesuit_education")
    shrine_idx = GameBuildings["library"]._value_
    to_add = jnp.array([0, 0, 0, 0, 2, 1, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        shrine_idx
    )
    shrine_idx = GameBuildings["public_school"]._value_
    to_add = jnp.array([0, 0, 0, 0, 2, 1, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        shrine_idx
    )
    shrine_idx = GameBuildings["university"]._value_
    to_add = jnp.array([0, 0, 0, 0, 2, 1, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        shrine_idx
    )
    shrine_idx = GameBuildings["laboratory"]._value_
    to_add = jnp.array([0, 0, 0, 0, 2, 1, 0, 0])
    religion_info = add_bldg_yields_for_existing_bldg(
        religion_info, 
        game.player_cities__buildings_owned, 
        to_add, 
        shrine_idx
    )

    return religion_info


def _sacred_sites(religion_info, game, player_id):
    """
    +2 culture, +2 gold, +2 faith from buildings unlocked with religious beliefs (e.g., pagodas)
    art hall writing aufitorium, vocal chamber
    """
    religion_info = add_religious_tenet_to_city(religion_info, "sacred_sites")
    to_add = jnp.array([0, 0, 2, 2, 2, 0, 0, 0])

    bldg_idx = GameBuildings["cathedral"]._value_
    religion_info = add_bldg_yields_for_existing_bldg(religion_info, game.player_cities__buildings_owned, to_add, bldg_idx)
    bldg_idx = GameBuildings["gurdwara"]._value_
    religion_info = add_bldg_yields_for_existing_bldg(religion_info, game.player_cities__buildings_owned, to_add, bldg_idx)
    bldg_idx = GameBuildings["mandir"]._value_
    religion_info = add_bldg_yields_for_existing_bldg(religion_info, game.player_cities__buildings_owned, to_add, bldg_idx)
    bldg_idx = GameBuildings["mosque"]._value_
    religion_info = add_bldg_yields_for_existing_bldg(religion_info, game.player_cities__buildings_owned, to_add, bldg_idx)
    bldg_idx = GameBuildings["pagoda"]._value_
    religion_info = add_bldg_yields_for_existing_bldg(religion_info, game.player_cities__buildings_owned, to_add, bldg_idx)
    bldg_idx = GameBuildings["vihara"]._value_
    religion_info = add_bldg_yields_for_existing_bldg(religion_info, game.player_cities__buildings_owned, to_add, bldg_idx)
    bldg_idx = GameBuildings["artist_house"]._value_
    religion_info = add_bldg_yields_for_existing_bldg(religion_info, game.player_cities__buildings_owned, to_add, bldg_idx)
    bldg_idx = GameBuildings["writer_house"]._value_
    religion_info = add_bldg_yields_for_existing_bldg(religion_info, game.player_cities__buildings_owned, to_add, bldg_idx)
    bldg_idx = GameBuildings["music_house"]._value_
    religion_info = add_bldg_yields_for_existing_bldg(religion_info, game.player_cities__buildings_owned, to_add, bldg_idx)
    return religion_info

def _swords_into_plowshares(religion_info, game, player_id):
    """
    +1 food, +20% growth accel in all cities
    """
    religion_info = add_religious_tenet_to_city(religion_info, "swords_into_plowshares")
    new_yields = religion_info.building_yields + jnp.array([1, 0, 0, 0, 0, 0, 0, 0])
    new_accel = religion_info.citywide_yield_accel.at[FOOD_IDX].add(0.2)
    return religion_info.replace(building_yields=new_yields, citywide_yield_accel=new_accel)

def _underground_sect(religion_info, game, player_id):
    """
    +1 science for every 4 followers of this religion
    """
    religion_info = add_religious_tenet_to_city(religion_info, "underground_sect")
    n_followers = (game.player_cities__religion_info__religious_population[:, :, player_id[0]]).sum() // 4

    new_yields = religion_info.building_yields + jnp.array([0, 0, 0, 0, 0, n_followers, 0, 0])
    return religion_info.replace(building_yields=new_yields)

def _work_ethic(religion_info, game, player_id):
    """
    Can use faith to purchase, workshops, windmills, factories, hydro plants
    """
    religion_info = add_religious_tenet_to_city(religion_info, "work_ethic")
    return religion_info


ALL_RELIGIOUS_TENETS_FNS = [do_nothing]
ALL_RELIGIOUS_TENETS_NAMES = ["_" + x.name for x in ReligiousTenets]


for tenet in ALL_RELIGIOUS_TENETS_NAMES:
    fn = getattr(sys.modules[__name__], tenet)
    ALL_RELIGIOUS_TENETS_FNS.append(fn)

# Finally need one more mapping that is dict(tenet_name: building_id) for all of the tenets that grant the ability to purchase
# buildings (e.g., pagodas, catherals)
TENET_TO_BUILDING = {
    "cathedrals": [GameBuildings["cathedral"]._value_],
    "gurdwaras": [GameBuildings["gurdwara"]._value_],
    "mandirs": [GameBuildings["mandir"]._value_, GameBuildings["monastery"]._value_],
    "mosques": [GameBuildings["mosque"]._value_],
    "pagodas": [GameBuildings["pagoda"]._value_],
    "synagogues": [GameBuildings["synagogue"]._value_],
    "viharas": [GameBuildings["vihara"]._value_],
    "houses_of_worship": [GameBuildings["artist_house"]._value_, 
                          GameBuildings["writer_house"]._value_,
                          GameBuildings["music_house"]._value_],
    "work_ethic": [GameBuildings["factory"]._value_]
}


def zero_out_fields_for_religion_update(pytree, names_to_zero, idx_0):
    """
    Returns a new pytree where specified fields have element [idx_0, idx_1] set to 0.
    Works in jitted context as long as names_to_zero is known at compile time.

    Args:
        pytree: flax.struct.dataclass instance
        names_to_zero: list of field names (must be static for JIT)
        idx_0, idx_1: integer indices

    Returns:
        New pytree with indexed elements zeroed in specified fields.
    """
    return type(pytree)(**{
        f.name: (
            getattr(pytree, f.name).at[idx_0].set(
                0 if ("accel" in f.name or "carryover" in f.name or "_dist_mod" in f.name) else 0
            )
            if f.name in names_to_zero
            else getattr(pytree, f.name)
        )
        for f in fields(pytree)
    })

def extract_fields(pytree, names_to_extract, player_id):
    """
    Extract a dict of selected fields from a dataclass-based pytree.
    To be called within a vmap-over-building-idx context

    Works under JAX JIT as long as `names_to_extract` is static.

    Returns:
        A flat dict {field_name: field_value} for the selected fields.
    """
    return {
        f.name: getattr(pytree, f.name)[player_id[0]]
        for f in fields(pytree)
        if f.name in names_to_extract
    }

def extract_map_fields(pytree, names_to_extract, player_id):
    """
    Extract a dict of selected fields from a dataclass-based pytree.
    To be called within a vmap-over-building-idx context

    Works under JAX JIT as long as `names_to_extract` is static.

    Returns:
        A flat dict {field_name: field_value} for the selected fields.
    """
    return {
        f.name: getattr(pytree, f.name)[player_id[0]]
        for f in fields(pytree)
        if f.name in names_to_extract
    }

def add_one_to_appropriate_fields(pytree, relevant_fields, idx_0):
    """
    Fields that are multipliers (e.g., any "accel") need a base one 1 added to them.
    """
    return type(pytree)(**{
        f.name: (
            getattr(pytree, f.name).at[idx_0].add(
                1 if ("accel" in f.name or "carryover" in f.name or "_dist_mod" in f.name) else 2 if ("missionary_spreads" in f.name) else 0
            )
            if f.name in relevant_fields
            else getattr(pytree, f.name)
        )
        for f in fields(pytree)
    })

player_religion_update_fn_nonmaps = make_update_fn_religion(TO_ZERO_OUT_FOR_RELIGION_STEP_SANS_MAPS, only_maps=False)
player_religion_update_fn_maps = make_update_fn_religion(TO_ZERO_OUT_FOR_RELIGION_STEP_ONLY_MAPS, only_maps=True)

KEEP_NAMES = (*TAKE_WITH_PLAYER_ID_AND_CITY_INT, *TAKE_WITH_PLAYER_ID, *PASS_THRUS)

def get_nested_attr(obj, dotted):
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj

def make_subset_dataclass_from_paths(paths, *, name="GameStateSubset"):
    """
    Build a flax.struct.dataclass whose fields correspond to `paths`.
    Each dotted path 'a.b.c' becomes a field 'a__b__c'.
    """
    def _sanitize(p): return p.replace(".", "__")
    annotations = { _sanitize(p): jnp.ndarray for p in paths }
    tmp_cls = type(name, (), {"__annotations__": annotations})
    return struct.dataclass(tmp_cls), {p: _sanitize(p) for p in paths}

GameStateSubset, PATH2FIELD = make_subset_dataclass_from_paths(KEEP_NAMES)

def slice_to_subset(batch, idx0, idx1):
    out = {}
    for path in KEEP_NAMES:
        leaf = get_nested_attr(batch, path)
        if path in TAKE_WITH_PLAYER_ID:
            leaf = leaf[idx0]
        elif path in TAKE_WITH_PLAYER_ID_AND_CITY_INT:
            leaf = leaf[idx0, idx1]
        # else passthrough
        out[PATH2FIELD[path]] = leaf
    return GameStateSubset(**out)


@partial(jax.vmap, in_axes=(0, None, 0, None, 0, 0, None))
def _religion_per_city_vmap(religion_info, 
                            player_id,
                            maj_religion_idx, 
                            gamewide_religious_tenets, 
                            has_maj_religion,
                            city_int,
                            game):
    """
    This function vmaps over cities and computes the religious bonuses based on the given citys' majority religion. 
    EXCEPTION: Founder beliefs can only be given to cities that the given player_id owns.

    Each belief needs to be filtered on two criteria:
        (1) a majority religion exists (if not, then no bonuses)
        (2) belief type X exists for majority religion (if not, then no bonuses)

    (2) is critical. If the majority relgion in a city is religion X, but religion X does not have an enhancer belief,
    then the city should not receive a bonus for the enhancer belief!

    Args:
        religion_info: see ReligionInfo class from primitives.py. All arrays are without max_num_cities here
        player_id: (1,)
        maj_religion_idx: () -- only tells us argmax() over religious_population, not if majority exists
        gamewide_religious_tenets: (6, len(ReligiousTenets))
        has_maj_religion: (6,) -- in this city, which religions are > half pop? will be .sum() > 0 if a major religion exists in this city 
        bldgs_owned: (len(GameBuildings),) --
        all_resource_map: ()
    """
    # (91,) (1,) () (6, 91) (6,)
    has_maj_religion = has_maj_religion.sum() > 0
    gamestate_subset = slice_to_subset(game, idx0=player_id[0], idx1=city_int)

    # This will be used to zero-out the dispatch int, thereby eliminating any religious bonuses that may erroneously occur
    # Now we grab the bool vector that described the tenets of the maj religion in this city
    # Need to zero-out the founder tenet, as that only goes to the player that founded the religion. Then,
    # we will add the founder tenet of player_id back into the back
    maj_religion_tenets = gamewide_religious_tenets[maj_religion_idx]
    gamewide_founder_tenet_idx = maj_religion_tenets[MAX_IDX_PANTHEON: MAX_IDX_FOUNDER].argmax() + MAX_IDX_PANTHEON
    this_player_founder_tenet = gamewide_religious_tenets[player_id[0]][MAX_IDX_PANTHEON: MAX_IDX_FOUNDER].argmax() + MAX_IDX_PANTHEON
    
    # Need to be careful here, in the event that the given player_id does **NOT** have a founder tenet, we need to
    # set  index to 0. 
    to_set_for_founder = jnp.where(
        gamewide_religious_tenets[player_id[0]][MAX_IDX_PANTHEON: MAX_IDX_FOUNDER].sum() > 0,
        1,
        0
    )
    religion_mask = maj_religion_tenets.at[gamewide_founder_tenet_idx].set(0).at[this_player_founder_tenet].set(to_set_for_founder)

    # If the given city does not have a majority religion, then we fall back to this player's chosen pantheon
    religion_mask_no_maj = gamewide_religious_tenets[player_id[0]].at[MAX_IDX_PANTHEON:].set(0)
    religion_mask = (
        has_maj_religion * religion_mask 
        + (1 - has_maj_religion) * religion_mask_no_maj
    )

    outs = [f(religion_info, gamestate_subset, player_id) for f in ALL_RELIGIOUS_TENETS_FNS[1:]]
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *outs)

    def weighted_sum_tree(stacked_leaves, weights):
        return jax.tree_util.tree_map( 
            lambda leaf: jnp.tensordot(weights, leaf, axes=((0,), (0,))),
            stacked_leaves
        )
    
    # religious pop need to be un-duplicated, as we do not zero it out prior to this function call
    # the same is happening with religious pressure accumulation
    _prev_religious_population = religion_info.religious_population
    _prev_player_perturn_influence_cumulative = religion_info.player_perturn_influence_cumulative
    _prev_cs_perturn_influence_cumulative = religion_info.cs_perturn_influence_cumulative

    religion_info = weighted_sum_tree(stacked, religion_mask.astype(jnp.float32))
    
    religion_info = religion_info.replace(
        religious_population=_prev_religious_population,
        player_perturn_influence_cumulative=_prev_player_perturn_influence_cumulative,
        cs_perturn_influence_cumulative=_prev_cs_perturn_influence_cumulative
    )
    
    return religion_info

def apply_religion_per_city(game, player_id, maj_religion_idx_per_city):
    """
    Args:
        game:
        player_id:
        maj_religion_idx_per_city: (max_num_cities,) --> gives idx of the religion with the most pop in each of player_id's cities
    Need to ensure that we give all bonuses EXCEPT for the founder bonus
    Each player will always get the bonuses of their Founder Belief

    Things that need to be re-done every turn (i.e., call to this fn):
    (1) religious_tenets_per_city
    (2) religion_building_yields
    (3) religion_yield_map
    """
    # First things first... zero-out the given fields...
    _religion_info = zero_out_fields_for_religion_update(game.player_cities.religion_info, TO_ZERO_OUT_FOR_RELIGION_STEP, player_id)
    game = game.replace(player_cities=game.player_cities.replace(religion_info=_religion_info))

    # Religion updates work slightly differently than building and social policy updates. With religion, we need to re-compute its 
    # contribution on a per-city basis, and other players' religions may effect any other players' cities.
    # bool for if the player has founded a relgiion yet
    has_founded_religion = game.religious_tenets[player_id[0]][MAX_IDX_PANTHEON:MAX_IDX_FOUNDER].sum() > 0

    # It could be the case that a given city does not have a majority religion (i.e., max < half pop)
    # (max_num_cities, 6) >= (max_num_cities,)[:, None]
    city_pop_halved = game.player_cities.population[player_id[0]] / 2
    has_maj_religion = game.player_cities.religion_info.religious_population[player_id[0]] > city_pop_halved[:, None]
    
    religion_info = _religion_per_city_vmap(
        jax.tree_map(lambda x: x[player_id[0]], game.player_cities.religion_info), 
        player_id,   # it may seem we do not need player_id due to arg 0 form. However, need for Founder tenet 
        maj_religion_idx_per_city, 
        game.religious_tenets,
        has_maj_religion,
        jnp.arange(0, game.player_cities.city_ids.shape[-1]),
        game)

    city_exists = game.player_cities.city_ids[player_id[0]] > 0
    
    # zero-out any contribution within cities that do not exist!
    religion_info = jax.tree_map(
        lambda x: (x * city_exists[(...,) + (None,) * (len(x.shape) - 1)]), religion_info
    )
    # Now everything has the leading (max_num_cities,...) axis
    new_religion_info = jax.tree_map(lambda x, y: x.at[player_id[0]].set(y), game.player_cities.religion_info, religion_info)
    new_religion_info = add_one_to_appropriate_fields(new_religion_info, TO_ZERO_OUT_FOR_RELIGION_STEP, player_id)
    game = game.replace(player_cities=game.player_cities.replace(religion_info=new_religion_info))
    
    """
    More one-offs...
    Players will always receive their founder bonus, so place these here.
    """
    # Messiah: great prophet earned with 25% fewer faith
    not_got_mes_b4 = game.prophet_threshold_from_messiah[player_id[0]] == 0
    has_messiah = game.religious_tenets[player_id[0], ReligiousTenets["messiah"]._value_] == 1
    give_mes_bonus = not_got_mes_b4 & has_messiah
    new_accel = (
        give_mes_bonus * (game.prophet_threshold_accel[player_id[0]] - 0.25)
        + (1 - give_mes_bonus) * game.prophet_threshold_accel[player_id[0]]
    )

    game = game.replace(
        prophet_threshold_from_messiah=game.prophet_threshold_from_messiah.at[player_id[0]].set(has_messiah),
        prophet_threshold_accel=game.prophet_threshold_accel.at[player_id[0]].set(new_accel)
    )

    def autopurchase_buildings(game, tenet_name):
        """
        This function will auto-purchase religious buildings (e.g., pagodas, catherals) IFF the given player_id
        has an enhancer (otherwise the auto-purchase will significantly slow down getting to enhancer level),
        has enough faith in reserves, and has the appropriate tenet type.
        
        This function will only purchase one buildings per turn per call, and the building will be placed in the 
        first available city slot.
        """
        has_enhancer = game.religious_tenets[player_id[0]].sum() > 4
        has_tenet = game.religious_tenets[player_id[0], ReligiousTenets[tenet_name]._value_] == 1
        has_enough_faith = game.faith_reserves[player_id[0]] >= (125 * game.culture_info.faith_purchase_accel[player_id[0]])

        if tenet_name == "work_ethic":
            has_tech = game.technologies[player_id[0], GameBuildings["factory"].prereq_tech] == 1
            to_construct = has_enhancer & has_tenet & has_enough_faith & has_tech
        else:
            to_construct = has_enhancer & has_tenet & has_enough_faith
        
        # Checking in each of player_id's cities to see if they have the building
        # Then, selecting the first city in the list (that is a city) that does 
        # not have the building
        bldg_idx = jnp.array(TENET_TO_BUILDING[tenet_name])

        building_not_in_city = jnp.any(
            game.player_cities.buildings_owned[player_id[0], :, bldg_idx] == 0, 
            axis=0  # along the building dimension
        )

        is_city = game.player_cities.city_ids[player_id[0]] > 0

        # First city that does not have the building
        city_idx = (building_not_in_city & is_city).argmax()

        # Does the building already exist in all cities?
        already_full = jnp.all(~building_not_in_city == is_city)
        to_construct = to_construct & ~already_full

        with_bldg_set = game.player_cities.buildings_owned.at[jnp.index_exp[
            player_id[0], city_idx, bldg_idx
        ]].set(1)

        with_faith_reserves = game.faith_reserves.at[player_id[0]].add(-125)
        
        new_bldgs = (
            to_construct * with_bldg_set
            + (1 - to_construct) * game.player_cities.buildings_owned 
        )
        new_faith_reserves = (
            to_construct * with_faith_reserves
            + (1 - to_construct) * game.faith_reserves
        )
        game = game.replace(
            faith_reserves=new_faith_reserves,
            player_cities=game.player_cities.replace(
                buildings_owned=new_bldgs
            )
        )
        return game
    
    game = autopurchase_buildings(game, "cathedrals")
    game = autopurchase_buildings(game, "gurdwaras")
    game = autopurchase_buildings(game, "mandirs")
    game = autopurchase_buildings(game, "mosques")
    game = autopurchase_buildings(game, "pagodas")
    game = autopurchase_buildings(game, "viharas")
    game = autopurchase_buildings(game, "houses_of_worship")
    game = autopurchase_buildings(game, "work_ethic")

    # Religious troubadours: +1 trade route
    not_got_trade_b4 = game.trade_route_from_troub[player_id[0]] == 0
    have_troub = game.religious_tenets[player_id[0], ReligiousTenets["religious_troubadours"]._value_] == 1 
    give_tr = not_got_trade_b4 & have_troub
    game = game.replace(
        num_trade_routes=game.num_trade_routes.at[player_id].add(give_tr),
        trade_route_from_troub=game.trade_route_from_troub.at[player_id[0]].set(have_troub)
    )

    # City of God: free gp
    not_got_gp_b4 = game.free_great_prophet_from_cog[player_id[0]] == 0
    have_cog = game.religious_tenets[player_id[0],  ReligiousTenets["city_of_god"]._value_] == 1
    give_gp = not_got_gp_b4 & have_cog

    game = game.replace(
        free_great_prophet_from_cog=game.free_great_prophet_from_cog.at[player_id[0]].set(have_cog)
    )

    # Now we can apply religious pressure
    # Need distance between each of player_id's cities to every other city in the game, including player_id's
    # own  cities
    # (6, max_num_cities, 2) and (6, max_num_cities) and (max_num_cities,)
    all_cities = game.player_cities.city_rowcols
    all_cs = game.cs_cities.city_rowcols
    cities_mask = game.player_cities.city_ids > 0
    cities_mask_cs = game.player_cities.city_ids[player_id[0]] > 0
    player_capital = game.player_cities.city_ids[player_id[0]] == 1
    players_cities = all_cities[player_id[0]]
    
    # Mask is
    def oddr_to_cube_single(rc):                      # (..., 2)
        r, q = rc[..., 0], rc[..., 1]
        x = q - ((r - (r & 1)) // 2)               # handle odd-row half-shift
        z = r
        y = -x - z
        return jnp.stack((x, y, z), axis=-1)       # (..., 3)

    def compute_dist_from_a_to_b_as_crow_flies_single(src_rc, tgt_rc):
        """
        src_rc : (N, 2)  odd-r coordinates
        tgt_rc : (M, 2)
        returns: (N, M)  Chebyshev (hex) distance between every pair
        """
        src = oddr_to_cube_single(src_rc)             # (N, 3)
        tgt = oddr_to_cube_single(tgt_rc)             # (M, 3)

        # Broadcasting:  src → (N, 1, 3);  tgt → (1, M, 3)  ⇒ diff (N, M, 3)
        diff = src[:, None, :] - tgt[None, :, :]

        # Hex distance in cube space = max-norm over (x,y,z)
        return jnp.max(jnp.abs(diff), axis=-1)     # (N, M)
    
    def oddr_to_cube_vec(rc): 
        r, q = rc[..., 0], rc[..., 1]
        x = q - ((r - (r & 1)) // 2)
        z = r
        y = -x - z
        return jnp.stack((x, y, z), axis=-1)    # (..., 3)

    def compute_dist_from_a_to_b_as_crow_flies_vec(src_rc, tgt_rc):          # (N,2) & (6,N,2)
        src = oddr_to_cube_vec(src_rc)                  # (N, 3)
        tgt = oddr_to_cube_vec(tgt_rc)                  # (6, N, 3)

        # Broadcast:  src -> (N,1,1,3),  tgt -> (1,6,N,3) -> diff (N,6,N,3)
        diff = src[:, None, None, :] - tgt[None, :, :, :]

        # Chebyshev norm in cube space gives hex distance
        dist = jnp.max(jnp.abs(diff), axis=-1)      # (N, 6, N)
        return dist         
    
    # Caravans
    # format of cumsum: from-city to-city
    # so we need all of player_id's caravans
    # all (max_num_units, )
    trade_pressure_sent = game.units.trade_yields[player_id[0], :, 1, -1]
    to_player_sent = game.units.trade_to_player_int[player_id[0]]
    to_player_sent_bool = game.units.trade_to_player_int[player_id[0]] < 6
    to_cs_sent = ~to_player_sent_bool
    to_city_sent = game.units.trade_to_city_int[player_id[0]]
    from_city_sent = game.units.trade_from_city_int[player_id[0]]


    ### PLAYER CITY PRESSURE ###
    # This workes by accumulating pressure over turns from each religion in the game.
    # Giving +6 dist (* modifier for capital holy city)
    # (max_num_cities, 6, max_num_cities) in from-to format
    distances = compute_dist_from_a_to_b_as_crow_flies_vec(players_cities, all_cities)
    distances = distances <= RELIGIOUS_PRESSURE_DIST_MAX

    pressure_mod_cap = (RELIGIOUS_PRESSURE_BASES * (HOLY_CITY_PRESSURE_ACCEL * player_capital)[: None, None])[..., None]

    distances_cap = distances * pressure_mod_cap
    distances_non = distances * RELIGIOUS_PRESSURE_BASES
    
    distances = distances_cap + distances_non
    distances = distances * cities_mask[None] * game.player_cities.religion_info.player_perturn_influence_accel[player_id[0]]  #/ RELIGIOUS_PERSSURE_THRESHOLD

    # (5, 6, 5) in (from, to, to) format
    distances = distances * has_founded_religion

    # the grand temple 2x religious pressure from a given city
    # We do +1, as False->1, True->2
    grand_temple_bonus = (game.player_cities.buildings_owned[player_id[0], :, GameBuildings["grand_temple"]._value_] == 1) + 1
    distances = distances * grand_temple_bonus[:, None, None]
    
    # We need to divide by 6 to account fo rthe summation over the axes
    new_cumsum = (game.player_cities.religion_info.player_perturn_influence_cumulative[player_id[0]]) + distances
    
    #with_trade = new_cumsum.sum(0).at[jnp.index_exp[to_player_sent, to_city_sent]].add(trade_pressure_sent * to_player_sent_bool)

    new_cumsum = new_cumsum.at[jnp.index_exp[
        from_city_sent, to_player_sent, to_city_sent    
    ]].add(trade_pressure_sent * to_player_sent_bool)

    #/ RELIGIOUS_PRESSURE_THRESHOLD
    # Need to now look at a per-city perspective...
    # We ultimately want to exert all of player_id's religious pressure across all other cities in the game
    # (5, 6, 5) => (6, 5)
    did_grow = (new_cumsum.sum(0) / RELIGIOUS_PRESSURE_THRESHOLD) >= 1

    # (6, 5, 6)
    new_religious_pop = game.player_cities.religion_info.religious_population.at[jnp.index_exp[:, :, player_id[0]]].add(did_grow)
    total_city_pop = game.player_cities.population  # Shape: (6, max_num_cities)

    # Check which cities exceed their population cap
    total_religious_after = new_religious_pop.sum(-1)  # (6, max_num_cities)
    exceeded_cap = total_religious_after > total_city_pop

    # For cities that exceed cap, find which religion to subtract from
    # First, check if any other religions have population > 0
    other_religions_mask = jnp.ones(6, dtype=bool).at[player_id[0]].set(False)
    other_religions_pop = new_religious_pop * other_religions_mask[None, None, :]  # Zero out player_id's religion

    # Check if there are any other religions with population > 0
    has_other_religions = (other_religions_pop > 0).any(-1)  # (6, max_num_cities)

    # Case 1: Other religions exist - subtract from the largest one
    other_religions_argmax = other_religions_pop.argmax(-1)  # (6, max_num_cities)

    # Case 2: Only player_id's religion exists - subtract from player_id's religion
    max_num_cities = game.player_cities.city_ids.shape[-1]
    player_religion_idx = jnp.full((6, max_num_cities), player_id[0])

    # Choose which religion to subtract from based on whether other religions exist
    religion_to_subtract = jnp.where(
        has_other_religions,
        other_religions_argmax,
        player_religion_idx
    )

    # Apply the subtraction only where we exceeded cap
    subtract_amount = exceeded_cap.astype(int)

    new_religious_pop = new_religious_pop.at[jnp.index_exp[
        jnp.arange(6)[:, None], 
        jnp.arange(max_num_cities)[None, :], 
        religion_to_subtract
    ]].add(-subtract_amount)

    # Zero out pressure for cities that grew
    # did_grow is (6, max_num_cities) - which cities received conversions
    # We need to zero out the cumulative pressure for those cities
    # new_cumsum is (max_num_cities, 6, max_num_cities) - pressure sent from player_id's cities
    new_cumsum = new_cumsum * (~did_grow)[None, :, :]  # Broadcasting: (1, 6, max_num_cities)
    
    game = game.replace(player_cities=game.player_cities.replace(religion_info=game.player_cities.religion_info.replace(
        player_perturn_influence_cumulative=game.player_cities.religion_info.player_perturn_influence_cumulative.at[player_id[0]].set(new_cumsum),
        religious_population=new_religious_pop
    )))
    
    ### CS PRESSURE ###
    # For CS, let's defer the pop conversion to the .step_cs() method!
    distances = compute_dist_from_a_to_b_as_crow_flies_single(players_cities, all_cs[:, 0])
    distances = distances <= RELIGIOUS_PRESSURE_DIST_MAX
    pressure_mod_cap = (RELIGIOUS_PRESSURE_BASES * (HOLY_CITY_PRESSURE_ACCEL * player_capital)[: None, None])


    distances_cap = distances * pressure_mod_cap
    distances_non = distances * RELIGIOUS_PRESSURE_BASES
    distances = distances_cap + distances_non
    distances = distances * cities_mask_cs[:, None] * game.player_cities.religion_info.cs_perturn_influence_accel[player_id[0]] # / RELIGIOUS_PERSSURE_THRESHOLD

    # (5, 12)
    distances = distances * has_founded_religion * grand_temple_bonus[:, None]

    # We need to divide by 6 to account fo rthe summation over the axes
    new_cumsum = (game.player_cities.religion_info.cs_perturn_influence_cumulative[player_id[0]]) + distances
    
    # Shapes here are slightly different... they are (max_num_cities, 12) whereas player-to-player is (max_num_cities, 6, max_num_cities)
    # Here we want to_player_sent for the 2nd idx,  as to_city_sent is always 0 for cs trade routes. We need to - 6 from this value
    # to get the correct idx.
    new_cumsum = new_cumsum.at[jnp.index_exp[
        from_city_sent, to_player_sent - 6   
    ]].add(trade_pressure_sent * to_cs_sent)
    
    game = game.replace(player_cities=game.player_cities.replace(religion_info=game.player_cities.religion_info.replace(
        cs_perturn_influence_cumulative=game.player_cities.religion_info.cs_perturn_influence_cumulative.at[player_id[0]].set(new_cumsum)
    )))
     
    return game
