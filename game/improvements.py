import enum
import jax.numpy as jnp

from game.constants import DESERT_IDX, FLATLAND_IDX, GRASSLAND_IDX, HILLS_IDX, PLAINS_IDX, TUNDRA_IDX
from game.resources import RESOURCE_TO_IDX
from game.techs import Technologies


class Improvements(enum.IntEnum):
    """
    name: id, tech prereq, terrain prereq, res prereq, must be on resources

    For oil well, let's just use "mine" on top of oil :)
    """
    farm = 0, Technologies.agriculture, [PLAINS_IDX, GRASSLAND_IDX, DESERT_IDX, TUNDRA_IDX], [], False
    pasture = 1, Technologies.animal_husbandry, [], [RESOURCE_TO_IDX["sheep"], RESOURCE_TO_IDX["cow"], RESOURCE_TO_IDX["horses"]], True
    mine = 2, Technologies.mining, [], [], False
    fishing_boat = 3, Technologies.sailing, [], [], True
    plantation = 4, Technologies.calendar, [], [], True
    camp = 5, Technologies.trapping, [], [], True
    quarry = 6, Technologies.masonry, [], [], True
    lumber_mill = 7, Technologies.construction, [], [], False
    fort = 8, Technologies.engineering, [], [], False
    trading_post = 9, Technologies.guilds, [], [], False
    road = 10, Technologies.the_wheel, [], [], False
    chop_forest =  11, Technologies.mining, [], [], False
    chop_jungle = 12, Technologies.bronze_working, [], [], False
    clear_marsh = 13, Technologies.masonry, [], [], False

    def __new__(cls, value: int, tech_prereq, terrain_prereq, res_prereq, requires_res):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.tech_prereq = tech_prereq
        obj.terrain_prereq = terrain_prereq
        obj.res_prereq = res_prereq
        obj.requires_res = requires_res
        return obj

ALL_IMPROVEMENT_TECHS = jnp.array([x.tech_prereq._value_ for x in Improvements], dtype=jnp.int32)

"""
All of these functions assume that we have already done the filtering for whether or not the given improvement type
can be built/performed on the given tile. Also, we assume that the worker has already been held in place for the correct
number of turns.

Here we need to alter two things:
    (1) The yield map
    (2) The feature map

For (1), since we use the "yield_map_players" as the base map upon which to build values (and this map changes per tech), we must access another
map entirely. This mad will be like the additional_yield_map attributes in other dataclasses. This will hold +/- for each yield type that 
tiles can have. Only one improvement can exist on a tile at a time, so there is no need to have a player-id indexed yield map.
For (2), we can directly modify the feature_map attribute, as this isn't reset or built upon anywhere.
"""
def add_yield_to_tile(additional_yield_map, rowcol, to_add):
    """This should be .set() instead of .add(). As no tile can have more than one type of improvement on it
        at a time, using .set() ensures we never stack yields, even if there is a bug somewhere else.
    """
    new_yields = additional_yield_map.at[rowcol[0], rowcol[1]].set(to_add)
    return new_yields

def add_improvement(improvement_map, rowcol, improvement):
    new_map = improvement_map.at[jnp.index_exp[rowcol[0], rowcol[1]]].set(Improvements[improvement]._value_ + 1)
    return new_map

def _farm(additional_yield_map, feature_map, improvement_map, resources_visible_to_player, elevation_map, terrain_map, road_map, rowcol):
    """
    +1 food everywhere
    """
    new_yields = add_yield_to_tile(additional_yield_map, rowcol, jnp.array([1, 0, 0, 0, 0, 0, 0]))
    new_map = add_improvement(improvement_map, rowcol, "farm")
    return new_yields, feature_map, new_map, road_map

def _pasture(additional_yield_map, feature_map, improvement_map, resources_visible_to_player, elevation_map, terrain_map, road_map, rowcol):
    """
    +1 prod on horse, cow
    +1 food on sheep
    """ 
    is_sheep = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["sheep"]
    is_horse = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["horses"]
    is_cow = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["cow"]
    is_horse_or_cow = is_horse | is_cow

    to_add = (
        is_sheep * jnp.array([1, 0, 0, 0, 0, 0, 0])
        + is_horse_or_cow * jnp.array([0, 1, 0, 0, 0, 0, 0])
    )

    new_yields = add_yield_to_tile(additional_yield_map, rowcol, to_add)
    new_map = add_improvement(improvement_map, rowcol, "pasture")
    return new_yields, feature_map, new_map, road_map

def _mine(additional_yield_map, feature_map, improvement_map, resources_visible_to_player, elevation_map, terrain_map, road_map, rowcol):
    """

    +1 gold, prod: lapis (hill and flat), jade, gems, jewelry, amber
    +1 prod: coal, aluminium, uranium, oil, iron, silver, gold, glass, copper
    +1 food, prod: salt
    """
    iron = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["iron"]
    coal = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["coal"]
    aluminium = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["aluminium"]
    uranium = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["uranium"]
    gold = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["gold"]
    silver = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["silver"]
    copper = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["copper"]
    gems = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["gems"]
    salt = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["salt"]
    oil = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["oil"]
    lapis = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["lapis"]
    jewelry = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["jewelry"]
    glass = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["glass"]
    amber = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["amber"]
    jade = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["jade"]

    goldprod = lapis | jade | gems | jewelry | amber
    prod = coal | aluminium | uranium | oil | iron | silver | gold  | glass | copper
    foodprod = salt
    base = ~goldprod & ~prod & ~foodprod

    to_add = (
        goldprod * jnp.array([0, 1, 1, 0, 0, 0, 0])
        + prod * jnp.array([0, 1, 0, 0, 0, 0, 0])
        + foodprod * jnp.array([1, 1, 0, 0, 0, 0, 0])
        + base * jnp.array([0, 1, 0, 0, 0, 0, 0])
    )

    new_yields = add_yield_to_tile(additional_yield_map, rowcol, to_add)
    new_map = add_improvement(improvement_map, rowcol, "mine")
    return new_yields, feature_map, new_map, road_map

def _fishing_boat(additional_yield_map, feature_map, improvement_map, resources_visible_to_player, elevation_map, terrain_map, road_map, rowcol):
    """
    +1 prod, gold: whales
    +1 food: fish
    +1 food, gold: crabs
    +1 prod, gold, culture: pearls
    +1 food, gold, faith: coral
    """
    whales = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["whales"]
    fish = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["fish"]
    crabs = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["crabs"]
    pearls = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["pearls"]
    coral = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["coral"]
    
    to_add = (
        whales * jnp.array([0, 1, 1, 0, 0, 0, 0])
        + fish * jnp.array([1, 0, 0, 0, 0, 0, 0])
        + crabs * jnp.array([1, 0, 1, 0, 0, 0, 0])
        + pearls * jnp.array([0, 1, 1, 0, 0, 0, 0])
        + coral * jnp.array([1, 0, 1, 1, 0, 0, 0])
    )

    new_yields = add_yield_to_tile(additional_yield_map, rowcol, to_add)
    new_map = add_improvement(improvement_map, rowcol, "fishing_boat")
    return new_yields, feature_map, new_map, road_map

def _plantation(additional_yield_map, feature_map, improvement_map, resources_visible_to_player, elevation_map, terrain_map, road_map, rowcol):
    """
    +1 prod, +2 gold: coffee
    +1 food, gold: olives
    +2 culture: perfume
    +1 prod, +1 gold: rubber, cotton
    +2 gold: tea, sugar
    +2 faith: tobacco, incense
    +1 food: bananas, citrus, chocolate, wine, (coconut)
    +1 culture: dyes, silk
    +1 food, +2 gold: spices
    """
    dyes = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["dyes"]
    wine = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["wine"]
    coconut = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["coconut"]
    tobacco = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["tobacco"]
    olives = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["olives"]
    sugar = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["sugar"]
    citrus = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["citrus"]
    cotton = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["cotton"]
    incense = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["incense"]
    coffee = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["coffee"]
    silk = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["silk"]
    perfume = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["perfume"]
    spices = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["spices"]
    chocolate = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["chocolate"]
    rubber = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["rubber"]
    tea = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["tea"]
    bananas = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["banana"]

    prod1gold2 = coffee
    food1gold1 = olives
    culture2 = perfume
    prod1gold1 = rubber | cotton
    gold2 = tea | sugar
    faith2 = tobacco | incense
    food1 = bananas | citrus | chocolate | wine | coconut
    culture1 = dyes | silk
    food1gold2 = spices

    to_add = (
        prod1gold2 * jnp.array([0, 1, 2, 0, 0, 0, 0])
        + food1gold1 * jnp.array([1, 0, 1, 0, 0, 0, 0])
        + culture2 * jnp.array([0, 0, 0, 0, 2, 0, 0])
        + prod1gold1 * jnp.array([0, 1, 1, 0, 0, 0, 0])
        + gold2 * jnp.array([0, 0, 2, 0, 0, 0, 0])
        + faith2 * jnp.array([0, 0, 0, 2, 0, 0, 0])
        + food1 * jnp.array([1, 0, 0, 0, 0, 0, 0])
        +  culture1 * jnp.array([0, 0, 0, 0, 1, 0, 0])
        + food1gold2 * jnp.array([1, 0, 2, 0, 0, 0, 0])
    )
    
    new_yields = add_yield_to_tile(additional_yield_map, rowcol, to_add)
    new_map = add_improvement(improvement_map, rowcol, "plantation")
    return new_yields, feature_map, new_map, road_map


def _camp(additional_yield_map, feature_map, improvement_map, resources_visible_to_player, elevation_map, terrain_map, road_map, rowcol):
    """
    +1 food: furs, truffles
    +1 food,  gold: ivory
    +1 prod: deer, bison
    """
    furs = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["furs"]
    truffles = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["truffles"]
    ivory = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["ivory"]
    deer = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["deer"]
    bison = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["bison"]
    
    food1 = furs | truffles
    food1gold1 = ivory
    prod1 = deer | bison
    to_add = (
        food1 * jnp.array([1, 0, 0, 0, 0, 0, 0])
        + food1gold1 * jnp.array([1, 0, 1, 0, 0, 0, 0])
        + prod1 * jnp.array([0, 1, 0, 0, 0, 0, 0])
    )

    new_yields = add_yield_to_tile(additional_yield_map, rowcol, to_add)
    new_map = add_improvement(improvement_map, rowcol, "camp")
    return new_yields, feature_map, new_map, road_map

def _quarry(additional_yield_map, feature_map, improvement_map, resources_visible_to_player, elevation_map, terrain_map, road_map, rowcol):
    """
    +1 prod: obsidian, stone
    +1 prod, gold: marble,  porcelain
    """
    obsidian = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["obsidian"]
    stone = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["stone"]
    marble = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["marble"]
    porcelain = resources_visible_to_player[rowcol[0], rowcol[1]] == RESOURCE_TO_IDX["porcelain"]

    prod1 = obsidian | stone
    prod1gold1 = marble | porcelain

    to_add = (
        prod1 * jnp.array([0, 1, 0, 0, 0, 0, 0])
        + prod1gold1 * jnp.array([0, 1, 1, 0, 0, 0, 0])
    )

    new_yields = add_yield_to_tile(additional_yield_map, rowcol, to_add)
    new_map = add_improvement(improvement_map, rowcol, "quarry")
    return new_yields, feature_map, new_map, road_map

def _lumber_mill(additional_yield_map, feature_map, improvement_map, resources_visible_to_player, elevation_map, terrain_map, road_map, rowcol):
    """
    +1 prod
    """
    to_add = jnp.array([0, 1, 0, 0, 0, 0, 0])
    new_yields = add_yield_to_tile(additional_yield_map, rowcol, to_add)
    new_map = add_improvement(improvement_map, rowcol, "lumber_mill")
    return new_yields, feature_map, new_map, road_map

def _fort(additional_yield_map, feature_map, improvement_map, resources_visible_to_player, elevation_map, terrain_map, road_map, rowcol):
    """
    Gives no additional yields, defensive bonus deferred to unit combat calcs
    """
    new_map = add_improvement(improvement_map, rowcol, "fort")
    return additional_yield_map, feature_map, new_map, road_map

def _trading_post(additional_yield_map, feature_map, improvement_map, resources_visible_to_player, elevation_map, terrain_map, road_map, rowcol):
    """
    +1 gold ?
    """
    to_add = jnp.array([0, 0, 1, 0, 0, 0, 0])
    new_yields = add_yield_to_tile(additional_yield_map, rowcol, to_add)
    new_map = add_improvement(improvement_map, rowcol, "trading_post")
    return new_yields, feature_map, new_map, road_map

def _road(additional_yield_map, feature_map, improvement_map, resources_visible_to_player, elevation_map, terrain_map, road_map, rowcol, player_id):
    """
    Gives no additional yields. Instead adds a flag of "player_id" to the road map.
    Using player_id instead of True/False helps us keep track of costs without needing to 6x the size
    of the road_map array
    """
    new_road_map = road_map.at[rowcol[0], rowcol[1]].set(player_id[0] + 1)
    return additional_yield_map, feature_map, improvement_map, new_road_map

def _chop_forest(additional_yield_map, feature_map, improvement_map, resources_visible_to_player, elevation_map, terrain_map, road_map, rowcol):
    """
    Removing forest causes the tile to go back to its base yields.
    Because we are using the additional_yield_map as an addition to the base yields, we need to do something interesting...

    - flatland plains: no change
    - flatland grass: +1 food, -1 prod
    - any hill: -1 food, +1 prod
    """
    flatland = elevation_map[rowcol[0], rowcol[1]] == FLATLAND_IDX
    hill = elevation_map[rowcol[0], rowcol[1]] == HILLS_IDX
    plains = terrain_map[rowcol[0], rowcol[1]] == PLAINS_IDX
    grass = terrain_map[rowcol[0], rowcol[1]] == GRASSLAND_IDX

    flat_plain = flatland & plains
    flat_grass = flatland & grass
    
    to_add = (
        flat_plain * jnp.array([0, 0, 0, 0, 0, 0, 0])
        + flat_grass * jnp.array([1, -1, 0, 0, 0, 0, 0])
        + hill * jnp.array([-1, 1, 0, 0, 0, 0, 0])
    )
    
    new_yields = add_yield_to_tile(additional_yield_map, rowcol, to_add)
    feature_map = feature_map.at[rowcol[0], rowcol[1]].set(0)
    return new_yields, feature_map, improvement_map, road_map


def _chop_jungle(additional_yield_map, feature_map, improvement_map, resources_visible_to_player, elevation_map, terrain_map, road_map, rowcol):
    """
    - flatland plains: -1 food
    - hill plains: -2 food +2 prod
    """
    flatland = elevation_map[rowcol[0], rowcol[1]] == FLATLAND_IDX
    hill = elevation_map[rowcol[0], rowcol[1]] == HILLS_IDX
    plains = terrain_map[rowcol[0], rowcol[1]] == PLAINS_IDX

    flat_plain = flatland & plains
    
    to_add = (
        flat_plain * jnp.array([-1, 0, 0, 0, 0, 0, 0])
        + hill * jnp.array([-2, 2, 0, 0, 0, 0, 0])
    )
    
    new_yields = add_yield_to_tile(additional_yield_map, rowcol, to_add)
    feature_map = feature_map.at[rowcol[0], rowcol[1]].set(0)
    return new_yields, feature_map, improvement_map, road_map


def _clear_marsh(additional_yield_map, feature_map, improvement_map, resources_visible_to_player, elevation_map, terrain_map, road_map, rowcol):
    """
    +1 food
    Gives no additional yields
    """
    to_add = jnp.array([1, 0, 0, 0, 0, 0, 0])
    new_yields = add_yield_to_tile(additional_yield_map, rowcol, to_add)
    feature_map = feature_map.at[rowcol[0], rowcol[1]].set(0)
    return new_yields, feature_map, improvement_map, road_map
