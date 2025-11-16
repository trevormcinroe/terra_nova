"""
Many things in this file are deprecated. E.g., instead of explicitly unlocking buildings/units, we defer
this check to building/unit prereq checks, and mask out unfulfilled options at the time of action selection.
However, we'll keep the file as is for now.
"""

import sys
from typing import List, Tuple
import jax
import jax.numpy as jnp
from enum import IntEnum
from functools import partial


class Technologies(IntEnum):
    """id, cost, era, prereq, [y, x]"""
    agriculture = 0, 20, "ancient", [], [5, 0]
    pottery = 1, 35, "ancient", [0], [1, 1]
    animal_husbandry = 2, 35, "ancient", [0], [4, 1]
    archery = 3, 35, "ancient", [0], [6, 1]
    mining = 4, 35, "ancient", [0], [8, 1]
    sailing = 5, 55, "ancient", [1], [0, 2]
    calendar = 6, 55, "ancient", [1], [1, 2]
    writing = 7, 55, "ancient", [1], [2, 2]
    trapping = 8, 55, "ancient", [2], [4, 2]
    the_wheel = 9, 55, "ancient", [2, 3], [6, 2]
    masonry = 10, 55, "ancient", [4], [8, 2]
    bronze_working = 11, 55, "ancient", [4], [9, 2]
    optics = 12, 68, "classical", [5], [0, 3]
    horseback_riding = 13, 105, "classical", [9, 8], [4, 3]
    mathematics = 14, 105, "classical", [9], [6, 3]
    construction = 15, 105, "classical", [10, 9], [8, 3]
    philosophy = 16, 175, "classical", [6, 7], [1, 4]
    drama = 17, 175, "classical", [7], [2, 4]
    currency = 18, 175, "classical", [14], [6, 4]
    engineering = 19, 175, "classical", [14, 15], [7, 4]
    iron_working = 20, 175, "classical", [11], [9, 4]
    theology = 21, 275, "medieval", [16, 17], [1, 5]
    civil_service = 22, 275, "medieval", [17, 13, 18], [4, 5]
    guilds = 23, 275, "medieval", [18], [6, 5]
    metal_casting = 24, 275, "medieval", [19, 20], [8, 5]
    compass = 25, 375, "medieval", [21, 12], [0, 6]
    education = 26, 485, "medieval", [21, 22], [2, 6]
    chivalry = 27, 485, "medieval", [22, 23], [5, 6]
    machinery = 28, 485, "medieval", [19, 23], [7, 6]
    physics = 29, 485, "medieval", [24], [8, 6]
    steel = 30, 485, "medieval", [24], [9, 6]
    astronomy = 31, 780, "renaissance", [25, 26], [1, 7]
    acoustics = 32, 780, "renaissance", [26], [3, 7]
    banking = 33, 780, "renaissance", [26, 27], [5, 7]
    printing_press = 34, 780, "renaissance", [28, 29, 27], [7, 7]
    gunpowder = 35, 780, "renaissance", [29, 30], [9, 7]
    navigation = 36, 1150, "renaissance", [31], [1 ,8]
    architecture = 37, 1150, "renaissance", [32, 33], [3, 8]
    economics = 38, 1150, "renaissance", [33, 34], [5, 8]
    metallurgy = 39, 1150, "renaissance", [35, 34], [7, 8]
    chemistry = 40, 1150, "renaissance", [35], [9, 8]
    archaeology = 41, 1600, "industrial", [36], [1, 9]
    scientific_theory = 42, 1600, "industrial", [38, 37], [3, 9]
    industrialization = 43, 1600, "industrial", [38], [5, 9]
    rifling = 44, 1600, "industrial", [39], [6, 9]
    military_science = 45, 1600, "industrial", [40, 39], [8, 9]
    fertilizer = 46, 1600, "industrial", [40], [9, 9]
    biology = 47, 2350, "industrial", [41, 42], [1, 10]
    electricity = 48, 2350, "industrial", [42], [2, 10]
    steam_power = 49, 2350, "industrial", [44, 43], [5, 10]
    dynamite = 50, 2350, "industrial", [46, 45], [8, 10]
    refrigeration = 51, 3100, "modern", [48, 47], [1, 11]
    radio = 52, 3100, "modern", [48], [2, 11]
    replaceable_parts = 53, 3100, "modern", [49, 48], [3, 11]
    flight = 54, 3100, "modern", [49], [5, 11]
    railroad = 55, 3100, "modern", [49, 50], [7, 11]
    plastic = 56, 4100, "modern", [52, 53], [2, 12]
    electronics = 57, 4100, "modern", [53, 54], [4, 12]
    ballistics = 58, 4100, "modern", [52, 54], [5, 12]
    combustion = 59, 4100, "modern", [55], [7, 12]
    penicilin = 60, 5100, "postmodern", [51, 56], [1, 13]
    atomic_theory = 61, 5100, "postmodern", [56, 57], [3, 13]
    radar = 62, 5100, "postmodern", [57, 58], [5, 13]
    combined_arms = 63, 5100, "postmodern", [58, 59], [7, 13]
    ecology = 64, 6400, "postmodern", [61, 60], [1, 14]
    nuclear_fission = 65, 6400, "postmodern", [61, 62], [3, 14]
    rocketry = 66, 6400, "postmodern", [62], [5, 14]
    computers = 67, 6400, "postmodern", [62, 63], [7, 14]
    telecom = 68, 8470, "future", [64], [1, 15]
    mobile_tactics = 69, 8470, "future", [64, 65], [2, 15]
    advanced_ballistics = 70, 8470, "future", [65, 66], [4, 15]
    satellites = 71, 8470, "future", [66], [5, 15]
    robotics = 72, 8470, "future", [66, 67], [6, 15]
    lasers = 73, 8470, "future", [67], [7, 15]
    internet = 74, 9680, "future", [68], [0, 16]
    globalization = 75, 9680, "future", [68], [1, 16]
    particle_physics = 76, 9680, "future", [68, 69, 70], [2, 16]
    nuclear_fusion = 77, 9680, "future", [70, 71, 72], [5, 16]
    nanotechnology = 78, 9680, "future", [72], [6, 16]
    stealth = 79, 9680, "future", [72, 73], [7, 16]
    future_tech = 80, 10500, "future", [74, 75, 76, 77, 78, 79], [4, 17]

    def __new__(cls, value: int, cost: int, era: str, prereq: List[int], grix_yx: List[int]):
        obj = int.__new__(cls, value)     # create the real `int`
        obj._value_ = value
        obj.cost = cost 
        obj.era = era
        obj.prereq = prereq
        obj.grid_yx = grix_yx
        return obj

ALL_TECH_COSTS = jnp.array([x.cost for x in Technologies])

era_to_integer = {
    "ancient": 0,
    "classical": 1,
    "medieval": 2,
    "renaissance": 3,
    "industrial": 4,
    "modern": 5,
    "postmodern": 6,
    "future": 7
}

TECH_TO_ERA_INT = jnp.array([era_to_integer[x.era] for x in Technologies])

"""All functions below are for when a technology finishes, written from the 
perspective of a single game instance
"""
def add_technology(game, player_id, technology):
    """"""
    techs = game.technologies.at[player_id[0], Technologies[technology]._value_].set(1)
    return game.replace(technologies=techs)


def unlock_building_in_all_cities(game, player_id, building_name):
    """
    We perform a lazy import here to avoid circular imports between this module and buildings
    """
    from game.buildings import GameBuildings

    new_cb = game.player_cities.can_build.at[jnp.index_exp[
        player_id[0], :, GameBuildings[building_name]._value_
    ]].set(1)
    return game.replace(player_cities=game.player_cities.replace(can_build=new_cb))


def add_trade_routes(game, player_id, num_routes):
    """"""
    new_tr = game.num_trade_routes[player_id[0]] + num_routes
    return game.replace(num_trade_routes=game.num_trade_routes.at[player_id[0]].set(new_tr))


def do_nothing(game, player_id):
    return game


def _agriculture(game, player_id):
    """
    This is already handed in game.action_space.do_settle()
    """
    return game


def _pottery(game, player_id):
    """
    unlocks granary and shrine in all cities
    """
    game = unlock_building_in_all_cities(game, player_id, "granary")
    game = unlock_building_in_all_cities(game, player_id, "shrine")
    game = add_technology(game, player_id, "pottery")
    return game

def _animal_husbandry(game, player_id):
    """
    build caravans
    reveals horses
    unlocks pastures improvement
    +1 trade routes
    """
    game = add_technology(game, player_id, "animal_husbandry")
    #game = unlock_building_in_all_cities(game, player_id, "caravans")
    game = add_trade_routes(game, player_id, 1)
    game = game.update_player_visible_resources_and_yields(player_id)

    return game

def _archery(game, player_id):
    """
    build archer
    build temple of artemis
    """
    game = add_technology(game, player_id, "archery")
    game = unlock_building_in_all_cities(game, player_id, "temple_artemis")
    return game

def _mining(game, player_id):
    """
    reveals iron
    unlocks mine improvement
    can chop forest
    """
    game = add_technology(game, player_id, "mining")
    game = game.update_player_visible_resources_and_yields(player_id)
    return game

def _sailing(game, player_id):
    """
    unlocks fishing boats improvement
    can build cargo shipts, trireme, 
    """
    game = add_technology(game, player_id, "sailing")
    return game

def _calendar(game, player_id):
    """
    can build stonehenge, stone works
    unlocks plantation improvement
    """
    game = add_technology(game, player_id, "calendar")
    game = unlock_building_in_all_cities(game, player_id, "stonehenge")
    return game

def _writing(game, player_id):
    """
    can build library, great library
    """
    game = add_technology(game, player_id, "writing")
    game = unlock_building_in_all_cities(game, player_id, "library")
    game = unlock_building_in_all_cities(game, player_id, "great_library")
    return game

def _trapping(game, player_id):
    """
    unlocks camp improvement
    can build circus
    """
    game = add_technology(game, player_id, "trapping")
    game = unlock_building_in_all_cities(game, player_id, "circus")
    return game

def _the_wheel(game, player_id):
    """
    can build chariot archer
    unlocks road improvement
    """
    game = add_technology(game, player_id, "the_wheel")
    return game

def _masonry(game, player_id):
    """
    can build walls, pyramids, masoleum of hal
    unlocks quarry improvement
    """
    game = add_technology(game, player_id, "masonry")
    game = unlock_building_in_all_cities(game, player_id, "walls")
    game = unlock_building_in_all_cities(game, player_id, "pyramid")
    game = unlock_building_in_all_cities(game, player_id, "mausoleum_halicarnassus")
    return game

def _bronze_working(game, player_id):
    """
    unlocks lumber mill improvement
    can chop jungle
    can build spearmen, barracks, statue of zeus
    """
    game = add_technology(game, player_id, "bronze_working")
    game = unlock_building_in_all_cities(game, player_id, "barracks")
    game = unlock_building_in_all_cities(game, player_id, "statue_zeus")
    return game

def _optics(game, player_id):
    """
    can build lighthouse, great lighthouse, galley
    """
    game = add_technology(game, player_id, "optics")
    game = unlock_building_in_all_cities(game, player_id, "lighthouse")
    game = unlock_building_in_all_cities(game, player_id, "great_lighthouse")
    return game

def _horseback_riding(game, player_id):
    """
    can build horseman, stable, circus maximus, caravansary
    """
    game = add_technology(game, player_id, "horseback_riding")
    game = unlock_building_in_all_cities(game, player_id, "stable")
    game = unlock_building_in_all_cities(game, player_id, "caravansary")
    game = unlock_building_in_all_cities(game, player_id, "circus_maximus")
    return game

def _mathematics(game, player_id):
    """
    can build catapult, courthouse, hanging gardens
    """
    game = add_technology(game, player_id, "mathematics")
    game = unlock_building_in_all_cities(game, player_id, "courthouse")
    game = unlock_building_in_all_cities(game, player_id, "hanging_garden")
    return game

def _construction(game, player_id):
    """
    can build comp bowman, colosseum, terracotta army
    """
    game = add_technology(game, player_id, "construction")
    game = unlock_building_in_all_cities(game, player_id, "colosseum")
    game = unlock_building_in_all_cities(game, player_id, "terracotta_army")
    return game

def _philosophy(game, player_id):
    """
    can build temple, national college, oracle, great mosque of djenne, 
    """
    game = add_technology(game, player_id, "philosophy")
    game = unlock_building_in_all_cities(game, player_id, "temple")
    game = unlock_building_in_all_cities(game, player_id, "national_college")
    game = unlock_building_in_all_cities(game, player_id, "oracle")
    game = unlock_building_in_all_cities(game, player_id, "mosque_of_djenne")
    return game

def _drama(game, player_id):
    """
    can build garden, national epic, amphitheater, writer's guild, parthenon
    """
    game = add_technology(game, player_id, "drama")
    game = unlock_building_in_all_cities(game, player_id, "garden")
    game = unlock_building_in_all_cities(game, player_id, "national_epic")
    game = unlock_building_in_all_cities(game, player_id, "amphitheater")
    game = unlock_building_in_all_cities(game, player_id, "writers_guild")
    game = unlock_building_in_all_cities(game, player_id, "parthenon")
    return game

def _currency(game, player_id):
    """
    can build mint, market, petra, 
    """
    game = add_technology(game, player_id, "currency")
    game = unlock_building_in_all_cities(game, player_id, "mint")
    game = unlock_building_in_all_cities(game, player_id, "market")
    game = unlock_building_in_all_cities(game, player_id, "petra")
    return game

def _engineering(game, player_id):
    """
    can build great wall, aqueduct
    unlocks fort improvement
    all roads over rivers now have bridges
    """
    game = add_technology(game, player_id, "engineering")
    game = unlock_building_in_all_cities(game, player_id, "great_wall")
    game = unlock_building_in_all_cities(game, player_id, "aqueduct")
    return game

def _iron_working(game, player_id):
    """
    can build swordsman, heroic epic, colossus
    """
    game = add_technology(game, player_id, "iron_working")
    game = unlock_building_in_all_cities(game, player_id, "heroic_epic")
    game = unlock_building_in_all_cities(game, player_id, "colossus")
    return game

def _theology(game, player_id):
    """
    can build hagia sophia, grand temple, borobudur, censer maker
    """
    game = add_technology(game, player_id, "theology")
    game = unlock_building_in_all_cities(game, player_id, "hagia_sophia")
    game = unlock_building_in_all_cities(game, player_id, "grand_temple")
    game = unlock_building_in_all_cities(game, player_id, "borobudur")
    game = unlock_building_in_all_cities(game, player_id, "censer")
    return game

def _civil_service(game, player_id):
    """
    +1 food on all farms next to rivers and lakes
    can build pikemen, chichen itza, lake wonder
    """
    game = add_technology(game, player_id, "civil_service")
    game = unlock_building_in_all_cities(game, player_id, "chichen_itza")
    game = unlock_building_in_all_cities(game, player_id, "lake_wonder")
    return game

def _guilds(game, player_id):
    """
    can build east india company, machu, artist's guild, brewery
    unlocks trading post improvement
    """
    game = add_technology(game, player_id, "guilds")
    game = unlock_building_in_all_cities(game, player_id, "national_treasury")
    game = unlock_building_in_all_cities(game, player_id, "machu_pichu")
    game = unlock_building_in_all_cities(game, player_id, "artists_guild")
    game = unlock_building_in_all_cities(game, player_id, "brewery")
    return game

def _metal_casting(game, player_id):
    """
    can build forge, workshop, althing
    """
    game = add_technology(game, player_id, "metal_casting")
    game = unlock_building_in_all_cities(game, player_id, "forge")
    game = unlock_building_in_all_cities(game, player_id, "workshop")
    game = unlock_building_in_all_cities(game, player_id, "althing")
    return game

def _compass(game, player_id):
    """
    can build galeass, harbor
    """
    game = add_technology(game, player_id, "compass")
    game = unlock_building_in_all_cities(game, player_id, "harbor")
    return game

def _education(game, player_id):
    """
    can build university, oxford, angkor wat
    """
    game = add_technology(game, player_id, "education")
    game = unlock_building_in_all_cities(game, player_id, "university")
    game = unlock_building_in_all_cities(game, player_id, "oxford_university")
    game = unlock_building_in_all_cities(game, player_id, "angkor_wat")
    return game

def _chivalry(game, player_id):
    """
    can build knight, castle, alhambra
    """
    game = add_technology(game, player_id, "chivalry")
    game = unlock_building_in_all_cities(game, player_id, "castle")
    game = unlock_building_in_all_cities(game, player_id, "alhambra")
    return game

def _machinery(game, player_id):
    """
    can build crossbowman, ironworks, gemcutter
    """
    game = add_technology(game, player_id, "machinery")
    game = unlock_building_in_all_cities(game, player_id, "ironworks")
    game = unlock_building_in_all_cities(game, player_id, "gemcutter")
    return game

def _physics(game, player_id):
    """
    can build trebuchet, notre dame
    """
    game = add_technology(game, player_id, "physics")
    game = unlock_building_in_all_cities(game, player_id, "notre_dame")
    return game

def _steel(game, player_id):
    """
    can build longswordsman, armory
    """
    game = add_technology(game, player_id, "steel")
    game = unlock_building_in_all_cities(game, player_id, "armory")
    return game

def _astronomy(game, player_id):
    """
    can build caravel, observatory
    """
    game = add_technology(game, player_id, "astronomy")
    game = unlock_building_in_all_cities(game, player_id, "observatory")
    return game

def _acoustics(game, player_id):
    """
    can build opera house, sistine, musician's guild
    """
    game = add_technology(game, player_id, "acoustics")
    game = unlock_building_in_all_cities(game, player_id, "opera_house")
    game = unlock_building_in_all_cities(game, player_id, "sistine_chapel")
    game = unlock_building_in_all_cities(game, player_id, "musicians_guild")
    return game

def _banking(game, player_id):
    """
    can build bank, constabulary, grocer
    """
    game = add_technology(game, player_id, "banking")
    game = unlock_building_in_all_cities(game, player_id, "bank")
    game = unlock_building_in_all_cities(game, player_id, "constable")
    game = unlock_building_in_all_cities(game, player_id, "grocer")
    game = add_trade_routes(game, player_id, 1)
    return game

def _printing_press(game, player_id):
    """
    can build zoo (called theatre), leaning tower, globe theater
    unlocks the world congress
    """
    game = add_technology(game, player_id, "printing_press")
    game = unlock_building_in_all_cities(game, player_id, "theatre")
    game = unlock_building_in_all_cities(game, player_id, "leaning_tower")
    game = unlock_building_in_all_cities(game, player_id, "globe_theater")
    return game

def _gunpowder(game, player_id):
    """
    can build musketmen, himeji
    """
    game = add_technology(game, player_id, "gunpowder")
    game = unlock_building_in_all_cities(game, player_id, "himeji_castle")
    return game

def _navigation(game, player_id):
    """
    can build frigate, privateer, seaport
    """
    game = add_technology(game, player_id, "navigation")
    game = unlock_building_in_all_cities(game, player_id, "seaport")
    return game

def _architecture(game, player_id):
    """
    can build hermitage, porcelain tower, taj mahal, uffizi
    """
    game = add_technology(game, player_id, "architecture")
    game = unlock_building_in_all_cities(game, player_id, "hermitage")
    game = unlock_building_in_all_cities(game, player_id, "porcelain_tower")
    game = unlock_building_in_all_cities(game, player_id, "taj_mahal")
    game = unlock_building_in_all_cities(game, player_id, "uffizi")
    return game

def _economics(game, player_id):
    """
    can build windmill, textile mill
    """
    game = add_technology(game, player_id, "economics")
    game = unlock_building_in_all_cities(game, player_id, "windmill")
    game = unlock_building_in_all_cities(game, player_id, "textile")
    return game

def _metallurgy(game, player_id):
    """
    can build lancer, arsenal, red fort
    """
    game = add_technology(game, player_id, "metallurgy")
    game = unlock_building_in_all_cities(game, player_id, "arsenal")
    game = unlock_building_in_all_cities(game, player_id, "red_fort")
    return game

def _chemistry(game, player_id):
    """
    can build cannon
    """
    game = add_technology(game, player_id, "chemistry")
    return game

def _archaeology(game, player_id):
    """
    can build museum, louvre, archeologist
    reveals antiquity sites
    unlocks arch dig
    """
    game = add_technology(game, player_id, "archaeology")
    game = unlock_building_in_all_cities(game, player_id, "museum")
    game = unlock_building_in_all_cities(game, player_id, "louvre")
    return game

def _scientific_theory(game, player_id):
    """
    can build public school
    """
    game = add_technology(game, player_id, "scientific_theory")
    game = unlock_building_in_all_cities(game, player_id, "public_school")
    return game

def _industrialization(game, player_id):
    """
    reveals coal
    can build gatling gun, factory, big ben, 
    """
    game = add_technology(game, player_id, "industrialization")
    game = unlock_building_in_all_cities(game, player_id, "factory")
    game = game.update_player_visible_resources_and_yields(player_id)
    return game

def _rifling(game, player_id):
    """
    can build rifleman, neu
    """
    game = add_technology(game, player_id, "rifling")
    game = unlock_building_in_all_cities(game, player_id, "neuschwanstein")
    return game

def _military_science(game, player_id):
    """
    can build cavalry, military academy, brandenburg gate
    """
    game = add_technology(game, player_id, "military_science")
    game = unlock_building_in_all_cities(game, player_id, "military_academy")
    game = unlock_building_in_all_cities(game, player_id, "brandenburg_gate")
    return game

def _fertilizer(game, player_id):
    """

    """
    game = add_technology(game, player_id, "fertilizer")
    return game

def _biology(game, player_id):
    """
    can build hospital, refinery
    reveals oil
    unlocks well improvement
    """
    game = add_technology(game, player_id, "biology")
    game = unlock_building_in_all_cities(game, player_id, "hospital")
    game = unlock_building_in_all_cities(game, player_id, "refinery")
    game = game.update_player_visible_resources_and_yields(player_id)
    return game

def _electricity(game, player_id):
    """
    reveals aluminium
    can build hydro plant, stock exchange, police station, 
    """
    game = add_technology(game, player_id, "electricity")
    game = unlock_building_in_all_cities(game, player_id, "hydro_plant")
    game = unlock_building_in_all_cities(game, player_id, "stock_exchange")
    game = unlock_building_in_all_cities(game, player_id, "police_station")
    game = game.update_player_visible_resources_and_yields(player_id)
    return game

def _steam_power(game, player_id):
    """
    can build ironclad, airship, defender
    """
    game = add_technology(game, player_id, "steam_power")
    #game = unlock_building_in_all_cities(game, player_id, "neuschwanstein")
    return game

def _dynamite(game, player_id):
    """
    can build artillery
    makes great wall movement penalty obsolete
    """
    game = add_technology(game, player_id, "dynamite")
    #game = unlock_building_in_all_cities(game, player_id, "neuschwanstein")
    return game


def _refrigeration(game, player_id):
    """
    can build submarine, stadium, hotel
    unlocks offshore oil well
    """
    game = add_technology(game, player_id, "refrigeration")
    game = unlock_building_in_all_cities(game, player_id, "stadium")
    game = unlock_building_in_all_cities(game, player_id, "hotel")
    return game

def _radio(game, player_id):
    """
    can build broadcast tower, eiffel tower, national intelligence agency, broadway
    """
    game = add_technology(game, player_id, "radio")
    game = unlock_building_in_all_cities(game, player_id, "broadcast_tower")
    game = unlock_building_in_all_cities(game, player_id, "eiffel_tower")
    game = unlock_building_in_all_cities(game, player_id, "intelligence_agency")
    game = unlock_building_in_all_cities(game, player_id, "broadway")
    return game


def _replaceable_parts(game, player_id):
    """
    can build great war infantry, military base, statue of liberty
    """
    game = add_technology(game, player_id, "replaceable_parts")
    game = unlock_building_in_all_cities(game, player_id, "military_base")
    game = unlock_building_in_all_cities(game, player_id, "statue_of_liberty")
    return game


def _flight(game, player_id):
    """
    can build triplane, great war bomber, prora
    """
    game = add_technology(game, player_id, "flight")
    game = unlock_building_in_all_cities(game, player_id, "prora_resort")
    return game

def _railroad(game, player_id):
    """
    can build kremlin, panama
    unlocks railroad improvement
    """
    game = add_technology(game, player_id, "railroad")
    game = unlock_building_in_all_cities(game, player_id, "kremlin")
    game = unlock_building_in_all_cities(game, player_id, "panama")
    game = add_trade_routes(game, player_id, 1)
    return game

def _plastic(game, player_id):
    """
    can build research lab, cristo
    """
    game = add_technology(game, player_id, "plastic")
    game = unlock_building_in_all_cities(game, player_id, "laboratory")
    game = unlock_building_in_all_cities(game, player_id, "cristo_redentor")
    return game

def _electronics(game, player_id):
    """
    can build carrier, battleship, infantry, medical lab
    """
    game = add_technology(game, player_id, "electronics")
    game = unlock_building_in_all_cities(game, player_id, "medical_lab")
    return game

def _ballistics(game, player_id):
    """
    can build anti-aircraft gun, machine gun
    """
    game = add_technology(game, player_id, "ballistics")
    #game = unlock_building_in_all_cities(game, player_id, "neuschwanstein")
    return game

def _combustion(game, player_id):
    """
    can build destroyer, landship
    """
    game = add_technology(game, player_id, "combustion")
    #game = unlock_building_in_all_cities(game, player_id, "neuschwanstein")
    return game

def _penicilin(game, player_id):
    """
    can build marine
    """
    game = add_technology(game, player_id, "penicilin")
    game = add_trade_routes(game, player_id, 1)
    #game = unlock_building_in_all_cities(game, player_id, "neuschwanstein")
    return game

def _atomic_theory(game, player_id):
    """
    unlocks manhattan project
    reveals uranium
    """
    game = add_technology(game, player_id, "atomic_theory")
    #game = unlock_building_in_all_cities(game, player_id, "neuschwanstein")
    game = game.update_player_visible_resources_and_yields(player_id)
    return game

def _radar(game, player_id):
    """
    can build bomber, fighter, paratrooper, cn tower, airport
    """
    game = add_technology(game, player_id, "radar")
    game = unlock_building_in_all_cities(game, player_id, "cn_tower")
    game = unlock_building_in_all_cities(game, player_id, "airport")
    return game

def _combined_arms(game, player_id):
    """
    can build tank, anti-tank gun, pentagon
    """
    game = add_technology(game, player_id, "combined_arms")
    game = unlock_building_in_all_cities(game, player_id, "pentagon")
    return game

def _ecology(game, player_id):
    """
    can build solar plant, sydney operahouse, recycling center
    """
    game = add_technology(game, player_id, "ecology")
    game = unlock_building_in_all_cities(game, player_id, "solar_plant")
    game = unlock_building_in_all_cities(game, player_id, "sydney_opera_house")
    game = unlock_building_in_all_cities(game, player_id, "recycling_center")
    return game

def _nuclear_fission(game, player_id):
    """
    can build bazooka, nuclear plant
    """
    game = add_technology(game, player_id, "nuclear_fission")
    game = unlock_building_in_all_cities(game, player_id, "nuclear_plant")
    return game

def _rocketry(game, player_id):
    """
    can build mobile SAM, rocket artillery
    unlocks Apollo program
    """
    game = add_technology(game, player_id, "rocketry")
    return game


def _computers(game, player_id):
    """
    can build helicopter gunship, great firewall
    """
    game = add_technology(game, player_id, "computers")
    game = unlock_building_in_all_cities(game, player_id, "great_firewall")
    return game

def _telecom(game, player_id):
    """
    can build nuclear submarine, bomb shelter, national visitor center
    """
    game = add_technology(game, player_id, "telecom")
    game = unlock_building_in_all_cities(game, player_id, "bomb_shelter")
    game = unlock_building_in_all_cities(game, player_id, "tourist_center")
    return game

def _mobile_tactics(game, player_id):
    """
    can build mechanized infantry
    """
    game = add_technology(game, player_id, "mobile_tactics")
    return game

def _advanced_ballistics(game, player_id):
    """
    can build SS booster, nuclear missile, guided missile
    """
    game = add_technology(game, player_id, "advanced_ballistics")
    return game


def _satellites(game, player_id):
    """
    can build SS cockpit
    """
    game = add_technology(game, player_id, "satellites")
    return game

def _robotics(game, player_id):
    """
    can build missile cruiser, spaceship factory
    """
    game = add_technology(game, player_id, "robotics")
    game = unlock_building_in_all_cities(game, player_id, "spaceship_factory")
    return game

def _lasers(game, player_id):
    """
    can build jet fighter, modern armor, heavy cruiser
    """
    game = add_technology(game, player_id, "lasers")
    return game

def _internet(game, player_id):
    """
    doubles tourism output of all cities
    Deferred to GameState.step_tourism()
    """
    game = add_technology(game, player_id, "internet")
    #game = unlock_building_in_all_cities(game, player_id, "bomb_shelter")
    #game = unlock_building_in_all_cities(game, player_id, "tourist_center")
    return game

def _globalization(game, player_id):
    """
    +1 delegate to world congress, +1 delegate per spy as diplomat
    """
    game = add_technology(game, player_id, "globalization")
    #game = unlock_building_in_all_cities(game, player_id, "bomb_shelter")
    #game = unlock_building_in_all_cities(game, player_id, "tourist_center")
    return game

def _particle_physics(game, player_id):
    """
    can build SS engine
    """
    game = add_technology(game, player_id, "particle_physics")
    #game = unlock_building_in_all_cities(game, player_id, "bomb_shelter")
    #game = unlock_building_in_all_cities(game, player_id, "tourist_center")
    return game

def _nuclear_fusion(game, player_id):
    """
    can build giant death robot
    """
    game = add_technology(game, player_id, "nuclear_fusion")
    #game = unlock_building_in_all_cities(game, player_id, "bomb_shelter")
    #game = unlock_building_in_all_cities(game, player_id, "tourist_center")
    return game

def _nanotechnology(game, player_id):
    """
    can build SS stasis chamber, xcom
    """
    game = add_technology(game, player_id, "nanotechnology")
    #game = unlock_building_in_all_cities(game, player_id, "bomb_shelter")
    #game = unlock_building_in_all_cities(game, player_id, "tourist_center")
    return game

def _stealth(game, player_id):
    """
    can build stealth bomber
    """
    game = add_technology(game, player_id, "stealth")
    #game = unlock_building_in_all_cities(game, player_id, "bomb_shelter")
    #game = unlock_building_in_all_cities(game, player_id, "tourist_center")
    return game


def _future_tech(game, player_id):
    """
    increases score
    """
    game = add_technology(game, player_id, "future_tech")
    return game


ALL_TECH_FINISHERS = [do_nothing]
ALL_TECH_NAMES = ["_" + repr(x).lower().replace("technologies.", "").replace("<", "").replace(">", "").replace(":", "").split(" ")[0] for x in Technologies]

ALL_TECH_TRADE_ROUTE_BONUS = []
all_tech_trade_route_bonus_helper = ["_animal_husbandry", "_banking", "_railroad", "_penicilin"]
for tech in ALL_TECH_NAMES:
    if tech in all_tech_trade_route_bonus_helper:
        ALL_TECH_TRADE_ROUTE_BONUS.append(1)
    else:
        ALL_TECH_TRADE_ROUTE_BONUS.append(0)
    
    fn = getattr(sys.modules[__name__], tech)
    ALL_TECH_FINISHERS.append(fn)

ALL_TECH_TRADE_ROUTE_BONUS = jnp.array(ALL_TECH_TRADE_ROUTE_BONUS)

# To save into a format loadable by the replay renderer
tech_list = [
    {
        "id": tech.value,  # numeric id
        "name": tech.name,  # enum member name
        "cost": tech.cost,
        "era": tech.era,
        "prereq": tech.prereq,  # still numeric ids → lighter payload
        "grid": tech.grid_yx,  # [row, col]  (Civ V calls it Y, X)
    }
    for tech in Technologies
]

ALL_TECH_COST = jnp.array([x.cost for x in Technologies])

def _check_prereq(mask: jnp.ndarray, req_indices: Tuple[int, ...]) -> bool:
    """
    mask : 1-D jnp array  (len == len(Building))
    req_indices : tuple of enum members
    """
    # prereqs can be an empty list
    if not req_indices:                
        return True
    req = jnp.asarray(req_indices, dtype=jnp.int32)
    return jnp.all(mask[req] == 1)

ALL_TECH_PREREQ_FN = []

for tech in Technologies:
    fn = partial(_check_prereq, req_indices=tech.prereq)
    ALL_TECH_PREREQ_FN.append(fn)

print(ALL_TECH_PREREQ_FN)
qqq
#import pathlib
#import json
#out_path = pathlib.Path(__file__).with_name("technologies.json")
#out_path.write_text(json.dumps(tech_list, indent=2), encoding="utf-8")
