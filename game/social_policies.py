"""
https://github.com/EnormousApplePie/Lekmod/tree/main/LEKMOD/Art/No%20Quitters%20Mod%20(v%2011)/social_policies
"""

from functools import partial
import jax.numpy as jnp
import jax
from typing import Tuple, List
import json
import pickle
import enum
import sys
from dataclasses import fields

from game.buildings import GameBuildings
from game.constants import HAPPINESS_IDX, TO_ZERO_OUT_FOR_POLICY_STEP, TO_ZERO_OUT_FOR_POLICY_STEP_ONLY_MAPS, TO_ZERO_OUT_FOR_POLICY_STEP_SAMS_MAPS, make_update_fn_policies, GOLDEN_AGE_TURNS, ARTIST_IDX, WRITER_IDX, MERCHANT_IDX, SCIENTIST_IDX, SECOND_GREAT_PROPHET_THRESHOLD
from game.techs import Technologies
from game.units import GameUnits


class SocialPolicies(enum.IntEnum):
    """
    name: id, tree, prereq, [y, x] (relative to the specific tree cell)
    """
    tradition_opener = 0, "tradition", [], [0, 0], [Technologies.agriculture._value_]
    aristocracy = 1, "tradition", [0], [1, 1], [Technologies.agriculture._value_]
    oligarchy = 2, "tradition", [0], [1, 5], [Technologies.agriculture._value_]
    legalism = 3, "tradition", [0], [1, 3], [Technologies.agriculture._value_]
    landed_elite = 4, "tradition", [3], [2, 2], [Technologies.agriculture._value_]
    monarchy = 5, "tradition", [3], [2, 4], [Technologies.agriculture._value_]
    
    liberty_opener = 6, "liberty", [], [0, 0], [Technologies.agriculture._value_]
    collective_rule = 7, "liberty", [8], [2, 1], [Technologies.agriculture._value_]
    republic = 8, "liberty", [6], [1, 1], [Technologies.agriculture._value_]
    citizenship = 9, "liberty", [6], [1, 4], [Technologies.agriculture._value_]
    meritocracy = 10, "liberty", [9], [2, 5], [Technologies.agriculture._value_]
    representation = 11, "liberty", [9], [2, 3], [Technologies.agriculture._value_]
    
    honor_opener = 12, "honor", [], [0, 0], [Technologies.agriculture._value_]
    warrior_code = 13, "honor", [12], [1, 2], [Technologies.agriculture._value_]
    professional_army = 14, "honor", [12], [1, 5], [Technologies.agriculture._value_]
    military_caste = 15, "honor", [13], [2, 1], [Technologies.agriculture._value_]
    military_tradition = 16, "honor", [14], [2, 5], [Technologies.agriculture._value_]
    discipline = 17, "honor", [13], [2, 3], [Technologies.agriculture._value_]

    piety_opener = 18, "piety", [], [0, 0], [Technologies.agriculture._value_]
    organized_religion = 19, "piety", [18], [1, 1], [Technologies.agriculture._value_]
    mandate_of_heaven = 20, "piety", [18], [1, 3], [Technologies.agriculture._value_]
    religious_tolerance = 21, "piety", [20], [2, 4], [Technologies.agriculture._value_]
    reformation = 22, "piety", [20], [2, 2], [Technologies.agriculture._value_]
    theocracy = 23, "piety", [18], [1, 5], [Technologies.agriculture._value_]

    patronage_opener = 24, "patronage", [], [0, 0], [x._value_ for x in Technologies if x.era == "medieval"]
    merchant_confederacy = 25, "patronage", [24], [1, 3], [x._value_ for x in Technologies if x.era == "medieval"]
    scholasticism = 26, "patronage", [25], [2, 1], [x._value_ for x in Technologies if x.era == "medieval"]
    cultural_diplomacy = 27, "patronage", [25], [2, 3], [x._value_ for x in Technologies if x.era == "medieval"]
    philanthropy = 28, "patronage", [25], [2, 5], [x._value_ for x in Technologies if x.era == "medieval"]
    consulates = 29, "patronage", [27], [3, 3], [x._value_ for x in Technologies if x.era == "medieval"]

    aesthetics_opener = 30, "aesthetics", [], [0, 0], [x._value_ for x in Technologies if x.era == "medieval"]
    cultural_centers = 31, "aesthetics", [30], [1, 3], [x._value_ for x in Technologies if x.era == "medieval"]
    ethics = 32, "aesthetics", [31], [2, 2], [x._value_ for x in Technologies if x.era == "medieval"]
    artistic_genius = 33, "aesthetics", [31], [2, 4], [x._value_ for x in Technologies if x.era == "medieval"]
    flourishing_of_arts = 34, "aesthetics", [32], [3, 2], [x._value_ for x in Technologies if x.era == "medieval"]
    fine_arts = 35, "aesthetics", [32, 33], [3, 4], [x._value_ for x in Technologies if x.era == "medieval"]

    commerce_opener = 36, "commerce", [], [0, 0], [x._value_ for x in Technologies if x.era == "medieval"]
    trade_unions = 37, "commerce", [36], [1, 5], [x._value_ for x in Technologies if x.era == "medieval"]
    caravans = 38, "commerce", [36], [1, 2], [x._value_ for x in Technologies if x.era == "medieval"]
    mercantilism = 39, "commerce", [38], [2, 3], [x._value_ for x in Technologies if x.era == "medieval"]
    entrepreneurship = 40, "commerce", [38], [2, 1], [x._value_ for x in Technologies if x.era == "medieval"]
    protectionism = 41, "commerce", [39], [3, 3], [x._value_ for x in Technologies if x.era == "medieval"]

    exploration_opener = 42, "exploration", [], [0, 0], [x._value_ for x in Technologies if x.era == "renaissance"]
    merchant_navy = 43, "exploration", [42], [1, 5], [x._value_ for x in Technologies if x.era == "renaissance"]
    maritime_infrastructure = 44, "exploration", [42], [1, 3], [x._value_ for x in Technologies if x.era == "renaissance"]
    naval_tradition = 45, "exploration", [42], [1, 1], [x._value_ for x in Technologies if x.era == "renaissance"]
    navigation_school = 46, "exploration", [44, 45], [2, 3], [x._value_ for x in Technologies if x.era == "renaissance"]
    treasure_fleets = 47, "exploration", [46], [3, 3], [x._value_ for x in Technologies if x.era == "renaissance"]

    rationalism_opener = 48, "rationalism", [], [0, 0], [x._value_ for x in Technologies if x.era == "renaissance"]
    sovereignity = 49, "rationalism", [48], [1, 1], [x._value_ for x in Technologies if x.era == "renaissance"]
    free_thought = 50, "rationalism", [48], [1, 3], [x._value_ for x in Technologies if x.era == "renaissance"]
    humanism = 51, "rationalism", [48], [1, 5], [x._value_ for x in Technologies if x.era == "renaissance"]
    scientific_revolution = 52, "rationalism", [49, 50], [2, 3], [x._value_ for x in Technologies if x.era == "renaissance"]
    secularism = 53, "rationalism", [52], [3, 3], [x._value_ for x in Technologies if x.era == "renaissance"]

    def __new__(cls, value: int, tree: str, prereq: List[int], grix_yx: List[int], tech_prereq: List[int]):
        obj = int.__new__(cls, value)     # create the real `int`
        obj._value_ = value
        obj.tree = tree 
        obj.prereq = prereq
        obj.grid_yx = grix_yx
        obj.tech_prereq = tech_prereq
        return obj

policy_list = [
    {
        "id": p.value,  # numeric id
        "name": p.name,  # enum member name
        "tree": p.tree,  # tree the sp belongs to
        "prereq": p.prereq,  # still numeric ids → lighter payload
        "grid": p.grid_yx,  # [row, col]  (Civ V calls it Y, X)
    }
    for p in SocialPolicies
]

#out_path = pathlib.Path(__file__).with_name("social_policies.json")
#print(out_path)
#out_path.write_text(json.dumps(policy_list, indent=2), encoding="utf-8")

"""
The below are "on-selection" effects from social policies.
Unfortunately, many of the require one-off attributes in e.g., the Cities dataclass...

For those that give bonus for buildings:
    If the building exists upon social policy choosing, then these functions will add the building yields.
    If the building does not exist upon sp choosing, then .step_buildings() will call functions below in the 
    "SP Effect on Buildings" section.
"""
def add_policy(game, player_id, policy):
    """"""
    policies = game.policies.at[player_id[0], policy].set(1)
    return policies


def do_nothing(game, player_id):
    return game.culture_info

def add_bldg_yields_for_existing_bldg(game, culture_info, player_id, to_add, bldg_idx):
    """"""
    b_already_owned = game.player_cities.buildings_owned[player_id[0], :, bldg_idx] == 1
    n_cities = game.player_cities.city_ids.shape[-1]
    to_add = jnp.concatenate([to_add[None] for _ in range(n_cities)], axis=0)
    new_bldg_yields = culture_info.building_yields[player_id[0]] + to_add * b_already_owned[:, None]
    new_bldg_yields = culture_info.building_yields.at[player_id[0]].set(new_bldg_yields)

    new_culture_info = culture_info.replace(building_yields=new_bldg_yields)
    return new_culture_info

def add_bldg_yields_for_existing_bldg_in_cap(game, culture_info, player_id, to_add, bldg_idx):
    """"""
    is_capital = (game.player_cities.city_ids[player_id[0]] == 1)[:, None]
    b_already_owned = game.player_cities.buildings_owned[player_id[0], :, bldg_idx] == 1
    n_cities = game.player_cities.city_ids.shape[-1]
    to_add = jnp.concatenate([to_add[None] for _ in range(n_cities)], axis=0)
    to_add = to_add * is_capital
    new_bldg_yields = culture_info.building_yields[player_id[0]] + to_add * b_already_owned[:, None]
    new_bldg_yields = culture_info.building_yields.at[player_id[0]].set(new_bldg_yields)

    new_culture_info = culture_info.replace(building_yields=new_bldg_yields)
    return new_culture_info


@partial(jax.vmap, in_axes=(None, None, None, 0, 0, 0, 0))
def add_tile_yields_vmaped_over_cities(game, player_id, to_add, bool_map, game_map_rowcols, bool_city_center, game_map_rowcols_city_center):
    # additional_yield_map (6, 42, 66, 7)
    _additional_yield_map = game.culture_info.additional_yield_map[player_id[0]]
    
    # (36,) => (36,7)
    city_ring_to_set = bool_map[:, None] * to_add[None]
    city_center_to_set = bool_city_center * to_add

    _additional_yield_map = _additional_yield_map.at[game_map_rowcols[:, 0], game_map_rowcols[:, 1]].add(city_ring_to_set)
    _additional_yield_map = _additional_yield_map.at[
        game_map_rowcols_city_center[0], game_map_rowcols_city_center[1]
    ].add(city_center_to_set)
    
    return _additional_yield_map

def _tradition_opener(game, player_id):
    """
    "greatly increases border expansion in all cities" -- what does this mean??
    +3 culture in the capital
    unlocks the hanging gardens (deferred to buildings prereq)
    """
    # We don't necessarily know which city is the capital, so we can mask with city_id
    # (max_num_cities,) -> (max_num_cities, 1)
    is_capital = (game.player_cities.city_ids[player_id[0]] == 1)[:, None]
    
    to_add = jnp.array([0, 0, 0, 0, 3, 0, 0, 0])
    to_add = jnp.concatenate([to_add[None] for _ in range(is_capital.shape[0])], axis=0)
    to_add = to_add * is_capital
    
    bldg_yields = game.culture_info.building_yields[player_id[0]] + to_add
    bldg_yields = game.culture_info.building_yields.at[player_id[0]].set(bldg_yields)

    new_border_accel = game.culture_info.border_growth_accel.at[player_id[0]].add(0.15)

    culture_info = game.culture_info.replace(building_yields=bldg_yields, border_growth_accel=new_border_accel)

    return culture_info

def _aristocracy(game, player_id):
    """
    +15% prod when building wonders
    +1 happiness for every 10 citizens in a city
    """
    new_wonder_accel = game.culture_info.wonder_accel.at[player_id[0]].add(0.15)
    # (max_num_cities,)
    population_threshold = jnp.round(game.player_cities.population[player_id[0]] / 10).astype(jnp.int32)

    new_bldg_yields = game.culture_info.building_yields.at[
        jnp.index_exp[player_id[0], jnp.arange(0, population_threshold.shape[0]), HAPPINESS_IDX]
    ].add(population_threshold)

    new_culture_info = game.culture_info.replace(
        wonder_accel=new_wonder_accel,
        building_yields=new_bldg_yields
    )
    return new_culture_info 

def _oligarchy(game, player_id):
    """
    Garrisoned units cost no maintenance
    Cities with a garrison gain +50% ranged combat str
    For simplicity, we'll do 1 free unit per city! This is deferred to 
    maths.utils.compute_unit_maintenance()
    """
    new_ranged_str = game.culture_info.city_ranged_strength_accel.at[player_id[0]].add(0.5)
    new_culture_info = game.culture_info.replace(city_ranged_strength_accel=new_ranged_str)
    return new_culture_info

def _legalism(game, player_id):
    """
    Free culture builing (monument?) in 1st 4 cities
    +2 culture from national wonders (deferred to step_empire without culture_nat_wonders_add)
    """
    new_culture_add = game.culture_info.culture_nat_wonders_add.at[
        player_id[0]
    ].add(2)
    new_culture_info = game.culture_info.replace(culture_nat_wonders_add=new_culture_add)
    return new_culture_info

def _landed_elite(game, player_id):
    """
    +15% growth, +2 food in cap
    """
    is_capital = (game.player_cities.city_ids[player_id[0]] == 1)[:, None]
    
    to_add = jnp.array([2, 0, 0, 0, 0, 0, 0, 0])
    to_add = jnp.concatenate([to_add[None] for _ in range(is_capital.shape[0])], axis=0)
    to_add = to_add * is_capital
    bldg_yields = game.culture_info.building_yields[player_id[0]] + to_add
    bldg_yields = game.culture_info.building_yields.at[player_id[0]].set(bldg_yields)

    to_add = jnp.array([0.15, 0, 0, 0, 0, 0, 0, 0])
    to_add = jnp.concatenate([to_add[None] for _ in range(is_capital.shape[0])], axis=0)
    to_add = to_add * is_capital
    yield_accel = game.culture_info.citywide_yield_accel[player_id[0]] + to_add
    citywide_yield_accel = game.culture_info.citywide_yield_accel.at[player_id[0]].set(yield_accel)

    new_culture_info = game.culture_info.replace(building_yields=bldg_yields, citywide_yield_accel=citywide_yield_accel)
    return new_culture_info

def _monarchy(game, player_id):
    """
    +1 gold, +1 happiness for every 2 pop in cap
    """
    is_capital = (game.player_cities.city_ids[player_id[0]] == 1)[:, None]
    
    population_threshold = jnp.round(game.player_cities.population[player_id[0]] / 2).astype(jnp.int32)[:, None]
    
    to_add = jnp.array([0, 0, 1, 0, 0, 0, 1, 0])
    to_add = jnp.concatenate([to_add[None] for _ in range(is_capital.shape[0])], axis=0)
    to_add = to_add * is_capital * population_threshold

    new_bldg_yields = game.culture_info.building_yields.at[
        player_id[0]
    ].add(to_add)
    
    new_culture_info = game.culture_info.replace(
        building_yields=new_bldg_yields
    )
    return new_culture_info
    
def _liberty_opener(game, player_id):
    """
    +1 culture per city
    Unlocks pyramids (deferred to building prereqs)
    """
    n_cities = game.player_cities.city_ids.shape[-1]
    is_city = game.player_cities.city_ids[player_id[0]] > 0
    to_add = jnp.array([0, 0, 0, 0, 1, 0, 0, 0])
    to_add = jnp.concatenate([to_add[None] for _ in range(n_cities)], axis=0) * is_city[:, None]
    new_bldg_yields = game.culture_info.building_yields.at[player_id[0]].add(to_add)
    new_culture_info = game.culture_info.replace(building_yields=new_bldg_yields)
    return new_culture_info

def _collective_rule(game, player_id):
    """
    +50% settler creation speed in the capital
    +1 free settler
    """
    is_capital = (game.player_cities.city_ids[player_id[0]] == 1)
    to_add = jnp.zeros_like(game.culture_info.settler_accel[player_id[0]]) + 0.5
    to_add = to_add * is_capital
    new_settler_accel = game.culture_info.settler_accel.at[player_id[0]].add(to_add)
    new_culture_info = game.culture_info.replace(settler_accel=new_settler_accel)
    return new_culture_info

def _republic(game, player_id):
    """
    +1 prod in all cities
    +5% prod towards buildings
    """
    n_cities = game.player_cities.city_ids.shape[-1]
    is_city = game.player_cities.city_ids[player_id[0]] > 0

    to_add = jnp.array([0, 1, 0, 0, 0, 0, 0, 0])
    to_add = jnp.concatenate([to_add[None] for _ in range(n_cities)], axis=0) * is_city[:, None]
    new_bldg_yields = game.culture_info.building_yields.at[player_id[0]].add(to_add)

    to_add = jnp.array([0.05])
    to_add = jnp.concatenate([to_add[None] for _ in range(n_cities)], axis=0) * is_city[:, None]
    new_bldg_accel = game.culture_info.bldg_accel.at[player_id[0]].add(to_add[:, 0])

    new_culture_info = game.culture_info.replace(building_yields=new_bldg_yields, bldg_accel=new_bldg_accel)
    return new_culture_info

def _citizenship(game, player_id):
    """
    +25% tile improvement rate
    +1 free worker in the capital
    """
    return game.culture_info

def _meritocracy(game, player_id):
    """
    +1 happiness, +15% gold from city connections
    -5% unhappiness from pop in non-occupied cities
    """
    return game.culture_info

def _representation(game, player_id):
    """
    Each city founded increases social policy cost by 33% less
    +1 gold from monuments
    Starts golden age
    """
    new_policy_cost = game.culture_info.policy_cost_accel.at[player_id[0]].add(-0.33)

    monument_idx = GameBuildings["monument"]._value_
    mnmt_already_owned = game.player_cities.buildings_owned[player_id[0], :, monument_idx] == 1
    to_add = jnp.array([0, 0, 1, 0, 0, 0, 0, 0])

    n_cities = game.player_cities.city_ids.shape[-1]
    to_add = jnp.concatenate([to_add[None] for _ in range(n_cities)], axis=0)
    to_add = to_add * mnmt_already_owned[..., None]

    new_bldg_yields = game.culture_info.building_yields[player_id[0]] + to_add
    new_bldg_yields = game.culture_info.building_yields.at[player_id[0]].set(new_bldg_yields)
    
    new_culture_info = game.culture_info.replace(policy_cost_accel=new_policy_cost, building_yields=new_bldg_yields)
    return new_culture_info

def _honor_opener(game, player_id):
    """
    +33% combat bonus versus barbs
    Gain culture for each unit killed
    Unlocks Temple of Artemis (deferred to building prereqs)
    We defer scaling the yields (per era) to step_units
    """
    new_ypk = game.culture_info.yields_per_kill.at[player_id[0]].add(jnp.array([0, 0, 0, 0, 1, 0, 0, 0]))
    combat_v_barbs_accel = game.culture_info.combat_v_barbs_accel.at[player_id[0]].add(0.33)
    new_culture_info = game.culture_info.replace(yields_per_kill=new_ypk, combat_v_barbs_accel=combat_v_barbs_accel)
    return new_culture_info

def _warrior_code(game, player_id):
    """
    4 units are maintenance free
    2 free warriors
    Gain gold for killed enemies
    We defer scaling the yields (per era) to step_units
    """
    new_ypk = game.culture_info.yields_per_kill.at[player_id[0]].add(jnp.array([0, 0, 1, 0, 0, 0, 0, 0]))
    new_culture_info = game.culture_info.replace(yields_per_kill=new_ypk)
    return new_culture_info

def _professional_army(game, player_id):
    """
    +100% prod towards barracks, armories, military academies
    Aforementioned buildings give +1 prod, +1 culture, +1 gold
    """
    new_mil_bldg_accel = game.culture_info.military_bldg_accel[player_id[0]] + 1.0
    new_mil_bldg_accel = game.culture_info.military_bldg_accel.at[player_id[0]].set(new_mil_bldg_accel)

    barracks_idx = GameBuildings["barracks"]._value_
    armory_idx = GameBuildings["armory"]._value_
    milacad_idx = GameBuildings["military_academy"]._value_
    
    b_already_owned = game.player_cities.buildings_owned[player_id[0], :, barracks_idx] == 1
    a_already_owned = game.player_cities.buildings_owned[player_id[0], :, armory_idx] == 1
    m_already_owned = game.player_cities.buildings_owned[player_id[0], :, milacad_idx] == 1

    to_add = jnp.array([0, 1, 1, 0, 1, 0, 0, 0])
    n_cities = game.player_cities.city_ids.shape[-1]
    to_add = jnp.concatenate([to_add[None] for _ in range(n_cities)], axis=0)
    
    new_bldg_yields = game.culture_info.building_yields[player_id[0]] + to_add * b_already_owned[:, None]
    new_bldg_yields = new_bldg_yields + to_add * a_already_owned[:, None]
    new_bldg_yields = new_bldg_yields + to_add * m_already_owned[:, None]
    new_bldg_yields = game.culture_info.building_yields.at[player_id[0]].set(new_bldg_yields)
    
    new_culture_info = game.culture_info.replace(
        military_bldg_accel=new_mil_bldg_accel, 
        building_yields=new_bldg_yields
    )

    return new_culture_info

def _military_caste(game, player_id):
    """
    Garrisoned units cost no maintenance
    +2 culture, +1 happiness per garrison
    Courthouses give +2 happiness
    +1 science per kill 
    """
    courthouse_idx = GameBuildings["courthouse"]._value_
    c_already_owned = game.player_cities.buildings_owned[player_id[0], :, courthouse_idx] == 1
    to_add = jnp.array([0, 0, 0, 0, 0, 0, 2, 0])
    n_cities = game.player_cities.city_ids.shape[-1]
    to_add = jnp.concatenate([to_add[None] for _ in range(n_cities)], axis=0)

    new_bldg_yields = game.culture_info.building_yields[player_id[0]] + to_add * c_already_owned[:, None]
    new_bldg_yields = game.culture_info.building_yields.at[player_id[0]].set(new_bldg_yields)
    new_culture_info = game.culture_info.replace(building_yields=new_bldg_yields)
    
    new_ypk = game.culture_info.yields_per_kill.at[player_id[0]].add(jnp.array([0, 0, 0, 0, 0, 1, 0, 0]))
    new_culture_info = game.culture_info.replace(yields_per_kill=new_ypk)
    return new_culture_info

def _military_tradition(game, player_id):
    """
    +50% prod towards Courthouses
    Courthouses give +3 food, +3 prod, +3 gold
    Citadels give +2 science, +2 culture, +1 food
    """
    new_courthouse_accel = game.culture_info.courthouse_accel[player_id[0]] + 0.5
    new_courthouse_accel = game.culture_info.courthouse_accel.at[player_id[0]].set(new_courthouse_accel)

    courthouse_idx = GameBuildings["courthouse"]._value_
    c_already_owned = game.player_cities.buildings_owned[player_id[0], :, courthouse_idx] == 1
    to_add = jnp.array([3, 3, 3, 0, 0, 0, 0, 0])
    n_cities = game.player_cities.city_ids.shape[1]
    to_add = jnp.concatenate([to_add[None] for _ in range(n_cities)], axis=0)
    
    new_bldg_yields = game.culture_info.building_yields[player_id[0]] + to_add * c_already_owned[:, None]
    new_bldg_yields = game.culture_info.building_yields.at[player_id[0]].set(new_bldg_yields)
    new_culture_info = game.culture_info.replace(building_yields=new_bldg_yields, courthouse_accel=new_courthouse_accel)
    return new_culture_info

def _discipline(game, player_id):
    """
    +100% bonus towards Heroic Epic
    Heroic Epic gives +4 happiness, +4 prod, +4 gold, +4 culture
    Military units (non air) get 50% more XP from combact
    """
    
    heroic_idx = GameBuildings["heroic_epic"]._value_
    c_already_owned = game.player_cities.buildings_owned[player_id[0], :, heroic_idx] == 1
    to_add = jnp.array([0, 4, 4, 0, 4, 0, 4, 0])
    n_cities = game.player_cities.city_ids.shape[1]
    to_add = jnp.concatenate([to_add[None] for _ in range(n_cities)], axis=0)
    new_bldg_yields = game.culture_info.building_yields[player_id[0]] + to_add * c_already_owned[:, None]
    new_bldg_yields = game.culture_info.building_yields.at[player_id[0]].set(new_bldg_yields)

    new_combat_xp_accel = game.culture_info.combat_xp_accel[player_id[0]] + 0.5
    new_combat_xp_accel = game.culture_info.combat_xp_accel.at[player_id[0]].set(new_combat_xp_accel)

    new_culture_info = game.culture_info.replace(building_yields=new_bldg_yields, combat_xp_accel=new_combat_xp_accel)
    return new_culture_info
    
def _piety_opener(game, player_id):
    """
    +1 culture, +1 faith in cap
    +100% prod towards shrines, temples
    Unlocks Mosque of Djenne (deferred to building prereqs)
    """

    is_capital = (game.player_cities.city_ids[player_id[0]] == 1)[:, None]
    to_add = jnp.array([0, 0, 0, 1, 1, 0, 0, 0])
    to_add = jnp.concatenate([to_add[None] for _ in range(is_capital.shape[0])], axis=0)
    to_add = to_add * is_capital
    
    bldg_yields = game.culture_info.building_yields[player_id[0]] + to_add
    bldg_yields = game.culture_info.building_yields.at[player_id[0]].set(bldg_yields)
    
    new_rel_bldg_accel = game.culture_info.religion_bldg_accel[player_id[0]] + 1.0
    new_rel_bldg_accel = game.culture_info.religion_bldg_accel.at[player_id[0]].set(new_rel_bldg_accel)
    
    new_culture_info = game.culture_info.replace(building_yields=bldg_yields, religion_bldg_accel=new_rel_bldg_accel)
    return new_culture_info

def _organized_religion(game, player_id):
    """
    +1 faith, +1 culture from shrines and temples
    """
    shrine_idx = GameBuildings["shrine"]._value_
    temple_idx = GameBuildings["temple"]._value_
    to_add = jnp.array([0, 0, 0, 1, 1, 0, 0, 0])

    new_culture_info = add_bldg_yields_for_existing_bldg(game, game.culture_info, player_id, to_add, shrine_idx)
    new_culture_info = add_bldg_yields_for_existing_bldg(game, new_culture_info, player_id, to_add, temple_idx)
    return new_culture_info

def _mandate_of_heaven(game, player_id):
    """
    +1 faith in the cap
    +1 happiness from each temple
    20% discount on purchase of religious units and buildings with faith
    """
    
    is_capital = (game.player_cities.city_ids[player_id[0]] == 1)[:, None]
    to_add = jnp.array([0, 0, 0, 1, 0, 0, 0, 0])
    to_add = jnp.concatenate([to_add[None] for _ in range(is_capital.shape[0])], axis=0)
    to_add = to_add * is_capital
    
    bldg_yields = game.culture_info.building_yields[player_id[0]] + to_add
    bldg_yields = game.culture_info.building_yields.at[player_id[0]].set(bldg_yields)
    new_culture_info = game.culture_info.replace(building_yields=bldg_yields)
    #game = game.replace(culture_info=new_culture_info)

    temple_idx = GameBuildings["temple"]._value_
    to_add = jnp.array([0, 0, 0, 0, 0, 0, 1, 0])
    new_culture_info = add_bldg_yields_for_existing_bldg(game, new_culture_info, player_id, to_add, temple_idx)

    new_fpa = new_culture_info.faith_purchase_accel.at[player_id[0]].add(-0.2)
    new_culture_info = new_culture_info.replace(faith_purchase_accel=new_fpa)
    return new_culture_info

def _religious_tolerance(game, player_id):
    """
    Cities with a majority religion also get pantheon belief bonus of the 2nd most popular religion 
    in the city
    +2 science from each temple
    +25% science from the Grand Temple
    """
    temple_idx = GameBuildings["temple"]._value_
    to_add = jnp.array([0, 0, 0, 0, 0, 2, 0, 0])
    new_culture_info = add_bldg_yields_for_existing_bldg(game, game.culture_info, player_id, to_add, temple_idx)
    new_accel = new_culture_info.grand_temple_science_accel.at[player_id[0]].add(0.25)
    new_culture_info = new_culture_info.replace(grand_temple_science_accel=new_accel)
    return new_culture_info

def _reformation(game, player_id):
    """
    If religion founded, gain bonus reformation belief
    +15% border growth from culture
    """
    new_bga = game.culture_info.border_growth_accel.at[player_id[0]].add(0.15)
    new_culture_info = game.culture_info.replace(border_growth_accel=new_bga)
    return new_culture_info

def _theocracy(game, player_id):
    """
    +1 gold from each shrine and temple
    Holy Sites provide +3 gold
    +33% gold output from Grand Temple
    """

    shrine_idx = GameBuildings["shrine"]._value_
    temple_idx = GameBuildings["temple"]._value_
    to_add = jnp.array([0, 0, 1, 0, 0, 0, 0, 0])
    new_culture_info = add_bldg_yields_for_existing_bldg(game, game.culture_info, player_id, to_add, shrine_idx)
    new_culture_info = add_bldg_yields_for_existing_bldg(game, new_culture_info, player_id, to_add, temple_idx)
    
    new_accel = new_culture_info.grand_temple_gold_accel.at[player_id[0]].add(0.33)
    new_culture_info = new_culture_info.replace(grand_temple_gold_accel=new_accel)
    return new_culture_info

def _patronage_opener(game, player_id):
    """
    Resting influence on CS +20
    Unlocks Forbidden Palace (deferred to building prereqs)
    """
    new_resting_infl = game.culture_info.cs_resting_influence.at[player_id[0]].add(20)
    new_culture_info = game.culture_info.replace(cs_resting_influence=new_resting_infl)
    return new_culture_info

def _merchant_confederacy(game, player_id):
    """
    +2 food, +2 prod, +1 influence per turn for trade routes with CS
    """
    # So we actually don't need to do this at all. Just maintain (6, max_num_cities, 10), then 
    # they are applied in the send_trade_route() function in action_space.py
    # Summing along axis -1 results in (players, max_num_cities) where the value of entry
    # <i,j> is the number of trade routes from the ith player's jth city to a cs
    #num_cs_tr = game.cs_trade_routes.sum(-1)[player_id[0]]
    to_add = jnp.array([[2, 2, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
    to_add = jnp.concatenate([to_add[None] for _ in range(game.player_cities.city_ids.shape[-1])], axis=0)
    
    new_tr_yields = game.culture_info.cs_trade_route_yields.at[player_id[0]].add(to_add)
    new_culture_info = game.culture_info.replace(cs_trade_route_yields=new_tr_yields)
    return new_culture_info

def _scholasticism(game, player_id):
    """
    Friend and Ally CS provide bonus science per turn based on current era
    Bonus from CS allies is higher
    Flat 50% boost to CS yields
    """
    new_relationship_bonus = game.culture_info.cs_relationship_bonus_accel.at[player_id[0]].add(0.5)
    new_culture_info = game.culture_info.replace(cs_relationship_bonus_accel=new_relationship_bonus)
    return new_culture_info

def _cultural_diplomacy(game, player_id):
    """
    Quantity of strat. resources from CS doubled
    Happiness from gifted lux +50%
    """
    new_relationship_bonus = game.culture_info.cs_relationship_bonus_accel.at[player_id[0]].add(0.25)
    new_culture_info = game.culture_info.replace(cs_relationship_bonus_accel=new_relationship_bonus)
    return new_culture_info

def _philanthropy(game, player_id):
    """
    Gold gifts to CS +25% influence
    influence degrades 25% slower
    """
    new_degrade = game.culture_info.cs_relationship_degrade_accel.at[player_id[0]].add(-0.25)
    new_culture_info = game.culture_info.replace(cs_relationship_degrade_accel=new_degrade)
    return new_culture_info

def _consulates(game, player_id):
    """
    +1 World Congress delegate
    +1 delegate per era beyond Renaissance
    """
    return game.culture_info

def _aesthetics_opener(game, player_id):
    """
    +25% gen of Great Writers, Artists, Musicians
    Unlocks Uffizi (deferred to building prereqs)
    [artist, musician, writer, engineer, merchant, scientist]
    """
    new_accel = game.culture_info.great_wam_accel.at[player_id[0], 0].add(0.25).at[player_id[0], 1].add(0.25).at[player_id[0], 2].add(0.25)
    new_culture_info = game.culture_info.replace(great_wam_accel=new_accel)
    return new_culture_info

def _cultural_centers(game, player_id):
    """
    +100% prod towards amphitheatres, opera houses, museums, broadcast towers
    +1 hapiness from writer/artist/musician guilds
    """
    new_accel = game.culture_info.culture_bldg_accel[player_id[0]] + 1.0
    new_accel = game.culture_info.culture_bldg_accel.at[player_id[0]].set(new_accel)
    new_culture_info = game.culture_info.replace(culture_bldg_accel=new_accel)

    wg_idx = GameBuildings["writers_guild"]._value_
    ag_idx = GameBuildings["artists_guild"]._value_
    mg_idx = GameBuildings["musicians_guild"]._value_
    to_add = jnp.array([0, 0, 0, 0, 0, 0, 1, 0])
    new_culture_info = add_bldg_yields_for_existing_bldg(game, new_culture_info, player_id, to_add, wg_idx)
    new_culture_info = add_bldg_yields_for_existing_bldg(game, new_culture_info, player_id, to_add, ag_idx)
    new_culture_info = add_bldg_yields_for_existing_bldg(game, new_culture_info, player_id, to_add, mg_idx)
    return new_culture_info

def _ethics(game, player_id):
    """
    Called "cultural exchange" in the game
    1 free great writer
    """
    return game.culture_info

def _artistic_genius(game, player_id):
    """
    1 free great artist
    +1 science from amphitheatres, opera houses, museums, broadcast towers
    +1.5 science from great works
    +2 science from festivals
    """

    a_idx = GameBuildings["amphitheater"]._value_
    o_idx = GameBuildings["opera_house"]._value_
    m_idx = GameBuildings["museum"]._value_
    b_idx = GameBuildings["broadcast_tower"]._value_
    to_add = jnp.array([0, 0, 0, 0, 0, 1, 0, 0])
    new_culture_info = add_bldg_yields_for_existing_bldg(game, game.culture_info, player_id, to_add, a_idx)
    new_culture_info = add_bldg_yields_for_existing_bldg(game, new_culture_info, player_id, to_add, o_idx)
    new_culture_info = add_bldg_yields_for_existing_bldg(game, new_culture_info, player_id, to_add, m_idx)
    new_culture_info = add_bldg_yields_for_existing_bldg(game, new_culture_info, player_id, to_add, b_idx)

    new_gw_add = new_culture_info.gw_yields_add.at[player_id[0]].add(jnp.array([0, 0, 0, 0, 0, 1.5, 0, 0]))
    new_culture_info = new_culture_info.replace(gw_yields_add=new_gw_add)
    return new_culture_info

def _flourishing_of_arts(game, player_id):
    """
    +1 culture, +0.15 tourism from every great work and world wonder
    Empire starts a golden age
    """
    new_gw_add = game.culture_info.gw_yields_add.at[player_id[0]].add(jnp.array([0, 0, 0, 0, 1, 0, 0, 0.15]))
    new_culture_info = game.culture_info.replace(gw_yields_add=new_gw_add)
    return new_culture_info

def _fine_arts(game, player_id):
    """
    Free scriptorium, gallery, conservatory in all cities
    +10% tourism from amphitheatres, opera houses, museums, broadcast towers
    """
    new_accel = game.culture_info.tourism_from_culture_bldgs_accel.at[player_id[0]].add(0.1)
    new_culture_info = game.culture_info.replace(tourism_from_culture_bldgs_accel=new_accel)
    return new_culture_info

def _commerce_opener(game, player_id):
    """
    +33% gen of Great Merchant
    Unlocks Big Ben (deferred to building prereqs)
    [artist, musician, writer, engineer, merchant, scientist]
    """
    new_accel = game.culture_info.great_merch_accel.at[player_id[0], 4].add(0.33)
    new_culture_info = game.culture_info.replace(great_merch_accel=new_accel)
    return new_culture_info

def _trade_unions(game, player_id):
    """
    All trade caravans produce +1 gold per turn, even when not trading
    """
    n_caravans = (game.units.unit_type[player_id[0]] == GameUnits["caravan"]._value_).sum()
    to_add = jnp.array([0, 0, 1, 0, 0, 0, 0, 0]) * n_caravans
    bldg_yields = game.culture_info.building_yields[player_id[0]] + to_add
    bldg_yields = game.culture_info.building_yields.at[player_id[0]].set(bldg_yields)
    return game.culture_info.replace(building_yields=bldg_yields)

def _caravans(game, player_id):
    """
    +50% prod towards markets, banks, and stock exchanges
    """
    new_accel = game.culture_info.econ_bldg_accel.at[player_id[0]].add(0.5)
    new_culture_info = game.culture_info.replace(econ_bldg_accel=new_accel)
    return new_culture_info

def _mercantilism(game, player_id):
    """
    -20% cost for gold purchases in cities
    +2 science from market, bank, stock exchange
    """
    a_idx = GameBuildings["market"]._value_
    o_idx = GameBuildings["bank"]._value_
    m_idx = GameBuildings["stock_exchange"]._value_
    to_add = jnp.array([0, 0, 0, 0, 0, 2, 0, 0])
    new_culture_info = add_bldg_yields_for_existing_bldg(game, game.culture_info, player_id, to_add, a_idx)
    new_culture_info = add_bldg_yields_for_existing_bldg(game, new_culture_info, player_id, to_add, o_idx)
    new_culture_info = add_bldg_yields_for_existing_bldg(game, new_culture_info, player_id, to_add, m_idx)

    new_accel = new_culture_info.gold_purchase_accel.at[player_id[0]].add(-0.2)
    new_culture_info = new_culture_info.replace(gold_purchase_accel=new_accel)
    return new_culture_info

def _entrepreneurship(game, player_id):
    """
    -25% maintenance on roads, railroads, buildings
    +2 trade routes
    """
    return game.culture_info

def _protectionism(game, player_id):
    """
    +4 prod, +4 culture, +4 happiness from East India Trade Company
    """
    m_idx = GameBuildings["national_treasury"]._value_
    to_add = jnp.array([0, 4, 0, 0, 4, 0, 4, 0])
    new_culture_info = add_bldg_yields_for_existing_bldg(game, game.culture_info, player_id, to_add, m_idx)
    return new_culture_info

def _exploration_opener(game, player_id):
    """
    +2 gold in all coastal cities
    """
    is_coastal = game.player_cities.is_coastal[player_id[0]]
    to_add = jnp.array([0, 0, 2, 0, 0, 0, 0, 0])
    to_add = to_add[None] * is_coastal[:, None]
    new_bldg_yields = game.culture_info.building_yields.at[player_id[0]].set(to_add)
    new_culture_info = game.culture_info.replace(building_yields=new_bldg_yields)
    return new_culture_info

def _merchant_navy(game, player_id):
    """
    Called "colonialism" in the GameBuildings XML table
    New cities start with 3 pop, free worker, 6 extra tiles, +2 happiness
    This is deffered to the add_settle() routine
    """
    return game.culture_info

def _maritime_infrastructure(game, player_id):
    """
    +3 prod in all coastal cities
    +50% prod towards lighthouse, harbor, seaport
    """
    is_coastal = game.player_cities.is_coastal[player_id[0]]
    previous_accel = game.culture_info.sea_bldg_accel[player_id[0]]
    new_accel = previous_accel + (0.5 * is_coastal)
    new_accel = game.culture_info.sea_bldg_accel.at[player_id[0]].set(new_accel)

    to_add = jnp.array([0, 3, 0, 0, 0, 0, 0, 0])
    to_add = to_add[None] * is_coastal[:, None]
    new_bldg_yields = game.culture_info.building_yields.at[player_id[0]].set(to_add)

    new_culture_info = game.culture_info.replace(sea_bldg_accel=new_accel, building_yields=new_bldg_yields)
    return new_culture_info

def _naval_tradition(game, player_id):
    """
    +1 happiness, +1 culture from lighthouse, harbor, seaport
    """
    a_idx = GameBuildings["lighthouse"]._value_
    o_idx = GameBuildings["harbor"]._value_
    m_idx = GameBuildings["seaport"]._value_
    to_add = jnp.array([0, 0, 0, 0, 1, 0, 1, 0])
    new_culture_info = add_bldg_yields_for_existing_bldg(game, game.culture_info, player_id, to_add, a_idx)
    new_culture_info = add_bldg_yields_for_existing_bldg(game, new_culture_info, player_id, to_add, o_idx)
    new_culture_info = add_bldg_yields_for_existing_bldg(game, new_culture_info, player_id, to_add, m_idx)
    return new_culture_info

def _navigation_school(game, player_id):
    """
    +2 science in all coastal cities
    +10% combat str for naval military units
    """
    is_coastal = game.player_cities.is_coastal[player_id[0]]
    to_add = jnp.array([0, 0, 0, 0, 0, 2, 0, 0])
    to_add = to_add[None] * is_coastal[:, None]
    new_bldg_yields = game.culture_info.building_yields.at[player_id[0]].set(to_add)

    new_strength = game.culture_info.naval_strength_add.at[player_id[0]].add(0.1)

    new_culture_info = game.culture_info.replace(building_yields=new_bldg_yields, naval_strength_add=new_strength)
    return new_culture_info

def _treasure_fleets(game, player_id):
    """
    +1 food, +1 prod, +1 gold from ocean tiles without resources and improvements
    """
    def bool_map_generator(game, player_id, city_int):
        this_city_rowcols = game.player_cities.potential_owned_rowcols[player_id[0], city_int]
        game_map_rowcols = game.idx_to_hex_rowcol[this_city_rowcols]
        
        def any_city_owns_tile(ownership_map, player_id, rowcols):
            """
            Args:
                ownership_map : (P, C, H, W) = (6, 10, 42, 66)
                player_id : () or (1) – dynamic scalar 0…5
                rowcols : (N, 2) – dynamic, here N = 36

            Returns:
                owned : (N,) bool
                    owned[i] = any_{c=0..C-1} ownership_map[player_id, c,
                                                            rowcols[i,0],
                                                            rowcols[i,1]] >= 2
            """
            # (1) pick the slice for this player  
            board_3d = jax.lax.dynamic_index_in_dim(            
                ownership_map,
                player_id[0],
                axis=0,
                keepdims=False
            )

            # (2) gather the  (C , N)  array of values 
            rows = rowcols[:, 0] 
            cols = rowcols[:, 1]

            # Advanced indexing is jit-friendly: produces shape (C, N)
            vals = board_3d[:, rows, cols]  # (10, 36)

            # (3) threshold & reduce over the city axis 
            return jnp.any(vals >= 2, axis=0)  # (36,) bool

        this_city_currently_owned = any_city_owns_tile(game.player_cities.ownership_map, player_id, game_map_rowcols)
        
        this_city_ocean = game.landmask_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] == 0
        this_city_resources = game.all_resource_type_map[game_map_rowcols[:, 0], game_map_rowcols[:, 1]] == 0
        this_city_ocean = this_city_ocean & this_city_resources

        this_city_center = game.player_cities.city_rowcols[player_id[0], city_int]
        this_city_center_ocean = game.landmask_map[this_city_center[0], this_city_center[1]] == 0

        return this_city_currently_owned * this_city_ocean, game_map_rowcols, this_city_center_ocean, this_city_center

    to_add = jnp.array([1, 1, 1, 0, 0, 0, 0])
    city_ints = jnp.arange(0, game.player_cities.city_ids.shape[-1])

    # (n, 36), (n, 36, 2), (n,), (n, 2)
    currently_owned, rowcols, citycenter, rowcol = jax.vmap(bool_map_generator, in_axes=(None, None, 0))(game, player_id, city_ints)
    
    # (n, 42, 55, 7)
    _yield_maps = add_tile_yields_vmaped_over_cities(game, player_id, to_add, currently_owned, rowcols, citycenter, rowcol)
    
    is_coastal = game.player_cities.is_coastal[player_id[0]][:, None, None, None]
    _yield_maps = _yield_maps * is_coastal
    _yield_map = _yield_maps.sum(0)
    new_additional_yield_map = game.culture_info.additional_yield_map.at[player_id[0]].set(_yield_map)
    new_culture_info = game.culture_info.replace(additional_yield_map=new_additional_yield_map)
    return new_culture_info

def _rationalism_opener(game, player_id):
    """
    +20% gen of Great Scientists
    Unlocks Porcelain Tower (deferred to buildings prereqs)
    [artist, musician, writer, engineer, merchant, scientist]
    """
    new_accel = game.culture_info.great_s_accel.at[player_id[0], 5].add(0.2)
    new_culture_info = game.culture_info.replace(great_s_accel=new_accel)
    return new_culture_info

def _sovereignity(game, player_id):
    """
    +1 gold from science buildings
    """
    lib_idx = GameBuildings["library"]._value_
    uni_idx = GameBuildings["university"]._value_
    obs_idx = GameBuildings["observatory"]._value_
    pub_idx = GameBuildings["public_school"]._value_
    lab_idx = GameBuildings["laboratory"]._value_
    to_add = jnp.array([0, 0, 1, 0, 0, 0, 0, 0])

    new_culture_info = add_bldg_yields_for_existing_bldg(game, game.culture_info, player_id, to_add, lib_idx)
    new_culture_info = add_bldg_yields_for_existing_bldg(game, new_culture_info, player_id, to_add, uni_idx)
    new_culture_info = add_bldg_yields_for_existing_bldg(game, new_culture_info, player_id, to_add, obs_idx)
    new_culture_info = add_bldg_yields_for_existing_bldg(game, new_culture_info, player_id, to_add, pub_idx)
    new_culture_info = add_bldg_yields_for_existing_bldg(game, new_culture_info, player_id, to_add, lab_idx)
    return new_culture_info

def _free_thought(game, player_id):
    """
    +1 science from barracks, armory, military academy
    """
    barracks_idx = GameBuildings["barracks"]._value_
    armory_idx = GameBuildings["armory"]._value_
    milacad_idx = GameBuildings["military_academy"]._value_
    
    to_add = jnp.array([0, 0, 0, 0, 0, 1, 0, 0])
    
    new_culture_info = add_bldg_yields_for_existing_bldg(game, game.culture_info, player_id, to_add, barracks_idx)
    new_culture_info = add_bldg_yields_for_existing_bldg(game, new_culture_info, player_id, to_add, armory_idx)
    new_culture_info = add_bldg_yields_for_existing_bldg(game, new_culture_info, player_id, to_add, milacad_idx)
    return new_culture_info

def _humanism(game, player_id):
    """
    +50% prod towards library, university, observatory, public school, research labs
    """
    new_accel = game.culture_info.science_bldg_accel.at[player_id[0]].add(0.5)
    new_culture_info = game.culture_info.replace(science_bldg_accel=new_accel)
    return new_culture_info

def _scientific_revolution(game, player_id):
    """
    1 free great scientist
    """
    return game.culture_info

def _secularism(game, player_id):
    """
    +2 science from each specialist
    The application of this Social Policy is deferred to GameState.step_specialists_and_great_people()
    """
    return game.culture_info


"""
Functions to be called when a social policy tree is completed.
Due to the design of the switch, these functions are only ever called once.
Ensure their design takes this fact into account!
This means that the fields **cannot** be zero'ed out like the fields in the other
social policy functions.
"""
def _tradition_finisher(game, player_id):
    """
    +15% growth, free aqueduct in first 4 cities
    """
    new_accel = game.growth_accel.at[player_id[0]].add(0.15)
    game = game.replace(growth_accel=new_accel)

    aq_idx = GameBuildings["aqueduct"]._value_
    new_bldgs = game.player_cities.buildings_owned.at[
        jnp.index_exp[player_id[0], jnp.arange(4), aq_idx]
    ].set(1)
    _pc = game.player_cities.replace(buildings_owned=new_bldgs)
    return game.replace(player_cities=_pc)

def _liberty_finisher(game, player_id):
    """
    +33% prod towards national wonders
    """
    new_accel = game.nat_wonder_accel.at[player_id[0]].add(0.33)
    return game.replace(nat_wonder_accel=new_accel)

def _honor_finisher(game, player_id):
    """
    Science for every enemy unit killed
    +10% food in all cities
    Great generals can be purchased with faith in industrial era +
    """
    new_accel = game.growth_accel.at[player_id[0]].add(0.1)
    new_ypk = game.culture_info.honor_finisher_yields_per_kill.at[player_id[0]].add(jnp.array([0, 0, 0, 0, 0, 1, 0, 0]))
    new_culture_info = game.culture_info.replace(yields_per_kill=new_ypk)
    return game.replace(growth_accel=new_accel, culture_info=new_culture_info)

def _piety_finisher(game, player_id):
    """
    Free great prophet
    Holy sites +1 food, +3 culture
    Free garden in first 4 citiese
    Can use faith to buy 1 great person of each type (excl. scientist) starting in industrial era
    """
    aq_idx = GameBuildings["garden"]._value_
    new_bldgs = game.player_cities.buildings_owned.at[
        jnp.index_exp[player_id[0], jnp.arange(4), aq_idx]
    ].set(1)
    _pc = game.player_cities.replace(buildings_owned=new_bldgs)
    amt_to_add = SECOND_GREAT_PROPHET_THRESHOLD
    return game.replace(player_cities=_pc, faith_reserves=game.faith_reserves.at[player_id[0]].add(amt_to_add))

def _patronage_finisher(game, player_id):
    """
    Allied CS occasionally gift great people
    +50% food, culture faith gifts from friends and allies
    Military CS give 2 units per gift
    """
    new_bonus = game.culture_info.patronage_finisher_bonus.at[player_id[0]].add(0.5)
    new_culture_info = game.culture_info.replace(patronage_finisher_bonus=new_bonus)
    return game.replace(culture_info=new_culture_info)

def _aesthetics_finisher(game, player_id):
    """
    +100% prod towards achaeologists
    Doubles theming bonus from muesums and wonders
    Hidden antiquity sites revealed
    Can purchase writers, artists, musicians with faith in industrial era
    1 free great person of choice
    """
    new_tourism_boost = game.aesthetics_finisher_bonus.at[player_id[0]].add(0.2)
    return game.replace(aesthetics_finisher_bonus=new_tourism_boost)

def _commerce_finisher(game, player_id):
    """
    2 free great merchants
    2x gold from great merchant trade missions
    +1 food from trading posts and caers
    Can purchase great merchants with faith in industrial era
    """
    amt_to_give = game.gp_threshold[player_id[0]]
    new_gpps = game.gpps.at[player_id[0], MERCHANT_IDX].add(amt_to_give * 2)  # doesn't actually spawn 2. Fix or fine?
    new_commerce_bonus = game.commerce_finisher_bonus.at[player_id[0]].add(1)
    return game.replace(gpps=new_gpps, commerce_finisher_bonus=new_commerce_bonus)

def _exploration_finisher(game, player_id):
    """
    +1 happiness from lux resources
    great admiral appears
    earn great admirals 2x as fast
    can purchase great admirals with faith starting in industrial era
    """
    new_hap = game.happiness_per_unique_lux.at[player_id[0]].add(1)
    return game.replace(happiness_per_unique_lux=new_hap)

def _rationalism_finisher(game, player_id):
    """
    +10% science while >0 happiness
    This is called in GameState.step_technologies(). Easier there than in step_empire()
    """
    new_accel = game.science_accel.at[player_id[0]].add(0.1)
    return game.replace(science_accel=new_accel)


def _check_prereq(mask: jnp.ndarray, techs: jnp.ndarray, req_indices: Tuple[int, ...], tech_prereq: List[int]) -> bool:
    """"""
    # handles empty lists (e.g., tree starters)
    if not req_indices:
        pol_prereq = True
    else:
        req = jnp.asarray(req_indices, dtype=jnp.int32)
        pol_prereq = jnp.all(mask[req] == 1)

    tech_prereq = jnp.asarray(tech_prereq, dtype=jnp.int32)

    return  pol_prereq & jnp.any(techs[tech_prereq] == 1)


ALL_SOCIAL_POLICY_PREREQ_FN = []

for policy in SocialPolicies:
    fn = partial(_check_prereq, req_indices=policy.prereq, tech_prereq=policy.tech_prereq)
    ALL_SOCIAL_POLICY_PREREQ_FN.append(fn)

ALL_SOCIAL_POLICY_FINISHERS = [do_nothing]
ALL_SOCIAL_POLICY_NAMES = ["_" + x.name for x in SocialPolicies]

for policy in ALL_SOCIAL_POLICY_NAMES:
    fn = getattr(sys.modules[__name__], policy)
    ALL_SOCIAL_POLICY_FINISHERS.append(fn)


def zero_out_fields_for_policy_update(pytree, names_to_zero, idx_0):
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
                1 if ("accel" in f.name or "carryover" in f.name or "_dist_mod" in f.name) else 6 if ("air_unit_capacity" in f.name) else 0
            )
            if f.name in relevant_fields
            else getattr(pytree, f.name)
        )
        for f in fields(pytree)
    })

player_policy_update_fn_nonmaps = make_update_fn_policies(TO_ZERO_OUT_FOR_POLICY_STEP_SAMS_MAPS, only_maps=False)
player_policy_update_fn_maps = make_update_fn_policies(TO_ZERO_OUT_FOR_POLICY_STEP_ONLY_MAPS, only_maps=True)


def apply_social_policies(game, player_id):
    """"""
    _policies_have = game.policies[player_id[0]]
    _policy_vmap_helper = jnp.arange(0, len(SocialPolicies)) + 1
    _policies_owned_vmap_idx = _policies_have * _policy_vmap_helper
    
    _culture_info = zero_out_fields_for_policy_update(game.culture_info, TO_ZERO_OUT_FOR_POLICY_STEP, player_id)
    game = game.replace(culture_info=_culture_info)

    @partial(jax.vmap, in_axes=(0,))
    def _vmap_helper(dispatch_idx):
        out_raw = jax.lax.switch(dispatch_idx, ALL_SOCIAL_POLICY_FINISHERS, game, player_id)

        # We unfortunately need to do something a little gross. The map-like fields are not stored on a 
        # per-city basis, so we need to extract those in a slightly different way... sorry about this.
        # Low priority fix.
        #out = extract_fields(out_raw.culture_info, TO_ZERO_OUT_FOR_POLICY_STEP_SAMS_MAPS, player_id)
        #out_maps = extract_map_fields(out_raw.culture_info, TO_ZERO_OUT_FOR_POLICY_STEP_ONLY_MAPS, player_id)
        out = extract_fields(out_raw, TO_ZERO_OUT_FOR_POLICY_STEP_SAMS_MAPS, player_id)
        out_maps = extract_map_fields(out_raw, TO_ZERO_OUT_FOR_POLICY_STEP_ONLY_MAPS, player_id)
        return {**out, **out_maps}
    
    # (len(SocialPolicies), 2)
    out = _vmap_helper(_policies_owned_vmap_idx)

    # There is an issue with .sum(0) --> with accel-like values that default to 1 instead of 0
    # so let's mask out effectively. So now we always set those to zero, and then +1 to them **after** the sum!
    has_this_policy_bool = game.policies[player_id[0]]

    out = jax.tree_map(
        lambda x: (x * has_this_policy_bool[(...,) + (None,) * (len(x.shape) - 1)]).sum(0).astype(x.dtype), out
    )

    # Repeating the folly mentioned in _vmap_helper
    _culture_info = player_policy_update_fn_nonmaps(game.culture_info, out, player_id[0])
    _culture_info = player_policy_update_fn_maps(_culture_info, out, player_id[0])
    _culture_info = add_one_to_appropriate_fields(_culture_info, TO_ZERO_OUT_FOR_POLICY_STEP, player_id[0])
    game = game.replace(culture_info=_culture_info)
    
    """one-offs
    Similar to the one-offs from buildings, we do the same here...
    Again, probably a cleaner/faster way to do this, but this should be fine for v1.0
    """
    # Legalism, set monument idx to 1
    # Need to do this outside of the finisher function, as only the fields within
    # TO_ZERO_OUT_FOR_POLICY_STEP will be updated!
    monument_idx = GameBuildings["monument"]._value_
    have_legalism = game.policies[player_id[0], SocialPolicies["legalism"]._value_] == 1
    have_monuments = game.player_cities.buildings_owned[player_id[0], :, monument_idx] == 1
    to_set_val = (have_legalism | have_monuments) & (game.player_cities.city_ids[player_id[0]] > 0)

    n_cities = game.player_cities.city_ids.shape[-1]
    n_cities = min(n_cities, 4)
    new_bldgs_owned = game.player_cities.buildings_owned.at[
        jnp.index_exp[player_id[0], jnp.arange(0, n_cities), monument_idx]
    ].set(to_set_val[:n_cities])

    new_player_cities = game.player_cities.replace(buildings_owned=new_bldgs_owned)
    game = game.replace(player_cities=new_player_cities)

    # Collective rule: free settler
    not_got_free_settler_cr_b4 = game.free_settler_from_collective_rule[player_id[0]] == 0
    have_cr = game.policies[player_id[0], SocialPolicies["collective_rule"]._value_] == 1
    give_free_settler = not_got_free_settler_cr_b4 & have_cr
    
    # Only can give settler if can settle again! Careful
    can_settle_again = (game.player_cities.city_ids[player_id] == 0).sum() > 0
    num_open_slots = (game.units.unit_type[player_id[0]] == 0).sum()
    open_slot_idx = game.units.unit_type[player_id[0]].argmin()
    cap_rowcol = game.player_cities.city_rowcols[player_id[0], 0]
    worked_id = GameUnits["settler"]._value_
    worker_ap = GameUnits["settler"].ap

    give_free_settler = (num_open_slots > 0) & give_free_settler & can_settle_again

    placed_rowcol = jnp.where(give_free_settler, cap_rowcol, game.units.unit_rowcol[player_id[0], open_slot_idx])
    _unit_id_ = jnp.where(give_free_settler, worked_id, game.units.unit_type[player_id[0], open_slot_idx])
    _unit_ap_ = jnp.where(give_free_settler, worker_ap, game.units.unit_ap[player_id[0], open_slot_idx])
    _unit_health_ = jnp.where(give_free_settler, 1, game.units.health[player_id[0], open_slot_idx])
    _combat_accel_ = jnp.where(give_free_settler, 0.0, game.units.combat_bonus_accel[player_id[0], open_slot_idx])
    
    game = game.replace(
       free_settler_from_collective_rule=game.free_settler_from_collective_rule.at[player_id[0]].set(have_cr),
        units=game.units.replace(
            unit_rowcol=game.units.unit_rowcol.at[player_id[0], open_slot_idx].set(placed_rowcol),
            unit_type=game.units.unit_type.at[player_id[0], open_slot_idx].set(_unit_id_),
            unit_ap=game.units.unit_ap.at[player_id[0], open_slot_idx].set(_unit_ap_),
            health=game.units.health.at[player_id[0], open_slot_idx].set(_unit_health_),
            combat_bonus_accel=game.units.combat_bonus_accel.at[player_id[0], open_slot_idx].set(_combat_accel_)
        )
    )

    # Citizenship: free worker and tile improvement
    not_got_free_worker_cit_b4 = game.free_worker_from_citizenship[player_id[0]] == 0
    have_cit = game.policies[player_id[0], SocialPolicies["citizenship"]._value_] == 1
    give_free_worker = not_got_free_worker_cit_b4 & have_cit
    
    no_improvement_speed_from_cit_b4 = game.tile_improvement_speed_from_citizenship[player_id[0]] == 0

    to_give_tile_improvement_speed = no_improvement_speed_from_cit_b4 & have_cit
    new_tile_improvement_speed = (
        to_give_tile_improvement_speed * (game.tile_improvement_speed[player_id[0]] - 0.25)
        + (1 - to_give_tile_improvement_speed) * game.tile_improvement_speed[player_id[0]]
    )
    
    num_open_slots = (game.units.unit_type[player_id[0]] == 0).sum()
    open_slot_idx = game.units.unit_type[player_id[0]].argmin()
    cap_rowcol = game.player_cities.city_rowcols[player_id[0], 0]
    worked_id = GameUnits["worker"]._value_
    worker_ap = GameUnits["worker"].ap

    give_free_worker = (num_open_slots > 0) & give_free_worker

    placed_rowcol = jnp.where(give_free_worker, cap_rowcol, game.units.unit_rowcol[player_id[0], open_slot_idx])
    _unit_id_ = jnp.where(give_free_worker, worked_id, game.units.unit_type[player_id[0], open_slot_idx])
    _unit_ap_ = jnp.where(give_free_worker, worker_ap, game.units.unit_ap[player_id[0], open_slot_idx])
    _unit_health_ = jnp.where(give_free_worker, 1, game.units.health[player_id[0], open_slot_idx])
    _combat_accel_ = jnp.where(give_free_worker, 0.0, game.units.combat_bonus_accel[player_id[0], open_slot_idx])
    
    game = game.replace(
       free_worker_from_citizenship=game.free_worker_from_citizenship.at[player_id[0]].set(have_cit),
       tile_improvement_speed_from_citizenship=game.tile_improvement_speed_from_citizenship.at[player_id[0]].set(have_cit),
       tile_improvement_speed=game.tile_improvement_speed.at[player_id[0]].set(new_tile_improvement_speed),
        units=game.units.replace(
            unit_rowcol=game.units.unit_rowcol.at[player_id[0], open_slot_idx].set(placed_rowcol),
            unit_type=game.units.unit_type.at[player_id[0], open_slot_idx].set(_unit_id_),
            unit_ap=game.units.unit_ap.at[player_id[0], open_slot_idx].set(_unit_ap_),
            health=game.units.health.at[player_id[0], open_slot_idx].set(_unit_health_),
            combat_bonus_accel=game.units.combat_bonus_accel.at[player_id[0], open_slot_idx].set(_combat_accel_)
        )
    )

    # Representation: start a golden age
    not_got_golden_age_b4 = game.golden_age_from_representation[player_id[0]] == 0
    have_rep = game.policies[player_id[0], SocialPolicies["representation"]._value_] == 1
    to_give_golden_age = not_got_golden_age_b4 & have_rep

    new_in_golden_age = jnp.where(to_give_golden_age, game.in_golden_age.at[player_id[0]].set(True), game.in_golden_age)
    new_golden_age_turns = jnp.where(
        to_give_golden_age,
        game.golden_age_turns.at[player_id[0]].add(GOLDEN_AGE_TURNS * game.golden_age_accel[player_id[0]]),
        game.golden_age_turns
    )

    game = game.replace(
        golden_age_from_representation=game.golden_age_from_representation.at[player_id[0]].set(have_rep),
        in_golden_age=new_in_golden_age,
        golden_age_turns=new_golden_age_turns,
    )

    # Warrior Code: 2 free warriors
    not_got_free_warriors = game.free_warriors_from_wc[player_id[0]] == 0
    have_wc = game.policies[player_id[0], SocialPolicies["warrior_code"]._value_] == 1
    give_free_warriors = not_got_free_warriors & have_wc
    
    num_open_slots = (game.units.unit_type[player_id[0]] == 0).sum()
    open_slot_idx = game.units.unit_type[player_id[0]].argmin()
    cap_rowcol = game.player_cities.city_rowcols[player_id[0], 0]
    worked_id = GameUnits["warrior"]._value_
    worker_ap = GameUnits["warrior"].ap

    give_free_warriors = (num_open_slots > 0) & give_free_warriors

    placed_rowcol = jnp.where(give_free_warriors, cap_rowcol, game.units.unit_rowcol[player_id[0], open_slot_idx])
    _unit_id_ = jnp.where(give_free_warriors, worked_id, game.units.unit_type[player_id[0], open_slot_idx])
    _unit_ap_ = jnp.where(give_free_warriors, worker_ap, game.units.unit_ap[player_id[0], open_slot_idx])
    _unit_health_ = jnp.where(give_free_warriors, 1, game.units.health[player_id[0], open_slot_idx])
    _unit_mil_ = jnp.where(give_free_warriors, 1, 0)
    _combat_accel_ = jnp.where(give_free_warriors, 1, game.units.combat_bonus_accel[player_id[0], open_slot_idx])

    game = game.replace(
        free_warriors_from_wc=game.free_warriors_from_wc.at[player_id[0]].set(have_wc),
        units=game.units.replace(
            unit_rowcol=game.units.unit_rowcol.at[player_id[0], open_slot_idx].set(placed_rowcol),
            unit_type=game.units.unit_type.at[player_id[0], open_slot_idx].set(_unit_id_),
            unit_ap=game.units.unit_ap.at[player_id[0], open_slot_idx].set(_unit_ap_),
            health=game.units.health.at[player_id[0], open_slot_idx].set(_unit_health_),
            military=game.units.military.at[player_id[0], open_slot_idx].set(_unit_mil_),
            combat_bonus_accel=game.units.combat_bonus_accel.at[player_id[0], open_slot_idx].set(_combat_accel_)
        )
    )
    
    num_open_slots = (game.units.unit_type[player_id[0]] == 0).sum()
    open_slot_idx = game.units.unit_type[player_id[0]].argmin()
    
    # This cascading of conditionals is OK, as we'll only ever give the 2nd warrior if we
    # gave the first warrior
    give_free_warriors = (num_open_slots > 0) & give_free_warriors

    placed_rowcol = jnp.where(give_free_warriors, cap_rowcol, game.units.unit_rowcol[player_id[0], open_slot_idx])
    _unit_id_ = jnp.where(give_free_warriors, worked_id, game.units.unit_type[player_id[0], open_slot_idx])
    _unit_ap_ = jnp.where(give_free_warriors, worker_ap, game.units.unit_ap[player_id[0], open_slot_idx])
    _unit_health_ = jnp.where(give_free_warriors, 1, game.units.health[player_id[0], open_slot_idx])
    _unit_mil_ = jnp.where(give_free_warriors, 1, 0)
    _combat_accel_ = jnp.where(give_free_warriors, 1, game.units.combat_bonus_accel[player_id[0], open_slot_idx])

    game = game.replace(
        units=game.units.replace(
            unit_rowcol=game.units.unit_rowcol.at[player_id[0], open_slot_idx].set(placed_rowcol),
            unit_type=game.units.unit_type.at[player_id[0], open_slot_idx].set(_unit_id_),
            unit_ap=game.units.unit_ap.at[player_id[0], open_slot_idx].set(_unit_ap_),
            health=game.units.health.at[player_id[0], open_slot_idx].set(_unit_health_),
            military=game.units.military.at[player_id[0], open_slot_idx].set(_unit_mil_),
            combat_bonus_accel=game.units.combat_bonus_accel.at[player_id[0], open_slot_idx].set(_combat_accel_)
        )
    )

    # Reformation: gets reformation belief
    # Already defered to step_religion()
    not_got_ref_b4 = game.reformation_belief_from_ref[player_id[0]] == 0
    have_ref = game.policies[player_id[0], SocialPolicies["reformation"]._value_] == 1
    to_give_ref = not_got_ref_b4 & have_ref

    game = game.replace(reformation_belief_from_ref=game.reformation_belief_from_ref.at[player_id[0]].set(to_give_ref))
    
    # Consulates: +1 delegate, +1 delegate per era beyond Ren
    not_got_delegates_b4 = game.delegates_from_consulates[player_id[0]] == 0
    have_cons = game.policies[player_id[0], SocialPolicies["consulates"]._value_] == 1
    to_give_dels = not_got_delegates_b4 & have_cons

    new_dels = (
        to_give_dels * (game.num_delegates[player_id[0]] + 1)
        + (1 - to_give_dels) * game.num_delegates[player_id[0]]
    )
    game = game.replace(
        num_delegates=game.num_delegates.at[player_id[0]].set(new_dels),
        delegates_from_consulates=game.delegates_from_consulates.at[player_id[0]].set(have_cons)
    )

    # Ethics: free great writer
    not_got_gw_b4 = game.free_great_writer_from_ethics[player_id[0]] == 0
    have_ethics = game.policies[player_id[0], SocialPolicies["ethics"]._value_] == 1
    give_free_gw = not_got_gw_b4 & have_ethics

    amt_to_add = jnp.where(
        give_free_gw,
        game.gp_threshold[player_id[0]],
        0
    )

    game = game.replace(
        free_great_writer_from_ethics=game.free_great_writer_from_ethics.at[player_id[0]].set(have_ethics),
        gpps=game.gpps.at[player_id[0], WRITER_IDX].add(amt_to_add)
    )

    # Artistic genius: free great artist
    not_got_free_ga_b4 = game.free_great_artist_from_art_genius[player_id[0]] == 0
    have_ag = game.policies[player_id[0], SocialPolicies["artistic_genius"]._value_] == 1
    give_free_ga = not_got_free_ga_b4 & have_ag
    
    amt_to_add = jnp.where(
        give_free_ga,
        game.gp_threshold[player_id[0]],
        0
    )

    game = game.replace(
        free_great_artist_from_art_genius=game.free_great_artist_from_art_genius.at[player_id[0]].set(have_ag),
        gpps=game.gpps.at[player_id[0], ARTIST_IDX].add(amt_to_add)
    )

    # Flourishing of Arts: start a golden age
    not_got_golden_age_b4 = game.golden_age_from_flourishing[player_id[0]] == 0
    have_flo = game.policies[player_id[0], SocialPolicies["flourishing_of_arts"]._value_] == 1
    to_give_golden_age = not_got_golden_age_b4 & have_flo
    
    new_in_golden_age = jnp.where(to_give_golden_age, game.in_golden_age.at[player_id[0]].set(True), game.in_golden_age)
    new_golden_age_turns = jnp.where(
        to_give_golden_age,
        game.golden_age_turns.at[player_id[0]].add(GOLDEN_AGE_TURNS * game.golden_age_accel[player_id[0]]),
        game.golden_age_turns
    )

    game = game.replace(
        golden_age_from_flourishing=game.golden_age_from_flourishing.at[player_id[0]].set(have_flo),
        in_golden_age=new_in_golden_age,
        golden_age_turns=new_golden_age_turns,
    )
    
    # Fine Arts: free scriptorium, gallery, conservatory
    have_fa = game.policies[player_id[0], SocialPolicies["fine_arts"]._value_] == 1
    n_cities = game.player_cities.city_ids.shape[-1]

    monument_idx = GameBuildings["scriptorium"]._value_
    have_monuments = game.player_cities.buildings_owned[player_id[0], :, monument_idx] == 1
    to_set_val = (have_fa | have_monuments) & (game.player_cities.city_ids[player_id[0]] > 0)

    new_bldgs_owned = game.player_cities.buildings_owned.at[
        jnp.index_exp[player_id[0], jnp.arange(0, n_cities), monument_idx]
    ].set(to_set_val)

    monument_idx = GameBuildings["gallery"]._value_
    have_monuments = game.player_cities.buildings_owned[player_id[0], :, monument_idx] == 1
    to_set_val = (have_fa | have_monuments) & (game.player_cities.city_ids[player_id[0]] > 0)

    new_bldgs_owned = new_bldgs_owned.at[
        jnp.index_exp[player_id[0], jnp.arange(0, n_cities), monument_idx]
    ].set(to_set_val)

    monument_idx = GameBuildings["conservatory"]._value_
    have_monuments = game.player_cities.buildings_owned[player_id[0], :, monument_idx] == 1
    to_set_val = (have_fa | have_monuments) & (game.player_cities.city_ids[player_id[0]] > 0)

    new_bldgs_owned = new_bldgs_owned.at[
        jnp.index_exp[player_id[0], jnp.arange(0, n_cities), monument_idx]
    ].set(to_set_val)

    new_player_cities = game.player_cities.replace(buildings_owned=new_bldgs_owned)
    game = game.replace(player_cities=new_player_cities)

    # Ent: +2 trade routes
    no_free_trade_routes_b4 = game.trade_routes_from_ent[player_id[0]] == 0
    have_ent = game.policies[player_id[0], SocialPolicies["entrepreneurship"]._value_] == 1
    to_give_free_trade_route = no_free_trade_routes_b4 & have_ent

    new_num_trade_routes = (
        to_give_free_trade_route * (game.num_trade_routes[player_id[0]] + 2)
        + (1 - to_give_free_trade_route) * game.num_trade_routes[player_id[0]]
    )

    game = game.replace(
        trade_routes_from_ent=game.trade_routes_from_ent.at[player_id[0]].set(have_ent),
        num_trade_routes=game.num_trade_routes.at[player_id[0]].set(new_num_trade_routes)
    )

    # Scientific Rev: free great scientist
    not_got_free_gs_b4 = game.free_great_scientist_from_sci_rev[player_id[0]] == 0
    have_scirev = game.policies[player_id[0], SocialPolicies["scientific_revolution"]._value_] == 1
    to_give_free_gs = not_got_free_gs_b4 & have_scirev
    
    amt_to_add = jnp.where(
        to_give_free_gs,
        game.gp_threshold[player_id[0]],
        0
    )

    game = game.replace(
        free_great_scientist_from_sci_rev=game.free_great_scientist_from_sci_rev.at[player_id[0]].set(have_scirev),
        gpps=game.gpps.at[player_id[0], SCIENTIST_IDX].add(amt_to_add)
    )

    """Policy Finishers
    Work in this fashion:
    (1) Build callable list [identity, finisher_fn]
    (2) dispatch_int = 1 if have all prereq policies && have not called the fn before
    (3) If called, set global flag for "have called before" to ensure non-repeat calls
    """
    # Tradition
    ALL_TRAD_OPTS = [lambda x, y: x, _tradition_finisher]
    have_all = game.policies[player_id[0], :SocialPolicies["liberty_opener"]._value_].mean() == 1
    not_done_b4 = game.tradition_finished[player_id[0]] == 0
    finisher_dispatch_bool = have_all & not_done_b4
    finisher_dispatch_int = 1 * finisher_dispatch_bool + (1 - finisher_dispatch_bool) * 0
    game = game.replace(tradition_finished=game.tradition_finished.at[player_id[0]].set(have_all))
    game = jax.lax.switch(finisher_dispatch_int, ALL_TRAD_OPTS, game, player_id)

    # Liberty
    ALL_LIB_OPTS = [lambda x, y: x, _liberty_finisher]
    have_all = game.policies[player_id[0], SocialPolicies["liberty_opener"]._value_: SocialPolicies["honor_opener"]._value_].mean() == 1
    not_done_b4 = game.liberty_finished[player_id[0]] == 0
    finisher_dispatch_bool = have_all & not_done_b4
    finisher_dispatch_int = 1 * finisher_dispatch_bool + (1 - finisher_dispatch_bool) * 0
    game = game.replace(liberty_finished=game.liberty_finished.at[player_id[0]].set(have_all))
    game = jax.lax.switch(finisher_dispatch_int, ALL_LIB_OPTS, game, player_id)

    # Honor
    ALL_HON_OPTS = [lambda x, y: x, _honor_finisher]
    have_all = game.policies[player_id[0], SocialPolicies["honor_opener"]._value_: SocialPolicies["piety_opener"]._value_].mean() == 1
    not_done_b4 = game.honor_finished[player_id[0]] == 0
    finisher_dispatch_bool = have_all & not_done_b4
    finisher_dispatch_int = 1 * finisher_dispatch_bool + (1 - finisher_dispatch_bool) * 0
    game = game.replace(honor_finished=game.honor_finished.at[player_id[0]].set(have_all))
    game = jax.lax.switch(finisher_dispatch_int, ALL_HON_OPTS, game, player_id)

    # Piety
    ALL_PIE_OPTS = [lambda x, y: x, _piety_finisher]
    have_all = game.policies[player_id[0], SocialPolicies["piety_opener"]._value_: SocialPolicies["patronage_opener"]._value_].mean() == 1
    not_done_b4 = game.piety_finished[player_id[0]] == 0
    finisher_dispatch_bool = have_all & not_done_b4
    finisher_dispatch_int = 1 * finisher_dispatch_bool + (1 - finisher_dispatch_bool) * 0
    game = game.replace(piety_finished=game.piety_finished.at[player_id[0]].set(have_all))
    game = jax.lax.switch(finisher_dispatch_int, ALL_PIE_OPTS, game, player_id)

    # Patronage
    ALL_PAT_OPTS = [lambda x, y: x, _patronage_finisher]
    have_all = game.policies[player_id[0], SocialPolicies["patronage_opener"]._value_: SocialPolicies["aesthetics_opener"]._value_].mean() == 1
    not_done_b4 = game.patronage_finished[player_id[0]] == 0
    finisher_dispatch_bool = have_all & not_done_b4
    finisher_dispatch_int = 1 * finisher_dispatch_bool + (1 - finisher_dispatch_bool) * 0
    game = game.replace(patronage_finished=game.patronage_finished.at[player_id[0]].set(have_all))
    game = jax.lax.switch(finisher_dispatch_int, ALL_PAT_OPTS, game, player_id)

    # Aesthetics
    ALL_AES_OPTS = [lambda x, y: x, _aesthetics_finisher]
    have_all = game.policies[player_id[0], SocialPolicies["aesthetics_opener"]._value_: SocialPolicies["commerce_opener"]._value_].mean() == 1
    not_done_b4 = game.aesthetics_finished[player_id[0]] == 0
    finisher_dispatch_bool = have_all & not_done_b4
    finisher_dispatch_int = 1 * finisher_dispatch_bool + (1 - finisher_dispatch_bool) * 0
    game = game.replace(aesthetics_finished=game.aesthetics_finished.at[player_id[0]].set(have_all))
    game = jax.lax.switch(finisher_dispatch_int, ALL_AES_OPTS, game, player_id)

    # Commerce
    ALL_COM_OPTS = [lambda x, y: x, _commerce_finisher]
    have_all = game.policies[player_id[0], SocialPolicies["commerce_opener"]._value_: SocialPolicies["exploration_opener"]._value_].mean() == 1
    not_done_b4 = game.commerce_finished[player_id[0]] == 0
    finisher_dispatch_bool = have_all & not_done_b4
    finisher_dispatch_int = 1 * finisher_dispatch_bool + (1 - finisher_dispatch_bool) * 0
    game = game.replace(commerce_finished=game.commerce_finished.at[player_id[0]].set(have_all))
    game = jax.lax.switch(finisher_dispatch_int, ALL_COM_OPTS, game, player_id)

    # Exploration
    ALL_EXP_OPTS = [lambda x, y: x, _exploration_finisher]
    have_all = game.policies[player_id[0], SocialPolicies["exploration_opener"]._value_: SocialPolicies["rationalism_opener"]._value_].mean() == 1
    not_done_b4 = game.exploration_finished[player_id[0]] == 0
    finisher_dispatch_bool = have_all & not_done_b4
    finisher_dispatch_int = 1 * finisher_dispatch_bool + (1 - finisher_dispatch_bool) * 0
    game = game.replace(exploration_finished=game.exploration_finished.at[player_id[0]].set(have_all))
    game = jax.lax.switch(finisher_dispatch_int, ALL_EXP_OPTS, game, player_id)

    # Rationalism
    ALL_RAT_OPTS = [lambda x, y: x, _rationalism_finisher]
    have_all = game.policies[player_id[0], SocialPolicies["rationalism_opener"]._value_:].mean() == 1
    not_done_b4 = game.rationalism_finished[player_id[0]] == 0
    finisher_dispatch_bool = have_all & not_done_b4
    finisher_dispatch_int = 1 * finisher_dispatch_bool + (1 - finisher_dispatch_bool) * 0
    game = game.replace(rationalism_finished=game.rationalism_finished.at[player_id[0]].set(have_all))
    game = jax.lax.switch(finisher_dispatch_int, ALL_RAT_OPTS, game, player_id)

    return game
