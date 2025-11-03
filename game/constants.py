"""
These constants need to be importable everywhere else in the codebase. TO ACCOMPLISH THIS, DO NOT IMPORT ANY
OTHER MODULES HERE, OR WE RISK CIRCULAR IMPORT ERRORS. 

If a specific mechanic has some constants (e.g., religion belief indices), then place those constants within 
the mechanic-specific file. 
"""
import jax.numpy as jnp

MAX_NUM_CITIES = 10
MAX_NUM_UNITS = 30

FOOD_IDX = 0
PROD_IDX = 1
GOLD_IDX = 2
FAITH_IDX = 3
CULTURE_IDX = 4
SCIENCE_IDX = 5
HAPPINESS_IDX = 6
TOURISM_IDX = 7

OCEAN_IDX = 0
GRASSLAND_IDX = 1
PLAINS_IDX = 2
DESERT_IDX = 3
TUNDRA_IDX = 4
SNOW_IDX = 5

FOREST_IDX = 1
JUNGLE_IDX = 2
MARSH_IDX = 3
OASIS_IDX = 4
FLOODPLAINS_IDX = 5
ICE_IDX = 6

FLATLAND_IDX = 1
HILLS_IDX = 2
MOUNTAIN_IDX = 3

EXCLUDE_FIELDS_FOR_PROJECTED_CITIES = ["ownership_map", "religion_yield_map"]

ANCIENT_ERA_IDX = 0
CLASSICAL_ERA_IDX = 1
MEDIEVAL_ERA_IDX = 2
RENAISSANCE_ERA_IDX = 3
INDUSTRIAL_ERA_IDX = 4
MODERN_ERA_IDX = 5
POSTMODERN_ERA_IDX = 6
FUTURE_ERA_IDX = 7

LAND_TRADEROUTE_RANGE = 15
SCIENCE_PER_ERA_TRADEROUTE = 2
INFLUENCE_PER_TURN_TRADEROUTE = 1.5  # two traderoutes in a row = friends 
INFLUECE_DEGRADE_PER_TURN = 1
INFLUENCE_LEVEL_FRIEND = 30
INFLUENCE_LEVEL_ALLY = 60
QUEST_CHANGE_TIMER = 30
QUEST_WINNER_INFLUENCE = 30

# [cultural, agricultural, mercantile, religious, scientific, militaristic]
# All friend bonuses are in the capital
# All ally bonuses are across all cities
FRIEND_BONUSES = jnp.array([
    [0, 0, 0, 0, 2, 0, 0],
    [2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 3],
    [0, 0, 0, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 0]
])
ALLY_BONUSES = jnp.array([
    [0, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0]
])
COMBAT_ACCEL_BONUS = 0.025
BASE_COMBAT_DAMAGE = 0.3
HILL_DEFENSE_BONUS = 0.25
JUNGLE_OR_FOREST_DEFENSE_BONUS = 0.25
FORT_DEFENSE_BONUS = 0.25
CITY_BASE_COMBAT = 8
CITY_COMBAT_BONUS_PER_5_POP = 2
CITY_IS_CAP_COMBAT_BONUS = 2.5
# [ancient, classical,  medieval, renaissance, industrial, modern, postmodern, future]
ERA_TO_INT_CITY_COMBAT_BONUS = jnp.array([1, 3, 8, 18, 33, 51, 73, 120])
DEAD_CITY_HEAL_LEVEL = 1.33

ERA_TO_NUM_SPIES = jnp.array([0, 0, 0, 1, 2, 3, 4, 5])

TECH_STEAL_CHANCE = 1/30

ROAD_MOVEMENT_DISCOUNT = 0.5


"""Used for border growth"""
H = 42
W = 66

deltas_even_row = jnp.array([
    [0, -1],  # west
    [-1, -1], # northwest
    [-1, 0],  # northeast
    [0, 1],   # east
    [1, 0],   # southeast
    [1, -1],  # southwest
])
deltas_odd_row = jnp.array([
    [0, -1],  # west
    [-1, 0],  # northwest
    [-1, 1],  # northeast
    [0, 1],   # east
    [1, 1],   # southeast
    [1, 0],   # southwest
])

# Create coordinate grids - (H, W) each
row_grid, col_grid = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij')

# Determine even rows - (H, W)
even_rows = (row_grid % 2 == 0)

# Expand grids to include neighbor dimension - (H, W, 1)
row_grid = row_grid[..., None]
col_grid = col_grid[..., None]
even_rows = even_rows[..., None, None]

# Expand deltas to broadcast - (1, 1, 6, 2)
deltas_even = deltas_even_row[None, None, :, :]
deltas_odd = deltas_odd_row[None, None, :, :]

# Select deltas based on row parity - (H, W, 6, 2)
deltas = jnp.where(even_rows, deltas_even, deltas_odd)

# Calculate all neighbor positions - (H, W, 6)
neighbor_rows = row_grid + deltas[..., 0]  # (H, W, 6)
NEIGHBOR_COLS = (col_grid + deltas[..., 1]) % W  # (H, W, 6) with wrapping

# Clip neighbor rows - (H, W, 6)
NEIGHBOR_ROWS = jnp.clip(neighbor_rows, 0, H - 1)


"""
We need to be a little careful here. All field names with things like `_accel` will have +1 to them in their respective mechanism step,
even if we do not have them explicitly in the "TO_ZERO_OUT*" list!
"""

TO_ZERO_OUT_FOR_BUILDINGS_STEP = ["additional_yield_map", "building_yields", "gw_slots", "growth_carryover", "bldg_accel", "specialist_slots", "bldg_maintenance", "unit_xp_add", "can_trade_food", "can_trade_prod", "citywide_yield_accel", "defense", "trade_gold_add_owner", "trade_land_dist_mod", "great_person_accel", "mounted_accel", "land_unit_accel", "trade_sea_dist_mod", "can_city_connect_over_water", "tech_steal_reduce_accel", "sea_unit_accel", "gw_tourism_accel", "culture_to_tourism", "air_unit_capacity", "spaceship_prod_accel", "trade_gold_add_dest", "naval_movement_add", "naval_sight_add", "city_connection_gold_accel", "armored_accel", "ranged_accel", "ranged_xp_add", "wonder_accel", "great_person_points"]
TO_ZERO_OUT_FOR_BUILDINGS_STEP_SANS_MAPS = ["building_yields", "gw_slots", "growth_carryover", "bldg_accel", "specialist_slots", "bldg_maintenance", "unit_xp_add", "can_trade_food", "can_trade_prod", "citywide_yield_accel", "defense", "trade_gold_add_owner", "trade_land_dist_mod", "great_person_accel", "mounted_accel", "land_unit_accel", "trade_sea_dist_mod", "can_city_connect_over_water", "tech_steal_reduce_accel", "sea_unit_accel", "gw_tourism_accel", "culture_to_tourism", "air_unit_capacity", "spaceship_prod_accel", "trade_gold_add_dest", "naval_movement_add", "naval_sight_add", "city_connection_gold_accel", "armored_accel", "ranged_accel", "ranged_xp_add", "wonder_accel", "great_person_points"]
TO_ZERO_OUT_FOR_BUILDINGS_STEP_ONLY_MAPS = ["additional_yield_map"]

TO_ZERO_OUT_FOR_POLICY_STEP = ["building_yields", "border_growth_accel", "wonder_accel", "city_ranged_strength_accel", "culture_nat_wonders_add", "citywide_yield_accel", "settler_accel", "bldg_accel", "policy_cost_accel", "yields_per_kill", "combat_v_barbs_accel", "military_bldg_accel", "courthouse_accel", "combat_xp_accel", "religion_bldg_accel", "faith_purchase_accel", "grand_temple_science_accel", "grand_temple_gold_accel", "cs_resting_influence", "cs_trade_route_yields", "great_wam_accel", "culture_bldg_accel", "gw_yields_add", "tourism_from_culture_bldgs_accel", "great_merch_accel", "econ_bldg_accel", "gold_purchase_accel", "naval_movement_add", "naval_sight_add", "sea_bldg_accel", "naval_strength_add", "additional_yield_map", "great_s_accel", "science_bldg_accel", "cs_relationship_bonus_accel", "cs_relationship_degrade_accel"]
TO_ZERO_OUT_FOR_POLICY_STEP_SAMS_MAPS = ["building_yields", "border_growth_accel", "wonder_accel", "city_ranged_strength_accel", "culture_nat_wonders_add", "citywide_yield_accel", "settler_accel", "bldg_accel", "policy_cost_accel", "yields_per_kill", "combat_v_barbs_accel", "military_bldg_accel", "courthouse_accel", "combat_xp_accel", "religion_bldg_accel", "faith_purchase_accel", "grand_temple_science_accel", "grand_temple_gold_accel", "cs_resting_influence", "cs_trade_route_yields", "great_wam_accel", "culture_bldg_accel", "gw_yields_add", "tourism_from_culture_bldgs_accel", "great_merch_accel", "econ_bldg_accel", "gold_purchase_accel", "naval_movement_add", "naval_sight_add", "sea_bldg_accel", "naval_strength_add", "additional_yield_map", "great_s_accel", "science_bldg_accel", "cs_relationship_bonus_accel", "cs_relationship_degrade_accel"]
TO_ZERO_OUT_FOR_POLICY_STEP_ONLY_MAPS = []

"""
Zeroing out for religion is slightly different. For the first three lists, follow the same protocol as the buildings
and social policy process. For the last three lists, we need to subset the outer GameState to a GameStateSubset. The fields
listed here will be availabe within the context of the religion tenet functions.

Everything here is only indexed by [player_id[0]], so we don't need to differentiate between MAPS and non-MAPS
"""
TO_ZERO_OUT_FOR_RELIGION_STEP = ["wonder_accel", "building_yields", "additional_yield_map", "religious_tenets_per_city", "city_ranged_strength_accel", "border_growth_accel", "missionary_spreads", "citywide_yield_accel", "cs_perturn_influence_accel", "player_perturn_influence_accel"]
TO_ZERO_OUT_FOR_RELIGION_STEP_SANS_MAPS = ["wonder_accel", "building_yields", "additional_yield_map", "religious_tenets_per_city", "city_ranged_strength_accel",  "border_growth_accel", "missionary_spreads", "citywide_yield_accel", "cs_perturn_influence_accel", "player_perturn_influence_accel"]
TO_ZERO_OUT_FOR_RELIGION_STEP_ONLY_MAPS = []

TAKE_WITH_PLAYER_ID_AND_CITY_INT = ["player_cities.buildings_owned",  "player_cities.potential_owned_rowcols", "player_cities.city_rowcols", "player_cities.ownership_map", "player_cities.population", "is_connected_to_cap"]
TAKE_WITH_PLAYER_ID = ["visible_resources_map_players"]

PASS_THRUS = ["all_resource_map", "idx_to_hex_rowcol", "terrain_map", "nw_map", "lake_map", "feature_map", "player_cities.religion_info.religious_population", "edge_river_map", "improvement_map", "road_map"]

RELIGIOUS_PRESSURE_BASES = 6
HOLY_CITY_PRESSURE_ACCEL = 3.5
RELIGIOUS_PRESSURE_DIST_MAX = 15
RELIGIOUS_PRESSURE_TRADEROUTE = 6
RELIGIOUS_PRESSURE_THRESHOLD = 100
GREAT_PROPHET_THRESHOLD = 134
SECOND_GREAT_PROPHET_THRESHOLD = 201
REFORMATION_THRESHOLD = 400

def make_update_fn(field_names_to_update: list, only_maps: bool):
    """
    Returns a function that updates only the specified fields in a flax dataclass.
    The updates dict must contain exactly those fields.
    """
    from dataclasses import fields
    field_names_to_update = tuple(field_names_to_update)  # ensure hashable/static
    
    if only_maps:
        def update_fn(pytree, updates_dict, idx_0, idx_1):
            return type(pytree)(**{
                f.name: (
                    getattr(pytree, f.name).at[idx_0].set(updates_dict[f.name])
                    if f.name in field_names_to_update else getattr(pytree, f.name)
                )
                for f in fields(pytree)
            })
    else:
        def update_fn(pytree, updates_dict, idx_0, idx_1):
            return type(pytree)(**{
                f.name: (
                    getattr(pytree, f.name).at[idx_0, idx_1].set(updates_dict[f.name])
                    if f.name in field_names_to_update else getattr(pytree, f.name)
                )
                for f in fields(pytree)
            })

    return update_fn

def make_update_fn_policies(field_names_to_update: list, only_maps: bool):
    """
    Returns a function that updates only the specified fields in a flax dataclass.
    The updates dict must contain exactly those fields.
    """
    from dataclasses import fields
    field_names_to_update = tuple(field_names_to_update)  # ensure hashable/static
    
    if only_maps:
        def update_fn(pytree, updates_dict, idx_0):
            return type(pytree)(**{
                f.name: (
                    getattr(pytree, f.name).at[idx_0].set(updates_dict[f.name])
                    if f.name in field_names_to_update else getattr(pytree, f.name)
                )
                for f in fields(pytree)
            })
    else:
        def update_fn(pytree, updates_dict, idx_0):
            return type(pytree)(**{
                f.name: (
                    getattr(pytree, f.name).at[idx_0].set(updates_dict[f.name])
                    if f.name in field_names_to_update else getattr(pytree, f.name)
                )
                for f in fields(pytree)
            })

    return update_fn

def make_update_fn_religion(field_names_to_update: list, only_maps: bool):
    """
    Returns a function that updates only the specified fields in a flax dataclass.
    The updates dict must contain exactly those fields.
    """
    from dataclasses import fields
    field_names_to_update = tuple(field_names_to_update)  # ensure hashable/static
    
    if only_maps:
        def update_fn(pytree, updates_dict, idx_0):
            return type(pytree)(**{
                f.name: (
                    getattr(pytree, f.name).at[idx_0].set(updates_dict[f.name])
                    if f.name in field_names_to_update else getattr(pytree, f.name)
                )
                for f in fields(pytree)
            })
    else:
        def update_fn(pytree, updates_dict, idx_0):
            return type(pytree)(**{
                f.name: (
                    getattr(pytree, f.name).at[idx_0].set(updates_dict[f.name])
                    if f.name in field_names_to_update else getattr(pytree, f.name)
                )
                for f in fields(pytree)
            })

    return update_fn

# [artist, musician, writer, engineer, merchant, scientist]
# food, prod, gold, faith, culture, science, happiness
SPECIALIST_YIELDS = jnp.array([
    jnp.array([0, 0, 0, 0, 3, 0, 0]),
    jnp.array([0, 0, 0, 0, 3, 0, 0]),
    jnp.array([0, 0, 0, 0, 3, 0, 0]),
    jnp.array([0, 2, 0, 0, 0, 0, 0]),
    jnp.array([0, 0, 2, 0, 0, 0, 0]),
    jnp.array([0, 0, 0, 0, 0, 3, 0]),
])
ARTIST_IDX = 0
MUSICIAN_IDX = 1
WRITER_IDX = 2
ENGINEER_IDX = 3
MERCHANT_IDX = 4
SCIENTIST_IDX = 5

SPECIALIST_GPPS = jnp.array([3, 3, 3, 3, 3, 3])

GOLDEN_AGE_TURNS = 10
GOLDEN_AGE_YIELD_ACCEL = jnp.array([1, 1.2, 1.05, 1, 1.2, 1, 1])

ERA_INT_TO_GREAT_MERCHANT_GOLD = jnp.array([300, 350, 400, 450, 500, 550, 600, 650])

CS_BORDER_GROWTH_THRESHOLD = 0.05

MAX_TRADE_DEALS = 10
TRADE_DEAL_LENGTH = 15
TRADE_DEAL_GPT_AMT = 5

DIPLO_VICTORY_THRESHOLD = 12
WC_MEETING_FREQ = 30
